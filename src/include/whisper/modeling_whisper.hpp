/// \file modeling_whisper.hpp
/// \brief modeling_whisper class
/// \author FastFlowLM Team
/// \date 2025-10-17
/// \version 0.9.24
/// \note This is a header file for the modeling_whisper class

#pragma once
#include "base64.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include "whisper/whisper_npu.hpp"
#include "nlohmann/json.hpp"
#include "utils/profiler.hpp"
#include "lm_config.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
// FFmpeg includes for image processing only
extern "C" {
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
}

#include <vector>
#include "typedef.hpp"
#include <complex>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <unordered_map>
#include "audio_process_utils/rfft.hpp"
#include <immintrin.h>  // For AVX intrinsics
#include "metrices.hpp"
#include "tokenizer/tokenizer.hpp"
#include "modules/sampler.hpp"
#include "whisper/language_table.hpp"

class FFT400 {
public:
    static constexpr int N = 400;

    FFT400() : out((N / 2 + 1) * 2) {}

    // Not copyable: compute_power() reuses the output buffer
    FFT400(const FFT400&) = delete;
    FFT400& operator=(const FFT400&) = delete;

    std::vector<float> compute_power(const std::vector<float>& input) const {
        if (input.size() != N) {
            throw std::runtime_error("FFT400 input must have 400 samples, got " + std::to_string(input.size()));
        }

        // Execute FFT (interleaved re, im output)
        audioproc::rfft_many(input.data(), out.data(), 1, N);

        // Extract power spectrum (magnitude squared)
        std::vector<float> power(N / 2 + 1);
        for (int k = 0; k <= N / 2; ++k) {
            float real = out[2 * k];
            float imag = out[2 * k + 1];
            power[k] = real * real + imag * imag;
        }

        return power;
    }

private:
    mutable std::vector<float> out;
};

/************              Whisper            **************/
class Whisper {
private:

    static constexpr int FS = 16000;
    static constexpr int WINDOW_LENGTH = 30; // seconds
    static constexpr int WINDOW_SAMPLES = FS * WINDOW_LENGTH;
    static constexpr int MAX_L = 448;
    static constexpr int start_of_transcript = 50258;
    static constexpr int first_language_token = 50259;
    static constexpr int last_language_token = 50358;
    static constexpr int start_of_prev_token = 50362;
    static constexpr int no_time_stamp_token = 50364;
    static constexpr int translate_token = 50359;
    static constexpr int transcribe_token = 50360;
    static constexpr int eot_token = 50257;
    static constexpr int max_initial_time_stamp = 50;  // <|1.00|>, as openai-whisper
    typedef enum {
        PREFILL_TIME,
        DECODING_TIME,
        SAMPLING_TIME,
        TKOEN_ENCODE_TIME,
        TKOEN_DECODE_TIME,
        TTFT_TIME,
        TOTAL_TIME,
        PROFILER_TYPE_NUM
    } profiler_type;
	std::vector<profiler> profiler_list;
	time_utils::time_with_unit last_prefill_time;
    std::vector<float> audio_buffer;
    buffer<bf16> mel_feature;
    float time_stamp;

    std::string model_path;
    std::unique_ptr<FFT400> fft_400;
    std::unique_ptr<whisper_npu> whisper_engine;
    std::unique_ptr<npu_xclbin_manager> npu;
	std::unique_ptr<Whisper_Config> lm_config = nullptr;
    flm_rt::device* device;

    bool enable_preemption;

	std::unique_ptr<Tokenizer> tokenizer = nullptr;
	std::unique_ptr<Sampler> sampler = nullptr;

	std::string bos_token;
    std::string eos_token;
    std::string boi_token;
    std::string eoi_token;
    bool has_bos_token;
    int bos_token_id;
    std::vector<int> eos_token_ids;
    std::vector<float> token_time_map;
    int token_time_map_offset;
    int total_time_stamps;


	std::string user_system_prompt = "";

    nlohmann::json extra_context;
    std::vector<float> _load_audio(std::string& audio_path);
    std::vector<float> _load_audio(std::vector<uint8_t>& audio_data);
    void _preprocess_audio(buffer<bf16>& mel_features, std::vector<float>& audio);

    inline float _S2T_(int   sample_idx ) { return (float)sample_idx / FS;}
    inline int   _T2S_(float time_second) { return int(time_second * FS);} 


    void _build_time_map();
    float _get_time(int& token_id){
        return this->token_time_map[token_id - token_time_map_offset];
    }
    int _get_token_id(float& time_stamp) {
        int rounded_time = (int) time_stamp * 50;
        return this->token_time_map_offset + rounded_time;
    }
    bool _is_time_stemp(int& token){
        return (token >= token_time_map_offset) && (token < (total_time_stamps + token_time_map_offset));
    }

    inline bool _is_normal_token(int& token){
        return token < 50257;
    }

    int _sample_in_language(buffer<bf16>& logits);
    int _language_token(const std::string& language);

    int _sample_in_time_stamp(buffer<bf16>& logits);
    void _apply_time_stamp_rules(buffer<bf16>& logits, int prev_idx, int last_idx, int last_time_stamp_idx);
    inline void _mask(buffer<bf16>& logits, int begin, int end){
        for (int i = begin; i < end; i++){
            logits[i] = -0x1.FEp127f;
        }
    }

    std::string _offset_time_stamp(std::string& time_stamp, float offset){
        float time;
        sscanf(time_stamp.c_str(), "<|%f|>", &time);
        time += offset;
        char buffer[128];
        sprintf(buffer, "<|%.2f|>", time);
        return std::string(buffer);
    }
public:
    typedef enum {
        e_translate = 0,
        e_transcribe = 1
    } whisper_task_type_t;

    Whisper(flm_rt::device* npu_device_inst);

    void load_model(std::string model_path, nlohmann::ordered_json model_inf, bool enable_preemption = false);
    //void toggle_enable_think() override;
    bool load_audio(std::string& audio_path);
    bool load_audio(std::vector<uint8_t>& audio_data);
    /// \param language ISO 639-1 code (e.g. "en"); empty to detect it per window
    /// \param prompt text to condition the decoder on, as OpenAI's `prompt`
    std::pair<std::string, std::string> generate(whisper_task_type_t task, bool enable_time_stamp, bool return_time_stamp, std::ostream& os,
                                                 const std::string& language = "", const std::string& prompt = "");
    void setup_tokenizer(std::string model_path);
};