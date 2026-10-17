/// \file whisper_npu.hpp
/// \brief whisper_npu class
/// \author FastFlowLM Team
/// \date 2025-10-17
/// \version 0.9.24
/// \note This is a source file for the modeling_whisper class
#include "whisper/modeling_whisper.hpp"


Whisper::Whisper(flm_rt::device* npu_device_inst){
    this->device = npu_device_inst;
    
    time_stamp = 0;
    audio_buffer.clear();
    fft_400 = std::make_unique<FFT400>();
    mel_feature = buffer<bf16>(128 * 3000);
    this->profiler_list.resize(PROFILER_TYPE_NUM);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i] = profiler();
        this->profiler_list[i].reset();
    }
    this->last_prefill_time = { 0, "us" };
}

void Whisper::load_model(std::string model_path, nlohmann::ordered_json model_info, bool enable_preemption) {
    header_print("FLM", "Loading model: " << model_path);
    this->npu = std::make_unique<npu_xclbin_manager>(npu_device::device_npu2, this->device, enable_preemption);
    this->enable_preemption = enable_preemption;
    this->model_path = model_path;
    this->enable_preemption = enable_preemption;

    this->lm_config = std::make_unique<Whisper_Config>();
    this->lm_config->from_pretrained(this->model_path);
    
    this->whisper_engine = std::make_unique<whisper_npu>(*this->lm_config, this->npu.get(), 448);
    std::unique_ptr<Q4NX> q4nx = std::make_unique<Q4NX>(this->model_path);

    this->whisper_engine->load_weights(*q4nx);

    // //free the q4nx
    q4nx.reset();
    this->whisper_engine->clear_context();
    this->setup_tokenizer(model_path);
    
    this->sampler.reset();

    sampler_config s_config;
    s_config.top_k = 1;
    s_config.top_p = 0.95;
    s_config.min_p = 0.1;
    s_config.temperature = 0.4;

    this->sampler = std::make_unique<Sampler>(this->lm_config->get("vocab_size"), s_config);
}

void Whisper::setup_tokenizer(std::string model_path) {
    // load tokenizer configurations
    #ifdef _WIN32
    std::string tokenizer_config_path = model_path + "\\tokenizer_config.json";
    #else
    std::string tokenizer_config_path = model_path + "/tokenizer_config.json";
    #endif
    std::ifstream fs_config(tokenizer_config_path, std::ios::in | std::ios::binary);
    if (fs_config.fail()) {
        std::cerr << "Cannot open " << tokenizer_config_path << std::endl;
        exit(1);
    }
    std::string data_config;
    fs_config.seekg(0, std::ios::end);
    size_t size_config = static_cast<size_t>(fs_config.tellg());
    fs_config.seekg(0, std::ios::beg);
    data_config.resize(size_config);
    fs_config.read(data_config.data(), size_config);
    fs_config.close();
    auto tokenizer_config = nlohmann::json::parse(data_config);
    // check if bos_token is null
    if (tokenizer_config["bos_token"].is_null()) {
        this->has_bos_token = false;
    }
    else {
        this->has_bos_token = true;
    }

    if (this->has_bos_token) {
        this->bos_token_id = tokenizer_config["bos_token_id"].get<int>();
    }
    else {
        this->bos_token_id = -1;
    }
    this->eos_token = tokenizer_config["eos_token"].get<std::string>();
    for (auto& token : tokenizer_config["eos_token_id"]) {
        this->eos_token_ids.push_back(token.get<int>());
    }
 
    this->tokenizer = std::make_unique<Tokenizer>(this->model_path);
    this->_build_time_map();
}

bool Whisper::load_audio(std::string& audio_path) {    
    this->audio_buffer = this->_load_audio(audio_path);
    header_print("info", "Length of audio: " + std::to_string(_S2T_(this->audio_buffer.size())) + " seconds");
    return true;
}

bool Whisper::load_audio(std::vector<uint8_t>& audio_data) {
    this->audio_buffer = this->_load_audio(audio_data);
    header_print("info", "Length of audio: " + std::to_string(_S2T_(this->audio_buffer.size())) + " seconds");
    return true;
}

std::pair<std::string, std::string> Whisper::generate(whisper_task_type_t task, bool enable_time_stamp, bool return_time_stamp, std::ostream& os,
                                                      const std::string& language, const std::string& prompt) {
    int length = this->audio_buffer.size();
    int current_idx = 0;
    int overlapping_samples = 5 * FS; // 
    int l_this_round = std::min(WINDOW_SAMPLES, length);
    int last_time_stamp = this->token_time_map_offset;
    int last_idx;
    bool last_chunk = false;
    std::string result;
    std::string language_detected;
    if ((!enable_time_stamp) && (return_time_stamp)){
        header_print("Error", "Return_time_stamp is true but timestamp is not enabled!");
        return std::make_pair("", "");
    }
    const int requested_language = language.empty() ? -1 : this->_language_token(language);
    std::vector<int> prompt_tokens;
    if (!prompt.empty()) {
        // As openai-whisper: leading space, and at most half the context.
        prompt_tokens = this->tokenizer->encode(" " + prompt);
        std::erase_if(prompt_tokens, [](int token) { return token >= 50257; });  // text only
        const size_t max_prompt_tokens = MAX_L / 2 - 1;
        if (prompt_tokens.size() > max_prompt_tokens) {
            prompt_tokens.erase(prompt_tokens.begin(), prompt_tokens.end() - max_prompt_tokens);
        }
    }
    const int prefix_length = prompt_tokens.empty() ? 0 : 1 + (int)prompt_tokens.size();
    auto emit = [&](const std::string& text) {
        os << text << std::flush;
        result += text;
    };
    while (current_idx < length){
        // std::cout << "Chunk " << _S2T_(current_idx) << "s to " << _S2T_(current_idx + l_this_round) << "s" << std::endl;
        float time_offset = _S2T_(current_idx);
        
        if (l_this_round == 0){
            break;
        }

        std::vector<float> audio_chunk(l_this_round);

        audio_chunk.insert(audio_chunk.begin(), this->audio_buffer.data() + current_idx, this->audio_buffer.data() + current_idx + l_this_round);
        
        _preprocess_audio(mel_feature, audio_chunk);
      
        // run whisper encoder
        this->whisper_engine->encode_audio(mel_feature); // encoded and pass kv-cache to decoder

        // decoder loop
        this->whisper_engine->clear_context();
        this->sampler->reset_penalties();

        if (!prompt_tokens.empty()) {
            this->whisper_engine->decode_audio(start_of_prev_token);
            for (int token : prompt_tokens) {
                this->whisper_engine->decode_audio(token);
            }
        }
        last_idx = start_of_transcript;
        buffer<bf16> logits = this->whisper_engine->decode_audio(last_idx);
        last_idx = requested_language >= 0 ? requested_language : this->_sample_in_language(logits);
      
        // std::cout << "Language detected: " << this->tokenizer->run_time_decoder(last_idx) << "(" << langmap::to_language_name(this->tokenizer->run_time_decoder(last_idx)) << ")" << std::endl;
        language_detected = this->tokenizer->run_time_decoder(last_idx);
        // The prompt is <|startoftranscript|><|lang|><|task|>.
        this->whisper_engine->decode_audio(last_idx);

        if (task == e_translate) {
            buffer<bf16> logits = this->whisper_engine->decode_audio(translate_token);
            last_idx = this->_sample_in_time_stamp(logits);
        }
        else if (task == e_transcribe) {
            buffer<bf16> logits = this->whisper_engine->decode_audio(transcribe_token);
            last_idx = this->_sample_in_time_stamp(logits);
        }
        else {
            header_print("Error", "Non-recongnized task!");
        }

        if (enable_time_stamp){
            if (return_time_stamp){
                std::string time_stamp = this->tokenizer->run_time_decoder(last_idx);
                std::string offset_time_stamp = this->_offset_time_stamp(time_stamp, time_offset);
                result += offset_time_stamp;
                os << offset_time_stamp << std::flush;
            }
        }
        else{
            buffer<bf16> logits = this->whisper_engine->decode_audio(no_time_stamp_token);
            last_idx = this->sampler->sample(logits);
            std::string token_str = this->tokenizer->run_time_decoder(last_idx);
            result += token_str;
            os << token_str << std::flush;
        }
        
        // A full window is followed by one that starts at its last timestamp,
        // so text after that timestamp would be decoded twice. Hold it, with
        // the timestamp that opens it, until a timestamp closes it, and drop
        // both if none does.
        const bool full_window = enable_time_stamp && l_this_round == WINDOW_SAMPLES;
        std::string pending;
        last_time_stamp = this->token_time_map_offset;
        int prev_idx = -1;
        int window_time_stamp = enable_time_stamp ? last_idx : -1;
        for (int i = 0; i < MAX_L - 4 - prefix_length; i++){
            buffer<bf16> logits = this->whisper_engine->decode_audio(last_idx);
            if (enable_time_stamp){
                this->_apply_time_stamp_rules(logits, prev_idx, last_idx, window_time_stamp);
            }
            prev_idx = last_idx;
            last_idx = this->sampler->sample(logits);
            std::string token_str = this->tokenizer->run_time_decoder(last_idx);
            
            if (_is_normal_token(last_idx)){
                if (full_window){
                    pending += token_str;
                }
                else{
                    emit(token_str);
                }
            }

            if (enable_time_stamp && _is_time_stemp(last_idx)){
                std::string ts = return_time_stamp ? this->_offset_time_stamp(token_str, time_offset) : "";
                if (full_window && _is_time_stemp(prev_idx)){
                    pending = ts;
                }
                else{
                    emit(pending);
                    pending.clear();
                    emit(ts);
                }
                last_time_stamp = last_idx;
                window_time_stamp = last_idx;
            }
         
            if (last_idx == 50257){
                break;
            }
        }

        
        if (l_this_round < WINDOW_SAMPLES){
            break;
        }
     
        if (enable_time_stamp){
            float end_time = _get_time(last_time_stamp);
            if (end_time == 0){
                // No timestamp to resume from: keep the text, skip the window.
                emit(pending);
                end_time = 30;
            }
            l_this_round = end_time * FS;
        }

        
        current_idx += l_this_round;
 
        l_this_round = std::min(WINDOW_SAMPLES, length - current_idx);
        l_this_round = std::max(l_this_round, 0);
        
    }
    return std::make_pair(result, langmap::to_language_name(language_detected));
}


void Whisper::_build_time_map(){
    this->token_time_map.clear();
    auto ids = this->tokenizer->encode("<|0.00|>");
    this->token_time_map_offset = ids[0];
    // <|0.00|> to <|30.00|>: 1501 tokens. Summing 0.02f drifts and stops at 29.98.
    for (int i = 0; i <= WINDOW_LENGTH * 50; i++){
        this->token_time_map.push_back(i / 50.0f);
    }

    this->total_time_stamps = this->token_time_map.size();
}

int Whisper::_sample_in_language(buffer<bf16>& logits){
    
    for (int i = 0; i < 50259; i++){
        logits[i] = -0x1.FEp127f;
    }
    for (int i = 50359; i < logits.size(); i++){
        logits[i] = -0x1.FEp127f;
    }
    return this->sampler->sample(logits);

}

int Whisper::_language_token(const std::string& language){
    std::vector<int> ids = this->tokenizer->encode("<|" + language + "|>");
    if (ids.size() != 1 || ids[0] < first_language_token || ids[0] > last_language_token){
        throw std::invalid_argument("unsupported language: " + language);
    }
    return ids[0];
}

int Whisper::_sample_in_time_stamp(buffer<bf16>& logits){
    // The first timestamp of a window, no later than <|1.00|>.
    this->_mask(logits, 0, this->token_time_map_offset);
    this->_mask(logits, this->token_time_map_offset + max_initial_time_stamp + 1, logits.size());
    return this->sampler->sample(logits);
}

// openai-whisper's ApplyTimestampRules: timestamps come in pairs around
// text, a segment is not empty, and time never goes back.
void Whisper::_apply_time_stamp_rules(buffer<bf16>& logits, int prev_idx, int last_idx, int last_time_stamp_idx){
    const int begin = this->token_time_map_offset;
    const int end = begin + this->total_time_stamps;
    const bool last_was_time_stamp = _is_time_stemp(last_idx);
    const bool penultimate_was_time_stamp = prev_idx < 0 || _is_time_stemp(prev_idx);
    if (last_was_time_stamp){
        if (penultimate_was_time_stamp){
            this->_mask(logits, begin, end);  // a segment needs text
        }
        else{
            this->_mask(logits, 0, eot_token);  // a closed segment: timestamp or EOT
        }
    }
    if (last_time_stamp_idx >= 0){
        int lowest = (last_was_time_stamp && !penultimate_was_time_stamp) ? last_time_stamp_idx : last_time_stamp_idx + 1;
        this->_mask(logits, begin, std::min(lowest, end));
    }
}