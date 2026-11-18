/// \file gemma4_open_frontend.cpp
/// \brief Provider-owned Gemma 4 request preparation for extension E2E tests.
#include "AutoModel/model_extension.h"
#include "audio/audio_reader.hpp"
#include "audio_process_utils/audioproc.hpp"
#include "base64.hpp"
#include "chat/chat_continue.hpp"
#include "image/image_reader.hpp"
#include "image_process_utils/imageproc.hpp"
#include "minja/chat-template.hpp"
#include "models/gemma4e/gemma4e_payload.hpp"
#include "tokenizers_cpp.h"
#include "typedef.hpp"

#include <nlohmann/json.hpp>

#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
// onig_sys is compiled by Cargo and retains one dynamic-UCRT import even when
// the enclosing static library requests crt-static. Resolve that import to the
// same static-CRT implementation used by this provider DLL.
extern "C" {
using flm_strcat_s_fn = errno_t(__cdecl*)(char*, std::size_t, const char*);
__declspec(selectany) flm_strcat_s_fn __imp_strcat_s = &strcat_s;
}
#endif

namespace gemma4_open {
namespace {

using json = nlohmann::ordered_json;

std::string String(flm_span_v1 value) {
    return value.data
        ? std::string(static_cast<const char*>(value.data), value.size)
        : std::string();
}

flm_span_v1 Span(const std::string& value) {
    return {value.data(), value.size()};
}

void SetError(const flm_error_sink_v1* sink, const std::string& message) {
    if (sink && sink->set_error) {
        sink->set_error(sink->context, message.data(), message.size());
    }
}

template <class Operation>
int Guard(const flm_error_sink_v1* error, Operation&& operation) {
    try {
        std::forward<Operation>(operation)();
        return FLM_MODEL_EXTENSION_STATUS_OK;
    } catch (const std::invalid_argument& exception) {
        SetError(error, exception.what());
        return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& exception) {
        SetError(error, exception.what());
    } catch (...) {
        SetError(error, "unknown Gemma4-open frontend exception");
    }
    return FLM_MODEL_EXTENSION_STATUS_ERROR;
}

std::string ReadFile(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open " + path.string());
    return {std::istreambuf_iterator<char>(stream),
            std::istreambuf_iterator<char>()};
}

std::string Download(const std::string& url) {
#if defined(_WIN32)
    (void)url;
    throw std::invalid_argument(
        "URL media is not supported by this Windows provider build");
#else
    int output_pipe[2];
    if (pipe(output_pipe) != 0) {
        throw std::runtime_error("cannot create URL download pipe");
    }
    const pid_t child = fork();
    if (child < 0) {
        close(output_pipe[0]);
        close(output_pipe[1]);
        throw std::runtime_error("cannot start URL downloader");
    }
    if (child == 0) {
        dup2(output_pipe[1], STDOUT_FILENO);
        close(output_pipe[0]);
        close(output_pipe[1]);
        execlp("curl", "curl", "--fail", "--location", "--silent",
               "--show-error", "--max-redirs", "5", "--max-time", "60",
               url.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }
    close(output_pipe[1]);
    std::string output;
    char buffer[16384];
    for (;;) {
        const ssize_t count = read(output_pipe[0], buffer, sizeof(buffer));
        if (count == 0) break;
        if (count < 0) {
            close(output_pipe[0]);
            waitpid(child, nullptr, 0);
            throw std::runtime_error("failed to read media URL");
        }
        output.append(buffer, static_cast<std::size_t>(count));
    }
    close(output_pipe[0]);
    int status = 0;
    if (waitpid(child, &status, 0) < 0 ||
        !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        throw std::runtime_error("failed to download media URL");
    }
    return output;
#endif
}

/// Owns the provider-local tokenizer loaded from the selected model package.
struct Tokenizer {
    explicit Tokenizer(const std::filesystem::path& path) {
        tokenizer = tokenizers::Tokenizer::FromBlobJSON(
            ReadFile(path / "tokenizer.json"));
        if (!tokenizer) throw std::runtime_error("cannot create tokenizer");
    }

    std::vector<int> Encode(const std::string& text) {
        return tokenizer->Encode(text);
    }

    std::string DecodeToken(int token) {
        return tokenizer->Decode({token});
    }

    std::unique_ptr<tokenizers::Tokenizer> tokenizer;
};

/// Preprocessing constants consumed by this provider's frontend and engine.
struct EngineConfig {
    int patch_size = 16;
    int pooling = 3;
    float rescale = 1.0f / 255.0f;
    float mean = 0.0f;
    float stddev = 1.0f;
    int audio_rate = 16000;
    int audio_kernel = 3;
    int audio_stride = 2;
    int audio_padding = 1;
    int image_budget = 280;
    int boi = 255999;
    int image = 258880;
    int eoi = 258882;
    int boa = 256000;
    int audio = 258881;
    int eoa = 258883;
};

template <class T>
T Value(const json& object, const char* key, T fallback) {
    const auto found = object.find(key);
    return found != object.end() && !found->is_null()
        ? found->get<T>() : fallback;
}

EngineConfig ParseEngineConfig(const json& config) {
    EngineConfig out;
    const json vision = config.value("vision_config", json::object());
    const json audio = config.value("audio_config", json::object());
    out.patch_size = Value(vision, "GEMMA4E_VISION_PATCH_SIZE", out.patch_size);
    out.pooling = Value(vision, "GEMMA4E_POOLING_KERNEL_SIZE", out.pooling);
    out.rescale = Value(
        vision, "GEMMA4E_VISION_RESCALE_FACTOR", out.rescale);
    out.mean = Value(vision, "GEMMA4E_VISION_IMAGE_MEAN", out.mean);
    out.stddev = Value(vision, "GEMMA4E_VISION_IMAGE_STD", out.stddev);
    out.audio_rate = Value(
        audio, "Gemma4E_Audio_audio_resample_rate", out.audio_rate);
    out.audio_kernel = Value(
        audio, "Gemma4E_conv2d_kernel_size", out.audio_kernel);
    out.audio_stride = Value(
        audio, "Gemma4E_conv2d_Stride", out.audio_stride);
    out.audio_padding = Value(
        audio, "Gemma4e_conv2d_Padding", out.audio_padding);
    out.image_budget = Value(
        config, "vision_soft_tokens_per_image", out.image_budget);
    out.boi = Value(config, "boi_token_id", out.boi);
    out.image = Value(config, "image_token_id", out.image);
    out.eoi = Value(config, "eoi_token_id", out.eoi);
    out.boa = Value(config, "boa_token_id", out.boa);
    out.audio = Value(config, "audio_token_id", out.audio);
    out.eoa = Value(config, "eoa_token_id", out.eoa);
    return out;
}

/// Owns token and multimodal views until the host releases a prepared request.
struct Prepared {
    std::vector<std::int32_t> tokens;
    gemma4e_multi_modal_payload_t payload;
    std::string prompt_tail;
    std::uint32_t first_payload_token_count = 0;
};

struct Cancelled : std::exception {
    const char* what() const noexcept override { return "cancelled"; }
};

/// Implements the stable frontend callbacks without calling FLM C++ objects.
class Frontend {
public:
    int Load(const flm_frontend_load_context_v1* context,
             flm_frontend_traits_v1* traits) {
        if (!context ||
            context->struct_size < FLM_FRONTEND_LOAD_CONTEXT_V1_MIN_SIZE ||
            !traits ||
            traits->struct_size < FLM_FRONTEND_TRAITS_V1_MIN_SIZE) {
            return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
        }
        model_path_ = String(context->model_path);
        const auto config = json::parse(
            ReadFile(model_path_ / "config.json"));
        const auto tokenizer_config = json::parse(
            ReadFile(model_path_ / "tokenizer_config.json"));
        config_ = ParseEngineConfig(config);
        vocabulary_size_ = Value(config, "vocab_size", 262144u);
        eos_.clear();
        const auto eos = tokenizer_config.at("eos_token_id");
        if (eos.is_array()) {
            for (const auto& value : eos) eos_.push_back(value.get<int>());
        } else {
            eos_.push_back(eos.get<int>());
        }
        has_bos_ = tokenizer_config.contains("bos_token_id") &&
                   tokenizer_config["bos_token_id"].is_number_integer();
        bos_ = has_bos_ ? tokenizer_config["bos_token_id"].get<int>() : -1;
        tokenizer_ = std::make_unique<Tokenizer>(model_path_);
        tokenizer_path_ = (model_path_ / "tokenizer.json").string();

        std::string source;
        const auto template_path = model_path_ / "chat_template.jinja";
        if (std::filesystem::exists(template_path)) {
            source = ReadFile(template_path);
        } else {
            source = tokenizer_config.at("chat_template").get<std::string>();
        }
        const std::string bos_token =
            tokenizer_config.value("bos_token", std::string());
        const std::string eos_token =
            tokenizer_config.at("eos_token").get<std::string>();
        chat_template_ = std::make_unique<minja::chat_template>(
            source, bos_token, eos_token);

        const std::size_t capacity = traits->struct_size;
        traits->struct_size = std::min(capacity, sizeof(*traits));
        traits->flags = CurrentFlags();
        traits->vocabulary_size = vocabulary_size_;
        traits->bos_token_id = bos_;
        traits->has_bos_token_id = has_bos_ ? 1u : 0u;
        traits->eos_token_ids = eos_.data();
        traits->eos_token_ids_size = eos_.size();
        traits->chat_format = {"gemma4", 6};
        traits->tokenizer_json_path = Span(tokenizer_path_);
        return FLM_MODEL_EXTENSION_STATUS_OK;
    }

    std::string Render(json messages, json tools, bool continued,
                       std::string* prompt_tail = nullptr) {
        NormalizeToolArguments(messages);
        minja::chat_template_options options;
        options.polyfill_tools = false;
        options.polyfill_tool_calls = false;
        options.polyfill_tool_responses = false;
        options.polyfill_object_arguments = false;

        minja::chat_template_inputs input;
        input.messages = messages;
        input.tools = tools;
        input.extra_context["enable_thinking"] = enable_thinking_;
        input.extra_context["conversational_mode"] = conversational_mode_;
        input.extra_context["user_system_prompt"] = system_prompt_;
        if (!continued) {
            input.add_generation_prompt = true;
            if (prompt_tail) prompt_tail->clear();
            return chat_template_->apply(input, options);
        }
        if (messages.empty() ||
            messages.back().value("role", "") != "assistant") {
            throw std::invalid_argument(
                "continue_final_message requires a final assistant message");
        }
        const auto render = [&](const json& value, bool generation_prompt) {
            auto copy = input;
            copy.messages = value;
            copy.add_generation_prompt = generation_prompt;
            return chat_template_->apply(copy, options);
        };
        chat_continued result = chat_continue_prompt(render, messages);
        if (prompt_tail) *prompt_tail = result.turn;
        return result.prompt;
    }

    Prepared* Prepare(flm_span_v1 request_json,
                      const flm_cancellation_v1* cancellation) {
        CheckCancelled(cancellation);
        const json request = json::parse(String(request_json));
        auto prepared = std::make_unique<Prepared>();
        json messages = request.value("messages", json::array());
        const json tools = request.value("tools", json::object());

        std::vector<std::pair<std::string, int>> images;
        std::vector<std::pair<std::string, int>> audios;
        if (!messages.empty()) {
            messages = NormalizeMessages(messages, images, audios);
        } else {
            const std::string prompt = request.value("prompt", std::string());
            if (prompt.empty()) {
                throw std::invalid_argument(
                    "request requires prompt or messages");
            }
            CollectTopLevel(
                request, "images", "image_payload_types", images);
            CollectTopLevel(
                request, "audios", "audio_payload_types", audios);
            json content = json::array();
            for (const auto& media : images) {
                content.push_back({{"type", "image"},
                                   {"image", media.first}});
            }
            for (const auto& media : audios) {
                content.push_back({{"type", "audio"},
                                   {"audio", media.first}});
            }
            content.push_back({{"type", "text"}, {"text", prompt}});
            messages.push_back({{"role", "user"}, {"content", content}});
        }

        for (const auto& image : images) {
            CheckCancelled(cancellation);
            ProcessImage(image.first, image.second,
                         prepared->payload.image_payload);
        }
        std::vector<audio_data_t> audio_inputs;
        std::vector<std::size_t> audio_chunk_counts;
        for (const auto& audio : audios) {
            CheckCancelled(cancellation);
            auto decoded = LoadAudio(audio.first, audio.second);
            auto chunks = ClipAudio(decoded, 30.0);
            audio_chunk_counts.push_back(chunks.size());
            audio_inputs.insert(
                audio_inputs.end(),
                std::make_move_iterator(chunks.begin()),
                std::make_move_iterator(chunks.end()));
        }
        ExpandAudioPlaceholders(messages, audio_chunk_counts);
        ProcessAudio(audio_inputs, prepared->payload.audio_payload);

        const bool continued =
            request.value("continue_final_message", false);
        const std::string prompt =
            Render(messages, tools, continued, &prepared->prompt_tail);
        std::vector<int> initial = tokenizer_->Encode(prompt);
        ExpandModalTokens(initial, *prepared);
        return prepared.release();
    }

    std::vector<std::int32_t> Encode(const std::string& text) {
        const auto encoded = tokenizer_->Encode(text);
        return {encoded.begin(), encoded.end()};
    }

    std::string Decode(int token) {
        return tokenizer_->DecodeToken(token);
    }

    int Configure(const std::string& name, const json& value) {
        if (name == "enable_think") {
            enable_thinking_ = value.get<bool>();
        } else if (name == "toggle_think") {
            enable_thinking_ = !enable_thinking_;
        } else if (name == "reasoning_effort") {
            const std::string effort = value.get<std::string>();
            if (effort != "none" && effort != "low" &&
                effort != "medium" && effort != "high") {
                throw std::invalid_argument("invalid reasoning_effort");
            }
            enable_thinking_ = effort != "none";
        } else if (name == "conversational_mode") {
            conversational_mode_ = value.get<bool>();
        } else if (name == "system_prompt") {
            system_prompt_ = value.get<std::string>();
        } else if (name == "image_max_tokens") {
            const int budget = value.get<int>();
            static constexpr int supported[] = {70, 140, 280, 560, 1120};
            if (std::find(std::begin(supported), std::end(supported), budget) ==
                std::end(supported)) {
                throw std::invalid_argument("unsupported image_max_tokens");
            }
            config_.image_budget = budget;
        } else {
            return FLM_MODEL_EXTENSION_STATUS_UNSUPPORTED;
        }
        return FLM_MODEL_EXTENSION_STATUS_OK;
    }

    std::uint32_t CurrentFlags() const {
        return FLM_FRONTEND_TRAIT_SUPPORTS_LOGPROBS |
               FLM_FRONTEND_TRAIT_SUPPORTS_CONTINUE_FINAL_MESSAGE |
               FLM_FRONTEND_TRAIT_TOOLS_ENABLED |
               (enable_thinking_
                    ? FLM_FRONTEND_TRAIT_THINKING_ENABLED : 0u);
    }

private:
    static void CheckCancelled(const flm_cancellation_v1* cancellation) {
        if (cancellation && cancellation->is_cancelled &&
            cancellation->is_cancelled(cancellation->context)) {
            throw Cancelled();
        }
    }

    static void NormalizeToolArguments(json& messages) {
        for (auto& message : messages) {
            if (!message.is_object() ||
                !message.contains("tool_calls")) continue;
            for (auto& call : message["tool_calls"]) {
                if (!call.contains("function") ||
                    !call["function"].contains("arguments")) continue;
                auto& arguments = call["function"]["arguments"];
                if (!arguments.is_string()) continue;
                const auto parsed = json::parse(
                    arguments.get<std::string>(), nullptr, false);
                if (parsed.is_object()) arguments = parsed;
            }
        }
    }

    static void CollectTopLevel(
        const json& request, const char* values_name, const char* types_name,
        std::vector<std::pair<std::string, int>>& output) {
        const json values = request.value(values_name, json::array());
        const json types = request.value(types_name, json::array());
        for (std::size_t index = 0; index < values.size(); ++index) {
            const int type = index < types.size() ? types[index].get<int>() : 0;
            output.emplace_back(values[index].get<std::string>(), type);
        }
    }

    static json NormalizeMessages(
        const json& input,
        std::vector<std::pair<std::string, int>>& images,
        std::vector<std::pair<std::string, int>>& audios) {
        json output = json::array();
        for (const auto& message : input) {
            if (!message.contains("images") && !message.contains("audios")) {
                output.push_back(message);
                continue;
            }
            json content = json::array();
            const json message_images =
                message.value("images", json::array());
            const json image_types =
                message.value("image_payload_types", json::array());
            for (std::size_t i = 0; i < message_images.size(); ++i) {
                const std::string value =
                    message_images[i].get<std::string>();
                const int type = i < image_types.size()
                    ? image_types[i].get<int>() : 1;
                images.emplace_back(value, type);
                content.push_back({{"type", "image"}, {"image", value}});
            }
            const json message_audios =
                message.value("audios", json::array());
            const json audio_types =
                message.value("audio_payload_types", json::array());
            for (std::size_t i = 0; i < message_audios.size(); ++i) {
                const std::string value =
                    message_audios[i].get<std::string>();
                const int type = i < audio_types.size()
                    ? audio_types[i].get<int>() : 1;
                audios.emplace_back(value, type);
                content.push_back({{"type", "audio"}, {"audio", value}});
            }
            content.push_back({
                {"type", "text"},
                {"text", message.value("content", std::string())}});
            output.push_back({
                {"role", message.value("role", "user")},
                {"content", content}});
        }
        return output;
    }

    static void ExpandAudioPlaceholders(
        json& messages, const std::vector<std::size_t>& chunk_counts) {
        std::size_t audio = 0;
        for (auto& message : messages) {
            if (!message.contains("content") ||
                !message["content"].is_array()) {
                continue;
            }
            json content = json::array();
            for (const auto& item : message["content"]) {
                if (!item.is_object() ||
                    item.value("type", std::string()) != "audio") {
                    content.push_back(item);
                    continue;
                }
                if (audio >= chunk_counts.size()) {
                    throw std::invalid_argument(
                        "audio placeholder/payload mismatch");
                }
                for (std::size_t chunk = 0;
                     chunk < chunk_counts[audio]; ++chunk) {
                    content.push_back(item);
                }
                ++audio;
            }
            message["content"] = std::move(content);
        }
        if (audio != chunk_counts.size()) {
            throw std::invalid_argument(
                "audio placeholder/payload mismatch");
        }
    }

    std::string MediaBytes(const std::string& value, int type) {
        if (type == 0) return ReadFile(value);
        if (type == 1) {
            const auto comma = value.find(',');
            return base64::from_base64(
                comma == std::string::npos ? value : value.substr(comma + 1));
        }
        if (type == 2) return Download(value);
        throw std::invalid_argument("unknown media payload type");
    }

    void ProcessImage(
        const std::string& value, int type,
        gemma4e_image_payload_t& payload) {
        image_data_t decoded;
        const bool loaded = type == 0
            ? image_reader_.load_image(value, decoded)
            : image_reader_.load_image_bytes(MediaBytes(value, type), decoded);
        if (!loaded) throw std::invalid_argument("cannot decode image");
        image_data_t planar;
        if (!image_reader_.reorder_hwc_to_chw(decoded, planar)) {
            image_reader_.recycle(decoded);
            throw std::invalid_argument("cannot reorder image");
        }
        image_reader_.recycle(decoded);

        const int side = config_.pooling * config_.patch_size;
        const int max_patches =
            config_.image_budget * config_.pooling * config_.pooling;
        const double factor = std::sqrt(
            static_cast<double>(max_patches * config_.patch_size *
                                config_.patch_size) /
            (planar.width * planar.height));
        int height = static_cast<int>(
            std::floor(factor * planar.height / side)) * side;
        int width = static_cast<int>(
            std::floor(factor * planar.width / side)) * side;
        const int max_side = max_patches /
            (config_.pooling * config_.pooling) * side;
        if (height == 0) {
            height = side;
            width = std::min(
                static_cast<int>(std::floor(
                    static_cast<double>(planar.width) / planar.height)) * side,
                max_side);
        }
        if (width == 0) {
            width = side;
            height = std::min(
                static_cast<int>(std::floor(
                    static_cast<double>(planar.height) / planar.width)) * side,
                max_side);
        }
        if (height <= 0 || width <= 0 ||
            height * width >
                max_patches * config_.patch_size * config_.patch_size) {
            image_reader_.recycle(planar);
            throw std::invalid_argument("invalid resized image dimensions");
        }
        std::vector<std::uint8_t> resized;
        if (height == planar.height && width == planar.width) {
            resized.assign(
                planar.pixels.data(),
                planar.pixels.data() + 3 * height * width);
        } else {
            resized = imgproc::resize_bicubic_antialias_rgb_planar_optimized(
                planar.pixels.data(), planar.width, planar.height,
                width, height, true);
        }
        image_reader_.recycle(planar);

        std::vector<float> normalized(
            static_cast<std::size_t>(3) * height * width);
        imgproc::rescale_and_normalize_optimized(
            resized.data(), normalized.data(), width, height, 3, true,
            config_.rescale, false, config_.mean, config_.stddev);
        const int patch_height = height / config_.patch_size;
        const int patch_width = width / config_.patch_size;
        const int patch_count = patch_height * patch_width;
        const int elements = config_.patch_size * config_.patch_size * 3;
        std::vector<bf16> pixels(
            static_cast<std::size_t>(max_patches) * elements,
            static_cast<bf16>(0.0f));
        for (int ph = 0; ph < patch_height; ++ph) {
            for (int pw = 0; pw < patch_width; ++pw) {
                const int patch = ph * patch_width + pw;
                for (int y = 0; y < config_.patch_size; ++y) {
                    for (int x = 0; x < config_.patch_size; ++x) {
                        for (int c = 0; c < 3; ++c) {
                            const int source =
                                (c * height + ph * config_.patch_size + y) *
                                width + pw * config_.patch_size + x;
                            const int target =
                                ((patch * config_.patch_size + y) *
                                 config_.patch_size + x) * 3 + c;
                            pixels[target] =
                                static_cast<bf16>(normalized[source]);
                        }
                    }
                }
            }
        }
        std::vector<int> grid(
            static_cast<std::size_t>(max_patches) * 2, -1);
        for (int ph = 0; ph < patch_height; ++ph) {
            for (int pw = 0; pw < patch_width; ++pw) {
                const int patch = ph * patch_width + pw;
                grid[patch * 2] = pw;
                grid[patch * 2 + 1] = ph;
            }
        }
        payload.image_patch__element_per_patch.emplace_back(
            max_patches, elements);
        payload.valid_patch_size_per_image.push_back(patch_count);
        payload.pixel_values.push_back(std::move(pixels));
        payload.image_grid_pairs_per_image.push_back(std::move(grid));
        payload.num_soft_tokens_per_image.push_back(
            patch_count / (config_.pooling * config_.pooling));
        payload.num_images++;
    }

    audio_data_t LoadAudio(const std::string& value, int type) {
        audio_data_t output;
        const bool loaded = type == 0
            ? audio_reader_.load_audio(
                value, output, config_.audio_rate, MonoDownmixMode::MEAN)
            : [&] {
                const std::string bytes = MediaBytes(value, type);
                return audio_reader_.load_audio_from_memory(
                    reinterpret_cast<const std::uint8_t*>(bytes.data()),
                    bytes.size(), output, config_.audio_rate,
                    MonoDownmixMode::MEAN);
              }();
        if (!loaded || output.channels != 1) {
            throw std::invalid_argument("cannot decode mono audio");
        }
        return output;
    }

    static std::vector<audio_data_t> ClipAudio(
        const audio_data_t& audio, double seconds) {
        std::vector<audio_data_t> chunks;
        const std::size_t frames =
            static_cast<std::size_t>(seconds * audio.sample_rate);
        for (std::size_t begin = 0; begin < audio.num_frames;
             begin += frames) {
            const std::size_t end =
                std::min(begin + frames, audio.num_frames);
            audio_data_t chunk;
            chunk.sample_rate = audio.sample_rate;
            chunk.original_sample_rate = audio.original_sample_rate;
            chunk.channels = audio.channels;
            chunk.original_channels = audio.original_channels;
            chunk.num_frames = end - begin;
            chunk.num_samples = chunk.num_frames;
            chunk.duration_seconds =
                static_cast<double>(chunk.num_frames) / chunk.sample_rate;
            chunk.samples.assign(
                audio.samples.begin() + begin, audio.samples.begin() + end);
            chunks.push_back(std::move(chunk));
        }
        return chunks;
    }

    void ProcessAudio(
        std::vector<audio_data_t>& inputs,
        gemma4e_audio_payload_t& payload) {
        payload.num_audios = static_cast<unsigned>(inputs.size());
        payload.mel_spectrograms.resize(inputs.size());
        payload.mel_spectrogram_frames_per_audio.resize(inputs.size());
        payload.mel_spectrogram_bins_per_audio.resize(inputs.size());
        constexpr int bins = 128;
        for (std::size_t index = 0; index < inputs.size(); ++index) {
            auto& input = inputs[index];
            const int frame_length =
                static_cast<int>(std::round(input.sample_rate * 0.020));
            const int hop = static_cast<int>(
                std::round(input.sample_rate * 0.010));
            int fft = 1;
            while (fft < frame_length) fft <<= 1;
            const int frequency_bins = fft / 2 + 1;
            const auto window = audioproc::window_function_optimized(
                frame_length, "hann", true);
            const auto filters = audioproc::mel_filter_bank_optimized(
                frequency_bins, bins, 0.0f, 8000.0f, input.sample_rate,
                false, false);
            const int left = frame_length / 2;
            std::vector<float> waveform(
                input.num_frames + left, 0.0f);
            std::copy(
                input.samples.begin(), input.samples.end(),
                waveform.begin() + left);
            const int frames =
                (static_cast<int>(waveform.size()) - frame_length - 1) /
                    hop + 1;
            if (frames <= 0) {
                throw std::invalid_argument("audio clip is too short");
            }
            std::vector<float> windowed(
                static_cast<std::size_t>(frames) * frame_length);
            audioproc::apply_window_frames_optimized(
                waveform.data(), window.data(), windowed.data(),
                frames, frame_length, hop);
            std::vector<float> magnitude(
                static_cast<std::size_t>(frames) * frequency_bins);
            audioproc::rfft_magnitude_batch_optimized(
                windowed.data(), magnitude.data(), frames,
                frame_length, fft);
            std::vector<float> mel(
                static_cast<std::size_t>(frames) * bins);
            audioproc::mel_spectrogram_optimized(
                magnitude.data(), filters.data(), mel.data(),
                frames, frequency_bins, bins);
            std::vector<float> log_mel(mel.size());
            audioproc::log_mel_floor_optimized(
                mel.data(), log_mel.data(), log_mel.size(), 1e-3f);
            auto& output = payload.mel_spectrograms[index];
            output.resize(log_mel.size());
            std::transform(
                log_mel.begin(), log_mel.end(), output.begin(),
                [](float value) { return static_cast<bf16>(value); });
            payload.mel_spectrogram_frames_per_audio[index] = frames;
            payload.mel_spectrogram_bins_per_audio[index] = bins;
            int length = frames;
            for (int layer = 0; layer < 2; ++layer) {
                length =
                    (length + 2 * config_.audio_padding -
                     config_.audio_kernel) /
                        config_.audio_stride + 1;
            }
            payload.num_soft_tokens_per_audio.push_back(length);
        }
    }

    void ExpandModalTokens(
        const std::vector<int>& input, Prepared& prepared) {
        std::size_t image = 0;
        std::size_t audio = 0;
        for (int token : input) {
            if (token == config_.image) {
                if (image >=
                    prepared.payload.image_payload
                        .num_soft_tokens_per_image.size()) {
                    throw std::invalid_argument(
                        "image placeholder/payload mismatch");
                }
                prepared.tokens.push_back(config_.boi);
                prepared.tokens.insert(
                    prepared.tokens.end(),
                    prepared.payload.image_payload
                        .num_soft_tokens_per_image[image],
                    config_.image);
                prepared.tokens.push_back(config_.eoi);
                prepared.first_payload_token_count =
                    static_cast<std::uint32_t>(prepared.tokens.size());
                ++image;
            } else if (token == config_.audio) {
                if (audio >=
                    prepared.payload.audio_payload
                        .num_soft_tokens_per_audio.size()) {
                    throw std::invalid_argument(
                        "audio placeholder/payload mismatch");
                }
                prepared.tokens.push_back(config_.boa);
                prepared.tokens.insert(
                    prepared.tokens.end(),
                    prepared.payload.audio_payload
                        .num_soft_tokens_per_audio[audio],
                    config_.audio);
                prepared.tokens.push_back(config_.eoa);
                prepared.first_payload_token_count =
                    static_cast<std::uint32_t>(prepared.tokens.size());
                ++audio;
            } else {
                prepared.tokens.push_back(token);
            }
        }
        if (image != prepared.payload.image_payload.num_images ||
            audio != prepared.payload.audio_payload.num_audios) {
            throw std::invalid_argument(
                "chat template did not emit all media placeholders");
        }
    }

    std::filesystem::path model_path_;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<minja::chat_template> chat_template_;
    ImageReader image_reader_;
    AudioReader audio_reader_;
    EngineConfig config_;
    std::vector<int> eos_;
    std::string tokenizer_path_;
    std::uint32_t vocabulary_size_ = 0;
    int bos_ = -1;
    bool has_bos_ = false;
    bool enable_thinking_ = false;
    bool conversational_mode_ = false;
    std::string system_prompt_;
};

int Load(void* object, const flm_frontend_load_context_v1* context,
         flm_frontend_traits_v1* traits, const flm_error_sink_v1* error) {
    if (!object) return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    return Guard(error, [&] {
        const int status =
            static_cast<Frontend*>(object)->Load(context, traits);
        if (status != FLM_MODEL_EXTENSION_STATUS_OK) {
            throw std::invalid_argument("invalid frontend load context");
        }
    });
}

int Prepare(void* object, flm_span_v1 request_json,
            const flm_cancellation_v1* cancellation,
            flm_prepared_request_v1* request,
            const flm_error_sink_v1* error) {
    if (!object || !request ||
        request->struct_size < FLM_PREPARED_REQUEST_V1_MIN_SIZE) {
        return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    }
    try {
        std::unique_ptr<Prepared> prepared(
            static_cast<Frontend*>(object)->Prepare(
                request_json, cancellation));
        const std::size_t capacity = request->struct_size;
        request->struct_size = std::min(capacity, sizeof(*request));
        request->tokens = {
            prepared->tokens.data(), prepared->tokens.size()};
        const bool has_payload =
            prepared->payload.image_payload.num_images != 0 ||
            prepared->payload.audio_payload.num_audios != 0;
        request->payload = {
            sizeof(flm_prefill_payload_v1), {}, &prepared->payload};
        request->has_payload = has_payload ? 1u : 0u;
        request->first_payload_token_count =
            has_payload ? prepared->first_payload_token_count : 0u;
        request->release_context = prepared.release();
        request->release = [](void* value) {
            delete static_cast<Prepared*>(value);
        };
        if (capacity >= FLM_PREPARED_REQUEST_V1_PROMPT_TAIL_SIZE) {
            auto* value = static_cast<Prepared*>(
                request->release_context);
            request->prompt_tail = Span(value->prompt_tail);
        }
        return FLM_MODEL_EXTENSION_STATUS_OK;
    } catch (const Cancelled&) {
        return FLM_MODEL_EXTENSION_STATUS_CANCELLED;
    } catch (const std::invalid_argument& exception) {
        SetError(error, exception.what());
        return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& exception) {
        SetError(error, exception.what());
        return FLM_MODEL_EXTENSION_STATUS_ERROR;
    } catch (...) {
        SetError(error, "unknown Gemma4-open prepare exception");
        return FLM_MODEL_EXTENSION_STATUS_ERROR;
    }
}

int Encode(void* object, flm_span_v1 text, flm_i32_view_v1* output,
           const flm_error_sink_v1* error) {
    if (!object || !output)
        return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    return Guard(error, [&] {
        thread_local std::vector<std::int32_t> tokens;
        tokens = static_cast<Frontend*>(object)->Encode(String(text));
        *output = {tokens.data(), tokens.size()};
    });
}

int Decode(void* object, std::int32_t token, flm_span_v1* output,
           const flm_error_sink_v1* error) {
    if (!object || !output)
        return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    return Guard(error, [&] {
        thread_local std::string text;
        text = static_cast<Frontend*>(object)->Decode(token);
        *output = Span(text);
    });
}

int ApplyTemplate(void* object, flm_span_v1 messages, flm_span_v1 tools,
                  std::uint32_t continued,
                  flm_chat_template_result_v1* result,
                  const flm_error_sink_v1* error) {
    if (!object || !result ||
        result->struct_size < FLM_CHAT_TEMPLATE_RESULT_V1_MIN_SIZE) {
        return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    }
    return Guard(error, [&] {
        thread_local std::string text;
        thread_local std::string tail;
        text = static_cast<Frontend*>(object)->Render(
            json::parse(String(messages)), json::parse(String(tools)),
            continued != 0, &tail);
        const std::size_t capacity = result->struct_size;
        result->struct_size = std::min(capacity, sizeof(*result));
        result->text = Span(text);
        result->prompt_tail = Span(tail);
    });
}

int Configure(void* object, flm_span_v1 name, flm_span_v1 value,
              const flm_error_sink_v1* error) {
    if (!object) return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    try {
        return static_cast<Frontend*>(object)->Configure(
            String(name), json::parse(String(value)));
    } catch (const std::exception& exception) {
        SetError(error, exception.what());
        return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    }
}

int QueryFlags(void* object, std::uint32_t* flags,
               const flm_error_sink_v1*) {
    if (!object || !flags)
        return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    *flags = static_cast<Frontend*>(object)->CurrentFlags();
    return FLM_MODEL_EXTENSION_STATUS_OK;
}

}  // namespace

int CreateFrontend(
    void*, const flm_frontend_create_context_v1* context,
    flm_model_frontend_v1* frontend, const flm_error_sink_v1* error) {
    if (!context ||
        context->struct_size < FLM_FRONTEND_CREATE_CONTEXT_V1_MIN_SIZE ||
        !frontend ||
        frontend->struct_size < FLM_MODEL_FRONTEND_V1_MIN_SIZE) {
        return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    }
    return Guard(error, [&] {
        const std::size_t capacity = frontend->struct_size;
        std::unique_ptr<Frontend> object(new Frontend);
        frontend->struct_size = std::min(capacity, sizeof(*frontend));
        frontend->object = object.release();
        frontend->destroy = [](void* value) {
            delete static_cast<Frontend*>(value);
        };
        frontend->load = Load;
        frontend->prepare = Prepare;
        frontend->encode = Encode;
        frontend->decode_token = Decode;
        frontend->apply_chat_template = ApplyTemplate;
        if (capacity >= FLM_MODEL_FRONTEND_V1_CONFIGURE_SIZE) {
            frontend->configure = Configure;
        }
        if (capacity >= FLM_MODEL_FRONTEND_V1_QUERY_FLAGS_SIZE) {
            frontend->query_flags = QueryFlags;
        }
    });
}

}  // namespace gemma4_open
