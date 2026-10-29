/// \file model_extension_frontend.cpp
/// \brief Adapts stable-C frontend providers to AutoModel
#include "AutoModel/model_extension_frontend.hpp"

#include "AutoModel/automodel.hpp"
#include "chat/chat_format.hpp"
#include "model_extension_detail.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace flm::frontend {
namespace {

using flm::extension::detail::copy_span;
using flm::extension::detail::destroy_provider_object;
using flm::extension::detail::ErrorCapture;
using flm::extension::detail::native_runtime_device_handle;
using flm::extension::detail::span;

void Check(const char *operation, int status, const ErrorCapture &error) {
  if (status == FLM_MODEL_EXTENSION_STATUS_OK)
    return;
  throw std::runtime_error(
      "Extension frontend " + std::string(operation) + " failed with status " +
      std::to_string(status) +
      (error.message.empty() ? std::string() : ": " + error.message));
}

class PreparedRequest {
public:
  PreparedRequest() { request_.struct_size = sizeof(request_); }
  ~PreparedRequest() {
    if (!request_.release)
      return;
    try {
      request_.release(request_.release_context);
    } catch (...) {
    }
  }
  PreparedRequest(const PreparedRequest &) = delete;
  PreparedRequest &operator=(const PreparedRequest &) = delete;
  flm_prepared_request_v1 &get() { return request_; }

private:
  flm_prepared_request_v1 request_{};
};

nlohmann::ordered_json RequestJson(const lm_uniform_input_t &input,
                                   bool continue_final_message) {
  nlohmann::ordered_json request{
      {"prompt", input.prompt},
      {"messages", input.messages},
      {"images", input.images},
      {"image_payload_types", input.image_payload_types},
      {"audios", input.audios},
      {"audio_payload_types", input.audio_payload_types},
      {"tools", input.tools},
      {"continue_final_message", continue_final_message},
  };
  if (input.requested_max_new_tokens) {
    request["requested_max_new_tokens"] = *input.requested_max_new_tokens;
  }
  return request;
}

/// Host-side generation adapter; model-specific preparation remains entirely
/// inside the provider and crosses the boundary only as an opaque payload.
class ExtensionAutoModel final : public AutoModel {
public:
  ExtensionAutoModel(flm_rt::device *device, std::string provider,
                     flm_model_frontend_v1 instance,
                     std::shared_ptr<void> lifetime)
      : AutoModel(device, "extension"), provider_(std::move(provider)),
        object_(instance.object), destroy_(instance.destroy),
        lifetime_(std::move(lifetime)) {
    std::memcpy(&api_, &instance, std::min(instance.struct_size, sizeof(api_)));
  }

  ~ExtensionAutoModel() override {
    destroy_provider_object(destroy_, object_);
  }

  void load_model(std::string path, json model_info, int context_length,
                  bool preemption, const std::string &backend) override {
    _shared_load_backend(path, model_info, context_length, preemption, backend,
                         nullptr, false, provider_);
    try {
      const std::string info = model_info.dump();
      model_info_ = info;
      flm_frontend_traits_v1 traits{};
      traits.struct_size = sizeof(traits);
      const std::string selected_backend = backend_id();
      const flm_frontend_load_context_v1 context{sizeof(context),
                                                 span(path),
                                                 span(info),
                                                 span(selected_backend),
                                                 span(backend_provider_),
                                                 MAX_L,
                                                 preemption ? 1u : 0u};
      ErrorCapture error;
      auto sink = error.sink();
      Check("load", api_.load(object_, &context, &traits, &sink), error);
      if (traits.struct_size < FLM_FRONTEND_TRAITS_V1_MIN_SIZE ||
          traits.vocabulary_size == 0 ||
          (traits.eos_token_ids_size != 0 && !traits.eos_token_ids)) {
        throw std::runtime_error(
            "Extension frontend returned incomplete traits");
      }
      traits_ = traits;
      eos_token_ids.clear();
      if (traits.eos_token_ids_size != 0) {
        eos_token_ids.assign(traits.eos_token_ids,
                             traits.eos_token_ids + traits.eos_token_ids_size);
      }
      if (const auto forced = backend_->forced_eos_ids();
          forced && *forced != eos_token_ids) {
        throw std::runtime_error(
            "Extension frontend EOS ids do not match its backend provider");
      }
      has_bos_token = traits.has_bos_token_id != 0;
      bos_token_id = has_bos_token ? traits.bos_token_id : -1;
      if (const auto forced = backend_->forced_bos_id();
          forced && (!has_bos_token || *forced != bos_token_id)) {
        throw std::runtime_error(
            "Extension frontend BOS id does not match its backend provider");
      }
      single_turn = (traits.flags & FLM_FRONTEND_TRAIT_SINGLE_TURN) != 0;
      chat_format_kind format;
      if (!chat_format_from_name(copy_span(traits.chat_format), format)) {
        throw std::runtime_error(
            "Extension frontend returned unknown chat format '" +
            copy_span(traits.chat_format) + "'");
      }
      chat_format = chat_format_make(format, thinking_enabled());
      sampler_config config;
      sampler = std::make_unique<Sampler>(
          static_cast<int>(traits.vocabulary_size), config);
    } catch (...) {
      lm_engine = nullptr;
      backend_.reset();
      backend_provider_.clear();
      sampler.reset();
      is_model_loaded = false;
      throw;
    }
  }

  void set_sampler(sampler_config &config) override {
    if (traits_.vocabulary_size == 0) {
      throw std::runtime_error("Extension frontend is not loaded");
    }
    sampler = std::make_unique<Sampler>(
        static_cast<int>(traits_.vocabulary_size), config);
  }

  std::string show_model_info() override { return model_info_; }

  bool insert(chat_meta_info_t &meta, lm_uniform_input_t &input,
              std::function<bool()> cancelled) override {
    const std::string request_json =
        RequestJson(input, meta.continue_final_message).dump();
    PreparedRequest prepared;
    auto &output = prepared.get();
    ErrorCapture error;
    auto sink = error.sink();
    const flm_cancellation_v1 cancellation{
        &cancelled, [](void *context) noexcept {
          try {
            return (*static_cast<std::function<bool()> *>(context))() ? 1 : 0;
          } catch (...) {
            return 1;
          }
        }};
    const int status = api_.prepare(object_, span(request_json), &cancellation,
                                    &output, &sink);
    if (status == FLM_MODEL_EXTENSION_STATUS_CANCELLED) {
      meta.stop_reason = CANCEL_DETECTED;
      return false;
    }
    Check("prepare", status, error);
    if (output.struct_size < FLM_PREPARED_REQUEST_V1_MIN_SIZE ||
        !output.release || (!output.tokens.data && output.tokens.size != 0) ||
        output.tokens.size >
            static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        output.first_payload_token_count > output.tokens.size ||
        (output.has_payload &&
         (output.payload.struct_size < FLM_PREFILL_PAYLOAD_V1_MIN_SIZE ||
          !output.payload.object))) {
      throw std::runtime_error(
          "Extension frontend returned an incomplete prepared request");
    }
    if (meta.continue_final_message) {
      if (output.struct_size < FLM_PREPARED_REQUEST_V1_PROMPT_TAIL_SIZE) {
        throw std::runtime_error(
            "Extension frontend did not return a continuation prompt tail");
      }
      continued_turn_ = copy_span(output.prompt_tail);
    }
    if (output.has_payload) {
      output.payload.provider = span(provider_);
      // Opaque attachments have no host-visible cache identity. Refill
      // the complete rendered request rather than reusing KV state that
      // may have been produced from different private data.
      clear_context();
    }
    std::vector<int> tokens;
    static_assert(sizeof(int) == sizeof(std::int32_t));
    if (output.tokens.size != 0) {
      tokens.assign(output.tokens.data,
                    output.tokens.data + output.tokens.size);
    }
    return _shared_insert(meta, tokens, std::move(cancelled),
                          output.has_payload ? &output.payload : nullptr,
                          static_cast<int>(output.first_payload_token_count),
                          input.requested_max_new_tokens);
  }

  std::string generate(chat_meta_info_t &meta, int limit, std::ostream &output,
                       std::function<bool()> cancelled) override {
    return _shared_generate(meta, limit, output, std::move(cancelled));
  }

  std::string generate_with_prompt(chat_meta_info_t &meta,
                                   lm_uniform_input_t &input, int limit,
                                   std::ostream &output) override {
    if (!insert(meta, input, [] { return false; }))
      return {};
    return generate(meta, limit, output, [] { return false; });
  }

  std::string apply_chat_template(nlohmann::ordered_json &messages,
                                  nlohmann::ordered_json tools,
                                  bool continue_final_message) override {
    const std::string messages_json = messages.dump();
    const std::string tools_json = tools.dump();
    flm_chat_template_result_v1 result{};
    result.struct_size = sizeof(result);
    ErrorCapture error;
    auto sink = error.sink();
    Check("apply_chat_template",
          api_.apply_chat_template(
              object_, span(messages_json), span(tools_json),
              continue_final_message ? 1u : 0u, &result, &sink),
          error);
    if (result.struct_size < FLM_CHAT_TEMPLATE_RESULT_V1_MIN_SIZE) {
      throw std::runtime_error(
          "Extension frontend returned an incomplete chat template result");
    }
    if (continue_final_message) {
      continued_turn_ = copy_span(result.prompt_tail);
    }
    return copy_span(result.text);
  }

  bool configure_parameter(std::string name, const std::any &value) override {
    if (!api_.configure) {
      return AutoModel::configure_parameter(std::move(name), value);
    }
    nlohmann::ordered_json stable_value;
    if (!value.has_value()) {
      stable_value = nullptr;
    } else if (value.type() == typeid(std::string)) {
      stable_value = std::any_cast<const std::string &>(value);
    } else if (value.type() == typeid(bool)) {
      stable_value = std::any_cast<bool>(value);
    } else if (value.type() == typeid(int)) {
      stable_value = std::any_cast<int>(value);
    } else if (value.type() == typeid(float)) {
      stable_value = std::any_cast<float>(value);
    } else if (value.type() == typeid(double)) {
      stable_value = std::any_cast<double>(value);
    } else {
      return false;
    }
    const std::string encoded = stable_value.dump();
    ErrorCapture error;
    auto sink = error.sink();
    const int status =
        api_.configure(object_, span(name), span(encoded), &sink);
    if (status == FLM_MODEL_EXTENSION_STATUS_UNSUPPORTED)
      return false;
    Check("configure", status, error);
    if (is_model_loaded) {
      auto updated =
          chat_format_make(chat_format.kind, thinking_enabled());
      if (updated.reasoning_start != chat_format.reasoning_start ||
          updated.reasoning_end != chat_format.reasoning_end) {
        grammar_engine_.reset();
      }
      chat_format = std::move(updated);
    }
    return true;
  }

  std::string token_text(int token) override {
    flm_span_v1 output{};
    ErrorCapture error;
    auto sink = error.sink();
    Check("decode_token", api_.decode_token(object_, token, &output, &sink),
          error);
    return copy_span(output);
  }

  std::vector<int> encode_text(const std::string &text) override {
    flm_i32_view_v1 output{};
    ErrorCapture error;
    auto sink = error.sink();
    Check("encode", api_.encode(object_, span(text), &output, &sink), error);
    if (!output.data && output.size != 0) {
      throw std::runtime_error(
          "Extension frontend encode returned a null token view");
    }
    if (output.size == 0)
      return {};
    return {output.data, output.data + output.size};
  }

  std::string tokenizer_json_path() const override {
    return copy_span(traits_.tokenizer_json_path);
  }

  std::pair<std::string, std::vector<int>> get_history() override {
    std::string text;
    for (int token : token_history)
      text += token_text(token);
    return {std::move(text), token_history};
  }

  bool supports_logprobs() const override {
    return (CurrentFlags() & FLM_FRONTEND_TRAIT_SUPPORTS_LOGPROBS) != 0;
  }
  bool supports_continue_final_message() const override {
    return (CurrentFlags() &
            FLM_FRONTEND_TRAIT_SUPPORTS_CONTINUE_FINAL_MESSAGE) != 0;
  }
  bool thinking_enabled() const override {
    return (CurrentFlags() & FLM_FRONTEND_TRAIT_THINKING_ENABLED) != 0;
  }
  bool tools_enabled() const override {
    return (CurrentFlags() & FLM_FRONTEND_TRAIT_TOOLS_ENABLED) != 0;
  }

private:
  std::uint32_t CurrentFlags() const {
    if (!api_.query_flags)
      return traits_.flags;
    std::uint32_t flags = 0;
    ErrorCapture error;
    auto sink = error.sink();
    Check("query_flags", api_.query_flags(object_, &flags, &sink), error);
    return flags;
  }

  std::string provider_;
  void *object_ = nullptr;
  void (*destroy_)(void *) = nullptr;
  flm_model_frontend_v1 api_{};
  flm_frontend_traits_v1 traits_{};
  std::string model_info_;
  std::shared_ptr<void> lifetime_;
};

} // namespace

FrontendFactory make_extension_frontend_factory(
    ExtensionFrontendRegistration registration, std::string extension_root,
    std::string asset_root, std::shared_ptr<void> provider_lifetime) {
  return [registration = std::move(registration),
          extension_root = std::move(extension_root),
          asset_root = std::move(asset_root),
          provider_lifetime = std::move(provider_lifetime)](void *raw_device) {
    auto *device = static_cast<flm_rt::device *>(raw_device);
    const flm_frontend_create_context_v1 context{
        sizeof(context),
        span(extension_root),
        span(asset_root),
        native_runtime_device_handle(device),
#if defined(FLM_USE_HRX)
        FLM_MODEL_EXTENSION_NPU_RUNTIME_HRX,
#else
        FLM_MODEL_EXTENSION_NPU_RUNTIME_XRT,
#endif
    };
    flm_model_frontend_v1 instance{};
    instance.struct_size = sizeof(instance);
    ErrorCapture error;
    auto sink = error.sink();
    int status = FLM_MODEL_EXTENSION_STATUS_ERROR;
    try {
      status = registration.create(registration.factory_context, &context,
                                   &instance, &sink);
    } catch (...) {
      destroy_provider_object(instance.destroy, instance.object);
      throw std::runtime_error("Extension frontend factory allowed an "
                               "exception across its C boundary");
    }
    if (status != FLM_MODEL_EXTENSION_STATUS_OK) {
      destroy_provider_object(instance.destroy, instance.object);
      Check("factory", status, error);
    }
    if (instance.struct_size < FLM_MODEL_FRONTEND_V1_MIN_SIZE ||
        !instance.object || !instance.destroy || !instance.load ||
        !instance.prepare || !instance.encode || !instance.decode_token ||
        !instance.apply_chat_template) {
      destroy_provider_object(instance.destroy, instance.object);
      throw std::runtime_error(
          "Extension frontend factory returned an incomplete frontend");
    }
    auto frontend = std::make_unique<ExtensionAutoModel>(
        device, registration.provider, instance, provider_lifetime);
    return own_frontend(std::move(frontend));
  };
}

} // namespace flm::frontend
