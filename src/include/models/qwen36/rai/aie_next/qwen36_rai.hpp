/// \file qwen36_rai.hpp
/// \brief Qwen3.6-35B-A3B text on ryzenai-corelib, from a Q8_0 GGUF
#pragma once

#include "causal_lm.hpp"
#include "rai/corelib_runtime.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>

namespace flm::qwen36 {

inline constexpr std::uint32_t kContextLimit = 262144;

/// \brief Qwen3.6 text through corelib's model API
/// \note Vision is not this engine. A prompt is one forward(); corelib runs
///       its first 4096 rows from position 0 as one pass and steps the rest.
///       The KV cache is sized once, at the capacity `max_length` rounds up
///       to in whole 4096-row pages.
class qwen36_rai final : public causal_lm {
public:
    qwen36_rai(std::filesystem::path gguf,
               std::shared_ptr<corelib::CorelibRuntime> runtime,
               std::uint32_t max_length = 4096);
    ~qwen36_rai() override;

    buffer<bf16> forward(int id) override;
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr) override;
    void set_context_length(int length) override;
    void load_weights(Q4NX&) override;
    void update_max_length(std::uint32_t max_length) override;
    void clear_context() override;
    buffer<bf16> get_k_cache(int layer, int index) override;
    buffer<bf16> get_v_cache(int layer, int index) override;
    int get_current_context_length() override;
    int checkpoint() override;
    int restore() override;
    bool poisoned() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace flm::qwen36
