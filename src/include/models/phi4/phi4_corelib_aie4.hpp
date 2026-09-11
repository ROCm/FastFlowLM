#pragma once

#include "causal_lm.hpp"
#include "corelib/corelib_runtime.hpp"
#include "lm_config.hpp"
#include "models/phi4/phi4_corelib_gguf.hpp"

#include <cstdint>
#include <memory>

namespace flm::phi4 {

class phi4_corelib_aie4 final : public causal_lm {
public:
    phi4_corelib_aie4(
        LM_Config config,
        std::shared_ptr<Phi4GgufPackage> package,
        std::shared_ptr<corelib::CorelibRuntime> runtime,
        std::uint32_t max_length = 4096);
    ~phi4_corelib_aie4() override;

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

}  // namespace flm::phi4
