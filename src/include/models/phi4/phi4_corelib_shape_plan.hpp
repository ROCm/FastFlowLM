#pragma once

#include "corelib/corelib_api.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace flm::phi4 {

struct Phi4RowExtents {
    std::int64_t query_rows;
    std::int64_t kv_rows;
    std::int64_t output_rows;
    std::int64_t ssmlp_rows;
    std::int64_t rmsnorm_rows;
    std::int64_t flat_mha_rows;
};

class Phi4ShapePlan final {
public:
    static Phi4ShapePlan Build(
        const std::shared_ptr<const corelib::CorelibApi>& api);
    const Phi4RowExtents& ForRows(std::size_t live_rows) const;
    const Phi4RowExtents& maximum_extents() const noexcept;
    const ryzenai_corelib_flat_mha_bf16_desc& attention_desc() const noexcept;
    const ryzenai_corelib_matmul_bf16_weights_desc& lm_head_desc() const noexcept;

private:
    std::vector<Phi4RowExtents> rows_;
    Phi4RowExtents maximum_extents_{};
    ryzenai_corelib_flat_mha_bf16_desc attention_desc_{};
    ryzenai_corelib_matmul_bf16_weights_desc lm_head_desc_{};
};

}  // namespace flm::phi4
