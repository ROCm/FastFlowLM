#pragma once

#include "rai/corelib_api.hpp"

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
    std::int64_t flat_mha_rows;
};

class Phi4ShapePlan final {
public:
    /// \brief interrogate corelib for the padded extent of every live row count
    /// \param api the corelib binding
    /// \param stream the stream the plan will be dispatched on
    /// \note corelib 0.5.0 takes the stream on every padding helper: the PDI
    ///       pair a stream was opened with selects the ELF, and a shape exists
    ///       under one pair and not another. So the plan is only meaningful
    ///       for the stream it was built against, and cannot be built before
    ///       one exists.
    static Phi4ShapePlan Build(
        const std::shared_ptr<const corelib::CorelibApi>& api,
        ryzenai_corelib_stream_ptr stream);
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
