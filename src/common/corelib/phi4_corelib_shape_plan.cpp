#include "models/phi4/phi4_corelib_shape_plan.hpp"

#include "models/phi4/phi4_corelib_constants.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace flm::phi4 {
namespace {

std::int64_t MatmulRows(const std::shared_ptr<const corelib::CorelibApi>& api,
                        std::int64_t rows, std::int64_t logical_k,
                        std::int64_t logical_n, const char* logical_name) {
    auto m = rows;
    auto k = logical_k;
    auto n = logical_n;
    const std::string call = std::string("ryzenai_corelib_matmul_bf16_pad_shape ") +
        logical_name + " [" + std::to_string(rows) + "," +
        std::to_string(logical_k) + "]x[" + std::to_string(logical_k) + "," +
        std::to_string(logical_n) + "]";
    api->Check(api->functions().matmul_pad_shape(
                   &m, &k, &n, kRequantizedGroupSize), call);
    if (k != logical_k || n != logical_n) {
        throw std::runtime_error(call + ": helper changed padded K/N");
    }
    return m;
}

}  // namespace

Phi4ShapePlan Phi4ShapePlan::Build(
    const std::shared_ptr<const corelib::CorelibApi>& api) {
    if (!api) throw std::invalid_argument("Phi4ShapePlan corelib API is null");

    Phi4ShapePlan plan;
    plan.attention_desc_ = {kQueryHeadCount, kKvHeadCount, kHeadSize,
                            kMaxSequenceLength, kRopeDimension};
    plan.lm_head_desc_ = {kHiddenSize, kVocabularySize,
                          kRequantizedGroupSize, false};
    plan.rows_.reserve(kMaxSequenceLength);

    for (std::int64_t rows = 1; rows <= kMaxSequenceLength; ++rows) {
        Phi4RowExtents extents{};
        extents.query_rows = MatmulRows(api, rows, kHiddenSize,
                                        kQueryDimension, "query");
        extents.kv_rows = MatmulRows(api, rows, kHiddenSize,
                                     kKvDimension, "key/value");
        extents.output_rows = MatmulRows(api, rows, kHiddenSize,
                                         kHiddenSize, "output");

        extents.ssmlp_rows = rows;
        const std::string ssmlp_call =
            "ryzenai_corelib_ssmlp_bf16_pad_rows [" + std::to_string(rows) +
            ",3072,8192]";
        api->Check(api->functions().ssmlp_pad_rows(
                       &extents.ssmlp_rows, kHiddenSize, kIntermediateSize,
                       kRequantizedGroupSize), ssmlp_call);

        extents.rmsnorm_rows = rows;
        const std::string rms_call =
            "ryzenai_corelib_rmsnorm_bf16_pad_rows [" + std::to_string(rows) +
            ",3072]";
        api->Check(api->functions().rmsnorm_pad_rows(
                       &extents.rmsnorm_rows, kHiddenSize), rms_call);

        extents.flat_mha_rows = rows;
        const std::string mha_call =
            "ryzenai_corelib_flat_mha_bf16_pad_rows [" + std::to_string(rows) +
            ",24,8,128,4096,96]";
        api->Check(api->functions().flat_mha_pad_rows(
                       &extents.flat_mha_rows, &plan.attention_desc_), mha_call);
        plan.maximum_extents_.query_rows = std::max(
            plan.maximum_extents_.query_rows, extents.query_rows);
        plan.maximum_extents_.kv_rows = std::max(
            plan.maximum_extents_.kv_rows, extents.kv_rows);
        plan.maximum_extents_.output_rows = std::max(
            plan.maximum_extents_.output_rows, extents.output_rows);
        plan.maximum_extents_.ssmlp_rows = std::max(
            plan.maximum_extents_.ssmlp_rows, extents.ssmlp_rows);
        plan.maximum_extents_.rmsnorm_rows = std::max(
            plan.maximum_extents_.rmsnorm_rows, extents.rmsnorm_rows);
        plan.maximum_extents_.flat_mha_rows = std::max(
            plan.maximum_extents_.flat_mha_rows, extents.flat_mha_rows);
        plan.rows_.push_back(extents);
    }

    (void)MatmulRows(api, 1, kHiddenSize, kVocabularySize, "lm_head");
    return plan;
}

const Phi4RowExtents& Phi4ShapePlan::ForRows(std::size_t live_rows) const {
    if (live_rows == 0 || live_rows > rows_.size()) {
        throw std::out_of_range("Phi-4 live rows must be in 1..4096");
    }
    return rows_[live_rows - 1];
}

const Phi4RowExtents& Phi4ShapePlan::maximum_extents() const noexcept {
    return maximum_extents_;
}

const ryzenai_corelib_flat_mha_bf16_desc&
Phi4ShapePlan::attention_desc() const noexcept {
    return attention_desc_;
}

const ryzenai_corelib_matmul_bf16_weights_desc&
Phi4ShapePlan::lm_head_desc() const noexcept {
    return lm_head_desc_;
}

}  // namespace flm::phi4
