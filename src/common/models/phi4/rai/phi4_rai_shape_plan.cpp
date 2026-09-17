#include "models/phi4/rai/phi4_rai_shape_plan.hpp"

#include "models/phi4/rai/phi4_rai_constants.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>

namespace flm::phi4 {
namespace {

std::int64_t MatmulRows(const std::shared_ptr<const corelib::CorelibApi>& api,
                        ryzenai_corelib_stream_ptr stream,
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
                   stream, &m, &k, &n, kRequantizedGroupSize), call);
    if (k != logical_k || n != logical_n) {
        throw std::runtime_error(call + ": helper changed padded K/N");
    }
    return m;
}

}  // namespace

Phi4ShapePlan Phi4ShapePlan::Build(
    const std::shared_ptr<const corelib::CorelibApi>& api,
    ryzenai_corelib_stream_ptr stream) {
    if (!api) throw std::invalid_argument("Phi4ShapePlan corelib API is null");
    if (!stream) throw std::invalid_argument("Phi4ShapePlan stream is null");

    Phi4ShapePlan plan;
    plan.attention_desc_ = {kQueryHeadCount, kKvHeadCount, kHeadSize,
                            kMaxSequenceLength, kRopeDimension};
    plan.lm_head_desc_ = {kHiddenSize, kVocabularySize,
                          kRequantizedGroupSize, false};
    plan.rows_.reserve(kMaxSequenceLength);
    constexpr std::array<std::int64_t, 8> execution_rows{
        1, 64, 128, 256, 512, 1024, 2048, 4096};

    for (const auto rows : execution_rows) {
        Phi4RowExtents extents{};
        extents.query_rows = MatmulRows(api, stream, rows, kHiddenSize,
                                        kQueryDimension, "query");
        extents.kv_rows = MatmulRows(api, stream, rows, kHiddenSize,
                                     kKvDimension, "key/value");
        extents.output_rows = MatmulRows(api, stream, rows, kHiddenSize,
                                         kHiddenSize, "output");

        // 0.5.0 takes the whole weights descriptor here rather than k/n/group
        // separately, because the activation and the post-feedforward norm
        // select a different ELF family and therefore a different padding.
        // Phi-4 is silu with no post-feedforward norm: both trailing fields 0.
        const ryzenai_corelib_ssmlp_bf16_weights_desc ssmlp_desc{
            kHiddenSize, kIntermediateSize, kRequantizedGroupSize, 0, 0};
        extents.ssmlp_rows = rows;
        const std::string ssmlp_call =
            "ryzenai_corelib_ssmlp_bf16_pad_rows [" + std::to_string(rows) +
            ",3072,8192]";
        api->Check(api->functions().ssmlp_pad_rows(
                       stream, &extents.ssmlp_rows, &ssmlp_desc), ssmlp_call);

        extents.flat_mha_rows = rows;
        const std::string mha_call =
            "ryzenai_corelib_flat_mha_bf16_pad_rows [" + std::to_string(rows) +
            ",24,8,128,4096,96]";
        api->Check(api->functions().flat_mha_pad_rows(
                       stream, &extents.flat_mha_rows, &plan.attention_desc_),
                   mha_call);
        plan.maximum_extents_.query_rows = std::max(
            plan.maximum_extents_.query_rows, extents.query_rows);
        plan.maximum_extents_.kv_rows = std::max(
            plan.maximum_extents_.kv_rows, extents.kv_rows);
        plan.maximum_extents_.output_rows = std::max(
            plan.maximum_extents_.output_rows, extents.output_rows);
        plan.maximum_extents_.ssmlp_rows = std::max(
            plan.maximum_extents_.ssmlp_rows, extents.ssmlp_rows);
        plan.maximum_extents_.flat_mha_rows = std::max(
            plan.maximum_extents_.flat_mha_rows, extents.flat_mha_rows);
        while (plan.rows_.size() < static_cast<std::size_t>(rows))
            plan.rows_.push_back(extents);
    }

    (void)MatmulRows(api, stream, 1, kHiddenSize, kVocabularySize, "lm_head");
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
