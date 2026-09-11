#include "models/phi4/phi4_corelib_shape_plan.hpp"
#include "fake_corelib.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <memory>
#include <string>

namespace {
using flm::corelib::CorelibApi;
using flm::phi4::Phi4ShapePlan;

std::shared_ptr<CorelibApi> Api() {
    return CorelibApi::ResolveForTest(fake_corelib::Resolver());
}

void TestShapePlanQueriesRows1Through4096AtGroup64() {
    fake_corelib::Reset();
    const auto plan = Phi4ShapePlan::Build(Api());
    const auto& state = fake_corelib::GetState();
    TEST_REQUIRE(state.matmul_pad_calls.size() == 3 * 4096 + 1);
    TEST_REQUIRE(state.rows_pad_calls.size() == 4096);
    TEST_REQUIRE(state.mha_pad_calls.size() == 4096);
    for (std::size_t row = 1; row <= 4096; ++row) {
        TEST_REQUIRE(state.matmul_pad_calls[(row - 1) * 3].m == static_cast<std::int64_t>(row));
        TEST_REQUIRE(state.matmul_pad_calls[(row - 1) * 3].group_size == 64);
        TEST_REQUIRE(state.rows_pad_calls[row - 1].helper == "ssmlp");
        TEST_REQUIRE(state.rows_pad_calls[row - 1].group_size == 64);
        TEST_REQUIRE(plan.ForRows(row).rmsnorm_rows == static_cast<std::int64_t>(row));
        TEST_REQUIRE(state.mha_pad_calls[row - 1].m == static_cast<std::int64_t>(row));
    }
    TEST_REQUIRE(plan.ForRows(65).query_rows == 128);
}

void TestShapePlanUsesExactQKvOutputSsmlpRmsAndLmHeadDimensions() {
    fake_corelib::Reset();
    const auto plan = Phi4ShapePlan::Build(Api());
    const auto& state = fake_corelib::GetState();
    const auto& q = state.matmul_pad_calls[0];
    const auto& kv = state.matmul_pad_calls[1];
    const auto& output = state.matmul_pad_calls[2];
    TEST_REQUIRE(q.k == 3072 && q.n == 3072);
    TEST_REQUIRE(kv.k == 3072 && kv.n == 1024);
    TEST_REQUIRE(output.k == 3072 && output.n == 3072);
    TEST_REQUIRE(state.rows_pad_calls[0].helper == "ssmlp");
    TEST_REQUIRE(state.rows_pad_calls[0].k == 3072);
    TEST_REQUIRE(state.rows_pad_calls[0].n == 8192);
    TEST_REQUIRE(plan.ForRows(1).rmsnorm_rows == 1);
    TEST_REQUIRE(plan.ForRows(4096).rmsnorm_rows == 4096);
    const auto& lm = state.matmul_pad_calls.back();
    TEST_REQUIRE(lm.m == 1 && lm.k == 3072 && lm.n == 200064 && lm.group_size == 64);
    TEST_REQUIRE(plan.lm_head_desc().k == 3072);
    TEST_REQUIRE(plan.lm_head_desc().n == 200064);
}

void TestShapePlanBuildsFlatMhaDescriptor24_8_128_4096_96() {
    fake_corelib::Reset();
    const auto plan = Phi4ShapePlan::Build(Api());
    const auto& desc = plan.attention_desc();
    TEST_REQUIRE(desc.num_heads == 24);
    TEST_REQUIRE(desc.kv_num_heads == 8);
    TEST_REQUIRE(desc.head_size == 128);
    TEST_REQUIRE(desc.max_seq == 4096);
    TEST_REQUIRE(desc.rope_dim == 96);
    TEST_REQUIRE(fake_corelib::GetState().mha_pad_calls.front().desc.rope_dim == 96);
}

void TestShapePlanRejectsPaddedKOrNChanges() {
    fake_corelib::Reset();
    fake_corelib::GetState().matmul_k_delta = 1;
    RequireContains(RequireThrows([&] { Phi4ShapePlan::Build(Api()); }), "padded K/N");
    fake_corelib::Reset();
    fake_corelib::GetState().matmul_n_delta = 1;
    RequireContains(RequireThrows([&] { Phi4ShapePlan::Build(Api()); }), "padded K/N");
}

void TestShapePlanRejectsRowsOutsideCachedRange() {
    fake_corelib::Reset();
    const auto plan = Phi4ShapePlan::Build(Api());
    RequireContains(RequireThrows([&] { plan.ForRows(0); }), "1..4096");
    RequireContains(RequireThrows([&] { plan.ForRows(4097); }), "1..4096");
}

void TestShapePlanFailureNamesHelperAndLogicalShape() {
    fake_corelib::Reset();
    auto api = Api();
    fake_corelib::GetState().statuses["ryzenai_corelib_ssmlp_bf16_pad_rows"] =
        ryzenai_corelib_status_unsupported;
    const auto error = RequireThrows([&] { Phi4ShapePlan::Build(api); });
    RequireContains(error, "ryzenai_corelib_ssmlp_bf16_pad_rows");
    RequireContains(error, "[1,3072,8192]");
}
}  // namespace

int main() {
#define RUN_TEST(name) RunTest(&name, #name)
    RUN_TEST(TestShapePlanQueriesRows1Through4096AtGroup64);
    RUN_TEST(TestShapePlanUsesExactQKvOutputSsmlpRmsAndLmHeadDimensions);
    RUN_TEST(TestShapePlanBuildsFlatMhaDescriptor24_8_128_4096_96);
    RUN_TEST(TestShapePlanRejectsPaddedKOrNChanges);
    RUN_TEST(TestShapePlanRejectsRowsOutsideCachedRange);
    RUN_TEST(TestShapePlanFailureNamesHelperAndLogicalShape);
#undef RUN_TEST
}
