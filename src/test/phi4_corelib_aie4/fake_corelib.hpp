#pragma once

#include "corelib/corelib_api.hpp"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace fake_corelib {

struct MatmulPadCall {
    std::int64_t m;
    std::int64_t k;
    std::int64_t n;
    std::uint32_t group_size;
};

struct RowsPadCall {
    std::string helper;
    std::int64_t m;
    std::int64_t k;
    std::int64_t n;
    std::uint32_t group_size;
};

struct MhaPadCall {
    std::int64_t m;
    ryzenai_corelib_flat_mha_bf16_desc desc;
};

struct TensorCreateRecord {
    ryzenai_corelib_data_type data_type;
    std::vector<std::int64_t> shape;
    void* object;
};

struct TensorWindowRecord {
    void* parent;
    std::vector<std::int64_t> shape;
    std::size_t offset;
    void* object;
};

struct WeightCreateRecord {
    std::string kind;
    std::int64_t k;
    std::int64_t n;
    std::uint32_t group_size;
    std::uint32_t threads;
    std::vector<const void*> pointers;
    std::vector<std::uint16_t> norm0;
    std::vector<std::uint16_t> norm1;
    std::uint16_t epsilon{};
};

struct DispatchRecord {
    std::string kind;
    void* stream;
    void* input;
    void* output;
    std::int64_t rows;
    std::int64_t position;
    std::size_t window_offset;
};

struct TensorWriteRecord {
    void* tensor;
    ryzenai_corelib_data_type source_type;
    std::size_t count;
    std::size_t offset;
    bool all_zero;
};

struct State {
    flm::corelib::CorelibVersion version{0, 3, 0};
    ryzenai_corelib_status selftest_status{ryzenai_corelib_status_success};
    ryzenai_corelib_status default_status{ryzenai_corelib_status_success};
    bool has_device_context{true};
    std::string detail;
    std::string status_text{"success"};
    std::string missing_symbol;
    std::vector<std::string> resolution_order;
    std::unordered_map<std::string, int> resolution_counts;
    std::unordered_map<std::string, int> call_counts;
    std::unordered_map<std::string, ryzenai_corelib_status> statuses;
    std::vector<std::string> lifetime_events;
    std::atomic<int> live_objects{0};
    std::atomic<int> releases{0};
    std::atomic<int> cleanup_calls{0};
    std::atomic<int> active_leases{0};
    std::atomic<int> maximum_active_leases{0};
    std::vector<MatmulPadCall> matmul_pad_calls;
    std::vector<RowsPadCall> rows_pad_calls;
    std::vector<MhaPadCall> mha_pad_calls;
    std::int64_t pad_multiple{64};
    std::int64_t matmul_k_delta{0};
    std::int64_t matmul_n_delta{0};
    std::vector<TensorCreateRecord> tensor_creates;
    std::vector<TensorWindowRecord> tensor_windows;
    std::vector<WeightCreateRecord> weight_creates;
    std::vector<DispatchRecord> dispatches;
    std::vector<TensorWriteRecord> tensor_writes;
    std::vector<std::string> call_log;
    std::atomic<int> active_weight_creates{0};
    std::atomic<int> maximum_active_weight_creates{0};
    bool work_in_flight{false};
    std::string fail_after_submit;
};

State& GetState();
void Reset();
flm::corelib::CorelibApi::Resolver Resolver();
std::vector<ryzenai_corelib_status> CallEveryResolvedFunction(
    const flm::corelib::CorelibFunctions& functions);
void* MakeObject();
void EnterLease();
void LeaveLease();

}  // namespace fake_corelib
