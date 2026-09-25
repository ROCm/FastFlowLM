#pragma once

#include "rai/corelib_api.hpp"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
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
    std::thread::id thread_id;
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

/// \brief one ..._weights_create_from_file call
struct WeightFromFileRecord {
    std::string kind;
    std::string path;
    std::uint64_t offset{};
    std::uint64_t size{};
};

struct State {
    /// \note Derived from the vendor header, not written out: the runtime
    ///       check in corelib_api.cpp compares against the same macros, so a
    ///       literal here would fail every test the day the pin moves.
    flm::corelib::CorelibVersion version{RYZENAI_CORELIB_VERSION_MAJOR,
                                         RYZENAI_CORELIB_VERSION_MINOR,
                                         RYZENAI_CORELIB_VERSION_PATCH};
    ryzenai_corelib_status selftest_status{ryzenai_corelib_status_success};
    ryzenai_corelib_status default_status{ryzenai_corelib_status_success};
    /// \brief whether get_device() hands back a device rather than NULL
    /// \note Still a bool here even though 0.5.0's entry point returns a
    ///       pointer: what a test wants to say is "this box has an NPU or it
    ///       does not", and the fake has no device object worth modelling.
    bool has_device_context{true};
    /// \brief the PDI pair the engine opened its stream with
    /// \note Required from 0.5.0 and deliberately undefaulted by corelib, so
    ///       it is worth being able to assert on.
    int stream_prefill_pdi{-1};
    int stream_token_pdi{-1};
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
    std::unordered_map<std::string,
        std::unordered_map<std::int64_t, std::int64_t>> pad_row_overrides;
    std::vector<TensorCreateRecord> tensor_creates;
    std::vector<TensorCreateRecord> host_view_creates;
    std::vector<TensorWindowRecord> tensor_windows;
    std::vector<WeightCreateRecord> weight_creates;
    std::vector<WeightFromFileRecord> weight_from_file;
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
