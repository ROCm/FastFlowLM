#pragma once

#include "core/corelib_api.hpp"

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
    std::vector<std::int64_t> strides;
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
    /// data_type of the matmul qweight or the ssmlp gate plane
    std::int32_t qweight_type{};
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

/// \brief one ..._weights_create call whose only source is a packed file slice
struct WeightFromFileRecord {
    std::string kind;
    std::string path;
    std::uint64_t offset{};
    std::uint64_t size{};
};

struct ModelForwardRecord {
    std::size_t rows{};
    std::size_t position{};
    bool logits{};
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
    /// \brief what npu_usable() answers
    bool npu_usable{true};
    /// \brief what testing_npu_arch() answers
    std::string npu_arch{"AIEM"};
    /// \brief the PDI lists the engine opened its stream with
    /// \note Required from 0.5.0 and deliberately undefaulted by corelib, so
    ///       it is worth being able to assert on.
    std::vector<int> stream_prefill_pdis;
    std::vector<int> stream_token_pdis;
    std::string detail;
    std::string status_text{"success"};
    std::string missing_symbol;
    std::vector<std::string> resolution_order;
    std::unordered_map<std::string, int> resolution_counts;
    std::unordered_map<std::string, int> call_counts;
    std::unordered_map<std::string, ryzenai_corelib_status> statuses;
    /// \brief how many more calls of a function fail before it answers as
    ///        `statuses` says
    std::unordered_map<std::string, int> fail_counts;
    std::vector<std::string> lifetime_events;
    std::atomic<int> live_objects{0};
    std::atomic<int> releases{0};
    std::atomic<int> cleanup_calls{0};
    std::atomic<int> active_leases{0};
    std::atomic<int> maximum_active_leases{0};
    std::vector<MatmulPadCall> matmul_pad_calls;
    std::vector<RowsPadCall> rows_pad_calls;
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
    std::vector<std::string> kernel_bundles;
    std::atomic<int> active_weight_creates{0};
    std::atomic<int> maximum_active_weight_creates{0};
    bool work_in_flight{false};
    std::string fail_after_submit;
    std::int64_t model_max_sequence_length{-1};
    std::int64_t model_vocab{8};
    std::size_t session_creates{};
    std::size_t state_saves{};
    std::size_t state_restores{};
    std::vector<ModelForwardRecord> model_forwards;
    /// \brief the name passed to each create_model_from_path
    std::vector<std::string> model_names;
    std::string model_path;
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
