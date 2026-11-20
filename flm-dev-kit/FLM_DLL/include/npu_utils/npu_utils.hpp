// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#if defined(FLM_USE_HRX)
/// \file npu_utils.hpp
/// \brief npu_utils class
/// \author FastFlowLM Team, Alfred
/// \date 2025-09-09
/// \note This file contains the classes for managing the npu device
#pragma once

/*
    The NPU is managed in this way:
    For each program, there should only be one npu_xclbin_manager, example:
    npu_xclbin_manager npu_mgr(device_npu2, 0);

    Then, multiple xclbins could be used by this npu and each xclbin may have multiple runtime sequences (applications)
    Therefore, each registered xclbin will have a npu_app_manager, example:

    npu_app_manager* mvm_i8_xclbin = npu_mgr.register_xclbin("mvm_i8.xclbin");

    Notice that the return value is a pointer to the npu_app_manager. This is to avoid duplicated npu hardware context,
    all xclbins are managed by the npu_xclbin_manager centrally.

    Finally, an application can be create from the npu_app_manager, example:

    npu_app app = mvm_i8_xclbin->create_app();

    In this version, the app do not have a name.
*/

#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdio.h>
#ifndef __WINDOWS__
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/types.h>
#include <drm/drm.h>
#include "amdxdna_accel.h"
#endif
#include "hrx_cpp/hrx_cpp.hpp"
#include "buffer.hpp"
#include "utils/debug_utils.hpp"
#include <cmath>

#include "npu_instr_utils.hpp"
#include "runtime_wait.hpp"

static_assert(flm::npu::to_ert_cmd_state(ERT_CMD_STATE_NEW) ==
              flm::npu::ErtCmdState::New);
static_assert(flm::npu::to_ert_cmd_state(ERT_CMD_STATE_QUEUED) ==
              flm::npu::ErtCmdState::Queued);
static_assert(flm::npu::to_ert_cmd_state(ERT_CMD_STATE_RUNNING) ==
              flm::npu::ErtCmdState::Running);
static_assert(flm::npu::to_ert_cmd_state(ERT_CMD_STATE_COMPLETED) ==
              flm::npu::ErtCmdState::Completed);
static_assert(flm::npu::to_ert_cmd_state(ERT_CMD_STATE_ERROR) ==
              flm::npu::ErtCmdState::Error);
static_assert(flm::npu::to_ert_cmd_state(ERT_CMD_STATE_ABORT) ==
              flm::npu::ErtCmdState::Abort);
static_assert(flm::npu::to_ert_cmd_state(ERT_CMD_STATE_SUBMITTED) ==
              flm::npu::ErtCmdState::Submitted);
static_assert(flm::npu::to_ert_cmd_state(ERT_CMD_STATE_TIMEOUT) ==
              flm::npu::ErtCmdState::Timeout);
static_assert(flm::npu::to_ert_cmd_state(ERT_CMD_STATE_NORESPONSE) ==
              flm::npu::ErtCmdState::NoResponse);
static_assert(flm::npu::to_ert_cmd_state(ERT_CMD_STATE_SKERROR) ==
              flm::npu::ErtCmdState::SkError);
static_assert(flm::npu::to_ert_cmd_state(ERT_CMD_STATE_SKCRASHED) ==
              flm::npu::ErtCmdState::SkCrashed);
static_assert(flm::npu::to_ert_cmd_state(ERT_CMD_STATE_MAX) ==
              flm::npu::ErtCmdState::Max);


class npu_app_manager;
class npu_xclbin_manager;


///@brief npu_app, a application that both xclbin and instruction are specified
///@param device the pointer to the device
///@param context the pointer to the context
///@param kernel_name the name of the kernel
///@see hrx::kernel, hrx::device

// Shared-Executable Collapse: a create_run/run_for_state/safe_run argument may be
// either a plain bytes&/buffer& (implicit byte offset 0) or a bound_arg carrying
// an explicit per-call byte offset into the buffer (e.g. a per-expert weight/bias
// base). The offset is forwarded to the dispatch as hrx_buffer_ref_t.offset so the
// TXN stays expert-independent and all experts share one executable generation.

// FLM_EXE_STATS counters (internal linkage): a function-local static of an inline
// member function becomes a process-wide GNU-unique symbol (nm 'u') emitted in
// every engine .so, which the symbol-clash check rejects. Housing the counters in
// an anonymous namespace gives each engine its own non-exported copy.
namespace {
inline size_t& flm_exe_acquire_calls() { static size_t v = 0; return v; }
inline size_t& flm_exe_builds()        { static size_t v = 0; return v; }
}  // namespace

struct bound_arg {
    bytes* buf;
    size_t byte_offset;
};
static inline bytes* bound_arg_buf(bytes& b) { return &b; }
static inline bytes* bound_arg_buf(const bound_arg& a) { return a.buf; }
static inline size_t bound_arg_off(bytes&) { return 0; }
static inline size_t bound_arg_off(const bound_arg& a) { return a.byte_offset; }

class npu_app {
private:
    // from external
    hrx::hw_context* context;
    hrx::device* device;
    std::string kernel_name;
    npu_device device_gen;
    bool enable_preemption;

    // self-managed
    bool module_valid;
    uint64_t module_version;
    hrx::executable_owner current_executable;
    std::unique_ptr<npu_sequence> ctrl_seq;

    hrx_executable_t acquire_current_executable(
        uint64_t revision, const uint32_t* control_words,
        size_t control_word_count, uint32_t* ordinal) {
        const bool rebuild = !current_executable.matches(revision);
        // Optional diagnostic (FLM_EXE_STATS, env-gated): under #196 each app owns
        // one current executable generation, so a "build" happens only when the
        // revision changes. The regeneration-skip keeps that rare; this counts
        // acquire calls vs actual build_executable invocations. NPU dispatch is
        // single-threaded on this path, so a plain static counter is sufficient.
        if (std::getenv("FLM_EXE_STATS")) {
            size_t& acquire_calls = flm_exe_acquire_calls();
            size_t& builds = flm_exe_builds();
            ++acquire_calls;
            if (rebuild) ++builds;
            std::fprintf(stderr,
                         "[FLM_EXE_STATS] acquire_calls=%zu builds=%zu %s words=%zu\n",
                         acquire_calls, builds, rebuild ? "NEW" : "reuse",
                         control_word_count);
            std::fflush(stderr);
        }
        if (rebuild) {
            uint32_t new_ordinal = 0;
            hrx_executable_t replacement = hrx::build_executable(
                this->context->xclbin_bytes(),
                this->context->xclbin_identity(), control_words,
                control_word_count, &new_ordinal);
            if (!replacement) return nullptr;
            current_executable.replace(replacement, revision, new_ordinal);
        }
        return current_executable.acquire(ordinal);
    }

    ///@brief Transfer all state from `o`.
    void move_from(npu_app&& o) noexcept {
        device_gen = o.device_gen;
        device = o.device;
        context = o.context;
        kernel_name = std::move(o.kernel_name);
        enable_preemption = o.enable_preemption;
        module_valid = o.module_valid;
        module_version = o.module_version;
        current_executable = std::move(o.current_executable);
        ctrl_seq = std::move(o.ctrl_seq);
        o.module_valid = false;
    }
    ///@brief Setup the kernel
    ///@note Builds an HRX XADX executable directly from the
    ///      ctrl_seq TXN stream (npu_sequence::dump()). aiebu/ELF are not used.
    void _setup_kernel(){
        std::pair<uint32_t*, size_t> data = this->ctrl_seq->dump();
        assert(data.first != nullptr);
        assert(data.second > 0);
        assert(this->context != nullptr);
        if (const char* dd = std::getenv("FLM_DUMP_TXN")) {
            // Host patch table (debug dump only): HRX now derives this itself from
            // the TXN stream inside hrx_amdxdna_executable_create(); we recompute
            // it here purely to dump for inspection.
            std::vector<uint32_t> patch = this->ctrl_seq->dump_patch_table();
            static int dump_idx = 0;
            std::string dir = (dd[0] && dd[0] != '1') ? dd : "/tmp/flm_dump";
            (void)system(("mkdir -p '" + dir + "'").c_str());
            int idx = dump_idx++;
            char fn[64];
            std::snprintf(fn, sizeof(fn), "/txn_%02d.bin", idx);
            std::ofstream tf(dir + fn, std::ios::binary);
            tf.write((const char*)data.first, data.second * sizeof(uint32_t));
            std::snprintf(fn, sizeof(fn), "/patch_%02d.bin", idx);
            std::ofstream pf(dir + fn, std::ios::binary);
            pf.write((const char*)patch.data(), patch.size() * sizeof(uint32_t));
            std::fprintf(stderr,
                         "[FLM_DUMP_TXN] kernel %d '%s': txn=%zu words patch=%zu triples\n",
                         idx, this->kernel_name.c_str(), data.second, patch.size() / 3);
        }
        uint32_t ord = 0;
        const uint64_t revision = this->ctrl_seq->sequence_version();
        hrx_executable_t exe = this->acquire_current_executable(
            revision, data.first, data.second, &ord);
        if (exe == nullptr){
            header_print_r("ERROR", "Failed to build HRX executable from ctrl_seq");
            exit(1);
        }
        // npu_app owns one reference; this temporary reference only verifies
        // that the immutable generation is ready for subsequent dispatches.
        hrx_executable_release(exe);
        this->module_valid = true;
        this->module_version = revision;
    }

    hrx::run _create_hrx_run() {
        std::pair<uint32_t*, size_t> data = this->ctrl_seq->dump();
        uint32_t ord = 0;
        hrx_executable_t exe = this->acquire_current_executable(
            this->module_version, data.first, data.second, &ord);
        if (exe == nullptr) {
            header_print_r("ERROR", "Failed to acquire HRX executable");
            exit(1);
        }
        return hrx::run(exe, ord);
    }

public:
    // enum ert_cmd_state {
    //     ERT_CMD_STATE_NEW = 1,
    //     ERT_CMD_STATE_QUEUED = 2,
    //     ERT_CMD_STATE_RUNNING = 3,
    //     ERT_CMD_STATE_COMPLETED = 4,
    //     ERT_CMD_STATE_ERROR = 5,
    //     ERT_CMD_STATE_ABORT = 6,
    //     ERT_CMD_STATE_SUBMITTED = 7,
    //     ERT_CMD_STATE_TIMEOUT = 8,
    //     ERT_CMD_STATE_NORESPONSE = 9,
    //     ERT_CMD_STATE_SKERROR = 10, //Check for error return code from Soft Kernel
    //     ERT_CMD_STATE_SKCRASHED = 11, //Soft kernel has crashed
    //     ERT_CMD_STATE_MAX, // Always the last one
    // };
    std::map<ert_cmd_state, std::string> cmd_state_map = {
        {ERT_CMD_STATE_NEW, "new"},
        {ERT_CMD_STATE_QUEUED, "queued"},
        {ERT_CMD_STATE_RUNNING, "running"},
        {ERT_CMD_STATE_COMPLETED, "completed"},
        {ERT_CMD_STATE_ERROR, "error"},
        {ERT_CMD_STATE_ABORT, "abort"},
        {ERT_CMD_STATE_SUBMITTED, "submitted"},
        {ERT_CMD_STATE_TIMEOUT, "timeout"},
        {ERT_CMD_STATE_NORESPONSE, "noresponse"},
        {ERT_CMD_STATE_SKERROR, "skerror"},
        {ERT_CMD_STATE_SKCRASHED, "skcrashed"},
        {ERT_CMD_STATE_MAX, "max"},
    };

    ///@brief Default Constructor
    ///@note Initialize the npu_app to nullptr
    npu_app() {
        this->device_gen = device_npu2;
        this->device = nullptr;
        this->context = nullptr;
        this->module_valid = false;
        this->module_version = UINT64_MAX;
    }

    ///@brief Constructor, this shall not invoke by user, it shall only be invoked by npu_app_manager
    ///@param device_gen the npu device
    ///@param device the pointer to the device
    ///@param context the pointer to the context
    ///@param kernel_name the name of the kernel
    ///@see hrx::device, hrx::hw_context, hrx::kernel
    npu_app(npu_device device_gen, hrx::device* device, hrx::hw_context* context, std::string kernel_name, bool enable_preemption = false):
        device_gen(device_gen), device(device), context(context), kernel_name(kernel_name), enable_preemption(enable_preemption){
        this->module_valid = false;
        this->ctrl_seq = std::make_unique<npu_sequence>(device_gen, enable_preemption);
        this->module_version = UINT64_MAX;
    }

    ~npu_app() = default;

    // The unique_ptr makes npu_app move-only. Each app owns only its current
    // executable generation; individual in-flight hrx::run values retain it.
    npu_app(const npu_app&) = delete;
    npu_app& operator=(const npu_app&) = delete;
    npu_app(npu_app&& other) noexcept { this->move_from(std::move(other)); }
    npu_app& operator=(npu_app&& other) noexcept {
        if (this != &other) {
            this->move_from(std::move(other));
        }
        return *this;
    }
  
    void update_ctrl_seq(){
        assert(this->ctrl_seq != nullptr);
        this->_setup_kernel();
    }

    void load_elf(std::string elf_name){
        // ELF-based control loading is not used on the HRX backend (control
        // code comes from ctrl_seq->dump()).
        (void)elf_name;
        header_print("warning", "npu_app::load_elf is not supported on the HRX backend (ignored)");
    }

    void store_elf(std::string elf_name){
        (void)elf_name;
        header_print("warning", "npu_app::store_elf is not supported on the HRX backend (ignored)");
    }

    ///@brief Name used in a completion failure. Includes the xclbin kernel when known.
    std::string completion_where(const char* what) const {
        std::string where = std::string("npu_app::") + what;
        if (this->kernel_name.empty()) return where;
        return this->kernel_name + " " + where;
    }

    ///@brief Run the kernel and return its terminal state, including a failure.
    ///       Hunyuan prefill uses this so a driver timeout can be retried from a
    ///       full switch. operator() throws on any state other than completed.
    ///@param args arguments, shall be the buffers with real bo
    ///@see hrx::run
    template<typename... BoArgs>
    ert_cmd_state run_for_state(BoArgs&&... args){
        if (this->module_valid == false || this->ctrl_seq->sequence_valid() == false || this->module_version != this->ctrl_seq->sequence_version()) {
            this->_setup_kernel();
        }
        hrx::run run = this->_create_hrx_run();
        std::array<bytes*, sizeof...(BoArgs)> bo_args = { bound_arg_buf(args)... };
        std::array<size_t, sizeof...(BoArgs)> bo_offs = { bound_arg_off(args)... };
        for (size_t i = 0; i < sizeof...(args); i++){
            assert(bo_offs[i] <= bo_args[i]->bo().size());
            run.add_binding(bo_args[i]->bo().handle(), bo_offs[i], bo_args[i]->bo().size() - bo_offs[i]);
        }
        run.start();
        return run.wait();
    }

    ///@brief Operator() for running the kernel
    ///@param args arguments, shall be the buffers with real bo
    ///@see hrx::run
    template<typename... BoArgs>
    ert_cmd_state operator()(BoArgs&&... args){
        ert_cmd_state state = this->run_for_state(std::forward<BoArgs>(args)...);
        flm::npu::require_completed_ert_state(
            flm::npu::to_ert_cmd_state(state), this->completion_where("operator()"));
        LOG_VERBOSE(2, "ending state: " << this->cmd_state_map[state]);
        return state;
    }

    
    ///@brief Operator() for running the kernel
    ///@param args arguments, shall be the buffers with real bo
    ///@see hrx::run
    template<typename... BoArgs>
    ert_cmd_state safe_run(BoArgs&&... args){
        if (this->module_valid == false || this->ctrl_seq->sequence_valid() == false || this->module_version != this->ctrl_seq->sequence_version()) {
            this->_setup_kernel();
        }
        std::array<bytes*, sizeof...(BoArgs)> bo_args = { bound_arg_buf(args)... };
        std::array<size_t, sizeof...(BoArgs)> bo_offs = { bound_arg_off(args)... };
        for (size_t i = 0; i < sizeof...(args); i++){
            bo_args[i]->sync_to_device();
        }
        hrx::run run = this->_create_hrx_run();
        for (size_t i = 0; i < sizeof...(args); i++){
            assert(bo_offs[i] <= bo_args[i]->bo().size());
            run.add_binding(bo_args[i]->bo().handle(), bo_offs[i], bo_args[i]->bo().size() - bo_offs[i]);
        }
        run.start();
        ert_cmd_state state =
            flm::npu::wait_for_completion(run, this->completion_where("safe_run"));
        for (size_t i = 0; i < sizeof...(args); i++){
            bo_args[i]->sync_from_device();
        }
        LOG_VERBOSE(2, "ending state: " << this->cmd_state_map[state]);
        return state;
    }
    
    ///@brief Create a run object
    ///@param args arguments, shall be the buffers with real bo
    ///@return a run object for waiting
    ///@see hrx::run
    template<typename... BoArgs>
    hrx::run create_run(BoArgs&&... args){
        if (this->module_valid == false || this->ctrl_seq->sequence_valid() == false || this->module_version != this->ctrl_seq->sequence_version()) {
            this->_setup_kernel();
        }
        hrx::run run = this->_create_hrx_run();
        // HRX bindings are the buffer args only, in order (arg 3 = output). The
        // (3,0,0) opcode scalars are encoded in the TXN stream, not as bindings.
        // Each arg is either a plain bytes&/buffer& (offset 0) or a bound_arg with
        // an explicit per-call byte offset (Shared-Executable Collapse).
        std::array<bytes*, sizeof...(BoArgs)> bo_args = { bound_arg_buf(args)... };
        std::array<size_t, sizeof...(BoArgs)> bo_offs = { bound_arg_off(args)... };
        for (size_t i = 0; i < sizeof...(args); i++){
            assert(bo_offs[i] <= bo_args[i]->bo().size());
            run.add_binding(bo_args[i]->bo().handle(), bo_offs[i], bo_args[i]->bo().size() - bo_offs[i]);
        }
        return run;
    }

    ///@brief Create a buffer with real bo
    ///@param size size of the buffer
    ///@see buffer
    template<typename T>
    buffer<T> create_bo_buffer(size_t size){
        assert(size > 0);
        LOG_VERBOSE(2, "Creating buffer buffer with size: " << size);
        return buffer<T>(*this->device, size);
    }

    npu_sequence* seq() {
        return this->ctrl_seq.get();
    }
};

class npu_app_manager {
private:
    npu_device device_gen;
    bool enable_preemption;
    hrx::device* device;
    std::unique_ptr<hrx::hw_context> context;
    std::string kernel_name;
    std::string xclbin_name;
    bool xclbin_valid;
public:
    ///@brief Default Constructor
    ///@param device_gen the npu device
    ///@param device device object
    ///@param xclbin_name name of the xclbin file
    ///@see hrx::device, hrx::xclbin
    ///@note The function will initialize the npu_app_manager to nullptr
    npu_app_manager(){
        this->device_gen = device_npu2;
        this->device = nullptr;
        this->context = nullptr;
        this->kernel_name = "";
        this->xclbin_name = "";
        this->xclbin_valid = false;
        this->enable_preemption = false;
    }

    ///@brief Copy constructor, this shall not invoke by user, it shall only be invoked by npu_xclbin_manager
    ///@param other the other npu_app_manager to copy from
    npu_app_manager(const npu_app_manager& other) {
        this->device_gen = other.device_gen;
        this->device = other.device;
        this->kernel_name = other.kernel_name;
        this->xclbin_name = other.xclbin_name;
        this->xclbin_valid = other.xclbin_valid;
        this->enable_preemption = other.enable_preemption;
        if (other.context) {
            this->context = std::make_unique<hrx::hw_context>(*other.context);
        } else {
            this->context = nullptr;
        }
    }

    ///@brief Copy assignment operator
    ///@param other the other npu_app_manager to copy from
    ///@return reference to this object
    npu_app_manager& operator=(const npu_app_manager& other) {
        if (this != &other) {
            this->device_gen = other.device_gen;
            this->device = other.device;
            this->kernel_name = other.kernel_name;
            this->xclbin_name = other.xclbin_name;
            this->xclbin_valid = other.xclbin_valid;
            this->enable_preemption = other.enable_preemption;
            if (other.context) {
                this->context = std::make_unique<hrx::hw_context>(*other.context);
            } else {
                this->context = nullptr;
            }
        }
        return *this;
    }

    ///@brief Move constructor
    ///@param other the other npu_app_manager to move from
    npu_app_manager(npu_app_manager&& other) noexcept {
        this->device_gen = other.device_gen;
        this->device = other.device;
        this->context = std::move(other.context);
        this->kernel_name = std::move(other.kernel_name);
        this->xclbin_name = std::move(other.xclbin_name);
        this->xclbin_valid = other.xclbin_valid;
        this->enable_preemption = other.enable_preemption;
        // Reset the moved-from object
        other.device = nullptr;
        other.xclbin_valid = false;
    }

    ///@brief Move assignment operator
    ///@param other the other npu_app_manager to move from
    ///@return reference to this object
    npu_app_manager& operator=(npu_app_manager&& other) noexcept {
        if (this != &other) {
            this->device_gen = other.device_gen;
            this->device = other.device;
            this->context = std::move(other.context);
            this->kernel_name = std::move(other.kernel_name);
            this->xclbin_name = std::move(other.xclbin_name);
            this->xclbin_valid = other.xclbin_valid;
            this->enable_preemption = other.enable_preemption;
            // Reset the moved-from object
            other.device = nullptr;
            other.xclbin_valid = false;
        }
        return *this;
    }

    ///@brief Constructor, this shall not invoke by user, it shall only be invoked by npu_xclbin_manager
    ///@param device_gen the npu device
    ///@param device device object
    ///@param xclbin_name name of the xclbin file
    ///@see hrx::device, hrx::xclbin
    npu_app_manager(npu_device device_gen, hrx::device* device, std::string xclbin_name, bool enable_preemption = false){
        assert(device != nullptr);
        assert(xclbin_name != "");
        this->device_gen = device_gen;
        this->device = device;
        this->xclbin_name = xclbin_name;
        this->enable_preemption = enable_preemption;
        #ifndef __WINDOWS__
        if (this->enable_preemption){
            header_print("warning", "Preemption is not supported on Linux host currently.");
            header_print("warning", "Preemption is set as disabled for NPU.");
            this->enable_preemption = false;
        }
        #endif
        LOG_VERBOSE(2, "Loading xclbin: " << xclbin_name);
        auto this_xclbin = hrx::xclbin(xclbin_name);
        // int verbosity = VERBOSE;
        std::string Node = "MLIR_AIE";
        auto xkernels = this_xclbin.get_kernels();
        auto xkernel = *std::find_if(
            xkernels.begin(), 
            xkernels.end(),
            [Node](hrx::xclbin::kernel &k) {
                auto name = k.get_name();
                return name.rfind(Node, 0) == 0;
            }
        );
        this->device->register_xclbin(this_xclbin);
        auto kernelName = xkernel.get_name();
        // Pass the xclbin (its raw bytes) into the context so npu_app can build
        // the HRX XADX executable that wraps it.
        this->context = std::make_unique<hrx::hw_context>(*this->device, this_xclbin);
        this->kernel_name = kernelName;
        this->xclbin_valid = true;
        LOG_VERBOSE(2, "Xclbin: " << xclbin_name << " loaded successfully!");
    }

    ~npu_app_manager() = default;

    ///@brief Create a npu_app
    ///@return a npu_app object
    ///@see npu_app
    npu_app create_app(){
        assert(this->xclbin_valid);
        return npu_app(this->device_gen, this->device, this->context.get(), this->kernel_name, this->enable_preemption);
    }

    ///@brief Create a buffer with real bo
    ///@param size size of the buffer
    ///@see buffer
    template<typename T>
    buffer<T> create_bo_buffer(size_t size){
        assert(size > 0);
        LOG_VERBOSE(2, "Creating buffer buffer with size: " << size);
        return buffer<T>(*this->device, size);
    }

    ///@brief Get the name of the xclbin
    ///@return the name of the xclbin
    std::string get_xclbin_name(){
        assert(this->xclbin_valid);
        return this->xclbin_name;
    }

    ///@brief Create a runlist
    ///@return a runlist object
    ///@see hrx::runlist
    hrx::runlist create_runlist(){
        assert(this->xclbin_valid);
        assert(this->enable_preemption == false); // preemption is not supported for runlist
        return hrx::runlist(*this->context);
    }
};

///@brief npu_xclbin_manager
///@note There should be only one npu_xclbin_manager inside main.
///@note It handles all xclbins
class npu_xclbin_manager{

private:
    std::vector<std::unique_ptr<npu_app_manager>> npu_xclbins;

    size_t xclbin_count;
    // the only device instance
    hrx::device* device;
    bool enable_preemption;
    npu_device npu_gen;
public:
    constexpr static int max_xclbins = 16; // hard constraint inherited from the NPU runtime
    
    ///@brief Constructor, this shall not invoke by user, it shall only be invoked by main
    ///@param device the npu device
    ///@param device_id the device id
    ///@see hrx::device
    npu_xclbin_manager(npu_device device = device_npu2, hrx::device* device_inst = nullptr, bool enable_preemption = false){
        this->device = device_inst;
        // this->npu_xclbins.resize(max_xclbins);
        this->npu_xclbins.reserve(max_xclbins);
        this->xclbin_count = 0;
        this->npu_gen = device;
        this->enable_preemption = enable_preemption;
    }

    ~npu_xclbin_manager() = default;

    ///@brief register an accel_user_desc to the npu_manager
    ///@param xclbin_name the name of the xclbin
    ///@return the npu_app_manager object
    ///@see npu_app_manager
    ///@note Different apps may have the same xclbin, but the sequence is unique.
    ///@note To avoid creating duplicated applications, the function checks if the xclbin is registered.
    ///@note If the xclbin is not registered, the function will register the xclbin and create a new application.
    npu_app_manager* register_xclbin(std::string xclbin_name){
        assert(xclbin_name != "");
        int xclbin_id = -1;
        for (size_t i = 0; i < this->xclbin_count; i++){
    
            if (this->npu_xclbins[i]->get_xclbin_name() == xclbin_name){
                xclbin_id = i;
                break;
            }
        }
        LOG_VERBOSE_IF_ELSE(2, xclbin_id > -1, 
            "Found xclbin: " << xclbin_name << "registered as id " << xclbin_id << "!",
            "Xclbin: " << xclbin_name << " not registered yet!"
        );
    
        if (xclbin_id == -1){ // the xclbin is not registered yet
            if (this->xclbin_count >= max_xclbins){
                throw std::runtime_error("Max number of xclbins reached");
            }
            this->npu_xclbins.push_back(std::make_unique<npu_app_manager>(this->npu_gen, this->device, xclbin_name, this->enable_preemption));
            xclbin_id = this->xclbin_count;
            LOG_VERBOSE(2, "Xclbin: " << xclbin_name << " registered as id " << xclbin_id << "!");
            this->xclbin_count++;
            xclbin_id = this->xclbin_count - 1;
        }
        return this->npu_xclbins[xclbin_id].get();
    }

    void list_xclbins(){
        for (size_t i = 0; i < this->xclbin_count; i++){
            std::cout << "Xclbin " << i << " name: " << this->npu_xclbins[i]->get_xclbin_name() << std::endl;
        }
    }

    /// @brief Helper function to write out the trace to a file
    /// @param traceOutPtr pointer to the trace
    /// @param trace_size size of the trace
    /// @param path path to the file
    static void write_out_trace(char *traceOutPtr, size_t trace_size, std::string path) {
        std::ofstream fout(path);
        LOG_VERBOSE(1, "Writing out trace to: " << path);
        uint32_t *traceOut = (uint32_t *)traceOutPtr;
        for (size_t i = 0; i < trace_size / sizeof(traceOut[0]); i++) {
          fout << std::setfill('0') << std::setw(8) << std::hex << (int)traceOut[i];
          fout << std::endl;
        }
        fout.close();
        LOG_VERBOSE(1, "Trace written successfully!");
    }
    
    ///@brief Create a buffer with real bo
    ///@param size size of the buffer
    ///@see buffer
    template<typename T>
    buffer<T> create_bo_buffer(size_t size){
        assert(size > 0);
        LOG_VERBOSE(2, "Creating buffer buffer with size: " << size);
        return buffer<T>(*this->device, size);
    }

    inline bool is_preemption_enabled(){
        return this->enable_preemption;
    }

    #ifndef __WINDOWS__
    ///@brief print the npu information
    ///@note The function will print the npu version, clock frequency, column count, row count, core info, mem info, shim info.
    ///@note The information is read via the IOCTL interface.
    void print_npu_info(){
        int fd = open("/dev/accel/accel0", O_RDWR);
        if (fd < 0) {
            perror("Failed to open amdgpu device");
            return;
        }
        amdxdna_drm_query_clock_metadata query_clock_metadata;
        amdxdna_drm_get_info get_info = {
            .param = DRM_AMDXDNA_QUERY_CLOCK_METADATA,
            .buffer_size = sizeof(amdxdna_drm_query_clock_metadata),
            .buffer = (unsigned long)&query_clock_metadata,
        };
        int ret = ioctl(fd, DRM_IOCTL_AMDXDNA_GET_INFO, &get_info);
        if (ret < 0) {
            std::cout << "Error code: " << ret << std::endl;
            perror("Failed to get telemetry information");
            close(fd);
            return;
        }

        amdxdna_drm_query_aie_metadata query_aie_metadata = {};
        get_info.param = DRM_AMDXDNA_QUERY_AIE_METADATA;
        get_info.buffer_size = sizeof(amdxdna_drm_query_aie_metadata);
        get_info.buffer = (unsigned long)&query_aie_metadata;
        ret = ioctl(fd, DRM_IOCTL_AMDXDNA_GET_INFO, &get_info);
        if (ret < 0) {
            std::cout << "Error code: " << ret << std::endl;
            perror("Failed to get telemetry information");
            close(fd);
            return;
        }

        close(fd);
        MSG_BONDLINE(40);
        MSG_BOX_LINE(40, "NPU version: " << query_aie_metadata.version.major << "." << query_aie_metadata.version.minor);
        MSG_BOX_LINE(40, "MP-NPU clock frequency: " << query_clock_metadata.mp_npu_clock.freq_mhz << " MHz");
        MSG_BOX_LINE(40, "H clock frequency: " << query_clock_metadata.h_clock.freq_mhz << " MHz");
        // What is the meaning of the column size?
        // std::cout << "NPU column size: " << query_aie_metadata.col_size << std::endl;
        MSG_BOX_LINE(40, "NPU column count: " << query_aie_metadata.cols);
        MSG_BOX_LINE(40, "NPU row count: " << query_aie_metadata.rows);
        MSG_BOX_LINE(40, "NPU core Info: ");
        MSG_BOX_LINE(40, "--Row count: " << query_aie_metadata.core.row_count);
        MSG_BOX_LINE(40, "--Row start: " << query_aie_metadata.core.row_start);
        MSG_BOX_LINE(40, "--DMA channel count: " << query_aie_metadata.core.dma_channel_count);
        MSG_BOX_LINE(40, "--Lock count: " << query_aie_metadata.core.lock_count);
        MSG_BOX_LINE(40, "--Event reg count: " << query_aie_metadata.core.event_reg_count);
        MSG_BOX_LINE(40, "NPU mem Info: ");
        MSG_BOX_LINE(40, "--Row count: " << query_aie_metadata.mem.row_count);
        MSG_BOX_LINE(40, "--Row start: " << query_aie_metadata.mem.row_start);
        MSG_BOX_LINE(40, "--DMA channel count: " << query_aie_metadata.mem.dma_channel_count);
        MSG_BOX_LINE(40, "--Lock count: " << query_aie_metadata.mem.lock_count);
        MSG_BOX_LINE(40, "--Event reg count: " << query_aie_metadata.mem.event_reg_count);
        MSG_BOX_LINE(40, "NPU shim Info: ");
        MSG_BOX_LINE(40, "--Row count: " << query_aie_metadata.shim.row_count);
        MSG_BOX_LINE(40, "--Row start: " << query_aie_metadata.shim.row_start);
        MSG_BOX_LINE(40, "--DMA channel count: " << query_aie_metadata.shim.dma_channel_count);
        MSG_BOX_LINE(40, "--Lock count: " << query_aie_metadata.shim.lock_count);
        MSG_BOX_LINE(40, "--Event reg count: " << query_aie_metadata.shim.event_reg_count);
        MSG_BONDLINE(40);
    }

    ///@brief get the npu power consumption
    ///@param print whether to print the power consumption
    ///@return the power consumption, unit is Watt
    float get_npu_power(bool print){
        // get the npu power consumption, unit is Watt
        int fd = open("/dev/accel/accel0", O_RDWR);
        if (fd < 0) {
            perror("Failed to open amdgpu device");
            return -1;
        }
        amdxdna_drm_query_sensor query_sensor;

        amdxdna_drm_get_info get_info = {
            .param = DRM_AMDXDNA_QUERY_SENSORS,
            .buffer_size = sizeof(amdxdna_drm_query_sensor),
            .buffer = (unsigned long)&query_sensor,
        };
        int ret = ioctl(fd, DRM_IOCTL_AMDXDNA_GET_INFO, &get_info);
        if (ret < 0) {
            std::cout << "Error code: " << ret << std::endl;
            perror("Failed to get telemetry information");
            close(fd);
            return -1;
        }
        if (print){
            MSG_BOX(40, "NPU power: " << query_sensor.input << " " << query_sensor.units);
        }
        close(fd);
        return (float)query_sensor.input * pow(10, query_sensor.unitm);
    }
    #endif
};
#else
/// \file npu_utils.hpp
/// \brief npu_utils class
/// \author FastFlowLM Team, Alfred
/// \date 2025-09-09
/// \note This file contains the classes for managing the npu device
#pragma once

/*
    The NPU is managed in this way:
    For each program, there should only be one npu_xclbin_manager, example:
    npu_xclbin_manager npu_mgr(device_npu2, 0);

    Then, multiple xclbins could be used by this npu and each xclbin may have multiple runtime sequences (applications)
    Therefore, each registered xclbin will have a npu_app_manager, example:

    npu_app_manager* mvm_i8_xclbin = npu_mgr.register_xclbin("mvm_i8.xclbin");

    Notice that the return value is a pointer to the npu_app_manager. This is to avoid duplicated npu hardware context,
    all xclbins are managed by the npu_xclbin_manager centrally.

    Finally, an application can be create from the npu_app_manager, example:

    npu_app app = mvm_i8_xclbin->create_app();

    In this version, the app do not have a name.
*/

#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdio.h>
#ifndef __WINDOWS__
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/types.h>
#include <drm/drm.h>
#include "amdxdna_accel.h"
#endif
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "buffer.hpp"
#include "utils/debug_utils.hpp"
#include <cmath>

#include "xrt/experimental/xrt_kernel.h"
#include "xrt/experimental/xrt_ext.h"
#include "xrt/experimental/xrt_module.h"
#include "xrt/experimental/xrt_elf.h"

#include "aiebu/aiebu.h"

#include "npu_instr_utils.hpp"
#include "runtime_wait.hpp"

static_assert(flm::npu::to_ert_cmd_state(ERT_CMD_STATE_NEW) ==
              flm::npu::ErtCmdState::New);
static_assert(flm::npu::to_ert_cmd_state(ERT_CMD_STATE_QUEUED) ==
              flm::npu::ErtCmdState::Queued);
static_assert(flm::npu::to_ert_cmd_state(ERT_CMD_STATE_RUNNING) ==
              flm::npu::ErtCmdState::Running);
static_assert(flm::npu::to_ert_cmd_state(ERT_CMD_STATE_COMPLETED) ==
              flm::npu::ErtCmdState::Completed);
static_assert(flm::npu::to_ert_cmd_state(ERT_CMD_STATE_ERROR) ==
              flm::npu::ErtCmdState::Error);
static_assert(flm::npu::to_ert_cmd_state(ERT_CMD_STATE_ABORT) ==
              flm::npu::ErtCmdState::Abort);
static_assert(flm::npu::to_ert_cmd_state(ERT_CMD_STATE_SUBMITTED) ==
              flm::npu::ErtCmdState::Submitted);
static_assert(flm::npu::to_ert_cmd_state(ERT_CMD_STATE_TIMEOUT) ==
              flm::npu::ErtCmdState::Timeout);
static_assert(flm::npu::to_ert_cmd_state(ERT_CMD_STATE_NORESPONSE) ==
              flm::npu::ErtCmdState::NoResponse);
static_assert(flm::npu::to_ert_cmd_state(ERT_CMD_STATE_SKERROR) ==
              flm::npu::ErtCmdState::SkError);
static_assert(flm::npu::to_ert_cmd_state(ERT_CMD_STATE_SKCRASHED) ==
              flm::npu::ErtCmdState::SkCrashed);
static_assert(flm::npu::to_ert_cmd_state(ERT_CMD_STATE_MAX) ==
              flm::npu::ErtCmdState::Max);


class npu_app_manager;
class npu_xclbin_manager;


///@brief npu_app, a application that both xclbin and instruction are specified
///@param device the pointer to the device
///@param context the pointer to the context
///@param kernel_name the name of the kernel
///@see xrt::kernel, xrt::device

// Shared-Executable Collapse: a create_run/run_for_state/safe_run argument may be
// either a plain bytes&/buffer& (implicit byte offset 0) or a bound_arg carrying
// an explicit per-call byte offset into the buffer (e.g. a per-expert weight/bias
// base). For XRT the offset binds a sub-buffer; the TXN stays expert-independent.
struct bound_arg {
    bytes* buf;
    size_t byte_offset;
};
static inline bytes* bound_arg_buf(bytes& b) { return &b; }
static inline bytes* bound_arg_buf(const bound_arg& a) { return a.buf; }
static inline size_t bound_arg_off(bytes&) { return 0; }
static inline size_t bound_arg_off(const bound_arg& a) { return a.byte_offset; }

class npu_app {
private:
    // from external
    xrt::hw_context* context;
    xrt::device* device;
    std::string kernel_name;
    npu_device device_gen;
    bool enable_preemption;

    // self-managed
    bool module_valid;
    uint64_t module_version;
    std::unique_ptr<xrt::module> module;
    std::unique_ptr<xrt::elf> elf;
    std::unique_ptr<xrt::ext::kernel> kernel;
    std::unique_ptr<npu_sequence> ctrl_seq;

    uint32_t _gen_elf(char** elf_buf, std::pair<uint32_t*, size_t>& instruction_data){
        uint32_t elf_buf_size = aiebu_assembler_get_elf(
            aiebu_assembler_buffer_type_blob_instr_transaction,
            (char*) instruction_data.first, instruction_data.second * sizeof(uint32_t),
            NULL, 0, (void**)elf_buf, NULL, 0, "", "", NULL, 0);
        assert(elf_buf_size > 0);
        if (elf_buf_size == 0){
            header_print_r("ERROR", "Failed to get elf from ctrl_seq");
            exit(1);
        }
        return elf_buf_size;
    }

    ///@brief Setup the kernel
    ///@note The function will create an elf file from the ctrl_seq
    ///@note The function will also update the module, elf, and kernel
    void _setup_kernel(){
        char* elf_buf;
        this->kernel.reset();
        this->module.reset();
        this->elf.reset();
        std::pair<uint32_t*, size_t> data = this->ctrl_seq->dump();
        assert(data.first != nullptr);
        assert(data.second > 0);
        uint32_t elf_buf_size = this->_gen_elf(&elf_buf, data);
        if (this->module_valid){
            this->module.reset();
            this->elf.reset();
            this->kernel.reset();
        }
        this->elf = std::make_unique<xrt::elf>(elf_buf, elf_buf_size);
        this->module = std::make_unique<xrt::module>(*this->elf);
        this->kernel = std::make_unique<xrt::ext::kernel>(*this->context, *this->module, this->kernel_name);
        this->module_valid = true;
        this->module_version = this->ctrl_seq->sequence_version();

        free((void*)elf_buf);
    }

public:
    // enum ert_cmd_state {
    //     ERT_CMD_STATE_NEW = 1,
    //     ERT_CMD_STATE_QUEUED = 2,
    //     ERT_CMD_STATE_RUNNING = 3,
    //     ERT_CMD_STATE_COMPLETED = 4,
    //     ERT_CMD_STATE_ERROR = 5,
    //     ERT_CMD_STATE_ABORT = 6,
    //     ERT_CMD_STATE_SUBMITTED = 7,
    //     ERT_CMD_STATE_TIMEOUT = 8,
    //     ERT_CMD_STATE_NORESPONSE = 9,
    //     ERT_CMD_STATE_SKERROR = 10, //Check for error return code from Soft Kernel
    //     ERT_CMD_STATE_SKCRASHED = 11, //Soft kernel has crashed
    //     ERT_CMD_STATE_MAX, // Always the last one
    // };
    std::map<ert_cmd_state, std::string> cmd_state_map = {
        {ERT_CMD_STATE_NEW, "new"},
        {ERT_CMD_STATE_QUEUED, "queued"},
        {ERT_CMD_STATE_RUNNING, "running"},
        {ERT_CMD_STATE_COMPLETED, "completed"},
        {ERT_CMD_STATE_ERROR, "error"},
        {ERT_CMD_STATE_ABORT, "abort"},
        {ERT_CMD_STATE_SUBMITTED, "submitted"},
        {ERT_CMD_STATE_TIMEOUT, "timeout"},
        {ERT_CMD_STATE_NORESPONSE, "noresponse"},
        {ERT_CMD_STATE_SKERROR, "skerror"},
        {ERT_CMD_STATE_SKCRASHED, "skcrashed"},
        {ERT_CMD_STATE_MAX, "max"},
    };

    ///@brief Default Constructor
    ///@note Initialize the npu_app to nullptr
    npu_app() {
        this->device_gen = device_npu2;
        this->kernel = nullptr;
        this->device = nullptr;
        this->module_valid = false;
        this->module = nullptr;
        this->elf = nullptr;
        this->kernel = nullptr;
        this->module_version = UINT64_MAX;
    }

    ///@brief Constructor, this shall not invoke by user, it shall only be invoked by npu_app_manager
    ///@param device_gen the npu device
    ///@param device the pointer to the device
    ///@param context the pointer to the context
    ///@param kernel_name the name of the kernel
    ///@see xrt::device, xrt::hw_context, xrt::kernel
    npu_app(npu_device device_gen, xrt::device* device, xrt::hw_context* context, std::string kernel_name, bool enable_preemption = false):
        device_gen(device_gen), device(device), context(context), kernel_name(kernel_name), enable_preemption(enable_preemption){
        this->module_valid = false;
        this->module = nullptr;
        this->elf = nullptr;
        this->kernel = nullptr;
        this->ctrl_seq = std::make_unique<npu_sequence>(device_gen, enable_preemption);
        this->module_version = UINT64_MAX;
    }

  
    void update_ctrl_seq(){
        assert(this->ctrl_seq != nullptr);
        this->_setup_kernel();
    }

    void load_elf(std::string elf_name){
        this->ctrl_seq->clear_cmds();
        this->ctrl_seq->cmds2seq(); // just for making sure the sequence is valid
        this->elf = std::make_unique<xrt::elf>(elf_name);
        this->module = std::make_unique<xrt::module>(*this->elf);
        this->kernel = std::make_unique<xrt::ext::kernel>(*this->context, *this->module, this->kernel_name);
        this->module_valid = true;
        this->module_version = this->ctrl_seq->sequence_version(); // force sync the sequence version
    }

    void store_elf(std::string elf_name){
        char* elf_buf;
        if (this->module_valid == false) {
            this->_setup_kernel();
        }
        std::pair<uint32_t*, size_t> data = this->ctrl_seq->dump();
        uint32_t elf_buf_size = this->_gen_elf(&elf_buf, data);
        std::ofstream fout(elf_name, std::ios::binary);
        if (fout.is_open() == false) {
            header_print_r("ERROR", "Failed to open file: " << elf_name);
            exit(1);
        }
        fout.write((char*)elf_buf, elf_buf_size);
        fout.close();
        free((void*)elf_buf);
    }

    ///@brief Name used in a completion failure. Includes the xclbin kernel when known.
    std::string completion_where(const char* what) const {
        std::string where = std::string("npu_app::") + what;
        if (this->kernel_name.empty()) return where;
        return this->kernel_name + " " + where;
    }

    ///@brief Run the kernel and return its terminal state, including a failure.
    ///       Hunyuan prefill uses this so a driver timeout can be retried from a
    ///       full switch. operator() throws on any state other than completed.
    ///@param args arguments, shall be the buffers with real bo
    ///@see xrt::run
    template<typename... BoArgs>
    ert_cmd_state run_for_state(BoArgs&&... args){
        if (this->module_valid == false || this->ctrl_seq->sequence_valid() == false || this->module_version != this->ctrl_seq->sequence_version()) {
            this->_setup_kernel();
        }
        // Route through create_run so plain bytes& and bound_arg (per-call byte
        // offset, Shared-Executable Collapse) are handled identically. bound_arg
        // has no .bo(), so the old kernel->operator()(3,0,0,args.bo()...) form
        // cannot take it; create_run binds a sub-buffer for offset args.
        xrt::run run = this->create_run(std::forward<BoArgs>(args)...);
        run.start();
        return run.wait();
    }

    ///@brief Operator() for running the kernel
    ///@param args arguments, shall be the buffers with real bo
    ///@see xrt::run
    template<typename... BoArgs>
    ert_cmd_state operator()(BoArgs&&... args){
        ert_cmd_state state = this->run_for_state(std::forward<BoArgs>(args)...);
        flm::npu::require_completed_ert_state(
            flm::npu::to_ert_cmd_state(state), this->completion_where("operator()"));
        LOG_VERBOSE(2, "ending state: " << this->cmd_state_map[state]);
        return state;
    }

    
    ///@brief Operator() for running the kernel
    ///@param args arguments, shall be the buffers with real bo
    ///@see xrt::run
    template<typename... BoArgs>
    ert_cmd_state safe_run(BoArgs&&... args){
        if (this->module_valid == false || this->ctrl_seq->sequence_valid() == false || this->module_version != this->ctrl_seq->sequence_version()) {
            this->_setup_kernel();
        }
        std::array<bytes*, sizeof...(BoArgs)> bo_args = { bound_arg_buf(args)... };
        for (size_t i = 0; i < sizeof...(args); i++){
            bo_args[i]->sync_to_device();
        }
        // Route through create_run (see run_for_state) so bound_arg offsets bind
        // sub-buffers; plain bytes& args keep offset 0 (unchanged path).
        xrt::run run = this->create_run(std::forward<BoArgs>(args)...);
        run.start();
        ert_cmd_state state =
            flm::npu::wait_for_completion(run, this->completion_where("safe_run"));
        for (size_t i = 0; i < sizeof...(args); i++){
            bo_args[i]->sync_from_device();
        }
        LOG_VERBOSE(2, "ending state: " << this->cmd_state_map[state]);
        return state;
    }
    
    ///@brief Create a run object
    ///@param args arguments, shall be the buffers with real bo
    ///@return a run object for waiting
    ///@see xrt::run
    template<typename... BoArgs>
    xrt::run create_run(BoArgs&&... args){
        if (this->module_valid == false || this->ctrl_seq->sequence_valid() == false || this->module_version != this->ctrl_seq->sequence_version()) {
            this->_setup_kernel();
        }
        xrt::run run = xrt::run(*this->kernel);
        run.set_arg(0, 3);
        run.set_arg(1, 0);
        run.set_arg(2, 0);
        // Each arg is either a plain bytes&/buffer& (offset 0) or a bound_arg with
        // an explicit per-call byte offset (Shared-Executable Collapse). For a
        // non-zero offset bind a sub-buffer [offset, size); offset 0 binds the
        // parent bo directly (common path, unchanged).
        std::array<bytes*, sizeof...(BoArgs)> bo_args = { bound_arg_buf(args)... };
        std::array<size_t, sizeof...(BoArgs)> bo_offs = { bound_arg_off(args)... };
        for (size_t i = 0; i < sizeof...(args); i++){
            if (bo_offs[i] == 0) {
                run.set_arg(3 + i, bo_args[i]->bo());
            } else {
                xrt::bo& parent = bo_args[i]->bo();
                assert(bo_offs[i] <= parent.size());
                run.set_arg(3 + i, xrt::bo(parent, parent.size() - bo_offs[i], bo_offs[i]));
            }
        }
        return run;
    }

    ///@brief Create a buffer with real bo
    ///@param size size of the buffer
    ///@see buffer
    template<typename T>
    buffer<T> create_bo_buffer(size_t size){
        assert(size > 0);
        LOG_VERBOSE(2, "Creating buffer buffer with size: " << size);
        return buffer<T>(*this->device, size);
    }

    npu_sequence* seq() {
        return this->ctrl_seq.get();
    }
};

class npu_app_manager {
private:
    npu_device device_gen;
    bool enable_preemption;
    xrt::device* device;
    std::unique_ptr<xrt::hw_context> context;
    std::string kernel_name;
    std::string xclbin_name;
    bool xclbin_valid;
public:
    ///@brief Default Constructor
    ///@param device_gen the npu device
    ///@param device device object
    ///@param xclbin_name name of the xclbin file
    ///@see xrt::device, xrt::xclbin
    ///@note The function will initialize the npu_app_manager to nullptr
    npu_app_manager(){
        this->device_gen = device_npu2;
        this->device = nullptr;
        this->context = nullptr;
        this->kernel_name = "";
        this->xclbin_name = "";
        this->xclbin_valid = false;
        this->enable_preemption = false;
    }

    ///@brief Copy constructor, this shall not invoke by user, it shall only be invoked by npu_xclbin_manager
    ///@param other the other npu_app_manager to copy from
    npu_app_manager(const npu_app_manager& other) {
        this->device_gen = other.device_gen;
        this->device = other.device;
        this->kernel_name = other.kernel_name;
        this->xclbin_name = other.xclbin_name;
        this->xclbin_valid = other.xclbin_valid;
        this->enable_preemption = other.enable_preemption;
        if (other.context) {
            this->context = std::make_unique<xrt::hw_context>(*other.context);
        } else {
            this->context = nullptr;
        }
    }

    ///@brief Copy assignment operator
    ///@param other the other npu_app_manager to copy from
    ///@return reference to this object
    npu_app_manager& operator=(const npu_app_manager& other) {
        if (this != &other) {
            this->device_gen = other.device_gen;
            this->device = other.device;
            this->kernel_name = other.kernel_name;
            this->xclbin_name = other.xclbin_name;
            this->xclbin_valid = other.xclbin_valid;
            this->enable_preemption = other.enable_preemption;
            if (other.context) {
                this->context = std::make_unique<xrt::hw_context>(*other.context);
            } else {
                this->context = nullptr;
            }
        }
        return *this;
    }

    ///@brief Move constructor
    ///@param other the other npu_app_manager to move from
    npu_app_manager(npu_app_manager&& other) noexcept {
        this->device_gen = other.device_gen;
        this->device = other.device;
        this->context = std::move(other.context);
        this->kernel_name = std::move(other.kernel_name);
        this->xclbin_name = std::move(other.xclbin_name);
        this->xclbin_valid = other.xclbin_valid;
        this->enable_preemption = other.enable_preemption;
        // Reset the moved-from object
        other.device = nullptr;
        other.xclbin_valid = false;
    }

    ///@brief Move assignment operator
    ///@param other the other npu_app_manager to move from
    ///@return reference to this object
    npu_app_manager& operator=(npu_app_manager&& other) noexcept {
        if (this != &other) {
            this->device_gen = other.device_gen;
            this->device = other.device;
            this->context = std::move(other.context);
            this->kernel_name = std::move(other.kernel_name);
            this->xclbin_name = std::move(other.xclbin_name);
            this->xclbin_valid = other.xclbin_valid;
            this->enable_preemption = other.enable_preemption;
            // Reset the moved-from object
            other.device = nullptr;
            other.xclbin_valid = false;
        }
        return *this;
    }

    ///@brief Constructor, this shall not invoke by user, it shall only be invoked by npu_xclbin_manager
    ///@param device_gen the npu device
    ///@param device device object
    ///@param xclbin_name name of the xclbin file
    ///@see xrt::device, xrt::xclbin
    npu_app_manager(npu_device device_gen, xrt::device* device, std::string xclbin_name, bool enable_preemption = false){
        assert(device != nullptr);
        assert(xclbin_name != "");
        this->device_gen = device_gen;
        this->device = device;
        this->xclbin_name = xclbin_name;
        this->enable_preemption = enable_preemption;
        #ifndef __WINDOWS__
        if (this->enable_preemption){
            header_print("warning", "Preemption is not supported on Linux host currently.");
            header_print("warning", "Preemption is set as disabled for NPU.");
            this->enable_preemption = false;
        }
        #endif
        LOG_VERBOSE(2, "Loading xclbin: " << xclbin_name);
        auto this_xclbin = xrt::xclbin(xclbin_name);
        // int verbosity = VERBOSE;
        std::string Node = "MLIR_AIE";
        auto xkernels = this_xclbin.get_kernels();
        auto xkernel = *std::find_if(
            xkernels.begin(), 
            xkernels.end(),
            [Node](xrt::xclbin::kernel &k) {
                auto name = k.get_name();
                return name.rfind(Node, 0) == 0;
            }
        );
        this->device->register_xclbin(this_xclbin);
        auto kernelName = xkernel.get_name();
        this->context = std::make_unique<xrt::hw_context>(*this->device, this_xclbin.get_uuid());
        this->kernel_name = kernelName;
        this->xclbin_valid = true;
        LOG_VERBOSE(2, "Xclbin: " << xclbin_name << " loaded successfully!");
    }

    ~npu_app_manager() = default;

    ///@brief Create a npu_app
    ///@return a npu_app object
    ///@see npu_app
    npu_app create_app(){
        assert(this->xclbin_valid);
        return npu_app(this->device_gen, this->device, this->context.get(), this->kernel_name, this->enable_preemption);
    }

    ///@brief Create a buffer with real bo
    ///@param size size of the buffer
    ///@see buffer
    template<typename T>
    buffer<T> create_bo_buffer(size_t size){
        assert(size > 0);
        LOG_VERBOSE(2, "Creating buffer buffer with size: " << size);
        return buffer<T>(*this->device, size);
    }

    ///@brief Get the name of the xclbin
    ///@return the name of the xclbin
    std::string get_xclbin_name(){
        assert(this->xclbin_valid);
        return this->xclbin_name;
    }

    ///@brief Create a runlist
    ///@return a runlist object
    ///@see xrt::runlist
    xrt::runlist create_runlist(){
        assert(this->xclbin_valid);
        assert(this->enable_preemption == false); // preemption is not supported for runlist
        return xrt::runlist(*this->context);
    }
};

///@brief npu_xclbin_manager
///@note There should be only one npu_xclbin_manager inside main.
///@note It handles all xclbins
class npu_xclbin_manager{

private:
    std::vector<std::unique_ptr<npu_app_manager>> npu_xclbins;

    size_t xclbin_count;
    // the only device instance
    xrt::device* device;
    bool enable_preemption;
    npu_device npu_gen;
public:
    constexpr static int max_xclbins = 16; // This is hard constraint from the XRT driver
    
    ///@brief Constructor, this shall not invoke by user, it shall only be invoked by main
    ///@param device the npu device
    ///@param device_id the device id
    ///@see xrt::device
    npu_xclbin_manager(npu_device device = device_npu2, xrt::device* device_inst = nullptr, bool enable_preemption = false){
        this->device = device_inst;
        // this->npu_xclbins.resize(max_xclbins);
        this->npu_xclbins.reserve(max_xclbins);
        this->xclbin_count = 0;
        this->npu_gen = device;
        this->enable_preemption = enable_preemption;
    }

    ~npu_xclbin_manager() = default;

    ///@brief register an accel_user_desc to the npu_manager
    ///@param xclbin_name the name of the xclbin
    ///@return the npu_app_manager object
    ///@see npu_app_manager
    ///@note Different apps may have the same xclbin, but the sequence is unique.
    ///@note To avoid creating duplicated applications, the function checks if the xclbin is registered.
    ///@note If the xclbin is not registered, the function will register the xclbin and create a new application.
    npu_app_manager* register_xclbin(std::string xclbin_name){
        assert(xclbin_name != "");
        int xclbin_id = -1;
        for (size_t i = 0; i < this->xclbin_count; i++){
    
            if (this->npu_xclbins[i]->get_xclbin_name() == xclbin_name){
                xclbin_id = i;
                break;
            }
        }
        LOG_VERBOSE_IF_ELSE(2, xclbin_id > -1, 
            "Found xclbin: " << xclbin_name << "registered as id " << xclbin_id << "!",
            "Xclbin: " << xclbin_name << " not registered yet!"
        );
    
        if (xclbin_id == -1){ // the xclbin is not registered yet
            if (this->xclbin_count >= max_xclbins){
                throw std::runtime_error("Max number of xclbins reached");
            }
            this->npu_xclbins.push_back(std::make_unique<npu_app_manager>(this->npu_gen, this->device, xclbin_name, this->enable_preemption));
            xclbin_id = this->xclbin_count;
            LOG_VERBOSE(2, "Xclbin: " << xclbin_name << " registered as id " << xclbin_id << "!");
            this->xclbin_count++;
            xclbin_id = this->xclbin_count - 1;
        }
        return this->npu_xclbins[xclbin_id].get();
    }

    void list_xclbins(){
        for (size_t i = 0; i < this->xclbin_count; i++){
            std::cout << "Xclbin " << i << " name: " << this->npu_xclbins[i]->get_xclbin_name() << std::endl;
        }
    }

    /// @brief Helper function to write out the trace to a file
    /// @param traceOutPtr pointer to the trace
    /// @param trace_size size of the trace
    /// @param path path to the file
    static void write_out_trace(char *traceOutPtr, size_t trace_size, std::string path) {
        std::ofstream fout(path);
        LOG_VERBOSE(1, "Writing out trace to: " << path);
        uint32_t *traceOut = (uint32_t *)traceOutPtr;
        for (size_t i = 0; i < trace_size / sizeof(traceOut[0]); i++) {
          fout << std::setfill('0') << std::setw(8) << std::hex << (int)traceOut[i];
          fout << std::endl;
        }
        fout.close();
        LOG_VERBOSE(1, "Trace written successfully!");
    }
    
    ///@brief Create a buffer with real bo
    ///@param size size of the buffer
    ///@see buffer
    template<typename T>
    buffer<T> create_bo_buffer(size_t size){
        assert(size > 0);
        LOG_VERBOSE(2, "Creating buffer buffer with size: " << size);
        return buffer<T>(*this->device, size);
    }

    inline bool is_preemption_enabled(){
        return this->enable_preemption;
    }

    #ifndef __WINDOWS__
    ///@brief print the npu information
    ///@note The function will print the npu version, clock frequency, column count, row count, core info, mem info, shim info.
    ///@note The information is read via the IOCTL interface.
    void print_npu_info(){
        int fd = open("/dev/accel/accel0", O_RDWR);
        if (fd < 0) {
            perror("Failed to open amdgpu device");
            return;
        }
        amdxdna_drm_query_clock_metadata query_clock_metadata;
        amdxdna_drm_get_info get_info = {
            .param = DRM_AMDXDNA_QUERY_CLOCK_METADATA,
            .buffer_size = sizeof(amdxdna_drm_query_clock_metadata),
            .buffer = (unsigned long)&query_clock_metadata,
        };
        int ret = ioctl(fd, DRM_IOCTL_AMDXDNA_GET_INFO, &get_info);
        if (ret < 0) {
            std::cout << "Error code: " << ret << std::endl;
            perror("Failed to get telemetry information");
            close(fd);
            return;
        }

        amdxdna_drm_query_aie_metadata query_aie_metadata = {};
        get_info.param = DRM_AMDXDNA_QUERY_AIE_METADATA;
        get_info.buffer_size = sizeof(amdxdna_drm_query_aie_metadata);
        get_info.buffer = (unsigned long)&query_aie_metadata;
        ret = ioctl(fd, DRM_IOCTL_AMDXDNA_GET_INFO, &get_info);
        if (ret < 0) {
            std::cout << "Error code: " << ret << std::endl;
            perror("Failed to get telemetry information");
            close(fd);
            return;
        }

        close(fd);
        MSG_BONDLINE(40);
        MSG_BOX_LINE(40, "NPU version: " << query_aie_metadata.version.major << "." << query_aie_metadata.version.minor);
        MSG_BOX_LINE(40, "MP-NPU clock frequency: " << query_clock_metadata.mp_npu_clock.freq_mhz << " MHz");
        MSG_BOX_LINE(40, "H clock frequency: " << query_clock_metadata.h_clock.freq_mhz << " MHz");
        // What is the meaning of the column size?
        // std::cout << "NPU column size: " << query_aie_metadata.col_size << std::endl;
        MSG_BOX_LINE(40, "NPU column count: " << query_aie_metadata.cols);
        MSG_BOX_LINE(40, "NPU row count: " << query_aie_metadata.rows);
        MSG_BOX_LINE(40, "NPU core Info: ");
        MSG_BOX_LINE(40, "--Row count: " << query_aie_metadata.core.row_count);
        MSG_BOX_LINE(40, "--Row start: " << query_aie_metadata.core.row_start);
        MSG_BOX_LINE(40, "--DMA channel count: " << query_aie_metadata.core.dma_channel_count);
        MSG_BOX_LINE(40, "--Lock count: " << query_aie_metadata.core.lock_count);
        MSG_BOX_LINE(40, "--Event reg count: " << query_aie_metadata.core.event_reg_count);
        MSG_BOX_LINE(40, "NPU mem Info: ");
        MSG_BOX_LINE(40, "--Row count: " << query_aie_metadata.mem.row_count);
        MSG_BOX_LINE(40, "--Row start: " << query_aie_metadata.mem.row_start);
        MSG_BOX_LINE(40, "--DMA channel count: " << query_aie_metadata.mem.dma_channel_count);
        MSG_BOX_LINE(40, "--Lock count: " << query_aie_metadata.mem.lock_count);
        MSG_BOX_LINE(40, "--Event reg count: " << query_aie_metadata.mem.event_reg_count);
        MSG_BOX_LINE(40, "NPU shim Info: ");
        MSG_BOX_LINE(40, "--Row count: " << query_aie_metadata.shim.row_count);
        MSG_BOX_LINE(40, "--Row start: " << query_aie_metadata.shim.row_start);
        MSG_BOX_LINE(40, "--DMA channel count: " << query_aie_metadata.shim.dma_channel_count);
        MSG_BOX_LINE(40, "--Lock count: " << query_aie_metadata.shim.lock_count);
        MSG_BOX_LINE(40, "--Event reg count: " << query_aie_metadata.shim.event_reg_count);
        MSG_BONDLINE(40);
    }

    ///@brief get the npu power consumption
    ///@param print whether to print the power consumption
    ///@return the power consumption, unit is Watt
    float get_npu_power(bool print){
        // get the npu power consumption, unit is Watt
        int fd = open("/dev/accel/accel0", O_RDWR);
        if (fd < 0) {
            perror("Failed to open amdgpu device");
            return -1;
        }
        amdxdna_drm_query_sensor query_sensor;

        amdxdna_drm_get_info get_info = {
            .param = DRM_AMDXDNA_QUERY_SENSORS,
            .buffer_size = sizeof(amdxdna_drm_query_sensor),
            .buffer = (unsigned long)&query_sensor,
        };
        int ret = ioctl(fd, DRM_IOCTL_AMDXDNA_GET_INFO, &get_info);
        if (ret < 0) {
            std::cout << "Error code: " << ret << std::endl;
            perror("Failed to get telemetry information");
            close(fd);
            return -1;
        }
        if (print){
            MSG_BOX(40, "NPU power: " << query_sensor.input << " " << query_sensor.units);
        }
        close(fd);
        return (float)query_sensor.input * pow(10, query_sensor.unitm);
    }
    #endif
};
#endif
#include "flm_runtime.hpp"
