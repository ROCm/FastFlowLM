// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file npu_utils.hpp
/// \brief npu_utils class
/// \author FastFlowLM Team
/// \date 2025-06-24
/// \version 0.9.9
/// \note This file contains the npu_utils class
#pragma once

#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdio.h>
#ifdef __LINUX__
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
#ifdef __LINUX__

#include "xrt/experimental/xrt_kernel.h"
#include "xrt/experimental/xrt_ext.h"
#include "xrt/experimental/xrt_module.h"
#include "xrt/experimental/xrt_elf.h"

#else

#include "experimental/xrt_kernel.h"
#include "experimental/xrt_ext.h"
#include "experimental/xrt_module.h"
#include "experimental/xrt_elf.h"
#endif

#include "npu_instr_utils.hpp"

///@brief accel_user_desc
///@param xclbin_name name of the xclbin file
///@param instr_seq instruction sequence, an object of npu_sequence
///@see npu_sequence
typedef struct {
    std::string xclbin_name;
    std::string app_name;
} npu_app_desc;

///@brief accel_xclbin_desc
///@param xclbin xclbin object
///@param kernel kernel object
///@param context hardware context
///@see xrt::xclbin, xrt::kernel, xrt::hw_context
typedef struct {
    xrt::xclbin xclbin;
    xrt::kernel kernel;
    xrt::hw_context context;
} accel_xclbin_desc;

///@brief accel_kernel_desc
///@param app_name name of the application
///@param kernel_desc kernel descriptor
///@param instr_seq instruction sequence
typedef struct {
    std::string app_name;
    std::unique_ptr<accel_xclbin_desc> xclbin_desc;
    npu_sequence instr_seq;
} accel_kernel_desc;

///@brief npu_app
///@param app_id application id
///@param instr_seq instruction sequence
///@param kernel kernel object
///@param device device object
///@see npu_sequence, xrt::kernel, xrt::device
class npu_app {
public:
    int app_id;
    npu_sequence* instr_seq;
private:
    xrt::kernel* kernel;
    xrt::device* device;
public:
    ///@brief Constructor
    ///@note Initialize the npu_app
    npu_app() {
        this->instr_seq = nullptr;
        this->kernel = nullptr;
        this->device = nullptr;
        this->app_id = -1;
    }

    ///@brief Constructor
    ///@param app_id application id
    ///@param instr_seq instruction sequence
    ///@param kernel kernel object
    ///@param device device object
    npu_app(int app_id, npu_sequence* instr_seq, xrt::kernel* kernel, xrt::device* device):
        app_id(app_id), instr_seq(instr_seq), kernel(kernel), device(device){}

    ///@brief Operator()
    ///@param args arguments
    ///@see xrt::run
    template<typename... BoArgs>
    void operator()(BoArgs&&... args){
        auto run = this->kernel->operator()(3, this->instr_seq->bo(), this->instr_seq->size(), args.bo()...);
        run.wait();
    }

    ///@brief Create a run
    ///@param args arguments
    ///@see xrt::run
    template<typename... BoArgs>
    xrt::run create_run(BoArgs&&... args){
        xrt::run run = xrt::run(*this->kernel);
        run.set_arg(0, 3);
        run.set_arg(1, this->instr_seq->bo());
        run.set_arg(2, this->instr_seq->size());
        std::array<bytes*, sizeof...(BoArgs)> bo_args = { &args... };
        for (size_t i = 0; i < sizeof...(args); i++){
            run.set_arg(3 + i, bo_args[i]->bo());
        }
        return run;
    }

    ///@brief Create a buffer
    ///@param size size of the buffer
    ///@param group_id group id
    ///@see buffer
    template<typename T>
    buffer<T> create_bo_buffer(size_t size, int group_id){
        assert(size > 0);
        assert(group_id >= 3);
        assert(group_id < 8);
        LOG_VERBOSE(2, "Creating buffer buffer with size: " << size << " and group_id: " << group_id);
        return buffer<T>(size, *this->device, *this->kernel, group_id);
    }
};



///@brief npu_manager
///@note There should be only one npu_manager inside main.
///@note It handles all xclbins and instr_sequences.
///@note Each xclbin may have multiple instr_sequences.
///@note Each xclbin and instr_sequence has a unique id.
///@note Both id shall be provided to run an accelerator.
///@note Therefore, the xclbin_name between different accel_descriptions may overlap, but the instr_name is unique.
class npu_manager{
public:
    constexpr static int max_xclbins = 16; // This is hard constraint from the XRT driver
    constexpr static int max_kernels = 64; // This is hard constraint from the XRT driver
    
    npu_manager(npu_device device = device_npu2, unsigned int device_id = 0U);

    npu_app create_app(npu_app_desc& desc);
    // ~npu_manager();
    int _load_xclbin(std::string xclbin_name);

    xrt::runlist create_runlist(npu_app& app);
    
    void list_kernels();
    void write_out_trace(char *traceOutPtr, size_t trace_size, std::string path);
    #ifdef __LINUX__
    void print_npu_info();
    float get_npu_power(bool print = true);
    #endif

private:
    std::vector<accel_xclbin_desc> xclbin_descs;
    std::vector<accel_kernel_desc> kernel_descs;
    std::vector<std::string> registered_xclbin_names;

    size_t kernel_desc_count;
    size_t xclbin_desc_count;

    // the only device instance
    xrt::device device;

    npu_device npu_gen;
};