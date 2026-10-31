/// \file options.hpp
/// \brief options file for the FastFlowLM project
/// \author FastFlowLM Team
/// \date 2026-02-24
/// \version 0.9.26
/// \note This file contains a struct for passing all user arguments from command line
/// \note This is to avoid keep add arguments to runner and serve
#pragma once

#include <string>

struct program_args_t {
    // common commands
    std::string command = "version";
    std::string model_tag = "model-faker";
    std::string power_mode = "performance";
    bool preemption = false;
    bool asr = false;
    bool embed = false;
    std::string decision = ""; // --decision <tag>: serve a decision model (POST /v1/systemone)
    bool json_output = false;
    int ctx_length = -1; // let model decide
    std::string backend = ""; // let the catalog decide; see AutoModel/model_backend.hpp
    int prefill_chunk_len = -1; // let model decide
    int prefix_cache_mb = -1;   // host memory (MiB) for saved prompt snapshots; -1 = the model's default
    int prefix_slots = -1;      // how many prompt snapshots to keep; -1 = the model's default, 0 = off
    int prefix_min_gain = -1;   // tokens a new end snapshot must add over a saved one covering its prompt; -1 = default (128), 0 = no minimum

    // handling input file
    std::string input_file_name = "";
    int iterations = 2;

    // specific commands
    int img_pre_resize = 3;

    // for list command
    std::string list_filter = "all";

    // for pull command
    bool force_redownload = false;
    
    // for download related command
    bool modelscope = false;

    // for serve command
    std::string host = "127.0.0.1";
    size_t max_socket_connections = 10;
    size_t max_npu_queue = 10;
    int port = -1; // default port
    bool cors = false;
    bool sub_process_mode = false;
    
    program_args_t() {}
};