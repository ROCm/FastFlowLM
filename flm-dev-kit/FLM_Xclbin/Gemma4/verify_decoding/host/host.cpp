// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "typedef.hpp"
#include "buffer.hpp"
#include "decoding_layer.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "utils/debug_utils.hpp"
#include "utils/vm_args.hpp"
#include "utils/utils.hpp"
#include "models.h"
#include "metrices.hpp"
#include "tensor_2d.hpp"
#include "lm_config.hpp"

#include <iostream>

#include "test_swa_non_skip.hpp"
#include "test_swa_skip.hpp"
#include "test_global_non_skip.hpp"
#include "test_global_skip.hpp"
#include "test_swa_non_skip_detail.hpp"
#include "test_verify8.hpp"
#include "test_proj8.hpp"
#include "test_rmsglu.hpp"
#include "test_ple8.hpp"
#include "test_fused8.hpp"
#include <sstream>

int main(int argc, char** argv) {
    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    arg_utils::add_default_options(desc);
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->default_value("GEMMA4_E2B"), "Model type, e.g., GEMMA4_E2B");
    desc.add_options()("layer,l", arg_utils::po::value<int>()->default_value(0), "Layer number to test, 0 - 35");
    desc.add_options()("test_seq_len,s", arg_utils::po::value<int>()->default_value(10), "Test sequence length for global skip layer");

    desc.add_options()("test,t", arg_utils::po::value<std::string>()->default_value("legacy"), "legacy | verify8 | chain8 | hammer8 | proj8 | rmsglu | ple8");
    desc.add_options()("layers", arg_utils::po::value<std::string>()->default_value("0,5,40,35"), "verify8: comma list of layers");
    desc.add_options()("cursors", arg_utils::po::value<std::string>()->default_value(""), "verify8: comma list of cursors (default: the S6.2 sweep)");
    desc.add_options()("repeat", arg_utils::po::value<int>()->default_value(1), "verify8: repeats of the 8-row path (must be bit-identical)");
    desc.add_options()("poison", "verify8: poison K/V rows above the cursor with NaN");
    desc.add_options()("timeout", arg_utils::po::value<int>()->default_value(20000), "dispatch timeout in ms");
    desc.add_options()("mode", arg_utils::po::value<std::string>()->default_value("oracle"), "verify8: oracle | verify");
    desc.add_options()("xclbin", arg_utils::po::value<std::string>()->default_value(""), "override the layer xclbin path");
    desc.add_options()("cos-tol", arg_utils::po::value<float>()->default_value(0.9998f), "verify8: min cosine per row (measured decode-vs-decode floor: 0.99988 worst, 0.99996 typical)");
    desc.add_options()("ulp-tol", arg_utils::po::value<float>()->default_value(6.0f), "verify8: max |err| in bf16 ulps of the row max (measured decode-vs-decode floor: 4.9 worst, 1-2 typical)");
    desc.add_options()("rows", arg_utils::po::value<int>()->default_value(8), "proj8: token rows per dispatch (1 = single-token core, 8 = 8-token core)");
    desc.add_options()("dir", arg_utils::po::value<std::string>()->default_value("proj8_data"), "proj8: directory with x_<phase>.bin, receives y_<phase>.bin");
    desc.add_options()("phases", arg_utils::po::value<std::string>()->default_value("qkv,o,up,down"), "proj8: phases to run");
    desc.add_options()("stage_name", arg_utils::po::value<std::string>()->default_value("full"), "ple8: gate | up | full (debug xclbins)");
    desc.add_options()("taps", "rmsglu: the xclbin was built with RMSGLU_TAPS=1, also capture the raw o/down/up|gate rounds");
    desc.add_options()("stage", "proj8: xclbin is the projstage design (y via the MT4 token-major stage and IT4)");
    desc.add_options()("fused-xclbin", arg_utils::po::value<std::string>()->default_value(""), "verify8 --mode verify: the fused8 verify xclbin (build/fused8_<stage>_L8/xclbins/fused8.xclbin)");
    desc.add_options()("fused-stage", arg_utils::po::value<std::string>()->default_value("full"), "fused8: attn | nople | full (what the xclbin outputs)");
    desc.add_options()("maxl", arg_utils::po::value<int>()->default_value(1536), "verify8: MAX_L of the KV cache");
    desc.add_options()("model_dir", arg_utils::po::value<std::string>()->default_value(""), "override the q4nx model directory");
    const int max_len = 1;

    arg_utils::parse_options(argc, argv, desc, vm);

    std::string model_type = vm["model"].as<std::string>();
    int layer_idx = vm["layer"].as<int>();
    int test_seq_len = vm["test_seq_len"].as<int>();


    header_print("info", "Testing model: " + model_type);

    model_list::model_t model  = model_list::get_model_type(model_type);
    std::string model_path     = model_list::get_model_path(model);
    if (!vm["model_dir"].as<std::string>().empty()) model_path = vm["model_dir"].as<std::string>();
    std::string reference_path = model_list::get_reference_path(model);
    header_print("info", "Model path: " + model_path);
    LM_Config lm_config;
    lm_config.from_pretrained(model_path);

    int D = lm_config.hidden_size;
    int DH = lm_config.head_dim;
    int DQ = lm_config.num_attention_heads * lm_config.head_dim;
    int DK = lm_config.num_key_value_heads * lm_config.head_dim;
    int DV = lm_config.num_key_value_heads * lm_config.head_dim;
    // overide the layer xclbin

    lm_config.layer_xclbin_name = model_list::get_layer_xclbin_path(model);
    if (!vm["xclbin"].as<std::string>().empty()) lm_config.layer_xclbin_name = vm["xclbin"].as<std::string>();
    {   // S6.1: RTP addresses come from the build next to the xclbin
        std::string x = lm_config.layer_xclbin_name; auto sl = x.find_last_of('/');
        std::string base = sl == std::string::npos ? x : x.substr(sl + 1);
        std::string stem = base.substr(0, base.find_last_of('.'));
        decoding_layer::rtp_file = (sl == std::string::npos ? std::string(".") : x.substr(0, sl)) + "/" + (stem == "layer" ? std::string("rtp_addresses.txt") : stem + "_rtp_addresses.txt");
    }
    decoding_layer::dispatch_timeout_ms = vm["timeout"].as<int>();
    header_print("info", "Layer xclbin path: " + lm_config.layer_xclbin_name);
    header_print("info", "Hidden size (D): " + std::to_string(D));


    // std::unique_ptr<Q4NX> q4nx_model = std::make_unique<Q4NX>(model_path);

    // buffer<bf16> embed;

    // q4nx_model->load_weights(embed, "model.embed_tokens.weight");

    // header_print("info", "Embedding weights:");
    // utils::print_matrix(embed, D, 10);
    header_print("info", "Testing decoding layer...");
    header_print("info", "Initializing NPU manager...");
    
    xrt::device npu_inst = xrt::device(0);
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_inst, false);
    npu_app_manager* layer_app_mgr = xclbin_mgr.register_xclbin(lm_config.layer_xclbin_name);
    // std::unique_ptr<SafeTensors> references = std::make_unique<SafeTensors>(reference_path);

    std::string ref_path = model_list::get_reference_path(model);

    std::unique_ptr<Q4NX> q4nx = std::make_unique<Q4NX>(model_path);
    if (vm["test"].as<std::string>() == "proj8") {
        proj8_opts po;
        po.layer = vm["layer"].as<int>(); po.rows = vm["rows"].as<int>(); po.dir = vm["dir"].as<std::string>(); po.phases = vm["phases"].as<std::string>();
        po.repeat = vm["repeat"].as<int>(); po.timeout_ms = vm["timeout"].as<int>(); po.stage = vm.count("stage") > 0;
        int rc = test_proj8(model, layer_app_mgr, lm_config, *q4nx, po);
        q4nx.reset();
        return rc;
    }
    if (vm["test"].as<std::string>() == "ple8") {
        ple8_opts pe;
        pe.layer = vm["layer"].as<int>(); pe.dir = vm["dir"].as<std::string>(); pe.repeat = vm["repeat"].as<int>(); pe.timeout_ms = vm["timeout"].as<int>(); pe.stage = vm["stage_name"].as<std::string>();
        int rc = test_ple8(model, layer_app_mgr, lm_config, *q4nx, pe);
        q4nx.reset();
        return rc;
    }
    if (vm["test"].as<std::string>() == "rmsglu") {
        rg_opts ro;
        ro.layer = vm["layer"].as<int>(); ro.dir = vm["dir"].as<std::string>(); ro.repeat = vm["repeat"].as<int>(); ro.timeout_ms = vm["timeout"].as<int>(); ro.taps = vm.count("taps") > 0;
        int rc = test_rmsglu(model, layer_app_mgr, lm_config, *q4nx, ro);
        q4nx.reset();
        return rc;
    }
    if (!vm["fused-xclbin"].as<std::string>().empty()) {
        f8::g_mgr = &xclbin_mgr; f8::g_xclbin = vm["fused-xclbin"].as<std::string>(); f8::g_stage = vm["fused-stage"].as<std::string>();
        f8::g_timeout_ms = vm["timeout"].as<int>(); f8::g_debug = std::getenv("FUSED8_DEBUG") != nullptr;
        g_verify_hook = [](decoding_layer& l, int c, const std::vector<buffer<bf16>>& x, const std::vector<int>& ids, std::vector<buffer<bf16>>& out) { return f8::run(l, c, x, ids, out); };
    }
    if (vm["test"].as<std::string>() == "hammer8") {
        auto split = [](const std::string& s) { std::vector<int> v; std::stringstream ss(s); std::string t; while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(std::stoi(t)); return v; };
        f8::hammer_opts ho;
        ho.layers = split(vm["layers"].as<std::string>());
        if (!vm["cursors"].as<std::string>().empty()) ho.cursors = split(vm["cursors"].as<std::string>());
        ho.count = vm["repeat"].as<int>(); ho.maxl = vm["maxl"].as<int>();
        int rc = f8::test_hammer8(model, layer_app_mgr, lm_config, *q4nx, ho);
        q4nx.reset();
        return rc;
    }
    if (vm["test"].as<std::string>() == "chain8") {
        auto split = [](const std::string& s) { std::vector<int> v; std::stringstream ss(s); std::string t; while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(std::stoi(t)); return v; };
        f8::chain_opts co;
        if (!vm["cursors"].as<std::string>().empty()) co.cursors = split(vm["cursors"].as<std::string>());
        co.seqs = vm["repeat"].as<int>(); co.maxl = vm["maxl"].as<int>();
        if (const char* nl = std::getenv("CHAIN_LAYERS")) co.layers = std::atoi(nl);
        int rc = f8::test_chain8(model, layer_app_mgr, lm_config, *q4nx, co);
        q4nx.reset();
        return rc;
    }
    if (vm["test"].as<std::string>() == "verify8") {
        auto split = [](const std::string& s) { std::vector<int> v; std::stringstream ss(s); std::string t; while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(std::stoi(t)); return v; };
        v8_opts o;
        o.mode = vm["mode"].as<std::string>();
        o.layers = split(vm["layers"].as<std::string>());
        if (!vm["cursors"].as<std::string>().empty()) o.cursors = split(vm["cursors"].as<std::string>());
        o.cos_tol = vm["cos-tol"].as<float>(); o.ulp_tol = vm["ulp-tol"].as<float>(); o.repeat = vm["repeat"].as<int>(); o.poison = vm.count("poison") > 0; o.maxl = vm["maxl"].as<int>();
        int rc = test_verify8(model, layer_app_mgr, lm_config, *q4nx, o);
        q4nx.reset();
        return rc;
    }
    test_swa_non_skip_layer(model, layer_app_mgr, lm_config, *q4nx, ref_path);
    test_global_non_skip_layer(model, layer_app_mgr, lm_config, *q4nx, ref_path);
    test_swa_skip_layer(model, layer_app_mgr, lm_config, *q4nx, ref_path);
    test_global_skip_layer(model, layer_app_mgr, lm_config, *q4nx, ref_path, test_seq_len);
    q4nx.reset();
    return 0;
}
    