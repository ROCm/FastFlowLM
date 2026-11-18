// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <boost/program_options.hpp>
#include <iostream>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "utils/vm_args.hpp"
#include "utils/debug_utils.hpp"
#include "utils/utils.hpp"
#include "npu_utils/npu_utils.hpp"

#include "SmolVLA/SmolVLA.hpp"
#include "SmolVLA/vlm/vision/SmolVLA_vision_def.hpp"

#include "tensor_utils/safe_tensors.hpp"
#include "metrices.hpp"

namespace po = boost::program_options;

/// @brief report whether OpenMP is available; if not, warn and wait for enter.
static void check_openmp();

/// @brief read a ref.safetensors tensor as a flat float vector (BF16 or F32).
static std::vector<float> ref_read_float(SafeTensors& ref, const std::string& name);

/// @brief round a float vector to bf16 and wrap in an owning buffer.
static buffer<bf16> to_bf16_buffer(const std::vector<float>& v);

/// @brief compare a computed float tensor against a ref tensor (both -> bf16).
static void compare_ref(const std::string& label,
                        const std::vector<float>& got_f,
                        SafeTensors& ref, const std::string& ref_name);

/// @brief compare two float tensors already in hand (both -> bf16).
static void compare_vec(const std::string& label,
                        const std::vector<float>& got_f,
                        const std::vector<float>& ref_f);

int main(int argc, char **argv) {
    // Fix the seed to ensure reproducibility in CI.
    srand(0);

    // Warn (and wait) if the CPU kernels have no OpenMP thread team.
    check_openmp();

    // Get Username for cross user
    const char* user_env = std::getenv("USER");
    std::string user_name = user_env ? user_env : "michyu";

    // Program arguments parsing
    po::options_description desc("Allowed options");
    po::variables_map vm;
    arg_utils::add_default_options(desc);

    // Add custom options
    desc.add_options()("model_name,n", po::value<std::string>()->default_value("SmolVLA"), "Model name");
    desc.add_options()("iterations,i", po::value<int>()->default_value(1), "Number of iterations");
    desc.add_options()("skip,s", po::value<bool>()->default_value(false), "Skip image part, use reference data");
    arg_utils::parse_options(argc, argv, desc, vm);

    // Initialize NPU
    xrt::device npu_device = xrt::device(0); // NPU device should always be 0
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);

    // User logic
    const std::string model_name = vm["model_name"].as<std::string>();
    const int iterations = vm["iterations"].as<int>();
    const bool skip_prefill = vm["skip"].as<bool>();

    std::string model_path = utils::path_join("/scratch", user_name, "models", model_name);

    header_print("info", "model path: " << model_path);
    header_print("info", "iterations: " << iterations);

    SmolVLA model(&xclbin_mgr);
    model.from_pretrained(model_path);
    model.load_weights();

    const SmolVLA_Expert_def& edef = model.expert->def;
    header_print("info", "expert D:          " << edef.D);
    header_print("info", "expert DQ/DK/DV:   " << edef.DQ << "/" << edef.DK << "/" << edef.DV);
    header_print("info", "expert heads/kv:   " << edef.num_attn_heads << "/" << edef.num_kv_heads);
    header_print("info", "expert head_dim:   " << edef.head_dim);
    header_print("info", "expert inter_size: " << edef.intermediate_size);
    header_print("info", "expert num_layers: " << edef.num_layers);
    header_print("info", "expert chunk_size: " << edef.chunk_size);
    header_print("info", "expert cross_layer bytes: " << edef.cross_layer_byte_size);
    header_print("info", "expert self_layer bytes:  " << edef.self_layer_byte_size);
    header_print("info", "expert non_layer bytes:   " << edef.non_layer_byte_size);

    const SmolVLA_Vision_def& vdef = model.vlm->vision->def;
    header_print("info", "vision D:          " << vdef.D);
    header_print("info", "vision DQ/DK/DV:   " << vdef.DQ << "/" << vdef.DK << "/" << vdef.DV);
    header_print("info", "vision heads/kv:   " << vdef.num_attn_heads << "/" << vdef.num_kv_heads);
    header_print("info", "vision head_dim:   " << vdef.head_dim);
    header_print("info", "vision inter_size: " << vdef.intermediate_size);
    header_print("info", "vision num_layers: " << vdef.num_layers);
    header_print("info", "vision layer bytes:     " << vdef.layer_byte_size);
    header_print("info", "vision non_layer bytes: " << vdef.non_layer_byte_size);

    const SmolVLA_Language_def& ldef = model.vlm->language->def;
    header_print("info", "language D:          " << ldef.D);
    header_print("info", "language DQ/DK/DV:   " << ldef.DQ << "/" << ldef.DK << "/" << ldef.DV);
    header_print("info", "language heads/kv:   " << ldef.num_attn_heads << "/" << ldef.num_kv_heads);
    header_print("info", "language head_dim:   " << ldef.head_dim);
    header_print("info", "language inter_size: " << ldef.intermediate_size);
    header_print("info", "language num_layers: " << ldef.num_layers);
    header_print("info", "language vocab_size: " << ldef.vocab_size);
    header_print("info", "language layer bytes:     " << ldef.layer_byte_size);
    header_print("info", "language non_layer bytes: " << ldef.non_layer_byte_size);

    const SmolVLA_Vlm_def& vlmdef = model.vlm->def;
    header_print("info", "vlm connector_in:   " << vlmdef.connector_in);
    header_print("info", "vlm non_layer bytes: " << vlmdef.non_layer_byte_size);
    header_print("info", "top-level adapter bytes: " << model.adapter_byte_size);

    // -----------------------------------------------------------------------
    // Build the policy input from the captured reference tensors. The pixel
    // values are stored post-resize/pad ([1, 3, 512, 512] per camera), the
    // language tokens are the tokenizer-padded ids ([1, 48], I32) and the state
    // is the pipeline-padded observation vector ([1, 8], BF16).
    // -----------------------------------------------------------------------
    SafeTensors ref(utils::path_join(model_path, "ref.safetensors"));

    smolvla_input_t input;
    input.pixel_values.resize(2);
    ref.load_weights(input.pixel_values[0], "vision.camera1.pixel_values");
    ref.load_weights(input.pixel_values[1], "vision.camera2.pixel_values");
    ref.load_weights(input.states, "input.processed.observation.state");

    buffer<int> input_text_tokens;
    ref.load_weights(input_text_tokens, "input.processed.observation.language.tokens");
    input.input_tokens.assign(input_text_tokens.begin(), input_text_tokens.end());

    header_print("info", "input pixel_values[0]: " << input.pixel_values[0].size());
    header_print("info", "input pixel_values[1]: " << input.pixel_values[1].size());
    header_print("info", "input tokens:          " << input.input_tokens.size());
    header_print("info", "input states:          " << input.states.size());

    // // predict() now returns the full language kv cache, laid out as
    // // (2, num_layers, MAX_L, DK): all layer keys first, then all layer values.
    buffer<bf16> kv = model.predict(input);

    const int DK         = (int)ldef.DK;
    const int MAX_L      = SmolVLA_language::MAX_L;   // matches the attn block / kv cache
    const int num_layers = (int)ldef.num_layers;
    const size_t layer_stride = (size_t)MAX_L * DK;
    const size_t value_base   = (size_t)num_layers * layer_stride;

    // // -----------------------------------------------------------------------
    // // Full policy step. The noise IS captured (input.noise == suffix.x_t, the
    // // step-0 denoise input), so the whole prefill + 10-step Euler denoise is
    // // reproducible: SmolVLA::forward runs the vlm prefill and integrates the
    // // denoise loop over the action expert, seeded with the captured noise.
    // // select_action returns the first action row sliced to the real action dim,
    // // so compare x[0, :out_dim] against result.pred_action.
    // // -----------------------------------------------------------------------
    // // Isolate the expert: overwrite the freshly computed language kv cache with
    // // the reference states (keys post-rope = k_rope, values pre-rope = v_proj)
    // // so the denoise attends the ground-truth VLM prefix instead of our own
    // // (upstream attention-mask error). Cache is (2, num_layers, MAX_L, DK): keys
    // // first, then values; the first P rows of each layer slot are the valid
    // // prefix. Ref tensors are [1, P, num_kv_heads, head_dim] == [P, DK].
    buffer<bf16>& kv_cache = model.vlm->language->kv_cache;
    const int P = model.vlm->language->valid_prefix_len;   // 151

    // Score our own prefill against the reference BEFORE overwriting it. Errors
    // compound down the stack, so the interesting number is the first layer that
    // breaks, not the last: read down the k/v columns and find where the error
    // stops looking like bf16 rounding (~1e-3) and starts looking like a bug.
    header_print("info", "=== prefill kv cache vs reference (pre-injection), P=" << P << " ===");
    for (int l = 0; l < num_layers; l++) {
        std::vector<float> ref_k = ref_read_float(ref, "vlm.layer." + std::to_string(l) + ".k_rope");
        std::vector<float> ref_v = ref_read_float(ref, "vlm.layer." + std::to_string(l) + ".v_proj");
        const bf16* k_slot = kv_cache.data() + (size_t)l * layer_stride;
        const bf16* v_slot = kv_cache.data() + value_base + (size_t)l * layer_stride;
        std::vector<float> got_k((size_t)P * DK), got_v((size_t)P * DK);
        for (int i = 0; i < P * DK; i++) got_k[i] = (float)k_slot[i];
        for (int i = 0; i < P * DK; i++) got_v[i] = (float)v_slot[i];
        compare_vec("vlm.layer." + std::to_string(l) + ".k_rope", got_k, ref_k);
        compare_vec("vlm.layer." + std::to_string(l) + ".v_proj", got_v, ref_v);
    }


    for (int l = 0; l < num_layers; l++) {
        std::vector<float> ref_k = ref_read_float(ref, "vlm.layer." + std::to_string(l) + ".k_rope");
        std::vector<float> ref_v = ref_read_float(ref, "vlm.layer." + std::to_string(l) + ".v_proj");
        bf16* k_slot = kv_cache.data() + (size_t)l * layer_stride;
        bf16* v_slot = kv_cache.data() + value_base + (size_t)l * layer_stride;
        for (int i = 0; i < P * DK; i++) k_slot[i] = (bf16)ref_k[i];
        for (int i = 0; i < P * DK; i++) v_slot[i] = (bf16)ref_v[i];
    }
    header_print("info", "injected reference kv cache for " << num_layers
                 << " layers, P=" << P);

    std::vector<float> noise = ref_read_float(ref, "input.noise");   // [1, 50, 32]
    buffer<bf16> noise_buf = to_bf16_buffer(noise);
    header_print("info", "running expert denoise over reference kv cache");
    buffer<bf16> x = model.forward(input, &noise_buf, skip_prefill);

    // select_action returns the first action row, sliced to the real action dim
    // (result.pred_action is [1, out_dim]). Normalization stats are identity for
    // this checkpoint (raw state == processed state), so the postprocessed action
    // matches the raw denoise output directly.
    std::vector<float> ref_pred = ref_read_float(ref, "result.pred_action");
    const int out_dim = (int)ref_pred.size();                       // 6
    std::vector<float> got_action(out_dim);
    for (int i = 0; i < out_dim; i++) got_action[i] = (float)x[i];
    compare_ref("denoise.pred_action", got_action, ref, "result.pred_action");

    std::cout << "[info]  pred_action (got): [";
    for (int i = 0; i < out_dim; i++)
        std::cout << got_action[i] << (i + 1 < out_dim ? ", " : "");
    std::cout << "]" << std::endl;
    std::cout << "[info]  pred_action (ref): [";
    for (int i = 0; i < out_dim; i++)
        std::cout << ref_pred[i] << (i + 1 < out_dim ? ", " : "");
    std::cout << "]" << std::endl;

    return 0;
}

// ---------------------------------------------------------------------------
// Helper implementations
// ---------------------------------------------------------------------------

/// The CPU kernels (cpu_func) rely on `#pragma omp parallel for` to spread the
/// prefill-shaped work over a thread team. Without OpenMP those pragmas are
/// ignored and everything runs single threaded, which is functionally correct
/// but a lot slower, so make the user acknowledge it before continuing.
static void check_openmp() {
#ifdef _OPENMP
    const int max_threads = omp_get_max_threads();
    if (max_threads > 1) {
        header_print("info", "OpenMP enabled: " << max_threads << " threads");
        return;
    }
    header_print("warn", "OpenMP is enabled but only 1 thread is available "
                 "(check OMP_NUM_THREADS / cpu affinity).");
#else
    header_print("warn", "OpenMP is NOT available: this binary was built "
                 "without -fopenmp / libgomp.");
#endif
    header_print("warn", "The CPU kernels will run single threaded and will be slow.");
    std::cout << "[warn]  Press enter to continue anyway (ctrl-c to abort)... " << std::flush;
    std::string line;
    if (!std::getline(std::cin, line)) {
        // stdin closed (non-interactive run): don't block, just continue.
        std::cout << std::endl;
    }
}

static std::vector<float> ref_read_float(SafeTensors& ref, const std::string& name) {
    tensor_metadata md = ref.get_tensor_metadata(name);
    buffer<u8> raw(md.byte_size);
    ref.load_weights(raw, name);
    size_t n = md.size;
    std::vector<float> out(n);
    if (md.dtype == "BF16") {
        bf16* p = reinterpret_cast<bf16*>(raw.data());
        for (size_t i = 0; i < n; i++) out[i] = float(p[i]);
    } else if (md.dtype == "F32") {
        float* p = reinterpret_cast<float*>(raw.data());
        for (size_t i = 0; i < n; i++) out[i] = p[i];
    } else {
        std::cerr << "ref tensor " << name << " unexpected dtype " << md.dtype << std::endl;
        exit(1);
    }
    return out;
}

static buffer<bf16> to_bf16_buffer(const std::vector<float>& v) {
    buffer<bf16> b(v.size());
    for (size_t i = 0; i < v.size(); i++) b[i] = bf16(v[i]);
    return b;
}

static void compare_ref(const std::string& label,
                        const std::vector<float>& got_f,
                        SafeTensors& ref, const std::string& ref_name) {
    compare_vec(label, got_f, ref_read_float(ref, ref_name));
}

static void compare_vec(const std::string& label,
                        const std::vector<float>& got_f,
                        const std::vector<float>& ref_f) {
    if (got_f.size() != ref_f.size()) {
        header_print("info", label << ": SIZE MISMATCH got " << got_f.size()
                     << " vs ref " << ref_f.size());
        return;
    }
    buffer<bf16> a = to_bf16_buffer(got_f);
    buffer<bf16> b = to_bf16_buffer(ref_f);
    error_metrics m = get_error_metrics(a, b);
    print_error_metrics(m, label);
}