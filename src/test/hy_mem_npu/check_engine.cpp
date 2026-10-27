/// \file check_engine.cpp
/// \brief numeric check of the hy_mem_npu engine against transformers
/// \note Loads the model through the normal app path (model_list, backend
///       registry, tokenizer, chat template), then compares with a transformers
///       reference on CPU (same messages as messages/model_card_demo.json):
///         hy_mem_e2e_reference_q4k.safetensors  ref_hy_mem_e2e_q4k.py, the q4nx
///             weights decoded: what the engine is responsible for (the default)
///         hy_mem_e2e_reference.safetensors      ref_hy_mem_e2e.py, the original
///             bf16 weights: how far Q4_K itself moves the model
///       Checks:
///         1. tokens   the app's chat template + tokenizer vs the reference ids
///         2. prefill  logits of the last prompt token
///         3. decode   teacher forced: feed the bf16 reference's greedy tokens,
///                     compare every step's logits
///         4. greedy   free-running argmax decode vs the reference's greedy reply
///       Even against the Q4_K reference the logits are not equal (the NPU rounds x
///       to int8 per 32 values in its Q4_K kernel and routes in fp32), so cos is
///       ~0.98-0.99 and the gate is on tokens, not on cos.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "AutoModel/modeling_hy_memory.hpp"
#include "AutoModel/flm_backend.hpp"
#include "tensor_utils/safe_tensors.hpp"
#include "model_list.hpp"

flm_rt::device npu_device_global;

/// \note builtin_backends.cpp names every engine and so links every prebuilt
///       library; this harness registers just the one it drives.
namespace flm::backend {
void register_builtin_backends(BackendRegistry& registry) {
    registry.register_backend("hy-memory", kFlmBackendId, flm_factory<hy_mem_npu>());
}
}  // namespace flm::backend

/// \brief HyMemory with its engine and tokenizer in reach (both are protected)
class HyMemoryProbe : public HyMemory {
public:
    using HyMemory::HyMemory;
    causal_lm* engine() { return this->lm_engine; }
    std::vector<int> encode(const std::string& text) { return this->tokenizer->encode(text); }
    std::string decode(const std::vector<int>& ids) { return this->tokenizer->decode(ids); }
};

static float cosine(const float* a, const float* b, size_t n) {
    double dot = 0.0, aa = 0.0, bb = 0.0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        aa += (double)a[i] * a[i];
        bb += (double)b[i] * b[i];
    }
    return (float)(dot / (std::sqrt(aa) * std::sqrt(bb)));
}

static size_t argmax(const float* v, size_t n) {
    return std::max_element(v, v + n) - v;
}

/// one step's comparison: cos, and whether the top-1 differs by more than a near tie
struct step_score {
    float cos;
    bool same_top1;
    bool real_mismatch;   // the engine's pick is clearly below the reference's best
    size_t top_got, top_want;
    float gap;            // reference logit of its best minus that of the engine's pick
};

static step_score score(const buffer<bf16>& got_bf16, const float* want, size_t vocab) {
    std::vector<float> got(vocab);
    for (size_t i = 0; i < vocab; i++) got[i] = (float)got_bf16[i];
    step_score s;
    s.cos = cosine(got.data(), want, vocab);
    const size_t top_got = argmax(got.data(), vocab);
    const size_t top_want = argmax(want, vocab);
    s.same_top1 = top_got == top_want;
    // a near tie can flip top-1 between two right answers; call it real only
    // when the pick is more than 1% (of the best logit) below the best
    const float gap = want[top_want] - want[top_got];
    s.real_mismatch = !s.same_top1 && gap > 0.01f * std::fabs(want[top_want]);
    s.top_got = top_got;
    s.top_want = top_want;
    s.gap = gap;
    return s;
}

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char* argv[]) {
    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->default_value("hy-memory:30b-a3b"), "Model tag");
    desc.add_options()("reference,r", arg_utils::po::value<std::string>()->default_value(
        "/scratch/michyu/Projects/FastFlowLM_IRON/models_simple/hy_mem_e2e_reference.safetensors"), "ref_hy_mem_e2e.py output");
    desc.add_options()("messages", arg_utils::po::value<std::string>()->default_value("../../../test/hy_mem_npu/messages/model_card_demo.json"),
        "the messages the reference was made from");
    arg_utils::po::store(arg_utils::po::parse_command_line(argc, argv, desc), vm);

    const std::string tag = vm["model"].as<std::string>();
    std::string list_path = utils::get_executable_directory() + "/model_list.json";
    std::string models_dir = utils::get_models_directory();
    model_list models(list_path, models_dir);
    const std::string model_path = models.get_model_path(tag);
    auto model_info = models.get_model_info(tag).second;

    HyMemoryProbe chat(&npu_device_global);
    npu_device_global = flm_rt::device(0);
    auto t_load = std::chrono::steady_clock::now();
    chat.load_model(model_path, model_info, -1, false);
    header_print("info", "model loaded in " << ms_since(t_load) / 1000.0 << " s");
    causal_lm* engine = chat.engine();

    SafeTensors ref(vm["reference"].as<std::string>());
    buffer<int32_t> ref_ids, ref_out;
    buffer<float> ref_logits, ref_step_logits;
    ref.load_weights(ref_ids, "input_ids");
    ref.load_weights(ref_out, "output_ids");
    ref.load_weights(ref_logits, "logits");
    ref.load_weights(ref_step_logits, "step_logits");
    const size_t vocab = ref_logits.size();
    const size_t n_out = ref_out.size();
    bool ok = true;

    // ---- 1. tokens ----
    std::ifstream mf(vm["messages"].as<std::string>());
    if (!mf.is_open()) {
        header_print("error", "cannot open " << vm["messages"].as<std::string>());
        return 1;
    }
    nlohmann::ordered_json messages = nlohmann::ordered_json::parse(mf);
    std::vector<int> ids = chat.encode(chat.apply_chat_template(messages));
    std::vector<int> want_ids(ref_ids.data(), ref_ids.data() + ref_ids.size());
    size_t first_diff = 0;
    while (first_diff < ids.size() && first_diff < want_ids.size() && ids[first_diff] == want_ids[first_diff]) first_diff++;
    const bool tokens_ok = ids == want_ids;
    ok = ok && tokens_ok;
    std::cout << "\n[1] tokens : app " << ids.size() << ", reference " << want_ids.size()
              << (tokens_ok ? "  -> identical" : "  -> DIFFER from token " + std::to_string(first_diff)) << std::endl;
    if (!tokens_ok) {
        std::cout << "    app template text:\n" << chat.apply_chat_template(messages) << std::endl;
    }

    // ---- 2. prefill (always the reference ids, so 3/4 are comparable) ----
    engine->clear_context();
    auto t0 = std::chrono::steady_clock::now();
    buffer<bf16> logits = engine->prefill(want_ids);
    const double prefill_ms = ms_since(t0);
    step_score s = score(logits, ref_logits.data(), vocab);
    // a coarse floor: a broken engine scores far lower; Q4_K vs Q4_K lands ~0.98
    ok = ok && s.cos > 0.95f && !s.real_mismatch;
    std::cout << "[2] prefill: " << want_ids.size() << " tokens in " << prefill_ms << " ms ("
              << prefill_ms / want_ids.size() << " ms/token), logits cos " << s.cos
              << ", top-1 " << (s.same_top1 ? "same" : (s.real_mismatch ? "MISMATCH" : "near tie")) << std::endl;

    // ---- 3. teacher-forced decode ----
    float min_cos = 1.0f, sum_cos = 0.0f;
    int same = 0, real_bad = 0, steps = 0, worst = -1;
    t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i + 1 < n_out; i++) {
        logits = engine->forward(ref_out[i]);
        s = score(logits, ref_step_logits.data() + (i + 1) * vocab, vocab);
        steps++;
        sum_cos += s.cos;
        same += s.same_top1;
        real_bad += s.real_mismatch;
        if (s.cos < min_cos) { min_cos = s.cos; worst = (int)i + 1; }
        if (!s.same_top1) {
            // where a greedy reply can part ways: was it a near tie in the reference?
            std::cout << "    step " << i + 1 << ": top-1 engine " << s.top_got << " " << std::quoted(chat.decode({(int)s.top_got}))
                      << ", reference " << s.top_want << " " << std::quoted(chat.decode({(int)s.top_want}))
                      << "; reference logit gap " << s.gap << " (best " << ref_step_logits[(i + 1) * vocab + s.top_want]
                      << ")" << (s.real_mismatch ? "  REAL" : "  near tie") << std::endl;
        }
    }
    const double decode_ms = ms_since(t0);
    // the worst step is reported, not gated: one expert pick that flips between the
    // NPU's fp32 router and transformers' bf16 one moves a single step's logits a lot
    ok = ok && real_bad == 0 && sum_cos / std::max(steps, 1) > 0.95f;
    std::cout << "[3] decode : " << steps << " teacher-forced steps, " << decode_ms / std::max(steps, 1) << " ms/token ("
              << 1000.0 * steps / decode_ms << " tok/s); logits cos mean " << sum_cos / std::max(steps, 1)
              << ", min " << min_cos << " (step " << worst << "); top-1 same " << same << "/" << steps
              << ", real mismatches " << real_bad << std::endl;

    // ---- 4. free-running greedy ----
    engine->clear_context();
    logits = engine->prefill(want_ids);
    std::vector<int> greedy;
    for (size_t i = 0; i < n_out + 8; i++) {
        std::vector<float> l(vocab);
        for (size_t j = 0; j < vocab; j++) l[j] = (float)logits[j];
        const int next = (int)argmax(l.data(), vocab);
        greedy.push_back(next);
        if (next == 120025 || next == 120026) break;   // eos / eod
        logits = engine->forward(next);
    }
    // the Q4_K reference carries its own greedy reply; the bf16 one's is output_ids
    buffer<int32_t> ref_greedy;
    ref.load_weights(ref_greedy, ref.has_tensor("greedy_ids") ? "greedy_ids" : "output_ids");
    size_t agree = 0;
    while (agree < greedy.size() && agree < ref_greedy.size() && greedy[agree] == ref_greedy[agree]) agree++;
    std::cout << "[4] greedy : " << greedy.size() << " tokens, first " << agree << " identical to the reference's greedy reply ("
              << ref_greedy.size() << " tokens)" << std::endl;
    // the same text can come out as different token splits ("\n" + "  " or "\n  ")
    const std::string ref_text = chat.decode(std::vector<int>(ref_greedy.data(), ref_greedy.data() + ref_greedy.size()));
    std::cout << "    text " << (chat.decode(greedy) == ref_text ? "identical" : "DIFFERS") << std::endl;
    std::cout << "    reference reply:\n" << chat.decode(std::vector<int>(ref_greedy.data(), ref_greedy.data() + ref_greedy.size())) << std::endl;
    std::cout << "    engine reply:\n" << chat.decode(greedy) << std::endl;

    // ---- 5. prefill on top of an existing cache ----
    // A new turn, or a prompt longer than one prefill chunk, prefills at pos0 > 0 and
    // attends to rows already in the cache, some of them written by the NPU decode.
    // The same sequence run three ways should give the same last logits:
    //   whole   one prefill of everything
    //   split   prefill the first half, then the rest              (pos0 > 0)
    //   mixed   prefill the first half, decode 5 tokens on the NPU, prefill the rest
    // split should be exact (same host math, row by row); mixed carries the NPU
    // decode's own rounding in 5 of the cache rows.
    {
        std::vector<int> seq = want_ids;
        for (size_t i = 0; i < std::min<size_t>(n_out, 16); i++) seq.push_back(ref_out[i]);
        const int half = (int)seq.size() / 2;
        auto run = [&](int decode_tokens) {
            engine->clear_context();
            std::vector<int> a(seq.begin(), seq.begin() + half);
            buffer<bf16> l = engine->prefill(a);
            if (half == (int)seq.size()) return l;
            for (int i = half; i < half + decode_tokens; i++) l = engine->forward(seq[i]);
            std::vector<int> b(seq.begin() + half + decode_tokens, seq.end());
            l = engine->prefill(b);
            return l;
        };
        auto to_float = [&](const buffer<bf16>& l) {
            std::vector<float> f(vocab);
            for (size_t j = 0; j < vocab; j++) f[j] = (float)l[j];
            return f;
        };
        engine->clear_context();
        const std::vector<float> whole = to_float(engine->prefill(seq));
        const std::vector<float> split = to_float(run(0));
        const std::vector<float> mixed = to_float(run(5));
        const float cos_split = cosine(split.data(), whole.data(), vocab);
        const float cos_mixed = cosine(mixed.data(), whole.data(), vocab);
        const size_t top_whole = argmax(whole.data(), vocab);
        const bool split_ok = cos_split > 0.9999f && argmax(split.data(), vocab) == top_whole;
        const bool mixed_ok = cos_mixed > 0.99f && argmax(mixed.data(), vocab) == top_whole;
        ok = ok && split_ok && mixed_ok;
        std::cout << "[5] resume : " << seq.size() << " tokens, split at " << half << ": split vs whole cos " << cos_split
                  << (split_ok ? "" : "  BAD") << "; with 5 NPU decode rows between, cos " << cos_mixed
                  << (mixed_ok ? "" : "  BAD") << std::endl;
    }

    std::cout << "\n" << (ok ? "PASS" : "FAIL")
              << " (tokens identical, no real top-1 mismatch, prefill cos and mean decode cos > 0.95,"
                 " resumed prefill = whole)" << std::endl;
    return ok ? 0 : 1;
}
