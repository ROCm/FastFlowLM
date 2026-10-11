#include "models/qwen35/rai/aie_next/qwen35_rai.hpp"

#include "rai/corelib_object.hpp"
#include "rai/host_ops.hpp"
#include "rai/kernel_grid.hpp"
#include "rai/weight_source.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <span>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace flm::qwen35 {
namespace {
using namespace flm::corelib;

int PadVectors(int vectors, int width) {
    const int* table = nullptr;
    int count = 0;
    if (width == 128) {
        static const int kRows[] = {2, 16, 32};
        table = kRows;
        count = 3;
    } else if (width == 256) {
        static const int kRows[] = {1, 8, 32};
        table = kRows;
        count = 3;
    } else {
        throw std::runtime_error("no RMSNorm vector table for width " + std::to_string(width));
    }
    for (int i = 0; i < count; ++i)
        if (table[i] >= vectors) return table[i];
    throw std::runtime_error("no RMSNorm kernel covers " + std::to_string(vectors) +
                             " vectors of " + std::to_string(width));
}

std::uint16_t FloatToHalfOne(float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    auto exponent = static_cast<std::int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    std::uint32_t mantissa = bits & 0x7fffffu;
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<std::uint16_t>(sign);
        mantissa |= 0x800000u;
        const auto shift = static_cast<std::uint32_t>(14 - exponent);
        std::uint32_t half = mantissa >> shift;
        const auto remainder = mantissa & ((1u << shift) - 1u);
        if (remainder > (1u << (shift - 1)) ||
            (remainder == (1u << (shift - 1)) && (half & 1u)))
            ++half;
        return static_cast<std::uint16_t>(sign | half);
    }
    if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    std::uint32_t half = mantissa >> 13;
    const auto remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u))) {
        ++half;
        if (half == 0x400u) {
            half = 0;
            ++exponent;
            if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
        }
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10) | half);
}

std::vector<std::uint16_t> FloatToHalf(const std::vector<float>& values) {
    std::vector<std::uint16_t> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) out[i] = FloatToHalfOne(values[i]);
    return out;
}

struct FullWeights {
    UniqueMatMulWeights q, gate, k, v, o;
    UniqueRmsNormWeights q_norm, k_norm;
    UniqueSsMlpWeights mlp;
};

struct LinearWeights {
    UniqueMatMulWeights qkv, beta, alpha, z, out;
    UniqueTensor neg_a;
    UniqueDwConvWeights conv;
    UniqueRmsNormWeights norm;
    UniqueSsMlpWeights mlp;
};

struct FullIo {
    UniqueTensor q_raw, q_buf, k_buf, mha_out, gated, k_cache, v_cache;
    UniqueTensor q_src[2], q_dst[2], q_live, k_live;
    std::vector<UniqueTensor> v_rows;
};

struct LinearIo {
    UniqueTensor qkv, qkv_row, beta, beta_live, g, g_live;
    UniqueTensor dw_q, dw_k, dw_v, lin_out, lin_flat, z_out;
    UniqueTensor conv[2], state[2];
};

}  // namespace

using namespace flm::corelib;

class qwen35_rai::Impl {
public:
    Impl(std::shared_ptr<Qwen35Gguf> package, std::shared_ptr<CorelibRuntime> runtime,
         std::uint32_t maximum)
        : package(std::move(package)), runtime(std::move(runtime)),
          api(this->runtime ? this->runtime->api() : nullptr), max_length(maximum) {
        if (!this->package) throw std::invalid_argument("Qwen3.5 GGUF package is null");
        if (!this->runtime || !api) throw std::invalid_argument("corelib runtime is null");
        if (!maximum || maximum > kMaxSequenceLength)
            throw std::invalid_argument("Qwen3.5 maximum length must be in 1..4096");
        cfg = &this->package->Config();
        const int layers = cfg->layers;

        auto lease = this->runtime->AcquireExecution();
        stream = corelib::OpenModelStream(api, kPrefillPdi, kTokenPdi, "qwen35");

        const auto matmul = ShapeGrid::Matmul(*api, stream.get());
        const auto sigmoid = ShapeGrid::MatmulAct(
            *api, stream.get(), ryzenai_corelib_matmul_activation_sigmoid);
        const auto softplus = ShapeGrid::MatmulAct(
            *api, stream.get(), ryzenai_corelib_matmul_activation_softplus_mulbcast);
        const auto silu = ShapeGrid::MatmulAct(
            *api, stream.get(), ryzenai_corelib_matmul_activation_silu_mul);
        const auto sigmoid_mul = ShapeGrid::MatmulAct(
            *api, stream.get(), ryzenai_corelib_matmul_activation_sigmoid_mul);
        const auto mlp_grid = ShapeGrid::SsMlp(*api, stream.get(), false);
        const auto require = [](const ShapeGrid& grid, int k, int n, const char* name) {
            if (!grid.Has(1, k, n, -1))
                throw std::runtime_error(std::string("no ") + name + " kernel ships at 1x" +
                                         std::to_string(k) + "x" + std::to_string(n));
        };
        require(matmul, cfg->hidden, cfg->q_dim(), "matmul");
        require(matmul, cfg->hidden, cfg->kv_dim(), "matmul");
        require(matmul, cfg->q_dim(), cfg->hidden, "matmul");
        require(matmul, cfg->hidden, cfg->conv_channels(), "matmul");
        require(matmul, cfg->lin_value_dim(), cfg->hidden, "matmul");
        require(matmul, cfg->hidden, cfg->vocab, "matmul");
        require(sigmoid, cfg->hidden, cfg->lin_v_heads, "sigmoid");
        require(softplus, cfg->hidden, cfg->lin_v_heads, "softplus_mulbcast");
        require(silu, cfg->hidden, cfg->lin_value_dim(), "silu_mul");
        require(sigmoid_mul, cfg->hidden, cfg->q_dim(), "sigmoid_mul");
        require(mlp_grid, cfg->hidden, cfg->intermediate, "fused MLP");
        beta_width = static_cast<int>(sigmoid.SmallestN(1, cfg->hidden, cfg->lin_v_heads));
        alpha_width = static_cast<int>(softplus.SmallestN(1, cfg->hidden, cfg->lin_v_heads));
        if (beta_width < 0 || alpha_width < 0)
            throw std::runtime_error("no gate kernel is wide enough for the linear-attention heads");

        cfg_eps = static_cast<float>(cfg->eps);
        epsilon_bf16 = rai::F32ToBf16(std::span<const float>(&cfg_eps, 1));
        lin_desc = {cfg->lin_k_heads, cfg->lin_v_heads, cfg->lin_k_dim, cfg->lin_v_dim};
        mha_desc = {cfg->q_heads, cfg->kv_heads, cfg->head_dim, kMaxSequenceLength, cfg->rope_dim,
                    0, 0, 0.0f};
        const auto started = std::chrono::steady_clock::now();
        std::cout << "[FLM]  Packing Qwen3.5 hidden " << cfg->hidden << ", " << layers
                  << " layers, from Q8_0" << std::endl;
        full.resize(static_cast<std::size_t>(layers));
        linear.resize(static_cast<std::size_t>(layers));
        norm0 = Norm("blk.0.attn_norm.weight", cfg->hidden);
        for (int layer = 0; layer < layers; ++layer) LoadLayer(layer);
        lm_head = Matmul(this->package->Blocks(this->package->LmHeadName(), cfg->hidden, cfg->vocab),
                         cfg->hidden, cfg->vocab, "lm_head");
        const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::cout << "[FLM]  Qwen3.5 weights ready in " << seconds << "s" << std::endl;

        Allocate();
        Reset();
        // A process's first M=1 decode is perturbed by a couple of bf16 ulps.
        // A throwaway token keeps that pass out of the caller's sequence.
        StepBody(0);
        Reset();
    }

    buffer<bf16> Forward(int id) {
        usable();
        auto lease = runtime->AcquireExecution();
        auto out = StepBody(id);
        history.push_back(id);
        return out;
    }

    buffer<bf16> Prefill(const std::vector<int>& ids) {
        usable();
        if (ids.empty()) throw std::invalid_argument("Qwen3.5 request contains no token IDs");
        auto lease = runtime->AcquireExecution();
        buffer<bf16> out;
        for (const int id : ids) {
            out = StepBody(id);
            history.push_back(id);
        }
        return out;
    }

    void Clear() {
        usable();
        auto lease = runtime->AcquireExecution();
        Reset();
        history.clear();
        saved.reset();
    }

    void Rewind(int to) {
        usable();
        if (to < 0 || static_cast<std::uint32_t>(to) > max_length)
            throw std::out_of_range("Qwen3.5 context length is out of range");
        if (static_cast<std::size_t>(to) > history.size())
            throw std::runtime_error("Qwen3.5 cannot rewind past the tokens it has run");
        if (to == position) return;
        auto lease = runtime->AcquireExecution();
        Reset();
        for (int i = 0; i < to; ++i) StepBody(history[static_cast<std::size_t>(i)]);
    }

    buffer<bf16> ReadCache(bool is_k, int layer, int index) {
        usable();
        if (layer < 0 || layer >= cfg->layers || !cfg->is_full(layer) || index < 0 ||
            index >= kMaxSequenceLength)
            throw std::out_of_range("Qwen3.5 cache index is out of range");
        auto lease = runtime->AcquireExecution();
        api->Check(api->functions().stream_synchronize(stream.get()),
                   "ryzenai_corelib_stream_synchronize cache read");
        const auto heads = static_cast<std::size_t>(cfg->kv_heads);
        const auto width = static_cast<std::size_t>(cfg->head_dim);
        buffer<bf16> out(heads * width);
        void* cache = is_k ? full_io[layer].k_cache.get() : full_io[layer].v_cache.get();
        for (std::size_t head = 0; head < heads; ++head) {
            const auto offset = (head * kMaxSequenceLength + static_cast<std::size_t>(index)) * width;
            api->Check(api->functions().tensor_read(cache, ryzenai_corelib_data_type_bf16,
                                                    out.data() + head * width, width, offset),
                       "ryzenai_corelib_tensor_read cache");
        }
        return out;
    }

    void usable() const {
        if (poisoned) throw std::runtime_error("Qwen3.5 corelib engine is poisoned");
    }

    std::uint32_t max_length;
    int position = 0;
    std::optional<int> saved;
    bool poisoned = false;

private:
    UniqueTensor Tensor(std::initializer_list<std::int64_t> shape, const char* label) {
        std::vector<std::int64_t> dims(shape);
        void* created = nullptr;
        api->Check(api->functions().create_device_tensor(ryzenai_corelib_data_type_bf16, dims.data(),
                                                         dims.size(), &created),
                   label);
        return UniqueTensor(api, created);
    }

    UniqueTensor Window(void* parent, std::initializer_list<std::int64_t> shape, std::size_t offset,
                        const char* label, std::initializer_list<std::int64_t> strides = {}) {
        std::vector<std::int64_t> dims(shape);
        std::vector<std::int64_t> steps(strides);
        void* created = nullptr;
        api->Check(api->functions().create_tensor_window(parent, dims.data(), dims.size(), offset,
                                                         steps.empty() ? nullptr : steps.data(),
                                                         steps.size(), &created),
                   label);
        return UniqueTensor(api, created);
    }

    void Write(void* tensor, const float* data, std::size_t count, const char* label) {
        api->Check(api->functions().tensor_write(tensor, ryzenai_corelib_data_type_fp32, data, count, 0),
                   label);
    }

    void Zero(void* tensor, std::size_t count, const char* label) {
        if (zeros.size() < count) zeros.assign(count, 0.0f);
        Write(tensor, zeros.data(), count, label);
    }

    UniqueMatMulWeights Matmul(std::span<const std::byte> blocks, int k, int n, const std::string& label) {
        const ryzenai_corelib_matmul_bf16_weights_desc desc{k, n, kGroup, false};
        const ryzenai_corelib_weights_source sources[]{corelib::GgufQ8(
            ryzenai_corelib_weights_role_qweight, blocks.data(), blocks.size(), n, k)};
        const auto options = corelib::PackOptions(kRequantizeThreads);
        void* created = nullptr;
        api->Check(api->functions().matmul_weights_create(
                       &desc, sources, std::size(sources), &options, &created),
                   "ryzenai_corelib_matmul_bf16_weights_create " + label);
        return UniqueMatMulWeights(api, created);
    }

    UniqueMatMulWeights MatmulOwned(std::vector<std::byte> blocks, int k, int n, const std::string& label) {
        return Matmul(blocks, k, n, label);
    }

    UniqueMatMulWeights MatmulBiased(std::vector<std::byte> blocks, int k, int n,
                                     const std::vector<float>& bias, const std::string& label) {
        const auto onnx = Q8ToOnnx(blocks, k, n, kGroup);
        const ryzenai_corelib_matmul_bf16_weights_desc desc{k, n, kGroup, true};
        const auto groups = k / kGroup;
        const ryzenai_corelib_weights_source sources[]{
            corelib::MemorySource(ryzenai_corelib_weights_role_qweight, onnx.qweight.data(),
                                  onnx.qweight.size(),
                                  ryzenai_corelib_weights_data_type_onnx_uint8, {n, k / 2}),
            corelib::Fp16(ryzenai_corelib_weights_role_scales, onnx.scales.data(),
                          onnx.scales.size(), {n, groups}),
            corelib::MemorySource(ryzenai_corelib_weights_role_qzeros, onnx.qzeros.data(),
                                  onnx.qzeros.size(),
                                  ryzenai_corelib_weights_data_type_onnx_uint8, {n, groups / 2}),
            corelib::Fp32(ryzenai_corelib_weights_role_bias, bias.data(), bias.size(), {n})};
        const auto options = corelib::PackOptions(kRequantizeThreads);
        void* created = nullptr;
        api->Check(api->functions().matmul_weights_create(&desc, sources, std::size(sources),
                                                          &options, &created),
                   "ryzenai_corelib_matmul_bf16_weights_create " + label);
        return UniqueMatMulWeights(api, created);
    }

    UniqueRmsNormWeights Norm(std::string_view name, int k) {
        const std::int64_t shape[] = {k};
        const auto scale = rai::F32ToBf16(package->F32(name, shape));
        const ryzenai_corelib_rmsnorm_bf16_weights_desc desc{k, static_cast<float>(cfg->eps)};
        const ryzenai_corelib_weights_source sources[]{
            corelib::Bf16(ryzenai_corelib_weights_role_scale, scale.data(), scale.size())};
        void* created = nullptr;
        api->Check(api->functions().rmsnorm_weights_create(&desc, sources, std::size(sources),
                                                           nullptr, &created),
                   "ryzenai_corelib_rmsnorm_bf16_weights_create " + std::string(name));
        return UniqueRmsNormWeights(api, created);
    }

    UniqueRmsNormWeights NormValues(const std::vector<float>& values, int k, const std::string& label) {
        const auto scale = rai::F32ToBf16(values);
        const ryzenai_corelib_rmsnorm_bf16_weights_desc desc{k, static_cast<float>(cfg->eps)};
        const ryzenai_corelib_weights_source sources[]{
            corelib::Bf16(ryzenai_corelib_weights_role_scale, scale.data(), scale.size())};
        void* created = nullptr;
        api->Check(api->functions().rmsnorm_weights_create(&desc, sources, std::size(sources),
                                                           nullptr, &created),
                   "ryzenai_corelib_rmsnorm_bf16_weights_create " + label);
        return UniqueRmsNormWeights(api, created);
    }

    UniqueSsMlpWeights Mlp(int layer) {
        const auto gate = package->Blocks("blk." + std::to_string(layer) + ".ffn_gate", cfg->hidden,
                                          cfg->intermediate);
        const auto up = package->Blocks("blk." + std::to_string(layer) + ".ffn_up", cfg->hidden,
                                        cfg->intermediate);
        const auto down = package->Blocks("blk." + std::to_string(layer) + ".ffn_down", cfg->intermediate,
                                          cfg->hidden);
        const std::int64_t norm_shape[] = {cfg->hidden};
        const auto norm0_scale = rai::F32ToBf16(package->F32(
            "blk." + std::to_string(layer) + ".post_attention_norm.weight", norm_shape));
        const auto next_name = layer + 1 == cfg->layers
                                   ? std::string("output_norm.weight")
                                   : "blk." + std::to_string(layer + 1) + ".attn_norm.weight";
        const auto norm1_scale = rai::F32ToBf16(package->F32(next_name, norm_shape));
        const ryzenai_corelib_ssmlp_bf16_weights_desc desc{cfg->hidden, cfg->intermediate, kGroup, 0, 0};
        const ryzenai_corelib_weights_source sources[]{
            corelib::Bf16(ryzenai_corelib_weights_role_epsilon, epsilon_bf16.data(), epsilon_bf16.size()),
            corelib::Bf16(ryzenai_corelib_weights_role_norm0, norm0_scale.data(), norm0_scale.size()),
            corelib::Bf16(ryzenai_corelib_weights_role_norm1, norm1_scale.data(), norm1_scale.size()),
            corelib::GgufQ8(ryzenai_corelib_weights_role_gate_qweight, gate.data(), gate.size(), cfg->intermediate, cfg->hidden),
            corelib::GgufQ8(ryzenai_corelib_weights_role_up_qweight, up.data(), up.size(), cfg->intermediate, cfg->hidden),
            corelib::GgufQ8(ryzenai_corelib_weights_role_down_qweight, down.data(), down.size(), cfg->hidden, cfg->intermediate)};
        const auto options = corelib::PackOptions(kRequantizeThreads);
        void* created = nullptr;
        api->Check(api->functions().ssmlp_weights_create(
                       &desc, sources, std::size(sources), &options, &created),
                   "ryzenai_corelib_ssmlp_bf16_weights_create layer " +
                       std::to_string(layer));
        return UniqueSsMlpWeights(api, created);
    }

    UniqueMatMulWeights SliceHeads(UniqueMatMulWeights& both, bool gate_half) {
        const int d = cfg->head_dim;
        std::vector<ryzenai_corelib_weights_range> ranges(static_cast<std::size_t>(cfg->q_heads));
        for (int head = 0; head < cfg->q_heads; ++head) {
            const std::int64_t begin = gate_half ? (2 * d * head + d) : (2 * d * head);
            ranges[static_cast<std::size_t>(head)] = {1, begin, begin + d};
        }
        void* created = nullptr;
        api->Check(api->functions().weights_slice(both.get(), ranges.data(), ranges.size(), &created),
                   "ryzenai_corelib_weights_slice attn_q");
        return UniqueMatMulWeights(api, created);
    }

    void LoadLayer(int layer) {
        const auto id = std::to_string(layer);
        if (cfg->is_full(layer)) {
            auto both = Matmul(package->Blocks("blk." + id + ".attn_q", cfg->hidden, 2 * cfg->q_dim()),
                               cfg->hidden, 2 * cfg->q_dim(), "blk." + id + ".attn_q");
            full[layer].q = SliceHeads(both, false);
            full[layer].gate = SliceHeads(both, true);
            both.reset();
            full[layer].k = Matmul(package->Blocks("blk." + id + ".attn_k", cfg->hidden, cfg->kv_dim()),
                                   cfg->hidden, cfg->kv_dim(), "blk." + id + ".attn_k");
            full[layer].v = Matmul(package->Blocks("blk." + id + ".attn_v", cfg->hidden, cfg->kv_dim()),
                                   cfg->hidden, cfg->kv_dim(), "blk." + id + ".attn_v");
            full[layer].o = Matmul(package->Blocks("blk." + id + ".attn_output", cfg->q_dim(), cfg->hidden),
                                   cfg->q_dim(), cfg->hidden, "blk." + id + ".attn_output");
            full[layer].q_norm = Norm("blk." + id + ".attn_q_norm.weight", cfg->head_dim);
            full[layer].k_norm = Norm("blk." + id + ".attn_k_norm.weight", cfg->head_dim);
            full[layer].mlp = Mlp(layer);
        } else {
            auto neg = package->PerHead(layer, "ssm_a");
            std::vector<float> operand(static_cast<std::size_t>(kSoftplusOperand), 0.0f);
            for (int head = 0; head < cfg->lin_v_heads && head < static_cast<int>(neg.size()); ++head)
                operand[static_cast<std::size_t>(head)] = neg[static_cast<std::size_t>(head)];
            linear[layer].neg_a = Tensor({kSoftplusOperand}, "neg_a");
            Write(linear[layer].neg_a.get(), operand.data(), operand.size(), "neg_a");
            linear[layer].qkv = MatmulOwned(package->QkvBlocks(layer), cfg->hidden, cfg->conv_channels(),
                                            "blk." + id + ".attn_qkv");
            linear[layer].beta = MatmulOwned(package->GateBlocks(layer, "beta"), cfg->hidden, cfg->lin_v_heads,
                                             "blk." + id + ".ssm_beta");
            linear[layer].alpha = MatmulBiased(package->GateBlocks(layer, "alpha"), cfg->hidden,
                                               cfg->lin_v_heads, package->PerHead(layer, "ssm_dt.bias"),
                                               "blk." + id + ".ssm_alpha");
            const auto taps = package->ConvTaps(layer);
            const auto weights = FloatToHalf(taps);
            std::vector<std::uint16_t> bias(static_cast<std::size_t>(cfg->conv_channels()), 0);
            const ryzenai_corelib_dwconv_bf16_weights_desc desc{
                cfg->conv_channels(), cfg->conv_width, cfg->lin_k_dim, cfg->lin_k_heads, cfg->lin_k_heads,
                cfg->lin_v_heads, 1.0e-6f, static_cast<float>(1.0 / std::sqrt(cfg->lin_k_dim)), 1.0f};
            const ryzenai_corelib_weights_source sources[]{
                corelib::Fp16(ryzenai_corelib_weights_role_weights, weights.data(), weights.size(),
                              {desc.channels, 1, desc.kernel_width}),
                corelib::Fp16(ryzenai_corelib_weights_role_bias, bias.data(), bias.size(),
                              {desc.channels})};
            void* created = nullptr;
            api->Check(api->functions().dwconv_weights_create(&desc, sources, std::size(sources),
                                                              nullptr, &created),
                       "ryzenai_corelib_dwconv_bf16_weights_create blk." + id);
            linear[layer].conv = UniqueDwConvWeights(api, created);
            linear[layer].z = MatmulOwned(package->ZBlocks(layer), cfg->hidden, cfg->lin_value_dim(),
                                          "blk." + id + ".attn_gate");
            linear[layer].norm = Norm("blk." + id + ".ssm_norm.weight", cfg->lin_v_dim);
            linear[layer].out = MatmulOwned(package->OutBlocks(layer), cfg->lin_value_dim(), cfg->hidden,
                                            "blk." + id + ".ssm_out");
            linear[layer].mlp = Mlp(layer);
        }
        std::cout << "[FLM]  layer " << (layer + 1) << "/" << cfg->layers << " packed ("
                  << (cfg->is_full(layer) ? "full" : "linear") << ")" << std::endl;
    }

    void Allocate() {
        const int layers = cfg->layers;
        const int half = cfg->q_heads / 2;
        if (cfg->q_heads % 2 != 0 || PadVectors(half, cfg->head_dim) != half)
            throw std::runtime_error("q_norm over " + std::to_string(cfg->q_heads) +
                                     " heads does not split into two shipped vector counts");
        q_vectors = std::max(PadVectors(cfg->q_heads, cfg->head_dim), cfg->q_heads);
        k_vectors = std::max(PadVectors(cfg->kv_heads, cfg->head_dim), cfg->kv_heads);
        input_rows = std::max(1, cfg->conv_width - 1);
        BuildRope();

        res_in = Tensor({1, cfg->hidden}, "residual");
        h_in = Tensor({1, cfg->hidden}, "hidden");
        logits = Tensor({1, cfg->vocab}, "logits");
        res.reserve(layers);
        h.reserve(layers);
        attn.reserve(layers);
        for (int layer = 0; layer < layers; ++layer) {
            res.push_back(Tensor({1, cfg->hidden}, "residual"));
            h.push_back(Tensor({1, cfg->hidden}, "hidden"));
            attn.push_back(Tensor({1, cfg->hidden}, "attention"));
        }
        full_io = std::vector<FullIo>(static_cast<std::size_t>(layers));
        linear_io = std::vector<LinearIo>(static_cast<std::size_t>(layers));
        parity.assign(static_cast<std::size_t>(layers), 0);
        for (int layer = 0; layer < layers; ++layer) {
            if (!cfg->is_full(layer)) {
                auto& io = linear_io[layer];
                io.qkv = Tensor({input_rows, cfg->conv_channels()}, "qkv");
                io.qkv_row = Window(io.qkv.get(), {1, cfg->conv_channels()}, 0, "qkv row");
                io.beta = Tensor({1, beta_width}, "beta");
                io.beta_live = Window(io.beta.get(), {1, cfg->lin_v_heads}, 0, "beta live");
                io.g = Tensor({1, alpha_width}, "gate");
                io.g_live = Window(io.g.get(), {1, cfg->lin_v_heads}, 0, "gate live");
                io.dw_q = Tensor({cfg->lin_k_heads, 1, cfg->lin_k_dim}, "dw q");
                io.dw_k = Tensor({cfg->lin_k_heads, 1, cfg->lin_k_dim}, "dw k");
                io.dw_v = Tensor({cfg->lin_v_heads, 1, cfg->lin_v_dim}, "dw v");
                io.lin_out = Tensor({1, cfg->lin_v_heads, cfg->lin_v_dim}, "linear out");
                io.lin_flat = Window(io.lin_out.get(), {1, cfg->lin_value_dim()}, 0, "linear flat");
                io.z_out = Tensor({1, cfg->lin_value_dim()}, "z");
                for (int side = 0; side < 2; ++side) {
                    io.conv[side] = Tensor({input_rows, cfg->conv_channels()}, "conv state");
                    io.state[side] = Tensor({cfg->lin_v_heads, cfg->lin_v_dim, cfg->lin_k_dim}, "recurrent state");
                }
                continue;
            }
            auto& io = full_io[layer];
            io.q_raw = Tensor({1, cfg->q_dim()}, "q");
            io.q_buf = Tensor({1, q_vectors, cfg->head_dim}, "q buffer");
            io.k_buf = Tensor({1, k_vectors, cfg->head_dim}, "k buffer");
            for (int part = 0; part < 2; ++part) {
                const auto offset = static_cast<std::size_t>(part * half * cfg->head_dim);
                io.q_src[part] = Window(io.q_raw.get(), {1, half, cfg->head_dim}, offset, "q source");
                io.q_dst[part] = Window(io.q_buf.get(), {1, half, cfg->head_dim}, offset, "q dest");
            }
            io.q_live = Window(io.q_buf.get(), {1, cfg->q_dim()}, 0, "q live");
            io.k_live = Window(io.k_buf.get(), {1, cfg->kv_dim()}, 0, "k live");
            io.mha_out = Tensor({1, cfg->q_dim()}, "mha");
            io.gated = Tensor({1, cfg->q_dim()}, "gated");
            io.k_cache = Tensor({cfg->kv_heads, kMaxSequenceLength, cfg->head_dim}, "k cache");
            io.v_cache = Tensor({cfg->kv_heads, kMaxSequenceLength, cfg->head_dim}, "v cache");
            io.v_rows.resize(static_cast<std::size_t>(kMaxSequenceLength));
        }
        void* cos_view = nullptr;
        void* sin_view = nullptr;
        const std::int64_t rope_shape[] = {kMaxSequenceLength, cfg->rope_dim / 2};
        api->Check(api->functions().create_host_view(ryzenai_corelib_data_type_fp32, rope_shape, 2,
                                                     cos_table.data(), &cos_view),
                   "ryzenai_corelib_create_host_view cos");
        api->Check(api->functions().create_host_view(ryzenai_corelib_data_type_fp32, rope_shape, 2,
                                                     sin_table.data(), &sin_view),
                   "ryzenai_corelib_create_host_view sin");
        cos = UniqueHostView(api, cos_view);
        sin = UniqueHostView(api, sin_view);
    }

    void BuildRope() {
        const int half = cfg->rope_dim / 2;
        cos_table.resize(static_cast<std::size_t>(kMaxSequenceLength) * half);
        sin_table.resize(cos_table.size());
        for (int pos = 0; pos < kMaxSequenceLength; ++pos) {
            for (int i = 0; i < half; ++i) {
                const double angle = pos / std::pow(cfg->rope_theta, (2.0 * i) / cfg->rope_dim);
                const auto at = static_cast<std::size_t>(pos) * half + i;
                cos_table[at] = static_cast<float>(std::cos(angle));
                sin_table[at] = static_cast<float>(std::sin(angle));
            }
        }
    }

    void Reset() {
        position = 0;
        const auto cache_count = static_cast<std::size_t>(cfg->kv_heads) * kMaxSequenceLength * cfg->head_dim;
        for (int layer = 0; layer < cfg->layers; ++layer) {
            if (cfg->is_full(layer)) {
                Zero(full_io[layer].k_cache.get(), cache_count, "zero k cache");
                Zero(full_io[layer].v_cache.get(), cache_count, "zero v cache");
                Zero(full_io[layer].q_buf.get(),
                     static_cast<std::size_t>(q_vectors) * cfg->head_dim, "zero q");
                Zero(full_io[layer].k_buf.get(),
                     static_cast<std::size_t>(k_vectors) * cfg->head_dim, "zero k");
            } else {
                auto& io = linear_io[layer];
                Zero(io.qkv.get(), static_cast<std::size_t>(input_rows) * cfg->conv_channels(), "zero qkv");
                const auto conv_count = static_cast<std::size_t>(input_rows) * cfg->conv_channels();
                const auto state_count = static_cast<std::size_t>(cfg->lin_v_heads) * cfg->lin_v_dim * cfg->lin_k_dim;
                for (int side = 0; side < 2; ++side) {
                    Zero(io.conv[side].get(), conv_count, "zero conv");
                    Zero(io.state[side].get(), state_count, "zero state");
                }
                parity[static_cast<std::size_t>(layer)] = 0;
            }
        }
    }

    void SubmitLinear(int layer, void* hidden) {
        auto& weights = linear[layer];
        auto& io = linear_io[layer];
        const auto side = parity[static_cast<std::size_t>(layer)];
        parity[static_cast<std::size_t>(layer)] = 1 - side;
        const auto& f = api->functions();
        api->Check(f.matmul(stream.get(), hidden, weights.qkv.get(), io.qkv_row.get()), "qkv");
        api->Check(f.matmul_act(stream.get(), ryzenai_corelib_matmul_activation_sigmoid, hidden,
                                weights.beta.get(), nullptr, io.beta.get()),
                   "beta");
        api->Check(f.matmul_act(stream.get(), ryzenai_corelib_matmul_activation_softplus_mulbcast, hidden,
                                weights.alpha.get(), weights.neg_a.get(), io.g.get()),
                   "alpha");
        api->Check(f.dwconv(stream.get(), io.qkv.get(), io.conv[side].get(), 1, weights.conv.get(),
                            io.dw_q.get(), io.dw_k.get(), io.dw_v.get(), io.conv[1 - side].get()),
                   "dwconv");
        api->Check(f.linear_attention(stream.get(), &lin_desc, io.state[side].get(), io.dw_q.get(), io.dw_k.get(),
                                      io.dw_v.get(), io.g_live.get(), io.beta_live.get(), 1, io.lin_out.get(),
                                      io.state[1 - side].get()),
                   "linear attention");
        api->Check(f.rmsnorm(stream.get(), io.lin_out.get(), weights.norm.get(), io.lin_out.get()),
                   "ssm norm");
        api->Check(f.matmul_act(stream.get(), ryzenai_corelib_matmul_activation_silu_mul, hidden,
                                weights.z.get(), io.lin_flat.get(), io.z_out.get()),
                   "z");
        api->Check(f.matmul(stream.get(), io.z_out.get(), weights.out.get(), attn[layer].get()),
                   "linear out");
    }

    void SubmitFull(int layer, void* hidden) {
        auto& weights = full[layer];
        auto& io = full_io[layer];
        const auto& f = api->functions();
        auto& v_row = io.v_rows[static_cast<std::size_t>(position)];
        if (!v_row) {
            v_row = Window(io.v_cache.get(), {cfg->kv_heads, 1, cfg->head_dim},
                           static_cast<std::size_t>(position) * cfg->head_dim, "v row",
                           {kMaxSequenceLength * cfg->head_dim, cfg->head_dim, 1});
        }
        api->Check(f.matmul(stream.get(), hidden, weights.q.get(), io.q_raw.get()), "q");
        api->Check(f.matmul(stream.get(), hidden, weights.k.get(), io.k_live.get()), "k");
        api->Check(f.matmul(stream.get(), hidden, weights.v.get(), v_row.get()), "v");
        for (int part = 0; part < 2; ++part) {
            api->Check(f.rmsnorm(stream.get(), io.q_src[part].get(), weights.q_norm.get(),
                                 io.q_dst[part].get()),
                       "q norm");
        }
        api->Check(f.rmsnorm(stream.get(), io.k_buf.get(), weights.k_norm.get(), io.k_buf.get()), "k norm");
        api->Check(f.flat_mha(stream.get(), &mha_desc, io.q_live.get(), io.k_live.get(), position, cos.get(),
                              sin.get(), io.k_cache.get(), io.v_cache.get(), io.mha_out.get()),
                   "flat mha");
        api->Check(f.matmul_act(stream.get(), ryzenai_corelib_matmul_activation_sigmoid_mul, hidden,
                                weights.gate.get(), io.mha_out.get(), io.gated.get()),
                   "output gate");
        api->Check(f.matmul(stream.get(), io.gated.get(), weights.o.get(), attn[layer].get()), "o proj");
    }

    buffer<bf16> StepBody(int token) {
        if (position >= kMaxSequenceLength)
            throw std::out_of_range("Qwen3.5 position is past the KV cache");
        if (static_cast<std::uint32_t>(position) >= max_length)
            throw std::out_of_range("Qwen3.5 request exceeds configured context capacity");
        const int id = token;
        const auto embedded = rai::DecodeQ8Rows(package->Embedding(), std::span<const int>(&id, 1));
        bool submitted = false;
        try {
            Write(res_in.get(), embedded.data(), embedded.size(), "embedding");
            api->Check(api->functions().rmsnorm(stream.get(), res_in.get(), norm0.get(), h_in.get()),
                       "input norm");
            submitted = true;
            void* residual = res_in.get();
            void* hidden = h_in.get();
            for (int layer = 0; layer < cfg->layers; ++layer) {
                if (cfg->is_full(layer)) SubmitFull(layer, hidden);
                else SubmitLinear(layer, hidden);
                const auto& f = api->functions();
                void* mlp = cfg->is_full(layer) ? full[layer].mlp.get() : linear[layer].mlp.get();
                api->Check(f.ssmlp(stream.get(), attn[layer].get(), residual, mlp, res[layer].get(),
                                   h[layer].get()),
                           "ssmlp");
                residual = res[layer].get();
                hidden = h[layer].get();
            }
            api->Check(api->functions().matmul(stream.get(), hidden, lm_head.get(), logits.get()), "lm_head");
            api->Check(api->functions().stream_synchronize(stream.get()),
                       "ryzenai_corelib_stream_synchronize");
            ++position;
            buffer<bf16> out(static_cast<std::size_t>(cfg->vocab));
            api->Check(api->functions().tensor_read(logits.get(), ryzenai_corelib_data_type_bf16, out.data(),
                                                    out.size(), 0),
                       "ryzenai_corelib_tensor_read logits");
            return out;
        } catch (...) {
            if (submitted) {
                (void)api->functions().stream_synchronize(stream.get());
                poisoned = true;
                position = 0;
                saved.reset();
                history.clear();
            }
            throw;
        }
    }

    std::shared_ptr<Qwen35Gguf> package;
    std::shared_ptr<CorelibRuntime> runtime;
    std::shared_ptr<const CorelibApi> api;
    const Qwen35Config* cfg = nullptr;
    float cfg_eps = 0;
    std::vector<std::uint16_t> epsilon_bf16;
    ryzenai_corelib_linear_attention_bf16_desc lin_desc{};
    ryzenai_corelib_flat_mha_bf16_desc mha_desc{};
    int beta_width = 0;
    int alpha_width = 0;
    int q_vectors = 0;
    int k_vectors = 0;
    int input_rows = 0;
    std::vector<float> cos_table, sin_table, zeros;
    std::vector<int> history;
    std::vector<int> parity;

    UniqueStream stream;
    UniqueRmsNormWeights norm0;
    UniqueMatMulWeights lm_head;
    UniqueHostView cos, sin;
    std::vector<FullWeights> full;
    std::vector<LinearWeights> linear;
    UniqueTensor res_in, h_in, logits;
    std::vector<UniqueTensor> res, h, attn;
    std::vector<FullIo> full_io;
    std::vector<LinearIo> linear_io;
};

qwen35_rai::qwen35_rai(LM_Config, std::shared_ptr<Qwen35Gguf> package,
                       std::shared_ptr<corelib::CorelibRuntime> runtime, std::uint32_t max_length)
    : impl_(std::make_unique<Impl>(std::move(package), std::move(runtime), max_length)) {}
qwen35_rai::~qwen35_rai() = default;

buffer<bf16> qwen35_rai::forward(int id) { return impl_->Forward(id); }
buffer<bf16> qwen35_rai::prefill(std::vector<int>& ids, void*) { return impl_->Prefill(ids); }

void qwen35_rai::set_context_length(int n) { impl_->Rewind(n); }

void qwen35_rai::load_weights(Q4NX&) {
    impl_->usable();
    throw std::runtime_error("Qwen3.5 rai weights are loaded only from GGUF");
}

void qwen35_rai::update_max_length(std::uint32_t n) {
    impl_->usable();
    if (!n || n > kMaxSequenceLength || n < static_cast<std::uint32_t>(impl_->position))
        throw std::out_of_range("Qwen3.5 maximum length is invalid");
    impl_->max_length = n;
}

void qwen35_rai::clear_context() { impl_->Clear(); }

buffer<bf16> qwen35_rai::get_k_cache(int layer, int index) { return impl_->ReadCache(true, layer, index); }
buffer<bf16> qwen35_rai::get_v_cache(int layer, int index) { return impl_->ReadCache(false, layer, index); }

int qwen35_rai::get_current_context_length() {
    impl_->usable();
    return impl_->position;
}

int qwen35_rai::checkpoint() {
    impl_->usable();
    impl_->saved = impl_->position;
    return impl_->position;
}

int qwen35_rai::restore() {
    impl_->usable();
    if (!impl_->saved) return -1;
    impl_->Rewind(*impl_->saved);
    return impl_->position;
}

bool qwen35_rai::poisoned() const noexcept { return impl_ && impl_->poisoned; }

}  // namespace flm::qwen35
