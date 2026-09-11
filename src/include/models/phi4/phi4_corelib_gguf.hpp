#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace flm::phi4 {

struct TensorView {
    std::string_view name;
    std::span<const std::byte> bytes;
    std::vector<std::int64_t> logical_shape;
    std::uint32_t ggml_type;
};

struct FloatTensorView {
    std::string_view name;
    std::span<const float> values;
    std::vector<std::int64_t> logical_shape;
};

struct ProjectionViews {
    std::array<TensorView, 3> values;
    std::size_t count;
};

struct GgufPhi4Metadata {
    std::string architecture;
    std::uint64_t layer_count;
    std::uint64_t hidden_size;
    std::uint64_t intermediate_size;
    std::uint64_t attention_head_count;
    std::uint64_t kv_head_count;
    std::uint64_t context_length;
    std::uint64_t rope_dimension_count;
    double rope_frequency_base;
    double rope_attention_factor;
    std::uint64_t rope_original_context_length;
    std::uint64_t tokenizer_vocabulary_size;
    bool add_bos_token;
};

class Phi4GgufPackage final {
public:
    static std::shared_ptr<Phi4GgufPackage> Open(
        const std::filesystem::path& gguf_path);
    ~Phi4GgufPackage();

    TensorView RequireQ8(
        std::string_view name,
        std::span<const std::int64_t> expected_shape) const;
    FloatTensorView RequireF32(
        std::string_view name,
        std::span<const std::int64_t> expected_shape) const;
    ProjectionViews AttentionQkv(std::size_t layer) const;
    ProjectionViews GateUp(std::size_t layer) const;
    GgufPhi4Metadata Metadata() const;
    void ValidatePhi4Contract(
        const nlohmann::json& config,
        const nlohmann::json& tokenizer,
        const nlohmann::json& tokenizer_config) const;

private:
    struct Impl;
    explicit Phi4GgufPackage(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace flm::phi4
