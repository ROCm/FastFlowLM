#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace gguf_fixture {

inline constexpr std::uint32_t kF32 = 0;
inline constexpr std::uint32_t kQ8_0 = 8;

enum class Mutation {
    None,
    TruncatedString,
    TruncatedDirectory,
    CountOverflow,
    ProductOverflow,
    OffsetOverflow,
    ZeroAlignment,
    NonPowerOfTwoAlignment,
    DuplicateName,
    OutOfFileRange,
    OverlappingRanges,
    UnsupportedMetadataType,
    DtypeMismatch,
    ShapeMismatch,
    PayloadLengthMismatch,
};

struct ArrayValue {
    std::uint32_t element_type;
    std::uint64_t count;
    std::vector<std::byte> encoded_elements;
};
using MetadataValue = std::variant<std::uint8_t, std::int8_t, std::uint16_t,
                                   std::int16_t, std::uint32_t, std::int32_t,
                                   float, bool, std::string, ArrayValue,
                                   std::uint64_t, std::int64_t, double>;

struct Tensor {
    std::string name;
    std::vector<std::uint64_t> logical_shape;
    std::uint32_t type;
    std::uint64_t offset = 0;
    bool explicit_offset = false;
};

struct TempFile {
    std::filesystem::path path;
    TempFile() = default;
    explicit TempFile(std::filesystem::path value) : path(std::move(value)) {}
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
    TempFile(TempFile&& other) noexcept : path(std::move(other.path)) {
        other.path.clear();
    }
    TempFile& operator=(TempFile&& other) noexcept {
        if (this != &other) {
            std::error_code ignored;
            if (!path.empty()) std::filesystem::remove(path, ignored);
            path = std::move(other.path);
            other.path.clear();
        }
        return *this;
    }
    ~TempFile() {
        std::error_code ignored;
        if (!path.empty()) std::filesystem::remove(path, ignored);
    }
};

template <typename T>
void Append(std::vector<std::byte>& out, T value) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto bytes = std::bit_cast<std::array<std::byte, sizeof(T)>>(value);
    out.insert(out.end(), bytes.begin(), bytes.end());
}

inline void AppendString(std::vector<std::byte>& out, const std::string& value) {
    Append(out, static_cast<std::uint64_t>(value.size()));
    for (const char c : value) out.push_back(static_cast<std::byte>(c));
}

inline std::uint64_t TensorBytes(const Tensor& tensor) {
    std::uint64_t elements = 1;
    for (const auto dimension : tensor.logical_shape) {
        if (dimension != 0 && elements > std::numeric_limits<std::uint64_t>::max() / dimension)
            throw std::overflow_error("fixture tensor product");
        elements *= dimension;
    }
    if (tensor.type == kF32) return elements * 4;
    if (tensor.type == kQ8_0) {
        if (elements % 32 != 0) throw std::runtime_error("fixture Q8_0 divisibility");
        return elements / 32 * 34;
    }
    return elements;
}

class Builder {
public:
    Builder() { AddContractMetadata(); }

    Builder& Alignment(std::uint32_t alignment) {
        alignment_ = alignment;
        SetMetadata("general.alignment", alignment);
        return *this;
    }

    Builder& AddMetadata(std::string key, MetadataValue value) {
        metadata_.emplace_back(std::move(key), std::move(value));
        return *this;
    }

    Builder& SetMetadata(std::string key, MetadataValue value) {
        for (auto& entry : metadata_) {
            if (entry.first == key) {
                entry.second = std::move(value);
                return *this;
            }
        }
        return AddMetadata(std::move(key), std::move(value));
    }

    Builder& RemoveMetadata(const std::string& key) {
        std::erase_if(metadata_, [&](const auto& entry) { return entry.first == key; });
        return *this;
    }

    Builder& AddTensor(std::string name, std::vector<std::uint64_t> logical_shape,
                       std::uint32_t type) {
        tensors_.push_back({std::move(name), std::move(logical_shape), type});
        return *this;
    }

    Builder& AddExactFixtureTensors() {
        AddTensor("token_embd.weight", {200064, 3072}, kQ8_0);
        AddTensor("output_norm.weight", {3072}, kF32);
        AddTensor("blk.0.attn_norm.weight", {3072}, kF32);
        AddTensor("blk.0.ffn_norm.weight", {3072}, kF32);
        AddTensor("blk.0.attn_qkv.weight", {5120, 3072}, kQ8_0);
        AddTensor("blk.0.attn_output.weight", {3072, 3072}, kQ8_0);
        AddTensor("blk.0.ffn_up.weight", {16384, 3072}, kQ8_0);
        AddTensor("blk.0.ffn_down.weight", {3072, 8192}, kQ8_0);
        AddTensor("rope_factors_short.weight", {48}, kF32);
        return *this;
    }

    Builder& AddFullContractTensors(bool short_rope = true) {
        AddTensor("token_embd.weight", {200064, 3072}, kQ8_0);
        AddTensor("output_norm.weight", {3072}, kF32);
        for (std::size_t layer = 0; layer < 32; ++layer) {
            const auto prefix = "blk." + std::to_string(layer);
            AddTensor(prefix + ".attn_norm.weight", {3072}, kF32);
            AddTensor(prefix + ".ffn_norm.weight", {3072}, kF32);
            AddTensor(prefix + ".attn_qkv.weight", {5120, 3072}, kQ8_0);
            AddTensor(prefix + ".attn_output.weight", {3072, 3072}, kQ8_0);
            AddTensor(prefix + ".ffn_up.weight", {16384, 3072}, kQ8_0);
            AddTensor(prefix + ".ffn_down.weight", {3072, 8192}, kQ8_0);
        }
        if (short_rope) AddTensor("rope_factors_short.weight", {48}, kF32);
        return *this;
    }

    Builder& MutateTensor(const std::string& name, std::uint32_t type,
                          std::vector<std::uint64_t> shape) {
        auto* tensor = FindTensor(name);
        if (!tensor) throw std::runtime_error("fixture tensor not found: " + name);
        tensor->type = type;
        tensor->logical_shape = std::move(shape);
        return *this;
    }

    Builder& RemoveTensor(const std::string& name) {
        std::erase_if(tensors_, [&](const Tensor& tensor) { return tensor.name == name; });
        return *this;
    }

    Builder& AddEverySkippableMetadataType() {
        AddMetadata("skip.u8", std::uint8_t{1});
        AddMetadata("skip.i8", std::int8_t{-1});
        AddMetadata("skip.u16", std::uint16_t{2});
        AddMetadata("skip.i16", std::int16_t{-2});
        AddMetadata("skip.u32", std::uint32_t{3});
        AddMetadata("skip.i32", std::int32_t{-3});
        AddMetadata("skip.f32", 1.25f);
        AddMetadata("skip.bool", true);
        AddMetadata("skip.string", std::string("value"));
        std::vector<std::byte> strings;
        AppendString(strings, "a"); AppendString(strings, "bc");
        AddMetadata("skip.array", ArrayValue{8, 2, std::move(strings)});
        AddMetadata("skip.u64", std::uint64_t{4});
        AddMetadata("skip.i64", std::int64_t{-4});
        AddMetadata("skip.f64", 2.5);
        return *this;
    }

    Builder& Apply(Mutation mutation) { mutation_ = mutation; return *this; }

    TempFile Write(std::string_view label = "fixture") const {
        static std::uint64_t serial = 0;
        auto path = std::filesystem::temp_directory_path() /
                    ("flm_phi4_" + std::string(label) + "_" +
                     std::to_string(++serial) + ".gguf");
        auto bytes = Encode();
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        if (!stream) throw std::runtime_error("cannot create fixture");
        stream.write(reinterpret_cast<const char*>(bytes.prefix.data()),
                     static_cast<std::streamsize>(bytes.prefix.size()));
        if (bytes.file_size > bytes.prefix.size()) {
            stream.seekp(static_cast<std::streamoff>(bytes.file_size - 1));
            const char zero = 0;
            stream.write(&zero, 1);
        }
        stream.close();
        return TempFile(path);
    }

private:
    struct Encoded { std::vector<std::byte> prefix; std::uint64_t file_size; };

    void AddContractMetadata() {
        AddMetadata("general.architecture", std::string("phi3"));
        AddMetadata("general.alignment", std::uint32_t{32});
        AddMetadata("phi3.block_count", std::uint32_t{32});
        AddMetadata("phi3.context_length", std::uint32_t{4096});
        AddMetadata("phi3.embedding_length", std::uint32_t{3072});
        AddMetadata("phi3.feed_forward_length", std::uint32_t{8192});
        AddMetadata("phi3.attention.head_count", std::uint32_t{24});
        AddMetadata("phi3.attention.head_count_kv", std::uint32_t{8});
        AddMetadata("phi3.attention.layer_norm_rms_epsilon", 1.0e-5f);
        AddMetadata("phi3.rope.dimension_count", std::uint32_t{96});
        AddMetadata("phi3.rope.freq_base", 10000.0f);
        AddMetadata("phi3.rope.scaling.attn_factor", 1.0f);
        AddMetadata("phi3.rope.scaling.original_context_length", std::uint32_t{4096});
        AddMetadata("tokenizer.ggml.tokens",
                    ArrayValue{0, 200064, std::vector<std::byte>(200064)});
        AddMetadata("tokenizer.ggml.add_bos_token", false);
        AddMetadata("tokenizer.ggml.eos_token_id", std::uint32_t{200020});
    }

    Tensor* FindTensor(const std::string& name) {
        const auto it = std::find_if(tensors_.begin(), tensors_.end(),
            [&](const Tensor& tensor) { return tensor.name == name; });
        return it == tensors_.end() ? nullptr : &*it;
    }

    static std::uint32_t TypeOf(const MetadataValue& value) {
        return static_cast<std::uint32_t>(value.index());
    }

    static void EncodeValue(std::vector<std::byte>& out, const MetadataValue& value) {
        std::visit([&](const auto& item) {
            using T = std::decay_t<decltype(item)>;
            if constexpr (std::is_same_v<T, std::string>) AppendString(out, item);
            else if constexpr (std::is_same_v<T, ArrayValue>) {
                Append(out, item.element_type); Append(out, item.count);
                out.insert(out.end(), item.encoded_elements.begin(), item.encoded_elements.end());
            } else if constexpr (std::is_same_v<T, bool>) Append(out, std::uint8_t(item));
            else Append(out, item);
        }, value);
    }

    Encoded Encode() const {
        auto metadata = metadata_;
        auto tensors = tensors_;
        std::uint32_t alignment = alignment_;
        if (mutation_ == Mutation::ZeroAlignment) alignment = 0;
        if (mutation_ == Mutation::NonPowerOfTwoAlignment) alignment = 24;
        for (auto& entry : metadata)
            if (entry.first == "general.alignment") entry.second = alignment;
        if (mutation_ == Mutation::DuplicateName && !tensors.empty()) tensors.push_back(tensors.front());
        if (mutation_ == Mutation::DtypeMismatch && !tensors.empty()) tensors.front().type = kF32;
        if (mutation_ == Mutation::ShapeMismatch && !tensors.empty()) tensors.front().logical_shape[0]--;

        std::vector<std::byte> out;
        Append(out, std::uint32_t{0x46554747}); Append(out, std::uint32_t{3});
        Append(out, mutation_ == Mutation::CountOverflow ? std::numeric_limits<std::uint64_t>::max()
                                                          : static_cast<std::uint64_t>(tensors.size()));
        Append(out, static_cast<std::uint64_t>(metadata.size()));
        for (const auto& [key, value] : metadata) {
            AppendString(out, key);
            if (mutation_ == Mutation::UnsupportedMetadataType && key == metadata.front().first) {
                Append(out, std::uint32_t{99});
            } else {
                Append(out, TypeOf(value)); EncodeValue(out, value);
            }
        }
        std::uint64_t running = 0;
        for (std::size_t index = 0; index < tensors.size(); ++index) {
            auto& tensor = tensors[index];
            if (mutation_ == Mutation::ProductOverflow && index == 0)
                tensor.logical_shape = {
                    static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()), 3};
            if (tensor.explicit_offset) running = tensor.offset;
            if (alignment != 0 && (alignment & (alignment - 1)) == 0)
                running = (running + alignment - 1) & ~(std::uint64_t(alignment) - 1);
            tensor.offset = running;
            const auto size = mutation_ == Mutation::ProductOverflow && index == 0
                                  ? 0 : TensorBytes(tensor);
            if (mutation_ == Mutation::OverlappingRanges && index == 1) {
                tensor.offset = 0;
                running += size;
            } else if (mutation_ == Mutation::OutOfFileRange && index == 0)
                tensor.offset = std::uint64_t{1} << 40;
            else if (mutation_ == Mutation::OffsetOverflow && index == 0)
                tensor.offset = std::numeric_limits<std::uint64_t>::max() - 31;
            else running += size;
            AppendString(out, tensor.name);
            Append(out, static_cast<std::uint32_t>(tensor.logical_shape.size()));
            for (auto it = tensor.logical_shape.rbegin(); it != tensor.logical_shape.rend(); ++it)
                Append(out, *it);
            Append(out, tensor.type); Append(out, tensor.offset);
        }
        if (mutation_ == Mutation::TruncatedDirectory && !out.empty()) {
            out.pop_back(); return {std::move(out), static_cast<std::uint64_t>(out.size())};
        }
        const auto data_start = alignment == 0 ? static_cast<std::uint64_t>(out.size())
            : (static_cast<std::uint64_t>(out.size()) + alignment - 1) & ~(std::uint64_t(alignment) - 1);
        out.resize(static_cast<std::size_t>(data_start), std::byte{0});
        std::uint64_t file_size = data_start + running;
        if (mutation_ == Mutation::PayloadLengthMismatch && file_size > data_start) --file_size;
        if (mutation_ == Mutation::TruncatedString) {
            const auto impossible = std::bit_cast<std::array<std::byte, sizeof(std::uint64_t)>>(
                std::numeric_limits<std::uint64_t>::max());
            std::copy(impossible.begin(), impossible.end(), out.begin() + 24);
        }
        return {std::move(out), file_size};
    }

    std::uint32_t alignment_ = 32;
    std::vector<std::pair<std::string, MetadataValue>> metadata_;
    std::vector<Tensor> tensors_;
    Mutation mutation_ = Mutation::None;
};

inline nlohmann::json ValidConfig() {
    return {{"model_type", "phi3"}, {"num_hidden_layers", 32},
            {"hidden_size", 3072}, {"intermediate_size", 8192},
            {"num_attention_heads", 24}, {"num_key_value_heads", 8},
            {"head_dim", 128}, {"vocab_size", 200064},
            {"rms_norm_eps", 1.0e-5}, {"original_max_position_embeddings", 4096},
            {"eos_token_id", 199999}};
}

inline nlohmann::json ValidTokenizer() {
    nlohmann::json vocab = nlohmann::json::object();
    for (int id = 0; id < 200062; ++id) vocab["t" + std::to_string(id)] = id;
    vocab["<|endoftext|>"] = 199999;
    vocab["<|end|>"] = 200020;
    return {{"model", {{"vocab", std::move(vocab)}}},
            {"added_tokens", nlohmann::json::array({
                {{"id", 200062}, {"content", "added-a"}},
                {{"id", 200063}, {"content", "added-b"}},
                {{"id", 200020}, {"content", "<|end|>"}},
                {{"id", 199999}, {"content", "<|endoftext|>"}}})}};
}

inline nlohmann::json ValidTokenizerConfig() {
    return {{"add_bos_token", false},
            {"chat_template", "<|user|>{{ message }}<|end|><|assistant|>"}};
}

}  // namespace gguf_fixture
