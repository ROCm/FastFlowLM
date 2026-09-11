#include "models/phi4/phi4_corelib_gguf.hpp"

#include "models/phi4/phi4_corelib_constants.hpp"

#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <variant>

namespace flm::phi4 {
namespace {
constexpr std::uint32_t kMagic = 0x46554747;
constexpr std::uint32_t kVersion = 3;
constexpr std::uint32_t kTypeF32 = 0;
constexpr std::uint32_t kTypeQ8_0 = 8;

[[noreturn]] void Fail(std::string_view field, std::string actual,
                       std::string expected) {
    throw std::runtime_error(std::string(field) + ": actual " + actual +
                             ", expected " + expected);
}

std::uint64_t CheckedAdd(std::uint64_t a, std::uint64_t b,
                         std::string_view field) {
    if (a > std::numeric_limits<std::uint64_t>::max() - b)
        throw std::runtime_error(std::string(field) + ": overflow in addition");
    return a + b;
}

std::uint64_t CheckedMultiply(std::uint64_t a, std::uint64_t b,
                              std::string_view field) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a)
        throw std::runtime_error(std::string(field) + ": overflow in product");
    return a * b;
}

std::uint64_t AlignUp(std::uint64_t value, std::uint64_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0)
        Fail("general.alignment", std::to_string(alignment), "a non-zero power of two");
    return CheckedAdd(value, alignment - 1, "alignment") & ~(alignment - 1);
}

std::span<const std::byte> RequireRange(std::span<const std::byte> file,
                                        std::uint64_t offset,
                                        std::uint64_t length,
                                        std::string_view field) {
    const auto end = CheckedAdd(offset, length, field);
    if (end > file.size() || offset > std::numeric_limits<std::size_t>::max() ||
        length > std::numeric_limits<std::size_t>::max())
        Fail(field, "out-of-file range", "range within mapped file");
    return file.subspan(static_cast<std::size_t>(offset),
                        static_cast<std::size_t>(length));
}

class Cursor {
public:
    Cursor(std::span<const std::byte> file, std::uint64_t offset = 0)
        : file_(file), offset_(offset) {}

    template <typename T>
    T Read(std::string_view field) {
        const auto bytes = RequireRange(file_, offset_, sizeof(T), field);
        T value;
        std::memcpy(&value, bytes.data(), sizeof(T));
        offset_ = CheckedAdd(offset_, sizeof(T), field);
        return value;
    }

    std::string ReadString(std::string_view field) {
        const auto length = Read<std::uint64_t>(field);
        const auto bytes = RequireRange(file_, offset_, length, field);
        std::string value(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        offset_ = CheckedAdd(offset_, length, field);
        return value;
    }

    void Skip(std::uint64_t length, std::string_view field) {
        RequireRange(file_, offset_, length, field);
        offset_ = CheckedAdd(offset_, length, field);
    }

    std::uint64_t offset() const noexcept { return offset_; }

private:
    std::span<const std::byte> file_;
    std::uint64_t offset_;
};

std::string MetadataTypeName(std::uint32_t type) {
    static constexpr const char* names[] = {"UINT8", "INT8", "UINT16", "INT16",
        "UINT32", "INT32", "FLOAT32", "BOOL", "STRING", "ARRAY", "UINT64",
        "INT64", "FLOAT64"};
    return type < std::size(names) ? names[type] : "unknown(" + std::to_string(type) + ")";
}

std::uint64_t FixedMetadataSize(std::uint32_t type) {
    switch (type) {
    case 0: case 1: case 7: return 1;
    case 2: case 3: return 2;
    case 4: case 5: case 6: return 4;
    case 10: case 11: case 12: return 8;
    default: return 0;
    }
}

struct ArrayInfo { std::uint32_t type; std::uint64_t count; };
using MetadataValue = std::variant<std::monostate, std::uint64_t, std::int64_t,
                                   double, bool, std::string, ArrayInfo>;

MetadataValue ReadMetadataValue(Cursor& cursor, std::uint32_t type,
                                std::string_view field, bool retain) {
    switch (type) {
    case 0: { auto v = cursor.Read<std::uint8_t>(field); return retain ? MetadataValue(std::uint64_t(v)) : MetadataValue{}; }
    case 1: { auto v = cursor.Read<std::int8_t>(field); return retain ? MetadataValue(std::int64_t(v)) : MetadataValue{}; }
    case 2: { auto v = cursor.Read<std::uint16_t>(field); return retain ? MetadataValue(std::uint64_t(v)) : MetadataValue{}; }
    case 3: { auto v = cursor.Read<std::int16_t>(field); return retain ? MetadataValue(std::int64_t(v)) : MetadataValue{}; }
    case 4: { auto v = cursor.Read<std::uint32_t>(field); return retain ? MetadataValue(std::uint64_t(v)) : MetadataValue{}; }
    case 5: { auto v = cursor.Read<std::int32_t>(field); return retain ? MetadataValue(std::int64_t(v)) : MetadataValue{}; }
    case 6: { auto v = cursor.Read<float>(field); return retain ? MetadataValue(double(v)) : MetadataValue{}; }
    case 7: { auto v = cursor.Read<std::uint8_t>(field); if (v > 1) Fail(field, std::to_string(v), "GGUF boolean 0 or 1"); return retain ? MetadataValue(bool(v)) : MetadataValue{}; }
    case 8: { auto v = cursor.ReadString(field); return retain ? MetadataValue(std::move(v)) : MetadataValue{}; }
    case 9: {
        const std::string array_field = std::string(field) + " array";
        const auto element_type = cursor.Read<std::uint32_t>(array_field);
        const auto count = cursor.Read<std::uint64_t>(array_field);
        if (element_type == 9 || element_type > 12)
            Fail(array_field, MetadataTypeName(element_type), "a skippable GGUF array element type");
        const auto fixed = FixedMetadataSize(element_type);
        if (fixed != 0) {
            cursor.Skip(CheckedMultiply(count, fixed, array_field), array_field);
        } else {
            const auto minimum = CheckedMultiply(count, std::uint64_t{8}, array_field);
            (void)minimum;
            for (std::uint64_t i = 0; i < count; ++i)
                (void)ReadMetadataValue(cursor, element_type, array_field, false);
        }
        return retain ? MetadataValue(ArrayInfo{element_type, count}) : MetadataValue{};
    }
    case 10: { auto v = cursor.Read<std::uint64_t>(field); return retain ? MetadataValue(v) : MetadataValue{}; }
    case 11: { auto v = cursor.Read<std::int64_t>(field); return retain ? MetadataValue(v) : MetadataValue{}; }
    case 12: { auto v = cursor.Read<double>(field); return retain ? MetadataValue(v) : MetadataValue{}; }
    default:
        Fail(field, MetadataTypeName(type), "a supported metadata type");
    }
}

bool IsRetainedKey(std::string_view key) {
    static constexpr std::string_view keys[] = {
        "general.architecture", "general.alignment", "phi3.block_count",
        "phi3.context_length", "phi3.embedding_length", "phi3.feed_forward_length",
        "phi3.attention.head_count", "phi3.attention.head_count_kv",
        "phi3.attention.layer_norm_rms_epsilon", "phi3.rope.dimension_count",
        "phi3.rope.freq_base", "phi3.rope.scaling.attn_factor",
        "phi3.rope.scaling.original_context_length", "tokenizer.ggml.tokens",
        "tokenizer.ggml.add_bos_token", "tokenizer.ggml.eos_token_id"};
    return std::find(std::begin(keys), std::end(keys), key) != std::end(keys);
}

std::string ShapeText(std::span<const std::int64_t> shape) {
    std::ostringstream out;
    out << '[';
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i) out << ',';
        out << shape[i];
    }
    return out.str() + ']';
}

std::string GgmlTypeName(std::uint32_t type) {
    if (type == kTypeF32) return "F32";
    if (type == kTypeQ8_0) return "Q8_0";
    return "GGML type " + std::to_string(type);
}

std::uint64_t ElementCount(std::span<const std::int64_t> shape,
                           std::string_view field) {
    std::uint64_t result = 1;
    for (const auto dimension : shape) {
        if (dimension <= 0) Fail(field, std::to_string(dimension), "positive dimensions");
        result = CheckedMultiply(result, static_cast<std::uint64_t>(dimension), field);
    }
    return result;
}

std::uint64_t TensorByteLength(std::uint32_t type,
                               std::span<const std::int64_t> shape,
                               std::string_view field) {
    const auto elements = ElementCount(shape, field);
    if (type == kTypeF32) return CheckedMultiply(elements, 4, field);
    if (type == kTypeQ8_0) {
        if (elements % 32 != 0)
            Fail(field, std::to_string(elements) + " elements", "Q8_0 element count divisible by 32");
        return CheckedMultiply(elements / 32, 34, field);
    }
    Fail(field, GgmlTypeName(type), "F32 or Q8_0");
}

std::string JsonText(const nlohmann::json& value) {
    return value.dump();
}

template <typename T>
void RequireJson(const nlohmann::json& object, std::string_view key,
                 const T& expected) {
    const auto it = object.find(std::string(key));
    if (it == object.end()) Fail(key, "missing", nlohmann::json(expected).dump());
    try {
        if (it->template get<T>() != expected)
            Fail(key, JsonText(*it), nlohmann::json(expected).dump());
    } catch (const nlohmann::json::exception&) {
        Fail(key, JsonText(*it), nlohmann::json(expected).dump());
    }
}

void RequireJsonDouble(const nlohmann::json& object, std::string_view key,
                       double expected) {
    const auto it = object.find(std::string(key));
    if (it == object.end() || !it->is_number())
        Fail(key, it == object.end() ? "missing" : JsonText(*it), std::to_string(expected));
    const auto actual = it->get<double>();
    if (!std::isfinite(actual) || actual != expected)
        Fail(key, JsonText(*it), std::to_string(expected));
}
}  // namespace

struct Phi4GgufPackage::Impl {
    struct TensorRecord {
        std::string name;
        std::span<const std::byte> bytes;
        std::vector<std::int64_t> shape;
        std::uint32_t type;
        std::uint64_t absolute_offset;
    };

    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE mapping = nullptr;
    const std::byte* data = nullptr;
    std::uint64_t size = 0;
    std::map<std::string, TensorRecord, std::less<>> tensors;
    std::map<std::string, MetadataValue, std::less<>> metadata;
    std::map<std::string, std::uint32_t, std::less<>> metadata_types;

    ~Impl() {
        if (data) UnmapViewOfFile(data);
        if (mapping) CloseHandle(mapping);
        if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
    }

    std::span<const std::byte> bytes() const {
        return {data, static_cast<std::size_t>(size)};
    }

    const TensorRecord& Tensor(std::string_view name) const {
        const auto it = tensors.find(name);
        if (it == tensors.end()) Fail(name, "missing", "present tensor");
        return it->second;
    }

    std::uint64_t Unsigned(std::string_view key) const {
        const auto it = metadata.find(key);
        if (it == metadata.end()) Fail(key, "missing", "unsigned integer metadata");
        if (const auto* value = std::get_if<std::uint64_t>(&it->second)) return *value;
        Fail(key, MetadataTypeName(metadata_types.at(std::string(key))), "unsigned integer metadata");
    }

    double Number(std::string_view key) const {
        const auto it = metadata.find(key);
        if (it == metadata.end()) Fail(key, "missing", "floating-point metadata");
        if (const auto* value = std::get_if<double>(&it->second)) return *value;
        Fail(key, MetadataTypeName(metadata_types.at(std::string(key))), "floating-point metadata");
    }

    bool Boolean(std::string_view key) const {
        const auto it = metadata.find(key);
        if (it == metadata.end()) Fail(key, "missing", "boolean metadata");
        if (const auto* value = std::get_if<bool>(&it->second)) return *value;
        Fail(key, MetadataTypeName(metadata_types.at(std::string(key))), "boolean metadata");
    }

    std::string String(std::string_view key) const {
        const auto it = metadata.find(key);
        if (it == metadata.end()) Fail(key, "missing", "string metadata");
        if (const auto* value = std::get_if<std::string>(&it->second)) return *value;
        Fail(key, MetadataTypeName(metadata_types.at(std::string(key))), "string metadata");
    }

    std::uint64_t ArrayCount(std::string_view key) const {
        const auto it = metadata.find(key);
        if (it == metadata.end()) Fail(key, "missing", "array metadata");
        if (const auto* value = std::get_if<ArrayInfo>(&it->second)) return value->count;
        Fail(key, MetadataTypeName(metadata_types.at(std::string(key))), "array metadata");
    }
};

Phi4GgufPackage::Phi4GgufPackage(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
Phi4GgufPackage::~Phi4GgufPackage() = default;

std::shared_ptr<Phi4GgufPackage> Phi4GgufPackage::Open(
    const std::filesystem::path& gguf_path) {
    auto impl = std::make_unique<Impl>();
    impl->file = CreateFileW(gguf_path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                             nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (impl->file == INVALID_HANDLE_VALUE)
        throw std::runtime_error("GGUF file: actual open failure " +
                                 std::to_string(GetLastError()) + ", expected readable file");
    LARGE_INTEGER size;
    if (!GetFileSizeEx(impl->file, &size) || size.QuadPart <= 0 ||
        static_cast<unsigned long long>(size.QuadPart) > std::numeric_limits<std::size_t>::max())
        Fail("GGUF file size", std::to_string(size.QuadPart), "positive mappable size");
    impl->size = static_cast<std::uint64_t>(size.QuadPart);
    impl->mapping = CreateFileMappingW(impl->file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!impl->mapping)
        throw std::runtime_error("GGUF mapping: actual CreateFileMappingW failure " +
                                 std::to_string(GetLastError()) + ", expected PAGE_READONLY mapping");
    impl->data = static_cast<const std::byte*>(
        MapViewOfFile(impl->mapping, FILE_MAP_READ, 0, 0, 0));
    if (!impl->data)
        throw std::runtime_error("GGUF mapping: actual MapViewOfFile failure " +
                                 std::to_string(GetLastError()) + ", expected FILE_MAP_READ view");

    const auto file = impl->bytes();
    Cursor cursor(file);
    if (cursor.Read<std::uint32_t>("GGUF header") != kMagic)
        Fail("GGUF magic", "mismatch", "0x46554747");
    const auto version = cursor.Read<std::uint32_t>("GGUF header");
    if (version != kVersion) Fail("GGUF version", std::to_string(version), "3");
    const auto tensor_count = cursor.Read<std::uint64_t>("tensor count");
    const auto metadata_count = cursor.Read<std::uint64_t>("metadata count");
    if (tensor_count > file.size() / 24) Fail("tensor count", std::to_string(tensor_count), "count fitting directory");
    if (metadata_count > file.size() / 12) Fail("metadata count", std::to_string(metadata_count), "count fitting metadata");

    for (std::uint64_t i = 0; i < metadata_count; ++i) {
        const auto key = cursor.ReadString("metadata key string");
        const auto type = cursor.Read<std::uint32_t>(key);
        const bool retain = IsRetainedKey(key);
        auto value = ReadMetadataValue(cursor, type, key, retain);
        if (retain) {
            if (!impl->metadata.emplace(key, std::move(value)).second)
                Fail(key, "duplicate metadata key", "unique metadata key");
            impl->metadata_types.emplace(key, type);
        }
    }

    const auto alignment = impl->Unsigned("general.alignment");
    if (alignment == 0 || (alignment & (alignment - 1)) != 0)
        Fail("general.alignment", std::to_string(alignment), "a non-zero power of two");

    struct DirectoryTensor {
        std::string name;
        std::vector<std::int64_t> shape;
        std::uint32_t type;
        std::uint64_t relative_offset;
        std::uint64_t length;
    };
    std::vector<DirectoryTensor> directory;
    directory.reserve(static_cast<std::size_t>(tensor_count));
    for (std::uint64_t i = 0; i < tensor_count; ++i) {
        auto name = cursor.ReadString("tensor directory name");
        const auto dimension_count = cursor.Read<std::uint32_t>("tensor directory dimensions");
        if (dimension_count == 0 || dimension_count > 4)
            Fail(name, std::to_string(dimension_count), "1..4 tensor dimensions");
        std::vector<std::int64_t> shape;
        shape.reserve(dimension_count);
        for (std::uint32_t d = 0; d < dimension_count; ++d) {
            const auto dimension = cursor.Read<std::uint64_t>("tensor directory dimension");
            if (dimension > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
                Fail(name, std::to_string(dimension), "dimension fitting int64");
            shape.push_back(static_cast<std::int64_t>(dimension));
        }
        std::reverse(shape.begin(), shape.end());
        const auto type = cursor.Read<std::uint32_t>("tensor directory type");
        const auto offset = cursor.Read<std::uint64_t>("tensor directory offset");
        const auto length = TensorByteLength(type, shape, name);
        directory.push_back({std::move(name), std::move(shape), type, offset, length});
    }

    const auto data_start = AlignUp(cursor.offset(), alignment);
    struct Range { std::uint64_t begin, end; std::string name; };
    std::vector<Range> ranges;
    ranges.reserve(directory.size());
    for (auto& tensor : directory) {
        if (tensor.relative_offset % alignment != 0)
            Fail(tensor.name, std::to_string(tensor.relative_offset), "offset aligned to " + std::to_string(alignment));
        const auto absolute = CheckedAdd(data_start, tensor.relative_offset, tensor.name);
        const auto bytes = RequireRange(file, absolute, tensor.length, tensor.name + " range");
        const auto end = CheckedAdd(absolute, tensor.length, tensor.name);
        ranges.push_back({absolute, end, tensor.name});
        auto [it, inserted] = impl->tensors.emplace(tensor.name,
            Impl::TensorRecord{tensor.name, bytes, std::move(tensor.shape), tensor.type, absolute});
        if (!inserted) Fail(tensor.name, "duplicate tensor name", "unique tensor name");
    }
    std::sort(ranges.begin(), ranges.end(), [](const Range& a, const Range& b) {
        return a.begin < b.begin;
    });
    for (std::size_t i = 1; i < ranges.size(); ++i) {
        if (ranges[i].begin < ranges[i - 1].end)
            Fail(ranges[i].name, "overlap with " + ranges[i - 1].name, "non-overlapping tensor range");
    }
    return std::shared_ptr<Phi4GgufPackage>(new Phi4GgufPackage(std::move(impl)));
}

TensorView Phi4GgufPackage::RequireQ8(
    std::string_view name, std::span<const std::int64_t> expected_shape) const {
    const auto& tensor = impl_->Tensor(name);
    if (tensor.type != kTypeQ8_0)
        Fail(name, GgmlTypeName(tensor.type), "Q8_0");
    if (!std::equal(tensor.shape.begin(), tensor.shape.end(), expected_shape.begin(), expected_shape.end()))
        Fail(name, ShapeText(tensor.shape), ShapeText(expected_shape));
    const auto expected_length = TensorByteLength(kTypeQ8_0, expected_shape, name);
    if (tensor.bytes.size() != expected_length)
        Fail(name, std::to_string(tensor.bytes.size()) + " bytes", std::to_string(expected_length) + " bytes");
    return {tensor.name, tensor.bytes, tensor.shape, tensor.type};
}

FloatTensorView Phi4GgufPackage::RequireF32(
    std::string_view name, std::span<const std::int64_t> expected_shape) const {
    const auto& tensor = impl_->Tensor(name);
    if (tensor.type != kTypeF32)
        Fail(name, GgmlTypeName(tensor.type), "F32");
    if (!std::equal(tensor.shape.begin(), tensor.shape.end(), expected_shape.begin(), expected_shape.end()))
        Fail(name, ShapeText(tensor.shape), ShapeText(expected_shape));
    const auto expected_length = TensorByteLength(kTypeF32, expected_shape, name);
    if (tensor.bytes.size() != expected_length)
        Fail(name, std::to_string(tensor.bytes.size()) + " bytes", std::to_string(expected_length) + " bytes");
    return {tensor.name,
            {reinterpret_cast<const float*>(tensor.bytes.data()),
             tensor.bytes.size() / sizeof(float)},
            tensor.shape};
}

ProjectionViews Phi4GgufPackage::AttentionQkv(std::size_t layer) const {
    if (layer >= static_cast<std::size_t>(kLayerCount))
        Fail("attention layer", std::to_string(layer), "0..31");
    const auto name = "blk." + std::to_string(layer) + ".attn_qkv.weight";
    const auto& tensor = impl_->Tensor(name);
    if (tensor.shape.size() == 2 && tensor.shape[1] % 32 != 0)
        Fail(name, std::to_string(tensor.shape[1]), "Q8_0 whole-row width divisible by 32");
    const auto fused = RequireQ8(name, std::array<std::int64_t, 2>{5120, 3072});
    const auto input_width = fused.logical_shape[1];
    const auto row_bytes = static_cast<std::size_t>(input_width / 32 * 34);
    ProjectionViews result{};
    result.count = 3;
    result.values[0] = {fused.name, fused.bytes.subspan(0, 3072 * row_bytes), {3072, 3072}, kTypeQ8_0};
    result.values[1] = {fused.name, fused.bytes.subspan(3072 * row_bytes, 1024 * row_bytes), {1024, 3072}, kTypeQ8_0};
    result.values[2] = {fused.name, fused.bytes.subspan(4096 * row_bytes, 1024 * row_bytes), {1024, 3072}, kTypeQ8_0};
    return result;
}

ProjectionViews Phi4GgufPackage::GateUp(std::size_t layer) const {
    if (layer >= static_cast<std::size_t>(kLayerCount))
        Fail("MLP layer", std::to_string(layer), "0..31");
    const auto name = "blk." + std::to_string(layer) + ".ffn_up.weight";
    const auto& tensor = impl_->Tensor(name);
    if (tensor.shape.size() == 2 && tensor.shape[1] % 32 != 0)
        Fail(name, std::to_string(tensor.shape[1]), "Q8_0 whole-row width divisible by 32");
    const auto fused = RequireQ8(name, std::array<std::int64_t, 2>{16384, 3072});
    const auto input_width = fused.logical_shape[1];
    const auto row_bytes = static_cast<std::size_t>(input_width / 32 * 34);
    ProjectionViews result{};
    result.count = 2;
    result.values[0] = {fused.name, fused.bytes.subspan(0, 8192 * row_bytes), {8192, 3072}, kTypeQ8_0};
    result.values[1] = {fused.name, fused.bytes.subspan(8192 * row_bytes, 8192 * row_bytes), {8192, 3072}, kTypeQ8_0};
    return result;
}

GgufPhi4Metadata Phi4GgufPackage::Metadata() const {
    return {impl_->String("general.architecture"),
            impl_->Unsigned("phi3.block_count"),
            impl_->Unsigned("phi3.embedding_length"),
            impl_->Unsigned("phi3.feed_forward_length"),
            impl_->Unsigned("phi3.attention.head_count"),
            impl_->Unsigned("phi3.attention.head_count_kv"),
            impl_->Unsigned("phi3.context_length"),
            impl_->Unsigned("phi3.rope.dimension_count"),
            impl_->Number("phi3.rope.freq_base"),
            impl_->Number("phi3.rope.scaling.attn_factor"),
            impl_->Unsigned("phi3.rope.scaling.original_context_length"),
            impl_->ArrayCount("tokenizer.ggml.tokens"),
            impl_->Boolean("tokenizer.ggml.add_bos_token")};
}

void Phi4GgufPackage::ValidatePhi4Contract(
    const nlohmann::json& config, const nlohmann::json& tokenizer,
    const nlohmann::json& tokenizer_config) const {
    const auto metadata = Metadata();
    const auto require_unsigned = [](std::string_view field, std::uint64_t actual,
                                     std::uint64_t expected) {
        if (actual != expected) Fail(field, std::to_string(actual), std::to_string(expected));
    };
    if (metadata.architecture != "phi3") Fail("general.architecture", metadata.architecture, "phi3");
    require_unsigned("phi3.block_count", metadata.layer_count, kLayerCount);
    require_unsigned("phi3.context_length", metadata.context_length, kMaxSequenceLength);
    require_unsigned("phi3.embedding_length", metadata.hidden_size, kHiddenSize);
    require_unsigned("phi3.feed_forward_length", metadata.intermediate_size, kIntermediateSize);
    require_unsigned("phi3.attention.head_count", metadata.attention_head_count, kQueryHeadCount);
    require_unsigned("phi3.attention.head_count_kv", metadata.kv_head_count, kKvHeadCount);
    require_unsigned("phi3.rope.dimension_count", metadata.rope_dimension_count, kRopeDimension);
    require_unsigned("phi3.rope.scaling.original_context_length", metadata.rope_original_context_length, kMaxSequenceLength);
    require_unsigned("tokenizer.ggml.tokens", metadata.tokenizer_vocabulary_size, kVocabularySize);
    if (metadata.add_bos_token) Fail("tokenizer.ggml.add_bos_token", "true", "false");
    const auto rms = impl_->Number("phi3.attention.layer_norm_rms_epsilon");
    if (!std::isfinite(rms) || rms != static_cast<double>(kRmsEpsilon))
        Fail("phi3.attention.layer_norm_rms_epsilon", std::to_string(rms), std::to_string(kRmsEpsilon));
    for (const auto [field, value] : std::array{
             std::pair<std::string_view, double>{"phi3.rope.freq_base", metadata.rope_frequency_base},
             std::pair<std::string_view, double>{"phi3.rope.scaling.attn_factor", metadata.rope_attention_factor}}) {
        if (!std::isfinite(value) || value <= 0) Fail(field, std::to_string(value), "finite positive value");
    }

    RequireQ8("token_embd.weight", std::array<std::int64_t, 2>{kVocabularySize, kHiddenSize});
    RequireF32("output_norm.weight", std::array<std::int64_t, 1>{kHiddenSize});
    if (impl_->tensors.contains("output.weight")) Fail("output.weight", "present", "absent (tied token_embd.weight)");
    if (impl_->tensors.contains("rope_factors_long.weight")) Fail("rope_factors_long.weight", "present", "absent for original 4096 window");
    for (std::size_t layer = 0; layer < static_cast<std::size_t>(kLayerCount); ++layer) {
        const auto prefix = "blk." + std::to_string(layer);
        RequireF32(prefix + ".attn_norm.weight", std::array<std::int64_t, 1>{kHiddenSize});
        RequireF32(prefix + ".ffn_norm.weight", std::array<std::int64_t, 1>{kHiddenSize});
        RequireQ8(prefix + ".attn_qkv.weight", std::array<std::int64_t, 2>{kQueryDimension + 2 * kKvDimension, kHiddenSize});
        RequireQ8(prefix + ".attn_output.weight", std::array<std::int64_t, 2>{kHiddenSize, kHiddenSize});
        RequireQ8(prefix + ".ffn_up.weight", std::array<std::int64_t, 2>{2 * kIntermediateSize, kHiddenSize});
        RequireQ8(prefix + ".ffn_down.weight", std::array<std::int64_t, 2>{kHiddenSize, kIntermediateSize});
    }
    if (impl_->tensors.contains("rope_factors_short.weight"))
        RequireF32("rope_factors_short.weight", std::array<std::int64_t, 1>{48});

    RequireJson(config, "model_type", std::string("phi3"));
    RequireJson(config, "num_hidden_layers", int(kLayerCount));
    RequireJson(config, "hidden_size", int(kHiddenSize));
    RequireJson(config, "intermediate_size", int(kIntermediateSize));
    RequireJson(config, "num_attention_heads", int(kQueryHeadCount));
    RequireJson(config, "num_key_value_heads", int(kKvHeadCount));
    RequireJson(config, "head_dim", int(kHeadSize));
    RequireJson(config, "vocab_size", int(kVocabularySize));
    RequireJsonDouble(config, "rms_norm_eps", 1.0e-5);
    RequireJson(config, "original_max_position_embeddings", int(kMaxSequenceLength));
    RequireJson(config, "eos_token_id", 199999);

    std::set<std::int64_t> vocabulary_ids;
    std::map<std::string, std::int64_t, std::less<>> token_ids;
    const auto add_token = [&](const std::string& token, std::int64_t id) {
        const auto [it, inserted] = token_ids.emplace(token, id);
        if (!inserted && it->second != id)
            Fail(token, std::to_string(id), std::to_string(it->second));
        vocabulary_ids.insert(id);
    };
    try {
        const auto& vocab = tokenizer.at("model").at("vocab");
        if (!vocab.is_object()) Fail("tokenizer.json model.vocab", JsonText(vocab), "object mapping tokens to IDs");
        for (auto it = vocab.begin(); it != vocab.end(); ++it) {
            const auto id = it.value().get<std::int64_t>();
            add_token(it.key(), id);
        }
        const auto added = tokenizer.find("added_tokens");
        if (added != tokenizer.end()) {
            if (!added->is_array()) Fail("tokenizer.json added_tokens", JsonText(*added), "array");
            for (const auto& item : *added) {
                const auto id = item.at("id").get<std::int64_t>();
                const auto content = item.at("content").get<std::string>();
                add_token(content, id);
            }
        }
    } catch (const nlohmann::json::exception& error) {
        Fail("tokenizer.json vocabulary", error.what(), "valid token-to-ID mappings");
    }
    const auto actual_count = vocabulary_ids.size();
    const auto actual_max_plus_one = vocabulary_ids.empty() ? 0 : *vocabulary_ids.rbegin() + 1;
    if (actual_count != static_cast<std::size_t>(kVocabularySize))
        Fail("tokenizer.json distinct vocabulary ID count", std::to_string(actual_count), std::to_string(kVocabularySize));
    if (actual_max_plus_one != kVocabularySize)
        Fail("tokenizer.json maximum ID plus one", std::to_string(actual_max_plus_one), std::to_string(kVocabularySize));
    for (const auto& [token, expected] : std::array{
             std::pair<std::string_view, std::int64_t>{"<|end|>", 200020},
             std::pair<std::string_view, std::int64_t>{"<|endoftext|>", 199999}}) {
        const auto it = token_ids.find(token);
        if (it == token_ids.end()) Fail(token, "missing", std::to_string(expected));
        if (it->second != expected) Fail(token, std::to_string(it->second), std::to_string(expected));
    }
    const auto gguf_eos = impl_->Unsigned("tokenizer.ggml.eos_token_id");
    if (gguf_eos != 200020) Fail("tokenizer.ggml.eos_token_id", std::to_string(gguf_eos), "200020");

    RequireJson(tokenizer_config, "add_bos_token", false);
    const auto template_it = tokenizer_config.find("chat_template");
    if (template_it == tokenizer_config.end() || !template_it->is_string())
        Fail("chat_template", template_it == tokenizer_config.end() ? "missing" : JsonText(*template_it), "string containing Phi-4 markers");
    const auto chat_template = template_it->get<std::string>();
    for (const auto marker : {"<|user|>", "<|end|>", "<|assistant|>"})
        if (chat_template.find(marker) == std::string::npos)
            Fail(marker, "missing from chat_template", "present in chat_template");
}

}  // namespace flm::phi4
