#pragma once

// A synthetic GGUF writer for Gemma 4's two shipped shapes (E2B and E4B).
//
// This is a second, independent implementation of the same GGUF v3 byte
// layout src/test/phi4_rai/gguf_fixture.hpp's Builder writes -- not an
// extraction of it. That header's Builder hardcodes Phi-4's own contract
// metadata in its constructor (general.architecture = "phi3", phi3.* keys, a
// 200064-entry tokenizer.ggml.tokens array...) and is owned by the agent
// working phi4_rai's corelib tests; reusing it here would mean either
// editing a file this task does not own or fighting its Phi-3-shaped
// defaults from the outside for every metadata key gemma4 needs instead.
// A from-scratch, model-agnostic writer -- generic over metadata and
// tensors, carrying no baked-in architecture -- is cleaner for what this
// fixture needs: two files whose overlap with Phi-4's fixture is the
// container format itself, nothing else.

#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <system_error>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#if defined(_WIN32)
#include <process.h>   // _getpid
#else
#include <unistd.h>    // getpid
#endif

namespace gemma4_fixture {

namespace detail {

/// \brief the process id, folded into every temp filename below so that two
///        instances of the *same* test binary running at once (`ctest -j`,
///        or a developer's scratch repro alongside the real suite) never
///        compute the same path. A monotonic per-process counter (below)
///        still tells two files apart within one process, same as before.
inline unsigned long CurrentProcessId() {
#if defined(_WIN32)
    return static_cast<unsigned long>(_getpid());
#else
    return static_cast<unsigned long>(getpid());
#endif
}

/// \brief a counter unique within this process, atomic because nothing here
///        promises the tests that call Write() stay single-threaded.
inline std::uint64_t NextSerial() {
    static std::atomic<std::uint64_t> serial{0};
    return serial.fetch_add(1, std::memory_order_relaxed) + 1;
}

/// \brief the process id encoded in one of Write()'s filenames
/// \return nullopt when `filename` is not one of ours, or carries no pid
/// \note The name is `flm_gemma4_<label>_<pid>_<serial>.gguf`, and a label is
///       free to contain underscores of its own, so the pid is read from the
///       END -- the second-to-last '_'-separated field -- and never by
///       counting forward from `flm_gemma4_`.
inline std::optional<unsigned long> PidInFixtureName(const std::string& filename) {
    if (filename.rfind("flm_gemma4_", 0) != 0) return std::nullopt;
    if (filename.size() < 6 || filename.compare(filename.size() - 5, 5, ".gguf") != 0)
        return std::nullopt;
    const auto stem = filename.substr(0, filename.size() - 5);
    const auto serial_underscore = stem.rfind('_');
    if (serial_underscore == std::string::npos || serial_underscore == 0)
        return std::nullopt;
    const auto pid_underscore = stem.rfind('_', serial_underscore - 1);
    if (pid_underscore == std::string::npos) return std::nullopt;
    const auto digits =
        stem.substr(pid_underscore + 1, serial_underscore - pid_underscore - 1);
    if (digits.empty() ||
        digits.find_first_not_of("0123456789") != std::string::npos)
        return std::nullopt;
    try {
        return std::stoul(digits);
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

/// \brief delete the fixtures a PREVIOUS run left in %TEMP%
/// \param cutoff only files last written strictly before this are removed
/// \param own_pid this process's id; its own files are never touched
/// \return how many files were removed
///
/// SELF-HEALING, AND SAFE ALONGSIDE A CONCURRENT RUN. The two filters are
/// what make it safe rather than merely lucky:
///
///   - the pid: a sibling `ctest -j` process writes `..._<its pid>_...`, and
///     this only ever removes names carrying a pid that is NOT ours and
///     whose mtime predates our own first write. The pid has been in the
///     filename since the fixture was written; this is the first thing that
///     reads it.
///   - the cutoff: the caller passes the moment this process started
///     writing, so a file a sibling creates AFTER we start is out of range
///     however long our run lasts.
///
/// WHAT IT STILL CANNOT RULE OUT, stated because the bullets above read as a
/// guarantee on their own: a sibling that was started before us, and that
/// wrote a fixture before our first write and has not yet opened it, has a
/// file this would remove. On Windows -- the platform this suite runs on --
/// an OPEN mapping already makes std::filesystem::remove fail, so the window
/// is the few milliseconds between that sibling's write and its open. It is
/// not closed here, and closing it properly needs a lock file or a pid
/// liveness probe, which is more machinery than a test fixture should carry.
inline std::size_t SweepStaleTempFixtures(std::filesystem::file_time_type cutoff,
                                          unsigned long own_pid) {
    std::size_t removed = 0;
    std::error_code error;
    std::filesystem::directory_iterator entry(std::filesystem::temp_directory_path(),
                                              error);
    if (error) return 0;
    const std::filesystem::directory_iterator end;
    for (; entry != end; entry.increment(error)) {
        if (error) break;
        const auto path = entry->path();
        const auto pid = PidInFixtureName(path.filename().string());
        if (!pid || *pid == own_pid) continue;
        std::error_code query_error;
        const auto written = std::filesystem::last_write_time(path, query_error);
        if (query_error || !(written < cutoff)) continue;
        std::error_code remove_error;
        if (std::filesystem::remove(path, remove_error) && !remove_error) ++removed;
    }
    return removed;
}

/// \brief remembers every path Write() has created and deletes all but the
///        newest, so a run of this suite occupies one fixture's worth of
///        %TEMP% rather than one per test
///
/// THE OLD DESIGN DELETED EVERYTHING IN THE DESTRUCTOR AND THAT WAS NOT
/// ENOUGH. A static destructor runs on a `return` from main -- the PASS path
/// -- and test_support.hpp's RunTest deliberately forces a FAILING process
/// down with std::_Exit(1), which bypasses atexit handlers and static
/// destructors by design so the artifact survives for postmortem. Since
/// these fixtures are zero-filled but full-shaped (~4.9 GB for E2B, ~8.2 GB
/// for E4B) and one run of test_gemma4_config writes six, a day of failing
/// runs left 240 files and 1.1 TB behind, filling a 1.9 TB volume to 99%.
///
/// So the sweep moved to where it is reached on every path: `Write()` itself
/// removes the generations before it, BEFORE it adds ~5 GB of its own. That
/// keeps the postmortem artifact -- the newest file, which on a failing run
/// is the one the failing test wrote -- and removes the rest. `Instance()`'s
/// first call additionally sweeps whatever a previous, crashed run left.
///
/// A path that CANNOT be removed is retained and retried at the next
/// `Write()` rather than dropped: on Windows a file another object still has
/// mapped refuses deletion, and a test that holds two fixtures open at once
/// is a legitimate thing to do.
class TempFileRegistry {
public:
    static TempFileRegistry& Instance() {
        static TempFileRegistry instance;
        return instance;
    }

    /// \brief delete every UNPINNED generation recorded so far
    void SweepTracked() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::filesystem::path> retained;
        for (auto& path : paths_) {
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
            std::error_code query_error;
            if (std::filesystem::exists(path, query_error) && !query_error)
                retained.push_back(std::move(path));
        }
        paths_.swap(retained);
    }

    /// \param pinned exempt this path from SweepTracked -- see Write()
    void Track(std::filesystem::path path, bool pinned) {
        std::lock_guard<std::mutex> lock(mutex_);
        (pinned ? pinned_paths_ : paths_).push_back(std::move(path));
    }

    ~TempFileRegistry() {
        std::error_code ignored;
        for (const auto& path : paths_) std::filesystem::remove(path, ignored);
        for (const auto& path : pinned_paths_) std::filesystem::remove(path, ignored);
    }

private:
    TempFileRegistry() {
        // Whatever is in %TEMP% now belongs to a run that is already over:
        // this constructor runs before this process has written anything, so
        // `now()` is a cutoff no file of ours can fall under.
        SweepStaleTempFixtures(std::filesystem::file_time_type::clock::now(),
                               CurrentProcessId());
    }

    std::mutex mutex_;
    std::vector<std::filesystem::path> paths_;
    /// \brief paths SweepTracked leaves alone
    /// \note The exemption is bounded: they are still removed when the
    ///       process returns from main, and by the next run's startup sweep
    ///       if it does not. What they survive is the PER-WRITE sweep, which
    ///       is what a test suite that opens one fixture many times needs.
    std::vector<std::filesystem::path> pinned_paths_;
};

}  // namespace detail

// GGML tensor payload type ids -- matches flm::rai::GgufFile's own
// kTypeF32/kTypeQ8_0 (src/common/rai/gguf_file.cpp), which is *not* the same
// id space as the GGUF metadata value type ids below.
inline constexpr std::uint32_t kF32 = 0;
inline constexpr std::uint32_t kQ8_0 = 8;
/// \brief BF16, ggml type 30 -- two bytes per element, no block structure
/// \note The ONE non-F32, non-Q8_0 type a Gemma 4 conversion ships, and it
///       ships exactly one tensor of it: `per_layer_model_proj.weight`,
///       measured at type 30 on both real rows. This fixture had it as
///       kQ8_0, which was a guess, and the guess was load-bearing: it meant
///       every test here validated readers against a file no vendor ships,
///       while `flm::rai::GgufFile::Open` could not even size a type-30
///       tensor and threw on both real GGUFs.
inline constexpr std::uint32_t kBf16 = 30;

// GGUF v3 metadata value type ids, per the spec flm::rai::GgufFile parses
// against.
inline constexpr std::uint32_t kMetaUint32 = 4;

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
    std::vector<std::uint64_t> shape;
    std::uint32_t type;
};

/// \brief the constant this fixture fills the `index`-th F32 tensor with
///
/// WHY THE F32 PAYLOAD IS NOT ZERO ANY MORE. Every norm in a zero-filled
/// fixture is indistinguishable from every other norm, so "layer i's ple was
/// packed with layer i+1's attn_norm" -- the off-by-one corelib.h says "is
/// not a crash ... still generates text" -- cannot be asserted at all: the
/// engine could pass any gamma and the bytes would match. Giving each F32
/// tensor a distinct constant turns "which tensor was handed to this packer"
/// into something a test can read back off the fake.
///
/// TWO PROPERTIES, AND BOTH ARE LOAD-BEARING:
///
///   - DISTINCT for every index. Derived from the index, not hashed from the
///     name: a hash over ~350 F32 tensors collides often enough (birthday)
///     that an assertion could pass against the WRONG tensor, which is the
///     exact failure mode this fill exists to catch. 128 mantissa values x
///     16 exponents is 2048 distinct constants and both shipped rows have
///     fewer tensors than that in total.
///   - EXACTLY REPRESENTABLE IN BF16. The low 16 bits are zero, so an
///     engine that converts F32 -> BF16 (round-to-nearest-even) on the way
///     to `ssmlp_bf16_weights_create_gguf_requantized` or
///     `rmsnorm_bf16_weights_create_reference` produces the top half of this
///     value unchanged, and a test can compare against it without an
///     epsilon.
inline float SignatureAt(std::size_t index) {
    const std::uint32_t mantissa = static_cast<std::uint32_t>(index % 128);
    const std::uint32_t exponent = 119u + static_cast<std::uint32_t>((index / 128) % 16);
    const std::uint32_t bits = (exponent << 23) | (mantissa << 16);
    return std::bit_cast<float>(bits);
}

/// \brief the BF16 half of SignatureAt, for comparing against a converted one
inline std::uint16_t SignatureAtBf16(std::size_t index) {
    return static_cast<std::uint16_t>(
        std::bit_cast<std::uint32_t>(SignatureAt(index)) >> 16);
}

/// \brief what this fixture writes at element `element` of the `index`-th
///        **2-D** F32 tensor
///
/// A CONSTANT FILL CANNOT SEE A TRANSPOSE, AND THAT IS WHY THIS EXISTS. The
/// transpose of a constant matrix is the same matrix, so with SignatureAt
/// alone no probe of any element of `inp_gate` or `proj` can tell the GGUF's
/// mapping from the orientation corelib's `ple_bf16_weights_pack` actually
/// requires ("gate FP32 [k, n] row-major ... Passing one the other way round
/// packs silently and produces noise", corelib.h). Adding the element's
/// POSITION makes `M[i][j] != M[j][i]`, which is exactly the property a
/// transpose check needs.
///
/// SCOPED TO 2-D F32 TENSORS, AND THAT SCOPE IS THE WHOLE POINT. In this
/// fixture the only 2-D F32 tensors are `blk.N.inp_gate.weight` and
/// `blk.N.proj.weight` -- every norm, `layer_output_scale`, `output_norm`,
/// `per_layer_proj_norm` and `rope_freqs` is 1-D or scalar and keeps its
/// per-tensor CONSTANT. So `SignatureOf` remains a faithful "the whole tensor
/// is this value" for everything the norm assertions lean on
/// (`post_norm.front() == post_norm.back()`, every `SignatureOfBf16`
/// comparison, `PackCallsForLayer`'s keying on `layer_output_scale`), and
/// nothing that used to hold stops holding. An earlier round declined this
/// fill on the grounds that it would collide with the `SignatureOf` contract;
/// that is true only of a fill applied to ALL F32 tensors.
///
/// ELEMENT 0 IS DELIBERATELY UNCHANGED (`+ 0`), so `SignatureOf` still names
/// the tensor for the `gate_first` / `proj_first` "which matrix went into
/// which port" check -- and it names it under BOTH orientations, since a
/// transpose maps element 0 to element 0. That check is about identity; the
/// probes at index 1 and beyond are about orientation, and the two are
/// separate questions.
///
/// \note EXACTNESS. The sum is a float32 addition and is not claimed to be
///       collision-free for arbitrarily large `element` -- at element ~4e5
///       the ULP exceeds the spacing between two neighbouring SignatureAt
///       constants. Every comparison against this fixture goes through
///       `Builder::ElementAt`, i.e. through this same expression, so it is
///       exact regardless; and the probe indices that matter (1, `n`, `k`)
///       are small enough that a wrong orientation is off by thousands of
///       ULPs, not by one.
inline float PositionalValueAt(std::size_t index, std::uint64_t element) {
    return SignatureAt(index) + static_cast<float>(element);
}

/// \brief the four-byte tag this fixture stamps at the head of the
///        `index`-th Q8_0 tensor's payload
///
/// EVERY ZERO-FILLED Q8_0 PAYLOAD IS BYTE-IDENTICAL TO EVERY OTHER, which is
/// why "which quantized tensor reached which corelib port" was unanswerable:
/// corelib's requantizing entry points take a bare `const void*` block
/// stream, so the only thing a fake can compare is the bytes, and there was
/// nothing in them. Swapping `ffn_gate` and `ffn_up` in `ssmlp`'s components
/// -- which computes `GELU(x @ up) * (x @ gate)`, a different function --
/// therefore left a 16/16 green suite.
///
/// NOT A VALID Q8_0 BLOCK, and it does not need to be: nothing in this suite
/// decodes the fixture's quantized payloads (the engine hands the pointer
/// straight to corelib, and the fake reads the tag and no further). A
/// synthetic block stream would be a second thing to keep honest for no gain.
/// The real files are the place quantized *content* is checked, and
/// test_gemma4_real_gguf.cpp does that with its own decoder.
///
/// Four bytes, at offset 0, so stamping it costs one memcpy per tensor rather
/// than a pass over ~5 GB. The high half is a recognizable marker so a zero
/// (i.e. never-stamped) payload cannot be mistaken for tensor 0's tag.
inline std::uint32_t BlockSignatureAt(std::size_t index) {
    return 0xB10C0000u | static_cast<std::uint32_t>(index);
}

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

inline std::uint64_t TensorElementCount(const std::vector<std::uint64_t>& shape) {
    std::uint64_t elements = 1;
    for (const auto dim : shape) {
        if (dim != 0 && elements > std::numeric_limits<std::uint64_t>::max() / dim)
            throw std::overflow_error("gemma4 fixture: tensor element-count overflow");
        elements *= dim;
    }
    return elements;
}

inline std::uint64_t TensorBytes(const Tensor& tensor) {
    const auto elements = TensorElementCount(tensor.shape);
    if (tensor.type == kF32) return elements * 4;
    if (tensor.type == kBf16) return elements * 2;
    if (tensor.type == kQ8_0) {
        if (elements % 32 != 0) {
            throw std::runtime_error("gemma4 fixture: Q8_0 tensor " + tensor.name +
                                     " has " + std::to_string(elements) +
                                     " elements, not divisible by 32");
        }
        return elements / 32 * 34;
    }
    throw std::runtime_error("gemma4 fixture: unsupported tensor type for " + tensor.name);
}

/// \brief a minimal, model-agnostic GGUF v3 writer
class Builder {
public:
    Builder() = default;

    Builder& SetMetadata(std::string key, MetadataValue value) {
        for (auto& entry : metadata_) {
            if (entry.first == key) {
                entry.second = std::move(value);
                return *this;
            }
        }
        metadata_.emplace_back(std::move(key), std::move(value));
        return *this;
    }

    Builder& RemoveMetadata(const std::string& key) {
        std::erase_if(metadata_, [&](const auto& entry) { return entry.first == key; });
        return *this;
    }

    Builder& AddTensor(std::string name, std::vector<std::uint64_t> shape,
                       std::uint32_t type) {
        tensors_.push_back({std::move(name), std::move(shape), type});
        return *this;
    }

    /// \brief how many tensors have been added so far
    /// \note Lets a caller assert `6 + layers * 17` while assembling the
    ///       fixture, not only after Write() has encoded it to disk.
    std::size_t TensorCount() const noexcept { return tensors_.size(); }

    /// \brief the constant every element of one F32 tensor was filled with
    /// \throws std::runtime_error naming the tensor, if it is absent or is
    ///         not an F32 tensor (only F32 payloads are filled -- a Q8_0 one
    ///         would need a block layout this fixture does not synthesize)
    /// \note The value depends only on the tensor's POSITION, so two
    ///       Builders made by the same `MakeBuilder(options)` call agree.
    ///       That is what lets a test compute the expected gamma without
    ///       holding the exact Builder the file was written from.
    float SignatureOf(std::string_view name) const {
        for (std::size_t i = 0; i < tensors_.size(); ++i) {
            if (tensors_[i].name != name) continue;
            if (tensors_[i].type != kF32) {
                throw std::runtime_error("gemma4 fixture: " + std::string(name) +
                                         " is not F32, so it carries no signature");
            }
            return SignatureAt(i);
        }
        throw std::runtime_error("gemma4 fixture: no tensor named " + std::string(name));
    }

    /// \brief SignatureOf, as the BF16 an engine would convert it to
    std::uint16_t SignatureOfBf16(std::string_view name) const {
        return static_cast<std::uint16_t>(
            std::bit_cast<std::uint32_t>(SignatureOf(name)) >> 16);
    }

    /// \brief the four-byte tag this fixture stamped at the head of one Q8_0
    ///        tensor's payload
    /// \throws std::runtime_error naming the tensor, if it is absent or is
    ///         not Q8_0
    /// \note The answer to "which quantized tensor reached this port". See
    ///       BlockSignatureAt.
    std::uint32_t BlockSignatureOf(std::string_view name) const {
        for (std::size_t i = 0; i < tensors_.size(); ++i) {
            if (tensors_[i].name != name) continue;
            if (tensors_[i].type != kQ8_0) {
                throw std::runtime_error("gemma4 fixture: " + std::string(name) +
                                         " is not Q8_0, so it carries no block tag");
            }
            return BlockSignatureAt(i);
        }
        throw std::runtime_error("gemma4 fixture: no tensor named " + std::string(name));
    }

    /// \brief the exact float this fixture wrote at element `element` of an
    ///        F32 tensor -- positional for a 2-D one, the constant otherwise
    /// \throws std::runtime_error naming the tensor, if it is absent, is not
    ///         F32, or `element` is past its end
    /// \note EXISTS SO A TEST NEVER RE-TYPES THE FILL FORMULA. An orientation
    ///       assertion says "corelib's `gate[1]` must be the mapping's
    ///       element `hidden`"; the arithmetic that turns that into a number
    ///       belongs here, once, beside the code that wrote it.
    float ElementAt(std::string_view name, std::uint64_t element) const {
        for (std::size_t i = 0; i < tensors_.size(); ++i) {
            if (tensors_[i].name != name) continue;
            if (tensors_[i].type != kF32) {
                throw std::runtime_error("gemma4 fixture: " + std::string(name) +
                                         " is not F32, so it carries no F32 elements");
            }
            if (element >= TensorElementCount(tensors_[i].shape)) {
                throw std::runtime_error("gemma4 fixture: element " +
                                         std::to_string(element) + " is past the end of " +
                                         std::string(name));
            }
            return tensors_[i].shape.size() == 2 ? PositionalValueAt(i, element)
                                                         : SignatureAt(i);
        }
        throw std::runtime_error("gemma4 fixture: no tensor named " + std::string(name));
    }

    /// \brief encode and write to a fresh temp file, returning its path
    /// \note The path is unique per process AND per call within a process
    ///       (pid + a monotonic counter), so two test binaries running at
    ///       once never collide on the same filename -- see
    ///       detail::TempFileRegistry for how (and when) the file this
    ///       returns gets cleaned up again.
    /// \note EVERY EARLIER GENERATION IS DELETED FIRST, before this call
    ///       adds a multi-gigabyte file of its own -- so the peak is two
    ///       fixtures rather than one per test, and a run that dies without
    ///       unwinding leaves one. A path a test still holds mapped survives
    ///       the attempt and is retried at the next Write; see
    ///       detail::TempFileRegistry.
    /// \param pinned exempt the result from that per-write sweep, for a
    ///        target that writes ONE fixture and opens it from many tests
    ///        (test_gemma4_engine does: the alternative is ~5 GB written per
    ///        test case). Still removed on a normal exit and by the next
    ///        run's startup sweep. Use it for a handful of files, never per
    ///        test -- the whole point of the sweep is that the leftovers are
    ///        bounded by a small constant rather than by the test count.
    std::filesystem::path Write(std::string_view label = "gemma4",
                                bool pinned = false) const {
        detail::TempFileRegistry::Instance().SweepTracked();
        auto path = std::filesystem::temp_directory_path() /
                    ("flm_gemma4_" + std::string(label) + "_" +
                     std::to_string(detail::CurrentProcessId()) + "_" +
                     std::to_string(detail::NextSerial()) + ".gguf");
        const auto encoded = Encode();
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        if (!stream) throw std::runtime_error("gemma4 fixture: cannot create " + path.string());
        stream.write(reinterpret_cast<const char*>(encoded.data()),
                     static_cast<std::streamsize>(encoded.size()));
        if (!stream) throw std::runtime_error("gemma4 fixture: write failed for " + path.string());
        detail::TempFileRegistry::Instance().Track(path, pinned);
        return path;
    }

private:
    std::vector<std::byte> Encode() const {
        std::vector<std::byte> out;
        Append(out, std::uint32_t{0x46554747});   // magic 'GGUF'
        Append(out, std::uint32_t{3});            // version 3
        Append(out, static_cast<std::uint64_t>(tensors_.size()));
        Append(out, static_cast<std::uint64_t>(metadata_.size()));

        for (const auto& [key, value] : metadata_) {
            AppendString(out, key);
            Append(out, static_cast<std::uint32_t>(value.index()));
            std::visit([&](const auto& item) {
                using T = std::decay_t<decltype(item)>;
                if constexpr (std::is_same_v<T, std::string>) {
                    AppendString(out, item);
                } else if constexpr (std::is_same_v<T, ArrayValue>) {
                    Append(out, item.element_type);
                    Append(out, item.count);
                    out.insert(out.end(), item.encoded_elements.begin(),
                              item.encoded_elements.end());
                } else if constexpr (std::is_same_v<T, bool>) {
                    Append(out, static_cast<std::uint8_t>(item));
                } else {
                    Append(out, item);
                }
            }, value);
        }

        // Every tensor's relative offset must land on an alignment boundary
        // (flm::rai::GgufFile::Open rejects one that does not), so the
        // running offset is rounded up after each tensor -- meaning the
        // offset recorded for the *next* tensor is always already aligned.
        constexpr std::uint64_t kAlignment = 32;
        std::vector<std::uint64_t> offsets(tensors_.size());
        std::uint64_t running = 0;
        for (std::size_t i = 0; i < tensors_.size(); ++i) {
            offsets[i] = running;
            running += TensorBytes(tensors_[i]);
            running = (running + kAlignment - 1) & ~(kAlignment - 1);
        }
        for (std::size_t i = 0; i < tensors_.size(); ++i) {
            const auto& tensor = tensors_[i];
            AppendString(out, tensor.name);
            Append(out, static_cast<std::uint32_t>(tensor.shape.size()));
            // GGUF stores dimensions fastest-varying first; flm::rai::GgufFile
            // reverses them back into outer-to-inner order on read, so
            // writing them reversed here is what makes AddTensor's shape
            // argument round-trip unchanged through Open()/ShapeOf().
            for (auto it = tensor.shape.rbegin(); it != tensor.shape.rend(); ++it)
                Append(out, *it);
            Append(out, tensor.type);
            Append(out, offsets[i]);
        }

        const auto header_size = out.size();
        const auto data_start = (header_size + kAlignment - 1) & ~(kAlignment - 1);
        out.resize(data_start, std::byte{0});
        if (!tensors_.empty()) {
            const auto total_payload = offsets.back() + TensorBytes(tensors_.back());
            out.resize(data_start + total_payload, std::byte{0});
        }
        // BF16 payloads stay zero -- nothing in this suite decodes them, and a
        // synthetic stream would be a second thing to keep honest. F32
        // payloads get a per-tensor constant (see SignatureAt for why a zero
        // norm makes the assertions that matter most unwritable), except for
        // 2-D ones, which get a POSITIONAL fill -- see PositionalValueAt.
        for (std::size_t i = 0; i < tensors_.size(); ++i) {
            if (tensors_[i].type != kF32) continue;
            const bool positional = tensors_[i].shape.size() == 2;
            const auto value = SignatureAt(i);
            const auto elements = TensorElementCount(tensors_[i].shape);
            auto* payload = reinterpret_cast<float*>(out.data() + data_start + offsets[i]);
            for (std::uint64_t element = 0; element < elements; ++element)
                payload[element] = positional ? PositionalValueAt(i, element) : value;
        }
        // Q8_0 payloads stay zero apart from a four-byte tag at their head --
        // see BlockSignatureAt for why they cannot all be identical and why
        // this is not (and need not be) a valid block stream. BF16 payloads
        // stay zero entirely: `per_layer_model_proj.weight` is the only one,
        // it is a HOST matmul, and no corelib port receives it.
        for (std::size_t i = 0; i < tensors_.size(); ++i) {
            if (tensors_[i].type != kQ8_0) continue;
            const auto tag = BlockSignatureAt(i);
            std::memcpy(out.data() + data_start + offsets[i], &tag, sizeof(tag));
        }
        return out;
    }

    std::vector<std::pair<std::string, MetadataValue>> metadata_;
    std::vector<Tensor> tensors_;
};

/// \brief the knobs that make an E2B or an E4B fixture, and nothing more
struct FixtureOptions {
    std::int64_t layers;
    std::int64_t hidden;
    std::int64_t kv_heads;
    std::int64_t full_attention_period;   // 5 on E2B, 6 on E4B

    /// When non-empty, gemma4.feed_forward_length is written as an ARRAY of
    /// exactly this many entries -- one width per layer, in layer order --
    /// and that same per-layer width is what the layer's ffn_gate/ffn_up/
    /// ffn_down tensors are shaped with. When empty, the metadata key is
    /// written as a SCALAR of `uniform_ffn`, and every layer's FFN tensors
    /// use `uniform_ffn` too. E2B is the array case, E4B the scalar case --
    /// the same metadata *key* moves type between conversions of one model
    /// family, and a reader must survive both.
    ///
    /// A later task that wants to prove its reader ignores this key entirely
    /// (rather than merely happening to agree with the tensor shapes) is not
    /// limited to these two fields: `MakeBuilder(options)` returns the
    /// Builder *before* it is written, so that task can
    /// `.SetMetadata("gemma4.feed_forward_length", something_wrong)` --
    /// changing only the metadata a reader might be tempted to trust, while
    /// every tensor this struct produced keeps the shape it was given here.
    std::vector<std::uint64_t> ffn_widths;
    std::uint64_t uniform_ffn;

    /// gemma4.attention.shared_kv_layers, written with the FILE's own
    /// semantics: how many layers SHARE a cache, not how many OWN one --
    /// 20 on E2B, 18 on E4B. NOT the 15/24 that count owners (owning =
    /// block_count - shared_kv_layers). See verified-gguf-facts.md's
    /// "correction that matters most": a fixture that wrote 15/24 here
    /// would be a fixture no vendor ships, and would validate a later
    /// task's reader against a fiction.
    std::int64_t shared_kv_layers;
};

inline constexpr std::int64_t kQueryHeadCount = 8;
inline constexpr std::int64_t kSlidingHeadDim = 256;
inline constexpr std::int64_t kFullHeadDim = 512;
inline constexpr std::int64_t kPleDim = 256;
inline constexpr std::int64_t kVocabularySize = 262144;

/// E2B: 35 layers, hidden 1536, 1 KV head, full-attention period 5,
/// feed_forward_length as a 35-element ARRAY (6144 x15 then 12288 x20),
/// gemma4.attention.shared_kv_layers = 20 (layers that SHARE, per the file's
/// own semantics -- 15 layers actually own a cache: 35 - 20 = 15).
inline FixtureOptions E2bOptions() {
    FixtureOptions options{};
    options.layers = 35;
    options.hidden = 1536;
    options.kv_heads = 1;
    options.full_attention_period = 5;
    options.ffn_widths.assign(15, 6144);
    options.ffn_widths.resize(35, 12288);
    options.uniform_ffn = 0;   // unused: ffn_widths is non-empty
    options.shared_kv_layers = 20;
    return options;
}

/// E4B: 42 layers, hidden 2560, 2 KV heads, full-attention period 6,
/// feed_forward_length as a SCALAR 10240, gemma4.attention.shared_kv_layers
/// = 18 (24 layers actually own a cache: 42 - 18 = 24).
inline FixtureOptions E4bOptions() {
    FixtureOptions options{};
    options.layers = 42;
    options.hidden = 2560;
    options.kv_heads = 2;
    options.full_attention_period = 6;
    // ffn_widths left empty: E4B is the scalar case.
    options.uniform_ffn = 10240;
    options.shared_kv_layers = 18;
    return options;
}

/// \brief which layers are full-attention, off the same period both real
///        files were measured to follow: last-in-period, not first.
/// \note E2B period 5 gives {4,9,14,19,24,29,34}; E4B period 6 gives
///       {5,11,17,23,29,35,41}. Confirmed against the real files in
///       verified-gguf-facts.md.
inline bool IsFullAttentionLayer(std::int64_t layer, std::int64_t full_attention_period) {
    return (layer % full_attention_period) == (full_attention_period - 1);
}

/// \brief assemble a Gemma 4 fixture's metadata and tensors, without writing
///        it to disk yet.
/// \note Exposed (not just `Write`) so a later task can mutate metadata --
///       via `SetMetadata`/`RemoveMetadata` -- independently of the tensor
///       shapes this function already committed to disk-ready form. See
///       `FixtureOptions::ffn_widths`'s doc comment for why that
///       independence matters.
inline Builder MakeBuilder(const FixtureOptions& options) {
    if (!options.ffn_widths.empty() &&
        static_cast<std::int64_t>(options.ffn_widths.size()) != options.layers) {
        throw std::runtime_error(
            "gemma4 fixture: ffn_widths has " + std::to_string(options.ffn_widths.size()) +
            " entries, expected one per layer (" + std::to_string(options.layers) + ")");
    }

    Builder builder;
    builder.SetMetadata("general.architecture", std::string("gemma4"));
    builder.SetMetadata("general.alignment", std::uint32_t{32});
    builder.SetMetadata("gemma4.block_count", static_cast<std::uint32_t>(options.layers));
    builder.SetMetadata("gemma4.embedding_length", static_cast<std::uint32_t>(options.hidden));
    builder.SetMetadata("gemma4.attention.head_count", static_cast<std::uint32_t>(kQueryHeadCount));
    builder.SetMetadata("gemma4.attention.head_count_kv", static_cast<std::uint32_t>(options.kv_heads));
    builder.SetMetadata("gemma4.attention.key_length", static_cast<std::uint32_t>(kFullHeadDim));
    builder.SetMetadata("gemma4.attention.key_length_swa", static_cast<std::uint32_t>(kSlidingHeadDim));
    builder.SetMetadata("gemma4.attention.shared_kv_layers",
                       static_cast<std::uint32_t>(options.shared_kv_layers));
    builder.SetMetadata("gemma4.attention.sliding_window", std::uint32_t{512});
    // Both shipped files state 131072.
    builder.SetMetadata("gemma4.context_length", std::uint32_t{131072});
    // 9.999999974752427e-07f is the exact float32 bit pattern the real files
    // carry for 1e-6 -- see verified-gguf-facts.md's "one value the plan
    // never stated". Not Phi-4's 1e-5 (phi4_rai_constants.hpp's kRmsEpsilon).
    builder.SetMetadata("gemma4.attention.layer_norm_rms_epsilon", 9.999999974752427e-07f);
    builder.SetMetadata("gemma4.rope.freq_base", 1000000.0f);       // full-attention layers
    builder.SetMetadata("gemma4.rope.freq_base_swa", 10000.0f);     // sliding layers
    builder.SetMetadata("gemma4.final_logit_softcapping", 30.0f);
    if (options.ffn_widths.empty()) {
        builder.SetMetadata("gemma4.feed_forward_length",
                           static_cast<std::uint32_t>(options.uniform_ffn));
    } else {
        std::vector<std::byte> encoded;
        for (const auto width : options.ffn_widths)
            Append(encoded, static_cast<std::uint32_t>(width));
        builder.SetMetadata("gemma4.feed_forward_length",
                           ArrayValue{kMetaUint32,
                                      static_cast<std::uint64_t>(options.ffn_widths.size()),
                                      std::move(encoded)});
    }
    builder.SetMetadata("tokenizer.ggml.eos_token_id", std::uint32_t{106});
    builder.SetMetadata("tokenizer.ggml.bos_token_id", std::uint32_t{2});
    builder.SetMetadata("tokenizer.ggml.add_bos_token", true);

    const auto hidden = static_cast<std::uint64_t>(options.hidden);
    const auto kv_heads = static_cast<std::uint64_t>(options.kv_heads);
    const auto ple_dim = static_cast<std::uint64_t>(kPleDim);
    const auto vocab = static_cast<std::uint64_t>(kVocabularySize);
    const auto layers = static_cast<std::uint64_t>(options.layers);

    // 6 file-level tensors.
    //
    // ORIENTATION. `AddTensor` takes the shape in `GgufFile::ShapeOf`'s
    // convention and writes it REVERSED into the file, because `ShapeOf`
    // reverses the file's dims back on read. A real Gemma 4 GGUF stores
    // dims fastest-varying (input) first, so on a real file `ShapeOf` is
    // [OUT, IN]: `token_embd.weight` is `[1536, 262144]` in the file and
    // `[262144, 1536]` here.
    //
    // THESE WERE ALL TRANSPOSED until Task R1. Task C1 transcribed
    // verified-gguf-facts.md's tables -- which quote FILE order and did not
    // say so -- straight into this argument, and Tasks C2-C5's readers were
    // then written to match the fixture. Fixture and readers confirmed each
    // other for four signed-off tasks; a real file breaks both. Phi-4,
    // which HAS run against a real GGUF on hardware, settles the convention
    // independently: phi4_rai.cpp requires `token_embd.weight` at
    // `{kVocabularySize, kHiddenSize}`.
    //
    // test_gemma4_real_gguf.cpp asserts every shape below against the real
    // files. If these two ever disagree again, that is what says so.
    builder.AddTensor("token_embd.weight", {vocab, hidden}, kQ8_0);
    builder.AddTensor("per_layer_token_embd.weight", {vocab, layers * ple_dim}, kQ8_0);
    // BF16 (ggml type 30), NOT Q8_0, AND THAT IS A MEASUREMENT. The one BF16
    // tensor in either real file -- the reference driver's `floats()` names
    // it as such, and a type histogram of both real GGUFs (2026-09-22) reads
    // 353 F32 / 247 Q8_0 / 1 BF16 on E2B and 423 / 296 / 1 on E4B. It was
    // kQ8_0 here, a guess carried from "the big projections are quantized",
    // and it is not cosmetic: `flm::rai::GgufFile` could not size a type-30
    // tensor at all, so `Open` threw on every real Gemma 4 GGUF and NO TEST
    // IN THIS PROJECT HAD EVER LOADED ONE. That blind spot is what let this
    // fixture's transposed dims survive four signed-off tasks.
    builder.AddTensor("per_layer_model_proj.weight", {layers * ple_dim, hidden}, kBf16);
    builder.AddTensor("per_layer_proj_norm.weight", {ple_dim}, kF32);
    builder.AddTensor("output_norm.weight", {hidden}, kF32);
    builder.AddTensor("rope_freqs.weight", {static_cast<std::uint64_t>(kFullHeadDim / 2)}, kF32);
    // Deliberately no "output.weight": lm_head is tied to token_embd on both
    // real models (verified-gguf-facts.md). A fixture that added one would
    // let a reader that wrongly requires it pass against a file no vendor
    // ships.

    for (std::int64_t layer = 0; layer < options.layers; ++layer) {
        const auto prefix = "blk." + std::to_string(layer);
        const bool is_full = IsFullAttentionLayer(layer, options.full_attention_period);
        const auto head_dim =
            static_cast<std::uint64_t>(is_full ? kFullHeadDim : kSlidingHeadDim);
        const auto ffn_width = options.ffn_widths.empty()
                                    ? options.uniform_ffn
                                    : options.ffn_widths[static_cast<std::size_t>(layer)];
        const auto q_width = static_cast<std::uint64_t>(kQueryHeadCount) * head_dim;
        const auto kv_width = kv_heads * head_dim;

        // 7 projection weights, in ShapeOf's [OUT, IN] convention (see the
        // file-level block above). attn_q's OUTPUT width steps with the
        // layer's own head_dim -- 2048 sliding / 4096 full, on both models,
        // whatever their hidden size -- and that, with ffn_gate's output
        // width, is what a real reader derives its geometry from.
        // attn_output and ffn_down are the two whose INPUT is not `hidden`.
        builder.AddTensor(prefix + ".attn_q.weight", {q_width, hidden}, kQ8_0);
        builder.AddTensor(prefix + ".attn_k.weight", {kv_width, hidden}, kQ8_0);
        builder.AddTensor(prefix + ".attn_v.weight", {kv_width, hidden}, kQ8_0);
        builder.AddTensor(prefix + ".attn_output.weight", {hidden, q_width}, kQ8_0);
        builder.AddTensor(prefix + ".ffn_gate.weight", {ffn_width, hidden}, kQ8_0);
        builder.AddTensor(prefix + ".ffn_up.weight", {ffn_width, hidden}, kQ8_0);
        builder.AddTensor(prefix + ".ffn_down.weight", {hidden, ffn_width}, kQ8_0);

        // 5 generic norms -- names and shapes (all {hidden}) confirmed
        // against a real layer-0 tensor dump of gemma-4-E2B-it-Q8_0.gguf.
        // The fifth is `post_norm`, NOT a per-layer-dim `per_layer_norm` --
        // that guess was wrong and is corrected here; see task-C1-report.md.
        builder.AddTensor(prefix + ".attn_norm.weight", {hidden}, kF32);
        builder.AddTensor(prefix + ".post_attention_norm.weight", {hidden}, kF32);
        builder.AddTensor(prefix + ".ffn_norm.weight", {hidden}, kF32);
        builder.AddTensor(prefix + ".post_ffw_norm.weight", {hidden}, kF32);
        builder.AddTensor(prefix + ".post_norm.weight", {hidden}, kF32);

        // 2 QK norms -- sized to THIS layer's own head_dim, so they differ
        // between a sliding and a full-attention layer just as attn_q does.
        builder.AddTensor(prefix + ".attn_q_norm.weight", {head_dim}, kF32);
        builder.AddTensor(prefix + ".attn_k_norm.weight", {head_dim}, kF32);

        // 3 PLE tensors -- inp_gate/proj confirmed verbatim against the real
        // file. layer_output_scale is a single scalar (shape [1]), not one
        // value per hidden channel -- also corrected from the earlier guess.
        //
        // F32, NOT Q8_0, AND THAT IS A MEASUREMENT. Re-dumped from
        // gemma-4-E2B-it-Q8_0.gguf and gemma-4-E4B-it-Q8_0.gguf on
        // 2026-09-22: every blk.N.inp_gate.weight and blk.N.proj.weight is
        // ggml type 0 (F32) on both rows. They were kQ8_0 here, which was a
        // guess carried from "the projections are quantized" and is wrong.
        // It is not cosmetic: ryzenai_corelib_ple_bf16_weights_pack takes
        // `const float*` for both matrices and requantizes them itself
        // (corelib.h's ple section; gemma4_driver.py:1481-1512 passes
        // `weights.floats(...)`), so there is no Q8_0 route for these two at
        // all. A fixture that quantized them would make the engine's
        // RequireF32 fail against the fixture and pass against nothing.
        //
        // In ShapeOf's [OUT, IN] convention these are {ple_dim, hidden} and
        // {hidden, ple_dim} -- the file stores them as [1536, 256] and
        // [256, 1536]. The two have the SAME element count, so nothing
        // anywhere catches them being swapped by size; only these exact
        // shapes do.
        builder.AddTensor(prefix + ".inp_gate.weight", {ple_dim, hidden}, kF32);
        builder.AddTensor(prefix + ".proj.weight", {hidden, ple_dim}, kF32);
        builder.AddTensor(prefix + ".layer_output_scale.weight", {std::uint64_t{1}}, kF32);
    }

    return builder;
}

/// \brief write a synthetic Gemma 4 GGUF and return its path
/// \note Tensor data is left zero-filled; these tests read shapes and
///       metadata, not values.
inline std::filesystem::path Write(const FixtureOptions& options) {
    return MakeBuilder(options).Write();
}

}  // namespace gemma4_fixture
