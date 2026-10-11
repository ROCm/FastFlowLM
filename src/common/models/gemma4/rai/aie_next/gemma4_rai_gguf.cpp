#include "models/gemma4/rai/aie_next/gemma4_rai_gguf.hpp"

#include "models/gemma4/rai/aie_next/gemma4_rai_constants.hpp"
#include "rai/gguf_file.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <utility>

namespace flm::gemma4 {
namespace {

// TWO ROWS, AND THE SECOND IS THE POINT OF THE FIRST. Everything that
// differs between E2B and E4B -- 42 layers against 35, hidden 2560 against
// 1536, two KV heads against one, a uniform FFN width against one that
// steps, a full-attention period of 6 against 5, a different pair of shared
// caches -- is READ FROM THE FILE by later tasks, not stated here. A row
// carries only what no GGUF key states: the vocabulary size, and the three
// group sizes DynamicDispatch's ELF set ships at PDI pair (8, 17). Those
// three were read off the ELF set itself -- not guessed from either model's
// shape -- which is exactly why an unrecognized shape below is rejected
// rather than assigned a plausible-looking row of its own.
constexpr std::array<Gemma4Row, 2> kRows{{
    {"E2B", 35, 1536, 8, 1, 256, 512, kVocabularySize, kGroupSize, kGroupSize, kGroupSize},
    {"E4B", 42, 2560, 8, 2, 256, 512, kVocabularySize, kGroupSize, kGroupSize, kGroupSize},
}};

}  // namespace

std::span<const Gemma4Row> Gemma4Rows() { return kRows; }

const Gemma4Row& RequireRowFor(std::int64_t layers, std::int64_t hidden,
                               std::int64_t q_heads, std::int64_t kv_heads,
                               std::int64_t head_dim,
                               std::int64_t global_head_dim) {
    for (const auto& row : kRows) {
        if (row.layers == layers && row.hidden == hidden &&
            row.q_heads == q_heads && row.kv_heads == kv_heads &&
            row.head_dim == head_dim && row.global_head_dim == global_head_dim) {
            return row;
        }
    }
    // Name every shape field the caller asked about, so whoever hits this
    // can tell at a glance whether it is a typo or a genuinely new size --
    // and, if the latter, exactly what to go measure before adding a row.
    throw std::runtime_error(
        "no Gemma 4 row matches layers=" + std::to_string(layers) +
        " hidden=" + std::to_string(hidden) +
        " q_heads=" + std::to_string(q_heads) +
        " kv_heads=" + std::to_string(kv_heads) +
        " head_dim=" + std::to_string(head_dim) +
        " global_head_dim=" + std::to_string(global_head_dim) +
        "; add a row once its group sizes have been read off the ELF set at "
        "PDI pair (8, 17). Guessing them would replace this message with a "
        "dispatch failure far from here.");
}

std::vector<std::int64_t> DeriveFfnWidths(const flm::rai::GgufFile& file,
                                          std::int64_t layers) {
    std::vector<std::int64_t> widths;
    widths.reserve(static_cast<std::size_t>(layers));
    for (std::int64_t layer = 0; layer < layers; ++layer) {
        const auto name = "blk." + std::to_string(layer) + ".ffn_gate.weight";
        const auto shape = file.Tensor(name).shape;
        if (shape.size() != 2) {
            throw std::runtime_error(
                name + " is not 2-D; Gemma 4's MLP width is read from this "
                "tensor's own shape, never from gemma4.feed_forward_length");
        }
        // shape[0], NOT shape[1]. `GgufFile::ShapeOf` REVERSES the GGUF's
        // own dims, and the file stores them fastest-varying (input) first,
        // so a real file's `ShapeOf` is [OUT, IN]: `blk.0.ffn_gate.weight`
        // reads [1536, 6144] in the file and [6144, 1536] here. Reading
        // shape[1] returns `hidden` on every layer of both models -- a
        // plausible, uniform, wrong width, which is precisely what the
        // transposed fixture made look right.
        widths.push_back(shape[0]);
    }
    return widths;
}

std::vector<bool> DeriveLayerIsSwa(const flm::rai::GgufFile& file,
                                   std::int64_t layers, std::int64_t q_heads,
                                   std::int64_t head_dim,
                                   std::int64_t global_head_dim) {
    // NOT a period. E2B's full-attention layers recur every 5th layer,
    // E4B's every 6th -- a pattern generated from one would fit the other's
    // weights nowhere. The Q projection's own width says which geometry a
    // layer is, and that width is the same on both models even though their
    // hidden sizes differ (see the header doc comment).
    const std::int64_t sliding_width = q_heads * head_dim;
    const std::int64_t full_width = q_heads * global_head_dim;
    std::vector<bool> is_swa;
    is_swa.reserve(static_cast<std::size_t>(layers));
    for (std::int64_t layer = 0; layer < layers; ++layer) {
        const auto name = "blk." + std::to_string(layer) + ".attn_q.weight";
        const auto shape = file.Tensor(name).shape;
        if (shape.size() != 2) {
            throw std::runtime_error(
                name + " is not 2-D; Gemma 4's attention geometry (sliding "
                "vs full-attention) is read from this tensor's own width, "
                "never from a derived period");
        }
        // shape[0] is the OUTPUT width -- see DeriveFfnWidths above for why.
        // This one at least fails loudly when read the other way round
        // (`hidden` matches neither geometry), which is how the transposition
        // was finally caught; DeriveFfnWidths just returns a wrong number.
        if (shape[0] == sliding_width) {
            is_swa.push_back(true);
        } else if (shape[0] == full_width) {
            is_swa.push_back(false);
        } else {
            throw std::runtime_error(
                name + " has width " + std::to_string(shape[0]) +
                ", which is neither the sliding geometry (" +
                std::to_string(sliding_width) + ") nor the full-attention one (" +
                std::to_string(full_width) + ")");
        }
    }
    return is_swa;
}

std::vector<std::int64_t> DeriveCacheOwners(const std::vector<bool>& layer_is_swa,
                                            std::int64_t kv_layers) {
    const auto layers = static_cast<std::int64_t>(layer_is_swa.size());
    if (kv_layers <= 0 || kv_layers > layers) {
        throw std::runtime_error(
            "kv_layers " + std::to_string(kv_layers) + " is outside 1.." +
            std::to_string(layers) + " -- this is the OWNING count "
            "(block_count - shared_kv_layers), not the raw "
            "gemma4.attention.shared_kv_layers metadata value");
    }
    // The last own-cache layer of each kind, tracked while scanning only the
    // owning prefix -- never a period, for the same reason DeriveLayerIsSwa
    // does not use one: the two shipped models disagree on it.
    std::int64_t last_sliding = -1;
    std::int64_t last_full = -1;
    std::vector<std::int64_t> owners(static_cast<std::size_t>(layers), -1);
    for (std::int64_t layer = 0; layer < kv_layers; ++layer) {
        owners[static_cast<std::size_t>(layer)] = layer;
        (layer_is_swa[static_cast<std::size_t>(layer)] ? last_sliding : last_full) = layer;
    }
    for (std::int64_t layer = kv_layers; layer < layers; ++layer) {
        const bool swa = layer_is_swa[static_cast<std::size_t>(layer)];
        const std::int64_t owner = swa ? last_sliding : last_full;
        if (owner < 0) {
            throw std::runtime_error(
                "layer " + std::to_string(layer) + " shares a " +
                (swa ? "sliding" : "full-attention") +
                " cache, but no layer among the first " +
                std::to_string(kv_layers) + " owns one of that kind");
        }
        owners[static_cast<std::size_t>(layer)] = owner;
    }
    return owners;
}

std::int64_t Gemma4Config::LayerHeadDim(std::int64_t layer) const {
    return layer_is_swa.at(static_cast<std::size_t>(layer)) ? head_dim : global_head_dim;
}

namespace {

[[noreturn]] void Fail(std::string_view field, std::string actual, std::string expected) {
    throw std::runtime_error(std::string(field) + ": actual " + actual +
                             ", expected " + expected);
}

std::string JsonText(const nlohmann::json& value) { return value.dump(); }

/// \note Not "== \"gemma4\"" on the nose: `config.json`'s own architecture
///       field is an `architectures` array of HF class names (e.g.
///       "Gemma4ForConditionalGeneration"), not the GGUF's short
///       `general.architecture` string. A `model_type` field, if present,
///       is checked directly instead.
bool ArchitectureNamesGemma4(const nlohmann::json& config) {
    const auto names_gemma4 = [](const std::string& text) {
        return text.find("Gemma4") != std::string::npos;
    };
    if (const auto it = config.find("architectures"); it != config.end() && it->is_array()) {
        for (const auto& entry : *it) {
            if (entry.is_string() && names_gemma4(entry.get<std::string>())) return true;
        }
    }
    if (const auto it = config.find("model_type"); it != config.end() && it->is_string()) {
        const auto model_type = it->get<std::string>();
        return model_type == "gemma4" || names_gemma4(model_type);
    }
    return false;
}

}  // namespace

std::vector<std::int64_t> ConfigEosIds(const nlohmann::json& config, std::int32_t gguf_eos) {
    std::vector<std::int64_t> ids{gguf_eos};
    const auto it = config.find("eos_token_id");
    if (it != config.end() && it->is_array()) {
        for (const auto& id : *it) {
            if (id.is_number_integer() &&
                std::find(ids.begin(), ids.end(), id.get<std::int64_t>()) == ids.end())
                ids.push_back(id.get<std::int64_t>());
        }
    }
    return ids;
}

namespace {

void RequireJsonUnsigned(const nlohmann::json& object, std::string_view key,
                         std::uint64_t expected) {
    const auto it = object.find(std::string(key));
    if (it == object.end()) Fail(key, "missing", std::to_string(expected));
    std::uint64_t actual;
    if (it->is_number_unsigned()) {
        actual = it->get<std::uint64_t>();
    } else if (it->is_number_integer()) {
        const auto signed_value = it->get<std::int64_t>();
        if (signed_value < 0)
            Fail(key, JsonText(*it), "non-negative integer " + std::to_string(expected));
        actual = static_cast<std::uint64_t>(signed_value);
    } else {
        Fail(key, JsonText(*it), "integer " + std::to_string(expected));
    }
    if (actual != expected) Fail(key, std::to_string(actual), std::to_string(expected));
}

/// \brief where the fields describing the TEXT model live
///
/// The real `config.json` is a MULTIMODAL config: `architectures`,
/// `model_type` and `eos_token_id` sit at the top level, but every field
/// that describes the text model -- `num_hidden_layers`, `vocab_size`,
/// `intermediate_size`, ... -- is nested one level down, under
/// `"text_config"` (confirmed against the real file; see
/// verified-gguf-facts.md's "config.json is MULTIMODAL" section). An
/// earlier version of this check read `num_hidden_layers` off the top
/// level directly and rejected every real Gemma 4 package as a result --
/// exactly the failure mode this contract exists to prevent, inverted.
///
/// Falls back to the top level itself when `text_config` is absent or not
/// an object, in case a hypothetical flat config ever ships. Every real
/// file measured so far has `text_config`, so that fallback is untested
/// insurance, not the expected path -- it exists only so a flat config does
/// not regress into "missing text_config" instead of "missing
/// num_hidden_layers".
const nlohmann::json& TextModelConfig(const nlohmann::json& config) {
    if (const auto it = config.find("text_config"); it != config.end() && it->is_object())
        return *it;
    return config;
}

}  // namespace

Gemma4Config ConfigFromMetadata(const flm::rai::GgufFile& file) {
    const auto architecture = file.String("general.architecture");
    if (architecture != "gemma4") {
        throw std::runtime_error("general.architecture: actual '" + architecture +
                                 "', expected 'gemma4'");
    }

    const auto layers = static_cast<std::int64_t>(file.Unsigned("gemma4.block_count"));
    const auto hidden = static_cast<std::int64_t>(file.Unsigned("gemma4.embedding_length"));
    const auto q_heads =
        static_cast<std::int64_t>(file.Unsigned("gemma4.attention.head_count"));
    const auto kv_heads =
        static_cast<std::int64_t>(file.Unsigned("gemma4.attention.head_count_kv"));

    // The GGUF's names run OPPOSITE to ours: key_length is the FULL-attention
    // layers' head size and key_length_swa the SLIDING one. Worth knowing at
    // this one place and nowhere else -- reading them the obvious way round
    // swaps 256 and 512 through the whole model, silently.
    const auto global_head_dim =
        static_cast<std::int64_t>(file.Unsigned("gemma4.attention.key_length"));
    const auto head_dim =
        static_cast<std::int64_t>(file.Unsigned("gemma4.attention.key_length_swa"));

    // gemma4.attention.shared_kv_layers counts the layers that SHARE a
    // cache, NOT the layers that OWN one -- the complement of what
    // DeriveCacheOwners's kv_layers parameter means. This subtraction is the
    // one correction that matters most in this whole file: taking the raw
    // key would give 20 on E2B and 18 on E4B where 15 and 24 are correct,
    // and every shared layer would silently bind the wrong KV cache. See
    // DeriveCacheOwners's doc comment and verified-gguf-facts.md.
    const auto shared_kv_layers =
        static_cast<std::int64_t>(file.Unsigned("gemma4.attention.shared_kv_layers"));
    const auto kv_layers = layers - shared_kv_layers;

    const auto sliding_window =
        static_cast<std::int64_t>(file.Unsigned("gemma4.attention.sliding_window"));
    // A file that does not state its context gets one page, as corelib's
    // driver does.
    const auto context_length = file.HasMetadata("gemma4.context_length")
        ? static_cast<std::int64_t>(file.Unsigned("gemma4.context_length"))
        : kMaxSequenceLength;
    const auto eps =
        static_cast<float>(file.Number("gemma4.attention.layer_norm_rms_epsilon"));
    const auto rope_theta = file.Number("gemma4.rope.freq_base");
    const auto rope_theta_swa = file.Number("gemma4.rope.freq_base_swa");
    const auto logit_softcap = static_cast<float>(file.Number("gemma4.final_logit_softcapping"));
    const auto eos_token_id =
        static_cast<std::int32_t>(file.Unsigned("tokenizer.ggml.eos_token_id"));
    const auto bos_token_id =
        static_cast<std::int32_t>(file.Unsigned("tokenizer.ggml.bos_token_id"));
    const auto add_bos = file.Boolean("tokenizer.ggml.add_bos_token");

    auto layer_intermediate = DeriveFfnWidths(file, layers);
    auto layer_is_swa = DeriveLayerIsSwa(file, layers, q_heads, head_dim, global_head_dim);
    auto layer_cache_owner = DeriveCacheOwners(layer_is_swa, kv_layers);

    const auto& row = RequireRowFor(layers, hidden, q_heads, kv_heads, head_dim, global_head_dim);

    // Named both numbers: which one is wrong (a missing/extra tensor vs. a
    // wrong layer count) is not obvious from either alone.
    const auto expected_tensor_count = static_cast<std::size_t>(6 + layers * 17);
    if (file.TensorCount() != expected_tensor_count) {
        throw std::runtime_error(
            "tensor count: actual " + std::to_string(file.TensorCount()) +
            ", expected " + std::to_string(expected_tensor_count) + " (6 + " +
            std::to_string(layers) + " layers * 17)");
    }

    // per_layer_token_embd's FILE dims are (layers * ple_dim, vocab) -- the
    // layer axis folded into the fastest-varying extent, not a separate
    // dimension. `ShapeOf` reverses those, so the layer axis is shape[1]
    // here and shape[0] is the vocabulary. ple_dim is 256 on both shipped
    // models: 8960 / 35 on E2B and 10752 / 42 on E4B.
    //
    // Reading shape[0] instead would divide 262144 by the layer count --
    // 7489.8 on E2B, which is not an integer, so THIS one would at least
    // throw; on a hypothetical model whose vocabulary happened to divide,
    // it would silently produce a ple_dim off by a factor of a thousand.
    const auto ple_shape = file.Tensor("per_layer_token_embd.weight").shape;
    if (ple_shape.size() != 2 || layers <= 0 || ple_shape[1] % layers != 0) {
        throw std::runtime_error(
            "per_layer_token_embd.weight's layer axis (" +
            (ple_shape.size() != 2 ? std::string("<not 2-D>") : std::to_string(ple_shape[1])) +
            ") is not a multiple of the layer count (" + std::to_string(layers) +
            "); ple_dim is derived as ShapeOf[1] / layers, ShapeOf being the "
            "reverse of the file's own dims");
    }
    const auto ple_dim = ple_shape[1] / layers;

    Gemma4Config config{};
    config.row_name = row.name;
    config.layers = layers;
    config.hidden = hidden;
    config.q_heads = q_heads;
    config.kv_heads = kv_heads;
    config.head_dim = head_dim;
    config.global_head_dim = global_head_dim;
    config.layer_intermediate = std::move(layer_intermediate);
    config.layer_is_swa = std::move(layer_is_swa);
    config.layer_cache_owner = std::move(layer_cache_owner);
    config.kv_layers = kv_layers;
    config.sliding_window = sliding_window;
    config.vocab = row.vocab;
    config.ple_dim = ple_dim;
    config.rope_theta = rope_theta;
    config.rope_theta_swa = rope_theta_swa;
    config.eps = eps;
    config.logit_softcap = logit_softcap;
    config.eos_token_id = eos_token_id;
    config.bos_token_id = bos_token_id;
    config.add_bos = add_bos;
    config.group = row.group;
    config.head_group = row.head_group;
    config.ple_group = row.ple_group;
    config.context_length = context_length;
    return config;
}

struct Gemma4GgufPackage::Impl {
    std::shared_ptr<flm::rai::GgufFile> file;
    Gemma4Config config;
};

Gemma4GgufPackage::Gemma4GgufPackage(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Gemma4GgufPackage::~Gemma4GgufPackage() = default;

std::shared_ptr<Gemma4GgufPackage> Gemma4GgufPackage::Open(
    const std::filesystem::path& gguf_path) {
    auto impl = std::make_unique<Impl>();
    impl->file = flm::rai::GgufFile::Open(gguf_path);
    impl->config = ConfigFromMetadata(*impl->file);
    return std::shared_ptr<Gemma4GgufPackage>(new Gemma4GgufPackage(std::move(impl)));
}

const Gemma4Config& Gemma4GgufPackage::Config() const noexcept { return impl_->config; }
const flm::rai::GgufFile& Gemma4GgufPackage::File() const noexcept { return *impl_->file; }
const std::filesystem::path& Gemma4GgufPackage::Path() const noexcept {
    return impl_->file->Path();
}

void Gemma4GgufPackage::ValidateGemma4Contract(const nlohmann::json& config,
                                               const nlohmann::json& tokenizer,
                                               const nlohmann::json& tokenizer_config) const {
    const auto& cfg = impl_->config;

    // architecture and model_type describe the whole (multimodal) package
    // and live at the top level even in the real, nested config.json.
    if (!ArchitectureNamesGemma4(config)) {
        Fail("config.json architecture",
             "neither 'architectures' nor 'model_type' names Gemma 4",
             "an entry containing \"Gemma4\" (or model_type == \"gemma4\")");
    }

    // Everything below describes the TEXT model and is read from
    // text_config, falling back to the top level -- see TextModelConfig's
    // doc comment for why, and verified-gguf-facts.md for the real file
    // structure that made this necessary.
    const auto& text_config = TextModelConfig(config);

    // num_hidden_layers is the one field below that is FATAL if missing,
    // and deliberately so: it is the ONLY cross-check field that can catch
    // an E2B/E4B swap at all. vocab_size (262144) and eos_token_id (106)
    // are IDENTICAL between the two shipped rows -- a config.json that
    // named the wrong row but was copy-pasted from within the same family
    // would still agree with the GGUF on both of those. Layer count (35 vs
    // 42) is where the two models actually disagree, so this is the field
    // this check cannot afford to make optional. A field that is silently
    // optional is a check that does not exist.
    RequireJsonUnsigned(text_config, "num_hidden_layers",
                        static_cast<std::uint64_t>(cfg.layers));

    // Deliberately NOT checked, ever: intermediate_size. It says 6144 for
    // E2B, which is true only of layers 0-14 -- layers 15-34 actually step
    // to 12288 (see DeriveFfnWidths's doc comment for the same trap in the
    // GGUF's own metadata key). Comparing it to anything derived here would
    // reject a correct E2B package.

    // vocab_size and eos_token_id are cross-checked WHEN PRESENT and
    // silently tolerated when absent -- a deliberate choice, not an
    // oversight: since both are identical across every row this port
    // supports (see the comment on num_hidden_layers above), neither one
    // is capable of catching a mismatched package on its own, so refusing
    // to load a package that simply omits them would cost real packages
    // for no detection gained in return. Their value is in catching a
    // config.json that disagrees with the GGUF for some OTHER reason --
    // hand-edited, corrupted, or copied from an unrelated model -- when it
    // does bother to state them.
    if (text_config.contains("vocab_size"))
        RequireJsonUnsigned(text_config, "vocab_size", static_cast<std::uint64_t>(cfg.vocab));
    // eos_token_id is top-level, not under text_config. Google's revisions
    // disagree on its type: 106 in some, [1, 106] in others. An array must
    // contain the GGUF's eos.
    if (config.contains("eos_token_id") && config.at("eos_token_id").is_array()) {
        const auto& ids = config.at("eos_token_id");
        const auto want = static_cast<std::int64_t>(cfg.eos_token_id);
        const bool found = std::any_of(ids.begin(), ids.end(), [&](const auto& id) {
            return id.is_number_integer() && id.template get<std::int64_t>() == want;
        });
        if (!found)
            Fail("eos_token_id", JsonText(ids), "an array containing " + std::to_string(want));
    } else if (config.contains("eos_token_id")) {
        RequireJsonUnsigned(config, "eos_token_id", static_cast<std::uint64_t>(cfg.eos_token_id));
    }
    // AND THE BOS ID SITS IN THE OTHER PLACE. It is nested under
    // `text_config` in every real file and absent from the top level, which
    // is the exact mirror of the eos above -- so `TextModelConfig` is the
    // object to ask, and asking `config` would silently check nothing. The
    // backend's forced_bos_id() rests on this comparison, because
    // tokenizer_config.json states no bos_token_id at all for either shipped
    // row and so cannot be the source.
    if (text_config.contains("bos_token_id"))
        RequireJsonUnsigned(text_config, "bos_token_id", static_cast<std::uint64_t>(cfg.bos_token_id));

    if (tokenizer.contains("model") && tokenizer.at("model").contains("vocab")) {
        const auto& vocab = tokenizer.at("model").at("vocab");
        if (vocab.is_object() && vocab.size() != static_cast<std::size_t>(cfg.vocab)) {
            Fail("tokenizer.json model.vocab size", std::to_string(vocab.size()),
                 std::to_string(cfg.vocab));
        }
        // The third source for each id: tokenizer_config.json NAMES the
        // token and tokenizer.json says which id that name has. Both are
        // checked only when the package bothers to state them, for the
        // reason given above vocab_size.
        const auto check_named_token = [&](const char* key, const char* what,
                                           const std::vector<std::int64_t>& accepted) {
            if (!vocab.is_object()) return;
            const auto named = tokenizer_config.find(key);
            if (named == tokenizer_config.end() || !named->is_string()) return;
            const auto token = named->get<std::string>();
            const auto token_it = vocab.find(token);
            if (token_it == vocab.end()) return;
            const auto id = token_it->get<std::int64_t>();
            if (std::find(accepted.begin(), accepted.end(), id) == accepted.end()) {
                std::string expected;
                for (const auto a : accepted)
                    expected += (expected.empty() ? "" : " or ") + std::to_string(a);
                Fail(what, std::to_string(id), expected);
            }
        };
        // Newer revisions name "<eos>" (1) here while the GGUF stops on
        // "<turn|>" (106); both are in config.json's eos_token_id array.
        check_named_token("eos_token", "tokenizer_config.json eos_token",
                          ConfigEosIds(config, cfg.eos_token_id));
        check_named_token("bos_token", "tokenizer_config.json bos_token",
                          {cfg.bos_token_id});
    }
}

}  // namespace flm::gemma4
