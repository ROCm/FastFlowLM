/// \file d3_contract.hpp
/// \brief The d3 (Decision 3.0) System One contract: request validation, the prompt the model reads,
///        the readout, and the answer objects. A byte-exact port of the upstream package's
///        d3_format.py, d3_runtime.py (normalize_question, prepare, product_answer, respond) and
///        d3_server.py (request checks) at vllm-sr/d3-lite v3.1.3.
/// \note  No NPU code here: the engine returns one hidden row per question and everything else is
///        this file. src/test/d3_contract checks it against the frozen goldens and fuzz requests.
#pragma once

#include <functional>
#include <memory>
#include <map>
#include <optional>
#include <utility>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "decision/d3_engine.hpp"
#include "decision/py_json.hpp"
#include "video/video_preprocess.hpp"

namespace flm::d3 {

using ojson = nlohmann::ordered_json;

constexpr int MAX_OPTIONS = 255;
constexpr const char* FORMAT_ID = "d3-code-readout-v1";
extern const char* const SYSTEM_PROMPT;

/// A problem with one question (`invalid_question`) or with a model output (`invalid_model_output`);
/// the message is upstream's ValueError text.
struct ContractError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// A request the server answers with 422 `{"detail": detail}` before running anything.
struct RequestError : std::runtime_error {
    ojson detail;
    explicit RequestError(ojson d) : std::runtime_error(d.is_string() ? d.get<std::string>() : d.dump()), detail(std::move(d)) {}
};

/// One validated question (d3_runtime.Question).
struct Question {
    std::string kind;                       //!< choice | noul | score
    std::vector<std::string> keys;          //!< answer keys, in option order ("0".."n-1" for score)
    std::vector<std::string> option_texts;  //!< the text after each code in the prompt
    std::vector<PyValue> levels;            //!< score: the original levels (legend)
    PyValue instructions;                   //!< as given (null when absent)
};

/// `describe`: strings verbatim, anything else as `json.dumps(v, ensure_ascii=False)`.
std::string describe(const PyValue& v);

/// `normalize_question`; throws ContractError with upstream's message.
Question normalize_question(const PyValue& question);

/// The 255 candidate codes in preference order (A..Z, AA..ZZ).
std::vector<std::string> candidate_codes();

/// prompt_layout `header-v1` (d3_format.HEADER_TITLE).
constexpr const char* HEADER_TITLE = "Questions in this request:";

/// `d3_format.header_lines` joined as the user prompt carries them: the title, one numbered line per question
/// (its instructions, or the default), the option texts (no codes) of choice / score questions as `   - text`
/// lines, then one empty line; the result ends with "\n\n" so that "State:" follows directly. `header` is every
/// normalizable question of the request, in request order; throws ContractError when it is empty.
std::string header_block(const std::vector<Question>& header);

/// `d3_format.user_prompt` (state may be null). `header` (header-v1: `header_block`) leads the text; empty: none.
std::string user_prompt(const PyValue& state, const Question& q, const std::vector<std::string>& codes,
                        const std::string& header = std::string());

/// The chat text of `[system, user]` (or `[user]` when system is empty) with
/// `add_generation_prompt=True, enable_thinking=False`, as the package's chat_template.jinja renders
/// it. AutoDecisionModel cross-checks this against minja at load.
std::string chat_text(const std::string& system, const std::string& user);

/// The whole prompt of one question (`d3_format.render`).
std::string render(const PyValue& state, const Question& q, const std::vector<std::string>& codes,
                   const std::string& header = std::string());

// ---- decision_config temperature_by_questions (d3_runtime.temperature_table / temperature_for) ----
/// `{"max_questions": [n_1..n_k], "temperature": [t_1..t_k, t_rest]}`.
struct TemperatureTable {
    std::vector<int> max_questions;
    std::vector<float> temperature;
    /// The temperature of a request with `n` choice / noul / score questions: the t_i of the first n <= n_i, else t_rest.
    float for_questions(int n) const;
};
/// `temperature_table`: null gives nullopt; throws std::runtime_error ("<source>: ...") with upstream's checks
/// (exactly the keys max_questions and temperature, strictly increasing integer counts >= 1, one more finite
/// positive temperature than counts).
std::optional<TemperatureTable> temperature_table(const ojson& value, const std::string& source);
/// `D3.temperature_for`: the table's value, or `base` without a table.
float temperature_for(const std::optional<TemperatureTable>& table, float base, int n_questions);

/// d3_runtime.DECISION_KINDS: the question types a request's temperature applies to.
bool is_decision_kind(const std::string& kind);

// ---- images (d3_runtime: load_image strict, visual_tokens, image_messages) ----------------------
constexpr const char* IMAGE_TOKEN = "<|image_pad|>";
constexpr const char* VIDEO_TOKEN = "<|video_pad|>";
constexpr long long MAX_IMAGE_BYTES = 8000000;           //!< decoded bytes of one image
constexpr long long MAX_IMAGE_SOURCE_PIXELS = 16000000;  //!< width x height before the resize
constexpr int IMAGE_MIN_PIXELS = 65536;                  //!< resize bounds (the processor's size)
constexpr int IMAGE_MAX_PIXELS = 1638400;
constexpr int IMAGE_PATCH_SIZE = 16;
constexpr int IMAGE_MERGE_SIZE = 2;

/// One request image as the server reads it before decoding pixels: the file bytes, the format Pillow
/// would identify ("PNG", "JPEG", "WEBP") and the size in its header.
struct ImageFile {
    std::string data;
    std::string format;
    int width = 0;
    int height = 0;
};

/// `load_image(value, strict=True)` up to decoding: a data URL string of type png/jpeg/jpg/webp, valid
/// base64 within MAX_IMAGE_BYTES, a PNG/JPEG/WebP file of at most MAX_IMAGE_SOURCE_PIXELS. Throws
/// ContractError with upstream's message (the caller adds "images[i]: "). Decoding and the integrity
/// check are the application's.
ImageFile inspect_image(const PyValue& value);

/// `_split_data_url(url, media)`: `data:<media>/<rest>;base64,<text>` gives `<rest>` (subtype with any parameters,
/// lower-cased) and `text` as given. Throws ContractError("expected a base64 data URL: data:<media>/<type>;base64,<data>").
std::string split_data_url(const std::string& url, const std::string& media, std::string& text);

/// `_decode_base64(text, limit)`: validated base64; with `limit` > 0 text longer than `limit` bytes can encode to and
/// decoded data over `limit` bytes are refused (upstream's texts). Throws ContractError.
std::string decode_base64(const std::string& text, long long limit);

/// The processor's `smart_resize` (factor = patch x merge): (height, width) of the resized image.
/// Throws ContractError("absolute aspect ratio must be smaller than 200, got <ratio>").
std::pair<int, int> smart_resize(int height, int width, int factor = IMAGE_PATCH_SIZE * IMAGE_MERGE_SIZE,
                                 int min_pixels = IMAGE_MIN_PIXELS, int max_pixels = IMAGE_MAX_PIXELS);

/// The prompt of one question with `n_images` image placeholders in front of the user text
/// (`image_messages` rendered by the chat template).
std::string render_image(const PyValue& state, const Question& q, const std::vector<std::string>& codes,
                         int n_images);

/// The prompt of one question with the image placeholders, then `n_videos` video placeholders, in front of the user
/// text (`video_messages` rendered by the chat template): `<|vision_start|><|video_pad|><|vision_end|>` each.
std::string render_video(const PyValue& state, const Question& q, const std::vector<std::string>& codes,
                         int n_images, int n_videos);

/// `d3_format.answer_codes`: the codes and token ids this tokenizer gives; throws ContractError if
/// fewer than 255 qualify.
void answer_codes(const std::function<std::vector<int>(const std::string&)>& encode,
                  std::vector<std::string>& codes, std::vector<int>& token_ids);

/// Masked code logits -> option probabilities, as the runtime computes them: fp32 hidden @ fp32
/// readout^T, codes >= count masked, / temperature, fp32 softmax, cut to count.
std::vector<float> readout_probabilities(const float* hidden, const float* readout, int hidden_size,
                                         int count, float temperature);

/// `product_answer`: the public answer object; throws ContractError (invalid_model_output) when the
/// probabilities are unusable.
ojson product_answer(const Question& q, const std::vector<double>& probabilities);

/// `{"type": <type as given>, "error": error, "message": message}`.
ojson error_answer(const ojson& type, const std::string& error, const std::string& message);

/// A PyValue as nlohmann JSON (for echoing request values in replies).
ojson to_ojson(const PyValue& v);

/// A System One request after validation and rendering (d3_runtime.Prepared, text, image and video paths).
struct Prepared {
    std::vector<std::string> keys;              //!< every question ID, request order
    PyValue state;                              //!< the request's state as given (null when absent)
    std::map<std::string, Question> questions;  //!< valid questions
    std::map<std::string, std::string> texts;   //!< rendered prompts of valid questions
    std::map<std::string, std::vector<int>> sequences;  //!< token ids of runnable questions
    std::map<std::string, ojson> errors;        //!< per-question error objects
    bool has_images = false;
    bool has_videos = false;
    /// the request's videos as decoded and sampled (shared by every question), with what the processor makes of them
    std::vector<std::shared_ptr<const video::VideoClip>> videos;
    std::vector<video::VideoGrid> video_grids;
    /// per video: its text with ONE `<|video_pad|>` per temporal group (`<t seconds><|vision_start|>..<|vision_end|>`
    /// per group), and each group's run length in input tokens; `tokenize` expands the pads
    std::vector<std::string> video_texts;
    std::vector<std::vector<int>> video_runs;  //!< per video: tokens_per_group of each temporal group
    std::vector<std::vector<int>> video_ids;   //!< per video, filled by `tokenize`: its expanded token ids
    std::vector<ImageFile> images;              //!< the request's images, in order (shared by every question)
    std::vector<std::pair<int, int>> image_grids; //!< per image: (grid_h, grid_w) in 16-pixel patches
    std::vector<int> image_tokens;              //!< per image: input tokens, grid_h * grid_w / 4
    /// prefix_bidir (filled by `tokenize` when it is given token byte lengths): per runnable question, the
    /// number S of leading tokens of its (image-expanded) sequence that form the state block
    std::map<std::string, int> state_len;
    bool reasons = false;   //!< `respond` adds the per-question error `reason` (PrepareOptions::reasons)

    std::vector<std::string> runnable() const;
};

/// Per-model behaviour of `prepare` and `tokenize`.
struct PrepareOptions {
    /// A prefix_bidir checkpoint (d3:turbo) whose build runs choice / noul / score questions on text and
    /// images only: box and track questions, `assets`, `long_context` and `videos` are refused with a 422
    /// that says so, naming this tag. Empty (d3:lite, d3:nano, d3:mini): upstream's messages, unchanged.
    std::string unsupported_for;
    /// With `unsupported_for`: this build also runs composed requests (`assets`, `box` questions: decision/
    /// d3_composed.hpp), so `assets` and plain `box` questions are accepted here and only what still has no
    /// engine (`long_context`, `videos`, `track`, a box at a video moment) is refused.
    bool composed = false;
    /// With `composed`: refuse `track` questions and a box at a video moment with a 422 (while no video reaches the
    /// engine). Off: they are answered per question like upstream does (`video_needed` / `track_unsupported`).
    bool refuse_video_questions = true;
    /// Error answers carry a `reason` (`question_malformed`, `question_tokens`, `model_output`; vllm-sr
    /// contract 3.1.0, which d3-turbo's runtime implements). Off for d3:lite / nano / mini: their upstream
    /// runtime does not add it, and their replies stay bit-identical.
    bool reasons = false;
    /// What this build answers, named in a refusal ("... by this build of d3:turbo (it answers <answers>)"); empty: the
    /// default text for the mode (choice, noul, score [and box] questions about text and images).
    std::string answers;
    /// decision_config `prompt_layout` "header-v1": a text request's prompts start with the header block listing every
    /// normalizable question (upstream applies it to the text path only; image, video and composed prompts keep the
    /// plain layout).
    bool header_v1 = false;
    /// Videos (d3:turbo with the video decoder on): loads `videos[i]` (see video/video_input.hpp: throws
    /// video::VideoError, or video::VideoBackendUnavailable when no decoder is there). Null: `videos` stays
    /// refused with the messages above, so nothing changes for builds without the decoder.
    std::function<std::shared_ptr<const video::VideoClip>(size_t, const PyValue&)> load_video;
    /// The video processor settings (the checkpoint's; defaults are d3-turbo's).
    video::VideoProcessorConfig video_config;
};

/// `d3_runtime.with_reasons`: every error answer without a `reason` gets its error's default.
void add_reasons(ojson& answers);

/// The server-level checks of `prepare` on a decoded body (d3_server.Decider.arguments plus this build's
/// refusals, see PrepareOptions): returns the body's `questions` object. Throws RequestError.
const PyValue& check_request(const PyValue& body, const PrepareOptions& options);

/// Server-level checks of a decoded body (d3_server.Decider.arguments + D3.prepare's request
/// checks) and per-question validation/rendering. Throws RequestError for what upstream answers 422.
/// Tokenization and the length limit are applied by `tokenize()`. Images are inspected in order, and
/// `decode` (when given) decodes each one right after its inspection, as upstream loads them one by
/// one: it throws ContractError with the message for "images[i]: ..." (integrity, decoding).
Prepared prepare(const PyValue& body, const std::vector<std::string>& codes,
                 const std::function<void(size_t, const ImageFile&)>& decode = nullptr,
                 const PrepareOptions& options = PrepareOptions());

/// Tokenize every rendered prompt; prompts over `max_length` (> 0) become max_length_exceeded. With
/// images, each `<|image_pad|>` becomes that image's run of input tokens (counted in the length).
/// With `token_bytes` (UTF-8 bytes one token id stands for) the prefix_bidir state length of every
/// runnable question is computed too (`Prepared::state_len`, `state_block_length`).
void tokenize(Prepared& p, const std::function<std::vector<int>(const std::string&)>& encode, int max_length,
              const std::function<size_t(int)>& token_bytes = nullptr);

// ---- prefix_bidir: the state block (d3_format.state_end_char / state_token_count / encode_split) ----
/// The state segment of a rendered prompt ends where the question block starts: `state_end_char`, as a
/// byte offset into the UTF-8 text. Throws ContractError (upstream's ValueError text) when the prompt has
/// no state segment followed by the question block.
size_t state_end_byte(const std::string& text, const PyValue& state);

/// `state_token_count` without an offsets mapping. The tokenizers C API gives token ids only; for a
/// byte-level BPE vocabulary a token's end offset is the running sum of `token_bytes(id)` (the UTF-8
/// bytes its text stands for), so "all the token's characters lie before end_char" is
/// `running_sum <= end` where `end` is the length in token bytes of the state segment: the sum over
/// `encode(text[:end_byte])`, which is `end_byte` for NFC text and the normalized length otherwise (the
/// tokenizer normalizes to NFC and upstream's offsets point back into the original text). A token that
/// straddles the boundary belongs to the question block. Throws ContractError for an empty segment.
int state_token_count(const std::vector<int>& ids, const std::function<size_t(int)>& token_bytes, size_t end);

/// `encode_split`'s count for one rendered prompt: `ids` are the prompt's tokens (image placeholders not
/// yet expanded), `media` the tokens the expansion adds (sum of image tokens - number of images).
/// `segment_cache` (optional) remembers the token-byte length of a state segment text across the
/// questions of one request.
/// Throws std::runtime_error when the state block is empty or covers the whole prompt (upstream's
/// RuntimeError: such a prompt cannot be split).
int state_block_length(const std::string& text, const PyValue& state, const std::vector<int>& ids,
                       const std::function<std::vector<int>(const std::string&)>& encode,
                       const std::function<size_t(int)>& token_bytes, int media, int expanded_length,
                       std::map<std::string, size_t>* segment_cache = nullptr);

/// `invalid_questions` of a prepared request, or an empty object.
ojson invalid_questions(const Prepared& p);

/// prefix_bidir plain request (no `assets`, no box / track) as ONE engine call: the state block shared by every
/// runnable question (the first S tokens of each sequence) and one block per question (the tokens after it, readout
/// at the last token). `images` are the request's images in order, all inside the state block (a plain question has
/// none of its own). Throws std::runtime_error when the questions do not share the state block's tokens (upstream's
/// _run_state_once refuses that too). Blocks are in `Prepared::runnable()` order; `EngineBlock::tag` is the key.
EngineStateRequest plain_state_request(const Prepared& p, std::vector<EngineImage> images, int image_token_id);

/// The option probabilities of every runnable question from the engine's rows (one hidden row per block, in the
/// order of `plain_state_request`): the fp32 readout of `readout_probabilities`.
std::map<std::string, std::vector<double>> plain_probabilities(const Prepared& p, const std::vector<std::vector<float>>& hidden,
                                                               const float* readout, int hidden_size, float temperature);

/// `D3.respond`: `{"model", "answers", "usage"}` from per-question probabilities.
/// For a prefix_bidir request (`Prepared::state_len` filled) `usage.input_tokens` counts the state block
/// once plus every question block (upstream's default state-once path; `state_once` false, D3_STATE_ONCE=0,
/// counts every whole prompt). Other models count the whole prompts, as ever.
ojson respond(const Prepared& p, const std::map<std::string, std::vector<double>>& probabilities,
              const std::string& model_name, bool state_once = true);

}  // namespace flm::d3
