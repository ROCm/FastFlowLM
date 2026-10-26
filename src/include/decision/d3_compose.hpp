/// \file d3_compose.hpp
/// \brief System One composed images (`assets`), Box answers and the video-window plan of the d3 package
///        runtime: a byte-exact port of upstream `d3_compose.py` (vllm-sr/d3-turbo v1.1).
/// \note  Pure functions, no NPU, no tokenizer library: token ids and token byte lengths come in as arguments.
///        src/test/d3_compose replays fixtures that gen_compose_fixtures.py renders with the upstream module.
///        Upstream names are kept so each function reads side by side with d3_compose.py.
#pragma once

#include <array>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "decision/d3_contract.hpp"

namespace flm::d3::compose {

constexpr int BINS = 128;                    //!< codes of one edge scale; bin k's centre is (k + 0.5) / BINS
constexpr int PRESENCE_BINS = 2;             //!< `visible`: A = no, B = yes
constexpr int MAX_REFERENCE_BYTES = 1024;
constexpr int MAX_REFERENCES_BYTES = 65536;
constexpr int WINDOW_FRAMES = 32;
constexpr double IOU_HIT = 0.5;
constexpr int CONFIDENCE_POINTS = 512;       //!< quasi-random draws of the box confidence
extern const char* const DEFAULT_INSTRUCTIONS;
extern const char* const REPLY;
extern const char* const HOISTED_LABEL_PREFIX;   //!< "[image "

/// One piece of a user turn: text, an image (a front image by index, or an asset reference) or a video.
struct Part {
    enum class Kind { Text, Image, Video };
    Kind kind = Kind::Text;
    std::string text;   //!< Text: the characters; Image with index < 0: the asset reference
    int index = -1;     //!< Image / Video: position among the request's images / videos; -1 for an asset reference
    bool operator==(const Part& o) const { return kind == o.kind && text == o.text && index == o.index; }
};
Part text_part(std::string s);
Part image_part(int index);                  //!< a front image
Part asset_part(std::string ref);            //!< an image given by asset reference (inline in a question)
Part video_part(int index);

/// `check_references`: validates the asset reference strings (the keys of `assets`) and returns them longest
/// first (code points), ties by code-point order. Throws ContractError with upstream's text.
std::vector<std::string> check_references(const std::vector<std::string>& references);

/// `split_string`: a value equal to a reference is that image whole; otherwise references are cut out left to
/// right without overlap, the first of `refs` (longest first) winning at one position.
std::vector<Part> split_string(const std::string& text, const std::vector<std::string>& refs);

/// `describe_parts`: a question value as parts; a non-string value is shown as its JSON text, a reference that is
/// a whole JSON string value replaces the string, quotes included. Throws ContractError where upstream raises
/// ValueError (stray reserved characters in the value).
std::vector<Part> describe_parts(const PyValue& value, const std::vector<std::string>& refs);

/// `hoist_state`: every reference in a string value of the state (never in keys) becomes the label `[image k]`,
/// k = front + number of uses so far (1-based); the used references, one entry per use, in reading order.
struct Hoisted {
    PyValue state;
    std::vector<std::string> uses;
};
Hoisted hoist_state(const PyValue& state, const std::vector<std::string>& refs, int front);

/// `_merge`: drops empty text parts and joins neighbouring text parts.
std::vector<Part> merge_parts(const std::vector<Part>& parts);

std::string format_seconds(double value);                  //!< `f"{value:.1f}"`
std::vector<std::string> markers(bool presence);           //!< ("visible",)? + ("top", "bottom", "left", "right")
std::string suffix(bool presence);                         //!< "top:\nbottom:\nleft:\nright:" (`visible:\n` first when presence)
/// `tail_text`: the Box lines after the question's instructions. `codes` are the answer codes (255).
std::string tail_text(const std::vector<std::string>& codes, std::optional<double> at, bool presence);

/// `user_parts`: the user turn as parts: front images, videos, `State:` + hoisted state, the question block (a Box
/// question: its lines). `question` is the question object as given; `at` is a video moment's time in seconds.
/// Throws ContractError for a question that does not fit (choice without criteria).
std::vector<Part> user_parts(const PyValue& state, const PyValue& question, const std::vector<std::string>& codes,
                             int n_front, int n_videos, const std::vector<std::string>& refs,
                             std::optional<double> at = std::nullopt);

/// `render`: the chat text of `[system, user(parts)]` with the generation prompt (thinking off), then `box_suffix`.
/// An image part is `<|vision_start|><|image_pad|><|vision_end|>`, a video part the same with `<|video_pad|>`.
std::string render_parts(const std::vector<Part>& parts, const std::string& box_suffix = std::string());

/// `readout_offsets` for a tokenizer without offsets: positions of the edge markers' colon tokens counted back from
/// the prompt's last token (box: 9, 6, 3, 0; track: 12, 9, 6, 3, 0). `ids` are the prompt's token ids (placeholders
/// not expanded; the markers lie at the end so expansion does not move them) and `token_bytes(id)` the UTF-8 bytes
/// one token stands for. Throws ContractError("a Box prompt must end with its edge markers") when `text` does not
/// end with `suffix(presence)`, and ContractError("an edge marker's colon is not the end of a token") when no token
/// ends right after a marker's colon (A11).
std::vector<int> readout_offsets(const std::vector<int>& ids, const std::function<size_t(int)>& token_bytes,
                                 const std::string& text, bool presence);

/// What the readout reads for a question: `count` code probabilities (box edges: BINS, `visible`: 2) at each offset.
struct Readout {
    std::vector<int> offsets;   //!< from the prompt's last token, in `markers(presence)` order
    std::vector<int> counts;
};
Readout readout_plan(const std::vector<int>& offsets, bool presence);

// ---- the 128-bin box readout ---------------------------------------------------------------------------------
struct Box {
    double top = 0, bottom = 0, left = 0, right = 0;
};
/// `expected`: the mean bin centre of one edge's distribution (Python float sums, sequential order).
double expected(const std::vector<double>& probabilities);
/// `order_edges`: (a, b) when a < b, else a pair centred on their mean, half a bin each side, clamped to [0, 1].
std::pair<double, double> order_edges(double a, double b, int bins = BINS);
/// `decode`: edges = the top, bottom, left, right distributions.
Box decode(const std::vector<std::vector<double>>& edges);
/// The 512 fixed R4 quasi-random points `_r4_points(512)`: four coordinates each.
const std::vector<std::array<double, 4>>& r4_points();
/// `confidence`: the share of the quasi-random draws from the edge distributions that give a valid box with IoU >=
/// 0.5 against `box` (numpy semantics: pairwise sums, `searchsorted(side="left")`).
double confidence(const std::vector<std::vector<double>>& edges, const Box& box);
/// A calibration table: knots (raw, calibrated), as decision_config.json `systemone.box_calibration`.
using Calibration = std::vector<std::pair<double, double>>;
double calibrate(double value, const Calibration& table);
/// `{"type": "box", "box": {top, bottom, left, right}, "confidence": c}`
ojson box_answer(const Box& box, double confidence);
/// A track entry: `{"time": round(t, 3), "box", "confidence", "visible"}` (M4 assembles them). `visible` is
/// `p[1] / sum(p)` over the two `visible` probabilities.
double visible_probability(const std::vector<double>& two);
/// Python's `round(x, 3)` (correctly rounded half-even of the decimal expansion).
double py_round3(double x);

// ---- video windows (M4 uses them; ported now so the structures exist) -----------------------------------------
/// `window_plan`: source frames of each track window.
std::vector<std::vector<int>> window_plan(int total, double fps, int rate);
/// `group_times`: the timestamp of each pair of frames, the mean of its two frame times.
std::vector<double> group_times(const std::vector<int>& indices, double fps);

}  // namespace flm::d3::compose
