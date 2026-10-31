/// \file d3_composed.hpp
/// \brief Composed System One requests (`assets`, `box` questions) of a prefix_bidir d3 checkpoint: validation, the
///        per-question rows, the engine requests, and the answers. The port of upstream
///        `D3._system_one_composed` / `_composed_question` / `_composed_row` / `refusal` (d3_runtime.py, v1.1)
///        around the d3_compose port (decision/d3_compose.hpp).
/// \note  CPU only. The engine is behind decision/d3_engine.hpp. Videos (M4): a plain question about the request's
///        videos is a row of the request's state; a box at a moment / a track owns one row per moment, and every
///        window of the clip (a `window_plan` entry, <= 32 frames) is a state of its own, run as one engine call
///        (`state_key` "window:<rate>:<w>") with the moment questions of every box / track question as its blocks.
#pragma once

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "decision/d3_compose.hpp"
#include "decision/d3_contract.hpp"
#include "decision/d3_engine.hpp"
#include "video/video_preprocess.hpp"

namespace flm::d3 {

/// A question refused with a narrower `reason` than the default (upstream `QuestionRefused`).
struct QuestionRefused : ContractError {
    std::string reason;
    QuestionRefused(std::string r, const std::string& message) : ContractError(message), reason(std::move(r)) {}
};

/// A prompt over the input limit (upstream `MaxLengthExceeded`); nothing is truncated.
struct MaxLengthExceeded : ContractError {
    using ContractError::ContractError;
};

/// What a checkpoint declares (decision_config.json `systemone`) and this build's switches.
struct ComposedConfig {
    std::string tag = "d3:turbo";             //!< catalog tag, named in refusals
    std::string model_name = "d3-turbo";      //!< the name in replies and in per-question refusals
    int max_length = 32768;                   //!< tokens per question prompt (long_context is refused: A6)
    std::vector<std::string> question_types = {"choice", "noul", "score", "box", "track"};   //!< `question_types` of the checkpoint
    std::vector<std::string> image_placements = {"state", "instructions", "criteria"};
    compose::Calibration box_calibration;     //!< systemone.box_calibration
    /// A2: upstream rejects `score` questions in any composed request ("score questions take no images here",
    /// d3_runtime.py:3039) although nothing about them is an image. true replicates it; false answers them.
    bool refuse_score = true;
    /// false: an engine without question-image support (no D3_CAP_QUESTION_IMAGES) cannot read the several hidden rows of a box
    /// question whose own text holds an image, so that question is refused inline (`question_images_unsupported`) instead of failing.
    bool box_question_images = true;
    bool video = false;                       //!< M4: videos (the request's `videos`, box at a moment, track)
    /// Without video: a `track` question, or a box at a video moment, refuses the whole request with a 422. Off, they
    /// are answered per question as upstream does (`video_needed`, or the refusal of a feature not declared).
    bool refuse_video_questions = true;
    /// A5 (FLM only): the moments of every box-at-a-moment and track question of one request are rows of the engine;
    /// more than this refuses the request with a 422 (a 60 s track at rate 2 is 120 moments).
    int moment_row_limit = video::DEFAULT_MOMENT_ROW_LIMIT;
    video::VideoProcessorConfig video_config;   //!< the checkpoint's video processor settings (d3-turbo's defaults)
    float temperature = 1.0f;
    /// decision_config temperature_by_questions: a choice / noul / score row reads the table at the request's number of such
    /// questions (`ComposedRequest::decisions`); box and track rows keep `temperature`.
    std::optional<TemperatureTable> temperature_by_questions;
};

/// What the contract needs from its host to prepare a request.
struct ComposedHooks {
    std::function<std::vector<int>(const std::string&)> encode;   //!< the tokenizer: token ids of a text
    std::function<size_t(int)> token_bytes;                       //!< UTF-8 bytes one token id stands for
    /// Decode one image (the host's pixels): called once per distinct source with its position in the request's image
    /// table, right after its header passes. Throws ContractError("...") for an image that cannot be decoded.
    std::function<void(size_t, const ImageFile&)> decode;
    /// `videos[i]` -> a sampled clip (video/video_input.hpp `make_video_loader`; throws video::VideoError,
    /// video::VideoBackendUnavailable). Called at most once per video, when a plain question needs it.
    std::function<std::shared_ptr<const video::VideoClip>(size_t, const PyValue&)> load_video;
    /// `videos[0]` decoded into its windows at `rate` (video::load_video_windows; same exceptions). Called at most once
    /// per rate.
    std::function<std::vector<video::VideoWindow>(const PyValue&, int)> load_windows;
};

/// One distinct image of a request: a request image, or an asset (used any number of times).
struct ComposedImage {
    std::string source;       //!< "images[2]" or the asset reference
    int width = 0, height = 0;
    int grid_h = 0, grid_w = 0;   //!< after the processor's resize (0 when rejected)
    int tokens = 0;           //!< input tokens: grid_h * grid_w / 4
    std::string reject;       //!< the processor's refusal (aspect ratio), "" when accepted
    /// A temporal group of a video (one entry per group, `tokens` its input tokens, the pixels of the host):
    /// of the request's video `clip` (>= 0), or of window `window` of the clip read at `rate` (> 0). `group` counts in the
    /// clip / window. Both are -1 / 0 for an image.
    int clip = -1;
    int rate = 0, window = -1;
    int group = 0;
    bool is_video() const { return clip >= 0 || window >= 0; }
};

/// One question as a row `[state | question block]` (upstream `_composed_row`).
struct ComposedRow {
    std::string key;
    ojson type;                       //!< the question's `type` as given
    std::string kind;                 //!< choice | noul | score | box | track
    Question question;                //!< choice / noul / score
    std::string text;                 //!< the rendered prompt (box: ends with the edge markers)
    std::vector<int> ids;             //!< token ids, image pad runs expanded
    int state_len = 0;                //!< S: tokens of the state block (front images' runs included)
    std::vector<int> state_images;    //!< image table index of each pad run in the state block, in order
    std::vector<int> block_images;    //!< image table index of each pad run in the question block, in order
    compose::Readout readout;         //!< offsets back from the last token, and how many codes each row reads
    std::string state_key = "request";   //!< rows with the same key share one state (a video window gets its own)
    std::optional<double> time;       //!< a video moment's time (the row of a box at a moment / one moment of a track)
    int rate = 0, window = -1;        //!< the window state of a moment row
};

/// A composed request after validation: per-question inline errors, the rows to run, the image table.
struct ComposedRequest {
    std::vector<std::string> keys;                    //!< every question ID, request order
    int decisions = 0;                                //!< questions of type choice / noul / score (valid or not): the temperature table's key
    std::map<std::string, ojson> errors;              //!< answers decided before the engine (refusals, invalid, too long)
    std::vector<ComposedRow> rows;                    //!< runnable rows, request order (M3: one per runnable question)
    std::vector<ComposedImage> images;                //!< distinct images (decoded by `ComposedHooks::decode`, same index)
    std::vector<int> front;                           //!< image table index of every state image, in pad-run order
    PyValue state;                                    //!< the state after hoisting (`[image k]` labels)
    int image_token_id = -1;                          //!< `<|image_pad|>` (-1 without images)
    int video_token_id = -1;                          //!< `<|video_pad|>` (-1 without videos)
    /// The request's videos as decoded for plain questions (one per `videos[i]`; null until a plain question needs it) and
    /// their grids, and the windows of the clip per rate (the state of every moment row; `pixels` are released as each
    /// window's call finishes, hence shared and mutable through a const request).
    std::vector<std::shared_ptr<const video::VideoClip>> clips;
    std::vector<video::VideoGrid> clip_grids;
    std::map<int, std::shared_ptr<std::vector<video::VideoWindow>>> windows;
    std::vector<size_t> rows_of(const std::string& key) const;
};

/// Whether a body is a composed request: it carries (non-empty) `assets` or a box / track question
/// (upstream `_system_one`'s dispatch).
bool is_composed(const PyValue& body);

/// `refusal`: the per-question answer for a feature this checkpoint does not declare (box / track, images inside a
/// question), or null.
ojson refusal(const PyValue* kind, const PyValue& question, const std::vector<std::string>& refs,
              const ComposedConfig& config);

/// Validate and prepare a composed request. Throws RequestError (HTTP 422, A1: request-level problems) for a bad
/// body, bad `assets`, an image that does not load (`images[i]: ...`, hoisted assets counted after the request's
/// images), videos / track (until M4) and long_context; every per-question problem becomes an inline answer.
ComposedRequest prepare_composed(const PyValue& body, const std::vector<std::string>& codes, const ComposedConfig& config,
                                 const ComposedHooks& hooks);

/// Rows that share a state, in request order: the unit of one engine call.
std::vector<std::vector<size_t>> state_groups(const ComposedRequest& request);

/// The engine request of one group of rows. `image` supplies the pixels of one table entry (a CPU test returns grids
/// only); a window group's pixels come from `request.windows`. Throws std::runtime_error when the rows do not share their state tokens.
EngineStateRequest build_engine_request(const ComposedRequest& request, const std::vector<size_t>& group,
                                        const std::function<EngineImage(size_t)>& image);

/// Code probabilities of one row from its hidden rows (`rows` x `hidden` floats, as the engine returns them): per
/// readout position, fp32 hidden @ the first `count` rows of `readout` ([255, hidden] fp32), softmax / temperature.
std::vector<std::vector<double>> row_probabilities(const ComposedRow& row, const std::vector<float>& hidden_rows,
                                                   const float* readout, int hidden_size, float temperature);

/// The answer of one row from its readout probabilities; throws ContractError (upstream ValueError) when the values
/// are unusable.
ojson row_answer(const ComposedRow& row, const std::vector<std::vector<double>>& values, const ComposedConfig& config);

/// `{"model", "answers", "usage"}`: every answer in request order (errors inline, each error with its `reason`),
/// usage = every distinct state once plus every answered question block. `values` holds the readout probabilities of
/// each row of `request.rows` (empty for a row that did not run).
ojson respond_composed(const ComposedRequest& request, const std::vector<std::vector<std::vector<double>>>& values,
                       const ComposedConfig& config);

/// Everything after `prepare_composed`: engine calls (one per state, in the order the states first appear), readout,
/// answers. A window's pixels are released as soon as its call returns. `progress` (optional) is called after each call
/// with (calls done, calls in all, state key, seconds the call took).
ojson run_composed(const ComposedRequest& request, ReadoutEngine& engine, const float* readout, const ComposedConfig& config,
                   const std::function<EngineImage(size_t)>& image,
                   const std::function<void(size_t, size_t, const std::string&, double)>& progress = nullptr);

}  // namespace flm::d3
