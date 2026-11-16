// Copyright (c) 2026 Advanced Micro Devices, Inc.

/// \file oga_engine.hpp
/// \brief A causal_lm over an OGA model, driven one AppendTokens at a time
#pragma once

#include "causal_lm.hpp"
#include "oga/oga_api.hpp"
#include "oga/oga_vision.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace flm::oga {

/// \brief how a package's prompts carry images
struct OgaVisionConfig {
    /// \brief the token an image's run is made of, one per merged patch
    int image_token_id = -1;
    /// \brief the token before each run, which is how OGA finds the images
    int vision_start_token_id = -1;
    /// \brief patches merged into one image token, along each side
    std::int64_t spatial_merge_size = 0;
    /// \brief values in one patch row of pixel_values
    std::int64_t patch_values = 0;
    /// \brief the most patches the images of one prompt may have together
    std::int64_t max_patches = kMaxImagePatches;
};

/// \brief how many generators an engine keeps, one per conversation, and the memory they may take
struct OgaSlotLimits {
    /// \brief the most generators; 0 counts as 1
    std::size_t count = 1;
    /// \brief bytes the generators beyond the first may take together
    std::uint64_t budget_bytes = 0;
    /// \brief bytes one generator takes per position of its maximum length; 0: unknown, no budget
    std::uint64_t bytes_per_position = 0;
};

/// \brief how many generators of `max_length` positions fit `limits`: always at least one
std::size_t SlotsWithin(const OgaSlotLimits& limits, std::uint32_t max_length);

/// \brief one OGA model and a few generators, each holding one conversation
/// \note A prefill goes to the generator (slot) that holds the longest prefix of
///       the logical token sequence it can reach: all it has run, when that is a
///       prefix, or its last checkpoint, which it rewinds to when the OGA can
///       rewind (kRewindOgaVersion). A checkpoint keeps the logits of its last
///       token, so a prompt equal to it runs nothing. No slot reaching a prefix:
///       a new generator replays the whole sequence, in a free slot or in place
///       of the least recently used one. OGA runs images only in a new
///       generator's first call, so a new image rebuilds too, and every rebuild
///       gives it all the images of the history again.
class OgaEngine final : public causal_lm {
public:
    /// \param model a model already checked to run on the GPU; the engine owns it
    /// \param vocab_size the logits width the package states
    /// \param context_limit the most positions the package supports
    /// \param max_length positions this engine may run: 1..context_limit
    /// \param vision how prompts carry images, when the package has a vision model
    /// \param empty_image_batch give a generator without images a batch of one image without
    ///        pixels: OGA shapes the empty image features of a text prompt by the image batch,
    ///        and with none it drops the batch dimension that a 3-D image_features input needs
    /// \param slots how many generators to keep (SlotsWithin at the current max_length)
    OgaEngine(std::shared_ptr<const OgaApi> api, OgaApi::Owned<OgaModel> model,
              std::string display_name, std::uint32_t vocab_size,
              std::uint32_t context_limit, std::uint32_t max_length,
              std::optional<OgaVisionConfig> vision = std::nullopt, bool empty_image_batch = false,
              OgaSlotLimits slots = {});
    ~OgaEngine() override;

    std::uint32_t max_length() const { return max_length_; }
    /// \brief how many generators it keeps at the current max_length
    std::size_t slot_limit() const { return SlotsWithin(limits_, max_length_); }
    /// \brief whether a prefill after restore() continues at the checkpoint
    /// \note It stops at the first rewind OGA refuses.
    bool rewinds() const noexcept { return rewinds_; }
    /// \brief tokens a rebuilt or rewound generator ran again, since construction
    std::uint64_t replayed_tokens() const noexcept { return replayed_; }
    /// \brief prefill tokens a slot had run already, so they were not run, since construction
    /// \note Counts only tokens past the logical sequence a prefill was given: the
    ///       tokens before it are the caller's, which replayed_tokens accounts for.
    std::uint64_t reused_tokens() const noexcept { return reused_; }
    /// \brief the tokens the next prefill calls will give together, in order
    /// \note So a prompt sent in chunks can reach a slot whose checkpoint lies past
    ///       the first chunk. While such a slot is ahead of the chunks, a prefill
    ///       runs nothing and returns zero logits; the prefill that ends the
    ///       expected tokens returns the real ones. A prefill that leaves the
    ///       expected tokens forgets them.
    void expect_prefill(const std::vector<int>& ids);

    buffer<bf16> forward(int id) override;
    /// \param payload an OgaVisionInput with the images of the runs of image
    ///        tokens in `ids`, or null when it has none
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr) override;
    void set_context_length(int length) override;
    void load_weights(Q4NX&) override;
    void update_max_length(std::uint32_t max_length) override;
    void clear_context() override;
    buffer<bf16> get_k_cache(int layer, int index) override;
    buffer<bf16> get_v_cache(int layer, int index) override;
    int get_current_context_length() override;
    int checkpoint() override;
    int restore() override;

private:
    /// \brief a generator and what it has run
    struct Live {
        OgaApi::Owned<OgaGenerator> generator;
        /// \brief max_length the generator was created with
        std::uint32_t capacity = 0;
        /// \brief every token the generator has been given
        std::vector<int> ran;
        /// \brief the length its recurrent state was snapshotted at, which it can rewind to
        std::optional<std::size_t> snapshot;
        /// \brief the logits of the snapshot's last token; set whenever `snapshot` is
        std::vector<bf16> snapshot_logits;
        /// \brief rewound to its snapshot and given nothing since, so OGA's logits are stale
        bool rewound = false;
        /// \brief the hash of each image it has been given, in order
        std::vector<std::uint64_t> images;
        /// \brief when it was last chosen, on the engine's clock
        std::uint64_t used = 0;
    };

    /// \brief one image's run of image tokens in `logical`
    struct ImageSpan {
        std::size_t start = 0;
        std::size_t length = 0;
        std::shared_ptr<const OgaImage> image;
        /// \brief a hash of the image's grid and pixels, which the run's tokens do not tell
        std::uint64_t hash = 0;
    };

    /// \brief `logical` and its images, as checkpoint() left them
    struct Saved {
        std::vector<int> tokens;
        std::vector<ImageSpan> images;
    };

    /// \brief how a pass reached the end of `logical`
    enum class Reuse { kSameTokens, kRewound, kRebuilt };

    /// \brief how far a slot can serve `target`, and whether it has to rewind for it
    struct Reach {
        std::size_t length = 0;
        bool rewind = false;
    };

    /// \brief run `fresh` after `logical`, and keep it there
    buffer<bf16> Run(const std::vector<int>& fresh, const OgaVisionInput* vision = nullptr);
    /// \brief the image runs of `fresh`, which goes at `offset` in `logical`, with their images
    /// \throws std::invalid_argument when the runs and the images do not match
    std::vector<ImageSpan> ImageSpans(const std::vector<int>& fresh, std::size_t offset,
                                      const OgaVisionInput* vision) const;
    /// \brief whether an image of `logical` ends after `position`
    bool ImageAfter(std::size_t position) const;
    /// \brief continue or rebuild a slot, then give it the rest of `logical`
    /// \param why set, on a rebuild, to why that slot was taken
    /// \return how, and how many tokens of `logical` were already run
    std::pair<Reuse, std::size_t> Append(const char** why);
    /// \brief choose the slot that reaches the longest prefix of the target (ties: the most
    ///        recently used), make it the active one and bring it there, or rebuild one
    Reuse Resume(const char** why);
    /// \brief what the slots are matched against: the expected tokens while `logical` is a
    ///        prefix of them, else `logical`
    const std::vector<int>& Target();
    /// \brief how much of `target` `slot` can serve
    Reach ReachOf(const Live& slot, const std::vector<int>& target) const;
    /// \brief an empty generator, which replays all of `logical`, in `slot` or, without one, in a
    ///        free slot or the least recently used one
    Reuse Rebuild(std::optional<std::size_t> slot, const char** why);
    std::size_t LeastRecentlyUsed() const;
    /// \brief remove a slot and its generator
    void Drop(std::size_t slot);
    /// \brief drop the least recently used slots until the limit holds
    void Trim();
    /// \brief " (slot i of n)" when there is more than one slot, for the log
    std::string SlotText(const char* why = nullptr) const;
    /// \brief the active slot
    Live& Active() { return slots_[*active_]; }
    OgaApi::Owned<OgaGenerator> NewGenerator(std::uint32_t max_length) const;
    void AppendTokens(OgaGenerator* generator, const int* ids, std::size_t count) const;
    /// \brief give an empty generator `ids` and `images` in one call
    /// \note No images: the batch of one image without pixels (empty_image_batch)
    void SetInputs(OgaGenerator* generator, const std::vector<int>& ids,
                   const std::vector<const OgaImage*>& images) const;
    /// \brief run the vision model once at max_patches, before the first real image
    void WarmUpVision();
    /// \brief a tensor that OGA owns, with `bytes` of `data` copied in when given
    OgaApi::Owned<OgaTensor> NewTensor(const std::vector<std::int64_t>& shape,
                                       OgaElementType type, const void* data,
                                       std::size_t bytes) const;
    /// \brief the last position's logits, as bf16
    buffer<bf16> Logits(OgaGenerator* generator) const;
    void CheckIds(const std::vector<int>& ids) const;

    // Declared first, so it is destroyed last: the API, whose entry points the
    // destructors of every OGA object below call, outlives them.
    std::shared_ptr<const OgaApi> api_;
    std::string display_name_;
    std::uint32_t vocab_size_;
    std::uint32_t context_limit_;
    std::uint32_t max_length_;
    std::optional<OgaVisionConfig> vision_;
    bool empty_image_batch_ = false;
    bool rewinds_ = false;
    bool vision_warm_ = false;
    OgaSlotLimits limits_;
    std::uint64_t replayed_ = 0;
    std::uint64_t reused_ = 0;
    /// \brief counts slot choices, for least recently used
    std::uint64_t clock_ = 0;
    std::vector<int> logical_;
    /// \brief what expect_prefill said `logical` will become, or empty
    std::vector<int> expected_;
    /// \brief the image runs of `logical`, in order
    std::vector<ImageSpan> images_;
    std::optional<Saved> saved_;
    // The generators are destroyed before the model they read.
    OgaApi::Owned<OgaModel> model_;
    std::vector<Live> slots_;
    /// \brief the slot the last pass ran in, which checkpoint() snapshots
    std::optional<std::size_t> active_;
};

}  // namespace flm::oga
