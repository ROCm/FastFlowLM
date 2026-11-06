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

/// \brief one OGA model and the tokens one generator has been given
/// \note Exact token prefixes continue the live generator. A divergent history
///       rebuilds it and replays the logical token sequence, unless the OGA can
///       rewind (kRewindOgaVersion) and the history still holds the generator's
///       last checkpoint: then it rewinds there. OGA runs images only in a new
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
    OgaEngine(std::shared_ptr<const OgaApi> api, OgaApi::Owned<OgaModel> model,
              std::string display_name, std::uint32_t vocab_size,
              std::uint32_t context_limit, std::uint32_t max_length,
              std::optional<OgaVisionConfig> vision = std::nullopt, bool empty_image_batch = false);
    ~OgaEngine() override;

    std::uint32_t max_length() const { return max_length_; }
    /// \brief whether a prefill after restore() continues at the checkpoint
    bool rewinds() const noexcept { return rewinds_; }
    /// \brief tokens a rebuilt or rewound generator ran again, since construction
    std::uint64_t replayed_tokens() const noexcept { return replayed_; }

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
        /// \brief the hash of each image it has been given, in order
        std::vector<std::uint64_t> images;
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

    /// \brief run `fresh` after `logical`, and keep it there
    buffer<bf16> Run(const std::vector<int>& fresh, const OgaVisionInput* vision = nullptr);
    /// \brief the image runs of `fresh`, which goes at `offset` in `logical`, with their images
    /// \throws std::invalid_argument when the runs and the images do not match
    std::vector<ImageSpan> ImageSpans(const std::vector<int>& fresh, std::size_t offset,
                                      const OgaVisionInput* vision) const;
    /// \brief whether an image of `logical` ends after `position`
    bool ImageAfter(std::size_t position) const;
    /// \brief continue or rebuild the generator, then give it the rest of `logical`
    /// \return how, and how many tokens of `logical` were already run
    std::pair<Reuse, std::size_t> Append();
    /// \brief make `live` stand for a prefix of `logical`, or rebuild it
    Reuse Resume();
    /// \brief replace the generator with an empty one, which replays all of `logical`
    Reuse Rebuild();
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
    std::uint64_t replayed_ = 0;
    std::vector<int> logical_;
    /// \brief the image runs of `logical`, in order
    std::vector<ImageSpan> images_;
    std::optional<Saved> saved_;
    // The generator is destroyed before the model it reads.
    OgaApi::Owned<OgaModel> model_;
    std::optional<Live> live_;
};

}  // namespace flm::oga
