/// \file modeling_qwen3_8mtp_image.cpp
/// \brief Qwen3.8-27B image preprocessing: file/base64 -> the engine's patches
/// \author FastFlowLM Team
/// \date 2026-09-22
/// \version 0.9.28
///
/// This is the host half of the vision path. It reproduces what the HF
/// `Qwen3_5ImageProcessor` does to a PIL image, so that what reaches
/// qwen3_8mtp_npu::encode_image() is byte-for-byte the `pixel_values` the
/// reference dump was captured with:
///
///   decode -> (optional pre-resize) -> HWC to CHW
///          -> smart_resize to a multiple of patch*merge
///          -> bicubic antialias resize
///          -> rescale by 1/255, normalise by mean/std
///          -> replicate the frame temporal_patch_size times
///          -> reorder into MERGE-BLOCK order
///
/// Modelled on modeling_qwen3_5vl_image.cpp; the differences are all in
/// preprocess_image() and are commented where they occur.

#include "AutoModel/modeling_qwen3_8mtp.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>


// ---------------------------------------------------------------------------
// decode
// ---------------------------------------------------------------------------

/// The pre-resize ladder, shared by both loaders. Factored out rather than
/// copied because the two Qwen3_5VL copies of it have already drifted -- one
/// logs with a trailing newline and the other does not.
static int _pre_resize_max_height(int level, int decoded_height) {
    switch (level) {
        case 1: return 480;
        case 2: return 720;
        case 3: return 1080;
        case 4: return 1440;
        case 5: return 2160;
        case 6: return 2880;
        case 7: return 3240;
        case 8: return 4320;
        default: return decoded_height;   // no resizing
    }
}

void Qwen3_8MTP::_apply_pre_resize(image_data_t& decoded) {
    if (this->image_pre_resize <= 0) return;

    const int max_height = _pre_resize_max_height(this->image_pre_resize,
                                                  decoded.height);
    if (decoded.height <= max_height) return;

    image_data_t resized_image;
    const float ratio = static_cast<float>(max_height) /
                        static_cast<float>(decoded.height);
    const int target_width  = static_cast<int>(static_cast<float>(decoded.width) * ratio);
    const int target_height = max_height;
    header_print_r("FLM", "Qwen3.8 resizing image from (" +
                          std::to_string(decoded.width) + ", " +
                          std::to_string(decoded.height) + ") to (" +
                          std::to_string(target_width) + ", " +
                          std::to_string(target_height) + ")\n");
    if (image_reader_.resize_image(decoded, target_width, target_height, resized_image)) {
        image_reader_.recycle(decoded);
        decoded = std::move(resized_image);
    }
}

/// Shared tail of both loaders: CHW reorder and hand-off into the host struct.
qwen3_8mtp_host_image_t Qwen3_8MTP::_finish_load(image_data_t& decoded) {
    qwen3_8mtp_host_image_t empty_result;
    image_data_t reordered;

    this->_apply_pre_resize(decoded);

    if (!image_reader_.reorder_hwc_to_chw(decoded, reordered)) {
        image_reader_.recycle(decoded);
        return empty_result;
    }
    image_reader_.recycle(decoded);

    qwen3_8mtp_host_image_t result;
    result.width  = reordered.width;
    result.height = reordered.height;
    result._data  = std::move(reordered.pixels);
    image_reader_.recycle(reordered);
    return result;
}

qwen3_8mtp_host_image_t Qwen3_8MTP::load_image(const std::string& filename) {
    image_data_t decoded;
    if (!image_reader_.load_image(filename, decoded))
        return qwen3_8mtp_host_image_t{};
    return this->_finish_load(decoded);
}

qwen3_8mtp_host_image_t Qwen3_8MTP::load_image_base64(const std::string& base64_string) {
    image_data_t decoded;
    if (!image_reader_.load_image_base64(base64_string, decoded))
        return qwen3_8mtp_host_image_t{};
    return this->_finish_load(decoded);
}


// ---------------------------------------------------------------------------
// smart_resize
// ---------------------------------------------------------------------------

/// Port of transformers' `smart_resize`. `factor` is patch_size * merge_size,
/// which for this checkpoint is 16 * 2 = 32: the grid must be a whole number
/// of patches AND a whole number of 2x2 merge blocks, or the merger's regroup
/// reads across an image boundary.
void Qwen3_8MTP::smart_resize(
    int height,
    int width,
    int& h_bar,
    int& w_bar,
    int factor,
    int min_pixels,
    int max_pixels
) {
    const double aspect_ratio = static_cast<double>(std::max(height, width)) /
                                static_cast<double>(std::min(height, width));
    if (aspect_ratio > 200.0) {
        header_print("WARNING", "absolute aspect ratio must be smaller than 200, got "
                                << aspect_ratio);
    }

    h_bar = static_cast<int>(std::round(static_cast<double>(height) / factor)) * factor;
    w_bar = static_cast<int>(std::round(static_cast<double>(width)  / factor)) * factor;

    const long long total_pixels = static_cast<long long>(h_bar) * w_bar;

    if (total_pixels > max_pixels) {
        const double beta = std::sqrt((static_cast<double>(height) * width) / max_pixels);
        h_bar = std::max(factor,
                         static_cast<int>(std::floor(height / beta / factor)) * factor);
        w_bar = std::max(factor,
                         static_cast<int>(std::floor(width  / beta / factor)) * factor);
    } else if (total_pixels < min_pixels) {
        const double beta = std::sqrt(static_cast<double>(min_pixels) /
                                      (static_cast<double>(height) * width));
        h_bar = static_cast<int>(std::ceil(height * beta / factor)) * factor;
        w_bar = static_cast<int>(std::ceil(width  * beta / factor)) * factor;
    }
}


// ---------------------------------------------------------------------------
// preprocess
// ---------------------------------------------------------------------------

/// \brief turn one decoded CHW image into the engine's `pixel_values` rows
/// \param image  decoded uint8 (3, H, W); its pixels are freed on the way out
/// \param pixel_values  appended to, fp32, [patches][3 * temporal * patch^2]
///
/// Two differences from the Qwen3_5VL version:
///
///  1. The output is fp32, because qwen3_8mtp_npu::encode_image() takes
///     `const float*`. The vision tower's ops carry activations across the
///     CPU/NPU seam as fp32 host buffers so that the two backends are drop-in
///     for each other, and pixel_values is just the first such buffer.
///
///  2. The geometry constants come from config.json's vision_config, with
///     Qwen3.8-27B's preprocessor_config.json values as the fallback. The
///     Qwen3_5VL path reads converter-injected QWEN3_5_* keys off the engine;
///     this checkpoint carries none of them.
///
/// The reorder itself is staged through bf16, because
/// imgproc::reorder_patches_inplace only writes bf16 and it is not worth a
/// second 90-line copy of that AVX-512 permutation to avoid the round trip.
/// In the tower's shipping mode that round trip is exactly inert: cpu_mm_op
/// rounds its `x` to bf16 before the first multiply anyway, so the patch
/// embedding consumes the same bits either way. It is NOT inert under
/// FLM_Q38_VISION_FP32=1, which is a debug control and not a shipping path.
void Qwen3_8MTP::preprocess_image(qwen3_8mtp_host_image_t& image,
                                  std::vector<float>& pixel_values) {
    const int width    = image.width;
    const int height   = image.height;
    const int channels = 3;   // RGB; vision_config.in_channels

    const unsigned patch_size    = this->vision_patch_size;
    const unsigned merge_size    = this->vision_merge_size;
    const unsigned temporal_size = this->vision_temporal_patch_size;

    int resized_height = 0;
    int resized_width  = 0;
    smart_resize(height, width,
                 resized_height, resized_width,
                 static_cast<int>(patch_size * merge_size),
                 static_cast<int>(this->vision_shortest_edge),
                 static_cast<int>(this->vision_longest_edge));

    const size_t single_frame_size = static_cast<size_t>(resized_height) *
                                     resized_width * channels;
    const size_t total_patch_size  = single_frame_size * temporal_size;
    const unsigned grid_h = static_cast<unsigned>(resized_height) / patch_size;
    const unsigned grid_w = static_cast<unsigned>(resized_width)  / patch_size;

    auto resize_image = imgproc::avx512::resize_bicubic_antialias_rgb_planar_avx512(
        image._data.data(), width, height, resized_width, resized_height, true);

    // Reused across calls: a multi-image prompt would otherwise allocate and
    // free two buffers of this size per image.
    static thread_local std::vector<float> patch_vector_scratch;
    static thread_local std::vector<bf16>  reorder_scratch;
    if (patch_vector_scratch.size() < total_patch_size)
        patch_vector_scratch.resize(total_patch_size);
    if (reorder_scratch.size() < total_patch_size)
        reorder_scratch.resize(total_patch_size);

    imgproc::avx512::rescale_and_normalize_avx512(
        resize_image.data(), patch_vector_scratch.data(),
        resized_width, resized_height, channels,
        true, this->vision_rescale_factor,
        true, this->vision_image_mean, this->vision_image_std);

    // A still image has one frame, which the processor replicates to fill the
    // temporal patch. grid_t therefore stays 1 while the per-patch row is
    // 3 * temporal_patch_size * patch^2 wide.
    for (unsigned l = 1; l < temporal_size; l++) {
        std::memcpy(patch_vector_scratch.data() + l * single_frame_size,
                    patch_vector_scratch.data(),
                    single_frame_size * sizeof(float));
    }

    imgproc::reorder_patches_inplace(
        patch_vector_scratch.data(),
        reorder_scratch.data(),
        1, 1,                       // batch_size, grid_t -- one still image
        static_cast<int>(temporal_size),
        channels,
        static_cast<int>(grid_h), static_cast<int>(grid_w),
        static_cast<int>(merge_size),
        static_cast<int>(patch_size));

    const size_t prev = pixel_values.size();
    pixel_values.resize(prev + total_patch_size);
    float* dst = pixel_values.data() + prev;
    for (size_t i = 0; i < total_patch_size; i++)
        dst[i] = static_cast<float>(reorder_scratch[i]);

    image.width_resized  = resized_width;
    image.height_resized = resized_height;
    image.grid_t         = 1;
    image.grid_h         = static_cast<int>(grid_h);
    image.grid_w         = static_cast<int>(grid_w);
    image._data.free();   // the uint8 source is dead from here on
}


bool Qwen3_8MTP::preprocess_image_file(const std::string& path,
                                       std::vector<float>& pixel_values,
                                       int& grid_t, int& grid_h, int& grid_w) {
    grid_t = grid_h = grid_w = 0;
    qwen3_8mtp_host_image_t image = this->load_image(path);
    if (image.width <= 0 || image.height <= 0) return false;

    const size_t offset = pixel_values.size();
    this->preprocess_image(image, pixel_values);
    if (image.grid_h <= 0 || image.grid_w <= 0) {
        pixel_values.resize(offset);
        return false;
    }
    grid_t = image.grid_t;
    grid_h = image.grid_h;
    grid_w = image.grid_w;
    return true;
}
