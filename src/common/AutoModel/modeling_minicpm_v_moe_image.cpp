/// \file modeling_minicpm_v_moe_image.cpp
/// \brief MiniCPM_V_MOE image preprocessing: MiniCPMV4ImageProcessorPil in C++.
/// \author FastFlowLM Team
/// \date 2026-10-05
/// \note Port of the checkpoint's own image_processing_pil_minicpmv4.py and of
///       the placeholder expansion in processing_minicpmv4.py, rule for rule.
///       Every number here (the 448 scale, the factor 4 in ensure_divide, the
///       slice-grid search) changes how many views and tokens an image produces,
///       and the engine checks that the prompt's image rows match its views
///       exactly -- a wrong rule fails loudly there rather than quietly here.
///
///       Layout handed to the engine (minicpm_v_4_7_moe_npu.cpp, _prefill_with_mm): one
///       minicpm_v_moe_image_t per VIEW, and `_data__processed` holding every view's
///       patches back to back -- raster order within a view, each patch flattened
///       (c, kh, kw) as 3*14*14 bf16. That is the NaViT `[3, 14, T*14]` tensor the
///       HF processor emits, transposed per patch.

#include "AutoModel/modeling_minicpm_v_moe.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

// ---- the processor's arithmetic, as Python computes it ------------------------
// Python's round() is half-to-even and int() truncates; nearbyint and a cast
// are those two exactly.

/// `max(round(length / divisor) * divisor, divisor)`
int ensure_divide(double length, int divisor) {
    return std::max((int)std::nearbyint(length / divisor) * divisor, divisor);
}

/// `find_best_resize()`: (height, width) in pixels, multiples of patch*4 -- two
/// successive 2x2 merges. The inputs are floats on the refine path.
void find_best_resize(double height, double width, int scale, int patch, bool allow_upscale,
                      int& best_h, int& best_w) {
    if (height * width > (double)scale * scale || allow_upscale) {
        const double ar = width / height;
        const int h = (int)(scale / std::sqrt(ar));
        const int w = (int)(h * ar);
        height = h;
        width = w;
    }
    best_w = ensure_divide(width, patch * 4);
    best_h = ensure_divide(height, patch * 4);
}

/// `get_sliced_grid()`: (rows, cols) of slices, or false for no slicing.
bool sliced_grid(int h, int w, int max_slices, int scale, int& rows, int& cols) {
    const double log_ratio = std::log((double)w / h);
    const double ratio = (double)w * h / ((double)scale * scale);
    const int multiple = std::min((int)std::ceil(ratio), max_slices);
    if (multiple <= 1) return false;

    rows = cols = 1;
    double min_error = std::numeric_limits<double>::infinity();
    for (int num_slices : {multiple - 1, multiple, multiple + 1}) {
        if (num_slices == 1 || num_slices > max_slices) continue;
        for (int r = 1; r <= num_slices; r++) {
            if (num_slices % r) continue;
            const int c = num_slices / r;
            const double error = std::fabs(log_ratio - std::log((double)c / r));
            if (error < min_error) { rows = r; cols = c; min_error = error; }
            else if (error == min_error && r > rows) { rows = r; cols = c; }
        }
    }
    return true;
}

// ---- Pillow's bicubic resample, bit for bit ------------------------------------
//
// The HF processor resizes the uint8 image with PIL (Image.resize, BICUBIC; the
// output is exactly PIL's, checked). The shared AVX-512 bicubic
// (imgproc::avx512::resize_bicubic_antialias_rgb_planar_avx512) is close but not
// the same function -- it disagrees with PIL by up to 29 uint8 levels at
// high-contrast pixels (relL2 0.013 on the reference photo), and SigLIP
// amplifies that ~9x: image_embeds relL2 0.116 through HF's own fp32 tower,
// more than every device rounding in the tower put together (0.080). So this
// tower gets Pillow's own algorithm (libImaging/Resample.c, ImagingResample):
// double-precision coefficients normalised per output pixel, quantised to
// 22-bit fixed point, a horizontal pass into a CLAMPED uint8 image, then a
// vertical pass.

constexpr int PIL_PRECISION_BITS = 32 - 8 - 2;

double pil_bicubic(double x) {
    constexpr double a = -0.5;
    if (x < 0.0) x = -x;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
}

/// precompute_coeffs() + normalize_coeffs_8bpc(): per output pixel, the first
/// input index, the tap count, and ksize fixed-point weights.
int pil_coeffs(int in_size, int out_size, std::vector<int>& bounds, std::vector<int32_t>& kk) {
    const double scale = (double)in_size / out_size;
    const double filterscale = std::max(scale, 1.0);
    const double support = 2.0 * filterscale;   // bicubic support 2
    const int ksize = (int)std::ceil(support) * 2 + 1;
    bounds.assign((size_t)out_size * 2, 0);
    kk.assign((size_t)out_size * ksize, 0);
    std::vector<double> k((size_t)ksize);
    for (int xx = 0; xx < out_size; xx++) {
        const double center = (xx + 0.5) * scale;
        const double ss = 1.0 / filterscale;
        int xmin = (int)(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = (int)(center + support + 0.5);
        if (xmax > in_size) xmax = in_size;
        xmax -= xmin;
        double ww = 0.0;
        for (int x = 0; x < xmax; x++) {
            const double w = pil_bicubic((x + xmin - center + 0.5) * ss);
            k[(size_t)x] = w;
            ww += w;
        }
        for (int x = 0; x < xmax; x++) {
            double w = ww != 0.0 ? k[(size_t)x] / ww : k[(size_t)x];
            kk[(size_t)xx * ksize + x] = w < 0 ? (int32_t)(-0.5 + w * (1 << PIL_PRECISION_BITS))
                                               : (int32_t)(0.5 + w * (1 << PIL_PRECISION_BITS));
        }
        bounds[(size_t)xx * 2] = xmin;
        bounds[(size_t)xx * 2 + 1] = xmax;
    }
    return ksize;
}

inline uint8_t pil_clip8(int32_t v) {
    v >>= PIL_PRECISION_BITS;   // arithmetic: floor, as Pillow's lookup table indexes
    return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
}

/// Planar uint8 [3][h][w] -> [3][oh][ow], identical to PIL's Image.resize(BICUBIC).
std::vector<uint8_t> pil_resize_bicubic(const uint8_t* src, int w, int h, int ow, int oh) {
    std::vector<int> bh, bv;
    std::vector<int32_t> kh, kv;
    const int ksh = pil_coeffs(w, ow, bh, kh);
    const int ksv = pil_coeffs(h, oh, bv, kv);
    const bool need_h = ow != w, need_v = oh != h;

    // Only the input rows the vertical pass reads, as ImagingResample does.
    const int y_first = bv[0];
    const int y_last = bv[(size_t)oh * 2 - 2] + bv[(size_t)oh * 2 - 1];

    std::vector<uint8_t> out((size_t)3 * oh * ow);
    for (int c = 0; c < 3; c++) {
        const uint8_t* in = src + (size_t)c * h * w;
        // horizontal pass, into a clamped uint8 temporary
        const int th = need_h ? (y_last - y_first) : h;
        const int y0 = need_h ? y_first : 0;
        std::vector<uint8_t> tmp;
        const uint8_t* mid = in;
        int mid_w = w;
        if (need_h) {
            tmp.resize((size_t)th * ow);
#pragma omp parallel for schedule(static)
            for (int y = 0; y < th; y++) {
                const uint8_t* row = in + (size_t)(y + y0) * w;
                for (int xx = 0; xx < ow; xx++) {
                    const int xmin = bh[(size_t)xx * 2], xmax = bh[(size_t)xx * 2 + 1];
                    const int32_t* k = &kh[(size_t)xx * ksh];
                    int32_t ss = 1 << (PIL_PRECISION_BITS - 1);
                    for (int x = 0; x < xmax; x++) ss += (int32_t)row[x + xmin] * k[x];
                    tmp[(size_t)y * ow + xx] = pil_clip8(ss);
                }
            }
            mid = tmp.data();
            mid_w = ow;
        }
        // vertical pass; bounds shifted by the rows the horizontal pass skipped
        uint8_t* o = out.data() + (size_t)c * oh * ow;
        if (need_v) {
            const int shift = need_h ? y_first : 0;
#pragma omp parallel for schedule(static)
            for (int yy = 0; yy < oh; yy++) {
                const int ymin = bv[(size_t)yy * 2] - shift, ymax = bv[(size_t)yy * 2 + 1];
                const int32_t* k = &kv[(size_t)yy * ksv];
                for (int xx = 0; xx < mid_w; xx++) {
                    int32_t ss = 1 << (PIL_PRECISION_BITS - 1);
                    for (int y = 0; y < ymax; y++) ss += (int32_t)mid[(size_t)(y + ymin) * mid_w + xx] * k[y];
                    o[(size_t)yy * ow + xx] = pil_clip8(ss);
                }
            }
        } else {
            std::copy(mid, mid + (size_t)oh * ow, o);
        }
    }
    return out;
}

/// Append one view's patches -- the region (y0, x0, vh, vw) of a normalised
/// planar [3][H][W] image -- as raster rows of (c, kh, kw).
void patchify(const float* img, int H, int W, int y0, int x0, int vh, int vw, int p,
              std::vector<bf16>& out) {
    const int gh = vh / p, gw = vw / p;
    const size_t row = (size_t)3 * p * p;
    const size_t base = out.size();
    out.resize(base + (size_t)gh * gw * row);
#pragma omp parallel for schedule(static)
    for (int t = 0; t < gh * gw; t++) {
        const int pr = t / gw, pc = t % gw;
        bf16* dst = out.data() + base + (size_t)t * row;
        for (int c = 0; c < 3; c++)
            for (int kh = 0; kh < p; kh++) {
                const float* src = img + ((size_t)c * H + (y0 + pr * p + kh)) * W + x0 + pc * p;
                for (int kw = 0; kw < p; kw++) *dst++ = (bf16)src[kw];
            }
    }
}

}  // namespace

bool MiniCPM_V_MOE::load_image(const std::string& source, bool base64, image_data_t& chw) {
    image_data_t decoded;
    const bool ok = base64 ? image_reader_.load_image_base64(source, decoded)
                           : image_reader_.load_image(source, decoded);
    if (!ok) return false;
    const bool reordered = image_reader_.reorder_hwc_to_chw(decoded, chw);
    image_reader_.recycle(decoded);
    return reordered && chw.width > 0 && chw.height > 0;
}

MiniCPM_V_MOE::image_layout MiniCPM_V_MOE::preprocess_image(const image_data_t& chw,
                                                    minicpm_v_moe_image_payload_t& payload) {
    auto* eng = reinterpret_cast<minicpm_v_4_7_moe_npu*>(this->lm_engine);
    const int patch = (int)eng->MINICPM_V_PATCH_SIZE;                      // 14
    const int max_slices = (int)eng->MINICPM_V_MAX_SLICE_NUMS;             // 9
    const int divisor = (int)eng->MINICPM_V_DOWNSAMPLE_FACTOR;             // 16 (or 4)
    constexpr int scale = 448;   // scale_resolution: the processor's default; not in any config
    const float rescale = eng->MINICPM_V_VISION_RESCALE_FACTOR;
    const float mean = eng->MINICPM_V_VISION_RESCALE_IMAGE_MEAN;
    const float stdv = eng->MINICPM_V_VISION_RESCALE_IMAGE_STD;

    const int H = chw.height, W = chw.width;
    const uint8_t* src = chw.pixels.data();

    image_layout l;
    const bool sliced = sliced_grid(H, W, max_slices, scale, l.rows, l.cols);
    if (!sliced) l.rows = l.cols = 0;
    const size_t before = payload._data__processed.size();

    // resize (Pillow's bicubic, exactly), rescale + normalise, return planar fp32
    std::vector<float> norm;
    auto resized = [&](int h, int w) {
        std::vector<uint8_t> r = pil_resize_bicubic(src, W, H, w, h);
        norm.resize((size_t)3 * h * w);
        imgproc::avx512::rescale_and_normalize_avx512(r.data(), norm.data(), w, h, 3, true, rescale,
                                                      true, mean, stdv);
    };
    auto push_view = [&](int y0, int x0, int vh, int vw, int img_h, int img_w) {
        patchify(norm.data(), img_h, img_w, y0, x0, vh, vw, patch, payload._data__processed);
        minicpm_v_moe_image_t v{};
        v.height = H;
        v.width = W;
        v.height_resized = vh;
        v.width_resized = vw;
        v.grid_h = vh / patch;
        v.grid_w = vw / patch;
        payload.images.push_back(v);
        return v.grid_h * v.grid_w / divisor;
    };

    // the source view: upscaled only when the image is not sliced
    int sh, sw;
    find_best_resize(H, W, scale, patch, /*allow_upscale=*/!sliced, sh, sw);
    resized(sh, sw);
    l.src_tokens = push_view(0, 0, sh, sw, sh, sw);
    l.views = 1;
    l.tokens = l.src_tokens;

    if (sliced) {
        // get_refine_size(): the whole image resized so it tiles exactly
        const int rw0 = ensure_divide(W, l.cols), rh0 = ensure_divide(H, l.rows);
        int bh, bw;
        find_best_resize((double)rh0 / l.rows, (double)rw0 / l.cols, scale, patch, true, bh, bw);
        const int rh = bh * l.rows, rw = bw * l.cols;
        resized(rh, rw);
        // divide_to_patches(): row-major tiles
        for (int r = 0; r < l.rows; r++)
            for (int c = 0; c < l.cols; c++) {
                l.slice_tokens = push_view(r * bh, c * bw, bh, bw, rh, rw);
                l.tokens += l.slice_tokens;
                l.views++;
            }
    }
    payload.num_images = (unsigned)payload.images.size();
    l.values = payload._data__processed.size() - before;
    return l;
}

std::vector<int> MiniCPM_V_MOE::image_placeholder(const image_layout& l, int index) {
    constexpr int image_pad = 248056;
    constexpr int image_start = 248078, image_end = 248079;        // <image> </image>
    constexpr int slice_start = 248088, slice_end = 248089;        // <slice> </slice>
    constexpr int image_id_start = 248090, image_id_end = 248091;  // <image_id> </image_id>
    constexpr int newline = 198;

    std::vector<int> t;
    t.push_back(image_id_start);
    for (int d : this->tokenizer->encode(std::to_string(index))) t.push_back(d);
    t.push_back(image_id_end);
    t.push_back(image_start);
    t.insert(t.end(), (size_t)l.src_tokens, image_pad);
    t.push_back(image_end);
    for (int r = 0; r < l.rows; r++) {
        if (r > 0) t.push_back(newline);   // "\n".join over slice rows
        for (int c = 0; c < l.cols; c++) {
            t.push_back(slice_start);
            t.insert(t.end(), (size_t)l.slice_tokens, image_pad);
            t.push_back(slice_end);
        }
    }
    return t;
}
