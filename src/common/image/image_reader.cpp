/// \file image_reader.cpp
/// \brief image_reader class
/// \author FastFlowLM Team
/// \date 2025-08-16
/// \version 0.9.24
/// \note This is a source file for the image_reader functions

#include "image/image_reader.hpp"
#include "typedef.hpp"
#include "base64.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <mutex>
#include "utils/debug_utils.hpp"
#include <csetjmp>
#include <cstdio>
#include <jpeglib.h>
#include "lodepng/lodepng.h"
#include <webp/decode.h>

namespace {
enum class ImageCodec { None, Jpeg, Png, Webp };

static uint32_t read_be32(const uint8_t* ptr) {
    return (static_cast<uint32_t>(ptr[0]) << 24) |
           (static_cast<uint32_t>(ptr[1]) << 16) |
           (static_cast<uint32_t>(ptr[2]) << 8) |
           static_cast<uint32_t>(ptr[3]);
}

static bool has_valid_png_structure(const uint8_t* data, size_t size) {
    if (!data || size < 8) {
        return false;
    }

    static const uint8_t kPngSig[8] = { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A };
    if (std::memcmp(data, kPngSig, sizeof(kPngSig)) != 0) {
        return false;
    }

    size_t offset = 8;
    bool saw_ihdr = false;
    bool saw_iend = false;

    while (offset + 12 <= size) {
        const uint32_t chunk_len = read_be32(data + offset);
        const uint8_t* chunk_type = data + offset + 4;
        const size_t full_chunk = static_cast<size_t>(chunk_len) + 12;

        if (offset + full_chunk > size) {
            return false;
        }

        const bool is_ihdr = chunk_type[0] == 'I' && chunk_type[1] == 'H' && chunk_type[2] == 'D' && chunk_type[3] == 'R';
        const bool is_iend = chunk_type[0] == 'I' && chunk_type[1] == 'E' && chunk_type[2] == 'N' && chunk_type[3] == 'D';

        if (!saw_ihdr) {
            if (!is_ihdr || chunk_len != 13) {
                return false;
            }
            saw_ihdr = true;
        }

        if (is_iend) {
            if (chunk_len != 0) {
                return false;
            }
            saw_iend = true;
            return offset + full_chunk == size;
        }

        offset += full_chunk;
    }

    return false;
}

static bool has_valid_jpeg_structure(const uint8_t* data, size_t size) {
    if (!data || size < 4) {
        return false;
    }

    if (!(data[0] == 0xFF && data[1] == 0xD8)) {
        return false;
    }

    bool found_eoi = false;

    for (size_t i = size - 2; i >= 2; --i) {
        if (data[i] == 0xFF && data[i + 1] == 0xD9) {
            found_eoi = true;
            break;
        }
    }

    if (!found_eoi) {
        return false;
    }

    bool saw_sos = false;
    size_t pos = 2;
    while (pos + 1 < size) {
        if (data[pos] != 0xFF) {
            ++pos;
            continue;
        }

        while (pos < size && data[pos] == 0xFF) {
            ++pos;
        }
        if (pos >= size) {
            return false;
        }

        const uint8_t marker = data[pos++];
        if (marker == 0xD9) {
            return saw_sos && (pos == size);
        }

        if (marker == 0xDA) {
            saw_sos = true;
            return true;
        }

        if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) {
            continue;
        }

        if (pos + 1 >= size) {
            return false;
        }

        const uint16_t seg_len = static_cast<uint16_t>((data[pos] << 8) | data[pos + 1]);
        if (seg_len < 2) {
            return false;
        }

        pos += static_cast<size_t>(seg_len);
    }

    return false;
}

static bool has_basic_image_integrity(const uint8_t* data, size_t size, ImageCodec codec) {
    if (!data || size == 0) {
        return false;
    }

    if (codec == ImageCodec::Png) {
        return has_valid_png_structure(data, size);
    }
    if (codec == ImageCodec::Jpeg) {
        return has_valid_jpeg_structure(data, size);
    }
    if (codec == ImageCodec::Webp) {
        return WebPGetInfo(data, size, nullptr, nullptr) != 0;   // libwebp validates the RIFF container
    }
    return false;
}

static ImageCodec detect_codec(const uint8_t* data, size_t size) {
    if (!data || size < 8) {
        return ImageCodec::None;
    }
    if (data[0] == 0xFF && data[1] == 0xD8) {
        return ImageCodec::Jpeg;
    }
    if (data[0] == 0x89 && data[1] == 0x50 && data[2] == 0x4E && data[3] == 0x47) {
        return ImageCodec::Png;
    }
    if (size >= 12 && std::memcmp(data, "RIFF", 4) == 0 && std::memcmp(data + 8, "WEBP", 4) == 0) {
        return ImageCodec::Webp;
    }
    return ImageCodec::None;
}

// Same limit as Pillow's DecompressionBombError (2 * Image.MAX_IMAGE_PIXELS):
// a few bytes of header can otherwise request a multi-gigabyte allocation.
constexpr size_t kMaxImagePixels = 178956970;

// libjpeg reports fatal errors through error_exit, whose default calls exit();
// jump back to the caller instead. Warnings (e.g. extraneous bytes) are muted:
// libjpeg still produces a full image for them, as the previous decoder did.
struct JpegErrorManager {
    jpeg_error_mgr pub;
    std::jmp_buf jump;
    char message[JMSG_LENGTH_MAX];
};

static void jpeg_error_exit(j_common_ptr cinfo) {
    auto* err = reinterpret_cast<JpegErrorManager*>(cinfo->err);
    (*cinfo->err->format_message)(cinfo, err->message);
    std::longjmp(err->jump, 1);
}

static void jpeg_output_message(j_common_ptr) {}

// Thin wrapper so that no object with a destructor lives inside the setjmp
// regions. Decodes with libjpeg's defaults (accurate integer IDCT, smooth
// chroma upsampling), which is what PIL uses.
class JpegDecoder {
public:
    JpegDecoder() {
        cinfo_.err = jpeg_std_error(&err_.pub);
        err_.pub.error_exit = jpeg_error_exit;
        err_.pub.output_message = jpeg_output_message;
        err_.message[0] = '\0';
    }
    ~JpegDecoder() {
        if (created_) {
            jpeg_destroy_decompress(&cinfo_);
        }
    }
    JpegDecoder(const JpegDecoder&) = delete;
    JpegDecoder& operator=(const JpegDecoder&) = delete;

    bool read_header(const uint8_t* data, size_t size) {
        if (setjmp(err_.jump)) {
            return false;
        }
        jpeg_create_decompress(&cinfo_);
        created_ = true;
        jpeg_mem_src(&cinfo_, data, static_cast<unsigned long>(size));
        jpeg_read_header(&cinfo_, TRUE);
        cmyk_ = cinfo_.jpeg_color_space == JCS_CMYK || cinfo_.jpeg_color_space == JCS_YCCK;
        cinfo_.out_color_space = cmyk_ ? JCS_CMYK : JCS_RGB;
        return true;
    }

    // Writes width*height pixels: RGB24 into rgb, or CMYK into cmyk for
    // CMYK/YCCK images (libjpeg can't convert those to RGB itself).
    bool read_pixels(uint8_t* rgb, uint8_t* cmyk) {
        if (setjmp(err_.jump)) {
            return false;
        }
        jpeg_start_decompress(&cinfo_);
        uint8_t* base = cmyk_ ? cmyk : rgb;
        const size_t stride = static_cast<size_t>(cinfo_.output_width) * cinfo_.output_components;
        while (cinfo_.output_scanline < cinfo_.output_height) {
            JSAMPROW row = base + static_cast<size_t>(cinfo_.output_scanline) * stride;
            jpeg_read_scanlines(&cinfo_, &row, 1);
        }
        jpeg_finish_decompress(&cinfo_);
        return true;
    }

    int width() const { return static_cast<int>(cinfo_.image_width); }
    int height() const { return static_cast<int>(cinfo_.image_height); }
    bool cmyk() const { return cmyk_; }
    // Adobe CMYK/YCCK JPEGs store inverted ink values; PIL undoes that.
    bool adobe() const { return cinfo_.saw_Adobe_marker != 0; }
    const char* error() const { return err_.message; }

private:
    jpeg_decompress_struct cinfo_{};
    JpegErrorManager err_{};
    bool created_ = false;
    bool cmyk_ = false;
};

// Pillow's CMYK -> RGB conversion (libImaging/Convert.c: cmyk2rgb).
static inline uint8_t muldiv255(int a, int b) {
    const int tmp = a * b + 128;
    return static_cast<uint8_t>(((tmp >> 8) + tmp) >> 8);
}

static void cmyk_to_rgb(const uint8_t* cmyk, uint8_t* rgb, size_t pixel_count, bool inverted) {
    for (size_t i = 0; i < pixel_count; ++i) {
        int c = cmyk[4 * i + 0];
        int m = cmyk[4 * i + 1];
        int y = cmyk[4 * i + 2];
        int k = cmyk[4 * i + 3];
        if (inverted) {
            c = 255 - c;
            m = 255 - m;
            y = 255 - y;
            k = 255 - k;
        }
        const int nk = 255 - k;
        rgb[3 * i + 0] = static_cast<uint8_t>(std::clamp(nk - muldiv255(c, nk), 0, 255));
        rgb[3 * i + 1] = static_cast<uint8_t>(std::clamp(nk - muldiv255(m, nk), 0, 255));
        rgb[3 * i + 2] = static_cast<uint8_t>(std::clamp(nk - muldiv255(y, nk), 0, 255));
    }
}

// ———————————————————————————————————————————————
// Bicubic resampling, ported from Pillow (libImaging/Resample.c) so resized
// pixels match PIL.Image.resize(..., Image.BICUBIC), which the HF image
// processors use: separable convolution with an antialiasing support that grows
// with the downscale factor, 22-bit fixed-point coefficients, horizontal pass
// first, and rounding to 8 bits between the passes.
// ———————————————————————————————————————————————
constexpr int kResamplePrecisionBits = 32 - 8 - 2;

static double bicubic_filter(double x) {
    constexpr double a = -0.5;
    if (x < 0.0) {
        x = -x;
    }
    if (x < 1.0) {
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    }
    if (x < 2.0) {
        return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    }
    return 0.0;
}

struct ResampleCoeffs {
    int ksize = 0;
    std::vector<int> bounds;   // per output index: first input index, tap count
    std::vector<int32_t> kk;   // ksize fixed-point taps per output index
};

static ResampleCoeffs precompute_coeffs(int in_size, int out_size) {
    constexpr double support_unscaled = 2.0;
    const double scale = static_cast<double>(in_size) / out_size;
    const double filterscale = scale < 1.0 ? 1.0 : scale;
    const double support = support_unscaled * filterscale;

    ResampleCoeffs c;
    c.ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
    c.bounds.resize(static_cast<size_t>(out_size) * 2);
    c.kk.assign(static_cast<size_t>(out_size) * c.ksize, 0);

    std::vector<double> k(c.ksize);
    for (int xx = 0; xx < out_size; ++xx) {
        const double center = (xx + 0.5) * scale;
        const double ss = 1.0 / filterscale;
        int xmin = static_cast<int>(center - support + 0.5);
        if (xmin < 0) {
            xmin = 0;
        }
        int xmax = static_cast<int>(center + support + 0.5);
        if (xmax > in_size) {
            xmax = in_size;
        }
        xmax -= xmin;

        double ww = 0.0;
        for (int x = 0; x < xmax; ++x) {
            const double w = bicubic_filter((x + xmin - center + 0.5) * ss);
            k[x] = w;
            ww += w;
        }
        int32_t* out = &c.kk[static_cast<size_t>(xx) * c.ksize];
        for (int x = 0; x < xmax; ++x) {
            const double w = ww != 0.0 ? k[x] / ww : k[x];
            const double scaled = w * (1 << kResamplePrecisionBits);
            out[x] = static_cast<int32_t>(w < 0 ? -0.5 + scaled : 0.5 + scaled);
        }
        c.bounds[static_cast<size_t>(xx) * 2] = xmin;
        c.bounds[static_cast<size_t>(xx) * 2 + 1] = xmax;
    }
    return c;
}

static inline uint8_t clip8(int32_t v) {
    v >>= kResamplePrecisionBits;
    return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
}

// RGB24 in, RGB24 out. Mirrors ImagingResampleInner for a full-image box.
static void resample_bicubic_rgb(const uint8_t* src, int in_w, int in_h,
                                 uint8_t* dst, int out_w, int out_h) {
    const bool need_horizontal = out_w != in_w;
    const bool need_vertical = out_h != in_h;
    const ResampleCoeffs horiz = precompute_coeffs(in_w, out_w);
    ResampleCoeffs vert = precompute_coeffs(in_h, out_h);

    // Rows of the source the vertical pass actually reads.
    const int ybox_first = vert.bounds[0];
    const int ybox_last = vert.bounds[static_cast<size_t>(out_h) * 2 - 2] + vert.bounds[static_cast<size_t>(out_h) * 2 - 1];

    std::vector<uint8_t> temp;
    const uint8_t* rows = src;
    int rows_w = in_w;

    if (need_horizontal) {
        for (int i = 0; i < out_h; ++i) {
            vert.bounds[static_cast<size_t>(i) * 2] -= ybox_first;
        }
        const int temp_h = ybox_last - ybox_first;
        uint8_t* out = dst;
        if (need_vertical) {
            temp.resize(static_cast<size_t>(out_w) * temp_h * 3);
            out = temp.data();
        }
        for (int yy = 0; yy < temp_h; ++yy) {
            const uint8_t* in_row = src + static_cast<size_t>(yy + ybox_first) * in_w * 3;
            uint8_t* out_row = out + static_cast<size_t>(yy) * out_w * 3;
            for (int xx = 0; xx < out_w; ++xx) {
                const int xmin = horiz.bounds[static_cast<size_t>(xx) * 2];
                const int xmax = horiz.bounds[static_cast<size_t>(xx) * 2 + 1];
                const int32_t* k = &horiz.kk[static_cast<size_t>(xx) * horiz.ksize];
                int32_t s0 = 1 << (kResamplePrecisionBits - 1);
                int32_t s1 = s0;
                int32_t s2 = s0;
                for (int x = 0; x < xmax; ++x) {
                    const uint8_t* p = in_row + static_cast<size_t>(x + xmin) * 3;
                    s0 += p[0] * k[x];
                    s1 += p[1] * k[x];
                    s2 += p[2] * k[x];
                }
                out_row[xx * 3 + 0] = clip8(s0);
                out_row[xx * 3 + 1] = clip8(s1);
                out_row[xx * 3 + 2] = clip8(s2);
            }
        }
        if (!need_vertical) {
            return;
        }
        rows = temp.data();
        rows_w = out_w;
    }

    if (need_vertical) {
        for (int yy = 0; yy < out_h; ++yy) {
            const int ymin = vert.bounds[static_cast<size_t>(yy) * 2];
            const int ymax = vert.bounds[static_cast<size_t>(yy) * 2 + 1];
            const int32_t* k = &vert.kk[static_cast<size_t>(yy) * vert.ksize];
            uint8_t* out_row = dst + static_cast<size_t>(yy) * out_w * 3;
            for (int xx = 0; xx < out_w * 3; ++xx) {
                int32_t s = 1 << (kResamplePrecisionBits - 1);
                for (int y = 0; y < ymax; ++y) {
                    s += rows[static_cast<size_t>(y + ymin) * rows_w * 3 + xx] * k[y];
                }
                out_row[xx] = clip8(s);
            }
        }
    } else {
        std::memcpy(dst, src, static_cast<size_t>(out_w) * out_h * 3);
    }
}
}

ImageMemoryPool::ImageMemoryPool(size_t max_cached_per_size)
    : max_cached_per_size_(max_cached_per_size) {
}

namespace {
static size_t round_up_to(size_t value, size_t alignment) {
    if (alignment == 0) {
        return value;
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

static size_t bucket_size_for_image(size_t requested_size) {
    if (requested_size <= 64 * 1024) {
        return round_up_to(requested_size, 4 * 1024);
    }
    if (requested_size <= 1024 * 1024) {
        return round_up_to(requested_size, 64 * 1024);
    }
    return round_up_to(requested_size, 1024 * 1024);
}
}

bytes ImageMemoryPool::acquire(size_t size) {
    if (size == 0) {
        return bytes();
    }

    const size_t bucket_size = bucket_size_for_image(size);

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = free_lists_.find(bucket_size);
    if (it != free_lists_.end() && !it->second.empty()) {
        bytes block = std::move(it->second.back());
        it->second.pop_back();
        memset(block.data(), 0, block.size());
        return block;
    }

    for (auto candidate = free_lists_.begin(); candidate != free_lists_.end(); ++candidate) {
        if (candidate->first >= bucket_size && !candidate->second.empty()) {
            bytes block = std::move(candidate->second.back());
            candidate->second.pop_back();
            memset(block.data(), 0, block.size());
            return block;
        }
    }

    return bytes(bucket_size);
}

void ImageMemoryPool::recycle(bytes&& block) {
    const size_t size = block.size();
    if (size == 0 || !block.is_owner()) {
        return;
    }

    const size_t bucket_size = bucket_size_for_image(size);

    std::lock_guard<std::mutex> lock(mutex_);
    auto& bucket = free_lists_[bucket_size];
    if (bucket.size() >= max_cached_per_size_) {
        return;
    }
    bucket.push_back(std::move(block));
}

ImageReader::ImageReader(size_t max_cached_per_size)
    : memory_pool_(max_cached_per_size) {
}

ImageReader::~ImageReader() = default;

bool ImageReader::decode_jpeg(const uint8_t* data, size_t size, image_data_t& out_image) {
    JpegDecoder decoder;
    if (!decoder.read_header(data, size)) {
        std::cerr << "Error: Could not read JPEG header: " << decoder.error() << std::endl;
        return false;
    }
    const int width = decoder.width();
    const int height = decoder.height();
    const size_t pixel_count = static_cast<size_t>(width) * height;
    if (width <= 0 || height <= 0 || pixel_count > kMaxImagePixels) {
        std::cerr << "Error: Unsupported JPEG dimensions " << width << "x" << height << std::endl;
        return false;
    }

    bytes buffer = memory_pool_.acquire(pixel_count * 3);
    if (buffer.size() == 0) {
        std::cerr << "Error: Could not allocate output image buffer" << std::endl;
        return false;
    }
    std::vector<uint8_t> cmyk_pixels(decoder.cmyk() ? pixel_count * 4 : 0);

    if (!decoder.read_pixels(buffer.data(), cmyk_pixels.data())) {
        std::cerr << "Error: Could not decode JPEG image: " << decoder.error() << std::endl;
        memory_pool_.recycle(std::move(buffer));
        return false;
    }
    if (decoder.cmyk()) {
        cmyk_to_rgb(cmyk_pixels.data(), buffer.data(), pixel_count, decoder.adobe());
    }

    recycle(out_image);
    out_image.width = width;
    out_image.height = height;
    out_image.pixels = std::move(buffer);
    return true;
}

bool ImageReader::decode_png(const uint8_t* data, size_t size, image_data_t& out_image) {
    // lodepng converts every PNG color type to 8-bit RGB the way PIL's
    // convert("RGB") does for the common cases: palette expanded, gray
    // replicated, alpha/tRNS dropped (not composited), 16-bit reduced to the
    // high byte. Gamma/ICC chunks are ignored, as PIL does by default.
    unsigned width = 0;
    unsigned height = 0;
    lodepng::State state;
    state.info_raw.colortype = LCT_RGB;
    state.info_raw.bitdepth = 8;
    if (lodepng_inspect(&width, &height, &state, data, size) != 0 || width == 0 || height == 0) {
        std::cerr << "Error: Could not read PNG header" << std::endl;
        return false;
    }
    if (static_cast<size_t>(width) * height > kMaxImagePixels) {
        std::cerr << "Error: Unsupported PNG dimensions " << width << "x" << height << std::endl;
        return false;
    }

    bytes buffer = memory_pool_.acquire(static_cast<size_t>(width) * height * 3);
    if (buffer.size() == 0) {
        std::cerr << "Error: Could not allocate output image buffer" << std::endl;
        return false;
    }

    unsigned char* decoded = nullptr;
    const unsigned err = lodepng_decode(&decoded, &width, &height, &state, data, size);
    if (err != 0 || !decoded) {
        std::cerr << "Error: Could not decode PNG image: " << lodepng_error_text(err) << std::endl;
        std::free(decoded);
        memory_pool_.recycle(std::move(buffer));
        return false;
    }
    std::memcpy(buffer.data(), decoded, static_cast<size_t>(width) * height * 3);
    std::free(decoded);

    recycle(out_image);
    out_image.width = static_cast<int>(width);
    out_image.height = static_cast<int>(height);
    out_image.pixels = std::move(buffer);
    return true;
}

bool ImageReader::decode_webp(const uint8_t* data, size_t size, image_data_t& out_image) {
    // libwebp, the library Pillow decodes with; RGB output drops alpha (not composited), as PIL's
    // convert("RGB") does. Animated files decode their first frame only through Pillow, and not here.
    int width = 0;
    int height = 0;
    if (!WebPGetInfo(data, size, &width, &height) || width <= 0 || height <= 0) {
        std::cerr << "Error: Could not read WebP header" << std::endl;
        return false;
    }
    const size_t pixel_count = static_cast<size_t>(width) * height;
    if (pixel_count > kMaxImagePixels) {
        std::cerr << "Error: Unsupported WebP dimensions " << width << "x" << height << std::endl;
        return false;
    }
    bytes buffer = memory_pool_.acquire(pixel_count * 3);
    if (buffer.size() == 0) {
        std::cerr << "Error: Could not allocate output image buffer" << std::endl;
        return false;
    }
    if (!WebPDecodeRGBInto(data, size, reinterpret_cast<uint8_t*>(buffer.data()), pixel_count * 3, width * 3)) {
        std::cerr << "Error: Could not decode WebP image" << std::endl;
        memory_pool_.recycle(std::move(buffer));
        return false;
    }
    recycle(out_image);
    out_image.width = width;
    out_image.height = height;
    out_image.pixels = std::move(buffer);
    return true;
}

bool ImageReader::decode_bytes(const uint8_t* data, size_t size, image_data_t& out_image) {
    const ImageCodec codec = detect_codec(data, size);
    if (codec == ImageCodec::None) {
        std::cerr << "Error: Unsupported image format (only JPEG/JPG, PNG and WebP supported)" << std::endl;
        return false;
    }

    if (!has_basic_image_integrity(data, size, codec)) {
        std::cerr << "Error: Image payload failed integrity check" << std::endl;
        return false;
    }

    if (codec == ImageCodec::Webp) return decode_webp(data, size, out_image);
    return codec == ImageCodec::Jpeg ? decode_jpeg(data, size, out_image)
                                     : decode_png(data, size, out_image);
}

bool ImageReader::load_image(const std::string& filename, image_data_t& out_image) {
    if (!std::filesystem::exists(filename)) {
        std::cerr << "Error: File does not exist: " << filename << std::endl;
        return false;
    }

    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << filename << std::endl;
        return false;
    }

    const std::streamsize file_size = file.tellg();
    if (file_size <= 0) {
        std::cerr << "Error: Invalid image file size" << std::endl;
        return false;
    }
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(static_cast<size_t>(file_size));
    if (!file.read(reinterpret_cast<char*>(data.data()), file_size)) {
        std::cerr << "Error: Could not read image file data" << std::endl;
        return false;
    }

    return decode_bytes(data.data(), data.size(), out_image);
}

bool ImageReader::load_image_base64(const std::string& base64_string, image_data_t& out_image) {
    std::string payload = base64_string;
    const std::size_t comma_pos = payload.find(',');
    if (comma_pos != std::string::npos) {
        payload = payload.substr(comma_pos + 1);
    }

    const std::string decoded = base64::from_base64(payload);
    if (decoded.size() < 8) {
        std::cerr << "Error: Invalid base64 image payload" << std::endl;
        return false;
    }

    return decode_bytes(reinterpret_cast<const uint8_t*>(decoded.data()), decoded.size(), out_image);
}

bool ImageReader::from_rgb8(const uint8_t* rgb, int height, int width, image_data_t& out_image) {
    if (!rgb || height <= 0 || width <= 0) {
        return false;
    }
    const size_t pixel_count = static_cast<size_t>(width) * height;
    if (pixel_count > kMaxImagePixels) {
        return false;
    }
    bytes buffer = memory_pool_.acquire(pixel_count * 3);
    if (buffer.size() == 0) {
        return false;
    }
    std::memcpy(buffer.data(), rgb, pixel_count * 3);
    recycle(out_image);
    out_image.width = width;
    out_image.height = height;
    out_image.pixels = std::move(buffer);
    return true;
}

bool ImageReader::resize_image(const image_data_t& input, int target_width, int target_height, image_data_t& output) {
    if (input.pixels.size() == 0 || input.width <= 0 || input.height <= 0 || target_width <= 0 || target_height <= 0) {
        return false;
    }
    if (input.pixels.size() < static_cast<size_t>(input.width) * input.height * 3) {
        return false;
    }

    const size_t out_size = static_cast<size_t>(target_width) * target_height * 3;
    bytes out_buffer = memory_pool_.acquire(out_size);
    if (out_buffer.size() == 0) {
        return false;
    }

    resample_bicubic_rgb(input.pixels.data(), input.width, input.height,
                         out_buffer.data(), target_width, target_height);

    recycle(output);
    output.width = target_width;
    output.height = target_height;
    output.pixels = std::move(out_buffer);
    return true;
}

bool ImageReader::reorder_hwc_to_chw(const image_data_t& input, image_data_t& output) {
    if (input.pixels.size() == 0 || input.width <= 0 || input.height <= 0) {
        return false;
    }

    const size_t pixel_count = static_cast<size_t>(input.width) * input.height;
    const size_t total_size = pixel_count * 3;
    if (input.pixels.size() < total_size) {
        return false;
    }

    bytes out_buffer = memory_pool_.acquire(total_size);
    if (out_buffer.size() == 0) {
        return false;
    }

    const uint8_t* src = input.pixels.data();
    uint8_t* dst = out_buffer.data();

    for (size_t i = 0; i < pixel_count; ++i) {
        const size_t hwc = i * 3;
        dst[i] = src[hwc];
        dst[pixel_count + i] = src[hwc + 1];
        dst[2 * pixel_count + i] = src[hwc + 2];
    }

    recycle(output);
    output.width = input.width;
    output.height = input.height;
    output.pixels = std::move(out_buffer);
    return true;
}

bool ImageReader::save_png(const std::string& filename, const image_data_t& image) {
    if (image.pixels.size() == 0 || image.width <= 0 || image.height <= 0) {
        return false;
    }

    std::vector<unsigned char> png;
    const unsigned err = lodepng::encode(png, image.pixels.data(),
                                         static_cast<unsigned>(image.width),
                                         static_cast<unsigned>(image.height), LCT_RGB, 8);
    if (err != 0) {
        std::cerr << "Error: Could not encode PNG: " << lodepng_error_text(err) << std::endl;
        return false;
    }

    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Error: Could not open output file: " << filename << std::endl;
        return false;
    }
    out.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
    return out.good();
}

void ImageReader::recycle(image_data_t& image) {
    if (image.pixels.size() > 0) {
        memory_pool_.recycle(std::move(image.pixels));
    }
    image.width = 0;
    image.height = 0;
}



bool save_image(const std::string& filename, const bytes& image) {
    try {
        // Calculate dimensions from image size (RGB24 = 3 bytes per pixel)
        if (image.size() % 3 != 0) {
            std::cerr << "Error: Invalid image data size (not divisible by 3)" << std::endl;
            return false;
        }
        
        int totalPixels = static_cast<int>(image.size() / 3);
        int width = static_cast<int>(sqrt(totalPixels));
        int height = totalPixels / width;
        
        if (width * height != totalPixels) {
            std::cerr << "Error: Image data does not represent a square image" << std::endl;
            return false;
        }

        // Write PPM format (Portable Pixmap)
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
            return false;
        }

        // Write PPM header: "P6\nwidth height\n255\n"
        file << "P6\n" << width << " " << height << "\n255\n";
        
        // Write RGB data
        file.write(reinterpret_cast<const char*>(image.data()), image.size());
        file.close();

        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error saving image: " << e.what() << std::endl;
        return false;
    }
}
