/// \file image_reader.hpp
/// \brief image_reader class
/// \author FastFlowLM Team
/// \date 2025-08-16
/// \version 0.9.24
/// \note This is a header file for the image_reader functions
#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <mutex>
#include "typedef.hpp"
#include "buffer.hpp"

struct image_data_t {
	bytes pixels;
	int height = 0;
	int width = 0;
};

class ImageMemoryPool {
public:
	explicit ImageMemoryPool(size_t max_cached_per_size = 16);
	bytes acquire(size_t size);
	void recycle(bytes&& block);

private:
	size_t max_cached_per_size_;
	std::unordered_map<size_t, std::vector<bytes>> free_lists_;
	std::mutex mutex_;
};

class ImageReader {
public:
	explicit ImageReader(size_t max_cached_per_size = 16);
	~ImageReader();

	bool load_image(const std::string& filename, image_data_t& out_image);
	bool load_image_base64(const std::string& base64_string, image_data_t& out_image);
	/// decode an encoded PNG, JPEG or WebP file held in memory to 8-bit RGB
	bool load_image_bytes(const std::string& data, image_data_t& out_image) {
		return decode_bytes(reinterpret_cast<const uint8_t*>(data.data()), data.size(), out_image);
	}
	/// copy 8-bit RGB pixels (HWC, 3 bytes per pixel, tightly packed) into `out_image`, as the decoders would leave them
	bool from_rgb8(const uint8_t* rgb, int height, int width, image_data_t& out_image);
	bool resize_image(const image_data_t& input, int target_width, int target_height, image_data_t& output);
	bool reorder_hwc_to_chw(const image_data_t& input, image_data_t& output);
	bool save_png(const std::string& filename, const image_data_t& image);

	void recycle(image_data_t& image);

private:
	bool decode_bytes(const uint8_t* data, size_t size, image_data_t& out_image);
	bool decode_jpeg(const uint8_t* data, size_t size, image_data_t& out_image);
	bool decode_png(const uint8_t* data, size_t size, image_data_t& out_image);
	bool decode_webp(const uint8_t* data, size_t size, image_data_t& out_image);

	ImageMemoryPool memory_pool_;
};

bool save_image(const std::string& filename, const bytes& image);

