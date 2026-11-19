/// \file oga_image_bytes.hpp
/// \brief the encoded image files an OGA package processor decodes itself
#pragma once

#include "base64.hpp"
#include "AutoModel/automodel.hpp"

#include <fstream>
#include <iterator>
#include <string>

namespace flm::oga {

/// \brief the file bytes of a base64 image, with or without a data-URL prefix
inline std::string ImageBytesFromBase64(std::string payload) {
    const auto marker = payload.find("base64,");
    if (marker != std::string::npos) payload.erase(0, marker + 7);
    std::string bytes = base64::from_base64(payload);
    if (bytes.size() < 8)
        throw ModelRequestError(400, false, "image payload is not a valid encoded image", "invalid_request_error");
    return bytes;
}

/// \brief the bytes of an image file given on the command line
inline std::string ImageBytesFromFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw ModelRequestError(400, false, "cannot read image " + path, "invalid_request_error");
    std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (bytes.size() < 8)
        throw ModelRequestError(400, false, "image file is empty: " + path, "invalid_request_error");
    return bytes;
}

}  // namespace flm::oga
