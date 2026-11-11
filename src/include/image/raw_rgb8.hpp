/// \file raw_rgb8.hpp
/// \brief decoded 8-bit RGB frames sent straight to a vision model, bypassing PNG/JPEG decoding
#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <memory>
#include <string>
#include <string_view>
#include <nlohmann/json.hpp>
#include "base64.hpp"
#ifdef _WIN32
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#if __has_include(<linux/openat2.h>)
#include <linux/openat2.h>
#endif
#endif

/// One request part: `frames` pictures of `height` x `width`, packed THWC, 3 bytes (R, G, B) per pixel.
/// The model reads frame i from data + i * frame_bytes(); every frame is then an independent picture.
struct raw_rgb8_batch_t {
	std::string data;
	int frames = 0;
	int height = 0;
	int width = 0;
	/// most slices a frame may be cut into (MiniCPM-V's `max_slice_nums`); 1 keeps every frame a single view, as the
	/// model's own video inference does, which is what keeps many frames inside the token budget
	int max_slice_nums = 1;

	size_t frame_bytes() const { return (size_t)height * (size_t)width * 3; }
	const uint8_t* frame(int index) const {
		return reinterpret_cast<const uint8_t*>(data.data()) + (size_t)index * frame_bytes();
	}
};

/// What one request may carry in raw frames, over all its parts.
struct raw_rgb8_limits_t {
	size_t max_frames = 128;
	size_t max_bytes = 128ull * 1024 * 1024;
	/// directory a frame file may be read from (`image_raw.path`); empty: file input is off
	std::string allow_path_dir;
};

namespace raw_rgb8_detail {

/// false for a path the server must not even open: on Windows a UNC or device path ("\\server\share\..",
/// "\\?\UNC\..", "\\.\.."), which would make the server connect to a host the client names (SMB, with the
/// service account's credentials) before any check could refuse it. Drive and relative paths are local.
inline bool is_local_path(const std::string& path) {
#ifdef _WIN32
	std::string p = path;
	for (char& c : p) if (c == '/') c = '\\';
	if (p.rfind("\\\\?\\", 0) == 0) {
		// "\\?\C:\x" names a drive; "\\?\UNC\server\.." and "\\?\Volume{..}" do not
		return p.size() >= 7 && std::isalpha((unsigned char)p[4]) && p[5] == ':' && p[6] == '\\';
	}
	return p.rfind("\\\\", 0) != 0 && p.rfind("\\??\\", 0) != 0;
#else
	(void)path;
	return true;
#endif
}

/// true if `file` is `root` or below it (both already resolved)
inline bool is_inside(const std::filesystem::path& file, const std::filesystem::path& root) {
	const std::filesystem::path rel = file.lexically_relative(root);
	return !rel.empty() && *rel.begin() != "..";
}

#ifdef _WIN32
/// Reads exactly `expected` bytes of the regular disk file `file` into `out`, checking the opened handle itself:
/// a path component swapped for a junction between the name check and the open cannot make us read elsewhere.
inline bool read_checked(const std::filesystem::path& root, const std::filesystem::path& file, unsigned long long expected,
                         std::string& out, unsigned long long& size, bool& wrong_size) {
	HANDLE h = CreateFileW(file.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
	                       nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
	if (h == INVALID_HANDLE_VALUE) return false;
	std::unique_ptr<void, decltype(&CloseHandle)> guard(h, &CloseHandle);
	BY_HANDLE_FILE_INFORMATION info;
	if (GetFileType(h) != FILE_TYPE_DISK || !GetFileInformationByHandle(h, &info) ||
	    (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
		return false;
	}
	std::wstring final_path(MAX_PATH, L'\0');
	for (;;) {
		const DWORD n = GetFinalPathNameByHandleW(h, final_path.data(), (DWORD)final_path.size(),
		                                          FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
		if (n == 0) return false;
		if (n < final_path.size()) { final_path.resize(n); break; }
		final_path.resize(n + 1);
	}
	if (final_path.rfind(L"\\\\?\\", 0) == 0) final_path = final_path.substr(4);
	if (final_path.rfind(L"UNC\\", 0) == 0 || !is_inside(std::filesystem::path(final_path), root)) return false;
	LARGE_INTEGER li;
	if (!GetFileSizeEx(h, &li)) return false;
	size = (unsigned long long)li.QuadPart;
	if (size != expected) { wrong_size = true; return false; }
	out.resize((size_t)expected);
	for (unsigned long long done = 0; done < expected;) {
		const DWORD want = (DWORD)(std::min<unsigned long long>)(expected - done, 1u << 30);
		DWORD got = 0;
		if (!ReadFile(h, out.data() + done, want, &got, nullptr) || got == 0) return false;  // truncated meanwhile
		done += got;
	}
	return true;
}
#else
/// As above, on Linux: the file is opened beneath the allowed directory (openat2 RESOLVE_BENEATH, so no symlink or
/// ".." swapped in meanwhile can lead out of it; O_NOFOLLOW and a check of the opened file's own path where openat2
/// is missing), and what was opened is checked and read through the same descriptor.
inline bool read_checked(const std::filesystem::path& root, const std::filesystem::path& file, unsigned long long expected,
                         std::string& out, unsigned long long& size, bool& wrong_size) {
	const int dir = open(root.c_str(), O_PATH | O_DIRECTORY | O_CLOEXEC);
	if (dir < 0) return false;
	const std::string rel = file.lexically_relative(root).string();
	const int flags = O_RDONLY | O_CLOEXEC | O_NOCTTY | O_NONBLOCK;
	int fd = -1;
	bool beneath = false;
#if defined(SYS_openat2) && defined(RESOLVE_BENEATH)
	open_how how{};
	how.flags = (unsigned long long)flags;
	how.resolve = RESOLVE_BENEATH | RESOLVE_NO_MAGICLINKS;
	fd = (int)syscall(SYS_openat2, dir, rel.c_str(), &how, sizeof(how));
	beneath = fd >= 0;
	const bool no_openat2 = fd < 0 && (errno == ENOSYS || errno == EPERM);  // old kernel, or a seccomp filter
#else
	const bool no_openat2 = true;
#endif
	if (fd < 0 && no_openat2) fd = openat(dir, rel.c_str(), flags | O_NOFOLLOW);
	close(dir);
	if (fd < 0) return false;
	std::unique_ptr<int, void (*)(int*)> guard(&fd, [](int* f) { close(*f); });
	struct stat st;
	if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) return false;
	if (!beneath) {
		std::error_code ec;
		const auto opened = std::filesystem::read_symlink("/proc/self/fd/" + std::to_string(fd), ec);
		if (ec || !is_inside(opened, root)) return false;
	}
	size = (unsigned long long)st.st_size;
	if (size != expected) { wrong_size = true; return false; }
	out.resize((size_t)expected);
	for (unsigned long long done = 0; done < expected;) {
		const ssize_t got = read(fd, out.data() + done, (size_t)(std::min<unsigned long long>)(expected - done, 1u << 30));
		if (got < 0 && errno == EINTR) continue;
		if (got <= 0) return false;  // truncated meanwhile
		done += (unsigned long long)got;
	}
	return true;
}
#endif

}  // namespace raw_rgb8_detail

/// Read `path` into `out` if, once symlinks and ".." are resolved, it is a regular file inside `allowed_dir` and is
/// exactly `expected` bytes. The same message answers a missing file and one outside the directory, so a client cannot
/// probe the rest of the disk. The name is checked first, then the opened file itself, which is what is read: the
/// file is read, not mapped (a file truncated while a request runs must not crash us), and nothing swapped in
/// between the two checks can make us read a file outside the directory. UNC and device paths are refused before
/// anything is opened.
inline bool read_raw_file(const std::string& allowed_dir, const std::string& path, unsigned long long expected,
                          std::string& out, std::string& error) {
	namespace fs = std::filesystem;
	if (allowed_dir.empty()) {
		error = "image_raw: 'path' input is off; start the server with --allow-raw-path <directory>";
		return false;
	}
	const std::string unreadable = "image_raw: 'path' is not a readable file inside the allowed directory";
	if (!raw_rgb8_detail::is_local_path(path)) {
		error = "image_raw: 'path' must be a local file path (not a network or device path) inside the allowed directory";
		return false;
	}
	std::error_code ec;
	const fs::path root = fs::canonical(fs::path(allowed_dir), ec);
	if (ec) { error = unreadable; return false; }
	const fs::path file = fs::canonical(fs::path(path), ec);
	if (ec || !raw_rgb8_detail::is_inside(file, root)) { error = unreadable; return false; }
	unsigned long long size = 0;
	bool wrong_size = false;
	if (!raw_rgb8_detail::read_checked(root, file, expected, out, size, wrong_size)) {
		out.clear();
		error = wrong_size ? "image_raw: the file holds " + std::to_string(size) + " bytes but 'shape' needs " +
		                         std::to_string(expected)
		                   : unreadable;
		return false;
	}
	return true;
}

/// Parse the `image_raw` object of a chat content part:
///   {"pixel_format":"RGB8","layout":"THWC","dtype":"uint8","shape":[T,H,W,3],"data":"<base64>"}
/// Everything is checked, nothing is converted: the format, layout and dtype must be exactly these, the shape must
/// match the byte count, and the totals must fit `limits` once `frames_so_far` / `bytes_so_far` (earlier parts of the
/// same request) are counted. On failure `error` says what is wrong and `out` is unusable.
inline bool parse_image_raw(const nlohmann::json& raw, const raw_rgb8_limits_t& limits,
                            size_t frames_so_far, size_t bytes_so_far,
                            raw_rgb8_batch_t& out, std::string& error) {
	auto fail = [&](const std::string& what) { error = "image_raw: " + what; return false; };
	if (!raw.is_object()) return fail("must be an object");
	auto text = [&](const char* key, const char* expected, std::string& why) {
		if (!raw.contains(key) || !raw.at(key).is_string() || raw.at(key).get<std::string>() != expected) {
			why = std::string("'") + key + "' must be \"" + expected + "\"";
			return false;
		}
		return true;
	};
	std::string why;
	if (!text("pixel_format", "RGB8", why) || !text("layout", "THWC", why) || !text("dtype", "uint8", why)) return fail(why);

	if (!raw.contains("shape") || !raw.at("shape").is_array() || raw.at("shape").size() != 4) {
		return fail("'shape' must be [T, H, W, 3]");
	}
	long long dims[4];
	for (int i = 0; i < 4; ++i) {
		const auto& d = raw.at("shape")[i];
		if (!d.is_number_integer() || d.get<long long>() <= 0 || d.get<long long>() > (1ll << 24)) {
			return fail("'shape' must hold positive integers");
		}
		dims[i] = d.get<long long>();
	}
	if (dims[3] != 3) return fail("the last dimension of 'shape' must be 3");

	int max_slice_nums = 1;
	if (raw.contains("max_slice_nums")) {
		const auto& m = raw.at("max_slice_nums");
		if (!m.is_number_integer() || m.get<long long>() < 1 || m.get<long long>() > 9) {
			return fail("'max_slice_nums' must be an integer from 1 to 9");
		}
		max_slice_nums = (int)m.get<long long>();
	}

	const bool has_data = raw.contains("data"), has_path = raw.contains("path");
	if (has_data == has_path) return fail("give exactly one of 'data' (base64) or 'path' (a file)");
	if (has_path && !raw.at("path").is_string()) return fail("'path' must be a string");
	if (has_data && !raw.at("data").is_string()) return fail("'data' must be a base64 string");
	static const std::string no_text;
	const std::string& b64 = has_data ? raw.at("data").get_ref<const std::string&>() : no_text;

	// unsigned long long throughout: each factor is below 2^24, the product of the first three can reach 2^72 and is
	// checked against the limits before it is multiplied by 3
	const unsigned long long frames = (unsigned long long)dims[0];
	if (frames + frames_so_far > limits.max_frames) {
		return fail("more than " + std::to_string(limits.max_frames) + " frames in one request");
	}
	const unsigned long long frame_pixels = (unsigned long long)dims[1] * (unsigned long long)dims[2];
	if (frame_pixels > limits.max_bytes / 3) return fail("a frame is larger than the request limit");
	const unsigned long long frame_bytes = frame_pixels * 3;
	if (frames > limits.max_bytes / frame_bytes
        || frames * frame_bytes + bytes_so_far > limits.max_bytes) {
		return fail("more than " + std::to_string(limits.max_bytes) + " bytes of raw frames in one request");
	}
	const unsigned long long total = frames * frame_bytes;

	out.max_slice_nums = max_slice_nums;
	if (has_path) {
		std::string file_error;
		if (!read_raw_file(limits.allow_path_dir, raw.at("path").get<std::string>(), total, out.data, file_error)) {
			error = file_error;
			return false;
		}
		out.frames = (int)dims[0];
		out.height = (int)dims[1];
		out.width = (int)dims[2];
		return true;
	}

	// the decoded size follows from the text alone, so a wrong length is refused before anything is allocated
	if ((b64.size() & 3) != 0) return fail("'data' is not valid base64");
	size_t padding = 0;
	for (size_t i = b64.size(); i > 0 && i + 2 >= b64.size() && b64[i - 1] == '='; --i) ++padding;
	if ((unsigned long long)(b64.size() / 4 * 3 - padding) != total) {
		return fail("'data' holds " + std::to_string(b64.size() / 4 * 3 - padding) + " bytes but 'shape' needs " +
		            std::to_string(total));
	}
	try {
		out.data = base64::from_base64(b64);
	}
	catch (const std::exception&) {
		return fail("'data' is not valid base64");
	}
	if (out.data.size() != total) return fail("'data' is not valid base64");
	out.frames = (int)dims[0];
	out.height = (int)dims[1];
	out.width = (int)dims[2];
	out.max_slice_nums = max_slice_nums;
	return true;
}
