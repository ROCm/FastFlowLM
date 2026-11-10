// parse_image_raw: what is accepted, and that each way of getting it wrong is refused with a message, not a crash.
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include "image/raw_rgb8.hpp"

using json = nlohmann::json;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

static json part(int t, int h, int w, const std::string& bytes) {
	return {{"pixel_format", "RGB8"}, {"layout", "THWC"}, {"dtype", "uint8"},
	        {"shape", {t, h, w, 3}}, {"data", base64::to_base64(bytes)}};
}

static bool parse(const json& j, raw_rgb8_batch_t& out, std::string& err, size_t frames_so_far = 0, size_t bytes_so_far = 0,
                  raw_rgb8_limits_t limits = raw_rgb8_limits_t()) {
	err.clear();
	return parse_image_raw(j, limits, frames_so_far, bytes_so_far, out, err);
}

static void refused(const char* name, const json& j, const char* needle, size_t frames_so_far = 0, size_t bytes_so_far = 0,
                    raw_rgb8_limits_t limits = raw_rgb8_limits_t()) {
	raw_rgb8_batch_t b;
	std::string err;
	const bool ok = parse(j, b, err, frames_so_far, bytes_so_far, limits);
	if (ok || err.find(needle) == std::string::npos) {
		std::printf("FAIL refused(%s): ok=%d err='%s' (wanted '%s')\n", name, (int)ok, err.c_str(), needle);
		++failures;
	}
}

int main() {
	// the happy path: 2 frames of 4x3, pixels counted so the frames can be told apart
	std::string bytes(2 * 4 * 3 * 3, 0);
	for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = (char)i;
	raw_rgb8_batch_t b;
	std::string err;
	CHECK(parse(part(2, 4, 3, bytes), b, err));
	CHECK(b.frames == 2 && b.height == 4 && b.width == 3);
	CHECK(b.frame_bytes() == 36);
	CHECK(b.data == bytes);
	CHECK(b.frame(1)[0] == 36 && b.frame(1)[35] == 71);

	// padding of 0, 1 and 2 bytes: 1 frame of 1x1 is 3 bytes (no padding), 1x2 is 6, 2x?: use odd byte counts via 1x1xN
	for (int w : {1, 2, 3, 4, 5}) {
		std::string px(3 * w, 7);
		raw_rgb8_batch_t c;
		CHECK(parse(part(1, 1, w, px), c, err));
		CHECK(c.data == px);
	}

	// max_slice_nums: 1 unless given, 1 to 9
	CHECK(b.max_slice_nums == 1);
	{ json j = part(2, 4, 3, bytes); j["max_slice_nums"] = 9; raw_rgb8_batch_t c; CHECK(parse(j, c, err)); CHECK(c.max_slice_nums == 9); }
	for (const json& bad : {json(0), json(10), json(2.5), json("1")}) {
		json j = part(2, 4, 3, bytes); j["max_slice_nums"] = bad; refused("max_slice_nums", j, "max_slice_nums");
	}

	// format, layout, dtype exactly
	for (const char* key : {"pixel_format", "layout", "dtype"}) {
		json j = part(2, 4, 3, bytes);
		j[key] = "other";
		refused(key, j, key);
		j.erase(key);
		refused(key, j, key);
	}
	// shape
	{ json j = part(2, 4, 3, bytes); j["shape"] = {4, 3, 3}; refused("3-dim", j, "shape"); }
	{ json j = part(2, 4, 3, bytes); j["shape"] = {2, 4, 3, 4}; refused("channels", j, "must be 3"); }
	{ json j = part(2, 4, 3, bytes); j["shape"] = {0, 4, 3, 3}; refused("zero", j, "positive"); }
	{ json j = part(2, 4, 3, bytes); j["shape"] = {2, -4, 3, 3}; refused("negative", j, "positive"); }
	{ json j = part(2, 4, 3, bytes); j["shape"] = {2.5, 4, 3, 3}; refused("float", j, "positive"); }
	{ json j = part(2, 4, 3, bytes); j["shape"] = {2, 4, "3", 3}; refused("string", j, "positive"); }
	{ json j = part(2, 4, 3, bytes); j["shape"] = {1ll << 40, 4, 3, 3}; refused("too big dim", j, "positive"); }
	{ json j = part(2, 4, 3, bytes); j["shape"] = {3, 4, 3, 3}; refused("short data", j, "needs 108"); }
	{ json j = part(2, 4, 3, bytes); j["shape"] = {1, 4, 3, 3}; refused("long data", j, "needs 36"); }
	{ json j = part(2, 4, 3, bytes); j.erase("shape"); refused("no shape", j, "shape"); }
	// products that would overflow 64 bits if multiplied naively must be refused by the limits, not wrap
	{ json j = part(2, 4, 3, bytes); j["shape"] = {1 << 24, 1 << 24, 1 << 24, 3}; refused("overflow", j, "128 frames"); }
	{ json j = part(2, 4, 3, bytes); j["shape"] = {1, 1 << 24, 1 << 24, 3}; refused("overflow 2", j, "larger than"); }
	// data
	{ json j = part(2, 4, 3, bytes); j["data"] = 5; refused("data int", j, "base64"); }
	{ json j = part(2, 4, 3, bytes); j["data"] = "abc"; refused("data len%4", j, "base64"); }
	{ json j = part(2, 4, 3, bytes); j["data"] = ""; refused("data empty", j, "holds 0"); }
	{ json j = part(2, 4, 3, bytes); j.erase("data"); refused("no data", j, "base64"); }
	refused("not object", json("x"), "object");
	refused("null", json(), "object");

	// limits, over the whole request: frames and bytes
	raw_rgb8_limits_t lim;
	lim.max_frames = 3;
	lim.max_bytes = 100;
	CHECK(parse(part(2, 4, 3, bytes), b, err, 0, 0, lim));
	refused("frames over", part(2, 4, 3, bytes), "3 frames", 2, 0, lim);
	CHECK(parse(part(2, 4, 3, bytes), b, err, 1, 0, lim));
	refused("bytes over", part(2, 4, 3, bytes), "100 bytes", 0, 30, lim);
	CHECK(parse(part(2, 4, 3, bytes), b, err, 0, 28, lim));   // 72 + 28 == 100: at the limit is fine
	lim.max_bytes = 71;
	refused("one byte over", part(2, 4, 3, bytes), "bytes", 0, 0, lim);

	// file input: off unless a directory is allowed, confined to it, exactly the size 'shape' says
	namespace fs = std::filesystem;
	const fs::path base = fs::temp_directory_path() / ("raw_rgb8_test_" + std::to_string((long long)std::hash<std::string>()(__FILE__)));
	fs::remove_all(base);
	fs::create_directories(base / "ok" / "sub");
	fs::create_directories(base / "out");
	auto write = [](const fs::path& f, const std::string& s) { std::ofstream(f, std::ios::binary).write(s.data(), (std::streamsize)s.size()); };
	write(base / "ok" / "f.rgb", bytes);
	write(base / "ok" / "sub" / "g.rgb", bytes);
	write(base / "out" / "secret.rgb", bytes);
	write(base / "ok" / "short.rgb", bytes.substr(1));
	std::error_code ec;
	fs::create_symlink(base / "out" / "secret.rgb", base / "ok" / "link.rgb", ec);
	const bool have_symlink = !ec;
	auto pathpart = [&](const fs::path& f) {
		json j = part(2, 4, 3, bytes); j.erase("data"); j["path"] = f.string(); return j;
	};
	raw_rgb8_limits_t fl;
	fl.allow_path_dir = (base / "ok").string();
	{ raw_rgb8_batch_t c; CHECK(parse(pathpart(base / "ok" / "f.rgb"), c, err, 0, 0, fl)); CHECK(c.data == bytes && c.frames == 2); }
	{ raw_rgb8_batch_t c; CHECK(parse(pathpart(base / "ok" / "sub" / "g.rgb"), c, err, 0, 0, fl)); CHECK(c.data == bytes); }
	{ raw_rgb8_batch_t c; CHECK(parse(pathpart(base / "ok" / "sub" / ".." / "f.rgb"), c, err, 0, 0, fl)); }
	refused("path off", pathpart(base / "ok" / "f.rgb"), "--allow-raw-path");
	refused("outside", pathpart(base / "out" / "secret.rgb"), "inside the allowed", 0, 0, fl);
	refused("traversal", pathpart(base / "ok" / ".." / "out" / "secret.rgb"), "inside the allowed", 0, 0, fl);
	if (have_symlink) refused("symlink out", pathpart(base / "ok" / "link.rgb"), "inside the allowed", 0, 0, fl);
	refused("missing", pathpart(base / "ok" / "nope.rgb"), "inside the allowed", 0, 0, fl);
	refused("directory", pathpart(base / "ok" / "sub"), "inside the allowed", 0, 0, fl);
	refused("sibling prefix", pathpart(base / "ok2"), "inside the allowed", 0, 0, fl);
	refused("short file", pathpart(base / "ok" / "short.rgb"), "needs 72", 0, 0, fl);
	{ json j = pathpart(base / "ok" / "f.rgb"); j["data"] = base64::to_base64(bytes); refused("both", j, "exactly one", 0, 0, fl); }
	{ json j = pathpart(base / "ok" / "f.rgb"); j.erase("path"); refused("neither", j, "exactly one", 0, 0, fl); }
	{ json j = pathpart(base / "ok" / "f.rgb"); j["path"] = 5; refused("path not string", j, "'path' must be a string", 0, 0, fl); }
	{ raw_rgb8_limits_t bad = fl; bad.allow_path_dir = (base / "no_such_dir").string(); refused("bad allowed dir", pathpart(base / "ok" / "f.rgb"), "inside the allowed", 0, 0, bad); }
	fs::remove_all(base);

	if (failures) { std::printf("%d FAILED\n", failures); return 1; }
	std::printf("all raw_rgb8 tests passed\n");
	return 0;
}
