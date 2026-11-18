// parse_image_raw: what is accepted, and that each way of getting it wrong is refused with a message, not a crash.
// Also json_log, which keeps such requests (megabytes of frame data) out of the server log.
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include "image/raw_rgb8.hpp"
#include "utils/json_log.hpp"
#ifndef _WIN32
#include <sys/stat.h>
#endif

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
	// links that stay inside are fine; a linked directory that leads out is not
	fs::create_symlink(base / "ok" / "f.rgb", base / "ok" / "inlink.rgb", ec);
	if (!ec) { raw_rgb8_batch_t c; CHECK(parse(pathpart(base / "ok" / "inlink.rgb"), c, err, 0, 0, fl)); CHECK(c.data == bytes); }
	fs::create_directory_symlink(base / "out", base / "ok" / "outdir", ec);
	if (!ec) refused("linked dir out", pathpart(base / "ok" / "outdir" / "secret.rgb"), "inside the allowed", 0, 0, fl);
	// the race: the name checked out, then a directory on it was swapped for a link that leads out before the open
	{
		fs::create_directories(base / "ok" / "swap");
		write(base / "ok" / "swap" / "secret.rgb", bytes);
		const fs::path root = fs::canonical(base / "ok");
		const fs::path checked = fs::canonical(base / "ok" / "swap" / "secret.rgb");
		fs::remove_all(base / "ok" / "swap");
		fs::create_directory_symlink(base / "out", base / "ok" / "swap", ec);
		if (!ec) {
			std::string got;
			unsigned long long size = 0;
			bool wrong_size = false;
			CHECK(!raw_rgb8_detail::read_checked(root, checked, bytes.size(), got, size, wrong_size));
			CHECK(!wrong_size);
		}
	}
	// a relative path is read from the server's working directory, as before
	{
		const fs::path cwd = fs::current_path();
		fs::current_path(base);
		raw_rgb8_batch_t c;
		CHECK(parse(pathpart(fs::path("ok") / "sub" / "g.rgb"), c, err, 0, 0, fl));
		refused("relative out", pathpart(fs::path("out") / "secret.rgb"), "inside the allowed", 0, 0, fl);
		fs::current_path(cwd);
	}
#ifndef _WIN32
	// a FIFO is refused at once: opening it must not wait for a writer
	if (mkfifo((base / "ok" / "fifo.rgb").c_str(), 0600) == 0) {
		refused("fifo", pathpart(base / "ok" / "fifo.rgb"), "inside the allowed", 0, 0, fl);
	}
#else
	// network and device paths are refused before anything is opened: no connection to a host the client names
	for (const char* p : {R"(\\no-such-host.invalid\share\f.rgb)", "//no-such-host.invalid/share/f.rgb",
	                      R"(\\?\UNC\no-such-host.invalid\share\f.rgb)", R"(\\.\PhysicalDrive0)",
	                      R"(\\?\Volume{00000000-0000-0000-0000-000000000000}\f.rgb)", R"(\??\C:\f.rgb)"}) {
		const auto t0 = std::chrono::steady_clock::now();
		refused(p, pathpart(p), "local file path", 0, 0, fl);
		CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(1));
	}
	// the long form of a drive path is local
	{
		raw_rgb8_batch_t c;
		CHECK(parse(pathpart(R"(\\?\)" + fs::canonical(base / "ok" / "f.rgb").string()), c, err, 0, 0, fl));
		CHECK(c.data == bytes);
	}
#endif
	fs::remove_all(base);

	// json_log: a 120-frame request is logged in a few hundred bytes, as valid UTF-8, with the short fields intact
	{
		const std::string frames((size_t)120 * 64 * 64 * 3, 'Z');
		json body = {{"model", "minicpm-v-4.7:1b"},
		             {"messages", {{{"role", "user"},
		                            {"content", {{{"type", "text"}, {"text", "describe"}},
		                                         {{"type", "image_raw"}, {"image_raw", part(120, 64, 64, frames)}}}}}}}};
		const std::string text = body.dump();
		const std::string logged = json_log::dump(json_log::parse(text));
		CHECK(text.size() > 1000000);
		CHECK(logged.size() < 2000);
		CHECK(logged.find("describe") != std::string::npos);
		CHECK(logged.find("minicpm-v-4.7:1b") != std::string::npos);
		CHECK(logged.find("bytes)...") != std::string::npos);
		CHECK(json::parse(logged)["messages"][0]["content"][1]["image_raw"]["shape"] == json({120, 64, 64, 3}));
		// cut on character boundaries: 300 two-byte and 300 four-byte characters stay valid UTF-8
		for (const char* ch : {"\xc3\xa9", "\xf0\x9f\x98\x80"}) {
			std::string s;
			for (int i = 0; i < 300; ++i) s += ch;
			for (size_t head : {0, 1, 31, 32, 33}) {
				const std::string c = json_log::cut(s, head, 15);
				CHECK(c.size() < 200);
				CHECK(json::parse(json(c).dump()) == c);   // dump throws on broken UTF-8
			}
		}
		CHECK(json_log::cut("short") == "short");
		CHECK(!json_log::dump(json(std::string("\xff\xfe"))).empty());   // never throws on broken UTF-8
		bool threw = false;
		try { json_log::parse("{\"a\": "); } catch (const std::exception&) { threw = true; }
		CHECK(threw);
		// a truncated body's parse error quotes everything after the last opening quote; json_log::what keeps the start
		threw = false;
		try { json::parse(text.substr(0, 100000)); }
		catch (const std::exception& e) {
			threw = true;
			const std::string w = json_log::what(e);
			CHECK(std::string(e.what()).size() > 90000);
			CHECK(w.size() < 400);
			CHECK(w.rfind("[json.exception.parse_error.101] parse error at line 1, column 100001", 0) == 0);
		}
		CHECK(threw);
	}

	if (failures) { std::printf("%d FAILED\n", failures); return 1; }
	std::printf("all raw_rgb8 tests passed\n");
	return 0;
}
