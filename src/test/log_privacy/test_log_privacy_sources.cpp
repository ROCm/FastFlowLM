/// \file test_log_privacy_sources.cpp
/// \brief WP-15 source checks: every "Model RAW Output" dump goes through AutoModel::print_raw_output (so the
///        content switch covers every model), log_raw_output starts from that switch, and no download turns
///        TLS verification off in code. Reads the source tree; needs no other header of the project.
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

namespace {
int failures = 0;
#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) {                                                              \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);           \
            failures++;                                                             \
        }                                                                           \
    } while (0)

std::string read(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}

size_t count(const std::string& text, const std::string& part) {
    size_t n = 0;
    for (size_t at = text.find(part); at != std::string::npos; at = text.find(part, at + part.size())) ++n;
    return n;
}

const fs::path kSrc = FLM_SOURCE_DIR;

void TestOneRawOutputHelper() {
    size_t prints = 0;
    for (const fs::path dir : {kSrc / "common", kSrc / "include", kSrc / "server", kSrc / "runner"}) {
        for (const auto& entry : fs::recursive_directory_iterator(dir)) {
            const auto ext = entry.path().extension();
            if (ext != ".cpp" && ext != ".hpp" && ext != ".h") continue;
            // code lines only: comments may name the marker
            std::string text;
            std::istringstream lines(read(entry.path()));
            for (std::string line; std::getline(lines, line);) {
                const size_t first = line.find_first_not_of(" \t");
                if (first != std::string::npos && (line.compare(first, 2, "//") == 0 || line[first] == '*')) continue;
                text += line + "\n";
            }
            const size_t n = count(text, "\"Model RAW Output:");
            if (n == 0) continue;
            prints += n;
            const bool helper = entry.path().filename() == "automodel.cpp";
            if (!helper) std::printf("  \"Model RAW Output:\" printed outside the helper: %s (%zu)\n",
                                     entry.path().string().c_str(), n);
            CHECK(helper);
        }
    }
    CHECK(prints == 1);
    const std::string automodel = read(kSrc / "common" / "AutoModel" / "automodel.cpp");
    CHECK(std::regex_search(automodel, std::regex(
        R"(AutoModel::print_raw_output\(const std::string& result\) const \{\s*if \(!this->log_raw_output\) return;)")));
    // the default comes from the switch, not a literal true
    const std::string header = read(kSrc / "include" / "AutoModel" / "automodel.hpp");
    CHECK(std::regex_search(header, std::regex(R"(bool log_raw_output = flm::log_privacy::raw_output_enabled\(\);)")));
    CHECK(!std::regex_search(header, std::regex(R"(bool log_raw_output = true)")));
}

void TestNoInsecureCurlOptions() {
    const std::regex off(R"(CURLOPT_SSL_VERIFY(PEER|HOST)\s*,\s*0)");
    for (const auto& entry : fs::recursive_directory_iterator(kSrc / "pull")) {
        if (entry.path().extension() != ".cpp" && entry.path().extension() != ".hpp") continue;
        const std::string text = read(entry.path());
        const bool insecure = std::regex_search(text, off);
        if (insecure) std::printf("  TLS verification turned off in %s\n", entry.path().string().c_str());
        CHECK(!insecure);
    }
    // every transfer in download_model.cpp sets its TLS options through apply_tls
    const std::string dl = read(kSrc / "pull" / "download_model.cpp");
    CHECK(count(dl, "curl_easy_init()") == count(dl, "apply_tls(curl, tls_settings());"));
}
}  // namespace

int main() {
    TestOneRawOutputHelper();
    TestNoInsecureCurlOptions();
    if (failures) { std::printf("%d FAILED\n", failures); return 1; }
    std::printf("all passed\n");
    return 0;
}
