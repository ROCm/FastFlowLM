/// \file test_gemma4_tokenizer.cpp
/// \brief Task R5 -- the doubled-BOS question, settled on the host.
///
/// THIS BINARY EXISTS TO REMOVE A STEP FROM THE HARDWARE CHECKLIST. C10's
/// box-44 list, and R4's revision of it, both carried a step 0b that said:
/// dump the encoded prompt on the NPU box and confirm the first id is 2 and
/// appears exactly once. Box 44 is shared and slow to iterate on, and that
/// question needs no NPU, no weights and no engine -- only `tokenizer.json`,
/// `tokenizer_config.json`'s chat template, and the two classes the product
/// actually uses to turn one into ids. So it is asked here instead.
///
/// WHY THE QUESTION IS NOT OBVIOUS. FastFlowLM and the reference driver reach
/// the same prompt by two genuinely different routes:
///
///   driver  `chat_wrap()` DELIBERATELY OMITS the BOS ("NO BOS IN THE STRING
///           ... it is an ID, prepended by encode_prompt") and `encode_prompt`
///           inserts `cfg.bos_token_id` in front of the ids.
///   flm     `Gemma4e::apply_chat_template` renders the model's own template,
///           which EMITS `<bos>` as TEXT, and lets the tokenizer resolve it.
///           `AutoModel::_shared_setup_tokenizer` hands minja
///           `has_bos_token ? bos_token : ""`, so the text form is the whole
///           mechanism.
///
/// Two routes, one of two things could go wrong, and neither raises anything:
/// a SECOND BOS (if the tokenizer's post-processor added one of its own on top
/// of the text form), or NO BOS AT ALL (if the tokenizer did not recognise the
/// literal `<bos>` and split it into ordinary subwords). Either is simply a
/// different prompt, one position out of alignment with the reference, still
/// generating fluent text. The tests below measure both routes on the real
/// packages and show they are byte-identical, and they pin the two properties
/// of `tokenizer.json` that make that true.
///
/// AND IT IS MEASURED THROUGH THE SHIPPED STACK. review-R4.md settled the same
/// question through the Python binding of the `tokenizers` crate and recorded
/// the caveat that this is not the `tokenizers-cpp` build FastFlowLM links.
/// This binary links `common/tokenizer/tokenizer.cpp` -- the production
/// `Tokenizer` class -- against the same tokenizers-cpp libraries the product
/// does, and renders through the production `minja::chat_template` with the
/// same three arguments `_shared_setup_tokenizer` passes it. The caveat is
/// closed.
///
/// WHAT IT STILL DOES NOT COVER: that a sampled `<bos>` is never echoed into
/// the OUTPUT. That is the other thing `bos_token_id` does
/// (`AutoModel::is_normal_token`), it needs a decode loop, and it stays on the
/// hardware list.
///
/// OPT-IN, on the same two environment variables the rest of this suite uses
/// for real packages -- the question is about files Google shipped and there
/// is nothing to ask of a synthetic one. With NEITHER set the process returns
/// 77 and ctest prints "Skipped"; with exactly ONE set it FAILS rather than
/// covering half of what it claims.
///
///   set FLM_GEMMA4_E2B_GGUF=C:/.../gemma-4-E2B-it/gemma-4-E2B-it-Q8_0.gguf
///   set FLM_GEMMA4_E4B_GGUF=C:/.../gemma-4-E4B-it/gemma-4-E4B-it-Q8_0.gguf
///
/// The package directory is the GGUF's parent, and every file in it is opened
/// READ-ONLY. Nothing here writes, moves or deletes anything.

#include "minja/chat-template.hpp"
#include "tokenizer/tokenizer.hpp"

#include "test_support.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

/// \brief the BOS id every real Gemma 4 package states, from three sources
/// \note It is what the TEST expects the files to say, hand-written from
///       verified-gguf-facts.md and never read back out of the subject.
constexpr int kBosId = 2;
constexpr const char* kBosToken = "<bos>";

/// \brief the one user turn, and exactly what gemma4_driver.py wraps it into
///
/// HAND-COPIED FROM THE REFERENCE DRIVER, NOT DERIVED FROM THE TEMPLATE.
/// `gemma4_driver.py:2513 chat_wrap()` returns
/// `f"<|turn>user\n{prompt}<turn|>\n<|turn>model\n"` and documents that
/// jinja2 on the model's own `chat_template`, with `add_generation_prompt=
/// True` on a single user message, produces exactly this string on both E2B
/// and E4B. Writing it out here is what lets the render assertion below be a
/// CROSS-CHECK between two independent implementations rather than a
/// tautology: minja renders Google's template, and the result has to equal
/// what the driver hardcodes.
///
/// The trailing newline after `model` is part of the generation prompt; the
/// template emits it and dropping it changes the first generated token.
constexpr const char* kPrompt = "What is the capital of France?";
constexpr const char* kDriverWrap =
    "<|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n";

std::filesystem::path g_e2b_dir;
std::filesystem::path g_e4b_dir;

std::string ReadFile(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + path.string());
    std::ostringstream text;
    text << input.rdbuf();
    return text.str();
}

/// \brief one real package, opened once
///
/// The `Tokenizer` constructor reads `<dir>/tokenizer.json` (32 MB) and
/// parses it twice -- once into the Rust tokenizer, once into nlohmann for
/// the decoder type -- so it is built once per package and shared, rather
/// than per test.
struct Package {
    std::string name;
    std::filesystem::path dir;
    nlohmann::json tokenizer_config;
    nlohmann::json tokenizer_json;
    std::unique_ptr<::Tokenizer> tokenizer;
};

std::vector<Package> g_packages;

Package OpenPackage(std::string name, const std::filesystem::path& dir) {
    Package package;
    package.name = std::move(name);
    package.dir = dir;
    package.tokenizer_config =
        nlohmann::json::parse(ReadFile(dir / "tokenizer_config.json"));
    // `_shared_setup_tokenizer`'s own precedence: chat_template.jinja wins
    // over tokenizer_config.json's `chat_template`, and current Gemma 4
    // packages ship only the file.
    if (std::filesystem::exists(dir / "chat_template.jinja"))
        package.tokenizer_config["chat_template"] = ReadFile(dir / "chat_template.jinja");
    package.tokenizer_json =
        nlohmann::json::parse(ReadFile(dir / "tokenizer.json"));
    package.tokenizer = std::make_unique<::Tokenizer>(dir.string());
    return package;
}

/// \brief what Gemma4e renders for one user turn, given a BOS token string
///
/// THE PRODUCTION PATH, REPRODUCED ARGUMENT FOR ARGUMENT. `AutoModel::
/// _shared_setup_tokenizer` builds
/// `minja::chat_template(tokenizer_config["chat_template"],
///                       has_bos_token ? tokenizer_config["bos_token"] : "",
///                       tokenizer_config["eos_token"])`
/// and `Gemma4e::apply_chat_template` applies it with
/// `add_generation_prompt = true` and the extra context
/// `_shared_setup_tokenizer` seeded. Passing `bos_token` here is therefore
/// not a test knob: "" is LITERALLY what the production code hands minja when
/// `has_bos_token` is false, which is what the Phi-4 idiom would have made it.
///
/// The template is whichever `_shared_setup_tokenizer` would pick, which
/// OpenPackage has already folded into `tokenizer_config`.
std::string Render(const Package& package, const std::string& bos_token,
                   bool enable_thinking) {
    minja::chat_template chat_tmpl(
        package.tokenizer_config.at("chat_template").get<std::string>(),
        bos_token,
        package.tokenizer_config.at("eos_token").get<std::string>());

    nlohmann::ordered_json messages = nlohmann::ordered_json::array();
    messages.push_back(
        {{"role", "user"},
         {"content", nlohmann::ordered_json::array(
                         {{{"type", "text"}, {"text", kPrompt}}})}});

    minja::chat_template_inputs inputs;
    inputs.add_generation_prompt = true;
    inputs.messages = messages;
    inputs.extra_context["user_system_prompt"] = "";
    inputs.extra_context["enable_thinking"] = enable_thinking;
    return chat_tmpl.apply(inputs);
}

/// \brief the ids FastFlowLM sends: the rendered text, BOS and all, encoded
std::vector<int> FlmRoute(const Package& package, bool enable_thinking = false) {
    return package.tokenizer->encode(
        Render(package, kBosToken, enable_thinking));
}

/// \brief the ids gemma4_driver.py sends: BOS as an ID in front of a wrap
///        that does not contain the text form
std::vector<int> DriverRoute(const Package& package) {
    std::vector<int> ids{kBosId};
    const auto rest = package.tokenizer->encode(Render(package, "", false));
    ids.insert(ids.end(), rest.begin(), rest.end());
    return ids;
}

std::size_t CountBos(const std::vector<int>& ids) {
    return static_cast<std::size_t>(
        std::count(ids.begin(), ids.end(), kBosId));
}

std::string Describe(const std::vector<int>& ids) {
    std::ostringstream text;
    text << '[';
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i) text << ',';
        text << ids[i];
    }
    text << ']';
    return text.str();
}

// ---------------------------------------------------------------------------
// 1. The template and the reference driver agree about the text.

void TestTheRenderedPromptIsTheDriversWrapWithABosInFront() {
    for (const auto& package : g_packages) {
        // minja on Google's template vs. the string gemma4_driver.py
        // hardcodes: two independent implementations of the same wrap. If
        // this ever disagrees, every id comparison below is comparing
        // FastFlowLM against itself.
        const auto with_bos = Render(package, kBosToken, false);
        const auto without_bos = Render(package, "", false);
        if (without_bos != kDriverWrap) {
            throw std::runtime_error(package.name + ": rendered '" +
                                     without_bos + "' but the driver wraps '" +
                                     kDriverWrap + "'");
        }
        TEST_REQUIRE(with_bos == std::string(kBosToken) + kDriverWrap);

        // And `{{- bos_token -}}` really is the FIRST thing the template
        // emits -- it sits ~8 KB into an 18 KB template, after the macro and
        // set blocks, so "opens with" is true of the output and not of the
        // file.
        TEST_REQUIRE(with_bos.rfind(kBosToken, 0) == 0);

        // The comparison above is with `enable_thinking` FALSE, which is
        // `Gemma4e::enable_think`'s default and therefore what the driver's
        // wrap corresponds to. It is not an arbitrary choice of knob:
        // `{%- set enable_thinking = enable_thinking | default(false) -%}`
        // is in the template itself, and with it TRUE the template emits an
        // extra `<|turn>system` block -- so the driver's hardcoded wrap
        // describes the default case only. What does NOT change is the BOS,
        // because `{{- bos_token -}}` is emitted before that branch; the
        // test below checks exactly that, both ways.
    }
}

// ---------------------------------------------------------------------------
// 2. The answer to checklist step 0b.

void TestTheRenderedPromptCarriesExactlyOneBosAtPositionZero() {
    for (const auto& package : g_packages) {
        // Both values of the one template knob the frontend sets on every
        // render. The thinking branch changes what follows the BOS and must
        // not change the BOS.
        for (bool thinking : {false, true}) {
            const auto ids = FlmRoute(package, thinking);
            if (ids.empty() || ids.front() != kBosId || CountBos(ids) != 1) {
                throw std::runtime_error(
                    package.name + (thinking ? " (thinking)" : "") + ": " +
                    Describe(ids) + " is not one BOS at position 0");
            }
        }
    }
}

void TestTheTextRouteAndTheDriversIdRouteAgreeExactly() {
    for (const auto& package : g_packages) {
        const auto flm = FlmRoute(package);
        const auto driver = DriverRoute(package);
        if (flm != driver) {
            throw std::runtime_error(package.name + ": flm " + Describe(flm) +
                                     " != driver " + Describe(driver));
        }
        // Not vacuously equal: a real prompt, not an empty one.
        TEST_REQUIRE(flm.size() > 8);
    }
}

// ---------------------------------------------------------------------------
// 3. Why the answer is what it is -- and that the other outcome was reachable.

void TestWithoutTheTextBosThereIsNoBosAtAll() {
    // THE PHI-4 IDIOM'S FAILURE MODE, MEASURED. `Phi4::setup_tokenizer` sets
    // `has_bos_token = false`, which makes _shared_setup_tokenizer hand minja
    // "" -- exactly the render below. Nothing downstream puts a BOS back:
    // tokenizer.json's post-processor adds none (see the next test), so the
    // prompt simply loses its first token, with no error anywhere. This is
    // what makes the test above non-vacuous: the id 2 in it is there because
    // the TEXT put it there.
    for (const auto& package : g_packages) {
        const auto ids = package.tokenizer->encode(Render(package, "", false));
        TEST_REQUIRE(CountBos(ids) == 0);
        TEST_REQUIRE(!ids.empty());
    }
}

void TestThePostProcessorCannotAddABos() {
    // THE PACKAGE HALF OF "NOTHING CAN ADD A SECOND BOS": TemplateProcessing
    // whose `single` is the sequence alone and whose `special_tokens` map is
    // EMPTY. A conversion that carried a real BOS post-processor would ask
    // for the token the template already emits as text.
    //
    // AND THIS IS THE ONLY TEST HERE THAT WOULD SEE SUCH A PACKAGE, which is
    // worth stating because it is not what one would guess. R5 built a copy
    // of the real E2B package with a `special_tokens: {"<bos>": ...}`
    // post-processor and ran this binary against it: the two id tests above
    // stayed GREEN and only this one went red. The reason is the second,
    // independent mechanism -- tokenizers-cpp's HFTokenizer calls
    // `Encode(text, /*add_special_tokens=*/false)` (third_party/
    // tokenizers-cpp/src/huggingface_tokenizer.cc), so the post-processor's
    // special tokens are never applied on this path at all, whatever the
    // file asks for. That is STRONGER than review-R4.md's argument, not
    // weaker: the doubled BOS is impossible here for two reasons rather than
    // one. But it does mean the file's shape has to be asserted directly,
    // because no id comparison on this path can notice it changing.
    for (const auto& package : g_packages) {
        const auto& post = package.tokenizer_json.at("post_processor");
        TEST_REQUIRE(post.at("type") == "TemplateProcessing");
        TEST_REQUIRE(post.at("special_tokens").empty());
        const auto& single = post.at("single");
        TEST_REQUIRE(single.size() == 1);
        TEST_REQUIRE(single.at(0).contains("Sequence"));
        TEST_REQUIRE(single.at(0).at("Sequence").at("id") == "A");
    }
}

void TestBosIsAnAddedSpecialTokenSoTheTextFormIsNeverSplit() {
    // THE OTHER HALF. `<bos>` resolves to one id rather than a handful of
    // ordinary subwords only because it is an ADDED token with `special:
    // true`; added tokens are matched before the BPE model ever sees the
    // text. Asserted from tokenizer.json and, independently, from the
    // tokenizer the product builds out of it.
    for (const auto& package : g_packages) {
        bool found = false;
        for (const auto& added : package.tokenizer_json.at("added_tokens")) {
            if (added.at("content") != kBosToken) continue;
            found = true;
            TEST_REQUIRE(added.at("id") == kBosId);
            TEST_REQUIRE(added.at("special") == true);
        }
        TEST_REQUIRE(found);
        TEST_REQUIRE(package.tokenizer->encode(kBosToken) ==
                     std::vector<int>{kBosId});
    }
}

}  // namespace

int main() {
    // BEFORE ANYTHING ELSE. Gemma 4 keeps no weight cache; only Phi-4's
    // engine reads this variable, and nothing here constructs an engine. It
    // stays pinned off so no cache could ever be written into the read-only
    // packages under C:/Users/chiz/work/models.
#ifdef _WIN32
    _putenv_s("FLM_RAI_WEIGHT_CACHE", "0");
#else
    setenv("FLM_RAI_WEIGHT_CACHE", "0", 1);
#endif

    const char* e2b = std::getenv("FLM_GEMMA4_E2B_GGUF");
    const char* e4b = std::getenv("FLM_GEMMA4_E4B_GGUF");
    const bool have_e2b = e2b != nullptr && *e2b != '\0';
    const bool have_e4b = e4b != nullptr && *e4b != '\0';

    if (!have_e2b && !have_e4b) {
        std::cout << "SKIP: neither FLM_GEMMA4_E2B_GGUF nor "
                     "FLM_GEMMA4_E4B_GGUF is set, so the doubled-BOS question "
                     "was NOT answered in this run and checklist step 0b is "
                     "back on the hardware list. Set both to the real "
                     "gemma-4-E{2,4}B-it-Q8_0.gguf paths; this binary reads "
                     "tokenizer.json and tokenizer_config.json from their "
                     "parent directories and never opens the GGUF.\n";
        return 77;
    }
    if (!have_e2b || !have_e4b) {
        std::cerr << "FAIL: exactly one of FLM_GEMMA4_E2B_GGUF / "
                     "FLM_GEMMA4_E4B_GGUF is set. Set both or neither -- the "
                     "two packages are here because each is a separate "
                     "conversion, and a half-covered run that reports green "
                     "is what this suite exists to prevent.\n";
        return 1;
    }

    g_e2b_dir = std::filesystem::path(e2b).parent_path();
    g_e4b_dir = std::filesystem::path(e4b).parent_path();
    for (const auto& dir : {g_e2b_dir, g_e4b_dir}) {
        for (const char* file : {"tokenizer.json", "tokenizer_config.json"}) {
            if (!std::filesystem::exists(dir / file)) {
                std::cerr << "FAIL: " << (dir / file).string()
                          << " does not exist.\n";
                return 1;
            }
        }
    }
    std::cout << "REAL PACKAGES: " << g_e2b_dir.string() << "\n               "
              << g_e4b_dir.string() << '\n';

    try {
        g_packages.push_back(OpenPackage("E2B", g_e2b_dir));
        g_packages.push_back(OpenPackage("E4B", g_e4b_dir));
    } catch (const std::exception& error) {
        std::cerr << "FAIL: cannot open a real package: " << error.what()
                  << '\n';
        return 1;
    }

#define RUN_TEST(name) RunTest(&name, #name)
    RUN_TEST(TestTheRenderedPromptIsTheDriversWrapWithABosInFront);
    RUN_TEST(TestTheRenderedPromptCarriesExactlyOneBosAtPositionZero);
    RUN_TEST(TestTheTextRouteAndTheDriversIdRouteAgreeExactly);
    RUN_TEST(TestWithoutTheTextBosThereIsNoBosAtAll);
    RUN_TEST(TestThePostProcessorCannotAddABos);
    RUN_TEST(TestBosIsAnAddedSpecialTokenSoTheTextFormIsNeverSplit);
#undef RUN_TEST

    std::cout << "test_gemma4_tokenizer: PASS\n";
}
