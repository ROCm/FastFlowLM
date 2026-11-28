// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file test_special_token_ids.cpp
/// \brief special_token_ids::checked_bos_id: the BOS id the engine uses comes from the tokenizer when tokenizer_config.json
///        states a different one (the gpt-oss packages state 2, which is "#", for <|startoftext|>, so every lone "#" was
///        dropped from the output)
///   test_special_token_ids <fixtures dir>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "tokenizer/special_token_ids.hpp"
#include "tokenizer/tokenizer.hpp"

static int failures = 0;
#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #cond << "\n"; \
            failures++;                                                                   \
        }                                                                                 \
    } while (0)

static nlohmann::json read_json(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return nlohmann::json::parse(ss.str());
}

// the rule on its own, with a lookup table for a tokenizer
static void test_rule() {
    const std::map<std::string, int> ids = {{"#", 2}, {"<|startoftext|>", 199998}, {"<s>", 1}};
    auto lookup = [&](const std::string& t) {
        const auto it = ids.find(t);
        return it == ids.end() ? -1 : it->second;
    };
    std::string warning;

    // the configured id is not the bos_token: the tokenizer's id wins, with a warning
    CHECK(special_token_ids::checked_bos_id("<|startoftext|>", 2, lookup, &warning) == 199998);
    CHECK(warning.find("199998") != std::string::npos);
    CHECK(warning.find("bos_token_id 2") != std::string::npos);

    // ids that agree: unchanged, no warning
    CHECK(special_token_ids::checked_bos_id("<s>", 1, lookup, &warning) == 1);
    CHECK(warning.empty());

    // an AddedToken object names the token by "content"
    const nlohmann::json added = {{"content", "<|startoftext|>"}, {"special", true}};
    CHECK(special_token_ids::checked_bos_id(added, 2, lookup, &warning) == 199998);
    CHECK(!warning.empty());

    // a token the tokenizer does not know, a null or empty entry, or no lookup: the configured id stays
    CHECK(special_token_ids::checked_bos_id("<bos>", 7, lookup, &warning) == 7);
    CHECK(warning.empty());
    CHECK(special_token_ids::checked_bos_id(nullptr, 7, lookup, &warning) == 7);
    CHECK(special_token_ids::checked_bos_id("", 7, lookup, &warning) == 7);
    CHECK(special_token_ids::checked_bos_id("<s>", 7, nullptr, &warning) == 7);
    CHECK(special_token_ids::checked_bos_id("<|startoftext|>", 2, lookup, nullptr) == 199998);

    CHECK(special_token_ids::text(nlohmann::json(5)).empty());
}

// with FastFlowLM's Tokenizer on a package laid out like the gpt-oss ones: "#" is the vocabulary token 2,
// <|startoftext|> an added token, and tokenizer_config.json states bos_token_id 2
static void test_package(const std::string& dir) {
    Tokenizer tok(dir);
    const nlohmann::json cfg = read_json(dir + "/tokenizer_config.json");

    CHECK(tok.token_to_id("#") == 2);
    CHECK(tok.token_to_id("<|startoftext|>") == 15);
    CHECK(tok.token_to_id("<no such token>") == -1);

    const std::vector<int> hash = tok.encode("#");
    CHECK(hash.size() == 1 && hash[0] == 2);  // a lone "#" is token 2, the id the config states for BOS

    std::string warning;
    const int bos = special_token_ids::checked_bos_id(
        cfg["bos_token"], cfg["bos_token_id"].get<int>(),
        [&](const std::string& t) { return tok.token_to_id(t); }, &warning);
    CHECK(bos == 15);
    CHECK(!warning.empty());
    CHECK(bos != hash[0]);  // so "#" is no longer taken for BOS and filtered out of the output
    CHECK(tok.decode({hash[0]}) == "#");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_special_token_ids <fixtures dir>\n";
        return 2;
    }
    test_rule();
    test_package(std::string(argv[1]) + "/gpt_oss_like");
    if (failures) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "special_token_ids: all checks passed\n";
    return 0;
}
