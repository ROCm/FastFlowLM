// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file tokenize_ids.cpp
/// \brief Writes the token ids of a text file, one per line, with the model's tokenizer (FastFlowLM's)
#include <fstream>
#include <iostream>
#include <sstream>
#include "tokenizer/tokenizer.hpp"

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "usage: tokenize_ids <model folder> <text file> <ids out file>\n";
        return 1;
    }
    std::ifstream in(argv[2], std::ios::binary);
    std::stringstream text;
    text << in.rdbuf();
    Tokenizer tokenizer(argv[1]);
    std::vector<int> ids = tokenizer.encode(text.str());
    std::ofstream out(argv[3]);
    for (int id : ids) {
        out << id << "\n";
    }
    std::cout << ids.size() << " tokens -> " << argv[3] << std::endl;
    return 0;
}
