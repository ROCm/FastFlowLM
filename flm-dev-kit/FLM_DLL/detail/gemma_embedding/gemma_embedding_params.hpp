// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once

constexpr int D = 768;
constexpr int DH = 256;
constexpr int DQ = DH * 3;
constexpr int DK = DH;
constexpr int DV = DH;
constexpr int INTERMEDIATE_SIZE = 1152;
constexpr int num_hidden_layers = 24;
constexpr int LC = 16;
constexpr int MAX_L = 2048;
constexpr int num_attention_heads = 3;
constexpr int num_key_value_heads = 1;
constexpr int head_dim = 256;
constexpr int hidden_size = 768;
constexpr int intermediate_size = 1152;
constexpr int dense_intermediate_size = 3072;
