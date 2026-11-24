// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __EMBEDDING_HPP__
#define __EMBEDDING_HPP__
#include "buffer.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "tensor_2d.hpp"
#include <immintrin.h>

class embedding_module{
private:
    buffer<bf16> weight;
    size_t vocabe_size;
    size_t dim;
    tensor_2d<bf16> tensor_weight;

    buffer<bf16> out_buffer;

    public:
    embedding_module(size_t vocab_size, size_t dim) {
        this->vocabe_size = vocab_size;
        this->dim = dim;
        this->weight = buffer<bf16>(vocab_size * dim);
        this->out_buffer = buffer<bf16>(dim);
        this->tensor_weight = tensor_2d<bf16>(weight, dim);
    }

    void init_weights(Q4NX& q4nx, const std::string& weight_name){
        q4nx.load_weights(this->weight, weight_name + ".weight");
    }

    buffer<bf16> forward(int idx){
        buffer<bf16> weight_row = tensor_weight[idx];
        return weight_row;
    }

    void forward(int idx, buffer<bf16>& out){
        buffer<bf16> weight_row = tensor_weight[idx];
        memcpy(out.data(), weight_row.data(), dim * sizeof(bf16));
    }
};

#endif