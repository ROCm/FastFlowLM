/// \file top_k.hpp
/// \brief What a request's `top_k` means for the sampler
#pragma once

/// \brief The number of tokens the sampler keeps for a requested `top_k`
/// \param requested the request's `top_k`
/// \param vocab_size the model's vocabulary size
/// \return `requested` clamped to [1, vocab_size]; 0 and below mean "off" (the whole vocabulary),
///         as in vLLM (0 or -1) and llama.cpp (0)
inline int effective_top_k(int requested, int vocab_size) {
    if (requested <= 0 || requested > vocab_size) {
        return vocab_size;
    }
    return requested;
}
