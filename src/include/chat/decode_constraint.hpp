/// \file decode_constraint.hpp
/// \brief Per request restrictions on which token may be sampled next
/// \note The decode loop asks for a mask before each sample and reports the
///       sampled token back. Masks are bitsets over the vocabulary so this
///       header needs nothing from the inference engine.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

/// \brief One request's constraint, advanced token by token
class decode_constraint {
public:
    virtual ~decode_constraint() = default;

    /// \brief Called once the prompt is in, before the first token is sampled
    /// \param prefill the part of the turn the template already wrote, see chat_format_prefill
    virtual void start(const std::string& /*prefill*/) {}

    /// \brief Allowed tokens for the next sample
    /// \param mask resized to cover the vocabulary; bit i of word i/32 set means
    ///        token i may be sampled
    /// \return false when every token is allowed, the mask is then left as is
    virtual bool fill_mask(std::vector<uint32_t>& mask) = 0;

    /// \brief Advance past the token that was sampled
    virtual void accept(int token) = 0;

    /// \brief Advance past a token the model was made to emit, not sampled
    /// \note A forced token may open what sampling was not allowed to, e.g. the
    ///       reasoning block of a thinking-only model whose request turned thinking off.
    virtual void accept_forced(int token) {
        accept(token);
    }

    /// \brief Undo the last n accepted tokens
    virtual void rollback(int /*n*/) {}
};

/// \brief Bans a fixed set of tokens, e.g. the tool call opener for tool_choice=none
class token_ban_constraint : public decode_constraint {
public:
    token_ban_constraint(size_t vocab_size, std::vector<int> banned)
        : vocab_size_(vocab_size), banned_(std::move(banned)) {}

    bool fill_mask(std::vector<uint32_t>& mask) override {
        if (banned_.empty()) {
            return false;
        }
        mask.assign((vocab_size_ + 31) / 32, 0xFFFFFFFFu);
        for (int token : banned_) {
            if (token >= 0 && static_cast<size_t>(token) < vocab_size_) {
                mask[token / 32] &= ~(1u << (token % 32));
            }
        }
        return true;
    }

    void accept(int) override {}

private:
    size_t vocab_size_;
    std::vector<int> banned_;
};
