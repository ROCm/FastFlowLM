/// \file chat_grammar.hpp
/// \brief Grammar constraints for tool calls and response_format, on xgrammar
/// \note The grammars are xgrammar structural tags built from the same
///       chat_format_spec the output parser uses. For Hermes JSON, Qwen XML and
///       Harmony they follow xgrammar's own builtin model styles (qwen_3,
///       qwen_3_5, harmony); Gemma4 and LFM2 have hand written argument grammars.
#pragma once

#include <memory>
#include <string>

#include <nlohmann/json.hpp>

#include "chat/chat_constraint.hpp"

/// \brief The structural tag a request needs
/// \return the structural tag JSON, or null when the request needs no grammar
/// \throws std::invalid_argument when the request cannot be expressed for this format
nlohmann::ordered_json chat_structural_tag(const chat_format_spec& spec, const chat_constraint_request& request);

/// \brief Per model grammar compiler; build once, it indexes the whole vocabulary
class chat_grammar_engine {
public:
    /// \param vocab the model's vocabulary; tokenizer_json must point at its tokenizer.json
    /// \param spec the model's format: markers its grammars match as text are kept
    ///        out of the stop tokens, whatever the eos list says
    chat_grammar_engine(const chat_vocab& vocab, const chat_format_spec& spec);
    ~chat_grammar_engine();

    /// \brief Compile the request's grammar into a fresh constraint
    /// \return null when the request needs no grammar
    /// \throws std::invalid_argument when the grammar cannot be built, e.g. a bad schema
    std::shared_ptr<decode_constraint> make(const chat_format_spec& spec, const chat_constraint_request& request);

    /// \brief Whether the request's grammar accepts a whole output, for tests and diagnostics
    /// \param prefill the prefill as the constraint would get it in start()
    /// \param text the generated text
    /// \param error set to where it failed
    bool accepts(const chat_format_spec& spec, const chat_constraint_request& request,
                 const std::string& prefill, const std::string& text, std::string& error);

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};
