/// \file chat_args.hpp
/// \brief Incremental conversion of tool call arguments to JSON
/// \note Every converter here is prefix stable: converting a prefix of some
///       text yields a prefix of what converting the whole text yields. That is
///       what lets the stream send arguments piece by piece. Values whose JSON
///       form depends on text not seen yet (a bare number, true/false) are held
///       back until they end; strings are passed through as they arrive.
#pragma once

#include <cstddef>
#include <string>

/// \brief Append s to out as JSON string content, without the quotes
void chat_json_escape_append(std::string& out, const std::string& s);

/// \brief Find where the JSON value starting at pos ends
/// \return the index just past the value, or npos if the text ends first
size_t chat_json_value_end(const std::string& s, size_t pos);

/// \brief Convert a bare scalar the way a model means it
/// \return JSON text: numbers, true/false/null (also Python's True/False/None)
///         as literals, anything else as a JSON string
std::string chat_scalar_to_json(const std::string& token);

/// \brief Syntax of the relaxed, not quite JSON argument formats
enum class chat_relaxed_dialect {
    GEMMA4, // {key:<|"|>text<|"|>,n:5,flag:true,list:[...]}
    PYTHON, // key="text", n=5, flag=True, list=[...]  (inside a call's parentheses)
};

/// \brief Result of converting relaxed arguments
struct chat_relaxed_result {
    std::string json; // a prefix of the final JSON object, complete when closed is true
    size_t end = 0;   // index just past what was consumed
    bool closed = false; // the argument list was properly terminated
};

/// \brief Convert relaxed arguments to a JSON object
/// \param s the text
/// \param pos for GEMMA4 the index of the opening '{'; for PYTHON the index just
///        past the call's '('
/// \param dialect the syntax
/// \param input_done true when no more text will come; unterminated values and
///        containers are then closed so the result is valid JSON
chat_relaxed_result chat_relaxed_args_to_json(const std::string& s, size_t pos,
                                              chat_relaxed_dialect dialect, bool input_done);
