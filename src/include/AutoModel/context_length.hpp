/// \file context_length.hpp
/// \brief The context length a model is loaded with, decided in one place
/// \note Read by the loader (resolve_context_length in automodel.cpp) and by
///       the capability record that GET /v1/models reports
///       (model_list::capabilities_of), so the listing cannot name a context
///       the loader would not use.
#pragma once

#include <algorithm>

namespace flm {

/// \brief whether an entry is a single-turn model (a dedicated translation
///        model, say), which keeps its own context whatever --ctx-len says
template <typename Json>
bool is_single_turn(const Json& model_info) {
    const auto label = model_info.find("label");
    return label != model_info.end() && label->is_array() &&
           std::find(label->begin(), label->end(), "single-turn") != label->end();
}

/// \brief the context length an entry is loaded with
/// \param model_info the catalog entry
/// \param ctx_length the --ctx-len the server was started with; negative when
///        it was started without one
/// \return the entry's default_context_length for a single-turn model or when
///         no --ctx-len was given, else ctx_length; 0 when the entry names no
///         default and one is needed
template <typename Json>
long long context_length_for(const Json& model_info, int ctx_length) {
    if (ctx_length >= 0 && !is_single_turn(model_info)) {
        return ctx_length;
    }
    const auto fixed = model_info.find("default_context_length");
    return fixed != model_info.end() && fixed->is_number_integer()
        ? fixed->template get<long long>() : 0;
}

}  // namespace flm
