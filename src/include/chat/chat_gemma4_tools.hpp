/// \file chat_gemma4_tools.hpp
/// \brief Prepare a request's tools for the Gemma4 chat template
#pragma once

#include <nlohmann/json.hpp>

/// \brief The tools with every nullable type, such as ["string", "null"], written as the
///        template's own form: the remaining type and nullable: true
/// \note The template upper-cases a parameter's type, which minja cannot do to a list, so
///       one nullable parameter used to fail the whole request. A type that still lists
///       more than one type after dropping "null" is left as it is.
nlohmann::ordered_json gemma4_template_tools(nlohmann::ordered_json tools);
