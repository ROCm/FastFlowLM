#ifndef FLM_OVERRIDE_HPP
#define FLM_OVERRIDE_HPP

/// \file flm_override.hpp
/// \brief Compile-time override points for a model's operator dispatch.
///
/// `FLM_OVERRIDE(name, expr, ...)` expands to `expr`. A build that points
/// `FLM_OVERRIDES` at a header may redefine it to dispatch on `name`; every
/// other build emits the code it would have emitted without the annotation.
///
///     g++ -DFLM_OVERRIDES='"iron_overrides.h"' ...
///
/// `name` is a token, not a string, so it costs nothing. The trailing
/// arguments carry values an override needs and the call itself does not. The
/// default expansion discards them unevaluated, so they must be free of side
/// effects, and a value computed only to be passed here belongs inside the
/// annotation too, or `-Wall` reports it unused.
///
/// A statement-position hook that only hands an override some context writes
/// `(void)0` as its expression.

#ifdef FLM_OVERRIDES
#include FLM_OVERRIDES
#endif

#ifndef FLM_OVERRIDE
#define FLM_OVERRIDE(name, expr, ...) (expr)
#endif

#endif  // FLM_OVERRIDE_HPP
