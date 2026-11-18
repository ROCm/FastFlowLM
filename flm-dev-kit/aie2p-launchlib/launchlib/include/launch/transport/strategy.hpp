//===- strategy.hpp - how the state of an app reaches the tiles ----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// Portions of this file consist of AI-generated content.
//
//===----------------------------------------------------------------------===//
//
// The parts of a kernel that depend on the transport (Strategy). The pipeline
// calls them in this order (kernel/pipeline.hpp):
//
//   reset part         (kernel/reset.hpp, the same for every transport) with
//                      the switch values of final_routes(prev) and
//                      load_routes(run), then adjust_reset
//   load(k)            the state of k.run into its tiles: program, data, locks,
//                      BDs, mux, chain starts; then the cores start
//   load_program(k)    a program-only warm kernel: per core of k.run, stop it,
//                      write its program, start it
//   runtime sequence   (not here)
//
//   final_routes(p)    the switch registers the array holds after p's load
//   load_routes(p)     the ones p's load needs while it runs (the reset part
//                      writes these); both p's routes unless the transport
//                      adds routes of its own (ctrlpkt_lane.hpp)
//
// One strategy object lives for one build(). A new transport is a new subclass.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "launch/kernel/context.hpp"
#include "launch/kernel/reset.hpp"

#include <map>

namespace launch::detail {

class TransportStrategy {
public:
  virtual ~TransportStrategy() = default;
  virtual const RouteMap &final_routes(const Package &p) { return routes_of(p); }
  virtual const RouteMap &load_routes(const Package &p) { return final_routes(p); }
  virtual void adjust_reset(KernelContext &, ResetTargets &) {}
  virtual void load(KernelContext &k) = 0;
  virtual void load_program(KernelContext &k) = 0;

protected:
  /// The routes of the package (kept for the build).
  const RouteMap &routes_of(const Package &p) {
    auto it = routes_.find(&p);
    if (it == routes_.end()) {
      RouteMap m;
      for (const auto &[t, init] : p.tiles)
        if (!init.routes.empty())
          m[t] = init.routes;
      it = routes_.emplace(&p, std::move(m)).first;
    }
    return it->second;
  }

private:
  std::map<const Package *, RouteMap> routes_;
};

} // namespace launch::detail
