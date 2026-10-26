/// \file d3_engine_stub.hpp
/// \brief A CPU stand-in for the d3 engine: returns scripted hidden rows and records every request.
#pragma once

#include <functional>
#include <stdexcept>

#include "decision/d3_engine.hpp"

namespace flm::d3 {

class ScriptedEngine : public ReadoutEngine {
public:
    /// (request, block index) -> `readout_from_end.size() * hidden_size` floats
    using Script = std::function<std::vector<float>(const EngineStateRequest&, size_t)>;

    ScriptedEngine(int hidden, Script script) : hidden_(hidden), script_(std::move(script)) {}

    int hidden_size() const override { return hidden_; }

    std::vector<std::vector<float>> encode_state_once(const EngineStateRequest& request) override {
        calls.push_back(request);
        std::vector<std::vector<float>> out;
        for (size_t b = 0; b < request.blocks.size(); b++) {
            std::vector<float> rows = script_(request, b);
            if (rows.size() != request.blocks[b].readout_from_end.size() * static_cast<size_t>(hidden_)) {
                throw std::runtime_error("scripted rows do not match the requested readout positions");
            }
            out.push_back(std::move(rows));
        }
        return out;
    }

    std::vector<EngineStateRequest> calls;   //!< every request, in order

private:
    int hidden_;
    Script script_;
};

}  // namespace flm::d3
