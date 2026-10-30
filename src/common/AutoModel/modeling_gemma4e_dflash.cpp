/// \file modeling_gemma4e_dflash.cpp
/// \brief Gemma4e_DFlash: DFlash speculative decoding for nemotron3.5-bolt-dflash:e4b
///
/// One cycle (dflash-bolt.md section 1.1): the engine drafts up to 7 tokens after the last sampled token, the target
/// verifies [last, drafts...] in one go and hands back one logits row per position. The wrapper then samples row i
/// with the user's sampler and accepts while the sample equals the draft (D11): every emitted token is a sample of the
/// target conditional on the accepted prefix, so sampling stays distribution-exact and only the tokens that are really
/// emitted touch the sampler's RNG and penalty history.
#include "AutoModel/modeling_gemma4e.hpp"
#include "AutoModel/automodel.hpp"
#include <cstdlib>
#include <cstring>

namespace {
/// FLM_DFLASH=0 turns speculation off (the tag then behaves like the plain model)
bool dflash_env_enabled() {
    const char* v = std::getenv("FLM_DFLASH");
    return !(v != nullptr && v[0] == '0' && v[1] == '\0');
}
}

gemma4e_engine_config_t Gemma4e_DFlash::engine_config() const {
    return read_gemma4e_engine_config<gemma4e_dflash>(this->lm_engine);
}

const flm_state_api* Gemma4e_DFlash::engine_state_api() const {
    return gemma4e_dflash_state_api(this->lm_engine);
}

void Gemma4e_DFlash::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption, const std::string& backend) {
    Gemma4e::load_model(model_path, model_info, default_context_length, enable_preemption, backend);
    auto* engine = dynamic_cast<gemma4e_dflash*>(this->lm_engine);
    if (engine == nullptr) {
        header_print("WARNING", "Gemma4e_DFlash: the engine is not gemma4e_dflash, speculation is off");
        return;
    }
    engine->set_stop_tokens(this->eos_token_ids);
    if (this->draft_model_path.empty()) {
        header_print("WARNING", "Gemma4e_DFlash: no draft package, decoding without speculation");
        return;
    }
    if (!dflash_env_enabled()) {
        header_print("FLM", "DFlash speculation disabled by FLM_DFLASH=0");
        return;
    }
    if (engine->load_draft(this->draft_model_path)) {
        header_print("FLM", "DFlash draft loaded from " << this->draft_model_path);
    } else {
        header_print("WARNING", "Gemma4e_DFlash: the draft package could not be loaded, decoding without speculation");
    }
}

std::string Gemma4e_DFlash::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    return this->_speculative_generate(meta_info, length_limit, os, is_cancelled);
}

std::string Gemma4e_DFlash::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    if (this->thinking_enabled()) {
        os << "<think>\n" << std::flush;
    }
    int checkpoint_idx = this->lm_engine->checkpoint();
    int restore_idx = this->lm_engine->restore();
    header_print_r("FLM", "Checkpoint before generation: " << checkpoint_idx << ", restore point: " << restore_idx << ", user context length: " << this->token_history.size());
    return this->_speculative_generate(meta_info, length_limit, os, [] { return false; });
}

void Gemma4e_DFlash::clear_context() {
    Gemma4e::clear_context();
    if (auto* engine = dynamic_cast<gemma4e_dflash*>(this->lm_engine)) {
        engine->reset_speculation_stats();
    }
}

std::string Gemma4e_DFlash::show_profile() {
    std::string profile = Gemma4e::show_profile();
    if (auto* engine = dynamic_cast<gemma4e_dflash*>(this->lm_engine)) {
        if (engine->speculation_cycles() > 0) {
            profile += engine->speculation_timing();
        }
    }
    return profile;
}

std::string Gemma4e_DFlash::_speculative_generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    std::string result;
    assert(this->last_token != -1);

    stop_reason_t reason = EOT_DETECTED;

    auto* engine = dynamic_cast<gemma4e_dflash*>(this->lm_engine);
    // Off under a decoding constraint (grammar / tool JSON, D14): a drafted token never passes the constraint's mask
    // through sample-and-compare unless every row is masked, which the plain loop already does cheaply.
    // draft_ready() is asked per cycle below: the engine's adaptive guard (it turns speculation off while it cannot win and re-probes
    // later) and the headroom check change from cycle to cycle, so only the static conditions are fixed for the whole turn.
    const bool spec_enabled = engine != nullptr && !meta_info.constraint && dflash_env_enabled() && engine->supports_speculation();
    const int SPEC_MAX_DRAFT = 7;
    const int vocab = engine != nullptr ? engine->vocab_size() : 0;
    if (spec_enabled) {
        header_print("FLM", "Speculative decoding enabled (DFlash draft).");
    }

    int last_sampled_token = this->last_token;
    this->token_history.push_back(this->last_token);
    // the sample insert() took is the reply's first token, and counts toward the limit
    meta_info.generated_tokens++;
    if (this->is_normal_token(last_sampled_token) && last_sampled_token != -1) {
        std::string token_str = this->tokenizer->run_time_decoder(last_sampled_token);
        result += token_str;
        os << token_str << std::flush;
    }
    if (this->is_eos(last_sampled_token)) {
        // Sampled from the prefill, so no engine has run it.
        this->token_history.pop_back();
        meta_info.stop_reason = reason;
        return result;
    }
    this->profiler_list[DECODING_TIME].reset();
    this->profiler_list[TKOEN_DECODE_TIME].reset();

    // Commit one token: stream it, record it, test the stop rules. Shared by the plain and the speculative branch.
    auto consume = [&](int token) -> bool {
        this->total_tokens++;
        last_sampled_token = token;

        this->profiler_list[TKOEN_DECODE_TIME].start();
        if (this->is_normal_token(token)) { // filter out special tokens
            std::string token_str = this->tokenizer->run_time_decoder(token);
            os << token_str << std::flush;
            result += token_str;
        }
        this->profiler_list[TKOEN_DECODE_TIME].stop(1);
        this->token_history.push_back(token);
        if (this->is_eos(token)) {
            meta_info.generated_tokens++;
            // The engine never ran the stop token: supply its KV row (it is the last token of a cycle, see below).
            this->lm_engine->forward(token);
            return false;
        }
        meta_info.generated_tokens++;
        if ((length_limit > 0) && (meta_info.generated_tokens >= length_limit)) {
            reason = MAX_LENGTH_REACHED;
            return false;
        }
        return true;
    };

    if (this->total_tokens >= this->MAX_L) {
        header_print("WARNING", "Max length reached, stopping generation...");
        reason = MAX_LENGTH_REACHED;
    } else if ((length_limit > 0) && (meta_info.generated_tokens >= length_limit)) {
        reason = MAX_LENGTH_REACHED;
    } else while (true) {
        if (this->total_tokens >= this->MAX_L) {
            reason = MAX_LENGTH_REACHED;
            break;
        }
        if (is_cancelled()) {
            reason = CANCEL_DETECTED;
            break;
        }

        // A cycle emits up to n_drafts + 1 tokens, all but the last already in the caches: cap it at the tokens left so
        // the limit lands on the last one. With one token left, decode it plainly.
        const int tokens_left = (length_limit > 0) ? length_limit - meta_info.generated_tokens : SPEC_MAX_DRAFT + 1;
        if (spec_enabled && tokens_left > 1 && engine->draft_ready()) {
            int drafts[SPEC_MAX_DRAFT];
            this->profiler_list[DECODING_TIME].start();
            int n_drafts = engine->dflash_draft(last_sampled_token, drafts);
            if (n_drafts > tokens_left - 1) n_drafts = tokens_left - 1;
            const bf16* rows = n_drafts > 0 ? engine->dflash_verify(last_sampled_token, drafts, n_drafts) : nullptr;
            this->profiler_list[DECODING_TIME].stop(0);   // time only; tokens are counted below

            if (rows != nullptr) {
                // sample-and-compare: s_i from row i; accept while s_i == drafts[i]; the first mismatch (or row n) is the bonus
                std::vector<int> emitted;
                emitted.reserve(n_drafts + 1);
                this->profiler_list[SAMPLING_TIME].start();
                for (int i = 0; i <= n_drafts; i++) {
                    buffer<bf16> row(const_cast<bf16*>(rows) + (size_t)i * vocab, (size_t)vocab);
                    const int s = this->_sample(row, meta_info);
                    emitted.push_back(s);
                    if (this->is_eos(s) || i == n_drafts || s != drafts[i]) break;
                }
                this->profiler_list[SAMPLING_TIME].stop((int)emitted.size());

                // rows kept in the caches: [last, accepted drafts...]; the last emitted token has no KV row (as after forward())
                const int n_keep = (int)emitted.size();
                this->profiler_list[DECODING_TIME].start();
                engine->dflash_commit(n_keep);
                this->profiler_list[DECODING_TIME].stop(n_keep);
                if (const uint64_t prime_us = engine->last_speculation_prime_us()) {
                    this->profiler_list[DECODING_TIME].add_time(-(int64_t)prime_us);
                    this->profiler_list[PREFILL_TIME].add_time((int64_t)prime_us);
                }

                bool go_on = true;
                for (int tok : emitted) {
                    if (!(go_on = consume(tok))) break;
                }
                if (!go_on) break;
                continue;
            }
            // declined (no room, draft not primed, ...): fall through to the plain step, state untouched
        }

        this->profiler_list[DECODING_TIME].start();
        buffer<bf16> y = this->lm_engine->forward(last_sampled_token);
        this->profiler_list[DECODING_TIME].stop(1);

        this->profiler_list[SAMPLING_TIME].start();
        int sampled_token = this->_sample(y, meta_info);
        this->profiler_list[SAMPLING_TIME].stop(1);

        if (!consume(sampled_token)) break;
    }
    meta_info.decoding_duration = (uint64_t)(time_utils::cast_to_us(this->profiler_list[DECODING_TIME].get_total_time()).first) * 1e3;
    meta_info.stop_reason = reason;
    if (this->total_tokens >= this->MAX_L) {
        header_print("WARNING", "Max length reached, stopping generation...");
    }
    std::cout << std::endl;
    header_print("FLM", "Model RAW Output: \n" + result);
    if (spec_enabled && engine->speculation_cycles() > 0) {
        header_print("FLM", engine->speculation_stats());
    }

    if (reason != EOT_DETECTED) {
        // Cut short after its last token was sampled: no engine has run that one, so the next prompt has to carry it.
        this->token_history.pop_back();
    }
    if (!this->thinking_enabled()) {
        // the history the checkpoint holds: an eos ending the turn was run, a cut-off token was not
        this->lm_engine->checkpoint();
        checkpoint_his = token_history;
    }
    return result;
}
