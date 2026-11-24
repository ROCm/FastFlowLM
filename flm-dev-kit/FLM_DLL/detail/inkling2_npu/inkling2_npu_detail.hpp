// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file inkling2_npu_detail.hpp
/// \brief inkling2_npu::Impl
#pragma once
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "models/inkling2/inkling2_npu.hpp"
#include "inkling2_fused.hpp"
#include "inkling2_host.hpp"
#include "inkling2_prefill.hpp"

/// \brief one persistent thread that runs the host model's compute.
///        Each thread that opens an OpenMP region becomes the master of its own team; a server
///        calls the engine from whichever io thread holds the request, so without this every io
///        thread grew a team of its own and decode fell from ~16 to ~10 tok/s once three teams
///        existed. Here there is one master, hence one team, for the engine's lifetime.
class compute_thread {
public:
    compute_thread() : th([this] { this->loop(); }) {}
    ~compute_thread() {
        {
            std::lock_guard<std::mutex> g(this->m);
            this->quit = true;
        }
        this->cv.notify_all();
        this->th.join();
    }
    compute_thread(const compute_thread&) = delete;
    compute_thread& operator=(const compute_thread&) = delete;

    /// \brief run f on the compute thread and wait for it; an exception of f is rethrown here
    void call(std::function<void()> f) {
        std::unique_lock<std::mutex> l(this->m);
        this->job = std::move(f);
        this->error = nullptr;
        this->done = false;
        this->cv.notify_all();
        this->cv.wait(l, [this] { return this->done; });
        if (this->error) std::rethrow_exception(this->error);
    }

private:
    void loop() {
        std::unique_lock<std::mutex> l(this->m);
        for (;;) {
            this->cv.wait(l, [this] { return this->quit || this->job; });
            if (this->quit) return;
            std::function<void()> f = std::move(this->job);
            this->job = nullptr;
            l.unlock();
            std::exception_ptr e;
            try {
                f();
            } catch (...) {
                e = std::current_exception();
            }
            l.lock();
            this->error = e;
            this->done = true;
            this->cv.notify_all();
        }
    }

    std::mutex m;
    std::condition_variable cv;
    std::function<void()> job;
    std::exception_ptr error;
    bool done = false, quit = false;
    std::thread th;   // last: started after the members it uses
};

struct inkling2_npu::Impl {
    LM_Config config;
    npu_xclbin_manager* npu = nullptr;
    int MAX_L = 0;

    inkling2_host::model model;
    std::unique_ptr<inkling2_prefill> prefill;   // the prompt matmuls on the NPU (null: on the host)
    // decode on the NPU (null: on the host). The host model keeps the sequence state; the first NPU token loads it
    // to the device, and the tokens the NPU runs go back to the host (sync_host) before anything else reads it.
    std::unique_ptr<inkling2_fused> fused;
    bool on_device = false;       // the device has the newest state (the host's is behind by dev_ids)
    std::vector<int> dev_ids;     // the tokens run on the device since its state was the host's
    std::vector<float> fused_x;   // the last token's final-normed x
    std::vector<float> logits_f32;
    buffer<bf16> logits;   // vocab_size; forward() / prefill() return a view of it

    std::unique_ptr<hybrid_engine_state> legacy_checkpoint;
    compute_thread compute;   // runs model.run(); see compute_thread

    Impl(LM_Config config, npu_xclbin_manager* npu, int MAX_L);

    /// \brief logits_f32 -> logits
    buffer<bf16>& publish_logits();

    std::unique_ptr<hybrid_engine_state> new_state();
    int read_state(hybrid_engine_state& st, int kv_from);
    int write_state(const hybrid_engine_state& st);
    int write_state_kv(const hybrid_engine_state& st);
    bool state_fits(const hybrid_engine_state& st) const;
    /// \brief one decode token on the NPU (host fallback if a run fails): logits_f32
    void decode_npu(int id);
    /// \brief the device's tokens back into the host model (call before the host model is read or changed)
    void sync_host();
    /// \brief (re)create the NPU decode for MAX_L (needs the prefill's weight buffers); INKLING2_NPU_DECODE=0, or a
    ///        context too long for its buffers, leaves decode on the host
    void make_fused();
};
