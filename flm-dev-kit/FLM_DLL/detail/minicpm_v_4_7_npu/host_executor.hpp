// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __MINICPM_V_HOST_EXECUTOR_HPP__
#define __MINICPM_V_HOST_EXECUTOR_HPP__
#pragma once

#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>

namespace mcv {

/// \brief One persistent thread that runs every prefill, so the OpenMP team the host stages use is
///        created once for the process.
/// \note  libgomp keeps its thread pool per *master* thread. FLM calls prefill() from whichever of its
///        I/O threads took the request, so each of those threads grew its own pool of workers the first
///        time it ran a prefill (flm thread count 35 -> 46 -> 57 -> 68 ... until every I/O thread had
///        one), and a given request could land on a pool with cold caches and a different set of cores.
///        Funnelling the call through one thread bounds it at one pool, and the pool stays warm between
///        requests. It is plain std::thread + condition_variable: no affinity, no OMP environment and no
///        platform API, so correctness does not depend on any of them (same behaviour on Windows).
///        The caller blocks until the work is done, so semantics are those of a direct call, including
///        exceptions, which are re-thrown on the caller's thread. Callers are serialised.
class HostExecutor {
public:
    static HostExecutor& instance() {
        static HostExecutor ex;
        return ex;
    }

    /// Run `fn` on the worker thread and wait for it. Runs `fn` inline when already on the worker.
    void run(const std::function<void()>& fn) {
        if (std::this_thread::get_id() == worker_id_) { fn(); return; }
        std::unique_lock<std::mutex> caller(call_mtx_);     // one job at a time
        std::exception_ptr err;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            job_ = &fn;
            err_ = &err;
            done_ = false;
        }
        cv_job_.notify_one();
        std::unique_lock<std::mutex> lk(mtx_);
        cv_done_.wait(lk, [this] { return done_; });
        if (err) std::rethrow_exception(err);
    }

    HostExecutor(const HostExecutor&) = delete;
    HostExecutor& operator=(const HostExecutor&) = delete;

private:
    HostExecutor() {
        worker_ = std::thread([this] { loop(); });
        worker_id_ = worker_.get_id();
    }
    ~HostExecutor() {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            stop_ = true;
        }
        cv_job_.notify_one();
        if (worker_.joinable()) worker_.join();
    }

    void loop() {
        std::unique_lock<std::mutex> lk(mtx_);
        for (;;) {
            cv_job_.wait(lk, [this] { return stop_ || job_ != nullptr; });
            if (stop_ && job_ == nullptr) return;
            const std::function<void()>* job = job_;
            std::exception_ptr* err = err_;
            job_ = nullptr;
            lk.unlock();
            try { (*job)(); } catch (...) { *err = std::current_exception(); }
            lk.lock();
            done_ = true;
            cv_done_.notify_one();
        }
    }

    std::mutex call_mtx_;
    std::mutex mtx_;
    std::condition_variable cv_job_, cv_done_;
    const std::function<void()>* job_ = nullptr;
    std::exception_ptr* err_ = nullptr;
    bool done_ = false;
    bool stop_ = false;
    std::thread worker_;
    std::thread::id worker_id_;
};

} // namespace mcv

#endif
