// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __D3_PAR_HPP__
#define __D3_PAR_HPP__

/// \file d3_par.hpp
/// \brief The host worker pool of the d3 engine: fork / join with workers that spin briefly, then sleep.
///
/// Why not OpenMP. A d3 prefill alternates short host loops with blocking NPU dispatches (~1 ms each,
/// about 24 layers of them). An OpenMP team keeps spinning through every dispatch, so the idle workers
/// of a 12-thread team burn ~11 cores for a request that needs ~4 (FLM issue #882; libgomp and the
/// Windows OpenMP runtime both do it, and neither lets a loaded library change the wait policy: libgomp
/// reads GOMP_SPINCOUNT before any library constructor runs). OpenMP also gives every calling thread
/// its own team, and the server calls from ten I/O threads, so a request lands on a cold team of ~11
/// threads, and a long-lived server holds ~120.
///
/// This pool is one set of threads for the whole process. A region is: wake the workers, run the
/// caller's share as tid 0, wait for the others. Between regions a worker spins for D3_PAR_SPIN_US
/// microseconds (default 30: longer than the gap between two host loops, far shorter than an NPU wait),
/// then sleeps on a futex / WaitOnAddress until the next region.
///
/// Regions run one at a time (callers take a lock), and a region's body must not start another region.
/// The partitioning is static, like `schedule(static)`; every loop here writes disjoint outputs, so the
/// results do not depend on it.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>
#ifdef D3_PAR_USE_OMP
#include <omp.h>
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#define D3_PAR_PAUSE() _mm_pause()
#elif defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define D3_PAR_PAUSE() _mm_pause()
#else
#define D3_PAR_PAUSE() std::this_thread::yield()
#endif

namespace d3_par {

/// Microseconds a waiting thread spins before it sleeps. D3_PAR_SPIN_US overrides; 0 = sleep at once.
inline int spin_us() {
    static const int us = [] {
        const char* e = std::getenv("D3_PAR_SPIN_US");
        const int v = e ? std::atoi(e) : 30;
        return v < 0 ? 0 : v;
    }();
    return us;
}

/// Spin until `done()` holds, then fall back to `sleep()` (which blocks until something changed) in a loop.
template <class Done, class Sleep>
inline void wait_for(Done&& done, Sleep&& sleep) {
    const auto limit = std::chrono::microseconds(spin_us());
    if (limit.count() > 0) {
        const auto t0 = std::chrono::steady_clock::now();
        for (unsigned i = 1; !done(); i++) {
            D3_PAR_PAUSE();
            if ((i & 31u) == 0 && std::chrono::steady_clock::now() - t0 > limit) break;
        }
    }
    while (!done()) sleep();
}

class Pool {
public:
    using Body = void (*)(void* ctx, int tid);

    Pool() = default;
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;
    ~Pool() { shutdown(); }

    /// Run body(ctx, tid) for tid in [0, nt) on nt threads; the caller is tid 0. Returns when all are done.
    /// A body that throws makes the region rethrow the first exception in the caller.
    void run(int nt, Body body, void* ctx) {
        std::lock_guard<std::mutex> region(region_);
        grow(nt - 1);
        body_ = body;
        ctx_ = ctx;
        nt_ = nt;
        error_ = nullptr;
        pending_.store((int)workers_.size(), std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_release);
        gen_.notify_all();

        try { body(ctx, 0); }
        catch (...) { record_error(); }

        wait_for([&] { return pending_.load(std::memory_order_acquire) == 0; },
                 [&] { const int v = pending_.load(std::memory_order_acquire); if (v != 0) pending_.wait(v); });
        if (error_) std::rethrow_exception(error_);
    }

    /// Stop and join every worker. The pool starts again on the next run().
    void shutdown() {
        std::lock_guard<std::mutex> region(region_);
        if (workers_.empty()) return;
        stop_.store(true, std::memory_order_release);
        gen_.fetch_add(1, std::memory_order_release);
        gen_.notify_all();
        for (auto& t : workers_) t.join();
        workers_.clear();
        stop_.store(false, std::memory_order_release);
    }

private:
    // Called with region_ held, before the first region of a width: add workers up to n.
    void grow(int n) {
        while ((int)workers_.size() < n) {
            const int tid = (int)workers_.size() + 1;
            // a worker starts from the generation current now, so it ignores regions that ran before it existed
            const uint32_t born = gen_.load(std::memory_order_acquire);
            workers_.emplace_back([this, tid, born] { worker(tid, born); });
        }
    }

    void worker(int tid, uint32_t seen) {
        for (;;) {
            wait_for([&] { return gen_.load(std::memory_order_acquire) != seen; },
                     [&] { gen_.wait(seen); });
            seen = gen_.load(std::memory_order_acquire);
            if (stop_.load(std::memory_order_acquire)) return;
            if (tid < nt_) {
                try { body_(ctx_, tid); }
                catch (...) { record_error(); }
            }
            // every worker reports, also the ones this region does not use: run() may only publish the
            // next region once nobody still reads this one's fields
            if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1) pending_.notify_one();
        }
    }

    void record_error() {
        std::lock_guard<std::mutex> lock(error_mutex_);
        if (!error_) error_ = std::current_exception();
    }

    std::mutex region_;
    std::vector<std::thread> workers_;
    std::atomic<uint32_t> gen_{0};
    std::atomic<int> pending_{0};
    std::atomic<bool> stop_{false};
    Body body_ = nullptr;
    void* ctx_ = nullptr;
    int nt_ = 0;
    std::mutex error_mutex_;
    std::exception_ptr error_;
};

/// The process-wide pool (one per library: this header is only included by the d3 engine).
inline Pool& pool() {
    static Pool p;
    return p;
}

/// Run f(tid) on nt threads (tid 0 is the caller). nt <= 1 runs f(0) inline.
template <class F>
inline void team(int nt, F&& f) {
    if (nt <= 1) { f(0); return; }
#ifdef D3_PAR_USE_OMP   // A/B against OpenMP with the same loop bodies (numerics checks)
    #pragma omp parallel num_threads(nt)
    f(omp_get_thread_num());
    return;
#endif
    auto* fp = &f;
    pool().run(nt, [](void* ctx, int tid) { (*static_cast<decltype(fp)>(ctx))(tid); }, fp);
}

/// f(i) for i in [0, n), split into contiguous blocks over up to nt threads (omp `schedule(static)`).
template <class F>
inline void for_static(int nt, long long n, F&& f) {
    if (n <= 0) return;
    const int t = (int)std::min<long long>(nt, n);
    if (t <= 1) { for (long long i = 0; i < n; i++) f(i); return; }
    team(t, [&](int tid) {
        const long long lo = n * tid / t, hi = n * (tid + 1) / t;
        for (long long i = lo; i < hi; i++) f(i);
    });
}

/// f(tid, i) for i in [0, n), threads taking `chunk` consecutive i at a time as they come free
/// (omp `schedule(dynamic, chunk)`). tid is the thread's index in [0, nt): per-thread scratch.
template <class F>
inline void for_dynamic(int nt, long long n, long long chunk, F&& f) {
    if (n <= 0) return;
    if (chunk < 1) chunk = 1;
    const int t = (int)std::min<long long>(nt, (n + chunk - 1) / chunk);
    std::atomic<long long> next{0};
    team(t, [&](int tid) {
        for (;;) {
            const long long lo = next.fetch_add(chunk, std::memory_order_relaxed);
            if (lo >= n) break;
            const long long hi = std::min(n, lo + chunk);
            for (long long i = lo; i < hi; i++) f(tid, i);
        }
    });
}

}  // namespace d3_par

#endif  // __D3_PAR_HPP__
