// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file image/npu/vision_lane.hpp
/// \brief One worker thread that runs the tower's device dispatches in order, so
///        the calling thread can do host work while the NPU is busy.
///
/// Plain C++ threading (std::thread, mutex, condition_variable, packaged_task):
/// no affinity, no OpenMP and nothing platform specific. Jobs run strictly in
/// submission order, one at a time, which is also all the NPU does with them, and
/// which is what makes the staging-slot discipline in `mm_op` sufficient: a slot
/// is touched by the host only while no job on it is queued or running.
///
/// Lifetime: the owner must not destroy the lane (or anything a queued job
/// touches) with jobs outstanding. The tower guarantees that by draining every
/// future it holds before leaving a streamed stretch, exception or not.
#ifndef __MINICPM_V_4_7_NPU_VISION_LANE_HPP__
#define __MINICPM_V_4_7_NPU_VISION_LANE_HPP__

#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <thread>

namespace mcv_npu {

class dispatch_lane {
public:
    dispatch_lane() = default;
    dispatch_lane(const dispatch_lane&) = delete;
    dispatch_lane& operator=(const dispatch_lane&) = delete;

    ~dispatch_lane() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        if (th_.joinable()) th_.join();
    }

    /// Queue `fn`; the future carries its exception, if any. The worker thread is
    /// started on first use, so a process that never streams never has one.
    std::future<void> submit(std::function<void()> fn) {
        std::packaged_task<void()> task(std::move(fn));
        std::future<void> f = task.get_future();
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (!th_.joinable()) th_ = std::thread([this] { _loop(); });
            q_.push_back(std::move(task));
        }
        cv_.notify_one();
        return f;
    }

private:
    void _loop() {
        for (;;) {
            std::packaged_task<void()> task;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this] { return stop_ || !q_.empty(); });
                if (q_.empty()) return;   // stop_ and drained
                task = std::move(q_.front());
                q_.pop_front();
            }
            task();
        }
    }

    std::mutex                           mu_;
    std::condition_variable              cv_;
    std::deque<std::packaged_task<void()>> q_;
    std::thread                          th_;
    bool                                 stop_ = false;
};

}  // namespace mcv_npu

#endif  // __MINICPM_V_4_7_NPU_VISION_LANE_HPP__
