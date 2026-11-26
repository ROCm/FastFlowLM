/// \file device_load_windows.cpp
/// \brief On-demand Windows accelerator sampling. Only the worker calls DXCore and PDH.
#ifdef _WIN32
#include "device_load.hpp"
#include <windows.h>
#include <pdh.h>
#include <pdhmsg.h>
#if __has_include(<dxcore.h>)
#include <initguid.h>
#include <dxcore.h>
#include <wrl/client.h>
#endif

#include <array>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace device_load {
namespace {
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;
constexpr auto max_age = 3s;

struct Identity { std::optional<AdapterId> id; std::optional<std::string> name; };

#if __has_include(<dxcore.h>)
using Microsoft::WRL::ComPtr;
using Property = DXCoreAdapterProperty;
// Stable DXCore attributes; older SDKs predate their declarations.
#ifdef NTDDI_WIN11_GE
const GUID& gpu_kind = DXCORE_HARDWARE_TYPE_ATTRIBUTE_GPU;
const GUID& npu_kind = DXCORE_HARDWARE_TYPE_ATTRIBUTE_NPU;
#else
constexpr GUID gpu_kind{0xb69eb219,0x3ded,0x4464,{0x97,0x9f,0xa0,0x0b,0xd4,0x68,0x70,0x06}};
constexpr GUID npu_kind{0xd46140c4,0xadd7,0x451b,{0x9e,0x56,0x06,0xfe,0x8c,0x3b,0x58,0xed}};
#endif

class AdapterList {
    ComPtr<IDXCoreAdapterList> list_;
    Identity selected_;
public:
    Identity select(IDXCoreAdapterFactory* factory, const GUID& kind) {
        if (list_ && !list_->IsStale()) return selected_;
        ComPtr<IDXCoreAdapterList> list;
        if (FAILED(factory->CreateAdapterList(1, &kind, list.GetAddressOf()))) return {};
        std::vector<Candidate> candidates;
        Identity candidate;
        for (unsigned i = 0; i < list->GetAdapterCount(); ++i) {
            ComPtr<IDXCoreAdapter> adapter;
            bool hardware = false;
            if (FAILED(list->GetAdapter(i, adapter.GetAddressOf())) ||
                FAILED(adapter->GetProperty(Property::IsHardware, &hardware))) return {};
            unsigned nodes = 0;
            constexpr auto physical_count = static_cast<Property>(15); // PhysicalAdapterCount
            if (hardware && adapter->IsPropertySupported(physical_count) &&
                FAILED(adapter->GetProperty(physical_count, &nodes))) return {};
            candidates.push_back({hardware, nodes});
        }
        if (const auto index = single_adapter(candidates)) {
            ComPtr<IDXCoreAdapter> adapter;
            LUID luid{};
            if (FAILED(list->GetAdapter(static_cast<unsigned>(*index), adapter.GetAddressOf())) ||
                FAILED(adapter->GetProperty(Property::InstanceLuid, &luid))) return {};
            candidate.id = AdapterId{static_cast<uint32_t>(luid.HighPart), luid.LowPart, 0};
            size_t size = 0;
            if (SUCCEEDED(adapter->GetPropertySize(Property::DriverDescription, &size)) && size && size <= 4096) {
                std::vector<char> name(size + 1);
                if (SUCCEEDED(adapter->GetProperty(Property::DriverDescription, size, name.data())))
                    candidate.name = name.data(); // DXCore specifies UTF-8.
            }
        }
        list_ = std::move(list); // cache ambiguity, not transient enumeration errors
        return selected_ = std::move(candidate);
    }
};

class Inventory {
    std::unique_ptr<std::remove_pointer_t<HMODULE>, decltype(&FreeLibrary)> module_{
        LoadLibraryExW(L"dxcore.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32), FreeLibrary};
    ComPtr<IDXCoreAdapterFactory> factory_;
    AdapterList gpu_, npu_;
public:
    std::pair<Identity, Identity> read() {
        if (!factory_) {
            using Create = HRESULT (WINAPI*)(REFIID, void**);
            auto create = module_ ? std::bit_cast<Create>(GetProcAddress(module_.get(), "DXCoreCreateAdapterFactory")) : nullptr;
            if (!create || FAILED(create(IID_PPV_ARGS(factory_.GetAddressOf())))) return {};
        }
        return {gpu_.select(factory_.Get(), gpu_kind), npu_.select(factory_.Get(), npu_kind)};
    }
};
#else
struct Inventory { std::pair<Identity, Identity> read() const { return {}; } };
#endif

// The wildcard set can change between sizing the array and reading it. Retry with a new size.
template <typename Fn>
void rows(PDH_HCOUNTER counter, DWORD format, Fn consume) {
    if (!counter) return;
    for (int attempt = 0; attempt < 3; ++attempt) {
        DWORD bytes = 0, count = 0;
        if (PdhGetFormattedCounterArrayW(counter, format, &bytes, &count, nullptr) != PDH_MORE_DATA ||
            !bytes || bytes > 32 * 1024 * 1024) return;
        std::vector<PDH_FMT_COUNTERVALUE_ITEM_W> data((bytes + sizeof(data[0]) - 1) / sizeof(data[0]));
        const auto rc = PdhGetFormattedCounterArrayW(counter, format, &bytes, &count, data.data());
        if (rc == PDH_MORE_DATA) continue;
        if (rc == ERROR_SUCCESS) {
            for (DWORD i = 0; i < count; ++i) {
                const auto& item = data[i];
                if (item.szName) consume(item.szName, item.FmtValue, item.FmtValue.CStatus == PDH_CSTATUS_VALID_DATA ||
                                                                    item.FmtValue.CStatus == PDH_CSTATUS_NEW_DATA);
            }
        }
        return;
    }
}

class Provider {
    PDH_HQUERY query_{};
    std::array<PDH_HCOUNTER, 3> counters_{};
    std::optional<Clock::time_point> previous_;
    bool reported_missing_ = false;
    Inventory inventory_;
public:
    ~Provider() {
        if (query_) PdhCloseQuery(query_);
    }
    Snapshot read() {
        Snapshot s;
        SYSTEMTIME utc; GetSystemTime(&utc);
        char stamp[32];
        std::snprintf(stamp, sizeof(stamp), "%04u-%02u-%02uT%02u:%02u:%02u.%03uZ", utc.wYear, utc.wMonth,
                      utc.wDay, utc.wHour, utc.wMinute, utc.wSecond, utc.wMilliseconds);
        s.sampled_at = stamp;
        MEMORYSTATUSEX mem{}; mem.dwLength = sizeof(mem);
        if (GlobalMemoryStatusEx(&mem)) s.available = mem.ullAvailPhys;
        const auto [gpu, npu] = inventory_.read();
        s.gpu.name = gpu.name; s.npu.name = npu.name;
        if ((!gpu.id || !npu.id) && !reported_missing_) {
            std::fprintf(stderr, "FLM device telemetry: no unambiguous %s adapter; unavailable readings are null.\n",
                         !gpu.id && !npu.id ? "GPU or NPU" : !gpu.id ? "GPU" : "NPU");
            reported_missing_ = true;
        }
        if (!gpu.id && !npu.id) return s;
        if (!query_ && PdhOpenQueryW(nullptr, 0, &query_) != ERROR_SUCCESS) return s;
        constexpr const wchar_t* paths[] = {LR"(\GPU Engine(*)\Utilization Percentage)",
            LR"(\GPU Adapter Memory(*)\Shared Usage)", LR"(\GPU Adapter Memory(*)\Dedicated Usage)"};
        for (size_t i = 0; i < counters_.size(); ++i)
            if (!counters_[i]) PdhAddEnglishCounterW(query_, paths[i], 0, &counters_[i]);
        const auto now = Clock::now();
        if (PdhCollectQueryData(query_) != ERROR_SUCCESS) {
            PdhCloseQuery(query_); query_ = nullptr; counters_ = {}; previous_.reset();
            return s;
        }
        if (previous_ && now - *previous_ <= max_age) {
            EngineUsage usage(GetCurrentProcessId());
            rows(counters_[0], PDH_FMT_DOUBLE, [&](auto name, const auto& v, bool valid) {
                usage.add(name, valid ? v.doubleValue : 0, valid);
            });
            if (gpu.id) s.gpu.usage = usage.get(*gpu.id);
            if (npu.id) s.npu.usage = usage.get(*npu.id);
        }
        previous_ = now;
        for (size_t i = 1; i <= 2; ++i) {
            rows(counters_[i], PDH_FMT_LARGE, [&](auto name, const auto& v, bool valid) {
                const auto id = parse_adapter(name);
                if (!id || !valid || v.largeValue < 0) return;
                auto* d = id == gpu.id ? &s.gpu : id == npu.id ? &s.npu : nullptr;
                if (d) (i == 1 ? d->shared : d->dedicated) = static_cast<uint64_t>(v.largeValue);
            });
        }
        return s;
    }
};

class Sampler {
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread worker_;
    bool stop_ = false;
    Snapshot latest_;
    std::optional<Clock::time_point> sampled_;
    Clock::time_point last_poll_{};
    void run() {
        Provider provider;
        std::unique_lock lock(mutex_);
        while (true) {
            cv_.wait(lock, [&] { return stop_ || Clock::now() - last_poll_ < 6s; });
            if (stop_) return;
            lock.unlock();
            const auto start = Clock::now();
            Snapshot next;
            try { next = provider.read(); } catch (const std::exception&) { /* best-effort telemetry */ }
            lock.lock();
            latest_ = std::move(next);
            sampled_ = start;
            if (cv_.wait_for(lock, 1s, [&] { return stop_; })) return;
        }
    }
public:
    ~Sampler() {
        { std::lock_guard lock(mutex_); stop_ = true; }
        cv_.notify_one();
        if (worker_.joinable()) worker_.join();
    }
    Snapshot get() {
        std::lock_guard lock(mutex_);
        const auto now = Clock::now();
        last_poll_ = now;
        if (!worker_.joinable()) {
            try { worker_ = std::thread(&Sampler::run, this); }
            catch (const std::system_error&) { return {}; }
        }
        cv_.notify_one();
        return sampled_ && now - *sampled_ <= max_age ? latest_ : Snapshot{};
    }
};
} // namespace
std::function<Snapshot()> make_reader() {
    return [sampler = std::make_shared<Sampler>()] { return sampler->get(); };
}
} // namespace device_load
#endif
