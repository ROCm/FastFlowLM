/// \file test_device_load.cpp
/// \brief Adapter counters are keyed by physical engine, not its descriptive type.
#include "device_load.hpp"

#include <array>
#include <iostream>
#include <limits>
#include <string_view>

int main() {
    using namespace device_load;
    constexpr AdapterId gpu{0, 0xd80bb6be, 0};
    if (single_adapter(std::array{Candidate{false, 1}, Candidate{true, 1}}) != 1 ||
        single_adapter(std::array{Candidate{true, 1}, Candidate{true, 1}}) ||
        single_adapter(std::array{Candidate{true, 2}}) || single_adapter(std::array{Candidate{true, 0}}) ||
        single_adapter(std::array{Candidate{false, 1}})) return 1;
    // A system that does not report PhysicalAdapterCount still has one adapter to read.
    if (single_adapter(std::array{Candidate{false, 1}, Candidate{true, std::nullopt}}) != 1 ||
        single_adapter(std::array{Candidate{true, std::nullopt}, Candidate{true, 1}})) return 1;
    const auto reason = [](auto candidates) { return std::string_view(no_adapter_reason(candidates)); };
    if (reason(std::array{Candidate{false, 1}}) != "no hardware adapter" ||
        reason(std::array{Candidate{true, 1}, Candidate{true, 1}}) != "several hardware adapters" ||
        reason(std::array{Candidate{true, 2}}) != "a linked adapter with several physical nodes") return 1;
    if (parse_adapter(L"luid_0x00000000_0xD80BB6BE_phys_0") != gpu ||
        parse_adapter(L"luid_0x00000000_0xD80BB6BE_phys_0suffix") ||
        parse_adapter(L"luid_0x100000000_0xD80BB6BE_phys_0")) return 1;

    EngineUsage usage(7);
    usage.add(L"pid_7_luid_0x0_0xd80bb6be_phys_0_eng_2_engtype_Copy", 30, true);
    usage.add(L"pid_8_luid_0x0_0xd80bb6be_phys_0_eng_2_engtype_Copy", 40, true);
    usage.add(L"pid_7_luid_0x0_0xd80bb6be_phys_0_eng_3_engtype_Copy", 42, true);
    usage.add(L"pid_8_luid_0x0_0xd80bb6be_phys_1_eng_2_engtype_Unknown", 95, true);
    usage.add(L"pid_8_luid_0x1_0xd80bb6be_phys_0_eng_2_engtype_Compute", 130, true);
    const auto [total, own] = usage.get(gpu);
    if (total != 70 || own != 42) {
        std::cerr << "expected overall 70 and own 42 on different busiest engines\n";
        return 1;
    }
    const auto [external, idle] = usage.get({1, gpu.low, 0});
    if (external != 100 || idle != 0 || usage.get({0, 1234, 0}).total) return 1;
    usage.add(L"pid_7_luid_0x0_0xd80bb6be_phys_1_eng_0_engtype_Copy", 0, false);
    if (usage.get({0, gpu.low, 1}).own || usage.get(gpu).total != 70) return 1;
    if (!usage.linked(gpu) || usage.linked({1, gpu.low, 0})) return 1;
    // A counter set with no valid value for this adapter does not prove the observer was idle.
    usage.add(L"pid_8_luid_0x0_0x1234_phys_0_eng_0_engtype_Copy", 0, false);
    if (usage.get({0, 0x1234, 0}).own) return 1;
    EngineUsage missing(7);
    for (auto malformed : {L"pid_7_luid_0x0_0xd80bb6be_phys_0_eng_0_engtype_",
                           L"pid_4294967296_luid_0x0_0xd80bb6be_phys_0_eng_0_engtype_Copy",
                           L"pid_7_luid_0x0_0xd80bb6be_phys_0_engtype_Copy"}) missing.add(malformed, 88, true);
    if (missing.get(gpu).total || missing.get(gpu).own) return 1;
    for (double invalid : {-1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        EngineUsage partial(7);
        partial.add(L"pid_7_luid_0x0_0xd80bb6be_phys_0_eng_1_engtype_Compute", 70, true);
        partial.add(L"pid_8_luid_0x0_0xd80bb6be_phys_0_eng_0_engtype_Copy", invalid, true);
        if (partial.get(gpu).total || partial.get(gpu).own != 70) return 1;
    }
#ifndef _WIN32
    auto snapshot = make_reader()();
    if (snapshot.sampled_at || snapshot.available || snapshot.gpu.name || snapshot.npu.name ||
        snapshot.gpu.usage.total || snapshot.npu.usage.total || snapshot.gpu.shared || snapshot.npu.dedicated) return 1;
#endif
}
