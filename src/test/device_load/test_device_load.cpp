/// \file test_device_load.cpp
/// \brief Adapter counters are keyed by physical engine, not its descriptive type.
#include "device_load.hpp"

#include <iostream>

int main() {
    using namespace device_load;
    constexpr AdapterId gpu{0, 0xd80bb6be, 0};
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
#ifndef _WIN32
    auto snapshot = make_reader()();
    if (snapshot.sampled_at || snapshot.available || snapshot.gpu.name || snapshot.npu.name ||
        snapshot.gpu.usage.total || snapshot.npu.usage.total || snapshot.gpu.shared || snapshot.npu.dedicated) return 1;
#endif
}
