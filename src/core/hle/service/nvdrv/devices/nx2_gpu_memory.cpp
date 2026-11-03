// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdlib>
#include <memory>
#include <vector>

#include "common/alignment.h"
#include "common/logging.h"
#include "core/core.h"
#include "core/hle/kernel/k_memory_manager.h"
#include "core/hle/kernel/k_page_group.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_system_resource.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/service/nvdrv/devices/nx2_gpu_memory.h"
#include "core/hle/service/nvdrv/devices/nx2_stub_device.h"
#include "core/memory.h"

namespace Service::Nvidia::Devices {

Nx2MeshScratch GetNx2MeshScratch() {
    const auto scratch = Nx2GpuMemoryRegistry::Get().MeshScratch();
    return {scratch.gpu_address, scratch.size};
}

void EnsureNx2MeshScratch(Core::System& system) {
    using Core::Memory::YUZU_PAGESIZE;
    // Users of the scratch: the task/mesh emulation and the batched chunk draws.
    static const bool enabled = [] {
        for (const char* name : {"EDEN_NX2_MESH_EMU", "EDEN_NX2_CHUNK_BATCH"}) {
            const char* value = std::getenv(name);
            if (value != nullptr && value[0] == '1') {
                return true;
            }
        }
        return false;
    }();
    if (!enabled) {
        return;
    }
    auto& registry = Nx2GpuMemoryRegistry::Get();
    if (registry.MeshScratch().size != 0) {
        return;
    }
    // EDEN_NX2_MESH_EMU_MIB: scratch size; one frame's task/mesh draws ring through it.
    static const u64 size = [] {
        const char* value = std::getenv("EDEN_NX2_MESH_EMU_MIB");
        const u64 mib = value ? std::strtoull(value, nullptr, 10) : 64;
        return std::clamp<u64>(mib, 4, 1024) << 20;
    }();
    auto* const process = system.ApplicationProcess();
    if (process == nullptr) {
        return;
    }
    auto& kernel = system.Kernel();
    auto pg = std::make_unique<Kernel::KPageGroup>(
        kernel, std::addressof(kernel.GetSystemSystemResource().GetBlockInfoManager()));
    // The graphics server's memory, like the GPU-only regions of nx2:gpu sub 214.
    if (R_FAILED(kernel.MemoryManager().AllocateAndOpen(
            pg.get(), size / YUZU_PAGESIZE,
            Kernel::KMemoryManager::EncodeOption(Kernel::KMemoryManager::Pool::SystemNonSecure,
                                                 Kernel::KMemoryManager::Direction::FromBack)))) {
        LOG_ERROR(Service_NVDRV, "Task/mesh scratch: could not allocate {:#x} bytes", size);
        return;
    }
    // CPU side: the alias code region, from its end (sub 214 backings fill it from the start).
    auto& page_table = process->GetPageTable();
    const u64 region_start = GetInteger(page_table.GetAliasCodeRegionStart());
    const u64 region_size = page_table.GetAliasCodeRegionSize();
    constexpr u64 Alignment = 0x200000;
    u64 cpu_address = 0;
    for (u64 offset = Common::AlignDown(region_size - size, Alignment); offset >= Alignment;
         offset -= Alignment) {
        const Common::ProcessAddress address{region_start + offset};
        if (R_SUCCEEDED(page_table.MapPageGroup(address, *pg, Kernel::KMemoryState::IoMemory,
                                                Kernel::KMemoryPermission::UserReadWrite))) {
            cpu_address = GetInteger(address);
            break;
        }
    }
    if (cpu_address == 0) {
        LOG_ERROR(Service_NVDRV, "Task/mesh scratch: no room for {:#x} bytes in the alias code region", size);
        pg->Close(kernel);
        return;
    }
    // GPU side: near the top of the 40-bit space, away from the client's identity mappings and
    // the addresses its allocator hands out; stepped down if something is already there.
    u64 gpu_address = 0xfe'0000'0000ULL;
    while (gpu_address >= 0xf0'0000'0000ULL && registry.IsGpuRangeMapped(gpu_address, size)) {
        gpu_address -= size;
    }
    if (gpu_address < 0xf0'0000'0000ULL) {
        LOG_ERROR(Service_NVDRV, "Task/mesh scratch: no free GPU range for {:#x} bytes", size);
        return;
    }
    // Kept for the session and deliberately never destroyed: a static vector would finalize the
    // page group at exit, after the kernel is gone.
    static auto& page_groups = *new std::vector<std::unique_ptr<Kernel::KPageGroup>>();
    page_groups.push_back(std::move(pg));
    registry.SetMeshScratch({gpu_address, cpu_address, size});
    registry.MapAt(gpu_address, cpu_address, size);
    LOG_INFO(Service_NVDRV, "Task/mesh scratch: {:#x} bytes at GPU {:#x} (CPU {:#x})", size,
             gpu_address, cpu_address);
}

} // namespace Service::Nvidia::Devices
