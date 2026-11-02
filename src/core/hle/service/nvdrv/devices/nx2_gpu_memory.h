// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"

namespace Core {
class System;
}

namespace Service::Nvidia::Devices {

// True inside a Switch 2 GPU-only region (nx2:gpu sub 214): the client never maps those, so
// nothing the GPU writes there is read back through host memory. Declared apart from
// nx2_stub_device.h so video_core can ask without pulling the service headers (whose NvResult
// names collide with X11 macros in renderer translation units).
bool IsNx2GpuOnlyAddress(u64 gpu_address);

// The emulator's own guest-visible memory for the task/mesh emulation (EDEN_NX2_MESH_EMU=1;
// video_core/nx2_mesh_emulation.h) and the batched chunk draws (EDEN_NX2_CHUNK_BATCH=1): kernel
// pages mapped into the application and at a fixed GPU address in every Switch 2 GPU address
// space. Zero until the address space exists.
struct Nx2MeshScratch {
    u64 gpu_address{};
    u64 size{};
};
Nx2MeshScratch GetNx2MeshScratch();

// Allocates and maps the scratch once per session; a no-op unless a user of it is enabled.
void EnsureNx2MeshScratch(Core::System& system);

} // namespace Service::Nvidia::Devices
