// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cmath>
#include <array>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <tuple>
#include <set>
#include <fstream>
#include <mutex>

#include <fmt/format.h>

#include "video_core/nx2.h"
#include "video_core/renderer_vulkan/renderer_vulkan.h"

#include <unordered_set>
#include "common/alignment.h"
#include "common/assert.h"
#include "common/string_util.h"
#include "shader_recompiler/exception.h"
#include "common/logging.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/hle/service/nvdrv/devices/nx2_gpu_memory.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/gpu_logging/gpu_logging.h"
#include "video_core/control/channel_state.h"
#include "video_core/native_query_trace.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/renderer_vulkan/blit_image.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_query_cache.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/shader_cache.h"
#include "video_core/texture_cache/texture_cache_base.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/nx2_profile.h"

namespace Vulkan {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;
using VideoCommon::ImageViewId;
using VideoCommon::ImageViewType;


namespace {
struct DrawParams {
    u32 base_instance;
    u32 num_instances;
    u32 base_vertex;
    u32 num_vertices;
    u32 first_index;
    bool is_indexed;
};

VkViewport GetViewportState(const Device& device, const Maxwell& regs, size_t index, float scale) {
    const auto& src = regs.viewport_transform[index];
    const auto conv = [scale](float value) {
        float new_value = value * scale;
        if (scale < 1.0f) {
            const bool sign = std::signbit(value);
            new_value = std::round(std::abs(new_value));
            new_value = sign ? -new_value : new_value;
        }
        return new_value;
    };
    const float x = conv(src.translate_x - src.scale_x);
    const float width = conv(src.scale_x * 2.0f);
    float y = conv(src.translate_y - src.scale_y);
    float height = conv(src.scale_y * 2.0f);

    const bool lower_left = regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft;
    const bool y_negate = !device.IsNvViewportSwizzleSupported() &&
                          src.swizzle.y == Maxwell::ViewportSwizzle::NegativeY;

    if (lower_left) {
        // Flip by surface clip height
        y += conv(static_cast<f32>(regs.surface_clip.height));
        height = -height;
    }

    if (y_negate) {
        // Flip by viewport height
        y += height;
        height = -height;
    }

    const float reduce_z = regs.depth_mode == Maxwell::DepthMode::MinusOneToOne ? 1.0f : 0.0f;
    VkViewport viewport{
        .x = x,
        .y = y,
        .width = width != 0.0f ? width : 1.0f,
        .height = height != 0.0f ? height : 1.0f,
        .minDepth = src.translate_z - src.scale_z * reduce_z,
        .maxDepth = src.translate_z + src.scale_z,
    };
    if (!device.IsExtDepthRangeUnrestrictedSupported()) {
        viewport.minDepth = std::clamp(viewport.minDepth, 0.0f, 1.0f);
        viewport.maxDepth = std::clamp(viewport.maxDepth, 0.0f, 1.0f);
    }
    return viewport;
}

VkRect2D GetScissorState(const Maxwell& regs, size_t index, u32 up_scale = 1, u32 down_shift = 0) {
    const auto& src = regs.scissor_test[index];
    VkRect2D scissor{};
    const auto scale_up = [&](s32 value) -> s32 {
        if (value == 0) {
            return 0U;
        }
        const s32 upset = value * up_scale;
        s32 acumm = 0;
        if ((up_scale >> down_shift) == 0) {
            acumm = upset % 2;
        }
        const s32 converted_value = (value * up_scale) >> down_shift;
        return value < 0 ? std::min<s32>(converted_value - acumm, -1)
                         : std::max<s32>(converted_value + acumm, 1);
    };

    const bool lower_left = regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft;
    const s32 clip_height = regs.surface_clip.height;

    // Flip coordinates if lower left
    s32 min_y = lower_left ? (clip_height - src.max_y) : src.min_y.Value();
    s32 max_y = lower_left ? (clip_height - src.min_y) : src.max_y.Value();

    // Bound to render area
    min_y = (std::max)(min_y, 0);
    max_y = (std::max)(max_y, 0);

    if (src.enable) {
        scissor.offset.x = scale_up(src.min_x);
        scissor.offset.y = scale_up(min_y);
        scissor.extent.width = scale_up(src.max_x - src.min_x);
        scissor.extent.height = scale_up(max_y - min_y);
    } else {
        scissor.offset.x = 0;
        scissor.offset.y = 0;
        scissor.extent.width = (std::numeric_limits<s32>::max)();
        scissor.extent.height = (std::numeric_limits<s32>::max)();
    }
    return scissor;
}

DrawParams MakeDrawParams(const Tegra::Engines::Maxwell3D::DrawManager::State& draw_state, u32 num_instances, bool is_indexed) {
    DrawParams params{
        .base_instance = draw_state.base_instance,
        .num_instances = num_instances,
        .base_vertex = is_indexed ? draw_state.base_index : draw_state.vertex_buffer.first,
        .num_vertices = is_indexed ? draw_state.index_buffer.count : draw_state.vertex_buffer.count,
        .first_index = is_indexed ? draw_state.index_buffer.first : 0,
        .is_indexed = is_indexed,
    };
    // 6 triangle vertices per quad, base vertex is part of the index
    // See BindQuadIndexBuffer for more details
    if (draw_state.topology == Maxwell::PrimitiveTopology::Quads) {
        params.num_vertices = (params.num_vertices / 4) * 6;
        params.base_vertex = 0;
        params.is_indexed = true;
    } else if (draw_state.topology == Maxwell::PrimitiveTopology::QuadStrip) {
        params.num_vertices = (params.num_vertices - 2) / 2 * 6;
        params.base_vertex = 0;
        params.is_indexed = true;
    }
    return params;
}

bool SupportsPrimitiveRestart(VkPrimitiveTopology topology) {
    switch (topology) {
    case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY:
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY:
    case VK_PRIMITIVE_TOPOLOGY_PATCH_LIST:
        return false;
    default:
        return true;
    }
}

bool IsPrimitiveRestartSupported(const Device& device, VkPrimitiveTopology topology) {
    return ((topology != VK_PRIMITIVE_TOPOLOGY_PATCH_LIST &&
             device.IsTopologyListPrimitiveRestartSupported()) ||
            SupportsPrimitiveRestart(topology) ||
            (topology == VK_PRIMITIVE_TOPOLOGY_PATCH_LIST &&
             device.IsPatchListPrimitiveRestartSupported()));
}
} // Anonymous namespace

RasterizerVulkan::RasterizerVulkan(Core::Frontend::EmuWindow& emu_window_, Tegra::GPU& gpu_,
                                   Tegra::MaxwellDeviceMemoryManager& device_memory_,
                                   const Device& device_, MemoryAllocator& memory_allocator_,
                                   StateTracker& state_tracker_, Scheduler& scheduler_)
    : gpu{gpu_}, device_memory{device_memory_}, device{device_},
      memory_allocator{memory_allocator_}, state_tracker{state_tracker_}, scheduler{scheduler_},
      staging_pool(device, memory_allocator, scheduler), descriptor_pool(device, scheduler),
      guest_descriptor_queue(device, UpdateDescriptorQueue::GUEST_FRAME_PAYLOAD_SIZE,
                             device.IsExtDescriptorBufferSupported()),
      compute_pass_descriptor_queue(device, UpdateDescriptorQueue::COMPUTE_FRAME_PAYLOAD_SIZE),
      descriptor_buffer_ring(device, memory_allocator),
      blit_image(device, scheduler, state_tracker, descriptor_pool), render_pass_cache(device),
      texture_cache_runtime{
          device,     scheduler,         memory_allocator, staging_pool,
          blit_image, render_pass_cache, descriptor_pool,  compute_pass_descriptor_queue},
      texture_cache(texture_cache_runtime, device_memory),
      buffer_cache_runtime(device, memory_allocator, scheduler, staging_pool,
                           guest_descriptor_queue, compute_pass_descriptor_queue, descriptor_pool),
      buffer_cache(device_memory, buffer_cache_runtime),
      query_cache_runtime(this, device_memory, buffer_cache, device, memory_allocator, scheduler,
                          staging_pool, compute_pass_descriptor_queue, descriptor_pool, texture_cache),
      query_cache(gpu, *this, device_memory, query_cache_runtime),
      pipeline_cache(device_memory, device, scheduler, descriptor_pool, guest_descriptor_queue,
                     descriptor_buffer_ring, render_pass_cache, buffer_cache, texture_cache,
                     gpu.ShaderNotify()),
      accelerate_dma(buffer_cache, texture_cache, scheduler),
      fence_manager(*this, gpu, texture_cache, buffer_cache, query_cache, device, scheduler),
      wfi_event(device.GetLogical().CreateEvent()) {
    scheduler.SetQueryCache(query_cache);
}

RasterizerVulkan::~RasterizerVulkan() {
    scheduler.WaitWorker();
    scheduler.Finish();
}

bool RasterizerVulkan::ResetIndirectRecordsOnCpu(bool is_indexed) {
    // Donkey Kong Bananza's terrain: a vertex program run over points resets each chunk's 5-word
    // indirect draw record to {triangles 0, copies 1, 0, 0, 0} through the pointers of a list
    // (c0[0x110]). Stores through addresses loaded from memory can't be mapped onto a storage
    // buffer, so the recompiler's global-memory fallback drops them: every record keeps 0 copies
    // and the colour pass draws nothing (the terrain is "invisible"). Do that program's stores
    // here instead. EDEN_NX2_RESETFIX=0 leaves them to the GPU. (Approach from zortos293/Eglantine.)
    static const bool reset_fix = [] {
        const char* value = std::getenv("EDEN_NX2_RESETFIX");
        return value == nullptr || value[0] != '0';
    }();
    if (!reset_fix || is_indexed || !maxwell3d->GetNativeMacroEngine().native_turing) {
        return false;
    }
    const auto& regs = maxwell3d->regs;
    const GPUVAddr program =
        (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
    if (!maxwell3d->IsIndirectResetProgram(program)) {
        return false;
    }
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    const auto& cbuf0 = maxwell3d->state.shader_stages[0].const_buffers[0];
    const GPUVAddr list = gpu_memory->Read<u64>(cbuf0.address + 0x110);
    const u32 count = std::min(draw_state.vertex_buffer.count, 1u << 20);
    const GPUVAddr list_begin = list + u64{draw_state.vertex_buffer.first} * sizeof(u64);
    const size_t list_bytes = size_t{count} * sizeof(u64);
    if (count != 0 && gpu_memory->IsFullyMappedRange(list_begin, list_bytes)) {
        if (gpu_memory->IsMemoryDirty(list_begin, list_bytes, VideoCommon::CacheType::BufferCache)) {
            gpu_memory->FlushRegion(list_begin, list_bytes, VideoCommon::CacheType::BufferCache);
        }
        std::vector<u64> pointers(count);
        gpu_memory->ReadBlockUnsafe(list_begin, pointers.data(), list_bytes);
        Tegra::Nx2IndirectRecords::Get().SetPointers(pointers);
        static constexpr std::array<u32, 5> record{0, 1, 0, 0, 0};
        const std::span<const u8> bytes{reinterpret_cast<const u8*>(record.data()), sizeof(record)};
        u32 written = 0;
        for (const u64 pointer : pointers) {
            if (pointer == 0 || !gpu_memory->IsFullyMappedRange(pointer, sizeof(record))) {
                continue;
            }
            AccelerateInlineToMemory(pointer, sizeof(record), bytes);
            ++written;
        }
        static u32 reset_logs{};
        if (reset_logs < 5) {
            ++reset_logs;
            LOG_INFO(HW_GPU, "Indirect draw reset: list {:#x}, {} records ({} written)", list_begin,
                     count, written);
        }
    }
    return true;
}

namespace {
// EDEN_NX2_CHUNK_BATCH: 0 off, 1 on, 2 on with every window exactly the range of its first draw
// (one draw per batch, bindings identical to the game's: isolates the batched draw path).
u32 ChunkBatchingMode() {
    static const u32 mode = [] {
        const char* value = std::getenv("EDEN_NX2_CHUNK_BATCH");
        return value != nullptr && value[0] >= '1' && value[0] <= '9' ? u32(value[0] - '0') : 0U;
    }();
    return mode;
}

// A window of memory around `pointer` that holds [pointer, pointer + need), as large as 32 MiB so
// that the following chunks fall inside it too. The device memory behind the window must be
// contiguous: the host buffer bound over it is one allocation.
std::optional<std::pair<GPUVAddr, u64>> MappedWindow(const Tegra::MemoryManager& gpu_memory,
                                                     GPUVAddr pointer, u64 need) {
    if (ChunkBatchingMode() == 2) {
        return gpu_memory.IsContinuousRange(pointer, need) ? std::optional{std::pair{pointer, need}}
                                                           : std::nullopt;
    }
    constexpr u64 Granule = 0x10000;
    constexpr u64 MaxWindow = u64{32} << 20;
    constexpr u64 MaxBefore = u64{8} << 20;
    const GPUVAddr first = Common::AlignDown(pointer, 0x1000);
    const GPUVAddr last = Common::AlignUp(pointer + need, 0x1000);
    if (!gpu_memory.IsContinuousRange(first, last - first)) {
        return std::nullopt;
    }
    // Lowest base and highest end that keep the range contiguous (both monotonic), by bisection.
    GPUVAddr lo = first - (std::min)(first, MaxBefore);
    GPUVAddr hi = first;
    while (lo < hi) {
        const GPUVAddr mid = lo + Common::AlignDown((hi - lo) / 2, Granule);
        if (gpu_memory.IsContinuousRange(mid, last - mid)) {
            hi = mid;
        } else {
            lo = mid + Granule;
        }
    }
    const GPUVAddr base = hi;
    lo = last;
    hi = base + MaxWindow;
    while (lo < hi) {
        const GPUVAddr mid = hi - Common::AlignDown((hi - lo) / 2, Granule);
        if (gpu_memory.IsContinuousRange(base, mid - base)) {
            lo = mid;
        } else {
            hi = mid - Granule;
        }
    }
    return std::pair{base, lo - base};
}
// The registers a chunk draw writes between two draws of a run.
constexpr u32 RegIndex(size_t byte_offset) {
    return static_cast<u32>(byte_offset / sizeof(u32));
}
namespace ChunkMethods {
constexpr u32 CbOffset = RegIndex(offsetof(Maxwell, const_buffer.offset));
constexpr u32 CbData = RegIndex(offsetof(Maxwell, const_buffer.buffer));
constexpr u32 CbDataWords = 16;
constexpr u32 IndexHigh = RegIndex(offsetof(Maxwell, index_buffer.start_addr_high));
constexpr u32 IndexLow = RegIndex(offsetof(Maxwell, index_buffer.start_addr_low));
constexpr u32 IndexFirst = RegIndex(offsetof(Maxwell, index_buffer.first));
constexpr u32 IndexCount = RegIndex(offsetof(Maxwell, index_buffer.count));
constexpr u32 DrawBegin = RegIndex(offsetof(Maxwell, draw.begin));
constexpr u32 DrawEnd = RegIndex(offsetof(Maxwell, draw.end));
// Per-chunk vertex buffers: stream addresses, legacy limits and the native byte sizes (0x600).
constexpr u32 StreamsBegin = RegIndex(offsetof(Maxwell, vertex_streams));
constexpr u32 StreamsEnd = StreamsBegin + Maxwell::NumVertexArrays * 4;
constexpr u32 LimitsBegin = RegIndex(offsetof(Maxwell, vertex_stream_limits));
constexpr u32 LimitsEnd = LimitsBegin + Maxwell::NumVertexArrays * 2;
constexpr u32 SizesBegin = 0x600 / sizeof(u32);
constexpr u32 SizesEnd = SizesBegin + Maxwell::NumVertexArrays * 2;

constexpr bool IsVertexStreamMethod(u32 method) {
    return (method >= StreamsBegin && method < StreamsEnd) || (method >= LimitsBegin && method < LimitsEnd) ||
           (method >= SizesBegin && method < SizesEnd);
}

std::bitset<Maxwell::NUM_REGS> Set() {
    std::bitset<Maxwell::NUM_REGS> set;
    for (const u32 method : {CbOffset, IndexHigh, IndexLow, IndexFirst, IndexCount, DrawBegin, DrawEnd}) {
        set.set(method);
    }
    for (u32 word = 0; word < CbDataWords; ++word) {
        set.set(CbData + word);
    }
    for (u32 method = 0; method < Maxwell::NUM_REGS; ++method) {
        if (IsVertexStreamMethod(method)) {
            set.set(method);
        }
    }
    return set;
}
} // namespace ChunkMethods
} // Anonymous namespace

std::optional<RasterizerVulkan::ChunkPattern> RasterizerVulkan::MatchChunkPattern() const {
    using namespace ChunkMethods;
    const auto& writes = maxwell3d->WritesSinceDraw();
    if (writes.empty() || writes.size() >= 256) {
        return std::nullopt;
    }
    // Replay CB_OFFSET through the inline data writes to find the patched bytes.
    u32 offset = ~0U;
    boost::container::static_vector<std::pair<u32, u32>, 8> ranges; // [begin, end) in the cbuf
    for (const auto& [method_raw, value] : writes) {
        const u32 method = method_raw & 0xffff;
        const u32 amount = method_raw >> 16;
        const u32 count = amount != 0 ? amount : 1;
        if (method == CbOffset && count == 1) {
            offset = value;
            continue;
        }
        if (method >= CbData && method + count <= CbData + CbDataWords) {
            if (offset == ~0U) {
                return std::nullopt;
            }
            const u32 end = offset + count * 4;
            if (!ranges.empty() && ranges.back().second == offset) {
                ranges.back().second = end;
            } else if (ranges.size() < ranges.capacity()) {
                ranges.push_back({offset, end});
            } else {
                return std::nullopt;
            }
            offset = end;
            continue;
        }
        if (count == 1 && (method == IndexHigh || method == IndexLow || method == IndexFirst ||
                           method == IndexCount || method == DrawBegin || method == DrawEnd)) {
            continue;
        }
        if (IsVertexStreamMethod(method) && IsVertexStreamMethod(method + count - 1)) {
            continue; // this chunk's vertex buffer
        }
        return std::nullopt;
    }
    if (ranges.empty() || ranges.size() > ChunkBatch::MaxSlots) {
        return std::nullopt;
    }
    ChunkPattern pattern{.cbuf_address = maxwell3d->regs.const_buffer.Address()};
    for (const auto& [begin, end] : ranges) {
        // A 64-bit pointer and its size (three or four words), 8-byte aligned.
        if (begin % 8 != 0 || end - begin < 12 || end - begin > 16) {
            return std::nullopt;
        }
        pattern.slots[pattern.num_slots++] = begin;
    }
    return pattern;
}

bool RasterizerVulkan::TryBatchChunkDraw(bool is_indexed, u32 instance_count) {
    if (ChunkBatchingMode() == 0) {
        return false;
    }
    maxwell3d->record_writes_since_draw = true;
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    // The first instance carries the draw's row in the table, so it must be free.
    const bool simple = is_indexed && instance_count <= 1 && draw_state.base_instance == 0 &&
                        draw_state.inline_index_draw_indexes.empty() &&
                        draw_state.draw_mode == Tegra::Engines::Maxwell3D::DrawManager::DrawMode::General;
    const auto pattern = simple ? MatchChunkPattern() : std::nullopt;
    if (pattern) {
        for (ChunkBatch& batch : chunk_batches) {
            if (ExtendChunkBatch(batch, *pattern)) {
                return true;
            }
        }
    }
    // EDEN_NX2_CHUNK_BATCH_OPEN: batches open at once (experiments).
    static const size_t max_open = [] {
        const char* value = std::getenv("EDEN_NX2_CHUNK_BATCH_OPEN");
        const size_t n = value ? static_cast<size_t>(std::strtoul(value, nullptr, 10)) : MaxOpenChunkBatches;
        return std::clamp<size_t>(n, 1, MaxOpenChunkBatches);
    }();
    if (!chunk_batches.empty() && (!pattern || chunk_batches.size() >= max_open)) {
        ++(pattern ? chunk_batch_stats.ended_by_miss : chunk_batch_stats.ended_by_draw);
        FlushChunkBatches();
    }
    if (!pattern || !BeginChunkBatch(*pattern)) {
        return false;
    }
    // EDEN_NX2_CHUNK_BATCH_EAGER=1: draw each batch as soon as it opens (experiments).
    static const bool eager = std::getenv("EDEN_NX2_CHUNK_BATCH_EAGER") != nullptr;
    if (eager) {
        FlushChunkBatches();
    }
    return true;
}

std::optional<std::pair<GPUVAddr, u64>> RasterizerVulkan::ChunkWindow(GPUVAddr pointer, u64 need) {
    if (ChunkBatchingMode() == 2) {
        return MappedWindow(*gpu_memory, pointer, need);
    }
    // Finding a window walks page tables over tens of MiB; remember it per 64 KiB of pointers.
    const GPUVAddr key = Common::AlignDown(pointer, 0x10000);
    if (const auto it = chunk_window_cache.find(key); it != chunk_window_cache.end()) {
        const auto [base, size] = it->second;
        if (pointer >= base && pointer + need <= base + size) {
            return it->second;
        }
    }
    const auto window = MappedWindow(*gpu_memory, pointer, need);
    if (window) {
        chunk_window_cache[key] = *window;
    }
    return window;
}

bool RasterizerVulkan::BeginChunkBatch(const ChunkPattern& pattern) {
    static u32 logged{};
    const auto fail = [&](std::string_view reason) {
        if (logged++ < 32) {
            LOG_WARNING(HW_GPU, "Chunk batch not started: {} (cbuf {:#x} slots {})", reason,
                        pattern.cbuf_address, pattern.num_slots);
        }
        return false;
    };
    GraphicsPipeline* const base = pipeline_cache.CurrentGraphicsPipeline();
    if (!base || base->IsMeshPipeline()) {
        return false;
    }
    const auto& regs = maxwell3d->regs;
    const GPUVAddr vertex_program =
        (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
    // Title 4's indirect-record reset program is run on the CPU (ResetIndirectRecordsOnCpu).
    if (maxwell3d->IsIndirectResetProgram(vertex_program)) {
        return false;
    }
    // The stage that binds the patched constant buffer and addresses storage buffers through
    // every patched slot.
    size_t program = ~size_t{0};
    size_t stage = 0;
    u32 bank = 0;
    // The vertex slot is left alone for now: a vertex program with gl_DrawID and no outputs
    // crashed the host driver's pipeline compiler (title 4's reset program); the chunk passes
    // patch the geometry stage.
    for (size_t candidate = 2; candidate < Maxwell::MaxShaderProgram && program == ~size_t{0}; ++candidate) {
        if (!regs.IsShaderConfigEnabled(candidate)) {
            continue;
        }
        const size_t candidate_stage = candidate - 1;
        const auto& info = base->StageInfos()[candidate_stage];
        for (u32 b = 0; b < Maxwell::MaxConstBuffers; ++b) {
            const auto& cbuf = maxwell3d->state.shader_stages[candidate_stage].const_buffers[b];
            if (!cbuf.enabled || cbuf.address != pattern.cbuf_address) {
                continue;
            }
            bool all = true;
            for (u32 k = 0; k < pattern.num_slots; ++k) {
                all &= std::ranges::any_of(info.storage_buffers_descriptors, [&](const auto& desc) {
                    return desc.cbuf_index == b && desc.cbuf_offset == pattern.slots[k];
                });
            }
            if (all) {
                program = candidate;
                stage = candidate_stage;
                bank = b;
                break;
            }
        }
    }
    if (program == ~size_t{0}) {
        return false; // the patches feed something other than storage buffers: not ours
    }
    // The draw's row travels in its first instance: no stage may read the instance builtins.
    for (size_t candidate = 1; candidate < Maxwell::MaxShaderProgram; ++candidate) {
        if (!regs.IsShaderConfigEnabled(candidate)) {
            continue;
        }
        const auto& loads = base->StageInfos()[candidate - 1].loads;
        if (loads[Shader::IR::Attribute::InstanceId] || loads[Shader::IR::Attribute::BaseInstance]) {
            return fail("a stage reads the instance id");
        }
    }
    const auto& cbuf = maxwell3d->state.shader_stages[stage].const_buffers[bank];
    if (cbuf.size == 0 || cbuf.size > Maxwell::MaxConstBufferSize) {
        return fail("constant buffer size");
    }
    ChunkBatch batch{};
    batch.cbuf_address = pattern.cbuf_address;
    batch.cbuf_size = cbuf.size;
    batch.program = program;
    batch.stage = stage;
    batch.bank = bank;
    // Spans: windows around this draw's pointers.
    for (u32 k = 0; k < pattern.num_slots; ++k) {
        const GPUVAddr pointer = gpu_memory->Read<u64>(pattern.cbuf_address + pattern.slots[k]);
        const u32 size = gpu_memory->Read<u32>(pattern.cbuf_address + pattern.slots[k] + 8);
        const auto window = ChunkWindow(pointer, (std::max)(size, 16U));
        if (!window) {
            return fail("storage buffer unmapped");
        }
        batch.span_base[k] = window->first;
        batch.span_size[k] = window->second;
        batch.table.slots[k] = {bank, pattern.slots[k]};
    }
    batch.table.num_slots = pattern.num_slots;
    const u32 table_slot_offset = Common::AlignUp(cbuf.size, 16U);
    batch.table.table_slot = {bank, table_slot_offset};
    batch.table.stride = ChunkBatch::MaxSlots * sizeof(u32);
    batch.index_format = static_cast<u32>(regs.index_buffer.format);
    batch.index_shift = batch.index_format == 0 ? 0 : batch.index_format == 1 ? 1 : 2;
    batch.topology = static_cast<u32>(maxwell3d->draw_manager.draw_state.topology);
    for (u32 i = 0; i < Maxwell::NumVertexArrays; ++i) {
        if (regs.vertex_streams[i].IsEnabled()) {
            batch.stream_mask |= 1U << i;
            // The row in the first instance would shift instanced attribute fetches.
            if (regs.vertex_stream_instances.is_instanced[i] != 0) {
                return fail("an instanced vertex stream");
            }
        }
    }
    // Scratch: [constant buffer copy with span bases and the table pointer | table].
    const u32 copy_size = table_slot_offset + 16;
    batch.table_offset = Common::AlignUp(copy_size, 256U);
    const u64 total = u64{batch.table_offset} + ChunkBatch::MaxDraws * batch.table.stride;
    batch.scratch = ReserveScratch(total);
    if (batch.scratch == 0) {
        return fail("no scratch");
    }
    batch.scratch_size = total;
    batch.cbuf_copy.assign(copy_size, 0);
    gpu_memory->ReadBlockUnsafe(pattern.cbuf_address, batch.cbuf_copy.data(), cbuf.size);
    const auto put64 = [&](u32 at, u64 value) { std::memcpy(batch.cbuf_copy.data() + at, &value, sizeof(value)); };
    for (u32 k = 0; k < pattern.num_slots; ++k) {
        put64(pattern.slots[k], batch.span_base[k]);
        put64(pattern.slots[k] + 8, batch.span_size[k]);
    }
    put64(table_slot_offset, batch.scratch + batch.table_offset);
    put64(table_slot_offset + 8, u64{ChunkBatch::MaxDraws} * batch.table.stride);

    batch.pipeline = pipeline_cache.BatchedGraphicsPipeline(program, batch.table);
    if (!batch.pipeline) {
        return fail("no batched pipeline");
    }
    if (!ExtendChunkBatch(batch, pattern)) {
        return fail("first draw outside its windows");
    }
    for (u32 k = 0; k < pattern.num_slots; ++k) {
        buffer_cache.storage_buffer_sizes[batch.span_base[k]] = static_cast<u32>(batch.span_size[k]);
    }
    if (logged < 32) {
        ++logged;
        LOG_INFO(HW_GPU, "Chunk batch started: vs {:#x} program slot {} bank {} cbuf {:#x} slots{} spans{} index format {} streams {:#x}",
                 vertex_program, program, bank, pattern.cbuf_address,
                 fmt::join(std::span{pattern.slots.data(), pattern.num_slots}, ","),
                 fmt::join(std::span{batch.span_size.data(), pattern.num_slots}, ","), batch.index_format,
                 batch.stream_mask);
    }
    // EDEN_NX2_CHUNK_BATCH_VERIFY=1: remember the state the batch opened with, to compare at the
    // flush (diagnostic).
    static const bool verify = std::getenv("EDEN_NX2_CHUNK_BATCH_VERIFY") != nullptr;
    if (verify) {
        const auto& draw_state = maxwell3d->draw_manager.draw_state;
        batch.regs_snapshot = std::make_unique<std::array<u32, Maxwell::NUM_REGS>>(regs.reg_array);
        batch.draw_snapshot = {static_cast<u32>(draw_state.topology), draw_state.base_index, draw_state.base_instance,
                               draw_state.instance_count, static_cast<u32>(draw_state.draw_mode),
                               static_cast<u32>(draw_state.inline_index_draw_indexes.size())};
        for (size_t st = 0; st < Maxwell::MaxShaderStage; ++st) {
            for (u32 b = 0; b < Maxwell::MaxConstBuffers; ++b) {
                const auto& c = maxwell3d->state.shader_stages[st].const_buffers[b];
                batch.cbuf_snapshot[st * Maxwell::MaxConstBuffers + b] = c.enabled ? c.address ^ (u64{c.size} << 48) : 0;
            }
        }
        batch.dirty_snapshot = maxwell3d->dirty.flags;
    }
    chunk_batches.push_back(std::move(batch));
    // Any other register write flushes the batches before it changes the state they ran with.
    auto& guard = maxwell3d->batched_draw_guard;
    if (!guard.flush) {
        guard.methods = ChunkMethods::Set();
        guard.flush = [this] { FlushChunkBatches(); };
    }
    guard.active = true;
    return true;
}

bool RasterizerVulkan::ExtendChunkBatch(ChunkBatch& batch, const ChunkPattern& pattern) {
    if (pattern.cbuf_address != batch.cbuf_address || pattern.num_slots != batch.table.num_slots) {
        return false;
    }
    for (u32 k = 0; k < pattern.num_slots; ++k) {
        if (pattern.slots[k] != batch.table.slots[k].cbuf_offset) {
            return false;
        }
    }
    // EDEN_NX2_CHUNK_BATCH_MAX_DRAWS: cap for experiments (1 = every draw alone, so only the
    // table's first row is ever read).
    static const u32 max_draws = [] {
        const char* value = std::getenv("EDEN_NX2_CHUNK_BATCH_MAX_DRAWS");
        const u32 n = value ? static_cast<u32>(std::strtoul(value, nullptr, 10)) : ChunkBatch::MaxDraws;
        return std::clamp<u32>(n, 1, ChunkBatch::MaxDraws);
    }();
    if (batch.draws.size() >= max_draws) {
        return false;
    }
    const auto& regs = maxwell3d->regs;
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    if (static_cast<u32>(regs.index_buffer.format) != batch.index_format ||
        static_cast<u32>(draw_state.topology) != batch.topology) {
        return false;
    }
    ChunkBatch::Draw draw{
        .index_address = regs.index_buffer.StartAddress(),
        .first_index = draw_state.index_buffer.first,
        .index_count = draw_state.index_buffer.count,
        .base_vertex = static_cast<s32>(draw_state.base_index),
    };
    if (draw.index_count == 0) {
        return false;
    }
    u32 streams = 0;
    const bool native = maxwell3d->GetNativeMacroEngine().native_turing;
    for (u32 i = 0; i < Maxwell::NumVertexArrays; ++i) {
        const auto& stream = regs.vertex_streams[i];
        if (!stream.IsEnabled()) {
            continue;
        }
        streams |= 1U << i;
        const GPUVAddr begin = stream.Address();
        const u64 end = native ? begin + regs.NativeVertexStreamSize(i) : regs.vertex_stream_limits[i].Address() + 1;
        if (end <= begin || end - begin > (std::numeric_limits<u32>::max)()) {
            ++chunk_batch_stats.miss_index;
            return false;
        }
        draw.vb_address[i] = begin;
        draw.vb_size[i] = static_cast<u32>(end - begin);
    }
    if (streams != batch.stream_mask) {
        return false;
    }
    std::array<u32, ChunkBatch::MaxSlots> deltas{};
    for (u32 k = 0; k < pattern.num_slots; ++k) {
        const GPUVAddr pointer = gpu_memory->Read<u64>(pattern.cbuf_address + pattern.slots[k]);
        const u32 size = gpu_memory->Read<u32>(pattern.cbuf_address + pattern.slots[k] + 8);
        if (pointer < batch.span_base[k] || pointer + size > batch.span_base[k] + batch.span_size[k]) {
            ++chunk_batch_stats.miss_slot[k];
            return false;
        }
        deltas[k] = static_cast<u32>(pointer - batch.span_base[k]);
    }
    batch.rows.insert(batch.rows.end(), deltas.begin(), deltas.end());
    batch.draws.push_back(draw);
    return true;
}

void RasterizerVulkan::FlushChunkBatches() {
    if (chunk_batches.empty()) {
        return;
    }
    maxwell3d->batched_draw_guard.active = false;
    // Taken out first: the uploads below pass through the flush hooks themselves.
    std::vector<ChunkBatch> batches;
    batches.swap(chunk_batches);
    FlushWork();
    for (ChunkBatch& batch : batches) {
        DrawChunkBatch(batch);
    }
    buffer_cache.storage_buffer_sizes.clear();
    auto& stats = chunk_batch_stats;
    stats.batches += batches.size();
    for (const ChunkBatch& batch : batches) {
        stats.draws += batch.draws.size();
    }
    ++stats.flushes;
    if (const auto now = std::chrono::steady_clock::now(); now - stats.window > std::chrono::seconds{10}) {
        LOG_INFO(HW_GPU,
                 "Chunk batches in the last 10 s: {} flushes, {} batches, {} draws ({:.0f} per batch); "
                 "flushed by a chunk outside every window {} (misses: indices {}, slots {}), by "
                 "another draw {}, by other work {}",
                 stats.flushes, stats.batches, stats.draws,
                 stats.batches ? static_cast<double>(stats.draws) / static_cast<double>(stats.batches) : 0.0,
                 stats.ended_by_miss, stats.miss_index, fmt::join(stats.miss_slot, ","),
                 stats.ended_by_draw, stats.flushes - stats.ended_by_miss - stats.ended_by_draw);
        stats = {.window = now};
    }
    gpu.TickWork();
}

void RasterizerVulkan::DrawChunkBatch(ChunkBatch& batch) {
    const u32 count = static_cast<u32>(batch.draws.size());
    if (count == 0) {
        return;
    }
    if (batch.regs_snapshot) {
        static u32 verified{};
        if (verified++ < 12) {
            const auto& regs = maxwell3d->regs;
            const auto& draw_state = maxwell3d->draw_manager.draw_state;
            std::string diff;
            u32 differing = 0;
            for (u32 i = 0; i < Maxwell::NUM_REGS; ++i) {
                if ((*batch.regs_snapshot)[i] != regs.reg_array[i]) {
                    if (differing++ < 12) {
                        diff += fmt::format(" r{:#x}:{:#x}->{:#x}", i, (*batch.regs_snapshot)[i], regs.reg_array[i]);
                    }
                }
            }
            std::string cbufs;
            for (size_t st = 0; st < Maxwell::MaxShaderStage; ++st) {
                for (u32 b = 0; b < Maxwell::MaxConstBuffers; ++b) {
                    const auto& c = maxwell3d->state.shader_stages[st].const_buffers[b];
                    const u64 now = c.enabled ? c.address ^ (u64{c.size} << 48) : 0;
                    if (now != batch.cbuf_snapshot[st * Maxwell::MaxConstBuffers + b]) {
                        cbufs += fmt::format(" s{}b{}", st, b);
                    }
                }
            }
            const std::array<u32, 6> now{static_cast<u32>(draw_state.topology), draw_state.base_index,
                                         draw_state.base_instance, draw_state.instance_count,
                                         static_cast<u32>(draw_state.draw_mode),
                                         static_cast<u32>(draw_state.inline_index_draw_indexes.size())};
            u32 dirty_new = 0;
            for (size_t f = 0; f < batch.dirty_snapshot.size(); ++f) {
                dirty_new += !batch.dirty_snapshot[f] && maxwell3d->dirty.flags[f];
            }
            LOG_INFO(HW_GPU,
                     "Chunk batch verify ({} draws): {} registers differ{}; cbuf bindings changed:{}; draw state "
                     "topology/base_index/base_instance/instances/mode/inline {} -> {}; dirty flags newly set {}",
                     count, differing, diff, cbufs.empty() ? " none" : cbufs, fmt::join(batch.draw_snapshot, "/"),
                     fmt::join(now, "/"), dirty_new);
        }
    }
    static u32 dumped{};
    if (dumped < 4) {
        ++dumped;
        const auto& d = batch.draws.front();
        LOG_INFO(HW_GPU,
                 "Chunk batch of {} draws: scratch {:#x} table +{:#x}; spans {:#x}+{:#x}, {:#x}+{:#x}; row 0 deltas "
                 "{:#x},{:#x}; draw 0 indices {:#x} first {} count {} base vertex {} vb0 {:#x}+{:#x}; last row deltas "
                 "{:#x},{:#x}",
                 count, batch.scratch, batch.table_offset, batch.span_base[0], batch.span_size[0], batch.span_base[1],
                 batch.span_size[1], batch.rows[0], batch.rows[1], d.index_address, d.first_index, d.index_count,
                 d.base_vertex, d.vb_address[0], d.vb_size[0], batch.rows[(count - 1) * ChunkBatch::MaxSlots],
                 batch.rows[(count - 1) * ChunkBatch::MaxSlots + 1]);
    }
    // The copy and the table go through guest memory like any CPU-written buffer, then the range
    // is obtained with synchronization so the upload is recorded before the draws.
    const u32 copy_size = static_cast<u32>(batch.cbuf_copy.size());
    const u32 rows_bytes = count * batch.table.stride;
    AccelerateInlineToMemory(batch.scratch, copy_size, batch.cbuf_copy);
    AccelerateInlineToMemory(batch.scratch + batch.table_offset, rows_bytes,
                             std::span{reinterpret_cast<const u8*>(batch.rows.data()), rows_bytes});
    gpu_memory->FlushCaching();
    // Configure once with the copy in the bank; the bank the game set is restored afterwards. The
    // vertex and index buffers are bound per draw below and marked dirty so the next draw re-binds
    // its own.
    auto& bound_cbuf = maxwell3d->state.shader_stages[batch.stage].const_buffers[batch.bank];
    const auto saved_cbuf = bound_cbuf;
    SCOPE_EXIT {
        bound_cbuf = saved_cbuf;
        buffer_cache.BindGraphicsUniformBuffer(batch.stage, batch.bank, saved_cbuf.address, saved_cbuf.size);
        auto& flags = maxwell3d->dirty.flags;
        flags[VideoCommon::Dirty::IndexBuffer] = true;
        flags[VideoCommon::Dirty::VertexBuffers] = true;
        for (u32 i = 0; i < Maxwell::NumVertexArrays; ++i) {
            if ((batch.stream_mask >> i) & 1) {
                flags[VideoCommon::Dirty::VertexBuffer0 + i] = true;
            }
        }
    };
    bound_cbuf = {.address = batch.scratch, .size = copy_size, .enabled = true};
    buffer_cache.BindGraphicsUniformBuffer(batch.stage, batch.bank, batch.scratch, copy_size);
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    [[maybe_unused]] const auto uploaded = buffer_cache.ObtainBuffer(
        batch.scratch, static_cast<u32>(batch.scratch_size), VideoCommon::ObtainBufferSynchronize::FullSynchronize,
        VideoCommon::ObtainBufferOperation::DoNothing);
    batch.pipeline->SetEngine(maxwell3d, gpu_memory);
    if (!batch.pipeline->Configure(true)) {
        LOG_WARNING(HW_GPU, "Chunk batch of {} draws dropped: configure failed", count);
        return;
    }
    UpdateDynamicStates();
    query_cache.NotifySegment(true);
    scheduler.RequestRenderpass(texture_cache.GetFramebuffer());
    const auto& regs = maxwell3d->regs;
    const auto topology = maxwell3d->draw_manager.draw_state.topology;
    const auto index_format = regs.index_buffer.format;
    for (u32 index = 0; index < count; ++index) {
        const ChunkBatch::Draw& draw = batch.draws[index];
        for (u32 i = 0; i < Maxwell::NumVertexArrays; ++i) {
            if (((batch.stream_mask >> i) & 1) == 0) {
                continue;
            }
            const auto [buffer, offset] = buffer_cache.ObtainBuffer(
                draw.vb_address[i], draw.vb_size[i], VideoCommon::ObtainBufferSynchronize::FullSynchronize,
                VideoCommon::ObtainBufferOperation::DoNothing);
            buffer_cache.runtime.BindVertexBuffer(i, buffer->Handle(), offset, draw.vb_size[i],
                                                  regs.vertex_streams[i].stride);
        }
        const u32 index_bytes = (draw.first_index + draw.index_count) << batch.index_shift;
        const auto [index_buffer, index_offset] = buffer_cache.ObtainBuffer(
            draw.index_address, index_bytes, VideoCommon::ObtainBufferSynchronize::FullSynchronize,
            VideoCommon::ObtainBufferOperation::DoNothing);
        buffer_cache.runtime.BindIndexBuffer(topology, index_format, draw.first_index, draw.index_count,
                                             index_buffer->Handle(), index_offset, index_bytes);
        scheduler.Record([draw, index](vk::CommandBuffer cmdbuf) {
            cmdbuf.DrawIndexed(draw.index_count, 1, draw.first_index, static_cast<u32>(draw.base_vertex), index);
        });
    }
}

GPUVAddr RasterizerVulkan::ReserveScratch(u64 bytes) {
    if (scratch_size == 0) {
        const auto scratch = Service::Nvidia::Devices::GetNx2MeshScratch();
        if (scratch.size == 0) {
            return 0;
        }
        scratch_base = scratch.gpu_address;
        scratch_size = scratch.size;
    }
    bytes = Common::AlignUp(bytes, 0x1000);
    if (bytes > scratch_size) {
        return 0;
    }
    if (scratch_next + bytes > scratch_size) {
        scratch_next = 0;
    }
    const GPUVAddr address = scratch_base + scratch_next;
    // The mapping lands on the GPU thread when the address space is created; until then, no.
    if (!gpu_memory->IsFullyMappedRange(address, bytes)) {
        return 0;
    }
    scratch_next += bytes;
    return address;
}

bool RasterizerVulkan::DrawEmulatedMesh(const VideoCommon::Nx2MeshEmulation::DrawDescription& draw) {
    namespace ME = VideoCommon::Nx2MeshEmulation;
    using Layout = Shader::Ampere::MeshEmulation::Layout;
    static u32 logged{};
    const auto fail = [&](std::string_view reason) {
        if (logged++ < 32) {
            LOG_WARNING(HW_GPU, "Task/mesh emulation: {} (task {:#x} mesh {:#x} workgroups {})", reason,
                        draw.task_program, draw.mesh_program, draw.workgroups);
        }
        return false;
    };
    const Layout layout{};
    const auto plan = ME::ScratchPlan::For(draw);
    const GPUVAddr scratch = ReserveScratch(plan.bytes);
    if (scratch == 0) {
        return fail("no scratch memory for the draw");
    }
    if (!gpu_memory->IsFullyMappedRange(draw.mesh_program, 128)) {
        return fail("mesh program header unmapped");
    }
    const auto& regs = maxwell3d->regs;
    static std::unordered_set<GPUVAddr> described;
    if (described.size() < 32 && described.insert(draw.mesh_program).second) {
        const auto& mesh = draw.mesh;
        LOG_INFO(HW_GPU,
                 "Task/mesh emulation: task {:#x} mesh {:#x} workgroups {} | control {:#x} topology {} "
                 "max_vertices {} max_primitives {} mesh_threads {} task_threads {} | replay {} slots x {} "
                 "workgroups, scratch {:#x} bytes",
                 draw.task_program, draw.mesh_program, draw.workgroups, mesh.control, mesh.topology,
                 mesh.max_vertices, mesh.max_primitives, mesh.mesh_threads, mesh.task_threads,
                 1U << plan.primitive_slots_shift, draw.MeshWorkgroups(), plan.bytes);
    }

    // Params at the head of this draw's region, through the inline-upload path so the buffer
    // cache carries the words to the host buffer before the dispatches read them.
    const Layout::Params params = plan.Params(scratch, draw);
    AccelerateInlineToMemory(scratch, sizeof(params),
                             std::span{reinterpret_cast<const u8*>(&params), sizeof(params)});
    gpu_memory->FlushCaching();

    // The rewritten programs run through the ordinary compute path with a launch description of
    // our own: slot 1's constant buffers plus Params as bank 6 (Nx2MeshEmulation::Launch).
    const auto& bound = maxwell3d->state.shader_stages[0].const_buffers;
    auto& qmd = kepler_compute->launch_description;
    const auto saved_qmd = qmd;
    SCOPE_EXIT {
        qmd = saved_qmd;
    };
    static constexpr VkMemoryBarrier SHADER_BARRIER{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
    };
    const auto dispatch = [&](GPUVAddr program, bool has_task, u32 workgroups, u32 threads) {
        qmd = ME::Launch(0, workgroups, threads, bound, scratch, layout);
        ComputePipeline* const pipeline =
            pipeline_cache.MeshEmulationComputePipeline(program, has_task, threads);
        if (!pipeline) {
            return fail("compute pipeline unavailable");
        }
        std::scoped_lock lock{texture_cache.mutex, buffer_cache.mutex};
        if (!pipeline->Configure(*kepler_compute, *gpu_memory, scheduler, buffer_cache, texture_cache)) {
            return fail("compute resources unavailable");
        }
        scheduler.RequestOutsideRenderPassOperationContext();
        scheduler.Record([pipeline, workgroups](vk::CommandBuffer cmdbuf) {
            cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER,
                                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, SHADER_BARRIER);
            if (pipeline->IsBound()) {
                cmdbuf.Dispatch(workgroups, 1, 1);
            }
        });
        return true;
    };
    const std::array<u32, 3> max_dim{device.GetMaxComputeWorkGroupCount()};
    if (draw.MeshWorkgroups() > max_dim[0]) {
        return fail("too many mesh workgroups");
    }
    if (draw.HasTask() && !dispatch(draw.task_program, false, draw.TaskWorkgroups(), draw.mesh.task_threads)) {
        return false;
    }
    if (!dispatch(draw.mesh_program, draw.HasTask(), draw.MeshWorkgroups(), draw.mesh.mesh_threads)) {
        return false;
    }
    scheduler.Record([](vk::CommandBuffer cmdbuf) {
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, SHADER_BARRIER);
    });

    // Replay: a triangle list over the mesh regions with the passthrough vertex program, the
    // title's fragment program and the draw's fixed state; Params is stage 0's bank 6.
    std::array<u8, 128> mesh_header{};
    gpu_memory->ReadBlock(draw.mesh_program, mesh_header.data(), mesh_header.size());
    const GPUVAddr fragment_program =
        regs.IsShaderConfigEnabled(5) && regs.rasterize_enable ? ME::SlotProgramAddress(regs, 5) : 0;
    auto& draw_state = maxwell3d->draw_manager.draw_state;
    const auto saved_topology = draw_state.topology;
    const auto saved_vertex_buffer = draw_state.vertex_buffer;
    const bool saved_draw_indexed = draw_state.draw_indexed;
    auto& cbuf6 = maxwell3d->state.shader_stages[0].const_buffers[layout.params_bank];
    const auto saved_cbuf6 = cbuf6;
    SCOPE_EXIT {
        draw_state.topology = saved_topology;
        draw_state.vertex_buffer = saved_vertex_buffer;
        draw_state.draw_indexed = saved_draw_indexed;
        cbuf6 = saved_cbuf6;
        if (saved_cbuf6.enabled) {
            buffer_cache.BindGraphicsUniformBuffer(0, layout.params_bank, saved_cbuf6.address, saved_cbuf6.size);
        } else {
            buffer_cache.DisableGraphicsUniformBuffer(0, layout.params_bank);
        }
    };
    const u32 vertices = plan.ReplayVertexCount(draw);
    draw_state.topology = Maxwell::PrimitiveTopology::Triangles;
    draw_state.vertex_buffer.first = 0;
    draw_state.vertex_buffer.count = vertices;
    draw_state.draw_indexed = false;
    cbuf6 = {.address = scratch, .size = ME::ScratchPlan::ParamsBytes, .enabled = true};
    buffer_cache.BindGraphicsUniformBuffer(0, layout.params_bank, scratch, ME::ScratchPlan::ParamsBytes);
    GraphicsPipeline* const pipeline = pipeline_cache.MeshReplayPipeline(mesh_header, fragment_program);
    if (!pipeline) {
        return fail("replay pipeline unavailable");
    }
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    pipeline->SetEngine(maxwell3d, gpu_memory);
    if (!pipeline->Configure(false)) {
        return fail("replay configure failed");
    }
    UpdateDynamicStates();
    scheduler.Record([vertices](vk::CommandBuffer cmdbuf) { cmdbuf.Draw(vertices, 1, 0, 0); });
    if (logged < 32 && frame_trace_state == 2) {
        LOG_INFO(HW_GPU, "frame trace:   mesh emulation task {:#x} mesh {:#x} workgroups {} replay {} vertices",
                 draw.task_program, draw.mesh_program, draw.workgroups, vertices);
    }
    return true;
}

template <typename Func>
void RasterizerVulkan::PrepareDraw(bool is_indexed, Func&& draw_func) {

    SCOPE_EXIT {
        gpu.TickWork();
    };
    bool submitted = false;
    SCOPE_EXIT {
        auto& native = maxwell3d->GetNativeMacroEngine();
        if (native.native_turing && !submitted) {
            native.native_execution_failed = true;
            LOG_DEBUG(HW_GPU, "Native graphics draw failed; refusing submission completion");
        }
    };
    namespace Profile = VideoCommon::Nx2Profile;
    {
        Profile::Scope scope{Profile::Zone::FlushWork};
        FlushWork();
        gpu_memory->FlushCaching();
    }

    const auto report_failure = [&](std::string_view reason) {
        static std::set<std::tuple<u64, u64, std::string_view>> seen;
        const auto& regs = maxwell3d->regs;
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        const u64 ps = (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1];
        if (seen.size() < 64 && seen.emplace(vs, ps, reason).second) {
            LOG_WARNING(HW_GPU, "Draw skipped ({}): vs={:#x} ps={:#x} rt0={:#x} {}x{}", reason, vs, ps,
                        regs.rt[0].Address(), regs.rt[0].width, regs.rt[0].height);
        }
        if (frame_trace_state == 2) {
            // Every enabled stage slot, so the rejected program behind the skip can be named.
            std::string stages;
            for (size_t index = 0; index < Maxwell::Regs::MaxShaderProgram; ++index) {
                if (regs.IsShaderConfigEnabled(index)) {
                    const auto& slot = regs.pipelines[index];
                    stages += fmt::format(" {}:{:#x}", index,
                                          (u64{slot.reserved[0] & 0xff} << 32) | slot.reserved[1]);
                }
            }
            LOG_INFO(HW_GPU, "frame trace:   draw skipped ({}) stages{}", reason, stages);
        }
    };
    if (VideoCommon::Nx2MeshEmulation::Enabled() && !is_indexed &&
        maxwell3d->GetNativeMacroEngine().native_turing && maxwell3d->regs.IsShaderConfigEnabled(1)) {
        namespace ME = VideoCommon::Nx2MeshEmulation;
        const auto kind = VideoCommon::ClassifyMeshProgram(
            *gpu_memory, ME::SlotProgramAddress(maxwell3d->regs, 1));
        if (const auto draw = ME::Describe(*maxwell3d, kind)) {
            submitted = DrawEmulatedMesh(*draw);
            if (!submitted) {
                report_failure("mesh emulation");
            }
            return;
        }
    }
    GraphicsPipeline* const pipeline{[&] {
        Profile::Scope scope{Profile::Zone::PipelineLookup};
        return pipeline_cache.CurrentGraphicsPipeline();
    }()};
    if (!pipeline) {
        report_failure("no pipeline");
        return;
    }
    // Task/mesh pipelines: the draw's vertex count is the number of task workgroups.
    mesh_draw = pipeline->IsMeshPipeline();
    // EDEN_NX2_NO_MESH=1: skip task/mesh draws as before they were supported (A/B comparisons).
    static const bool no_mesh{std::getenv("EDEN_NX2_NO_MESH") != nullptr};
    if (mesh_draw && (no_mesh || !device.IsExtMeshShaderSupported())) {
        report_failure(no_mesh ? "EDEN_NX2_NO_MESH" : "no VK_EXT_mesh_shader");
        return;
    }
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    if (ResetIndirectRecordsOnCpu(is_indexed)) {
        submitted = true;
        return;
    }
    // update engine as channel may be different.
    pipeline->SetEngine(maxwell3d, gpu_memory);
    if (Profile::Scope scope{Profile::Zone::Configure}; !pipeline->Configure(is_indexed)) {
        report_failure("configure failed");
        return;
    }

    {
        Profile::Scope scope{Profile::Zone::DynamicState};
        UpdateDynamicStates();
    }

    query_cache.NotifySegment(true);
    HandleTransformFeedback();
    query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64, maxwell3d->regs.zpass_pixel_count_enable);
    {
        Profile::Scope scope{Profile::Zone::DrawRecord};
        draw_func();
    }
    submitted = true;
    static u32 native_capture_count{};
    if (maxwell3d->GetNativeMacroEngine().native_turing &&
        native_capture_count < 16 && [] {
            static const bool capture = std::getenv("EDEN_NATIVE_DRAW_CAPTURE") != nullptr;
            return capture;
        }()) {
        const u32 capture = native_capture_count++;
        const auto prefix = fmt::format("native-draw-{}", capture);
        const auto dump = [&](const std::string& name, GPUVAddr address, size_t size) {
            if (!size || !gpu_memory->IsFullyMappedRange(address, size)) return;
            std::vector<u8> bytes(size);
            gpu_memory->ReadBlockUnsafe(address, bytes.data(), bytes.size());
            std::ofstream file(prefix + name + ".bin", std::ios::binary);
            file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            LOG_INFO(HW_GPU, "Native draw capture {}{}: address={:#x}, bytes={}, nonzero={}",
                     capture, name, address, size,
                     std::count_if(bytes.begin(), bytes.end(), [](u8 byte) { return byte != 0; }));
        };
        const auto& regs = maxwell3d->regs;
        const auto& rt = regs.rt[0];
        const size_t size = static_cast<size_t>(rt.array_pitch) * 4;
        if (size && size <= 16 * 1024 * 1024) {
            if (const auto address = gpu_memory->GpuToCpuAddress(rt.Address(), size)) {
                LOG_INFO(HW_GPU, "Native draw {} render target physical address={:#x}", capture, *address);
                texture_cache.DownloadMemory(*address, size);
                dump("-rt0", rt.Address(), size);
            }
        }
        std::ofstream registers(prefix + "-registers.bin", std::ios::binary);
        registers.write(reinterpret_cast<const char*>(&regs), sizeof(regs));
        for (size_t stream = 0; stream < 4; ++stream) {
            if (regs.vertex_streams[stream].IsEnabled())
                dump(fmt::format("-vertex-{}", stream), regs.vertex_streams[stream].Address(), 128);
        }
        for (size_t stage : {0U, 4U}) {
            for (size_t bank : {2U, 3U}) {
                const auto& cb = maxwell3d->state.shader_stages[stage].const_buffers[bank];
                if (cb.enabled) dump(fmt::format("-cbuf-{}-{}", stage, bank), cb.address, std::min<size_t>(cb.size, 128));
            }
        }
    }
}


namespace {
// EDEN_NX2_WINDOW_TRACE: describe clears and draws into screen-sized colour targets (debug aid).
void TraceWindowTarget(const Tegra::Engines::Maxwell3D& maxwell3d, Tegra::MemoryManager& memory,
                       const char* kind, u32 count) {
    static const bool trace = std::getenv("EDEN_NX2_WINDOW_TRACE") != nullptr;
    static u32 traced{};
    const auto& regs = maxwell3d.regs;
    static const u64 skip = [] {
        const char* value = std::getenv("EDEN_NX2_WINDOW_TRACE_SKIP");
        return value != nullptr ? std::strtoull(value, nullptr, 10) : 0ULL;
    }();
    static u64 seen_events{};
    static const u64 only_ps = [] {
        const char* value = std::getenv("EDEN_NX2_TRACE_PS");
        return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ULL;
    }();
    const u64 pixel_shader =
        (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1];
    if (!trace || regs.rt[0].width != 3840 || traced >= 400 || seen_events++ < skip ||
        (only_ps != 0 && pixel_shader != only_ps)) {
        return;
    }
    ++traced;
    const auto& vp = regs.viewport_transform[0];
    const auto& sc = regs.scissor_test[0];
    LOG_INFO(HW_GPU,
             "nx2 window {} rt0={:#x} count={} clear={:#x} color=({},{},{},{}) mask_common={} "
             "mask0={:#x} rt_count={} vp scale=({},{}) translate=({},{}) scissor en={} x={}..{} "
             "y={}..{} blend0={} vs={:#x} ps={:#x} rasterize={}",
             kind, regs.rt[0].Address(), count, regs.clear_surface.raw, regs.clear_color[0],
             regs.clear_color[1], regs.clear_color[2], regs.clear_color[3], regs.color_mask_common,
             regs.color_mask[0].raw, regs.rt_control.count.Value(), vp.scale_x, vp.scale_y,
             vp.translate_x, vp.translate_y, sc.enable, sc.min_x.Value(), sc.max_x.Value(),
             sc.min_y.Value(), sc.max_y.Value(), regs.blend.enable[0],
             (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1],
             (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1],
             regs.rasterize_enable);
    if (kind[0] != 'd') {
        return;
    }
    // Pixel-stage constant buffer 3: colour constants of the loading-screen passes.
    if (const auto& cb3 = maxwell3d.state.shader_stages[4].const_buffers[3]; cb3.enabled) {
        std::array<f32, 8> values{};
        memory.ReadBlockUnsafe(cb3.address, values.data(), sizeof(values));
        LOG_INFO(HW_GPU, "nx2 window ps c[3]@{:#x} size={}: {}", cb3.address, cb3.size,
                 fmt::join(values, " "));
    }
    static u32 texture_traces{};
    if (texture_traces++ > 200) {
        return;
    }
    // Texture used by the pixel shader's first bound TEX (handle in c[2][0x20]) and its TIC.
    const auto& cb = maxwell3d.state.shader_stages[4].const_buffers[2];
    const u32 handle = cb.enabled ? memory.Read<u32>(cb.address + 0x20) : 0;
    const u32 tic_index = handle & 0xfffff;
    const GPUVAddr tic_address = regs.tex_header.Address() + u64{tic_index} * 32;
    std::array<u32, 8> tic{};
    memory.ReadBlockUnsafe(tic_address, tic.data(), sizeof(tic));
    LOG_INFO(HW_GPU,
             "nx2 window texture: cb2={:#x} en={} handle={:#x} tic pool={:#x} limit={} "
             "sampler pool={:#x} tic[{}]={:08x}",
             cb.address, cb.enabled, handle, regs.tex_header.Address(), regs.tex_header.limit,
             regs.tex_sampler.Address(), tic_index, fmt::join(tic, " "));
    // First bytes of the texture's memory (block-linear, so just a sample).
    const GPUVAddr texture = (GPUVAddr{tic[2] & 0xffff} << 32) | tic[1];
    std::array<u32, 16> texels{};
    memory.ReadBlockUnsafe(texture, texels.data(), sizeof(texels));
    LOG_INFO(HW_GPU, "nx2 window texture {:#x} {}x{}: {:08x}", texture, (tic[4] & 0xffff) + 1,
             (tic[5] & 0xffff) + 1, fmt::join(texels, " "));
    // EDEN_NX2_TEXTURE_DUMP=<file>: save the first 4 MiB of the last traced texture's memory.
    if (const char* path = std::getenv("EDEN_NX2_TEXTURE_DUMP"); path != nullptr) {
        std::vector<u8> bytes(2048u * 10240u);
        memory.ReadBlockUnsafe(texture, bytes.data(), bytes.size());
        if (std::FILE* file = std::fopen(path, "wb")) {
            std::fwrite(bytes.data(), 1, bytes.size(), file);
            std::fclose(file);
        }
    }
}
} // Anonymous namespace

// EDEN_NX2_DRAW_STATS=1: every 600 frames, the vertex/pixel program pairs that drew the most
// vertices (their addresses name the dumped programs: <key>-gfx-<address>.bin).
void RasterizerVulkan::RecordDrawStats(u64 draws, u64 vertices, bool indirect) {
    const auto& regs = maxwell3d->regs;
    const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
    const u64 ps = (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1];
    const u32 depth = (regs.depth_test_enable ? 0x100 : 0) | (regs.depth_write_enabled ? 0x200 : 0) |
                      (indirect ? 0x400 : 0) | (static_cast<u32>(regs.depth_test_func) & 0xff);
    auto& entry = draw_stats[(vs << 24) ^ ps ^ (u64{depth} << 52)];
    entry.vs = vs;
    entry.ps = ps;
    entry.depth = depth;
    entry.rt_count = regs.rt_control.count;
    entry.zeta = regs.zeta_enable ? static_cast<u32>(regs.zeta.format) : 0;
    entry.color0 = static_cast<u32>(regs.rt[0].format);
    entry.draws += draws;
    entry.vertices += vertices;
}

void RasterizerVulkan::Draw(bool is_indexed, u32 instance_count) {
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Draw};
    SCOPE_EXIT {
        if (ChunkBatchingMode() != 0) {
            maxwell3d->ClearWritesSinceDraw();
        }
    };
    if (TryBatchChunkDraw(is_indexed, instance_count)) {
        return;
    }
    TraceWindowTarget(*maxwell3d, *gpu_memory, is_indexed ? "draw-indexed" : "draw",
                      is_indexed ? maxwell3d->draw_manager.draw_state.index_buffer.count
                                 : maxwell3d->draw_manager.draw_state.vertex_buffer.count);
    if (static const bool stats = std::getenv("EDEN_NX2_DRAW_STATS") != nullptr; stats) {
        const u64 count = is_indexed ? maxwell3d->draw_manager.draw_state.index_buffer.count
                                     : maxwell3d->draw_manager.draw_state.vertex_buffer.count;
        RecordDrawStats(1, count * (std::max)(instance_count, 1U), false);
    }
    // EDEN_NX2_DRAW_DELTA_LOG=<vertex program address, hex>: for the first draws with that
    // program, the register writes since the previous draw (what a batch of them would have to
    // carry), with the draw parameters.
    if (static const u64 delta_vs = [] {
            const char* value = std::getenv("EDEN_NX2_DRAW_DELTA_LOG");
            return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ULL;
        }();
        delta_vs != 0) {
        const auto& regs = maxwell3d->regs;
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        static u32 logged{};
        if (vs == delta_vs && logged < 12) {
            ++logged;
            const auto& ds = maxwell3d->draw_manager.draw_state;
            std::string writes;
            for (const auto& w : maxwell3d->WritesSinceDraw()) {
                writes += fmt::format(" {:#x}={:#x}", w.first, w.second);
            }
            LOG_INFO(HW_GPU, "Draw delta: vs {:#x} indexed {} first {} count {} base_vertex {} base_instance {} instances {} index {:#x} fmt {} topology {}; {} writes since the last draw:{}",
                     vs, is_indexed, is_indexed ? ds.index_buffer.first : ds.vertex_buffer.first,
                     is_indexed ? ds.index_buffer.count : ds.vertex_buffer.count, ds.base_index,
                     ds.base_instance, instance_count, regs.index_buffer.StartAddress(),
                     static_cast<u32>(regs.index_buffer.format), static_cast<u32>(ds.topology),
                     maxwell3d->WritesSinceDraw().size(), writes);
        }
        maxwell3d->ClearWritesSinceDraw();
    }
    // EDEN_NX2_CB24_TRACE: for draws whose vertex shader reads bank 24, show the inline
    // constant buffer upload target (CB_ADDRESS) and its first words.
    if (static const bool trace = std::getenv("EDEN_NX2_CB24_TRACE") != nullptr; trace) {
        const auto& regs = maxwell3d->regs;
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        static u32 logged{};
        if ((vs == 0x11471c9a00 || vs == 0x11471ca600 || vs == 0x11471cd300) && logged++ < 3) {
            std::array<u32, 24> words{};
            gpu_memory->ReadBlockUnsafe(regs.const_buffer.Address(), words.data(), sizeof(words));
            LOG_INFO(HW_GPU, "bank24 draw vs={:#x}: CB_ADDRESS {:#x} size {:#x} offset {:#x}: {:08x}", vs,
                     regs.const_buffer.Address(), regs.const_buffer.size, regs.const_buffer.offset,
                     fmt::join(words, " "));
            // Search every bound vertex-stage buffer for render-target-size values.
            const auto& stage = maxwell3d->state.shader_stages[0];
            for (u32 slot = 0; slot < Tegra::Engines::Maxwell3D::Regs::MaxConstBuffers; ++slot) {
                const auto& buffer = stage.const_buffers[slot];
                if (!buffer.enabled || buffer.size == 0) continue;
                std::vector<u32> data(std::min<u32>(buffer.size, 0x10000) / 4);
                gpu_memory->ReadBlockUnsafe(buffer.address, data.data(), data.size() * 4);
                for (size_t i = 0; i < data.size(); ++i) {
                    const u32 w = data[i];
                    if (w == 0x45200000 || w == 0x44b40000 || w == 0x39cccccd || w == 0x3a360b61 ||
                        w == 0x44a00000 || w == 0x44340000 || w == 2559 || w == 1439) {
                        LOG_INFO(HW_GPU, "  slot {} +{:#x} = {:#x}", slot, i * 4, w);
                    }
                }
            }
        }
    }
    // EDEN_NX2_TRACE_VS=<address>: describe the next few draws using that vertex shader like the
    // after-upscale trace does (render target, blend, fragment textures).
    // EDEN_NX2_TRACE_VS_MASK=<hex> compares only those address bits (shader pools move between
    // runs; the offset inside the pool does not).
    static const u64 trace_vs = [] {
        const char* value = std::getenv("EDEN_NX2_TRACE_VS");
        return value ? std::strtoull(value, nullptr, 16) : 0ULL;
    }();
    static const u64 trace_vs_mask = [] {
        const char* value = std::getenv("EDEN_NX2_TRACE_VS_MASK");
        return value ? std::strtoull(value, nullptr, 16) : ~0ULL;
    }();
    if (trace_vs != 0) {
        const auto& regs = maxwell3d->regs;
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        static u32 traced{};
        if ((vs & trace_vs_mask) == (trace_vs & trace_vs_mask) && traced++ % 200 == 0 && traced < 1000) {
            Tegra::Engines::g_nx2_trace_draws_after_upscale = 1;
        }
        // EDEN_NX2_TRACE_FRAME_AFTER=<frame>: once that vertex shader is seen in frame >= n,
        // list every draw and clear of the following frame in order.
        static const u32 trace_frame_after = [] {
            const char* value = std::getenv("EDEN_NX2_TRACE_FRAME_AFTER");
            return value ? static_cast<u32>(std::strtoul(value, nullptr, 10)) : 0U;
        }();
        if (trace_frame_after != 0 && frame_trace_state == 0 && frame_counter >= trace_frame_after &&
            (vs & trace_vs_mask) == (trace_vs & trace_vs_mask)) {
            frame_trace_state = 1;
        }
    }
    if (frame_trace_state == 2) {
        const auto& regs = maxwell3d->regs;
        const auto& ds = maxwell3d->draw_manager.draw_state;
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        const u64 ps = (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1];
        LOG_INFO(HW_GPU,
                 "frame trace: draw vs {:#x} ps {:#x} count {} inst {} | depth {}{}func {:#x} | rts {} rt0 "
                 "{:#x} {}x{} fmt {:#x} layers {} vol {} base {} pitch {:#x} | zeta {:#x} {}x{} fmt {:#x} en {} | "
                 "vp {}x{} | cull {} mask {:#x} | "
                 "blend {} per_target {} eq {:#x} src {:#x} dst {:#x} a_eq {:#x} a_src {:#x} a_dst {:#x} | "
                 "logic {} {:#x}",
                 vs, ps, is_indexed ? ds.index_buffer.count : ds.vertex_buffer.count, instance_count,
                 regs.depth_test_enable ? "T" : "", regs.depth_write_enabled ? "W" : "",
                 static_cast<u32>(regs.depth_test_func), regs.rt_control.count.Value(),
                 regs.rt[0].Address(), regs.rt[0].width, regs.rt[0].height,
                 static_cast<u32>(regs.rt[0].format), regs.rt[0].depth.Value(),
                 regs.rt[0].volume.Value(), regs.rt[0].base_layer, regs.rt[0].array_pitch,
                 regs.zeta.Address(), regs.zeta_size.width,
                 regs.zeta_size.height, static_cast<u32>(regs.zeta.format), regs.zeta_enable,
                 regs.viewport_transform[0].scale_x * 2, regs.viewport_transform[0].scale_y * -2,
                 regs.gl_cull_test_enabled, regs.color_mask[0].raw, regs.blend.enable[0],
                 regs.blend_per_target_enabled,
                 static_cast<u32>(regs.blend_per_target_enabled ? regs.blend_per_target[0].color_op : regs.blend.color_op),
                 static_cast<u32>(regs.blend_per_target_enabled ? regs.blend_per_target[0].color_source : regs.blend.color_source),
                 static_cast<u32>(regs.blend_per_target_enabled ? regs.blend_per_target[0].color_dest : regs.blend.color_dest),
                 static_cast<u32>(regs.blend_per_target_enabled ? regs.blend_per_target[0].alpha_op : regs.blend.alpha_op),
                 static_cast<u32>(regs.blend_per_target_enabled ? regs.blend_per_target[0].alpha_source : regs.blend.alpha_source),
                 static_cast<u32>(regs.blend_per_target_enabled ? regs.blend_per_target[0].alpha_dest : regs.blend.alpha_dest),
                 regs.logic_op.enable, static_cast<u32>(regs.logic_op.op));
        for (u32 i = 1; i < regs.rt_control.count; ++i) {
            LOG_INFO(HW_GPU, "frame trace:   rt{} {:#x} {}x{} fmt {:#x} layers {} base {} mask {:#x}", i,
                     regs.rt[i].Address(), regs.rt[i].width, regs.rt[i].height,
                     static_cast<u32>(regs.rt[i].format), regs.rt[i].depth.Value(),
                     regs.rt[i].base_layer, regs.color_mask[i].raw);
        }
        const auto& sc = regs.scissor_test[0];
        LOG_INFO(HW_GPU,
                 "frame trace:   render_enable mode {} ovr {} exec {} | scissor en {} x {}-{} y {}-{} | "
                 "rasterize {} | depth_bounds en {} {} {} | user_clip {:#x} | vp0 z {} {} near {} far {} | "
                 "stencil {} | clip_ctl {:#x} | ib {:#x} first {} base_v {} | va0 {:#x} | alpha_test {} "
                 "func {:#x} ref {} | aa {} a2c {} | topology {:#x} | poly_off {}{}{} bias {} slope {} "
                 "clamp {} | depth_mode {} z01 {} pix_z {}{}",
                 static_cast<u32>(regs.render_enable.mode), static_cast<u32>(regs.render_enable_override),
                 maxwell3d->ShouldExecute(), sc.enable, sc.min_x.Value(), sc.max_x.Value(),
                 sc.min_y.Value(), sc.max_y.Value(), regs.rasterize_enable, regs.depth_bounds_enable,
                 regs.depth_bounds[0], regs.depth_bounds[1], regs.user_clip_enable.raw,
                 regs.viewport_transform[0].scale_z, regs.viewport_transform[0].translate_z,
                 regs.viewports[0].depth_range_near, regs.viewports[0].depth_range_far,
                 regs.stencil_enable, static_cast<u32>(regs.viewport_clip_control.geometry_clip.Value()),
                 ds.index_buffer.StartAddress(),
                 ds.index_buffer.first, regs.global_base_vertex_index,
                 regs.vertex_streams[0].Address(), regs.alpha_test_enabled,
                 static_cast<u32>(regs.alpha_test_func), regs.alpha_test_ref, regs.anti_alias_enable,
                 regs.anti_alias_alpha_control.alpha_to_coverage.Value(),
                 static_cast<u32>(ds.topology), regs.polygon_offset_fill_enable,
                 regs.polygon_offset_point_enable, regs.polygon_offset_line_enable, regs.depth_bias,
                 regs.slope_scale_depth_bias, regs.depth_bias_clamp,
                 static_cast<u32>(regs.depth_mode),
                 regs.viewport_clip_control.depth_0_to_1.Value(),
                 regs.viewport_clip_control.pixel_min_z.Value(),
                 regs.viewport_clip_control.pixel_max_z.Value());
        // EDEN_NX2_TRACE_REG_DIFF_VS=<hex low20>: when the vertex program changes to or from that
        // one, list every 3D register that differs from the previous draw (finds state the named
        // fields above do not cover).
        static const u32 reg_diff_vs = [] {
            const char* value = std::getenv("EDEN_NX2_TRACE_REG_DIFF_VS");
            return value ? static_cast<u32>(std::strtoul(value, nullptr, 16)) : 0U;
        }();
        if (reg_diff_vs != 0) {
            static std::array<u32, Maxwell::NUM_REGS> previous{};
            static bool previous_valid = false;
            static u32 previous_vs = 0;
            static u32 diffs_logged = 0;
            const u32 vs_low = static_cast<u32>(vs & 0xfffff);
            if (previous_valid && diffs_logged < 12 &&
                static_cast<u32>(regs.viewport_transform[0].scale_x * 2) == 1920 &&
                ((vs_low == reg_diff_vs) != (previous_vs == reg_diff_vs))) {
                ++diffs_logged;
                std::string diff;
                for (u32 i = 0; i < Maxwell::NUM_REGS; ++i) {
                    if (regs.reg_array[i] != previous[i]) {
                        diff += fmt::format(" {:#x}:{:#x}->{:#x}", i * 4, previous[i], regs.reg_array[i]);
                    }
                }
                LOG_INFO(HW_GPU, "frame trace:   reg diff {:#x} -> {:#x}:{}", previous_vs, vs_low, diff);
            }
            std::memcpy(previous.data(), regs.reg_array.data(), sizeof(previous));
            previous_valid = true;
            previous_vs = vs_low;
        }
    }
    // With EDEN_NX2_VS_TEX_TRACE: once the traced image is known, summarise its guest memory as
    // half floats (every 600 frames, to see whether it changes and whether the CPU writes it).
    if (const u64 traced = Tegra::g_nx2_trace_dst_start.load(std::memory_order_relaxed);
        traced != 0 && traced_image_frame + 600 <= frame_counter) {
        traced_image_frame = frame_counter;
        const u64 size = (std::min)(Tegra::g_nx2_trace_dst_size.load(), u64{0x10000});
        std::vector<u16> halfs(size / 2);
        gpu_memory->ReadBlock(traced, halfs.data(), size);
        u32 zeros = 0, nans = 0, negatives = 0, big = 0;
        float min = 1e30f, max = -1e30f;
        for (const u16 h : halfs) {
            if (h == 0) {
                ++zeros;
                continue;
            }
            const u32 exp = (h >> 10) & 0x1f;
            if (exp == 0x1f) {
                ++nans;
                continue;
            }
            const float sign = (h & 0x8000) ? -1.0f : 1.0f;
            const float v = exp == 0 ? sign * std::ldexp(static_cast<float>(h & 0x3ff), -24)
                                     : sign * std::ldexp(static_cast<float>((h & 0x3ff) | 0x400),
                                                         static_cast<int>(exp) - 25);
            min = (std::min)(min, v);
            max = (std::max)(max, v);
            negatives += v < 0;
            big += std::abs(v) > 5.0f;
        }
        LOG_INFO(HW_GPU,
                 "traced image {:#x} guest memory: {} halfs, {} zero, {} nan/inf, {} negative, {} |v|>5, "
                 "min {} max {}, first {:#06x} {:#06x} {:#06x} {:#06x} {:#06x} {:#06x} {:#06x} {:#06x}",
                 traced, halfs.size(), zeros, nans, negatives, big, min, max, halfs[0], halfs[1],
                 halfs[2], halfs[3], halfs[4], halfs[5], halfs[6], halfs[7]);
    }
    // EDEN_NX2_DUMP_VS=<hex low 20 bits | program key prefix>[,...]: write the index buffer, every
    // enabled vertex stream (from base_vertex) and the VS/PS constant buffers of the first 6 draws of
    // each listed vertex program to EDEN_NX2_DUMP_DIR (default /tmp) as
    // vsdump-<vs>-<n>-{ib,vb<b>,cb<i>,pscb<i>}.bin plus a .txt summary. Entries longer than five
    // hex digits are matched against the native program key (stable across sessions).
    static const std::vector<u64> dump_vs = [] {
        std::vector<u64> keys;
        if (const char* value = std::getenv("EDEN_NX2_DUMP_VS")) {
            std::vector<std::string> parts;
            Common::SplitString(value, ',', parts);
            for (const auto& part : parts) {
                if (part.size() <= 5) {
                    keys.push_back(std::strtoull(part.c_str(), nullptr, 16));
                }
            }
        }
        return keys;
    }();
    static const std::vector<std::string> dump_vs_keys = [] {
        std::vector<std::string> keys;
        if (const char* value = std::getenv("EDEN_NX2_DUMP_VS")) {
            std::vector<std::string> parts;
            Common::SplitString(value, ',', parts);
            for (const auto& part : parts) {
                if (part.size() > 5) {
                    keys.push_back(part);
                }
            }
        }
        return keys;
    }();
    static const u32 dump_after = [] {
        const char* value = std::getenv("EDEN_NX2_DUMP_AFTER");
        return value != nullptr ? static_cast<u32>(std::strtoul(value, nullptr, 10)) : 0u;
    }();
    // EDEN_NX2_DUMP_VP=<width>: only draws whose viewport is that wide (same low bits, other pass).
    static const u32 dump_viewport = [] {
        const char* value = std::getenv("EDEN_NX2_DUMP_VP");
        return value != nullptr ? static_cast<u32>(std::strtoul(value, nullptr, 10)) : 0u;
    }();
    if ((!dump_vs.empty() || !dump_vs_keys.empty()) && frame_counter >= dump_after) {
        const auto& regs = maxwell3d->regs;
        const u64 vs_low = regs.pipelines[1].reserved[1] & 0xfffff;
        const auto program_key = [&regs](size_t pipeline) {
            return fmt::format("{:016x}",
                               Tegra::Nx2ProgramKey((u64{regs.pipelines[pipeline].reserved[0] & 0xff} << 32) |
                                                    regs.pipelines[pipeline].reserved[1]));
        };
        const bool key_match = std::any_of(dump_vs_keys.begin(), dump_vs_keys.end(), [&](const auto& k) {
            return program_key(1).starts_with(k);
        });
        if ((key_match || std::find(dump_vs.begin(), dump_vs.end(), vs_low) != dump_vs.end()) &&
            vs_dump_counts[vs_low] < 6 &&
            (dump_viewport == 0 ||
             static_cast<u32>(regs.viewport_transform[0].scale_x * 2) == dump_viewport)) {
            const u32 n = vs_dump_counts[vs_low]++;
            const char* dir = std::getenv("EDEN_NX2_DUMP_DIR");
            const std::string base =
                fmt::format("{}/vsdump-{:05x}-{}", dir ? dir : "/tmp", vs_low, n);
            const auto& ds = maxwell3d->draw_manager.draw_state;
            const auto dump = [&](const std::string& name, GPUVAddr addr, u64 size) {
                std::vector<u8> data(size);
                gpu_memory->ReadBlock(addr, data.data(), size);
                std::ofstream out(base + "-" + name + ".bin", std::ios::binary);
                out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(size));
            };
            std::ofstream txt(base + ".txt");
            const u32 index_size = is_indexed ? static_cast<u32>(regs.index_buffer.FormatSizeInBytes()) : 0;
            const u32 count = is_indexed ? ds.index_buffer.count : ds.vertex_buffer.count;
            if (is_indexed) {
                dump("ib", regs.index_buffer.StartAddress() + u64{ds.index_buffer.first} * index_size,
                     u64{count} * index_size);
            }
            u32 max_index = 0;
            if (is_indexed) {
                std::vector<u8> idx(u64{count} * index_size);
                gpu_memory->ReadBlock(regs.index_buffer.StartAddress() + u64{ds.index_buffer.first} * index_size,
                                      idx.data(), idx.size());
                for (u32 i = 0; i < count; ++i) {
                    u32 v = 0;
                    std::memcpy(&v, idx.data() + u64{i} * index_size, index_size);
                    max_index = (std::max)(max_index, v);
                }
            } else {
                max_index = ds.vertex_buffer.first + count - 1;
            }
            const u32 base_vertex = is_indexed ? ds.base_index : 0;
            const u64 vertex_count = u64{max_index} + 1;
            txt << fmt::format("vs {:#x} ({}) ps {:#x} ({}) indexed {} index_size {} first {} count {} "
                               "base_vertex {} max_index {} ib {:#x}\n",
                               regs.pipelines[1].reserved[1], program_key(1),
                               regs.pipelines[5].reserved[1], program_key(5), is_indexed,
                               index_size, ds.index_buffer.first, count, base_vertex, max_index,
                               regs.index_buffer.StartAddress());
            u32 streams_used = 0;
            for (u32 i = 0; i < 32; ++i) {
                const auto& attr = regs.vertex_attrib_format[i];
                if (attr.constant || !regs.vertex_streams[attr.buffer].enable) {
                    continue;
                }
                streams_used |= 1u << attr.buffer;
                txt << fmt::format("attrib[{}] buffer {} offset {:#x} size {:#x} type {}\n", i,
                                   attr.buffer.Value(), attr.offset.Value(),
                                   static_cast<u32>(attr.size.Value()),
                                   static_cast<u32>(attr.type.Value()));
            }
            u64 vb_size = 0;
            for (u32 b = 0; b < 32; ++b) {
                if (((streams_used >> b) & 1) == 0) {
                    continue;
                }
                const auto& stream = regs.vertex_streams[b];
                const u64 stride = stream.stride.Value();
                const u64 size = vertex_count * stride;
                dump(fmt::format("vb{}", b), stream.Address() + u64{base_vertex} * stride, size);
                vb_size += size;
                txt << fmt::format("stream{} {:#x} stride {} dumped from vertex {} ({} vertices)\n", b,
                                   stream.Address(), stride, base_vertex, vertex_count);
            }
            for (const u32 stage : {0u, 4u}) {
                const auto& cbufs = maxwell3d->state.shader_stages[stage].const_buffers;
                for (u32 i = 0; i < cbufs.size(); ++i) {
                    if (!cbufs[i].enabled || cbufs[i].size == 0) {
                        continue;
                    }
                    const u64 size = (std::min)(u64{cbufs[i].size}, u64{0x10000});
                    dump(fmt::format("{}cb{}", stage == 0 ? "" : "ps", i), cbufs[i].address, size);
                    txt << fmt::format("{}cbuf{} {:#x} size {:#x}\n", stage == 0 ? "" : "ps ", i,
                                       cbufs[i].address, cbufs[i].size);
                }
            }
            // EDEN_NX2_DUMP_SSBO=<hex offset in PS cb0>: also dump the storage buffer whose
            // address/size descriptor lives there (first 1 MiB).
            if (const char* ssbo_text = std::getenv("EDEN_NX2_DUMP_SSBO")) {
                const auto& cb0 = maxwell3d->state.shader_stages[4].const_buffers[0];
                const u64 offset = std::strtoull(ssbo_text, nullptr, 16);
                if (cb0.enabled && offset + 12 <= cb0.size) {
                    u64 ssbo_addr = 0;
                    u32 ssbo_size = 0;
                    gpu_memory->ReadBlock(cb0.address + offset, &ssbo_addr, sizeof(ssbo_addr));
                    gpu_memory->ReadBlock(cb0.address + offset + 8, &ssbo_size, sizeof(ssbo_size));
                    txt << fmt::format("ps ssbo cb0+{:#x}: {:#x} size {:#x}\n", offset, ssbo_addr, ssbo_size);
                    if (ssbo_addr != 0 && ssbo_size != 0) {
                        dump("psssbo", ssbo_addr, (std::min)(u64{ssbo_size}, u64{1} << 20));
                    }
                }
            }
            txt << fmt::format("viewport scale {} {} {} translate {} {} {} | depth {}{} func {:#x}\n",
                               regs.viewport_transform[0].scale_x, regs.viewport_transform[0].scale_y,
                               regs.viewport_transform[0].scale_z, regs.viewport_transform[0].translate_x,
                               regs.viewport_transform[0].translate_y, regs.viewport_transform[0].translate_z,
                               regs.depth_test_enable ? "T" : "", regs.depth_write_enabled ? "W" : "",
                               static_cast<u32>(regs.depth_test_func));
            LOG_INFO(HW_GPU, "vs dump {} written ({} indices, vb {:#x} bytes)", base, count, vb_size);
        }
    }
    // EDEN_NX2_TRACE_COUNT=<n>: describe the first draws of exactly n indices/vertices (to compare
    // the passes that draw one mesh).
    static const u64 trace_count = [] {
        const char* value = std::getenv("EDEN_NX2_TRACE_COUNT");
        return value ? std::strtoull(value, nullptr, 10) : 0ULL;
    }();
    if (trace_count != 0) {
        const u64 count = is_indexed ? maxwell3d->draw_manager.draw_state.index_buffer.count
                                     : maxwell3d->draw_manager.draw_state.vertex_buffer.count;
        static u32 traced{};
        if (count == trace_count && traced++ < 60) {
            Tegra::Engines::g_nx2_trace_draws_after_upscale = 1;
        }
    }
    if (Tegra::Engines::g_nx2_trace_draws_after_upscale > 0) {
        --Tegra::Engines::g_nx2_trace_draws_after_upscale;
        const auto& regs = maxwell3d->regs;
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        const u64 ps = (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1];
        LOG_INFO(HW_GPU, "after upscale: draw rt0 {:#x} {}x{} fmt {:#x} block {}/{}/{} pitch_linear {} depth {} array_pitch {:#x} ctrl {:#x} vs {:#x} ps {:#x}",
                 regs.rt[0].Address(), regs.rt[0].width, regs.rt[0].height,
                 static_cast<u32>(regs.rt[0].format), regs.rt[0].tile_mode.block_width.Value(),
                 regs.rt[0].tile_mode.block_height.Value(), regs.rt[0].tile_mode.block_depth.Value(),
                 regs.rt[0].tile_mode.is_pitch_linear.Value(), regs.rt[0].depth.Value(),
                 regs.rt[0].array_pitch, regs.rt_control.count.Value(), vs, ps);
        // Fragment-stage texture handles (bound buffer 2) and the images they name.
        const auto& handles = maxwell3d->state.shader_stages[4].const_buffers[2];
        if (handles.enabled) {
            for (u32 i = 0; i < 24; ++i) {
                const u32 handle = gpu_memory->Read<u32>(handles.address + i * 4);
                if (handle == 0 || handle == 0xffffffff) continue;
                Tegra::Texture::TICEntry tic{};
                gpu_memory->ReadBlockUnsafe(regs.tex_header.Address() + u64{handle & 0xfffff} * sizeof(tic),
                                            &tic, sizeof(tic));
                LOG_INFO(HW_GPU, "  fs handle[{}] = {:#x}: {:#x} {}x{} format {:#x}", i, handle,
                         tic.Address(), tic.Width(), tic.Height(), static_cast<u32>(tic.format.Value()));
            }
        }
        // Vertex input: enabled attributes (format type/size/offset/buffer) and their streams.
        for (u32 i = 0; i < Maxwell::NumVertexAttributes; ++i) {
            const auto& attrib = regs.vertex_attrib_format[i];
            if (attrib.constant != 0 || attrib.hex == 0) continue;
            const auto& stream = regs.vertex_streams[attrib.buffer];
            LOG_INFO(HW_GPU, "  attrib[{}] = {:#x}: buffer {} offset {:#x} size {:#x} type {} bgra {} | stream stride {} enabled {} at {:#x}",
                     i, attrib.hex, attrib.buffer.Value(), attrib.offset.Value(),
                     static_cast<u32>(attrib.size.Value()), static_cast<u32>(attrib.type.Value()),
                     attrib.bgra.Value(), stream.stride.Value(), stream.enable.Value(), stream.Address());
        }
        LOG_INFO(HW_GPU, "  topology {:#x} index format {} provoking {} draw mode {}",
                 static_cast<u32>(regs.draw.topology.Value()), static_cast<u32>(regs.index_buffer.format),
                 static_cast<u32>(regs.provoking_vertex), static_cast<u32>(regs.draw.instance_id.Value()));
        {
            const auto& ds = maxwell3d->draw_manager.draw_state;
            const auto& vp = regs.viewport_transform[0];
            const auto& vs_cbufs = maxwell3d->state.shader_stages[0].const_buffers;
            LOG_INFO(HW_GPU,
                     "  index buffer {:#x} first {} count {} base_vertex {} | vertex first {} count {} | "
                     "instances {} base_instance {}",
                     regs.index_buffer.StartAddress(), ds.index_buffer.first, ds.index_buffer.count,
                     ds.base_index, ds.vertex_buffer.first, ds.vertex_buffer.count,
                     ds.instance_count, ds.base_instance);
            LOG_INFO(HW_GPU,
                     "  depth test {} write {} func {:#x} | polygon offset fill {} point {} line {} "
                     "units {} factor {} clamp {} | cull {} face {} front {} | viewport scale {} {} {} "
                     "translate {} {} {} | depth_mode {} clip {:#x}",
                     regs.depth_test_enable, regs.depth_write_enabled,
                     static_cast<u32>(regs.depth_test_func), regs.polygon_offset_fill_enable,
                     regs.polygon_offset_point_enable, regs.polygon_offset_line_enable,
                     regs.depth_bias, regs.slope_scale_depth_bias, regs.depth_bias_clamp,
                     regs.gl_cull_test_enabled, static_cast<u32>(regs.gl_cull_face),
                     static_cast<u32>(regs.gl_front_face), vp.scale_x, vp.scale_y, vp.scale_z,
                     vp.translate_x, vp.translate_y, vp.translate_z,
                     static_cast<u32>(regs.depth_mode), static_cast<u32>(regs.viewport_clip_control.geometry_clip.Value()));
            LOG_INFO(HW_GPU, "  vs cbuf3 {:#x}/{:#x} cbuf5 {:#x}/{:#x} cbuf2 {:#x} | zeta {:#x} {}x{} fmt {:#x} | host depth bias enabled {}",
                     vs_cbufs[3].address, vs_cbufs[3].size, vs_cbufs[5].address, vs_cbufs[5].size,
                     vs_cbufs[2].address, regs.zeta.Address(), regs.zeta_size.width,
                     regs.zeta_size.height, static_cast<u32>(regs.zeta.format), last_depth_bias_enable);
        }
        const auto& bt = regs.blend_per_target[0];
        LOG_INFO(HW_GPU, "  blend enable {} per-target {} global op {} src {:#x} dst {:#x} | rt0 op {} src {:#x} dst {:#x} alpha {:#x}/{:#x} mask {:#x}",
                 regs.blend.enable[0], regs.blend_per_target_enabled,
                 static_cast<u32>(regs.blend.color_op), static_cast<u32>(regs.blend.color_source),
                 static_cast<u32>(regs.blend.color_dest), static_cast<u32>(bt.color_op),
                 static_cast<u32>(bt.color_source), static_cast<u32>(bt.color_dest),
                 static_cast<u32>(bt.alpha_source), static_cast<u32>(bt.alpha_dest),
                 regs.color_mask[0].raw);
    }
    // EDEN_NX2_SKIP_VS=<address>[,<address>...]: drop draws using those vertex shaders (debug aid).
    static const std::vector<u64> skip_list = [] {
        std::vector<u64> list;
        if (const char* value = std::getenv("EDEN_NX2_SKIP_VS")) {
            for (char* end = nullptr; *value; value = *end ? end + 1 : end) {
                list.push_back(std::strtoull(value, &end, 16));
                if (end == value) break;
            }
        }
        return list;
    }();
    if (!skip_list.empty()) {
        const auto& regs = maxwell3d->regs;
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        if (std::find(skip_list.begin(), skip_list.end(), vs) != skip_list.end()) return;
    }
    // EDEN_NX2_RT_TRACE: log the first draw to every distinct colour target (debug aid).
    if (static const bool trace = std::getenv("EDEN_NX2_RT_TRACE") != nullptr; trace) {
        static std::set<std::tuple<GPUVAddr, u32, u32, u64, u64>> seen;
        const auto& regs = maxwell3d->regs;
        const auto& rt = regs.rt[0];
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        const u64 ps = (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1];
        const auto key = std::make_tuple(rt.Address(), rt.width, static_cast<u32>(rt.format), vs, ps);
        if (rt.width >= 640 && seen.size() < 600 && seen.insert(key).second) {
            LOG_INFO(HW_GPU, "nx2 draw target {:#x} {}x{} format={:#x} vs={:#x} ps={:#x} blend={} (pairs so far {})",
                     rt.Address(), rt.width, rt.height, static_cast<u32>(rt.format), vs, ps,
                     regs.blend.enable[0], seen.size());
        }
    }
    // Draws whose depth target lies in the EDEN_NX2_TRACE_DST range: who renders a shadow atlas.
    if (const auto& regs = maxwell3d->regs;
        regs.zeta_enable && Tegra::Nx2TracedDestination(regs.zeta.Address(), 1)) {
        static std::set<std::tuple<u64, u64, u32>> seen;
        static u32 draws{};
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        const u64 ps = (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1];
        ++draws;
        if (seen.size() < 200 &&
            seen.insert({vs, ps, static_cast<u32>(regs.depth_test_func)}).second) {
            LOG_INFO(HW_GPU,
                     "nx2 draw into traced zeta {:#x} {}x{} layers {} fmt {:#x} vs={:#x} ({:016x}) ps={:#x} "
                     "depth test {} write {} func {} | rt0 {:#x} fmt {:#x} rt count {} | viewport {}x{} | "
                     "draw {} so far",
                     regs.zeta.Address(), regs.zeta_size.width, regs.zeta_size.height,
                     regs.zeta_size.depth, static_cast<u32>(regs.zeta.format), vs,
                     Tegra::Nx2ProgramKey(vs), ps, regs.depth_test_enable, regs.depth_write_enabled,
                     static_cast<u32>(regs.depth_test_func), regs.rt[0].Address(),
                     static_cast<u32>(regs.rt[0].format), regs.rt_control.count.Value(),
                     static_cast<s32>(regs.viewport_transform[0].scale_x * 2.0f),
                     static_cast<s32>(std::abs(regs.viewport_transform[0].scale_y) * 2.0f), draws);
        }
    }
    // Likewise for a colour target in the range (what fills a texture nobody uploads).
    if (const auto& regs = maxwell3d->regs; true) {
        for (u32 i = 0; i < regs.rt_control.count; ++i) {
            if (!Tegra::Nx2TracedDestination(regs.rt[i].Address(), 1)) {
                continue;
            }
            static u32 traced{};
            if (traced++ < 60) {
                const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
                const u64 ps = (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1];
                LOG_INFO(HW_GPU,
                         "nx2 draw into traced rt{} {:#x} {}x{} fmt {:#x} vs={:#x} ({:016x}) ps={:#x} ({:016x}) "
                         "rt count {} viewport {}x{} indexed {} instances {}",
                         i, regs.rt[i].Address(), regs.rt[i].width, regs.rt[i].height,
                         static_cast<u32>(regs.rt[i].format), vs, Tegra::Nx2ProgramKey(vs), ps,
                         Tegra::Nx2ProgramKey(ps), regs.rt_control.count.Value(),
                         static_cast<s32>(regs.viewport_transform[0].scale_x * 2.0f),
                         static_cast<s32>(std::abs(regs.viewport_transform[0].scale_y) * 2.0f),
                         is_indexed, instance_count);
            }
            break;
        }
    }
    Nx2DumpRenderTarget(false);
    PrepareDraw(is_indexed, [this, is_indexed, instance_count] {
        const auto& draw_state = maxwell3d->draw_manager.draw_state;
        const u32 num_instances{instance_count};
        const DrawParams draw_params{MakeDrawParams(draw_state, num_instances, is_indexed)};

        if (mesh_draw) {
            scheduler.Record([groups = draw_params.num_vertices](vk::CommandBuffer cmdbuf) {
                cmdbuf.DrawMeshTasksEXT(groups, 1, 1);
            });
            return;
        }
        scheduler.Record([draw_params](vk::CommandBuffer cmdbuf) {
            if (draw_params.is_indexed) {
                cmdbuf.DrawIndexed(draw_params.num_vertices, draw_params.num_instances,
                                   draw_params.first_index, draw_params.base_vertex,
                                   draw_params.base_instance);
            } else {
                cmdbuf.Draw(draw_params.num_vertices, draw_params.num_instances,
                            draw_params.base_vertex, draw_params.base_instance);
            }
        });

        // Log draw call
        if (GPU::Logging::IsActive() &&
            Settings::values.gpu_log_vulkan_calls.GetValue()) {
            const std::string params = is_indexed ?
                fmt::format("vertices={}, instances={}, firstIndex={}, baseVertex={}, baseInstance={}",
                    draw_params.num_vertices, draw_params.num_instances,
                    draw_params.first_index, draw_params.base_vertex, draw_params.base_instance) :
                fmt::format("vertices={}, instances={}, firstVertex={}, firstInstance={}",
                    draw_params.num_vertices, draw_params.num_instances,
                    draw_params.base_vertex, draw_params.base_instance);
            GPU::Logging::GPULogger::GetInstance().LogVulkanCall(
                is_indexed ? "vkCmdDrawIndexed" : "vkCmdDraw", params, VK_SUCCESS);
        }
    });
    Nx2DumpRenderTarget(true);
}

// EDEN_NX2_RT_DUMP_EVERY=<n> (with EDEN_NX2_IMAGE_DUMP_DIR): while a frame trace is running, read
// colour target 0 back after every n-th draw into it that targets a >= 1280 wide image, so the
// draw that paints a region wrong can be found from the file numbering (`rtdraw-<draw index>`).
// Depth-only passes leave the file unchanged, so the first dumps show the previous frame.
// EDEN_NX2_RT_DUMP_TAIL=1 restricts this to the draws from the first blended one on (the
// post-process and UI tail, whose indices move from scene to scene), with one extra dump of the
// state before that draw.
void RasterizerVulkan::Nx2DumpRenderTarget(bool after_draw) {
    static const u32 every = [] {
        const char* value = std::getenv("EDEN_NX2_RT_DUMP_EVERY");
        return value ? static_cast<u32>(std::strtoul(value, nullptr, 10)) : 0U;
    }();
    static const bool tail = std::getenv("EDEN_NX2_RT_DUMP_TAIL") != nullptr;
    if (frame_trace_state != 2) {
        return;
    }
    const auto& regs = maxwell3d->regs;
    // Every colour target of a traced-frame draw that renders into the EDEN_NX2_TRACE_DST range,
    // after the draw (the image a later pass samples from that range, as that draw left it).
    static u32 traced_draws{};
    if (after_draw && traced_draws < 16) {
        bool traced = false;
        for (u32 i = 0; i < regs.rt_control.count; ++i) {
            if (!Tegra::Nx2TracedDestination(regs.rt[i].Address(), 1)) {
                continue;
            }
            traced = true;
            if (const auto id = texture_cache.Nx2RenderTargetImage(i)) {
                LOG_INFO(HW_GPU, "frame trace: traced rt{} {:#x} is image {} ({}x{} layers {})", i,
                         regs.rt[i].Address(), id.index, texture_cache.Nx2ImageInfo(id).size.width,
                         texture_cache.Nx2ImageInfo(id).size.height,
                         texture_cache.Nx2ImageInfo(id).resources.layers);
                texture_cache.DumpImage(id, fmt::format("traced-rt{}-{:02}", i, traced_draws));
            }
        }
        traced_draws += traced ? 1 : 0;
    }
    if (every == 0 || regs.rt_control.count == 0 || regs.rt[0].width < 1280) {
        return;
    }
    static u32 draws{};
    static bool in_tail{};
    const auto image_id = texture_cache.Nx2RenderTargetImage(0);
    if (!after_draw) {
        if (tail && !in_tail && regs.blend.enable[0]) {
            in_tail = true;
            if (image_id) {
                texture_cache.DumpImage(image_id, fmt::format("rtdraw-{:04}-before", draws));
            }
        }
        return;
    }
    const u32 index = draws++;
    if ((tail && !in_tail) || index % every != 0 || !image_id) {
        return;
    }
    const u64 ps = (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1];
    LOG_INFO(HW_GPU, "frame trace: rt0 dump after draw {} (ps {:#x})", index, ps);
    texture_cache.DumpImage(image_id, fmt::format("rtdraw-{:04}", index));
}

void RasterizerVulkan::DrawIndirect() {
    if (!chunk_batches.empty()) {
        FlushChunkBatches();
    }
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Draw};
    const auto& params = maxwell3d->draw_manager.indirect_state;
    if (static const bool stats = std::getenv("EDEN_NX2_DRAW_STATS") != nullptr; stats) {
        RecordDrawStats(params.max_draw_counts, 0, true);
    }
    buffer_cache.SetDrawIndirect(&params);
    PrepareDraw(params.is_indexed, [this, &params] {
        if (mesh_draw) {
            // Indirect task/mesh draws would need vkCmdDrawMeshTasksIndirectEXT's layout.
            static bool logged{};
            if (!std::exchange(logged, true)) {
                LOG_WARNING(Render_Vulkan, "Indirect task/mesh draw skipped");
            }
            return;
        }
        const auto indirect_buffer = buffer_cache.GetDrawIndirectBuffer();
        const auto& buffer = indirect_buffer.first;
        const auto& offset = indirect_buffer.second;
        if (params.is_byte_count) {
            scheduler.Record([buffer_obj = buffer->Handle(), offset,
                              stride = params.stride](vk::CommandBuffer cmdbuf) {
                cmdbuf.DrawIndirectByteCountEXT(1, 0, buffer_obj, offset, 0,
                                                static_cast<u32>(stride));
            });
            return;
        }
        if (params.include_count) {
            const auto count = buffer_cache.GetDrawIndirectCount();
            const auto& draw_buffer = count.first;
            const auto& offset_base = count.second;
            scheduler.Record([draw_buffer_obj = draw_buffer->Handle(),
                              buffer_obj = buffer->Handle(), offset_base, offset,
                              params](vk::CommandBuffer cmdbuf) {
                if (params.is_indexed) {
                    cmdbuf.DrawIndexedIndirectCount(
                        buffer_obj, offset, draw_buffer_obj, offset_base,
                        static_cast<u32>(params.max_draw_counts), static_cast<u32>(params.stride));
                } else {
                    cmdbuf.DrawIndirectCount(buffer_obj, offset, draw_buffer_obj, offset_base,
                                             static_cast<u32>(params.max_draw_counts),
                                             static_cast<u32>(params.stride));
                }
            });
            return;
        }
        scheduler.Record([buffer_obj = buffer->Handle(), offset, params](vk::CommandBuffer cmdbuf) {
            if (params.is_indexed) {
                cmdbuf.DrawIndexedIndirect(buffer_obj, offset,
                                           static_cast<u32>(params.max_draw_counts),
                                           static_cast<u32>(params.stride));
            } else {
                cmdbuf.DrawIndirect(buffer_obj, offset, static_cast<u32>(params.max_draw_counts),
                                    static_cast<u32>(params.stride));
            }
        });

        // Log indirect draw call
        if (GPU::Logging::IsActive() &&
            Settings::values.gpu_log_vulkan_calls.GetValue()) {
            const std::string log_params = fmt::format("drawCount={}, stride={}",
                params.max_draw_counts, params.stride);
            GPU::Logging::GPULogger::GetInstance().LogVulkanCall(
                params.is_indexed ? "vkCmdDrawIndexedIndirect" : "vkCmdDrawIndirect",
                log_params, VK_SUCCESS);
        }
    });
    buffer_cache.SetDrawIndirect(nullptr);
}

void RasterizerVulkan::DrawTexture() {
    if (!chunk_batches.empty()) {
        FlushChunkBatches();
    }

    SCOPE_EXIT {
        gpu.TickWork();
    };
    FlushWork();

    std::scoped_lock l{texture_cache.mutex};
    texture_cache.SynchronizeDescriptors(false);
    texture_cache.UpdateRenderTargets(false);

    UpdateDynamicStates();

    query_cache.NotifySegment(true);
    query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64, maxwell3d->regs.zpass_pixel_count_enable);
    const auto& draw_texture_state = maxwell3d->draw_manager.draw_texture_state;
    const auto& sampler = texture_cache.GetSampler(draw_texture_state.src_sampler, false);
    const auto& texture = texture_cache.GetImageView(draw_texture_state.src_texture);
    const auto* framebuffer = texture_cache.GetFramebuffer();

    const bool src_rescaling = texture_cache.IsRescaling() && texture.IsRescaled();
    const bool dst_rescaling = texture_cache.IsRescaling() && framebuffer->IsRescaled();

    const auto ScaleSrc = [&](auto dim_f) -> s32 {
        auto dim = static_cast<s32>(dim_f);
        return src_rescaling ? Settings::values.resolution_info.ScaleUp(dim) : dim;
    };

    const auto ScaleDst = [&](auto dim_f) -> s32 {
        auto dim = static_cast<s32>(dim_f);
        return dst_rescaling ? Settings::values.resolution_info.ScaleUp(dim) : dim;
    };

    Region2D dst_region = {Offset2D{.x = ScaleDst(draw_texture_state.dst_x0),
                                    .y = ScaleDst(draw_texture_state.dst_y0)},
                           Offset2D{.x = ScaleDst(draw_texture_state.dst_x1),
                                    .y = ScaleDst(draw_texture_state.dst_y1)}};
    Region2D src_region = {Offset2D{.x = ScaleSrc(draw_texture_state.src_x0),
                                    .y = ScaleSrc(draw_texture_state.src_y0)},
                           Offset2D{.x = ScaleSrc(draw_texture_state.src_x1),
                                    .y = ScaleSrc(draw_texture_state.src_y1)}};
    Extent3D src_size = {static_cast<u32>(ScaleSrc(texture.size.width)),
                         static_cast<u32>(ScaleSrc(texture.size.height)), texture.size.depth};
    blit_image.BlitColor(framebuffer, texture.RenderTarget(), texture.ImageHandle(),
                         sampler->Handle(), dst_region, src_region, src_size);
}

void RasterizerVulkan::Clear(u32 layer_count) {
    if (!chunk_batches.empty()) {
        FlushChunkBatches();
    }
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Clear};
    if (Tegra::Engines::g_nx2_trace_draws_after_upscale > 0) {
        const auto& regs = maxwell3d->regs;
        LOG_INFO(HW_GPU, "after upscale: clear rt0 {:#x} {}x{} fmt {:#x} surface {:#x} colour {} {} {} {}",
                 regs.rt[0].Address(), regs.rt[0].width, regs.rt[0].height,
                 static_cast<u32>(regs.rt[0].format), regs.clear_surface.raw, regs.clear_color[0],
                 regs.clear_color[1], regs.clear_color[2], regs.clear_color[3]);
    }
    if (frame_trace_state == 2) {
        const auto& regs = maxwell3d->regs;
        LOG_INFO(HW_GPU,
                 "frame trace: clear surface {:#x} depth {} color {} {} {} {} | rt0 {:#x} {}x{} fmt {:#x} | zeta {:#x} {}x{} "
                 "fmt {:#x} en {} | scissor {}",
                 regs.clear_surface.raw, regs.clear_depth, regs.clear_color[0],
                 regs.clear_color[1], regs.clear_color[2], regs.clear_color[3],
                 regs.rt[0].Address(), regs.rt[0].width,
                 regs.rt[0].height, static_cast<u32>(regs.rt[0].format), regs.zeta.Address(),
                 regs.zeta_size.width, regs.zeta_size.height, static_cast<u32>(regs.zeta.format),
                 regs.zeta_enable, regs.clear_control.use_scissor.Value());
    }
    TraceWindowTarget(*maxwell3d, *gpu_memory, "clear", layer_count);
    FlushWork();
    gpu_memory->FlushCaching();

    auto& regs = maxwell3d->regs;
    const bool use_color = regs.clear_surface.R || regs.clear_surface.G || regs.clear_surface.B ||
                           regs.clear_surface.A;
    const bool use_depth = regs.clear_surface.Z;
    const bool use_stencil = regs.clear_surface.S;
    if (!use_color && !use_depth && !use_stencil) {
        return;
    }
    if (use_depth && regs.zeta_enable && Tegra::Nx2TracedDestination(regs.zeta.Address(), 1)) {
        static u32 traced{};
        if (traced++ < 300) {
            LOG_INFO(HW_GPU, "nx2 clear of traced zeta {:#x} {}x{} layers {}: depth {} layer {} scissor {} ({}x{} at {},{})",
                     regs.zeta.Address(), regs.zeta_size.width, regs.zeta_size.height,
                     regs.zeta_size.depth, regs.clear_depth, regs.clear_surface.layer.Value(),
                     regs.clear_control.use_scissor.Value(),
                     regs.scissor_test[0].max_x - regs.scissor_test[0].min_x,
                     regs.scissor_test[0].max_y - regs.scissor_test[0].min_y,
                     regs.scissor_test[0].min_x, regs.scissor_test[0].min_y);
        }
    }
    if (use_color && regs.clear_surface.RT < regs.rt_control.count &&
        Tegra::Nx2TracedDestination(regs.rt[regs.clear_surface.RT].Address(), 1)) {
        static u32 traced{};
        if (traced++ < 60) {
            const auto& rt = regs.rt[regs.clear_surface.RT];
            LOG_INFO(HW_GPU, "nx2 clear of traced rt{} {:#x} {}x{} fmt {:#x}: color {} {} {} {} layer {} scissor {}",
                     regs.clear_surface.RT.Value(), rt.Address(), rt.width, rt.height,
                     static_cast<u32>(rt.format), regs.clear_color[0], regs.clear_color[1],
                     regs.clear_color[2], regs.clear_color[3], regs.clear_surface.layer.Value(),
                     regs.clear_control.use_scissor.Value());
        }
    }

    std::scoped_lock lock{texture_cache.mutex};
    texture_cache.UpdateRenderTargets(true);
    const Framebuffer* const framebuffer = texture_cache.GetFramebuffer();
    const VkExtent2D render_area = framebuffer->RenderArea();

    constexpr bool ENABLE_DEFERRED_CLEAR = true;
    const bool color_full_channels = regs.clear_surface.R && regs.clear_surface.G &&
                                     regs.clear_surface.B && regs.clear_surface.A;
    const bool stencil_partial = use_stencil && framebuffer->HasAspectStencilBit() &&
                                 regs.stencil_front_mask != 0xFF && regs.stencil_front_mask != 0;
    const bool ds_used = use_depth || use_stencil;
    const bool ds_deferrable =
        !ds_used || ((!framebuffer->HasAspectDepthBit() || use_depth) &&
                     (!framebuffer->HasAspectStencilBit() || use_stencil) && !stencil_partial);
    u32 up_scale = 1;
    u32 down_shift = 0;
    if (texture_cache.IsRescaling()) {
        up_scale = Settings::values.resolution_info.up_scale;
        down_shift = Settings::values.resolution_info.down_shift;
    }
    VkRect2D default_scissor{};
    default_scissor.offset.x = 0;
    default_scissor.offset.y = 0;
    default_scissor.extent.width = (std::numeric_limits<s32>::max)();
    default_scissor.extent.height = (std::numeric_limits<s32>::max)();

    VkClearRect clear_rect{
        .rect = regs.clear_control.use_scissor ? GetScissorState(regs, 0, up_scale, down_shift)
                                               : default_scissor,
        .baseArrayLayer = regs.clear_surface.layer,
        .layerCount = layer_count,
    };
    const auto clamp_rect_to_render_area = [render_area](VkRect2D& rect) -> bool {
        const auto clamp_axis = [](s32& offset, u32& extent, u32 limit) {
            auto clamp_offset = [&offset, limit]() {
                if (limit == 0) {
                    offset = 0;
                    return;
                }
                offset = std::clamp(offset, 0, static_cast<s32>(limit));
            };

            if (extent == 0) {
                clamp_offset();
                return;
            }
            if (offset < 0) {
                const u32 shrink = (std::min)(extent, static_cast<u32>(-offset));
                extent -= shrink;
                offset = 0;
            }
            if (limit == 0) {
                extent = 0;
                offset = 0;
                return;
            }
            if (offset >= s32(limit)) {
                offset = s32(limit);
                extent = 0;
                return;
            }
            const u64 end_coord = u64(offset) + extent;
            if (end_coord > limit) {
                extent = limit - u32(offset);
            }
        };

        clamp_axis(rect.offset.x, rect.extent.width, render_area.width);
        clamp_axis(rect.offset.y, rect.extent.height, render_area.height);
        return rect.extent.width != 0 && rect.extent.height != 0;
    };
    if (!clamp_rect_to_render_area(clear_rect.rect)) {
        return;
    }

    const bool clear_covers_render_area =
        clear_rect.rect.offset.x == 0 && clear_rect.rect.offset.y == 0 &&
        clear_rect.rect.extent.width >= render_area.width &&
        clear_rect.rect.extent.height >= render_area.height;
    const bool can_defer_clear = ENABLE_DEFERRED_CLEAR && (!regs.clear_control.use_scissor || clear_covers_render_area) &&
                                 regs.clear_surface.layer == 0 &&
                                 !scheduler.IsRenderPassActive() &&
                                 (!use_color || color_full_channels) && ds_deferrable;
    if (!can_defer_clear) {
        scheduler.RequestRenderpass(framebuffer);
    }

    query_cache.NotifySegment(true);
    query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64, maxwell3d->regs.zpass_pixel_count_enable);
    UpdateViewportsState(regs);

    const u32 color_attachment = regs.clear_surface.RT;
    if (use_color && framebuffer->HasAspectColorBit(color_attachment)) {
        const auto format = VideoCore::Surface::PixelFormatFromRenderTargetFormat(regs.rt[color_attachment].format);
        bool is_integer = IsPixelFormatInteger(format);
        bool is_signed = IsPixelFormatSignedInteger(format);
        size_t int_size = PixelComponentSizeBitsInteger(format);
        VkClearValue clear_value{};
        if (!is_integer) {
            std::memcpy(clear_value.color.float32, regs.clear_color.data(), regs.clear_color.size() * sizeof(f32));
        } else if (!is_signed) {
            for (size_t i = 0; i < 4; i++)
                clear_value.color.uint32[i] = u32(f32(u64(int_size) << 1U) * regs.clear_color[i]);
        } else {
            for (size_t i = 0; i < 4; i++)
                clear_value.color.int32[i] = s32(f32(s64(int_size - 1) << 1) * (regs.clear_color[i] - 0.5f));
        }

        if (color_full_channels) {
            if (can_defer_clear) {
                scheduler.DeferColorClear(framebuffer, color_attachment, clear_value);
            } else {
                scheduler.Record([color_attachment, clear_value, clear_rect](vk::CommandBuffer cmdbuf) {
                    const VkClearAttachment attachment{
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .colorAttachment = color_attachment,
                        .clearValue = clear_value,
                    };
                    cmdbuf.ClearAttachments(attachment, clear_rect);
                });
            }
        } else {
            u8 color_mask = u8(regs.clear_surface.R | regs.clear_surface.G << 1 | regs.clear_surface.B << 2 | regs.clear_surface.A << 3);
            Region2D dst_region = {
                Offset2D{.x = clear_rect.rect.offset.x, .y = clear_rect.rect.offset.y},
                Offset2D{.x = clear_rect.rect.offset.x + s32(clear_rect.rect.extent.width),
                         .y = clear_rect.rect.offset.y + s32(clear_rect.rect.extent.height)}};
            blit_image.ClearColor(framebuffer, color_mask, regs.clear_color, dst_region);
        }
    }

    if (!use_depth && !use_stencil) {
        return;
    }
    VkImageAspectFlags aspect_flags = 0;
    if (use_depth && framebuffer->HasAspectDepthBit()) {
        aspect_flags |= VK_IMAGE_ASPECT_DEPTH_BIT;
    }
    if (use_stencil && framebuffer->HasAspectStencilBit()) {
        aspect_flags |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    if (aspect_flags == 0) {
        return;
    }

    if (use_stencil && framebuffer->HasAspectStencilBit() && regs.stencil_front_mask != 0xFF &&
        regs.stencil_front_mask != 0) {
        Region2D dst_region = {
            Offset2D{.x = clear_rect.rect.offset.x, .y = clear_rect.rect.offset.y},
            Offset2D{.x = clear_rect.rect.offset.x + s32(clear_rect.rect.extent.width),
                     .y = clear_rect.rect.offset.y + s32(clear_rect.rect.extent.height)}};
        blit_image.ClearDepthStencil(framebuffer, use_depth, regs.clear_depth,
                                     u8(regs.stencil_front_mask), regs.clear_stencil,
                                     regs.stencil_front_func_mask, dst_region);
    } else if (can_defer_clear) {
        VkClearValue ds_value{};
        ds_value.depthStencil.depth = regs.clear_depth;
        ds_value.depthStencil.stencil = regs.clear_stencil;
        scheduler.DeferDepthStencilClear(framebuffer, ds_value);
    } else {
        scheduler.Record([clear_depth = regs.clear_depth, clear_stencil = regs.clear_stencil,
                          clear_rect, aspect_flags](vk::CommandBuffer cmdbuf) {
            VkClearAttachment attachment;
            attachment.aspectMask = aspect_flags;
            attachment.colorAttachment = 0;
            attachment.clearValue.depthStencil.depth = clear_depth;
            attachment.clearValue.depthStencil.stencil = clear_stencil;
            cmdbuf.ClearAttachments(attachment, clear_rect);
        });
    }
}

void RasterizerVulkan::DispatchCompute() {
    if (!chunk_batches.empty()) {
        FlushChunkBatches();
    }
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Compute};
    if (frame_trace_state == 2) {
        const auto& qmd = kepler_compute->launch_description;
        LOG_INFO(HW_GPU, "frame trace: dispatch {}x{}x{} block {}x{}x{} program {:#x}",
                 qmd.grid_dim_x.Value(), qmd.grid_dim_y.Value(), qmd.grid_dim_z.Value(),
                 qmd.block_dim_x.Value(), qmd.block_dim_y.Value(), qmd.block_dim_z.Value(),
                 qmd.program_start);
    }
    FlushWork();
    gpu_memory->FlushCaching();

    ComputePipeline* const pipeline{pipeline_cache.CurrentComputePipeline()};
    if (!pipeline) {
        if (kepler_compute->IsNativeMode()) throw Shader::RuntimeError("Native compute pipeline unavailable");
        return;
    }
    std::scoped_lock lock{texture_cache.mutex, buffer_cache.mutex};
    if (!pipeline->Configure(*kepler_compute, *gpu_memory, scheduler, buffer_cache,
                             texture_cache)) {
        if (kepler_compute->IsNativeMode()) throw Shader::RuntimeError("Native compute resources unavailable");
        return;
    }

    const auto& qmd{kepler_compute->launch_description};
    auto indirect_address = kepler_compute->GetIndirectComputeAddress();
    if (indirect_address) {
        // DispatchIndirect
        static constexpr auto sync_info = VideoCommon::ObtainBufferSynchronize::FullSynchronize;
        const auto post_op = VideoCommon::ObtainBufferOperation::DiscardWrite;
        const auto [buffer, offset] =
            buffer_cache.ObtainBuffer(*indirect_address, 12, sync_info, post_op);
        scheduler.RequestOutsideRenderPassOperationContext();
        scheduler.Record([pipeline, indirect_buffer = buffer->Handle(),
                          indirect_offset = offset](vk::CommandBuffer cmdbuf) {
            if (!pipeline->IsBound()) {
                return;
            }
            cmdbuf.DispatchIndirect(indirect_buffer, indirect_offset);
        });
        return;
    }
    const std::array<u32, 3> dim{qmd.grid_dim_x, qmd.grid_dim_y, qmd.grid_dim_z};
    const std::array<u32, 3> max_dim{device.GetMaxComputeWorkGroupCount()};
    if (dim[0] > max_dim[0] || dim[1] > max_dim[1] || dim[2] > max_dim[2]) {
        return;
    }
    scheduler.RequestOutsideRenderPassOperationContext();
    static constexpr VkMemoryBarrier READ_BARRIER{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
    };
    scheduler.Record([](vk::CommandBuffer cmdbuf) { cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               0, READ_BARRIER); });
    scheduler.Record([pipeline, dim](vk::CommandBuffer cmdbuf) {
        if (!pipeline->IsBound()) {
            return;
        }
        cmdbuf.Dispatch(dim[0], dim[1], dim[2]);
    });

    // Log compute dispatch
    if (GPU::Logging::IsActive() &&
        Settings::values.gpu_log_vulkan_calls.GetValue()) {
        const std::string params = fmt::format("groupCountX={}, groupCountY={}, groupCountZ={}",
            dim[0], dim[1], dim[2]);
        GPU::Logging::GPULogger::GetInstance().LogVulkanCall(
            "vkCmdDispatch", params, VK_SUCCESS);
    }
}

void RasterizerVulkan::ResetCounter(VideoCommon::QueryType type) {
    switch (type) {
    case VideoCommon::QueryType::ZPassPixelCount64:
    case VideoCommon::QueryType::StreamingByteCount:
    case VideoCommon::QueryType::StreamingPrimitivesSucceeded:
    case VideoCommon::QueryType::VtgPrimitivesOut:
        query_cache.CounterReset(type);
        return;
    default:
        LOG_DEBUG(Render_Vulkan, "Unimplemented counter reset={}", type);
        return;
    }
}

void RasterizerVulkan::Query(GPUVAddr gpu_addr, VideoCommon::QueryType type,
                             VideoCommon::QueryPropertiesFlags flags, u32 payload, u32 subreport) {
    if (!chunk_batches.empty()) {
        FlushChunkBatches();
    }
    query_cache.CounterReport(gpu_addr, type, flags, payload, subreport);
}

void RasterizerVulkan::BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr,
                                                 u32 size) {
    buffer_cache.BindGraphicsUniformBuffer(stage, index, gpu_addr, size);
}

void Vulkan::RasterizerVulkan::DisableGraphicsUniformBuffer(size_t stage, u32 index) {
    buffer_cache.DisableGraphicsUniformBuffer(stage, index);
}

void RasterizerVulkan::FlushAll() {}

void RasterizerVulkan::FlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (!chunk_batches.empty()) {
        FlushChunkBatches();
    }
    if (addr == 0 || size == 0) {
        return;
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.DownloadMemory(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::BufferCache))) {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.DownloadMemory(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::QueryCache))) {
        query_cache.FlushRegion(addr, size);
    }
}

bool RasterizerVulkan::MustFlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if ((True(which & VideoCommon::CacheType::BufferCache))) {
        std::scoped_lock lock{buffer_cache.mutex};
        if (buffer_cache.IsRegionGpuModified(addr, size)) {
            return true;
        }
    }
    if (!Settings::IsGPULevelHigh()) {
        return false;
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        return texture_cache.IsRegionGpuModified(addr, size);
    }
    return false;
}

VideoCore::RasterizerDownloadArea RasterizerVulkan::GetFlushArea(DAddr addr, u64 size) {
    {
        std::scoped_lock lock{texture_cache.mutex};
        auto area = texture_cache.GetFlushArea(addr, size);
        if (area) {
            return *area;
        }
    }
    VideoCore::RasterizerDownloadArea new_area{
        .start_address = Common::AlignDown(addr, Core::DEVICE_PAGESIZE),
        .end_address = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE),
        .preemtive = true,
    };
    return new_area;
}

void RasterizerVulkan::InvalidateRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (!chunk_batches.empty()) {
        FlushChunkBatches();
    }
    if (addr == 0 || size == 0) {
        return;
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::BufferCache))) {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::QueryCache))) {
        query_cache.InvalidateRegion(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::ShaderCache))) {
        pipeline_cache.InvalidateRegion(addr, size);
    }
}

void RasterizerVulkan::InnerInvalidation(std::span<const std::pair<DAddr, std::size_t>> sequences) {
    {
        std::scoped_lock lock{texture_cache.mutex};
        for (const auto& [addr, size] : sequences) {
            texture_cache.WriteMemory(addr, size);
        }
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        for (const auto& [addr, size] : sequences) {
            buffer_cache.WriteMemory(addr, size);
        }
    }
    {
        for (const auto& [addr, size] : sequences) {
            query_cache.InvalidateRegion(addr, size);
            pipeline_cache.InvalidateRegion(addr, size);
        }
    }
}

bool RasterizerVulkan::OnCPUWrite(DAddr addr, u64 size) {
    DEBUG_ASSERT(addr != 0 || size != 0);
    {
        std::scoped_lock lock{buffer_cache.mutex};
        if (buffer_cache.OnCPUWrite(addr, size)) {
            return true;
        }
    }
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    pipeline_cache.InvalidateRegion(addr, size);
    return false;
}

void RasterizerVulkan::OnCacheInvalidation(DAddr addr, u64 size) {
    if (!chunk_batches.empty()) {
        FlushChunkBatches();
    }
    if (addr == 0 || size == 0) {
        return;
    }

    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size);
    }
    pipeline_cache.InvalidateRegion(addr, size);
}

bool RasterizerVulkan::TryCacheInvalidation(DAddr addr, u64 size) {
    if (addr == 0 || size == 0) return true;
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        if (buffer_cache.IsRegionGpuModified(addr, size)) return false;
        texture_cache.WriteMemory(addr, size);
        buffer_cache.WriteMemory(addr, size);
    }
    pipeline_cache.InvalidateRegion(addr, size);
    return true;
}

void RasterizerVulkan::InvalidateGPUCache() {
    if (!chunk_batches.empty()) {
        FlushChunkBatches();
    }
    gpu.InvalidateGPUCache();
}

void RasterizerVulkan::UnmapMemory(DAddr addr, u64 size) {
    chunk_window_cache.clear();
    if (!chunk_batches.empty()) {
        FlushChunkBatches();
    }
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.UnmapMemory(addr, size);
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size);
    }
    pipeline_cache.OnCacheInvalidation(addr, size);
}

void RasterizerVulkan::ModifyGPUMemory(size_t as_id, GPUVAddr addr, u64 size) {
    chunk_window_cache.clear();
    if (!chunk_batches.empty()) {
        FlushChunkBatches();
    }
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.UnmapGPUMemory(as_id, addr, size);
    }
    buffer_cache.UnmapGPUMemory(as_id, addr, size);
}

void RasterizerVulkan::SignalFence(std::function<void()>&& func) {
    if (!chunk_batches.empty()) {
        FlushChunkBatches();
    }
    fence_manager.SignalFence(std::move(func));
}

void RasterizerVulkan::SyncOperation(std::function<void()>&& func) {
    if (!chunk_batches.empty()) {
        FlushChunkBatches();
    }
    fence_manager.SyncOperation(std::move(func));
}

void RasterizerVulkan::SignalSyncPoint(u32 value) {
    if (!chunk_batches.empty()) {
        FlushChunkBatches();
    }
    fence_manager.SignalSyncPoint(value);
}

void RasterizerVulkan::SignalReference() {
    if (!chunk_batches.empty()) {
        FlushChunkBatches();
    }
    fence_manager.SignalReference();
}

void RasterizerVulkan::ReleaseFences(bool force) {
    if (!chunk_batches.empty()) {
        FlushChunkBatches();
    }
    fence_manager.WaitPendingFences(force);
}

void RasterizerVulkan::FlushAndInvalidateRegion(DAddr addr, u64 size,
                                                VideoCommon::CacheType which) {
    if (!chunk_batches.empty()) {
        FlushChunkBatches();
    }
    if (Settings::IsGPULevelHigh()) {
        FlushRegion(addr, size, which);
    }
    InvalidateRegion(addr, size, which);
}

void RasterizerVulkan::WaitForIdle() {
    if (!chunk_batches.empty()) {
        FlushChunkBatches();
    }
    // Everything but wait pixel operations. This intentionally includes FRAGMENT_SHADER_BIT because
    // fragment shaders can still write storage buffers.
    VkPipelineStageFlags flags =
        VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT |
        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT |
        VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT |
        VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    if (device.IsExtTransformFeedbackSupported()) {
        flags |= VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT;
    }

    query_cache.NotifyWFI();

    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([event = *wfi_event, flags](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetEvent(event, flags);
        cmdbuf.WaitEvents(event, flags, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, {}, {}, {});
    });
    fence_manager.SignalOrdering();
}

void RasterizerVulkan::FragmentBarrier() {
    if (!chunk_batches.empty()) {
        FlushChunkBatches();
    }
    // We already put barriers when a render pass finishes
    scheduler.RequestOutsideRenderPassOperationContext();
}

void RasterizerVulkan::TiledCacheBarrier() {
    if (!chunk_batches.empty()) {
        FlushChunkBatches();
    }
    // TODO: Implementing tiled barriers requires rewriting a good chunk of the Vulkan backend
}

void RasterizerVulkan::FlushCommands() {
    if (!chunk_batches.empty()) {
        FlushChunkBatches();
    }
    if (draw_counter == 0) {
        return;
    }
    draw_counter = 0;
    scheduler.Flush();
}

void RasterizerVulkan::TickFrame() {
    if (!chunk_batches.empty()) {
        FlushChunkBatches();
    }
    draw_counter = 0;
    ++frame_counter;
    // EDEN_NX2_TRACE_TRIGGER=<path>: trace the frame after that file appears (`touch` it from
    // outside, so the trace lands on the scene currently on screen).
    static const char* const trigger = std::getenv("EDEN_NX2_TRACE_TRIGGER");
    if (trigger != nullptr && frame_trace_state == 0 && frame_counter % 8 == 0 &&
        std::filesystem::exists(trigger)) {
        frame_trace_state = 1;
    }
    if (frame_trace_state == 2) {
        LOG_INFO(HW_GPU, "frame trace: end of frame {}", frame_counter);
        frame_trace_state = 3;
        Tegra::g_nx2_frame_trace_active.store(false, std::memory_order_relaxed);
    } else if (frame_trace_state == 1) {
        LOG_INFO(HW_GPU, "frame trace: start of frame {}", frame_counter);
        frame_trace_state = 2;
        Tegra::g_nx2_frame_trace_active.store(true, std::memory_order_relaxed);
    }
    if (!draw_stats.empty() && ++draw_stats_frames == 600) {
        std::vector<const DrawStats*> top;
        u64 total_draws = 0;
        u64 total_indirect = 0;
        for (const auto& [key, entry] : draw_stats) {
            total_draws += entry.draws;
            total_indirect += (entry.depth & 0x400) ? entry.draws : 0;
            if (entry.draws >= 300 || (entry.depth & 0x400)) {
                top.push_back(&entry);
            }
        }
        static auto last_dump = std::chrono::steady_clock::now();
        const auto now = std::chrono::steady_clock::now();
        LOG_INFO(HW_GPU, "Draw stats: {} draws ({} indirect) in {} program pairs over 600 frames, {:.1f} s",
                 total_draws, total_indirect, draw_stats.size(),
                 std::chrono::duration<double>(now - last_dump).count());
        last_dump = now;
        std::ranges::sort(top, std::greater{}, &DrawStats::vertices);
        for (const DrawStats* entry : top) {
            LOG_INFO(HW_GPU,
                     "Draw stats: vs={:#x} ps={:#x} depth {}{}func {:#x} zeta {:#x} rts {} rt0 {:#x}{}: "
                     "{} draws, {} vertices in 600 frames",
                     entry->vs, entry->ps, (entry->depth & 0x100) ? "test " : "",
                     (entry->depth & 0x200) ? "write " : "", entry->depth & 0xff, entry->zeta,
                     (entry->depth & 0x400) ? " indirect" : "",
                     entry->rt_count, entry->color0, entry->draws, entry->vertices);
        }
        draw_stats.clear();
        draw_stats_frames = 0;
    }
    guest_descriptor_queue.TickFrame();
    compute_pass_descriptor_queue.TickFrame();
    descriptor_buffer_ring.TickFrame();
    fence_manager.TickFrame();
    staging_pool.TickFrame();
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.TickFrame();
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.TickFrame();
    }
}

bool RasterizerVulkan::AccelerateConditionalRendering() {
    gpu_memory->FlushCaching();
    return query_cache.AccelerateHostConditionalRendering();
}

bool RasterizerVulkan::HasDrawTransformFeedback() {
    return device.IsTransformFeedbackDrawSupported();
}

bool RasterizerVulkan::AccelerateSurfaceCopy(const Tegra::Engines::Fermi2D::Surface& src,
                                             const Tegra::Engines::Fermi2D::Surface& dst,
                                             const Tegra::Engines::Fermi2D::Config& copy_config) {
    // Mario Kart World fills its 8192x8192x68 ASTC terrain arrays (sparse regions) with 2D-engine
    // copies of each layer's levels as 1024x1024 128-bit surfaces. On the host those would land
    // in an RGBA32 image the ASTC array can neither alias nor copy from, so the array would stay
    // black. The software blitter writes guest memory instead, which backs the sparse blocks and
    // re-uploads the array's layer.
    if (!copy_config.must_accelerate && Tegra::Nx2InSparseRegion(dst.Address(), 1)) {
        static u32 logged{};
        if (logged++ < 8) {
            LOG_INFO(Render_Vulkan,
                     "2D copy into sparse region {:#x} ({}x{} format {:#x}) done in guest memory",
                     dst.Address(), dst.width, dst.height, static_cast<u32>(dst.format));
        }
        return false;
    }
    std::scoped_lock lock{texture_cache.mutex};
    return texture_cache.BlitImage(dst, src, copy_config);
}

Tegra::Engines::AccelerateDMAInterface& RasterizerVulkan::AccessAccelerateDMA() {
    return accelerate_dma;
}

void RasterizerVulkan::AccelerateInlineToMemory(GPUVAddr address, size_t copy_size,
                                                std::span<const u8> memory) {
    if (!chunk_batches.empty()) {
        FlushChunkBatches();
    }
    auto cpu_addr = gpu_memory->GpuToCpuAddress(address);
    if (gpu_memory->UsesNativeTiling()) {
        u32 first_word{};
        if (memory.size() >= sizeof(first_word)) std::memcpy(&first_word, memory.data(), sizeof(first_word));
        VideoCommon::TraceNativeQuery(11, gpu_memory->GetID(), address, cpu_addr.value_or(0),
            0, first_word, static_cast<u32>(copy_size));
        static const GPUVAddr watch = [] {
            const char* value = std::getenv("EDEN_NX2_SEM_WATCH");
            return value ? GPUVAddr{std::strtoull(value, nullptr, 16)} : GPUVAddr{};
        }();
        if (watch != 0 && address < watch + 16 && address + copy_size > watch) {
            LOG_INFO(Render_Vulkan, "Semaphore watch: inline write {:#x} size {} first word {:#x}",
                     address, copy_size, first_word);
        }
    }
    if (!cpu_addr) [[unlikely]] {
        gpu_memory->WriteBlock(address, memory.data(), copy_size);
        return;
    }
    gpu_memory->WriteBlockUnsafe(address, memory.data(), copy_size);
    // EDEN_NX2_INLINE_DEFER=0: invalidate every cache at once (for comparisons).
    static const bool defer = [] {
        const char* value = std::getenv("EDEN_NX2_INLINE_DEFER");
        return value == nullptr || value[0] != '0';
    }();
    // One cache at a time: holding both while waiting for either (std::lock) waited longer, the
    // fencing thread and CPU write traps holding one of them most of the time.
    bool uploaded;
    {
        std::scoped_lock lock{buffer_cache.mutex};
        uploaded = buffer_cache.InlineMemory(*cpu_addr, copy_size, memory);
        if (!uploaded && !defer) {
            buffer_cache.WriteMemory(*cpu_addr, copy_size);
        }
    }
    // Pending query results must not land over the new words later: drop them now.
    query_cache.InvalidateRegion(*cpu_addr, copy_size);
    if (!uploaded && defer) {
        // Title 1 makes thousands of small inline writes between draws (~10% of the GPU
        // thread invalidating every cache for each). Guest memory is written; the buffer,
        // texture and shader caches see the merged ranges at the next FlushCaching, which runs
        // before anything of theirs is used. Writes into GPU-written buffer data were uploaded
        // in place above, as before.
        gpu_memory->AccumulateInvalidation(address, copy_size);
        return;
    }
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(*cpu_addr, copy_size);
    }
    pipeline_cache.InvalidateRegion(*cpu_addr, copy_size);
}

std::optional<FramebufferTextureInfo> RasterizerVulkan::AccelerateDisplay(
    const Tegra::FramebufferConfig& config, DAddr framebuffer_addr, u32 pixel_stride) {
    if (!framebuffer_addr) {
        return {};
    }
    std::scoped_lock lock{texture_cache.mutex};
    const auto [image_view, scaled] =
        texture_cache.TryFindFramebufferImageView(config, framebuffer_addr);
    static u32 display_trace_count{};
    static const bool frame_capture = std::getenv("EDEN_NATIVE_FRAME_CAPTURE") != nullptr;
    if (frame_capture && display_trace_count++ < 64) {
        LOG_INFO(HW_GPU, "Native display source: address={:#x}, offset={:#x}, dimensions={}x{}, stride={}, accelerated={}, mask={:#x}",
                 framebuffer_addr, config.offset, config.width, config.height, pixel_stride,
                 image_view != nullptr, config.layer_stack_mask);
    }
    if (!image_view) {
        return {};
    }

    query_cache.NotifySegment(false);

    const auto& resolution = Settings::values.resolution_info;

    FramebufferTextureInfo info{};
    info.image = image_view->ImageHandle();
    info.image_view = image_view->Handle(Shader::TextureType::Color2D);
    info.width = image_view->size.width;
    info.height = image_view->size.height;
    info.scaled_width = scaled ? resolution.ScaleUp(info.width) : info.width;
    info.scaled_height = scaled ? resolution.ScaleUp(info.height) : info.height;
    return info;
}

void RasterizerVulkan::LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                                         const VideoCore::DiskResourceLoadCallback& callback) {
    pipeline_cache.LoadDiskResources(title_id, stop_loading, callback);
}

void RasterizerVulkan::FlushWork() {
#ifdef __ANDROID__
    static constexpr u32 DRAWS_TO_DISPATCH = 512;
    static constexpr u32 CHECK_MASK = 3;
#else
    // Every 32 draws (was 8): each dispatch woke the worker thread, and Switch 2 titles make
    // ~55,000 draws a second. A full chunk is dispatched on its own (Scheduler::Record).
    static constexpr u32 DRAWS_TO_DISPATCH = 4096;
    static constexpr u32 CHECK_MASK = 31;
#endif // __ANDROID__

    static_assert(DRAWS_TO_DISPATCH % (CHECK_MASK + 1) == 0);
    if ((++draw_counter & CHECK_MASK) != CHECK_MASK) {
        return;
    }
    if (draw_counter < DRAWS_TO_DISPATCH) {
        scheduler.DispatchWork();
        return;
    }
    scheduler.Flush();
    draw_counter = 0;
}

AccelerateDMA::AccelerateDMA(BufferCache& buffer_cache_, TextureCache& texture_cache_,
                             Scheduler& scheduler_)
    : buffer_cache{buffer_cache_}, texture_cache{texture_cache_}, scheduler{scheduler_} {}

bool AccelerateDMA::BufferClear(GPUVAddr src_address, u64 amount, u32 value) {
    std::scoped_lock lock{buffer_cache.mutex};
    return buffer_cache.DMAClear(src_address, amount, value);
}

bool AccelerateDMA::BufferCopy(GPUVAddr src_address, GPUVAddr dest_address, u64 amount) {
    std::scoped_lock lock{buffer_cache.mutex};
    return buffer_cache.DMACopy(src_address, dest_address, amount);
}

template <bool IS_IMAGE_UPLOAD>
bool AccelerateDMA::DmaBufferImageCopy(const Tegra::DMA::ImageCopy& copy_info,
                                       const Tegra::DMA::BufferOperand& buffer_operand,
                                       const Tegra::DMA::ImageOperand& image_operand) {
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    const auto image_id = texture_cache.DmaImageId(image_operand, IS_IMAGE_UPLOAD);
    if (image_id == VideoCommon::NULL_IMAGE_ID) {
        return false;
    }
    const u32 buffer_size = static_cast<u32>(buffer_operand.pitch * buffer_operand.height);
    static constexpr auto sync_info = VideoCommon::ObtainBufferSynchronize::FullSynchronize;
    const auto post_op = IS_IMAGE_UPLOAD ? VideoCommon::ObtainBufferOperation::DoNothing
                                         : VideoCommon::ObtainBufferOperation::MarkAsWritten;
    const auto [buffer, offset] =
        buffer_cache.ObtainBuffer(buffer_operand.address, buffer_size, sync_info, post_op);

    const auto [image, copy] = texture_cache.DmaBufferImageCopy(
        copy_info, buffer_operand, image_operand, image_id, IS_IMAGE_UPLOAD);
    const std::span copy_span{&copy, 1};

    if constexpr (IS_IMAGE_UPLOAD) {
        texture_cache.PrepareImage(image_id, true, false);
        image->UploadMemory(buffer->Handle(), offset, copy_span);
    } else {
        if (offset % BytesPerBlock(image->info.format)) {
            return false;
        }
        texture_cache.DownloadImageIntoBuffer(image, buffer->Handle(), offset, copy_span,
                                              buffer_operand.address, buffer_size);
    }
    return true;
}

bool AccelerateDMA::ImageToBuffer(const Tegra::DMA::ImageCopy& copy_info,
                                  const Tegra::DMA::ImageOperand& image_operand,
                                  const Tegra::DMA::BufferOperand& buffer_operand) {
    return DmaBufferImageCopy<false>(copy_info, buffer_operand, image_operand);
}

bool AccelerateDMA::BufferToImage(const Tegra::DMA::ImageCopy& copy_info,
                                  const Tegra::DMA::BufferOperand& buffer_operand,
                                  const Tegra::DMA::ImageOperand& image_operand) {
    return DmaBufferImageCopy<true>(copy_info, buffer_operand, image_operand);
}

void RasterizerVulkan::UpdateDynamicStates() {
    auto& regs = maxwell3d->regs;
    auto& flags = maxwell3d->dirty.flags;
    const auto topology = maxwell3d->draw_manager.draw_state.topology;
    const bool topology_changed = state_tracker.ChangePrimitiveTopology(topology);
    if (topology_changed) {
        flags[Dirty::DepthBiasEnable] = true;
        flags[Dirty::PrimitiveRestartEnable] = true;
    }

    // EDEN_NX2_BIAS_VS=<hex low 20 bits> with EDEN_NX2_BIAS_UNITS=<n>: push the draws of that
    // vertex program back by n depth units (diagnostic: how much nearer is its geometry?).
    static const u64 bias_vs = [] {
        const char* value = std::getenv("EDEN_NX2_BIAS_VS");
        return value != nullptr ? std::strtoull(value, nullptr, 16) : ~0ULL;
    }();
    if (bias_vs != ~0ULL) {
        const bool match = (regs.pipelines[1].reserved[1] & 0xfffff) == bias_vs;
        if (match != bias_vs_active) {
            bias_vs_active = match;
            flags[Dirty::DepthBias] = true;
            flags[Dirty::DepthBiasEnable] = true;
        }
    }

    UpdateViewportsState(regs);
    UpdateScissorsState(regs);
    UpdateDepthBias(regs);
    UpdateBlendConstants(regs);
    UpdateDepthBounds(regs);
    UpdateStencilFaces(regs);
    UpdateLineWidth(regs);

    if (device.IsExtExtendedDynamicStateSupported()) {
        UpdateCullMode(regs);
        UpdateDepthCompareOp(regs);
        UpdateFrontFace(regs);
        UpdateStencilOp(regs);
        if (state_tracker.TouchStateEnable()) {
            UpdateDepthBoundsTestEnable(regs);
            UpdateDepthTestEnable(regs);
            UpdateDepthWriteEnable(regs);
            UpdateStencilTestEnable(regs);
        }
        if (topology_changed) {
            scheduler.Record([topology_vk = MaxwellToVK::PrimitiveTopology(device, topology)](
                                 vk::CommandBuffer cmdbuf) {
                cmdbuf.SetPrimitiveTopologyEXT(topology_vk);
            });
        }
    }

    if (device.IsExtExtendedDynamicState2Supported()) {
        UpdatePrimitiveRestartEnable(regs);
        UpdateRasterizerDiscardEnable(regs);
        UpdateDepthBiasEnable(regs);
    }

    if (device.IsExtExtendedDynamicState2ExtrasSupported()) {
        UpdateLogicOp(regs);
    }

    if (device.IsExtExtendedDynamicState3EnablesSupported()) {
        using namespace Tegra::Engines;
        // AMD Workaround: LogicOp incompatible with float render targets
        if (device.GetDriverID() == VkDriverIdKHR::VK_DRIVER_ID_AMD_OPEN_SOURCE ||
            device.GetDriverID() == VkDriverIdKHR::VK_DRIVER_ID_AMD_PROPRIETARY) {
            const auto has_float = std::any_of(
                regs.vertex_attrib_format.begin(), regs.vertex_attrib_format.end(),
                [](const auto& attrib) {
                    return attrib.type == Maxwell3D::Regs::VertexAttribute::Type::Float;
                }
            );
            if (regs.logic_op.enable) {
                regs.logic_op.enable = static_cast<u32>(!has_float);
            }
        }
        UpdateLogicOpEnable(regs);
        UpdateDepthClampEnable(regs);
        UpdateLineRasterizationMode(regs);
        UpdateLineStippleEnable(regs);
        UpdateConservativeRasterizationMode(regs);
        UpdateAlphaToCoverageEnable(regs);
        UpdateAlphaToOneEnable(regs);
    }

    if (device.IsExtExtendedDynamicState3BlendingSupported()) {
        UpdateBlending(regs);
    } else if (device.IsExtColorWriteEnableSupported()) {
        UpdateColorWriteEnable(regs);
    }

    if (device.IsExtVertexInputDynamicStateSupported()) {
        if (auto* gp = pipeline_cache.CurrentGraphicsPipeline(); gp && gp->HasDynamicVertexInput()) {
            UpdateVertexInput(regs);
        }
    }
}

void RasterizerVulkan::HandleTransformFeedback() {
    static std::once_flag warn_unsupported;

    const auto& regs = maxwell3d->regs;
    if (!device.IsExtTransformFeedbackSupported()) {
        if (regs.transform_feedback_enabled != 0) {
            std::call_once(warn_unsupported, [&] {
                LOG_WARNING(Render_Vulkan, "Transform feedback requested by guest but VK_EXT_transform_feedback is unavailable; queries disabled");
            });
        } else {
            std::call_once(warn_unsupported, [&] {
                LOG_INFO(Render_Vulkan, "VK_EXT_transform_feedback not available on device");
            });
        }
        return;
    }
    query_cache.CounterEnable(VideoCommon::QueryType::StreamingByteCount,
                              regs.transform_feedback_enabled);
    if (regs.transform_feedback_enabled != 0) {
        // Log extension usage for transform feedback
        if (GPU::Logging::IsActive()) {
            GPU::Logging::GPULogger::GetInstance().LogExtensionUsage(
                "VK_EXT_transform_feedback", "HandleTransformFeedback");
        }
        UNIMPLEMENTED_IF(regs.IsShaderConfigEnabled(Maxwell::ShaderType::TessellationInit) ||
                         regs.IsShaderConfigEnabled(Maxwell::ShaderType::Tessellation));
    }
}

void RasterizerVulkan::UpdateViewportsState(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchViewports()) {
        return;
    }

    maxwell3d->dirty.flags[Dirty::Scissors] = true;

    if (!regs.viewport_scale_offset_enabled) {
        float x = static_cast<float>(regs.surface_clip.x);
        float y = static_cast<float>(regs.surface_clip.y);
        float width = (std::max)(1.0f, static_cast<float>(regs.surface_clip.width));
        float height = (std::max)(1.0f, static_cast<float>(regs.surface_clip.height));
        if (regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft) {
            y += height;
            height = -height;
        }
        VkViewport viewport{
            .x = x,
            .y = y,
            .width = width,
            .height = height,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        scheduler.Record([this, viewport](vk::CommandBuffer cmdbuf) {
            const u32 num_viewports = std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
            std::array<VkViewport, Maxwell::NumViewports> viewport_list{};
            viewport_list.fill(viewport);
            const vk::Span<VkViewport> viewports(viewport_list.data(), num_viewports);
            cmdbuf.SetViewport(0, viewports);
        });
        return;
    }
    const bool is_rescaling{texture_cache.IsRescaling()};
    const float scale = is_rescaling ? Settings::values.resolution_info.up_factor : 1.0f;
    const std::array viewport_list{
        GetViewportState(device, regs, 0, scale),  GetViewportState(device, regs, 1, scale),
        GetViewportState(device, regs, 2, scale),  GetViewportState(device, regs, 3, scale),
        GetViewportState(device, regs, 4, scale),  GetViewportState(device, regs, 5, scale),
        GetViewportState(device, regs, 6, scale),  GetViewportState(device, regs, 7, scale),
        GetViewportState(device, regs, 8, scale),  GetViewportState(device, regs, 9, scale),
        GetViewportState(device, regs, 10, scale), GetViewportState(device, regs, 11, scale),
        GetViewportState(device, regs, 12, scale), GetViewportState(device, regs, 13, scale),
        GetViewportState(device, regs, 14, scale), GetViewportState(device, regs, 15, scale),
    };
    scheduler.Record([this, viewport_list](vk::CommandBuffer cmdbuf) {
        const u32 num_viewports = std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
        const vk::Span<VkViewport> viewports(viewport_list.data(), num_viewports);
        cmdbuf.SetViewport(0, viewports);
    });
}

void RasterizerVulkan::UpdateScissorsState(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchScissors()) {
        return;
    }
    if (!regs.viewport_scale_offset_enabled) {
        u32 x = regs.surface_clip.x;
        u32 y = regs.surface_clip.y;
        u32 width = (std::max)(1u, static_cast<u32>(regs.surface_clip.width));
        u32 height = (std::max)(1u, static_cast<u32>(regs.surface_clip.height));
        if (regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft) {
            y = regs.surface_clip.height - (y + height);
        }
        VkRect2D scissor{};
        scissor.offset.x = static_cast<int32_t>(x);
        scissor.offset.y = static_cast<int32_t>(y);
        scissor.extent.width  = width;
        scissor.extent.height = height;
        scheduler.Record([this, scissor](vk::CommandBuffer cmdbuf) {
            const u32 num_scissors = std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
            std::array<VkRect2D, Maxwell::NumViewports> scissor_list{};
            scissor_list.fill(scissor);
            const vk::Span<VkRect2D> scissors(scissor_list.data(), num_scissors);
            cmdbuf.SetScissor(0, scissors);
        });
        return;
    }
    u32 up_scale = 1;
    u32 down_shift = 0;
    if (texture_cache.IsRescaling()) {
        up_scale = Settings::values.resolution_info.up_scale;
        down_shift = Settings::values.resolution_info.down_shift;
    }
    const std::array scissor_list{
        GetScissorState(regs, 0, up_scale, down_shift),
        GetScissorState(regs, 1, up_scale, down_shift),
        GetScissorState(regs, 2, up_scale, down_shift),
        GetScissorState(regs, 3, up_scale, down_shift),
        GetScissorState(regs, 4, up_scale, down_shift),
        GetScissorState(regs, 5, up_scale, down_shift),
        GetScissorState(regs, 6, up_scale, down_shift),
        GetScissorState(regs, 7, up_scale, down_shift),
        GetScissorState(regs, 8, up_scale, down_shift),
        GetScissorState(regs, 9, up_scale, down_shift),
        GetScissorState(regs, 10, up_scale, down_shift),
        GetScissorState(regs, 11, up_scale, down_shift),
        GetScissorState(regs, 12, up_scale, down_shift),
        GetScissorState(regs, 13, up_scale, down_shift),
        GetScissorState(regs, 14, up_scale, down_shift),
        GetScissorState(regs, 15, up_scale, down_shift),
    };
    scheduler.Record([this, scissor_list](vk::CommandBuffer cmdbuf) {
        const u32 num_scissors = std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
        const vk::Span<VkRect2D> scissors(scissor_list.data(), num_scissors);
        cmdbuf.SetScissor(0, scissors);
    });
}

void RasterizerVulkan::UpdateDepthBias(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthBias()) {
        return;
    }
    float units = regs.depth_bias / 2.0f;
    const bool is_d24 = regs.zeta.format == Tegra::DepthFormat::Z24_UNORM_S8_UINT ||
                        regs.zeta.format == Tegra::DepthFormat::X8Z24_UNORM ||
                        regs.zeta.format == Tegra::DepthFormat::S8Z24_UNORM ||
                        regs.zeta.format == Tegra::DepthFormat::V8Z24_UNORM;

    if (is_d24 && !device.SupportsD24DepthBuffer()) {
        static constexpr const size_t length = sizeof(NEEDS_D24) / sizeof(NEEDS_D24[0]);

        static constexpr const u64* start = NEEDS_D24;
        static constexpr const u64* end = NEEDS_D24 + length;

        const u64* it = std::find(start, end, program_id);

        if (it != end) {
            // the base formulas can be obtained from here:
            //   https://docs.microsoft.com/en-us/windows/win32/direct3d11/d3d10-graphics-programming-guide-output-merger-stage-depth-bias
            const double rescale_factor =
                static_cast<double>(1ULL << (32 - 24)) / (static_cast<double>(0x1.ep+127));
            units = static_cast<float>(static_cast<double>(units) * rescale_factor);
        }
    }

    float factor_override = regs.slope_scale_depth_bias;
    if (bias_vs_active) {
        static const float bias_units = [] {
            const char* value = std::getenv("EDEN_NX2_BIAS_UNITS");
            return value != nullptr ? std::strtof(value, nullptr) : 4.0f;
        }();
        units = bias_units;
        factor_override = 0.0f;
        static bool logged = false;
        if (!logged) {
            logged = true;
            LOG_INFO(HW_GPU, "depth bias override active: {} units on vs {:#x}", units,
                     regs.pipelines[1].reserved[1]);
        }
    }

    scheduler.Record([constant = units, clamp = regs.depth_bias_clamp,
                      factor = factor_override, this](vk::CommandBuffer cmdbuf) {
        if (device.IsExtDepthBiasControlSupported()) {
            static VkDepthBiasRepresentationInfoEXT bias_info{
                .sType = VK_STRUCTURE_TYPE_DEPTH_BIAS_REPRESENTATION_INFO_EXT,
                .pNext = nullptr,
                .depthBiasRepresentation =
                    VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT,
                .depthBiasExact = VK_FALSE,
            };

            cmdbuf.SetDepthBias(constant, clamp, factor, &bias_info);
        } else {
            cmdbuf.SetDepthBias(constant, clamp, factor);
        }
    });
}

void RasterizerVulkan::UpdateBlendConstants(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchBlendConstants()) {
        return;
    }
    const std::array blend_color = {regs.blend_color.r, regs.blend_color.g, regs.blend_color.b,
                                    regs.blend_color.a};
    scheduler.Record(
        [blend_color](vk::CommandBuffer cmdbuf) { cmdbuf.SetBlendConstants(blend_color.data()); });
}

void RasterizerVulkan::UpdateDepthBounds(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthBounds()) {
        return;
    }
    scheduler.Record([min = regs.depth_bounds[0], max = regs.depth_bounds[1]](
                         vk::CommandBuffer cmdbuf) { cmdbuf.SetDepthBounds(min, max); });
}

void RasterizerVulkan::UpdateStencilFaces(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchStencilProperties()) {
        return;
    }
    bool update_references = state_tracker.TouchStencilReference();
    bool update_write_mask = state_tracker.TouchStencilWriteMask();
    bool update_compare_masks = state_tracker.TouchStencilCompare();
    if (state_tracker.TouchStencilSide(regs.stencil_two_side_enable != 0)) {
        update_references = true;
        update_write_mask = true;
        update_compare_masks = true;
    }
    if (update_references) {
        [&]() {
            bool changed;
            if (regs.stencil_two_side_enable) {
                const bool front_changed =
                    state_tracker.CheckStencilReferenceFront(regs.stencil_front_ref);
                const bool back_changed =
                    state_tracker.CheckStencilReferenceBack(regs.stencil_back_ref);
                changed = front_changed || back_changed;
            } else {
                const bool front_changed =
                    state_tracker.CheckStencilReferenceFront(regs.stencil_front_ref);
                const bool back_changed =
                    state_tracker.CheckStencilReferenceBack(regs.stencil_front_ref);
                changed = front_changed || back_changed;
            }
            if (!changed) {
                return;
            }
            scheduler.Record([front_ref = regs.stencil_front_ref, back_ref = regs.stencil_back_ref,
                              two_sided = regs.stencil_two_side_enable](vk::CommandBuffer cmdbuf) {
                const bool set_back = two_sided && front_ref != back_ref;
                // Front face
                cmdbuf.SetStencilReference(set_back ? VK_STENCIL_FACE_FRONT_BIT
                                                    : VK_STENCIL_FACE_FRONT_AND_BACK,
                                           front_ref);
                if (set_back) {
                    cmdbuf.SetStencilReference(VK_STENCIL_FACE_BACK_BIT, back_ref);
                }
            });
        }();
    }
    if (update_write_mask) {
        [&]() {
            bool changed;
            if (regs.stencil_two_side_enable) {
                const bool front_changed =
                    state_tracker.CheckStencilWriteMaskFront(regs.stencil_front_mask);
                const bool back_changed =
                    state_tracker.CheckStencilWriteMaskBack(regs.stencil_back_mask);
                changed = front_changed || back_changed;
            } else {
                const bool front_changed =
                    state_tracker.CheckStencilWriteMaskFront(regs.stencil_front_mask);
                const bool back_changed =
                    state_tracker.CheckStencilWriteMaskBack(regs.stencil_front_mask);
                changed = front_changed || back_changed;
            }
            if (!changed) {
                return;
            }
            scheduler.Record([front_write_mask = regs.stencil_front_mask,
                              back_write_mask = regs.stencil_back_mask,
                              two_sided = regs.stencil_two_side_enable](vk::CommandBuffer cmdbuf) {
                const bool set_back = two_sided && front_write_mask != back_write_mask;
                // Front face
                cmdbuf.SetStencilWriteMask(set_back ? VK_STENCIL_FACE_FRONT_BIT
                                                    : VK_STENCIL_FACE_FRONT_AND_BACK,
                                           front_write_mask);
                if (set_back) {
                    cmdbuf.SetStencilWriteMask(VK_STENCIL_FACE_BACK_BIT, back_write_mask);
                }
            });
        }();
    }
    if (update_compare_masks) {
        [&]() {
            bool changed;
            if (regs.stencil_two_side_enable) {
                const bool front_changed =
                    state_tracker.CheckStencilCompareMaskFront(regs.stencil_front_func_mask);
                const bool back_changed =
                    state_tracker.CheckStencilCompareMaskBack(regs.stencil_back_func_mask);
                changed = front_changed || back_changed;
            } else {
                const bool front_changed =
                    state_tracker.CheckStencilCompareMaskFront(regs.stencil_front_func_mask);
                const bool back_changed =
                    state_tracker.CheckStencilCompareMaskBack(regs.stencil_front_func_mask);
                changed = front_changed || back_changed;
            }
            if (!changed) {
                return;
            }
            scheduler.Record([front_test_mask = regs.stencil_front_func_mask,
                              back_test_mask = regs.stencil_back_func_mask,
                              two_sided = regs.stencil_two_side_enable](vk::CommandBuffer cmdbuf) {
                const bool set_back = two_sided && front_test_mask != back_test_mask;
                // Front face
                cmdbuf.SetStencilCompareMask(set_back ? VK_STENCIL_FACE_FRONT_BIT
                                                      : VK_STENCIL_FACE_FRONT_AND_BACK,
                                             front_test_mask);
                if (set_back) {
                    cmdbuf.SetStencilCompareMask(VK_STENCIL_FACE_BACK_BIT, back_test_mask);
                }
            });
        }();
    }
    state_tracker.ClearStencilReset();
}

void RasterizerVulkan::UpdateLineWidth(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchLineWidth()) {
        return;
    }
    const float width =
        regs.line_anti_alias_enable ? regs.line_width_smooth : regs.line_width_aliased;
    scheduler.Record([width](vk::CommandBuffer cmdbuf) { cmdbuf.SetLineWidth(width); });
}

void RasterizerVulkan::UpdateCullMode(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchCullMode()) {
        return;
    }
    scheduler.Record([enabled = regs.gl_cull_test_enabled,
                      cull_face = regs.gl_cull_face](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetCullModeEXT(enabled ? MaxwellToVK::CullFace(cull_face) : VK_CULL_MODE_NONE);
    });
}

void RasterizerVulkan::UpdateDepthBoundsTestEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthBoundsTestEnable()) {
        return;
    }
    bool enabled = regs.depth_bounds_enable;
    if (enabled && !device.IsDepthBoundsSupported()) {
        LOG_WARNING(Render_Vulkan, "Depth bounds is enabled but not supported");
        enabled = false;
    }
    scheduler.Record([enable = enabled](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetDepthBoundsTestEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdateDepthTestEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthTestEnable()) {
        return;
    }
    scheduler.Record([enable = regs.depth_test_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetDepthTestEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdateDepthWriteEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthWriteEnable()) {
        return;
    }
    scheduler.Record([enable = regs.depth_write_enabled](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetDepthWriteEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdatePrimitiveRestartEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchPrimitiveRestartEnable()) {
        return;
    }

    bool enable = regs.primitive_restart.enabled != 0;
    if (device.IsMoltenVK()) {
        enable = true;
    } else if (enable) {
        const auto topology = MaxwellToVK::PrimitiveTopology(device, maxwell3d->draw_manager.draw_state.topology);
        enable = IsPrimitiveRestartSupported(device, topology);
    }

    scheduler.Record([enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetPrimitiveRestartEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdateRasterizerDiscardEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchRasterizerDiscardEnable()) {
        return;
    }
    scheduler.Record([disable = regs.rasterize_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetRasterizerDiscardEnableEXT(disable == 0);
    });
}

void RasterizerVulkan::UpdateConservativeRasterizationMode(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchConservativeRasterizationMode()) {
        return;
    }

    if (!device.SupportsDynamicState3ConservativeRasterizationMode()) {
        return;
    }

    scheduler.Record([enable = regs.conservative_raster_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetConservativeRasterizationModeEXT(
            enable ? VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT
                   : VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT);
    });
}

void RasterizerVulkan::UpdateLineStippleEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchLineStippleEnable()) {
        return;
    }

    if (!device.SupportsDynamicState3LineStippleEnable()) {
        return;
    }

    scheduler.Record([enable = regs.line_stipple_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetLineStippleEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdateLineRasterizationMode(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!device.IsExtLineRasterizationSupported()) {
        return;
    }
    if (!state_tracker.TouchLineRasterizationMode()) {
        return;
    }

    if (!device.SupportsDynamicState3LineRasterizationMode()) {
        static std::once_flag warn_missing_rect;
        std::call_once(warn_missing_rect, [] {
            LOG_WARNING(Render_Vulkan,
                        "Driver lacks rectangular line rasterization support; skipping dynamic "
                        "line state updates");
        });
        return;
    }

    const bool wants_smooth = regs.line_anti_alias_enable != 0;
    VkLineRasterizationModeEXT mode = VK_LINE_RASTERIZATION_MODE_RECTANGULAR_EXT;
    if (wants_smooth) {
        if (device.SupportsSmoothLines()) {
            mode = VK_LINE_RASTERIZATION_MODE_RECTANGULAR_SMOOTH_EXT;
        } else {
            static std::once_flag warn_missing_smooth;
            std::call_once(warn_missing_smooth, [] {
                LOG_WARNING(Render_Vulkan,
                            "Line anti-aliasing requested but smoothLines feature unavailable; "
                            "using rectangular rasterization");
            });
        }
    }
    scheduler.Record([mode](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetLineRasterizationModeEXT(mode);
    });
}

void RasterizerVulkan::UpdateDepthBiasEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthBiasEnable()) {
        return;
    }
    constexpr size_t POINT = 0;
    constexpr size_t LINE = 1;
    constexpr size_t POLYGON = 2;
    static constexpr std::array POLYGON_OFFSET_ENABLE_LUT = {
        POINT,   // Points
        LINE,    // Lines
        LINE,    // LineLoop
        LINE,    // LineStrip
        POLYGON, // Triangles
        POLYGON, // TriangleStrip
        POLYGON, // TriangleFan
        POLYGON, // Quads
        POLYGON, // QuadStrip
        POLYGON, // Polygon
        LINE,    // LinesAdjacency
        LINE,    // LineStripAdjacency
        POLYGON, // TrianglesAdjacency
        POLYGON, // TriangleStripAdjacency
        POLYGON, // Patches
    };
    const std::array enabled_lut{
        regs.polygon_offset_point_enable,
        regs.polygon_offset_line_enable,
        regs.polygon_offset_fill_enable,
    };
    const u32 topology_index = u32(maxwell3d->draw_manager.draw_state.topology);
    // EDEN_NX2_NO_DEPTH_BIAS=1: never enable polygon offset (diagnostic).
    static const bool no_depth_bias = std::getenv("EDEN_NX2_NO_DEPTH_BIAS") != nullptr;
    const u32 enable = bias_vs_active ? 1
                       : no_depth_bias ? 0
                                       : enabled_lut[POLYGON_OFFSET_ENABLE_LUT[topology_index]];
    last_depth_bias_enable = enable != 0;
    scheduler.Record([enable](vk::CommandBuffer cmdbuf) { cmdbuf.SetDepthBiasEnableEXT(enable != 0); });
}

void RasterizerVulkan::UpdateLogicOpEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchLogicOpEnable()) {
        return;
    }
    if (!device.SupportsDynamicState3LogicOpEnable()) {
        return;
    }
    scheduler.Record([enable = regs.logic_op.enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetLogicOpEnableEXT(enable != 0);
    });
}

void RasterizerVulkan::UpdateDepthClampEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthClampEnable()) {
        return;
    }
    if (!device.SupportsDynamicState3DepthClampEnable()) {
        return;
    }
    bool is_enabled = !(regs.viewport_clip_control.geometry_clip ==
                            Maxwell::ViewportClipControl::GeometryClip::Passthrough ||
                        regs.viewport_clip_control.geometry_clip ==
                            Maxwell::ViewportClipControl::GeometryClip::FrustumXYZ ||
                        regs.viewport_clip_control.geometry_clip ==
                            Maxwell::ViewportClipControl::GeometryClip::FrustumZ);
    scheduler.Record(
        [is_enabled](vk::CommandBuffer cmdbuf) { cmdbuf.SetDepthClampEnableEXT(is_enabled); });
}

void RasterizerVulkan::UpdateAlphaToCoverageEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchAlphaToCoverageEnable()) {
        return;
    }
    if (!device.SupportsDynamicState3AlphaToCoverageEnable()) {
        return;
    }
    GraphicsPipeline* const pipeline = pipeline_cache.CurrentGraphicsPipeline();
    const bool enable = pipeline != nullptr && pipeline->SupportsAlphaToCoverage() &&
                        regs.anti_alias_alpha_control.alpha_to_coverage != 0;
    scheduler.Record([enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetAlphaToCoverageEnableEXT(enable ? VK_TRUE : VK_FALSE);
    });
}

void RasterizerVulkan::UpdateAlphaToOneEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchAlphaToOneEnable()) {
        return;
    }
    if (!device.SupportsDynamicState3AlphaToOneEnable()) {
        static std::once_flag warn_alpha_to_one;
        std::call_once(warn_alpha_to_one, [] {
            LOG_WARNING(Render_Vulkan,
                        "Alpha-to-one is not supported on this device; forcing it disabled");
        });
        return;
    }
    GraphicsPipeline* const pipeline = pipeline_cache.CurrentGraphicsPipeline();
    const bool enable = pipeline != nullptr && pipeline->SupportsAlphaToOne() &&
                        regs.anti_alias_alpha_control.alpha_to_one != 0;
    scheduler.Record([enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetAlphaToOneEnableEXT(enable ? VK_TRUE : VK_FALSE);
    });
}

void RasterizerVulkan::UpdateDepthCompareOp(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthCompareOp()) {
        return;
    }
    scheduler.Record([func = regs.depth_test_func](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetDepthCompareOpEXT(MaxwellToVK::DepthComparisonOp(func));
    });
}

void RasterizerVulkan::UpdateFrontFace(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchFrontFace()) {
        return;
    }

    VkFrontFace front_face = MaxwellToVK::FrontFace(regs.gl_front_face);
    if (regs.window_origin.flip_y != 0) {
        front_face = front_face == VK_FRONT_FACE_CLOCKWISE ? VK_FRONT_FACE_COUNTER_CLOCKWISE
                                                           : VK_FRONT_FACE_CLOCKWISE;
    }
    scheduler.Record(
        [front_face](vk::CommandBuffer cmdbuf) { cmdbuf.SetFrontFaceEXT(front_face); });
}

void RasterizerVulkan::UpdateStencilOp(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchStencilOp()) {
        return;
    }
    const Maxwell::StencilOp::Op fail = regs.stencil_front_op.fail;
    const Maxwell::StencilOp::Op zfail = regs.stencil_front_op.zfail;
    const Maxwell::StencilOp::Op zpass = regs.stencil_front_op.zpass;
    const Maxwell::ComparisonOp compare = regs.stencil_front_op.func;
    if (regs.stencil_two_side_enable) {
        // Separate stencil op per face
        const Maxwell::StencilOp::Op back_fail = regs.stencil_back_op.fail;
        const Maxwell::StencilOp::Op back_zfail = regs.stencil_back_op.zfail;
        const Maxwell::StencilOp::Op back_zpass = regs.stencil_back_op.zpass;
        const Maxwell::ComparisonOp back_compare = regs.stencil_back_op.func;
        scheduler.Record([fail, zfail, zpass, compare, back_fail, back_zfail, back_zpass,
                          back_compare](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetStencilOpEXT(VK_STENCIL_FACE_FRONT_BIT, MaxwellToVK::StencilOp(fail),
                                   MaxwellToVK::StencilOp(zpass), MaxwellToVK::StencilOp(zfail),
                                   MaxwellToVK::ComparisonOp(compare));
            cmdbuf.SetStencilOpEXT(VK_STENCIL_FACE_BACK_BIT, MaxwellToVK::StencilOp(back_fail),
                                   MaxwellToVK::StencilOp(back_zpass),
                                   MaxwellToVK::StencilOp(back_zfail),
                                   MaxwellToVK::ComparisonOp(back_compare));
        });
    } else {
        // Front face defines the stencil op of both faces
        scheduler.Record([fail, zfail, zpass, compare](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetStencilOpEXT(VK_STENCIL_FACE_FRONT_AND_BACK, MaxwellToVK::StencilOp(fail),
                                   MaxwellToVK::StencilOp(zpass), MaxwellToVK::StencilOp(zfail),
                                   MaxwellToVK::ComparisonOp(compare));
        });
    }
}

void RasterizerVulkan::UpdateLogicOp(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchLogicOp()) {
        return;
    }
    const auto op_value = static_cast<u32>(regs.logic_op.op);
    auto op = op_value >= 0x1500 && op_value < 0x1510 ? static_cast<VkLogicOp>(op_value - 0x1500)
                                                      : VK_LOGIC_OP_NO_OP;
    scheduler.Record([op](vk::CommandBuffer cmdbuf) { cmdbuf.SetLogicOpEXT(op); });
}

void RasterizerVulkan::UpdateBlending(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchBlending()) {
        return;
    }

    if (state_tracker.TouchColorMask()) {
        std::array<VkColorComponentFlags, Maxwell::NumRenderTargets> setup_masks{};
        for (size_t index = 0; index < Maxwell::NumRenderTargets; index++) {
            const auto& mask = regs.color_mask[regs.color_mask_common ? 0 : index];
            auto& current = setup_masks[index];
            if (mask.R) {
                current |= VK_COLOR_COMPONENT_R_BIT;
            }
            if (mask.G) {
                current |= VK_COLOR_COMPONENT_G_BIT;
            }
            if (mask.B) {
                current |= VK_COLOR_COMPONENT_B_BIT;
            }
            if (mask.A) {
                current |= VK_COLOR_COMPONENT_A_BIT;
            }
        }
        scheduler.Record([setup_masks](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetColorWriteMaskEXT(0, setup_masks);
        });
    }

    if (state_tracker.TouchBlendEnable()) {
        std::array<VkBool32, Maxwell::NumRenderTargets> setup_enables{};
        for (size_t index = 0; index < Maxwell::NumRenderTargets; index++) {
            bool is_integer = false;
            if (regs.rt[index].format != Tegra::RenderTargetFormat::NONE) {
                const auto format =
                    VideoCore::Surface::PixelFormatFromRenderTargetFormat(regs.rt[index].format);
                is_integer = IsPixelFormatInteger(format);
            }
            setup_enables[index] =
                (!is_integer && regs.blend.enable[index] != 0) ? VK_TRUE : VK_FALSE;
        }
        scheduler.Record([setup_enables](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetColorBlendEnableEXT(0, setup_enables);
        });
    }

    if (state_tracker.TouchBlendEquations()) {
        std::array<VkColorBlendEquationEXT, Maxwell::NumRenderTargets> setup_blends{};

        const auto blend_setup = [&](auto& host_blend, const auto& guest_blend) {
            host_blend.srcColorBlendFactor = MaxwellToVK::BlendFactor(guest_blend.color_source);
            host_blend.dstColorBlendFactor = MaxwellToVK::BlendFactor(guest_blend.color_dest);
            host_blend.colorBlendOp = MaxwellToVK::BlendEquation(guest_blend.color_op);
            host_blend.srcAlphaBlendFactor = MaxwellToVK::BlendFactor(guest_blend.alpha_source);
            host_blend.dstAlphaBlendFactor = MaxwellToVK::BlendFactor(guest_blend.alpha_dest);
            host_blend.alphaBlendOp = MaxwellToVK::BlendEquation(guest_blend.alpha_op);
        };

        // Single blend equation for all targets
        if (!regs.blend_per_target_enabled) {
            // Temporary workaround for games that use iterated blending
            if (regs.iterated_blend.enable && Settings::values.use_squashed_iterated_blend) {
                setup_blends[0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
                setup_blends[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
                setup_blends[0].colorBlendOp = VK_BLEND_OP_ADD;
                setup_blends[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
                setup_blends[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
                setup_blends[0].alphaBlendOp = VK_BLEND_OP_ADD;
            } else {
                blend_setup(setup_blends[0], regs.blend);
            }

            // Copy first blend state to all other targets
            for (size_t index = 1; index < Maxwell::NumRenderTargets; index++) {
                setup_blends[index] = setup_blends[0];
            }
        } else {
            // Per-target blending
            for (size_t index = 0; index < Maxwell::NumRenderTargets; index++) {
                blend_setup(setup_blends[index], regs.blend_per_target[index]);
            }
        }

        scheduler.Record([setup_blends](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetColorBlendEquationEXT(0, setup_blends);
        });
    }
}

void RasterizerVulkan::UpdateColorWriteEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchColorMask()) {
        return;
    }
    std::array<VkBool32, Maxwell::NumRenderTargets> setup_enables{};
    for (size_t index = 0; index < Maxwell::NumRenderTargets; index++) {
        const auto& mask = regs.color_mask[regs.color_mask_common ? 0 : index];
        setup_enables[index] = (mask.R || mask.G || mask.B || mask.A) ? VK_TRUE : VK_FALSE;
    }
    scheduler.Record([setup_enables](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetColorWriteEnableEXT(setup_enables);
    });
}

void RasterizerVulkan::UpdateStencilTestEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchStencilTestEnable()) {
        return;
    }
    scheduler.Record([enable = regs.stencil_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetStencilTestEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdateVertexInput(Tegra::Engines::Maxwell3D::Regs& regs) {
    auto& dirty{maxwell3d->dirty.flags};
    const bool vertex_input_dirty = dirty[Dirty::VertexInput];
    const bool vertex_buffers_dirty = dirty[VideoCommon::Dirty::VertexBuffers];
    if (!vertex_input_dirty && !vertex_buffers_dirty) {
        return;
    }
    dirty[Dirty::VertexInput] = false;

    boost::container::static_vector<VkVertexInputBindingDescription2EXT, 32> bindings;
    boost::container::static_vector<VkVertexInputAttributeDescription2EXT, 32> attributes;

    const u32 max_attributes =
        static_cast<u32>(std::min<size_t>(Maxwell::NumVertexAttributes,
                                          device.GetMaxVertexInputAttributes()));
    const u32 max_bindings =
        static_cast<u32>(std::min<size_t>(Maxwell::NumVertexArrays,
                                          device.GetMaxVertexInputBindings()));


    for (u32 index = 0; index < max_attributes; ++index) {
        const Maxwell::VertexAttribute attribute{regs.vertex_attrib_format[index]};
        const u32 binding{attribute.buffer};
        if (attribute.constant || binding >= max_bindings) {
            continue;
        }
        attributes.push_back({
            .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
            .pNext = nullptr,
            .location = index,
            .binding = binding,
            .format = MaxwellToVK::VertexFormat(device, attribute.type, attribute.size),
            .offset = attribute.offset,
        });
    }

    for (u32 binding = 0; binding < max_bindings; ++binding) {
        const auto& input_binding{regs.vertex_streams[binding]};
        const bool is_instanced{regs.vertex_stream_instances.IsInstancingEnabled(binding)};
        bindings.push_back({
            .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
            .pNext = nullptr,
            .binding = binding,
            .stride = input_binding.stride,
            .inputRate = is_instanced ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX,
            .divisor = is_instanced ? input_binding.frequency : 1,
        });
    }

    for (size_t index = 0; index < Maxwell::NumVertexAttributes; ++index) {
        dirty[Dirty::VertexAttribute0 + index] = false;
    }
    for (size_t index = 0; index < Maxwell::NumVertexArrays; ++index) {
        dirty[Dirty::VertexBinding0 + index] = false;
    }

    scheduler.Record([bindings, attributes](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetVertexInputEXT(bindings, attributes);
    });
}

void RasterizerVulkan::InitializeChannel(Tegra::Control::ChannelState& channel) {
    CreateChannel(channel);
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        texture_cache.CreateChannel(channel);
        buffer_cache.CreateChannel(channel);
    }
    pipeline_cache.CreateChannel(channel);
    query_cache.CreateChannel(channel);
    state_tracker.SetupTables(channel);
}

void RasterizerVulkan::BindChannel(Tegra::Control::ChannelState& channel) {
    const s32 channel_id = channel.bind_id;
    BindToChannel(channel_id);
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        texture_cache.BindToChannel(channel_id);
        buffer_cache.BindToChannel(channel_id);
    }
    pipeline_cache.BindToChannel(channel_id);
    query_cache.BindToChannel(channel_id);
    state_tracker.ChangeChannel(channel);
    state_tracker.InvalidateState();
}

void RasterizerVulkan::ReleaseChannel(s32 channel_id) {
    EraseChannel(channel_id);
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        texture_cache.EraseChannel(channel_id);
        buffer_cache.EraseChannel(channel_id);
    }
    pipeline_cache.EraseChannel(channel_id);
    query_cache.EraseChannel(channel_id);
}

} // namespace Vulkan
