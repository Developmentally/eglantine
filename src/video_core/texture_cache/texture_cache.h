// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <numeric>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <map>
#include <vector>
#include <optional>
#include <bit>
#include "common/container/unordered_map.h"
#include <boost/container/small_vector.hpp>

#include "common/alignment.h"
#include "common/settings.h"
#include "common/slot_vector.h"
#include "common/diag_trace.h"
#include "core/hle/service/nvdrv/devices/nx2_gpu_memory.h"
#include "video_core/control/channel_state.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/guest_memory.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/texture_cache/collector_accounting.h"
#include "video_core/texture_cache/image_view_base.h"
#include "video_core/texture_cache/samples_helper.h"
#include "video_core/texture_cache/texture_cache_base.h"
#include "video_core/texture_cache/util.h"
#include "video_core/nx2.h"
#include "video_core/nx2_profile.h"
#include "video_core/textures/decoders.h"

namespace VideoCommon {

using Tegra::Texture::TICEntry;
using Tegra::Texture::TSCEntry;
using VideoCore::Surface::GetFormatType;
using VideoCore::Surface::PixelFormat;
using VideoCore::Surface::SurfaceType;
using namespace Common::Literals;

template <class P>
TextureCache<P>::TextureCache(Runtime& runtime_, Tegra::MaxwellDeviceMemoryManager& device_memory_)
    : runtime{runtime_}, device_memory{device_memory_} {
    // Configure null sampler
    TSCEntry sampler_descriptor{};
    sampler_descriptor.min_filter.Assign(Tegra::Texture::TextureFilter::Linear);
    sampler_descriptor.mag_filter.Assign(Tegra::Texture::TextureFilter::Linear);
    sampler_descriptor.mipmap_filter.Assign(Tegra::Texture::TextureMipmapFilter::Linear);
    sampler_descriptor.cubemap_anisotropy.Assign(1);

    // These values were chosen based on typical peak swizzle data sizes seen in some titles
    constexpr size_t SWIZZLE_DATA_BUFFER_INITIAL_CAPACITY = 8_MiB;
    constexpr size_t UNSWIZZLE_DATA_BUFFER_INITIAL_CAPACITY = 1_MiB;
    swizzle_data_buffer.resize_destructive(SWIZZLE_DATA_BUFFER_INITIAL_CAPACITY);
    unswizzle_data_buffer.resize_destructive(UNSWIZZLE_DATA_BUFFER_INITIAL_CAPACITY);

    // Make sure the first index is reserved for the null resources
    // This way the null resource becomes a compile time constant
    void(slot_images.insert(NullImageParams{}));
    const ImageViewId null_view_id = slot_image_views.insert(runtime, NullImageViewParams{});
    // D6b: the null view gets a tracked generation too, so DiagViewGenOf can
    // never read a previous occupant's gen for this slot.
    Common::DiagTrace::Observe(
        [&] { diag_view_gen_[null_view_id.index] = ++diag_view_next_gen_; });
    void(slot_samplers.insert(runtime, sampler_descriptor));

    if constexpr (HAS_DEVICE_MEMORY_INFO) {
        const s64 device_local_memory = static_cast<s64>(runtime.GetDeviceLocalMemory());
        const s64 min_spacing_expected = device_local_memory - 1_GiB;
        const s64 min_spacing_critical = device_local_memory - 512_MiB;
        const s64 mem_threshold = (std::min)(device_local_memory, TARGET_THRESHOLD);
        const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
        const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
        expected_memory = static_cast<u64>(
            (std::max)((std::min)(device_local_memory - min_vacancy_expected, min_spacing_expected),
                     DEFAULT_EXPECTED_MEMORY));
        critical_memory = static_cast<u64>(
            (std::max)((std::min)(device_local_memory - min_vacancy_critical, min_spacing_critical),
                     DEFAULT_CRITICAL_MEMORY));
        minimum_memory = static_cast<u64>((device_local_memory - mem_threshold) / 2);
    } else {
        expected_memory = DEFAULT_EXPECTED_MEMORY + 512_MiB;
        critical_memory = DEFAULT_CRITICAL_MEMORY + 1_GiB;
        minimum_memory = 0;
    }
    ApplyThresholdOverrides();

    const bool gpu_unswizzle_enabled = Settings::values.gpu_unswizzle_enabled.GetValue();

    if (gpu_unswizzle_enabled) {
        switch (Settings::values.gpu_unswizzle_texture_size.GetValue()) {
            case Settings::GpuUnswizzleSize::VerySmall:    gpu_unswizzle_maxsize = 16_MiB; break;
            case Settings::GpuUnswizzleSize::Small:        gpu_unswizzle_maxsize = 32_MiB; break;
            case Settings::GpuUnswizzleSize::Normal:       gpu_unswizzle_maxsize = 128_MiB; break;
            case Settings::GpuUnswizzleSize::Large:        gpu_unswizzle_maxsize = 256_MiB; break;
            case Settings::GpuUnswizzleSize::VeryLarge:    gpu_unswizzle_maxsize = 512_MiB; break;
            default:                                       gpu_unswizzle_maxsize = 128_MiB; break;
        }

        switch (Settings::values.gpu_unswizzle_stream_size.GetValue()) {
            case Settings::GpuUnswizzle::VeryLow: swizzle_chunk_size = 4_MiB; break;
            case Settings::GpuUnswizzle::Low:     swizzle_chunk_size = 8_MiB; break;
            case Settings::GpuUnswizzle::Normal:  swizzle_chunk_size = 16_MiB; break;
            case Settings::GpuUnswizzle::Medium:  swizzle_chunk_size = 32_MiB; break;
            case Settings::GpuUnswizzle::High:    swizzle_chunk_size = 64_MiB; break;
            default:                              swizzle_chunk_size = 16_MiB;
        }

        switch (Settings::values.gpu_unswizzle_chunk_size.GetValue()) {
            case Settings::GpuUnswizzleChunk::VeryLow: swizzle_slices_per_batch = 32; break;
            case Settings::GpuUnswizzleChunk::Low:     swizzle_slices_per_batch = 64; break;
            case Settings::GpuUnswizzleChunk::Normal:  swizzle_slices_per_batch = 128; break;
            case Settings::GpuUnswizzleChunk::Medium:  swizzle_slices_per_batch = 256; break;
            case Settings::GpuUnswizzleChunk::High:    swizzle_slices_per_batch = 512; break;
            default:                                   swizzle_slices_per_batch = 128;
        }
    } else {
        gpu_unswizzle_maxsize = 0;
        swizzle_chunk_size = 0;
        swizzle_slices_per_batch = 0;
    }
}

template <class P>
void TextureCache<P>::RunGarbageCollector() {
    bool high_priority_mode = false;
    bool aggressive_mode = false;
    u64 ticks_to_destroy = 0;
    size_t num_iterations = 0;
    const auto Configure = [&](bool allow_aggressive) {
        high_priority_mode = total_used_memory >= expected_memory;
        aggressive_mode = allow_aggressive && total_used_memory >= critical_memory;
        ticks_to_destroy = aggressive_mode ? 10ULL : high_priority_mode ? 25ULL : 50ULL;
        // EDEN_NX2_TC_MIN_AGE=<frames>: never evict images used more recently (thrash tests).
        static const u64 min_age = [] {
            const char* value = std::getenv("EDEN_NX2_TC_MIN_AGE");
            return value != nullptr ? std::strtoull(value, nullptr, 10) : 0ULL;
        }();
        ticks_to_destroy = (std::max)(ticks_to_destroy, min_age);
        num_iterations = aggressive_mode ? 40 : (high_priority_mode ? 20 : 10);
    };
    bool allow_costly = false;
    const auto Cleanup = [this, &num_iterations, &high_priority_mode, &aggressive_mode, &allow_costly](ImageId image_id) {
        if (num_iterations == 0) {
            return true;
        }
        --num_iterations;
        auto& image = slot_images[image_id];
        if (True(image.flags & ImageFlagBits::IsDecoding)) {
            return false;
        }
        const bool must_download = IsDownloadable(image) && False(image.flags & ImageFlagBits::BadOverlap);
        if ((!allow_costly && True(image.flags & ImageFlagBits::CostlyLoad)) || (!high_priority_mode && must_download)) {
            return false;
        }
        if (must_download) {
            auto map = runtime.DownloadStagingBuffer(image.unswizzled_size_bytes);
            const auto copies = FullDownloadCopies(image.info);
            image.DownloadMemory(map, copies);
            runtime.Finish();
            SwizzleImage(*gpu_memory, image.gpu_addr, image.info, copies, map.mapped_span, swizzle_data_buffer);
            ++gc_download_deletes;
        }
        if (True(image.flags & ImageFlagBits::Tracked)) {
            UntrackImage(image, image_id);
        }
        if (image.guest_size_bytes >= 64_MiB) {
            static u32 logged{};
            if (logged++ < 40) {
                LOG_INFO(HW_GPU, "Texture collector evicts {} MiB image {:#x} ({}x{}x{} layers {} levels {} format {}, costly {}, {} frames unused; collector counts {} MiB, expected {} MiB, critical {} MiB)",
                         image.guest_size_bytes >> 20, image.gpu_addr, image.info.size.width,
                         image.info.size.height, image.info.size.depth, image.info.resources.layers,
                         image.info.resources.levels, static_cast<u32>(image.info.format),
                         True(image.flags & ImageFlagBits::CostlyLoad),
                         frame_tick - lru_cache.GetTick(image.lru_index), total_used_memory >> 20,
                         expected_memory >> 20, critical_memory >> 20);
            }
        }
        ++(True(image.flags & ImageFlagBits::CostlyLoad) ? gc_costly_deletes : gc_deletes);
        UnregisterImage(image_id);
        DeleteImage(image_id, image.scale_tick > frame_tick + 5, "collector");
        if (aggressive_mode && total_used_memory < critical_memory) {
            num_iterations >>= 2;
            aggressive_mode = false;
        } else if (high_priority_mode && total_used_memory < expected_memory) {
            num_iterations >>= 1;
            high_priority_mode = false;
        }
        return false;
    };
    Configure(false);
    lru_cache.ForEachItemBelow(frame_tick - ticks_to_destroy, Cleanup);
    if (total_used_memory >= critical_memory) {
        Configure(true);
        lru_cache.ForEachItemBelow(frame_tick - ticks_to_destroy, Cleanup);
    }
    // Costly images (big layered arrays: seconds of unswizzling to bring back) only when still
    // over the line, and only ones unused for a while. Evicting one Mario Kart World used 10
    // frames earlier made the collector re-upload 260 MiB every few seconds, the GPU back
    // thread unswizzling while both channels waited: 4-10 s stalls in the race.
    if (total_used_memory >= critical_memory) {
        Configure(true);
        allow_costly = true;
        static const u64 costly_age = [] {
            const char* value = std::getenv("EDEN_NX2_TC_COSTLY_AGE");
            return value != nullptr ? std::strtoull(value, nullptr, 10) : 90ULL;
        }();
        lru_cache.ForEachItemBelow(frame_tick - (std::max)(ticks_to_destroy, costly_age), Cleanup);
    }
}

template <class P>
void TextureCache<P>::RetryUnbackedImages() {
    // An image the allocator left unbacked under the UMA hard limit reads as black until something
    // recreates it. Once there is room for it (and a margin for what follows), drop it: its next
    // use builds it again from guest memory, which an unbacked image never changed. Images too
    // big to ever back (the allocator's huge rule) and images unbacked on purpose (sparse layers,
    // unwritten sparse regions) stay as they are.
    if constexpr (requires { runtime.OverUmaHardLimit(u64{}); }) {
        constexpr auto deliberate = ImageFlagBits::IsDecoding | ImageFlagBits::SparseLayers;
        u32 retried = 0;
        for (size_t i = 0; i < unbacked_image_ids.size() && retried < 2;) {
            const ImageId image_id = unbacked_image_ids[i];
            Image& image = slot_images[image_id];
            const ImageInfo& info = image.info;
            const u64 texels = u64{info.size.width} * info.size.height * info.size.depth *
                               static_cast<u64>(info.resources.layers);
            // UnwrittenSparse also marks hard-limit refusals (their first GPU write replaces
            // them); only images over a region nobody has written stay as they are.
            if (texels > (u64{1} << 28) || True(image.flags & deliberate) ||
                (True(image.flags & ImageFlagBits::UnwrittenSparse) &&
                 Tegra::Nx2SparseUnwritten(image.gpu_addr, image.guest_size_bytes))) {
                ++i;
                continue;
            }
            const u64 size = True(image.flags & ImageFlagBits::Converted)
                                 ? CalculateConvertedSizeBytes(info)
                                 : CalculateUnswizzledSizeBytes(info);
            if (runtime.OverUmaHardLimit(size + 256_MiB)) {
                return;
            }
            if (unbacked_retries++ < 32) {
                LOG_INFO(HW_GPU, "Retrying unbacked image {:#x} {}x{}x{} layers {} format {} ({} MiB)",
                         image.gpu_addr, info.size.width, info.size.height, info.size.depth,
                         info.resources.layers, static_cast<u32>(info.format), size >> 20);
            }
            if (True(image.flags & ImageFlagBits::Tracked)) {
                UntrackImage(image, image_id);
            }
            // UnregisterImage removes the id from unbacked_image_ids, so i stays.
            UnregisterImage(image_id);
            DeleteImage(image_id, false, "retry");
            ++retried;
        }
    }
}

template <class P>
void TextureCache<P>::ApplyThresholdOverrides(bool log) {
    // EDEN_NX2_TC_EXPECTED_MIB / EDEN_NX2_TC_CRITICAL_MIB override the collector's thresholds.
    for (const auto& [name, target] : {std::pair{"EDEN_NX2_TC_EXPECTED_MIB", &expected_memory},
                                       std::pair{"EDEN_NX2_TC_CRITICAL_MIB", &critical_memory}}) {
        if (const char* value = std::getenv(name)) {
            *target = std::strtoull(value, nullptr, 10) << 20;
        }
    }
    if (log) {
        LOG_INFO(HW_GPU, "Texture cache thresholds: expected {} MiB, critical {} MiB, minimum {} MiB",
                 expected_memory >> 20, critical_memory >> 20, minimum_memory >> 20);
    }
}

template <class P>
void TextureCache<P>::TickFrame() {
    // Upload buffers grown for one huge image (title 1's 341 MiB terrain layer pools) are freed
    // rather than kept for the rest of the session.
    for (auto* const scratch : {&swizzle_data_buffer, &unswizzle_data_buffer}) {
        if (scratch->capacity() > LargeScratchCapacity) {
            scratch->release();
        }
    }
    if constexpr (HAS_DEVICE_MEMORY_INFO) {
        // Switch 2 titles (native tiling) stream far more texture data than Eden's thresholds
        // were tuned for: title 1's attract loop alone keeps ~5.3 GB resident, so the collector
        // sat in aggressive mode evicting decoded ASTC (and decoding it again). Total use is
        // measured from the driver, so collect only near the device budget instead.
        if (!native_thresholds_applied && gpu_memory != nullptr && gpu_memory->UsesNativeTiling()) {
            native_thresholds_applied = true;
            const u64 device_local = runtime.GetDeviceLocalMemory();
            critical_memory = (std::max)(critical_memory, device_local - 256_MiB);
            expected_memory = (std::max)(expected_memory, device_local - 1280_MiB);
            ApplyThresholdOverrides();
            native_critical_memory = critical_memory;
            native_expected_memory = expected_memory;
        }
        // Follow the driver's budget (re-read now and then): it shrinks when other programs use
        // VRAM (another game: 7.0 -> 6.4 GB), and collecting only past it means the driver pages
        // textures out. It is also the room to use: Eden's device memory figure is capped near
        // 7 GB, which on a 12 GB card (11.2 GB budget) kept title 1 evicting textures with 3 GB
        // unused. EDEN_NX2_TC_*_MIB overrides still win.
        if constexpr (requires { runtime.GetDeviceMemoryBudget(); }) {
            if (native_thresholds_applied && budget_check_frames++ % 60 == 0) {
                const u64 budget = runtime.GetDeviceMemoryBudget();
                if (budget > 2_GiB) {
                    const u64 old_critical = critical_memory;
                    critical_memory = budget - 768_MiB;
                    expected_memory = budget - 1536_MiB;
                    // Log only real changes (the budget moves by a few MiB all the time).
                    const u64 change = critical_memory > old_critical
                                           ? critical_memory - old_critical
                                           : old_critical - critical_memory;
                    ApplyThresholdOverrides(change >= 256_MiB);
                }
            }
        }
    }
    // If we can obtain the memory info, use it instead of the estimate.
    if (runtime.CanReportMemoryUsage()) {
        driver_used_memory = runtime.GetDeviceMemoryUsage();
        // Images the collector deleted keep their memory for TICKS_TO_DESTROY frames: counting
        // it kept the collector in aggressive mode, deleting 40+ images a frame for frames after
        // it had freed enough (title 1: frames over 100 ms for seconds at a time).
        // The discount reuses the same estimates the non-driver path books (RegisterImage/
        // DeleteImage) and saturates at the driver total: saturation avoids underflow, it does
        // not prove estimates equal committed bytes. Collector decisions use the discounted
        // figure; the raw driver figure stays in driver_used_memory for honest telemetry.
        // The aggregate only feeds the clamp against the driver figure, so clamp the running
        // sum at that figure: multiple individually valid slots can never wrap u64, and the
        // result is identical to min(driver, exact sum).
        u64 sentenced_pending_bytes = 0;
        for (const u64 bytes : sentenced_image_bytes) {
            sentenced_pending_bytes += (std::min)(bytes, driver_used_memory - sentenced_pending_bytes);
        }
        total_used_memory = driver_used_memory - sentenced_pending_bytes;
    }
    // Every 600 frames (every 200 under EDEN_NX2_GPU_PROFILE): device memory in use against the
    // collector's thresholds, and what the images themselves account for (VRAM pressure makes the
    // collector thrash).
    static const bool profile = std::getenv("EDEN_NX2_GPU_PROFILE") != nullptr;
    static const u64 stats_interval = profile ? 200 : 600;
    // EDEN_NX2_MEM_TRACE=1: the same every 60 frames, to see what fills the device during a load.
    static const bool mem_trace = std::getenv("EDEN_NX2_MEM_TRACE") != nullptr;
    if (static u64 trace_frames{}; mem_trace && ++trace_frames % 60 == 0) {
        if (runtime.CanReportMemoryUsage()) {
            LOG_INFO(HW_GPU,
                     "mem trace: device memory {} MiB (collector counts {} MiB, {} MiB pending destruction), {} images, {} unbacked",
                     driver_used_memory >> 20, total_used_memory >> 20,
                     (driver_used_memory - (std::min)(driver_used_memory, total_used_memory)) >> 20, slot_images.size(),
                     unbacked_image_ids.size());
        } else {
            LOG_INFO(HW_GPU, "mem trace: estimated memory {} MiB, {} images, {} unbacked",
                     total_used_memory >> 20, slot_images.size(), unbacked_image_ids.size());
        }
        if constexpr (requires { runtime.LogMemoryStatistics(); }) {
            runtime.LogMemoryStatistics();
        }
    }
    DumpRequestedImages();
    if (profile) {
        // Under the profile, every 256 MiB step up or down as well: the Deck's GTT pool fills
        // in seconds when a title loads, between two regular reports.
        static u64 last_logged{};
        const bool reported = runtime.CanReportMemoryUsage();
        const u64 device_now = reported ? driver_used_memory : total_used_memory;
        if (device_now + (256_MiB) < last_logged || device_now > last_logged + (256_MiB)) {
            last_logged = device_now;
            if (reported) {
                LOG_INFO(HW_GPU, "Texture cache: device memory now {} MiB (collector counts {} MiB)",
                         driver_used_memory >> 20, total_used_memory >> 20);
            } else {
                LOG_INFO(HW_GPU, "Texture cache: estimated memory now {} MiB",
                         total_used_memory >> 20);
            }
            if constexpr (requires { runtime.LogMemoryStatistics(); }) {
                runtime.LogMemoryStatistics();
            }
        }
    }
    if (static u64 frames{}; ++frames % stats_interval == 0) {
        u64 image_bytes{};
        u64 host_bytes{};
        size_t images{};
        std::vector<std::pair<u64, const Image*>> largest;
        struct FormatTotal {
            u64 guest_bytes{};
            u64 host_bytes{};
            u32 count{};
        };
        std::map<u32, FormatTotal> by_format;
        // Guest bytes are what the collector estimates; the host side (ASTC recompressed to BCn,
        // converted formats, scaled copies) is what the device actually holds.
        const auto host_size = [](const Image& image) -> u64 {
            if constexpr (requires { image.HostMemoryBytes(); }) {
                return image.HostMemoryBytes();
            } else {
                return 0;
            }
        };
        for (const auto [image_id, image] : slot_images) {
            const u64 bytes = GetScaledImageSizeBytes(*image);
            const u64 host = host_size(*image);
            image_bytes += bytes;
            host_bytes += host;
            ++images;
            largest.emplace_back((std::max)(bytes, host), image);
            auto& format = by_format[static_cast<u32>(image->info.format)];
            format.guest_bytes += bytes;
            format.host_bytes += host;
            ++format.count;
        }
        if (runtime.CanReportMemoryUsage()) {
            LOG_INFO(HW_GPU, "Texture cache: device memory {} MiB, collector counts {} MiB (expected {} MiB, critical {} MiB), {} images {} MiB guest / {} MiB host",
                     driver_used_memory >> 20, total_used_memory >> 20, expected_memory >> 20,
                     critical_memory >> 20, images, image_bytes >> 20, host_bytes >> 20);
        } else {
            LOG_INFO(HW_GPU, "Texture cache: estimated memory {} MiB (expected {} MiB, critical {} MiB), {} images {} MiB guest / {} MiB host",
                     total_used_memory >> 20, expected_memory >> 20, critical_memory >> 20, images,
                     image_bytes >> 20, host_bytes >> 20);
        }
        std::partial_sort(largest.begin(), largest.begin() + std::min<size_t>(12, largest.size()),
                          largest.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        for (size_t i = 0; i < std::min<size_t>(12, largest.size()); ++i) {
            const Image& image = *largest[i].second;
            LOG_INFO(HW_GPU, "  {} MiB guest / {} MiB host: {:#x} {}x{}x{} layers {} levels {} format {} type {} flags {:#x}",
                     GetScaledImageSizeBytes(image) >> 20, host_size(image) >> 20, image.gpu_addr,
                     image.info.size.width, image.info.size.height, image.info.size.depth,
                     image.info.resources.layers, image.info.resources.levels,
                     static_cast<u32>(image.info.format), static_cast<u32>(image.info.type),
                     static_cast<u32>(image.flags));
        }
        std::vector<std::pair<u64, u32>> formats;
        for (const auto& [format, total] : by_format) {
            formats.emplace_back((std::max)(total.guest_bytes, total.host_bytes), format);
        }
        std::sort(formats.rbegin(), formats.rend());
        for (size_t i = 0; i < std::min<size_t>(8, formats.size()); ++i) {
            const FormatTotal& total = by_format[formats[i].second];
            LOG_INFO(HW_GPU, "  format {}: {} MiB guest / {} MiB host in {} images", formats[i].second,
                     total.guest_bytes >> 20, total.host_bytes >> 20, total.count);
        }
        if constexpr (requires { runtime.LogMemoryStatistics(); }) {
            runtime.LogMemoryStatistics();
        }
        LOG_INFO(HW_GPU, "Texture cache last {} frames: created {} images, deleted {} (collector {}, {} costly, {} downloaded), {} async decodes, ASTC {} opaque / {} alpha",
                 stats_interval, images_created, images_deleted, gc_deletes + gc_costly_deletes, gc_costly_deletes,
                 gc_download_deletes,
                 async_decodes_queued, astc_opaque_images, astc_alpha_images);
        LOG_INFO(HW_GPU, "Texture cache last 600 frames: {} uploads from guest memory, {} MiB (largest {} MiB, {} by layer), {} written back for the CPU in {} flushes, {} render targets re-created at another size ({} revived, {} MiB dormant)",
                 images_uploaded, uploaded_bytes >> 20, largest_upload >> 20, layer_uploads,
                 images_written_back, download_requests, render_target_resizes,
                 render_target_revivals, dormant_bytes >> 20);
        images_uploaded = uploaded_bytes = largest_upload = layer_uploads = images_written_back =
            download_requests = render_target_resizes = render_target_revivals = 0;
        astc_opaque_images = astc_alpha_images = 0;
        std::vector<std::pair<u32, u64>> shapes;
        for (const auto& [key, value] : created_shapes) shapes.emplace_back(value.first, key);
        std::sort(shapes.rbegin(), shapes.rend());
        for (size_t i = 0; i < std::min<size_t>(10, shapes.size()); ++i) {
            const u64 key = shapes[i].second;
            LOG_DEBUG(HW_GPU, "  created {}x: format {} type {} levels {} layers {} {}x{} (last at {:#x})",
                     shapes[i].first, key >> 56, (key >> 52) & 0xF, (key >> 48) & 0xF,
                     (key >> 36) & 0xFFF, (key >> 18) & 0x3FFFF, key & 0x3FFFF,
                     created_shapes[key].second);
        }
        created_shapes.clear();
        gc_deletes = gc_costly_deletes = gc_download_deletes = async_decodes_queued = 0;
        images_created = images_deleted = 0;
    }
    if (total_used_memory > minimum_memory) {
        RunGarbageCollector();
    }
    EvictDormantImages();
    RetryUnbackedImages();
    sentenced_images.Tick();
    sentenced_image_index = (sentenced_image_index + 1) % TICKS_TO_DESTROY;
    sentenced_image_bytes[sentenced_image_index] = 0;
    sentenced_framebuffers.Tick();
    sentenced_image_view.Tick();
    TickAsyncDecode();
    TickAsyncUnswizzle();

    runtime.TickFrame();
    if (traced_writes_this_frame != 0) {
        static u32 traced_frames{};
        if (traced_frames++ < 3000) {
            LOG_INFO(HW_GPU, "nx2 traced range frame {}: {} GPU writes into{}", frame_tick,
                     traced_writes_this_frame, traced_images_text);
        }
        traced_writes_this_frame = 0;
        traced_images_this_frame.clear();
        traced_images_text.clear();
    }
    ++frame_tick;
    Common::DiagTrace::TickFlush(frame_tick);
    if (Common::DiagTrace::Enabled() && !diag_snapshot_done &&
        Common::DiagTrace::InWindow(frame_tick)) {
        // Pre-window identity snapshot: the live table state when capture begins.
        // Observer-only containment: a fault here can never escape TickFrame.
        // Drain any pre-existing ring contents first: a full ring at snapshot
        // start would overwrite older events before the first chunk flush
        // (still not a guarantee against concurrent emitters - losses are
        // counted, not prevented).
        Common::DiagTrace::Flush(frame_tick, "identity-snap-start");
        Common::DiagTrace::Observe([&] {
            diag_snapshot_done = true;
            // D7: state-1 (deleted) records are skipped - their lineage is
            // already evidenced by ImageErase - and COUNTED, reported in the
            // follow-up Config event. The burst is chunked with intermediate
            // flushes so a large table cannot overflow the ring in one burst.
            u64 diag_snap_emitted = 0;
            u64 diag_snap_deleted_skipped = 0;
            for (const auto& [slot, rec] : diag_images_) {
                if (rec.state == 1) {
                    ++diag_snap_deleted_skipped;
                    continue;
                }
                char diag_tag[40];
                // asid tags print MemoryManager id + 1; 0 would mean untracked.
                std::snprintf(diag_tag, sizeof(diag_tag), "gen%u as%llx", rec.generation,
                              (unsigned long long)(rec.asid + 1));
                Common::DiagTrace::Emit(Common::DiagTrace::Ev::IdentitySnap, frame_tick,
                                        rec.trace_obj, slot, rec.gpu_addr, rec.state, diag_tag);
                if (++diag_snap_emitted % 1024 == 0) {
                    Common::DiagTrace::Flush(frame_tick, "identity-snap-chunk");
                }
            }
            Common::DiagTrace::Emit(Common::DiagTrace::Ev::Config, frame_tick,
                                    diag_images_.size(),
                                    Common::DiagTrace::S().identity_skipped.load(
                                        std::memory_order_relaxed),
                                    diag_view_gen_.size(),
                                    Common::DiagTrace::S().view_gen_skipped.load(
                                        std::memory_order_relaxed),
                                    "identity-snapshot-done");
            Common::DiagTrace::Emit(Common::DiagTrace::Ev::Config, frame_tick,
                                    diag_snap_emitted, diag_snap_deleted_skipped, 0, 0,
                                    "identity-snapshot-states");
        });
    }

    if constexpr (IMPLEMENTS_ASYNC_DOWNLOADS) {
        for (auto& buffer : async_buffers_death_ring) {
            runtime.FreeDeferredStagingBuffer(buffer);
        }
        async_buffers_death_ring.clear();
    }
}

template <class P>
const typename P::ImageView& TextureCache<P>::GetImageView(ImageViewId id) const noexcept {
    return slot_image_views[id];
}

template <class P>
typename P::ImageView& TextureCache<P>::GetImageView(ImageViewId id) noexcept {
    return slot_image_views[id];
}

template <class P>
typename P::ImageView& TextureCache<P>::GetImageView(u32 index) noexcept {
    // Not compute!
    const auto image_view_id = VisitImageView(index, false);
    return slot_image_views[image_view_id];
}

template <class P>
void TextureCache<P>::MarkModification(ImageId id) noexcept {
    MarkModification(slot_images[id]);
}

template <class P>
void TextureCache<P>::FillImageViews(std::span<ImageViewInOut> views, bool compute, bool blacklist) {
    bool has_blacklisted = false;
    do {
        has_deleted_images = false;
        if (blacklist) {
            has_blacklisted = false;
        }
        for (ImageViewInOut& view : views) {
            view.id = VisitImageView(view.index, compute);
            // `blacklist` is set on written storage images. One over an unwritten sparse region
            // reads zero and has no memory to store into: replace it before the program runs
            // (render targets get the same treatment in PrepareImage).
            if (view.blacklist && view.id != NULL_IMAGE_VIEW_ID &&
                ReplaceUnwrittenSparseStorageImage(view.id)) {
                break;
            }
            if (blacklist) {
                if (view.blacklist && view.id != NULL_IMAGE_VIEW_ID) {
                    const ImageViewBase& image_view = slot_image_views[view.id];
                    auto& image = slot_images[image_view.image_id];
                    has_blacklisted |= ScaleDown(image);
                    image.scale_rating = 0;
                }
            }
        }
    } while (has_deleted_images || (blacklist && has_blacklisted));
}

template <class P>
void TextureCache<P>::CheckFeedbackLoop(std::span<const ImageViewInOut> views) {
    if (!Settings::values.barrier_feedback_loops.GetValue()) {
        return;
    }

    if (render_targets_serial == last_feedback_loop_serial &&
        texture_bindings_serial == last_feedback_texture_serial) {
        if (last_feedback_loop_result) {
            runtime.BarrierFeedbackLoop();
        }
        return;
    }

    if (rt_active_mask == 0) {
        last_feedback_loop_serial = render_targets_serial;
        last_feedback_texture_serial = texture_bindings_serial;
        last_feedback_loop_result = false;
        return;
    }
    const u32 depth_bit = 1u << NUM_RT;
    const bool depth_active = (rt_active_mask & depth_bit) != 0;

    const bool requires_barrier = [&] {
        for (const auto& view : views) {
            if (!view.id) {
                continue;
            }

            {
                bool is_continue = false;
                for (size_t i = 0; i < 8; ++i)
                    is_continue |= (rt_active_mask & (1u << i)) && view.id == render_targets.color_buffer_ids[i];
                if (is_continue)
                    continue;
            }

            if (depth_active && view.id == render_targets.depth_buffer_id)
                continue;

            const ImageId view_image_id = slot_image_views[view.id].image_id;
            {
                bool is_continue = false;
                for (size_t i = 0; i < 8; ++i)
                    is_continue |= (rt_active_mask & (1u << i)) && view_image_id == rt_image_id[i];
                if (is_continue)
                    continue;
            }
            if (depth_active && view_image_id == rt_depth_image_id) {
                return true;
            }
        }

        return false;
    }();

    last_feedback_loop_serial = render_targets_serial;
    last_feedback_texture_serial = texture_bindings_serial;
    last_feedback_loop_result = requires_barrier;
    if (requires_barrier) {
        runtime.BarrierFeedbackLoop();
    }
}

template <class P>
typename P::Sampler* TextureCache<P>::GetSampler(u32 index, bool compute) {
    return &slot_samplers[GetSamplerId(index, compute)];
}

template <class P>
SamplerId TextureCache<P>::GetSamplerId(u32 index, bool compute) {
    auto& table = compute ? channel_state->compute_sampler_table : channel_state->graphics_sampler_table;
    if (index > table.current_limit) {
        LOG_DEBUG(HW_GPU, "Invalid sampler index={}", index);
        return NULL_SAMPLER_ID;
    }
    auto const map_index = index | (compute ? Common::SlotId::TAGGED_VALUE : 0);
    auto const [descriptor, is_new] = table.Read(*gpu_memory, index);
    if (is_new) {
        auto const id = FindSampler(descriptor, compute);
        channel_state->sampler_ids.insert_or_assign(map_index, id);
        return id;
    }
    return channel_state->sampler_ids.find(map_index)->second;
}

template <class P>
const typename P::Sampler& TextureCache<P>::GetSampler(SamplerId id) const noexcept {
    return slot_samplers[id];
}

template <class P>
typename P::Sampler& TextureCache<P>::GetSampler(SamplerId id) noexcept {
    return slot_samplers[id];
}

template <class P>
void TextureCache<P>::SynchronizeDescriptors(bool compute) {
    if (compute) {
        const bool linked_tsc = kepler_compute->launch_description.linked_tsc;
        const u32 tic_limit = kepler_compute->regs.tic.limit;
        const u32 tsc_limit = linked_tsc ? tic_limit : kepler_compute->regs.tsc.limit;
        bool bindings_changed = false;
        if (channel_state->compute_sampler_table.Synchronize(kepler_compute->regs.tsc.Address(), tsc_limit))
            bindings_changed = true;
        if (channel_state->compute_image_table.Synchronize(kepler_compute->regs.tic.Address(), tic_limit))
            bindings_changed = true;
        if (bindings_changed) {
            ++texture_bindings_serial;
        }
    } else {
        const bool linked_tsc = maxwell3d->regs.sampler_binding == Tegra::Engines::Maxwell3D::Regs::SamplerBinding::ViaHeaderBinding;
        const u32 tic_limit = maxwell3d->regs.tex_header.limit;
        const u32 tsc_limit = linked_tsc ? tic_limit : maxwell3d->regs.tex_sampler.limit;
        bool bindings_changed = false;
        if (channel_state->graphics_sampler_table.Synchronize(maxwell3d->regs.tex_sampler.Address(), tsc_limit))
            bindings_changed = true;
        if (channel_state->graphics_image_table.Synchronize(maxwell3d->regs.tex_header.Address(), tic_limit))
            bindings_changed = true;
        if (bindings_changed) {
            ++texture_bindings_serial;
        }
    }
}

template <class P>
bool TextureCache<P>::RescaleRenderTargets() {
    auto& flags = maxwell3d->dirty.flags;
    u32 scale_rating = 0;
    bool rescaled = false;
    std::array<ImageId, NUM_RT> tmp_color_images{};
    ImageId tmp_depth_image{};
    do {
        flags[Dirty::RenderTargets] = false;

        has_deleted_images = false;
        // Render target control is used on all render targets, so force look ups when this one is
        // up
        const bool force = flags[Dirty::RenderTargetControl];
        flags[Dirty::RenderTargetControl] = false;

        scale_rating = 0;
        bool any_rescaled = false;
        bool can_rescale = true;
        const auto check_rescale = [&](ImageViewId view_id, ImageId& id_save) {
            if (view_id != NULL_IMAGE_VIEW_ID && view_id != ImageViewId{}) {
                const auto& view = slot_image_views[view_id];
                const auto image_id = view.image_id;
                id_save = image_id;
                auto& image = slot_images[image_id];
                can_rescale &= ImageCanRescale(image);
                any_rescaled |= True(image.flags & ImageFlagBits::Rescaled) ||
                                GetFormatType(image.info.format) != SurfaceType::ColorTexture;
                scale_rating = std::max<u32>(scale_rating, image.scale_tick <= frame_tick
                                                               ? image.scale_rating + 1U
                                                               : image.scale_rating);
            } else {
                id_save = CORRUPT_ID;
            }
        };
        for (size_t index = 0; index < NUM_RT; ++index) {
            ImageViewId& color_buffer_id = render_targets.color_buffer_ids[index];
            if (flags[Dirty::ColorBuffer0 + index] || force) {
                flags[Dirty::ColorBuffer0 + index] = false;
                BindRenderTarget(&color_buffer_id, FindColorBuffer(index));
            }
            check_rescale(color_buffer_id, tmp_color_images[index]);
        }
        if (flags[Dirty::ZetaBuffer] || force) {
            flags[Dirty::ZetaBuffer] = false;
            BindRenderTarget(&render_targets.depth_buffer_id, FindDepthBuffer());
        }
        check_rescale(render_targets.depth_buffer_id, tmp_depth_image);

        if (can_rescale) {
            rescaled = any_rescaled || scale_rating >= 2;
            const auto scale_up = [this](ImageId image_id) {
                if (image_id != CORRUPT_ID) {
                    Image& image = slot_images[image_id];
                    ScaleUp(image);
                }
            };
            if (rescaled) {
                for (size_t index = 0; index < NUM_RT; ++index) {
                    scale_up(tmp_color_images[index]);
                }
                scale_up(tmp_depth_image);
                scale_rating = 2;
            }
        } else {
            rescaled = false;
            const auto scale_down = [this](ImageId image_id) {
                if (image_id != CORRUPT_ID) {
                    Image& image = slot_images[image_id];
                    ScaleDown(image);
                }
            };
            for (size_t index = 0; index < NUM_RT; ++index) {
                scale_down(tmp_color_images[index]);
            }
            scale_down(tmp_depth_image);
            scale_rating = 1;
        }
    } while (has_deleted_images);
    const auto set_rating = [this, scale_rating](ImageId image_id) {
        if (image_id != CORRUPT_ID) {
            Image& image = slot_images[image_id];
            image.scale_rating = scale_rating;
            if (image.scale_tick <= frame_tick) {
                image.scale_tick = frame_tick + 1;
            }
        }
    };
    for (size_t index = 0; index < NUM_RT; ++index) {
        set_rating(tmp_color_images[index]);
    }
    set_rating(tmp_depth_image);

    return rescaled;
}

template <class P>
void TextureCache<P>::UpdateRenderTargets(bool is_clear) {
    using namespace VideoCommon::Dirty;
    auto& flags = maxwell3d->dirty.flags;
    if (!flags[Dirty::RenderTargets]) {
        for (size_t index = 0; index < NUM_RT; ++index) {
            ImageViewId& color_buffer_id = render_targets.color_buffer_ids[index];
            PrepareImageView(color_buffer_id, true, is_clear && IsFullClear(color_buffer_id));
        }
        const ImageViewId depth_buffer_id = render_targets.depth_buffer_id;
        PrepareImageView(depth_buffer_id, true, is_clear && IsFullClear(depth_buffer_id));
        return;
    }

    const VideoCommon::RenderTargets previous_render_targets = render_targets;
    const bool rescaled = RescaleRenderTargets();
    if (is_rescaling != rescaled) {
        flags[Dirty::RescaleViewports] = true;
        flags[Dirty::RescaleScissors] = true;
        is_rescaling = rescaled;
    }

    for (size_t index = 0; index < NUM_RT; ++index) {
        ImageViewId& color_buffer_id = render_targets.color_buffer_ids[index];
        PrepareImageView(color_buffer_id, true, is_clear && IsFullClear(color_buffer_id));
    }
    const ImageViewId depth_buffer_id = render_targets.depth_buffer_id;

    PrepareImageView(depth_buffer_id, true, is_clear && IsFullClear(depth_buffer_id));

    rt_active_mask = 0;
    rt_image_id = {};
    for (size_t i = 0; i < rt_image_id.size(); ++i) {
        if (ImageViewId const view = render_targets.color_buffer_ids[i]; view) {
            rt_active_mask |= 1u << i;
            rt_image_id[i] = slot_image_views[view].image_id;
        }
    }
    if (depth_buffer_id) {
        rt_active_mask |= (1u << NUM_RT);
        rt_depth_image_id = slot_image_views[depth_buffer_id].image_id;
    } else {
        rt_depth_image_id = ImageId{};
    }

    for (size_t index = 0; index < NUM_RT; ++index) {
        render_targets.draw_buffers[index] = static_cast<u8>(maxwell3d->regs.rt_control.Map(index));
    }
    u32 up_scale = 1;
    u32 down_shift = 0;
    if (is_rescaling) {
        up_scale = Settings::values.resolution_info.up_scale;
        down_shift = Settings::values.resolution_info.down_shift;
    }
    render_targets.size = Extent2D{
        (maxwell3d->regs.surface_clip.width * up_scale) >> down_shift,
        (maxwell3d->regs.surface_clip.height * up_scale) >> down_shift,
    };
    render_targets.is_rescaled = is_rescaling;

    if (render_targets != previous_render_targets) {
        ++render_targets_serial;
    }

    flags[Dirty::DepthBiasGlobal] = true;
}

template <class P>
typename P::Framebuffer* TextureCache<P>::GetFramebuffer() {
    // F1: tie a recorded draw to its render targets. Selected draws only
    // (volume bound, same gate as full per-bind DescFinal); emitted at
    // draw-configure RT resolution on both the cache-hit and the new path.
    if (Common::DiagTrace::tl_selected_draw) {
        Common::DiagTrace::Observe([&] {
            for (size_t rt = 0; rt < NUM_RT; ++rt) {
                const ImageViewId id = render_targets.color_buffer_ids[rt];
                if (!id) {
                    continue;
                }
                char diag_rt_tag[40];
                std::snprintf(diag_rt_tag, sizeof(diag_rt_tag), "v%u g%llu", id.index,
                              (unsigned long long)DiagViewGenOf(id));
                Common::DiagTrace::Emit(Common::DiagTrace::Ev::RTBind, frame_tick,
                                        Common::DiagTrace::tl_draw_seq,
                                        DiagTraceObjOf(slot_image_views[id].image_id), rt, 0,
                                        diag_rt_tag);
            }
            if (const ImageViewId id = render_targets.depth_buffer_id) {
                char diag_rt_tag[40];
                std::snprintf(diag_rt_tag, sizeof(diag_rt_tag), "v%u g%llu", id.index,
                              (unsigned long long)DiagViewGenOf(id));
                Common::DiagTrace::Emit(Common::DiagTrace::Ev::RTBind, frame_tick,
                                        Common::DiagTrace::tl_draw_seq,
                                        DiagTraceObjOf(slot_image_views[id].image_id), 0, 1,
                                        diag_rt_tag);
            }
        });
    }
    if (last_framebuffer_id && last_framebuffer_serial == render_targets_serial) {
        return &slot_framebuffers[last_framebuffer_id];
    }
    const FramebufferId framebuffer_id = GetFramebufferId(render_targets);
    last_framebuffer_id = framebuffer_id;
    last_framebuffer_serial = render_targets_serial;
    return &slot_framebuffers[framebuffer_id];
}

template <class P>
ImageViewId TextureCache<P>::VisitImageView(u32 index, bool compute) {
    auto& table = compute ? channel_state->compute_image_table : channel_state->graphics_image_table;
    if (index > table.current_limit) {
        LOG_DEBUG(HW_GPU, "Invalid image view index={}", index);
        return NULL_IMAGE_VIEW_ID;
    }
    auto const map_index = index | (compute ? Common::SlotId::TAGGED_VALUE : 0);
    // Is new (on the tegra engine side)?
    auto const [descriptor, is_new] = table.Read(*gpu_memory, index);
    if (!is_new) {
        auto const it = channel_state->image_view_ids.find(map_index);
        // A view of a dormant image is resolved again below: that revives the image (or
        // builds the one that owns the address now).
        if (!IsDormantView(it->second)) {
            if (it->second != NULL_IMAGE_VIEW_ID)
                PrepareImageView(it->second, false, false);
            return it->second;
        }
    }
    if (IsValidEntry(*gpu_memory, descriptor)) {
        // Is new (registered view) on the texture cache side?
        const auto [pair, is_new_tc] = channel_state->image_views.try_emplace(descriptor);
        if (is_new_tc || IsDormantView(pair->second))
            pair->second = CreateImageView(descriptor);
        PrepareImageView(pair->second, false, false);
        channel_state->image_view_ids.insert_or_assign(map_index, pair->second);
        return pair->second;
    }
    channel_state->image_view_ids.insert_or_assign(map_index, NULL_IMAGE_VIEW_ID);
    return NULL_IMAGE_VIEW_ID;
}

template <class P>
FramebufferId TextureCache<P>::GetFramebufferId(const RenderTargets& key) {
    const auto [pair, is_new] = framebuffers.try_emplace(key);
    FramebufferId& framebuffer_id = pair->second;
    if (!is_new) {
        return framebuffer_id;
    }
    std::array<ImageView*, NUM_RT> color_buffers;
    std::ranges::transform(key.color_buffer_ids, color_buffers.begin(), [this](ImageViewId id) {
        return id ? &slot_image_views[id] : nullptr;
    });
    ImageView* const depth_buffer = key.depth_buffer_id ? &slot_image_views[key.depth_buffer_id] : nullptr;
    runtime.FlushDeferredClear();
    framebuffer_id = slot_framebuffers.insert(runtime, color_buffers, depth_buffer, key);
    return framebuffer_id;
}

template <class P>
void TextureCache<P>::WriteMemory(DAddr cpu_addr, size_t size) {
    ForEachImageInRegion(cpu_addr, size, [this, cpu_addr, size](ImageId image_id, Image& image) {
        if (True(image.flags & ImageFlagBits::UnwrittenSparse)) {
            // Written at last: a backed image replaces it where it is used next.
            image.flags |= ImageFlagBits::CpuModified | ImageFlagBits::Remapped;
            return;
        }
        if (UploadsByLayer(image)) {
            MarkLayersWritten(image, cpu_addr, size);
            return;
        }
        if (True(image.flags & ImageFlagBits::CpuModified)) {
            return;
        }
        image.flags |= ImageFlagBits::CpuModified;
        if (True(image.flags & ImageFlagBits::Tracked)) {
            UntrackImage(image, image_id);
        }
    });
}

template <class P>
bool TextureCache<P>::IsDownloadable(const ImageBase& image) const noexcept {
    if (!image.IsSafeGpuCopy() || True(image.flags & (ImageFlagBits::Unbacked | ImageFlagBits::TrimmedLevels))) {
        return false;
    }
    // Switch 2 GPU-only regions (nx2:gpu 214) are never mapped by the client: a write-back would
    // only commit hundreds of MB of host RAM per render target.
    if (Service::Nvidia::Devices::IsNx2GpuOnlyAddress(image.gpu_addr)) {
        return false;
    }
    if (image.info.num_samples == 1) {
        return true;
    }
    if constexpr (P::HAS_MSAA_DOWNLOADS) {
        return runtime.CanDownloadMsaa(image.info);
    } else {
        return false;
    }
}

// EDEN_NX2_IMAGE_DUMP=<hex gpu address>[,<hex address>...] with EDEN_NX2_IMAGE_DUMP_DIR=<dir>:
// every EDEN_NX2_IMAGE_DUMP_EVERY frames (300), read back the images starting at those addresses
// (DumpImage), up to EDEN_NX2_IMAGE_DUMP_MAX (100) files.
template <class P>
void TextureCache<P>::DumpRequestedImages() {
    static const char* const spec = std::getenv("EDEN_NX2_IMAGE_DUMP");
    if (spec == nullptr || std::getenv("EDEN_NX2_IMAGE_DUMP_DIR") == nullptr) {
        return;
    }
    static const u64 interval = [] {
        const char* const every = std::getenv("EDEN_NX2_IMAGE_DUMP_EVERY");
        return every != nullptr ? (std::max)(std::strtoull(every, nullptr, 10), 1ull) : 300ull;
    }();
    static u64 frames{};
    if (++frames % interval != 0) {
        return;
    }
    static const std::vector<GPUVAddr> addresses = [] {
        std::vector<GPUVAddr> result;
        for (const char* cursor = spec; *cursor != '\0';) {
            char* end{};
            result.push_back(std::strtoull(cursor, &end, 16));
            cursor = *end == ',' ? end + 1 : end;
            if (end == cursor && *cursor != '\0') {
                break;
            }
        }
        return result;
    }();
    for (const auto [image_id, image] : slot_images) {
        if (std::ranges::find(addresses, image->gpu_addr) != addresses.end()) {
            DumpImage(image_id, "image");
        }
    }
}

// Read an image back and write it raw (host format, linear, every level and layer) as
// <EDEN_NX2_IMAGE_DUMP_DIR>/<tag>-<address>-<image id>-<frame>.bin with a .txt describing the
// layout, from frame EDEN_NX2_IMAGE_DUMP_AFTER on. Returns false once EDEN_NX2_IMAGE_DUMP_MAX
// (100) files were written.
template <class P>
bool TextureCache<P>::DumpImage(ImageId image_id, std::string_view tag) {
    static const char* const dir = [] {
        const char* const value = std::getenv("EDEN_NX2_IMAGE_DUMP_DIR");
        if (value != nullptr) {
            std::error_code ec;
            std::filesystem::create_directories(value, ec);
        }
        return value;
    }();
    static const u32 max_dumps = [] {
        const char* const max = std::getenv("EDEN_NX2_IMAGE_DUMP_MAX");
        return max != nullptr ? static_cast<u32>(std::strtoul(max, nullptr, 10)) : 100u;
    }();
    // EDEN_NX2_IMAGE_DUMP_AFTER=<frame>: nothing before that frame (the title screen binds the
    // same programs with other textures).
    static const u64 first_frame = [] {
        const char* const after = std::getenv("EDEN_NX2_IMAGE_DUMP_AFTER");
        return after != nullptr ? std::strtoull(after, nullptr, 10) : 0ull;
    }();
    static u32 dumps{};
    if (dir == nullptr || dumps >= max_dumps) {
        return false;
    }
    if (frame_tick < first_frame) {
        return true;
    }
    Image& image = slot_images[image_id];
    if (True(image.flags & (ImageFlagBits::Unbacked | ImageFlagBits::Remapped))) {
        return true;
    }
    boost::container::small_vector<BufferImageCopy, 16> copies;
    size_t total_bytes = image.unswizzled_size_bytes;
    const char* host_format = "guest";
    if (True(image.flags & ImageFlagBits::Converted)) {
        // GPU-decoded ASTC is BC1 (opaque), BC3 or RGBA8 on the host: lay the levels out
        // tightly in that format instead of the guest's.
        const bool recompressed = Settings::values.astc_recompression.GetValue() !=
                                  Settings::AstcRecompression::Uncompressed;
        const bool bc1 = recompressed && (image.info.astc_opaque ||
                                          Settings::values.astc_recompression.GetValue() ==
                                              Settings::AstcRecompression::Bc1);
        const u32 block = recompressed ? 4 : 1;
        const size_t block_bytes = !recompressed ? 4 : bc1 ? 8 : 16;
        host_format = !recompressed ? "rgba8" : bc1 ? "bc1" : "bc3";
        size_t offset = 0;
        for (s32 level = 0; level < image.info.resources.levels; ++level) {
            const Extent3D size = MipSize(image.info.size, level);
            const size_t bytes = size_t{Common::DivCeil(size.width, block)} *
                                 Common::DivCeil(size.height, block) * size.depth * block_bytes *
                                 image.info.resources.layers;
            copies.push_back(BufferImageCopy{
                .buffer_offset = offset,
                .buffer_size = bytes,
                .buffer_row_length = 0,
                .buffer_image_height = 0,
                .image_subresource = {.base_level = level,
                                      .base_layer = 0,
                                      .num_layers = image.info.resources.layers},
                .image_offset = {},
                .image_extent = size,
            });
            offset += Common::AlignUp(bytes, size_t{64});
        }
        total_bytes = offset;
    } else {
        copies = FullDownloadCopies(image.info);
    }
    auto map = runtime.DownloadStagingBuffer(total_bytes);
    image.DownloadMemory(map, copies);
    runtime.Finish();
    const std::string base = fmt::format("{}/{}-{:x}-{}-{}", dir, tag, image.gpu_addr,
                                         image_id.index, frame_tick);
    if (std::FILE* file = std::fopen((base + ".bin").c_str(), "wb")) {
        std::fwrite(map.mapped_span.data(), 1, total_bytes, file);
        std::fclose(file);
    }
    if (std::FILE* file = std::fopen((base + ".txt").c_str(), "w")) {
        fmt::print(file, "{}x{}x{} layers {} levels {} format {} flags {:#x} id {} host {}\n",
                   image.info.size.width, image.info.size.height, image.info.size.depth,
                   image.info.resources.layers, image.info.resources.levels,
                   static_cast<u32>(image.info.format), static_cast<u32>(image.flags),
                   image_id.index, host_format);
        for (const auto& copy : copies) {
            fmt::print(file, "level {} offset {} size {} {}x{}x{} layers {} row {} rows {}\n",
                       copy.image_subresource.base_level, copy.buffer_offset, copy.buffer_size,
                       copy.image_extent.width, copy.image_extent.height,
                       copy.image_extent.depth, copy.image_subresource.num_layers,
                       copy.buffer_row_length, copy.buffer_image_height);
        }
        std::fclose(file);
    }
    // The guest's own data as well (linear, guest format), to check what was decoded.
    if (gpu_memory != nullptr && image.guest_size_bytes <= (64_MiB)) {
        std::vector<u8> swizzled(image.guest_size_bytes);
        gpu_memory->ReadBlockUnsafe(image.gpu_addr, swizzled.data(), swizzled.size());
        std::vector<u8> linear(image.unswizzled_size_bytes);
        const auto guest_copies =
            UnswizzleImage(*gpu_memory, image.gpu_addr, image.info, swizzled, linear);
        if (std::FILE* file = std::fopen((base + ".guest.bin").c_str(), "wb")) {
            std::fwrite(linear.data(), 1, linear.size(), file);
            std::fclose(file);
        }
        if (std::FILE* file = std::fopen((base + ".guest.txt").c_str(), "w")) {
            for (const auto& copy : guest_copies) {
                fmt::print(file, "level {} offset {} size {} {}x{}x{} layers {} row {} rows {}\n",
                           copy.image_subresource.base_level, copy.buffer_offset,
                           copy.buffer_size, copy.image_extent.width, copy.image_extent.height,
                           copy.image_extent.depth, copy.image_subresource.num_layers,
                           copy.buffer_row_length, copy.buffer_image_height);
            }
            std::fclose(file);
        }
    }
    ++dumps;
    LOG_INFO(HW_GPU, "nx2 image dump: {}.bin ({} bytes)", base, total_bytes);
    return true;
}

template <class P>
void TextureCache<P>::DownloadMemory(DAddr cpu_addr, size_t size) {
    ++download_requests;
    boost::container::small_vector<ImageId, 16> images;
    ForEachImageInRegion(cpu_addr, size, [this, &images](ImageId image_id, ImageBase& image) {
        if (!IsDownloadable(image)) {
            return;
        }
        image.flags &= ~ImageFlagBits::GpuModified;
        images.push_back(image_id);
    });
    if (images.empty()) {
        return;
    }
    std::ranges::sort(images, [this](ImageId lhs, ImageId rhs) {
        return slot_images[lhs].modification_tick < slot_images[rhs].modification_tick;
    });
    for (const ImageId image_id : images) {
        Image& image = slot_images[image_id];
        if (Tegra::Nx2TracedDestination(image.gpu_addr, image.guest_size_bytes)) {
            static u32 traced{};
            if (traced++ < 400) {
                LOG_INFO(HW_GPU, "nx2 image in traced range written back to guest memory: {:#x} {}x{} format {}",
                         image.gpu_addr, image.info.size.width, image.info.size.height,
                         static_cast<u32>(image.info.format));
            }
        }
        ++images_written_back;
        auto map = runtime.DownloadStagingBuffer(image.unswizzled_size_bytes);
        const auto copies = FullDownloadCopies(image.info);
        image.DownloadMemory(map, copies);
        runtime.Finish();
        SwizzleImage(*gpu_memory, image.gpu_addr, image.info, copies, map.mapped_span,
                     swizzle_data_buffer);
    }
}

template <class P>
std::optional<VideoCore::RasterizerDownloadArea> TextureCache<P>::GetFlushArea(DAddr cpu_addr,
                                                                               u64 size) {
    std::optional<VideoCore::RasterizerDownloadArea> area{};
    ForEachImageInRegion(cpu_addr, size, [&](ImageId, ImageBase& image) {
        if (False(image.flags & ImageFlagBits::GpuModified)) {
            return;
        }
        if (!area) {
            area.emplace();
            area->start_address = cpu_addr;
            area->end_address = cpu_addr + size;
            area->preemtive = true;
        }
        area->start_address = (std::min)(area->start_address, image.cpu_addr);
        area->end_address = (std::max)(area->end_address, image.cpu_addr_end);
        for (auto image_view_id : image.image_view_ids) {
            auto& image_view = slot_image_views[image_view_id];
            image_view.flags |= ImageViewFlagBits::PreemtiveDownload;
        }
        area->preemtive &= image.info.forced_flushed;
        image.info.forced_flushed = true;
    });
    return area;
}

template <class P>
void TextureCache<P>::UnmapMemory(DAddr cpu_addr, size_t size) {
    boost::container::small_vector<ImageId, 16> deleted_images;
    ForEachImageInRegion(cpu_addr, size, [&](ImageId id, Image&) { deleted_images.push_back(id); });
    for (const ImageId id : deleted_images) {
        Image& image = slot_images[id];
        if (True(image.flags & ImageFlagBits::Tracked)) {
            UntrackImage(image, id);
        }
        UnregisterImage(id);
        DeleteImage(id, false, "unmap");
    }
}

template <class P>
void TextureCache<P>::UnmapGPUMemory(size_t as_id, GPUVAddr gpu_addr, size_t size) {
    boost::container::small_vector<ImageId, 16> deleted_images;
    ForEachImageInRegionGPU(as_id, gpu_addr, size,
                            [&](ImageId id, Image&) { deleted_images.push_back(id); });
    for (const ImageId id : deleted_images) {
        Image& image = slot_images[id];
        if (False(image.flags & ImageFlagBits::CpuModified)) {
            image.flags |= ImageFlagBits::CpuModified;
            if (True(image.flags & ImageFlagBits::Tracked)) {
                UntrackImage(image, id);
            }
        }
        image.dirty_layers.clear(); // all of it
        if (True(image.flags & ImageFlagBits::Remapped)) {
            continue;
        }
        image.flags |= ImageFlagBits::Remapped;
    }
}

template <class P>
bool TextureCache<P>::BlitImage(const Tegra::Engines::Fermi2D::Surface& dst,
                                const Tegra::Engines::Fermi2D::Surface& src,
                                const Tegra::Engines::Fermi2D::Config& copy) {
    const auto result = GetBlitImages(dst, src, copy);
    if (!result) {
        return false;
    }
    const BlitImages images = *result;
    const ImageId dst_id = images.dst_id;
    const ImageId src_id = images.src_id;

    PrepareImage(src_id, false, false);
    PrepareImage(dst_id, true, false);

    Image& dst_image = slot_images[dst_id];
    Image& src_image = slot_images[src_id];
    if (True(dst_image.flags & ImageFlagBits::Unbacked)) {
        static u32 logged{};
        if (logged++ < 64) {
            LOG_INFO(HW_GPU, "2D blit into unbacked image {:#x}: dst {:#x} {}x{} from {:#x} {}x{}",
                     dst_image.gpu_addr, dst.Address(), dst.width, dst.height, src.Address(),
                     src.width, src.height);
        }
    }
    bool is_src_rescaled = True(src_image.flags & ImageFlagBits::Rescaled);
    bool is_dst_rescaled = True(dst_image.flags & ImageFlagBits::Rescaled);

    const bool is_resolve = src_image.info.num_samples != 1 && dst_image.info.num_samples == 1;
    if (is_src_rescaled != is_dst_rescaled) {
        if (ImageCanRescale(src_image)) {
            ScaleUp(src_image);
            is_src_rescaled = True(src_image.flags & ImageFlagBits::Rescaled);
            if (is_resolve) {
                dst_image.info.rescaleable = true;
                for (const auto& alias : dst_image.aliased_images) {
                    Image& other_image = slot_images[alias.id];
                    other_image.info.rescaleable = true;
                }
            }
        }
        if (ImageCanRescale(dst_image)) {
            ScaleUp(dst_image);
            is_dst_rescaled = True(dst_image.flags & ImageFlagBits::Rescaled);
        }
    }
    if (is_resolve && (is_src_rescaled != is_dst_rescaled)) {
        // A resolve requires both images to be the same dimensions. Resize down if needed.
        ScaleDown(src_image);
        ScaleDown(dst_image);
        is_src_rescaled = True(src_image.flags & ImageFlagBits::Rescaled);
        is_dst_rescaled = True(dst_image.flags & ImageFlagBits::Rescaled);
    }
    const auto& resolution = Settings::values.resolution_info;
    const auto scale_region = [&](Region2D& region) {
        region.start.x = resolution.ScaleUp(region.start.x);
        region.start.y = resolution.ScaleUp(region.start.y);
        region.end.x = resolution.ScaleUp(region.end.x);
        region.end.y = resolution.ScaleUp(region.end.y);
    };

    // TODO: Deduplicate
    const std::optional src_base = src_image.TryFindBase(src.Address());
    const SubresourceRange src_range{.base = src_base.value(), .extent = {1, 1}};
    const ImageViewInfo src_view_info(ImageViewType::e2D, images.src_format, src_range);
    const auto [src_framebuffer_id, src_view_id] = RenderTargetFromImage(src_id, src_view_info);
    const auto [src_samples_x, src_samples_y] = SamplesLog2(src_image.info.num_samples);
    Region2D src_region{
        Offset2D{.x = copy.src_x0 >> src_samples_x, .y = copy.src_y0 >> src_samples_y},
        Offset2D{.x = copy.src_x1 >> src_samples_x, .y = copy.src_y1 >> src_samples_y},
    };
    if (is_src_rescaled) {
        scale_region(src_region);
    }

    const std::optional dst_base = dst_image.TryFindBase(dst.Address());
    const SubresourceRange dst_range{.base = dst_base.value(), .extent = {1, 1}};
    const ImageViewInfo dst_view_info(ImageViewType::e2D, images.dst_format, dst_range);
    const auto [dst_framebuffer_id, dst_view_id] = RenderTargetFromImage(dst_id, dst_view_info);
    const auto [dst_samples_x, dst_samples_y] = SamplesLog2(dst_image.info.num_samples);
    Region2D dst_region{
        Offset2D{.x = copy.dst_x0 >> dst_samples_x, .y = copy.dst_y0 >> dst_samples_y},
        Offset2D{.x = copy.dst_x1 >> dst_samples_x, .y = copy.dst_y1 >> dst_samples_y},
    };
    if (is_dst_rescaled) {
        scale_region(dst_region);
    }

    // F1 (CopyDep): the blit is issued below in either constexpr branch; both
    // ends have real texture-cache identity here. d = 0 (no count at this site).
    Common::DiagTrace::Emit(Common::DiagTrace::Ev::CopyDep, frame_tick,
                            DiagTraceObjOf(src_id), DiagTraceObjOf(dst_id), 0, 0, "blit");
    // Always call this after src_framebuffer_id was queried, as the address might be invalidated.
    Framebuffer* const dst_framebuffer = &slot_framebuffers[dst_framebuffer_id];
    if constexpr (FRAMEBUFFER_BLITS) {
        // OpenGL blits from framebuffers, not images
        Framebuffer* const src_framebuffer = &slot_framebuffers[src_framebuffer_id];
        runtime.BlitFramebuffer(dst_framebuffer, src_framebuffer, dst_region, src_region,
                                copy.filter, copy.operation);
    } else {
        // Vulkan can blit images, but it lacks format reinterpretations
        // Provide a framebuffer in case it's necessary
        ImageView& dst_view = slot_image_views[dst_view_id];
        ImageView& src_view = slot_image_views[src_view_id];
        runtime.BlitImage(dst_framebuffer, dst_view, src_view, dst_region, src_region, copy.filter,
                          copy.operation);
    }
    return true;
}

template <class P>
std::pair<typename P::ImageView*, bool> TextureCache<P>::TryFindFramebufferImageView(
    const Tegra::FramebufferConfig& config, DAddr cpu_addr) {
    // TODO: Properly implement this
    const auto it = page_table.find(cpu_addr >> YUZU_PAGEBITS);
    if (it == page_table.end()) {
        return {};
    }
    const auto view_format = [&]() {
        switch (config.pixel_format) {
        case Service::android::PixelFormat::Rgb565:
            return PixelFormat::R5G6B5_UNORM;
        case Service::android::PixelFormat::Bgra8888:
            return PixelFormat::B8G8R8A8_UNORM;
        default:
            return PixelFormat::A8B8G8R8_UNORM;
        }
    }();

    const auto& image_map_ids = it->second;
    boost::container::small_vector<ImageId, 4> valid_image_ids;
    for (const ImageMapId map_id : image_map_ids) {
        const ImageMapView& map = slot_map_views[map_id];
        const ImageBase& image = slot_images[map.image_id];
        if (image.cpu_addr != cpu_addr) {
            continue;
        }
        if (image.image_view_ids.empty()) {
            continue;
        }
        if (!VideoCore::Surface::IsViewCompatible(image.info.format, view_format, false, true)) {
            static u32 rejected_alias_count{};
            if (std::getenv("EDEN_NATIVE_DRAW_DIAGNOSTICS") && rejected_alias_count++ < 8) {
                LOG_WARNING(HW_GPU, "Ignoring incompatible framebuffer alias: address={:#x}, image={}, view={}",
                            cpu_addr, image.info.format, view_format);
            }
            continue;
        }
        valid_image_ids.push_back(map.image_id);
    }

    const auto GetImageViewForFramebuffer = [&](ImageId image_id) {
        ImageViewInfo info{ImageViewType::e2D, view_format};
        if (config.blending == Tegra::BlendMode::Opaque) {
            info.x_source = static_cast<u8>(SwizzleSource::R);
            info.y_source = static_cast<u8>(SwizzleSource::G);
            info.z_source = static_cast<u8>(SwizzleSource::B);
            info.w_source = static_cast<u8>(SwizzleSource::OneFloat);
        }
        return std::make_pair(&slot_image_views[FindOrEmplaceImageView(image_id, info)],
                              slot_images[image_id].IsRescaled());
    };

    if (valid_image_ids.size() == 1) [[likely]] {
        return GetImageViewForFramebuffer(valid_image_ids.front());
    }

    if (valid_image_ids.size() > 0) [[unlikely]] {
        auto most_recent = std::ranges::max_element(valid_image_ids, [&](auto a, auto b) {
            return slot_images[a].modification_tick < slot_images[b].modification_tick;
        });
        return GetImageViewForFramebuffer(*most_recent);
    }

    return {};
}

template <class P>
bool TextureCache<P>::HasUncommittedFlushes() const noexcept {
    return !uncommitted_downloads.empty();
}

template <class P>
bool TextureCache<P>::HasAsyncFlushWork() const noexcept {
    return !uncommitted_downloads.empty() || !uncommitted_async_buffers.empty();
}

template <class P>
bool TextureCache<P>::ShouldWaitAsyncFlushes() const noexcept {
    return !committed_downloads.empty() && !committed_downloads.front().empty();
}

template <class P>
void TextureCache<P>::CommitAsyncFlushes() {
    // This is intentionally passing the value by copy
    if constexpr (IMPLEMENTS_ASYNC_DOWNLOADS) {
        auto& download_ids = uncommitted_downloads;
        const size_t redundant_images = DeduplicatePendingImageDownloads(download_ids, slot_images);
        if (download_ids.empty()) {
            committed_downloads.emplace_back(std::move(uncommitted_downloads));
            uncommitted_downloads.clear();
            async_buffers.emplace_back(std::move(uncommitted_async_buffers));
            uncommitted_async_buffers.clear();
            return;
        }
        size_t total_size_bytes = 0;
        size_t last_async_buffer_id = uncommitted_async_buffers.size();
        bool any_none_dma = false;
        for (PendingDownload& download_info : download_ids) {
            if (download_info.is_swizzle) {
                total_size_bytes +=
                    Common::AlignUp(download_info.GetImage(slot_images).unswizzled_size_bytes, 64);
                any_none_dma = true;
                download_info.async_buffer_id = last_async_buffer_id;
            }
        }

        if (any_none_dma) {
            if (std::getenv("EDEN_NATIVE_DRAW_DIAGNOSTICS") && total_size_bytes >= 64_MiB) {
                LOG_WARNING(HW_GPU, "Large image readback batch: bytes={}, entries={}, duplicates_removed={}, submitted_batches={}, retained_buffers={}",
                            total_size_bytes, download_ids.size(), redundant_images,
                            committed_downloads.size(), async_buffers_death_ring.size());
                size_t traced_images = 0;
                for (const auto& download : download_ids) {
                    if (!download.is_swizzle || traced_images++ >= 8) continue;
                    const auto& image = download.GetImage(slot_images);
                    LOG_WARNING(HW_GPU, "Readback image: gpu={:#x}, cpu={:#x}, format={}, size={}x{}x{}, levels={}, layers={}, layer_stride={}, guest_bytes={}, download_bytes={}",
                                image.gpu_addr, image.cpu_addr, image.info.format,
                                image.info.size.width, image.info.size.height, image.info.size.depth,
                                image.info.resources.levels, image.info.resources.layers,
                                image.info.layer_stride, image.guest_size_bytes, image.unswizzled_size_bytes);
                }
            }
            auto download_map = runtime.DownloadStagingBuffer(total_size_bytes, true);
            for (const PendingDownload& download_info : download_ids) {
                if (download_info.is_swizzle) {
                    Image& image = download_info.GetImage(slot_images);
                    const auto copies = FullDownloadCopies(image.info);
                    image.DownloadMemory(download_map, copies);
                    download_map.offset += Common::AlignUp(image.unswizzled_size_bytes, 64);
                }
            }
            uncommitted_async_buffers.emplace_back(download_map);
        }

        async_buffers.emplace_back(std::move(uncommitted_async_buffers));
        uncommitted_async_buffers.clear();
    }
    committed_downloads.emplace_back(std::move(uncommitted_downloads));
    uncommitted_downloads.clear();
}

template <class P>
void TextureCache<P>::PopAsyncFlushes() {
    if (committed_downloads.empty()) {
        return;
    }
    if constexpr (IMPLEMENTS_ASYNC_DOWNLOADS) {
        const auto& download_ids = committed_downloads.front();
        if (download_ids.empty()) {
            committed_downloads.pop_front();
            async_buffers.pop_front();
            return;
        }
        auto download_map = std::move(async_buffers.front());
        for (size_t i = download_ids.size(); i > 0; i--) {
            auto& download_info = download_ids[i - 1];
            auto& download_buffer = download_map[download_info.async_buffer_id];
            if (download_info.is_swizzle) {
                const ImageBase& image = download_info.GetImage(slot_images);
                const auto copies = FullDownloadCopies(image.info);
                download_buffer.offset -= Common::AlignUp(image.unswizzled_size_bytes, 64);
                std::span<u8> download_span =
                    download_buffer.mapped_span.subspan(download_buffer.offset);
                SwizzleImage(*gpu_memory, image.gpu_addr, image.info, copies, download_span, swizzle_data_buffer);
            } else {
                const BufferDownload& buffer_info = slot_buffer_downloads[download_info.object_id];
                std::span<u8> download_span =
                    download_buffer.mapped_span.subspan(download_buffer.offset);
                gpu_memory->WriteBlockUnsafe(buffer_info.address, download_span.data(),
                                             buffer_info.size);
                slot_buffer_downloads.erase(download_info.object_id);
            }
        }
        for (auto& download_buffer : download_map) {
            async_buffers_death_ring.emplace_back(download_buffer);
        }
        committed_downloads.pop_front();
        async_buffers.pop_front();
    } else {
        const auto& download_ids = committed_downloads.front();
        if (download_ids.empty()) {
            committed_downloads.pop_front();
            return;
        }
        size_t total_size_bytes = 0;
        for (const PendingDownload& download_info : download_ids) {
            if (download_info.is_swizzle) {
                total_size_bytes += download_info.GetImage(slot_images).unswizzled_size_bytes;
            }
        }
        auto download_map = runtime.DownloadStagingBuffer(total_size_bytes);
        const size_t original_offset = download_map.offset;
        for (const PendingDownload& download_info : download_ids) {
            if (!download_info.is_swizzle) {
                continue;
            }
            Image& image = download_info.GetImage(slot_images);
            const auto copies = FullDownloadCopies(image.info);
            image.DownloadMemory(download_map, copies);
            download_map.offset += image.unswizzled_size_bytes;
        }
        // Wait for downloads to finish
        runtime.Finish();
        download_map.offset = original_offset;
        std::span<u8> download_span = download_map.mapped_span;
        for (const PendingDownload& download_info : download_ids) {
            if (!download_info.is_swizzle) {
                continue;
            }
            const ImageBase& image = download_info.GetImage(slot_images);
            const auto copies = FullDownloadCopies(image.info);
            SwizzleImage(*gpu_memory, image.gpu_addr, image.info, copies, download_span, swizzle_data_buffer);
            download_map.offset += image.unswizzled_size_bytes;
            download_span = download_span.subspan(image.unswizzled_size_bytes);
        }
        committed_downloads.pop_front();
    }
}

template <class P>
ImageId TextureCache<P>::DmaImageId(const Tegra::DMA::ImageOperand& operand, bool is_upload) {
    const ImageInfo dst_info(operand);
    const ImageId dst_id = FindDMAImage(dst_info, operand.address);
    if (!dst_id) {
        return NULL_IMAGE_ID;
    }
    auto& image = slot_images[dst_id];
    if (False(image.flags & ImageFlagBits::GpuModified)) {
        // No need to waste time on an image that's synced with guest
        return NULL_IMAGE_ID;
    }
    if (image.info.type == ImageType::e3D) {
        // Don't accelerate 3D images.
        return NULL_IMAGE_ID;
    }
    if (!is_upload && !image.info.dma_downloaded) {
        // Force a full sync.
        image.info.dma_downloaded = true;
        return NULL_IMAGE_ID;
    }
    const auto base = image.TryFindBase(operand.address);
    if (!base) {
        return NULL_IMAGE_ID;
    }
    return dst_id;
}

template <class P>
bool TextureCache<P>::IsRescaling() const noexcept {
    return is_rescaling;
}

template <class P>
bool TextureCache<P>::IsRescaling(const ImageViewBase& image_view) const noexcept {
    if (image_view.type == ImageViewType::Buffer) {
        return false;
    }
    const ImageBase& image = slot_images[image_view.image_id];
    return True(image.flags & ImageFlagBits::Rescaled);
}

template <class P>
bool TextureCache<P>::IsRegionGpuModified(DAddr addr, size_t size) {
    bool is_modified = false;
    ForEachImageInRegion(addr, size, [&is_modified](ImageId, ImageBase& image) {
        if (False(image.flags & ImageFlagBits::GpuModified)) {
            return false;
        }
        is_modified = true;
        return true;
    });
    return is_modified;
}

template <class P>
std::pair<typename TextureCache<P>::Image*, BufferImageCopy> TextureCache<P>::DmaBufferImageCopy(
    const Tegra::DMA::ImageCopy& copy_info, const Tegra::DMA::BufferOperand& buffer_operand,
    const Tegra::DMA::ImageOperand& image_operand, ImageId image_id, bool modifies_image) {
    const auto [level, base] = PrepareDmaImage(image_id, image_operand.address, modifies_image);
    auto* image = &slot_images[image_id];
    const u32 buffer_size = static_cast<u32>(buffer_operand.pitch * buffer_operand.height);
    const u32 bpp = VideoCore::Surface::BytesPerBlock(image->info.format);
    const auto convert = [old_bpp = image_operand.bytes_per_pixel, bpp](u32 value) {
        return (old_bpp * value) / bpp;
    };
    const u32 base_x = convert(image_operand.params.origin.x.Value());
    const u32 base_y = image_operand.params.origin.y.Value();
    const u32 length_x = convert(copy_info.length_x);
    const u32 length_y = copy_info.length_y;

    const BufferImageCopy copy{
        .buffer_offset = 0,
        .buffer_size = buffer_size,
        .buffer_row_length = convert(buffer_operand.pitch),
        .buffer_image_height = buffer_operand.height,
        .image_subresource =
            {
                .base_level = static_cast<s32>(level),
                .base_layer = static_cast<s32>(base),
                .num_layers = 1,
            },
        .image_offset =
            {
                .x = static_cast<s32>(base_x),
                .y = static_cast<s32>(base_y),
                .z = 0,
            },
        .image_extent =
            {
                .width = length_x,
                .height = length_y,
                .depth = 1,
            },
    };
    return {image, copy};
}

template <class P>
void TextureCache<P>::DownloadImageIntoBuffer(typename TextureCache<P>::Image* image,
                                              typename TextureCache<P>::BufferType buffer,
                                              size_t buffer_offset,
                                              std::span<const VideoCommon::BufferImageCopy> copies,
                                              GPUVAddr address, size_t size) {
    if constexpr (IMPLEMENTS_ASYNC_DOWNLOADS) {
        const BufferDownload new_buffer_download{address, size};
        auto slot = slot_buffer_downloads.insert(new_buffer_download);
        const PendingDownload new_download{false, uncommitted_async_buffers.size(), slot};
        uncommitted_downloads.emplace_back(new_download);
        auto download_map = runtime.DownloadStagingBuffer(size, true);
        uncommitted_async_buffers.emplace_back(download_map);
        std::array buffers{
            buffer,
            download_map.buffer,
        };
        std::array<size_t, 2> buffer_offsets{
            buffer_offset,
            download_map.offset,
        };
        image->DownloadMemory(buffers, buffer_offsets, copies);
    } else {
        image->DownloadMemory(buffer, buffer_offset, copies);
    }
}

template <class P>
void TextureCache<P>::RefreshContents(Image& image, ImageId image_id) {
    if (False(image.flags & ImageFlagBits::CpuModified)) {
        // Only upload modified images
        return;
    }
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::TextureUpload};

    image.flags &= ~ImageFlagBits::CpuModified;

    // Images uploaded by layer stay tracked while some of their layers are dirty.
    if (False(image.flags & ImageFlagBits::Tracked)) {
        TrackImage(image, image_id);
    }

    if (True(image.flags & ImageFlagBits::SparseLayers)) {
        RefreshSparseLayers(image);
        return;
    }
    if (!image.dirty_layers.empty()) {
        u64 dirty_count = 0; // diag only: counted BEFORE UploadWrittenLayers clears the vector
        for (const u64 word : image.dirty_layers) {
            dirty_count += static_cast<u64>(std::popcount(word));
        }
        if (UploadWrittenLayers(image)) {
            Common::DiagTrace::Emit(Common::DiagTrace::Ev::ImageUpload, frame_tick,
                                    DiagTraceObjOf(image_id), 0, dirty_count, 0, "by-layer");
            runtime.InsertUploadMemoryBarrier();
            return;
        }
    }
    image.dirty_layers.clear();

    if (True(image.flags & ImageFlagBits::Unbacked)) {
        // Nothing to upload into (and staging it could take tens of GB).
        runtime.TransitionImageLayout(image);
        return;
    }

    if (image.info.num_samples > 1 && !runtime.CanUploadMSAA()) {
        LOG_WARNING(HW_GPU, "MSAA image uploads are not implemented");
        runtime.TransitionImageLayout(image);
        return;
    }
    if (True(image.flags & ImageFlagBits::AsynchronousDecode)) {
        QueueAsyncDecode(image, image_id);
        return;
    }

    const bool gpu_unswizzle_enabled = Settings::values.gpu_unswizzle_enabled.GetValue();

    if (gpu_unswizzle_enabled &&
        IsPixelFormatBCn(image.info.format) &&
        image.info.type == ImageType::e3D &&
        image.info.resources.levels == 1 &&
        image.info.resources.layers == 1 &&
        MapSizeBytes(image) >= gpu_unswizzle_maxsize &&
        False(image.flags & ImageFlagBits::GpuModified)) {

        QueueAsyncUnswizzle(image, image_id);
        return;
    }
    ++images_uploaded;
    uploaded_bytes += image.guest_size_bytes;
    largest_upload = (std::max<u64>)(largest_upload, image.guest_size_bytes);
    auto staging = runtime.UploadStagingBuffer(MapSizeBytes(image));
    UploadImageContents(image, staging);
    Common::DiagTrace::Emit(Common::DiagTrace::Ev::ImageUpload, frame_tick,
                            DiagTraceObjOf(image_id), 0, 0, 0, "staging");
    runtime.InsertUploadMemoryBarrier();
}

template <class P>
bool TextureCache<P>::UploadsByLayer(const Image& image) const noexcept {
    // Title 1's terrain arrays (8192x8192x68 ASTC, 1.45 GB each) get a few layers streamed in at a
    // time; re-decoding all of them for each took a 2 GiB staging buffer and stalled the GPU.
    if constexpr (requires { &Runtime::AccelerateImageUploadLayers; }) {
        if (True(image.flags & ImageFlagBits::SparseLayers)) {
            return true;
        }
        return runtime.CanAccelerateImageUploadLayers(image) &&
               image.info.type == ImageType::e2D && image.info.resources.layers > 1 &&
               image.guest_size_bytes >= LayerUploadMinSize &&
               False(image.flags & (ImageFlagBits::Unbacked | ImageFlagBits::Sparse |
                                    ImageFlagBits::Remapped));
    } else {
        return false;
    }
}

template <class P>
void TextureCache<P>::MarkLayersWritten(Image& image, DAddr cpu_addr, size_t size) {
    if (True(image.flags & ImageFlagBits::CpuModified) && image.dirty_layers.empty()) {
        return; // all of it is dirty already
    }
    const u32 layers = static_cast<u32>(image.info.resources.layers);
    if (False(image.flags & ImageFlagBits::CpuModified)) {
        image.flags |= ImageFlagBits::CpuModified;
        image.dirty_layers.assign(Common::DivCeil(layers, 64U), 0);
    }
    const DAddr begin = (std::max<DAddr>)(cpu_addr, image.cpu_addr) - image.cpu_addr;
    const DAddr end = (std::min<DAddr>)(cpu_addr + size, image.cpu_addr + image.guest_size_bytes) -
                      image.cpu_addr;
    const u32 stride = image.info.layer_stride;
    if (end <= begin || stride == 0) {
        return;
    }
    const u32 last = (std::min)(static_cast<u32>((end - 1) / stride), layers - 1);
    for (u32 layer = static_cast<u32>(begin / stride); layer <= last; ++layer) {
        image.dirty_layers[layer / 64] |= u64{1} << (layer % 64);
    }
}

template <class P>
bool TextureCache<P>::UploadWrittenLayers(Image& image) {
    if constexpr (requires { &Runtime::AccelerateImageUploadLayers; }) {
        const u32 num_layers = static_cast<u32>(image.info.resources.layers);
        std::vector<u32> layers;
        for (u32 layer = 0; layer < num_layers; ++layer) {
            if ((image.dirty_layers[layer / 64] >> (layer % 64)) & 1) {
                layers.push_back(layer);
            }
        }
        image.dirty_layers.clear();
        if (layers.empty() || layers.size() == num_layers) {
            return false;
        }
        UploadLayers(image, layers);
        return true;
    } else {
        return false;
    }
}

template <class P>
void TextureCache<P>::RefreshSparseLayers(Image& image) {
    if constexpr (requires { &Image::BindLayers; }) {
        // The layers written since the last upload, or, for a new image, every layer any of whose
        // blocks has been written.
        const u32 num_layers = static_cast<u32>(image.info.resources.layers);
        const u64 stride = image.info.layer_stride;
        std::vector<u32> layers;
        for (u32 layer = 0; layer < num_layers; ++layer) {
            const bool written =
                image.dirty_layers.empty()
                    ? !Tegra::Nx2SparseUnwritten(image.gpu_addr + layer * stride, stride)
                    : ((image.dirty_layers[layer / 64] >> (layer % 64)) & 1) != 0;
            if (written) {
                layers.push_back(layer);
            }
        }
        image.dirty_layers.clear();
        const std::vector<u32> bound = layers.empty() ? std::vector<u32>{} : image.BindLayers(layers);
        if (bound.empty()) {
            runtime.TransitionImageLayout(image);
            return;
        }
        UploadLayers(image, bound);
        runtime.InsertUploadMemoryBarrier();
    }
}

template <class P>
void TextureCache<P>::UploadLayers(Image& image, std::span<const u32> layers) {
    if constexpr (requires { &Runtime::AccelerateImageUploadLayers; }) {
        if (!runtime.CanAccelerateImageUploadLayers(image)) {
            // Plain formats (and the generic mip-tail trim, whose UploadMemory drops the copies
            // into trimmed-away levels): each layer unswizzled on its own into its layer.
            ImageInfo layer_info = image.info;
            layer_info.resources.layers = 1;
            const size_t layer_size = CalculateUnswizzledSizeBytes(layer_info);
            auto staging = runtime.UploadStagingBuffer(layer_size * layers.size());
            std::vector<BufferImageCopy> copies;
            for (size_t i = 0; i < layers.size(); ++i) {
                const GPUVAddr layer_addr = image.gpu_addr + u64{layers[i]} * image.info.layer_stride;
                Tegra::Memory::GpuGuestMemory<u8, Tegra::Memory::GuestMemoryFlags::UnsafeRead>
                    swizzle_data(*gpu_memory, layer_addr, image.info.layer_stride,
                                 &swizzle_data_buffer);
                auto layer_copies =
                    UnswizzleImage(*gpu_memory, layer_addr, layer_info, swizzle_data,
                                   staging.mapped_span.subspan(i * layer_size, layer_size));
                for (BufferImageCopy copy : layer_copies) {
                    copy.buffer_offset += i * layer_size;
                    copy.image_subresource.base_layer = static_cast<s32>(layers[i]);
                    copies.push_back(copy);
                }
            }
            image.UploadMemory(staging, copies);
            ++images_uploaded;
            uploaded_bytes += u64{image.info.layer_stride} * layers.size();
            ++layer_uploads;
            return;
        }
        // Each layer's mip chain from the first level the host keeps on (as UploadImageContents).
        auto uploads = FullUploadSwizzles(image.info);
        const size_t level_offset = uploads[image.host_level_offset].buffer_offset;
        const size_t tail = image.info.layer_stride - level_offset;
        auto staging = runtime.UploadStagingBuffer(tail * layers.size());
        for (size_t i = 0; i < layers.size(); ++i) {
            gpu_memory->ReadBlock(image.gpu_addr + u64{layers[i]} * image.info.layer_stride +
                                      level_offset,
                                  staging.mapped_span.data() + i * tail, tail,
                                  VideoCommon::CacheType::NoTextureCache);
        }
        uploads.erase(uploads.begin(), uploads.begin() + image.host_level_offset);
        for (auto& upload : uploads) {
            upload.buffer_offset -= level_offset;
        }
        runtime.AccelerateImageUploadLayers(image, staging, uploads, layers);
        ++images_uploaded;
        uploaded_bytes += tail * layers.size();
        ++layer_uploads;
    }
}

template <class P>
template <typename StagingBuffer>
void TextureCache<P>::UploadImageContents(Image& image, StagingBuffer& staging) {
    const std::span<u8> mapped_span = staging.mapped_span;
    const GPUVAddr gpu_addr = image.gpu_addr;

    if (True(image.flags & ImageFlagBits::AcceleratedUpload) && image.host_level_offset != 0) {
        // Only levels >= host_level_offset exist on the host: read each layer's tail of the mip
        // chain into a compact buffer (layer stride = that tail's size) and decode those levels.
        auto uploads = FullUploadSwizzles(image.info);
        const size_t level_offset = uploads[image.host_level_offset].buffer_offset;
        const size_t tail = image.info.layer_stride - level_offset;
        for (s32 layer = 0; layer < image.info.resources.layers; ++layer) {
            gpu_memory->ReadBlock(gpu_addr + layer * image.info.layer_stride + level_offset,
                                  mapped_span.data() + layer * tail, tail,
                                  VideoCommon::CacheType::NoTextureCache);
        }
        uploads.erase(uploads.begin(), uploads.begin() + image.host_level_offset);
        for (auto& upload : uploads) {
            upload.buffer_offset -= level_offset;
        }
        runtime.AccelerateImageUpload(image, staging, uploads, 0, 0);
        return;
    }
    if (True(image.flags & ImageFlagBits::AcceleratedUpload)) {
        gpu_memory->ReadBlock(gpu_addr, mapped_span.data(), mapped_span.size_bytes(),
                              VideoCommon::CacheType::NoTextureCache);
        const auto uploads = FullUploadSwizzles(image.info);
        runtime.AccelerateImageUpload(image, staging, uploads, 0, 0);
        return;
    }

    Tegra::Memory::GpuGuestMemory<u8, Tegra::Memory::GuestMemoryFlags::UnsafeRead> swizzle_data(
        *gpu_memory, gpu_addr, image.guest_size_bytes, &swizzle_data_buffer);
    if (True(image.flags & ImageFlagBits::Converted)) {
        unswizzle_data_buffer.resize_destructive(image.unswizzled_size_bytes);
        auto copies = UnswizzleImage(*gpu_memory, gpu_addr, image.info, swizzle_data, unswizzle_data_buffer);
        ConvertImage(unswizzle_data_buffer, image.info, mapped_span, copies);
        image.UploadMemory(staging, copies);
    } else {
        const auto copies = UnswizzleImage(*gpu_memory, gpu_addr, image.info, swizzle_data, mapped_span);
        image.UploadMemory(staging, copies);
    }
}

template <class P>
ImageViewId TextureCache<P>::CreateImageView(const TICEntry& raw_config) {
    const bool native = gpu_memory->UsesNativeTiling();
    if (native && raw_config.BaseLayer() && std::getenv("EDEN_NATIVE_DRAW_DIAGNOSTICS")) {
        static u32 texture_view_traces{};
        if (texture_view_traces++ < 8) {
            LOG_WARNING(HW_GPU, "Native texture view header: address={:#x}, dimensions={}x{}x{}, type={}, legacy_layer={}, raw={:016x}/{:016x}/{:016x}/{:016x}",
                        raw_config.Address(), raw_config.Width(), raw_config.Height(), raw_config.Depth(),
                        static_cast<u32>(raw_config.texture_type.Value()), raw_config.BaseLayer(),
                        raw_config.raw[0], raw_config.raw[1], raw_config.raw[2], raw_config.raw[3]);
        }
    }
    const TICEntry config = TextureViewHeader(raw_config, native);
    const ImageInfo info(config);
    if (info.type == ImageType::Buffer) {
        const ImageViewInfo view_info(config, 0);
        const ImageViewId buffer_view_id =
            slot_image_views.insert(runtime, info, view_info, config.Address());
        // D6b: every slot_image_views insert sets a GLOBALLY monotonic
        // generation (fault-contained); the counter advances even when the
        // entry is cap-skipped so numbers never repeat across occupants.
        Common::DiagTrace::Observe([&] {
            const u64 vgen = ++diag_view_next_gen_;
            if (diag_view_gen_.size() < 262144 ||
                diag_view_gen_.contains(buffer_view_id.index)) {
                diag_view_gen_[buffer_view_id.index] = vgen;
            } else {
                Common::DiagTrace::S().view_gen_skipped.fetch_add(1, std::memory_order_relaxed);
            }
            Common::DiagTrace::Emit(Common::DiagTrace::Ev::ViewCreate, frame_tick,
                                    buffer_view_id.index, 0, vgen,
                                    static_cast<u64>(info.type), "buffer");
        });
        return buffer_view_id;
    }
    const u32 layer_offset = config.BaseLayer() * info.layer_stride;
    const GPUVAddr image_gpu_addr = config.Address() - layer_offset;
    const ImageId image_id = FindOrInsertImage(info, image_gpu_addr);
    if (!image_id) {
        return NULL_IMAGE_VIEW_ID;
    }
    ImageBase& image = slot_images[image_id];
    const SubresourceBase base = image.TryFindBase(config.Address()).value();
    ASSERT(base.level == 0);
    const ImageViewInfo view_info(config, base.layer);
    const ImageViewId image_view_id = FindOrEmplaceImageView(image_id, view_info);
    ImageViewBase& image_view = slot_image_views[image_view_id];
    image_view.flags |= ImageViewFlagBits::Strong;
    image.flags |= ImageFlagBits::Strong;
    return image_view_id;
}

template <class P>
ImageId TextureCache<P>::FindOrInsertImage(const ImageInfo& info, GPUVAddr gpu_addr,
                                           RelaxedOptions options) {
    if (const ImageId image_id = FindImage(info, gpu_addr, options); image_id) {
        return image_id;
    }
    return InsertImage(info, gpu_addr, options);
}

template <class P>
ImageId TextureCache<P>::FindImage(const ImageInfo& info, GPUVAddr gpu_addr,
                                   RelaxedOptions options) {
    const std::optional<DAddr> cpu_addr = ImageCpuAddress(gpu_addr, CalculateGuestSizeInBytes(info));
    if (!cpu_addr) {
        return ImageId{};
    }
    const bool broken_views =
        runtime.HasBrokenTextureViewFormats() || True(options & RelaxedOptions::ForceBrokenViews);
    const bool native_bgr = runtime.HasNativeBgr();
    const bool flexible_formats = True(options & RelaxedOptions::Format);
    ImageId image_id{};
    boost::container::small_vector<ImageId, 8> image_ids;
    const auto lambda = [&](ImageId existing_image_id, ImageBase& existing_image) {
        if (True(existing_image.flags & ImageFlagBits::Remapped)) {
            return false;
        }
        if (info.type == ImageType::Linear || existing_image.info.type == ImageType::Linear)
            [[unlikely]] {
            const bool strict_size = False(options & RelaxedOptions::Size) &&
                                     True(existing_image.flags & ImageFlagBits::Strong);
            const ImageInfo& existing = existing_image.info;
            if (existing_image.gpu_addr == gpu_addr && existing.type == info.type &&
                existing.pitch == info.pitch &&
                IsPitchLinearSameSize(existing, info, strict_size) &&
                IsViewCompatible(existing.format, info.format, broken_views, native_bgr)) {
                image_id = existing_image_id;
                image_ids.push_back(existing_image_id);
                return !flexible_formats && existing.format == info.format;
            }
        } else if (IsSubresource(info, existing_image, gpu_addr, options, broken_views,
                                 native_bgr)) {
            image_id = existing_image_id;
            image_ids.push_back(existing_image_id);
            return !flexible_formats && existing_image.info.format == info.format;
        }
        return false;
    };
    ForEachImageInRegion(*cpu_addr, CalculateGuestSizeInBytes(info), lambda);
    if (image_ids.size() <= 1) [[likely]] {
        return image_id;
    }
    auto image_ids_compare = [this](ImageId a, ImageId b) {
        auto& image_a = slot_images[a];
        auto& image_b = slot_images[b];
        return image_a.modification_tick < image_b.modification_tick;
    };
    return *std::ranges::max_element(image_ids, image_ids_compare);
}

template <class P>
bool TextureCache<P>::ImageCanRescale(ImageBase& image) {
    if (!image.info.rescaleable) {
        return false;
    }
    if (Settings::values.resolution_info.downscale && !image.info.downscaleable) {
        return false;
    }
    if (True(image.flags & (ImageFlagBits::Rescaled | ImageFlagBits::CheckingRescalable))) {
        return true;
    }
    if (True(image.flags & ImageFlagBits::IsRescalable)) {
        return true;
    }
    image.flags |= ImageFlagBits::CheckingRescalable;
    for (const auto& alias : image.aliased_images) {
        Image& other_image = slot_images[alias.id];
        if (!ImageCanRescale(other_image)) {
            image.flags &= ~ImageFlagBits::CheckingRescalable;
            return false;
        }
    }
    image.flags &= ~ImageFlagBits::CheckingRescalable;
    image.flags |= ImageFlagBits::IsRescalable;
    return true;
}

template <class P>
void TextureCache<P>::InvalidateScale(Image& image) {
    if (image.scale_tick <= frame_tick) {
        image.scale_tick = frame_tick + 1;
    }
    const std::span<const ImageViewId> image_view_ids = image.image_view_ids;
    auto& dirty = maxwell3d->dirty.flags;
    dirty[Dirty::RenderTargets] = true;
    dirty[Dirty::ZetaBuffer] = true;
    for (size_t rt = 0; rt < NUM_RT; ++rt) {
        dirty[Dirty::ColorBuffer0 + rt] = true;
    }
    for (const ImageViewId image_view_id : image_view_ids) {
        std::ranges::replace(render_targets.color_buffer_ids, image_view_id, ImageViewId{});
        if (render_targets.depth_buffer_id == image_view_id) {
            render_targets.depth_buffer_id = ImageViewId{};
        }
    }
    RemoveImageViewReferences(image_view_ids);
    RemoveFramebuffers(image_view_ids);
    for (const ImageViewId image_view_id : image_view_ids) {
        sentenced_image_view.Push(std::move(slot_image_views[image_view_id]));
        Common::DiagTrace::Observe([&] {
            const u64 old_gen = DiagViewGenOf(image_view_id);
            Common::DiagTrace::Emit(Common::DiagTrace::Ev::ViewErase, frame_tick,
                                    image_view_id.index, 0, 0, old_gen);
            // D6b: clear the gen entry; gens are globally monotonic, so a
            // later occupant gets a fresh number, never this one.
            diag_view_gen_.erase(image_view_id.index);
        });
        slot_image_views.erase(image_view_id);
    }
    image.image_view_ids.clear();
    image.image_view_infos.clear();
    for (size_t c : active_channel_ids) {
        auto& channel_info = channel_storage[c];

        if constexpr (ENABLE_VALIDATION)
            for (auto& e : channel_info.image_view_ids)
                e.second = CORRUPT_ID;
        channel_info.graphics_image_table.Invalidate();
        channel_info.compute_image_table.Invalidate();
    }
    has_deleted_images = true;
}

template <class P>
u64 TextureCache<P>::GetScaledImageSizeBytes(const ImageBase& image) {
    const u64 scale_up = static_cast<u64>(Settings::values.resolution_info.up_scale *
                                          Settings::values.resolution_info.up_scale);
    const u64 down_shift = static_cast<u64>(Settings::values.resolution_info.down_shift +
                                            Settings::values.resolution_info.down_shift);
    const u64 image_size_bytes =
        static_cast<u64>((std::max)(image.guest_size_bytes, image.unswizzled_size_bytes));
    const u64 tentative_size = (image_size_bytes * scale_up) >> down_shift;
    const u64 fitted_size = Common::AlignUp(tentative_size, 1024);
    return fitted_size;
}

template <class P>
void TextureCache<P>::QueueAsyncDecode(Image& image, ImageId image_id) {
    UNIMPLEMENTED_IF(False(image.flags & ImageFlagBits::Converted));
    LOG_DEBUG(HW_GPU, "Queuing async texture decode {:#x}", image.gpu_addr);

    image.flags |= ImageFlagBits::IsDecoding;
    ++async_decodes_queued;
    auto decode = std::make_unique<AsyncDecodeContext>();
    auto* decode_ptr = decode.get();
    decode->image_id = image_id;
    decode->trace_obj = DiagTraceObjOf(image_id); // diag: identity frozen at queue time
    async_decodes.push_back(std::move(decode));

    std::vector<u8> local_unswizzle_data_buffer(image.unswizzled_size_bytes, 0);
    Tegra::Memory::GpuGuestMemory<u8, Tegra::Memory::GuestMemoryFlags::UnsafeRead> swizzle_data(*gpu_memory, image.gpu_addr, image.guest_size_bytes, &swizzle_data_buffer);
    auto copies = UnswizzleImage(*gpu_memory, image.gpu_addr, image.info, swizzle_data, local_unswizzle_data_buffer);
    const size_t out_size = MapSizeBytes(image);
    if (False(image.flags & ImageFlagBits::ZeroFilled)) {
        // Until the decode lands (a second or more while title 1 streams ~40 textures a second)
        // a new image showed whatever its memory last held, often other textures. Zero blocks
        // read as opaque black (BC1) or transparent (BC3) instead.
        image.flags |= ImageFlagBits::ZeroFilled;
        auto zero_copies = copies;
        ConvertedLayout(image.info, zero_copies);
        auto staging = runtime.UploadStagingBuffer(out_size);
        std::memset(staging.mapped_span.data(), 0, out_size);
        image.UploadMemory(staging, zero_copies);
        Common::DiagTrace::Emit(Common::DiagTrace::Ev::ImageUpload, frame_tick,
                                DiagTraceObjOf(image_id), 0, 0, 0, "zero-fill");
        runtime.InsertUploadMemoryBarrier();
    }

    auto func = [out_size, copies, info = image.info,
                 input = std::move(local_unswizzle_data_buffer),
                 async_decode = decode_ptr]() mutable {
        async_decode->decoded_data.resize_destructive(out_size);
        std::span copies_span{copies.data(), copies.size()};
        ConvertImage(input, info, async_decode->decoded_data, copies_span);

        // TODO: Do we need this lock?
        std::unique_lock lock{async_decode->mutex};
        async_decode->copies = std::move(copies);
        async_decode->complete = true;
    };
    texture_decode_worker.QueueWork(std::move(func));
}

template <class P>
void TextureCache<P>::QueueAsyncUnswizzle(Image& image, ImageId image_id) {
    if (True(image.flags & ImageFlagBits::IsDecoding)) {
        return;
    }

    image.flags |= ImageFlagBits::IsDecoding;

    unswizzle_queue.push_back({
        .image_id = image_id,
        .info = image.info
    });
    unswizzle_queue.back().trace_obj = DiagTraceObjOf(image_id); // diag: frozen at queue time
    Common::DiagTrace::Emit(Common::DiagTrace::Ev::ImageUpload, frame_tick,
                            unswizzle_queue.back().trace_obj, 0, 0, 0, "unswizzle-queued");
}

template <class P>
void TextureCache<P>::TickAsyncDecode() {
    bool has_uploads{};
    auto i = async_decodes.begin();
    while (i != async_decodes.end()) {
        auto* async_decode = i->get();
        {
            // Release the lock before the context is erased below: unlocking it afterwards wrote
            // into freed memory (heap corruption, crashes minutes into title 1).
            std::unique_lock lock{async_decode->mutex};
            if (!async_decode->complete) {
                ++i;
                continue;
            }
        }
        if (async_decode->cancelled) {
            i = async_decodes.erase(i);
            continue;
        }
        Image& image = slot_images[async_decode->image_id];
        if (async_decode->decoded_data.size() > MapSizeBytes(image)) {
            LOG_ERROR(HW_GPU, "Async decode of {} bytes for a {}-byte image at {:#x}, dropped",
                      async_decode->decoded_data.size(), MapSizeBytes(image), image.gpu_addr);
            i = async_decodes.erase(i);
            continue;
        }
        auto staging = runtime.UploadStagingBuffer(MapSizeBytes(image));
        std::memcpy(staging.mapped_span.data(), async_decode->decoded_data.data(),
                    async_decode->decoded_data.size());
        image.UploadMemory(staging, async_decode->copies);
        image.flags &= ~ImageFlagBits::IsDecoding;
        const char* up_tag; // diag: re-check the queue-time identity against the record
        if (async_decode->trace_obj == 0) {
            up_tag = "async-decode-unknown";
        } else if (DiagImageStateOf(async_decode->image_id) < 0) {
            up_tag = "async-decode-missing";
        } else if (DiagImageStateOf(async_decode->image_id) == 1) {
            up_tag = "async-decode-erased"; // guard: cancelled should have dropped it;
                                            // if this fires, the cancel path missed
        } else if (DiagTraceObjOf(async_decode->image_id) != async_decode->trace_obj) {
            up_tag = "async-decode-superseded";
        } else {
            up_tag = "async-decode";
        }
        Common::DiagTrace::Emit(Common::DiagTrace::Ev::ImageUpload, frame_tick,
                                async_decode->trace_obj, 0, 0, 0, up_tag);
        has_uploads = true;
        i = async_decodes.erase(i);
    }
    if (has_uploads) {
        runtime.InsertUploadMemoryBarrier();
    }
}

template <class P>
void TextureCache<P>::TickAsyncUnswizzle() {
    if (unswizzle_queue.empty()) {
        return;
    }

    if(current_unswizzle_frame > 0) {
        current_unswizzle_frame--;
        return;
    }

    PendingUnswizzle& task = unswizzle_queue.front();
    Image& image = slot_images[task.image_id];

    if (!task.initialized) {
        task.total_size = MapSizeBytes(image);
        task.staging_buffer = runtime.UploadStagingBuffer(task.total_size, true);

        const auto& info = image.info;
        const u32 bytes_per_block = BytesPerBlock(info.format);
        const u32 width_blocks = Common::DivCeil(info.size.width, 4u);
        const u32 height_blocks = Common::DivCeil(info.size.height, 4u);

        const u32 stride = width_blocks * bytes_per_block;
        const u32 aligned_height = height_blocks;
        task.bytes_per_slice = static_cast<size_t>(stride) * aligned_height;
        task.last_submitted_offset = 0;
        task.initialized = true;
    }

    // Read data
    if (task.current_offset < task.total_size) {
        const size_t remaining = task.total_size - task.current_offset;

        size_t copy_amount = (std::min)(swizzle_chunk_size, remaining);

        if (remaining > swizzle_chunk_size) {
            copy_amount = (copy_amount / task.bytes_per_slice) * task.bytes_per_slice;
            if (copy_amount == 0) copy_amount = task.bytes_per_slice;
        }

        gpu_memory->ReadBlock(image.gpu_addr + task.current_offset,
                              task.staging_buffer.mapped_span.data() + task.current_offset,
                              copy_amount);
        task.current_offset += copy_amount;
    }

    const bool is_final_batch = task.current_offset >= task.total_size;
    const size_t bytes_ready = task.current_offset - task.last_submitted_offset;
    const u32 complete_slices = static_cast<u32>(bytes_ready / task.bytes_per_slice);

    if (complete_slices >= swizzle_slices_per_batch || (is_final_batch && complete_slices > 0)) {
        const u32 z_start = static_cast<u32>(task.last_submitted_offset / task.bytes_per_slice);
        const u32 slices_to_process = (std::min)(complete_slices, swizzle_slices_per_batch);
        const u32 z_count = (std::min)(slices_to_process, image.info.size.depth - z_start);

        if (z_count > 0) {
            const auto uploads = FullUploadSwizzles(task.info);
            runtime.AccelerateImageUpload(image, task.staging_buffer, uploads, z_start, z_count);
            task.last_submitted_offset += (static_cast<size_t>(z_count) * task.bytes_per_slice);
        }
    }

    // Check if complete
    const u32 slices_submitted = static_cast<u32>(task.last_submitted_offset / task.bytes_per_slice);
    const bool all_slices_submitted = slices_submitted >= image.info.size.depth;

    if (is_final_batch && all_slices_submitted) {
        runtime.FreeDeferredStagingBuffer(task.staging_buffer);
        image.flags &= ~ImageFlagBits::IsDecoding;
        const char* up_tag; // diag: re-check the queue-time identity against the record
        if (task.trace_obj == 0) {
            up_tag = "unswizzle-submitted-unknown";
        } else if (DiagImageStateOf(task.image_id) < 0) {
            up_tag = "unswizzle-submitted-missing";
        } else if (DiagImageStateOf(task.image_id) == 1) {
            up_tag = "unswizzle-submitted-erased"; // pre-existing hazard: DeleteImage
                                                   // never touches unswizzle_queue
        } else if (DiagTraceObjOf(task.image_id) != task.trace_obj) {
            up_tag = "unswizzle-submitted-superseded";
        } else {
            up_tag = "unswizzle-submitted"; // final batch ISSUED, not GPU-complete
        }
        Common::DiagTrace::Emit(Common::DiagTrace::Ev::ImageUpload, frame_tick,
                                task.trace_obj, 0, 0, 0, up_tag);
        unswizzle_queue.pop_front();

        // Wait 4 frames to process the next entry
        current_unswizzle_frame = 4u;
    }
}

template <class P>
bool TextureCache<P>::ScaleUp(Image& image) {
    const bool has_copy = image.HasScaled();
    const bool rescaled = image.ScaleUp();
    if (!has_copy && image.HasScaled()) {
        total_used_memory += GetScaledImageSizeBytes(image);
    }
    if (!rescaled) {
        return false;
    }
    InvalidateScale(image);
    return true;
}

template <class P>
bool TextureCache<P>::ScaleDown(Image& image) {
    const bool rescaled = image.ScaleDown();
    if (!rescaled) {
        return false;
    }
    InvalidateScale(image);
    return true;
}

template <class P>
std::optional<DAddr> TextureCache<P>::ImageCpuAddress(GPUVAddr gpu_addr, size_t size) const {
    if (const std::optional<DAddr> cpu_addr = gpu_memory->GpuToCpuAddress(gpu_addr)) {
        return cpu_addr;
    }
    if (const std::optional<DAddr> cpu_addr = gpu_memory->GpuToCpuAddress(gpu_addr, size)) {
        return cpu_addr;
    }
    // No CPU backing (title 1 renders a bloom chain at GPU 0xf0000-0x15b600): a stand-in address
    // fixed by the GPU address, at or above ~(1 << 40) so TrackImage skips it. A fresh address per
    // lookup made every use create another image (thousands a minute, never found again).
    constexpr GPUVAddr StandInRange = 1ULL << 40;
    if (gpu_addr >= StandInRange || size > StandInRange - gpu_addr - 1) {
        return std::nullopt;
    }
    return ~(1ULL << 40ULL) + gpu_addr;
}

template <class P>
ImageId TextureCache<P>::InsertImage(const ImageInfo& info, GPUVAddr gpu_addr,
                                     RelaxedOptions options) {
    std::optional<DAddr> cpu_addr = ImageCpuAddress(gpu_addr, CalculateGuestSizeInBytes(info));
    if (!cpu_addr) {
        const DAddr fake_addr = ~(1ULL << 40ULL) + virtual_invalid_space;
        virtual_invalid_space += Common::AlignUp(CalculateGuestSizeInBytes(info), 32);
        cpu_addr = std::optional<DAddr>(fake_addr);
    }
    ASSERT_MSG(cpu_addr, "Tried to insert an image to an invalid gpu_addr={:#x}", gpu_addr);
    ImageInfo storage_info = info;
    storage_info.native_tiling = gpu_memory->UsesNativeTiling();
    if (storage_info.native_tiling && info.type == ImageType::e3D) {
        // Switch 2 3D images (lighting/fog volumes) are rare: log each one's shape.
        LOG_DEBUG(HW_GPU, "3D image at {:#x}: {}x{}x{} levels {} format {} block {}x{}x{}",
                 gpu_addr, info.size.width, info.size.height, info.size.depth, info.resources.levels,
                 static_cast<u32>(info.format), info.block.width, info.block.height, info.block.depth);
    }
    const ImageId image_id = JoinImages(storage_info, gpu_addr, *cpu_addr);
    const Image& image = slot_images[image_id];
    // Using "image.gpu_addr" instead of "gpu_addr" is important because it might be different
    const auto [it, is_new] = image_allocs_table.try_emplace(image.gpu_addr);
    if (is_new) {
        it->second = slot_image_allocs.insert();
    }
    slot_image_allocs[it->second].images.push_back(image_id);
    return image_id;
}

template <class P>
ImageId TextureCache<P>::JoinImages(const ImageInfo& info, GPUVAddr gpu_addr, DAddr cpu_addr) {
    ImageInfo new_info = info;
    const size_t size_bytes = CalculateGuestSizeInBytes(new_info);
    const bool broken_views = runtime.HasBrokenTextureViewFormats();
    const bool native_bgr = runtime.HasNativeBgr();
    join_overlap_ids.clear();
    join_overlaps_found.clear();
    join_left_aliased_ids.clear();
    join_right_aliased_ids.clear();
    join_ignore_textures.clear();
    join_bad_overlap_ids.clear();
    join_copies_to_do.clear();
    join_alias_indices.clear();
    const bool this_is_linear = info.type == ImageType::Linear;
    const auto region_check = [&](ImageId overlap_id, ImageBase& overlap) {
        // What the GPU rendered into stays a render target when a texture lookup rebuilds it.
        new_info.render_target |= overlap.info.render_target;
        if (True(overlap.flags & ImageFlagBits::Remapped)) {
            join_ignore_textures.insert(overlap_id);
            return;
        }
        const bool overlap_is_linear = overlap.info.type == ImageType::Linear;
        if (this_is_linear != overlap_is_linear) {
            return;
        }
        if (this_is_linear && overlap_is_linear) {
            if (info.pitch == overlap.info.pitch && gpu_addr == overlap.gpu_addr) {
                // Alias linear images with the same pitch
                join_left_aliased_ids.push_back(overlap_id);
            }
            return;
        }
        join_overlaps_found.insert(overlap_id);
        constexpr bool strict_size = true;
        const std::optional<OverlapResult> solution = ResolveOverlap(
            new_info, gpu_addr, cpu_addr, overlap, strict_size, broken_views, native_bgr);
        if (solution) {
            gpu_addr = solution->gpu_addr;
            cpu_addr = solution->cpu_addr;
            new_info.resources = solution->resources;
            join_overlap_ids.push_back(overlap_id);
            join_copies_to_do.emplace_back(JoinCopy{false, overlap_id});
            return;
        }
        // A Switch 2 render target re-created at another size at the same address (dynamic
        // resolution changes it every frame: 15 sizes in a second in Mario Kart World's menus).
        // Aliasing would keep every size alive until the collector gets to it, and that filled
        // the Deck's GTT. The old one is dead on the console as well; copy what overlaps and
        // drop it.
        if (overlap.gpu_addr == gpu_addr && new_info.type == ImageType::e2D &&
            overlap.info.type == ImageType::e2D && new_info.resources.levels == 1 &&
            overlap.info.resources.levels == 1 &&
            (new_info.size.width != overlap.info.size.width ||
             new_info.size.height != overlap.info.size.height) &&
            Service::Nvidia::Devices::IsNx2GpuOnlyAddress(gpu_addr)) {
            join_copies_to_do.emplace_back(JoinCopy{false, overlap_id, true});
            return;
        }
        constexpr auto options = RelaxedOptions::Size | RelaxedOptions::Format;
        const ImageBase new_image_base(new_info, gpu_addr, cpu_addr);
        if (IsSubresource(new_info, overlap, gpu_addr, options, broken_views, native_bgr)) {
            join_left_aliased_ids.push_back(overlap_id);
            overlap.flags |= ImageFlagBits::Alias;
            join_copies_to_do.emplace_back(JoinCopy{true, overlap_id});
        } else if (IsSubresource(overlap.info, new_image_base, overlap.gpu_addr, options,
                                 broken_views, native_bgr)) {
            join_right_aliased_ids.push_back(overlap_id);
            overlap.flags |= ImageFlagBits::Alias;
            join_copies_to_do.emplace_back(JoinCopy{true, overlap_id});
        } else {
            join_bad_overlap_ids.push_back(overlap_id);
        }
    };
    ForEachImageInRegion(cpu_addr, size_bytes, region_check);
    const auto region_check_gpu = [&](ImageId overlap_id, ImageBase& overlap) {
        if (!join_overlaps_found.contains(overlap_id)) {
            if (True(overlap.flags & ImageFlagBits::Remapped)) {
                join_ignore_textures.insert(overlap_id);
            }
            if (overlap.gpu_addr == gpu_addr && overlap.guest_size_bytes == size_bytes) {
                join_ignore_textures.insert(overlap_id);
            }
        }
    };
    ForEachSparseImageInRegion(channel_state->gpu_memory.GetID(), gpu_addr, size_bytes,
                               region_check_gpu);

    bool can_rescale = info.rescaleable;
    bool any_rescaled = false;
    for (const auto& copy : join_copies_to_do) {
        if (!can_rescale) {
            break;
        }
        Image& sibling = slot_images[copy.id];
        can_rescale &= ImageCanRescale(sibling);
        any_rescaled |= True(sibling.flags & ImageFlagBits::Rescaled);
    }

    can_rescale &= any_rescaled;

    if (can_rescale) {
        for (const auto& copy : join_copies_to_do) {
            Image& sibling = slot_images[copy.id];
            ScaleUp(sibling);
        }
    } else {
        for (const auto& copy : join_copies_to_do) {
            Image& sibling = slot_images[copy.id];
            ScaleDown(sibling);
        }
    }

    if constexpr (requires { runtime.UseOpaqueAstcFormat(); }) {
        // Opaque ASTC goes to BC1 rather than BC3 when it is re-compressed (the Vulkan runtime
        // decides). The guest data is already there when an image is first used.
        constexpr size_t MaxScanBytes = 64_MiB;
        if (IsPixelFormatASTC(new_info.format) && runtime.UseOpaqueAstcFormat()) {
            const size_t scan_bytes = CalculateGuestSizeInBytes(new_info);
            if (scan_bytes <= MaxScanBytes) {
                astc_scan_buffer.resize(scan_bytes);
                gpu_memory->ReadBlockUnsafe(gpu_addr, astc_scan_buffer.data(), scan_bytes);
                new_info.astc_opaque = AstcDataLooksOpaque(astc_scan_buffer);
                ++(new_info.astc_opaque ? astc_opaque_images : astc_alpha_images);
            }
        }
    }
    // A size this address already went through (the resize path below) comes back as it was.
    const bool resizing = std::ranges::any_of(join_copies_to_do, [](const JoinCopy& copy) {
        return copy.resize;
    });
    const ImageId revived_id = resizing ? ReviveDormantImage(new_info, gpu_addr, cpu_addr) : ImageId{};
    const ImageId new_image_id =
        revived_id ? revived_id : slot_images.insert(runtime, new_info, gpu_addr, cpu_addr);
    Image& new_image = slot_images[new_image_id];
    u64 diag_trace_obj = 0;
    // Diagnostic identity tracking only; skipped entirely when the master
    // switch is off, and fault-contained (map inserts can allocate/throw) so
    // an observer fault can never escape into image creation.
    Common::DiagTrace::Observe([&] {
        // D2: which address space this lineage belongs to (evidence only).
        // Computed inside the Enabled-gated block; tags print it + 1 so that
        // 0 stays reserved for "untracked" (raw manager ids start at 0).
        const u64 diag_asid = channel_state->gpu_memory.GetID();
        if (revived_id) {
        // Same slot object revived: keep its trace identity, bump the generation.
        if (auto dit = diag_images_.find(new_image_id.index); dit != diag_images_.end()) {
            dit->second.state = 0;
            ++dit->second.generation;
            diag_trace_obj = dit->second.trace_obj;
            // ReviveDormantImage matches on gpu/cpu addr + shape only, NOT on
            // address space. Disclose the observed current asid and, when it
            // differs from the creation-time record, update the association
            // and say so (production revive policy unchanged).
            char diag_revive_tag[64];
            if (dit->second.asid != diag_asid) {
                std::snprintf(diag_revive_tag, sizeof(diag_revive_tag),
                              "same-object as%llx was%llx",
                              (unsigned long long)(diag_asid + 1),
                              (unsigned long long)(dit->second.asid + 1));
                dit->second.asid = diag_asid;
            } else {
                std::snprintf(diag_revive_tag, sizeof(diag_revive_tag), "same-object as%llx",
                              (unsigned long long)(diag_asid + 1));
            }
            Common::DiagTrace::Emit(Common::DiagTrace::Ev::ImageRevive, frame_tick,
                                    diag_trace_obj, new_image_id.index, gpu_addr,
                                    dit->second.generation, diag_revive_tag);
        } else {
            diag_trace_obj =
                Common::DiagTrace::S().next_obj_id.fetch_add(1, std::memory_order_relaxed);
            if (diag_images_.size() < 262144 || diag_images_.contains(new_image_id.index)) {
                diag_images_[new_image_id.index] =
                    DiagImageRec{diag_trace_obj, gpu_addr, frame_tick, 0, 0, diag_asid};
            } else {
                Common::DiagTrace::S().identity_skipped.fetch_add(1, std::memory_order_relaxed);
            }
            char diag_revive_tag[48];
            std::snprintf(diag_revive_tag, sizeof(diag_revive_tag), "no-prior-record as%llx",
                          (unsigned long long)(diag_asid + 1));
            Common::DiagTrace::Emit(Common::DiagTrace::Ev::ImageRevive, frame_tick,
                                    diag_trace_obj, new_image_id.index, gpu_addr, 0,
                                    diag_revive_tag);
        }
    } else {
        diag_trace_obj =
            Common::DiagTrace::S().next_obj_id.fetch_add(1, std::memory_order_relaxed);
        if (diag_images_.size() < 262144 || diag_images_.contains(new_image_id.index)) {
            diag_images_[new_image_id.index] =
                DiagImageRec{diag_trace_obj, gpu_addr, frame_tick, 0, 0, diag_asid};
        } else {
            Common::DiagTrace::S().identity_skipped.fetch_add(1, std::memory_order_relaxed);
        }
        // D9: flags at construction (Sparse is OR-ed in after this emit); D2: ASID.
        char diag_shape[64];
        const int diag_shape_len = std::snprintf(
            diag_shape, sizeof(diag_shape), "fmt%u ty%u lv%u la%u f%x as%llx",
            static_cast<u32>(new_info.format), static_cast<u32>(new_info.type),
            new_info.resources.levels, new_info.resources.layers,
            static_cast<u32>(new_image.flags), (unsigned long long)(diag_asid + 1));
        if (diag_shape_len < 0 || static_cast<size_t>(diag_shape_len) >= sizeof(diag_shape)) {
            Common::DiagTrace::S().tag_truncated.fetch_add(1, std::memory_order_relaxed);
            std::memcpy(diag_shape + sizeof(diag_shape) - 4, "~TR", 4);
        }
        Common::DiagTrace::Emit(Common::DiagTrace::Ev::ImageCreate, frame_tick,
                                diag_trace_obj, new_image_id.index, gpu_addr,
                                new_image.guest_size_bytes, diag_shape);
        }
    });

    new_image.allocation_tick = frame_tick;
    if (!revived_id) {
        // Which shapes are being created (periodic log): format, type, levels, layers, size.
        const u64 key = (u64{static_cast<u32>(new_info.format)} << 56) |
                        (u64{static_cast<u32>(new_info.type)} << 52) |
                        (u64{static_cast<u32>(new_info.resources.levels) & 0xF} << 48) |
                        (u64{static_cast<u32>(new_info.resources.layers) & 0xFFF} << 36) |
                        (u64{new_info.size.width & 0x3FFFF} << 18) | (new_info.size.height & 0x3FFFF);
        auto& shape = created_shapes[key];
        ++shape.first;
        shape.second = gpu_addr;
        ++images_created;
    }

    if (!gpu_memory->IsContinuousRange(new_image.gpu_addr, new_image.guest_size_bytes) &&
        new_info.is_sparse) {
        new_image.flags |= ImageFlagBits::Sparse;
    }

    for (const ImageId overlap_id : join_ignore_textures) {
        Image& overlap = slot_images[overlap_id];
        if (True(overlap.flags & ImageFlagBits::GpuModified)) {
            UNIMPLEMENTED();
        }
        if (True(overlap.flags & ImageFlagBits::Tracked)) {
            UntrackImage(overlap, overlap_id);
        }
        UnregisterImage(overlap_id);
        if (const auto dit = diag_images_.find(overlap_id.index);
            dit != diag_images_.end()) {
            char diag_evict_tag[48];
            std::snprintf(diag_evict_tag, sizeof(diag_evict_tag), "join-overlap-delete as%llx",
                          (unsigned long long)(dit->second.asid + 1));
            Common::DiagTrace::Emit(Common::DiagTrace::Ev::RelatedEvict, frame_tick,
                                    dit->second.trace_obj, diag_trace_obj, overlap.gpu_addr, 0,
                                    diag_evict_tag);
        }
        DeleteImage(overlap_id, false, "join-overlap");
    }

    // TODO: Only upload what we need
    RefreshContents(new_image, new_image_id);

    if (can_rescale) {
        ScaleUp(new_image);
    } else {
        ScaleDown(new_image);
    }

    std::ranges::sort(join_copies_to_do, [this](const JoinCopy& lhs, const JoinCopy& rhs) {
        const ImageBase& lhs_image = slot_images[lhs.id];
        const ImageBase& rhs_image = slot_images[rhs.id];
        return lhs_image.modification_tick < rhs_image.modification_tick;
    });

    ImageBase& new_image_base = new_image;
    for (const ImageId aliased_id : join_right_aliased_ids) {
        ImageBase& aliased = slot_images[aliased_id];
        size_t alias_index = new_image_base.aliased_images.size();
        if (!AddImageAlias(new_image_base, aliased, new_image_id, aliased_id)) {
            continue;
        }
        join_alias_indices.emplace(aliased_id, alias_index);
        new_image.flags |= ImageFlagBits::Alias;
    }
    for (const ImageId aliased_id : join_left_aliased_ids) {
        ImageBase& aliased = slot_images[aliased_id];
        size_t alias_index = new_image_base.aliased_images.size();
        if (!AddImageAlias(aliased, new_image_base, aliased_id, new_image_id)) {
            continue;
        }
        join_alias_indices.emplace(aliased_id, alias_index);
        new_image.flags |= ImageFlagBits::Alias;
    }
    for (const ImageId aliased_id : join_bad_overlap_ids) {
        ImageBase& aliased = slot_images[aliased_id];
        aliased.overlapping_images.push_back(new_image_id);
        new_image.overlapping_images.push_back(aliased_id);
        if (aliased.info.resources.levels == 1 && aliased.info.block.depth == 0 &&
            aliased.overlapping_images.size() > 1) {
            aliased.flags |= ImageFlagBits::BadOverlap;
        }
        if (new_image.info.resources.levels == 1 && new_image.info.block.depth == 0 &&
            new_image.overlapping_images.size() > 1) {
            new_image.flags |= ImageFlagBits::BadOverlap;
        }
    }

    for (const auto& copy_object : join_copies_to_do) {
        Image& overlap = slot_images[copy_object.id];
        if (copy_object.is_alias) {
            if (!overlap.IsSafeGpuCopy()) {
                continue;
            }
            if (overlap.info.num_samples != new_image.info.num_samples) {
                continue;
            }
            const auto alias_pointer = join_alias_indices.find(copy_object.id);
            if (alias_pointer == join_alias_indices.end()) {
                continue;
            }
            const AliasedImage& aliased = new_image.aliased_images[alias_pointer->second];
            CopyImage(new_image_id, aliased.id, aliased.copies);
            new_image.modification_tick = overlap.modification_tick;
            continue;
        }
        if (copy_object.resize) {
            if (overlap.IsSafeGpuCopy() && overlap.info.format == new_image.info.format &&
                overlap.info.num_samples == new_image.info.num_samples &&
                False(overlap.flags & ImageFlagBits::Unbacked) &&
                False(new_image.flags & ImageFlagBits::Unbacked)) {
                new_image.flags |= ImageFlagBits::GpuModified;
                const s32 layers = (std::min)(overlap.info.resources.layers,
                                              new_image.info.resources.layers);
                const ImageCopy copy{
                    .src_subresource = {.base_level = 0, .base_layer = 0, .num_layers = layers},
                    .dst_subresource = {.base_level = 0, .base_layer = 0, .num_layers = layers},
                    .src_offset = {},
                    .dst_offset = {},
                    .extent = {(std::min)(overlap.info.size.width, new_image.info.size.width),
                               (std::min)(overlap.info.size.height, new_image.info.size.height), 1},
                };
                runtime.CopyImage(new_image, overlap, std::array{copy});
                // F1 (CopyDep): the actual resize copy; d = layers copied, NOT bytes.
                Common::DiagTrace::Emit(Common::DiagTrace::Ev::CopyDep, frame_tick,
                                        DiagTraceObjOf(copy_object.id), diag_trace_obj,
                                        static_cast<u64>(layers), 0, "join-resize");
                new_image.modification_tick = overlap.modification_tick;
            }
            ++render_target_resizes;
            if (static u32 logged{}; logged++ < 8) {
                LOG_INFO(HW_GPU, "Render target {:#x} resized {}x{} -> {}x{} (format {} -> {}); dropping the old one",
                         gpu_addr, overlap.info.size.width, overlap.info.size.height,
                         new_image.info.size.width, new_image.info.size.height,
                         static_cast<u32>(overlap.info.format), static_cast<u32>(new_image.info.format));
            }
        } else if (True(overlap.flags & ImageFlagBits::GpuModified)) {
            new_image.flags |= ImageFlagBits::GpuModified;
            const auto& resolution = Settings::values.resolution_info;
            const SubresourceBase base = new_image.TryFindBase(overlap.gpu_addr).value();
            const u32 up_scale = can_rescale ? resolution.up_scale : 1;
            const u32 down_shift = can_rescale ? resolution.down_shift : 0;
            auto copies = MakeShrinkImageCopies(new_info, overlap.info, base, up_scale, down_shift);
            // F1 (CopyDep): the actual shrink copy; d = copy-region count, NOT bytes.
            if (overlap.info.num_samples != new_image.info.num_samples) {
                runtime.CopyImageMSAA(new_image, overlap, copies);
                Common::DiagTrace::Emit(Common::DiagTrace::Ev::CopyDep, frame_tick,
                                        DiagTraceObjOf(copy_object.id), diag_trace_obj,
                                        static_cast<u64>(copies.size()), 0, "join-shrink-msaa");
            } else {
                runtime.CopyImage(new_image, overlap, copies);
                Common::DiagTrace::Emit(Common::DiagTrace::Ev::CopyDep, frame_tick,
                                        DiagTraceObjOf(copy_object.id), diag_trace_obj,
                                        static_cast<u64>(copies.size()), 0, "join-shrink");
            }
            new_image.modification_tick = overlap.modification_tick;
        }
        if (True(overlap.flags & ImageFlagBits::Tracked)) {
            UntrackImage(overlap, copy_object.id);
        }
        UnregisterImage(copy_object.id);
        // D4: join-copy replacement edge (eviction relation, NOT proven same
        // semantic object) for both the resize->dormant and the delete route.
        // Tag is a fixed string + 16 hex chars, bounded by construction.
        if (const auto dit = diag_images_.find(copy_object.id.index);
            dit != diag_images_.end()) {
            char diag_evict_tag[48];
            std::snprintf(diag_evict_tag, sizeof(diag_evict_tag), "%s as%llx",
                          copy_object.resize ? "join-copy-resize-dormant" : "join-copy-delete",
                          (unsigned long long)(dit->second.asid + 1));
            Common::DiagTrace::Emit(Common::DiagTrace::Ev::RelatedEvict, frame_tick,
                                    dit->second.trace_obj, diag_trace_obj,
                                    dit->second.gpu_addr, 0, diag_evict_tag);
        }
        if (copy_object.resize) {
            MakeDormant(copy_object.id);
        } else {
            DeleteImage(copy_object.id, false, "join-copy");
        }
    }

    RegisterImage(new_image_id);
    return new_image_id;
}

template <class P>
void TextureCache<P>::MakeDormant(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT(False(image.flags & (ImageFlagBits::Registered | ImageFlagBits::Tracked)));
    // The views stay valid (and the framebuffers built on them), so only what must not be
    // reached through them changes: bound render targets are resolved again, and alias and
    // overlap links go, as they would for a deleted image. Views still cached for texture
    // descriptors are caught by IsDormantView when a lookup hits one.
    auto& dirty = maxwell3d->dirty.flags;
    dirty[Dirty::RenderTargets] = true;
    dirty[Dirty::ZetaBuffer] = true;
    for (size_t rt = 0; rt < NUM_RT; ++rt) {
        dirty[Dirty::ColorBuffer0 + rt] = true;
    }
    for (const ImageViewId image_view_id : image.image_view_ids) {
        std::ranges::replace(render_targets.color_buffer_ids, image_view_id, ImageViewId{});
        if (render_targets.depth_buffer_id == image_view_id) {
            render_targets.depth_buffer_id = ImageViewId{};
        }
    }
    for (const AliasedImage& alias : image.aliased_images) {
        ImageBase& other_image = slot_images[alias.id];
        std::erase_if(other_image.aliased_images, [image_id](const AliasedImage& other_alias) {
            return other_alias.id == image_id;
        });
        other_image.CheckAliasState();
    }
    image.aliased_images.clear();
    for (const ImageId overlap_id : image.overlapping_images) {
        ImageBase& other_image = slot_images[overlap_id];
        std::erase(other_image.overlapping_images, image_id);
        other_image.CheckBadOverlapState();
    }
    image.overlapping_images.clear();
    image.flags &= ~(ImageFlagBits::Alias | ImageFlagBits::BadOverlap);

    const u64 bytes = Common::AlignUp(HostImageSizeBytes(image), 1024);
    // Same policy as DeleteImage (see there): the estimate can exceed the refreshed total.
    if (CollectorSubtract(total_used_memory, bytes)) [[unlikely]] {
        LOG_WARNING(HW_GPU, "dormant image's estimate {} MiB exceeds collector total; saturating",
                    bytes >> 20);
    }
    dormant_bytes += bytes;
    dormant_images[image.gpu_addr].push_back(DormantImage{image_id, frame_tick});
    // D3: dormant is a distinct recorded state (was indistinguishable from
    // "live" in IdentitySnap). Same unguarded-find shape as DeleteImage:
    // snprintf + Emit run only when a record exists, so it is fault-safe in
    // the same way (no map mutation; node write cannot rehash).
    if (const auto dit = diag_images_.find(image_id.index); dit != diag_images_.end()) {
        char diag_dorm_tag[24];
        std::snprintf(diag_dorm_tag, sizeof(diag_dorm_tag), "as%llx",
                      (unsigned long long)(dit->second.asid + 1));
        Common::DiagTrace::Emit(Common::DiagTrace::Ev::ImageDormant, frame_tick,
                                dit->second.trace_obj, image_id.index, dit->second.gpu_addr,
                                dit->second.created_frame, diag_dorm_tag);
        dit->second.state = 2;
    }
}

template <class P>
ImageId TextureCache<P>::ReviveDormantImage(const ImageInfo& info, GPUVAddr gpu_addr,
                                            DAddr cpu_addr) {
    const auto it = dormant_images.find(gpu_addr);
    if (it == dormant_images.end()) {
        return ImageId{};
    }
    auto& candidates = it->second;
    const auto match = std::ranges::find_if(candidates, [&](const DormantImage& dormant) {
        const Image& image = slot_images[dormant.id];
        const ImageInfo& old = image.info;
        return image.cpu_addr == cpu_addr && old.format == info.format && old.type == info.type &&
               old.size == info.size && old.resources == info.resources &&
               old.block == info.block && old.num_samples == info.num_samples &&
               old.native_tiling == info.native_tiling && old.is_sparse == info.is_sparse;
    });
    if (match == candidates.end()) {
        return ImageId{};
    }
    const ImageId image_id = match->id;
    candidates.erase(match);
    if (candidates.empty()) {
        dormant_images.erase(it);
    }
    Image& image = slot_images[image_id];
    const u64 bytes = Common::AlignUp(HostImageSizeBytes(image), 1024);
    dormant_bytes -= bytes;
    // RegisterImage adds it to total_used_memory again.
    ++render_target_revivals;
    return image_id;
}

template <class P>
void TextureCache<P>::EvictDormantImages() {
    // A size not bound for two seconds is not coming back soon; and the pool is bounded.
    constexpr u64 MaxIdleTicks = 120;
    constexpr u64 MaxBytes = 256_MiB;
    if (dormant_images.empty()) {
        return;
    }
    u64 oldest_tick = frame_tick;
    if (dormant_bytes > MaxBytes) {
        // Over budget: the oldest ones go, whatever their age.
        for (const auto& [gpu_addr, candidates] : dormant_images) {
            for (const DormantImage& dormant : candidates) {
                oldest_tick = (std::min)(oldest_tick, dormant.tick);
            }
        }
    }
    for (auto it = dormant_images.begin(); it != dormant_images.end();) {
        auto& candidates = it->second;
        const auto removed = std::remove_if(candidates.begin(), candidates.end(),
                                            [&](const DormantImage& dormant) {
            const bool idle = frame_tick - dormant.tick > MaxIdleTicks;
            const bool over_budget = dormant_bytes > MaxBytes && dormant.tick == oldest_tick;
            if (!idle && !over_budget) {
                return false;
            }
            Image& image = slot_images[dormant.id];
            const u64 bytes = Common::AlignUp(HostImageSizeBytes(image), 1024);
            dormant_bytes -= bytes;
            total_used_memory += bytes; // DeleteImage takes it off again
            DeleteImage(dormant.id, false, "dormant-evict");
            return true;
        });
        candidates.erase(removed, candidates.end());
        it = candidates.empty() ? dormant_images.erase(it) : std::next(it);
    }
}

template <class P>
bool TextureCache<P>::IsDormantView(ImageViewId id) const noexcept {
    if (!id || id == NULL_IMAGE_VIEW_ID) {
        return false;
    }
    const ImageViewBase& view = slot_image_views[id];
    if (view.IsBuffer() || !view.image_id) {
        return false;
    }
    return False(slot_images[view.image_id].flags & ImageFlagBits::Registered);
}

template <class P>
std::optional<typename TextureCache<P>::BlitImages> TextureCache<P>::GetBlitImages(
    const Tegra::Engines::Fermi2D::Surface& dst, const Tegra::Engines::Fermi2D::Surface& src,
    const Tegra::Engines::Fermi2D::Config& copy) {

    constexpr auto FIND_OPTIONS = RelaxedOptions::Samples;
    const GPUVAddr dst_addr = dst.Address();
    const GPUVAddr src_addr = src.Address();
    ImageInfo dst_info(dst);
    ImageInfo src_info(src);
    const bool can_be_depth_blit =
        dst_info.format == src_info.format && copy.filter == Tegra::Engines::Fermi2D::Filter::Point;
    ImageId dst_id;
    ImageId src_id;
    RelaxedOptions try_options = FIND_OPTIONS;
    if (can_be_depth_blit) {
        try_options |= RelaxedOptions::Format;
    }
    do {
        has_deleted_images = false;
        src_id = FindImage(src_info, src_addr, try_options);
        dst_id = FindImage(dst_info, dst_addr, try_options);
        if (!copy.must_accelerate) {
            do {
                if (!src_id && !dst_id) {
                    return std::nullopt;
                }
                if (src_id && True(slot_images[src_id].flags & ImageFlagBits::GpuModified)) {
                    break;
                }
                if (dst_id && True(slot_images[dst_id].flags & ImageFlagBits::GpuModified)) {
                    break;
                }
                return std::nullopt;
            } while (false);
        }
        const ImageBase* const src_image = src_id ? &slot_images[src_id] : nullptr;
        if (src_image && src_image->info.num_samples > 1) {
            RelaxedOptions find_options{FIND_OPTIONS | RelaxedOptions::ForceBrokenViews};
            src_id = FindOrInsertImage(src_info, src_addr, find_options);
            dst_id = FindOrInsertImage(dst_info, dst_addr, find_options);
            if (has_deleted_images) {
                continue;
            }
            break;
        }
        if (can_be_depth_blit) {
            const ImageBase* const dst_image = dst_id ? &slot_images[dst_id] : nullptr;
            DeduceBlitImages(dst_info, src_info, dst_image, src_image);
            if (GetFormatType(dst_info.format) != GetFormatType(src_info.format)) {
                continue;
            }
        }
        if (!src_id) {
            src_id = InsertImage(src_info, src_addr, RelaxedOptions{});
        }
        if (!dst_id) {
            dst_id = InsertImage(dst_info, dst_addr, RelaxedOptions{});
        }
    } while (has_deleted_images);
    const ImageBase& src_image = slot_images[src_id];
    const ImageBase& dst_image = slot_images[dst_id];
    const bool native_bgr = runtime.HasNativeBgr();
    if (GetFormatType(dst_info.format) != GetFormatType(dst_image.info.format) ||
        GetFormatType(src_info.format) != GetFormatType(src_image.info.format) ||
        !VideoCore::Surface::IsViewCompatible(dst_info.format, dst_image.info.format, false,
                                              native_bgr) ||
        !VideoCore::Surface::IsViewCompatible(src_info.format, src_image.info.format, false,
                                              native_bgr)) {
        // Make sure the images match the expected format.
        do {
            has_deleted_images = false;
            src_id = FindOrInsertImage(src_info, src_addr, RelaxedOptions{});
            dst_id = FindOrInsertImage(dst_info, dst_addr, RelaxedOptions{});
        } while (has_deleted_images);
    }
    return {BlitImages{
        .dst_id = dst_id,
        .src_id = src_id,
        .dst_format = dst_info.format,
        .src_format = src_info.format,
    }};
}

template <class P>
ImageId TextureCache<P>::FindDMAImage(const ImageInfo& info, GPUVAddr gpu_addr) {
    const std::optional<DAddr> cpu_addr = ImageCpuAddress(gpu_addr, CalculateGuestSizeInBytes(info));
    if (!cpu_addr) {
        return ImageId{};
    }
    ImageId image_id{};
    boost::container::small_vector<ImageId, 8> image_ids;
    const auto lambda = [&](ImageId existing_image_id, ImageBase& existing_image) {
        if (True(existing_image.flags & ImageFlagBits::Remapped)) {
            return false;
        }
        if (info.type == ImageType::Linear || existing_image.info.type == ImageType::Linear)
            [[unlikely]] {
            const bool strict_size = True(existing_image.flags & ImageFlagBits::Strong);
            const ImageInfo& existing = existing_image.info;
            if (existing_image.gpu_addr == gpu_addr && existing.type == info.type &&
                existing.pitch == info.pitch &&
                IsPitchLinearSameSize(existing, info, strict_size) &&
                IsViewCompatible(existing.format, info.format, false, true)) {
                image_id = existing_image_id;
                image_ids.push_back(existing_image_id);
                return true;
            }
        } else if (IsSubCopy(info, existing_image, gpu_addr)) {
            if (True(existing_image.flags & ImageFlagBits::Unbacked)) {
                static u32 logged{};
                if (logged++ < 64) {
                    LOG_INFO(HW_GPU, "DMA copy at {:#x} ({}x{}) into unbacked image {:#x}", gpu_addr,
                             info.size.width, info.size.height, existing_image.gpu_addr);
                }
            }
            image_id = existing_image_id;
            image_ids.push_back(existing_image_id);
            return true;
        }
        return false;
    };
    ForEachImageInRegion(*cpu_addr, CalculateGuestSizeInBytes(info), lambda);
    if (image_ids.size() <= 1) [[likely]] {
        return image_id;
    }
    auto image_ids_compare = [this](ImageId a, ImageId b) {
        auto& image_a = slot_images[a];
        auto& image_b = slot_images[b];
        return image_a.modification_tick < image_b.modification_tick;
    };
    return *std::ranges::max_element(image_ids, image_ids_compare);
}

template <class P>
std::pair<u32, u32> TextureCache<P>::PrepareDmaImage(ImageId dst_id, GPUVAddr base_addr,
                                                     bool mark_as_modified) {
    const auto& image = slot_images[dst_id];
    const auto base = image.TryFindBase(base_addr);
    PrepareImage(dst_id, mark_as_modified, false);
    const auto& new_image = slot_images[dst_id];
    lru_cache.Touch(new_image.lru_index, frame_tick);
    return std::make_pair(base->level, base->layer);
}

template <class P>
SamplerId TextureCache<P>::FindSampler(const TSCEntry& config, bool compute) {
    if (std::ranges::all_of(config.raw, [](u64 value) { return value == 0; })) {
        return NULL_SAMPLER_ID;
    }
    const auto [pair, is_new] = channel_state->samplers.try_emplace(config);
    if (is_new) {
        pair->second = slot_samplers.insert(runtime, config);
        // EDEN_NX2_TSC_LOG=1: every distinct sampler descriptor as it is first seen.
        if (static const bool log_tsc = std::getenv("EDEN_NX2_TSC_LOG") != nullptr; log_tsc) {
            LOG_INFO(HW_GPU,
                     "nx2 sampler {}: raw {:016x} {:016x} {:016x} {:016x} | compare {} func {} | "
                     "filters mag {} min {} mip {} | wrap {} {} {} | lod {}..{} bias {} aniso {} | "
                     "border {} {} {} {}",
                     pair->second.index, config.raw[0], config.raw[1], config.raw[2], config.raw[3],
                     config.depth_compare_enabled.Value(),
                     static_cast<u32>(config.depth_compare_func.Value()),
                     static_cast<u32>(config.mag_filter.Value()),
                     static_cast<u32>(config.min_filter.Value()),
                     static_cast<u32>(config.mipmap_filter.Value()),
                     static_cast<u32>(config.wrap_u.Value()), static_cast<u32>(config.wrap_v.Value()),
                     static_cast<u32>(config.wrap_p.Value()), config.MinLod(), config.MaxLod(),
                     config.LodBias(), config.MaxAnisotropy(), config.border_color[0],
                     config.border_color[1], config.border_color[2], config.border_color[3]);
        }
    }
    return pair->second;
}

template <class P>
ImageViewId TextureCache<P>::FindColorBuffer(size_t index) {
    const auto& regs = maxwell3d->regs;
    if (index >= regs.rt_control.count) {
        return ImageViewId{};
    }
    const auto& rt = regs.rt[index];
    const GPUVAddr gpu_addr = rt.Address();
    if (gpu_addr == 0) {
        return ImageViewId{};
    }
    if (rt.format == Tegra::RenderTargetFormat::NONE) {
        return ImageViewId{};
    }
    if (!gpu_memory->IsWithinGPUAddressRange(gpu_addr)) {
        LOG_WARNING(HW_GPU, "Ignoring color buffer {} outside the GPU address space: {:#x}", index,
                    gpu_addr);
        return ImageViewId{};
    }
    const ImageInfo info(regs.rt[index], regs.anti_alias_samples_mode);
    return FindRenderTargetView(info, gpu_addr);
}

template <class P>
ImageViewId TextureCache<P>::FindDepthBuffer() {
    const auto& regs = maxwell3d->regs;
    if (!regs.zeta_enable) {
        return ImageViewId{};
    }
    const GPUVAddr gpu_addr = regs.zeta.Address();
    if (gpu_addr == 0) {
        return ImageViewId{};
    }
    if (!gpu_memory->IsWithinGPUAddressRange(gpu_addr)) {
        LOG_WARNING(HW_GPU, "Ignoring depth buffer outside the GPU address space: {:#x}", gpu_addr);
        return ImageViewId{};
    }
    const ImageInfo info(regs.zeta, regs.zeta_size, regs.anti_alias_samples_mode);
    return FindRenderTargetView(info, gpu_addr);
}

template <class P>
ImageViewId TextureCache<P>::FindRenderTargetView(const ImageInfo& info, GPUVAddr gpu_addr) {
    ImageId image_id{};
    bool delete_state = has_deleted_images;
    do {
        has_deleted_images = false;
        image_id = FindOrInsertImage(info, gpu_addr);
        delete_state |= has_deleted_images;
        // Rendering into an unwritten sparse region (Mario Kart World's baked shadow atlas, a
        // 512x512x101 D16 array whose tiles are drawn on demand): give it memory now, or this
        // draw is lost and the tile reads as depth 0, which the ESM lookup takes as full shadow.
        // The flag also marks images refused under the UMA hard limit: only a region nobody wrote
        // yet qualifies, and marking it written keeps this from repeating.
        if (image_id) {
            const Image& found = slot_images[image_id];
            if (True(found.flags & ImageFlagBits::UnwrittenSparse) &&
                Tegra::Nx2SparseUnwritten(found.gpu_addr, found.guest_size_bytes) &&
                ReplaceUnwrittenSparseImage(image_id)) {
                has_deleted_images = true;
            }
        }
    } while (has_deleted_images);
    has_deleted_images = delete_state;
    if (!image_id) {
        return NULL_IMAGE_VIEW_ID;
    }
    Image& image = slot_images[image_id];
    const ImageViewType view_type = RenderTargetImageViewType(info);
    SubresourceBase base;
    if (image.info.type == ImageType::Linear) {
        base = SubresourceBase{.level = 0, .layer = 0};
    } else {
        base = image.TryFindBase(gpu_addr).value();
    }
    const s32 layers = image.info.type == ImageType::e3D ? info.size.depth : info.resources.layers;
    if (True(image.flags & ImageFlagBits::Unbacked)) {
        static u32 logged{};
        if (logged++ < 64) {
            LOG_INFO(HW_GPU, "Render target in unbacked image {:#x} at {:#x}: {}x{} level {} layer {}",
                     image.gpu_addr, gpu_addr, info.size.width, info.size.height, base.level, base.layer);
        }
    }
    if constexpr (requires { image.RenderTargetBelowTrim(base.level); }) {
        if (True(image.flags & ImageFlagBits::TrimmedLevels) &&
            False(image.flags & ImageFlagBits::Remapped) && image.RenderTargetBelowTrim(base.level)) {
            // This draw still lands on the trimmed image; the next lookup re-creates it whole.
            image.flags |= ImageFlagBits::CpuModified | ImageFlagBits::Remapped;
        }
    }
    const SubresourceRange range{
        .base = base,
        .extent = {.levels = 1, .layers = layers},
    };
    return FindOrEmplaceImageView(image_id, ImageViewInfo(view_type, info.format, range));
}

template <class P>
template <typename Func>
void TextureCache<P>::ForEachImageInRegion(DAddr cpu_addr, size_t size, Func&& func) {
    using FuncReturn = typename std::invoke_result<Func, ImageId, Image&>::type;
    constexpr bool BOOL_BREAK = std::is_same_v<FuncReturn, bool>;
    boost::container::small_vector<ImageId, 32> images;
    boost::container::small_vector<ImageMapId, 32> maps;
    ForEachCPUPage(cpu_addr, size, [this, &images, &maps, cpu_addr, size, func](u64 page) {
        const auto it = page_table.find(page);
        if (it == page_table.end()) {
            if constexpr (BOOL_BREAK) {
                return false;
            } else {
                return;
            }
        }
        for (const ImageMapId map_id : it->second) {
            ImageMapView& map = slot_map_views[map_id];
            if (map.picked) {
                continue;
            }
            if (!map.Overlaps(cpu_addr, size)) {
                continue;
            }
            map.picked = true;
            maps.push_back(map_id);
            Image& image = slot_images[map.image_id];
            if (True(image.flags & ImageFlagBits::Picked)) {
                continue;
            }
            image.flags |= ImageFlagBits::Picked;
            images.push_back(map.image_id);
            if constexpr (BOOL_BREAK) {
                if (func(map.image_id, image)) {
                    return true;
                }
            } else {
                func(map.image_id, image);
            }
        }
        if constexpr (BOOL_BREAK) {
            return false;
        }
    });
    for (const ImageId image_id : images) {
        slot_images[image_id].flags &= ~ImageFlagBits::Picked;
    }
    for (const ImageMapId map_id : maps) {
        slot_map_views[map_id].picked = false;
    }
}

template <class P>
template <typename Func>
void TextureCache<P>::ForEachImageInRegionGPU(size_t as_id, GPUVAddr gpu_addr, size_t size,
                                              Func&& func) {
    using FuncReturn = typename std::invoke_result<Func, ImageId, Image&>::type;
    constexpr bool BOOL_BREAK = std::is_same_v<FuncReturn, bool>;
    boost::container::small_vector<ImageId, 8> images;
    auto storage_id = getStorageID(as_id);
    if (!storage_id) {
        return;
    }
    auto& gpu_page_table = gpu_page_table_storage[*storage_id * 2];
    ForEachGPUPage(gpu_addr, size,
                   [this, &gpu_page_table, &images, gpu_addr, size, func](u64 page) {
                       const auto it = gpu_page_table.find(page);
                       if (it == gpu_page_table.end()) {
                           if constexpr (BOOL_BREAK) {
                               return false;
                           } else {
                               return;
                           }
                       }
                       for (const ImageId image_id : it->second) {
                           Image& image = slot_images[image_id];
                           if (True(image.flags & ImageFlagBits::Picked)) {
                               continue;
                           }
                           if (!image.OverlapsGPU(gpu_addr, size)) {
                               continue;
                           }
                           image.flags |= ImageFlagBits::Picked;
                           images.push_back(image_id);
                           if constexpr (BOOL_BREAK) {
                               if (func(image_id, image)) {
                                   return true;
                               }
                           } else {
                               func(image_id, image);
                           }
                       }
                       if constexpr (BOOL_BREAK) {
                           return false;
                       }
                   });
    for (const ImageId image_id : images) {
        slot_images[image_id].flags &= ~ImageFlagBits::Picked;
    }
}

template <class P>
template <typename Func>
void TextureCache<P>::ForEachSparseImageInRegion(size_t as_id, GPUVAddr gpu_addr, size_t size,
                                                 Func&& func) {
    using FuncReturn = typename std::invoke_result<Func, ImageId, Image&>::type;
    constexpr bool BOOL_BREAK = std::is_same_v<FuncReturn, bool>;
    boost::container::small_vector<ImageId, 8> images;
    auto storage_id = getStorageID(as_id);
    if (!storage_id) {
        return;
    }
    auto& sparse_page_table = gpu_page_table_storage[*storage_id * 2 + 1];
    ForEachGPUPage(gpu_addr, size,
                   [this, &sparse_page_table, &images, gpu_addr, size, func](u64 page) {
                       const auto it = sparse_page_table.find(page);
                       if (it == sparse_page_table.end()) {
                           if constexpr (BOOL_BREAK) {
                               return false;
                           } else {
                               return;
                           }
                       }
                       for (const ImageId image_id : it->second) {
                           Image& image = slot_images[image_id];
                           if (True(image.flags & ImageFlagBits::Picked)) {
                               continue;
                           }
                           if (!image.OverlapsGPU(gpu_addr, size)) {
                               continue;
                           }
                           image.flags |= ImageFlagBits::Picked;
                           images.push_back(image_id);
                           if constexpr (BOOL_BREAK) {
                               if (func(image_id, image)) {
                                   return true;
                               }
                           } else {
                               func(image_id, image);
                           }
                       }
                       if constexpr (BOOL_BREAK) {
                           return false;
                       }
                   });
    for (const ImageId image_id : images) {
        slot_images[image_id].flags &= ~ImageFlagBits::Picked;
    }
}

template <class P>
template <typename Func>
void TextureCache<P>::ForEachSparseSegment(ImageBase& image, Func&& func) {
    using FuncReturn = typename std::invoke_result<Func, GPUVAddr, DAddr, size_t>::type;
    constexpr bool RETURNS_BOOL = std::is_same_v<FuncReturn, bool>;
    const auto segments = gpu_memory->GetSubmappedRange(image.gpu_addr, image.guest_size_bytes);
    for (const auto& [gpu_addr, size] : segments) {
        std::optional<DAddr> cpu_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
        ASSERT(cpu_addr);
        if constexpr (RETURNS_BOOL) {
            if (func(gpu_addr, *cpu_addr, size)) {
                break;
            }
        } else {
            func(gpu_addr, *cpu_addr, size);
        }
    }
}

template <class P>
ImageViewId TextureCache<P>::FindOrEmplaceImageView(ImageId image_id, const ImageViewInfo& info) {
    Image& image = slot_images[image_id];
    if (const ImageViewId image_view_id = image.FindView(info); image_view_id) {
        Common::DiagTrace::Emit(Common::DiagTrace::Ev::ViewCacheHit, frame_tick,
                                image_view_id.index, DiagTraceObjOf(image_id),
                                DiagViewGenOf(image_view_id), 0);
        return image_view_id;
    }
    if (True(image.flags & ImageFlagBits::Unbacked) &&
        False(image.flags & ImageFlagBits::SparseLayers)) {
        // How the game uses an image too large to back (e.g. a sparse virtual texture).
        LOG_INFO(HW_GPU, "View of unbacked image {:#x}: type {} format {} levels {}+{} layers {}+{}",
                 image.gpu_addr, static_cast<u32>(info.type), static_cast<u32>(info.format),
                 info.range.base.level, info.range.extent.levels, info.range.base.layer,
                 info.range.extent.layers);
    }
    const ImageViewId image_view_id = slot_image_views.insert(runtime, info, image_id, image, slot_images);
    u64 diag_vgen = 0;
    // Bounded like the image identity map: existing keys always update;
    // new keys beyond the cap are skipped and counted. Fault-contained.
    Common::DiagTrace::Observe([&] {
        const u64 vgen = ++diag_view_next_gen_; // advances even when cap-skipped
        if (diag_view_gen_.size() < 262144 || diag_view_gen_.contains(image_view_id.index)) {
            diag_view_gen_[image_view_id.index] = vgen;
            diag_vgen = vgen;
        } else {
            Common::DiagTrace::S().view_gen_skipped.fetch_add(1, std::memory_order_relaxed);
        }
    });
    Common::DiagTrace::Emit(Common::DiagTrace::Ev::ViewCreate, frame_tick,
                            image_view_id.index, DiagTraceObjOf(image_id), diag_vgen,
                            static_cast<u64>(info.type));
    image.InsertView(info, image_view_id);
    return image_view_id;
}

template <class P>
void TextureCache<P>::RegisterImage(ImageId image_id) {
    ImageBase& image = slot_images[image_id];
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered),
               "Trying to register an already registered image");
    if (Tegra::Nx2TracedDestination(image.gpu_addr, image.guest_size_bytes)) {
        static u32 traced{};
        if (traced++ < 400) {
            LOG_INFO(HW_GPU, "nx2 image created in traced range (frame {}): {:#x} {}x{}x{} layers {} levels {} format {} flags {:#x} id {}",
                     frame_tick, image.gpu_addr, image.info.size.width, image.info.size.height,
                     image.info.size.depth, image.info.resources.layers, image.info.resources.levels,
                     static_cast<u32>(image.info.format), static_cast<u32>(image.flags), image_id.index);
        }
    }
    image.flags |= ImageFlagBits::Registered;
    total_used_memory += Common::AlignUp(HostImageSizeBytes(image), 1024);
    image.lru_index = lru_cache.Insert(image_id, frame_tick);
    if (True(image.flags & ImageFlagBits::Unbacked)) {
        unbacked_image_ids.push_back(image_id);
    }

    ForEachGPUPage(image.gpu_addr, image.guest_size_bytes, [this, image_id](u64 page) {
        (*channel_state->gpu_page_table)[page].push_back(image_id);
    });
    if (False(image.flags & ImageFlagBits::Sparse)) {
        auto map_id =
            slot_map_views.insert(image.gpu_addr, image.cpu_addr, image.guest_size_bytes, image_id);
        ForEachCPUPage(image.cpu_addr, image.guest_size_bytes,
                       [this, map_id](u64 page) { page_table[page].push_back(map_id); });
        image.map_view_id = map_id;
        return;
    }
    boost::container::small_vector<ImageViewId, 16> sparse_maps;
    ForEachSparseSegment(
        image, [this, image_id, &sparse_maps](GPUVAddr gpu_addr, DAddr cpu_addr, size_t size) {
            auto map_id = slot_map_views.insert(gpu_addr, cpu_addr, size, image_id);
            ForEachCPUPage(cpu_addr, size,
                           [this, map_id](u64 page) { page_table[page].push_back(map_id); });
            sparse_maps.push_back(map_id);
        });
    sparse_views.emplace(image_id, std::move(sparse_maps));
    ForEachGPUPage(image.gpu_addr, image.guest_size_bytes, [this, image_id](u64 page) {
        (*channel_state->sparse_page_table)[page].push_back(image_id);
    });
}

template <class P>
void TextureCache<P>::UnregisterImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(True(image.flags & ImageFlagBits::Registered),
               "Trying to unregister an already registered image");
    image.flags &= ~ImageFlagBits::Registered;
    image.flags &= ~ImageFlagBits::BadOverlap;
    lru_cache.Free(image.lru_index);
    if (True(image.flags & ImageFlagBits::Unbacked)) {
        std::erase(unbacked_image_ids, image_id);
    }
    const auto& clear_page_table =
        [image_id](u64 page, ::Common::unordered_map<u64, std::vector<ImageId>, Common::IdentityHash<u64>>& selected_page_table) {
            const auto page_it = selected_page_table.find(page);
            if (page_it == selected_page_table.end()) {
                ASSERT_MSG(false, "Unregistering unregistered page={:#x}", page << YUZU_PAGEBITS);
                return;
            }
            std::vector<ImageId>& image_ids = page_it->second;
            const auto vector_it = std::ranges::find(image_ids, image_id);
            if (vector_it == image_ids.end()) {
                ASSERT_MSG(false, "Unregistering unregistered image in page={:#x}",
                           page << YUZU_PAGEBITS);
                return;
            }
            image_ids.erase(vector_it);
        };
    ForEachGPUPage(image.gpu_addr, image.guest_size_bytes, [this, &clear_page_table](u64 page) {
        clear_page_table(page, (*channel_state->gpu_page_table));
    });
    if (False(image.flags & ImageFlagBits::Sparse)) {
        const auto map_id = image.map_view_id;
        ForEachCPUPage(image.cpu_addr, image.guest_size_bytes, [this, map_id](u64 page) {
            const auto page_it = page_table.find(page);
            if (page_it == page_table.end()) {
                ASSERT_MSG(false, "Unregistering unregistered page={:#x}", page << YUZU_PAGEBITS);
                return;
            }
            std::vector<ImageMapId>& image_map_ids = page_it->second;
            const auto vector_it = std::ranges::find(image_map_ids, map_id);
            if (vector_it == image_map_ids.end()) {
                ASSERT_MSG(false, "Unregistering unregistered image in page={:#x}",
                           page << YUZU_PAGEBITS);
                return;
            }
            image_map_ids.erase(vector_it);
        });
        slot_map_views.erase(map_id);
        return;
    }
    ForEachGPUPage(image.gpu_addr, image.guest_size_bytes, [this, &clear_page_table](u64 page) {
        clear_page_table(page, (*channel_state->sparse_page_table));
    });
    auto it = sparse_views.find(image_id);
    ASSERT(it != sparse_views.end());
    auto& sparse_maps = it->second;
    for (auto& map_view_id : sparse_maps) {
        const auto& map_range = slot_map_views[map_view_id];
        const DAddr cpu_addr = map_range.cpu_addr;
        const std::size_t size = map_range.size;
        ForEachCPUPage(cpu_addr, size, [this, image_id](u64 page) {
            const auto page_it = page_table.find(page);
            if (page_it == page_table.end()) {
                ASSERT_MSG(false, "Unregistering unregistered page={:#x}", page << YUZU_PAGEBITS);
                return;
            }
            std::vector<ImageMapId>& image_map_ids = page_it->second;
            auto vector_it = image_map_ids.begin();
            while (vector_it != image_map_ids.end()) {
                ImageMapView& map = slot_map_views[*vector_it];
                if (map.image_id != image_id) {
                    vector_it++;
                    continue;
                }
                if (!map.picked) {
                    map.picked = true;
                }
                vector_it = image_map_ids.erase(vector_it);
            }
        });
        slot_map_views.erase(map_view_id);
    }
    sparse_views.erase(it);
}

template <class P>
void TextureCache<P>::TrackImage(ImageBase& image, ImageId image_id) {
    ASSERT(False(image.flags & ImageFlagBits::Tracked));
    image.flags |= ImageFlagBits::Tracked;
    if (False(image.flags & ImageFlagBits::Sparse)) {
        if (image.cpu_addr < ~(1ULL << 40)) {
            device_memory.UpdatePagesCachedCount(image.cpu_addr, image.guest_size_bytes, 1);
        }
        return;
    }
    if (True(image.flags & ImageFlagBits::Registered)) {
        auto it = sparse_views.find(image_id);
        ASSERT(it != sparse_views.end());
        auto& sparse_maps = it->second;
        for (auto& map_view_id : sparse_maps) {
            const auto& map = slot_map_views[map_view_id];
            const DAddr cpu_addr = map.cpu_addr;
            const std::size_t size = map.size;
            device_memory.UpdatePagesCachedCount(cpu_addr, size, 1);
        }
        return;
    }
    ForEachSparseSegment(image,
                         [this]([[maybe_unused]] GPUVAddr gpu_addr, DAddr cpu_addr, size_t size) {
                             device_memory.UpdatePagesCachedCount(cpu_addr, size, 1);
                         });
}

template <class P>
void TextureCache<P>::UntrackImage(ImageBase& image, ImageId image_id) {
    ASSERT(True(image.flags & ImageFlagBits::Tracked));
    image.flags &= ~ImageFlagBits::Tracked;
    if (False(image.flags & ImageFlagBits::Sparse)) {
        if (image.cpu_addr < ~(1ULL << 40)) {
            device_memory.UpdatePagesCachedCount(image.cpu_addr, image.guest_size_bytes, -1);
        }
        return;
    }
    ASSERT(True(image.flags & ImageFlagBits::Registered));
    auto it = sparse_views.find(image_id);
    ASSERT(it != sparse_views.end());
    auto& sparse_maps = it->second;
    for (auto& map_view_id : sparse_maps) {
        const auto& map = slot_map_views[map_view_id];
        const DAddr cpu_addr = map.cpu_addr;
        const std::size_t size = map.size;
        device_memory.UpdatePagesCachedCount(cpu_addr, size, -1);
    }
}

template <class P>
void TextureCache<P>::DeleteImage(ImageId image_id, bool immediate_delete,
                                  const char* diag_cause) {
    ImageBase& image = slot_images[image_id];
    ++images_deleted;
    if (Tegra::Nx2TracedDestination(image.gpu_addr, image.guest_size_bytes)) {
        static u32 traced{};
        if (traced++ < 400) {
            LOG_INFO(HW_GPU, "nx2 image deleted in traced range (frame {}): {:#x} {}x{}x{} layers {} flags {:#x} id {}",
                     frame_tick, image.gpu_addr, image.info.size.width, image.info.size.height,
                     image.info.size.depth, image.info.resources.layers,
                     static_cast<u32>(image.flags), image_id.index);
        }
    }
    // Drop every decode still queued for it, whatever IsDecoding says: an image re-decoded while
    // a decode was in flight has two, and the first to finish clears the flag. A decode left
    // behind was uploaded into the next image to take this slot (another texture's data: garbage
    // textures, and a staging overrun when that image was smaller).
    for (auto& async_decode : async_decodes) {
        if (async_decode->image_id == image_id) {
            async_decode->cancelled = true;
        }
    }
    // total_used_memory is refreshed from the (discounted) driver figure every TickFrame and
    // adjusted by booked estimates in between (the two are different units: an estimate is not
    // a driver byte count), so an image's estimate can exceed the current total. Saturate
    // instead of wrapping: a wrapped total reads as over-critical to every collector threshold
    // until the next refresh. Saturation is containment, not correction: the absorbed excess
    // biases the counter toward under-collection for the rest of the frame, the next refresh
    // re-derives the anchor (it does not recover the absorbed divergence), and every saturating
    // step is logged so the mismatch stays observable. Additions stay unchecked.
    const u64 deleted_scaled_bytes = image.HasScaled() ? GetScaledImageSizeBytes(image) : 0;
    const u64 deleted_host_bytes = Common::AlignUp(HostImageSizeBytes(image), 1024);
    if (CollectorSubtract(total_used_memory, deleted_scaled_bytes)) [[unlikely]] {
        LOG_WARNING(HW_GPU, "deleted image's scaled estimate {} MiB exceeds collector total; saturating",
                    deleted_scaled_bytes >> 20);
    }
    if (CollectorSubtract(total_used_memory, deleted_host_bytes)) [[unlikely]] {
        LOG_WARNING(HW_GPU, "deleted image's host estimate {} MiB exceeds collector total; saturating",
                    deleted_host_bytes >> 20);
    }
    // Mirror of the two subtractions above, for the driver-reported path in TickFrame: the
    // driver keeps counting this memory until the ring destroys the image. Unbacked images were
    // never backed by device memory, and sparse images commit only populated pages, so their
    // estimate does not track driver bytes: keep both out of the discount. Under-discounting is
    // the safe direction - it only leaves the collector stricter, never looser.
    // Checked scaled+host sum: on (physically implausible) overflow drop the discount
    // entirely, same direction as the slot-accumulation guard below.
    u64 freed_bytes = 0;
    if (False(image.flags & ImageFlagBits::Unbacked) && False(image.flags & ImageFlagBits::Sparse)) {
        if (deleted_scaled_bytes > (std::numeric_limits<u64>::max)() - deleted_host_bytes) [[unlikely]] {
            LOG_ERROR(HW_GPU, "sentenced-image size overflowed; discount dropped");
        } else {
            freed_bytes = deleted_scaled_bytes + deleted_host_bytes;
        }
    }
    const GPUVAddr gpu_addr = image.gpu_addr;
    const auto alloc_it = image_allocs_table.find(gpu_addr);
    if (alloc_it == image_allocs_table.end()) {
        ASSERT_MSG(false, "Trying to delete an image alloc that does not exist in address {:#x}",
                   gpu_addr);
        return;
    }
    const ImageAllocId alloc_id = alloc_it->second;
    std::vector<ImageId>& alloc_images = slot_image_allocs[alloc_id].images;
    const auto alloc_image_it = std::ranges::find(alloc_images, image_id);
    if (alloc_image_it == alloc_images.end()) {
        ASSERT_MSG(false, "Trying to delete an image that does not exist");
        return;
    }
    ASSERT_MSG(False(image.flags & ImageFlagBits::Tracked), "Image was not untracked");
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered), "Image was not unregistered");

    // Mark render targets as dirty
    auto& dirty = maxwell3d->dirty.flags;
    dirty[Dirty::RenderTargets] = true;
    dirty[Dirty::ZetaBuffer] = true;
    for (size_t rt = 0; rt < NUM_RT; ++rt) {
        dirty[Dirty::ColorBuffer0 + rt] = true;
    }
    const std::span<const ImageViewId> image_view_ids = image.image_view_ids;
    for (const ImageViewId image_view_id : image_view_ids) {
        std::ranges::replace(render_targets.color_buffer_ids, image_view_id, ImageViewId{});
        if (render_targets.depth_buffer_id == image_view_id) {
            render_targets.depth_buffer_id = ImageViewId{};
        }
    }
    RemoveImageViewReferences(image_view_ids);
    RemoveFramebuffers(image_view_ids);

    for (const AliasedImage& alias : image.aliased_images) {
        ImageBase& other_image = slot_images[alias.id];
        [[maybe_unused]] const size_t num_removed_aliases =
            std::erase_if(other_image.aliased_images, [image_id](const AliasedImage& other_alias) {
                return other_alias.id == image_id;
            });
        other_image.CheckAliasState();
        ASSERT_MSG(num_removed_aliases == 1, "Invalid number of removed aliases: {}",
                   num_removed_aliases);
    }
    for (const ImageId overlap_id : image.overlapping_images) {
        ImageBase& other_image = slot_images[overlap_id];
        [[maybe_unused]] const size_t num_removed_overlaps = std::erase_if(
            other_image.overlapping_images,
            [image_id](const ImageId other_overlap_id) { return other_overlap_id == image_id; });
        other_image.CheckBadOverlapState();
        ASSERT_MSG(num_removed_overlaps == 1, "Invalid number of removed overlapps: {}",
                   num_removed_overlaps);
    }
    for (const ImageViewId image_view_id : image_view_ids) {
        if (!immediate_delete) {
            sentenced_image_view.Push(std::move(slot_image_views[image_view_id]));
        }
        Common::DiagTrace::Observe([&] {
            const u64 old_gen = DiagViewGenOf(image_view_id);
            // DeleteImage-erased views were previously silent.
            Common::DiagTrace::Emit(Common::DiagTrace::Ev::ViewErase, frame_tick,
                                    image_view_id.index, 0, 0, old_gen);
            diag_view_gen_.erase(image_view_id.index);
        });
        slot_image_views.erase(image_view_id);
    }
    const bool retained = RetainPendingImageDownloads(image_id, slot_images[image_id],
                                                       uncommitted_downloads, committed_downloads);
    if (retained) {
        LOG_INFO(Render_Vulkan, "Retained image for queued download: id={}", image_id.Value());
    } else if (!immediate_delete) {
        sentenced_images.Push(std::move(slot_images[image_id]));
        // The slot accumulates booked estimates, which can legitimately exceed physical
        // VRAM, so no physical bound applies. On (physically implausible) u64 overflow,
        // keep the previous total: this delete's discount is dropped and the collector
        // behaves as if it had not been discounted - the unported behavior.
        u64& slot_bytes = sentenced_image_bytes[sentenced_image_index];
        if (freed_bytes > (std::numeric_limits<u64>::max)() - slot_bytes) [[unlikely]] {
            LOG_ERROR(HW_GPU, "sentenced-image byte accounting overflowed; discount dropped");
        } else {
            slot_bytes += freed_bytes;
        }
    }
    if (const auto dit = diag_images_.find(image_id.index); dit != diag_images_.end()) {
        // Emitted at the actual slot-free point; address taken from the record because
        // the image may already be moved-from (sentenced).
        // D5: c= caller-supplied cause; r= backing disposition - the slot is
        // freed here, but sentenced/retained backing outlives this event.
        char diag_erase_tag[72];
        const int diag_erase_len = std::snprintf(
            diag_erase_tag, sizeof(diag_erase_tag), "as%llx c=%s r=%s",
            (unsigned long long)(dit->second.asid + 1), diag_cause,
            retained          ? "retained"
            : immediate_delete ? "immediate"
                               : "sentenced");
        if (diag_erase_len < 0 || static_cast<size_t>(diag_erase_len) >= sizeof(diag_erase_tag)) {
            Common::DiagTrace::S().tag_truncated.fetch_add(1, std::memory_order_relaxed);
            std::memcpy(diag_erase_tag + sizeof(diag_erase_tag) - 4, "~TR", 4);
        }
        Common::DiagTrace::Emit(Common::DiagTrace::Ev::ImageErase, frame_tick,
                                dit->second.trace_obj, image_id.index, dit->second.gpu_addr,
                                dit->second.created_frame, diag_erase_tag);
        dit->second.state = 1;
    }
    slot_images.erase(image_id);

    alloc_images.erase(alloc_image_it);
    if (alloc_images.empty()) {
        image_allocs_table.erase(alloc_it);
        slot_image_allocs.erase(alloc_id);
    }
    for (size_t c : active_channel_ids) {
        auto& channel_info = channel_storage[c];
        if constexpr (ENABLE_VALIDATION)
            for (auto& e : channel_info.image_view_ids)
                e.second = CORRUPT_ID;
        channel_info.graphics_image_table.Invalidate();
        channel_info.compute_image_table.Invalidate();
    }
    has_deleted_images = true;
}

template <class P>
void TextureCache<P>::RemoveImageViewReferences(std::span<const ImageViewId> removed_views) {
    for (size_t c : active_channel_ids) {
        auto& channel_info = channel_storage[c];
        auto it = channel_info.image_views.begin();
        while (it != channel_info.image_views.end()) {
            const auto found = std::ranges::find(removed_views, it->second);
            if (found != removed_views.end()) {
                it = channel_info.image_views.erase(it);
            } else {
                ++it;
            }
        }
    }
}

template <class P>
void TextureCache<P>::RemoveFramebuffers(std::span<const ImageViewId> removed_views) {
    runtime.FlushDeferredClear();
    auto it = framebuffers.begin();
    while (it != framebuffers.end()) {
        if (it->first.Contains(removed_views)) {
            auto framebuffer_id = it->second;
            ASSERT(framebuffer_id);
            if (framebuffer_id == last_framebuffer_id) {
                last_framebuffer_id = {};
                last_framebuffer_serial = 0;
            }
            sentenced_framebuffers.Push(std::move(slot_framebuffers[framebuffer_id]));
            slot_framebuffers.erase(framebuffer_id);
            it = framebuffers.erase(it);
        } else {
            ++it;
        }
    }
}

template <class P>
void TextureCache<P>::MarkModification(ImageBase& image) noexcept {
    image.flags |= ImageFlagBits::GpuModified;
    image.modification_tick = ++modification_tick;
    // The images aliasing this one will have to copy from it: let their SynchronizeAliases
    // know without walking every alias on every draw (Switch 2 titles sample images with
    // ~40 aliases each; the walk was 5% of the GPU back thread).
    for (const AliasedImage& aliased : image.aliased_images) {
        slot_images[aliased.id].alias_max_tick = image.modification_tick;
    }
}

template <class P>
void TextureCache<P>::SynchronizeAliases(ImageId image_id) {
    boost::container::small_vector<const AliasedImage*, 8> aliased_images;
    Image& image = slot_images[image_id];
    if (VideoCommon::Nx2Profile::Enabled()) [[unlikely]] {
        VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AliasSyncs, 1);
    }
    if (image.alias_max_tick <= image.modification_tick) {
        return; // no alias has been modified since this image was
    }
    if (VideoCommon::Nx2Profile::Enabled()) [[unlikely]] {
        VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AliasSyncWalked, image.aliased_images.size());
    }
    bool any_rescaled = True(image.flags & ImageFlagBits::Rescaled);
    bool any_modified = True(image.flags & ImageFlagBits::GpuModified);
    u64 most_recent_tick = image.modification_tick;
    for (const AliasedImage& aliased : image.aliased_images) {
        ImageBase& aliased_image = slot_images[aliased.id];
        if (image.modification_tick < aliased_image.modification_tick) {
            most_recent_tick = (std::max)(most_recent_tick, aliased_image.modification_tick);
            aliased_images.push_back(&aliased);
            any_rescaled |= True(aliased_image.flags & ImageFlagBits::Rescaled);
            any_modified |= True(aliased_image.flags & ImageFlagBits::GpuModified);
        }
    }
    if (aliased_images.empty()) {
        return;
    }
    const bool can_rescale = ImageCanRescale(image);
    if (any_rescaled) {
        if (can_rescale) {
            ScaleUp(image);
        } else {
            ScaleDown(image);
        }
    }
    image.modification_tick = most_recent_tick;
    if (any_modified) {
        image.flags |= ImageFlagBits::GpuModified;
    }
    std::ranges::sort(aliased_images, [this](const AliasedImage* lhs, const AliasedImage* rhs) {
        const ImageBase& lhs_image = slot_images[lhs->id];
        const ImageBase& rhs_image = slot_images[rhs->id];
        return lhs_image.modification_tick < rhs_image.modification_tick;
    });
    const auto& resolution = Settings::values.resolution_info;
    for (const AliasedImage* const aliased : aliased_images) {
        if (!resolution.active || !any_rescaled) {
            CopyImage(image_id, aliased->id, aliased->copies);
            continue;
        }
        Image& aliased_image = slot_images[aliased->id];
        if (!can_rescale) {
            ScaleDown(aliased_image);
            CopyImage(image_id, aliased->id, aliased->copies);
            continue;
        }
        ScaleUp(aliased_image);
        CopyImage(image_id, aliased->id, aliased->copies);
    }
}

template <class P>
void TextureCache<P>::PrepareImage(ImageId image_id, bool is_modification, bool invalidate) {
    Image& image = slot_images[image_id];
    if (False(image.flags & ImageFlagBits::Registered)) {
        // Dormant: not in the LRU list (touching its freed slot would corrupt the list). The
        // lookups that can still reach one re-resolve it instead of getting here.
        static u32 logged{};
        if (logged++ < 8) {
            LOG_WARNING(HW_GPU, "PrepareImage on a dormant image {:#x} {}x{}", image.gpu_addr,
                        image.info.size.width, image.info.size.height);
        }
        return;
    }
    if (invalidate) {
        image.flags &= ~(ImageFlagBits::CpuModified | ImageFlagBits::GpuModified);
        if (False(image.flags & ImageFlagBits::Tracked)) {
            TrackImage(image, image_id);
        }
    } else {
        if (True(image.flags & ImageFlagBits::CpuModified) &&
            Tegra::Nx2TracedDestination(image.gpu_addr, image.guest_size_bytes)) {
            static u32 traced{};
            if (traced++ < 400) {
                LOG_INFO(HW_GPU, "nx2 CPU upload into image in traced range: {:#x} {}x{} format {}",
                         image.gpu_addr, image.info.size.width, image.info.size.height,
                         static_cast<u32>(image.info.format));
            }
        }
        RefreshContents(image, image_id);
        if (!image.aliased_images.empty()) {
            SynchronizeAliases(image_id);
        }
    }
    if (is_modification) {
        if (Tegra::Nx2TracedDestination(image.gpu_addr, image.guest_size_bytes)) {
            static u32 traced{};
            if (traced++ < 40) {
                LOG_INFO(HW_GPU, "nx2 GPU write into image in traced range: {:#x} {}x{}x{} layers {} format {} flags {:#x}",
                         image.gpu_addr, image.info.size.width, image.info.size.height,
                         image.info.size.depth, image.info.resources.layers,
                         static_cast<u32>(image.info.format), static_cast<u32>(image.flags));
            }
            // Per-frame summary (TickFrame): which images in the range the GPU wrote this frame.
            ++traced_writes_this_frame;
            if (std::ranges::find(traced_images_this_frame, image_id) == traced_images_this_frame.end() &&
                traced_images_this_frame.size() < 8) {
                traced_images_this_frame.push_back(image_id);
                traced_images_text += fmt::format(" {:#x}/{}x{}x{}/{}", image.gpu_addr,
                                                  image.info.size.width, image.info.size.height,
                                                  image.info.resources.layers, image_id.index);
            }
        }
        if constexpr (requires { &Image::BindLayers; }) {
            if (True(image.flags & ImageFlagBits::SparseLayers)) {
                std::vector<u32> layers(static_cast<size_t>(image.info.resources.layers));
                std::iota(layers.begin(), layers.end(), 0U);
                // Layers over the image's memory cap stay unbacked (logged once).
                image.BindLayers(layers);
            }
        }
        if (True(image.flags & ImageFlagBits::UnwrittenSparse) &&
            False(image.flags & ImageFlagBits::Remapped)) {
            // The GPU renders into it: these writes are lost, later uses get a backed image.
            LOG_WARNING(HW_GPU, "GPU write into unwritten sparse image {:#x}; replacing it",
                        image.gpu_addr);
            Tegra::Nx2SparseMarkWritten(image.gpu_addr, image.guest_size_bytes);
            image.flags |= ImageFlagBits::CpuModified | ImageFlagBits::Remapped;
        }
        MarkModification(image);
    }
    lru_cache.Touch(image.lru_index, frame_tick);
}

template <class P>
bool TextureCache<P>::ReplaceUnwrittenSparseStorageImage(ImageViewId image_view_id) {
    const ImageViewBase& image_view = slot_image_views[image_view_id];
    if (image_view.IsBuffer()) {
        return false;
    }
    return ReplaceUnwrittenSparseImage(image_view.image_id);
}

template <class P>
bool TextureCache<P>::ReplaceUnwrittenSparseImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    if (False(image.flags & ImageFlagBits::UnwrittenSparse)) {
        return false;
    }
    static u32 logged{};
    if (logged++ < 32) {
        LOG_WARNING(HW_GPU, "GPU write into unwritten sparse image {:#x} {}x{}x{} layers {} format {}; replacing it",
                    image.gpu_addr, image.info.size.width, image.info.size.height,
                    image.info.size.depth, image.info.resources.layers,
                    static_cast<u32>(image.info.format));
    }
    Tegra::Nx2SparseMarkWritten(image.gpu_addr, image.guest_size_bytes);
    if (True(image.flags & ImageFlagBits::Tracked)) {
        UntrackImage(image, image_id);
    }
    UnregisterImage(image_id);
    DeleteImage(image_id, false, "replace-unwritten-sparse");
    return true;
}

template <class P>
void TextureCache<P>::PrepareImageView(ImageViewId image_view_id, bool is_modification,
                                       bool invalidate) {
    if (!image_view_id) {
        return;
    }
    const ImageViewBase& image_view = slot_image_views[image_view_id];
    if (image_view.IsBuffer()) {
        return;
    }
    PrepareImage(image_view.image_id, is_modification, invalidate);
}

template <class P>
void TextureCache<P>::CopyImage(ImageId dst_id, ImageId src_id, std::vector<ImageCopy> copies) {
    Image& dst = slot_images[dst_id];
    Image& src = slot_images[src_id];
    if (dst.info.astc_opaque != src.info.astc_opaque) {
        // BC1 and BC3 blocks differ in size, so the host can't copy between them; both hold the
        // same guest data, which is uploaded from guest memory instead.
        return;
    }
    const bool is_rescaled = True(src.flags & ImageFlagBits::Rescaled);
    if (is_rescaled) {
        ASSERT(True(dst.flags & ImageFlagBits::Rescaled));
        const bool both_2d{src.info.type == ImageType::e2D && dst.info.type == ImageType::e2D};
        const auto& resolution = Settings::values.resolution_info;
        for (auto& copy : copies) {
            copy.src_offset.x = resolution.ScaleUp(copy.src_offset.x);
            copy.dst_offset.x = resolution.ScaleUp(copy.dst_offset.x);
            copy.extent.width = resolution.ScaleUp(copy.extent.width);
            if (both_2d) {
                copy.src_offset.y = resolution.ScaleUp(copy.src_offset.y);
                copy.dst_offset.y = resolution.ScaleUp(copy.dst_offset.y);
                copy.extent.height = resolution.ScaleUp(copy.extent.height);
            }
        }
    }
    const auto dst_format_type = GetFormatType(dst.info.format);
    const auto src_format_type = GetFormatType(src.info.format);
    // F1 (CopyDep): emitted at the actual issue route, AFTER the astc_opaque
    // early return (which uploads from guest memory instead - no copy, no
    // event). c = copy-region count, NOT bytes. The tag names the ROUTE, not
    // the caller (join-alias, SynchronizeAliases and every other caller all
    // flow through here).
    if (src_format_type == dst_format_type) {
        if constexpr (HAS_EMULATED_COPIES) {
            if (!runtime.CanImageBeCopied(dst, src)) {
                Common::DiagTrace::Emit(Common::DiagTrace::Ev::CopyDep, frame_tick,
                                        DiagTraceObjOf(src_id), DiagTraceObjOf(dst_id),
                                        static_cast<u64>(copies.size()), 0, "copy-emulated");
                return runtime.EmulateCopyImage(dst, src, copies);
            }
        }
        Common::DiagTrace::Emit(Common::DiagTrace::Ev::CopyDep, frame_tick,
                                DiagTraceObjOf(src_id), DiagTraceObjOf(dst_id),
                                static_cast<u64>(copies.size()), 0, "copy-direct");
        return runtime.CopyImage(dst, src, copies);
    }
    UNIMPLEMENTED_IF(dst.info.type != ImageType::e2D);
    UNIMPLEMENTED_IF(src.info.type != ImageType::e2D);
    if (runtime.ShouldReinterpret(dst, src)) {
        Common::DiagTrace::Emit(Common::DiagTrace::Ev::CopyDep, frame_tick,
                                DiagTraceObjOf(src_id), DiagTraceObjOf(dst_id),
                                static_cast<u64>(copies.size()), 0, "copy-reinterpret");
        return runtime.ReinterpretImage(dst, src, copies);
    }
    Common::DiagTrace::Emit(Common::DiagTrace::Ev::CopyDep, frame_tick,
                            DiagTraceObjOf(src_id), DiagTraceObjOf(dst_id),
                            static_cast<u64>(copies.size()), 0, "copy-convert");
    for (const ImageCopy& copy : copies) {
        UNIMPLEMENTED_IF(copy.dst_subresource.num_layers != 1);
        UNIMPLEMENTED_IF(copy.src_subresource.num_layers != 1);
        UNIMPLEMENTED_IF(copy.src_offset != Offset3D{});
        UNIMPLEMENTED_IF(copy.dst_offset != Offset3D{});

        const SubresourceBase dst_base{
            .level = copy.dst_subresource.base_level,
            .layer = copy.dst_subresource.base_layer,
        };
        const SubresourceBase src_base{
            .level = copy.src_subresource.base_level,
            .layer = copy.src_subresource.base_layer,
        };
        const SubresourceExtent dst_extent{.levels = 1, .layers = 1};
        const SubresourceExtent src_extent{.levels = 1, .layers = 1};
        const SubresourceRange dst_range{.base = dst_base, .extent = dst_extent};
        const SubresourceRange src_range{.base = src_base, .extent = src_extent};
        PixelFormat dst_format = dst.info.format;
        if (GetFormatType(src.info.format) == SurfaceType::DepthStencil &&
            GetFormatType(dst_format) == SurfaceType::ColorTexture &&
            BytesPerBlock(dst_format) == 4) {
            dst_format = PixelFormat::A8B8G8R8_UNORM;
        }
        const ImageViewInfo dst_view_info(ImageViewType::e2D, dst_format, dst_range);
        const ImageViewInfo src_view_info(ImageViewType::e2D, src.info.format, src_range);
        const auto [dst_framebuffer_id, dst_view_id] = RenderTargetFromImage(dst_id, dst_view_info);
        Framebuffer* const dst_framebuffer = &slot_framebuffers[dst_framebuffer_id];
        const ImageViewId src_view_id = FindOrEmplaceImageView(src_id, src_view_info);
        ImageView& dst_view = slot_image_views[dst_view_id];
        ImageView& src_view = slot_image_views[src_view_id];
        [[maybe_unused]] const Extent3D expected_size{
            .width = (std::min)(dst_view.size.width, src_view.size.width),
            .height = (std::min)(dst_view.size.height, src_view.size.height),
            .depth = (std::min)(dst_view.size.depth, src_view.size.depth),
        };
        const Extent3D scaled_extent = [is_rescaled, expected_size]() {
            if (!is_rescaled) {
                return expected_size;
            }
            const auto& resolution = Settings::values.resolution_info;
            return Extent3D{
                .width = resolution.ScaleUp(expected_size.width),
                .height = resolution.ScaleUp(expected_size.height),
                .depth = expected_size.depth,
            };
        }();
        UNIMPLEMENTED_IF(copy.extent != scaled_extent);

        runtime.ConvertImage(dst_framebuffer, dst_view, src_view);
    }
}

template <class P>
void TextureCache<P>::BindRenderTarget(ImageViewId* old_id, ImageViewId new_id) {
    if (*old_id == new_id) {
        return;
    }
    if (new_id) {
        const ImageViewBase& old_view = slot_image_views[new_id];
        if (True(old_view.flags & ImageViewFlagBits::PreemtiveDownload) &&
            False(slot_images[old_view.image_id].flags & (ImageFlagBits::Unbacked | ImageFlagBits::TrimmedLevels))) {
            const PendingDownload new_download{true, 0, old_view.image_id};
            uncommitted_downloads.emplace_back(new_download);
        }
    }
    *old_id = new_id;
}

template <class P>
std::pair<FramebufferId, ImageViewId> TextureCache<P>::RenderTargetFromImage(
    ImageId image_id, const ImageViewInfo& view_info) {
    const ImageViewId view_id = FindOrEmplaceImageView(image_id, view_info);
    const ImageBase& image = slot_images[image_id];
    const bool is_rescaled = True(image.flags & ImageFlagBits::Rescaled);
    const bool is_color = GetFormatType(image.info.format) == SurfaceType::ColorTexture;
    const ImageViewId color_view_id = is_color ? view_id : ImageViewId{};
    const ImageViewId depth_view_id = is_color ? ImageViewId{} : view_id;
    Extent3D extent = MipSize(image.info.size, view_info.range.base.level);
    if (is_rescaled) {
        const auto& resolution = Settings::values.resolution_info;
        extent.width = resolution.ScaleUp(extent.width);
        if (image.info.type == ImageType::e2D) {
            extent.height = resolution.ScaleUp(extent.height);
        }
    }
    const u32 num_samples = image.info.num_samples;
    const auto [samples_x, samples_y] = SamplesLog2(num_samples);
    const FramebufferId framebuffer_id = GetFramebufferId(RenderTargets{
        .color_buffer_ids = {color_view_id},
        .depth_buffer_id = depth_view_id,
        .size = {extent.width >> samples_x, extent.height >> samples_y},
        .is_rescaled = is_rescaled,
    });
    return {framebuffer_id, view_id};
}

template <class P>
bool TextureCache<P>::IsFullClear(ImageViewId id) {
    if (!id) {
        return true;
    }
    const ImageViewBase& image_view = slot_image_views[id];
    const ImageBase& image = slot_images[image_view.image_id];
    const Extent3D size = image_view.size;
    const auto& regs = maxwell3d->regs;
    const auto& scissor = regs.scissor_test[0];
    if (image.info.resources.levels > 1 || image.info.resources.layers > 1) {
        // Images with multiple resources can't be cleared in a single call
        return false;
    }
    if (regs.clear_control.use_scissor == 0) {
        // If scissor testing is disabled, the clear is always full
        return true;
    }
    // Make sure the clear covers all texels in the subresource
    return scissor.min_x == 0 && scissor.min_y == 0 && scissor.max_x >= size.width &&
           scissor.max_y >= size.height;
}

template <class P>
void TextureCache<P>::CreateChannel(struct Tegra::Control::ChannelState& channel) {
    VideoCommon::ChannelSetupCaches<TextureCacheChannelInfo>::CreateChannel(channel);
    const auto it = channel_map.find(channel.bind_id);
    auto* this_state = &channel_storage[it->second];
    const auto& this_as_ref = address_spaces[channel.memory_manager->GetID()];
    this_state->gpu_page_table = &gpu_page_table_storage[this_as_ref.storage_id * 2];
    this_state->sparse_page_table = &gpu_page_table_storage[this_as_ref.storage_id * 2 + 1];
}

/// Bind a channel for execution.
template <class P>
void TextureCache<P>::OnGPUASRegister([[maybe_unused]] size_t map_id) {
    gpu_page_table_storage.emplace_back();
    gpu_page_table_storage.emplace_back();
}

} // namespace VideoCommon
