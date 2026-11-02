# Eglantine on the Steam Deck

This is the state of my fork of [Eglantine](https://github.com/Djihads80/Eglantine) as of 1 November 2026, three days after I started it. The goal is one thing: run Switch 2 games natively on a Steam Deck, with the Deck's own controls, the way the Deck already runs Switch 1 games. Everything below was measured on one Deck running SteamOS 3.8, with `eden-cli` (the SDL frontend, no Qt).

## Where it is

- **Donkey Kong Bananza** boots, gets through the menus, and is playable in the opening cave at 15-29 fps depending on the area. The diggable terrain renders (that took two days and three wrong fixes, see below). It used to freeze on every big terrain break (the pressure plate at the start): the present call returned after the *previous* frame's fence so the game could run a frame ahead, and within a frame of a big break the game recycled command memory the GPU thread had not read yet (a submit-time hash of every pushbuffer entry shows the entries rewritten 34 ms after submit). Any frame of overlap does it, even waiting for the previous frame on every channel; the present now waits for the current frame's fence and the breaks play.
- **Mario Kart World** boots, plays its opening movie, goes through the Grand Prix menus and runs a full race with 24 karts, items, HUD and minimap. 18 fps in the race on the scripted Mushroom Cup run with the default present pacing (the call waits for the frame's own fence, which Donkey Kong Bananza needs, see below); 25 fps with `EDEN_NX2_FRAMES_AHEAD=a`, which lets the game run one frame ahead of the GPU thread once every GPU channel has finished the previous frame — safe for Mario Kart World in 45 minutes of checks, not for DK. The ~90 ms stall every six or so frames turned out to be a stale release prediction on the GPU front thread, now off by default. Trail ribbons, culling quads and a few splats are missing because they are task/mesh programs we skip. Some ground textures are darker than they should be.
- **Memory holds.** With the fences described below, a full MKW race sits at 4.6-5.6 GB of GTT and the Deck stays up; before them it locked the machine hard every time it reached the menus.
- Other titles have barely been tried. Welcome Tour and Splatoon Raiders have been launched but not worked on.

Nothing here is "console smooth". It is a Deck playing Switch 2 games at half to a third of their target frame rate, with bugs.

## Why the Deck is a hard target

A Switch 2 is a 12 GB LPDDR5X machine (about 9 GB of it available to the game) with an NVIDIA Ampere GPU and eight Cortex-A78C cores. The games are built against that: NVN2, the Ampere shader ISA, task/mesh shaders, tensor-core upscalers, sparse textures with multi-gigabyte address-space reservations, and GPU-written command lists.

A Steam Deck is a 16 GB LPDDR5 machine with four Zen 2 cores (eight threads) and an 8 CU RDNA2 GPU, driven by RADV. That 16 GB is one pool shared by the CPU and the GPU, and `amdgpu` only lets the GPU take half of it: the GTT pool is 8 GB.

Those numbers collide like this:

- **A Switch 2 game's guest memory alone is 7+ GB of host RSS.** Donkey Kong Bananza asks for a ~9.5 GiB heap at boot (the Switch 1 kernel limits in the emulator rejected it; we bound them by the intended memory size instead). On a 16 GB machine that already leaves less than half for the emulator, the JIT, the shader caches and every host-side GPU resource.
- **Every GPU resource costs twice on a UMA host.** The guest's texture pool lives in RAM as guest memory, and its Vulkan image lives in RAM again as the host copy. Mario Kart World keeps ~2.6 GB of virtual-texture pools resident every frame (four 8192² RG16 and three 16384² R8 pools at 341 MiB each) plus 0.5-1.1 GB of GPU-decoded ASTC, and reserves 17 GB of GPU address space it never fully touches. A desktop GPU with its own VRAM never notices; the Deck does.
- **Hitting the GTT pool is not a slowdown, it is a lockup.** When the GPU pool fills, `amdgpu_gem_create` starts evicting and swapping inside the ioctl, the GPU thread goes into D state, `kill -9` leaves a zombie holding 8 GB of GTT, SSH stops answering and only the power button helps. I rebooted the Deck a lot the first two nights.
- **GPU accuracy affects threading.** On this branch, High GPU accuracy (the
  default in some saved configurations) prevents the GPU front/back split from
  starting. Low GPU accuracy allows the split with default DMA accuracy and
  Vulkan; look for "GPU thread split: engine methods run on a second thread"
  in the log. This changes emulation accuracy, so check rendering and
  stability before using it for a game.
- **The CPU is thin.** Four guest cores, a GPU front thread, a GPU back thread, a video decoder and audio all share four physical cores. Switch 2 titles are heavier on the CPU than Switch 1 titles. Check that SMT is actually on: a PowerTools profile had quietly turned it off on my Deck, and every number from the first two days was measured on four threads.
- **RDNA2 is not Ampere.** Eglantine lowers Ampere programs to Maxwell IR and then to SPIR-V. Anything that has no Maxwell equivalent (task/mesh shaders, HMMA tensor ops, some newer opcodes) has to be emulated or replaced, and RADV has its own opinions about sparse images on an integrated GPU.

## What I did about it

Most of this is on by default only when the Vulkan device is integrated (`Device::GetUmaMemoryBudgetCap() != 0`), so it should not change behaviour on a desktop GPU. Each one has an environment knob for A/B testing; see the commit messages.

Memory:

- Texture collector budget = 35% of physical RAM instead of the driver's "budget", which on an APU is the whole machine.
- Hard limit: an image of 8 MiB or more that would push device memory past the cap is created unbacked (reads zero) instead of taking the machine down. Render targets, depth images, layered material arrays and guest-memory textures are exempt.
- Staging pool capped at 768 MiB; big uploads release their staging as soon as the GPU is done.
- Images living in GPU-only regions are never written back to guest memory when evicted (nothing on the CPU ever reads them).
- GPU-decoded ASTC over 8 Mi texels drops its top mips; at a downscaled render resolution, textures of 64 MiB and up drop the mip levels derivative LOD can never reach. MKW's nine 341 MiB pools become 85 MiB each.
- Sparse address-space reservations (the 17 GB) are left unbacked; sparse blocks get memory only where the GPU writes, and terrain arrays are bound layer by layer (this part came from upstream's `nx2-experimental`).
- Switch 2 processes get a 16 GiB heap region, and zero-filled heap pages are released instead of committed.

GPU command stream (the front/back split that lets the pushbuffer parse run ahead of the GPU):

- Donkey Kong's diggable terrain: its indirect-draw records are reset by a vertex program that stores through pointers loaded from memory, which the recompiler cannot map to a storage buffer; we do those stores on the CPU and flush the record words before the front reads them as macro parameters. Ported from zortos293's fork, then narrowed to only the record entries so it does not drain the back thousands of times a second.
- Pushbuffer desync hardening: an out-of-range syncpoint id or an unknown class bind now drops the torn entry and logs instead of crashing the GPU back thread (this was the DK "freeze" in the pause menu, which was really a segfault followed by four minutes of core dumping).
- The front notes every copy-engine, P2MF and inline upload it sends and drains the back before parsing a command list one of them writes.
- The compute QMD release mirror only predicts releases for the QMD it was signalled on; the SDK patches chained QMDs after submitting them and the front mirror was reading them too early. This is the MKW stall fix and is still being validated.

Shaders:

- ISBERD was lowered as `slot / vertex_count`; the GS actually computes `slot = base * count + i`, so DK's culling geometry shader saw vertices (0, 1, 0) of every triangle and collected the wrong indices. Fixed to a plain register move. Upstream has the same bug open as [#158](https://github.com/Djihads80/Eglantine/issues/158).
- A passthrough geometry shader with an empty output map still rasterizes: it only culls via `ViewportMask 0`. My earlier "emits nothing" reading was wrong and is now behind a knob.
- `SURED` (0x59f) lowered to Maxwell's surface reduction; geometry-stage Ampere ops (`SR_INVOCATION_INFO`, `LEPC`, `BMOV`).
- The two HMMA tensor-core upscalers in DK are replaced by a bilinear Fermi2D blit.
- `nx2-shader-check` (`tools/nx2/`, built with `-DENABLE_DEBUG_TOOLS=ON`) lowers a directory of dumped native programs (`EDEN_NX2_DUMP_DIR`) to SPIR-V offline, so a translator change can be checked against thousands of real programs in seconds before a game is launched.

Video: the FFmpeg decoder runs on its own thread, skips frames the client is already late for, and uses four decode threads.

## Building for the Deck

SteamOS has a read-only root and no compiler, so build in a rootless distrobox on the Deck itself (Ubuntu 24.04 matches SteamOS 3.8's glibc 2.39 / libstdc++ 13) and run the resulting `eden-cli` on the host. The general [build guide](Build.md) applies; the Ubuntu specifics are: CMake 3.31 is needed and Ubuntu ships 3.28, so install it under `/opt`; do not install `libfmt-dev` or `nlohmann-json3-dev`, or CMake picks Ubuntu's fmt 9 over the bundled fmt 12 and the build fails in `common/logging.h`. A full build takes about 45 minutes at `-j5`. The host's ffmpeg and boost sonames moved in SteamOS 3.8, so copy the container's `libavcodec`, `libavutil`, `libswscale` and `libboost_*` next to the binary and add that directory to `LD_LIBRARY_PATH`, or build them statically. Stop the emulator before a large rebuild: a Switch 2 title holds 7+ GB and the compilers push the Deck into swap until SSH drops.

## Measured against upstream

Before publishing I ran upstream's own `nx2-experimental` build 113 (`51d798ff03`, the Linux x86_64 PGO AppImage from the release page) on the same Deck, same settings, same save, driven by the same scripted Mario Kart World Mushroom Cup run as this fork. Two days earlier plain upstream could not boot either title on the Deck (`SetHeapSize` for a ~9.5 GiB heap was rejected, abort 2001-0101); build 113 carries the same heap-limit fix, so it now boots and races.

| MKW, Mushroom Cup 50cc, 1 Nov 2026 | this fork (two runs) | upstream build 113 |
|---|---|---|
| Race fps samples | 16-23, average 24.2 and 22.1 (18 with the present wait DK needs; 25 with `EDEN_NX2_FRAMES_AHEAD=a`) | 15-17, average 18.4 |
| Menu fps peak | 60 | 51 |
| Emulator RSS + swap at the end of the race | 3.5-5.4 GB + 2.7-2.9 GB | 2.0 GB + 1.1 GB |
| Draws skipped (no pipeline yet) | 15-18 | 64 |
| Guest panics | 0 in the day's runs | 1 of 2 runs (`2162-0001` while idle in the main menu) |
| Race renders | yes | yes |

So for Mario Kart World: this fork is somewhat faster and did not crash, and upstream uses noticeably less memory (its new iGPU allocator work is real). Neither is a large gap.

Donkey Kong Bananza is a different story. Same build 113, same Deck, same boot-to-gameplay script, two runs:

| DK Bananza, 1 Nov 2026 | this fork | upstream build 113 |
|---|---|---|
| Reaches gameplay | yes, 14-23 fps in the opening cave | no: black screen, then `SIGSEGV` at about 3 minutes, both runs |
| Emulator RSS | ~6.2 GB, no swap | 2 GB at 60 s, 5.5 GB at 90 s, 11-13.5 GB at 2 minutes, 10.9 GB peak + 2.7 GB swap |
| Shaders rejected | 0 | 2 (`op 305`, native float-to-integer conversion mode) |
| `Storage buffer index 16/17 exceeds maximum` | 0 | ~1900 per run (`NUM_STORAGE_BUFFERS` is 16 upstream, 32 here) |

The memory curve is the game committing its ~9.5 GiB heap: on a 16 GB machine that is fatal unless zero-filled heap pages are released instead of committed, which is what the kernel/memory changes below do. Upstream's own iGPU work does not cover it yet. This is the clearest case of the fork doing something a Deck needs that upstream does not; the Deck toolchain and the pad mapping in `eden-cli` are the other parts upstream does not have at all.

## Regressions and divergence from upstream

This is the part I want to be upfront about. Upstream has had many people landing commits every hour while I had one Deck and a narrow goal. The fork is based on `origin/nx2` from 30 October plus a merge of `origin/nx2-experimental` from 31 October. Since then:

- `origin/nx2` is about 100 commits ahead of that base. Things it has that this fork does not: sparse texture tile mapping (`nx2:gpu 216`), the `MappedBacking` fix for the GPU/guest page-table deadlock (we fixed the same deadlock differently), the Switch 2 heap-size limit (same fix as ours, landed independently), VRAM kept under the driver budget and an iGPU budget within its memory share (again, parallel to our fences), GPU buffer readback to the CPU, AV1 movies, FSR 3/4 and frame generation, the AMD green/blue patch fix, the Mario Kart World text fix, Pokopia booting, loose-dump discovery and icons, the macOS and Linux aarch64 fixes, PGO builds, NUP containers.
- `origin/nx2-experimental` is about 310 commits ahead of this fork and already has what I was about to build next: task/mesh programs drawn through `VK_EXT_mesh_shader` (RADV on the Deck supports it), roughly fifty more Ampere opcode lowerings, 16 MiB memory blocks and a stream buffer that stays inside a small iGPU heap, command lists executed as submitted, and a long list of Switch 2 service fixes.
- 38 files were changed on both sides. A trial merge of `origin/nx2` compiles after about a dozen conflict resolutions, mostly in `nx2_gsv.h`, the texture cache and the Vulkan device, but has not been run on the Deck yet. One deliberate deviation in it: upstream disables unbound sparse images on RADV; on the Deck that would back the 17 GB terrain arrays whole and lock the machine, so the fork keeps them allowed.
- Only the Deck has been tested. Mac builds compile; Windows, Android and the Qt frontend have not been built or run since the fork. Some changes are not gated to integrated GPUs and apply everywhere: the passthrough geometry shader semantics, the Switch 2 command-list re-read being off by default (`EDEN_NX2_CMDFLUSH=1` restores it), the 16 GiB heap region for Switch 2 processes, and the Qt frontend no longer asking for keys at startup.
- About 60 `EDEN_NX2_*` environment knobs were added, most of them diagnostics in the rasterizer and texture cache. All default off, but they are noise in hot code and I would not want them upstreamed as they are.
- Things that are still simply wrong: MKW's mesh draws are missing, some MKW ground textures are dark, two MKW compute shaders are rejected (opcode 0x927), `nx2:camera 29` and a few `nx2:gsv` commands are auto-stubbed.

If you want the newest emulator, use upstream. If you want to run it on a Deck today, or work on the Deck-specific problems, this fork is where that work is.

## Disclosure

This fork was written with heavy use of AI coding assistants (Claude), with me directing, testing on the hardware and deciding what to chase. The commits do not yet carry the `Assisted-by:` trailer that [Eglantine's AI policy](policies/AI.md) asks for; I will add it to anything I propose upstream, and nothing from here should be merged upstream without a human reading it first.

## Help

I am one person with one Deck. The things that would move this the most:

- Someone with a second Deck (or any 16 GB APU Linux machine) to reproduce the numbers above and tell me what differs.
- Rebasing onto current `nx2-experimental`, which means re-deriving the memory fences on top of upstream's new allocator work and dropping our mesh-shader plans in favour of theirs.
- RADV people: sparse image behaviour on Van Gogh, and whether there is a saner way than the hard limit to keep a UMA game under 8 GB of GTT.
- Anyone who understands the NVN2 QMD chaining and the GPU semaphore handshake better than I do; the DK freeze and the MKW stall are both there.

The repository is [MFS-code/Eglantine-steamdeck](https://github.com/MFS-code/Eglantine-steamdeck).
