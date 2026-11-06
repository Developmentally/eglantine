#!/bin/bash
# Reproducible local build+run of the criterion-2 / DiagTrace fixture.
#
# Pins (fetched by --fetch; versions from cpmfile.json + release pages):
#   Catch2 v3.13.0 (amalgamated hpp+cpp), fmt 12.1.0, boost 1.91.0 (full
#   release), Vulkan-Headers v1.4.345, VulkanMemoryAllocator v3.3.0
#   (renderer TUs only, not this fixture).
# Checkout policy: with EXPECT_SHA set, the exact SHA and a clean tree are
# REQUIRED; without it the build is local iteration and is labeled "not
# exact-SHA evidence" in the output.
# EDEN_NX2_TRACE_TEST_HOOKS is defined for the fixture build ONLY; shipping
# builds contain zero hook code.
set -euo pipefail
REPO=${REPO:-/tmp/wt-test3}
CXX=${CXX:-g++}

if [ "${1:-}" = "--fetch" ]; then
  cd /tmp
  mkdir -p catchshim/catch2
  curl -fsSL -o catchshim/catch_amalgamated.hpp https://github.com/catchorg/Catch2/releases/download/v3.13.0/catch_amalgamated.hpp
  curl -fsSL -o catchshim/catch_amalgamated.cpp https://github.com/catchorg/Catch2/releases/download/v3.13.0/catch_amalgamated.cpp
  printf '#pragma once\n#include "../catch_amalgamated.hpp"\n' > catchshim/catch2/catch_test_macros.hpp
  rm -rf fmt fmt-12.1.0 && curl -fsSL https://github.com/fmtlib/fmt/archive/refs/tags/12.1.0.tar.gz | tar xz && mv fmt-12.1.0 fmt
  rm -rf boost_1_91_0 && curl -fsSL -o boost_1_91_0.tar.gz https://archives.boost.io/release/1.91.0/source/boost_1_91_0.tar.gz && tar xzf boost_1_91_0.tar.gz && rm boost_1_91_0.tar.gz
  rm -rf vulkan-headers && git clone -q --depth 1 --branch v1.4.345 https://github.com/KhronosGroup/Vulkan-Headers.git vulkan-headers
  curl -fsSL -o vk_mem_alloc.h https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/v3.3.0/include/vk_mem_alloc.h
  # Resolved-ref + content-hash manifest: later non-fetch runs VERIFY against it.
  M=/tmp/fixture-deps.sha256
  {
    echo "# fixture dependency manifest - resolved refs and content hashes"
    echo "catch_amalgamated.hpp v3.13.0 $(sha256sum catchshim/catch_amalgamated.hpp | cut -d' ' -f1)"
    echo "catch_amalgamated.cpp v3.13.0 $(sha256sum catchshim/catch_amalgamated.cpp | cut -d' ' -f1)"
    echo "catch_test_macros.hpp wrapper $(sha256sum catchshim/catch2/catch_test_macros.hpp | cut -d' ' -f1)"
    echo "fmt 12.1.0 $(find fmt -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
    echo "boost 1.91.0 $(find boost_1_91_0 -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
    echo "vulkan-headers $(git -C vulkan-headers rev-parse HEAD) $(find vulkan-headers/include -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
    echo "vk_mem_alloc.h v3.3.0 $(sha256sum vk_mem_alloc.h | cut -d' ' -f1)"
  } > "$M"
  echo "externals fetched; manifest written to $M"
  # Expected-anchor: pinned reproduction can require the freshly resolved
  # manifest to equal an externally supplied hash, so the fetch itself is
  # anchored instead of self-attesting.
  if [ -n "${EXPECT_DEPS_SHA256:-}" ]; then
    actual_m=$(sha256sum "$M" | cut -d' ' -f1)
    [ "$actual_m" = "$EXPECT_DEPS_SHA256" ] || { echo "FATAL: fetched manifest $actual_m != expected anchor $EXPECT_DEPS_SHA256"; exit 1; }
    echo "fetch anchor verified: manifest sha256 matches EXPECT_DEPS_SHA256"
  fi
  exit 0
fi

ACTUAL_SHA=$(cd "$REPO" && git rev-parse HEAD)
STATUS_OUT=$(cd "$REPO" && git status --porcelain)
echo "invocation: $0 $* REPO=$REPO CXX=$CXX EXPECT_SHA=${EXPECT_SHA:-<unset>} (pid $$, $(date -u +%FT%TZ))"
echo "script sha256: $(sha256sum "$0" | cut -d' ' -f1)"
echo "git-status --porcelain sha256: $(printf '%s' "$STATUS_OUT" | sha256sum | cut -d' ' -f1) (empty string hashes e3b0c442... = clean)"
if [ -n "${EXPECT_SHA:-}" ]; then
  # Externally pinned reproduction: exact checkout AND clean tree required.
  EXPECT_SHA_WAS_SUPPLIED=1
  echo "pin: exact (EXPECT_SHA supplied)"
  [ "$ACTUAL_SHA" = "$EXPECT_SHA" ] || { echo "FATAL: checkout $ACTUAL_SHA != expected $EXPECT_SHA"; exit 1; }
  [ -z "$STATUS_OUT" ] || { echo "FATAL: dirty tree"; exit 1; }
else
  EXPECT_SHA=$ACTUAL_SHA
  echo "pin: none (local iteration; not exact-SHA evidence)"
  [ -z "$STATUS_OUT" ] && TREE=clean || TREE="DIRTY (local iteration; not exact-SHA evidence)"
  echo "NOTE: no EXPECT_SHA supplied - tree is $TREE"
fi
# Verify fetched dependencies against the manifest. In pinned mode the
# manifest is REQUIRED - an absent manifest silently skipping verification
# was a provenance gap.
if [ -f /tmp/fixture-deps.sha256 ]; then
  dep_fail=0
  # Required entry set: a self-consistent but incomplete manifest (empty or a
  # subset) must NOT pass pinned verification.
  for req in catch_amalgamated.hpp catch_amalgamated.cpp catch_test_macros.hpp fmt boost vulkan-headers vk_mem_alloc.h; do
    grep -q "^$req " /tmp/fixture-deps.sha256 || { echo "FATAL: manifest missing required entry $req"; dep_fail=1; }
  done
  while read -r name ref hash; do
    case "$name" in ''|\#*) continue;; esac
    case "$name" in
      catch_amalgamated.hpp) actual=$(cd /tmp && sha256sum catchshim/catch_amalgamated.hpp | cut -d' ' -f1);;
      catch_amalgamated.cpp) actual=$(cd /tmp && sha256sum catchshim/catch_amalgamated.cpp | cut -d' ' -f1);;
      catch_test_macros.hpp) actual=$(cd /tmp && sha256sum catchshim/catch2/catch_test_macros.hpp | cut -d' ' -f1);;
      fmt) actual=$(cd /tmp && find fmt -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1);;
      boost) actual=$(cd /tmp && find boost_1_91_0 -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1);;
      vulkan-headers)
        actual=$(cd /tmp && find vulkan-headers/include -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
        actual_ref=$(git -C /tmp/vulkan-headers rev-parse HEAD)
        if [ "$actual_ref" != "$ref" ]; then echo "FATAL: vulkan-headers ref mismatch (manifest $ref, actual $actual_ref)"; dep_fail=1; fi;;
      vk_mem_alloc.h) actual=$(cd /tmp && sha256sum vk_mem_alloc.h | cut -d' ' -f1);;
      *) continue;;
    esac
    if [ "$actual" != "$hash" ]; then echo "FATAL: dependency $name content mismatch (manifest $hash, actual $actual)"; dep_fail=1; fi
  done < /tmp/fixture-deps.sha256
  [ "$dep_fail" = 0 ] || exit 1
  manifest_actual=$(sha256sum /tmp/fixture-deps.sha256 | cut -d' ' -f1)
  echo "dependency hashes verified against /tmp/fixture-deps.sha256 (sha256 $manifest_actual)"
  if [ -n "${EXPECT_SHA_WAS_SUPPLIED:-}" ]; then
    if [ -z "${EXPECT_DEPS_SHA256:-}" ]; then
      echo "FATAL: pinned mode requires EXPECT_DEPS_SHA256 (the externally supplied manifest anchor)"; exit 1
    fi
    [ "$manifest_actual" = "$EXPECT_DEPS_SHA256" ] || { echo "FATAL: manifest $manifest_actual != pinned anchor EXPECT_DEPS_SHA256 $EXPECT_DEPS_SHA256"; exit 1; }
    echo "pinned anchor verified: manifest sha256 matches EXPECT_DEPS_SHA256"
  fi
elif [ -n "${EXPECT_SHA_WAS_SUPPLIED:-}" ]; then
  echo "FATAL: pinned mode (EXPECT_SHA supplied) but /tmp/fixture-deps.sha256 is missing - dependencies cannot be verified"; exit 1
else
  echo "NOTE: /tmp/fixture-deps.sha256 missing - dependency contents UNVERIFIED (local iteration only)"
fi

OBJ=/tmp/tobj-${EXPECT_SHA:0:12}
rm -rf "$OBJ" && mkdir -p "$OBJ"   # clean per-SHA build dir: no stale object reuse
echo "fixture build: repo=$REPO sha=$EXPECT_SHA objdir=$OBJ"
echo "compiler: $($CXX --version | head -1)"
SHIM_HASH=$(sha256sum /tmp/fixture-link-shims.cpp | cut -d' ' -f1)
EXPECT_SHIM_HASH=${EXPECT_SHIM_HASH:-b6c9d74395e652341da266b8e4d1110ed0bc22bbdad40793cd71713560b1ae67}
echo "shims sha256: $SHIM_HASH"
if [ "$SHIM_HASH" != "$EXPECT_SHIM_HASH" ]; then
  echo "FATAL: fixture-link-shims.cpp hash mismatch (expect $EXPECT_SHIM_HASH)"; exit 1
fi
FLAGS="-O1 -std=c++23 -DEDEN_NX2_TRACE_TEST_HOOKS -I $REPO/src -I /tmp/catchshim -I /tmp/fmt/include -I /tmp/boost_1_91_0"
UNITS="src/core/device_memory.cpp src/video_core/host1x/gpu_device_memory_manager.cpp \
src/video_core/buffer_cache/buffer_cache.cpp src/video_core/control/channel_state_cache.cpp \
src/common/logging.cpp src/common/host_memory.cpp src/common/assert.cpp \
src/common/fs/fs.cpp src/common/fs/file.cpp src/common/fs/fs_util.cpp \
src/common/fs/path_util.cpp src/common/fs/symlink.cpp src/common/settings.cpp \
src/common/settings_common.cpp src/common/sparse_large_vector.cpp src/common/time_zone.cpp"
cd "$REPO"
MANIFEST="$OBJ/build-manifest.txt"
{
  echo "# build-input record for sha=$EXPECT_SHA tree=${TREE:-clean-pinned}"
  echo "compiler: $($CXX --version | head -1)"
  echo "flags: $FLAGS"
  echo "fixture source: src/tests/core/gpu_dirty_memory_manager.cpp $(sha256sum src/tests/core/gpu_dirty_memory_manager.cpp | cut -d' ' -f1)"
  echo "link: $CXX -O1 OBJ/fixture_test.o OBJ/catch_amalgamated.o OBJS -o /tmp/fixture_test"
  echo "deps manifest: /tmp/fixture-deps.sha256 sha256 $(sha256sum /tmp/fixture-deps.sha256 2>/dev/null | cut -d' ' -f1)"
  sed 's/^/dep: /' /tmp/fixture-deps.sha256 2>/dev/null || echo "dep: <manifest missing>"
  echo "shims: /tmp/fixture-link-shims.cpp sha256 $SHIM_HASH"
  # Record the actual relevant env at build/run time; never assume unset.
  echo "env: EDEN_NX2_CHUNK_BATCH=${EDEN_NX2_CHUNK_BATCH:-<unset>} EDEN_NX2_TRACE=${EDEN_NX2_TRACE:-<unset>} EDEN_NX2_TRACE_DRAWS=${EDEN_NX2_TRACE_DRAWS:-<unset>} EDEN_NX2_TRACE_FROM_FRAME=${EDEN_NX2_TRACE_FROM_FRAME:-<unset>} EDEN_NX2_TRACE_TO_FRAME=${EDEN_NX2_TRACE_TO_FRAME:-<unset>}"
} > "$MANIFEST"
OBJS=""
for f in $UNITS /tmp/fixture-link-shims.cpp; do
  n=$(echo "$f" | tr '/' '_'); n=${n%.cpp}
  echo "$CXX -c $FLAGS $f -o $OBJ/$n.o" >> "$MANIFEST"
  $CXX -c $FLAGS "$f" -o "$OBJ/$n.o"
  OBJS="$OBJS $OBJ/$n.o"
done
$CXX -c -O1 -std=c++23 -I /tmp/fmt/include /tmp/fmt/src/format.cc -o "$OBJ/fmt_format.o"; OBJS="$OBJS $OBJ/fmt_format.o"
$CXX -c -O1 -std=c++23 -I /tmp/fmt/include /tmp/fmt/src/os.cc -o "$OBJ/fmt_os.o"; OBJS="$OBJS $OBJ/fmt_os.o"
for f in /tmp/boost_1_91_0/libs/filesystem/src/*.cpp; do
  n=boostfs_$(basename "$f" .cpp)
  $CXX -c -O1 -std=c++23 -I /tmp/boost_1_91_0 "$f" -o "$OBJ/$n.o"
  OBJS="$OBJS $OBJ/$n.o"
done
$CXX -c -O1 -std=c++23 -I /tmp/catchshim /tmp/catchshim/catch_amalgamated.cpp -o "$OBJ/catch_amalgamated.o"
$CXX -c $FLAGS src/tests/core/gpu_dirty_memory_manager.cpp -o "$OBJ/fixture_test.o"
$CXX -O1 "$OBJ/fixture_test.o" "$OBJ/catch_amalgamated.o" $OBJS -o /tmp/fixture_test
echo "build-input record (inputs + hashes, not a complete command transcript): $MANIFEST (sha256 $(sha256sum "$MANIFEST" | cut -d' ' -f1))"

# Tagged batches: ReadEnvOnce latches per process, so trace-on, trace-off and
# init-fail cases need separate processes. Extra args select one named batch.
if [ $# -gt 0 ]; then
  shift 0
  exec /tmp/fixture_test "$@"
fi
echo "== batch [trace-init-fail] =="
env -u EDEN_NX2_TRACE_DRAWS -u EDEN_NX2_TRACE_FROM_FRAME -u EDEN_NX2_TRACE_TO_FRAME -u EDEN_NX2_TRACE_FLUSH_EVERY -u EDEN_NX2_TRACE_MAX_LINES EDEN_NX2_TRACE=1 EDEN_NX2_TRACE_TEST_FAIL_INIT=1 /tmp/fixture_test "[trace-init-fail]" --reporter compact -s
echo "== batch [trace-on] =="
env -u EDEN_NX2_TRACE_TEST_FAIL_INIT -u EDEN_NX2_TRACE_DRAWS -u EDEN_NX2_TRACE_FROM_FRAME -u EDEN_NX2_TRACE_TO_FRAME -u EDEN_NX2_TRACE_FLUSH_EVERY -u EDEN_NX2_TRACE_MAX_LINES EDEN_NX2_TRACE=1 /tmp/fixture_test "[trace-on]" --reporter compact -s
echo "== batch [trace-off]+[gpu_dirty] =="
env -u EDEN_NX2_TRACE -u EDEN_NX2_TRACE_TEST_FAIL_INIT -u EDEN_NX2_TRACE_DRAWS -u EDEN_NX2_TRACE_FROM_FRAME -u EDEN_NX2_TRACE_TO_FRAME -u EDEN_NX2_TRACE_FLUSH_EVERY -u EDEN_NX2_TRACE_MAX_LINES /tmp/fixture_test "[trace-off],[gpu_dirty],[diagtrace]" --reporter compact -s
