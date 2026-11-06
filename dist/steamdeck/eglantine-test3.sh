#!/usr/bin/env bash
# eglantine test 3 - diagnostic capture launcher.
#
# Evidence-gathering build for the missing-text investigation: it adds trace
# instrumentation only. It is NOT a fix and NOT an FPS change.
#
# Wraps eglantine-deck.sh with:
#   - its own settings file (eglantine-test3.ini) so the run never rewrites
#     the working ini;
#   - an explicit, fail-closed seed: the run REFUSES to start until you seed
#     eglantine-test3.ini from your actual working settings with
#       ./eglantine-test3.sh --seed-from <path to your working eglantine-deck.ini>
#     which byte-copies the file and writes a receipt (source path, both
#     hashes, time) into eglantine-test3-sessions.log;
#   - the trace environment (EDEN_NX2_TRACE=1; window/draw vars pass through);
#   - a REFUSAL of chunk-batch mode in all three ways it can appear (env,
#     flag, deck.sh's dk/bananza folder-name auto-default): batched draws
#     skip the per-draw instrumentation and the capture would silently lose
#     draw-to-resource correlation. The default draw mode is required.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
die() { echo "eglantine-test3: $*" >&2; exit 1; }
sessions="$here/eglantine-test3-sessions.log"

# NEVER inherit EGLANTINE_SETTINGS: it is the standard launcher's variable and
# could point at the WORKING ini, which the build rewrites at startup. The
# diagnostic build always uses this kit's own eglantine-test3.ini.
if [ -n "${EGLANTINE_SETTINGS:-}" ]; then
    echo "eglantine-test3: ignoring inherited EGLANTINE_SETTINGS=$EGLANTINE_SETTINGS (it could be your working ini); using this kit's eglantine-test3.ini" | tee -a "$sessions" >&2
    unset EGLANTINE_SETTINGS
fi
export EGLANTINE_SETTINGS="$here/eglantine-test3.ini"
[ -f "$EGLANTINE_SETTINGS" ] || die "settings file $EGLANTINE_SETTINGS is missing; it comes with this kit"

# Explicit seed action: byte-copy the given working ini over the kit's
# diagnostic ini and write a receipt. This is the ONLY way to seed - no
# path guessing, no equality heuristics. Seeding from the kit's own packaged
# default is allowed and recorded as such (a deliberate choice).
if [ "${1:-}" = "--seed-from" ]; then
    src="${2:-}"
    [ -n "$src" ] || die "--seed-from needs the path to your working eglantine-deck.ini"
    [ -f "$src" ] || die "seed source $src does not exist or is not a file"
    cp -- "$src" "$EGLANTINE_SETTINGS" || die "seed copy from $src failed"
    src_hash="$(sha256sum "$src" | cut -d' ' -f1)"
    dst_hash="$(sha256sum "$EGLANTINE_SETTINGS" | cut -d' ' -f1)"
    echo "eglantine-test3: seeded settings from '$src' src-sha256=$src_hash dst-sha256=$dst_hash" | tee -a "$sessions" >&2
    exit 0
fi

# Fail-closed: refuse to run until an explicit seed receipt exists.
if ! grep -q '^eglantine-test3: seeded settings from ' "$sessions" 2>/dev/null; then
    die "settings not seeded yet. Run: ./eglantine-test3.sh --seed-from <path to your working eglantine-deck.ini> (see README-test3.txt)"
fi

# Chunk-batch refusals: env var, explicit flag, and the standard launcher's
# folder-name auto-default (it would turn chunk-batch on AFTER these checks).
# The folder check mirrors eglantine-deck.sh's argument parser and title
# resolution exactly, so options (including --appimage's value) can appear in
# any position without hiding the game folder or being mistaken for it.
[ -z "${EDEN_NX2_CHUNK_BATCH:-}" ] || die "EDEN_NX2_CHUNK_BATCH is set; the diagnostic capture needs the default draw mode (batched draws skip the per-draw trace)"
for a in "$@"; do
    [ "$a" != "--chunk-batch" ] || die "--chunk-batch is unsupported for the diagnostic capture (batched draws skip the per-draw trace)"
done
roms="${EGLANTINE_ROMS:-$HOME/Emulation/roms/switch2}"
game=""
remaining=("$@")
while [ ${#remaining[@]} -gt 0 ]; do
    case "${remaining[0]}" in
    --frames-ahead|--chunk-batch|--no-mesh|--windowed) remaining=("${remaining[@]:1}") ;;
    --appimage) remaining=("${remaining[@]:2}") ;;
    --) break ;;
    -*) remaining=("${remaining[@]:1}") ;;
    *) [ -z "$game" ] && game="${remaining[0]}"; remaining=("${remaining[@]:1}") ;;
    esac
done
if [ -n "$game" ]; then
    if [ -f "$game" ]; then
        ttitle="$(basename "$(dirname "$(dirname "$game")")")"
    else
        # a folder path, or a bare folder name resolved under $roms by deck.sh
        ttitle="$(basename "$game")"
    fi
    case "$ttitle" in
    *dk*|*bananza*|*DK*|*Bananza*)
        die "game folder name '$ttitle' matches the dk/bananza pattern: the standard launcher would auto-enable chunk-batch, which skips the per-draw trace. Rename the folder or use your usual MKW folder."
        ;;
    esac
fi

# This launcher is a capture launcher: refuse to run with the capture
# inherited OFF - a silent EDEN_NX2_TRACE=0 run would look like a capture and
# record nothing.
if [ -n "${EDEN_NX2_TRACE:-}" ] && [ "$EDEN_NX2_TRACE" != "1" ]; then
    die "EDEN_NX2_TRACE=$EDEN_NX2_TRACE inherited; the capture would be OFF. Unset it for a capture run, or use the standard launcher for non-capture runs."
fi
export EDEN_NX2_TRACE=1

appimage="$here/eglantine test 3.AppImage"
[ -f "$appimage" ] || die "'eglantine test 3.AppImage' is missing from $here"

ini_hash_before="$(sha256sum "$EGLANTINE_SETTINGS" | cut -d' ' -f1)"
echo "eglantine-test3: diagnostic capture (evidence only, not a fix); trace=$EDEN_NX2_TRACE draws=${EDEN_NX2_TRACE_DRAWS:-<all>} window=[${EDEN_NX2_TRACE_FROM_FRAME:-0},${EDEN_NX2_TRACE_TO_FRAME:-end}] settings=$(basename "$EGLANTINE_SETTINGS") sha256-before=$ini_hash_before" | tee -a "$sessions" >&2

"$here/eglantine-deck.sh" --appimage "$appimage" "$@"
rc=$?
ini_hash_after="$(sha256sum "$EGLANTINE_SETTINGS" | cut -d' ' -f1)"
echo "eglantine-test3: session ended rc=$rc settings-sha256 before=$ini_hash_before after=$ini_hash_after" | tee -a "$sessions" >&2
exit $rc
