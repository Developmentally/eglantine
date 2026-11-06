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
inherited_settings="${EGLANTINE_SETTINGS:-}"
if [ -n "$inherited_settings" ]; then
    echo "eglantine-test3: ignoring inherited EGLANTINE_SETTINGS=$inherited_settings (it could be your working ini); using this kit's eglantine-test3.ini. Recorded as a possible seed-source clue, never used as a destination" | tee -a "$sessions" >&2
    unset EGLANTINE_SETTINGS
fi
export EGLANTINE_SETTINGS="$here/eglantine-test3.ini"
[ -f "$EGLANTINE_SETTINGS" ] || die "settings file $EGLANTINE_SETTINGS is missing; it comes with this kit"
receipt="$here/eglantine-test3.seeded"

# Explicit seed action: atomically byte-copy the given working ini over the
# kit's diagnostic ini (temp file + rename, so a symlinked or busy dest is
# replaced, never written through, and a failed seed never truncates it) and
# record a receipt. The receipt marker is REMOVED first and rewritten only
# after the copy completes - an interrupted seed leaves no marker and the
# launcher stays fail-closed. No path guessing, no equality heuristics.
# Seeding from the kit's own packaged default is allowed and recorded as
# such (a deliberate choice).
if [ "${1:-}" = "--seed-from" ]; then
    [ $# -eq 2 ] || die "--seed-from takes exactly one argument: the path to your working eglantine-deck.ini"
    src="$2"
    [ -f "$src" ] || die "seed source $src does not exist or is not a file"
    [ "$(realpath -m -- "$src")" != "$(realpath -m -- "$EGLANTINE_SETTINGS")" ] || die "seed source IS this kit's eglantine-test3.ini itself - a self-copy proves nothing. Point --seed-from at your WORKING build's eglantine-deck.ini"
    tmp="$here/.eglantine-test3.ini.tmp.$$"
    rm -f "$receipt"
    if ! cp -- "$src" "$tmp" || ! mv -f -- "$tmp" "$EGLANTINE_SETTINGS"; then
        rm -f "$tmp"
        die "seed copy from $src failed; the previous settings file is untouched and the seed marker is cleared - seed again before running"
    fi
    src_hash="$(sha256sum "$src" | cut -d' ' -f1)"
    dst_hash="$(sha256sum "$EGLANTINE_SETTINGS" | cut -d' ' -f1)"
    echo "eglantine-test3: seeded settings from '$src' src-sha256=$src_hash dst-sha256=$dst_hash" | tee -a "$sessions" > "$receipt"
    cat "$receipt" >&2
    exit 0
fi

# Fail-closed: refuse to run until an explicit seed receipt marker exists.
if [ ! -f "$receipt" ]; then
    clue=""
    [ -z "$inherited_settings" ] || clue=" (inherited EGLANTINE_SETTINGS=$inherited_settings is recorded above as a possible source clue - verify it is your working ini before using it)"
    die "settings not seeded yet. Run: ./eglantine-test3.sh --seed-from <path to your working eglantine-deck.ini>$clue (see README-test3.txt)"
fi
# Run-time guard: a symlinked OR hardlinked settings file would make the
# build's startup rewrite write through to whatever shares the inode
# (possibly the working ini). Refuse; reseed to fix. (The seed's temp+rename
# always installs a fresh, singly-linked regular file.)
[ ! -L "$EGLANTINE_SETTINGS" ] || die "eglantine-test3.ini is a symlink - the build rewrites its settings and would write through the link. Run the --seed-from step to replace it with a real copy."
[ "$(stat -c %h "$EGLANTINE_SETTINGS")" -eq 1 ] || die "eglantine-test3.ini has more than one hard link - the build rewrites its settings and would write through the link. Run the --seed-from step to install a fresh copy."


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
    --appimage) die "--appimage is owned by this launcher (it selects 'eglantine test 3.AppImage'); passing your own would silently run a different binary" ;;
    --) die "a literal -- is unsupported by this launcher; pass the game folder and known switches only" ;;
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

sid="$(date +%Y%m%d-%H%M%S).$$"
ini_hash_before="$(sha256sum "$EGLANTINE_SETTINGS" | cut -d' ' -f1)"
effective_env="$(env | grep '^EDEN_NX2_' | sort | tr '\n' ' ')"
echo "eglantine-test3: [$sid] diagnostic capture (evidence only, not a fix); settings=$(basename "$EGLANTINE_SETTINGS") sha256-before=$ini_hash_before env: $effective_env" | tee -a "$sessions" >&2

# deck.sh applies its own defaults AFTER the wrapper's record above (e.g.
# EDEN_NX2_AUTO_STUB=1, --frames-ahead -> EDEN_NX2_FRAMES_AHEAD=a) and echoes
# the final effective env to stderr. Capture ALL deck/app stderr to a temp
# file and append it to the sessions log AFTER the run: synchronous (no lost
# tail, no ordering ambiguity), bounded to the last 128 KiB.
deck_err="$here/.eglantine-test3-stderr.$sid"
"$here/eglantine-deck.sh" --appimage "$appimage" "$@" 2> "$deck_err"
rc=$?
ini_hash_after="$(sha256sum "$EGLANTINE_SETTINGS" | cut -d' ' -f1)"
{
    echo "eglantine-test3: [$sid] deck/app stderr follows (bounded to the last 131072 bytes):"
    tail -c 131072 "$deck_err"
    echo "eglantine-test3: [$sid] end of deck/app stderr"
} | tee -a "$sessions" >&2
rm -f "$deck_err"
echo "eglantine-test3: [$sid] session ended rc=$rc settings-sha256 before=$ini_hash_before after=$ini_hash_after" | tee -a "$sessions" >&2
exit $rc
