#!/usr/bin/env bash
# eglantine test 3 - diagnostic capture launcher.
#
# Evidence-gathering build for the missing-text investigation: it adds trace
# instrumentation only. It is NOT a fix and NOT an FPS change.
#
# Wraps eglantine-deck.sh with:
#   - its own settings file (eglantine-test3.ini) so the run never rewrites
#     the working ini;
#   - a seed preflight: the run REFUSES to start while eglantine-test3.ini
#     still matches the packaged default, so the capture uses your ACTUAL
#     current settings, not the fork baseline;
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

# Seed preflight: refuse while the diagnostic ini is still the packaged
# default. Your working build rewrites its own ini at every launch, so the
# packaged default is not your actual current settings.
if [ -f "$here/eglantine-deck.ini" ] && cmp -s "$EGLANTINE_SETTINGS" "$here/eglantine-deck.ini" && [ "${EGLANTINE_TEST3_ALLOW_DEFAULT_INI:-}" != "1" ]; then
    die "eglantine-test3.ini is still the packaged default. Copy the eglantine-deck.ini from your WORKING build folder over it first (see README-test3.txt), so the capture uses your actual settings. To run with the packaged default on purpose: EGLANTINE_TEST3_ALLOW_DEFAULT_INI=1"
fi

# Chunk-batch refusals: env var, explicit flag, and the standard launcher's
# folder-name auto-default (it would turn chunk-batch on AFTER these checks).
[ -z "${EDEN_NX2_CHUNK_BATCH:-}" ] || die "EDEN_NX2_CHUNK_BATCH is set; the diagnostic capture needs the default draw mode (batched draws skip the per-draw trace)"
for a in "$@"; do
    [ "$a" != "--chunk-batch" ] || die "--chunk-batch is unsupported for the diagnostic capture (batched draws skip the per-draw trace)"
done
target="${1:-}"
if [ -n "$target" ] && [ "${target#--}" = "$target" ]; then
    if [ -f "$target" ]; then
        ttitle="$(basename "$(dirname "$(dirname "$target")")")"
    else
        ttitle="$(basename "$target")"
    fi
    case "$ttitle" in
    *dk*|*bananza*|*DK*|*Bananza*)
        die "game folder name '$ttitle' matches the dk/bananza pattern: the standard launcher would auto-enable chunk-batch, which skips the per-draw trace. Rename the folder or use your usual MKW folder."
        ;;
    esac
fi

appimage="$here/eglantine test 3.AppImage"
[ -f "$appimage" ] || die "'eglantine test 3.AppImage' is missing from $here"

export EDEN_NX2_TRACE="${EDEN_NX2_TRACE:-1}"
ini_hash_before="$(sha256sum "$EGLANTINE_SETTINGS" | cut -d' ' -f1)"
echo "eglantine-test3: diagnostic capture (evidence only, not a fix); trace=$EDEN_NX2_TRACE draws=${EDEN_NX2_TRACE_DRAWS:-<all>} window=[${EDEN_NX2_TRACE_FROM_FRAME:-0},${EDEN_NX2_TRACE_TO_FRAME:-end}] settings=$(basename "$EGLANTINE_SETTINGS") sha256-before=$ini_hash_before" | tee -a "$sessions" >&2

"$here/eglantine-deck.sh" --appimage "$appimage" "$@"
rc=$?
ini_hash_after="$(sha256sum "$EGLANTINE_SETTINGS" | cut -d' ' -f1)"
echo "eglantine-test3: session ended rc=$rc settings-sha256 before=$ini_hash_before after=$ini_hash_after" | tee -a "$sessions" >&2
exit $rc
