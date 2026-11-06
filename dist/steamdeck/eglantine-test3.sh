#!/usr/bin/env bash
# eglantine test 3 - diagnostic capture launcher.
#
# Evidence-gathering build for the missing-text investigation: it adds trace
# instrumentation only. It is NOT a fix and NOT an FPS change.
#
# Wraps eglantine-deck.sh with:
#   - its own settings file (eglantine-test3.ini, a verbatim copy of the working
#     settings) so the run never rewrites the working ini;
#   - the trace environment (EDEN_NX2_TRACE=1; window/draw vars pass through);
#   - a REFUSAL of chunk-batch mode: batched draws skip the per-draw
#     instrumentation, and the capture would silently lose draw-to-resource
#     correlation. The default draw mode is required.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
die() { echo "eglantine-test3: $*" >&2; exit 1; }

[ -z "${EDEN_NX2_CHUNK_BATCH:-}" ] || die "EDEN_NX2_CHUNK_BATCH is set; the diagnostic capture needs the default draw mode (batched draws skip the per-draw trace)"
for a in "$@"; do
    [ "$a" != "--chunk-batch" ] || die "--chunk-batch is unsupported for the diagnostic capture (batched draws skip the per-draw trace)"
done

appimage="$here/eglantine test 3.AppImage"
[ -f "$appimage" ] || die "'eglantine test 3.AppImage' is missing from $here"

export EGLANTINE_SETTINGS="${EGLANTINE_SETTINGS:-$here/eglantine-test3.ini}"
[ -f "$EGLANTINE_SETTINGS" ] || die "settings file $EGLANTINE_SETTINGS is missing; it comes with this kit"

export EDEN_NX2_TRACE="${EDEN_NX2_TRACE:-1}"
echo "eglantine-test3: diagnostic capture (evidence only, not a fix); trace=$EDEN_NX2_TRACE draws=${EDEN_NX2_TRACE_DRAWS:-<all>} window=[${EDEN_NX2_TRACE_FROM_FRAME:-0},${EDEN_NX2_TRACE_TO_FRAME:-end}] settings=$(basename "$EGLANTINE_SETTINGS")" >&2
exec "$here/eglantine-deck.sh" --appimage "$appimage" "$@"
