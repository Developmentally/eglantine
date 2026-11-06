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

# Hash WITHOUT a pipeline: sha256sum's own exit status is checked (a
# `| cut` pipeline would mask a nonzero exit behind cut's status), and the
# digest must BE a 64-char lowercase-hex string - a plausible-looking but
# invalid digest is rejected.
hash_of() { # path -> validated 64-hex digest on stdout, or die
    local out h
    if ! out="$(sha256sum -- "$1")"; then
        die "could not hash '$1' (sha256sum exited nonzero)"
    fi
    h="${out%% *}"
    case "$h" in
        ""|*[!0-9a-f]*) die "sha256sum output for '$1' is not a 64-char lowercase-hex digest ('$h')" ;;
    esac
    [ "${#h}" -eq 64 ] || die "sha256sum output for '$1' is not a 64-char lowercase-hex digest ('$h')"
    printf '%s' "$h"
}

# Explicit seed action: atomically byte-copy the given working ini over the
# kit's diagnostic ini (temp file + rename, so a symlinked or busy dest is
# replaced, never written through, and a failed seed never truncates it) and
# record a receipt. The receipt marker is REMOVED first (checked; a marker
# that cannot be removed is chmod 000 so the launch gate still refuses) and
# published only after the copy, the status-checked hash match, and the
# sessions-log receipt line all succeed - atomically (temp + rename, then
# verified), so a failed or partial write never leaves a marker. No path
# guessing, no equality heuristics.
# Seeding from the kit's own packaged default is allowed and recorded as
# such (a deliberate choice).
if [ "${1:-}" = "--seed-from" ]; then
    [ $# -eq 2 ] || die "--seed-from takes exactly one argument: the path to your working eglantine-deck.ini"
    src="$2"
    [ -f "$src" ] || die "seed source $src does not exist or is not a file"
    [ "$(realpath -m -- "$src")" != "$(realpath -m -- "$EGLANTINE_SETTINGS")" ] || die "seed source IS this kit's eglantine-test3.ini itself - a self-copy proves nothing. Point --seed-from at your WORKING build's eglantine-deck.ini"
    tmp="$here/.eglantine-test3.ini.tmp.$$"
    # Clear any previous marker FIRST, and CHECK the clearing. rm only ever
    # unlinks the marker path itself - this script NEVER changes permissions
    # of anything at that path, because a symlink or hardlink there would
    # make chmod hit an unrelated inode (an executed r34 review finding).
    # If removal is refused (e.g. unwritable kit directory), die loudly:
    # the surviving marker is exactly the last good seed's marker and the
    # config is untouched, so a launch would still match the recorded
    # receipt, but the user MUST see the failure and remove it manually.
    if [ -e "$receipt" ] || [ -L "$receipt" ]; then
        rm -f -- "$receipt" 2>/dev/null
        if [ -e "$receipt" ] || [ -L "$receipt" ]; then
            die "could not remove the previous seed marker '$receipt' - seed aborted before copying; no permissions were changed; the old marker is still the last good seed's, but remove it yourself and seed again"
        fi
    fi
    if ! cp -- "$src" "$tmp" || ! mv -f -- "$tmp" "$EGLANTINE_SETTINGS"; then
        rm -f "$tmp"
        die "seed copy from $src failed; the previous settings file is untouched and the seed marker is cleared - seed again before running"
    fi
    # Fail-closed receipt: every step must succeed, the hashes must MATCH, the
    # evidence line must land in the sessions log BEFORE the marker exists, and
    # the marker must be published ATOMICALLY (temp + rename, verified after).
    # Any failure dies nonzero with no valid marker, so a broken seed can
    # never permit a launch.
    src_hash="$(hash_of "$src")" || exit 1
    dst_hash="$(hash_of "$EGLANTINE_SETTINGS")" || exit 1
    [ "$src_hash" = "$dst_hash" ] || \
        die "seed hash mismatch (source $src_hash != installed $dst_hash) - the copy did not land intact; no marker written"
    receipt_line="eglantine-test3: seeded settings from '$src' src-sha256=$src_hash dst-sha256=$dst_hash"
    if ! printf '%s\n' "$receipt_line" >> "$sessions"; then
        die "could not append the seed receipt to the sessions log '$sessions' - marker NOT written; fix the sessions log and seed again"
    fi
    # Atomic publish: write the marker to a temp file and rename it over the
    # marker path, so a failed or partial write NEVER leaves a regular marker
    # file behind (the launch gate checks only this path). Then VERIFY the
    # published marker is a regular file with exactly the receipt line.
    receipt_tmp="$here/.eglantine-test3.seeded.tmp.$$"
    if ! printf '%s\n' "$receipt_line" > "$receipt_tmp"; then
        rm -f "$receipt_tmp"
        die "could not write the seed receipt (temp) - the evidence line is in the sessions log, but NO marker was published; the launcher stays fail-closed"
    fi
    if ! mv -f -- "$receipt_tmp" "$receipt"; then
        rm -f "$receipt_tmp"
        die "could not publish the seed receipt marker '$receipt' - the evidence line is in the sessions log, but NO marker was published; the launcher stays fail-closed"
    fi
    if [ ! -f "$receipt" ] || ! grep -qxF "$receipt_line" "$receipt"; then
        rm -f "$receipt" "$receipt/${receipt_tmp##*/}"
        die "seed receipt marker verification failed after publication (a non-file such as a directory may shadow '$receipt') - the launcher stays fail-closed"
    fi
    printf '%s\n' "$receipt_line" >&2
    exit 0
fi

# Fail-closed launch gate. The marker must be:
#   (a) a READABLE, NON-EMPTY, SINGLY-LINKED regular file (never a symlink or
#       a hardlink to another name - an aliased marker is refused, never
#       modified) with the exact receipt shape - empty, partial, unreadable,
#       or malformed markers refuse;
#   (b) IDENTICAL to the latest seed receipt line in the sessions log - a
#       hand-written marker without its recorded evidence does not count;
#   (c) consistent with the INSTALLED CONFIG: the ini must either still be
#       the freshly seeded bytes (first launch after seeding) or match the
#       recorded end-state of the last completed session (the build rewrites
#       its own ini at startup, so a diff after a run is the documented
#       startup normalization). Anything else means the config was changed
#       outside the recorded flow: refuse and ask for a re-seed.
marker_ok=0
if [ -f "$receipt" ] && [ ! -L "$receipt" ] && [ -r "$receipt" ] && [ -s "$receipt" ] && [ "$(stat -c %h "$receipt")" -eq 1 ]; then
    if grep -qE "^eglantine-test3: seeded settings from '.+' src-sha256=[0-9a-f]{64} dst-sha256=[0-9a-f]{64}$" "$receipt" 2>/dev/null; then
        marker_ok=1
    fi
fi
if [ "$marker_ok" != 1 ]; then
    clue=""
    [ -z "$inherited_settings" ] || clue=" (inherited EGLANTINE_SETTINGS=$inherited_settings is recorded above as a possible source clue - verify it is your working ini before using it)"
    die "settings not seeded yet (no valid seed receipt marker). Run: ./eglantine-test3.sh --seed-from <path to your working eglantine-deck.ini>$clue (see README-test3.txt)"
fi
marker_line="$(cat -- "$receipt")"
last_seed="$(grep -h '^eglantine-test3: seeded settings from ' "$sessions" 2>/dev/null | tail -1)"
if [ -z "$last_seed" ] || [ "$marker_line" != "$last_seed" ]; then
    die "the seed marker does not match the latest seed receipt in '$sessions' - the marker or the log was changed outside the recorded flow. Seed again: ./eglantine-test3.sh --seed-from <path to your working eglantine-deck.ini>"
fi
ini_now="$(hash_of "$EGLANTINE_SETTINGS")" || exit 1
dst_hash="${marker_line##* dst-sha256=}"
config_ok=0
if [ "$ini_now" = "$dst_hash" ]; then
    config_ok=1
else
    seed_ln="$(grep -nF "$marker_line" "$sessions" | tail -1 | cut -d: -f1)"
    end_line="$(grep -n ' session ended rc=' "$sessions" | tail -1)"
    end_ln="${end_line%%:*}"
    after_hash="${end_line##* after=}"
    case "$end_ln" in ''|*[!0-9]*) end_ln=0 ;; esac
    case "$seed_ln" in ''|*[!0-9]*) seed_ln=0 ;; esac
    if [ "$end_ln" -gt "$seed_ln" ] && [ -n "$after_hash" ] && [ "$ini_now" = "$after_hash" ]; then
        config_ok=1
    fi
fi
if [ "$config_ok" != 1 ]; then
    die "eglantine-test3.ini matches neither the seeded bytes nor the recorded end-state of the last session - the config was changed outside the recorded flow. Seed again: ./eglantine-test3.sh --seed-from <path to your working eglantine-deck.ini>"
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
# the final effective env to stderr AT STARTUP. Capture ALL deck/app stderr to
# a temp file; after the run, append the deck env line(s) as their own record
# FIRST (they survive any run volume), then the bounded tail. The temp file
# grows unbounded DURING the run; it is removed only after a successful
# append, so a wrapper kill (or a failed append) never loses the only copy.
deck_err="$here/.eglantine-test3-stderr.$sid"
"$here/eglantine-deck.sh" --appimage "$appimage" "$@" 2> "$deck_err"
rc=$?
ini_hash_after="$(sha256sum "$EGLANTINE_SETTINGS" | cut -d' ' -f1)"
# Every required read/write is status-checked explicitly: a producer failure
# (missing/unreadable temp, grep read error, tail failure) or a consumer
# failure (tee/sessions log) keeps the FULL temp on disk - the only copy is
# never deleted after an incomplete capture. grep no-match (rc 1) is a
# truthful "no env line" note, NOT a failure; grep rc >=2 is a read error.
append_ok=1
if [ ! -r "$deck_err" ]; then
    echo "eglantine-test3: [$sid] WARNING: stderr temp $deck_err missing or unreadable before append; nothing captured" | tee -a "$sessions" >&2
    append_ok=0
else
    {
        echo "eglantine-test3: [$sid] deck effective-env record (kept separate so it survives any run volume):"
        total_bytes=$(wc -c < "$deck_err") || { echo "eglantine-test3: [$sid] WARNING: wc failed on stderr temp"; exit 5; }
        grep -a '^eglantine-deck:' "$deck_err"; grc=$?
        if [ $grc -ge 2 ]; then
            echo "eglantine-test3: [$sid] WARNING: grep read error (rc=$grc) on stderr temp"
            exit 3
        fi
        [ $grc -eq 0 ] || echo "eglantine-test3: [$sid] (no eglantine-deck: env line found in captured stderr)"
        if [ "$total_bytes" -gt 131072 ]; then
            echo "eglantine-test3: [$sid] deck/app stderr follows (LAST 131072 bytes of $total_bytes total; $((total_bytes-131072)) earlier bytes are only in the temp file, which is removed after this append succeeds):"
        else
            echo "eglantine-test3: [$sid] deck/app stderr follows ($total_bytes bytes, complete):"
        fi
        tail -c 131072 "$deck_err" || { echo "eglantine-test3: [$sid] WARNING: tail failed (rc=$?) on stderr temp"; exit 4; }
        echo "eglantine-test3: [$sid] end of deck/app stderr"
    } | tee -a "$sessions" >&2
    ps=("${PIPESTATUS[@]}")
    if [ "${ps[0]}" != 0 ] || [ "${ps[1]}" != 0 ]; then
        append_ok=0
    fi
fi
if [ $append_ok = 1 ]; then
    rm -f "$deck_err"
else
    echo "eglantine-test3: [$sid] WARNING: sessions-log append FAILED or incomplete; FULL stderr retained at $deck_err - send that file too" | tee -a "$sessions" >&2
fi
echo "eglantine-test3: [$sid] session ended rc=$rc settings-sha256 before=$ini_hash_before after=$ini_hash_after" | tee -a "$sessions" >&2
exit $rc
