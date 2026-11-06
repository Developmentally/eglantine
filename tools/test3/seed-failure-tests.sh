#!/bin/bash
# eglantine test 3 - seed-branch failure-injection tests (r34; extends r33's
# suite after independent review found residual fail-open paths).
#
# Drives the REAL dist/steamdeck/eglantine-test3.sh seed branch in scratch kit
# folders and injects failures. Every case asserts the fail-closed contract:
# nonzero exit, NO valid launch marker, and the diagnostic ini untouched unless
# the case says otherwise. Exits nonzero if any case's actuals differ.
#
# Cases:
#   1 control:               clean seed -> rc 0, marker, receipt line in sessions
#   2 launch-refusal:        no marker -> launch dies "no valid seed receipt"
#   3 copy failure:          unreadable source -> rc!=0, no marker, ini untouched
#   4 src hash failure:      stubbed sha256sum exits 1 on the source -> rc!=0, no marker
#   5 dst hash failure:      stubbed sha256sum exits 1 on the installed ini -> rc!=0, no marker
#   6 hash mismatch:         stubbed sha256sum lies (valid 64-hex) for the installed ini -> rc!=0, no marker
#   7 evidence append fail:  unwritable sessions log -> rc!=0, no marker
#   8 marker path is a dir:  clearing refuses (not a regular file) -> rc!=0, no marker file
#   9 stale-marker reseed:   good seed, then reseed with unwritable sessions ->
#                            rc!=0 AND the old marker is GONE (launch refuses)
#  10 hash output+nonzero:   stub prints VALID-LOOKING 64-hex then exits 1
#                            (editor's executed r33 attack: cut pipeline masked it) -> rc!=0, no marker
#  11 hash format:           stub prints 63-char hex, exit 0, consistent for both files
#                            (equality would pass) -> rc!=0 "not a 64-char", no marker
#  12 partial write:         exported printf function writes 'partial' + returns 1 when
#                            stdout is the receipt temp (editor's executed r33 attack:
#                            a regular partial marker survived) -> rc!=0, NO marker, no tmp litter
#  13 cleanup failure:       good seed, kit dir chmod 555, reseed -> rc!=0 naming the
#                            removal failure, NO permissions changed anywhere; the
#                            surviving marker still matches the recorded receipt and the
#                            untouched config, so launch proceeds with the last good
#                            seed's settings (documented behavior, loudly failed seed)
#  14 garbage markers:       empty file / 'partial' / valid-shape-wrong-length marker,
#                            no seed -> launch refuses each
#  15 symlink side effect:   receipt is a symlink to an unrelated 0640 file, kit dir
#                            555 (editor's executed r34 finding: r34 chmod 000 the
#                            TARGET) -> rc!=0, target mode/content UNCHANGED, launch
#                            refuses the symlinked marker
#  16 hardlink side effect:  receipt is a hardlink to an unrelated 0640 file, dir 555
#                            -> rc!=0, target mode UNCHANGED, launch refuses the
#                            multi-linked marker
#  17 hand-written marker:   valid-shape line with fabricated 64-hex digests and no
#                            matching sessions receipt -> launch refuses
#  18 fabricated receipt:    after a real seed, marker replaced by a valid-shape line
#                            with unequal 64-hex digests -> launch refuses
#  19 config tamper:         after a real seed, ini modified outside the recorded flow
#                            -> launch refuses (neither seeded bytes nor recorded
#                            end-state)
#  18b unequal-pair receipt: valid-shape src!=dst digests (dst = the REAL ini hash)
#                            with a forged matching sessions line -> launch refuses
#                            (no real seed ever writes unequal hashes)
#  20 legit continuation:    dummy AppImage rewrites the ini like the real build's
#                            startup normalization; launch 1 and launch 2 BOTH pass the
#                            gate (recorded end-state branch) - guards over-refusal
#  21 log loss:              sessions log deleted, or the receipt line truncated out,
#                            after a real seed -> launch refuses (re-seed recovers)
#  22 stale-marker matrix:   with a VALID marker already installed, each injected
#                            reseed failure (src/dst hash failure, hash
#                            output-then-nonzero, hash mismatch, partial publish)
#                            leaves NO marker and launch refuses. (Distinct
#                            disclosed limit: if the filesystem refuses to DELETE
#                            the old marker at all, the last-good marker survives
#                            by design - case 13.)
set -u
# The injected permission failures (chmod 555/444/000) only bind a NON-ROOT
# user; running this harness as root would make them silently ineffective.
[ "$(id -u)" -ne 0 ] || { echo "FATAL: run this harness as a NON-ROOT user; the permission injections are no-ops for root"; exit 1; }
REPO=${REPO:-/tmp/wt-test3}
KIT_SRC="$REPO/dist/steamdeck"
WORK=$(mktemp -d /tmp/seed-fail-test.XXXXXX)
trap 'chmod -R u+w "$WORK" 2>/dev/null; rm -rf "$WORK"' EXIT
fails=0

ncases=0
note() { case "$1" in CASE*) ncases=$((ncases+1));; esac; echo "--- $*"; }
nchecks=0
expect() { # desc actual expected
  nchecks=$((nchecks+1))
  if [ "$2" = "$3" ]; then echo "  OK: $1"; else echo "  FAIL: $1 (expected [$3], got [$2])"; fails=$((fails+1)); fi
}

mk_kit() { # dir -> fresh kit with dummy AppImage and distinctive ini bytes
  local d="$1"
  mkdir -p "$d"
  cp "$KIT_SRC/eglantine-test3.sh" "$KIT_SRC/eglantine-deck.sh" "$KIT_SRC/eglantine-deck.ini" "$d/"
  cp "$KIT_SRC/eglantine-deck.ini" "$d/eglantine-test3.ini"
  printf '#!/bin/bash\nexit 1\n' > "$d/eglantine test 3.AppImage"
  chmod +x "$d/eglantine test 3.AppImage" "$d/eglantine-test3.sh" "$d/eglantine-deck.sh"
}

SRCINI="$WORK/working-eglantine-deck.ini"
printf '[Controls]\\nseed_marker=working-bytes-12345\\n' > "$SRCINI"

mk_stub() { # dir mode: writes a sha256sum stub; mode = fail-src|fail-dst|lie-dst|output-then-fail|short-hex
  local stubbin="$1" mode="$2"
  mkdir -p "$stubbin"
  cat > "$stubbin/sha256sum" <<STUB
#!/bin/bash
last="\${!#}"
case "\$last" in
STUB
  case "$mode" in
    fail-src) echo '  */working-eglantine-deck.ini) exit 1;;' >> "$stubbin/sha256sum";;
    fail-dst) echo '  */eglantine-test3.ini) exit 1;;' >> "$stubbin/sha256sum";;
    lie-dst)  echo '  */eglantine-test3.ini) echo "0000000000000000000000000000000000000000000000000000000000000000  $last"; exit 0;;' >> "$stubbin/sha256sum";;
    # editor's executed r33 attack: plausible-looking digest on stdout, then nonzero exit
    output-then-fail) echo '  *) echo "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  $last"; exit 1;;' >> "$stubbin/sha256sum";;
    # consistent 63-char hex for BOTH files: equality passes, format must reject
    short-hex) echo '  *) echo "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb  $last"; exit 0;;' >> "$stubbin/sha256sum";;
  esac
  cat >> "$stubbin/sha256sum" <<STUB
  *) exec /usr/bin/sha256sum "\$@";;
esac
STUB
  chmod +x "$stubbin/sha256sum"
}

note "CASE 1 control: clean seed"
D="$WORK/c1"; mk_kit "$D"
out=$("$D/eglantine-test3.sh" --seed-from "$SRCINI" 2>&1); rc=$?
expect "seed rc 0" "$rc" "0"
expect "marker exists" "$([ -f "$D/eglantine-test3.seeded" ] && echo yes || echo no)" "yes"
expect "sessions has receipt" "$(grep -c 'seeded settings from' "$D/eglantine-test3-sessions.log")" "1"
expect "ini bytes seeded" "$(sha256sum "$D/eglantine-test3.ini" | cut -d' ' -f1)" "$(sha256sum "$SRCINI" | cut -d' ' -f1)"

note "CASE 2 launch-refusal: no marker -> refuses"
D="$WORK/c2"; mk_kit "$D"
out=$("$D/eglantine-test3.sh" 2>&1); rc=$?
expect "launch rc 1" "$rc" "1"
expect "refusal message" "$(echo "$out" | grep -c 'no valid seed receipt marker')" "1"

note "CASE 3 copy failure: unreadable source"
D="$WORK/c3"; mk_kit "$D"
UNREAD="$WORK/unreadable.ini"; printf 'x\n' > "$UNREAD"; chmod 000 "$UNREAD"
ini_before=$(sha256sum "$D/eglantine-test3.ini" | cut -d' ' -f1)
out=$("$D/eglantine-test3.sh" --seed-from "$UNREAD" 2>&1); rc=$?
[ "$rc" = 0 ] && rc_note=0 || rc_note=nonzero
expect "seed rc nonzero" "$rc_note" "nonzero"
expect "no marker" "$([ -f "$D/eglantine-test3.seeded" ] && echo yes || echo no)" "no"
expect "ini untouched" "$(sha256sum "$D/eglantine-test3.ini" | cut -d' ' -f1)" "$ini_before"
chmod 644 "$UNREAD"

note "CASE 4 src hash failure (stubbed sha256sum fails on source)"
D="$WORK/c4"; mk_kit "$D"; mk_stub "$WORK/c4bin" fail-src
out=$(PATH="$WORK/c4bin:$PATH" "$D/eglantine-test3.sh" --seed-from "$SRCINI" 2>&1); rc=$?
[ "$rc" = 0 ] && rc_note=0 || rc_note=nonzero
expect "seed rc nonzero" "$rc_note" "nonzero"
expect "message names hashing" "$(echo "$out" | grep -c "could not hash")" "1"
expect "no marker" "$([ -f "$D/eglantine-test3.seeded" ] && echo yes || echo no)" "no"

note "CASE 5 dst hash failure (stubbed sha256sum fails on installed ini)"
D="$WORK/c5"; mk_kit "$D"; mk_stub "$WORK/c5bin" fail-dst
out=$(PATH="$WORK/c5bin:$PATH" "$D/eglantine-test3.sh" --seed-from "$SRCINI" 2>&1); rc=$?
[ "$rc" = 0 ] && rc_note=0 || rc_note=nonzero
expect "seed rc nonzero" "$rc_note" "nonzero"
expect "message names installed hashing" "$(echo "$out" | grep -c "could not hash")" "1"
expect "no marker" "$([ -f "$D/eglantine-test3.seeded" ] && echo yes || echo no)" "no"

note "CASE 6 hash mismatch (stubbed sha256sum lies for installed ini)"
D="$WORK/c6"; mk_kit "$D"; mk_stub "$WORK/c6bin" lie-dst
out=$(PATH="$WORK/c6bin:$PATH" "$D/eglantine-test3.sh" --seed-from "$SRCINI" 2>&1); rc=$?
[ "$rc" = 0 ] && rc_note=0 || rc_note=nonzero
expect "seed rc nonzero" "$rc_note" "nonzero"
expect "message names mismatch" "$(echo "$out" | grep -c 'hash mismatch')" "1"
expect "no marker" "$([ -f "$D/eglantine-test3.seeded" ] && echo yes || echo no)" "no"

note "CASE 7 evidence append failure (unwritable sessions log)"
D="$WORK/c7"; mk_kit "$D"
touch "$D/eglantine-test3-sessions.log"; chmod 444 "$D/eglantine-test3-sessions.log"
out=$("$D/eglantine-test3.sh" --seed-from "$SRCINI" 2>&1); rc=$?
[ "$rc" = 0 ] && rc_note=0 || rc_note=nonzero
expect "seed rc nonzero" "$rc_note" "nonzero"
expect "message names sessions append" "$(echo "$out" | grep -c 'could not append the seed receipt')" "1"
expect "no marker" "$([ -f "$D/eglantine-test3.seeded" ] && echo yes || echo no)" "no"

note "CASE 8 marker path is a directory (clearing refuses early)"
D="$WORK/c8"; mk_kit "$D"
mkdir "$D/eglantine-test3.seeded"
out=$("$D/eglantine-test3.sh" --seed-from "$SRCINI" 2>&1); rc=$?
[ "$rc" = 0 ] && rc_note=0 || rc_note=nonzero
expect "seed rc nonzero" "$rc_note" "nonzero"
expect "message names removal failure" "$(echo "$out" | grep -c 'could not remove the previous seed marker')" "1"
expect "sessions did NOT get the line" "$(grep -c 'seeded settings from' "$D/eglantine-test3-sessions.log" 2>/dev/null || echo 0)" "0"
out=$("$D/eglantine-test3.sh" 2>&1); rc=$?
expect "launch still refuses (dir is not a marker file)" "$rc" "1"

note "CASE 9 stale-marker failing reseed (old marker must NOT permit launch)"
D="$WORK/c9"; mk_kit "$D"
"$D/eglantine-test3.sh" --seed-from "$SRCINI" >/dev/null 2>&1
expect "initial seed marker" "$([ -f "$D/eglantine-test3.seeded" ] && echo yes || echo no)" "yes"
chmod 444 "$D/eglantine-test3-sessions.log"
out=$("$D/eglantine-test3.sh" --seed-from "$SRCINI" 2>&1); rc=$?
[ "$rc" = 0 ] && rc_note=0 || rc_note=nonzero
expect "reseed rc nonzero" "$rc_note" "nonzero"
expect "stale marker removed" "$([ -f "$D/eglantine-test3.seeded" ] && echo yes || echo no)" "no"
out=$("$D/eglantine-test3.sh" 2>&1); rc=$?
expect "launch refuses after failed reseed" "$rc" "1"

note "CASE 10 hash output+nonzero (editor's executed r33 attack)"
D="$WORK/c10"; mk_kit "$D"; mk_stub "$WORK/c10bin" output-then-fail
out=$(PATH="$WORK/c10bin:$PATH" "$D/eglantine-test3.sh" --seed-from "$SRCINI" 2>&1); rc=$?
[ "$rc" = 0 ] && rc_note=0 || rc_note=nonzero
expect "seed rc nonzero" "$rc_note" "nonzero"
expect "message names hash failure" "$(echo "$out" | grep -c 'exited nonzero')" "1"
expect "no marker" "$([ -f "$D/eglantine-test3.seeded" ] && echo yes || echo no)" "no"

note "CASE 11 hash format (consistent 63-char hex, equality would pass)"
D="$WORK/c11"; mk_kit "$D"; mk_stub "$WORK/c11bin" short-hex
out=$(PATH="$WORK/c11bin:$PATH" "$D/eglantine-test3.sh" --seed-from "$SRCINI" 2>&1); rc=$?
[ "$rc" = 0 ] && rc_note=0 || rc_note=nonzero
expect "seed rc nonzero" "$rc_note" "nonzero"
expect "message names format" "$(echo "$out" | grep -c 'not a 64-char lowercase-hex digest')" "1"
expect "no marker" "$([ -f "$D/eglantine-test3.seeded" ] && echo yes || echo no)" "no"

note "CASE 12 partial write after open (editor's executed r33 attack)"
D="$WORK/c12"; mk_kit "$D"
# bash's export builtin rejects the BASH_FUNC_ name syntax; set it via env(1).
out=$(env 'BASH_FUNC_printf%%=() { if [[ $(readlink /proc/$$/fd/1) == *eglantine-test3.seeded* ]]; then builtin printf partial; return 1; else builtin printf "$@"; fi; }' "$D/eglantine-test3.sh" --seed-from "$SRCINI" 2>&1); rc=$?
[ "$rc" = 0 ] && rc_note=0 || rc_note=nonzero
expect "seed rc nonzero" "$rc_note" "nonzero"
expect "no marker survives" "$([ -e "$D/eglantine-test3.seeded" ] && echo yes || echo no)" "no"
expect "no tmp litter" "$(ls -a "$D" | grep -c 'eglantine-test3.seeded.tmp')" "0"
expect "sessions got the line" "$(grep -c 'seeded settings from' "$D/eglantine-test3-sessions.log")" "1"
out=$("$D/eglantine-test3.sh" 2>&1); rc=$?
expect "launch refuses" "$rc" "1"
expect "refusal names the invalid marker" "$(echo "$out" | grep -c 'no valid seed receipt marker')" "1"
expect "no capture started" "$(echo "$out" | grep -c 'diagnostic capture')" "0"

note "CASE 13 cleanup failure (kit dir unwritable during reseed; no chmod anywhere)"
D="$WORK/c13"; mk_kit "$D"
"$D/eglantine-test3.sh" --seed-from "$SRCINI" >/dev/null 2>&1
expect "initial seed marker" "$([ -f "$D/eglantine-test3.seeded" ] && echo yes || echo no)" "yes"
chmod 555 "$D"
out=$("$D/eglantine-test3.sh" --seed-from "$SRCINI" 2>&1); rc=$?
[ "$rc" = 0 ] && rc_note=0 || rc_note=nonzero
expect "reseed rc nonzero" "$rc_note" "nonzero"
expect "message names removal failure" "$(echo "$out" | grep -c 'could not remove the previous seed marker')" "1"
expect "message states no permissions changed" "$(echo "$out" | grep -c 'no permissions were changed')" "1"
expect "stale marker mode UNCHANGED (644)" "$(stat -c %a "$D/eglantine-test3.seeded")" "644"
# documented behavior: the surviving marker still matches the recorded receipt
# and the untouched config, so launch proceeds with the last good seed's
# settings (as if the failed re-seed was never attempted) - the failure was
# loud (nonzero exit + message), nothing was silently hidden
out=$("$D/eglantine-test3.sh" 2>&1); rc=$?
expect "launch proceeds with last-good settings (rc 1 from dummy AppImage)" "$rc" "1"
expect "capture started with recorded settings" "$(echo "$out" | grep -c 'diagnostic capture')" "1"
chmod 755 "$D"

note "CASE 14 garbage markers never permit launch"
for shape in empty partial wronghash; do
  D="$WORK/c14-$shape"; mk_kit "$D"
  case "$shape" in
    empty) : > "$D/eglantine-test3.seeded";;
    partial) printf 'partial' > "$D/eglantine-test3.seeded";;
    wronghash) printf "eglantine-test3: seeded settings from '/x' src-sha256=abc dst-sha256=def\n" > "$D/eglantine-test3.seeded";;
  esac
  out=$("$D/eglantine-test3.sh" 2>&1); rc=$?
  expect "launch refuses $shape marker" "$rc" "1"
  expect "refusal names the invalid marker ($shape)" "$(echo "$out" | grep -c 'no valid seed receipt marker')" "1"
  expect "no capture started ($shape)" "$(echo "$out" | grep -c 'diagnostic capture')" "0"
done

note "CASE 15 symlink side effect (editor's executed r34 finding)"
D="$WORK/c15"; mk_kit "$D"
TARGET="$WORK/unrelated-target.ini"; printf 'unrelated\n' > "$TARGET"; chmod 640 "$TARGET"
ln -s "$TARGET" "$D/eglantine-test3.seeded"
chmod 555 "$D"
out=$("$D/eglantine-test3.sh" --seed-from "$SRCINI" 2>&1); rc=$?
[ "$rc" = 0 ] && rc_note=0 || rc_note=nonzero
expect "seed rc nonzero" "$rc_note" "nonzero"
expect "target mode UNCHANGED (640)" "$(stat -c %a "$TARGET")" "640"
expect "target content unchanged" "$(cat "$TARGET")" "unrelated"
expect "target link count UNCHANGED (1)" "$(stat -c %h "$TARGET")" "1"
out=$("$D/eglantine-test3.sh" 2>&1); rc=$?
expect "launch refuses symlinked marker" "$rc" "1"
expect "refusal names invalid marker" "$(echo "$out" | grep -c 'no valid seed receipt marker')" "1"
expect "no capture started" "$(echo "$out" | grep -c 'diagnostic capture')" "0"
chmod 755 "$D"

note "CASE 16 hardlink side effect (editor's executed r34 finding, hardlink variant)"
D="$WORK/c16"; mk_kit "$D"
TARGET="$WORK/unrelated-target2.ini"; printf 'unrelated2\n' > "$TARGET"; chmod 640 "$TARGET"
ln "$TARGET" "$D/eglantine-test3.seeded"
chmod 555 "$D"
out=$("$D/eglantine-test3.sh" --seed-from "$SRCINI" 2>&1); rc=$?
[ "$rc" = 0 ] && rc_note=0 || rc_note=nonzero
expect "seed rc nonzero" "$rc_note" "nonzero"
expect "target mode UNCHANGED (640)" "$(stat -c %a "$TARGET")" "640"
expect "target content unchanged" "$(cat "$TARGET")" "unrelated2"
expect "target link count UNCHANGED (2)" "$(stat -c %h "$TARGET")" "2"
out=$("$D/eglantine-test3.sh" 2>&1); rc=$?
expect "launch refuses multi-linked marker" "$rc" "1"
expect "no capture started" "$(echo "$out" | grep -c 'diagnostic capture')" "0"
chmod 755 "$D"

note "CASE 17 hand-written valid-shape marker, no recorded receipt"
D="$WORK/c17"; mk_kit "$D"
printf "eglantine-test3: seeded settings from '/tmp/fake' src-sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa dst-sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n" > "$D/eglantine-test3.seeded"
out=$("$D/eglantine-test3.sh" 2>&1); rc=$?
expect "launch refuses (rc 1)" "$rc" "1"
expect "message names marker/log mismatch" "$(echo "$out" | grep -c 'does not match the latest seed receipt')" "1"
expect "no capture started" "$(echo "$out" | grep -c 'diagnostic capture')" "0"

note "CASE 18 fabricated receipt: valid shape, digests not matching the recorded receipt, after a real seed"
D="$WORK/c18"; mk_kit "$D"
"$D/eglantine-test3.sh" --seed-from "$SRCINI" >/dev/null 2>&1
printf "eglantine-test3: seeded settings from '/tmp/fake' src-sha256=cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc dst-sha256=cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\n" > "$D/eglantine-test3.seeded"
out=$("$D/eglantine-test3.sh" 2>&1); rc=$?
expect "launch refuses (rc 1)" "$rc" "1"
expect "message names marker/log mismatch" "$(echo "$out" | grep -c 'does not match the latest seed receipt')" "1"
expect "no capture started" "$(echo "$out" | grep -c 'diagnostic capture')" "0"

note "CASE 18b unequal src/dst invariant: valid-shape pair, dst matches the real ini, sessions line forged to match"
D="$WORK/c18b"; mk_kit "$D"
"$D/eglantine-test3.sh" --seed-from "$SRCINI" >/dev/null 2>&1
real_ini_hash="$(sha256sum "$D/eglantine-test3.ini" | cut -d' ' -f1)"
forged="eglantine-test3: seeded settings from '/tmp/fake' src-sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa dst-sha256=$real_ini_hash"
printf '%s\n' "$forged" > "$D/eglantine-test3.seeded"
printf '%s\n' "$forged" >> "$D/eglantine-test3-sessions.log"
out=$("$D/eglantine-test3.sh" 2>&1); rc=$?
expect "launch refuses unequal-pair receipt (rc 1)" "$rc" "1"
expect "message names unequal src/dst" "$(echo "$out" | grep -c 'unequal src/dst hashes')" "1"
expect "no capture started" "$(echo "$out" | grep -c 'diagnostic capture')" "0"

note "CASE 21 sessions log lost or truncated after a real seed"
D="$WORK/c21"; mk_kit "$D"
"$D/eglantine-test3.sh" --seed-from "$SRCINI" >/dev/null 2>&1
rm "$D/eglantine-test3-sessions.log"
out=$("$D/eglantine-test3.sh" 2>&1); rc=$?
expect "launch refuses after log loss (rc 1)" "$rc" "1"
expect "message names marker/log mismatch" "$(echo "$out" | grep -c 'does not match the latest seed receipt')" "1"
expect "no capture started" "$(echo "$out" | grep -c 'diagnostic capture')" "0"
D="$WORK/c21b"; mk_kit "$D"
"$D/eglantine-test3.sh" --seed-from "$SRCINI" >/dev/null 2>&1
grep -v 'seeded settings from' "$D/eglantine-test3-sessions.log" > "$D/truncated"; mv "$D/truncated" "$D/eglantine-test3-sessions.log"  # grep rc 1 = every line removed = intended here
out=$("$D/eglantine-test3.sh" 2>&1); rc=$?
expect "launch refuses after receipt truncated out (rc 1)" "$rc" "1"
expect "no capture started (truncated)" "$(echo "$out" | grep -c 'diagnostic capture')" "0"

note "CASE 22 stale VALID marker + injected reseed failure (Marcelo's criterion:"
#       "no failed hash/evidence write may leave a launch-authorizing receipt")
stale_case() { # name stubmode|printf-inject
  local name="$1" mode="$2"
  local D="$WORK/c22-$name"; mk_kit "$D"
  "$D/eglantine-test3.sh" --seed-from "$SRCINI" >/dev/null 2>&1
  expect "[$name] initial valid marker" "$([ -f "$D/eglantine-test3.seeded" ] && echo yes || echo no)" "yes"
  local out rc
  case "$mode" in
    printf-inject)
      out=$(env 'BASH_FUNC_printf%%=() { if [[ $(readlink /proc/$$/fd/1) == *eglantine-test3.seeded* ]]; then builtin printf partial; return 1; else builtin printf "$@"; fi; }' "$D/eglantine-test3.sh" --seed-from "$SRCINI" 2>&1); rc=$?;;
    *)
      mk_stub "$WORK/c22bin-$name" "$mode"
      out=$(PATH="$WORK/c22bin-$name:$PATH" "$D/eglantine-test3.sh" --seed-from "$SRCINI" 2>&1); rc=$?;;
  esac
  [ "$rc" = 0 ] && rc=ok0 || rc=nonzero
  expect "[$name] reseed rc nonzero" "$rc" "nonzero"
  expect "[$name] marker GONE (no launch-authorizing receipt)" "$([ -f "$D/eglantine-test3.seeded" ] && echo yes || echo no)" "no"
  out=$("$D/eglantine-test3.sh" 2>&1); rc=$?
  expect "[$name] launch refuses" "$rc" "1"
  expect "[$name] refusal names invalid marker" "$(echo "$out" | grep -c 'no valid seed receipt marker')" "1"
  expect "[$name] no capture started" "$(echo "$out" | grep -c 'diagnostic capture')" "0"
}
stale_case src-hash-fail fail-src
stale_case dst-hash-fail fail-dst
stale_case hash-output-nonzero output-then-fail
stale_case hash-mismatch lie-dst
stale_case partial-publish printf-inject

note "CASE 19 config changed outside the recorded flow"
D="$WORK/c19"; mk_kit "$D"
"$D/eglantine-test3.sh" --seed-from "$SRCINI" >/dev/null 2>&1
printf '# external edit\n' >> "$D/eglantine-test3.ini"
out=$("$D/eglantine-test3.sh" 2>&1); rc=$?
expect "launch refuses (rc 1)" "$rc" "1"
expect "message names config mismatch" "$(echo "$out" | grep -c 'matches neither the seeded bytes nor the recorded end-state')" "1"
expect "no capture started" "$(echo "$out" | grep -c 'diagnostic capture')" "0"

note "CASE 20 legitimate continuation (build rewrites its own ini at startup)"
D="$WORK/c20"; mk_kit "$D"
# deck.sh requires <game>/exefs/main to exec the AppImage; dummy game folder:
mkdir -p "$D/MKW/exefs"; : > "$D/MKW/exefs/main"
printf '#!/bin/bash\necho "# startup-normalized" >> "%s/eglantine-test3.ini"\nexit 1\n' "$D" > "$D/eglantine test 3.AppImage"
chmod +x "$D/eglantine test 3.AppImage"
"$D/eglantine-test3.sh" --seed-from "$SRCINI" >/dev/null 2>&1
out=$("$D/eglantine-test3.sh" "$D/MKW" 2>&1); rc=$?
expect "launch 1 passes gate (rc 1 from dummy)" "$rc" "1"
expect "launch 1 capture started" "$(echo "$out" | grep -c 'diagnostic capture')" "1"
expect "ini was rewritten by the run" "$(grep -c 'startup-normalized' "$D/eglantine-test3.ini")" "1"
expect "session end recorded after-hash" "$(grep -c 'session ended rc=1' "$D/eglantine-test3-sessions.log")" "1"
out=$("$D/eglantine-test3.sh" "$D/MKW" 2>&1); rc=$?
expect "launch 2 passes gate via recorded end-state (rc 1 from dummy)" "$rc" "1"
expect "launch 2 capture started" "$(echo "$out" | grep -c 'diagnostic capture')" "1"

echo ""
if [ "$fails" = 0 ]; then
  echo "SEED FAILURE-INJECTION: all expectations met ($ncases case groups, $nchecks expectations)"
else
  echo "SEED FAILURE-INJECTION: $fails expectation(s) FAILED"
  exit 1
fi
