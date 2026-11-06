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
#                            clearing failure, stale marker chmod 000, launch still refuses
#  14 garbage markers:       empty file / 'partial' / valid-shape-wrong-length marker,
#                            no seed -> launch refuses each
set -u
REPO=${REPO:-/tmp/wt-test3}
KIT_SRC="$REPO/dist/steamdeck"
WORK=$(mktemp -d /tmp/seed-fail-test.XXXXXX)
trap 'chmod -R u+w "$WORK" 2>/dev/null; rm -rf "$WORK"' EXIT
fails=0

note() { echo "--- $*"; }
expect() { # desc actual expected
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
expect "message names clearing failure" "$(echo "$out" | grep -c 'could not clear the previous seed marker')" "1"
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

note "CASE 13 cleanup failure (kit dir unwritable during reseed)"
D="$WORK/c13"; mk_kit "$D"
"$D/eglantine-test3.sh" --seed-from "$SRCINI" >/dev/null 2>&1
expect "initial seed marker" "$([ -f "$D/eglantine-test3.seeded" ] && echo yes || echo no)" "yes"
chmod 555 "$D"
out=$("$D/eglantine-test3.sh" --seed-from "$SRCINI" 2>&1); rc=$?
[ "$rc" = 0 ] && rc_note=0 || rc_note=nonzero
expect "reseed rc nonzero" "$rc_note" "nonzero"
expect "message names clearing failure" "$(echo "$out" | grep -c 'could not clear the previous seed marker')" "1"
expect "stale marker neutralized (chmod 000)" "$(stat -c %a "$D/eglantine-test3.seeded")" "0"
out=$("$D/eglantine-test3.sh" 2>&1); rc=$?
expect "launch refuses (unreadable marker)" "$rc" "1"
expect "refusal message" "$(echo "$out" | grep -c 'no valid seed receipt marker')" "1"
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

echo ""
if [ "$fails" = 0 ]; then
  echo "SEED FAILURE-INJECTION: all expectations met (14 cases)"
else
  echo "SEED FAILURE-INJECTION: $fails expectation(s) FAILED"
  exit 1
fi
