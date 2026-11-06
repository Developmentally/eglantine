eglantine test 3 - diagnostic build (evidence gathering for the missing-text
investigation). This is NOT a fix and NOT an FPS improvement; it adds trace
instrumentation only. It never rewrites your working build's settings file.

INSTALL - a FRESH, SEPARATE FOLDER IS MANDATORY:
  Extract this kit into a NEW, EMPTY folder (e.g. ~/Downloads/eglantine-test-3).
  NEVER extract it over your existing working build folder: the kit carries an
  eglantine-deck.ini and would overwrite your working settings with the
  packaged default.

Folder layout:
  eglantine test 3.AppImage     the diagnostic build
  eglantine-deck.sh             the standard launcher (unchanged)
  eglantine-test3.sh            the diagnostic capture launcher - use THIS one
  eglantine-deck.ini            the packaged default settings (reference)
  eglantine-test3.ini           the settings the diagnostic build reads and
                                rewrites - never your working ini
  eglantine-test3-sessions.log  appears after the first run: one evidence line
                                per session (exact environment, settings
                                hashes, exit code)

STEP 1 - seed the diagnostic settings from your ACTUAL working settings:
  The packaged eglantine-test3.ini is only the fork's default. Your working
  build rewrites its own ini at every launch, so yours may differ. Seed the
  diagnostic settings from your working build BEFORE the first launch:
    ./eglantine-test3.sh --seed-from "/home/deck/Downloads/eglantine-steamdeck-dev-tc-memory-ports-d551c07/eglantine-deck.ini"
  (Recovered default CANDIDATE: your working shortcut's Start In folder is
  /home/deck/Downloads/eglantine-steamdeck-dev-tc-memory-ports-d551c07, so
  its eglantine-deck.ini should be your working settings - but the shortcut
  proves the folder, not the file's current bytes. If you ever launch with
  EGLANTINE_SETTINGS pointing at a different file, seed from THAT file
  instead.)
  This atomically byte-copies the file (temp + rename; a symlinked or busy
  target is replaced, never written through, and a failed seed leaves the
  old file untouched) and records a receipt (source path, both hashes) in
  eglantine-test3-sessions.log plus the eglantine-test3.seeded marker. The
  launcher REFUSES to run until the marker exists, and also refuses if
  eglantine-test3.ini is a symlink - there is no bypass. (To run with the
  packaged default on purpose, seed from this folder's eglantine-deck.ini;
  the receipt records that choice.)

STEP 2 - launch (Desktop Mode first):
  1. Make the AppImage executable: Properties -> Permissions -> "Is executable".
  2. In this folder, run the test 3 launcher the same way your working
     shortcut launches the game (same game folder, same --frames-ahead):
       ./eglantine-test3.sh "/home/deck/Downloads/base" --frames-ahead
     --frames-ahead is part of your working launch; keep it identical so the
     capture matches what you actually play.
  3. Game Mode: same shape as your working shortcut - Target "/usr/bin/bash",
     Start In = this folder, Launch Options:
       "<this folder>/eglantine-test3.sh" "/home/deck/Downloads/base" --frames-ahead
     The exact Steam argument line is NOT yet verified, so dry-run it in
     Desktop Mode first; if it misbehaves, send the session lines from
     eglantine-test3-sessions.log.
  4. The capture turns itself on (EDEN_NX2_TRACE=1). Trace lines (T3TRACE) go
     to the usual log: ~/.local/share/eden/log/eden_log.txt
  5. Send the log before relaunching - a new launch rotates it (one previous
     generation is kept as eden_log.txt.old.txt).
  6. Every session is recorded in eglantine-test3-sessions.log in this folder
     (exact environment and settings hashes before/after); those lines are
     part of the evidence.

Limits you will hit on purpose:
  - --chunk-batch / EDEN_NX2_CHUNK_BATCH is REFUSED, and so is a game folder
    whose name contains "dk" or "bananza" (the standard launcher auto-enables
    chunk-batch there): batched draws skip the per-draw instrumentation and
    the capture would silently lose the draw-to-resource correlation.
  - If EGLANTINE_SETTINGS is set in your environment, the launcher IGNORES it
    (loudly) and always uses this kit's eglantine-test3.ini - an inherited
    value could point at your working ini, which must never be rewritten.
  - After the first run the build rewrites eglantine-test3.ini at startup, so
    the seed receipt's hash documents the SEEDED state, not the current one.
    Each session records the settings hash before startup and after exit, so
    a diff against the receipt is the startup normalization, not tampering.
    Re-seed (same command) whenever your working settings change.
  - The session log carries the launcher's own pre-launch env line, then
    the standard launcher's "eglantine-deck:" FINAL effective-env record,
    kept as its own lines so it survives any run volume (deck.sh applies its
    own defaults, e.g. AUTO_STUB). Deck/app stderr goes to a temp file during
    the run and only the LAST 128 KiB is appended after the run (replayed to
    the screen too) - earlier stderr beyond that window is not in the log,
    and live feedback is delayed until the app exits. The temp file grows
    unbounded DURING the run and the session log grows with every session;
    fine for a short diagnostic campaign. If the launcher is killed mid-run
    (terminal closed, Deck shutdown, hang), the append never happens and the
    FULL stderr stays in a hidden .eglantine-test3-stderr.* file in this
    folder - send that file too; it is never deleted automatically. Every
    line group carries a session id like [20261106-003000.1234].
  - eglantine-test3.ini must stay a real, singly-linked file: the launcher
    REFUSES to run if it is a symlink or has extra hard links (either would
    make the build's startup rewrite write through to the file it aliases).
    Seeding from the diagnostic ini itself is also refused (a self-copy
    proves nothing).
  - A literal -- argument and a user --appimage are REFUSED (the launcher
    owns the binary choice and the argument shape).
  - Run ONE session at a time: two simultaneous runs share eglantine-test3.ini
    (the build rewrites it at startup) and the sessions log, and the settings
    file could interleave-corrupt - the settings themselves are not protected.
  - The capture window defaults to everything. Instinct may ask you to set
    EDEN_NX2_TRACE_FROM_FRAME / EDEN_NX2_TRACE_TO_FRAME / EDEN_NX2_TRACE_DRAWS
    for a focused capture.
  - Tracing overhead is UNMEASURED - we cannot say whether any extra stutter
    comes from the instrumentation or from something else. If the game feels
    different, note when and where and include that with the logs; it is
    evidence, not a known regression in your working build.
  - The diagnostic build SHARES your Eden data folders (cache, saves, and the
    log at ~/.local/share/eden/log/eden_log.txt) with the working build.
    Sessions can add cache/save/log entries there, and the log is shared -
    send it before relaunching either build.

Rollback:
  Delete this folder. That removes the diagnostic build, its launcher, its
  settings copy, and the session record. Your working build and its settings
  file are never touched by this kit. What deletion does NOT remove: entries
  the diagnostic sessions added to the shared Eden data (cache, saves, log);
  those live in the shared Eden folders, the same as any session's.
