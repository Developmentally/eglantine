eglantine test 3 - diagnostic build (evidence gathering for the missing-text
investigation). This is NOT a fix and NOT an FPS improvement; it adds trace
instrumentation only. Your working build stays untouched.

Folder layout (all five files in one folder, e.g. ~/Downloads/eglantine-test-3):
  eglantine test 3.AppImage   the diagnostic build
  eglantine-deck.sh           the standard launcher (unchanged)
  eglantine-test3.sh          the diagnostic capture launcher - use THIS one
  eglantine-deck.ini          the working settings (reference, unchanged)
  eglantine-test3.ini         a verbatim COPY of the working settings; the
                              diagnostic build reads and rewrites THIS file,
                              never your working ini

Launch (first time, Desktop Mode):
  1. Extract this kit next to "eglantine test 3.AppImage" (all five files,
     one folder).
  2. Make the AppImage executable: Properties -> Permissions -> "Is executable".
  3. Launch with the test 3 launcher and your usual game folder:
       ./eglantine-test3.sh "/home/deck/Downloads/base"
     Game Mode: same kind of Steam shortcut as before - add the AppImage as a
     non-Steam game, then Target "/usr/bin/bash", Start In = this folder,
     Launch Options: "<this folder>/eglantine-test3.sh" "/home/deck/Downloads/base"
  4. The capture turns itself on (EDEN_NX2_TRACE=1). Trace lines (T3TRACE) go
     to the usual log: ~/.local/share/eden/log/eden_log.txt
  5. Send the log before relaunching - a new launch rotates it.
  6. The launcher echoes its exact environment when it starts
     ("eglantine-test3: ..."); that line is part of the evidence.

Limits you will hit on purpose:
  - --chunk-batch / EDEN_NX2_CHUNK_BATCH is REFUSED: batched draws skip the
    per-draw instrumentation and the capture would silently lose the
    draw-to-resource correlation. (Your Mario Kart World setup does not use
    it, so nothing changes for you.)
  - The capture window defaults to everything. Instinct may ask you to set
    EDEN_NX2_TRACE_FROM_FRAME / EDEN_NX2_TRACE_TO_FRAME / EDEN_NX2_TRACE_DRAWS
    for a focused capture.
  - Tracing overhead is UNMEASURED. If the game stutters more than usual,
    that is the instrumentation, not a regression in your working build.
  - The shared log file (~/.local/share/eden/log/eden_log.txt) is also used
    by your working build; send it before relaunching either build.

Rollback:
  Delete the "eglantine test 3" folder. Your working build, its settings,
  and your Eden data (saves, keys, firmware) are untouched - the diagnostic
  build only ever wrote its own ini copy and the shared log file.
