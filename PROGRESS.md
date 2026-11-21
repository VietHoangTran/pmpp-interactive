# Progress

Running state across sessions. Newest first. Decisions are dated.

## State on 2026-11-21: every chapter built; final pass done

All 29 chapters and modules are built (waves 1 to 5b below), each was reviewed on its own,
and the whole-course consistency pass is done. What is still open is the "Still open" list
under the final pass below. The fresh verify report
(reports/verify-20261121-120703-darwin-emu+race+metal.md) passes 226 of 226 reference
solutions on emu, race and Metal and all 21 emulator self-tests. CUDA and HIP remain
unverified on hardware.

## State on 2026-11-20 (later): building the rest, in waves

The reader approved the pilot ("looks good... go ahead and build the rest") without
CUDA/ROCm reports yet; CUDA and HIP therefore stay unverified on hardware for everything.
The reader also said: add narrated animations or any other content wherever it helps,
without rationing usage.

**How the build runs.** One orchestrated run per wave (the Workflow tool, script saved
by the harness under the session's `workflows/scripts/build-chapters-*.js`). Per chapter:
facts check against current docs -> design -> section builders and lab builders in
parallel -> integration -> independent review -> fixes. The main session then generates
narration audio for the new scripts, runs `./verify.sh` and `npm run e2e`, looks at the
pages, and commits. Per-chapter hints given to the designers are in the wave's `args`.

**Waves** (tick when committed)
- [x] Wave 0 (committed c0ec3fe): on-demand loading of chapters and sections, chapter-building rules in
      CLAUDE.md (done); shared labs-page kit `app/src/components/LabKit.tsx`; emulator
      extension (warps, warp intrinsics, atomics, constant memory, cooperative groups,
      streams, a ThreadSanitizer `race` backend); `notes/today/infra-facts.md`.
- [x] Wave 1 (run wf_ec5c6999-f65, finished on Opus 5.5): ch02, ch03, ch04 (Deep); ch01, ch24
      (Condensed). Built, reviewed and fixed; 13 labs (2.1–2.4, 3.1–3.5, 4.1–4.4); every
      browser check for ch01–ch05 and ch24 passes. Narration audio is committed separately.
- [x] Wave 2 (run wf_52b4ecac-e87, Opus 5.5): ch06 (Deep), ch07 (Standard), ch08 (Condensed),
      ch09 (Standard), ch10 (Deep), appC (Standard). Built, reviewed and fixed; 15 labs (6.1–6.5,
      7.1–7.2, 8.1, 9.1–9.3, 10.1–10.3, appC lab 1); milestones M2 tuned (6.4), M4 softmax and
      argmax (10.3), M5 token counts (9.3). Every browser check passes. Audio committed separately.
- Builder agents run on Opus 5.5 (the reader's request, 2026-11-20). Agents inherit the
  session model; new wave scripts should also pass `model: 'opus'` to every agent() call.
- [x] Wave 3 (run wf_a2cea698-3c2, Opus 5.5): ch15, ch11, ch12 (Deep), ch14 (Standard, 3 labs:
      accepted), ch13 (Condensed). Built, reviewed and fixed; 22 labs (11.1–11.6, 12.1–12.5,
      13.1, 14.1–14.3, 15.1–15.7); milestones M3 (15.6) and the M5 pieces (11.6 top-p mask,
      12.4 candidates, 14.3 sorted rows). Labs 11.5 and 12.5 are CUDA-family only (Metal
      documents no forward progress between threadgroups). Every browser check passes.
- [x] Wave 4 (run wf_ccca9f22-a5f, resumed after the limit reset and finished on Opus 5.5):
      ch20 (Deep, 5 labs: M6 causal softmax and flash attention, a bug hunt, M7 decode and
      GQA decode), x1 (Deep, 4 labs: top-k, top-p/min-p, the M5 sampler, a bug hunt), ch22
      (Standard, M8 batching study), ch19 (Standard, conv as GEMM), appA (Standard, summation
      error). Every review's must-fix items fixed; all checks pass. Narration with
      eleven_v4_turbo through the API (credits were low). Lab 20.5's starter carries lab
      20.4's reference kernel (finish 20.4 first, the page says).
- [x] Wave 5a (run wf_63d11bbd-bd6, resumed after the limit reset and finished on Opus 5.5):
      ch16, ch17, ch18, ch21, ch23 (Condensed: start, one or two pages, labs, wrap; one lab
      each; two narrated scenes each). Every review's must-fix items fixed; all checks pass.
      Metal benchmarks for 16.1, 21.1 and 23.1 were recorded on a loaded machine: rerun them
      on a quiet one. Lab 18.1 part B (grid-wide barrier) is CUDA-family only.
- [x] Wave 5b (run wf_df00289e-445, paused once at the reader's request, resumed and
      finished on Opus 5.5): x2 quantised kernels (Deep, 4 labs: Q8_0 GEMV, Q4_0 GEMV on
      llama.cpp's block layout, an integer-path bug hunt, M9 the quantised decode step) and
      x3 reading real backends (Standard, 2 labs: the missing llama pieces RMSNorm, RoPE and
      SwiGLU, and M10, a tiny deterministic Llama-shaped model generated in the repo, with
      greedy tokens compared to a CPU reference; nothing downloads). Narration: 50 clips with
      eleven_v4_turbo through the API. M9 does not yet plug into M10 (launchQLinear's
      signature differs from LinearFn): the pages say so; an adapter is on the final list.
- [ ] Final pass (2026-11-21: per-chapter reviews of all 29 chapters and modules, then this
      whole-course consistency pass). The to-do list collected along the way, sorted:
      **Done**
      - [x] M9 to M10 adapter (labs/x3/lab2_tinyllama/provided/m9_linear.*: float weight
        pointer -> QMatrix table; Q8_0 and Q4_0 tokens tested against the CPU reference on the
        same quantised weights; verified on Metal, emulator-checked; x2 and x3 pages updated).
      - [x] Lab 5.5 benchmark times 512 to 4096 (every width lab 5.2 can pick); the labs page
        compares at lab 5.2's width only (a pre-2026-11-21 lab 5.5 report shows no comparison
        until re-benchmarked; the "about half the speed" finding below was measured across widths).
      - [x] ch12 lab 1's libhipcxx comment (solution comment and README; starter left as it was).
      - [x] Shared infrastructure asked for by waves 1 to 5b, checked in the code on 2026-11-21:
        the stub and emulators have char4, __dp4a, __byte_perm, __umulhi, __float_as_uint,
        __half (not __nv_bfloat16), CUDA math functions and stream priorities; cu2emu accepts
        parentheses in __shared__ bounds; the emulator enforces per-dimension block and grid
        limits; pmpp.py accepts appendix lab numbers (C.1, A.1), `--reps`, PMPP_CMAKE_ARGS; Lab.cmake
        has SHARED_DIR and a "no Metal build" message; doctor prints asyncEngineCount and
        concurrentKernels; LabKit has RaceStatus, solutionTabStatus and a race column;
        NoteGroup (components/notes) and useNarrow (lib/narrow) are shared; browser.mjs has
        answerQuestion and setRange.
      - [x] Wrap pages say "counting your latest answer to each"; the right option is no longer
        the longest in most questions (question-balance.test.ts passes for every chapter).
      - [x] Whole-course consistency pass (this list's last item, 2026-11-21): every in-app
        link resolves and names the right section number and title (crawled all 248 pages);
        milestone statements checked M0 to M10 across wraps and labs pages (registry PLAN rows
        and ch24's module cards now give X2 as GEMVs plus the decode step and X3 as M10; x3's
        "composes your milestones M0 to M8" now names what M10 really runs, reference solutions
        of M0, M2 to M5, M7, M8, not M1 or M6; ch22's M10 sentence says M5 is used when present);
        "next in this course's order" corrected on the x2, x3 and appC wraps against
        sim/bookmap courseOrder(); start-page checks drawing on later chapters fixed (x1's
        "chapters 10 and 20" label; ch14 says chapter 13 comes later in the course order);
        "verified" kept to the vocabulary (ch02 2.2, ch14 14.9, x3 X3.5); ch20 S203 and S205
        lab tabs use solutionTabStatus; the HIP __syncwarp note in 20.5 is qualified; stale
        comments fixed in solution, shared and snippet files (ch06 lab 3, ch14 lab 2 and
        snippet, ch15 labs 3 to 7, ch23 snippets and probe, appA lab 1 incl. its driver message,
        x1 lab 3, x2 labs 3 and 4 and shared headers) and dated updates added to x1-, x2- and
        appA-facts; PLAN.md's M9, M10, X3 and ch23 rows, §8 captions and narration, and §9
        status brought up to what was built (§9's order is unchanged; a test reads it);
        README.md describes the finished course.
      - [x] Final verify report: reports/verify-20261121-120703-darwin-emu+race+metal.md, run
        on a quiet machine: 226 of 226 reference solutions pass (emu, race, Metal), every
        untouched starter fails as it should, 21 of 21 emulator self-tests, 443 .cu files pass
        the front-end check. It also filled reports/runs/emu.json's device record. CUDA and HIP
        stay unverified until reports come from those machines.
      - [x] Lab X3.2 on the emulator: pmpp.py's test limit is now 2400 s on the emulators
        (900 s on devices; PMPP_TEST_TIMEOUT overrides). It passes 93/93 in the verify run.
      - [x] Narration: the 3 stale clips (ch02/round-trip-6, ch07/halo-from-cache-5,
        ch21/scatter-gather-6) regenerated with eleven_v4; ch07's script reworded after a
        mishearing.
      **Still open, with reasons**
      - [ ] Stale comments in starter/ files are left, because editing a starter turns every
        untouched work/ copy into "edited" (pmpp.py work_state) and work/ is never overwritten:
        parentheses in __shared__ bounds (ch06 lab 3, ch14 lab 2, ch15 lab 4), __umulhi (x1 lab 3),
        "declares no CUDA Math API functions" (ch10 lab 3, ch20 labs 1 and 4, ch21 lab 1, x1 labs
        2 to 4, x3 lab 1; the solutions, shared copies and ch21 snippets carry it too, kept so
        that copies stay byte-identical, and the ch10 README says the comment is out of date).
      - [ ] Some lesson tabs still build their status by hand: ch07 S72, S74, S75, ch08 S84,
        ch19 S192 and S193, ch04 S48, ch06 S67, ch15 S153 and S158, ch16 S166 read the reports
        and match LabKit's words when a run passes, but say "no passing ... run" where LabKit
        says "failed on this machine"; ch05 S54 and ch12 S122 to S127 are fixed strings that do
        not read the reports at all (true on this machine).
      - [ ] LabKit prints "verified on CUDA" or "verified on HIP" from a passing run in
        reports/runs, while CLAUDE.md asks for a reports/verify-*.md from that machine (shared
        decision; no CUDA or HIP runs exist, so nothing shows it today).
      - [ ] Lab 14.2's planted missing-barrier defect has not been re-run on `race` since the
        two-token barrier fix; the race backend once hung after a SEGV in a cooperative launch
        under heavy load (not reproduced).
      - [ ] Metal benchmarks for 16.1, 21.1 and 23.1 were recorded on a loaded machine: rerun on
        a quiet one.
      - [ ] Optional requests not checked in this pass: a choice-type Predict, a source label and
        custom line numbers for Code, a function-excerpt option for useLabFile, a bench-row
        picker in lib/reports.ts, a `.threads.wrap` modifier, lab 4.1's kernel named like lab
        3.3's, source_hash covering provided/shared sources, Code tabs' status text squeezed at
        390 px.
- [ ] Final: spaced-review pool check, final verify report, hand-off (the whole-course
      consistency pass is done; see above).

## State on 2026-11-20: pilot complete, waiting for review

**Phase:** chapter 5 (the pilot) is built to the finished standard. Nothing after the
pilot starts until the reader approves it and has run `./verify.sh` on a CUDA machine and
a ROCm machine and brought the reports back.

### Done
- Whole book read as page images; notes for all 24 chapters and appendices A to C in
  `notes/` (chapter 5 by the main session, the rest by per-chapter readers).
  `notes/pagemap.json` maps PDF pages to book pages.
- `notes/today/ch05-facts.md`: 28 CUDA / HIP / Metal claims checked against current docs.
- `PLAN.md`, `CLAUDE.md`, `README.md`.
- Lab infrastructure: CMake backend detection, runtime, CPU emulator with
  AddressSanitizer and three thread schedules, CUDA front-end check, `labs/pmpp.py`,
  `./verify.sh`.
- Chapter 5 in the app: nine pages (before you start, 5.1 to 5.6, labs, summary), twelve
  interactives, one narrated animation (six clips), 28 questions feeding spaced review.
- Chapter 5 labs 5.1 to 5.5 with starters, working copies and reference solutions.
- Checks: 49 unit tests (`cd app && npm test`), 402 browser checks in nine scripts
  (`cd app && npm run e2e`, app running).

### Verification status of chapter 5 reference solutions
| Lab | Metal | emu | CUDA | HIP |
|---|---|---|---|---|
| 5.1 roofline | verified on Metal | emulator-checked | front-end-checked; unverified on device | unverified |
| 5.2 tiled | verified on Metal | emulator-checked | front-end-checked; unverified on device | unverified |
| 5.3 gemm | verified on Metal | emulator-checked | front-end-checked; unverified on device | unverified |
| 5.4 race hunt | verified on Metal | emulator-checked | front-end-checked; unverified on device | unverified |
| 5.5 dynamic | verified on Metal | emulator-checked | front-end-checked; unverified on device | unverified |

Evidence: `reports/verify-20261120-033114-darwin-emu+metal.md` (Apple M3 Max). The CUDA
and HIP CMake paths and `pmpp/vendors/hip.h` have never been compiled. The HIP build
settings follow AMD's and CMake's documentation (facts file, B6 and B7).

### Not verified, stated plainly
- Nothing has run on a CUDA or HIP device.
- Narration audio was transcribed back with a local Whisper model and matches the
  scripts word for word; nobody has listened to it for tone or pacing.
- Sections 5.2, 5.3, 5.5, 5.6 and the labs page were written by section builders from
  the notes and the facts file. The main session reviewed their reports, their browser
  checks and screenshots of each interactive, not every sentence.
- The browser checks ran in headless Chrome 157; the Chrome extension was not connected,
  so nothing was checked in a visible browser window.

### Decisions
- 2026-11-20: when ElevenLabs credits run low, narration may use `eleven_v4_turbo` through
  the API directly (tools/tts-api.mjs), same voice (the reader's instruction, asking for a
  'v4 flash': the API has no such model; `eleven_v4_turbo` is the fast v4). Measured on
  one 228-character clip: 48 credits through the API with turbo, about 219 per take
  through the MCP tool with eleven_v4; a 25-character test cost 6 (turbo) and 11
  (eleven_v4) through the API. The supplied key can generate speech but lacks user_read
  and models_read, so the balance is read through the MCP tool. The key lives only in the
  session scratchpad and the environment of the commands that use it.
- 2026-11-20: choice options are shown in a stable shuffled order per question id
  (components/Ask.tsx displayOrder; `fixedOrder` keeps a scale in order). Builders had put
  the right answer first in up to 81% of a chapter's questions. Answers are stored by the
  written index; browser checks set window.__PMPP_WRITTEN_ORDER__ (e2e/browser.mjs).
- 2026-11-20: the race emulator alternates two happens-before tokens per barrier (and per
  grid sync). One shared token let a thread leaving barrier k sync with a neighbour already
  at barrier k+1 and hid races between them (a new self-test case shows it: missed 3 of 5
  runs before, reported 6 of 6 after). All 36 race-enabled lab solutions still pass.
- 2026-11-20: shared fixes from wave 2: pmpp.py locks reports/runs while merging and replaces
  files atomically (concurrent runs had emptied emu.json); the labs-page emulator paragraph
  now matches labs/README.md (0xCD shared memory per block, warps modelled); realBackends
  leaves out `race`; `__launch_bounds__` in the stub and emulator; cudaMallocManaged and
  cudaMemAttachGlobal in hip.h (appC-facts B6); the CUDA-family device report adds
  totalConstMem and l2CacheSize; the HIP build links libhipcxx::libhipcxx when ROCm has it
  (infra-facts B2, unverified); the highlighter no longer colours `.x` members as variables.
- 2026-11-20: condensed chapters may carry more than one or two pages of interactives
  (ch01 and ch24 do), on the reader's word that more content is welcome. Their reviewers
  flagged the extra density; it is accepted, not a defect.
- 2026-11-20: shared fixes from wave 1 requests: scene part buttons wrap on phones; a
  scene drawing keeps a readable width and scrolls in its frame below 640 px unless its
  svg has data-layout="narrow"; a missing narration manifest is served as null (no 404);
  a benchmark-only run keeps the last test result when the sources hash the same; the
  emulator and the front-end stub accept the typed `cudaMalloc(T**, size_t)` that the
  CUDA runtime reference lists (ch02-facts A2; it was wrongly removed during the pilot).
- 2026-11-20: the dev server serves a not-yet-written section as a module that throws, so
  one half-built chapter cannot take the app down while builders work.
- 2026-11-20: captions on narrated animations are an option, off by default and
  remembered (reader's instruction).
- 2026-11-20: narration model is `eleven_v4` through the MCP tool. The reader asked for
  `eleven_v4_turbo`; the MCP server offers only `eleven_v4`, `eleven_v3`,
  `eleven_turbo_v2_5`, `eleven_flash_v2_5`, `eleven_multilingual_v2`. Reader chose
  `eleven_v4`. Voice `tnSpp4vdxKPjI9w0GnoV` ("Hope - upbeat and clear").
- 2026-11-20: CPU emulation uses our own small emulator, not AMD's HIP-CPU (does not build
  with Apple clang, needs GCC + TBB, unmaintained since 2024-03).
- 2026-11-20: pilot chapter is 5 (memory and data locality, tiled matmul).

### Findings worth remembering
- On the M3 Max (appendix A's probe, wave 4): Metal flushed FP32 denormals, rounded to
  nearest, fused a*b+c, and its default math mode deleted the Kahan compensation;
  `#pragma METAL fp math_mode(safe)` (upper-case METAL) restored it. One machine only.
- On the M3 Max, chapter 15's M3 GEMM (the book's 128x128 tile, register tiling, padding,
  pipelining) measured slower than chapter 5's M2 at batch 256 and far below MPS: the
  book's NVIDIA-shaped tuning does not transfer to Apple's GPU as is. Pages show only
  measurements; module X3 and the final pass should not claim otherwise.
- On the Apple M3 Max used for the pilot, tiling gave about 1.7x to 1.9x over the naive
  kernel and a 4x4 tile was slower than naive; the naive kernel ran about 10x above the
  book's 0.25 FLOP/B model ceiling. Caches serve most of the naive kernel's reads.
  Lessons have the reader predict and measure, and never promise the book's 32x.
- With the tile width as a run-time value (lab 5.5) the kernel ran at about half the
  speed of the compile-time version on the same machine.
- Missing zero-fill on only one of the two tiles is invisible to tests unless a slot was
  never written and holds NaN or infinity. Lab 5.4's defect omits both.
- Book errata found in chapter 5: the 25.1 FLOP/B on p. 97, phase numbering in Fig. 5.8,
  the thread index on p. 113, several errors in Fig. 5.14, `sharedMemPerBlock` on p. 116,
  "per SM" in exercise 12. All are in `notes/ch05.md` and on the summary page.
- The book never mentions HIP, Metal, sampling or quantised kernels, and gives no measured
  timings in chapters 15 or 20.
- ElevenLabs: about 440 credits per clip of 40 to 55 words with the tool's default of two
  takes; the six pilot clips cost about 3,000 credits. Generating six clips at once hits
  a concurrency limit and the queue drains by itself.

### Costs a later session should expect
- A chapter the size of chapter 5 took three section builders about 20 to 25 minutes each
  in parallel, plus integration.

### Open questions for the reader
See the hand-off message at the end of the pilot session; unanswered ones carry over:
1. Approve the pilot's format and depth as the template for the remaining chapters?
2. Captions default (currently off).
3. Whether `reports/verify-*` from each machine should be committed to the repo.
4. Order of the remaining chapters (PLAN.md section 9).
