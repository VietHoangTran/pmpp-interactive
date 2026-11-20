# Progress

Running state across sessions. Newest first. Decisions are dated.

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
- [ ] Wave 0: on-demand loading of chapters and sections, chapter-building rules in
      CLAUDE.md (done); shared labs-page kit `app/src/components/LabKit.tsx`; emulator
      extension (warps, warp intrinsics, atomics, constant memory, cooperative groups,
      streams, a ThreadSanitizer `race` backend); `notes/today/infra-facts.md`.
- [ ] Wave 1: ch02, ch03, ch04 (Deep); ch01, ch24 (Condensed).
- [ ] Wave 2: ch06 (Deep), ch07 (Standard), ch08 (Condensed), ch09 (Standard), ch10 (Deep), appC (Standard).
- [ ] Wave 3: ch15, ch11, ch12 (Deep), ch14 (Standard), ch13 (Condensed).
- [ ] Wave 4: ch20 (Deep), ch22, ch19, appA (Standard), x1 sampling (Deep).
- [ ] Wave 5: x2 quantised kernels (Deep), x3 reading real backends (Standard),
      ch16, ch17, ch18, ch21, ch23 (Condensed).
- [ ] Final pass to-do list (collected along the way):
      lab 5.5's prediction compares GFLOP/s at different matrix widths (2048 vs lab 5.2's
      adaptive width): make the two benchmarks use the same width or reword the prompt;
      the labs page uses two wordings for the same statuses (table vs tabs) and the
      kernel.cu tab says "no run" when a run failed; `app-review.mjs` only visits
      chapter 5 routes: extend to every built chapter.
- [ ] Final: whole-course consistency pass, spaced-review pool check, final verify
      report, hand-off.

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
