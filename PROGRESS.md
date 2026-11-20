# Progress

Running state across sessions. Newest first. Decisions are dated.

## State on 2026-11-20

**Phase:** pilot (chapter 5) under construction. Nothing after the pilot starts until the
reader approves it and has run `./verify.sh` on a CUDA and a ROCm machine.

### Done
- Whole book read as page images; notes for all 24 chapters and appendices A to C in
  `notes/` (chapter 5 by the main session, the rest by per-chapter readers).
- `notes/pagemap.json`: PDF page = book page + offset (28, 27, 26, 25 or 24 by chapter).
- `notes/today/ch05-facts.md`: 28 CUDA / HIP / Metal claims checked against current docs.
- `PLAN.md`, `CLAUDE.md`.
- Lab infrastructure: CMake backend detection, runtime, CPU emulator, CUDA front-end
  check, `labs/pmpp.py`, `./verify.sh`.
- Chapter 5 labs 5.1 to 5.4 with starters and reference solutions.

### Verification status of chapter 5 reference solutions
| Lab | Metal | emu | CUDA | HIP |
|---|---|---|---|---|
| 5.1 roofline | verified on Metal | emulator-checked | front-end-checked, unverified on device | unverified |
| 5.2 tiled | verified on Metal | emulator-checked | front-end-checked, unverified on device | unverified |
| 5.3 gemm | verified on Metal | emulator-checked | front-end-checked, unverified on device | unverified |
| 5.4 race hunt | verified on Metal | emulator-checked | front-end-checked, unverified on device | unverified |

Evidence: `reports/verify-20261120-*-darwin-emu+metal.md`. The CUDA and HIP CMake paths
and `pmpp/vendors/hip.h` have never been compiled.

### In progress
- The app and the chapter 5 lessons.
- Narration clips for chapter 5.

### Decisions
- 2026-11-20: narration model is `eleven_v4` through the MCP tool. The reader asked for
  `eleven_v4_turbo`; the MCP server offers only `eleven_v4`, `eleven_v3`,
  `eleven_turbo_v2_5`, `eleven_flash_v2_5`, `eleven_multilingual_v2`. Reader chose
  `eleven_v4`.
- 2026-11-20: CPU emulation uses our own small emulator, not AMD's HIP-CPU (does not build
  with Apple clang, needs GCC + TBB, unmaintained since 2024-03).
- 2026-11-20: pilot chapter is 5 (memory and data locality, tiled matmul).

### Findings worth remembering
- On the Apple M3 Max used for the pilot, tiling gave about 1.9x over the naive kernel and
  a 4x4 tile was slower than naive; the naive kernel ran about 10x above the book's
  0.25 FLOP/B model ceiling. Caches serve most of the naive kernel's reads. Lessons must
  have the reader predict and measure, not promise the book's 32x.
- Missing zero-fill on only one of the two tiles is invisible to tests (the other operand
  is zero) unless the stale value is NaN or infinity. Lab 5.4's defect omits both.
- The book never mentions HIP, Metal, sampling or quantised kernels, and gives no measured
  timings in chapters 15 or 20.

### Open questions for the reader
See the hand-off message at the end of the pilot.
