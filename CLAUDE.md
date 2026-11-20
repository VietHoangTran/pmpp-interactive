# Project conventions

An interactive textbook built from *Programming Massively Parallel Processors*, 5th ed.,
for one reader: a senior AI/ML engineer new to GPU kernels who wants to write LLM-inference
kernels on CUDA, HIP and Metal. Read `PLAN.md` for the curriculum and `PROGRESS.md` for the
current state before doing anything.

## Ground rules

- **Ground everything in the PDF, through the notes.** Work from `notes/chNN.md`; do not
  write from memory of the book or of earlier editions. Every lesson section carries its
  book section and pages (`<Source>`). If the notes are not enough, read the PDF pages
  (as images: code listings exist only as images) and update the notes.
- **Own words.** Explain and re-illustrate. Never paste the book's prose or figures. Short
  code quotes are fine where the exact expression is the point.
- **The book has errata.** Each notes file ends with "Unclear in the PDF". Present
  corrected, compiled code and say where it differs from the page.
- **Not in the book = say so.** HIP, Metal, sampling and quantised kernels are ours. Label
  them, and check every HIP/Metal/"today" claim against current docs first
  (`notes/today/`). Do not assert vendor facts from memory.
- **No device-specific numbers** in lessons or labs (memory sizes, core counts, timings).
  The book's H100 figures appear only as the book's example. Reader values come from
  `python3 labs/pmpp.py doctor` and lab benchmarks (`reports/runs/*.json`).
- **Every simulation states what it simplifies** (`<Simplification>`).
- **Active over passive.** Each section has the reader predict, manipulate, write or debug.
  Questions need reasoning, not recall, and feed the spaced-review pool.

## Verification vocabulary (use exactly these)

`verified on Metal` (compiled and run here), `emulator-checked` (CPU emulator, logic only),
`front-end-checked` (clang CUDA parse against stubs), `unverified`. A reference solution
counts as verified on CUDA or HIP only when a `reports/verify-*.md` from that machine
shows it passing. Never write "verified" for anything else.

## Commands

- `./start.sh` : run the app (installs dependencies on first run).
- `./verify.sh` : build, test and benchmark every lab on every available backend and
  write `reports/verify-*.md`. `--quick` skips benchmarks.
- `python3 labs/pmpp.py doctor | run 5.2 [--bench] [--solution] [--backend emu] | reset 5.2 --yes`
- `node tools/narration.mjs status` : which narration clips are missing or stale.
- `cd app && npm test` : unit tests for the simulators.

## Labs

- Layout: `labs/chNN/labM_name/{starter,work,solution}/{kernel.cu,kernel.metal,launch.mm}`,
  plus `main.cpp` (driver) and a one-line `CMakeLists.txt` calling `pmpp_add_lab`.
- `work/` starts as a copy of `starter/` and belongs to the reader. Never overwrite it.
- One `.cu` source serves CUDA, HIP (through `pmpp/vendors/hip.h`) and the emulator
  (through `tools/cu2emu.py` and `pmpp/emu/cuda_emu.h`). Student-facing code uses the real
  API of its backend: plain CUDA in `.cu`, MSL in `.metal`, Metal calls in `.mm`.
- A new lab is done only when: the solution passes on Metal and the emulator, the
  untouched starter builds and fails, and each planted defect is shown to fail alone.
- Test against a CPU reference with a tolerance (Metal does not default to strict IEEE).

## App

- `app/src/chapters/chNN/` holds a chapter: `index.ts` (metadata, section list) and one
  component per section. Chapter 5 is the template.
- Shared pieces live in `app/src/components` (Source, Today, Simplification, Predict,
  Quiz, CodeTabs, LabCard, ScenePlayer) and `app/src/sim` (simulators with unit tests).
- Progress lives in `progress/progress.json` via `/api/progress`. Keys are stable ids
  (`ch05/5.1/q-flopb`); never rename an id once shipped.
- Style: plain, direct prose. No decoration for its own sake. Dark and light themes.

## Narration

- Only the ElevenLabs MCP speech tool, model `eleven_v4`, voice `tnSpp4vdxKPjI9w0GnoV`
  (the reader's decision on 2026-11-20 after `eleven_v4_turbo` proved unavailable through
  the MCP server). Never another model or voice, never the API directly. If the tool is
  unavailable, stop and say so.
- Script `content/chNN/narration/<clip>.txt` beside `<clip>.mp3`; register with
  `node tools/narration.mjs add <chapter> <clip> <audio-file>`. Regenerate only clips the
  status command reports as stale. Commit the audio.
- Write for the ear: short sentences, no symbols read aloud, numbers spelled the way they
  are said.

## Repository

- The PDF and `.book/` (extracted text) are git-ignored and must stay local.
- Commit `reports/verify-*` you want to keep; `reports/runs/` is per-machine and ignored.
