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
- Shared pieces live in `app/src/components`: `notes.tsx` (Source, Today, Simplification,
  Beyond, Aside, Bench, TierLegend), `Ask.tsx` (Ask, Predict), `Code.tsx` (Code,
  useLabFile), `Scene.tsx` (the narrated-animation player), `LabKit.tsx` (everything a
  labs page needs; read its header comment). Simulators with unit tests live in
  `app/src/sim`.
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

## Writing a lesson section (chapter 5 is the template)

- One file per section, `app/src/chapters/chNN/S<id>.tsx`, exporting the component and a
  `questions...` record; `index.tsx` lists the sections and merges the questions.
- The component returns a fragment whose children are laid out by the `.prose` grid:
  paragraphs and headings in the text column, `<Bench>` and `<Code>` full width, margin
  notes (`<Source>`, `<Today>`, `<Simplification>`, `<Beyond>`, `<Aside>`) in the margin.
- **A margin note goes immediately after the paragraph it belongs to**, never after a
  bench or code block (it would open a blank gap). Keep notes short.
- Colour only for memory tiers (`--reg`, `--shared`, `--global`, `--const`) and failure
  (`--fail`) / success (`--ok`). Use the CSS variables; never hard-code colours.
- Interactives read device numbers from `useReports()`; the book's example device is
  offered as "the book's example" and nothing else is hard-coded.
- Each interactive gets a browser check in `app/e2e/<section>.mjs` that drives it and
  asserts what it shows. `node e2e/run-all.mjs` runs them all against the running app.
- **Captions are an option on every narrated animation** (the reader's instruction,
  2026-11-20): a "captions" switch, off by default and remembered in the browser, showing
  the clip's script. Use the shared `<Scene>` player, which provides it; never build a
  video or animation with narration that lacks the switch. A beat with no audio shows its
  script regardless.

## Building a chapter (after the pilot)

Chapter 5 is the template: read `app/src/chapters/ch05/` and `labs/ch05/` before writing.
`PLAN.md` gives each chapter's depth tier and project milestone.

**Where things go** (NN = chapter id: `ch06`, `appA`, `x1`, ...)
- `app/src/chapters/NN/index.tsx`: metadata and one `load` per section. Section ids are
  `start`, the book's section numbers (`6.1`), `labs`, `wrap`. Never import a section
  statically anywhere; sections and chapters load on demand.
- `app/src/chapters/NN/S<id>.tsx`: one section each, exporting a component and a
  `questions...` record. Question ids: `NN/<section>/q-<slug>`; predictions `.../p-<slug>`.
- `app/src/sim/<name>.ts` + `<name>.test.ts`: any model an interactive runs on. Models
  are tested; components do not re-implement kernel logic.
- `app/e2e/NN-<section>.mjs`: a browser check per section that drives every interactive
  and asserts what it shows. It must intercept `PUT /api/progress` (see ch05 scripts).
- `labs/NN/labM_name/{starter,work,solution}` + `main.cpp` + `CMakeLists.txt` + `README.md`.
- `content/NN/narration/<scene>-<n>.txt`: narration scripts.
- `notes/today/NN-facts.md`: HIP / Metal / "today" facts checked against current docs.
- Do not edit `registry.ts`, shared components, `app.css`, the lab runtime or another
  chapter's files. If a shared piece needs to change, say so in your report.

**Depth**
- Deep: every book section gets a page; several interactives; 3 or more labs.
- Standard: every book section gets a page; 1 or 2 labs.
- Condensed: one or two pages that keep the transferable technique and name plainly
  what was left out and where it is in the book; at most one optional lab.
- Always: a `start` page (what the chapter is for, a prerequisite check drawn from
  earlier chapters' questions and notes) and a `wrap` page (what to carry forward, a
  mixed check, a map of the book's exercises to where they are worked, the PDF's
  apparent errata for the chapter, the project milestone).

**Choosing the format for an idea**
- Something the reader should discover by changing a parameter: an explorer.
- A process that unfolds step by step under the reader's control: a stepper.
- A process that unfolds in time and is best watched once with an explanation: a narrated
  animation (`<Scene>`). Use them freely where they fit; budget is not the constraint
  (reader's instruction, 2026-11-20). They must show something a still picture cannot,
  and every beat's drawing must be computed from the same tested model as the
  interactives, not hand-drawn numbers.
- A claim the reader can test on their own GPU: a lab with a prediction made first.
- Bridge to what the reader already knows: where this sits under a PyTorch call, an
  attention layer, a sampler, a quantised model. One concrete bridge per section at most.

**Narration**
- Authors write scripts and the `<Scene>` beats only. Audio is generated centrally by the
  main session through the ElevenLabs MCP tool (`eleven_v4`, the fixed voice). Never
  call a speech tool or API from a chapter build.
- A script is 35 to 60 words per beat, written for the ear: short sentences, digits for
  numbers, no symbols read aloud, `[short pause]` sparingly, single letters for names
  (M, N, T) are fine.

**Labs**
- Student-facing code is the real API of its backend. One `.cu` serves CUDA, HIP and the
  emulators; Metal gets `kernel.metal` and `launch.mm`. If Metal cannot express the
  lab (say so, with the fact that shows it), the lab may be CUDA-family only.
- Use your own build directory and never the default one:
  `export PMPP_BUILD_DIR=$PWD/labs/build-NN` before any `labs/pmpp.py` call.
  Do not run `./verify.sh` (it rewrites shared reports); run labs with `pmpp.py run`.
- Done means: solution passes on Metal and on the emulators that apply; untouched starter
  builds and fails; every planted defect fails on its own; `.cu` files pass the CUDA
  front-end check; the lab has a README with a prediction to make before benchmarking.
- Benchmarks report what was measured. No expected values, no device constants.
- Project milestones follow these conventions so later chapters can compose them:
  row-major `float` tensors unless the lab is about another type; batch dimension first;
  raw device pointers and explicit sizes as kernel arguments; a `launch<Name>` host
  function beside each kernel; the CPU reference lives in `main.cpp`.

**Running the app while building**
- Start your own server on your own port and point checks at it:
  `cd app && PORT=<port> node server.mjs &` then `APP_URL=http://localhost:<port> node e2e/NN-x.mjs`.
  Stop it when done. Screenshots: `APP_URL=... node e2e/shot.mjs NN/6.1 <file outside the repo>`.
- `npx tsc --noEmit` covers the whole app; other chapters may be mid-build, so judge
  only errors in your own files. Run your own unit tests with `npx vitest run <path>`.

**Accuracy**
- What the book says comes from `notes/NN.md` only, with pages. What vendors say comes
  from `notes/today/*.md` only. Anything else is labelled as ours (`<Beyond>`), and
  anything unchecked is labelled unverified.
- Present the book's device figures as the book's example. Reader figures come from
  `useReports()`.
- Code shown in a lesson is a lab file loaded with `useLabFile`, or a snippet stored in
  `labs/NN/snippets/` with a unit test tying the lesson copy to the file.
- Where the PDF looks wrong, say so (the notes list these) and show the corrected,
  tested version.
