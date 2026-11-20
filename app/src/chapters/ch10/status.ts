/**
 * What has been checked for each listing in chapter 10, in the course's fixed vocabulary
 * (CLAUDE.md, "Verification vocabulary"). One place, used by every section.
 *
 * These are fixed when the chapter is built, as chapter 5 does, and are not read from
 * reports/runs/*.json: those files hold whatever `pmpp.py run` wrote last on this machine,
 * and "verified on CUDA" or "verified on HIP" may only come from a reports/verify-*.md
 * written on such a machine. No such report exists for chapter 10.
 *
 * Checked on 2026-11-20: every lab's reference solution passed its tests on Metal and on
 * both emulators (emu, race); every .cu file under labs/ch10 passed the CUDA front-end
 * check; the snippets run on the emulator through their unit tests.
 */
export const STATUS = {
  /** A lab's reference solution in its .cu file, which is written to compile for CUDA and HIP. */
  cu: 'front-end-checked; emulator-checked; unverified on CUDA and HIP hardware',
  /** A lab's Metal kernel or host code. */
  metal: 'verified on Metal',
  /** A correct snippet in CUDA, run on the emulator by its unit test. */
  snippet: 'front-end-checked; emulator-checked; unverified on CUDA and HIP hardware',
  /** A book listing kept as printed, with the 32-bit mask literal HIP rejects. */
  printed: 'front-end-checked; emulator-checked; unverified on CUDA hardware; as printed, CUDA only',
  /** A snippet that is wrong on purpose: its behaviour, not its correctness, was checked. */
  wrong: 'front-end-checked; wrong on purpose (behaviour emulator-checked)',
} as const;
