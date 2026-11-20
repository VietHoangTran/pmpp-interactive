import { solutionStatus, type Lab } from '../../components/LabKit';
import type { Reports } from '../../lib/reports';

/**
 * The status line of a lab's CUDA file, which also serves HIP and the emulators, from this
 * machine's reports: "emulator-checked; unverified on CUDA and HIP hardware" until a CUDA
 * or HIP run of the reference solution passes, then that run's verdict.
 */
export function cuStatus(reports: Reports, lab: Lab): string {
  const e = solutionStatus(reports, 'emu', lab);
  const c = solutionStatus(reports, 'cuda', lab);
  const h = solutionStatus(reports, 'hip', lab);
  const dev = [c, h].every((s) => s.tone !== 'ok')
    ? 'unverified on CUDA and HIP hardware'
    : [c.tone === 'ok' ? c.text : 'unverified on CUDA', h.tone === 'ok' ? h.text : 'unverified on HIP'].join('; ');
  return `${e.tone === 'ok' ? e.text : 'emulator: ' + e.text}; ${dev}`;
}
