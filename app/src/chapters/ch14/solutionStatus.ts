import { labStatus, useReports, type Reports } from '../../lib/reports';

// The status a lesson shows on a tab that holds part of a lab's reference solution, in the
// course's fixed vocabulary and computed from this machine's reports exactly as the labs page
// computes it (components/LabKit.tsx, SolutionCode), so a lesson and the labs page never
// disagree about the same file.

/** kernel.cu (CUDA, HIP and the emulator) and the Metal files of one lab's reference solution. */
export function solutionStatus(reports: Reports, lab: string): { cu: string; metal: string } {
  const st = (b: string) => labStatus(reports[b], lab, 'solution');
  const got = [
    st('cuda') === 'pass' ? 'verified on CUDA' : '',
    st('hip') === 'pass' ? 'verified on HIP' : '',
    st('emu') === 'pass' ? 'emulator-checked' : '',
  ].filter(Boolean);
  const lacking = [st('cuda') === 'pass' ? '' : 'CUDA', st('hip') === 'pass' ? '' : 'HIP'].filter(Boolean);
  const cu = got.length
    ? got.join(', ') + (lacking.length ? `; unverified on ${lacking.join(' and ')}` : '')
    : 'unverified (no run on this machine)';
  const metal =
    st('metal') === 'pass'
      ? 'verified on Metal'
      : st('metal') === 'fail'
        ? 'unverified (failed on this machine)'
        : 'unverified (no run on this machine)';
  return { cu, metal };
}

/** The same, read from /api/reports for a lesson page. */
export function useSolutionStatus(lab: string): { cu: string; metal: string } {
  const { reports } = useReports();
  return solutionStatus(reports, lab);
}
