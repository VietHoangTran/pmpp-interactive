import { solutionTabStatus } from '../../components/LabKit';
import { useReports, type Reports } from '../../lib/reports';

// The status a lesson shows on a tab that holds part of a lab's reference solution: the labs
// page's own wording (components/LabKit.tsx, solutionTabStatus), computed from this machine's
// reports, so a lesson and the labs page never disagree about the same file.

/** kernel.cu (CUDA, HIP and the emulator) and the Metal files of one lab's reference solution. */
export function solutionStatus(reports: Reports, lab: string): { cu: string; metal: string } {
  return { cu: solutionTabStatus(reports, lab, 'kernel.cu'), metal: solutionTabStatus(reports, lab, 'kernel.metal') };
}

/** The same, read from /api/reports for a lesson page. */
export function useSolutionStatus(lab: string): { cu: string; metal: string } {
  const { reports } = useReports();
  return solutionStatus(reports, lab);
}
