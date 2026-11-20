// Which Unified Memory paradigm a CUDA (or HIP) device offers, from three device
// attributes. Appendix C, §C.7 (book pp. 627–628) and the labs page.
//
// Not in the book. The table is NVIDIA's (CUDA Programming Guide 2.6.2.1, quoted in
// notes/today/appC-facts.md A14):
//
//   | Paradigm                                   | Attributes                                      |
//   | Limited (Windows, WSL, some Tegra)         | ConcurrentManagedAccess = 0                     |
//   | Full, managed allocations only             | ConcurrentManagedAccess = 1, PageableMemoryAccess = 0 |
//   | Full, all memory, software coherence (HMM) | both 1, PageableMemoryAccessUsesHostPageTables = 0 |
//   | Full, all memory, hardware coherence (ATS) | all three 1                                     |
//
// The capability checklist comes from A10 (the limited paradigm's rules), A11 (full
// support: migration on use, oversubscription, page or cache-line granularity) and
// A12 (system memory needs PageableMemoryAccess; cudaMalloc memory is never
// host-accessible, even with hardware coherence). HIP has attributes with the same
// names and meanings (B5); Metal has none of them (C5).
//
// Two combinations are not rows of NVIDIA's table: UsesHostPageTables = 1 with
// PageableMemoryAccess = 0, and PageableMemoryAccess = 1 with ConcurrentManagedAccess
// = 0. The model reports them as unlisted and does not guess what they mean.
//
// No imports and only erasable TypeScript, so browser checks can load this file
// directly with Node.

export interface Attrs {
  /** cudaDevAttrConcurrentManagedAccess */
  concurrentManagedAccess: boolean;
  /** cudaDevAttrPageableMemoryAccess */
  pageableMemoryAccess: boolean;
  /** cudaDevAttrPageableMemoryAccessUsesHostPageTables */
  usesHostPageTables: boolean;
}

export type ParadigmKey = 'limited' | 'managed' | 'hmm' | 'ats';

export interface Row {
  key: ParadigmKey;
  name: string;
  /** The attribute condition, as NVIDIA's table writes it. */
  condition: string;
  /** Where NVIDIA says you find it. */
  where: string;
  /** Facts items. */
  source: string;
}

export const ROWS: Row[] = [
  {
    key: 'limited',
    name: 'Limited',
    condition: 'ConcurrentManagedAccess = 0',
    where: 'Windows, including WSL, and some Tegra systems; devices of compute capability below 6.0',
    source: 'A10, A14',
  },
  {
    key: 'managed',
    name: 'Full, managed allocations only',
    condition: 'ConcurrentManagedAccess = 1, PageableMemoryAccess = 0',
    where: 'compute capability 6.0 or later without pageable memory access, for example Linux without HMM',
    source: 'A12, A14',
  },
  {
    key: 'hmm',
    name: 'Full, all memory, software coherence (HMM)',
    condition: 'both 1, PageableMemoryAccessUsesHostPageTables = 0',
    where: 'Linux 6.1.24+, 6.2.11+ or 6.3+, compute capability 7.5 or later, driver 535+ with the open kernel modules',
    source: 'A12, A14',
  },
  {
    key: 'ats',
    name: 'Full, all memory, hardware coherence (ATS)',
    condition: 'all three 1',
    where: 'Grace Hopper and Grace Blackwell (NVLink chip-to-chip)',
    source: 'A12, A14',
  },
];

export interface Paradigm {
  row: Row;
  /** False for a combination NVIDIA's table does not list. */
  listed: boolean;
  /** Why it is unlisted, when it is. */
  note?: string;
}

export function paradigm(a: Attrs): Paradigm {
  const row = (k: ParadigmKey) => ROWS.find((r) => r.key === k) as Row;
  if (!a.concurrentManagedAccess) {
    if (a.pageableMemoryAccess || a.usesHostPageTables)
      return {
        row: row('limited'),
        listed: false,
        note: 'NVIDIA lists the limited paradigm by ConcurrentManagedAccess = 0 alone and does not say what the other two attributes mean alongside it.',
      };
    return { row: row('limited'), listed: true };
  }
  if (!a.pageableMemoryAccess) {
    if (a.usesHostPageTables)
      return {
        row: row('managed'),
        listed: false,
        note: 'UsesHostPageTables = 1 with PageableMemoryAccess = 0 is not a row of NVIDIA\'s table. Read as managed only, since system memory needs PageableMemoryAccess.',
      };
    return { row: row('managed'), listed: true };
  }
  return { row: row(a.usesHostPageTables ? 'ats' : 'hmm'), listed: true };
}

export type Answer = 'yes' | 'no' | 'not stated';

export interface Check {
  key: 'malloc' | 'concurrent' | 'oversubscribe' | 'fault' | 'cudamalloc-host';
  what: string;
  answer: Answer;
  /** One line of why, with the facts item. */
  why: string;
}

/** What works on a machine with these attributes. */
export function checklist(a: Attrs): Check[] {
  const full = a.concurrentManagedAccess;
  const malloc: Answer = full ? (a.pageableMemoryAccess ? 'yes' : 'no') : a.pageableMemoryAccess ? 'not stated' : 'no';
  return [
    {
      key: 'malloc',
      what: 'A kernel uses a pointer from malloc, new or mmap',
      answer: malloc,
      why:
        malloc === 'yes'
          ? 'PageableMemoryAccess = 1: all system memory behaves as unified memory, file-backed mmap included (A12).'
          : malloc === 'no'
            ? 'PageableMemoryAccess = 0: only memory allocated as managed behaves as unified memory (A12).'
            : 'An unlisted combination; NVIDIA does not say (A14).',
    },
    {
      key: 'concurrent',
      what: 'The CPU touches managed memory while a kernel runs',
      answer: full ? 'yes' : 'no',
      why: full
        ? 'ConcurrentManagedAccess = 1: coherent concurrent access (A11, A14).'
        : 'Not while the GPU is active; concurrent access can end in a segmentation fault (A10).',
    },
    {
      key: 'oversubscribe',
      what: 'Managed allocations larger than GPU memory',
      answer: full ? 'yes' : 'no',
      why: full ? 'Oversubscription is allowed (A11).' : 'No more managed memory than the GPU physically has (A10).',
    },
    {
      key: 'fault',
      what: 'Managed pages move on demand, when the GPU touches them',
      answer: full ? 'yes' : 'no',
      why: full
        ? 'Migrated when used by another processor, by fault and migrate by default (A11, A13).'
        : 'All managed memory generally moves to the GPU at each launch, and back at synchronisation (A10).',
    },
    {
      key: 'cudamalloc-host',
      what: 'Host code dereferences a cudaMalloc pointer',
      answer: 'no',
      why: 'Never, on any paradigm: hardware coherence does not open GPU-only allocations to the host (A12).',
    },
  ];
}

/** The unit in which managed data moves or stays coherent (A10, A11). */
export function granularityText(a: Attrs): string {
  const k = paradigm(a).row.key;
  return k === 'ats'
    ? 'cache lines: hardware coherence (A11)'
    : k === 'limited'
      ? 'all managed memory, moved in bulk at launch and synchronisation (A10)'
      : 'pages: software coherence (A11)';
}

// ---------------------------------------------------------------- lab C.1's 'paradigm' bench row

/** The minimal shape of a backend's run in reports/runs/*.json (see app/src/lib/reports.ts). */
export interface RunLike {
  results: { lab: string; variant: string; bench: Record<string, string | number>[] }[];
}

const norm = (k: string) => k.toLowerCase().replace(/[^a-z]/g, '');
const truthy = (v: string | number | undefined): boolean | undefined => {
  if (v == null) return undefined;
  if (typeof v === 'number') return isFinite(v) ? v !== 0 : undefined;
  const s = v.trim().toLowerCase();
  if (s === '1' || s === 'true' || s === 'yes') return true;
  if (s === '0' || s === 'false' || s === 'no') return false;
  return undefined;
};

/**
 * Reads the three attributes from a bench row. Field names are matched loosely: lab
 * C.1 (labs/appC/lab1_feeding/main.cpp) writes concurrent_managed, pageable_access and
 * uses_host_page_tables, and the full attribute names (concurrentManagedAccess,
 * cudaDevAttrConcurrentManagedAccess, hipDeviceAttributeConcurrentManagedAccess, ...)
 * work too.
 * Returns null unless all three are present and read as 0 or 1.
 */
export function parseParadigmRow(row: Record<string, string | number>): Attrs | null {
  let cma: boolean | undefined;
  let pma: boolean | undefined;
  let host: boolean | undefined;
  for (const [k, v] of Object.entries(row)) {
    const n = norm(k);
    if (n.includes('useshostpagetables')) host = truthy(v);
    else if (n.includes('concurrentmanaged')) cma = truthy(v);
    else if (n.includes('pageable')) pma = truthy(v);
  }
  if (cma == null || pma == null || host == null) return null;
  return { concurrentManagedAccess: cma, pageableMemoryAccess: pma, usesHostPageTables: host };
}

/** Lab C.1's directory, as the lab runtime names results: "appC/lab1_feeding". */
export const isLabC1 = (lab: string) => /^appC\/lab1(_|$)/.test(lab);

/**
 * The attributes lab C.1 reported on one backend, from its bench row named
 * "paradigm" (solution first, then work), or null when the lab has not run there.
 */
export function paradigmFromRun(run: RunLike | undefined): Attrs | null {
  if (!run) return null;
  const order = ['solution', 'work', 'starter'];
  const rs = run.results
    .filter((r) => isLabC1(r.lab))
    .sort((a, b) => order.indexOf(a.variant) - order.indexOf(b.variant));
  for (const r of rs)
    for (const row of r.bench) {
      if (String(row.name ?? '').toLowerCase() !== 'paradigm') continue;
      const a = parseParadigmRow(row);
      if (a) return a;
    }
  return null;
}
