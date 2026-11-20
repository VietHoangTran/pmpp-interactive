// Appendix C as data: who may dereference which pointer, era by era.
//
// The book tells the history of CUDA's memory model as a sequence of steps
// (pp. 623–629). This file turns it into a table: for each era, each kind of
// pointer (a row) and each place that might dereference it (a site), one verdict
// class, one sentence on cost, and the source of the rule. The book's own
// assembled table is in notes/appC.md ("Who can dereference what"); the rules
// for today, for HIP and for Metal come from notes/today/appC-facts.md (items
// A1 to D5 and its table "The notes' dereference table, brought up to date").
// A source written as "A12" means that facts file; "ch02-facts A17" means
// notes/today/ch02-facts.md, item A17; "p. 625" means the book.
//
// What the model simplifies, and what the page says wherever it is shown:
//   - One cell is one documented rule. Real systems add conditions (driver
//     version, operating system, device attributes) that the cell's sentence
//     names only when they decide the answer.
//   - Costs are classes, not numbers: local, over the interconnect, pages moving
//     on first touch, pages moved in bulk at launch.
//   - The eras follow the book's order of sections, which is not always the
//     order of the dates (CUDA 2.2 shipped before Fermi products did, A2, A3).
//   - "Windows, WSL and some Tegra systems today" behave like the CUDA 6 era on
//     Kepler and Maxwell (A10); the era's label says so instead of repeating it.
//   - A row the book never discusses (registered memory before unified
//     addressing) is "not documented" rather than a guess.
//
// No runtime imports, so browser checks can load this file directly.

export type EraId = 'copies' | 'fermi' | 'zerocopy' | 'uva' | 'kepler' | 'um-limited' | 'pascal' | 'vmm' | 'hmm';
export type Platform = 'cuda' | 'hip' | 'metal';
export type RowId = 'malloc' | 'pinned' | 'registered' | 'device' | 'peer' | 'managed' | 'shared' | 'vmm';
export type SiteId = 'host' | 'arg' | 'chased';
export type Verdict =
  | 'local'
  | 'remote'
  | 'fault'
  | 'bulk'
  | 'other-ptr'
  | 'not-while-running'
  | 'after-enable'
  | 'no'
  | 'not-yet'
  | 'unknown';

export interface Cell {
  verdict: Verdict;
  /** One sentence: what it costs or why not. */
  cost: string;
  /** Book pages and/or facts items, in the notation described above. */
  source: string;
}

export interface Era {
  id: EraId;
  /** Label on the timeline strip: the book's section, or where the era comes from. */
  strip: string;
  title: string;
  /** The release or architecture that brought it. */
  vehicle: string;
  /** The date as the book gives it, or null when the era is not in the book. */
  book: string | null;
  /** The date as our sources give it, with the facts item, when it differs or the book gives none. */
  corrected: { date: string; item: string } | null;
  /** Book pages, or null for the era beyond the book. */
  pages: string | null;
  /** What this step changes, in one sentence. */
  adds: string;
  /** A remark that belongs to this era (an ordering caveat, the systems that still behave this way). */
  note?: string;
}

export const ERAS: Era[] = [
  {
    id: 'copies',
    strip: 'p. 623',
    title: 'Copies only',
    vehicle: 'early GPU generations, and chapter 2',
    book: 'early generations',
    corrected: null,
    pages: '623',
    adds: 'Two physical memories. Data goes in with cudaMemcpy before a kernel and comes out after it.',
  },
  {
    id: 'fermi',
    strip: 'C.1',
    title: 'One address space on the GPU',
    vehicle: 'Fermi',
    book: '2009',
    corrected: { date: 'whitepaper 2009; products reached the market in March 2010', item: 'A2' },
    pages: '624',
    adds: 'Shared, local and global memory share one address space, so one pointer type serves all three.',
  },
  {
    id: 'zerocopy',
    strip: 'C.2',
    title: 'Zero-copy host memory',
    vehicle: 'CUDA 2.2',
    book: '2009',
    corrected: { date: 'May 2009, as the book says', item: 'A3' },
    pages: '624–625',
    adds: 'A kernel reads pinned host memory in place, over the interconnect, through a second pointer value.',
    note: 'The map follows the book\'s order. By the dates, CUDA 2.2 (May 2009) came before Fermi products (March 2010).',
  },
  {
    id: 'uva',
    strip: 'C.3',
    title: 'Unified virtual address space',
    vehicle: 'CUDA 4',
    book: '2011',
    corrected: { date: 'May 2011, as the book says', item: 'A5' },
    pages: '625',
    adds: 'One virtual address space for host and devices: a pointer\'s value says where it lives. Who may dereference it barely changes.',
  },
  {
    id: 'kepler',
    strip: 'C.4–C.5',
    title: 'Large addresses and peer GPUs',
    vehicle: 'Kepler',
    book: '2013',
    corrected: { date: 'March 2012; and Fermi already had 64-bit addressing and more than 4 GB', item: 'A6' },
    pages: '625–626',
    adds: 'Addresses wide enough for more than 4 GB, and one GPU dereferencing another GPU\'s memory.',
  },
  {
    id: 'um-limited',
    strip: 'C.6',
    title: 'Unified Memory, limited',
    vehicle: 'CUDA 6 on Kepler and Maxwell',
    book: '2013',
    corrected: { date: 'announced November 2013, shipped April 2014', item: 'A9' },
    pages: '626–627',
    adds: 'Managed memory: one pointer for CPU and GPU, moved to the GPU in bulk at every launch.',
    note: 'Windows, WSL and some Tegra systems still behave exactly like this today (A10).',
  },
  {
    id: 'pascal',
    strip: 'C.7–C.8',
    title: 'GPU page faults',
    vehicle: 'Pascal and CUDA 8',
    book: '2016',
    corrected: { date: 'CUDA 8 shipped in September 2016; the book gives 2016 for Pascal', item: 'A11' },
    pages: '627–629',
    adds: '49-bit addresses and GPU page faults: managed pages move when first touched, not at launch.',
    note: 'This is Linux with a GPU of compute capability 6.0 or later and no HMM (A12, A14).',
  },
  {
    id: 'vmm',
    strip: 'C.9',
    title: 'Address space control',
    vehicle: 'the virtual memory management calls',
    book: 'CUDA 11, no year',
    corrected: { date: 'CUDA 10.2, November 2019', item: 'A15' },
    pages: '629',
    adds: 'Reserve a range of addresses, create memory, map it into the range, and grant access to it.',
  },
  {
    id: 'hmm',
    strip: 'beyond the book',
    title: 'All host memory',
    vehicle: 'HMM on Linux, or ATS on Grace Hopper and Grace Blackwell',
    book: null,
    corrected: { date: 'NVIDIA\'s HMM post is from August 2023', item: 'A12' },
    pages: null,
    adds: 'Plain malloc memory works in kernels through the host\'s own pointer.',
    note: 'HMM needs Linux 6.1.24 or later, compute capability 7.5 or later and driver 535 or later with the open kernel modules (A12).',
  },
];

export const ERA_ORDER: EraId[] = ERAS.map((e) => e.id);
export const era = (id: EraId): Era => ERAS.find((e) => e.id === id)!;

export const ROWS: RowId[] = ['malloc', 'pinned', 'registered', 'device', 'peer', 'managed', 'shared', 'vmm'];
export const SITES: SiteId[] = ['host', 'arg', 'chased'];

export const ROW_LABEL: Record<Platform, Record<RowId, string>> = {
  cuda: {
    malloc: "malloc'd host memory",
    pinned: 'pinned host memory (cudaHostAlloc, cudaMallocHost)',
    registered: 'registered host memory (cudaHostRegister)',
    device: 'cudaMalloc on this GPU',
    peer: 'cudaMalloc on a peer GPU',
    managed: 'managed memory (cudaMallocManaged)',
    shared: 'a __shared__ variable',
    vmm: 'a VMM-mapped range (cuMemMap)',
  },
  hip: {
    malloc: "malloc'd host memory",
    pinned: 'pinned host memory (hipHostMalloc)',
    registered: 'registered host memory (hipHostRegister)',
    device: 'hipMalloc on this GPU',
    peer: 'hipMalloc on a peer GPU',
    managed: 'managed memory (hipMallocManaged)',
    shared: 'a __shared__ variable',
    vmm: 'a VMM-mapped range (hipMemMap)',
  },
  metal: {
    malloc: "malloc'd memory",
    pinned: 'a shared buffer (no pinned memory documented for Metal)',
    registered: 'your page-aligned memory, wrapped by newBufferWithBytesNoCopy',
    device: 'a private buffer',
    peer: "another GPU's buffer",
    managed: 'managed storage mode (deprecated)',
    shared: 'a threadgroup variable',
    vmm: 'a placement sparse buffer (Metal 4)',
  },
};

export const SITE_LABEL: Record<Platform, Record<SiteId, string>> = {
  cuda: { host: 'host code', arg: 'kernel, given the pointer as an argument', chased: 'kernel, pointer read from memory' },
  hip: { host: 'host code', arg: 'kernel, given the pointer as an argument', chased: 'kernel, pointer read from memory' },
  metal: { host: 'CPU code, through contents', arg: 'kernel, buffer bound as an argument', chased: 'kernel, address read from memory' },
};

export const VERDICT_LABEL: Record<Verdict, string> = {
  local: 'yes, local',
  remote: 'yes, over the interconnect',
  fault: 'yes, pages move on first touch',
  bulk: 'yes, moved in bulk at launch',
  'other-ptr': 'only through a different pointer value',
  'not-while-running': 'not while a kernel runs',
  'after-enable': 'only after access is enabled',
  no: 'no',
  'not-yet': 'does not exist yet',
  unknown: 'not documented',
};

/** The verdicts under which the dereference works, at some cost. */
export const WORKS: Verdict[] = ['local', 'remote', 'fault', 'bulk'];
export const works = (v: Verdict) => WORKS.includes(v);

/** Colour use: success and failure only; slow and conditional verdicts stay neutral. */
export function tone(v: Verdict): 'ok' | 'fail' | 'neutral' | 'absent' {
  if (v === 'local') return 'ok';
  if (v === 'no') return 'fail';
  if (v === 'not-yet') return 'absent';
  return 'neutral';
}

// ---------------------------------------------------------------- CUDA, era by era
type Grid = Record<RowId, Record<SiteId, Cell>>;
type Patch = Partial<Record<RowId, Partial<Record<SiteId, Cell>>>>;
const c = (verdict: Verdict, cost: string, source: string): Cell => ({ verdict, cost, source });

const COPIES: Grid = {
  malloc: {
    host: c('local', 'Ordinary host memory, read at host-memory speed.', 'p. 623'),
    arg: c('no', 'Kernels reach only device memory: copy the data in with cudaMemcpy first.', 'p. 623'),
    chased: c('no', 'A host pointer means nothing to a kernel in this model.', 'p. 623'),
  },
  pinned: {
    host: c('local', 'Host memory the operating system may not page out.', 'ch. 23, pp. 562–563'),
    arg: c('no', 'In the copy model even pinned memory is reached only through copies.', 'p. 623'),
    chased: c('no', 'In the copy model even pinned memory is reached only through copies.', 'p. 623'),
  },
  registered: {
    host: c('local', 'Your own malloc memory, page-locked where it is (not in the book).', 'A3'),
    arg: c('no', 'In the copy model kernels reach only device memory.', 'p. 623'),
    chased: c('no', 'In the copy model kernels reach only device memory.', 'p. 623'),
  },
  device: {
    host: c('no', 'Host code may only hand the pointer to API calls and kernels; dereferencing it is undefined.', 'ch. 2 §2.4, p. 30; ch02-facts A17'),
    arg: c('local', 'Device memory: the fast path.', 'p. 623'),
    chased: c('local', 'A device pointer stored in device memory can be followed.', 'p. 625'),
  },
  peer: {
    host: c('no', 'Host code cannot dereference any GPU\'s cudaMalloc memory.', 'ch. 2 §2.4, p. 30'),
    arg: c('no', 'Another GPU\'s data arrives only through copies the host starts.', 'p. 626'),
    chased: c('no', 'Another GPU\'s data arrives only through copies the host starts.', 'p. 626'),
  },
  managed: {
    host: c('not-yet', 'Managed memory arrives with CUDA 6 (C.6).', 'p. 626'),
    arg: c('not-yet', 'Managed memory arrives with CUDA 6 (C.6).', 'p. 626'),
    chased: c('not-yet', 'Managed memory arrives with CUDA 6 (C.6).', 'p. 626'),
  },
  shared: {
    host: c('no', 'Shared memory exists only on the GPU, one copy per block, while the block runs.', 'ch. 5 §5.2, pp. 98–104'),
    arg: c('no', 'A pointer could point only into global memory, so a helper could not take a pointer to shared data.', 'p. 624'),
    chased: c('no', 'A pointer could point only into global memory.', 'p. 624'),
  },
  vmm: {
    host: c('not-yet', 'Reserve-and-map control arrives with C.9.', 'p. 629'),
    arg: c('not-yet', 'Reserve-and-map control arrives with C.9.', 'p. 629'),
    chased: c('not-yet', 'Reserve-and-map control arrives with C.9.', 'p. 629'),
  },
};

const PATCHES: Record<EraId, Patch> = {
  copies: {},
  fermi: {
    shared: {
      arg: c('local', 'One generic pointer: the address falls in the shared window, so the load reads the block\'s shared memory.', 'p. 624; A2'),
      chased: c('local', 'A generic pointer read back from memory works too, inside the block that owns the variable.', 'A2; ch. 5 §5.2'),
    },
  },
  zerocopy: {
    pinned: {
      host: c('local', 'cudaHostAlloc with cudaHostAllocMapped: pinned host memory, read at host speed.', 'p. 624'),
      arg: c('other-ptr', 'The kernel must be given the value from cudaHostGetDevicePointer, and every access crosses the interconnect.', 'pp. 624–625'),
      chased: c('other-ptr', 'Only if the host wrote device-side values into the structure, not its own pointers.', 'pp. 624–625'),
    },
    registered: {
      arg: c('unknown', 'The book names only cudaHostAlloc; our sources do not say how registered memory behaved in 2009.', 'not in the book'),
      chased: c('unknown', 'The book names only cudaHostAlloc; our sources do not say how registered memory behaved in 2009.', 'not in the book'),
    },
  },
  uva: {
    malloc: {
      arg: c('no', 'One address space is not one access right: a kernel still cannot dereference malloc memory.', 'p. 625'),
      chased: c('no', 'A malloc pointer that the host stored is invalid on the GPU.', 'p. 625'),
    },
    pinned: {
      arg: c('remote', 'The same pointer now works directly; every access crosses the interconnect.', 'p. 625; A3'),
      chased: c('remote', 'The book\'s one exception: a pointer read from memory is safe if it points into cudaHostAlloc memory.', 'p. 625'),
    },
    registered: {
      arg: c('other-ptr', 'An exception to unified addressing: the device address comes from cudaHostGetDevicePointer, unless the device reports CanUseHostPointerForRegisteredMem.', 'A3, A5'),
      chased: c('other-ptr', 'A pointer the host stored holds the host address, which is not this memory\'s device address.', 'A3, A5'),
    },
    device: {
      host: c('no', 'Unified addresses, but host code still cannot dereference cudaMalloc memory.', 'p. 625'),
    },
    peer: {
      arg: c('no', 'The book dates direct peer access to Kepler; NVIDIA\'s account of UVA suggests earlier, and the first architecture could not be confirmed.', 'p. 626; A8'),
    },
  },
  kepler: {
    peer: {
      arg: c('after-enable', 'Pass the pointer and the kernel loads over PCIe or NVLink, once cudaDeviceEnablePeerAccess has granted access.', 'p. 626; A8'),
      chased: c('after-enable', 'The peer pointer has the same value on both GPUs, so a stored one can be followed once access is enabled.', 'A8'),
    },
  },
  'um-limited': {
    managed: {
      host: c('not-while-running', 'The CPU may touch it only while no kernel runs; touching it during one is a segmentation fault.', 'p. 627; A10'),
      arg: c('bulk', 'Every managed page goes to the GPU at launch, and the total cannot exceed GPU memory.', 'p. 627; A10'),
      chased: c(
        'bulk',
        'Our inference: one pointer value on both sides, and every managed page on the GPU from the launch, so a host-built list in managed memory should be followable. The book claims it only from Pascal on.',
        'ours, inferred from p. 626 (one virtual address for every variable); the book dates linked-structure traversal to Pascal, pp. 628–629',
      ),
    },
  },
  pascal: {
    managed: {
      host: c('fault', 'Allowed while kernels run; a page that lives on the GPU is serviced at longer latency.', 'p. 628; A10'),
      arg: c('fault', 'Nothing moves at launch: each first touch faults, and the page migrates or is mapped.', 'p. 628; A11'),
      chased: c('fault', 'Host-built linked structures in managed memory can be followed.', 'pp. 628–629; A12'),
    },
    malloc: {
      arg: c('no', 'Still no: Pascal\'s page faults cover managed allocations only; plain malloc memory needs HMM or ATS.', 'A12'),
      chased: c('no', 'A malloc pointer the host stored still cannot be followed; the book leaves this open, NVIDIA does not.', 'pp. 628–629; A12'),
    },
  },
  vmm: {
    vmm: {
      host: c('unknown', 'Our sources describe access grants to devices only.', 'A15'),
      arg: c('after-enable', 'Mapping alone is not enough: a kernel may touch the range only after cuMemSetAccess.', 'p. 629; A15'),
      chased: c('after-enable', 'One range and one pointer, even across devices, once access is granted.', 'p. 629; A15'),
    },
  },
  hmm: {
    malloc: {
      arg: c('fault', 'The host pointer works in kernels; with HMM a first touch faults the page over, with ATS access is coherent in hardware.', 'A12'),
      chased: c('fault', 'Host-built lists in plain malloc memory can be followed.', 'A12'),
    },
    registered: {
      arg: c('remote', 'All host memory is reachable through host pointers; pinned pages do not migrate.', 'A3; D1'),
      chased: c('remote', 'All host memory is reachable through host pointers; pinned pages do not migrate.', 'A3; D1'),
    },
    device: {
      host: c('no', 'Even with hardware coherence, host code cannot dereference cudaMalloc memory.', 'A12'),
    },
  },
};

function applyPatch(g: Grid, p: Patch): Grid {
  const out = {} as Grid;
  for (const r of ROWS) out[r] = { ...g[r], ...(p[r] ?? {}) };
  return out;
}

const CUDA: Record<EraId, Grid> = (() => {
  const all = {} as Record<EraId, Grid>;
  let g = COPIES;
  for (const id of ERA_ORDER) {
    g = applyPatch(g, PATCHES[id]);
    all[id] = g;
  }
  return all;
})();

// ---------------------------------------------------------------- HIP today (ROCm 10.2)
function hipGrid(xnack: boolean): Grid {
  const noX = 'Not GPU-visible without XNACK: it needs HSA_XNACK=1 and an HMM driver, on CDNA2 or later.';
  const withX = 'A first GPU access causes a recoverable page fault; CDNA2 and later only.';
  return {
    malloc: {
      host: c('local', 'Ordinary host memory.', 'B5'),
      arg: xnack ? c('fault', withX, 'B5') : c('no', noX, 'B5'),
      chased: xnack ? c('fault', withX, 'B5') : c('no', noX, 'B5'),
    },
    pinned: {
      host: c('local', 'Pinned host memory.', 'B2'),
      arg: c('remote', 'The same pointer works in kernels; traffic crosses the interconnect, much slower than on-device bandwidth.', 'B2'),
      chased: c('remote', 'AMD: the pointer can be used on both host and device.', 'B2'),
    },
    registered: {
      host: c('local', 'Your own malloc memory, page-locked where it is.', 'B5'),
      arg: c('unknown', 'Device-accessible after hipHostRegister; AMD does not say which pointer the kernel should use.', 'B5'),
      chased: c('unknown', 'Device-accessible after hipHostRegister; AMD does not say which pointer the kernel should use.', 'B5'),
    },
    device: {
      host: c('remote', 'Allowed, unlike CUDA: AMD documents host access to hipMalloc memory as zero copy over the interconnect.', 'B3; ch02-facts B7'),
      arg: c('local', 'Device memory: the fast path.', 'B3'),
      chased: c('local', 'A device pointer stored in device memory can be followed.', 'B3'),
    },
    peer: {
      host: c('remote', 'As for any hipMalloc memory: zero copy over the interconnect.', 'B3; ch02-facts B7'),
      arg: c('after-enable', 'Direct loads and stores once hipDeviceEnablePeerAccess has been called.', 'B4'),
      chased: c('after-enable', 'Direct loads and stores once hipDeviceEnablePeerAccess has been called.', 'B4'),
    },
    managed: xnack
      ? {
          host: c('unknown', 'AMD describes managed pages as migrating automatically; the CPU side is not spelled out.', 'B3, B5'),
          arg: c('fault', 'A GPU access to a page outside device memory faults, and the page migrates.', 'B5'),
          chased: c('fault', 'A GPU access to a page outside device memory faults, and the page migrates.', 'B5'),
        }
      : {
          host: c('local', 'Without XNACK it behaves as pinned host memory: the pages stay in host memory.', 'B5'),
          arg: c('remote', 'Without XNACK it behaves as pinned host memory, read over the interconnect.', 'B5'),
          chased: c('remote', 'Without XNACK it behaves as pinned host memory, read over the interconnect.', 'B5'),
        },
    shared: {
      host: c('no', 'Shared memory (LDS) exists only on the GPU.', 'B1'),
      arg: c('local', 'A generic "flat" pointer: the address falls in the LDS aperture.', 'B1'),
      chased: c('local', 'A generic "flat" pointer read back from memory works too, inside the owning block.', 'B1'),
    },
    vmm: {
      host: c('unknown', 'Our sources describe access grants to devices only.', 'B7'),
      arg: c('after-enable', 'Usable once hipMemMap and hipMemSetAccess have run.', 'B7'),
      chased: c('after-enable', 'Usable once hipMemMap and hipMemSetAccess have run.', 'B7'),
    },
  };
}

// ---------------------------------------------------------------- Metal today (Apple silicon)
const NOT_DOC = 'Kernels receive buffers by binding; our sources say nothing about following CPU-built pointers to them.';
const METAL: Grid = {
  malloc: {
    host: c('local', 'Ordinary memory.', 'C2'),
    arg: c('no', 'A kernel sees only buffers you bind: wrap page-aligned memory with newBufferWithBytesNoCopy first.', 'C2; ch02-facts C2'),
    chased: c('no', 'Not GPU-visible until it is wrapped in a buffer.', 'C2'),
  },
  pinned: {
    host: c('local', 'contents is the CPU\'s pointer to it: one memory, no copy.', 'ch05-facts C6'),
    arg: c('local', 'Bound by object and read in place: on Apple silicon there is no interconnect to cross.', 'C2'),
    chased: c('unknown', NOT_DOC, 'C3'),
  },
  registered: {
    host: c('local', 'Your own page-aligned memory, used in place.', 'ch02-facts C2'),
    arg: c('local', 'Bound like any buffer; llama.cpp wraps memory-mapped weights this way.', 'C2; D4'),
    chased: c('unknown', NOT_DOC, 'C3'),
  },
  device: {
    host: c('no', 'contents is NULL for a buffer with private storage.', 'ch05-facts C6'),
    arg: c('local', 'GPU-only storage, filled by a blit copy.', 'C2; ch02-facts C3'),
    chased: c('unknown', NOT_DOC, 'C3'),
  },
  peer: {
    host: c('no', 'No peer GPUs on Apple silicon; peer groups are deprecated there.', 'C4'),
    arg: c('no', 'No peer GPUs on Apple silicon; peer groups are deprecated there.', 'C4'),
    chased: c('no', 'No peer GPUs on Apple silicon; peer groups are deprecated there.', 'C4'),
  },
  managed: {
    host: c('local', 'Apple: managed storage has no effect on Apple silicon; use shared.', 'ch05-facts C6; C5'),
    arg: c('local', 'Nothing migrates: there is one memory.', 'C5'),
    chased: c('unknown', NOT_DOC, 'C3'),
  },
  shared: {
    host: c('no', 'Threadgroup memory exists only on the GPU, one copy per threadgroup.', 'C1'),
    arg: c('local', 'Through a helper whose parameter says threadgroup, called from a kernel: one overload per address space.', 'C1'),
    chased: c('unknown', 'Not documented in our sources.', 'C1'),
  },
  vmm: {
    host: c('unknown', 'Not documented in our sources.', 'C6'),
    arg: c('local', 'Once its pages are mapped from a placement heap with updateBufferMappings; there is no separate access grant.', 'C6'),
    chased: c('unknown', 'Not documented in our sources.', 'C6'),
  },
};

export interface Options {
  /** HIP only: HSA_XNACK=1 with an HMM driver on a GPU that supports it. */
  xnack?: boolean;
}

/** The whole grid for a platform; `eraId` matters for CUDA only (HIP and Metal are today). */
export function grid(platform: Platform, eraId: EraId, opts: Options = {}): Grid {
  if (platform === 'hip') return hipGrid(!!opts.xnack);
  if (platform === 'metal') return METAL;
  return CUDA[eraId];
}

export function cell(platform: Platform, eraId: EraId, row: RowId, site: SiteId, opts: Options = {}): Cell {
  return grid(platform, eraId, opts)[row][site];
}

/** The cells whose verdict differs from the era before (CUDA only). */
export function changed(eraId: EraId): { row: RowId; site: SiteId; from: Verdict; to: Verdict }[] {
  const i = ERA_ORDER.indexOf(eraId);
  if (i <= 0) return [];
  const before = CUDA[ERA_ORDER[i - 1]];
  const now = CUDA[eraId];
  const out: { row: RowId; site: SiteId; from: Verdict; to: Verdict }[] = [];
  for (const r of ROWS)
    for (const s of SITES)
      if (before[r][s].verdict !== now[r][s].verdict) out.push({ row: r, site: s, from: before[r][s].verdict, to: now[r][s].verdict });
  return out;
}

/** The first CUDA era in which a cell's verdict is one of `vs`, or null. */
export function firstEra(row: RowId, site: SiteId, vs: Verdict[]): EraId | null {
  return ERA_ORDER.find((id) => vs.includes(CUDA[id][row][site].verdict)) ?? null;
}

/** Every facts item a cell or an era cites, for the check that each one exists. */
export function citedItems(): string[] {
  const texts: string[] = [];
  for (const id of ERA_ORDER) for (const r of ROWS) for (const s of SITES) texts.push(CUDA[id][r][s].source);
  for (const x of [false, true]) {
    const g = hipGrid(x);
    for (const r of ROWS) for (const s of SITES) texts.push(g[r][s].source);
  }
  for (const r of ROWS) for (const s of SITES) texts.push(METAL[r][s].source);
  for (const e of ERAS) if (e.corrected) texts.push(e.corrected.item);
  return texts;
}
