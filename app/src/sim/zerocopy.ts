// Section C.2: what reading host memory in place (zero-copy, today "mapped
// memory") does to a memory-bound kernel, and when copying the data first pays.
//
// A bandwidth-only model. Time is bytes moved divided by bandwidth, with device
// bandwidth = 1. The link (PCIe, say) has bandwidth rho, a fraction of device
// bandwidth: the book's rule of thumb is "typically less than 10%" (p. 625).
//
//   mixed kernel, a fraction f of its bytes from mapped host memory:
//     no overlap       t = (1 - f) + f / rho
//     perfect overlap  t = max(1 - f, f / rho)
//   (relative to the same kernel with every byte in device memory)
//
//   a table of T bytes read k times, a fraction s of it per pass, in units of
//   T / device bandwidth:
//     read in place    k * s / rho
//     copy, then read  1 / rho + k * s
//   copying first wins once k > 1 / (s * (1 - rho)), and reading in place wins
//   exactly when the bytes read over the whole life, k * s tables, stay below
//   1 / (1 - rho) tables: a little more than one table's worth.
//
// The break-even rule is ours, derived from the book's rule that mapped memory
// suits data a kernel touches "occasionally and sparsely" (p. 625).
//
// What the model leaves out (the page says so too): latency; transactions (a
// link moves fixed-size pieces, so a sparse read costs more than its bytes);
// any caching of mapped data on the GPU; overlap of the copy with compute; and
// the "yours" ratio uses copy bandwidths as stand-ins for in-kernel reads.
//
// No imports, so browser checks can load this file directly.

/** Time of a memory-bound kernel relative to the all-device version. */
export function mixedTime(f: number, rho: number, overlap: 'none' | 'perfect'): number {
  const dev = 1 - f;
  const link = f / rho;
  return overlap === 'none' ? dev + link : Math.max(dev, link);
}

/** Read the table in place for k passes, s of it per pass (units: table bytes / device bandwidth). */
export const mappedCost = (k: number, s: number, rho: number) => (k * s) / rho;

/** Copy the whole table over the link once, then read from device memory. */
export const copyFirstCost = (k: number, s: number, rho: number) => 1 / rho + k * s;

/** The pass count above which copying first is cheaper. */
export const breakEven = (s: number, rho: number) => 1 / (s * (1 - rho));

/** Reading in place wins while the bytes read over the whole life stay below this many tables. */
export const lifetimeTables = (rho: number) => 1 / (1 - rho);

export function better(k: number, s: number, rho: number): 'mapped' | 'copy' | 'tie' {
  const m = mappedCost(k, s, rho);
  const c = copyFirstCost(k, s, rho);
  if (Math.abs(m - c) <= 1e-12 * Math.max(m, c)) return 'tie';
  return m < c ? 'mapped' : 'copy';
}

/** Bytes that cross the link when you copy the table, divided by those when you read the touched part in place. */
export const bytesRatio = (tableBytes: number, touchedBytes: number) => tableBytes / touchedBytes;

// ---------------------------------------------------------------- link ratios
/** The book's rule of thumb (p. 625): the link has under 10% of device bandwidth. */
export const BOOK_RULE = 0.1;
/**
 * The book's example device, H100 SXM: 3.35 TB/s of device memory bandwidth
 * (book ch. 5 p. 94) and PCIe Gen5 at 128 GB/s (NVIDIA's page, appC-facts A4).
 * The ratio is our arithmetic, about 3.8%.
 */
export const H100 = { deviceGBps: 3350, linkGBps: 128 };
export const H100_RATIO = H100.linkGBps / H100.deviceGBps;

/** The ratio from lab C.1's probe row on a CUDA or HIP backend, or null when it is missing or nonsense. */
export function ratioFromProbe(row: Record<string, unknown> | undefined): number | null {
  if (!row) return null;
  const l = row.link_gb_per_s;
  const d = row.device_gb_per_s;
  if (typeof l !== 'number' || typeof d !== 'number' || !(l > 0) || !(d > 0)) return null;
  return l / d;
}

// ---------------------------------------------------------------- curves for the plots
export function mixedCurve(rho: number, overlap: 'none' | 'perfect', n = 101): { f: number; t: number }[] {
  return Array.from({ length: n }, (_, i) => {
    const f = i / (n - 1);
    return { f, t: mixedTime(f, rho, overlap) };
  });
}

/** Pass counts from 1 to kMax, evenly spaced on a log scale, with both costs. */
export function passCurve(s: number, rho: number, kMax = 1000, n = 61): { k: number; mapped: number; copy: number }[] {
  return Array.from({ length: n }, (_, i) => {
    const k = Math.pow(kMax, i / (n - 1));
    return { k, mapped: mappedCost(k, s, rho), copy: copyFirstCost(k, s, rho) };
  });
}
