// Cutoff binning (book §21.5, pp. 523–527).
//
// What it models:
//   - Square bins of side b; a block's points are exactly one bin (the book's
//     simplification, p. 525).
//   - The per-block neighbourhood: the bins whose nearest point lies inside the "super
//     circle" centred on the block's bin, of radius cutoff + half the bin diagonal (the
//     text and Fig. 21.14) or cutoff + the full diagonal (Fig. 21.13's label); as relative
//     offsets and as absolute bin coordinates, with fully and partly covered counts. The
//     same counts in 3D for the book's worked case (p. 525).
//   - Binning atoms with a fixed capacity: padded slots (dummy atoms of charge 0) and an
//     overflow list summed afterwards by the host (p. 527).
//   - A block's walk over its list: per entry, the staged slots, each thread's accept or
//     reject for each staged atom, per-warp accept masks, and running tallies.
//   - Pairs considered against direct summation, and the far-field contribution the cutoff
//     drops at a point.
//
// What it leaves out, and what the app says wherever this model is used:
//   - The explorer is 2D: points and atoms in one plane, with the atoms a fixed height
//     above it only so that no distance is zero. The cutoff test uses the distance in the
//     plane. Atoms are uniformly random (seeded).
//   - The far field is simply dropped; the book hands it to an "implicit method" it does
//     not describe.
//   - Neighbour bins outside the box are skipped (a bound check the book does not show).
//   - No time and no memory traffic: tests and pairs are counted, not timed.

import { LAYER_HEIGHT, rng } from './dcs';

// ---------------------------------------------------------------- neighbourhoods

export type RadiusMode = 'half' | 'full';

/** Super-circle radius: cutoff plus half the bin diagonal (text, Fig. 21.14) or the full diagonal (Fig. 21.13's label). */
export function superRadius(cutoff: number, b: number, dims: 2 | 3, mode: RadiusMode): number {
  const diag = b * Math.sqrt(dims);
  return cutoff + (mode === 'half' ? diag / 2 : diag);
}

export interface Offset {
  dx: number;
  dy: number;
  dz: number;
  /** Fully inside the super circle (sphere), or only partly. */
  cover: 'full' | 'partial';
  /** Distances, in bin widths, from the centre bin's centre to the bin's nearest and farthest points. */
  near: number;
  far: number;
}

const EPS = 1e-12;

/**
 * The offsets of the bins in a block's neighbourhood. R is in bin widths. A bin is in the
 * list when its nearest point is strictly inside the circle (a bin that only touches it
 * contributes nothing); it is fully covered when its farthest point is inside or on it.
 * Order: by dz, then dy, then dx, so the 2D list reads row by row like Fig. 21.14.
 */
export function neighbourhood(R: number, dims: 2 | 3 = 2): Offset[] {
  const out: Offset[] = [];
  const reach = Math.ceil(R + 0.5);
  const zs = dims === 3 ? reach : 0;
  for (let dz = -zs; dz <= zs; dz++)
    for (let dy = -reach; dy <= reach; dy++)
      for (let dx = -reach; dx <= reach; dx++) {
        const comps = dims === 3 ? [dx, dy, dz] : [dx, dy];
        const near = Math.hypot(...comps.map((d) => Math.max(0, Math.abs(d) - 0.5)));
        const far = Math.hypot(...comps.map((d) => Math.abs(d) + 0.5));
        if (near < R - EPS) out.push({ dx, dy, dz, cover: far <= R + EPS ? 'full' : 'partial', near, far });
      }
  return out;
}

export function counts(R: number, dims: 2 | 3 = 2): { total: number; full: number; partial: number } {
  const n = neighbourhood(R, dims);
  const full = n.filter((o) => o.cover === 'full').length;
  return { total: n.length, full, partial: n.length - full };
}

/**
 * The share of the atoms a thread tests that pass its cutoff test, at uniform density:
 * the cutoff disc (ball) over the area (volume) of the bins it scans.
 */
export function acceptedShare(cutoff: number, b: number, dims: 2 | 3, mode: RadiusMode = 'half'): number {
  const R = superRadius(cutoff, b, dims, mode) / b;
  const bins = counts(R, dims).total;
  const inside = dims === 2 ? Math.PI * cutoff ** 2 : (4 / 3) * Math.PI * cutoff ** 3;
  return inside / (bins * b ** dims);
}

/** The radii (in bin widths) at which the 2D list changes size: every distinct "nearest point" distance up to rMax. */
export function criticalRadii(rMax: number): number[] {
  const s = new Set<number>();
  for (const o of neighbourhood(rMax + 1, 2)) if (o.near > 0 && o.near <= rMax) s.add(Math.round(o.near * 1e9) / 1e9);
  return [...s].sort((a, b) => a - b);
}

// ---------------------------------------------------------------- the 2D world of the explorer

export interface Atom2 {
  /** Position in grid units; grid point (i, j) sits at (i, j). */
  x: number;
  y: number;
  q: number;
}

export interface World {
  binsX: number;
  binsY: number;
  /** Points per bin side: a bin (and a block) is side × side points. */
  side: number;
  atoms: Atom2[];
}

export const worldW = (w: World) => w.binsX * w.side;
export const worldH = (w: World) => w.binsY * w.side;

/** Uniform random atoms over the box (seeded), charges of either sign with magnitude 0.25 to 1. */
export function makeWorld(binsX: number, binsY: number, side: number, nAtoms: number, seed: number): World {
  const r = rng(seed * 104729 + 3);
  const W = binsX * side;
  const H = binsY * side;
  const atoms: Atom2[] = [];
  for (let k = 0; k < nAtoms; k++) {
    const x = -0.5 + r() * W;
    const y = -0.5 + r() * H;
    const mag = 0.25 + r() * 0.75;
    atoms.push({ x, y, q: r() < 0.5 ? -mag : mag });
  }
  return { binsX, binsY, side, atoms };
}

/** The bin an atom (or any position) falls in. Bin (bx, by) holds points bx·side .. bx·side + side − 1. */
export function binOf(w: World, x: number, y: number): [number, number] {
  const bx = Math.min(w.binsX - 1, Math.max(0, Math.floor((x + 0.5) / w.side)));
  const by = Math.min(w.binsY - 1, Math.max(0, Math.floor((y + 0.5) / w.side)));
  return [bx, by];
}

/** Centre of a bin, in grid units. */
export const binCentre = (w: World, bx: number, by: number): [number, number] => [bx * w.side + (w.side - 1) / 2, by * w.side + (w.side - 1) / 2];

export interface Binning {
  capacity: number;
  /** Per bin (row-major by, bx): `capacity` slots, each an atom index or -1 for a dummy atom of charge 0. */
  slots: number[][];
  /** Atoms whose home bin was full, in the order they were met. */
  overflow: number[];
  /** Atoms per bin before the cap. */
  occupancy: number[];
  real: number;
  dummy: number;
}

export const binIndex = (w: World, bx: number, by: number) => by * w.binsX + bx;

export function maxOccupancy(w: World): number {
  const occ = new Array<number>(w.binsX * w.binsY).fill(0);
  for (const a of w.atoms) occ[binIndex(w, ...binOf(w, a.x, a.y))]++;
  return Math.max(0, ...occ);
}

/** The smallest capacity that leaves at most `share` of the atoms on the overflow list (the book's "under 3 %", p. 527). */
export function capacityFor(w: World, share = 0.03): number {
  const max = maxOccupancy(w);
  for (let c = 1; c <= max; c++) if (binAtoms(w, c).overflow.length <= share * w.atoms.length) return c;
  return Math.max(1, max);
}

/** Sort atoms into bins of fixed capacity: the first `capacity` atoms of a bin take its slots, the rest overflow. */
export function binAtoms(w: World, capacity: number): Binning {
  const n = w.binsX * w.binsY;
  const slots: number[][] = Array.from({ length: n }, () => []);
  const occupancy = new Array<number>(n).fill(0);
  const overflow: number[] = [];
  w.atoms.forEach((a, k) => {
    const b = binIndex(w, ...binOf(w, a.x, a.y));
    occupancy[b]++;
    if (slots[b].length < capacity) slots[b].push(k);
    else overflow.push(k);
  });
  let real = 0;
  for (const s of slots) {
    real += s.length;
    while (s.length < capacity) s.push(-1);
  }
  return { capacity, slots, overflow, occupancy, real, dummy: n * capacity - real };
}

// ---------------------------------------------------------------- one block's walk

export interface ListEntry {
  /** Position in the list. */
  k: number;
  rel: Offset;
  /** The block's bin plus the offset. */
  bx: number;
  by: number;
  inBox: boolean;
}

/** A block's list: relative offsets turned into absolute bin coordinates (Fig. 21.14, bottom). */
export function blockList(w: World, R: number, bx: number, by: number): ListEntry[] {
  return neighbourhood(R, 2).map((rel, k) => {
    const x = bx + rel.dx;
    const y = by + rel.dy;
    return { k, rel, bx: x, by: y, inBox: x >= 0 && y >= 0 && x < w.binsX && y < w.binsY };
  });
}

/** The points of a block (its bin), as (i, j), in thread order: thread t has tx = t mod side, ty = t div side. */
export function blockPoints(w: World, bx: number, by: number): [number, number][] {
  const out: [number, number][] = [];
  for (let ty = 0; ty < w.side; ty++) for (let tx = 0; tx < w.side; tx++) out.push([bx * w.side + tx, by * w.side + ty]);
  return out;
}

export type Outcome = 'accept' | 'reject' | 'dummy';

export interface WalkEntry extends ListEntry {
  /** The staged slots (atom index or -1), empty for a bin outside the box. */
  slots: number[];
  /** outcome[slot][thread]. */
  outcome: Outcome[][];
  tested: number;
  accepted: number;
  dummyTests: number;
  /** Running totals up to and including this entry. */
  cumTested: number;
  cumAccepted: number;
}

export interface Walk {
  list: WalkEntry[];
  points: [number, number][];
  tested: number;
  accepted: number;
  dummyTests: number;
  /** Per thread: accepted atoms and their summed contribution (the kernel's result before the host pass). */
  perThread: { accepted: number; sum: number }[];
  /** What direct summation would evaluate for the same points: points × all atoms. */
  directPairs: number;
}

/** Planar distance between a grid point and an atom. */
export const planar = (i: number, j: number, a: Atom2) => Math.hypot(i - a.x, j - a.y);
/** An atom's contribution at a point: q / r, with the atoms LAYER_HEIGHT above the plane. */
export const contribution = (i: number, j: number, a: Atom2) => a.q / Math.sqrt((i - a.x) ** 2 + (j - a.y) ** 2 + LAYER_HEIGHT ** 2);

/**
 * All threads of block (bx, by) walk the list: per entry they stage the bin's slots, and
 * each thread tests every staged atom against the cutoff around its own point. cutoff is
 * in grid units, R in bin widths.
 */
export function blockWalk(w: World, binning: Binning, cutoff: number, R: number, bx: number, by: number): Walk {
  const points = blockPoints(w, bx, by);
  const perThread = points.map(() => ({ accepted: 0, sum: 0 }));
  let cumTested = 0;
  let cumAccepted = 0;
  let dummyTests = 0;
  const list = blockList(w, R, bx, by).map((e): WalkEntry => {
    const slots = e.inBox ? binning.slots[binIndex(w, e.bx, e.by)] : [];
    let accepted = 0;
    let dummies = 0;
    const outcome = slots.map((s) =>
      points.map(([i, j], t): Outcome => {
        if (s < 0) {
          dummies++;
          return 'dummy';
        }
        const a = w.atoms[s];
        if (planar(i, j, a) < cutoff) {
          accepted++;
          perThread[t].accepted++;
          perThread[t].sum += contribution(i, j, a);
          return 'accept';
        }
        return 'reject';
      }),
    );
    const tested = slots.length * points.length;
    cumTested += tested;
    cumAccepted += accepted;
    dummyTests += dummies;
    return { ...e, slots, outcome, tested, accepted, dummyTests: dummies, cumTested, cumAccepted };
  });
  return { list, points, tested: cumTested, accepted: cumAccepted, dummyTests, perThread, directPairs: points.length * w.atoms.length };
}

/** For one staged slot of one entry: per warp, which lanes accept. Lanes past the block are left out. */
export function acceptMasks(entry: WalkEntry, slot: number, warpWidth: number): boolean[][] {
  const row = entry.outcome[slot] ?? [];
  const out: boolean[][] = [];
  for (let w0 = 0; w0 < row.length; w0 += warpWidth) out.push(row.slice(w0, w0 + warpWidth).map((o) => o === 'accept'));
  return out;
}

/** A warp diverges on a staged atom when some of its lanes accept and some do not. */
export const diverges = (mask: boolean[]) => mask.some((x) => x) && mask.some((x) => !x);

// ---------------------------------------------------------------- the whole grid

export interface GridResult {
  /** Device result per point (row-major): accepted contributions of binned atoms only. */
  device: number[];
  /** After the host adds the overflow atoms' contributions within the cutoff. */
  withHost: number[];
  /** The exact cutoff sum (every atom within the cutoff), and the direct sum (every atom). */
  cutoffSum: number[];
  direct: number[];
  /** Tests made by all blocks (dummy slots included), tests of real atoms, accepted pairs. */
  tests: number;
  realTests: number;
  accepted: number;
  /** Points × atoms. */
  directPairs: number;
  /** Atom-point pairs the host's overflow pass evaluates (overflow atoms × points within the cutoff). */
  hostPairs: number;
}

/**
 * Every block of the grid walks its list. With sums = false only the counts are made
 * (tests, real tests, accepted pairs), which is what the explorer's pair counts need.
 */
export function runGrid(w: World, binning: Binning, cutoff: number, R: number, sums = true): GridResult {
  const W = worldW(w);
  const H = worldH(w);
  const device = new Array<number>(W * H).fill(0);
  let tests = 0;
  let realTests = 0;
  let accepted = 0;
  for (let by = 0; by < w.binsY; by++)
    for (let bx = 0; bx < w.binsX; bx++) {
      const walk = blockWalk(w, binning, cutoff, R, bx, by);
      tests += walk.tested;
      realTests += walk.tested - walk.dummyTests;
      accepted += walk.accepted;
      walk.points.forEach(([i, j], t) => (device[j * W + i] = walk.perThread[t].sum));
    }
  const withHost = [...device];
  let hostPairs = 0;
  const directPairs = W * H * w.atoms.length;
  if (!sums) return { device, withHost, cutoffSum: [], direct: [], tests, realTests, accepted, directPairs, hostPairs };
  for (const k of binning.overflow) {
    const a = w.atoms[k];
    for (let j = 0; j < H; j++)
      for (let i = 0; i < W; i++)
        if (planar(i, j, a) < cutoff) {
          withHost[j * W + i] += contribution(i, j, a);
          hostPairs++;
        }
  }
  const cutoffSum = new Array<number>(W * H).fill(0);
  const direct = new Array<number>(W * H).fill(0);
  for (let j = 0; j < H; j++)
    for (let i = 0; i < W; i++)
      for (const a of w.atoms) {
        const c = contribution(i, j, a);
        direct[j * W + i] += c;
        if (planar(i, j, a) < cutoff) cutoffSum[j * W + i] += c;
      }
  return { device, withHost, cutoffSum, direct, tests, realTests, accepted, directPairs, hostPairs };
}

/** At one point: the part the cutoff keeps, the far field it drops, and the direct sum. */
export function nearFar(w: World, i: number, j: number, cutoff: number): { near: number; far: number; direct: number; nearAtoms: number } {
  let near = 0;
  let far = 0;
  let nearAtoms = 0;
  for (const a of w.atoms) {
    const c = contribution(i, j, a);
    if (planar(i, j, a) < cutoff) {
      near += c;
      nearAtoms++;
    } else far += c;
  }
  return { near, far, direct: near + far, nearAtoms };
}

// ---------------------------------------------------------------- the binned-block scene

/**
 * The scene's case: a 9 × 9-bin box of 4 × 4-point bins, 160 atoms (about 2 per bin), the
 * block in the middle, a cutoff that makes the super-circle radius exactly 2 bin widths
 * (21 offsets), and a capacity of 4 atoms per bin.
 */
export const SCENE = {
  binsX: 9,
  binsY: 9,
  side: 4,
  atoms: 160,
  seed: 1,
  capacity: 4,
  block: [4, 4] as [number, number],
  /** Super-circle radius in bin widths. */
  R: 2,
};

export function sceneCase() {
  const w = makeWorld(SCENE.binsX, SCENE.binsY, SCENE.side, SCENE.atoms, SCENE.seed);
  const b = w.side;
  // R = cutoff + half the diagonal  =>  cutoff = R·b − b·√2/2 (grid units)
  const cutoff = SCENE.R * b - (b * Math.SQRT2) / 2;
  const binning = binAtoms(w, SCENE.capacity);
  const walk = blockWalk(w, binning, cutoff, SCENE.R, ...SCENE.block);
  return { world: w, cutoff, binning, walk };
}
