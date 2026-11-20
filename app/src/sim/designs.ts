// One power budget spent two ways: a model of the book's cost argument for
// Fig. 1.1 (chapter 1, p. 5). It runs the "Spend a power budget" explorer, the
// p-crowd prediction and every beat of the "One budget, two designs" animation.
//
// The rule, as the book states it: doubling throughput by doubling the number of
// arithmetic units costs twice the power; halving the latency of a unit may cost
// four times the power. Here a design takes a fixed budget and halves the latency
// of its units some number of times. Each halving makes a unit twice as fast and
// four times as expensive, so the budget buys a quarter as many of them.
//
// The work is a number of chains. A chain is a sequence of operations in which
// each one needs the result of the one before, so one chain can only ever keep
// one unit busy. Different chains do not depend on each other.
//
// What this leaves out:
//  - The single step (half the latency for four times the power) is the book's.
//    Applying it again and again is our extrapolation. Real circuits do not follow
//    one exponent.
//  - Area, which the book says grows by "more than" two per halving, is not
//    modelled. Neither are caches, control logic, memory channels or pipelining.
//  - A unit runs one chain at a time, start to finish. There is no time-slicing,
//    and no stock of waiting threads as on a real GPU (chapter 4, section 4.6).
//  - The budget is an abstract number, not a device.
//
// This is not the warp-scheduling model of chapter 4; that chapter builds its own.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.

export interface Design {
  budget: number;
  /** How many times the latency of a unit has been halved. */
  halvings: number;
  /** Units the budget buys: floor(budget / unitPower). */
  units: number;
  /** Operations one unit finishes per time unit: 2^halvings. */
  speed: number;
  /** Power one unit draws: 4^halvings. */
  unitPower: number;
  /** Operations per time unit with every unit busy: units * speed. */
  throughput: number;
}

export function design(budget: number, halvings: number): Design {
  const speed = 2 ** halvings;
  const unitPower = 4 ** halvings;
  const units = Math.floor(budget / unitPower);
  return { budget, halvings, units, speed, unitPower, throughput: units * speed };
}

export interface Finish {
  /** When the first chain finishes: length / speed. */
  first: number;
  /** When the last chain finishes: waves * length / speed. */
  last: number;
  /** Chains start in waves of `units`; this is how many waves the work takes. */
  waves: number;
  /** Units that never get a chain. */
  idle: number;
  /** Operations per time unit actually delivered on this work: chains * length / last. */
  delivered: number;
  /** Share of the available unit-time spent working, 0 to 1. */
  busy: number;
}

export function finish(d: Design, chains: number, length: number): Finish {
  if (d.units < 1 || chains < 1)
    return { first: Infinity, last: Infinity, waves: Infinity, idle: d.units, delivered: 0, busy: 0 };
  const waves = Math.ceil(chains / d.units);
  const first = length / d.speed;
  const last = (waves * length) / d.speed;
  return {
    first,
    last,
    waves,
    idle: Math.max(0, d.units - chains),
    delivered: (chains * length) / last,
    busy: chains / (d.units * waves),
  };
}

export interface Segment {
  unit: number;
  chain: number;
  wave: number;
  start: number;
  end: number;
}

/** Which unit runs which chain, and when. Chain c goes to unit c mod units in wave floor(c / units). */
export function timeline(d: Design, chains: number, length: number): Segment[] {
  const out: Segment[] = [];
  if (d.units < 1) return out;
  const step = length / d.speed;
  for (let chain = 0; chain < chains; chain++) {
    const wave = Math.floor(chain / d.units);
    out.push({ unit: chain % d.units, chain, wave, start: wave * step, end: (wave + 1) * step });
  }
  return out;
}

export interface Wave {
  wave: number;
  start: number;
  end: number;
  /** Units working in this wave (the last wave may be partly filled). */
  units: number;
  firstChain: number;
}

/** The same schedule as timeline(), one entry per wave. For drawing when there are too many segments. */
export function waves(d: Design, chains: number, length: number): Wave[] {
  const out: Wave[] = [];
  if (d.units < 1) return out;
  const step = length / d.speed;
  for (let w = 0; w * d.units < chains; w++)
    out.push({ wave: w, start: w * step, end: (w + 1) * step, units: Math.min(d.units, chains - w * d.units), firstChain: w * d.units });
  return out;
}

export interface Snapshot {
  /** Operations finished by time t, per chain. */
  ops: number[];
  /** Chains finished by time t. */
  done: number;
  /** Chains being worked on at time t. */
  running: number;
}

/** Where the work stands at time t. */
export function snapshot(d: Design, chains: number, length: number, t: number): Snapshot {
  const ops = timeline(d, chains, length).map((s) =>
    Math.max(0, Math.min(length, Math.floor((t - s.start) * d.speed + 1e-9))),
  );
  const done = ops.filter((o) => o >= length).length;
  const running = timeline(d, chains, length).filter((s) => t >= s.start && t < s.end).length;
  return { ops, done, running };
}

export interface Race {
  winner: 'a' | 'b' | 'tie';
  /** How many times sooner the winner's last chain finishes; 1 for a tie. */
  factor: number;
  a: Finish;
  b: Finish;
}

/**
 * Which design finishes all the work first. Both finishing times are
 * waves * length / speed, so the length cancels: the comparison is done on
 * waves_a * speed_b against waves_b * speed_a, which are whole numbers.
 */
export function race(a: Design, b: Design, chains: number, length: number): Race {
  const fa = finish(a, chains, length);
  const fb = finish(b, chains, length);
  const ta = fa.waves * b.speed; // proportional to a's finishing time
  const tb = fb.waves * a.speed; // proportional to b's finishing time
  if (ta === tb) return { winner: 'tie', factor: 1, a: fa, b: fb };
  return ta < tb ? { winner: 'a', factor: tb / ta, a: fa, b: fb } : { winner: 'b', factor: ta / tb, a: fa, b: fb };
}

export interface SweepPoint {
  chains: number;
  /** Finishing time of each design for this many chains. */
  a: number;
  b: number;
  winner: 'a' | 'b' | 'tie';
  factor: number;
}

/** Finishing time of both designs for every number of chains from 1 to maxChains. */
export function sweep(a: Design, b: Design, maxChains: number, length: number): SweepPoint[] {
  const out: SweepPoint[] = [];
  for (let chains = 1; chains <= maxChains; chains++) {
    const r = race(a, b, chains, length);
    out.push({ chains, a: r.a.last, b: r.b.last, winner: r.winner, factor: r.factor });
  }
  return out;
}

export interface Region {
  winner: 'a' | 'b' | 'tie';
  from: number;
  to: number;
}

/** Runs of consecutive chain counts with the same winner, in order. */
export function regions(points: SweepPoint[]): Region[] {
  const out: Region[] = [];
  for (const p of points) {
    const last = out[out.length - 1];
    if (last && last.winner === p.winner) last.to = p.chains;
    else out.push({ winner: p.winner, from: p.chains, to: p.chains });
  }
  return out;
}

export interface Plot {
  /** Side of the square the budget is drawn as, in budget cells. */
  side: number;
  /** One square per unit: its side equals its speed, so its area is its power. */
  cells: { unit: number; x: number; y: number; side: number }[];
}

/** The design drawn as a floor plan: `budget` equal cells, grouped into units of unitPower cells. */
export function floorplan(d: Design): Plot {
  const side = Math.ceil(Math.sqrt(d.budget));
  const s = d.speed; // sqrt(unitPower)
  const perRow = Math.floor(side / s);
  const cells = [];
  for (let unit = 0; unit < d.units && perRow > 0; unit++)
    cells.push({ unit, x: (unit % perRow) * s, y: Math.floor(unit / perRow) * s, side: s });
  return { side, cells };
}
