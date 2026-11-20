import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  EPS_EXPONENTS,
  bookPanels,
  busyFraction,
  captionText,
  diffPanels,
  elimBusyFraction,
  equation,
  printedA11,
  system,
  trace,
  type Panel,
  type SystemDef,
} from './gauss';

const R = (s: string) => s.split(' ');

// Fig. A.10 as printed (p. 596), transcribed from notes/appA.md (zeros written as 0).
const printedA10: Pick<Panel, 'label' | 'rows'>[] = [
  { label: 'Original', rows: [R('3 5 2 19'), R('2 3 1 11'), R('1 2 2 11')] },
  { label: 'Step 1', rows: [R('1 5/3 2/3 19/3'), R('1 3/2 1/2 11/2'), R('1 2 2 11')] },
  { label: 'Step 2', rows: [R('1 5/3 2/3 19/3'), R('0 -1/6 -1/6 -5/6'), R('0 1/3 4/3 14/3')] },
  { label: 'Step 3', rows: [R('1 5/3 2/3 19/3'), R('0 1 1 5'), R('0 1 4 14')] },
  { label: 'Step 4', rows: [R('1 5/3 2/3 19/3'), R('0 1 1 5'), R('0 0 3 9')] },
  { label: 'Step 5', rows: [R('1 5/3 2/3 19/3'), R('0 1 1 5'), R('0 0 1 3')] },
  { label: 'Step 6', rows: [R('1 5/3 2/3 19/3'), R('0 1 0 2'), R('0 0 1 3')] },
  { label: 'Step 7', rows: [R('1 0 0 1'), R('0 1 0 2'), R('0 0 1 3')] },
];

describe('Figs. A.9 and A.10: the book\'s first system (pp. 594-596)', () => {
  const t = trace(system('book1'), { pivot: false, arith: 'exact' });
  it('solves to X = 1, Y = 2, Z = 3', () => {
    expect(t.solution.map((c) => c.text)).toEqual(['1', '2', '3']);
    expect(t.finite).toBe(true);
  });
  it('every panel of Fig. A.10, as printed', () => {
    const panels = bookPanels('A.10');
    expect(panels.map((p) => p.label)).toEqual(printedA10.map((p) => p.label));
    panels.forEach((p, i) => expect(p.rows, p.label).toEqual(printedA10[i].rows));
  });
  it('the captions: divide by 3 and 2, by -1/6 and 1/3, by 3; subtract; substitute Z, then Y and Z', () => {
    const c = bookPanels('A.10').map((p) => captionText(p.caption));
    expect(c).toEqual([
      '',
      'divide row 1 by 3 and row 2 by 2',
      'subtract row 1 from row 2 and row 3',
      'divide row 2 by −1/6 and row 3 by 1/3',
      'subtract row 2 from row 3',
      'divide row 3 by 3',
      'substitute Z into row 2',
      'substitute Y and Z into row 1',
    ]);
    expect(bookPanels('A.10').map((p) => p.solved ?? '')).toEqual(['', '', '', '', '', 'Z', 'Y', 'X']);
  });
  it('Fig. A.9 is the same steps as equations', () => {
    const p = bookPanels('A.10');
    expect(p[0].rows.map((r) => equation(r, ['X', 'Y', 'Z']))).toEqual(['3X + 5Y + 2Z = 19', '2X + 3Y + Z = 11', 'X + 2Y + 2Z = 11']);
    expect(p[2].rows.slice(1).map((r) => equation(r, ['X', 'Y', 'Z']))).toEqual(['−1/6Y − 1/6Z = −5/6', '1/3Y + 4/3Z = 14/3']);
    expect(equation(p[4].rows[2], ['X', 'Y', 'Z'])).toBe('3Z = 9');
  });
  it('barriers follow every divide and every subtract; one more thread is idle after each subtract', () => {
    const kinds = t.phases.map((p) => p.kind);
    kinds.forEach((k, i) => {
      if (k === 'divide' || k === 'subtract') expect(kinds[i + 1]).toBe('barrier');
    });
    const subs = t.phases.filter((p) => p.kind === 'subtract');
    expect(subs.map((p) => p.threads.filter((s) => s === 'idle').length)).toEqual([1, 2]);
    const divs = t.phases.filter((p) => p.kind === 'divide');
    expect(divs.map((p) => p.threads.filter((s) => s === 'working').length)).toEqual([3, 2, 1]);
  });
  it('back substitution runs upward, one row per step', () => {
    const bs = t.phases.filter((p) => p.kind === 'backsub');
    expect(bs.map((p) => p.col)).toEqual([1, 0]);
    expect(bs.map((p) => p.threads)).toEqual([
      ['idle', 'working', 'finished'],
      ['working', 'finished', 'finished'],
    ]);
  });
});

describe('the second system (p. 597): 5Y + 2Z = 16 replaces equation 1', () => {
  it('without pivoting, step 1 divides by zero: infinity, and 0 / 0 is NaN', () => {
    const t = trace(system('book2'), { pivot: false, arith: 'exact' });
    const d1 = t.phases.find((p) => p.kind === 'divide')!;
    expect(d1.divisors![0]!.text).toBe('0');
    expect(d1.m[0].map((c) => c.text)).toEqual(['NaN', 'inf', 'inf', 'inf']);
    expect(d1.note).toContain('0 / 0 gives NaN');
  });
  it('at the next subtraction NaN spreads into every row; the answer is lost', () => {
    const t = trace(system('book2'), { pivot: false, arith: 'exact' });
    const s1 = t.phases.find((p) => p.kind === 'subtract')!;
    expect(s1.m.every((r) => Number.isNaN(r[0].v))).toBe(true);
    expect(t.finite).toBe(false);
    expect(t.solution.every((c) => Number.isNaN(c.v))).toBe(true);
  });
  it('with pivoting it solves to (1, 2, 3): row 2 wins the max reduction and swaps to the top', () => {
    const t = trace(system('book2'), { pivot: true, arith: 'exact' });
    expect(t.solution.map((c) => c.text)).toEqual(['1', '2', '3']);
    const pv = t.phases.find((p) => p.kind === 'pivot')!;
    expect(pv.pivot).toMatchObject({ rows: [0, 1, 2], winner: 1, levels: [[0, 1, 2], [1, 2], [1]] });
    expect(pv.pivot!.abs.map((c) => c.text)).toEqual(['0', '2', '1']);
    expect(t.phases.filter((p) => p.kind === 'swap').map((p) => p.swap)).toEqual([[0, 1]]);
  });
  it('Fig. A.11, step 1: row 2 already lacks X, so its thread skips the division and the subtraction (p. 598)', () => {
    const t = trace(system('book2'), { pivot: true, arith: 'exact' });
    const [d1] = t.phases.filter((p) => p.kind === 'divide');
    const [s1] = t.phases.filter((p) => p.kind === 'subtract');
    expect(d1.threads).toEqual(['working', 'skipping', 'working']);
    expect(s1.threads).toEqual(['idle', 'skipping', 'working']);
  });
});

describe('Fig. A.11 as printed against Fig. A.11 computed (the erratum hunt)', () => {
  it('they differ exactly at step 1\'s caption and at row 1 of steps 5 and 6', () => {
    const d = diffPanels(printedA11, bookPanels('A.11'));
    expect(d.filter((x) => x.what === 'caption')).toEqual([
      { panel: 'Step 1', what: 'caption', printed: 'divide row 1 by 3', computed: 'divide row 1 by 2' },
    ]);
    const cells = d.filter((x) => x.what === 'cell');
    expect(cells.map((x) => `${x.panel} r${x.row} c${x.col}`)).toEqual([
      'Step 5 r0 c1', 'Step 5 r0 c2', 'Step 5 r0 c3', 'Step 6 r0 c1', 'Step 6 r0 c2', 'Step 6 r0 c3',
    ]);
    expect(cells.map((x) => x.printed)).toEqual(['5/3', '2/3', '19/3', '5/3', '2/3', '19/3']);
    expect(cells.map((x) => x.computed)).toEqual(['3/2', '1/2', '11/2', '3/2', '1/2', '11/2']);
  });
  it('the printed row 1 of steps 5 and 6 is row 1 of Fig. A.10 (the first system\'s)', () => {
    expect(printedA11[6].rows[0]).toEqual(bookPanels('A.10')[5].rows[0]);
  });
  it('q-a11: row 1 is [2 3 1 11] before step 1 and [1 3/2 1/2 11/2] after, a division by 2', () => {
    const p = bookPanels('A.11');
    expect(p[1].rows[0]).toEqual(R('2 3 1 11'));
    expect(p[2].rows[0]).toEqual(R('1 3/2 1/2 11/2'));
    expect(p[2].caption).toEqual({ op: 'divide', by: [[1, '2']] });
  });
});

describe('a tiny pivot: [[ε, 1], [1, 1]] x = [1, 2] (ours, p. 597\'s subtler case)', () => {
  it('q-tiny: FP32, ε = 1e-8: x1 = 0 without pivoting, 1 with it', () => {
    expect(trace(system('tiny', 8), { pivot: false, arith: 'fp32' }).solution[0].v).toBe(0);
    expect(trace(system('tiny', 8), { pivot: true, arith: 'fp32' }).solution[0].v).toBe(1);
  });
  it('exact fractions get x1 = 10^8 / (10^8 - 1) either way', () => {
    for (const pivot of [false, true]) {
      const t = trace(system('tiny', 8), { pivot, arith: 'exact' });
      expect(t.solution[0].text).toBe('≈1.00000001');
      expect(t.solution[0].v).toBeCloseTo(1e8 / (1e8 - 1), 12);
    }
  });
  it('FP16: ε = 1e-4 loses x1 without pivoting and keeps it with; ε = 1e-8 rounds to 0, so without pivoting it is 0 / 0', () => {
    expect(trace(system('tiny', 4), { pivot: false, arith: 'fp16' }).solution[0].v).toBe(0);
    expect(trace(system('tiny', 4), { pivot: true, arith: 'fp16' }).solution[0].v).toBe(1);
    expect(trace(system('tiny', 8), { pivot: false, arith: 'fp16' }).solution[0].v).toBeNaN();
    expect(trace(system('tiny', 8), { pivot: true, arith: 'fp16' }).solution[0].v).toBe(1);
  });
  it('FP16, ε = 1e-5 to 1e-7, even with pivoting: the book\'s design divides the ε row by ε, and 1/ε overflows FP16 (largest 65,504)', () => {
    for (const k of [5, 6, 7]) {
      const t = trace(system('tiny', k), { pivot: true, arith: 'fp16' });
      const d1 = t.phases.find((p) => p.kind === 'divide')!;
      expect(d1.m[1][1].v, `k=${k}`).toBe(Infinity);
      expect(t.solution[0].v, `k=${k}`).toBeNaN();
      expect(1 / 10 ** -k).toBeGreaterThan(65504);
    }
    expect(trace(system('tiny', 4), { pivot: true, arith: 'fp16' }).finite).toBe(true);
  });
  it('with pivoting every ε gives x1 within 1e-6 of the truth in FP32', () => {
    for (const k of EPS_EXPONENTS) {
      const t = trace(system('tiny', k), { pivot: true, arith: 'fp32' });
      expect(Math.abs(t.solution[0].v - 1 / (1 - 10 ** -k)), `k=${k}`).toBeLessThan(1e-6);
    }
  });
  it('without pivoting the FP32 error in x1 grows as ε shrinks', () => {
    const err = (k: number) => Math.abs(trace(system('tiny', k), { pivot: false, arith: 'fp32' }).solution[0].v - 1 / (1 - 10 ** -k));
    expect(err(2)).toBeLessThan(1e-4);
    expect(err(6)).toBeGreaterThan(err(2));
    expect(err(8)).toBeCloseTo(1, 6);
  });
});

describe('thread utilisation (p. 597): one more thread idle after each subtraction', () => {
  const dense = (n: number): SystemDef => ({
    id: 'book1',
    label: 'dense',
    vars: Array.from({ length: n }, (_, i) => `x${i}`),
    rows: Array.from({ length: n }, (_, i) =>
      Array.from({ length: n + 1 }, (_, j) => [BigInt(((i * 7 + j * 13) % 17) + 1 + (i === j ? 40 : 0)), 1n] as [bigint, bigint]),
    ),
    solution: [],
  });
  it('3 rows: 9 of 15 thread slots busy, n / (2n - 1)', () => {
    expect(busyFraction(trace(system('book1'), { pivot: false, arith: 'exact' }))).toBe(0.6);
    expect(elimBusyFraction(3)).toBe(0.6);
  });
  it('a dense 24 × 24 system matches the formula', () => {
    expect(busyFraction(trace(dense(24), { pivot: false, arith: 'fp32' }))).toBeCloseTo(elimBusyFraction(24), 12);
  });
  it('q-idle: 8 equations: 64 of 120 slots, 53.33% busy', () => {
    expect(elimBusyFraction(8)).toBe(64 / 120);
    expect(Math.abs(elimBusyFraction(8) * 100 - 53.33)).toBeLessThan(53.33 * 0.01);
    expect(Math.abs(50 - 53.33)).toBeGreaterThan(53.33 * 0.01); // "about half" is not accepted
    expect(busyFraction(trace(dense(8), { pivot: false, arith: 'fp32' }))).toBeCloseTo(elimBusyFraction(8), 12);
  });
  it('1,024 equations: about 50% busy, tending to 1/2', () => {
    expect(elimBusyFraction(1024)).toBeCloseTo(0.5, 3);
    expect(Math.abs(elimBusyFraction(1024) * 100 - 50)).toBeLessThan(50 * 0.05);
  });
});

describe('the scene "One thread per row" (content/appA/narration/gauss-threads-*.txt quote these)', () => {
  const script = (n: number) => fs.readFileSync(path.join(__dirname, `../../../content/appA/narration/gauss-threads-${n}.txt`), 'utf8');
  const t1 = trace(system('book1'), { pivot: false, arith: 'exact' });
  it('thread 0 has 3, 5, 2, 19; the divisors are 3, 2, 1', () => {
    expect(t1.phases[0].m[0].map((c) => c.text)).toEqual(['3', '5', '2', '19']);
    expect(script(1)).toMatch(/Thread 0 has the row 3, 5, 2, 19/);
    expect(t1.phases[1].divisors!.map((d) => d!.text)).toEqual(['3', '2', '1']);
    expect(script(2)).toMatch(/thread 0 by 3, thread 1 by 2, thread 2 by 1/);
  });
  it('thread 1 finds Y = 2, then thread 0 finds X = 1', () => {
    const bs = t1.phases.filter((p) => p.kind === 'backsub');
    expect(bs.map((p) => [p.col, p.m[p.col][3].text])).toEqual([[1, '2'], [0, '1']]);
    expect(script(4)).toMatch(/thread 1 finds that Y is 2\. Then thread 0 finds that X is 1/);
  });
  it('the second system: row 0, 5, 2, 16; thread 1\'s row wins with a 2', () => {
    expect(system('book2').rows[0].map(([n]) => Number(n))).toEqual([0, 5, 2, 16]);
    expect(script(5)).toMatch(/0, 5, 2, 16/);
    const pv = trace(system('book2'), { pivot: true, arith: 'exact' }).phases.find((p) => p.kind === 'pivot')!;
    expect(pv.pivot!.winner).toBe(1);
    expect(script(6)).toMatch(/Thread 1's row wins with a 2/);
  });
  it('every script is 35 to 60 words with no symbols read aloud', () => {
    for (let n = 1; n <= 6; n++) {
      const s = script(n).replace(/\[[^\]]*\]\s*/g, '').trim();
      const w = s.split(/\s+/).length;
      expect(w, `gauss-threads-${n}`).toBeGreaterThanOrEqual(35);
      expect(w, `gauss-threads-${n}`).toBeLessThanOrEqual(60);
      expect(s, `gauss-threads-${n}`).not.toMatch(/[<>=×%/_(){}^*+]/);
    }
  });
});
