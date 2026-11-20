import { describe, expect, it } from 'vitest';
import {
  TOY,
  exactRow,
  makeProblem,
  panels,
  rowOk,
  runBlock,
  runKernel,
  tiles,
  withDefects,
  type KConfig,
  type LaneSchedule,
  type WarpSchedule,
} from './flashkernel';

const LANES: LaneSchedule[] = ['lockstep', 'apart'];
const WARPS: WarpSchedule[] = ['together', 'ahead'];
const SCHEDULES = LANES.flatMap((l) => WARPS.map((w) => [l, w] as const));
const prob12 = makeProblem(12, 8);

describe('the corrected kernel matches exact attention', () => {
  it('toy size, every schedule', () => {
    for (const [l, w] of SCHEDULES) {
      const r = runKernel(TOY, prob12, l, w);
      expect(r.correct, `${l}/${w}`).toBe(true);
      expect(r.violations, `${l}/${w}`).toEqual([]);
    }
  });
  it('N not a multiple of B_r or B_c (10 rows: ceiling, zero-fill, masks by index, rows ≥ N not stored)', () => {
    const c = { ...TOY, n: 10 };
    const p = makeProblem(10, 8);
    expect(panels(c)).toBe(3);
    expect(tiles(c)).toBe(3);
    for (const [l, w] of SCHEDULES) expect(runKernel(c, p, l, w).correct, `${l}/${w}`).toBe(true);
    const last = runBlock(c, p, 2, 'lockstep', 'together');
    expect(Object.keys(last.stored).map(Number)).toEqual([8, 9]);
  });
  it('one row per warp, B_c 2 and 6 (lanes without a column bring the identity), other sizes', () => {
    const cases: Partial<KConfig>[] = [{ brWarp: 1 }, { bc: 2 }, { bc: 6 }, { n: 8 }, { n: 9, bc: 6, brWarp: 1 }];
    for (const over of cases) {
      const c = { ...TOY, ...over };
      const p = makeProblem(c.n, c.d);
      for (const [l, w] of SCHEDULES) expect(runKernel(c, p, l, w).correct, `${JSON.stringify(over)} ${l}/${w}`).toBe(true);
    }
  });
  it('a finite start with right-to-left tiles is still correct', () => {
    const c = { ...TOY, order: 'rtl' as const };
    for (const [l, w] of SCHEDULES) expect(runKernel(c, prob12, l, w).correct).toBe(true);
  });
  it('stores L = m + log D (FlashAttention-2’s logsumexp)', () => {
    const tr = runBlock(TOY, prob12, 1, 'lockstep', 'together');
    for (const [r, v] of Object.entries(tr.stored)) {
      expect(Math.abs(v.L - prob12.L[Number(r)])).toBeLessThan(1e-12);
      expect(rowOk(prob12, Number(r), v)).toBe(true);
    }
  });
});

describe('each defect fails as documented', () => {
  it('no __syncwarp: wrong only when lanes run apart, with stale P reads flagged', () => {
    const c = withDefects(TOY, ['noSyncwarp']);
    for (const w of WARPS) {
      expect(runKernel(c, prob12, 'lockstep', w).correct, `lockstep/${w}`).toBe(true);
      const r = runKernel(c, prob12, 'apart', w);
      expect(r.correct, `apart/${w}`).toBe(false);
      expect(r.violations.some((v) => v.kind === 'stale-P')).toBe(true);
    }
  });
  it('update_m_and_D returning the new max: wrong on every schedule, with no bad read at all', () => {
    const c = withDefects(TOY, ['newMax']);
    for (const [l, w] of SCHEDULES) {
      const r = runKernel(c, prob12, l, w);
      expect(r.correct, `${l}/${w}`).toBe(false);
      expect(r.violations).toEqual([]);
    }
  });
  it('no barrier after the warp loop: wrong when warp 0 runs ahead (it reloads the tiles warp 1 is reading)', () => {
    const c = withDefects(TOY, ['noBarrierAfter']);
    expect(runKernel(c, prob12, 'lockstep', 'together').correct).toBe(true);
    for (const l of LANES) {
      const r = runKernel(c, prob12, l, 'ahead');
      expect(r.correct, l).toBe(false);
      expect(r.violations.some((v) => v.kind === 'kt-tile' || v.kind === 'v-tile')).toBe(true);
    }
  });
  it('minus infinity and right to left: NaN in the first panel on every schedule', () => {
    const c = withDefects(TOY, ['negInfRtl']);
    for (const [l, w] of SCHEDULES) {
      const r = runKernel(c, prob12, l, w);
      expect(r.correct).toBe(false);
      expect(r.O.slice(0, 8).every(Number.isNaN)).toBe(true);
    }
  });
  it('minus infinity alone (left to right) is correct: the book’s order avoids the hazard', () => {
    const c = { ...TOY, init: 'neg-inf' as const };
    for (const [l, w] of SCHEDULES) expect(runKernel(c, prob12, l, w).correct).toBe(true);
  });
});

describe('the trace the stepper and the scene draw from', () => {
  const tr = runBlock(TOY, prob12, 1, 'lockstep', 'together');
  it('one snapshot per step, plus the start', () => {
    expect(tr.snaps.length).toBe(tr.steps.length + 1);
  });
  it('lockstep: 2 block barriers and 2 warp barriers per tile, 3 tiles', () => {
    const rel = (op: string) => tr.steps.filter((s) => s.op === op).length;
    expect(rel('release_syncthreads')).toBe(6);
    expect(rel('release_syncwarp')).toBe(3 * 2 * 2);
  });
  it('panel 1, tile 2 lies above the diagonal: every P element is 0 and the max does not move', () => {
    const lastP = tr.steps.filter((s) => s.op === 'compute_P_and_update_D' && s.tile === 2);
    expect(lastP.length).toBe(4);
    const at = lastP[lastP.length - 1].index + 1;
    const S = tr.snaps[at].S;
    expect(S.every((x, i) => (tr.snaps[at].Stile[i] === 2 ? x === 0 : true))).toBe(true);
  });
  it('after each compute_O, O~/D equals exact attention over the tiles merged so far', () => {
    for (const s of tr.steps.filter((x) => x.op === 'compute_O')) {
      const snap = tr.snaps[s.index + 1];
      const w = s.warp;
      const ii = s.ii!;
      const lr = w * TOY.brWarp + ii;
      const row = 4 + lr;
      const cols = snap.merged[lr][0].flatMap((t) => [0, 1, 2, 3].map((k) => t * 4 + k));
      const ex = exactRow(prob12, row, cols)!;
      for (let dd = 0; dd < 8; dd++) {
        const reg = snap.regs[w * 4 + (dd % 4)];
        expect(reg.O[ii][Math.floor(dd / 4)] / reg.D[ii]).toBeCloseTo(ex[dd], 10);
      }
    }
  });
  it('the narration of a step names the row, the tile and the values', () => {
    const s = tr.steps.find((x) => x.op === 'update_m_and_D' && x.tile === 1)!;
    expect(s.text).toMatch(/^Warp \d, row \d+: the max (rises|stays)/);
    expect(tr.steps[0].text).toContain('loads its rows of Q (rows 4 and 5)');
  });
  it('apart: one lane per event, lane 0 first', () => {
    const a = runBlock(TOY, prob12, 0, 'apart', 'together');
    expect(a.steps.filter((s) => s.warp >= 0 && !s.op.startsWith('release')).every((s) => s.lanes.length === 1)).toBe(true);
    expect(a.steps[0].lanes).toEqual([0]);
  });
});
