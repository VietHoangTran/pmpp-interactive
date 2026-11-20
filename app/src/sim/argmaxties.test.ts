import { describe, expect, it } from 'vitest';
import {
  BINS, FLT_MAX, IMPLS, INT_MAX, OFFSETS, STAGES, STAGE_LIST, SYSTEMS, TIE_VALUE, WARP,
  argmaxBy, cellOf, cudaArgmax, cudaButterfly, hostGreedy, m4Argmax, metalArgmax, metalMaxLanes, tieRow, torchArgmax,
} from './argmaxties';

describe('the tie row', () => {
  it('has 32 distinct values below the planted maximum', () => {
    const r = tieRow([]);
    expect(r).toHaveLength(WARP);
    expect(new Set(r).size).toBe(WARP);
    expect(Math.max(...r)).toBeLessThan(TIE_VALUE);
  });
  it('without a tie every rule agrees', () => {
    const r = tieRow([9]);
    for (const k of IMPLS) expect(argmaxBy(k.id, r)).toBe(9);
    const s = tieRow([]);
    const top = s.indexOf(Math.max(...s));
    for (const k of IMPLS) expect(argmaxBy(k.id, s)).toBe(top);
  });
});

describe("ch10 D2's worked examples", () => {
  it('columns 1 and 2 tied give 2 on CUDA', () => {
    expect(cudaArgmax(tieRow([1, 2]))).toBe(2);
  });
  it('the reason: lane 0 takes lane 2 at offset 2 and keeps it against lane 1 at offset 1', () => {
    const st = cudaButterfly(tieRow([1, 2]));
    expect(st.map((s) => s.offset)).toEqual([null, ...OFFSETS]);
    const atOffset = (o: number) => st.find((s) => s.offset === o)!;
    expect(atOffset(4).lanes[0].v).toBeLessThan(TIE_VALUE); // no maximum has reached lane 0 yet
    expect(atOffset(2).lanes[0]).toEqual({ v: TIE_VALUE, i: 2 });
    expect(atOffset(2).changed).toContain(0);
    expect(atOffset(1).lanes[0]).toEqual({ v: TIE_VALUE, i: 2 });
    expect(atOffset(1).lanes[1]).toEqual({ v: TIE_VALUE, i: 1 }); // lane 1 kept its own: lanes disagree
  });
  it('columns 0 and 1 give 1 on Metal and 0 on CUDA', () => {
    const r = tieRow([0, 1]);
    expect(metalArgmax(r)).toBe(1);
    expect(metalMaxLanes(r)).toEqual([0, 1]);
    expect(cudaArgmax(r)).toBe(0);
  });
  it('a row of -inf gives -1 on CUDA', () => {
    expect(cudaArgmax(Array(WARP).fill(-Infinity))).toBe(-1);
  });
  it('NaN is skipped on CUDA', () => {
    const r = tieRow([5]);
    r[3] = NaN;
    expect(cudaArgmax(r)).toBe(5);
    r[5] = NaN;
    expect(cudaArgmax(r)).toBe(tieRow([]).indexOf(Math.max(...tieRow([]))));
  });
  it('the host greedy sampler keeps the lowest index', () => {
    expect(hostGreedy(tieRow([1, 2]))).toBe(1);
    expect(hostGreedy(tieRow([0, 1]))).toBe(0);
  });
});

describe('torch.argmax and M4', () => {
  it('both return the first maximum on ties', () => {
    for (const planted of [[1, 2], [0, 1], [3, 7, 30], [31, 16]]) {
      const r = tieRow(planted);
      expect(torchArgmax(r)).toBe(Math.min(...planted));
      expect(m4Argmax(r)).toBe(Math.min(...planted));
    }
  });
  it('torch.argmax lets NaN win; M4 skips it', () => {
    const r = tieRow([6]);
    r[10] = NaN;
    expect(torchArgmax(r)).toBe(10);
    expect(m4Argmax(r)).toBe(6);
  });
  it('M4 on a row of -FLT_MAX returns column 0, and on a row of -inf its identity index', () => {
    expect(m4Argmax(Array(WARP).fill(-FLT_MAX))).toBe(0);
    expect(m4Argmax(Array(WARP).fill(-Infinity))).toBe(INT_MAX);
  });
});

describe('the CUDA answer depends on lane positions, not on first or last', () => {
  it('is neither the lowest nor the highest index for some pairs', () => {
    let notLowest = 0;
    let notHighest = 0;
    for (let a = 0; a < WARP; a++)
      for (let b = a + 1; b < WARP; b++) {
        const g = cudaArgmax(tieRow([a, b]));
        expect([a, b]).toContain(g);
        if (g !== a) notLowest++;
        if (g !== b) notHighest++;
      }
    expect(notLowest).toBeGreaterThan(0);
    expect(notHighest).toBeGreaterThan(0);
  });
  it('the planted pair 0 and 16 meets at the first offset and lane 0 keeps its own', () => {
    expect(cudaArgmax(tieRow([0, 16]))).toBe(0);
    expect(cudaArgmax(tieRow([16, 17]))).toBe(16);
  });
  it('Metal always returns the highest tied index', () => {
    for (const planted of [[1, 2], [0, 31], [4, 9, 12]]) expect(metalArgmax(tieRow(planted))).toBe(Math.max(...planted));
  });
});

describe('where each stage runs', () => {
  it('has one cited cell per system and stage', () => {
    expect(STAGES).toHaveLength(SYSTEMS.length * STAGE_LIST.length);
    for (const s of SYSTEMS)
      for (const st of STAGE_LIST) {
        const c = cellOf(s.id, st);
        expect(c, `${s.id} ${st}`).toBeDefined();
        expect(c!.cite).toMatch(/(x1|ch09|ch11|ch12|ch14) D\d/);
        expect(BINS.map((b) => b.id)).toContain(c!.bin);
      }
  });
  it('llama.cpp defaults run everything on the host; with -bs, in its graph', () => {
    expect(STAGE_LIST.every((s) => cellOf('llama-default', s)!.bin === 'host')).toBe(true);
    expect(STAGE_LIST.every((s) => cellOf('llama-bs', s)!.bin === 'kernels')).toBe(true);
    expect(STAGE_LIST.every((s) => cellOf('transformers', s)!.bin === 'framework')).toBe(true);
    expect(STAGE_LIST.every((s) => cellOf('mlx-lm', s)!.bin === 'framework')).toBe(true);
  });
});
