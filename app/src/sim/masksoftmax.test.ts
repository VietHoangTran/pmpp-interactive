import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { causalSoftmaxProblem } from './attnref';
import { functionSource } from './toolchain';
import {
  BARRIER_LINES,
  CORRECT_SOFTMAX,
  LINE,
  NEG_IDENTITY,
  SCHEDULES,
  VARIANTS,
  columnsOf,
  outputColumnsOf,
  runAllRows,
  runRow,
  stateAt,
  unfusedTraffic,
  type SoftmaxConfig,
} from './masksoftmax';

const cfg = (over: Partial<SoftmaxConfig> = {}, n = 8, block = 4): SoftmaxConfig => ({ ...CORRECT_SOFTMAX, n, block, ...over });
const input = causalSoftmaxProblem(8, 0);
const input90 = causalSoftmaxProblem(8, 90);

describe('who reads what (Fig. 20.4 lines 7 and 16)', () => {
  it('BLOCK_SIZE 4, N 8: in block 5, thread 1 reads columns 1 and 5 (question q-cols)', () => {
    expect(columnsOf(cfg(), 5, 1)).toEqual([1, 5]);
  });
  it('in block 2, thread 3 reads nothing (question q-idle) and contributes the identities', () => {
    expect(columnsOf(cfg(), 2, 3)).toEqual([]);
    const tr = runRow(cfg(), input, 2, 'in-order');
    const st = tr.steps.find((s) => s.thread === -1 && s.line === 11)!.after;
    expect(st.threads[3].privMax).toBe(NEG_IDENTITY);
    const st2 = tr.steps.find((s) => s.thread === -1 && s.line === 20)!.after;
    expect(st2.threads[3].privSum).toBe(0);
  });
  it('block r has r + 1 live columns; the output loop covers all N', () => {
    for (let r = 0; r < 8; r++) {
      const all = [0, 1, 2, 3].flatMap((t) => columnsOf(cfg(), r, t)).sort((a, b) => a - b);
      expect(all).toEqual(Array.from({ length: r + 1 }, (_, i) => i));
    }
    expect([0, 1, 2, 3].flatMap((t) => outputColumnsOf(cfg(), t)).sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
  });
});

describe('the corrected kernel', () => {
  it('is right in every thread order, for several sizes', () => {
    for (const [n, block] of [[4, 2], [8, 4], [10, 8], [6, 8]] as const) {
      const inp = causalSoftmaxProblem(n, 0);
      for (const s of SCHEDULES) expect(runAllRows(cfg({}, n, block), inp, s).correct).toBe(true);
    }
  });
  it('the reduction result is valid in thread 0 only until it is broadcast', () => {
    const tr = runRow(cfg(), input, 6, 'in-order');
    const k = tr.steps.findIndex((s) => s.thread === -1 && s.line === 11);
    const after = tr.steps[k].after;
    expect(Number.isNaN(after.threads[0].maxRow)).toBe(false);
    expect(after.threads.slice(1).every((x) => Number.isNaN(x.maxRow))).toBe(true);
    const end = stateAt(cfg(), tr, tr.steps.length);
    expect(new Set(end.threads.map((x) => x.maxRow)).size).toBe(1);
    expect(end.shared.holds).toBe('sum');
  });
  it('stateAt(0) is the state before anything ran', () => {
    const tr = runRow(cfg(), input, 3, 'lockstep');
    const s0 = stateAt(cfg(), tr, 0);
    expect(s0.P.every((x) => x === null)).toBe(true);
    expect(s0.shared.holds).toBe('unset');
  });
});

describe('each defect fails as documented', () => {
  const outcome = (over: Partial<SoftmaxConfig>, inp = input) => SCHEDULES.map((s) => runAllRows(cfg(over), inp, s).correct);

  it('without line 13 the readers can get ahead of thread 0 (read after write): lockstep and in-order hide it', () => {
    expect(outcome({ barrier13: false })).toEqual([true, true, false, false]);
    const r = runAllRows(cfg({ barrier13: false }), input, 'reversed');
    expect(r.kinds).toContain('read-before-write');
  });
  it('without line 22, the same for the sum', () => {
    expect(outcome({ barrier22: false })).toEqual([true, true, false, false]);
  });
  it('without line 19 every order is still right: the reduction itself orders thread 0\'s write after every read', () => {
    expect(outcome({ barrier19: false })).toEqual([true, true, true, true]);
    for (const s of SCHEDULES) expect(runAllRows(cfg({ barrier19: false }), input, s).kinds).toEqual([]);
  });
  it('without line 10 every order is right too: nothing before it needs protecting in a one-row block', () => {
    expect(outcome({ barrier10: false })).toEqual([true, true, true, true]);
  });
  it('without 13 and 19, CUB\'s rule for reusing temp_store is broken (a contract check, not a wrong value)', () => {
    const r = runAllRows(cfg({ barrier13: false, barrier19: false }), input, 'lockstep');
    expect(r.kinds).toContain('temp-reuse');
  });
  it('with idx < N the max and sum include the future: every row but the last is wrong', () => {
    for (const s of SCHEDULES) {
      const r = runAllRows(cfg({ bound: 'n' }), input, s);
      expect(r.wrongRows).toEqual([0, 1, 2, 3, 4, 5, 6]);
      expect(r.kinds).toContain('masked-read');
    }
  });
  it('without the max subtraction nothing changes at ordinary logits...', () => {
    expect(outcome({ subtractMax: false })).toEqual([true, true, true, true]);
  });
  it('...and at +90 exp overflows float32 and the rows turn into NaN', () => {
    const r = runAllRows(cfg({ subtractMax: false }), input90, 'lockstep');
    expect(r.correct).toBe(false);
    expect(r.kinds).toContain('overflow');
    const tr = runRow(cfg({ subtractMax: false }), input90, 7, 'lockstep');
    expect(tr.P.slice(0, 8).some(Number.isNaN)).toBe(true);
    // The corrected kernel handles the same logits.
    for (const s of SCHEDULES) expect(runAllRows(cfg(), input90, s).correct).toBe(true);
  });
  it('the outcome table\'s variants are the ones above', () => {
    expect(VARIANTS.map((v) => v.id)).toEqual(['as-corrected', 'no10', 'no13', 'no19', 'no22', 'bound', 'nomax', 'nomax90']);
  });
});

// Section 20.3 shows labs/ch20/snippets/s203_softmax_corrected.cu (loaded at run time, so
// there is no lesson copy to drift) and highlights the line this model is on. These pin the
// snippet's lines to the model's, and to the book's numbering (Fig. 20.4, 28 lines).
describe('the snippet the stepper shows', () => {
  const src = fs.readFileSync(path.resolve(__dirname, '../../../labs/ch20/snippets/s203_softmax_corrected.cu'), 'utf8');
  const fn = functionSource(src, 'softmax_kernel')!;
  const line = fn.text.split('\n').map((l) => l.trim());
  const at = (n: number) => line[n - 1];
  it('keeps the book\'s 28 lines', () => {
    expect(line).toHaveLength(28);
    expect(at(1)).toBe('__global__ void softmax_kernel(const float* S, float* L, float* P, int N) {');
    expect(at(28)).toBe('}');
  });
  it('has the barriers where the model has them', () => {
    for (const n of BARRIER_LINES) expect(at(n)).toBe('__syncthreads();');
    expect(line.filter((l) => l === '__syncthreads();')).toHaveLength(4);
  });
  it('has every operation on the line the model gives it', () => {
    expect(at(6)).toBe('float max_val_thread = -FLT_MAX;');
    expect(at(7)).toBe('for (int idx = threadIdx.x; idx <= blockIdx.x; idx += blockDim.x) {');
    expect(at(LINE.max)).toBe('if (S_row[idx] > max_val_thread) max_val_thread = S_row[idx];');
    expect(at(LINE.reduceMax)).toBe('float max_val_row = BlockReduce(temp_store).Reduce(max_val_thread, cuda::maximum<>{});');
    expect(at(LINE.writeMax)).toBe('if (threadIdx.x == 0) max_or_sum = max_val_row;');
    expect(at(LINE.readMax)).toBe('max_val_row = max_or_sum;');
    expect(at(16)).toBe(at(7));
    expect(at(LINE.sum)).toBe('sum_thread += expf(S_row[idx] - max_val_row);');
    expect(at(LINE.reduceSum)).toBe('float sum_row = BlockReduce(temp_store).Reduce(sum_thread, cuda::std::plus<>{});');
    expect(at(LINE.writeSum)).toBe('if (threadIdx.x == 0) max_or_sum = sum_row;');
    expect(at(LINE.readSum)).toBe('sum_row = max_or_sum;');
    expect(at(24)).toBe('for (int idx = threadIdx.x; idx < N; idx += blockDim.x) {');
    expect(at(LINE.out)).toBe('P[blockIdx.x * N + idx] = idx <= blockIdx.x ? expf(S_row[idx] - max_val_row) / sum_row : 0.0f;');
    expect(at(LINE.lse)).toBe('if (threadIdx.x == 0) L[blockIdx.x] = max_val_row + logf(sum_row);');
  });
  it('the identity in the file is the model\'s', () => {
    expect(NEG_IDENTITY).toBe(-3.4028234663852886e38); // FLT_MAX
  });
});

describe('traffic of the unfused pipeline', () => {
  it('one fp32 head: the softmax moves 10 N^2 + 10 N bytes (three passes over the visible half, P and L written)', () => {
    for (const n of [4, 64, 4096]) {
      const t = unfusedTraffic(n, 128, 1, 4);
      expect(t.softmax.read + t.softmax.write).toBe(10 * n * n + 10 * n);
    }
  });
  it('hand count for N = 4, d = 2, one fp32 head', () => {
    const t = unfusedTraffic(4, 2, 1, 4);
    expect(t.gemm1).toEqual({ read: 2 * 4 * 2 * 4, write: 16 * 4 });
    expect(t.gemm2).toEqual({ read: (16 + 8) * 4, write: 8 * 4 });
    expect(t.irreducible).toBe(4 * 4 * 2 * 4);
  });
  it('32 heads, N = 4096, fp32: S and P hold 4 GiB (question q-sp-bytes); one S is 64 MiB (q-s-bytes)', () => {
    expect(unfusedTraffic(4096, 128, 32, 4).sp).toBe(4 * 2 ** 30);
    expect(unfusedTraffic(4096, 128, 1, 4).sp / 2).toBe(64 * 2 ** 20);
  });
  it('the traffic beyond the irreducible grows like N / d (about 1.125 N / d at any precision)', () => {
    const excess = (n: number, d: number, b: number) => {
      const t = unfusedTraffic(n, d, 1, b);
      return (t.total - t.irreducible) / t.irreducible;
    };
    expect(excess(4096, 128, 4) / excess(1024, 128, 4)).toBeCloseTo(4, 2);
    expect(excess(4096, 64, 4) / excess(4096, 128, 4)).toBeCloseTo(2, 2);
    expect(excess(4096, 128, 4) / (4096 / 128)).toBeCloseTo(1.125, 2);
  });
});
