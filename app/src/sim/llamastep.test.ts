import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  FLOAT, HEAD, STORIES15M, STORIES260K, SYNTHETIC, launchCount, llama2cBytes, llama2cFloats, stepLaunches, totals, validate, type Config, type Job,
} from './llamastep';

/** llama2.c's forward as lab X3.2 launches it (x3-facts D1; labs/x3/lab2_tinyllama/llama.h), one layer. */
const LAYER: Job[] = ['rmsnorm', 'q', 'k', 'v', 'rope-q', 'rope-k', 'append', 'attention', 'wo', 'residual', 'rmsnorm', 'w1', 'w3', 'swiglu', 'w2', 'residual'];

describe('the launch list', () => {
  it('follows llama2.c’s forward: embed, the layers, final norm, LM head, argmax', () => {
    const ls = stepLaunches(SYNTHETIC, 0);
    const jobs = ls.map((l) => l.job);
    expect(jobs[0]).toBe('embed');
    for (let l = 0; l < SYNTHETIC.layers; l++) expect(jobs.slice(1 + 16 * l, 17 + 16 * l)).toEqual(LAYER);
    expect(jobs.slice(-4)).toEqual(['final-norm', 'lm-head', 'argmax', 'advance']);
    expect(ls.length).toBe(launchCount(SYNTHETIC.layers));
    expect(ls.map((l) => l.n)).toEqual(ls.map((_, i) => i));
  });
  it('counts 1 + 16 L + 4 launches per token', () => {
    expect(launchCount(4)).toBe(69);
    expect(launchCount(30)).toBe(485);
  });
  it('only RMSNorm, RoPE and SwiGLU (lab X3.1) and the advance (lab X3.2) are new', () => {
    const missing = new Set(stepLaunches(SYNTHETIC, 3).filter((l) => !l.exists).map((l) => l.job));
    expect([...missing].sort()).toEqual(['advance', 'final-norm', 'rmsnorm', 'rope-k', 'rope-q', 'swiglu']);
  });
  it('names the supplying labs from the project notes', () => {
    const ls = stepLaunches(SYNTHETIC, 0);
    const by = (j: Job) => ls.find((l) => l.job === j)!;
    expect(by('embed').fn).toBe('launchEmbed');
    expect(by('embed').lab).toBe('lab 22.2');
    expect(by('append').fn).toBe('launchKvAppend');
    expect(by('attention').fn).toBe('launchDecodeAttentionGrouped'); // 4 query heads over 2 KV heads
    expect(by('attention').lab).toBe('lab 20.5');
    expect(by('residual').fn).toBe('launchAddResidual');
    expect(by('argmax').fn).toBe('launchArgmaxRows');
    expect(stepLaunches({ ...SYNTHETIC, kvHeads: 4 }, 0).find((l) => l.job === 'attention')!.lab).toBe('lab 20.4');
    expect(stepLaunches(SYNTHETIC, 0, 'sample').at(-2)!.fn).toBe('launchSampleRows');
  });
  it('the RoPE launch touches q and k only, and the append writes row pos', () => {
    const ls = stepLaunches(SYNTHETIC, 9);
    const ropes = ls.filter((l) => l.layer === 0 && l.kind === 'rope');
    expect(ropes.map((r) => r.inputs.split(' ')[0])).toEqual(['q', 'k']);
    expect(ropes.map((r) => r.inputs).join(' ')).toContain('as 4 heads');
    expect(ropes.map((r) => r.inputs).join(' ')).toContain('as 2 heads');
    expect(ls.find((l) => l.job === 'append')!.outputs).toBe('K, V cache row 9');
  });
  it('rejects shapes M7 or llama2.c would not take', () => {
    expect(validate({ ...SYNTHETIC, kvHeads: 3 })).toMatch(/multiple/);
    expect(validate({ ...SYNTHETIC, d: 500 })).toMatch(/heads × head/);
    expect(HEAD).toBe(128);
  });
});

describe('bytes per token', () => {
  // E1: floats per file and bytes on the Hub for the two tinyllamas headers.
  it("reproduces E1's file sizes", () => {
    expect(llama2cFloats(STORIES15M)).toBe(15_204_000);
    expect(llama2cBytes(STORIES15M)).toBe(60_816_028);
    expect(llama2cFloats(STORIES260K)).toBe(264_128);
    expect(llama2cBytes(STORIES260K)).toBe(1_056_540);
  });
  it('the step reads every weight once, plus the embedding row, minus the skipped tables', () => {
    for (const c of [STORIES15M, STORIES260K, SYNTHETIC] as Config[]) {
      const t = totals(stepLaunches(c, 0));
      expect(t.weightBytes).toBe(FLOAT * (llama2cFloats(c) - c.seqLen * c.head) + FLOAT * c.d);
    }
  });
  it('the LM head share is vocab·d over all weights read', () => {
    const t = totals(stepLaunches(STORIES15M, 0));
    expect(t.lmHeadBytes).toBe(32000 * 288 * 4);
    expect(t.lmHeadShare).toBeCloseTo((32000 * 288) / (15_204_000 - 256 * 48 + 288), 12);
    expect(t.lmHeadShare).toBeCloseTo(0.6068, 3);
    expect(totals(stepLaunches(SYNTHETIC, 0)).lmHeadShare).toBeCloseTo(0.1531, 3);
  });
  it('attention reads grow with pos; weights do not', () => {
    const a = totals(stepLaunches(SYNTHETIC, 0));
    const b = totals(stepLaunches(SYNTHETIC, 200));
    expect(b.weightBytes).toBe(a.weightBytes);
    expect(b.cacheBytes - a.cacheBytes).toBe(SYNTHETIC.layers * 2 * 200 * SYNTHETIC.kvHeads * HEAD * 4);
    expect(b.byKind.attention).toBeGreaterThan(a.byKind.attention);
  });
  it('the RoPE tables read per token are one row per rotated buffer per layer', () => {
    expect(totals(stepLaunches(SYNTHETIC, 5)).tableBytes).toBe(SYNTHETIC.layers * 2 * HEAD * 4);
  });
  it('SwiGLU for one token of SmolLM2 (hidden 1536) moves 18432 bytes', () => {
    const c: Config = { d: 576, hidden: 1536, layers: 1, heads: 9, kvHeads: 3, vocab: 49152, head: 64, seqLen: 8 };
    expect(stepLaunches(c, 0).find((l) => l.job === 'swiglu')!.actBytes).toBe(18432);
  });
});

// Lab X3.2 is written in parallel with this lesson. Once its header exists, every course launch
// function the model names must be declared there (the integrator also checks the README's order).
const header = path.resolve(__dirname, '../../../labs/x3/lab2_tinyllama/llama.h');
describe("lab X3.2's header", () => {
  it.skipIf(!fs.existsSync(header))('declares every launch function of the step', () => {
    const src = fs.readFileSync(header, 'utf8');
    const fns = new Set(stepLaunches(SYNTHETIC, 0).map((l) => l.fn).filter((f) => /^launch\w+$/.test(f) && f !== 'launchDecodeAttentionGrouped'));
    for (const f of fns) expect(src, f).toMatch(new RegExp('\\b' + f + '\\s*\\('));
  });
});
