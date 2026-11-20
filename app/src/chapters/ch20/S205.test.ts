import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { tileClasses } from '../../sim/flashplan';
import { PROB, RANGES, TRACE } from './FlashBlockScene';
import { MERGED, SCENE, SPLIT } from './OnlineMergeScene';

// The narration of section 20.5's two scenes states numbers the scenes draw from the
// models; if a model or the scene data changes, the audio would go on saying the old
// numbers. This ties each number in the scripts to the value drawn now, and the quoted
// llama.cpp lines to the facts file they were read into.
const repo = path.resolve(__dirname, '../../../..');
const script = (clip: string) => fs.readFileSync(path.join(repo, `content/ch20/narration/${clip}.txt`), 'utf8').trim();
const words = (s: string) => s.replace(/\[[^\]]*\]/g, '').split(/\s+/).filter(Boolean).length;
const CLIPS = [1, 2, 3, 4, 5, 6].flatMap((n) => [`online-merge-${n}`, `flash-block-${n}`]);

describe('every script is written for the ear', () => {
  it('35 to 60 words a beat, and no symbols read aloud', () => {
    for (const c of CLIPS) {
      expect(words(script(c)), c).toBeGreaterThanOrEqual(35);
      expect(words(script(c)), c).toBeLessThanOrEqual(60);
      expect(script(c), c).not.toMatch(/[=×*+/<>%−]/);
    }
  });
});

describe('online-merge narration agrees with the model', () => {
  const [t1, t2, t3] = SCENE.steps;
  it('twelve scores, tiles of four', () => {
    expect(SCENE.config.scores.length).toBe(12);
    expect(SCENE.steps.map((s) => s.cols.length)).toEqual([4, 4, 4]);
    expect(script('online-merge-1')).toContain('one row of twelve attention scores');
  });
  it('tile one: max 1.6', () => {
    expect(t1.mB).toBeCloseTo(1.6, 6);
    expect(script('online-merge-2')).toContain('Its own maximum is 1.6.');
  });
  it('tile two: max 2.9, factor about 0.27', () => {
    expect(t2.mB).toBeCloseTo(2.9, 6);
    expect(t2.rose).toBe(true);
    expect(t2.aFactor).toBeGreaterThan(0.265);
    expect(t2.aFactor).toBeLessThan(0.275);
    expect(script('online-merge-3')).toContain('a larger score, 2.9');
    expect(script('online-merge-3')).toContain('about 0.27');
  });
  it('tile three: max 2.1, factor exactly 1', () => {
    expect(t3.mB).toBeCloseTo(2.1, 6);
    expect(t3.aFactor).toBe(1);
    expect(script('online-merge-4')).toContain('largest score is 2.1');
    expect(script('online-merge-4')).toContain('exactly 1');
  });
  it('the end: a few parts in a hundred million', () => {
    expect(SCENE.nan).toBe(false);
    expect(SCENE.diff).toBeGreaterThan(1e-9);
    expect(SCENE.diff).toBeLessThan(1e-7);
    expect(script('online-merge-5')).toContain('a few parts in a hundred million');
  });
  it('the split: two blocks of columns merge to the same result', () => {
    SCENE.final.forEach((x, c) => expect(MERGED.O[c] / MERGED.D).toBeCloseTo(x, 6));
    expect(SPLIT.A.m).toBeGreaterThan(SPLIT.B.m);
    expect(script('online-merge-6')).toContain('the maximum and the sum');
  });
});

describe('flash-block narration agrees with the model', () => {
  const steps = TRACE.steps;
  it('block 1 is rows 4 to 7; warps of four lanes, two rows each', () => {
    expect(TRACE.panel).toBe(1);
    expect(TRACE.config.lanes).toBe(4);
    expect(TRACE.config.brWarp).toBe(2);
    expect(steps[0].text).toContain('rows 4 and 5');
    expect(script('flash-block-1')).toContain('rows 4 to 7');
    expect(script('flash-block-1')).toContain('lane 0 takes columns 0 and 4, lane 1 takes 1 and 5');
  });
  it('eight threads load four rows of K and V, then a block barrier', () => {
    const [a, b] = RANGES[1];
    const loads = steps.slice(a, b).filter((s) => s.op === 'load_KT_and_V');
    expect(loads.flatMap((s) => s.lanes).length).toBe(8);
    expect(TRACE.config.bc).toBe(4);
    expect(steps[b - 1].op).toBe('release_syncthreads');
    expect(script('flash-block-2')).toContain('All eight threads copy four rows of K');
  });
  it('beat 4 contains the warp barrier between P and O', () => {
    const [a, b] = RANGES[3];
    const ops = steps.slice(a, b).map((s) => s.op);
    expect(ops.indexOf('release_syncwarp')).toBeGreaterThan(ops.indexOf('compute_P_and_update_D'));
    expect(ops.indexOf('compute_O')).toBeGreaterThan(ops.indexOf('release_syncwarp'));
  });
  it('tile 2 lies above the diagonal of panel 1, and every P in it is 0', () => {
    expect(tileClasses(12, 4, 4).grid[1][2]).toBe('skip');
    const lastP = steps.filter((s) => s.op === 'compute_P_and_update_D' && s.tile === 2).pop()!;
    const snap = TRACE.snaps[lastP.index + 1];
    snap.S.forEach((x, i) => {
      if (snap.Stile[i] === 2) expect(x).toBe(0);
    });
    expect(script('flash-block-5')).toContain('tile 2 lies entirely above the diagonal');
  });
  it('the stored rows are correct and L is written', () => {
    for (const [r, v] of Object.entries(TRACE.stored)) {
      expect(Math.abs(v.L - PROB.L[Number(r)])).toBeLessThan(1e-12);
    }
    expect(Object.keys(TRACE.stored).map(Number)).toEqual([4, 5, 6, 7]);
    expect(script('flash-block-6')).toContain('the max plus the log of D');
  });
});

describe('the quoted llama.cpp lines are the ones in the facts file', () => {
  it('every line of the quote appears in notes/today/ch20-facts.md D1', () => {
    const src = fs.readFileSync(path.join(__dirname, 'S205.tsx'), 'utf8');
    const m = /const LLAMA = `([\s\S]*?)`;/.exec(src);
    expect(m).not.toBeNull();
    const facts = fs.readFileSync(path.join(repo, 'notes/today/ch20-facts.md'), 'utf8');
    for (const line of m![1].split('\n').filter((l) => l.trim())) expect(facts, line).toContain(line);
  });
});
