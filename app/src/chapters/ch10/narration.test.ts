import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { FIG103B, TIE_ROW, builderTree, f32bits, treeStats } from '../../sim/reducetree';
import { smallValues } from '../../sim/shuffle';
import { SCENE } from './ShuffleScene';
import { SCENE_FLOATS, pairTree } from './TreeFoldScene';

// The narration of "Lanes exchanging values" (section 10.7) and "A reduction tree folding"
// (section 10.2) states numbers that the scenes draw from the models. If a model or a
// scene's data changes, the audio would go on saying the old numbers; this test ties each
// number in the scripts to the value the scene now draws. (The commit-order scripts have
// the same check in S109.snippets.test.ts.)
const repo = path.resolve(__dirname, '../../../..');
const script = (clip: string) => fs.readFileSync(path.join(repo, `content/ch10/narration/${clip}.txt`), 'utf8').trim();
const words = (s: string) => s.replace(/\[[^\]]*\]/g, '').split(/\s+/).filter(Boolean).length;

describe('every script is written for the ear', () => {
  const clips = [1, 2, 3, 4, 5, 6].flatMap((n) => [`shuffle-${n}`, `tree-fold-${n}`]);
  it('35 to 60 words a beat, and no symbols read aloud', () => {
    for (const c of clips) {
      expect(words(script(c)), c).toBeGreaterThanOrEqual(35);
      expect(words(script(c)), c).toBeLessThanOrEqual(60);
      expect(script(c), c).not.toMatch(/[=×*+/<>%]/);
    }
  });
});

describe('the shuffle narration agrees with the model', () => {
  const down = SCENE.down;
  const xor = SCENE.xor;
  const strides = (t: typeof down) => t.steps.map((s) => s.stride);
  it('the scene reduces smallValues(32), whose sum is 162', () => {
    expect(down.initial).toEqual(smallValues(32));
    expect(smallValues(32).reduce((a, b) => a + b, 0)).toBe(162);
  });
  it('part 1: 32 lanes; at distance 16, lanes 0 to 15 receive lanes 16 to 31, the top 16 keep their own', () => {
    expect(down.hardwareWidth).toBe(32);
    const first = down.steps[0];
    expect(first.stride).toBe(16);
    for (let l = 0; l < 16; l++) expect([first.lanes[l].outcome, first.lanes[l].src]).toEqual(['moved', l + 16]);
    for (let l = 16; l < 32; l++) expect(first.lanes[l].outcome).toBe('own');
    expect(script('shuffle-1')).toContain('Here are 32 lanes of one warp.');
    expect(script('shuffle-1')).toContain('the value held 16 lanes above it. Lanes 0 to 15 receive the values of lanes 16 to 31. The top 16 lanes');
  });
  it('part 2: distances 16, 8, 4, 2, 1; after 5 steps lane 0 holds the sum', () => {
    expect(strides(down)).toEqual([16, 8, 4, 2, 1]);
    expect(down.lane0).toBe(down.total);
    expect(script('shuffle-2')).toContain('the distance halves: 8, then 4, then 2, then 1.');
    expect(script('shuffle-2')).toContain(`After ${down.steps.length} steps, lane 0 holds the sum of all ${down.hardwareWidth} values. Here that is ${down.lane0}.`);
  });
  it('part 3: lane-operations, the cone and the waste', () => {
    expect(down.laneOps).toBe(down.hardwareWidth * down.steps.length);
    expect(down.useful).toBe(down.hardwareWidth - 1);
    expect(script('shuffle-3')).toContain(`${down.laneOps} operations in all. Only ${down.useful} of them feed lane 0`);
    expect(script('shuffle-3')).toContain(`The other ${down.wasted} produce sums that nobody reads.`);
  });
  it('part 4: the butterfly, same strides, every lane holds the total', () => {
    expect(strides(xor)).toEqual([16, 8, 4, 2, 1]);
    expect(xor.holders).toHaveLength(32);
    expect(script('shuffle-4')).toContain(`Same ${xor.steps.length} steps, different partners.`);
    expect(script('shuffle-4')).toContain('16 apart, then 8, 4, 2 and 1.');
    expect(script('shuffle-4')).toContain(`every lane ends up holding the total, ${xor.total}, not just lane 0.`);
  });
  it('part 5: lanes from 20 on returned; after the second step an undefined value reaches lane 0', () => {
    expect(SCENE.tailFrom).toBe(20);
    const lane0 = SCENE.gone.steps.map((s) => s.lanes[0].after);
    expect(lane0[0]).not.toBeNull();
    expect(lane0[1]).toBeNull();
    expect(SCENE.zero.lane0).toBe(SCENE.zero.total);
    expect(script('shuffle-5')).toContain(`the data ends at lane ${SCENE.tailFrom}, and lanes ${SCENE.tailFrom} and up have already returned.`);
    expect(script('shuffle-5')).toContain('after the second step one reaches lane 0');
  });
  it('part 6: a 64-lane wavefront with the loop for 32: lanes 0 and 32 hold 32 each, nobody holds 64', () => {
    const w = SCENE.wave64;
    expect([w.hardwareWidth, w.steps[0].stride, w.final[0], w.final[32], w.total]).toEqual([64, 16, 32, 32, 64]);
    expect(w.holders).toEqual([]);
    expect(script('shuffle-6')).toContain(`On an AMD wavefront of ${w.hardwareWidth} lanes, the loop written for 32 still starts at ${w.steps[0].stride}.`);
    expect(script('shuffle-6')).toContain(`each holds ${w.final[0]}, and no lane holds ${w.total}.`);
  });
});

describe('the tree-fold narration agrees with the model', () => {
  const loop = builderTree(FIG103B, 'sum', 'loop');
  const tree = builderTree(FIG103B, 'sum', 'neighbours');
  const level = (k: number) => tree.levels[k].map((nd) => nd.value);
  const list = (v: (string | number)[]) => `${v.slice(0, -1).join(', ')} and ${v[v.length - 1]}`;
  it('part 1: the eight values and the loop\'s running total', () => {
    expect(script('tree-fold-1')).toContain(`on 8 values: ${list(FIG103B)}.`);
    const running = loop.levels.slice(1).map((l) => l[0].value);
    expect(running).toHaveLength(7);
    expect(script('tree-fold-1')).toContain(
      `The running total reads ${FIG103B[0]}, then ${running.slice(0, -1).join(', ')}, and finally ${running[running.length - 1]}.`,
    );
    expect([loop.work, loop.span]).toEqual([7, 7]);
    expect(script('tree-fold-1')).toContain('That is 7 additions, in 7 steps, 1 at a time.');
  });
  it('part 2: the pairs, step by step', () => {
    expect(script('tree-fold-2')).toContain(`4 adders work at once and produce ${list(level(1))}.`);
    expect(script('tree-fold-2')).toContain(`2 adders produce ${list(level(2))}.`);
    expect(script('tree-fold-2')).toContain(`a single adder produces ${level(3)[0]}.`);
    expect(tree.opsPerStep).toEqual([4, 2, 1]);
  });
  it('part 3: the same work, a shorter span', () => {
    expect([tree.work, tree.span]).toEqual([loop.work, 3]);
    expect(script('tree-fold-3')).toContain(`The loop did ${loop.work} additions, and so did the tree`);
    expect(script('tree-fold-3')).toContain(`${loop.span} steps against ${tree.span}.`);
  });
  it('part 4: 1024 values, 512 adders down to one in step 10, about 102 on average', () => {
    const s = treeStats(1024);
    expect(s.opsPerStep.slice(0, 3)).toEqual([512, 256, 128]);
    expect([s.span, s.opsPerStep[s.span - 1]]).toEqual([10, 1]);
    expect(script('tree-fold-4')).toContain(`The first step needs ${s.opsPerStep[0]} adders, then ${s.opsPerStep[1]}, then ${s.opsPerStep[2]}, down to a single adder in step ${s.span}.`);
    expect(script('tree-fold-4')).toContain(`about ${Math.floor(s.average)} are busy`);
  });
  it('part 5: eight floats whose loop and tree sums differ in their bits', () => {
    expect(SCENE_FLOATS.values).toHaveLength(8);
    expect(f32bits(SCENE_FLOATS.loop)).not.toBe(f32bits(SCENE_FLOATS.tree));
    expect(script('tree-fold-5')).toContain('Here are 8 float values from the model.');
    expect(script('tree-fold-5')).toContain('the 2 sums differ in their last bits');
  });
  it('part 6: two entries tie; keep-left depends on the bracket, lower index does not', () => {
    const top = Math.max(...TIE_ROW);
    expect(TIE_ROW.filter((v) => v === top)).toHaveLength(2);
    const win = (rule: 'left' | 'lower', shape: 'neighbours' | 'halves') => pairTree(TIE_ROW, rule, shape).root.value.split('#')[1];
    expect(win('left', 'neighbours')).not.toBe(win('left', 'halves'));
    expect(win('lower', 'neighbours')).toBe(win('lower', 'halves'));
    expect(script('tree-fold-6')).toContain('2 entries tie for the top.');
  });
});
