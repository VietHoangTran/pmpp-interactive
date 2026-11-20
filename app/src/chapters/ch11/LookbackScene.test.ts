import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { SCENE_G, SCENE_R } from '../../sim/lookback';
import { SCENE, SCENE_WIDE } from './LookbackScene';

// The narration of "Waiting for the block before you" (section 11.9) states numbers the scene
// draws from sim/lookback.ts. If the model or the scene's seeds change, the audio would go on
// saying the old numbers; this test ties each number in the scripts to the value drawn now.
const repo = path.resolve(__dirname, '../../../..');
const script = (n: number) => fs.readFileSync(path.join(repo, `content/ch11/narration/lookback-${n}.txt`), 'utf8').trim();
const words = (s: string) => s.replace(/\[[^\]]*\]/g, '').split(/\s+/).filter(Boolean).length;

describe('every lookback script is written for the ear', () => {
  it('35 to 60 words a beat, and no symbols read aloud', () => {
    for (let n = 1; n <= 6; n++) {
      expect(words(script(n)), `lookback-${n}`).toBeGreaterThanOrEqual(35);
      expect(words(script(n)), `lookback-${n}`).toBeLessThanOrEqual(60);
      expect(script(n), `lookback-${n}`).not.toMatch(/[=×*+/<>%]/);
    }
  });
});

describe('the lookback narration agrees with the model', () => {
  const S = SCENE.single;
  it('part 1: 6 blocks, 3 at a time', () => {
    expect([SCENE_G, SCENE_R]).toEqual([6, 3]);
    expect(script(1)).toContain(`Here are ${SCENE_G} blocks, and a device that can hold ${SCENE_R} of them at a time.`);
  });
  it('part 2: indices 0, 1 and 2 in start order; block 1 done before block 0, spins on its flag', () => {
    expect(S.dispatch.slice(0, 3).map((hw) => S.blocks[hw].idx)).toEqual([0, 1, 2]);
    const b0 = S.blocks.find((b) => b.idx === 0)!;
    const b1 = S.blocks.find((b) => b.idx === 1)!;
    const localEnd = (b: typeof b0) => b.spans.find((s) => s.phase === 'local')!.to;
    expect(localEnd(b1)).toBeLessThan(localEnd(b0));
    expect(b1.waitedOn).toEqual([0]);
    expect(script(2)).toContain('The blocks take indices 0, 1 and 2 in the order they start.');
    expect(script(2)).toContain("Block 1 finishes its segment before block 0 does, and spins on block 0's flag.");
  });
  it('part 3: 5 links for 6 blocks, 31 for 32 resident blocks', () => {
    expect(S.criticalPath).toBe(SCENE_G - 1);
    expect([SCENE_WIDE.config.G, SCENE_WIDE.config.R, SCENE_WIDE.criticalPath]).toEqual([32, 32, 31]);
    expect(script(3)).toContain(`With ${SCENE_G} blocks the chain has ${S.criticalPath} links.`);
    expect(script(3)).toContain(`With ${SCENE_WIDE.config.G} blocks all on the device at once, it has ${SCENE_WIDE.criticalPath},`);
  });
  it('part 4: blocks 3, 4 and 5 start first; 3 spins on 2, not started; deadlock', () => {
    const D = SCENE.deadlock;
    expect(D.dispatch.slice(0, 3)).toEqual([3, 4, 5]);
    expect(D.outcome).toBe('deadlock');
    expect(D.stuck[0]).toEqual({ hw: 3, idx: 3, on: 2, onStarted: false });
    expect(D.stuck.slice(1).every((s) => s.onStarted)).toBe(true);
    expect(script(4)).toContain('Suppose the scheduler starts blocks 3, 4 and 5 first. Block 3 spins on block 2, which has not started.');
  });
  it('part 5: waiting drops from 5 rounds to 1; 4 extra additions', () => {
    const D = SCENE.decoupled;
    expect([S.waitTotal, D.waitTotal, D.repeated]).toEqual([5, 1, 4]);
    expect(script(5)).toContain(`Here the waiting drops from ${S.waitTotal} rounds to ${D.waitTotal}, and the price is ${D.repeated} extra additions.`);
  });
  it('part 6: without a progress guarantee the same claimed-index launch stalls', () => {
    expect(SCENE.metal.outcome).toBe('stall');
    // Not a fixed number: lab 11.4 recurses on the block sums until they fit one segment.
    expect(script(6)).toContain('so on Metal the scan runs as separate dispatches, one after another.');
    expect(script(6)).not.toMatch(/\b3 dispatches|three dispatches/);
  });
});
