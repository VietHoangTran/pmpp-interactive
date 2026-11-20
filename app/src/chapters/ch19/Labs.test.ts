import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { LAB_LAYERS, layerFacts, llamaFor } from './Labs';

// The labs page lists the benchmark's five layers and computes their GEMM shapes and
// expansion ratios. This test ties those layers to lab 19.1's main.cpp and the computed
// figures to the lab README, so neither can drift from the page.
const repo = path.resolve(__dirname, '../../../..');
const lab = path.join(repo, 'labs/ch19/lab1_convgemm');
const main = fs.readFileSync(path.join(lab, 'main.cpp'), 'utf8');
const readme = fs.readFileSync(path.join(lab, 'README.md'), 'utf8');

describe('labs page, chapter 19', () => {
  it("the five layers are main.cpp's benchmark shapes, in its order", () => {
    const rows = [...main.matchAll(/\{"(\w+)", (\d+), (\d+), (\d+), (\d+), (\d+), (\d+)\}/g)].map((m) => ({
      id: m[1],
      shape: { N: +m[2], C: +m[3], H: +m[4], W: +m[5], M: +m[6], K: +m[7] },
    }));
    expect(rows).toEqual(LAB_LAYERS.map((L) => ({ id: L.id, shape: L.shape })));
  });

  it("the computed GEMM shapes and expansion ratios match the README's table", () => {
    for (const L of LAB_LAYERS) {
      const f = layerFacts(L.shape);
      const row = readme.split('\n').find((l) => l.startsWith(`| ${L.id} |`));
      if (L.id === 'pointwise') {
        expect(f.expansion).toBe(1);
        continue;
      }
      expect(row, L.id).toBeTruthy();
      const cells = row!.split('|').map((c) => c.trim());
      expect(cells[3]).toBe(f.gemm.replace(/ × /g, ' x '));
      expect(cells[4]).toBe(f.expansion.toFixed(2));
    }
  });

  it('the prediction readings use the right tile counts', () => {
    const late = layerFacts(LAB_LAYERS[2].shape);
    expect([late.tileRows, late.tileCols, late.phases]).toEqual([16, 9, 144]);
    const early = layerFacts(LAB_LAYERS[0].shape);
    expect([early.phases, early.lastPhase]).toEqual([2, 11]);
    expect(layerFacts(LAB_LAYERS[4].shape).launches).toBe(64);
  });

  it("no lab layer goes to llama.cpp's direct kernel by its shape rule", () => {
    expect(LAB_LAYERS.map((L) => llamaFor(L.shape))).not.toContain('direct');
  });
});
