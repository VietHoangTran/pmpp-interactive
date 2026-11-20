import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { IMPLICIT_SCENE, PRINTED, CORRECTED, huntShape, phasesOf, runKernel, warpView } from '../../sim/implicitgemm';
import { PRESETS, UNFOLD_SCENE, bToX, expansion, preset, sizes, xToB } from '../../sim/unfold';
import { PROVIDED, REGIONS, SOLUTION_FILES, questionsS193, region } from './S193';

// Section 19.3's contracts with the files around it:
//  - the provided Fig. 19.11 it shows exists in CUDA and Metal and is the corrected
//    kernel the lesson describes (ceiling phases, three bounds checks, the derived launch);
//  - the reference solution carries the [lesson:unroll] and [lesson:gemm-path] pieces the
//    lesson cuts out, and the GEMM call is the one q-roles teaches;
//  - the lab benchmarks the layers the explorers call "the lab's";
//  - the narration says the models' numbers, for the ear;
//  - every numeric answer is the model's number.
const repo = path.resolve(__dirname, '../../../..');
const lab = (rel: string) => fs.readFileSync(path.join(repo, 'labs', rel), 'utf8');

describe('the provided Fig. 19.11 (labs/ch19/lab1_convgemm/provided)', () => {
  for (const file of [PROVIDED.cu, PROVIDED.metal]) {
    it(`${file} exists and is the corrected kernel: ceiling phases, zero fill on both loads, a guarded store`, () => {
      const src = lab(file);
      expect(src).toContain('for (int ph = 0; ph < (CKK + TILE_WIDTH - 1) / TILE_WIDTH; ++ph)');
      expect(src).toContain('(Row < M && fc < CKK) ? F[Row * CKK + fc] : 0.0f');
      expect(src).toContain('if (u < CKK && v < P)');
      expect(src).toContain('if (Row < M && Col < P)');
      expect(src).toMatch(/H_out = H - K \+ 1, W_out = W - K \+ 1/);
      // The gather is line 28 with W_out named.
      expect(src.replace(/\s+/g, ' ')).toContain('X[bz * (C * H * W) + (u / (K * K)) * (H * W) + ((u % (K * K)) / K + v / W_out) * W + ((u % (K * K)) % K + v % W_out)]');
    });
  }
  it('the CUDA launch is the one the lesson derives: x pixel tiles, y map tiles, z samples', () => {
    const s = lab(PROVIDED.cu).replace(/\s+/g, ' ');
    expect(s).toContain('dim3 dimGrid((P + TILE_WIDTH - 1) / TILE_WIDTH, // x');
    expect(s).toContain('(M + TILE_WIDTH - 1) / TILE_WIDTH, // y');
    expect(s).toMatch(/N\); \/\/ z: the sample/);
  });
});

describe('the reference solution pieces shown beside q-roles', () => {
  const sol = Object.fromEntries(SOLUTION_FILES.map((f) => [f, lab(`ch19/lab1_convgemm/solution/${f}`)]));
  it('unroll is marked in kernel.cu and kernel.metal; gemm-path in kernel.cu and launch.mm', () => {
    const found = REGIONS.flatMap((r) => SOLUTION_FILES.filter((f) => region(sol[f], r)).map((f) => `${r}:${f}`));
    expect(found).toEqual(['unroll:kernel.cu', 'unroll:kernel.metal', 'gemm-path:kernel.cu', 'gemm-path:launch.mm']);
  });
  it('the slices start on the line after the marker and keep the file\'s numbering', () => {
    const r = region(sol['kernel.cu'], 'unroll')!;
    expect(sol['kernel.cu'].split('\n')[r.firstLine - 2].trim()).toBe('// [lesson:unroll]');
    expect(r.code).toContain('__global__ void unrollKernel');
  });
  it('gemm-path passes X = F, W = Bᵀ, batch = M, in = C·K·K, out = P (q-roles\' answer)', () => {
    expect(region(sol['kernel.cu'], 'gemm-path')!.code).toContain('launchLinear(F, Bt, Y + (size_t)n * M * P, M, C * K * K, P);');
    expect(region(sol['launch.mm'], 'gemm-path')!.code).toMatch(/encodeLinearAt\(enc, linearPipeline, F, 0, Bt, 0, Y, \(NSUInteger\)n \* M \* P \* sizeof\(float\),\s*M, C \* K \* K, P, tileWidth\);/);
  });
  it('the unroll writes Bᵀ: element (v, u) at v·C·K·K + u, gathered with Eq. 19.5', () => {
    const code = region(sol['kernel.cu'], 'unroll')!.code;
    expect(code).toContain('int v = i / CKK, u = i % CKK;');
    expect(code).toContain('Bt[i] = Xn[(c * H + h) * W + w];');
  });
  it('a missing marker gives null, not a wrong slice', () => {
    expect(region('// [lesson:unroll]\nint x;\n', 'unroll')).toBeNull();
    expect(region(null, 'unroll')).toBeNull();
    expect(region('// could not load labs/x', 'unroll')).toBeNull();
  });
});

describe("the lab's layers are the explorers' layers", () => {
  it('main.cpp benchmarks early, middle, late and pointwise at the presets\' sizes', () => {
    const main = lab('ch19/lab1_convgemm/main.cpp');
    for (const id of ['early', 'middle', 'late', 'pointwise'] as const) {
      const m = new RegExp(`\\{"${id}", (\\d+), (\\d+), (\\d+), (\\d+), (\\d+), (\\d+)\\}`).exec(main);
      expect(m, id).not.toBeNull();
      const [N, C, H, W, M, K] = m!.slice(1).map(Number);
      expect(preset(id).shape, id).toEqual({ N, C, H, W, M, K });
    }
    expect(PRESETS.map((p) => p.id)).toEqual(['book', 'c1', 'c3', 'c5', 'early', 'middle', 'late', 'pointwise']);
  });
});

describe('the narration is written for the ear and says the models\' numbers', () => {
  const script = (name: string) => fs.readFileSync(path.join(repo, `content/ch19/narration/${name}.txt`), 'utf8').trim();
  const words = (s: string) => s.replace(/\[[^\]]*\]/g, '').split(/\s+/).filter(Boolean).length;
  const all = [1, 2, 3, 4, 5, 6].flatMap((n) => [`unfold-${n}`, `implicit-tiles-${n}`]);
  it('twelve scripts of 35 to 60 words, with no symbols read aloud', () => {
    for (const name of all) {
      expect(words(script(name)), name).toBeGreaterThanOrEqual(35);
      expect(words(script(name)), name).toBeLessThanOrEqual(60);
      expect(script(name), name).not.toMatch(/[=×*+/<>%_·²]/);
    }
  });
  it('unfold: 7, 5 and 2 make 14; 48 entries from 27 values; LeNet C1 at about 19 of 25', () => {
    expect(UNFOLD_SCENE.perChannel).toEqual([7, 5, 2]);
    expect(script('unfold-1')).toContain('7, then 5, then 2. Together they give 14.');
    expect(script('unfold-4')).toContain(`B ends up with ${UNFOLD_SCENE.entries} entries made from ${UNFOLD_SCENE.inputs} values.`);
    expect(script('unfold-4')).toContain('The centre pixel is copied 4 times, each edge pixel twice, each corner once.');
    expect(Math.round(UNFOLD_SCENE.c1Ratio)).toBe(19);
    expect(script('unfold-6')).toContain('25 for the first layer of LeNet, which is already at about 19.');
    expect(script('unfold-5')).toContain('12 products, and the sum is 14');
    expect(UNFOLD_SCENE.Y[0]).toBe(14);
  });
  it('implicit tiles: the block owns 14, 20, 12, 24; the toy counts are 96, 48 and 48', () => {
    expect(IMPLICIT_SCENE.finals).toEqual([14, 20, 12, 24]);
    expect(script('implicit-tiles-1')).toContain('14 and 20 from map 0, 12 and 24 from map 1');
    expect(script('implicit-tiles-5')).toContain('they read 14, 20, 12 and 24');
    const t = IMPLICIT_SCENE.traffic;
    expect([t.direct.x, t.explicit.writesB, t.explicit.gemmReadsB, t.implicit.gathersX]).toEqual([96, 48, 48, 48]);
    expect(script('implicit-tiles-6')).toContain('The direct kernel read 96 input values. Explicit unrolling read 48 to write the 48 entries of B, then read them back. Implicit unrolling gathered 48');
    const g = IMPLICIT_SCENE.block.phases[0].b;
    expect(new Set(g.map((x) => x.idx)).size).toBe(3);
    expect(script('implicit-tiles-2')).toContain('3 input pixels fill 4 slots, and 1 of them lands in 2 slots.');
    expect(IMPLICIT_SCENE.block.phases.length).toBe(6);
    expect(script('implicit-tiles-5')).toContain('Phases 2 and 3 gather from the second input map, and phases 4 and 5 from the third.');
  });
});

describe('every numeric answer is the model\'s number', () => {
  const num = (id: string) => {
    const q = questionsS193[id];
    if (q.kind !== 'numeric') throw new Error(id + ' is not numeric');
    return q.answer;
  };
  it('the question ids are the designed ones', () => {
    expect(Object.keys(questionsS193).sort()).toEqual(
      ['q-belems', 'q-noM', 'q-map', 'q-copies', 'q-phases', 'q-loads', 'q-roles', 'q-wholebatch', 'q-thin', 'q-vit', 'q-divmod', 'q-ex4']
        .map((s) => 'ch19/19.3/' + s)
        .sort(),
    );
    for (const [id, q] of Object.entries(questionsS193)) {
      expect(q.id).toBe(id);
      if (q.kind === 'choice') expect(q.options.filter((o) => o.correct).length, id).toBe(1);
    }
  });
  it('q-belems 15000, q-map 27, q-copies 4', () => {
    expect(num('ch19/19.3/q-belems')).toBe(sizes(preset('c3').shape).bSample);
    const s = { C: 3, H: 4, W: 4, K: 2, M: 1, N: 1 };
    expect(num('ch19/19.3/q-map')).toBe(bToX(s, 7, 5).flat);
    expect(num('ch19/19.3/q-copies')).toBe(xToB(s, 1, 1, 2).length);
  });
  it('q-phases 6, q-loads 20', () => {
    const c3 = huntShape('c3').shape;
    const r = runKernel(c3, PRINTED);
    expect(num('ch19/19.3/q-phases')).toBe(r.terms - r.termsUsed);
    expect(num('ch19/19.3/q-loads')).toBe(2 * phasesOf(c3, CORRECTED));
  });
  it('q-vit 1, q-ex4 2', () => {
    expect(num('ch19/19.3/q-vit')).toBe(expansion({ C: 3, H: 224, W: 224, K: 16, M: 1, N: 1, stride: 16 }));
    expect(num('ch19/19.3/q-ex4')).toBe(warpView(huntShape('c3').shape, 0, 0, 0, 0).xRunsByTy[0].runs.length);
  });
});
