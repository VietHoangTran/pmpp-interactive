import { describe, expect, it } from 'vitest';
import { kneeOf } from './batching';
import {
  CODE_SWITCHES,
  DEFAULT_SHAPE,
  MMVF_MAX_BATCH_SIZE,
  MMVQ_MAX_BATCH_SIZE,
  PROJECTS,
  ROUTES,
  SHAPE_RULE_READ,
  canUseGemv,
  readerKnees,
  route,
  type Shape,
} from './matmulroute';

const at = (over: Partial<Shape>): Shape => ({ ...DEFAULT_SHAPE, ...over });

describe('one token', () => {
  it('takes a GEMV path first in every project whose shape rule the facts read, for every weight type', () => {
    for (const p of SHAPE_RULE_READ)
      for (const weight of ['f32', 'f16', 'quant', 'kquant'] as const) {
        const r = route(p, at({ M: 1, weight }));
        expect(r.kind, `${p} ${weight}`).toBe('gemv');
      }
    expect(route('llama-cuda', at({ M: 1, weight: 'quant' })).candidate).toBe('mmvq');
  });
  it('PyTorch CUDA hands it to a library whose choice is not readable', () => {
    const r = route('torch-cuda', at({ M: 1 }));
    expect(r.kind).toBe('library');
    expect(r.certain).toBe(true);
  });
  it('llama.cpp CUDA, F32 activations, F16 weights (q-one-token): mmvf first, but not certain', () => {
    const r = route('llama-cuda', at({ M: 1, weight: 'f16', inputF32: true }));
    expect(r.candidate).toBe('mmvf');
    expect(r.certain).toBe(false);
    expect(r.alternatives).toContain('cublas');
  });
  it('a non-F32 input goes straight to cuBLAS', () => {
    const r = route('llama-cuda', at({ M: 1, inputF32: false }));
    expect(r.candidate).toBe('cublas');
    expect(r.certain).toBe(true);
  });
  it('llama.cpp Metal: one token is mul_mv, certainly', () => {
    const r = route('llama-metal', at({ M: 1 }));
    expect([r.candidate, r.certain]).toEqual(['mul_mv', true]);
  });
});

describe('the batch thresholds', () => {
  it('the constants are 8 and 8 (ch22 D6)', () => {
    expect(MMVF_MAX_BATCH_SIZE).toBe(8);
    expect(MMVQ_MAX_BATCH_SIZE).toBe(8);
    expect(CODE_SWITCHES.map((c) => c.tokens)).toEqual([8, 5]);
  });
  it('9 tokens rule out mmvf and mmvq (q-nine); mmf, mmq or cuBLAS remain', () => {
    for (const weight of ['f16', 'quant'] as const) {
      const r = route('llama-cuda', at({ M: 9, weight }));
      expect(r.checks.find((c) => c.route === 'mmvf')!.verdict).toBe('false');
      expect(r.checks.find((c) => c.route === 'mmvq')!.verdict).toBe('false');
      expect(r.alternatives).toContain('cublas');
    }
    expect(route('llama-cuda', at({ M: 9, weight: 'f16' })).candidate).toBe('mmf');
    expect(route('llama-cuda', at({ M: 9, weight: 'quant' })).candidate).toBe('mmq');
  });
  it('8 tokens keep mmvf a candidate', () => {
    expect(route('llama-cuda', at({ M: 8 })).candidate).toBe('mmvf');
  });
  it('a quantised weight at 4 tokens: mmvq is the first candidate, mmq next', () => {
    const r = route('llama-cuda', at({ M: 4, weight: 'quant' }));
    expect(r.checks.find((c) => c.route === 'mmvf')!.verdict).toBe('false');
    expect(r.checks.find((c) => c.route === 'mmf')!.verdict).toBe('false');
    expect(r.candidate).toBe('mmvq');
    expect(r.alternatives[0]).toBe('mmq');
  });
  it('llama.cpp Metal: 2 to 8 tokens may use mul_mv_ext, 4 to 8 for k-quants', () => {
    expect(route('llama-metal', at({ M: 2, weight: 'f16' })).candidate).toBe('mul_mv_ext');
    expect(route('llama-metal', at({ M: 3, weight: 'kquant' })).candidate).toBe('mul_mm');
    expect(route('llama-metal', at({ M: 4, weight: 'kquant' })).candidate).toBe('mul_mv_ext');
    expect(route('llama-metal', at({ M: 9 })).candidate).toBe('mul_mm');
  });
});

describe('MLX', () => {
  it('can_use_gemv as quoted from gemv.cu line 188', () => {
    expect(canUseGemv(1, 4096, false, true)).toBe(true);
    expect(canUseGemv(1, 4096, false, false)).toBe(false);
    expect(canUseGemv(7, 1, false, false)).toBe(true);
    expect(canUseGemv(7, 1, true, false)).toBe(false);
  });
  it('CUDA, x @ W.T with one row (q-mlx-transposed): GEMV, certainly', () => {
    const r = route('mlx-cuda', at({ M: 1, bTransposed: true }));
    expect([r.candidate, r.certain]).toEqual(['mlx-gemv', true]);
  });
  it('CUDA: a bias sends even one token to cuBLASLt', () => {
    expect(route('mlx-cuda', at({ M: 1, bias: true })).candidate).toBe('cublaslt');
    expect(route('mlx-cuda', at({ M: 2 })).candidate).toBe('cublaslt');
  });
  it('Metal: 4 vectors route to the wide GEMV, 6 to steel', () => {
    const four = route('mlx-metal', at({ M: 4 }));
    expect(four.candidate).toBe('wide');
    expect(four.alternatives).toEqual(['steel']);
    expect(route('mlx-metal', at({ M: 6 })).candidate).toBe('steel');
  });
  it('Metal: 1 vector is the wide GEMV or gemv, both GEMV kernels', () => {
    const r = route('mlx-metal', at({ M: 1 }));
    expect(r.candidate).toBe('wide');
    expect(r.alternatives[0]).toBe('gemv');
    expect(ROUTES.gemv.kind).toBe('gemv');
  });
});

describe('PyTorch MPS', () => {
  it('rank-1 shapes go to the GEMV kernels', () => {
    expect(route('torch-mps', at({ M: 1 })).candidate).toBe('mps-gemv');
    expect(route('torch-mps', at({ M: 2 })).candidate).toBe('mps-matmul');
  });
});

describe('what is never evaluated', () => {
  it('mmf, mmq, FP8 and use_mm are never decided by tokens: not modelled at every count for the weights their names fit', () => {
    for (let M = 1; M <= 512; M *= 2) {
      for (const weight of ['f32', 'f16', 'quant', 'kquant'] as const) {
        const c = route('llama-cuda', at({ M, weight })).checks;
        const q = weight === 'quant' || weight === 'kquant';
        expect(c.find((x) => x.route === 'mmf')!.verdict).toBe(q ? 'false' : 'not-modelled');
        expect(c.find((x) => x.route === 'mmq')!.verdict).toBe(q ? 'not-modelled' : 'false');
        expect(c.find((x) => x.route === 'fp8')!.verdict).toBe('not-modelled');
        expect(route('llama-metal', at({ M, weight })).checks.find((x) => x.route === 'mul_mm')!.verdict).toBe('not-modelled');
      }
    }
  });
  it('every project returns a route with a file', () => {
    for (const p of PROJECTS) for (const M of [1, 3, 9, 512]) expect(ROUTES[route(p, at({ M })).candidate].where).toMatch(/\//);
  });
});

describe("the reader's knee, chapter 22's definition", () => {
  const rows = (gemm: string, pts: [number, number][]) =>
    pts.map(([batch, seconds]) => ({ kind: 'bench', name: 'step', gemm, batch, seconds }));
  const reports = {
    metal: {
      device: { name: 'Test GPU' },
      results: [
        {
          lab: 'ch22/lab2_batching',
          variant: 'solution',
          bench: [
            ...rows('m3', [[1, 0.048], [8, 0.046], [128, 0.047], [256, 0.088]]),
            ...rows('m2', [[1, 0.0042], [16, 0.0043], [32, 0.0081]]),
            ...rows('gemv', [[1, 0.0027], [8, 0.0037]]),
          ],
        },
      ],
    },
    emu: { device: {}, results: [{ lab: 'ch22/lab2_batching', variant: 'solution', bench: rows('m3', [[1, 1], [2, 9]]) }] },
  };
  it('first batch at 1.5 times the smallest batch, per kernel; emulators ignored', () => {
    const k = readerKnees(reports);
    expect(k.map((x) => x.backend)).toEqual(['metal', 'metal', 'metal']);
    const by = (s: string) => k.find((x) => x.label.includes(s))!;
    expect(by('M3').knee).toBe(256);
    expect(by('M2').knee).toBe(32);
    expect(by('gather').knee).toBe(null);
    expect(k.map((x) => x.short)).toEqual(expect.arrayContaining(['M3', 'M2']));
    expect(by('M3').knee).toBe(kneeOf([{ batch: 1, seconds: 0.048 }, { batch: 256, seconds: 0.088 }]));
  });
  it('no reports: no knees', () => {
    expect(readerKnees({})).toEqual([]);
  });
});
