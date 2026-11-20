import { describe, expect, it } from 'vitest';
import { PROJECTS, attentionHome } from './backendmap';
import {
  ATTENTION_HOME,
  DEFAULT_INPUT,
  KV_CACHE_STEP,
  OWN_EVERYWHERE,
  CUDA_HOME_OF,
  cudaHome,
  SDP_PRIORITY_ORDER,
  attnRoute,
  type AttnInput,
} from './attnroute';

const at = (over: Partial<AttnInput>): AttnInput => ({ ...DEFAULT_INPUT, ...over, use: { ...DEFAULT_INPUT.use, ...(over.use ?? {}) } });

describe('MLX on CUDA (q-decode-100)', () => {
  it('one query, no mask, 100 keys: its own sdpa_vector', () => {
    const r = attnRoute('mlx-cuda', at({ queries: 1, keys: 100, mask: 'none' }));
    expect(r.kernel.home).toBe('own');
    expect(r.kernel.name).toContain('sdpa_vector');
  });
  it('300 keys: cuDNN, if supports_sdpa_cudnn accepts', () => {
    expect(attnRoute('mlx-cuda', at({ queries: 1, keys: 300 })).kernel.home).toBe('library');
    expect(attnRoute('mlx-cuda', at({ queries: 1, keys: 300, cudnnSupports: false })).kernel.name).toContain('sdpa_vector');
  });
  it('the switch is mlx-lm cache step, 256 keys', () => {
    expect(KV_CACHE_STEP).toBe(256);
    expect(attnRoute('mlx-cuda', at({ keys: 255 })).kernel.home).toBe('own');
    expect(attnRoute('mlx-cuda', at({ keys: 256 })).kernel.home).toBe('library');
  });
  it('an array mask keeps a decode on its own kernel at any length', () => {
    expect(attnRoute('mlx-cuda', at({ keys: 8192, mask: 'array' })).kernel.home).toBe('own');
  });
  it('prefill goes to cuDNN when supported, whatever the length', () => {
    expect(attnRoute('mlx-cuda', at({ queries: 128, keys: 10 })).kernel.home).toBe('library');
    expect(attnRoute('mlx-cuda', at({ queries: 128, cudnnSupports: false })).kernel.home).toBe('unknown');
  });
});

describe('MLX on Metal (q-2pass)', () => {
  it('GQA, one query, 5000 keys, device class neither d nor s: the 2-pass kernel', () => {
    const r = attnRoute('mlx-metal', at({ queries: 1, keys: 5000, gqa: true, deviceDS: false }));
    expect(r.kernel.name).toContain('2pass');
  });
  it('without GQA the same shape is one pass; on a d or s device from 1024 keys it is two', () => {
    expect(attnRoute('mlx-metal', at({ keys: 5000, gqa: false })).kernel.name).not.toContain('2pass');
    expect(attnRoute('mlx-metal', at({ keys: 1024, deviceDS: true })).kernel.name).toContain('2pass');
    expect(attnRoute('mlx-metal', at({ keys: 1023, deviceDS: true })).kernel.name).not.toContain('2pass');
    expect(attnRoute('mlx-metal', at({ keys: 4095, gqa: true })).kernel.name).not.toContain('2pass');
  });
  it('8 queries are still vector kernels, 9 go to steel attention', () => {
    expect(attnRoute('mlx-metal', at({ queries: 8 })).kernel.name).toContain('sdpa_vector');
    expect(attnRoute('mlx-metal', at({ queries: 9 })).kernel.name).toContain('steel');
  });
});

describe('PyTorch on CUDA (q-order)', () => {
  it('default order: flash, efficient, math, cuDNN', () => {
    expect(SDP_PRIORITY_ORDER.slice(0, 4)).toEqual(['flash', 'efficient', 'math', 'cudnn']);
    expect(attnRoute('torch-cuda', at({})).order!.slice(0, 4)).toEqual(['flash', 'efficient', 'math', 'cudnn']);
  });
  it('a shape flash and efficient reject falls to math before cuDNN', () => {
    const r = attnRoute('torch-cuda', at({ use: { flash: false, efficient: false, math: true, cudnn: true, overrideable: false } }));
    expect(r.kernel.name).toContain('math');
  });
  it('cuDNN first when preferred', () => {
    const r = attnRoute('torch-cuda', at({ preferCudnn: true }));
    expect(r.order![0]).toBe('cudnn');
    expect(r.kernel.home).toBe('library');
  });
  it('flash lives in a submodule, not in the tree', () => {
    const r = attnRoute('torch-cuda', at({}));
    expect(r.kernel.home).toBe('submodule');
    expect(r.kernel.where).toContain('third_party/flash-attention');
  });
});

describe('where attention lives (the start page prediction)', () => {
  it('only llama.cpp runs its own kernels on every GPU backend it supports', () => {
    expect(OWN_EVERYWHERE).toEqual(['llama.cpp']);
  });
  it('llama.cpp is own on every input; MLX on Metal is own for every shape', () => {
    for (const queries of [1, 4, 64]) for (const keys of [10, 300, 5000]) {
      expect(attnRoute('llama', at({ queries, keys })).kernel.home).toBe('own');
      expect(attnRoute('mlx-metal', at({ queries, keys })).kernel.home).toBe('own');
    }
    expect(ATTENTION_HOME.filter((r) => r.allOwn).map((r) => `${r.project} ${r.backend}`)).toEqual([
      'llama.cpp CUDA and HIP', 'llama.cpp Metal', 'MLX Metal', 'PyTorch MPS',
    ]);
  });
});

describe("the start page's prediction (x3/start/p-attention-home): CUDA attention", () => {
  it('llama.cpp its own, MLX mixed, PyTorch a submodule first', () => {
    expect([cudaHome('llama'), cudaHome('mlx-cuda'), cudaHome('torch-cuda')]).toEqual(['own', 'mixed', 'submodule']);
  });
  it("agrees with the start page's own answer key (sim/backendmap.ts)", () => {
    for (const p of PROJECTS) expect(cudaHome(CUDA_HOME_OF[p])).toBe(attentionHome(p));
  });
});
