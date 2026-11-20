import { describe, expect, it } from 'vitest';
import {
  FEATURES, TARGETS, cell, cuda13Targets, hasLdgsts, hasTcgen05, hasTma, hasWgmma, hasWmma, summarize, target,
} from './gemmhw';

const status = (id: string, f: Parameters<typeof cell>[1]) => cell(target(id), f).status;

describe('CUDA rules from ch15-facts', () => {
  it('WGMMA only when built for sm_90a (A6)', () => {
    expect(hasWgmma(9.0, true)).toBe(true);
    expect(hasWgmma(9.0, false)).toBe(false);
    expect(hasWgmma(10.0, true)).toBe(false);
    expect(status('cuda-9.0a', 'bigMma')).toBe('yes');
    expect(status('cuda-9.0', 'bigMma')).toBe('no');
    expect(cell(target('cuda-9.0a'), 'bigMma').what).toContain('cannot run on later generations');
  });
  it('tcgen05 with TMEM on the sm_100 and sm_110 families only, not on 12.x (A6)', () => {
    expect(hasTcgen05(10.0)).toBe(true);
    expect(hasTcgen05(10.3)).toBe(true);
    expect(hasTcgen05(11.0)).toBe(true);
    expect(hasTcgen05(12.0)).toBe(false);
    expect(hasTcgen05(9.0)).toBe(false);
    expect(status('cuda-10.0', 'bigMma')).toBe('yes');
    expect(cell(target('cuda-10.0'), 'bigMma').what).toContain('TMEM');
    expect(status('cuda-12', 'bigMma')).toBe('no');
    expect(cell(target('cuda-12'), 'bigMma').what).toContain('No TMEM');
  });
  it('LDGSTS from 8.0 (A7), TMA from 9.0 (A8)', () => {
    expect([7.5, 8.0, 8.6, 9.0].map(hasLdgsts)).toEqual([false, true, true, true]);
    expect([8.0, 8.9, 9.0, 12.0].map(hasTma)).toEqual([false, false, true, true]);
    expect(status('cuda-7.5', 'asyncCopy')).toBe('no');
    expect(status('cuda-8.6', 'asyncCopy')).toBe('yes');
    expect(status('cuda-8.6', 'bulkCopy')).toBe('no');
    expect(status('cuda-9.0', 'bulkCopy')).toBe('yes');
    expect(cell(target('cuda-9.0'), 'bulkCopy').what).toContain('cuTensorMapEncodeTiled');
    expect(cell(target('cuda-8.0'), 'asyncCopy').what).toContain('does not guarantee');
  });
  it('CUDA 13 targets from 7.5; WMMA from 7.0 and still present on every later target (A5, A6)', () => {
    expect(cuda13Targets(7.0)).toBe(false);
    expect(cuda13Targets(7.5)).toBe(true);
    expect(summarize('cuda-7.0').toolkit?.status).toBe('no');
    expect(summarize('cuda-7.5').toolkit?.status).toBe('yes');
    expect(hasWmma(7.0)).toBe(true);
    for (const t of TARGETS.filter((x) => x.backend === 'CUDA')) expect(cell(t, 'warpMma').status).toBe('yes');
  });
  it('the cuBLAS baseline is FP32 by default; BF16x9 is mentioned only on 10.0 (A10)', () => {
    expect(cell(target('cuda-10.0'), 'vendorGemm').what).toContain('BF16x9');
    expect(cell(target('cuda-12'), 'vendorGemm').what).not.toContain('BF16x9');
    expect(cell(target('cuda-8.0'), 'vendorGemm').what).toContain('keeps FP32');
  });
});

describe('HIP rules from ch15-facts', () => {
  it('MFMA on CDNA, with no xf32 on gfx950 (B4)', () => {
    for (const id of ['hip-cdna2', 'hip-cdna3', 'hip-cdna4']) expect(cell(target(id), 'warpMma').what).toContain('MFMA builtins');
    expect(cell(target('hip-cdna3'), 'warpMma').what).toContain('xf32 form exists');
    expect(cell(target('hip-cdna4'), 'warpMma').what).toContain('does not support xf32');
  });
  it('WMMA builtins on RDNA3 and RDNA4, and no MFMA there (B4)', () => {
    expect(cell(target('hip-rdna3'), 'warpMma').what).toContain('wmma');
    expect(cell(target('hip-rdna4'), 'warpMma').what).toContain('gfx12');
    expect(cell(target('hip-rdna3'), 'warpMma').what).toContain('No MFMA');
    expect(status('hip-rdna3', 'warpMma')).toBe('yes');
  });
  it('no documented asynchronous copy or bulk copy anywhere on HIP (B5)', () => {
    for (const t of TARGETS.filter((x) => x.backend === 'HIP')) {
      expect(cell(t, 'asyncCopy').status).toBe('undocumented');
      expect(cell(t, 'bulkCopy').status).toBe('undocumented');
    }
    expect(cell(target('hip-cdna3'), 'asyncCopy').what).toContain('global_load_lds');
    expect(cell(target('hip-rdna3'), 'asyncCopy').what).not.toContain('exists in the compiler');
  });
});

describe('Metal rules from ch15-facts', () => {
  it('simdgroup_matrix on Apple7 and later, with or without Metal 4 (C5)', () => {
    expect(cell(target('metal-apple7'), 'warpMma').what).toContain('simdgroup_float8x8');
    expect(cell(target('metal-apple7-m4'), 'warpMma').status).toBe('yes');
  });
  it('MPP matmul2d only with Metal 4 (C5)', () => {
    expect(status('metal-apple7-m4', 'bigMma')).toBe('yes');
    expect(status('metal-apple7', 'bigMma')).toBe('no');
    expect(cell(target('metal-apple7-m4'), 'bigMma').what).toContain('float x float -> float');
  });
  it('no documented asynchronous copy on Metal (ch06-facts C7)', () => {
    expect(status('metal-apple7', 'asyncCopy')).toBe('undocumented');
    expect(status('metal-apple7-m4', 'asyncCopy')).toBe('undocumented');
  });
  it('never calls Apple\'s matrix interfaces tensor cores', () => {
    for (const t of TARGETS.filter((x) => x.backend === 'Metal'))
      for (const f of FEATURES) expect(cell(t, f.id).what.replace('not "tensor cores"', '')).not.toMatch(/tensor core/i);
  });
});

describe('the table is complete', () => {
  it('every target has every feature, each with a facts item', () => {
    for (const t of TARGETS) {
      const s = summarize(t.id);
      for (const f of FEATURES) {
        expect(s.cells[f.id].what.length).toBeGreaterThan(10);
        expect(s.cells[f.id].items.length).toBeGreaterThan(0);
      }
    }
  });
  it('unknown targets are rejected', () => expect(() => target('cuda-6.0')).toThrow());
});
