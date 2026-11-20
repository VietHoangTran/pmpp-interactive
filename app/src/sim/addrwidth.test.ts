import { describe, expect, it } from 'vitest';
import {
  ELEMENTS,
  GiB,
  MARKS,
  PiB,
  TiB,
  arrayLimit,
  binary,
  coverage,
  memoryFigures,
  reach,
  smallestWidth,
  times,
} from './addrwidth';

describe('reach: bytes an address width can name', () => {
  it('32 bits reach 4 GiB, the cap the book gives for early GPUs (pp. 625–626)', () => {
    expect(reach(32)).toBe(4 * GiB);
    expect(binary(reach(32)).text).toBe('4 GiB');
  });
  it("40 bits reach 1 TiB, Fermi's unified space: 'a Terabyte' (appC-facts A6)", () => {
    expect(reach(40)).toBe(TiB);
    expect(binary(reach(40)).text).toBe('1 TiB');
  });
  it('48 bits reach 256 TiB, the 4-level paging limit (A7)', () => {
    expect(binary(reach(48)).text).toBe('256 TiB');
  });
  it('49 bits reach 512 TiB, and 57 bits reach 128 PiB (A7)', () => {
    expect(binary(reach(49)).text).toBe('512 TiB');
    expect(reach(57)).toBe(128 * PiB);
    expect(binary(reach(57)).text).toBe('128 PiB');
  });
  it('64 bits reach 16 EiB, still exact as a double', () => {
    expect(binary(reach(64)).text).toBe('16 EiB');
    expect(reach(64)).toBe(18446744073709551616);
  });
  it('each extra bit doubles the reach', () => {
    for (let b = 16; b < 64; b++) expect(reach(b + 1)).toBe(2 * reach(b));
  });
  it("every labelled mark's label is its reach", () => {
    for (const m of MARKS) expect(binary(reach(m.bits)).text).toBe(m.label);
  });
  it('binary() keeps three significant digits and handles small values', () => {
    expect(binary(1536).text).toBe('1.5 KiB');
    expect(binary(1000).text).toBe('1000 B');
    expect(binary(0).text).toBe('0 B');
    expect(binary(80 * GiB).text).toBe('80 GiB');
    expect(binary(1023.9 * 1024).text).toBe('1020 KiB');
  });
});

describe('arrayLimit: the largest array a kernel can index', () => {
  it('an int index over floats stops at 2^31 elements, 8 GiB (the C.4 question)', () => {
    const l = arrayLimit('int32', 4);
    expect(l.elements).toBe(2 ** 31);
    expect(l.bytes).toBe(8 * GiB);
    expect(l.bytes / GiB).toBeCloseTo(8, 10);
    expect(l.binds).toBe('index');
  });
  it('the same int over 2-byte, 1-byte and 4-bit elements: 4 GiB, 2 GiB, 1 GiB', () => {
    expect(arrayLimit('int32', 2).bytes).toBe(4 * GiB);
    expect(arrayLimit('int32', 1).bytes).toBe(2 * GiB);
    expect(arrayLimit('int32', 0.5).bytes).toBe(GiB);
  });
  it('unsigned int doubles it: 16 GiB of floats', () => {
    expect(arrayLimit('uint32', 4).bytes).toBe(16 * GiB);
  });
  it('a 64-bit index over floats outruns a 64-bit address: the address binds', () => {
    const l = arrayLimit('int64', 4);
    expect(l.indexBytes).toBe(2 ** 65);
    expect(l.binds).toBe('address');
    expect(l.bytes).toBe(reach(64));
  });
  it('with 32-bit addresses even an int index over floats is capped at 4 GiB by the address', () => {
    const l = arrayLimit('int32', 4, 32);
    expect(l.binds).toBe('address');
    expect(l.bytes).toBe(4 * GiB);
  });
  it('offers the four element sizes the explorer lists', () => {
    expect(ELEMENTS.map((e) => e.bytes)).toEqual([4, 2, 1, 0.5]);
  });
  it('times() compares a device size to the limit', () => {
    expect(times(24 * GiB, 8 * GiB)).toBe(3);
    expect(Number.isNaN(times(1, 0))).toBe(true);
  });
});

describe("memoryFigures: the reader's devices, from reports/runs/*.json", () => {
  it('reads totalGlobalMem from CUDA or HIP and the two Metal figures, skipping emulators and zeros', () => {
    const f = memoryFigures({
      emu: { device: { name: 'emulator', totalGlobalMem: '0' } },
      race: { device: { name: 'race', totalGlobalMem: '123' } },
      cuda: { device: { name: 'Some GPU', totalGlobalMem: String(24 * GiB) } },
      metal: { device: { name: 'Some Mac', maxBufferLength: '22613000192', recommendedMaxWorkingSetSize: '30150672384', physicalMemory: '1' } },
    });
    expect(f.map((x) => [x.backend, x.field, x.bytes])).toEqual([
      ['cuda', 'totalGlobalMem', 24 * GiB],
      ['metal', 'maxBufferLength', 22613000192],
      ['metal', 'recommendedMaxWorkingSetSize', 30150672384],
    ]);
    expect(f[0].name).toBe('Some GPU');
  });
  it('is empty with no reports', () => {
    expect(memoryFigures({})).toEqual([]);
  });
});

describe('coverage and smallestWidth: does a CPU space plus GPU memory fit (§C.7)', () => {
  it('two disjoint ranges of 2^48 bytes need 49 bits: the C.7 question', () => {
    expect(smallestWidth(48, 2 ** 48)).toBe(49);
    expect(reach(48) + reach(48)).toBe(reach(49));
  });
  it('49 is the smallest width for a 48-bit CPU space plus any GPU memory up to 2^48 bytes', () => {
    expect(smallestWidth(48, 1)).toBe(49);
    expect(smallestWidth(48, 80 * GiB)).toBe(49);
    expect(smallestWidth(48, 2 ** 48 + 1)).toBe(50);
    expect(smallestWidth(48, 0)).toBe(48);
  });
  it("Linux's default 47-bit user space plus GPU memory fits in 48 bits", () => {
    expect(smallestWidth(47, 80 * GiB)).toBe(48);
    expect(coverage(48, 47, 80 * GiB).fits).toBe(true);
  });
  it('48 bits cannot hold a 48-bit CPU space and any GPU memory', () => {
    const c = coverage(48, 48, GiB);
    expect(c.fits).toBe(false);
    expect(c.over).toBe(GiB);
    expect(c.cpuOutside).toBe(0);
    expect(c.gpuOutside).toBe(GiB);
  });
  it('49 bits hold it with room to spare', () => {
    const c = coverage(49, 48, 80 * GiB);
    expect(c.fits).toBe(true);
    expect(c.spare).toBe(reach(49) - reach(48) - 80 * GiB);
  });
  it('32 bits cannot even hold the CPU range: all of the GPU range and most of the CPU range fall outside', () => {
    const c = coverage(32, 48, 16 * GiB);
    expect(c.fits).toBe(false);
    expect(c.cpuOutside).toBe(reach(48) - reach(32));
    expect(c.gpuOutside).toBe(16 * GiB);
  });
  it('56-bit user space with 5-level paging does not fit in 49 bits', () => {
    expect(coverage(49, 56, GiB).fits).toBe(false);
    expect(smallestWidth(56, GiB)).toBe(57);
  });
});
