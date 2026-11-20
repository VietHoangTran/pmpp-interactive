import { describe, expect, it } from 'vitest';
import {
  BOOK_RULE,
  BOOK_TARGET,
  CUDA_DOCUMENTED,
  FIG31,
  FORMULAS,
  LAUNCH_ERROR,
  blockCount,
  blockFromLabel,
  blockLabel,
  checkLaunch,
  coverage,
  describeViolation,
  dim3,
  dimensionality,
  discriminates,
  drillTargets,
  formatBlockLabel,
  formatDim3,
  formatThreadLabel,
  globalCoord,
  inside,
  isDim3,
  limitsFromReport,
  linearThread,
  linearThreadWorking,
  misread,
  sameIndex,
  sizeGrid,
  sizeGridMinusOne,
  threadAt,
  threadFromLabel,
  threadLabel,
  threadsPerBlock,
  totalThreads,
  workingInBlock,
  type Dim3,
} from './grid';

describe('dim3 arithmetic', () => {
  it('leaves unused dimensions at 1, like the constructor', () => {
    expect(dim3(42)).toEqual({ x: 42, y: 1, z: 1 });
    expect(dim3(16, 32)).toEqual({ x: 16, y: 32, z: 1 });
    expect(dim3()).toEqual({ x: 1, y: 1, z: 1 });
  });
  it('Fig. 3.1 (p. 48): grid (2,2,1), block (4,2,2) is 4 blocks of 16 threads, 64 in all', () => {
    expect(blockCount(FIG31.grid)).toBe(4n);
    expect(threadsPerBlock(FIG31.block)).toBe(16n);
    expect(totalThreads(FIG31.grid, FIG31.block)).toBe(64n);
  });
  it('the 1D example on p. 46: 32 blocks of 128 threads is 4096 threads', () => {
    expect(totalThreads(dim3(32, 1, 1), dim3(128, 1, 1))).toBe(4096n);
    expect(totalThreads(dim3(32), dim3(128))).toBe(4096n);
  });
  it('a grid at its documented limits is counted exactly', () => {
    expect(blockCount(dim3(2 ** 31 - 1, 65535, 65535))).toBe(2147483647n * 65535n * 65535n);
  });
  it('grid and block need not have the same dimensionality (p. 47)', () => {
    expect(dimensionality(FIG31.grid)).toBe(2);
    expect(dimensionality(FIG31.block)).toBe(3);
    expect(dimensionality(dim3(1, 1, 1))).toBe(1);
  });
  it('a field is an unsigned 32-bit integer', () => {
    expect(isDim3(dim3(0xffffffff, 1, 1))).toBe(true);
    expect(isDim3(dim3(2 ** 32, 1, 1))).toBe(false);
    expect(isDim3(dim3(1.5, 1, 1))).toBe(false);
    expect(isDim3(dim3(-1, 1, 1))).toBe(false);
  });
});

describe('limits', () => {
  const one = dim3();
  const legal = (block: Dim3, l = CUDA_DOCUMENTED) => checkLaunch(one, block, l).legal;

  it("the book's four block shapes (p. 47) come out the same under the book's rule and under CUDA's documented limits", () => {
    for (const l of [BOOK_RULE, CUDA_DOCUMENTED]) {
      expect(legal(dim3(512, 1, 1), l)).toBe(true);
      expect(legal(dim3(8, 16, 4), l)).toBe(true);
      expect(legal(dim3(32, 16, 2), l)).toBe(true);
      expect(legal(dim3(32, 32, 2), l)).toBe(false);
    }
    expect(threadsPerBlock(dim3(512, 1, 1))).toBe(512n);
    expect(threadsPerBlock(dim3(8, 16, 4))).toBe(512n);
    expect(threadsPerBlock(dim3(32, 16, 2))).toBe(1024n);
    expect(threadsPerBlock(dim3(32, 32, 2))).toBe(2048n);
  });
  it('(32,32,2) breaks exactly the product clause', () => {
    const v = checkLaunch(one, dim3(32, 32, 2), CUDA_DOCUMENTED);
    expect(v.violations).toEqual([{ clause: 'block-threads', field: 'threads per block', value: 2048n, limit: 1024 }]);
    expect(describeViolation(v.violations[0])).toBe('A block would hold 2,048 threads. The limit is 1,024.');
  });
  it("(1,1,128) has 128 threads, is legal under the book's rule and illegal under CUDA's (z cap 64)", () => {
    const b = dim3(1, 1, 128);
    expect(threadsPerBlock(b)).toBe(128n);
    const book = checkLaunch(one, b, BOOK_RULE);
    expect(book.legal).toBe(true);
    expect(book.unchecked).toEqual(['block-dim']);
    const cuda = checkLaunch(one, b, CUDA_DOCUMENTED);
    expect(cuda.legal).toBe(false);
    expect(cuda.unchecked).toEqual([]);
    expect(cuda.violations).toEqual([{ clause: 'block-dim', field: 'blockDim.z', value: 128n, limit: 64 }]);
    expect(describeViolation(cuda.violations[0])).toBe('blockDim.z is 128. The cap for that dimension of a block is 64.');
  });
  it('the per-dimension caps are 1024, 1024 and 64, each exactly reachable', () => {
    expect(legal(dim3(1024, 1, 1))).toBe(true);
    expect(legal(dim3(1, 1024, 1))).toBe(true);
    expect(legal(dim3(1, 1, 64))).toBe(true);
    expect(legal(dim3(1, 1, 65))).toBe(false);
    expect(legal(dim3(16, 1, 64))).toBe(true);
    expect(legal(dim3(1025, 1, 1))).toBe(false);
    // 1025 in x breaks two clauses at once.
    expect(checkLaunch(one, dim3(1025, 1, 1), CUDA_DOCUMENTED).violations.map((v) => v.clause)).toEqual(['block-threads', 'block-dim']);
  });
  it('grid ranges (p. 47): x up to 2^31 - 1, y and z up to 65535', () => {
    for (const l of [BOOK_RULE, CUDA_DOCUMENTED]) {
      expect(checkLaunch(dim3(2 ** 31 - 1, 65535, 65535), one, l).legal).toBe(true);
      expect(checkLaunch(dim3(2 ** 31, 1, 1), one, l).violations).toEqual([
        { clause: 'grid-dim', field: 'gridDim.x', value: 2147483648n, limit: 2147483647 },
      ]);
      expect(checkLaunch(dim3(1, 65536, 1), one, l).legal).toBe(false);
      expect(checkLaunch(dim3(1, 1, 65536), one, l).violations[0].field).toBe('gridDim.z');
    }
    expect(describeViolation(checkLaunch(dim3(1, 65536, 1), one, CUDA_DOCUMENTED).violations[0])).toBe(
      'gridDim.y is 65,536. The cap for that dimension of a grid is 65,535.',
    );
  });
  it('a zero anywhere is a violation under every rule', () => {
    const v = checkLaunch(dim3(2, 0, 1), dim3(4, 2, 0), BOOK_RULE);
    expect(v.legal).toBe(false);
    expect(v.violations.map((x) => x.field)).toEqual(['gridDim.y', 'blockDim.z']);
    expect(describeViolation(v.violations[0])).toBe('gridDim.y is 0. Every dimension must be at least 1.');
  });
  it('the Fig. 3.1 launch and the 32 by 128 launch are legal', () => {
    expect(checkLaunch(FIG31.grid, FIG31.block, CUDA_DOCUMENTED)).toEqual({ legal: true, violations: [], unchecked: [] });
    expect(checkLaunch(dim3(32), dim3(128), CUDA_DOCUMENTED).legal).toBe(true);
  });
  it('both runtimes name the same error, with the value 9', () => {
    expect(LAUNCH_ERROR.cuda).toEqual({ name: 'cudaErrorInvalidConfiguration', value: 9, query: 'cudaGetLastError()' });
    expect(LAUNCH_ERROR.hip.name).toBe('hipErrorInvalidConfiguration');
    expect(LAUNCH_ERROR.hip.value).toBe(9);
  });
});

describe('limits read from a device report', () => {
  // The shape of reports/runs/<backend>.json: every value is a string.
  const cudaLike = {
    name: 'some device',
    maxThreadsPerBlock: '1024',
    'maxThreadsDim.x': '1024',
    'maxThreadsDim.y': '1024',
    'maxThreadsDim.z': '64',
    'maxGridSize.x': '2147483647',
    'maxGridSize.y': '65535',
    'maxGridSize.z': '65535',
  };
  it('reads the three CUDA and HIP property groups', () => {
    expect(limitsFromReport(cudaLike)).toEqual(CUDA_DOCUMENTED);
  });
  it('a record that caps z at 1024 accepts (1,1,128), as the course emulator does', () => {
    const emu = limitsFromReport({ ...cudaLike, 'maxThreadsDim.z': '1024', 'maxGridSize.x': '65535' })!;
    expect(checkLaunch(dim3(), dim3(1, 1, 128), emu).legal).toBe(true);
    expect(checkLaunch(dim3(), dim3(32, 32, 2), emu).legal).toBe(false);
    expect(checkLaunch(dim3(65536), dim3(), emu).legal).toBe(false);
  });
  it('a Metal record gives per-dimension caps only; total and grid stay unchecked', () => {
    const metal = limitsFromReport({
      name: 'a Metal device',
      'maxThreadsPerThreadgroup.width': '1024',
      'maxThreadsPerThreadgroup.height': '1024',
      'maxThreadsPerThreadgroup.depth': '1024',
    })!;
    expect(metal).toEqual({ maxThreadsPerBlock: null, maxBlockDim: { x: 1024, y: 1024, z: 1024 }, maxGridDim: null });
    const v = checkLaunch(dim3(1 << 20), dim3(32, 32, 2), metal);
    expect(v.legal).toBe(true);
    expect(v.unchecked).toEqual(['block-threads', 'grid-dim']);
    expect(checkLaunch(dim3(), dim3(2048, 1, 1), metal).legal).toBe(false);
  });
  it('is null when the record has none of the fields, and ignores unusable values', () => {
    expect(limitsFromReport({ name: 'x', hasUnifiedMemory: '1' })).toBeNull();
    expect(limitsFromReport({ maxThreadsPerBlock: '0', 'maxThreadsDim.x': '1024' })).toBeNull();
    expect(limitsFromReport({ maxThreadsPerBlock: true })).toBeNull();
    expect(limitsFromReport({ maxThreadsPerBlock: '512' })).toEqual({ maxThreadsPerBlock: 512, maxBlockDim: null, maxGridDim: null });
  });
});

describe("labels: the book's order is the reverse of the constructor's (p. 48)", () => {
  it('Block (1,0) has blockIdx.y = 1 and blockIdx.x = 0', () => {
    expect(blockFromLabel([1, 0])).toEqual({ x: 0, y: 1, z: 0 });
    expect(blockLabel({ x: 0, y: 1, z: 0 })).toEqual([1, 0]);
    expect(formatBlockLabel({ x: 0, y: 1, z: 0 })).toBe('Block (1,0)');
  });
  it('Thread (1,0,2) has z = 1, y = 0, x = 2', () => {
    expect(threadFromLabel([1, 0, 2])).toEqual({ x: 2, y: 0, z: 1 });
    expect(threadLabel({ x: 2, y: 0, z: 1 })).toEqual([1, 0, 2]);
    expect(formatThreadLabel({ x: 2, y: 0, z: 1 })).toBe('Thread (1,0,2)');
    expect(formatDim3({ x: 2, y: 0, z: 1 })).toBe('(2, 0, 1)');
    expect(BOOK_TARGET).toEqual({ block: { x: 0, y: 1, z: 0 }, thread: { x: 2, y: 0, z: 1 } });
  });
  it('labels round-trip for every thread and block of Fig. 3.1', () => {
    for (let n = 0; n < 16; n++) {
      const t = threadAt(n, FIG31.block);
      expect(threadFromLabel(threadLabel(t))).toEqual(t);
    }
    for (let y = 0; y < 2; y++) for (let x = 0; x < 2; x++) expect(blockFromLabel(blockLabel({ x, y, z: 0 }))).toEqual({ x, y, z: 0 });
  });
  it('a grid with more than one z layer gets a three-part block label', () => {
    expect(formatBlockLabel({ x: 3, y: 2, z: 1 }, dim3(4, 4, 2))).toBe('Block (1,2,3)');
    expect(formatBlockLabel({ x: 3, y: 2, z: 0 }, dim3(4, 4, 1))).toBe('Block (2,3)');
  });
  it("Fig. 3.1's front face reads Thread (0,0,0) to (0,0,3) then (0,1,0) to (0,1,3)", () => {
    const front = [0, 1, 2, 3, 4, 5, 6, 7].map((n) => formatThreadLabel(threadAt(n, FIG31.block)));
    expect(front).toEqual([
      'Thread (0,0,0)', 'Thread (0,0,1)', 'Thread (0,0,2)', 'Thread (0,0,3)',
      'Thread (0,1,0)', 'Thread (0,1,1)', 'Thread (0,1,2)', 'Thread (0,1,3)',
    ]);
  });
});

describe('linear thread number in a block: x fastest, then y, then z', () => {
  it("the book's Thread (1,0,2) in a (4,2,2) block is number 10", () => {
    const t = threadFromLabel([1, 0, 2]);
    expect(linearThread(t, dim3(4, 2, 2))).toBe(10);
    expect(linearThreadWorking(t, dim3(4, 2, 2))).toBe('2 + 0×4 + 1×8 = 10');
  });
  it('threadIdx.y has a stride of blockDim.x and threadIdx.z a stride of blockDim.x * blockDim.y', () => {
    const b = dim3(8, 16, 4);
    const at = (x: number, y: number, z: number) => linearThread({ x, y, z }, b);
    expect(at(1, 0, 0) - at(0, 0, 0)).toBe(1);
    expect(at(0, 1, 0) - at(0, 0, 0)).toBe(8);
    expect(at(0, 0, 1) - at(0, 0, 0)).toBe(128);
  });
  it('numbers every thread of a block once, 0 to threads - 1, and threadAt inverts it', () => {
    for (const b of [dim3(4, 2, 2), dim3(8, 16, 4), dim3(1, 1, 128), dim3(5, 3, 1)]) {
      const n = Number(threadsPerBlock(b));
      const seen = new Set<number>();
      for (let z = 0; z < b.z; z++)
        for (let y = 0; y < b.y; y++)
          for (let x = 0; x < b.x; x++) {
            const k = linearThread({ x, y, z }, b);
            expect(threadAt(k, b)).toEqual({ x, y, z });
            seen.add(k);
          }
      expect(seen.size).toBe(n);
      expect(Math.max(...seen)).toBe(n - 1);
    }
  });
  it('inside and sameIndex', () => {
    expect(inside({ x: 3, y: 1, z: 1 }, FIG31.block)).toBe(true);
    expect(inside({ x: 4, y: 0, z: 0 }, FIG31.block)).toBe(false);
    expect(inside({ x: 1, y: 0, z: 2 }, FIG31.block)).toBe(false); // the book's label read in constructor order
    expect(sameIndex({ x: 1, y: 2, z: 3 }, { x: 1, y: 2, z: 3 })).toBe(true);
    expect(sameIndex({ x: 1, y: 2, z: 3 }, { x: 3, y: 2, z: 1 })).toBe(false);
  });
});

describe('the find-the-thread drill', () => {
  it("starts with the book's example and then gives distinct targets inside the launch", () => {
    for (const seed of [1, 2, 3, 99]) {
      const ts = drillTargets(FIG31.grid, FIG31.block, 3, seed);
      expect(ts).toHaveLength(3);
      expect(ts[0]).toEqual(BOOK_TARGET);
      for (const t of ts) {
        expect(inside(t.block, FIG31.grid)).toBe(true);
        expect(inside(t.thread, FIG31.block)).toBe(true);
        expect(discriminates(t)).toBe(true);
      }
      const keys = ts.map((t) => formatThreadLabel(t.thread) + formatBlockLabel(t.block));
      expect(new Set(keys).size).toBe(3);
    }
  });
  it('is deterministic for a seed and changes with it', () => {
    expect(drillTargets(FIG31.grid, FIG31.block, 3, 7)).toEqual(drillTargets(FIG31.grid, FIG31.block, 3, 7));
    const a = JSON.stringify(drillTargets(FIG31.grid, FIG31.block, 3, 1));
    expect([2, 3, 4, 5, 6].some((s) => JSON.stringify(drillTargets(FIG31.grid, FIG31.block, 3, s)) !== a)).toBe(true);
  });
  it('still terminates on a launch too small to discriminate', () => {
    expect(drillTargets(dim3(1), dim3(2), 3, 1)).toHaveLength(2);
    expect(drillTargets(dim3(1), dim3(1), 3, 1)).toHaveLength(1);
  });
});

describe('reading a click in the drill', () => {
  const at = (bx: number, by: number, tx: number, ty: number, tz: number) => ({
    block: { x: bx, y: by, z: 0 },
    thread: { x: tx, y: ty, z: tz },
  });
  it('the target itself is right on both counts', () => {
    expect(misread(BOOK_TARGET, at(0, 1, 2, 0, 1))).toEqual({ block: 'right', thread: 'right' });
  });
  it('Block (1,0) taken as x = 1, y = 0 is the reversed reading', () => {
    expect(misread(BOOK_TARGET, at(1, 0, 2, 0, 1))).toEqual({ block: 'reversed', thread: 'right' });
  });
  it('Thread (0,1,3) taken as x = 0, y = 1, z = 3 does not exist; taken as (1,1,0) for label (0,1,1) it does', () => {
    const target = { block: { x: 1, y: 0, z: 0 }, thread: threadFromLabel([0, 1, 1]) };
    expect(misread(target, at(1, 0, 0, 1, 1))).toEqual({ block: 'right', thread: 'reversed' });
    expect(misread(target, at(1, 0, 3, 1, 1))).toEqual({ block: 'right', thread: 'other' });
  });
  it('anything else is other', () => {
    expect(misread(BOOK_TARGET, at(1, 1, 0, 0, 0))).toEqual({ block: 'other', thread: 'other' });
  });
});

describe('sizing a 1D grid', () => {
  const blocks = (n: number, b: number) => FORMULAS.map((f) => sizeGrid(n, b, f.key));

  it('the four formulas, in the order the explorer lists them', () => {
    expect(FORMULAS.map((f) => f.code)).toEqual(['n / b', 'n / b + 1', '(n + b - 1) / b', 'ceil(n / (double)b)']);
  });
  it("the book's numbers (p. 46): n = 1000 needs 4 blocks of 256, n = 4000 needs 16", () => {
    expect(blocks(1000, 256)).toEqual([3, 4, 4, 4]);
    expect(blocks(4000, 256)).toEqual([15, 16, 16, 16]);
    expect(coverage(1000, 256, 4)).toEqual({ blocks: 4, threads: 1024, working: 1000, surplus: 24, uncovered: 0, idleBlocks: 0 });
    expect(coverage(4000, 256, 16).surplus).toBe(96);
  });
  it("chapter 2's other worked numbers (pp. 37 to 39): 750, 2,000,000, and 100 with blocks of 32", () => {
    expect(sizeGrid(750, 256, 'ceil')).toBe(3);
    expect(sizeGrid(2_000_000, 256, 'ceil')).toBe(7813);
    expect(coverage(100, 32, sizeGrid(100, 32, 'round-up'))).toMatchObject({ blocks: 4, threads: 128, surplus: 28 });
  });
  it('chapter 2, exercise 4: 8000 elements in blocks of 1024 is 8 blocks, 8192 threads, 192 of them surplus', () => {
    const c = coverage(8000, 1024, sizeGrid(8000, 1024, 'round-up'));
    expect(c).toEqual({ blocks: 8, threads: 8192, working: 8000, surplus: 192, uncovered: 0, idleBlocks: 0 });
  });
  it('chapter 2, exercise 9: (200000 + 127) / 128 is 1563 blocks, 200,064 threads, 64 surplus', () => {
    const c = coverage(200000, 128, sizeGrid(200000, 128, 'round-up'));
    expect([c.blocks, c.threads, c.surplus]).toEqual([1563, 200064, 64]);
  });
  it('truncation leaves elements with no thread: 232 of 1000', () => {
    expect(coverage(1000, 256, sizeGrid(1000, 256, 'truncate'))).toEqual({
      blocks: 3, threads: 768, working: 768, surplus: 0, uncovered: 232, idleBlocks: 0,
    });
  });
  it('n / b + 1 launches a block in which no thread works when b divides n', () => {
    expect(coverage(1024, 256, sizeGrid(1024, 256, 'plus-one'))).toEqual({
      blocks: 5, threads: 1280, working: 1024, surplus: 256, uncovered: 0, idleBlocks: 1,
    });
    expect(workingInBlock(1024, 256, 3)).toBe(256);
    expect(workingInBlock(1024, 256, 4)).toBe(0);
    expect(workingInBlock(1000, 256, 3)).toBe(232);
  });
  it('over the whole range of the explorer: the rules the lesson states', () => {
    // 160,000 cases: collect what breaks and assert once, so the loop stays fast.
    const broken: string[] = [];
    const need = (ok: boolean, what: string, n: number, b: number) => {
      if (!ok && broken.length < 5) broken.push(`${what} at n = ${n}, b = ${b}`);
    };
    for (let b = 32; b <= 1024; b += 32)
      for (let n = 1; n <= 5000; n++) {
        const [t, p, r, c] = blocks(n, b);
        const divides = n % b === 0;
        need(r === c, 'round-up and ceil agree', n, b);
        need(sizeGridMinusOne(n, b) === r, 'the form of exercise 3 agrees with them', n, b);
        const cr = coverage(n, b, r);
        need(cr.uncovered === 0 && cr.surplus < b && cr.idleBlocks === 0, 'round-up covers everything with no idle block', n, b);
        need(coverage(n, b, t).uncovered === n % b, 'truncation leaves n mod b elements out', n, b);
        const cp = coverage(n, b, p);
        need(cp.uncovered === 0, 'n / b + 1 covers everything', n, b);
        need(cp.idleBlocks === (divides ? 1 : 0), 'n / b + 1 has one idle block exactly when b divides n', n, b);
        need((p === r) === !divides, 'n / b + 1 equals round-up unless b divides n', n, b);
      }
    expect(broken).toEqual([]);
  });
  it('per-block working counts add up to the working total', () => {
    for (const [n, b] of [[1000, 256], [4000, 256], [1024, 256], [77, 32]] as const)
      for (const f of FORMULAS) {
        const c = coverage(n, b, sizeGrid(n, b, f.key));
        let sum = 0;
        for (let k = 0; k < c.blocks; k++) sum += workingInBlock(n, b, k);
        expect(sum).toBe(c.working);
        expect(c.working + c.surplus).toBe(c.threads);
        expect(c.working + c.uncovered).toBe(n);
      }
  });
});

describe('the book, exercise 3 (p. 65)', () => {
  const M = 150;
  const N = 300;
  const bd = dim3(16, 32);
  const gd = dim3(sizeGridMinusOne(N, 16), sizeGridMinusOne(M, 32));
  it('(a) 512 threads per block', () => expect(threadsPerBlock(bd)).toBe(512n));
  it('the grid is (19, 5, 1)', () => expect(gd).toEqual({ x: 19, y: 5, z: 1 }));
  it('(c) 95 blocks', () => expect(blockCount(gd)).toBe(95n));
  it('(b) 48,640 threads', () => expect(totalThreads(gd, bd)).toBe(48640n));
  it('the grid spans 160 rows by 304 columns of threads', () => {
    expect(gd.y * bd.y).toBe(160);
    expect(gd.x * bd.x).toBe(304);
  });
  it('the launch is legal', () => expect(checkLaunch(gd, bd, CUDA_DOCUMENTED).legal).toBe(true));
});

describe('global coordinates (chapter 2, p. 34)', () => {
  it('blockIdx.x = 3, blockDim.x = 256, threadIdx.x = 10 handles element 778', () => {
    expect(globalCoord(dim3(3, 0, 0), dim3(256, 1, 1), dim3(10, 0, 0)).x).toBe(778);
  });
  it('blocks of 256 cover 0 to 255, 256 to 511, 512 to 767', () => {
    const at = (b: number, t: number) => globalCoord(dim3(b, 0, 0), dim3(256), dim3(t, 0, 0)).x;
    expect([at(0, 0), at(0, 255), at(1, 0), at(1, 255), at(2, 0), at(2, 255)]).toEqual([0, 255, 256, 511, 512, 767]);
  });
});
