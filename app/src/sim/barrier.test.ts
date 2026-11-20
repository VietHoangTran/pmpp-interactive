import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { EXERCISE_4, KERNELS, finding, run, timeline, type KernelId, type Op } from './barrier';

describe('timeline: threads reaching one barrier (Fig. 4.3, exercise 4)', () => {
  it('exercise 4: released at 3.0, 4.1 of 24.0 spent waiting, 17.1%', () => {
    const t = timeline(EXERCISE_4);
    expect(t.release).toBe(3.0);
    expect(t.waiting).toBeCloseTo(4.1, 9);
    expect(t.total).toBeCloseTo(24.0, 9);
    expect(Math.round(t.share * 1000) / 10).toBe(17.1);
    expect(t.slowest).toEqual([2]);
    expect(t.waits.map((w) => Math.round(w * 10) / 10)).toEqual([1.0, 0.7, 0, 0.2, 0.6, 1.1, 0.4, 0.1]);
  });
  it('the slowest thread sets everyone\'s wait', () => {
    const t = timeline([1, 1, 1, 5]);
    expect(t.waits).toEqual([4, 4, 4, 0]);
    expect(t.share).toBe(12 / 20);
  });
  it('equal times mean no waiting', () => {
    const t = timeline([2.5, 2.5, 2.5]);
    expect(t.waiting).toBe(0);
    expect(t.share).toBe(0);
    expect(t.slowest).toEqual([0, 1, 2]);
  });
  it('no threads, no time', () => {
    expect(timeline([])).toMatchObject({ release: 0, waiting: 0, total: 0, share: 0 });
  });
});

const last = <T,>(a: T[]) => a[a.length - 1];
const N = 4;

describe('run: Fig. 4.4 (p. 72)', () => {
  const prog = KERNELS.incorrect_barrier.build(N, 0);
  for (const rule of ['all-arrive', 'exited-excluded'] as const)
    it(`even threads wait at line 5 and odd threads at line 9, and nobody is released (${rule})`, () => {
      const r = run(prog, rule);
      expect(r.outcome).toBe('stuck');
      expect(r.reason).toBe('different-sites');
      expect(last(r.steps).waiting).toEqual([
        { site: 5, threads: [0, 2] },
        { site: 9, threads: [1, 3] },
      ]);
      expect(r.releases).toBe(0);
      expect(finding(r)).toBe('stuck-sites');
      expect(r.sites).toEqual([5, 9]);
    });
  it('the same holds for a larger block and for the other thread order', () => {
    const r = run(KERNELS.incorrect_barrier.build(8, 0), 'exited-excluded', 'descending');
    expect(last(r.steps).waiting).toEqual([
      { site: 5, threads: [0, 2, 4, 6] },
      { site: 9, threads: [1, 3, 5, 7] },
    ]);
  });
});

describe('run: a condition that is the same for the whole block', () => {
  it('in a block where it is true, all four arrive and are released together, under both rules', () => {
    for (const rule of ['all-arrive', 'exited-excluded'] as const) {
      const r = run(KERNELS.barrier_uniform.build(N, 1), rule);
      expect(r.outcome).toBe('finished');
      expect(r.releases).toBe(1);
      expect(r.partialReleases).toBe(0);
      expect(r.ahead).toEqual([]);
      expect(finding(r)).toBe('clean');
      const rel = r.steps.find((s) => s.kind === 'release')!;
      expect(rel).toMatchObject({ site: 6, released: [0, 1, 2, 3], absent: [] });
    }
  });
  it('in block 0 nobody calls the barrier: nothing hangs, and nothing separates the two phases', () => {
    const r = run(KERNELS.barrier_uniform.build(N, 0), 'all-arrive');
    expect(r.outcome).toBe('finished');
    expect(r.releases).toBe(0);
    expect(r.ahead[0]).toEqual({ thread: 0, phase: 1, behind: [1, 2, 3] });
    expect(finding(r)).toBe('ran-ahead');
  });
});

describe('run: a condition that differs between threads of the block', () => {
  const prog = KERNELS.barrier_per_thread.build(N, 0);
  it('thread 0 walks past while the others wait: stuck if every thread must arrive', () => {
    const r = run(prog, 'all-arrive');
    expect(r.outcome).toBe('stuck');
    expect(r.reason).toBe('ended-thread');
    expect(last(r.steps)).toMatchObject({ absent: [0], waiting: [{ site: 6, threads: [1, 2, 3] }] });
  });
  it('if ended threads do not block, the rest are released, and thread 0 has run ahead', () => {
    const r = run(prog, 'exited-excluded');
    expect(r.outcome).toBe('finished');
    expect(r.partialReleases).toBe(1);
    expect(r.ahead).toEqual([{ thread: 0, phase: 1, behind: [1, 2, 3] }]);
    expect(finding(r)).toBe('ran-ahead');
  });
  it('with the other thread order the same broken kernel shows nothing wrong', () => {
    const r = run(prog, 'exited-excluded', 'descending');
    expect(r.outcome).toBe('finished');
    expect(r.ahead).toEqual([]);
    expect(finding(r)).toBe('partial');
  });
});

describe('run: out-of-range threads return before the barrier', () => {
  const prog = KERNELS.barrier_early_return.build(N, 0);
  it('stuck when every thread of the block must arrive', () => {
    const r = run(prog, 'all-arrive');
    expect(r.outcome).toBe('stuck');
    expect(r.reason).toBe('ended-thread');
    expect(last(r.steps)).toMatchObject({ absent: [3], waiting: [{ site: 6, threads: [0, 1, 2] }] });
    expect(finding(r)).toBe('stuck-ended');
  });
  it('released when ended threads do not block, with nobody running ahead, in either thread order', () => {
    for (const order of ['ascending', 'descending'] as const) {
      const r = run(prog, 'exited-excluded', order);
      expect(r.outcome).toBe('finished');
      expect(r.releases).toBe(1);
      expect(r.partialReleases).toBe(1);
      expect(r.ahead).toEqual([]);
      expect(finding(r)).toBe('partial');
      expect(r.steps.find((s) => s.kind === 'release')).toMatchObject({ released: [0, 1, 2], absent: [3] });
    }
  });
  it('the rewrite that guards the work and not the barrier is clean under both rules', () => {
    for (const rule of ['all-arrive', 'exited-excluded'] as const) {
      const r = run(KERNELS.barrier_guarded.build(N, 0), rule);
      expect(finding(r)).toBe('clean');
      expect(r.steps.find((s) => s.kind === 'release')).toMatchObject({ released: [0, 1, 2, 3], absent: [] });
    }
  });
});

describe('run: a barrier in a loop whose trip count depends on the thread', () => {
  const prog = KERNELS.barrier_loop.build(N, 0);
  it('round one releases everybody; in round two the odd threads wait for even threads that have ended', () => {
    const r = run(prog, 'all-arrive');
    expect(r.releases).toBe(1);
    expect(r.outcome).toBe('stuck');
    expect(r.reason).toBe('ended-thread');
    expect(last(r.steps)).toMatchObject({ absent: [0, 2], waiting: [{ site: 7, threads: [1, 3] }] });
  });
  it('if ended threads do not block it finishes, but the even threads ran ahead, in either thread order', () => {
    for (const order of ['ascending', 'descending'] as const) {
      const r = run(prog, 'exited-excluded', order);
      expect(r.outcome).toBe('finished');
      expect(r.releases).toBe(2);
      expect(r.partialReleases).toBe(1);
      expect(r.ahead.length).toBeGreaterThan(0);
      expect(r.ahead.every((a) => a.thread % 2 === 0 && a.phase === 2)).toBe(true);
      expect(finding(r)).toBe('ran-ahead');
    }
  });
});

describe('run: bookkeeping', () => {
  it('every step carries the state of every thread, and passed counts the releases', () => {
    const r = run(KERNELS.barrier_uniform.build(N, 1), 'all-arrive');
    expect(r.steps.every((s) => s.states.length === N)).toBe(true);
    expect(r.steps.map((s) => s.kind)).toEqual(['advance', 'advance', 'advance', 'advance', 'release', 'advance', 'advance', 'advance', 'advance', 'finished']);
    expect(r.steps[0]).toMatchObject({ thread: 0, lines: [4, 6] });
    expect(r.steps[0].states[0]).toMatchObject({ status: 'waiting', site: 6, phase: 0 });
    expect(last(r.steps).states.every((s) => s.status === 'ended' && s.passed === 1)).toBe(true);
  });
});

// The model's programs are written by hand; these checks keep them honest against
// the files a reader sees, and keep the lesson's copies equal to those files.
const repo = path.resolve(__dirname, '../../..');
const snippet = (id: string) => fs.readFileSync(path.join(repo, 'labs/ch04/snippets', id + '.cu'), 'utf8');

describe('the model and the lesson agree with labs/ch04/snippets', () => {
  const ids = Object.keys(KERNELS) as KernelId[];
  for (const id of ids)
    it(`${id}: barrier lines hold __syncthreads(), exit lines a return, work lines code`, () => {
      const lines = snippet(id).split('\n');
      const callLines = lines.flatMap((l, i) => (l.includes('__syncthreads()') ? [i + 1] : []));
      expect(KERNELS[id].sites).toEqual(callLines);
      const ops: Op[] = [0, 1].flatMap((block) => KERNELS[id].build(N, block).threads.flat());
      for (const op of ops) {
        if (op.kind === 'barrier') expect(callLines).toContain(op.site);
        else if (op.kind === 'exit') expect(lines[op.line - 1]).toContain('return;');
        else {
          expect(lines[op.line - 1].trim().length).toBeGreaterThan(0);
          expect(lines[op.line - 1]).not.toContain('__syncthreads');
        }
      }
    });
  it('Fig. 4.4 keeps the printed shape: 11 lines, the if on line 3, the calls on lines 5 and 9', () => {
    const lines = snippet('incorrect_barrier').trimEnd().split('\n');
    expect(lines.length).toBe(11);
    expect(lines[0]).toContain('__global__');
    expect(lines[2]).toContain('if (threadIdx.x % 2 == 0)');
    expect(lines[6]).toContain('else');
  });
  it('section 4.3 shows these files unchanged', () => {
    const file = path.join(repo, 'app/src/chapters/ch04/S43.tsx');
    const src = fs.readFileSync(file, 'utf8');
    const m = /const SNIPPETS: Record<KernelId, string> = \{([\s\S]*?)\n\};/.exec(src);
    if (!m) throw new Error('no SNIPPETS in S43.tsx');
    for (const id of ids) {
      const c = new RegExp(id + ': `([\\s\\S]*?)`,').exec(m[1]);
      if (!c) throw new Error(`no ${id} in SNIPPETS`);
      expect(c[1].trim()).toBe(snippet(id).trim());
    }
  });
});
