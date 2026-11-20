import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  DEFECTS,
  EMU_FILL,
  EMU_FILL_BYTE,
  FLOAT_BYTES,
  METAL_ENCODE,
  N,
  SENTINEL,
  boundaryTraffic,
  chainOps,
  chainTimes,
  expected,
  groupOfKey,
  inputs,
  lanes,
  listing,
  mutate,
  outcome,
  phaseTimes,
  program,
  result,
  run,
  showCell,
  toSeconds,
  type Backend,
  type Iface,
  type Program,
} from './offload';

const repo = path.resolve(__dirname, '../../..');
const read = (rel: string) => fs.readFileSync(path.join(repo, rel), 'utf8');
const SUMS = [5, 8, 5, 9, 7, 17, 3, 14];
const ALL: [Backend, Iface][] = [['cuda', 'buffers'], ['hip', 'buffers'], ['metal', 'buffers'], ['metal', 'host-arrays']];

describe('the test data', () => {
  it('has sums that are never 0 and never the sentinel, and no zero in B', () => {
    expect(expected()).toEqual(SUMS);
    expect(SUMS.every((v) => v !== 0 && v !== SENTINEL)).toBe(true);
    expect(inputs().B.every((v) => v !== 0)).toBe(true);
  });
});

describe('the correct stub', () => {
  it('CUDA: ten calls, returns A + B, moves exactly 12n bytes in three copies, frees three arrays', () => {
    const p = program('cuda');
    expect(p.calls.map((c) => c.id)).toEqual([
      'alloc-A', 'alloc-B', 'alloc-C', 'copy-in-A', 'copy-in-B', 'launch', 'copy-back', 'free-A', 'free-B', 'free-C',
    ]);
    const s = run(p);
    expect(result(s)).toEqual(SUMS);
    expect(s.bytesMoved).toBe(12 * N);
    expect(s.copies).toBe(3);
    expect(s.frees).toBe(3);
    expect(s.apiCalls).toBe(9); // the launch statement is not an API call
    expect(s.finding).toBeNull();
    expect(s.hostRead.status).toBe('valid');
  });

  it('CUDA: the byte counter reads 4n, 8n, 8n, 12n after each copy and the launch', () => {
    const p = program('cuda');
    expect([3, 4, 5, 6, 7].map((k) => run(p, k).bytesMoved)).toEqual([0, 4 * N, 8 * N, 8 * N, 12 * N]);
  });

  it('HIP: the same ten steps with hip names, and hipFree marked as synchronising', () => {
    const c = program('cuda');
    const h = program('hip');
    expect(h.calls.map((x) => x.id)).toEqual(c.calls.map((x) => x.id));
    expect(h.calls.map((x) => x.code[0])).toEqual(c.calls.map((x) => x.code[0].replaceAll('cuda', 'hip')));
    expect(h.calls.filter((x) => x.group === 'free').every((x) => /implicit hipDeviceSynchronize/.test(x.note ?? ''))).toBe(true);
    expect(c.calls.filter((x) => x.group === 'free').every((x) => /may perform implicit synchronization/.test(x.note ?? ''))).toBe(true);
    const s = run(h);
    expect(result(s)).toEqual(SUMS);
    expect(s.bytesMoved).toBe(12 * N);
  });

  it('Metal with the data in shared buffers: 0 bytes moved, 0 copies, 0 frees, right sums', () => {
    const p = program('metal', 'buffers');
    expect(p.memory).toBe('shared');
    expect(p.calls.map((c) => c.id)).toEqual(['alloc-A', 'alloc-B', 'alloc-C', 'fill-A', 'fill-B', 'encode', 'launch', 'wait', 'read']);
    const s = run(p);
    expect(result(s)).toEqual(SUMS);
    expect(s.bytesMoved).toBe(0);
    expect(s.copies).toBe(0);
    expect(s.frees).toBe(0);
    expect(s.host).toBeNull(); // no host arrays exist: the data was created in the buffers
    expect(p.calls.some((c) => c.group === 'free')).toBe(false);
  });

  it('Metal with host arrays: 12n bytes through three copies come back, still no frees', () => {
    const p = program('metal', 'host-arrays');
    const s = run(p);
    expect(result(s)).toEqual(SUMS);
    expect(s.bytesMoved).toBe(12 * N);
    expect(s.copies).toBe(3);
    expect(s.frees).toBe(0);
    expect(p.calls.filter((c) => c.group === 'copy in')).toHaveLength(2);
    expect(p.calls.filter((c) => c.group === 'copy back')).toHaveLength(1);
  });

  it('Metal makes more calls than CUDA for fewer bytes', () => {
    expect(run(program('metal', 'buffers')).apiCalls).toBe(16);
    expect(run(program('metal', 'host-arrays')).apiCalls).toBe(14);
    expect(METAL_ENCODE).toHaveLength(8);
  });

  it('scales with n', () => {
    for (const n of [4, 8, 16]) {
      const s = run(program('cuda', 'buffers', n));
      expect(result(s)).toEqual(expected(n));
      expect(s.bytesMoved).toBe(12 * n);
    }
  });
});

describe('fresh memory', () => {
  it('is uncleared on CUDA and HIP, zero on Metal', () => {
    expect(run(program('cuda'), 3).device.C!.cells).toEqual(Array(N).fill('uncleared'));
    expect(run(program('hip'), 3).device.C!.cells).toEqual(Array(N).fill('uncleared'));
    expect(run(program('metal'), 3).device.C!.cells).toEqual(Array(N).fill(0));
  });
  it('is the 0xA5 fill in the emulator, as the emulator header says, and vanishes when added to an input', () => {
    expect(read('labs/common/include/pmpp/emu/emu_host.h')).toMatch(/std::memset\(\*devPtr, 0xA5, size\)/);
    expect(EMU_FILL_BYTE).toBe(0xa5);
    expect(EMU_FILL).toBeLessThan(0);
    expect(Math.abs(EMU_FILL + 2.87352e-16)).toBeLessThan(1e-20);
    expect(Math.fround(3 + EMU_FILL)).toBe(3);
    expect(run(program('cuda'), 1, 'emu').device.A!.cells[0]).toBe(EMU_FILL);
    expect(showCell(EMU_FILL)).toBe('A5');
    expect(showCell('uncleared')).toBe('?');
    expect(showCell('undetermined')).toBe('~');
    expect(showCell(17)).toBe('17');
  });
});

describe('pointers, the host thread and the queue', () => {
  it('device pointers refer to nothing, then device memory, then freed memory, and are never for the host to use', () => {
    const p = program('cuda');
    const ptr = (k: number, name: string) => run(p, k).pointers.find((x) => x.name === name)!;
    expect(ptr(0, 'A_d').refersTo).toBe('nothing yet');
    expect(ptr(1, 'A_d').refersTo).toBe('device memory');
    expect(ptr(8, 'A_d').refersTo).toBe('freed device memory');
    expect(run(p, 6).pointers.filter((x) => x.name.endsWith('_d')).every((x) => !x.hostMayUse)).toBe(true);
    expect(run(p, 6).pointers.filter((x) => x.name.endsWith('_h')).every((x) => x.hostMayUse && x.refersTo === 'host memory')).toBe(true);
  });
  it('on Metal every pointer refers to the one shared memory and the host may use it', () => {
    const s = run(program('metal'), 3);
    expect(s.pointers.map((x) => x.name)).toEqual(['A.contents', 'B.contents', 'C.contents']);
    expect(s.pointers.every((x) => x.hostMayUse && x.refersTo === 'the one shared memory')).toBe(true);
  });
  it('CUDA: the host waits in the three copies and nowhere else; the launch returns with the kernel queued', () => {
    const p = program('cuda');
    const blocked = p.calls.map((_, i) => run(p, i + 1).hostThread === 'blocked');
    expect(blocked).toEqual([false, false, false, true, true, false, true, false, false, false]);
    const afterLaunch = run(p, 6);
    expect(afterLaunch.queue).toEqual(['vecAddKernel']);
    expect(afterLaunch.kernel).toBe('issued');
    expect(afterLaunch.device.C!.inFlight).toBe(true);
    expect(afterLaunch.device.C!.cells).toEqual(Array(N).fill('uncleared'));
    const afterCopy = run(p, 7);
    expect(afterCopy.ran).toEqual(['vecAddKernel']); // the kernel finished while the host was inside the copy
    expect(afterCopy.queue).toEqual([]);
    expect(afterCopy.device.C!.cells).toEqual(SUMS);
  });
  it('Metal: encoding sends nothing, commit queues the kernel, only the wait blocks', () => {
    const p = program('metal');
    expect(run(p, 6).queue).toEqual([]);
    expect(run(p, 6).kernel).toBe('not issued');
    expect(run(p, 7).queue).toEqual(['add_arrays']);
    expect(p.calls.map((_, i) => run(p, i + 1).hostThread === 'blocked')).toEqual([false, false, false, false, false, false, false, true, false]);
    expect(run(p, 8).ran).toEqual(['add_arrays']);
  });
});

describe('lanes: who waits for whom', () => {
  it('CUDA: the host is blocked in the three copies; the kernel is pending after the launch and finishes inside the copy back', () => {
    const l = lanes(program('cuda'));
    expect(l.map((x) => x.id)).toEqual(program('cuda').calls.map((c) => c.id));
    expect(l.filter((x) => x.host === 'blocked').map((x) => x.id)).toEqual(['copy-in-A', 'copy-in-B', 'copy-back']);
    expect(l.filter((x) => x.pending.length).map((x) => x.id)).toEqual(['launch']);
    expect(l.filter((x) => x.finishes.length).map((x) => x.id)).toEqual(['copy-back']);
    expect(l.map((x) => x.bytesMoved)).toEqual([0, 0, 0, 32, 64, 64, 96, 96, 96, 96]);
  });
  it('Metal with the data in buffers: the only blocked call is the wait, and no byte moves', () => {
    const l = lanes(program('metal', 'buffers'));
    expect(l.filter((x) => x.host === 'blocked').map((x) => x.id)).toEqual(['wait']);
    expect(l.filter((x) => x.pending.length).map((x) => x.id)).toEqual(['launch']);
    expect(l.filter((x) => x.finishes.length).map((x) => x.id)).toEqual(['wait']);
    expect(l.every((x) => x.bytesMoved === 0)).toBe(true);
  });
  it('the bill of the scene: 8 additions for 96 bytes, 12 bytes per element', () => {
    const s = run(program('cuda'));
    expect(s.bytesMoved).toBe(96);
    expect(s.bytesMoved / N).toBe(12);
    expect(run(program('cuda'), 5).bytesMoved).toBe(8 * N); // before any arithmetic
  });
});

describe('reading the result from the host', () => {
  it.each(ALL)('%s (%s): never valid before the step that waits, valid from it on', (backend, iface) => {
    const p = program(backend, iface);
    const waits = p.calls.findIndex((c) => (backend === 'metal' ? c.id === 'wait' : c.id === 'copy-back'));
    expect(waits).toBeGreaterThan(0);
    for (let k = 0; k <= waits; k++) expect(run(p, k).hostRead.status).not.toBe('valid');
    for (let k = waits + 1; k <= p.calls.length; k++) expect(run(p, k).hostRead.status).toBe('valid');
  });
  it('CUDA after the launch: impossible, and C_h still holds the sentinel', () => {
    const r = run(program('cuda'), 6).hostRead;
    expect(r.status).toBe('impossible');
    expect(r.values).toEqual(Array(N).fill(SENTINEL));
  });
  it('Metal after commit: possible but stale; before any buffer exists: impossible', () => {
    const p = program('metal');
    expect(run(p, 0).hostRead.status).toBe('impossible');
    expect(run(p, 5).hostRead.status).toBe('stale');
    const r = run(p, 7).hostRead;
    expect(r.status).toBe('stale');
    expect(r.values).toEqual(Array(N).fill(0));
    expect(run(p, 8).hostRead.values).toEqual(SUMS);
  });
});

describe('listing', () => {
  it('gives every call a line range that covers exactly its code', () => {
    for (const [b, i] of ALL) {
      const p = program(b, i);
      const { lines, range } = listing(p);
      expect(range).toHaveLength(p.calls.length);
      p.calls.forEach((c, k) => {
        const [first, last] = range[k];
        expect(lines.slice(first - 1, last).map((l) => l.trim())).toEqual(c.code);
      });
      expect(lines.at(-1)).toBe('}');
    }
  });
});

describe('the code the stepper shows is the code in labs/ch02/snippets', () => {
  /** Every line of the program appears in the file, in the same order. */
  const inOrder = (p: Program, file: string, from = 0) => {
    const src = read(file).split('\n').map((l) => l.trim());
    let at = from;
    for (const line of listing(p).lines.slice(0, -1)) {
      const found = src.indexOf(line.trim(), at);
      expect(found, `"${line.trim()}" not found in ${file} after line ${at}`).toBeGreaterThanOrEqual(0);
      at = found + 1;
    }
    return at;
  };
  it('CUDA: vecadd_stub.cu', () => {
    inOrder(program('cuda'), 'labs/ch02/snippets/vecadd_stub.cu');
  });
  it('Metal: both functions of vecadd_stub_metal.mm, which is compiled and run on Metal', () => {
    const after = inOrder(program('metal', 'buffers'), 'labs/ch02/snippets/vecadd_stub_metal.mm');
    inOrder(program('metal', 'host-arrays'), 'labs/ch02/snippets/vecadd_stub_metal.mm', after);
  });
  it('HIP: every renamed call is a rename the course header makes', () => {
    const hip = read('labs/common/include/pmpp/vendors/hip.h');
    for (const name of ['Malloc', 'Free', 'Memcpy', 'MemcpyHostToDevice', 'MemcpyDeviceToHost'])
      expect(hip).toMatch(new RegExp(`#define cuda${name}\\s+hip${name}\\b`));
  });
});

describe('defects: one at a time, each with its own symptom', () => {
  it('there are seven, and each changes the CUDA and HIP stubs', () => {
    expect(DEFECTS).toHaveLength(7);
    for (const d of DEFECTS)
      for (const b of ['cuda', 'hip'] as const) {
        const m = mutate(program(b), d)!;
        expect(m.defect).toBe(d);
        expect(listing(m).lines).not.toEqual(listing(program(b)).lines);
      }
  });

  it('a count in elements on the copy back leaves exactly the first n/4 elements right', () => {
    for (const env of ['cuda', 'hip', 'emu'] as const) {
      const o = outcome('bytes-as-elements', env);
      expect(o.klass).toBe('wrong-values');
      expect(o.result).toEqual([5, 8, SENTINEL, SENTINEL, SENTINEL, SENTINEL, SENTINEL, SENTINEL]);
      expect(o.wrong).toEqual([2, 3, 4, 5, 6, 7]);
      expect(o.stoppedAt).toBeNull();
    }
    const o16 = outcome('bytes-as-elements', 'cuda', 16);
    expect(o16.wrong).toEqual(Array.from({ length: 12 }, (_, i) => i + 4));
  });

  it('allocations in elements: out of bounds; undetermined on a device, stopped by the emulator at the first copy', () => {
    const o = outcome('alloc-in-elements', 'cuda');
    expect(o.klass).toBe('out-of-bounds');
    expect(o.result).toEqual(Array(N).fill('undetermined'));
    expect(o.finding!.step).toBe(4);
    expect(o.finding!.detail).toBe('the copy writes 32 bytes into an allocation of 8');
    const e = outcome('alloc-in-elements', 'emu');
    expect(e.klass).toBe('out-of-bounds');
    expect(e.stoppedAt).toBe(4);
    expect(e.result).toEqual([]);
  });

  it('B never copied in: uncleared on CUDA, A itself in the emulator and on Metal', () => {
    const o = outcome('missing-copy-in', 'cuda');
    expect(o.klass).toBe('wrong-values');
    expect(o.result).toEqual(Array(N).fill('uncleared'));
    expect(o.wrong).toHaveLength(N);
    expect(outcome('missing-copy-in', 'emu').result).toEqual(inputs().A);
    const m = outcome('missing-copy-in', 'metal');
    expect(m.klass).toBe('wrong-values');
    expect(m.result).toEqual(inputs().A);
    expect(m.wrong).toHaveLength(N);
  });

  it('copy back before the launch: what the memory held before the kernel wrote it', () => {
    expect(outcome('copy-back-early', 'cuda').result).toEqual(Array(N).fill('uncleared'));
    expect(outcome('copy-back-early', 'emu').result).toEqual(Array(N).fill(EMU_FILL));
    const m = outcome('copy-back-early', 'metal');
    expect(m.result).toEqual(Array(N).fill(0));
    expect(m.klass).toBe('wrong-values');
    // the kernel still ran afterwards; its result was simply never fetched
    const s = run(mutate(program('cuda'), 'copy-back-early')!);
    expect(s.device.C!.cells).toEqual(SUMS);
    expect(s.hostRead.status).toBe('stale');
  });

  it('free before the copy back: use after free; stopped by the emulator at the copy', () => {
    const o = outcome('free-before-copy-back', 'cuda');
    expect(o.klass).toBe('use-after-free');
    expect(o.result).toEqual(Array(N).fill('undetermined'));
    const e = outcome('free-before-copy-back', 'emu');
    expect(e.klass).toBe('use-after-free');
    expect(e.stoppedAt).toBe(8);
    expect(e.program!.calls[7].id).toBe('copy-back');
  });

  it('swapped pointers with the kind unchanged: undefined per the documentation; the emulator leaves the sentinel', () => {
    const o = outcome('swapped-pointers', 'cuda');
    expect(o.klass).toBe('undefined-per-docs');
    expect(o.errorCode).toBeNull();
    expect(outcome('swapped-pointers', 'hip').klass).toBe('undefined-per-docs');
    // nothing about either array the call was handed is known afterwards
    expect(o.result).toEqual(Array(N).fill('undetermined'));
    const s = run(o.program!);
    expect(s.device.C!.cells).toEqual(Array(N).fill('undetermined'));
    expect(s.hostRead.status).toBe('stale');
    const e = outcome('swapped-pointers', 'emu');
    expect(e.klass).toBe('wrong-values');
    expect(e.result).toEqual(Array(N).fill(SENTINEL));
    expect(e.stoppedAt).toBeNull();
  });

  it('host dereference: right answer everywhere; not allowed on CUDA, slow zero copy on HIP, invisible to the emulator, normal on Metal', () => {
    const c = outcome('host-dereference', 'cuda');
    expect(c.klass).toBe('not-documented');
    expect(c.result).toEqual(SUMS);
    expect(outcome('host-dereference', 'hip').klass).toBe('slow-zero-copy');
    expect(outcome('host-dereference', 'emu').klass).toBe('passes');
    const m = outcome('host-dereference', 'metal');
    expect(m.klass).toBe('passes');
    expect(m.result).toEqual(SUMS);
  });

  it('no defect has a documented, guaranteed error code', () => {
    for (const d of DEFECTS) for (const env of ['cuda', 'hip', 'emu', 'metal'] as const) expect(outcome(d, env).errorCode).toBeNull();
  });

  it('the emulator stops exactly two of the seven and passes exactly one', () => {
    const emu = DEFECTS.map((d) => outcome(d, 'emu'));
    expect(emu.filter((o) => o.stoppedAt != null).map((o) => o.defect)).toEqual(['alloc-in-elements', 'free-before-copy-back']);
    expect(emu.filter((o) => o.klass === 'passes').map((o) => o.defect)).toEqual(['host-dereference']);
  });

  it('what a test sees in the emulator is different for every defect', () => {
    const seen = DEFECTS.map((d) => {
      const o = outcome(d, 'emu');
      return JSON.stringify([o.stoppedAt, o.finding?.kind ?? null, o.result]);
    });
    expect(new Set(seen).size).toBe(DEFECTS.length);
  });

  it('four of the seven cannot be written on Metal with the data in shared buffers', () => {
    const none = DEFECTS.filter((d) => outcome(d, 'metal').klass === 'no-counterpart');
    expect(none).toEqual(['bytes-as-elements', 'alloc-in-elements', 'free-before-copy-back', 'swapped-pointers']);
  });

  it('every outcome names its facts and says something', () => {
    for (const d of DEFECTS)
      for (const env of ['cuda', 'hip', 'emu', 'metal'] as const) {
        const o = outcome(d, env);
        expect(o.summary.length).toBeGreaterThan(20);
        expect(o.facts).toMatch(/^[ABC]\d/);
      }
  });

  it('Metal: reading after commit and before the wait gives values the model will not put a number on', () => {
    const p = program('metal');
    const calls = p.calls.slice();
    const [read] = calls.splice(8, 1);
    calls.splice(7, 0, read); // read moved between commit and wait
    const s = run({ ...p, calls });
    expect(s.consumed).toEqual(Array(N).fill('undetermined'));
  });
});

describe('boundary traffic', () => {
  it('transparent: 3 transfers per operation; resident: k + 2', () => {
    for (let k = 1; k <= 8; k++) {
      const t = boundaryTraffic(k, 'transparent', 'separate', 1024);
      const r = boundaryTraffic(k, 'resident', 'separate', 1024);
      expect(t.transfers).toHaveLength(3 * k);
      expect(t.transfersIn).toBe(2 * k);
      expect(t.transfersOut).toBe(k);
      expect(r.transfers).toHaveLength(k + 2);
      expect(r.transfersIn).toBe(k + 1);
      expect(r.transfersOut).toBe(1);
      expect(t.kernelBytes).toBe(r.kernelBytes); // the kernels' own work is identical
      expect(t.kernelBytes).toBe(k * 3 * 1024 * FLOAT_BYTES);
    }
  });
  it('the prediction of section 2.3: four operations are 12 transfers transparent and 6 resident', () => {
    expect(boundaryTraffic(4, 'transparent', 'separate', 1 << 20).transfers).toHaveLength(12);
    expect(boundaryTraffic(4, 'resident', 'separate', 1 << 20).transfers).toHaveLength(6);
  });
  it('bytes and ratio for the default chain of three', () => {
    const n = 1 << 20;
    const t = boundaryTraffic(3, 'transparent', 'separate', n);
    expect(t.bytesAcross).toBe(9 * 4 * n);
    expect(t.ratio).toBe(1); // every byte a kernel touches crossed the boundary once
    const r = boundaryTraffic(3, 'resident', 'separate', n);
    expect(r.bytesAcross).toBe(5 * 4 * n);
    expect(r.ratio).toBeCloseTo(5 / 9, 12);
    expect(t.ops).toEqual(['add bias', 'scale', 'add residual']);
  });
  it('one vector addition as a transparent offload moves 12 bytes per addition', () => {
    const n = 4096;
    expect(boundaryTraffic(1, 'transparent', 'separate', n).bytesAcross / n).toBe(12);
    expect(boundaryTraffic(1, 'transparent', 'separate', n).bytesAcross).toBe(run(program('cuda', 'buffers', 16)).bytesMoved / 16 * n);
  });
  it('one shared memory: nothing crosses in either strategy, and the kernels do the same work', () => {
    for (const s of ['transparent', 'resident'] as const) {
      const t = boundaryTraffic(5, s, 'shared', 4096);
      expect(t.transfers).toEqual([]);
      expect(t.bytesAcross).toBe(0);
      expect(t.ratio).toBe(0);
      expect(t.kernelBytes).toBe(5 * 3 * 4096 * FLOAT_BYTES);
    }
  });
  it('names operations by layer once the chain is longer than three', () => {
    expect(chainOps(4).map((o) => o.name)).toEqual(['add bias 1', 'scale 1', 'add residual 1', 'add bias 2']);
  });
});

describe('benchmark rows of labs 2.2 and 2.3', () => {
  it('maps a phase column to its step group and ignores anything that is not a time', () => {
    expect(groupOfKey('alloc_ms')).toBe('allocate');
    expect(groupOfKey('h2d_ms')).toBe('copy in');
    expect(groupOfKey('copy_in_ms')).toBe('copy in');
    expect(groupOfKey('copy_in_s')).toBe('copy in');
    expect(groupOfKey('kernel_ms')).toBe('launch');
    expect(groupOfKey('wait_s')).toBe('wait');
    expect(groupOfKey('d2h_ms')).toBe('copy back');
    expect(groupOfKey('copy_back_ms')).toBe('copy back');
    expect(groupOfKey('free_ms')).toBe('free');
    expect(groupOfKey('min_s')).toBeNull();
    expect(groupOfKey('n')).toBeNull();
    expect(groupOfKey('name')).toBeNull();
  });
  it('converts by the unit a column name ends with', () => {
    expect(toSeconds('alloc_s', 0.25)).toBe(0.25);
    expect(toSeconds('alloc_ms', 250)).toBe(0.25);
    expect(toSeconds('alloc_us', 250000)).toBe(0.25);
    expect(toSeconds('n', 1024)).toBeNull();
    expect(toSeconds('alloc_s', 'fast')).toBeNull();
  });
  it('reads a phases row into seconds per step group, adding columns of one group', () => {
    const t = phaseTimes({ name: 'phases', n: 1 << 20, alloc_s: 0.001, copy_in_s: 0.004, kernel_s: 0.0005, copy_back_s: 0.002, free_s: 0.0001 });
    expect(t).toEqual({ allocate: 0.001, 'copy in': 0.004, launch: 0.0005, 'copy back': 0.002, free: 0.0001 });
    expect(phaseTimes({ name: 'phases', h2d_a_ms: 1, h2d_b_ms: 2 })['copy in']).toBeCloseTo(0.003, 12);
    expect(phaseTimes({ name: 'phases', n: 8 })).toEqual({});
  });
  it('reads the rows lab 2.2 really writes, on the CUDA family and on Metal', () => {
    // Column names from labs/ch02/lab2_timing/main.cpp.
    const main = read('labs/ch02/lab2_timing/main.cpp');
    for (const col of ['alloc_s', 'h2d_s', 'kernel_s', 'd2h_s', 'free_s', 'wrap_s', 'encode_s', 'wait_s', 'readback_s', 'gpu_s', 'total_s'])
      expect(main).toContain(`"${col}"`);
    const cuda = phaseTimes({ name: 'phases', n: 65536, sync: 0, alloc_s: 1, h2d_s: 2, kernel_s: 3, d2h_s: 4, free_s: 5, total_s: 15, kernel_share_pct: 20, nosync_kernel_pct: 1 });
    expect(cuda).toEqual({ allocate: 1, 'copy in': 2, launch: 3, 'copy back': 4, free: 5 });
    const metal = phaseTimes({ name: 'phases', n: 65536, wrap_s: 1, encode_s: 2, wait_s: 3, readback_s: 4, gpu_s: 0.5, total_s: 10 });
    expect(metal).toEqual({ 'copy in': 1, launch: 2, wait: 3, 'copy back': 4 });
  });
  it('reads a chain row only when it has both times', () => {
    expect(chainTimes({ name: 'chain', n: 4096, resident_s: 0.002, round_trips_s: 0.009 })).toEqual({ resident: 0.002, roundTrips: 0.009 });
    const ms = chainTimes({ name: 'chain', resident_ms: 2, transparent_ms: 8 })!;
    expect(ms.resident).toBeCloseTo(0.002, 12);
    expect(ms.roundTrips).toBeCloseTo(0.008, 12);
    // The row lab 2.3 really writes (labs/ch02/lab3_elementwise/main.cpp).
    const main = read('labs/ch02/lab3_elementwise/main.cpp');
    expect(main).toContain('bench_row(a, "chain")');
    for (const col of ['resident_s', 'roundtrips_s']) expect(main).toContain(`"${col}"`);
    expect(chainTimes({ name: 'chain', rows: 16384, d: 4096, resident_s: 0.09, roundtrips_s: 0.19, ratio: 2.1, roundtrip: 'host arrays copied' }))
      .toEqual({ resident: 0.09, roundTrips: 0.19 });
    expect(chainTimes({ name: 'chain', resident_s: 0.002 })).toBeNull();
    expect(chainTimes({ name: 'chain', n: 4096 })).toBeNull();
  });
});
