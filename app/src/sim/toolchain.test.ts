import { describe, expect, it } from 'vitest';
import {
  ANSWERS,
  BACKENDS,
  CALLEES,
  CALLERS,
  DYNAMIC_PARALLELISM,
  METAL_FNS,
  PROGRAM,
  QUALIFIERS,
  canRunOnNewArchitecture,
  cleanArchs,
  compilesOnUsersMachine,
  executedBy,
  functionSource,
  grade,
  legalCall,
  metalCall,
  placement,
  spacesOf,
  stages,
  type Answer,
  type Caller,
  type Qualifiers,
} from './toolchain';

const fn = (name: string) => PROGRAM.find((f) => f.name === name)!;
const ids = (list: { id: string }[]) => list.map((s) => s.id);
const chipNames = (box: { chips: { fn: string }[] }) => box.chips.map((c) => c.fn);

describe('the caller by callee table (book Fig. 2.11, pp. 35-36)', () => {
  // The twelve cells, written out by hand from the figure: [legality, who executes].
  const FIG: Record<Caller, Record<string, [string, string | null]>> = {
    'host-fn': {
      none: ['legal', 'caller-host-thread'],
      global: ['legal', 'new-grid'],
      device: ['illegal', null],
      'host-device': ['legal', 'caller-host-thread'],
    },
    kernel: {
      none: ['illegal', null],
      global: ['dynamic-parallelism', 'new-grid'],
      device: ['legal', 'caller-device-thread'],
      'host-device': ['legal', 'caller-device-thread'],
    },
    'device-fn': {
      none: ['illegal', null],
      global: ['dynamic-parallelism', 'new-grid'],
      device: ['legal', 'caller-device-thread'],
      'host-device': ['legal', 'caller-device-thread'],
    },
  };
  it('has three callers and four callee columns: twelve cells', () => {
    expect(CALLERS).toHaveLength(3);
    expect(CALLEES).toHaveLength(4);
  });
  for (const caller of CALLERS)
    for (const callee of CALLEES)
      it(`${caller} calls ${callee}`, () => {
        const r = legalCall(caller, callee);
        expect([r.book, r.executedBy]).toEqual(FIG[caller][callee]);
        expect(r.facts.length).toBeGreaterThan(0);
        expect(r.row.length).toBeGreaterThan(0);
      });
  it('no qualifier and __host__ are the same rule', () => {
    for (const caller of CALLERS) {
      const a = legalCall(caller, 'none');
      const b = legalCall(caller, 'host');
      expect([a.book, a.executedBy, a.legal]).toEqual([b.book, b.executedBy, b.legal]);
    }
  });
  it('an unqualified function called from a kernel is illegal, and the reason names the default', () => {
    const r = legalCall('kernel', 'none');
    expect(r.legal).toBe(false);
    expect(r.why).toContain('No qualifier means __host__');
  });
  it('executedBy: code runs where its caller runs, unless the callee is a kernel', () => {
    for (const caller of CALLERS)
      for (const callee of QUALIFIERS) {
        const who = executedBy(caller, callee);
        if (callee === 'global') expect(who).toBe('new-grid');
        else if (who !== null) expect(who).toBe(caller === 'host-fn' ? 'caller-host-thread' : 'caller-device-thread');
      }
  });
});

describe('a kernel launched from device code (dynamic parallelism)', () => {
  it('CUDA yes, HIP no, Metal none documented, each with its fact', () => {
    expect(DYNAMIC_PARALLELISM.cuda).toMatchObject({ supported: true, fact: 'A13' });
    expect(DYNAMIC_PARALLELISM.hip).toMatchObject({ supported: false, fact: 'ch05 B4' });
    expect(DYNAMIC_PARALLELISM.metal).toMatchObject({ supported: null, fact: 'C7' });
  });
  it('the cell compiles on CUDA and not on HIP', () => {
    expect(legalCall('kernel', 'global', 'cuda').legal).toBe(true);
    expect(legalCall('kernel', 'global', 'hip').legal).toBe(false);
    expect(legalCall('device-fn', 'global', 'hip').legal).toBe(false);
    expect(legalCall('host-fn', 'global', 'hip').legal).toBe(true);
  });
  it('on Metal the qualifier table does not apply at all', () => {
    for (const caller of CALLERS)
      for (const callee of CALLEES) {
        const r = legalCall(caller, callee, 'metal');
        expect(r.applies).toBe(false);
        expect(r.legal).toBeNull();
        expect(r.facts).toEqual(['C7']);
      }
  });
});

describe('marking an answer', () => {
  it('every plain cell has exactly one right answer', () => {
    for (const caller of CALLERS)
      for (const callee of CALLEES) {
        const marks = ANSWERS.map((a) => grade(caller, callee, a));
        if (legalCall(caller, callee).book === 'dynamic-parallelism') {
          expect(marks.filter((m) => m === 'depends')).toHaveLength(2);
          expect(marks.filter((m) => m === 'right')).toHaveLength(0);
        } else {
          expect(marks.filter((m) => m === 'right')).toHaveLength(1);
          expect(marks.filter((m) => m === 'depends')).toHaveLength(0);
        }
      }
  });
  it('the cells readers get wrong', () => {
    expect(grade('kernel', 'none', 'error')).toBe('right');
    expect(grade('kernel', 'none', 'caller-device-thread')).toBe('wrong');
    expect(grade('host-fn', 'device', 'error')).toBe('right');
    expect(grade('host-fn', 'global', 'new-grid')).toBe('right');
    expect(grade('host-fn', 'global', 'caller-host-thread')).toBe('wrong');
    expect(grade('kernel', 'host-device', 'caller-device-thread')).toBe('right');
    expect(grade('host-fn', 'host-device', 'caller-host-thread')).toBe('right');
    expect(grade('kernel', 'global', 'error')).toBe('depends');
    expect(grade('device-fn', 'global', 'new-grid')).toBe('depends');
    expect(grade('device-fn', 'global', 'caller-device-thread' as Answer)).toBe('wrong');
  });
});

describe('which compilation sees a function', () => {
  it('qualifiers decide the side', () => {
    expect(spacesOf('none')).toEqual(['host']);
    expect(spacesOf('host')).toEqual(['host']);
    expect(spacesOf('device')).toEqual(['device']);
    expect(spacesOf('global')).toEqual(['device']);
    expect(spacesOf('host-device')).toEqual(['host', 'device']);
  });
  it('the program is main, vecAdd, vecAddKernel and a shared helper', () => {
    expect(PROGRAM.map((f) => [f.name, f.quals])).toEqual([
      ['main', 'none'], ['vecAdd', 'none'], ['vecAddKernel', 'global'], ['add1', 'host-device'],
    ]);
  });
  it('__host__ __device__ yields two placements on CUDA and on HIP', () => {
    for (const b of ['cuda', 'hip'] as const) {
      const p = placement(fn('add1'), b);
      expect(p.map((x) => x.space)).toEqual(['host', 'device']);
      expect(p.every((x) => x.facts.includes('A7'))).toBe(true);
      expect(p.some((x) => x.writtenAgain)).toBe(false);
    }
  });
  it('on Metal the shared helper is two hand-written functions in two files', () => {
    const p = placement(fn('add1'), 'metal');
    expect(p.map((x) => x.file)).toEqual(['launch.mm', 'kernel.metal']);
    expect(p[1].writtenAgain).toBe(true);
    expect(p.every((x) => x.facts.includes('C7'))).toBe(true);
  });
  it('main and vecAdd are host only, the kernel is device only, on every backend', () => {
    for (const b of BACKENDS) {
      expect(placement(fn('main'), b).map((x) => x.space)).toEqual(['host']);
      expect(placement(fn('vecAdd'), b).map((x) => x.space)).toEqual(['host']);
      expect(placement(fn('vecAddKernel'), b).map((x) => x.space)).toEqual(['device']);
    }
    expect(placement(fn('vecAddKernel'), 'cuda')[0].compiler).toContain('nvcc');
    expect(placement(fn('vecAddKernel'), 'hip')[0].compiler).toContain('amdclang++');
    expect(placement(fn('vecAddKernel'), 'metal')[0].file).toBe('kernel.metal');
  });
});

describe('stage lists, in the documented order', () => {
  it('CUDA: split, host object and PTX, cubin through ptxas, fatbin, then run time (A14)', () => {
    const s = stages('cuda');
    expect(ids(s)).toEqual(['source', 'split', 'compile', 'cubin', 'fatbin', 'run']);
    expect(s.map((x) => x.when)).toEqual(['build', 'build', 'build', 'build', 'build', 'run']);
    expect(s[3].tool).toBe('ptxas');
    expect(s.every((x) => x.facts.includes('A14'))).toBe(true);
  });
  it('CUDA: the helper is on both sides from the split on', () => {
    const split = stages('cuda')[1];
    expect(chipNames(split.boxes[0])).toEqual(['main', 'vecAdd', 'add1']);
    expect(chipNames(split.boxes[1])).toEqual(['vecAddKernel', 'add1']);
    expect(split.boxes[0].chips.every((c) => c.copy === 'host')).toBe(true);
    expect(split.boxes[1].chips.every((c) => c.copy === 'device')).toBe(true);
  });
  it('CUDA: the fatbin holds PTX and a cubin inside the executable', () => {
    const fat = stages('cuda')[4];
    expect(fat.boxes.map((b) => b.title)).toEqual(['host code', 'fatbin: PTX', 'fatbin: cubin']);
    expect(new Set(fat.boxes.map((b) => b.group))).toEqual(new Set(['the executable']));
  });
  it('CUDA at run time: a suitable cubin is loaded, otherwise the driver compiles the PTX', () => {
    const fits = stages('cuda', { cubinFits: true })[5];
    expect(fits.title).toBe('Run time: the cubin is loaded');
    expect(fits.boxes.find((b) => b.id === 'ptx')!.state).toBe('unused');
    expect(fits.boxes.some((b) => b.id === 'jit')).toBe(false);
    const jit = stages('cuda', { cubinFits: false })[5];
    expect(jit.title).toBe('Run time: the driver compiles the PTX');
    expect(jit.tool).toBe('the device driver');
    expect(jit.boxes.find((b) => b.id === 'jit')!.form).toBe('GPU machine code for this GPU');
    expect(jit.boxes.some((b) => b.id === 'cubin')).toBe(false);
  });
  it('HIP: host pass, one device pass per named architecture, embedded machine code, then run time (B4, B6)', () => {
    const s = stages('hip', { archs: ['archA', 'archB'], deviceArch: 'archA' });
    expect(ids(s)).toEqual(['source', 'host-pass', 'device-passes', 'embed', 'run']);
    expect(s[2].boxes.filter((b) => b.id.startsWith('arch-')).map((b) => b.title)).toEqual(['device code: archA', 'device code: archB']);
    expect(s[2].title).toBe('Device passes: archA, archB');
    expect(s[3].boxes.every((b) => b.group === 'the executable')).toBe(true);
    expect(s.some((x) => x.text.includes('PTX') && x.id !== 'device-passes')).toBe(false);
  });
  it('HIP at run time: runnable only on a named architecture', () => {
    const run = (archs: string[], deviceArch: string) => stages('hip', { archs, deviceArch })[4];
    expect(run(['archA', 'archB'], 'archB').title).toBe('Run time on archB: runs');
    expect(run(['archA', 'archB'], 'archB').boxes.find((b) => b.id === 'arch-archB')!.state).toBe('ok');
    expect(run(['archA', 'archB'], 'archB').boxes.find((b) => b.id === 'arch-archA')!.state).toBe('unused');
    const no = run(['archA'], 'archC');
    expect(no.title).toBe('Run time on archC: not runnable');
    expect(no.boxes.find((b) => b.id === 'missing')!.state).toBe('fail');
    expect(run(['archA'], '').title).toBe('Run time: which GPU?');
    expect(stages('hip')[2].title).toBe('Device passes: no architecture named yet');
  });
  it('the list of architectures is what the reader typed, cleaned', () => {
    expect(cleanArchs([' a ', '', 'b', 'a'])).toEqual(['a', 'b']);
    expect(cleanArchs()).toEqual([]);
  });
  it('Metal: two files, a library at build time or run time, a function object, then the pipeline state (C6)', () => {
    const run = stages('metal');
    expect(ids(run)).toEqual(['source', 'library', 'function', 'pipeline']);
    expect(run.map((x) => x.when)).toEqual(['build', 'run', 'run', 'run']);
    expect(run[1].tool).toBe('newLibraryWithSource:options:error:');
    const build = stages('metal', { metalLibrary: 'build' });
    expect(build.map((x) => x.when)).toEqual(['build', 'build', 'run', 'run']);
    expect(build[1].tool).toBe('xcrun metal');
    // Whichever route, the last stage is at run time and is for this GPU.
    for (const s of [run, build]) {
      expect(s[3].when).toBe('run');
      expect(s[3].boxes.find((b) => b.id === 'pipeline')!.form).toBe('GPU machine code for this GPU');
    }
  });
  it('Metal: the two source files hold the helper once each', () => {
    const src = stages('metal')[0];
    expect(src.boxes.map((b) => b.title)).toEqual(['launch.mm', 'kernel.metal']);
    expect(chipNames(src.boxes[0])).toEqual(['main', 'vecAdd', 'add1']);
    expect(chipNames(src.boxes[1])).toEqual(['vecAddKernel', 'add1']);
  });
  it('every stage of every backend cites at least one fact and names its tool', () => {
    for (const b of BACKENDS)
      for (const s of stages(b, { archs: ['x'], deviceArch: 'x' })) {
        expect(s.facts.length).toBeGreaterThan(0);
        expect(s.tool.length).toBeGreaterThan(0);
        expect(s.text.length).toBeGreaterThan(20);
      }
  });
  it('which backends finish compiling on the machine that runs the program', () => {
    expect(compilesOnUsersMachine('cuda')).toBe('when-needed');
    expect(compilesOnUsersMachine('hip')).toBe('never');
    expect(compilesOnUsersMachine('metal')).toBe('always');
  });
});

describe('a GPU architecture that appears after you ship (question q-future)', () => {
  it('CUDA with PTX in the fatbin: yes (A14)', () => {
    expect(canRunOnNewArchitecture('cuda')).toMatchObject({ runs: 'yes', facts: ['A14'] });
    expect(canRunOnNewArchitecture('cuda', { ptx: true }).runs).toBe('yes');
  });
  it('CUDA without PTX: the facts do not say', () => {
    expect(canRunOnNewArchitecture('cuda', { ptx: false }).runs).toBe('not-documented');
  });
  it('HIP: only the architectures named at build time (B6)', () => {
    expect(canRunOnNewArchitecture('hip', { archs: ['archA'], newArch: 'archB' })).toMatchObject({ runs: 'no', facts: ['B6'] });
    expect(canRunOnNewArchitecture('hip').runs).toBe('no');
    expect(canRunOnNewArchitecture('hip', { archs: ['archA'], newArch: 'archA' }).runs).toBe('yes');
  });
  it('Metal: yes, compilation for the GPU always happens at pipeline creation (C6)', () => {
    expect(canRunOnNewArchitecture('metal')).toMatchObject({ runs: 'yes', facts: ['C6'] });
  });
});

describe('the same question on Metal, where there are no qualifiers (C7)', () => {
  it('nine cells', () => {
    const table = METAL_FNS.map((a) => METAL_FNS.map((b) => metalCall(a, b).result));
    expect(table).toEqual([
      ['call', 'dispatch', 'no'],
      ['no', 'no', 'call'],
      ['no', 'no', 'call'],
    ]);
  });
  it('a plain MSL function plays the part of __device__', () => {
    expect(metalCall('kernel', 'msl')).toMatchObject({ result: 'call', executedBy: 'caller-device-thread' });
  });
  it('the host reaches a kernel by dispatching, not by calling', () => {
    expect(metalCall('host', 'kernel')).toMatchObject({ result: 'dispatch', executedBy: 'new-grid' });
  });
  it('a kernel cannot call a kernel', () => {
    expect(metalCall('kernel', 'kernel').result).toBe('no');
    expect(metalCall('kernel', 'kernel').facts).toContain('C7');
  });
});

describe('functionSource: one function out of a file', () => {
  const SRC = [
    '// A file with three functions.',
    '#include <cstdio>',
    'float sq(float x);   // a declaration: not what we want',
    '',
    '// Squares a number.',
    '// Two comment lines belong to it.',
    '__host__ __device__ inline float sq(float x) {',
    '    return x * x;   // } a brace in a comment',
    '}',
    '',
    '__global__',
    'void squareKernel(float* a, int n)',
    '{',
    '    int i = blockIdx.x * blockDim.x + threadIdx.x;',
    '    if (i < n) { a[i] = sq(a[i]); }',
    '}',
    '',
    'int main() {',
    '    printf("sq(2) = %f {\\n", sq(2));',
    '    squareKernel<<<1, 4>>>(nullptr, 0);',
    '    return 0;',
    '}',
  ].join('\n');
  it('finds a definition and skips the declaration and the call sites', () => {
    const f = functionSource(SRC, 'sq')!;
    expect(f.firstLine).toBe(5);
    expect(f.lastLine).toBe(9);
    expect(f.text.split('\n')).toHaveLength(5);
    expect(f.quals).toBe('host-device');
  });
  it('handles a qualifier and a brace on lines of their own', () => {
    const f = functionSource(SRC, 'squareKernel')!;
    expect([f.firstLine, f.lastLine]).toEqual([11, 16]);
    expect(f.quals).toBe('global');
    expect(f.text.startsWith('__global__\nvoid squareKernel')).toBe(true);
  });
  it('is not fooled by braces in strings', () => {
    const f = functionSource(SRC, 'main')!;
    expect([f.firstLine, f.lastLine]).toEqual([18, 22]);
    expect(f.quals).toBe('none');
  });
  it('returns null for a function the file does not define', () => {
    expect(functionSource(SRC, 'cube')).toBeNull();
    expect(functionSource(SRC, 'printf')).toBeNull();
    expect(functionSource('// could not load labs/x', 'sq')).toBeNull();
  });
  it('reads the qualifiers the model gives the program', () => {
    const q: Qualifiers[] = ['none', 'global', 'host-device'];
    expect([functionSource(SRC, 'main')!.quals, functionSource(SRC, 'squareKernel')!.quals, functionSource(SRC, 'sq')!.quals]).toEqual(q);
  });
});
