// Rule model for chapter 2, section 2.7 (and the qualifier table of section 2.5):
// which compilation sees each function of a small program, what each backend's
// toolchain does with it stage by stage, and who may call a function and who
// then executes it. It runs the "Follow one source file" stepper, the "Who may
// call it, who runs it" classification, the Metal call table and the section's
// questions.
//
// Every rule carries the id of the fact it comes from, so the lesson can cite it.
// Ids without a prefix are items of notes/today/ch02-facts.md; "ch05 B4" and the
// like are items of notes/today/ch05-facts.md; "Fig. 2.11" and page numbers are
// the book (notes/ch02.md).
//  - Fig. 2.11, pp. 35-36: the three qualifiers, where each function may be called
//    from, where it runs and which thread runs it; __host__ is the default;
//    __host__ __device__ together give two compiled versions.
//  - A7: the same table in NVIDIA's current guide, and the __CUDA_ARCH__ macro.
//  - A13, ch05 B4, C7: a kernel launched from device code (dynamic parallelism):
//    CUDA yes, HIP no, Metal none documented.
//  - A14: nvcc splits host from device code; device code becomes PTX, then a cubin
//    through ptxas; both travel in a fatbin; the device driver compiles PTX at run
//    time only when it has to.
//  - B4, B6: amdclang++ makes one host pass and one device pass per architecture
//    named with --offload-arch; the machine code is embedded; nothing like PTX is
//    shipped by default.
//  - C1, C4, C6, C7: on Metal the kernel is a separate file in another language;
//    a library is built at build time or at run time; creating the pipeline state
//    finishes compilation for the GPU at hand; there are no qualifiers.
//
// What this leaves out:
//  - Linking, and the host compiler's own stages.
//  - Caches of just-in-time results, and the first-call cost of initialising a runtime.
//  - Run-time compilation as an API (nvrtc, hiprtc).
//  - CUDA Tile kernels (__tile__, __tile_global__), which add two rows to NVIDIA's
//    qualifier table (A7) and are not used in this course.
//  - Whether a cubin built for one GPU architecture also runs on another. The
//    facts file has no statement on it, so the model does not answer it.
//  - The host-side launch stub a compiler generates for a kernel: a kernel is
//    counted as device code only.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.

export type Backend = 'cuda' | 'hip' | 'metal';
export const BACKENDS: readonly Backend[] = ['cuda', 'hip', 'metal'];
export const BACKEND_NAME: Record<Backend, string> = { cuda: 'CUDA', hip: 'HIP', metal: 'Metal' };

// ---------------------------------------------------------------- functions and qualifiers
/** The qualifier sets a function can carry. 'none' and 'host' mean the same thing (Fig. 2.11). */
export type Qualifiers = 'none' | 'host' | 'device' | 'global' | 'host-device';
export const QUALIFIERS: readonly Qualifiers[] = ['none', 'host', 'device', 'global', 'host-device'];
export const QUALIFIER_TEXT: Record<Qualifiers, string> = {
  none: 'no qualifier',
  host: '__host__',
  device: '__device__',
  global: '__global__',
  'host-device': '__host__ __device__',
};
/** The side of the program a compiled function belongs to. */
export type Space = 'host' | 'device';

/** Where a function with these qualifiers is compiled for: the host, the device, or both. */
export function spacesOf(q: Qualifiers): Space[] {
  if (q === 'none' || q === 'host') return ['host'];
  if (q === 'host-device') return ['host', 'device'];
  return ['device'];
}

export interface Fn {
  name: string;
  quals: Qualifiers;
  does: string;
}
/** The vector addition program of the chapter, with one shared helper added. */
export const PROGRAM: readonly Fn[] = [
  { name: 'main', quals: 'none', does: 'fills the arrays and calls vecAdd' },
  { name: 'vecAdd', quals: 'none', does: 'the host stub: allocates, copies, launches the kernel, copies back' },
  { name: 'vecAddKernel', quals: 'global', does: 'the kernel: one addition per thread' },
  { name: 'add1', quals: 'host-device', does: 'a one-line helper called by the kernel and by the CPU reference' },
];

export interface Placement {
  space: Space;
  /** The compiler, or the compiler pass, that sees the function. */
  compiler: string;
  /** On Metal: the file the function has to be written in. */
  file?: string;
  /** On Metal, for a function wanted on both sides: it is a second, hand-written copy. */
  writtenAgain?: boolean;
  facts: string[];
}

const COMPILER: Record<Backend, Record<Space, { compiler: string; facts: string[] }>> = {
  cuda: {
    host: { compiler: 'the host C++ compiler, called by nvcc', facts: ['A14'] },
    device: { compiler: "nvcc's GPU compiler", facts: ['A14'] },
  },
  hip: {
    host: { compiler: 'amdclang++, host pass', facts: ['B4', 'B6'] },
    device: { compiler: 'amdclang++, one device pass per architecture', facts: ['B4', 'B6'] },
  },
  metal: {
    host: { compiler: 'the host compiler', facts: ['C1'] },
    device: { compiler: 'the Metal compiler', facts: ['C1', 'C6'] },
  },
};
export const METAL_FILE: Record<Space, string> = { host: 'launch.mm', device: 'kernel.metal' };

/**
 * Which compilation(s) see a function. On CUDA and HIP the qualifiers decide, and
 * __host__ __device__ gives two. Metal has no qualifiers (C7): the same intent is
 * expressed by which file the function is written in, and a function wanted on
 * both sides is written twice.
 */
export function placement(fn: Fn, backend: Backend): Placement[] {
  const spaces = spacesOf(fn.quals);
  return spaces.map((space, k): Placement => {
    const c = COMPILER[backend][space];
    if (backend !== 'metal') return { space, compiler: c.compiler, facts: fn.quals === 'host-device' ? [...c.facts, 'A7'] : c.facts };
    return {
      space,
      compiler: c.compiler,
      file: METAL_FILE[space],
      ...(spaces.length > 1 && k > 0 ? { writtenAgain: true } : {}),
      facts: spaces.length > 1 ? [...c.facts, 'C7'] : c.facts,
    };
  });
}

// ---------------------------------------------------------------- stages
export type When = 'build' | 'run';
export interface Chip {
  fn: string;
  /** Which compiled version of the function this is, or null while it is still source. */
  copy: Space | null;
}
export interface Box {
  id: string;
  /** Boxes with the same group are drawn inside one frame with this title. */
  group?: string;
  title: string;
  /** What the contents are at this point: source, PTX, machine code for an architecture. */
  form: string;
  chips: Chip[];
  /** 'fail': this cannot run on the device at hand. 'unused': present but not what runs. */
  state?: 'ok' | 'fail' | 'unused';
}
export interface Stage {
  id: string;
  when: When;
  /** The tool or call that does this stage. */
  tool: string;
  title: string;
  text: string;
  boxes: Box[];
  facts: string[];
}
export interface Options {
  /** CUDA: the fatbin holds a cubin that suits the GPU the program runs on. Default true. */
  cubinFits?: boolean;
  /** HIP: the architecture names given to --offload-arch. Default none. */
  archs?: string[];
  /** HIP: the architecture of the GPU the program runs on. */
  deviceArch?: string;
  /** Metal: when the library is built. Default 'run', which is what the course's labs do. */
  metalLibrary?: 'build' | 'run';
}

const chipsFor = (program: readonly Fn[], space: Space, compiled: boolean): Chip[] =>
  program.filter((f) => spacesOf(f.quals).includes(space)).map((f) => ({ fn: f.name, copy: compiled ? space : null }));
const sourceChips = (program: readonly Fn[]): Chip[] => program.map((f) => ({ fn: f.name, copy: null }));
/** Trimmed, non-empty, no repeats, in the order typed. */
export const cleanArchs = (archs: string[] = []): string[] => [...new Set(archs.map((a) => a.trim()).filter(Boolean))];

function cudaStages(o: Options, program: readonly Fn[]): Stage[] {
  const host = chipsFor(program, 'host', true);
  const dev = chipsFor(program, 'device', true);
  const fits = o.cubinFits !== false;
  const exe = 'the executable';
  return [
    {
      id: 'source', when: 'build', tool: 'you', title: 'One source file',
      text: 'One .cu file holds host code and device code side by side. Only the qualifiers say which is which.',
      boxes: [{ id: 'cu', title: 'vecadd.cu', form: 'CUDA C++ source', chips: sourceChips(program) }],
      facts: ['A14'],
    },
    {
      id: 'split', when: 'build', tool: 'nvcc', title: 'nvcc splits the file',
      text: 'nvcc separates device code from host code and sends each to its own compiler. A function marked __host__ __device__ goes both ways, so from here on there are two of it.',
      boxes: [
        { id: 'host-src', title: 'host code', form: 'plain C++', chips: chipsFor(program, 'host', true) },
        { id: 'dev-src', title: 'device code', form: 'C++ for the GPU compiler', chips: chipsFor(program, 'device', true) },
      ],
      facts: ['A14', 'A7'],
    },
    {
      id: 'compile', when: 'build', tool: 'host compiler and GPU compiler', title: 'Host object and PTX',
      text: 'The host compiler turns the host code into an ordinary object file. The GPU compiler turns the device code into PTX: assembly for a virtual GPU instruction set, not yet for any real one.',
      boxes: [
        { id: 'host-obj', title: 'host object', form: 'CPU machine code', chips: host },
        { id: 'ptx', title: 'PTX', form: 'virtual GPU instruction set', chips: dev },
      ],
      facts: ['A14'],
    },
    {
      id: 'cubin', when: 'build', tool: 'ptxas', title: 'ptxas makes a cubin',
      text: 'ptxas compiles the PTX to a cubin: machine code for a real GPU instruction set. By default the target is the earliest architecture the toolkit supports. The PTX is kept.',
      boxes: [
        { id: 'host-obj', title: 'host object', form: 'CPU machine code', chips: host },
        { id: 'ptx', title: 'PTX', form: 'virtual GPU instruction set', chips: dev },
        { id: 'cubin', title: 'cubin', form: 'GPU machine code for one architecture', chips: dev },
      ],
      facts: ['A14'],
    },
    {
      id: 'fatbin', when: 'build', tool: 'nvcc', title: 'A fatbin inside the executable',
      text: 'PTX and cubin are packed together in a fatbin, and the fatbin travels inside the executable beside the host machine code. This is what you ship.',
      boxes: [
        { id: 'host-code', group: exe, title: 'host code', form: 'CPU machine code', chips: host },
        { id: 'ptx', group: exe, title: 'fatbin: PTX', form: 'virtual GPU instruction set', chips: dev },
        { id: 'cubin', group: exe, title: 'fatbin: cubin', form: 'GPU machine code for one architecture', chips: dev },
      ],
      facts: ['A14'],
    },
    {
      id: 'run', when: 'run', tool: fits ? 'the CUDA runtime' : 'the device driver', title: fits ? 'Run time: the cubin is loaded' : 'Run time: the driver compiles the PTX',
      text: fits
        ? 'The host code runs as a normal process. The fatbin holds a cubin that suits this GPU, so it is loaded as it is and nothing is compiled.'
        : 'The host code runs as a normal process. No cubin in the fatbin suits this GPU, so the device driver compiles the PTX for it, just in time, and keeps a copy of the result. This is how a binary built today can run on a GPU that did not exist at build time.',
      boxes: [
        { id: 'host-code', title: 'host process', form: 'CPU machine code, running', chips: host, state: 'ok' },
        { id: 'ptx', title: 'fatbin: PTX', form: fits ? 'not needed this time' : 'compiled now by the driver', chips: dev, state: fits ? 'unused' : 'ok' },
        fits
          ? { id: 'cubin', title: 'fatbin: cubin', form: 'loaded into device memory', chips: dev, state: 'ok' }
          : { id: 'jit', title: 'driver output', form: 'GPU machine code for this GPU', chips: dev, state: 'ok' },
      ],
      facts: ['A14', 'A5'],
    },
  ];
}

function hipStages(o: Options, program: readonly Fn[]): Stage[] {
  const host = chipsFor(program, 'host', true);
  const dev = chipsFor(program, 'device', true);
  const archs = cleanArchs(o.archs);
  const device = (o.deviceArch ?? '').trim();
  const exe = 'the executable';
  const archBoxes = (group?: string): Box[] =>
    archs.map((a) => ({ id: 'arch-' + a, ...(group ? { group } : {}), title: 'device code: ' + a, form: 'GPU machine code for ' + a, chips: dev }));
  const listed = archs.includes(device);
  const none = archs.length === 0;
  return [
    {
      id: 'source', when: 'build', tool: 'you', title: 'One source file',
      text: 'The same single file as on CUDA, with the same qualifiers. It includes hip/hip_runtime.h, which HIP requires for the built-in variables.',
      boxes: [{ id: 'src', title: 'vecadd.hip', form: 'HIP C++ source', chips: sourceChips(program) }],
      facts: ['B4', 'ch05 B6'],
    },
    {
      id: 'host-pass', when: 'build', tool: 'amdclang++', title: 'Host pass',
      text: 'amdclang++ makes one pass over the file for the host code. There is no second compiler: one Clang-based compiler does both halves.',
      boxes: [{ id: 'host-obj', title: 'host object', form: 'CPU machine code', chips: host }],
      facts: ['B4', 'B6'],
    },
    {
      id: 'device-passes', when: 'build', tool: 'amdclang++ --offload-arch', title: none ? 'Device passes: no architecture named yet' : `Device ${archs.length === 1 ? 'pass' : 'passes'}: ${archs.join(', ')}`,
      text: none
        ? 'The compiler makes one more pass over the device code for each GPU architecture named with --offload-arch. Type at least one name above to see it.'
        : `The compiler makes one more pass over the device code for each GPU architecture named with --offload-arch: here ${archs.length}. Each pass ends in machine code for that architecture. There is no stage like PTX in between.`,
      boxes: [{ id: 'host-obj', title: 'host object', form: 'CPU machine code', chips: host }, ...archBoxes()],
      facts: ['B4', 'B6'],
    },
    {
      id: 'embed', when: 'build', tool: 'amdclang++', title: 'Device code embedded',
      text: 'The compiled device code is embedded into the host object file. What ships is machine code for the named architectures and nothing a driver could compile later.',
      boxes: [{ id: 'host-code', group: exe, title: 'host code', form: 'CPU machine code', chips: host }, ...archBoxes(exe)],
      facts: ['B6'],
    },
    {
      id: 'run', when: 'run', tool: 'the HIP runtime',
      title: device === '' ? 'Run time: which GPU?' : listed ? `Run time on ${device}: runs` : `Run time on ${device}: not runnable`,
      text:
        device === ''
          ? 'Type the architecture of the GPU the program runs on to see what happens there.'
          : listed
            ? `The binary carries machine code for ${device}, so the kernel is loaded and runs. Nothing is compiled at run time.`
            : `The binary carries no machine code for ${device}. Machine code for one AMD GPU architecture runs only on the physical architectures it is compatible with, and there is no portable form to compile from: the fix is to rebuild with ${device} in the list.`,
      boxes: [
        { id: 'host-code', title: 'host process', form: 'CPU machine code, running', chips: host, state: 'ok' },
        ...archs.map((a): Box => ({
          id: 'arch-' + a, title: 'device code: ' + a, form: a === device ? 'loaded on this GPU' : 'not for this GPU', chips: dev,
          state: a === device ? 'ok' : 'unused',
        })),
        ...(device !== '' && !listed ? [{ id: 'missing', title: 'device code: ' + device, form: 'not in the binary', chips: [], state: 'fail' } as Box] : []),
      ],
      facts: ['B6'],
    },
  ];
}

function metalStages(o: Options, program: readonly Fn[]): Stage[] {
  const host = chipsFor(program, 'host', true);
  const dev = chipsFor(program, 'device', true);
  const atBuild = o.metalLibrary === 'build';
  return [
    {
      id: 'source', when: 'build', tool: 'you', title: 'Two source files from the start',
      text: 'There is nothing to split. Host code is in an ordinary C++ or Objective-C++ file. The kernel is in a .metal file, written in Metal Shading Language. A helper wanted on both sides has to be written in both files.',
      boxes: [
        { id: 'host-src', title: METAL_FILE.host, form: 'Objective-C++ source', chips: chipsFor(program, 'host', true) },
        { id: 'dev-src', title: METAL_FILE.device, form: 'Metal Shading Language source', chips: chipsFor(program, 'device', true) },
      ],
      facts: ['C1', 'C7'],
    },
    {
      id: 'library', when: atBuild ? 'build' : 'run', tool: atBuild ? 'xcrun metal' : 'newLibraryWithSource:options:error:',
      title: atBuild ? 'Build time: a .metallib' : 'Run time: a library from source',
      text: atBuild
        ? 'The metal compiler turns kernel.metal into an intermediate representation and links it into a .metallib, which the app loads when it runs. The host file is compiled by the host compiler as usual.'
        : 'The host file was compiled by the host compiler as usual and carries the kernel source as text. When the program runs, newLibraryWithSource compiles that text into a library. This is the route the labs in this course take.',
      boxes: [
        { id: 'host-code', title: 'host executable', form: 'CPU machine code', chips: host },
        { id: 'library', title: atBuild ? 'default.metallib' : 'library object', form: 'intermediate representation, not for any one GPU', chips: dev },
      ],
      facts: ['C6'],
    },
    {
      id: 'function', when: 'run', tool: 'newFunctionWithName:', title: 'Run time: a function object',
      text: 'The app asks the library for the kernel by name and gets a function object. It stands for the MSL function and is not executable code yet.',
      boxes: [
        { id: 'host-code', title: 'host process', form: 'CPU machine code, running', chips: host, state: 'ok' },
        { id: 'function', title: 'function object', form: 'a proxy for the kernel, not executable', chips: dev.filter((c) => program.find((f) => f.name === c.fn)?.quals === 'global') },
      ],
      facts: ['C6', 'C5'],
    },
    {
      id: 'pipeline', when: 'run', tool: 'newComputePipelineStateWithFunction:error:', title: 'Run time: the pipeline state',
      text: 'Creating the pipeline state finishes compiling the kernel for this specific GPU. It happens on the machine that runs the program, in both routes, and it takes a while: do it once, outside anything you time.',
      boxes: [
        { id: 'host-code', title: 'host process', form: 'CPU machine code, running', chips: host, state: 'ok' },
        { id: 'pipeline', title: 'pipeline state', form: 'GPU machine code for this GPU', chips: dev, state: 'ok' },
      ],
      facts: ['C6', 'C4'],
    },
  ];
}

/** The documented stages of one backend's toolchain, with where each function of the program is after each. */
export function stages(backend: Backend, options: Options = {}, program: readonly Fn[] = PROGRAM): Stage[] {
  if (backend === 'cuda') return cudaStages(options, program);
  if (backend === 'hip') return hipStages(options, program);
  return metalStages(options, program);
}

/** Does any stage after shipping compile device code on the machine that runs the program? */
export function compilesOnUsersMachine(backend: Backend): 'always' | 'when-needed' | 'never' {
  if (backend === 'metal') return 'always';
  return backend === 'hip' ? 'never' : 'when-needed';
}

// ---------------------------------------------------------------- a GPU architecture that appears later
export interface Artefacts {
  /** CUDA: the shipped fatbin carries PTX (the default, A14). */
  ptx?: boolean;
  /** HIP: the architectures named at build time. */
  archs?: string[];
  /** HIP: the name of the architecture that appeared later. */
  newArch?: string;
}
export interface NewArchVerdict {
  /** Can the shipped binary's kernel be compiled for the new architecture without a rebuild? */
  runs: 'yes' | 'no' | 'not-documented';
  why: string;
  facts: string[];
}
/** The three documented answers to "a new GPU architecture appears after you ship". */
export function canRunOnNewArchitecture(backend: Backend, a: Artefacts = {}): NewArchVerdict {
  if (backend === 'cuda')
    return a.ptx === false
      ? { runs: 'not-documented', why: 'Without PTX there is nothing for the driver to compile. Whether a cubin built for an older architecture runs is not covered by the facts checked for this course.', facts: ['A14'] }
      : { runs: 'yes', why: 'The fatbin carries PTX, and the device driver can compile PTX just in time for a target that did not exist at build time.', facts: ['A14'] };
  if (backend === 'hip') {
    const named = cleanArchs(a.archs).includes((a.newArch ?? '').trim()) && (a.newArch ?? '').trim() !== '';
    return named
      ? { runs: 'yes', why: 'The architecture was named at build time, so its machine code is in the binary.', facts: ['B6'] }
      : { runs: 'no', why: 'A HIP binary holds machine code for the architectures named with --offload-arch and no portable form a driver could compile later. A new architecture needs a rebuild.', facts: ['B6'] };
  }
  return { runs: 'yes', why: 'Compilation for the specific GPU always happens when the pipeline state is created, on the machine that runs the program.', facts: ['C6'] };
}

// ---------------------------------------------------------------- who may call, who runs (book Fig. 2.11)
export type Caller = 'host-fn' | 'kernel' | 'device-fn';
export const CALLERS: readonly Caller[] = ['host-fn', 'kernel', 'device-fn'];
export const CALLER_TEXT: Record<Caller, string> = {
  'host-fn': 'a host function',
  kernel: 'a kernel',
  'device-fn': 'a device function',
};
/** The four columns of the classification: no qualifier and __host__ are one column, as in the book's table. */
export const CALLEES: readonly Qualifiers[] = ['none', 'global', 'device', 'host-device'];
export const CALLEE_TEXT: Record<Qualifiers, string> = {
  none: 'no qualifier, or __host__',
  host: '__host__',
  global: '__global__',
  device: '__device__',
  'host-device': '__host__ __device__',
};

export type Executor = 'caller-host-thread' | 'caller-device-thread' | 'new-grid';
export const EXECUTORS: readonly Executor[] = ['caller-host-thread', 'caller-device-thread', 'new-grid'];
export const EXECUTOR_TEXT: Record<Executor, string> = {
  'caller-host-thread': 'the calling host thread',
  'caller-device-thread': 'the calling device thread',
  'new-grid': 'a new grid of device threads',
};
/** 'dynamic-parallelism': legal only where the backend lets device code launch a kernel. */
export type Legality = 'legal' | 'illegal' | 'dynamic-parallelism';

export interface DynamicParallelism {
  /** true, false, or null where the vendor documents nothing like it. */
  supported: boolean | null;
  text: string;
  fact: string;
}
export const DYNAMIC_PARALLELISM: Record<Backend, DynamicParallelism> = {
  cuda: { supported: true, text: 'supported: the current version is called CDP2', fact: 'A13' },
  hip: { supported: false, text: 'not supported: a kernel cannot be called from the device', fact: 'ch05 B4' },
  metal: { supported: null, text: 'none documented: MSL has no dispatch call and kernels cannot call kernels', fact: 'C7' },
};

const callerSpace = (c: Caller): Space => (c === 'host-fn' ? 'host' : 'device');

/** Who executes the callee when the call is allowed by the book's table; null when it is not. */
export function executedBy(caller: Caller, callee: Qualifiers): Executor | null {
  const from = callerSpace(caller);
  if (callee === 'global') return 'new-grid';
  if (!spacesOf(callee).includes(from)) return null;
  return from === 'host' ? 'caller-host-thread' : 'caller-device-thread';
}

export interface CallRule {
  caller: Caller;
  callee: Qualifiers;
  backend: Backend;
  /** False on Metal, where these qualifiers do not exist (C7): use metalCall() there. */
  applies: boolean;
  /** What the book's table says. */
  book: Legality;
  /** Whether the call compiles on this backend; null where the rule does not apply. */
  legal: boolean | null;
  /** Who executes the callee when the call is legal. */
  executedBy: Executor | null;
  /** The row of Fig. 2.11 the answer is read from. */
  row: string;
  why: string;
  facts: string[];
}

const ROW: Record<Qualifiers, string> = {
  none: '__host__ (default): callable from the host, executed on the host by the caller host thread',
  host: '__host__ (default): callable from the host, executed on the host by the caller host thread',
  global: '__global__: callable from the host (or the device), executed on the device by a new grid of device threads',
  device: '__device__: callable from the device, executed on the device by the caller device thread',
  'host-device': '__host__ and __device__ together: one version compiled for each side (p. 36)',
};

/** One cell of the caller by callee table: book Fig. 2.11, resolved for one backend. */
export function legalCall(caller: Caller, callee: Qualifiers, backend: Backend = 'cuda'): CallRule {
  const from = callerSpace(caller);
  const who = executedBy(caller, callee);
  const base = { caller, callee, backend, row: ROW[callee] };
  const applies = backend !== 'metal';
  const finish = (book: Legality, why: string, facts: string[]): CallRule => {
    const legal = !applies ? null : book === 'legal' ? true : book === 'illegal' ? false : DYNAMIC_PARALLELISM[backend].supported === true;
    return {
      ...base,
      applies,
      book,
      legal,
      executedBy: book === 'illegal' ? null : who,
      why: applies ? why : 'Metal has no such qualifiers: every function in a .metal file is GPU code, and host functions live in another file and another language.',
      facts: applies ? facts : ['C7'],
    };
  };
  if (callee === 'global') {
    if (from === 'host')
      return finish('legal', 'Calling a kernel from host code is the launch: it starts a new grid of device threads, and every one of them runs the kernel.', ['Fig. 2.11', 'A7']);
    return finish(
      'dynamic-parallelism',
      'Device code launching a kernel is dynamic parallelism. Where the backend supports it the call starts a new grid; where it does not, the call is an error.',
      ['Fig. 2.11', 'A13', 'ch05 B4', 'C7'],
    );
  }
  if (callee === 'none' || callee === 'host') {
    if (from === 'host') return finish('legal', 'An ordinary C++ call: the calling host thread runs the function.', ['Fig. 2.11', 'A7']);
    return finish(
      'illegal',
      'No qualifier means __host__. A host function exists only as CPU code, so device code cannot call it. This is the usual first compile error: mark the helper __device__, or __host__ __device__ if the host needs it too.',
      ['Fig. 2.11', 'A7'],
    );
  }
  if (callee === 'device') {
    if (from === 'device') return finish('legal', 'The calling device thread runs the function itself. No thread is created.', ['Fig. 2.11', 'A7']);
    return finish('illegal', 'A __device__ function exists only as GPU code. Host code cannot call it.', ['Fig. 2.11', 'A7']);
  }
  return finish(
    'legal',
    from === 'host'
      ? 'The compiler made two versions of this function. A host caller gets the host version, run by the calling host thread.'
      : 'The compiler made two versions of this function. A device caller gets the device version, run by the calling device thread.',
    ['p. 36', 'A7'],
  );
}

/** What a reader may answer for one cell: it does not compile, or who executes the callee. */
export type Answer = 'error' | Executor;
export const ANSWERS: readonly Answer[] = ['error', ...EXECUTORS];
export const ANSWER_TEXT: Record<Answer, string> = { error: 'compile error', ...EXECUTOR_TEXT };

/**
 * Marks an answer against the book's table. The dynamic-parallelism cells have two
 * defensible answers (an error without the feature, a new grid with it): both are
 * 'depends', and anything else is wrong.
 */
export function grade(caller: Caller, callee: Qualifiers, answer: Answer): 'right' | 'wrong' | 'depends' {
  const rule = legalCall(caller, callee, 'cuda');
  if (rule.book === 'dynamic-parallelism') return answer === 'error' || answer === 'new-grid' ? 'depends' : 'wrong';
  if (rule.book === 'illegal') return answer === 'error' ? 'right' : 'wrong';
  return answer === rule.executedBy ? 'right' : 'wrong';
}

// ---------------------------------------------------------------- the same question on Metal
export type MetalFn = 'host' | 'kernel' | 'msl';
export const METAL_FNS: readonly MetalFn[] = ['host', 'kernel', 'msl'];
export const METAL_FN_TEXT: Record<MetalFn, string> = {
  host: 'a host function (in the .mm or .cpp file)',
  kernel: 'a kernel function (kernel void, in the .metal file)',
  msl: 'a plain function in the .metal file',
};
export interface MetalCall {
  caller: MetalFn;
  callee: MetalFn;
  /** 'dispatch': not a call, but the host can run it through a pipeline and an encoder. */
  result: 'call' | 'dispatch' | 'no';
  executedBy: Executor | null;
  why: string;
  facts: string[];
}
/** Who can reach what on Metal, where the file a function is written in plays the part of the qualifier. */
export function metalCall(caller: MetalFn, callee: MetalFn): MetalCall {
  const cell = (result: MetalCall['result'], executor: Executor | null, why: string, facts: string[]): MetalCall => ({
    caller, callee, result, executedBy: executor, why, facts,
  });
  if (caller === 'host') {
    if (callee === 'host') return cell('call', 'caller-host-thread', 'An ordinary call.', []);
    if (callee === 'kernel')
      return cell('dispatch', 'new-grid', 'Not a function call: the host builds a pipeline state from the kernel and dispatches a grid of threads through an encoder.', ['C1', 'C4']);
    return cell('no', null, 'Only functions with an attribute such as kernel are visible to the app.', ['C1', 'C7']);
  }
  if (callee === 'host') return cell('no', null, 'MSL code is a separate file in a separate language. It cannot call into the host program, and no function is compiled for both sides.', ['C7']);
  if (callee === 'kernel') return cell('no', null, 'Kernel functions cannot be called by other shader functions: it is a compile error. Nothing like dynamic parallelism is documented.', ['C1', 'C7']);
  return cell('call', 'caller-device-thread', 'A function with no attribute plays the part of a __device__ helper: the calling GPU thread runs it.', ['C7']);
}

// ---------------------------------------------------------------- one function out of a source file
export interface FunctionSource {
  /** The definition, with the comment lines directly above it. */
  text: string;
  /** 1-based line of the first line of `text` in the file. */
  firstLine: number;
  lastLine: number;
  /** Read from the definition's own qualifiers. 'none' in a .metal or .mm file, which have none. */
  quals: Qualifiers;
}

/** Removes comments and string literals from one line, so that braces inside them are not counted. */
function codeOnly(line: string, state: { block: boolean }): string {
  let out = '';
  let i = 0;
  while (i < line.length) {
    if (state.block) {
      const end = line.indexOf('*/', i);
      if (end < 0) return out;
      state.block = false;
      i = end + 2;
    } else if (line.startsWith('//', i)) return out;
    else if (line.startsWith('/*', i)) {
      state.block = true;
      i += 2;
    } else if (line[i] === '"' || line[i] === "'") {
      const q = line[i];
      i++;
      while (i < line.length && line[i] !== q) i += line[i] === '\\' ? 2 : 1;
      i++;
    } else out += line[i++];
  }
  return out;
}

/**
 * Finds the definition of a function in C-family source (CUDA C++, MSL or
 * Objective-C++) and returns its text. Lessons use it to show one function of a
 * lab file without copying it. Null when the file does not define the function.
 */
export function functionSource(source: string, name: string): FunctionSource | null {
  const lines = source.replace(/\r\n/g, '\n').split('\n');
  const state = { block: false };
  const code = lines.map((l) => codeOnly(l, state));
  const depthBefore: number[] = [];
  let depth = 0;
  for (const c of code) {
    depthBefore.push(depth);
    for (const ch of c) depth += ch === '{' ? 1 : ch === '}' ? -1 : 0;
  }
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const named = new RegExp('(^|[^\\w])' + escaped + '\\s*\\(');
  for (let k = 0; k < lines.length; k++) {
    if (depthBefore[k] !== 0 || !named.test(code[k])) continue;
    // From the name on, the first of ';' and '{' decides: a declaration or a definition.
    let open = -1;
    for (let j = k; j < lines.length; j++) {
      const text = j === k ? code[j].slice(code[j].search(named)) : code[j];
      const semi = text.indexOf(';');
      const brace = text.indexOf('{');
      if (brace >= 0 && (semi < 0 || brace < semi)) open = j;
      if (brace >= 0 || semi >= 0) break;
    }
    if (open < 0) continue;
    // The definition may start above the name: a qualifier or a template header on its own line.
    let first = k;
    while (first > 0 && depthBefore[first - 1] === 0 && code[first - 1].trim() !== '' && !/[;}]\s*$/.test(code[first - 1]) && !code[first - 1].trim().startsWith('#')) first--;
    const head = first;
    while (first > 0 && lines[first - 1].trim().startsWith('//')) first--;
    // The end: the line on which the braces opened at `open` are closed again.
    let d = 0;
    let last = open;
    for (let j = open; j < lines.length; j++) {
      for (const ch of code[j]) d += ch === '{' ? 1 : ch === '}' ? -1 : 0;
      if (d === 0) {
        last = j;
        break;
      }
    }
    const headText = code.slice(head, open + 1).join(' ');
    const isHost = /\b__host__\b/.test(headText);
    const isDevice = /\b__device__\b/.test(headText);
    const quals: Qualifiers = /\b__global__\b/.test(headText) ? 'global' : isHost && isDevice ? 'host-device' : isDevice ? 'device' : isHost ? 'host' : 'none';
    return { text: lines.slice(first, last + 1).join('\n'), firstLine: first + 1, lastLine: last + 1, quals };
  }
  return null;
}
