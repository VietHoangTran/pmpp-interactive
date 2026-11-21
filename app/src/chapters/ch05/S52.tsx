import { useState, type ReactNode } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, TierLegend, Today, type Tier } from '../../components/notes';
import { Code } from '../../components/Code';
import { realBackends, useReports } from '../../lib/reports';
import { href } from '../../lib/router';

// ---------------------------------------------------------------- memory map
// Our own drawing of the device memory model (the book's is Fig. 5.1, p. 98):
// one grid, two blocks, two threads each. Laid out by where things physically
// are: on the chip, off the chip on the device, and on the host.
type Kind = 'thread' | 'host' | 'reg' | 'local' | 'shared' | 'global' | 'const';
interface Box {
  id: string;
  kind: Kind;
  b: number;
  t: number;
  x: number;
  y: number;
  w: number;
  h: number;
}
const VW = 560;
const VH = 512;
const BLOCK_X = [16, 288];
const BLOCK_W = 256;
const COL_W = 112;
const BAR_X = 16;
const BAR_W = 528;
const colX = (b: number, t: number) => BLOCK_X[b] + 12 + t * 120;
const PAIRS: [number, number][] = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
];
// Order matters twice: it is the tab order, and later boxes sit on top of earlier
// ones (the local slices are drawn inside global memory).
const BOXES: Box[] = [
  ...PAIRS.map(([b, t]): Box => ({ id: `t${b}${t}`, kind: 'thread', b, t, x: colX(b, t), y: 62, w: COL_W, h: 42 })),
  ...PAIRS.map(([b, t]): Box => ({ id: `r${b}${t}`, kind: 'reg', b, t, x: colX(b, t), y: 112, w: COL_W, h: 42 })),
  ...[0, 1].map((b): Box => ({ id: `s${b}`, kind: 'shared', b, t: 0, x: BLOCK_X[b] + 12, y: 170, w: BLOCK_W - 24, h: 44 })),
  { id: 'g', kind: 'global', b: 0, t: 0, x: BAR_X, y: 280, w: BAR_W, h: 98 },
  ...PAIRS.map(([b, t]): Box => ({ id: `l${b}${t}`, kind: 'local', b, t, x: colX(b, t), y: 318, w: COL_W, h: 46 })),
  { id: 'c', kind: 'const', b: 0, t: 0, x: BAR_X, y: 390, w: BAR_W, h: 42 },
  { id: 'h', kind: 'host', b: 0, t: 0, x: BAR_X, y: 462, w: BAR_W, h: 42 },
];
const isActor = (x: Box) => x.kind === 'thread' || x.kind === 'host';
const TIER_OF: Record<Exclude<Kind, 'thread' | 'host'>, Tier> = {
  reg: 'reg',
  local: 'global', // local memory is physically part of global memory
  shared: 'shared',
  global: 'global',
  const: 'const',
};

type Access = 'rw' | 'r' | 'none';
function access(actor: Box, mem: Box): Access {
  if (actor.kind === 'host') return mem.kind === 'global' || mem.kind === 'const' ? 'rw' : 'none';
  switch (mem.kind) {
    case 'reg':
    case 'local':
      return actor.b === mem.b && actor.t === mem.t ? 'rw' : 'none';
    case 'shared':
      return actor.b === mem.b ? 'rw' : 'none';
    case 'global':
      return 'rw';
    case 'const':
      return 'r';
    default:
      return 'none';
  }
}

type Look = 'plain' | 'selected' | 'idle' | Access;
function lookOf(x: Box, sel: Box | null): Look {
  if (!sel) return 'plain';
  if (x.id === sel.id) return 'selected';
  if (isActor(x) === isActor(sel)) return 'idle';
  return isActor(x) ? access(x, sel) : access(sel, x);
}

function boxName(x: Box): string {
  switch (x.kind) {
    case 'thread':
      return `Thread ${x.t} of block ${x.b}`;
    case 'host':
      return 'The host';
    case 'reg':
      return `Registers of thread ${x.t}, block ${x.b}`;
    case 'local':
      return `Local memory of thread ${x.t}, block ${x.b}`;
    case 'shared':
      return `Shared memory of block ${x.b}`;
    case 'global':
      return 'Global memory';
    case 'const':
      return 'Constant memory';
  }
}
const LABEL: Record<Kind, (x: Box) => string> = {
  thread: (x) => `thread ${x.t}`,
  host: () => 'host',
  reg: () => 'registers',
  local: () => 'local memory',
  shared: (x) => `shared memory of block ${x.b}`,
  global: () => 'global memory',
  const: () => 'constant memory',
};
// Shown in a box while nothing is selected.
const RESTING: Record<Kind, string> = {
  thread: '',
  host: 'the CPU side of the program',
  reg: 'private',
  local: 'private slice',
  shared: 'one per block',
  global: 'off-chip DRAM, one for everybody',
  const: 'kernels can only read it',
};
function statusText(x: Box, look: Look): string {
  if (look === 'plain' || look === 'idle') return RESTING[x.kind];
  if (look === 'selected') return isActor(x) ? '' : 'selected';
  if (look === 'none') return 'no access';
  if (isActor(x)) return look === 'rw' ? 'reads, writes' : 'reads only';
  return look === 'rw' ? 'read and write' : 'read only';
}

function MapDetail({ sel }: { sel: Box | null }) {
  if (!sel)
    return (
      <>
        <p>
          <b>Nothing selected.</b>
        </p>
        <p style={{ margin: 0 }}>
          Every box is a button. Select a thread or the host to see what it can reach. Select a memory to see who can
          reach it. Select the same box again to clear.
        </p>
      </>
    );
  const other = 1 - sel.t;
  const otherB = 1 - sel.b;
  const title = (
    <p>
      <b>{boxName(sel)}</b>
    </p>
  );
  switch (sel.kind) {
    case 'thread':
      return (
        <>
          {title}
          <p style={{ marginBottom: '0.3em' }}>Reads and writes:</p>
          <ul>
            <li>
              <span className="chip reg">its own registers</span>
            </li>
            <li>
              <span className="chip global">its own local memory</span>, which sits inside global memory
            </li>
            <li>
              <span className="chip shared">the shared memory of block {sel.b}</span>, together with thread {other}
            </li>
            <li>
              <span className="chip global">global memory</span>, together with every other thread and the host
            </li>
          </ul>
          <p>
            Reads, cannot write: <span className="chip const">constant memory</span>
          </p>
          <p style={{ margin: 0 }}>
            Cannot reach: the registers and local memory of any other thread, including thread {other} of its own
            block; the shared memory of block {otherB}.
          </p>
        </>
      );
    case 'host':
      return (
        <>
          {title}
          <p>
            Reads and writes <span className="chip global">global memory</span> and{' '}
            <span className="chip const">constant memory</span>.
          </p>
          <p style={{ margin: 0 }}>
            Cannot reach anything a kernel keeps to itself: no registers, no local memory, no shared memory. Kernels
            cannot write constant memory either, so every result a kernel produces comes back to the host through
            global memory.
          </p>
        </>
      );
    case 'reg':
      return (
        <>
          {title}
          <p>
            Read and written by that one thread. Thread {other} of the same block cannot see them, and neither can the
            host.
          </p>
          <p style={{ margin: 0 }}>
            On the chip, inside the processor, which is why using them is cheap. There is one set per thread: four in
            this picture.
          </p>
        </>
      );
    case 'local':
      return (
        <>
          {title}
          <p>Private to that one thread, exactly like its registers.</p>
          <p style={{ margin: 0 }}>
            Look at where it is drawn, though: inside global memory, off the chip, and about as slow to reach. It holds
            what the thread could not keep in registers: arrays declared in the kernel body, spilled registers and the
            rest of the call stack.
          </p>
        </>
      );
    case 'shared':
      return (
        <>
          {title}
          <p>
            Read and written by both threads of block {sel.b}, and by nobody else. Block {otherB} has its own; the host
            sees neither.
          </p>
          <p style={{ margin: 0 }}>
            On the chip. One copy per block: two in this picture. It is the only on-chip place where one thread can
            leave a value for another.
          </p>
        </>
      );
    case 'global':
      return (
        <>
          {title}
          <p>Read and written by every thread of every block, and by the host.</p>
          <p style={{ margin: 0 }}>
            Off-chip DRAM: the memory whose bandwidth section 5.1 was about. One copy, however many threads there are.
            The four dashed slices inside it are the threads' local memories, each closed to everyone but its owner.
          </p>
        </>
      );
    case 'const':
      return (
        <>
          {title}
          <p>Every thread can read it. No thread can write it. The host can do both.</p>
          <p style={{ margin: 0 }}>
            The device serves it with short latency and high bandwidth; the book explains how in chapter 7.
          </p>
        </>
      );
  }
}

function MemoryMap() {
  const [selId, setSelId] = useState<string | null>(null);
  const [hover, setHover] = useState<string | null>(null);
  const sel = BOXES.find((x) => x.id === selId) ?? null;

  const drawBox = (x: Box) => {
    const look = lookOf(x, sel);
    const status = statusText(x, look);
    const actor = isActor(x);
    const hot = hover === x.id && look !== 'selected';
    const dim = look === 'idle' ? 0.42 : 1;
    const off = look === 'none';
    const wide = x.w > 300;
    let fill: string;
    let stroke: string;
    let ink = 'var(--ink)';
    if (actor) {
      fill = look === 'selected' ? 'var(--ink)' : 'var(--page)';
      stroke = off ? 'var(--ink-3)' : 'var(--ink)';
      if (look === 'selected') ink = 'var(--page)';
    } else {
      const tier = TIER_OF[x.kind as keyof typeof TIER_OF];
      fill = off ? 'none' : `var(--${tier}-wash)`;
      stroke = `var(--${tier})`;
    }
    if (off) ink = 'var(--ink-3)';
    const sw = (look === 'selected' ? 3.5 : look === 'rw' || look === 'r' ? 2.5 : 1.5) + (hot ? 1 : 0);
    const dash = off ? '4 3' : x.kind === 'local' && look !== 'selected' ? '7 3' : undefined;
    const strong = look === 'rw' || look === 'r' || look === 'selected';
    return (
      <g key={x.id} data-id={x.id} data-look={look} opacity={dim}>
        {/* a page-coloured base so the wash reads the same on any background */}
        {!actor && x.kind !== 'local' ? <rect x={x.x} y={x.y} width={x.w} height={x.h} rx="4" fill="var(--page)" /> : null}
        <rect
          x={x.x}
          y={x.y}
          width={x.w}
          height={x.h}
          rx="4"
          fill={fill}
          stroke={stroke}
          strokeWidth={sw}
          strokeDasharray={dash}
          strokeOpacity={off ? 0.55 : 1}
        />
        {wide ? (
          <>
            <text x={x.x + 12} y={x.kind === 'global' ? x.y + 22 : x.y + x.h / 2 + 4.5} fontSize="13" fontWeight="600" style={{ fill: ink }}>
              {LABEL[x.kind](x)}
            </text>
            <text
              x={x.x + x.w - 12}
              y={x.kind === 'global' ? x.y + 22 : x.y + x.h / 2 + 4.5}
              fontSize="11.5"
              textAnchor="end"
              fontWeight={strong ? 600 : 400}
              style={{ fill: strong ? ink : 'var(--ink-2)' }}
            >
              {status}
            </text>
          </>
        ) : (
          <>
            <text x={x.x + 10} y={status ? x.y + 18 : x.y + x.h / 2 + 4.5} fontSize="12.5" fontWeight="600" style={{ fill: ink }}>
              {LABEL[x.kind](x)}
            </text>
            <text
              x={x.x + 10}
              y={x.y + 33}
              fontSize="11"
              fontWeight={strong ? 600 : 400}
              style={{ fill: strong ? ink : off ? 'var(--ink-3)' : 'var(--ink-2)' }}
            >
              {status}
            </text>
          </>
        )}
      </g>
    );
  };

  return (
    <Bench
      title="Memory map"
      task="Select a thread to see what it can reach, or a memory to see who can reach it."
      foot="Boxes show who may read and write what. They are not to scale, and they say nothing about speed. Texture memory and caches are not drawn."
    >
      <div className="cols plot">
        <div>
          <div style={{ position: 'relative', width: '100%', maxWidth: VW }}>
            <svg
              viewBox={`0 0 ${VW} ${VH}`}
              role="img"
              aria-label="Memory map of one grid: two blocks of two threads, each thread with registers, each block with shared memory; below them global memory containing each thread's local memory, constant memory, and the host."
              style={{ width: '100%', display: 'block' }}
            >
              {/* the grid and its two blocks */}
              <rect x="6" y="8" width="548" height="232" rx="6" fill="none" stroke="var(--ink-3)" strokeDasharray="2 4" />
              <text x="18" y="26" fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>
                grid: one launch of a kernel
              </text>
              {[0, 1].map((b) => (
                <g key={b}>
                  <rect x={BLOCK_X[b]} y="34" width={BLOCK_W} height="196" rx="5" fill="var(--page)" stroke="var(--rule)" />
                  <text x={BLOCK_X[b] + 12} y="53" fontSize="12" fontWeight="600" style={{ fill: 'var(--ink-2)' }}>
                    block {b}
                  </text>
                </g>
              ))}
              {/* each thread sits on its own registers */}
              {PAIRS.map(([b, t]) => (
                <line key={`${b}${t}`} x1={colX(b, t) + COL_W / 2} x2={colX(b, t) + COL_W / 2} y1="104" y2="112" stroke="var(--ink-3)" strokeWidth="1.5" />
              ))}
              {/* chip boundary */}
              <line x1="6" x2="554" y1="258" y2="258" stroke="var(--ink-2)" strokeDasharray="6 4" />
              <text x="554" y="252" fontSize="11" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
                above this line: on the GPU chip
              </text>
              <text x="554" y="272" fontSize="11" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
                below: off the chip, in the device's memory
              </text>
              {/* device / host boundary */}
              <line x1="6" x2="554" y1="442" y2="442" stroke="var(--ink-2)" strokeDasharray="6 4" />
              <text x="554" y="455" fontSize="11" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
                below this line: the host
              </text>
              {BOXES.map(drawBox)}
            </svg>
            {BOXES.map((x) => (
              <button
                key={x.id}
                type="button"
                data-pick={x.id}
                aria-label={boxName(x)}
                aria-pressed={selId === x.id}
                onClick={() => setSelId(selId === x.id ? null : x.id)}
                onMouseEnter={() => setHover(x.id)}
                onMouseLeave={() => setHover((h) => (h === x.id ? null : h))}
                style={{
                  position: 'absolute',
                  left: `${(x.x / VW) * 100}%`,
                  top: `${(x.y / VH) * 100}%`,
                  width: `${(x.w / VW) * 100}%`,
                  height: `${(x.h / VH) * 100}%`,
                  background: 'transparent',
                  border: 0,
                  padding: 0,
                  borderRadius: 4,
                }}
              />
            ))}
          </div>
          <div style={{ marginTop: '0.7rem' }}>
            <TierLegend tiers={['reg', 'shared', 'global', 'const']} />
          </div>
        </div>
        <div className="panel" data-map-detail aria-live="polite">
          <MapDetail sel={sel} />
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- where does it live?
// Our own kernel, in the manner of the book's exercise 11 (pp. 120-121).
// Front-end-checked with clang's CUDA mode and run on the CPU emulator against a
// CPU reference, in all three emulator schedules (2026-11-20).
const G = 48;
const T = 256;
const TAPS = 4;
const SMOOTH = `#define TAPS 4

__constant__ float taps[TAPS];      // filter weights, filled in by the host before the launch

__global__ void smooth(const float* in, const float* gain, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float window[TAPS];
    __shared__ float blockGain;
    __shared__ float stage[256];

    stage[threadIdx.x] = (i < n) ? in[i] : 0.0f;
    if (threadIdx.x == 0) blockGain = gain[blockIdx.x];
    __syncthreads();

    for (int t = 0; t < TAPS; ++t) {
        unsigned j = threadIdx.x + t;
        window[t] = (j < blockDim.x) ? stage[j] : 0.0f;
    }
    float acc = 0.0f;
    for (int t = 0; t < TAPS; ++t) {
        acc += taps[t] * window[t];
    }
    if (i < n) out[i] = blockGain * acc;
}

void launchSmooth(const float* in, const float* gain, float* out, int n) {
    smooth<<<48, 256>>>(in, gain, out, n);      // 48 blocks of 256 threads
}`;

type Mem = 'register' | 'local' | 'shared' | 'global' | 'constant';
type Scope = 'thread' | 'block' | 'grid';
const MEMS: [Mem, string][] = [
  ['register', 'a register'],
  ['local', 'local memory'],
  ['shared', 'shared memory'],
  ['global', 'global memory'],
  ['constant', 'constant memory'],
];
const SCOPES: [Scope, string][] = [
  ['thread', 'one thread'],
  ['block', 'the threads of one block'],
  ['grid', 'every thread of every block'],
];

interface Decl {
  /** The identifier, as spelled in the code. */
  id: string;
  /** What the row asks about. */
  what: ReactNode;
  decl: string;
  mem: Mem;
  scope: Scope;
  copies: number;
  /** Colour the identifier takes once placed. */
  tier: Tier;
  memWhy: Partial<Record<Mem, ReactNode>>;
  scopeWhy: Partial<Record<Scope, ReactNode>>;
  done: ReactNode;
}

const NOT_SHARED = (
  <>
    Nothing is shared unless it says <code>__shared__</code>. Without the qualifier, a variable declared in the kernel
    body is private to the thread.
  </>
);
const DECLS: Decl[] = [
  {
    id: 'taps',
    what: (
      <>
        the array <code>taps</code>
      </>
    ),
    decl: '__constant__ float taps[TAPS];',
    mem: 'constant',
    scope: 'grid',
    copies: 1,
    tier: 'const',
    memWhy: {
      global: (
        <>
          Close. The book says constant variables are stored in global memory but cached. The memory type this
          declaration selects is still constant memory, and that is what makes it read-only for kernels.
        </>
      ),
      register: <>Registers hold scalars declared inside a kernel. This is declared outside any function, with a qualifier.</>,
      local: <>Local memory holds arrays declared inside a kernel. This one is declared outside any function, with a qualifier.</>,
      shared: (
        <>
          Shared memory is per block and needs <code>__shared__</code>. The qualifier here is a different one.
        </>
      ),
    },
    scopeWhy: {
      thread: <>No thread owns it. Every thread reads the same four weights.</>,
      block: <>Nothing ties it to a block. It is declared at file scope, and every thread of every grid can read it.</>,
    },
    done: <>One array for the whole application, readable by every thread, writable only by the host.</>,
  },
  {
    id: 'in',
    what: (
      <>
        the array <code>in</code> points to
      </>
    ),
    decl: 'const float* in',
    mem: 'global',
    scope: 'grid',
    copies: 1,
    tier: 'global',
    memWhy: {
      register: (
        <>
          The pointer itself is a per-thread scalar, true. The row asks about the array it points to, which the host
          allocated in global memory.
        </>
      ),
      constant: (
        <>
          <code>const float*</code> is a C++ promise that this kernel will not write through the pointer. CUDA's
          constant memory is something else and needs <code>__constant__</code>.
        </>
      ),
      local: <>Local memory holds a thread's private arrays. This array came from the host and is the same for every thread.</>,
      shared: <>Shared memory belongs to one block and is filled by the kernel itself. This array arrives from outside.</>,
    },
    scopeWhy: {
      thread: <>Every thread receives the same address, so every thread reaches the same array.</>,
      block: (
        <>
          Every block receives the same address. Each block happens to read its own slice here, but nothing stops a
          thread reading any element.
        </>
      ),
    },
    done: <>One array in global memory. All 12,288 threads receive its address, and every load from it is global-memory traffic.</>,
  },
  {
    id: 'i',
    what: <code>i</code>,
    decl: 'int i = ...;',
    mem: 'register',
    scope: 'thread',
    copies: G * T,
    tier: 'reg',
    memWhy: {
      local: (
        <>
          Local memory is for what cannot go in a register, arrays above all. <code>i</code> is a scalar declared in
          the kernel body, so by the book's rule it gets a register.
        </>
      ),
      shared: NOT_SHARED,
      global: (
        <>
          <code>i</code> is not data passed in from outside. Each thread computes it from its own indices and keeps it
          in a register.
        </>
      ),
      constant: (
        <>
          Constant memory needs <code>__constant__</code> outside any function, and kernels cannot write it.{' '}
          <code>i</code> is assigned inside the kernel.
        </>
      ),
    },
    scopeWhy: {
      block: <>Being in the same block shares nothing by itself. Each thread has its own value of i: it has to, the values differ.</>,
      grid: <>Only data in global or constant memory is visible grid-wide. This is declared in the kernel body.</>,
    },
    done: <>A scalar declared in the kernel body: one copy per thread, each in a register.</>,
  },
  {
    id: 'window',
    what: (
      <>
        the array <code>window</code>
      </>
    ),
    decl: 'float window[TAPS];',
    mem: 'local',
    scope: 'thread',
    copies: G * T,
    tier: 'global',
    memWhy: {
      register: (
        <>
          That is the rule for scalars. An array declared in the kernel body goes to local memory by default. The
          exception is for arrays accessed only with constant indices, and this one is indexed by the loop variable.
        </>
      ),
      shared: NOT_SHARED,
      global: (
        <>
          Close: that is where it physically is. But the memory type is local memory, and the difference matters: it is
          private to the thread, which global memory is not.
        </>
      ),
      constant: <>Each thread writes it, and kernels cannot write constant memory.</>,
    },
    scopeWhy: {
      block: <>No qualifier, so no sharing. Each thread fills its own window with its own four samples.</>,
      grid: <>It sits in global memory physically, but only its owner can reach it.</>,
    },
    done: (
      <>
        An array declared in the kernel body: one per thread, in local memory. It takes the global colour because
        global memory is where local memory physically is.
      </>
    ),
  },
  {
    id: 'blockGain',
    what: <code>blockGain</code>,
    decl: '__shared__ float blockGain;',
    mem: 'shared',
    scope: 'block',
    copies: G,
    tier: 'shared',
    memWhy: {
      register: (
        <>
          It is a scalar, but the <code>__shared__</code> qualifier overrides the default for scalars.
        </>
      ),
      local: <>Local memory is private to a thread. Thread 0 writes this variable and the other 255 threads read it.</>,
      global: (
        <>
          <code>gain</code>, the parameter, points into global memory. <code>blockGain</code> is the block's own copy
          of one element of it.
        </>
      ),
      constant: <>Thread 0 writes it, and kernels cannot write constant memory.</>,
    },
    scopeWhy: {
      thread: (
        <>
          <code>__shared__</code> exists to break thread privacy. Thread 0 stores the value and every thread of the
          block reads it.
        </>
      ),
      grid: <>Shared memory stops at the block boundary. Another block has a different copy, holding a different gain.</>,
    },
    done: <>One float per block. Thread 0 fetches it from global memory once, and 256 threads use it.</>,
  },
  {
    id: 'stage',
    what: (
      <>
        the array <code>stage</code>
      </>
    ),
    decl: '__shared__ float stage[256];',
    mem: 'shared',
    scope: 'block',
    copies: G,
    tier: 'shared',
    memWhy: {
      local: (
        <>
          Arrays go to local memory only when they are plain automatic variables. <code>__shared__</code> puts this one
          in shared memory.
        </>
      ),
      register: <>It is an array, and it carries a qualifier that names its memory.</>,
      global: (
        <>
          Its contents come from global memory, but the array itself is the block's on-chip copy. That is the whole
          reason it exists.
        </>
      ),
      constant: <>Every thread writes one element of it, and kernels cannot write constant memory.</>,
    },
    scopeWhy: {
      thread: <>Each thread writes one element and then reads its neighbours' elements. It could not if the array were private.</>,
      grid: <>Shared memory stops at the block boundary. That is why a thread at the end of a block cannot see the next block's samples.</>,
    },
    done: <>One 256-element array per block, on the chip, visible to that block's threads and nobody else.</>,
  },
];

function copiesHint(d: Decl, v: number): ReactNode {
  const perThread = G * T;
  if (!Number.isFinite(v) || v < 0 || !Number.isInteger(v)) return <>A count of copies is a whole number.</>;
  if (d.copies === perThread) {
    if (v === G) return <>48 is the number of blocks. This one is private to a thread, so count threads.</>;
    if (v === T) return <>256 is the number of threads in one block, and each of the 48 blocks has its own 256.</>;
    if (v === 1) return <>There is one declaration in the source, but every thread that runs the kernel gets its own copy.</>;
    if (d.id === 'window' && v === perThread * TAPS)
      return <>That counts floats. The row asks how many arrays exist: one per thread.</>;
  } else if (d.copies === G) {
    if (v === perThread)
      return d.id === 'stage' ? (
        <>12,288 would be one per thread. (It is also the number of floats in all the copies together.) There is one array per block.</>
      ) : (
        <>
          That would be one per thread. <code>__shared__</code> makes it one per block.
        </>
      );
    if (v === T) return <>256 is how many threads use each copy. Count the copies: one per block.</>;
    if (v === 1) return <>A single copy for the whole grid would need a declaration outside the kernel. This one is per block.</>;
  } else {
    if (d.id === 'taps' && v === TAPS) return <>Four is the number of elements. The row asks how many arrays exist.</>;
    if (v === G) return <>Blocks do not get their own. All 48 read the same array.</>;
    if (v === perThread)
      return d.id === 'in' ? (
        <>Each thread holds the address, 12,288 pointers in all. The array behind them exists once.</>
      ) : (
        <>Threads do not get their own. One copy serves all 12,288 of them.</>
      );
  }
  return <>Start from who can access it: how many threads, or blocks, or grids each hold their own?</>;
}

interface RowState {
  mem: Mem | '';
  scope: Scope | '';
  copies: string;
  /** The answers as they were when Check was pressed. */
  checked: { mem: Mem; scope: Scope; copies: number } | null;
}
const blankRows = (): Record<string, RowState> =>
  Object.fromEntries(DECLS.map((d) => [d.id, { mem: '', scope: '', copies: '', checked: null }]));
const isSolved = (d: Decl, s: RowState) =>
  !!s.checked && s.checked.mem === d.mem && s.checked.scope === d.scope && s.checked.copies === d.copies;

function WhereItLives() {
  const [rows, setRows] = useState<Record<string, RowState>>(blankRows);
  const set = (id: string, patch: Partial<RowState>) =>
    setRows((r) => ({ ...r, [id]: { ...r[id], checked: null, ...patch } }));
  const solved = DECLS.filter((d) => isSolved(d, rows[d.id]));
  const tiers: Record<string, Tier> = Object.fromEntries(solved.map((d) => [d.id, d.tier]));

  return (
    <Bench
      title="Where does it live?"
      task={`One launch: ${G} blocks of ${T} threads. Place all six names.`}
      foot="Answers follow the book's declaration rules. A name takes its memory's colour in the code once all three answers for it are right."
    >
      <div data-where>
        <Code
          tabs={[{ label: 'CUDA', code: SMOOTH, tiers, status: 'front-end-checked (labs/ch05/snippets/where_does_it_live.cu)' }]}
          caption="A smoothing filter of our own, written for this exercise. Each thread combines its sample and the three after it inside its block, then scales by a per-block gain."
        />
        <div className="panel">
          <div className="row" style={{ justifyContent: 'space-between', marginBottom: '0.2rem' }}>
            <b>Six names</b>
            <span className="muted" data-where-score>
              {solved.length} of {DECLS.length} placed
            </span>
          </div>
          {DECLS.map((d, n) => {
            const s = rows[d.id];
            const done = isSolved(d, s);
            const ready = s.mem !== '' && s.scope !== '' && s.copies.trim() !== '';
            const c = s.checked;
            const check = () =>
              set(d.id, { checked: { mem: s.mem as Mem, scope: s.scope as Scope, copies: Number(s.copies) } });
            return (
              <div
                key={d.id}
                data-row={d.id}
                data-solved={done ? 'true' : 'false'}
                style={{ padding: '0.7rem 0', borderTop: n ? '1px solid var(--rule-soft)' : undefined }}
              >
                <div className="row" style={{ alignItems: 'flex-end' }}>
                  <div style={{ flex: '1 1 11rem', minWidth: '11rem' }}>
                    <p style={{ margin: 0 }}>{d.what}</p>
                    <div className="muted mono nowrap" style={{ fontSize: '0.74rem' }}>
                      {d.decl}
                    </div>
                  </div>
                  <label className="field">
                    Memory
                    <select
                      aria-label={`Memory for ${d.id}`}
                      value={s.mem}
                      disabled={done}
                      onChange={(e) => set(d.id, { mem: e.target.value as Mem | '' })}
                    >
                      <option value="">choose</option>
                      {MEMS.map(([v, label]) => (
                        <option key={v} value={v}>
                          {label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="field">
                    Who can access it
                    <select
                      aria-label={`Who can access ${d.id}`}
                      value={s.scope}
                      disabled={done}
                      onChange={(e) => set(d.id, { scope: e.target.value as Scope | '' })}
                    >
                      <option value="">choose</option>
                      {SCOPES.map(([v, label]) => (
                        <option key={v} value={v}>
                          {label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="field">
                    Copies in this launch
                    <input
                      type="number"
                      min="0"
                      step="1"
                      aria-label={`Copies of ${d.id}`}
                      value={s.copies}
                      disabled={done}
                      onChange={(e) => set(d.id, { copies: e.target.value })}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && ready) check();
                      }}
                    />
                  </label>
                  <div style={{ minWidth: '4.6rem' }}>
                    {done ? (
                      <span className="chip ok">placed</span>
                    ) : (
                      <button className="primary" aria-label={`Check ${d.id}`} disabled={!ready} onClick={check}>
                        Check
                      </button>
                    )}
                  </div>
                </div>
                {c ? (
                  <div data-feedback={d.id} role="status" style={{ marginTop: '0.5rem' }}>
                    {done ? (
                      <p style={{ margin: 0 }}>{d.done}</p>
                    ) : (
                      <ul style={{ margin: 0, paddingLeft: 0, listStyle: 'none' }}>
                        <li>
                          {c.mem === d.mem ? (
                            <span className="chip ok">memory is right</span>
                          ) : (
                            <>
                              <span className="chip fail">memory</span>{' '}
                              {d.memWhy[c.mem] ?? <>Look at where the declaration sits and which qualifier it carries.</>}
                            </>
                          )}
                        </li>
                        <li>
                          {c.scope === d.scope ? (
                            <span className="chip ok">access is right</span>
                          ) : (
                            <>
                              <span className="chip fail">access</span>{' '}
                              {d.scopeWhy[c.scope] ?? <>Ask which threads could name this variable and get the same storage.</>}
                            </>
                          )}
                        </li>
                        <li style={{ marginBottom: 0 }}>
                          {c.copies === d.copies ? (
                            <span className="chip ok">count is right</span>
                          ) : (
                            <>
                              <span className="chip fail">count</span> {copiesHint(d, c.copies)}
                            </>
                          )}
                        </li>
                      </ul>
                    )}
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>
        <div className="row" style={{ marginTop: '0.8rem', justifyContent: 'space-between' }}>
          <TierLegend tiers={['reg', 'shared', 'global', 'const']} />
          <button className="quiet" onClick={() => setRows(blankRows())}>
            Start again
          </button>
        </div>
        {solved.length === DECLS.length ? (
          <div className="panel" style={{ marginTop: '0.8rem' }} data-where-done>
            <p style={{ margin: 0 }}>
              <b>All six placed.</b> Now read the kernel by colour. Per thread it asks global memory for one element of{' '}
              <code>in</code> and stores one element of <code>out</code>; one thread per block also fetches a gain. But{' '}
              <code>window</code> has the global colour too: if the compiler leaves that array in local memory, each of
              its eight accesses per thread pays global-memory latency that the source never shows. Four scalars would
              have gone to registers.
            </p>
          </div>
        ) : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- your devices
// What ./verify.sh recorded from the reader's own devices, for the query row of
// the comparison table. Nothing is shown that a device did not report.
const DEVICE_FIELDS: [string, string][] = [
  ['hasUnifiedMemory', 'CPU and GPU share all memory'],
  ['maxThreadgroupMemoryLength', 'threadgroup memory available to a kernel, bytes'],
  ['sharedMemPerBlock', 'shared memory per block, bytes'],
  ['sharedMemPerMultiprocessor', 'shared memory per multiprocessor, bytes'],
  ['totalConstMem', 'constant memory, bytes'],
  ['totalGlobalMem', 'global memory, bytes'],
];
function YourDevices() {
  const { reports, loading } = useReports();
  const rows = realBackends(reports).flatMap((b) =>
    DEVICE_FIELDS.filter(([k]) => reports[b].device[k] !== undefined).map(([k, meaning]) => {
      const raw = String(reports[b].device[k]);
      const value = k === 'hasUnifiedMemory' ? (raw === '1' || raw === 'true' ? 'yes' : 'no') : Number(raw).toLocaleString('en-US');
      return { b, name: String(reports[b].device.name ?? b), k, meaning, value };
    }),
  );
  return (
    <div className="panel" data-your-devices style={{ marginBottom: '1.4rem' }}>
      <p style={{ margin: '0 0 0.4rem' }} className="ui">
        <b>What your own devices answered</b>
      </p>
      {rows.length ? (
        <table className="data" style={{ margin: 0 }}>
          <thead>
            <tr>
              <th>Backend</th>
              <th>Device</th>
              <th>Property</th>
              <th>Meaning</th>
              <th className="num">Reported</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.b + r.k}>
                <td>{r.b}</td>
                <td>{r.name}</td>
                <td>
                  <code>{r.k}</code>
                </td>
                <td>{r.meaning}</td>
                <td className="num">{r.value}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p style={{ margin: 0 }} className="ui muted">
          {loading
            ? 'Reading reports.'
            : 'No device report yet. ./verify.sh records what each of your devices reports, and this table fills in.'}
        </p>
      )}
    </div>
  );
}

// Row headers of a data table: keep the heavier weight, not the heavier rule.
const ROW_HEAD = { borderBottom: '1px solid var(--rule-soft)' };

// ---------------------------------------------------------------- questions
export const questionsS52: Record<string, Question> = {
  'ch05/5.2/q-copies-thread': {
    kind: 'numeric',
    id: 'ch05/5.2/q-copies-thread',
    prompt: (
      <>
        A kernel body declares <code>float acc = 0.0f;</code>. The kernel is launched once, as 1000 blocks of 512
        threads. How many separate versions of <code>acc</code> exist during that launch?
      </>
    ),
    answer: 512000,
    unit: 'versions',
    explain: (
      <p>
        An automatic variable has thread scope: every thread gets a private version, so 1000 × 512 = 512,000. Each one
        occupies a register of its thread. Had it been an automatic array, the count would be the same and the place
        would be local memory: 512,000 private regions of global memory.
      </p>
    ),
    source: 'ch. 5 exercise 6, p. 119; §5.2, p. 102',
  },
  'ch05/5.2/q-copies-block': {
    kind: 'numeric',
    id: 'ch05/5.2/q-copies-block',
    prompt: (
      <>
        The same kernel declares <code>__shared__ float stage[512];</code>. Same launch: 1000 blocks of 512 threads.
        How many versions of the array <code>stage</code> exist? Count arrays, not elements.
      </>
    ),
    answer: 1000,
    unit: 'versions',
    explain: (
      <p>
        Block scope: one per block, so 1000, each visible to the 512 threads of its block and to no one else. Divide
        the previous answer by this one and you get 512: that many threads can read a value once it sits in shared
        memory, where a register serves exactly one. Section 5.3 turns that into fewer fetches from global memory.
      </p>
    ),
    source: 'ch. 5 exercise 7, p. 119; §5.2, pp. 102–103',
  },
  'ch05/5.2/q-why-shared': {
    kind: 'choice',
    id: 'ch05/5.2/q-why-shared',
    prompt: (
      <>
        Suppose a device had registers to spare, so capacity is no argument. A block's threads keep needing the same
        values from global memory. Why hold those values in shared memory and not in registers?
      </>
    ),
    options: [
      {
        label: 'Reading shared memory is faster than reading a register, so the arithmetic waits less for operands.',
        why: 'It is the other way round: shared memory needs a load instruction, a register operand does not.',
      },
      {
        label:
          'A register serves only its own thread. A value fetched once into shared memory can be read by every thread of the block.',
        correct: true,
      },
      {
        label: 'A value held in a register still uses global-memory bandwidth each time it is read, as the register is refilled.',
        why: 'A value kept in a register uses none. That is exactly why registers raise the FLOP/B ratio.',
      },
      {
        label: 'Registers lose their contents whenever the SM switches to another warp, so the value is fetched again.',
        why: 'The register file keeps the registers of every scheduled thread. That is what makes the switch free.',
      },
    ],
    explain: (
      <p>
        Registers are private. If 512 threads each need the same element, 512 registers mean 512 fetches from global
        memory. Shared memory is visible to the whole block, so one thread can fetch the element and the others read
        the on-chip copy. The saving comes from the sharing, not from the speed of the memory: a register is faster
        than shared memory and still could not do this job.
      </p>
    ),
    source: 'ch. 5 exercise 4, p. 119; §5.2, pp. 99–101',
  },
  'ch05/5.2/q-unified': {
    kind: 'choice',
    id: 'ch05/5.2/q-unified',
    prompt: (
      <>
        You port a CUDA program to Metal on Apple silicon. The CUDA version copies the input to the device, launches
        the kernel and copies the result back. In the Metal version the input and the output are buffers in shared
        storage mode. What happens to the two copies?
      </>
    ),
    options: [
      {
        label:
          'They are still needed. Shared storage mode only saves you from allocating the buffer twice, once per processor.',
        why: 'There is one allocation and both processors address it. There is nothing to copy between.',
      },
      {
        label:
          'They disappear, and the CPU may read the output as soon as it has encoded and committed the work.',
        why: 'Committing schedules the work. The output is not there until the GPU has finished writing it.',
      },
      {
        label:
          'They disappear: CPU and GPU address the same allocation. What remains is ordering: the CPU must wait until the GPU work that writes the output has completed.',
        correct: true,
      },
      {
        label: 'They are replaced by a call that tells Metal which byte range of the buffer the CPU modified.',
        why: 'That call belongs to managed storage mode, which Apple has deprecated and which has no effect on Apple silicon.',
      },
    ],
    explain: (
      <>
        <p>
          With unified memory a shared-mode buffer is system memory that both processors can address, so a transfer
          has nothing to move and no API call stands in for it. The pointer the CPU holds is valid at any moment and
          waits for nothing, so the ordering is yours to enforce. Apple's documentation puts it as making sure the
          changes scheduled on one processor complete before the other accesses the resource.
        </p>
        <p>
          Nothing changes inside the kernel. It still has values private to a thread, threadgroup memory for the
          group, and buffers for everything else, and sharing a buffer with the CPU does not tell you how fast a kernel
          can read it. Lab 5.1 measures that.
        </p>
      </>
    ),
    source:
      'not in the book. Apple Metal documentation on storage modes, checked 2026-11-20 (notes/today/ch05-facts.md, C6)',
  },
};

const OPERANDS = `// both operands already in registers: one instruction
fadd r1, r2, r3

// one operand still in memory: load it into r2 first, then the same add
load r2, r4, offset
fadd r1, r2, r3`;

export function S52() {
  return (
    <>
      <p>
        From PyTorch you know one question about where data is: host or device, settled with <code>.to(device)</code>.
        Inside a kernel the question is finer. A variable can be in one of five memories, and the one you declare it
        into decides two things at once: which threads can see it, and how long a thread waits to use it.
      </p>
      <Source section="5.2" pages="98" figs="Fig. 5.1" />

      <h2>Five memories</h2>
      <p>
        Start with the two that the host can also reach. <b>Global memory</b> is where the arrays you hand to a kernel
        are. Every thread of every block can read and write it, and so can the host. It is also the slow one: off-chip
        DRAM, the memory whose bandwidth limited everything in section 5.1. <b>Constant memory</b> is visible to every
        thread too, and the host can read and write it, but kernel code can only read it. In return the device serves
        it with short latency and high bandwidth. The book leaves the how to chapter 7, in{' '}
        <a href={href('ch07', '7.4')}>section 7.4, Constant memory and caching</a>.
      </p>
      <Source section="5.2" pages="93, 98" />
      <p>
        <b>Registers</b> are on the chip, inside the processor. They are very fast and each thread has its own: no
        thread can read another thread's registers. <b>Shared memory</b> is on the chip as well, slower than a register
        and much faster than global memory. It is allocated per block. Every thread of a block can read and write its
        block's shared memory, and no thread can reach another block's. It is how the threads of a block work on the
        same inputs and pass intermediate results to each other.
      </p>
      <Source section="5.2" pages="98–99" />
      <p>
        The fifth, <b>local memory</b>, is the one whose name misleads. It is private to a thread, like registers, but
        it is not on the chip: it is placed in global memory and costs about what global memory costs. It holds
        whatever a thread cannot keep in registers: arrays declared in the kernel body, registers the compiler had to
        spill, and the rest of the call stack. "Local" says who can see it, not where it is.
      </p>
      <Source section="5.2" pages="98" />
      <p>
        The map below has one grid of two blocks with two threads each. Before you select anything, decide: thread 0
        and thread 1 of block 0 need to pass a value between them. Which boxes could carry it?
      </p>
      <Simplification>
        <p>
          Texture memory is left out, as it is in the book's own figure. The map shows permissions only. It has no
          caches, no sizes and no speeds, and a real block has far more than two threads.
        </p>
      </Simplification>
      <MemoryMap />
      <p>
        Two boxes could: the shared memory of block 0, and global memory. Registers could not, although both threads
        have them and they are the fastest thing in the picture, because a thread's registers are closed even to its
        neighbour in the same block. And the on-chip option ends at the block boundary: select the shared memory of
        block 1 and the threads of block 0 go dark.
      </p>

      <h2>Why a register operand is cheap</h2>
      <p>
        A processor's arithmetic instructions take their operands from registers. If both operands are already there,
        an addition is one instruction. If one of them is still in memory, the processor first has to execute a load
        that brings it into a register, and only then can it add. The book's two cases:
      </p>
      <Source section="5.2" pages="99–100" figs="Fig. 5.2" />
      <Code
        tabs={[{ label: 'pseudo-assembly', code: OPERANDS }]}
        caption="The book's illustration (p. 100), as schematic assembly. The register file is inside the processor; memory is outside it."
      />
      <p>
        So a value that stays in a register saves three things. It saves an instruction. It saves bandwidth: a register
        access is not global-memory traffic, which raises the FLOP/B ratio of section 5.1, and the book puts the
        combined bandwidth of the register files across all SMs at two orders of magnitude or more above that of global
        memory. And it saves energy: by the book's figure, reading a register costs at least ten times less than
        reading global memory. Take both magnitudes as the book's rough lower bounds, not as measurements of any
        device.
      </p>
      <Source section="5.2" pages="100–101" />
      <p>
        The catch is capacity. A thread has a limited number of registers, and a kernel that needs more per thread
        than the device can give at full occupancy runs with fewer threads resident. You met that budget in chapter 4.
      </p>
      <Source section="5.2" pages="101" />

      <h3>Why GPU register files are so large</h3>
      <p>
        A CPU core saves a thread's registers to memory when it switches away from the thread and restores them when
        it switches back. A GPU does neither. The register file of a processing block holds the registers of every
        thread scheduled on it, all the time, so moving from one warp to another copies nothing. This is the
        zero-overhead scheduling of chapter 4, and it is the reason a GPU's register file is much larger than a CPU's.
        The file can also be divided either way: fewer registers per thread and more threads, or the reverse.
      </p>
      <Source section="5.2" pages="99" figs="sidebar on CPU and GPU register architecture" />

      <h2>Shared memory is a scratchpad</h2>
      <p>
        Shared memory is on the chip, but it is memory and not a register file. To use a value from it the processor
        still executes a load. That load has lower latency and far higher throughput than a load from global memory,
        and longer latency and less bandwidth than a register access. The architectural name for a memory like this is
        a scratchpad. An SM has several processing units, the threads of a block are spread over them, and the hardware
        lets those units access the block's shared memory at the same time.
      </p>
      <Source section="5.2" pages="101" figs="Fig. 5.3" />
      <p>
        One extension, in one sentence: on recent NVIDIA architectures blocks can be grouped into a cluster, and a
        thread can then reach the shared memory of every block in its cluster, which the book calls distributed shared
        memory.
      </p>
      <Source section="5.2" pages="102" />
      <p>
        Nothing in this chapter or its labs uses it. Every kernel here keeps to the rule the map showed: shared memory
        ends at the block boundary. If you meet clusters later, treat them as a feature to test for on the device in
        front of you. HIP's headers declare cluster groups and launch attributes, but which AMD devices support them
        we have not checked; Metal documents nothing like them (notes/today/infra-facts.md, "Where the backends really
        differ").
      </p>
      <Today checked="2026-11-20 against the CUDA 13.5 guide (notes/today/ch05-facts.md, A6)">
        <p>
          NVIDIA's guide agrees with the book's dating: compute capability 9.0 (Hopper) or later.{' '}
          <code>cudaDeviceProp.clusterLaunch</code> reports support.
        </p>
      </Today>

      <h2>Scope, lifetime and number of copies</h2>
      <p>
        Two more words describe a variable. Its <b>scope</b> is the set of threads that can access it: one thread, the
        threads of one block, or all threads of all grids. Its <b>lifetime</b> is how long its value survives. Declared
        inside the kernel body, a variable lives for one launch: nothing carries over to the next launch, and every
        launch has to initialise it again. Declared outside any function, it lives as long as the application and
        every kernel can see it.
      </p>
      <Source section="5.2" pages="102" />
      <p>
        Scope also tells you how many copies exist, which is the part people get wrong. Thread scope means one private
        copy per thread: launch a million threads and there are a million of them. Block scope means one per block.
        The declaration decides all three properties:
      </p>
      <Source section="5.2" pages="102" figs="Fig. 5.4" />
      <table className="data">
        <thead>
          <tr>
            <th>What you declare</th>
            <th>Memory</th>
            <th>Scope</th>
            <th>Lifetime</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>a scalar in the kernel body</td>
            <td>
              <span className="chip reg">register</span>
            </td>
            <td>thread</td>
            <td>one launch</td>
          </tr>
          <tr>
            <td>an array in the kernel body</td>
            <td>
              <span className="chip global">local</span>
            </td>
            <td>thread</td>
            <td>one launch</td>
          </tr>
          <tr>
            <td>
              <code>__shared__</code> in the kernel body
            </td>
            <td>
              <span className="chip shared">shared</span>
            </td>
            <td>block</td>
            <td>one launch</td>
          </tr>
          <tr>
            <td>
              <code>__device__</code> outside any function
            </td>
            <td>
              <span className="chip global">global</span>
            </td>
            <td>all grids</td>
            <td>application</td>
          </tr>
          <tr>
            <td>
              <code>__constant__</code> outside any function
            </td>
            <td>
              <span className="chip const">constant</span>
            </td>
            <td>all grids</td>
            <td>application</td>
          </tr>
        </tbody>
      </table>
      <p>Each row has a footnote worth knowing.</p>
      <ul>
        <li>
          <b>Scalars.</b> Registers are the cheap case, but every scalar is one more register per thread, and too many
          cost occupancy.
        </li>
        <li>
          <b>Arrays.</b> An array in the kernel body looks like a stack variable and behaves like global memory, long
          delays included. The compiler may keep a small one in registers when every access uses a constant index.
          Chapter 6 comes back to this in <a href={href('ch06', '6.6')}>section 6.6, Loop unrolling</a>.
        </li>
        <li>
          <b>
            <code>__shared__</code>.
          </b>{' '}
          One copy per block, visible to all its threads, gone when the kernel finishes. Its usual job is to hold the
          piece of global data that the block uses heavily during one phase of the computation. Often the algorithm
          has to be reorganised to have such phases at all, which is what section 5.3 does.
        </li>
        <li>
          <b>
            <code>__device__</code> alone.
          </b>{' '}
          An ordinary global variable in global memory: slow, visible to every kernel, and persistent. It can carry
          data from one launch to the next without a parameter, and the book calls that bad style.
        </li>
        <li>
          <b>
            <code>__constant__</code>.
          </b>{' '}
          Must be declared outside any function, and kernel code cannot change it. It is stored in global memory but
          cached, which is where its speed comes from. The book gives the total size of all constant variables as
          65,536 bytes "currently".
        </li>
      </ul>
      <Source section="5.2" pages="102–104" />
      <p>
        A limit that a book calls current is one to query, not to memorise. The same goes for every capacity in this
        chapter: constant memory here, shared memory per block in section 5.6, registers per thread in chapter 4. A
        kernel that hard-codes one of them is only known to fit the device it was written on.
      </p>
      <Today checked="2026-11-20 against the CUDA 13.5 guide (notes/today/ch05-facts.md, A4)">
        <p>
          Still 64 KB, which NVIDIA's guide defines as 65,536 bytes, for every compute capability it lists. The guide
          says "typically", so read <code>totalConstMem</code> from <code>cudaDeviceProp</code>.
        </p>
      </Today>

      <h2>Where does it live?</h2>
      <p>
        Now apply the table. The kernel below smooths a signal: each thread combines four consecutive samples with four
        weights and scales the result by a gain that belongs to its block. For each of six names, say which memory it
        is in, who can access it, and how many copies exist during the one launch shown at the bottom. Work the counts
        out. When an answer is wrong, the feedback names the rule you tripped over.
      </p>
      <Simplification>
        <p>
          The answers follow the book's rules for declarations. A real compiler may do better, for example by keeping a
          small array in registers. The kernel is ours, in the manner of the book's exercise 11 (pp. 120–121), whose
          printed listing has typesetting errors.
        </p>
      </Simplification>
      <WhereItLives />
      <p>The same counting at the scale of a real launch, and then the question this section exists to set up.</p>
      <Ask q={questionsS52['ch05/5.2/q-copies-thread']} />
      <Ask q={questionsS52['ch05/5.2/q-copies-block']} />
      <Ask q={questionsS52['ch05/5.2/q-why-shared']} />

      <h2>The same ideas in HIP and Metal</h2>
      <p>
        Everything so far is CUDA, because the book is. The ideas carry over to the other two targets of this course.
        The spellings carry over to one of them.
      </p>
      <Beyond>
        <p>
          The book covers CUDA only. The table is built from vendor documentation read on 2026-11-20
          (notes/today/ch05-facts.md: A1, B2, B4, B9, C1, C2, C3, C6, C8), not from running code.
        </p>
      </Beyond>
      <div className="wide" style={{ maxWidth: '62rem' }}>
        <table className="data" data-compare>
          <thead>
            <tr>
              <th style={{ width: '14%' }}>Concept</th>
              <th style={{ width: '28%' }}>CUDA (the book)</th>
              <th style={{ width: '25%' }}>HIP</th>
              <th>Metal</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th scope="row" style={ROW_HEAD}>Scratch memory for a block</th>
              <td>
                <code>__shared__</code> on a variable in the kernel. The group of threads is a block.
              </td>
              <td>
                <code>__shared__</code>, same spelling and same meaning: visible to the threads of one block, for the
                lifetime of that block.
              </td>
              <td>
                The <code>threadgroup</code> address space: a <code>threadgroup</code> variable in the kernel body, or
                a <code>threadgroup</code> pointer parameter marked <code>[[threadgroup(n)]]</code> that the host
                sizes. The group is a threadgroup.
              </td>
            </tr>
            <tr>
              <th scope="row" style={ROW_HEAD}>Barrier</th>
              <td>
                <code>__syncthreads()</code>
              </td>
              <td>
                <code>__syncthreads()</code>
              </td>
              <td>
                <code>threadgroup_barrier(mem_flags::mem_threadgroup)</code>
              </td>
            </tr>
            <tr>
              <th scope="row" style={ROW_HEAD}>Global buffers</th>
              <td>Pointer parameters of the kernel that point into the device's global memory.</td>
              <td>
                The same. <code>__global__</code> and <code>__device__</code> keep their spelling, and llama.cpp
                builds its HIP backend from the same kernel files as its CUDA one.
              </td>
              <td>
                Buffers passed as kernel arguments, in what the Metal Shading Language calls device memory. On the
                host each one is an <code>MTLBuffer</code>. The lab kernels show the declaration.
              </td>
            </tr>
            <tr>
              <th scope="row" style={ROW_HEAD}>Host to device</th>
              <td>
                An explicit copy. The direction is named by <code>cudaMemcpyHostToDevice</code>.
              </td>
              <td>
                An explicit copy, <code>hipMemcpyHostToDevice</code>.
              </td>
              <td>
                On Apple silicon, none. A buffer in shared storage mode, the default there, is one allocation both
                processors address; <code>contents()</code> is the CPU's pointer to it. Ordering the accesses is your
                job.
              </td>
            </tr>
            <tr>
              <th scope="row" style={ROW_HEAD}>Asking for limits</th>
              <td>
                <code>cudaGetDeviceProperties</code> fills a <code>cudaDeviceProp</code>:{' '}
                <code>sharedMemPerBlock</code>, <code>sharedMemPerMultiprocessor</code>, <code>totalConstMem</code>,{' '}
                <code>totalGlobalMem</code>.
              </td>
              <td>
                <code>hipGetDeviceProperties</code> fills a <code>hipDeviceProp_t</code> with the same field names.
                What AMD devices report in them was not checked.
              </td>
              <td>
                <code>MTLDevice.maxThreadgroupMemoryLength</code> and <code>MTLDevice.hasUnifiedMemory</code>; per
                compiled kernel, <code>staticThreadgroupMemoryLength</code> and{' '}
                <code>maxTotalThreadsPerThreadgroup</code> on the pipeline state.
              </td>
            </tr>
          </tbody>
        </table>
        <YourDevices />
      </div>
      <p>
        HIP is the easy column. It keeps CUDA's spellings for <code>__shared__</code>, <code>__syncthreads()</code>,{' '}
        <code>__global__</code>, <code>__device__</code>, <code>__constant__</code> and the index built-ins, so the
        kernels of this chapter, which use only threads, blocks, shared and constant memory and barriers, are HIP
        kernels as they stand. One difference to remember: HIP's compiler does not define the built-ins implicitly, so the source has
        to include <code>hip/hip_runtime.h</code>. The sameness stops at warp intrinsics, cooperative groups,
        half-precision headers and matrix-core intrinsics, which later chapters meet one at a time. This course's HIP
        build is unverified: no HIP machine has compiled it yet.
      </p>
      <p>
        Metal changes the vocabulary, and on Apple silicon it changes the bottom of the memory map. In the map the
        host sits below a boundary, and in CUDA data crosses that boundary by a copy. On Apple silicon the CPU and the
        GPU share system memory. A buffer created in shared storage mode is a single allocation that both can address, so
        a host-to-device copy has nothing to move, and no call replaces it. What stays with you is ordering: when one
        side has work scheduled that changes the buffer, the other side must not touch it until that work has
        completed. In practice that means waiting for the command buffer to finish before the CPU reads a result.
      </p>
      <Beyond title="Apple's wording">
        <p>
          "Ensure that all changes you schedule on either the CPU or GPU for a resource that uses shared memory
          complete before accessing that resource on the other processor." The command-buffer reading of it is ours
          (notes/today/ch05-facts.md, C6).
        </p>
      </Beyond>
      <p>
        The top of the map does not change. A Metal kernel still chooses between memory private to a thread,
        threadgroup memory, and buffers, and the Metal Shading Language specification gives the same reason the book
        does: sharing through threadgroup memory "is faster on most devices" than sharing through device memory. What
        unified memory does to the cost of reading a buffer from a kernel is not something the documentation settles.
        You measured it in <a href={href('ch05', 'labs')}>lab 5.1</a>, or will.
      </p>
      <Ask q={questionsS52['ch05/5.2/q-unified']} />
      <p>
        Carry one thing into section 5.3. Of the five memories, exactly one is on the chip, writable by a kernel and
        visible to more than one thread. The naive kernel does not use it at all.
      </p>
    </>
  );
}
