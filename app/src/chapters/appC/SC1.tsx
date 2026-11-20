import { useState, type MouseEvent } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { href } from '../../lib/router';
import {
  BIN_LABEL,
  CARDS,
  STRIP_TOP,
  VARIABLES,
  WINDOWS,
  classify,
  judge,
  type Bin,
  type Card,
  type Rules,
} from '../../sim/addrspace';

// Lesson copies of labs/appC/snippets/c1_helper_generic.cu and c1_helper_overloads.metal;
// SC1.snippets.test.ts fails if they drift from the files.
const GENERIC_CU = `#define TILE 256

// A plain pointer: the signature names no memory. Under generic addressing
// the address it is given decides whether these loads read shared or global memory.
__device__ float rowMax(const float* p, int n) {
    float m = p[0];
    for (int i = 1; i < n; ++i) m = p[i] > m ? p[i] : m;
    return m;
}

// One block per row of x (row-major, n columns, 1 <= n <= TILE). Thread 0 takes
// the row's maximum twice: from global memory, and from the block's shared copy.
__global__ void rowMaxTwice(const float* x, float* fromGlobal, float* fromShared, int n) {
    __shared__ float tile[TILE];
    const float* row = x + (size_t)blockIdx.x * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) tile[i] = row[i];
    __syncthreads();
    if (threadIdx.x == 0) {
        fromGlobal[blockIdx.x] = rowMax(row, n);    // an address in global memory
        fromShared[blockIdx.x] = rowMax(tile, n);   // an address in the shared window
    }
}

void launchRowMaxTwice(const float* x, float* fromGlobal, float* fromShared, int rows, int n) {
    rowMaxTwice<<<rows, TILE>>>(x, fromGlobal, fromShared, n);
}`;

const OVERLOADS_METAL = `#include <metal_stdlib>
using namespace metal;

#define TILE 256

// MSL has no generic address space: every pointer names its memory, and the
// address space is part of the overload. One definition per memory you call it on.
float rowMax(device const float* p, uint n) {
    float m = p[0];
    for (uint i = 1; i < n; ++i) m = p[i] > m ? p[i] : m;
    return m;
}

float rowMax(threadgroup const float* p, uint n) {
    float m = p[0];
    for (uint i = 1; i < n; ++i) m = p[i] > m ? p[i] : m;
    return m;
}

// One threadgroup per row of x (row-major, n columns, 1 <= n <= TILE). Thread 0
// takes the row's maximum twice: from device memory, and from the threadgroup copy.
kernel void row_max_twice(device const float* x            [[buffer(0)]],
                          device float* fromDevice         [[buffer(1)]],
                          device float* fromThreadgroup    [[buffer(2)]],
                          constant uint& n                 [[buffer(3)]],
                          uint row  [[threadgroup_position_in_grid]],
                          uint t    [[thread_position_in_threadgroup]],
                          uint size [[threads_per_threadgroup]]) {
    threadgroup float tile[TILE];
    device const float* r = x + row * n;
    for (uint i = t; i < n; i += size) tile[i] = r[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        fromDevice[row] = rowMax(r, n);            // the device overload
        fromThreadgroup[row] = rowMax(tile, n);    // the threadgroup overload
    }
}`;

// `p` has no colour on purpose: which memory it reads is decided by the address it is given.
const CU_TIERS: Record<string, Tier> = {
  tile: 'shared', x: 'global', row: 'global', fromGlobal: 'global', fromShared: 'global', m: 'reg', i: 'reg',
};
const METAL_TIERS: Record<string, Tier> = {
  tile: 'shared', x: 'global', r: 'global', fromDevice: 'global', fromThreadgroup: 'global', m: 'reg', i: 'reg', t: 'reg',
};

const hex = (a: number) => '0x' + a.toString(16).padStart(4, '0');
const WINDOW_NAME = { local: 'local window', shared: 'shared window', global: 'outside every window: global' };

// ---------------------------------------------------------------- explorer: which window?
function WhichWindow() {
  const [addr, setAddr] = useState(VARIABLES[1].addr);
  const k = classify(addr);
  const picked = VARIABLES.find((v) => v.addr === addr);
  const W = 640;
  const H = 100;
  const pad = 12;
  const X = (a: number) => pad + (a / STRIP_TOP) * (W - 2 * pad);
  const onStrip = (ev: MouseEvent<SVGSVGElement>) => {
    const box = ev.currentTarget.getBoundingClientRect();
    const frac = ((ev.clientX - box.left) / box.width) * W;
    const a = Math.round((((frac - pad) / (W - 2 * pad)) * STRIP_TOP) / 4) * 4;
    setAddr(Math.max(0, Math.min(STRIP_TOP - 4, a)));
  };
  return (
    <Bench title="Which window?" task="Pick a variable, or move the address along the generic address space.">
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="ui muted">The helper is given a pointer to</span>
        <span className="seg" role="group" aria-label="Variable">
          {VARIABLES.map((v) => (
            <button key={v.id} data-var={v.id} aria-pressed={picked?.id === v.id} onClick={() => setAddr(v.addr)}>
              {v.code}
            </button>
          ))}
        </span>
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="The generic address space, with the local and shared windows"
        style={{ width: '100%', maxWidth: W, display: 'block', cursor: 'pointer' }}
        onClick={onStrip}
      >
        <rect x={pad} y={18} width={W - 2 * pad} height={28} fill="var(--global-wash)" stroke="var(--global)" />
        {WINDOWS.map((w) => {
          // Local memory lives in global memory, so its window keeps the global colour
          // (as in chapter 5) and is told apart by a dashed outline and its label.
          const local = w.window === 'local';
          const ink = local ? 'var(--global)' : 'var(--shared)';
          return (
            <g key={w.window} data-window-rect={w.window}>
              <rect
                x={X(w.lo)}
                y={18}
                width={X(w.hi) - X(w.lo)}
                height={28}
                fill={local ? 'var(--global-wash)' : 'var(--shared-wash)'}
                stroke={ink}
                strokeWidth={local ? 2 : 1}
                strokeDasharray={local ? '4 3' : undefined}
              />
              <text x={(X(w.lo) + X(w.hi)) / 2} y={62} fontSize="13" textAnchor="middle" style={{ fill: ink }}>
                {w.window}
              </text>
              {local ? (
                <text x={(X(w.lo) + X(w.hi)) / 2} y={76} fontSize="10" textAnchor="middle" style={{ fill: ink }}>
                  in global memory
                </text>
              ) : null}
            </g>
          );
        })}
        <text x={X(0x4000)} y={62} fontSize="13" textAnchor="middle" style={{ fill: 'var(--global)' }}>
          global
        </text>
        <text x={pad} y={62} fontSize="11">0x0000</text>
        <text x={W - pad} y={62} fontSize="11" textAnchor="end">0xffff</text>
        <line x1={X(addr)} x2={X(addr)} y1={10} y2={52} stroke="var(--ink)" strokeWidth="2" />
        <polygon points={`${X(addr) - 5},4 ${X(addr) + 5},4 ${X(addr)},11`} fill="var(--ink)" />
        <text x={Math.min(W - 40, Math.max(40, X(addr)))} y={95} fontSize="13" textAnchor="middle" fontWeight="600">
          {hex(addr)}
        </text>
      </svg>
      <label className="field ui" style={{ marginTop: '0.5rem', width: '100%', maxWidth: W }}>
        Address {hex(addr)}
        <input type="range" min={0} max={STRIP_TOP - 4} step={4} value={addr} aria-label="address" onChange={(e) => setAddr(+e.target.value)} />
      </label>
      <div className="narrate" data-window={k.window} style={{ marginTop: '0.7rem' }}>
        <p style={{ margin: 0 }}>
          {picked ? (
            <>
              <code>{picked.code}</code>, {picked.label}, is at <code>{hex(addr)}</code>.{' '}
            </>
          ) : (
            <>
              Address <code>{hex(addr)}</code>.{' '}
            </>
          )}
          It falls {k.window === 'global' ? '' : 'in the '}
          <span className={'chip ' + k.tier}>{WINDOW_NAME[k.window]}</span>.
          {k.window === 'local' ? (
            <>
              {' '}
              Local memory is private to one thread but lives in global memory, so it keeps the global colour, as in{' '}
              <a href={href('ch05', '5.2')}>chapter 5</a>.
            </>
          ) : null}
        </p>
        <p style={{ margin: '0.35rem 0 0' }}>
          The helper's load is a generic <code>ld</code> with no state space in the instruction. For this address it
          goes to the <code data-state>{k.state}</code> state space, as <code data-load>{k.load}</code> would.
        </p>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- classification
const LANG_LABEL = { cuda: 'CUDA', hip: 'HIP', msl: 'MSL' };
const BINS: Bin[] = ['global', 'shared', 'error'];
const BIN_SHORT: Record<Bin, string> = { global: 'reads global / device', shared: 'reads shared / threadgroup', error: 'does not compile' };

function CardBox({ card, rules, pick, onPick }: { card: Card; rules: Rules; pick: Bin | undefined; onPick: (b: Bin) => void }) {
  const o = judge(card, rules);
  const done = pick !== undefined;
  const right = pick === o.bin;
  return (
    <div className="panel" data-card={card.id} data-picked={pick ?? ''} data-result={done ? (right ? 'right' : 'wrong') : ''}>
      <p className="ui" style={{ margin: '0 0 0.3rem', fontSize: '0.8rem' }}>
        <b>{LANG_LABEL[card.lang]}</b>
      </p>
      <pre className="mono" style={{ margin: '0 0 0.4rem', fontSize: '0.76rem', whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>
        {card.helper}
      </pre>
      <pre className="mono muted" style={{ margin: '0 0 0.6rem', fontSize: '0.76rem', whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>
        {card.call}
      </pre>
      <div className="row" style={{ gap: '0.35rem' }}>
        {BINS.map((b) => (
          <button key={b} data-bin={b} aria-pressed={pick === b} className="quiet" style={{ fontSize: '0.78rem', padding: '0.2rem 0.45rem' }} onClick={() => onPick(b)}>
            {BIN_SHORT[b]}
          </button>
        ))}
      </div>
      {done ? (
        <p className="ui" style={{ margin: '0.55rem 0 0', fontSize: '0.82rem' }} data-feedback>
          <span className={'chip ' + (right ? 'ok' : 'fail')}>{right ? 'right' : 'not this bin'}</span>{' '}
          {right ? null : <>It {o.bin === 'error' ? 'does not compile' : BIN_LABEL[o.bin].replace('compiles, ', 'compiles and ')}. </>}
          {o.rule} <span className="muted">({o.source})</span>
        </p>
      ) : null}
    </div>
  );
}

function SortCards() {
  const [rules, setRules] = useState<Rules>('spec');
  const [picks, setPicks] = useState<Record<string, Bin>>({});
  const answered = CARDS.filter((c) => picks[c.id] !== undefined);
  const right = answered.filter((c) => picks[c.id] === judge(c, rules).bin);
  return (
    <Bench
      title="Does it compile, and which memory does it read?"
      task="Sort each card into one of the three bins."
      foot={
        <>
          The specification rules are MSL 4.1 sections 1.5.1 and 4 as quoted in appC-facts C1; the Metal compiler on
          the build machine agreed with every card when asked for MSL 4.0 or earlier. At its default, MSL 4.1, it
          disagreed on one card (labs/appC/snippets/check.py compiles each MSL card both ways). Nothing here was
          compiled for an AMD GPU.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="ui muted">Metal rules:</span>
        <span className="seg" role="group" aria-label="Metal rules">
          <button aria-pressed={rules === 'spec'} data-rules="spec" onClick={() => setRules('spec')}>
            the specification (and -std=metal4.0)
          </button>
          <button aria-pressed={rules === 'observed-4.1'} data-rules="observed-4.1" onClick={() => setRules('observed-4.1')}>
            the compiler's default here, MSL 4.1 (observed)
          </button>
        </span>
        <span className="ui" data-score>
          {right.length} of {CARDS.length} sorted correctly
        </span>
      </div>
      <div className="cols">
        {CARDS.map((c) => (
          <CardBox key={c.id} card={c} rules={rules} pick={picks[c.id]} onPick={(b) => setPicks({ ...picks, [c.id]: b })} />
        ))}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSC1: Record<string, Question> = {
  'appC/C.1/q-who-decides': {
    kind: 'choice',
    id: 'appC/C.1/q-who-decides',
    prompt: (
      <>
        A CUDA helper takes <code>const float* p</code> and loads <code>p[i]</code>. One call passes a{' '}
        <code>__shared__</code> tile, another a global array. What makes the first call's load read shared memory?
      </>
    ),
    options: [
      { label: 'The address itself: it falls in the shared window of the generic address space, and the hardware maps it there.', correct: true },
      {
        label: 'The compiler emits a separate copy of the helper for each memory.',
        why: 'Nothing requires that. One compiled function, with one generic load, handles both.',
      },
      { label: 'The driver checks the pointer at launch and patches the helper.', why: 'The helper is not even called at launch; the decision is made per load, by the address.' },
      { label: 'You must cast the argument to a __shared__ float*.', why: 'That was the situation before Fermi, when a pointer could point only into global memory.' },
    ],
    explain: (
      <p>
        Generic addressing: a load with no state space in the instruction reads whatever memory its address lies in.
        Shared and local memory are windows in one address space, and everything outside them is global.
      </p>
    ),
    source: 'book p. 624; appC-facts A2',
  },
  'appC/C.1/q-overloads': {
    kind: 'numeric',
    id: 'appC/C.1/q-overloads',
    prompt: (
      <>
        You port to MSL a helper that a kernel calls both on a <code>threadgroup</code> tile and on a{' '}
        <code>device</code> buffer. It must compile under every MSL version, 4.0 and earlier included. What is the
        smallest number of function definitions you need?
      </>
    ),
    answer: 2,
    unit: 'definitions',
    explain: (
      <p>
        One taking <code>device const float*</code> and one taking <code>threadgroup const float*</code>. MSL extends
        overloading to include the address space of an argument, so the two can share a name and each call picks its
        own. In CUDA and HIP the answer is 1.
      </p>
    ),
    source: 'MSL specification §1.5.1 and §4, appC-facts C1; compiled on the build machine with -std=metal4.0',
  },
  'appC/C.1/q-missing-space': {
    kind: 'choice',
    id: 'appC/C.1/q-missing-space',
    prompt: (
      <>
        MSL, compiled with <code>-std=metal4.0</code>: <code>{'float rowMax(const float* p, uint n) { ... }'}</code>.
        What happens?
      </>
    ),
    options: [
      { label: 'A compile error: every pointer must carry an address space.', correct: true },
      { label: 'It compiles, and p defaults to device.', why: 'There is no default in MSL 4.0: the compiler says "pointer type must have explicit address space qualifier".' },
      { label: 'It compiles, and p defaults to thread.', why: 'There is no default in MSL 4.0: the compiler says "pointer type must have explicit address space qualifier".' },
      {
        label: 'It compiles, and p becomes generic, as in CUDA.',
        why: 'Not under 4.0. At its default, MSL 4.1, the compiler on the build machine did accept it as generic; that is observed, not documented (appC-facts C1).',
      },
    ],
    explain: (
      <p>
        The specification: "If an address space attribute is missing on a pointer or reference type declaration, a
        compilation error occurs." The compiler on the build machine enforces this up to MSL 4.0.
      </p>
    ),
    source: 'MSL specification §4, appC-facts C1; compiled on the build machine',
  },
  'appC/C.1/q-speed': {
    kind: 'choice',
    id: 'appC/C.1/q-speed',
    prompt: <>The same CUDA helper, with the same n, is called once on shared data and once on global data. Same time?</>,
    options: [
      {
        label: 'No. It is one instruction stream, but the loads go to different memories, and the book says the call is faster on shared memory.',
        correct: true,
      },
      { label: 'Yes, because generic addressing hides the difference.', why: 'It hides the spelling, not the cost.' },
      { label: 'Yes, because both run the same compiled code.', why: 'Same instructions, different memories behind the loads.' },
      { label: 'It depends only on n.', why: 'n sets how many loads there are; the memory sets what each one costs.' },
    ],
    explain: (
      <p>
        Placement stays your job, and so does the optimisation of chapter 5: stage what a block reuses in shared memory,
        then let one helper read it there. NVIDIA's pages do not put a number on the difference.
      </p>
    ),
    source: 'book p. 624; appC-facts A2',
  },
};

// ---------------------------------------------------------------- the page
export function SC1() {
  return (
    <>
      <p>
        On the first CUDA GPUs, shared memory, local memory and global memory were three separate address spaces, each
        with its own load and store instructions. A pointer could point only into global memory. A device function that
        took a pointer therefore needed a separate version for each kind of memory it might be given, for each pointer
        argument.
      </p>
      <Source section="C.1" pages="624" />
      <p>
        Fermi put the three into one address space, reached by one set of load and store instructions. You decide which
        memory holds the data once, when you declare or allocate it, and never again. A function that takes a pointer
        accepts data in any of the three, so device code composes the way C++ does elsewhere. The book adds that this
        lowered the cost of building production-quality CUDA libraries and brought full C and C++ pointer support to
        kernels.
      </p>
      <Source section="C.1" pages="624" />
      <p>
        What one address space does not do is make the memories equally fast. In the book's words, the call is faster
        when the pointer refers to shared memory and slower when it refers to global memory. Deciding where data lives,
        and staging it in shared memory as chapter 5 did, is still your job.
      </p>
      <Today checked="2026-11-20 against NVIDIA's Fermi whitepaper and the PTX ISA (notes/today/appC-facts.md, A2)">
        <p>
          This is PTX's generic addressing (PTX 2.0, <code>sm_20</code> and later). <code>.shared</code>,{' '}
          <code>.local</code>, <code>.const</code> and <code>.param</code> are windows in one generic space; an address
          outside every window is global. <code>cvta</code> converts between generic and state-space addresses, and{' '}
          <code>isspacep</code> tests which window an address is in. The whitepaper is from 2009; NVIDIA says Fermi
          reached the market in March 2010. NVIDIA does not quantify the speed difference.
        </p>
      </Today>

      <h2>Which window?</h2>
      <p>
        Here is what that means for one load. The helper's load instruction names no memory. Pick a variable, or move
        the address along the strip, and see which window the address falls in and which memory the load therefore
        reaches.
      </p>
      <Simplification>
        <p>
          Window positions and sizes are invented for the picture; the real ones are set by the hardware and not given
          in our sources. One dimension, no caches, and the <code>.const</code> and <code>.param</code> windows left
          out.
        </p>
      </Simplification>
      <WhichWindow />
      <p>
        The address, not the source code, picks the memory. That is why one compiled <code>rowMax</code> serves a global
        row, a shared tile and a thread's local array.
      </p>
      <Ask q={questionsSC1['appC/C.1/q-who-decides']} />

      <h2>HIP: the same arrangement</h2>
      <p>
        AMD GPUs work the same way under other names. The compiler HIP uses documents a generic "flat" address space
        with two apertures, one for LDS (AMD's name for shared memory) and one for scratch (private memory). A flat load
        reaches global memory, LDS or scratch depending on which range its address is in. So a helper that takes{' '}
        <code>const float*</code> works unchanged in HIP. One condition from the same document: only 64-bit address
        spaces are supported for the <code>amdgcn</code> target.
      </p>
      <Beyond>
        <p>
          The source is the LLVM AMDGPU documentation, not HIP's own, which does not discuss this (appC-facts B1).
        </p>
      </Beyond>

      <h2>Metal: one helper per address space</h2>
      <p>
        The Metal Shading Language specification goes the other way. It describes <code>device</code>,{' '}
        <code>constant</code>, <code>thread</code> and <code>threadgroup</code> as disjoint address spaces and requires
        every pointer and reference to name one; a missing address space is a compile error. What makes this workable is
        overloading. MSL includes an argument's address space in the overloading rules, so you write{' '}
        <code>rowMax</code> once for <code>device const float*</code> and once for <code>threadgroup const float*</code>,
        and each call picks its own. That is exactly the one-version-per-memory arrangement the book says Fermi removed.
        One more rule: a helper may take a <code>threadgroup</code> pointer only if a kernel calls it.
      </p>
      <Beyond>
        <p>MSL specification 4.1, sections 1.5.1 and 4, as quoted in appC-facts C1.</p>
      </Beyond>
      <p>
        One thing the specification does not tell you. On the machine this course was built on, Apple's Metal compiler
        at its default language version, MSL 4.1, accepted <code>rowMax(const float* p, uint n)</code> with no address
        space, and a kernel that called it on threadgroup, device and thread memory returned the right maxima on the GPU.
        Asked for MSL 4.0 or earlier with <code>-std=metal4.0</code>, the same compiler rejects the line: "pointer type
        must have explicit address space qualifier". The specification says a missing address space is a compile error,
        so treat this as observed, not documented, and write the overloads. They compile under every version.
      </p>
      <Beyond title="Observed on the build machine">
        <p>
          Recorded in appC-facts C1 as observed, not documented: Apple metal 32023.921, macOS 27 SDK, 2026-11-20.{' '}
          <code>labs/appC/snippets/check.py</code> repeats the compiles and the run.
        </p>
      </Beyond>

      <h2>Does it compile, and which memory does it read?</h2>
      <p>
        Eight call sites in CUDA, HIP and MSL. For each, decide whether it compiles and, if it does, which memory the
        helper's loads read. The switch at the top changes the Metal rules from the specification to what the compiler
        here did at its default. Before you flip it, predict which cards will move.
      </p>
      <SortCards />
      <p>
        CUDA and HIP need one helper; Metal, by its specification, needs one per address space. A wrong overload is not
        a slow path or a silent conversion: it is a compile error.
      </p>
      <Ask q={questionsSC1['appC/C.1/q-overloads']} />
      <Ask q={questionsSC1['appC/C.1/q-missing-space']} />

      <h2>The helper, in both languages</h2>
      <p>
        Here is the helper with its two call sites. In the CUDA tab, <code>p</code> carries no colour: which memory it
        reads is decided by the address it is given. The Metal tab has the pair of overloads.
      </p>
      <Code
        tabs={[
          { label: 'CUDA and HIP', code: GENERIC_CU, tiers: CU_TIERS, status: 'front-end-checked' },
          { label: 'Metal', code: OVERLOADS_METAL, tiers: METAL_TIERS, status: 'verified on Metal' },
        ]}
        caption={
          <>
            labs/appC/snippets/c1_helper_generic.cu and c1_helper_overloads.metal. The Metal file compiles with every MSL
            version from 2.0 to 4.1 on the build machine, and <code>python3 labs/appC/snippets/check.py</code> runs its
            kernel on the GPU against a CPU reference. The CUDA file has not run on an NVIDIA or AMD GPU.
          </>
        }
      />
      <p>
        Why this matters for inference kernels: attention and sampling are full of small reductions, the maximum and the
        sum inside a softmax, a dot product, an argmax. You will want each written once and called both on a tile staged
        in shared memory and on a row read straight from global memory. In CUDA and HIP that is one function. In MSL,
        as specified, it is one per address space.
      </p>
      <Beyond>
        <p>This connection to attention and sampling is ours.</p>
      </Beyond>
      <Ask q={questionsSC1['appC/C.1/q-speed']} />
      <p>
        Back to <a href={href('appC', 'start')}>the map</a>: this step is the one marked C.1, and it changes only the
        row for a <code>__shared__</code> variable.
      </p>
    </>
  );
}
