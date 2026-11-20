import { useState, type MouseEvent } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { benchRows, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { ALLOCS, HOP_LABEL, NODE_VALUES, SETTINGS, addresses, chase, hex, type Alloc, type HopVerdict, type Setting } from '../../sim/ptrchase';
import { REGISTERED, SPACE_LABEL, TOP, hex4, inferKind, ranges, type Mode, type Range } from '../../sim/uvacopy';
import { LAB } from './SC2';

// Lesson copy of labs/appC/snippets/c3_chase.cu; SC3.snippets.test.ts fails if it drifts.
const CHASE_CU = `// A linked list that the host built. The kernel can follow it only if every
// node is in memory a kernel may dereference; under UVA (CUDA 4) that means
// cudaHostAlloc or device memory, never plain malloc memory.
struct Node {
    float value;
    Node* next;
};

// One thread walks the list and sums it: pure pointer chasing, no parallelism.
__global__ void sumList(const Node* head, float* sum) {
    float s = 0.0f;
    for (const Node* p = head; p != nullptr; p = p->next) s += p->value;
    *sum = s;
}

void launchSumList(const Node* head, float* sum) {
    sumList<<<1, 1>>>(head, sum);
}`;
const CHASE_TIERS: Record<string, Tier> = { s: 'reg', p: 'reg' };

// ---------------------------------------------------------------- explorer: which way does the copy go?
function rangeFill(r: Range): { fill: string; stroke: string; dash?: string } {
  if (r.registered) return { fill: 'var(--page)', stroke: 'var(--ink)', dash: '4 3' };
  if (r.space === 'host') return { fill: 'var(--bench)', stroke: 'var(--ink-3)' };
  return { fill: 'var(--global-wash)', stroke: 'var(--global)' };
}

function CopyDirection() {
  const [mode, setMode] = useState<Mode>('uva');
  const [reg, setReg] = useState(false);
  const [src, setSrc] = useState(0x3000);
  const [dst, setDst] = useState(0x9000);
  const [placing, setPlacing] = useState<'src' | 'dst'>('src');
  const registered = mode === 'uva' && reg;
  const rs = ranges(mode, registered);
  const r = inferKind(dst, src, mode, registered);
  const W = 640;
  const pad = 14;
  const lanes = mode === 'uva' ? [null] : (['host', 'gpu0', 'gpu1'] as const);
  const laneH = 30;
  const top = 26;
  const bottom = top + lanes.length * (laneH + 8) - 8;
  const H = bottom + 36;
  const X = (a: number) => pad + (a / TOP) * (W - 2 * pad);
  const place = (ev: MouseEvent<SVGSVGElement>) => {
    const box = ev.currentTarget.getBoundingClientRect();
    const x = ((ev.clientX - box.left) / box.width) * W;
    const a = Math.max(0, Math.min(TOP - 0x100, Math.round((((x - pad) / (W - 2 * pad)) * TOP) / 0x100) * 0x100));
    if (placing === 'src') setSrc(a);
    else setDst(a);
  };
  const laneY = (i: number) => top + i * (laneH + 8);
  const marker = (a: number, label: string) => (
    <g key={label}>
      <line x1={X(a)} x2={X(a)} y1={label === 'src' ? top - 6 : top} y2={label === 'src' ? bottom : bottom + 6} stroke="var(--ink)" strokeWidth="2" />
      <text x={Math.min(W - 50, Math.max(50, X(a)))} y={label === 'src' ? top - 10 : bottom + 19} fontSize="13" textAnchor="middle" fontWeight="650">
        {label} {hex4(a)}
      </text>
    </g>
  );
  return (
    <Bench title="Which way does the copy go?" task="Place src and dst, then switch between one address space and separate ones.">
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label="Address spaces">
          <button aria-pressed={mode === 'uva'} data-mode="uva" onClick={() => setMode('uva')}>
            one shared address space (CUDA 4 and later)
          </button>
          <button aria-pressed={mode === 'separate'} data-mode="separate" onClick={() => setMode('separate')}>
            separate spaces (before)
          </button>
        </span>
        <label className="check ui" style={{ opacity: mode === 'uva' ? 1 : 0.5 }}>
          <input type="checkbox" checked={reg} disabled={mode !== 'uva'} onChange={(e) => setReg(e.target.checked)} />
          add a buffer registered with cudaHostRegister
        </label>
      </div>
      <div className="row ui" style={{ marginBottom: '0.4rem' }}>
        <span className="muted">Clicking the map places</span>
        <span className="seg" role="group" aria-label="Marker to place">
          <button aria-pressed={placing === 'src'} onClick={() => setPlacing('src')}>
            src
          </button>
          <button aria-pressed={placing === 'dst'} onClick={() => setPlacing('dst')}>
            dst
          </button>
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Address map" style={{ width: '100%', maxWidth: W, display: 'block', cursor: 'crosshair' }} onClick={place}>
        {lanes.map((lane, i) => (
          <g key={lane ?? 'one'}>
            <rect x={pad} y={laneY(i)} width={W - 2 * pad} height={laneH} fill="none" stroke="var(--rule)" />
            {rs
              .filter((x) => (lane === null ? true : x.space === lane))
              .map((x) => {
                const st = rangeFill(x);
                return (
                  <g key={x.label}>
                    <rect x={X(x.lo)} y={laneY(i) + (x.registered ? 4 : 0)} width={X(x.hi) - X(x.lo)} height={laneH - (x.registered ? 8 : 0)} fill={st.fill} stroke={st.stroke} strokeDasharray={st.dash} />
                    <text x={(X(x.lo) + X(x.hi)) / 2} y={laneY(i) + laneH / 2 + 5} fontSize={x.registered ? 11 : 13} textAnchor="middle">
                      {x.registered ? (x.registered === 'host-side' ? 'registered' : 'its device address') : lane ? SPACE_LABEL[x.space] + "'s space" : SPACE_LABEL[x.space]}
                    </text>
                  </g>
                );
              })}
          </g>
        ))}
        {marker(src, 'src')}
        {marker(dst, 'dst')}
        <text x={pad} y={H - 3} fontSize="11">0x0000</text>
        <text x={W - pad} y={H - 3} fontSize="11" textAnchor="end">0xffff</text>
      </svg>
      <div className="row" style={{ marginTop: '0.5rem' }}>
        <label className="field ui" style={{ flex: '1 1 14rem' }}>
          src = {hex4(src)}
          <input type="range" min={0} max={TOP - 0x100} step={0x100} value={src} aria-label="src address" onChange={(e) => setSrc(+e.target.value)} />
        </label>
        <label className="field ui" style={{ flex: '1 1 14rem' }}>
          dst = {hex4(dst)}
          <input type="range" min={0} max={TOP - 0x100} step={0x100} value={dst} aria-label="dst address" onChange={(e) => setDst(+e.target.value)} />
        </label>
      </div>
      <div className={'narrate' + (r.kind === 'ambiguous' || r.kind === 'unmapped' ? ' bad' : '')} data-kind={r.kind} style={{ marginTop: '0.7rem' }}>
        <p style={{ margin: 0 }}>
          <code>
            cudaMemcpy(dst, src, n, cudaMemcpyDefault)
          </code>{' '}
          →{' '}
          <b>
            {r.kind.startsWith('cudaMemcpy') ? <code>{r.kind}</code> : r.kind === 'ambiguous' ? 'cannot tell' : r.kind === 'unmapped' ? 'no allocation' : 'one buffer, two addresses'}
          </b>
        </p>
        <p style={{ margin: '0.3rem 0 0' }} data-detail>
          {r.detail}
        </p>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- stepper: follow the list
const HOP_TONE: Record<HopVerdict, string> = { local: 'chip ok', remote: 'chip', fault: 'chip', invalid: 'chip fail' };
const neutralChip = { border: '1px solid var(--rule)', background: 'var(--page)', color: 'var(--ink)' };

function FollowList() {
  const [setting, setSetting] = useState<Setting>('uva');
  const [nodes, setNodes] = useState<Alloc[]>(['cudaMalloc', 'cudaHostAlloc', 'malloc', 'cudaMalloc', 'cudaHostAlloc']);
  const [step, setStep] = useState(0);
  const hops = chase(nodes, setting);
  const addr = addresses(nodes);
  const shown = hops.slice(0, step);
  const cur = shown[shown.length - 1];
  const done = step >= hops.length;
  const failAt = hops.findIndex((h) => h.verdict === 'invalid');
  const label = SETTINGS.find((s) => s.id === setting)!.label;
  const reset = () => setStep(0);
  return (
    <Bench title="Follow the list" task="Choose where each node lives and the system, then step the kernel through the list.">
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field" style={{ maxWidth: '100%' }}>
          System
          <select
            style={{ maxWidth: '100%' }}
            value={setting}
            aria-label="system"
            onChange={(e) => {
              setSetting(e.target.value as Setting);
              reset();
            }}
          >
            {SETTINGS.map((s) => (
              <option key={s.id} value={s.id}>
                {s.label}
              </option>
            ))}
          </select>
        </label>
        <button className="primary" onClick={() => setStep((x) => Math.min(hops.length, x + 1))} disabled={done}>
          Step
        </button>
        <button onClick={() => setStep(failAt >= 0 ? failAt + 1 : hops.length)} disabled={done}>
          Run to failure
        </button>
        <button className="quiet" onClick={reset} disabled={step === 0}>
          Reset
        </button>
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', alignItems: 'stretch', marginBottom: '0.8rem' }}>
        {nodes.map((a, i) => {
          const h = shown[i];
          return (
            <div
              key={i}
              className="panel"
              data-node={i}
              data-verdict={h?.verdict ?? ''}
              style={{
                flex: '1 1 9rem',
                minWidth: 0,
                borderColor: h ? (h.verdict === 'invalid' ? 'var(--fail)' : h.verdict === 'local' ? 'var(--ok)' : 'var(--ink-3)') : undefined,
                background: h?.verdict === 'invalid' ? 'var(--fail-wash)' : undefined,
              }}
            >
              <p className="ui" style={{ margin: 0, fontSize: '0.78rem' }}>
                <b>node {i + 1}</b> · value {NODE_VALUES[i]}
                {i < nodes.length - 1 ? ' · next →' : ' · next = null'}
              </p>
              <select
                value={a}
                aria-label={`node ${i + 1} allocator`}
                style={{ width: '100%', margin: '0.3rem 0', fontSize: '0.8rem' }}
                onChange={(e) => {
                  const n = [...nodes];
                  n[i] = e.target.value as Alloc;
                  setNodes(n);
                  reset();
                }}
              >
                {ALLOCS.map((x) => (
                  <option key={x} value={x}>
                    {x}
                  </option>
                ))}
              </select>
              <p className="mono" style={{ margin: 0, fontSize: '0.72rem', overflowWrap: 'anywhere' }}>
                {hex(addr[i])}
              </p>
            </div>
          );
        })}
      </div>
      <div className={'narrate' + (cur?.verdict === 'invalid' ? ' bad' : '')} data-narrate>
        {!cur ? (
          <p style={{ margin: 0 }}>
            {label}. The kernel has been launched with the address of node 1 as its argument. Before you step, say at
            which node the walk will stop, if it stops.
          </p>
        ) : (
          <>
            <p style={{ margin: 0 }}>
              Hop {cur.index + 1}: the kernel dereferences <code>{hex(cur.address)}</code>, which it got from{' '}
              {cur.via === 'argument' ? 'its argument' : `node ${cur.index}'s next field`}. That address lies in{' '}
              {cur.range}.{' '}
              <span className={HOP_TONE[cur.verdict]} style={cur.verdict === 'remote' || cur.verdict === 'fault' ? neutralChip : undefined}>
                {HOP_LABEL[cur.verdict]}
              </span>
            </p>
            <p style={{ margin: '0.3rem 0 0' }}>
              {cur.rule.cost}{' '}
              {cur.verdict === 'invalid' ? (
                <b>
                  The walk stops here. The kernel read {cur.index} value{cur.index === 1 ? '' : 's'} correctly; the sum so far is {cur.sum}.
                </b>
              ) : done ? (
                <b>The kernel reached the end of the list: the sum is {cur.sum}.</b>
              ) : (
                <>Sum so far: {cur.sum}.</>
              )}
            </p>
          </>
        )}
      </div>
      {shown.length ? (
        <table className="threads" style={{ marginTop: '0.7rem' }} data-hops>
          <thead>
            <tr>
              <th>hop</th>
              <th>address</th>
              <th>range</th>
              <th>result</th>
              <th>sum</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((h) => (
              <tr key={h.index} className={h === cur ? 'cur' : undefined} data-hop={h.index} data-verdict={h.verdict}>
                <td>{h.index + 1}</td>
                <td>{hex(h.address)}</td>
                <td>{h.range}</td>
                <td>{HOP_LABEL[h.verdict]}</td>
                <td>{h.sum}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- your Mac's addresses (lab C.1)
const truthy = (v: unknown) => v === 1 || v === true || v === '1' || v === 'true' || v === 'yes';

export function addressRow(reports: Reports): { shared: boolean; nocopy: boolean; when: string } | null {
  const run = reports.metal;
  const row = benchRows(run, LAB).find((r) => r.name === 'addresses');
  if (!row || row.shared_same === undefined || row.nocopy_same === undefined) return null;
  return { shared: truthy(row.shared_same), nocopy: truthy(row.nocopy_same), when: run.generated.slice(0, 10) };
}

function YourAddresses() {
  const { reports, loading } = useReports();
  if (loading) return null;
  const a = addressRow(reports);
  if (!a)
    return (
      <p data-addresses="none">
        <b>Your Mac:</b> no observation yet. Run lab C.1 with <code>--bench</code> on the Metal backend (
        <code>python3 labs/pmpp.py run appC/lab1 --bench</code>) and reload; its "addresses" row compares the two values
        for you.
      </p>
    );
  const same = (b: boolean) => (b ? 'the same value' : 'different values');
  return (
    <p data-addresses="some" data-shared-same={a.shared ? '1' : '0'} data-nocopy-same={a.nocopy ? '1' : '0'}>
      <b>Observed on your machine, not documented</b> (lab C.1, {a.when}): for a shared buffer, <code>contents</code> and{' '}
      <code>gpuAddress</code> were {same(a.shared)}; for a buffer made with <code>newBufferWithBytesNoCopy</code>, they
      were {same(a.nocopy)}.
      {!a.shared && !a.nocopy ? (
        <>
          {' '}
          On your machine, then, the bytes are shared but the CPU and the GPU name them with different addresses: the
          picture of section C.2, two pointer values for one piece of data, not the picture of this section.
        </>
      ) : null}
    </p>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSC3: Record<string, Question> = {
  'appC/C.3/q-direction': {
    kind: 'choice',
    id: 'appC/C.3/q-direction',
    prompt: (
      <>
        Under UVA: <code>cudaMemcpy(dst, src, n, cudaMemcpyDefault)</code>, where <code>dst</code> came from{' '}
        <code>cudaMalloc</code> on GPU 1 and <code>src</code> from <code>cudaMallocHost</code>. What does the runtime do?
      </>
    ),
    options: [
      { label: 'A host-to-device copy into GPU 1, inferred from the two addresses.', correct: true },
      { label: 'It fails, because the kind of copy is missing.', why: 'cudaMemcpyDefault is the kind: it asks the runtime to infer the direction, which unified addressing allows.' },
      { label: 'A copy to the current device, whatever dst says.', why: 'The destination value says GPU 1; the current device does not matter for where the bytes go.' },
      { label: 'A device-to-host copy.', why: 'The direction is read off the values: src is host memory, dst is GPU memory.' },
    ],
    explain: (
      <p>
        Under unified addressing every allocation lies in one virtual address space, each processor in its own range, so
        a pointer's value says where it lives. <code>cudaMemcpyDefault</code> (value 4) "requires unified virtual
        addressing".
      </p>
    ),
    source: 'book p. 625; notes/today/ch02-facts.md A4; appC-facts A5',
  },
  'appC/C.3/q-address-not-access': {
    kind: 'choice',
    id: 'appC/C.3/q-address-not-access',
    prompt: <>A teammate says: "With UVA every pointer works everywhere." Which is a counterexample on CUDA today?</>,
    options: [
      { label: 'Host code dereferencing a cudaMalloc pointer.', correct: true },
      { label: 'A kernel reading cudaMallocHost memory.', why: 'Allowed: under unified addressing that pointer works in kernels directly, over the interconnect.' },
      { label: 'A kernel reading managed memory.', why: 'Allowed: that is what managed memory is for (section C.6).' },
      { label: 'cudaMemcpyDefault between two GPUs.', why: 'Allowed: the direction is inferred from the two values.' },
    ],
    explain: (
      <p>
        One address space is not one access right. Host code still may not dereference <code>cudaMalloc</code> memory,
        and NVIDIA says so for every kind of system today, hardware coherence included.
      </p>
    ),
    source: 'book p. 625; appC-facts A3, A12',
  },
  'appC/C.3/q-chase': {
    kind: 'numeric',
    id: 'appC/C.3/q-chase',
    prompt: (
      <>
        CUDA 4. The host builds a four-node list: nodes 1 and 2 in <code>cudaHostAlloc</code> memory, node 3 in{' '}
        <code>malloc</code> memory, node 4 in <code>cudaHostAlloc</code> memory, and passes the head to a kernel. How
        many node values does the kernel read correctly before its first invalid dereference?
      </>
    ),
    answer: 2,
    unit: 'values',
    explain: (
      <p>
        Nodes 1 and 2 are read over the interconnect. Reading node 2's <code>next</code> field is fine too, because the
        field itself lives in mapped memory. Dereferencing that value is not: it points into <code>malloc</code> memory,
        which a kernel may not touch under CUDA 4. Node 4 is never reached.
      </p>
    ),
    source: 'book p. 625',
  },
  'appC/C.3/q-hip-host': {
    kind: 'choice',
    id: 'appC/C.3/q-hip-host',
    prompt: (
      <>
        To debug a sampler, a teammate adds <code>printf("%f\n", d_logits[0]);</code> to host code, where{' '}
        <code>d_logits</code> came from <code>hipMalloc</code>. On the AMD machine it prints the right value. The same
        file also builds for CUDA, where <code>d_logits</code> comes from <code>cudaMalloc</code>. What do you tell them?
      </>
    ),
    options: [
      {
        label:
          'It works on HIP because AMD documents host reads of hipMalloc memory, as slow zero copy over the interconnect. CUDA forbids host code to dereference cudaMalloc memory, so the line is wrong in the CUDA build. Copy the value out with a memcpy call in both.',
        correct: true,
      },
      {
        label: 'It is fine on both: unified addressing lets the host read any device pointer.',
        why: 'Unified addressing tells the runtime where a pointer lives; it grants no access. CUDA host code still may not dereference cudaMalloc memory.',
      },
      {
        label: 'It is fine on both, only slow, because every read crosses the interconnect.',
        why: 'Slow is HIP\'s documented behaviour. On CUDA the read is not allowed at all.',
      },
      {
        label: 'It only worked by luck on HIP: host reads of device memory are undefined there too.',
        why: 'AMD documents it: "CPUs can access memory allocated via hipMalloc()". This is one of the places where HIP and CUDA really differ.',
      },
    ],
    explain: (
      <p>
        AMD's unified memory page: "CPUs can access memory allocated via hipMalloc()". CUDA forbids the same read, and
        NVIDIA's documentation does not say what happens when host code breaks the rule, so do not count on a clean
        error. A line like this passes every test on the AMD machine and is wrong the moment the file builds for CUDA:
        keep host reads of device memory out of code that must build for both, and out of hot loops on HIP.
      </p>
    ),
    source: 'not in the book; appC-facts B3; notes/today/ch02-facts.md A17 and B7',
  },
};

// ---------------------------------------------------------------- the page
export function SC3() {
  return (
    <>
      <p>
        Before CUDA 4, host and device each had their own virtual address space, each mapped onto its own physical
        memory. One physical location could therefore have two different virtual addresses, one on each side. That is
        exactly the situation of section C.2, where the host's pointer and the kernel's pointer to the same mapped
        bytes were different values.
      </p>
      <Source section="C.3" pages="625" />
      <p>
        CUDA 4, in 2011, introduced the <b>unified virtual address space</b>: one virtual address space shared by host
        and devices, in which, in the book's words, each physical address maps to only one virtual location. The runtime
        can now tell from a pointer's value whether it refers to host or device memory. So <code>cudaMemcpy</code> no
        longer needs to be told the direction: pass <code>cudaMemcpyDefault</code> and it reads the direction off the two
        pointers.
      </p>
      <Source section="C.3" pages="625" />
      <Today checked="2026-11-20 against the CUDA 13.5 guide and runtime reference (notes/today/appC-facts.md, A5; ch02-facts A4)">
        <p>
          CUDA 4 is from May 2011. Today's guide: "A single virtual address space is used for all host memory and all
          global memory on all GPUs", including memory from <code>malloc</code> and <code>mmap</code>, and it is on
          automatically in 64-bit processes. <code>cudaPointerGetAttributes</code> tells you where a pointer lives. The
          exceptions to "one location": registered and write-combined host memory get a distinct device address.
        </p>
      </Today>
      <p>
        Two names, one feature: appendix C says UVAS, and chapter 2's footnote on page 35 says UVA. NVIDIA uses both.
        The book also says UVAS came from GMAC, the research library of the 2010 paper it cites; that is the book's
        account, and NVIDIA's own pages do not mention it.
      </p>
      <Source pages="625" figs="UVA: ch. 2, p. 35, footnote 3" />

      <h2>Which way does the copy go?</h2>
      <p>
        Place a source and a destination on the address map and read off what <code>cudaMemcpyDefault</code> would do.
        Then switch to separate spaces, the situation before CUDA 4, and try again.
      </p>
      <Simplification>
        <p>
          Range bases and sizes are invented and tiny. One host and two GPUs. The model infers the direction only; it does
          not perform or time a copy.
        </p>
      </Simplification>
      <CopyDirection />
      <p>
        With separate spaces a value names a location in every space at once, so the direction has to come from you:
        that is why <code>cudaMemcpy</code> has a kind argument at all. Under one address space the value is enough.
        Add the registered buffer to see the documented exception: one buffer at two addresses (from{' '}
        {hex4(REGISTERED.host[0])} and from {hex4(REGISTERED.device[0])} on the map), the host's and the device's.
      </p>
      <Ask q={questionsSC3['appC/C.3/q-direction']} />

      <h2>An address is not an access right</h2>
      <p>
        What the unified address space does not give you is access. Under CUDA 4, host code still cannot dereference a
        pointer from <code>cudaMalloc</code>, and a kernel still cannot dereference ordinary host memory. The exception is
        memory from <code>cudaHostAlloc</code>: host code can pass that pointer straight to a kernel, and each access goes
        over PCIe. Pointers that a kernel reads from memory, such as the links of a list the host built, are safe only if
        they too point into <code>cudaHostAlloc</code> memory.
      </p>
      <Source section="C.3" pages="625" />
      <Today checked="2026-11-20 against the CUDA 13.5 guide (notes/today/appC-facts.md, A12)">
        <p>
          Still true of <code>cudaMalloc</code> memory on every kind of system: "Hardware coherency does not enable access
          to GPU-only allocations such as those made with cudaMalloc from the host."
        </p>
      </Today>
      <Ask q={questionsSC3['appC/C.3/q-address-not-access']} />

      <h2>Follow the list</h2>
      <p>
        Here is the kernel the book has in mind: one thread walking a linked list that the host built, summing as it
        goes. Every hop after the first dereferences a pointer the kernel read from memory.
      </p>
      <Code
        tabs={[{ label: 'CUDA', code: CHASE_CU, tiers: CHASE_TIERS, status: 'front-end-checked' }]}
        caption="labs/appC/snippets/c3_chase.cu. It has not run on an NVIDIA or AMD GPU."
      />
      <p>
        Now choose where each node lives and on which kind of system, and step the kernel through the list. Look for
        two things: a list that is fine for two hops and fails on the third, and what it takes for a list with a{' '}
        <code>malloc</code> node to work at all. Section C.8 settles the book's open question here, and the map on the{' '}
        <a href={href('appC', 'start')}>start page</a> shows the same rules.
      </p>
      <Simplification>
        <p>
          One thread and no timing. "Invalid dereference" stands for undefined behaviour, which on a GPU may crash the
          kernel or quietly return garbage. Addresses are invented, one range per allocator.
        </p>
      </Simplification>
      <FollowList />
      <p>
        A structure built by the host is safe for a kernel only if every node lives in memory the kernel may access.
        Pascal's page faults help only for managed nodes. A node in plain <code>malloc</code> memory needs HMM or ATS,
        which arrived years later.
      </p>
      <Ask q={questionsSC3['appC/C.3/q-chase']} />

      <h2>HIP: further than CUDA</h2>
      <p>
        HIP has the same pieces: <code>hipMemcpyDefault</code> and <code>hipPointerGetAttributes</code>. It goes one step
        further than CUDA on access. AMD documents that the CPU may read <code>hipMalloc</code> memory directly, as "zero
        copy" over the interconnect: slow, but allowed, where CUDA forbids it. And one trap: the HIP header comments{' '}
        <code>hipDeviceAttributeUnifiedAddressing</code> as "Cuda only", so do not use it to test for unified addressing
        on AMD.
      </p>
      <Beyond>
        <p>AMD's HIP documentation and the ROCm 10.2 header (appC-facts B3).</p>
      </Beyond>
      <Ask q={questionsSC3['appC/C.3/q-hip-host']} />

      <h2>Metal: buffers, not pointers</h2>
      <p>
        Metal never asks you to hand a kernel a raw CPU pointer. A kernel receives buffers by binding them, so the
        question of whether the CPU's address means the same thing on the GPU does not come up in ordinary code. Apple
        does document a 64-bit GPU virtual address per buffer, <code>gpuAddress</code>, and says nothing about how it
        relates to the buffer's <code>contents</code> pointer.
      </p>
      <Beyond>
        <p>Apple's Metal headers, macOS 27 SDK (appC-facts C3). The fact check found the relation undocumented.</p>
      </Beyond>
      <YourAddresses />
      <p>
        If you build a linked structure for a Metal kernel, put it inside one buffer and link the records by offsets
        from the buffer's start, not by pointers. An offset means the same thing to the CPU and to the GPU whatever the
        two addresses are.
      </p>
      <Beyond>
        <p>The advice to use offsets is ours.</p>
      </Beyond>

      <h2>Where this shows up in your code</h2>
      <p>
        When you write a copy helper that accepts either host or device tensors, <code>cudaMemcpyDefault</code> is the
        right kind: the runtime works out the direction from the pointers, and you cannot get it wrong. For managed
        pointers NVIDIA says to prefer it over a kind that does not match where the data actually is.
      </p>
      <Beyond>
        <p>
          NVIDIA: "Prefer cudaMemcpyDefault over an inaccurate cudaMemcpyKind hint" (appC-facts A14). The copy-helper
          framing is ours.
        </p>
      </Beyond>
      <p>
        Back to <a href={href('appC', 'start')}>the map</a>: this step is C.3. Compare its cells with C.2's: the pointer
        values changed, the access rules hardly did.
      </p>
    </>
  );
}
