import { useEffect, useMemo, useState, type CSSProperties, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { useNarrow } from '../../lib/narrow';
import { href } from '../../lib/router';
import {
  FIG_HEAD,
  FIG_PAGES,
  PROGRAM_FILE,
  PROGRAM_NAME,
  moved,
  pageList,
  runFig,
  steppable,
  type Counters,
  type FigLine,
  type FigRun,
  type Policy,
  type Program,
  type Step,
  type View,
} from '../../sim/managedmem';
import { TwoUnifiedScene } from './TwoUnifiedScene';

// Section C.6, Unified memory (book pp. 626–627). The stepper runs Fig. C.1 and our
// explicit-copy port on src/sim/managedmem.ts; the listings are the model's lines, which
// SC6.snippets.test.ts ties to labs/appC/snippets/c6_sortfile_*.cu character for character.

type Mode = Extract<Policy, 'bulk' | 'migrate' | 'one'>;
const MODES: Mode[] = ['bulk', 'migrate', 'one'];
const MODE_LABEL: Record<Mode, string> = {
  bulk: 'limited: CUDA 6 on Kepler/Maxwell; today Windows, WSL, some Tegra',
  migrate: 'full: compute capability 6.0 and later on Linux',
  one: 'Apple silicon, Metal shared buffer',
};
const MODE_SHORT: Record<Mode, string> = { bulk: 'limited', migrate: 'full', one: 'Apple silicon' };
const PROGRAMS: Program[] = ['cpu', 'managed', 'explicit']; // Fig. C.1's order, then our third column

// The three listings, aligned row by row like a side-by-side diff: [cpu, managed, explicit]
// line indices into figLines(), or null where a listing has no such line.
const ROWS: (number | null)[][] = [
  [0, 0, 0], // declaration
  [1, 1, 1], // malloc / cudaMallocManaged / malloc
  [null, null, 2], // cudaMalloc
  [2, 2, 3], // fread
  [null, null, 4], // copy in
  [3, 3, 5], // sort
  [4, 4, null], // (blank) / cudaDeviceSynchronize
  [null, null, 6], // copy out
  [5, 5, 7], // use_data
  [null, null, 8], // cudaFree(d_data)
  [6, 6, 9], // free / cudaFree
];

const plural = (n: number, one: string, many = one + 's') => `${n} ${n === 1 ? one : many}`;

// Listing styles (inline, like the rest of the app's interactives).
const MONO: CSSProperties = { fontFamily: 'var(--sans)', fontVariationSettings: "'MONO' 1, 'CASL' 0" };
const lnStyle = (chg: boolean, del: boolean, hl: boolean): CSSProperties => ({
  ...MONO,
  display: 'block',
  whiteSpace: 'pre',
  padding: '0 0.45rem 0 0.25rem',
  minHeight: '1.6em',
  fontWeight: chg ? 680 : undefined,
  textDecoration: del ? 'line-through' : undefined,
  color: del ? 'var(--fail)' : undefined,
  background: hl ? 'var(--rule-soft)' : undefined,
  boxShadow: hl ? 'inset 3px 0 0 var(--ink)' : undefined,
});

/** One sentence or three about what a line just did, from the model's events. */
function describe(program: Program, mode: Mode, line: FigLine, step: Step | null, fr: FigRun): ReactNode {
  const ev = step?.events ?? [];
  const count = (e: string, extra?: (x: (typeof ev)[number]) => boolean) => ev.filter((x) => x.e === e && (!extra || extra(x))).length;
  const crashed = step?.skipped;
  if (crashed) return <>The program has already stopped: this line never runs.</>;
  const err = step?.errors[0];
  switch (line.role) {
    case 'decl':
      return program === 'explicit'
        ? <>Two pointers: one for a CPU buffer, one for a GPU buffer. Chapter 2's model needs both.</>
        : <>Declares the pointer. No memory yet.</>;
    case 'alloc':
      if (line.op?.op === 'alloc' && line.op.kind === 'managed')
        return mode === 'one' ? (
          <>
            On Apple silicon the counterpart is a buffer with shared storage (<code>newBufferWithLength:options:</code> with{' '}
            <code>MTLResourceStorageModeShared</code>); its <code>contents</code> pointer is what the CPU uses. There is no
            managed mode to choose: managed storage "has no effect on Apple Silicon".
          </>
        ) : (
          <>
            <code>cudaMallocManaged</code> returns one pointer that both the CPU and the GPU may use. In this model no page is
            placed yet: each lands where it is first touched.
          </>
        );
      if (line.op?.op === 'alloc' && line.op.kind === 'device')
        return mode === 'one' ? (
          <>The counterpart on Metal is a buffer with private storage: in the same physical memory, but GPU only.</>
        ) : (
          <>
            <code>cudaMalloc</code>: {FIG_PAGES} pages of GPU memory. Host code may not dereference <code>d_data</code>.
          </>
        );
      return <><code>malloc</code>: ordinary CPU memory, not yet touched.</>;
    case 'read-file':
      return (
        <>
          <code>fread</code> writes all N bytes through <code>data</code> on the CPU. {plural(count('populate'), 'page')} placed in{' '}
          {mode === 'one' ? 'the one memory' : 'CPU memory'}, where {count('populate') === 1 ? 'it was' : 'they were'} first touched.
        </>
      );
    case 'copy-in':
      return mode === 'one' ? (
        <>
          The copy fills the GPU buffer from the CPU buffer: {plural(step!.counters.copiedInMemory, 'page')} copied inside one
          physical memory. No link to cross, but it is still a copy (on Metal, a blit into a private buffer).
        </>
      ) : (
        <>
          <code>cudaMemcpy</code> moves all {plural(count('move'), 'page')} to the GPU buffer, whatever the kernel will touch.
        </>
      );
    case 'sort':
      if (program === 'cpu') return <>The CPU sorts in place. Nothing moves.</>;
      if (program === 'explicit') return <>The launch returns at once. The kernel sorts <code>d_data</code> in GPU memory: local accesses only.</>;
      if (mode === 'one')
        return (
          <>
            On Metal: encode the dispatch and commit the command buffer. The GPU reads and writes the same pages in place.
            Nothing moves.
          </>
        );
      if (mode === 'bulk')
        return (
          <>
            The launch returns at once. Before the kernel runs, every managed page whose copy is in CPU memory moves to the GPU:{' '}
            {plural(count('move', (x) => x.why === 'bulk'), 'page')}, whatever the kernel touches. From now until the synchronise the
            CPU must not touch any managed memory.
          </>
        );
      return (
        <>
          The launch returns at once and nothing moves yet. Each page the kernel touches faults on its first touch and migrates:{' '}
          {plural(count('fault'), 'GPU fault')}, {plural(count('move'), 'page')} moved.
          {line.op?.op === 'launch' && line.op.reads.length < FIG_PAGES ? ' Untouched pages stay in CPU memory.' : ''}
        </>
      );
    case 'sync':
      return mode === 'one' ? (
        <>On Metal, <code>waitUntilCompleted</code> on the command buffer: the one duty one memory leaves you.</>
      ) : (
        <>
          <code>cudaDeviceSynchronize</code> waits until the kernel has finished. Now the CPU may touch the data
          {mode === 'bulk' ? ', and in limited mode, touch managed memory at all' : ''}.
        </>
      );
    case 'copy-out':
      return mode === 'one' ? (
        <>Waits for the kernel, then copies {plural(FIG_PAGES, 'page')} back inside the same memory.</>
      ) : (
        <>
          The second <code>cudaMemcpy</code> waits for the kernel, then moves all {plural(count('move'), 'page')} back.
        </>
      );
    case 'use':
      if (err?.kind === 'segfault')
        return (
          <>
            <b className="t-fail">Segmentation fault.</b> <code>use_data</code> touches managed memory while the kernel may
            still be running, which limited systems do not allow. The program stops here.
          </>
        );
      if (err?.kind === 'race')
        return (
          <>
            <b className="t-fail">A race.</b> No crash: the CPU may touch managed memory while a kernel runs. But nothing makes{' '}
            <code>use_data</code> wait for the sort, so it may read unsorted or partly sorted bytes in {pageList(err.pages)}.{' '}
            {count('fault') ? `(${plural(count('fault'), 'CPU fault')} pulled those pages back mid-sort.)` : ''}
          </>
        );
      if (err?.kind === 'ordering')
        return (
          <>
            <b className="t-fail">An ordering error.</b> Nothing moves on one memory, so nothing crashes either: the CPU reads{' '}
            <code>contents</code> before waiting for the command buffer, and may see unsorted bytes in {pageList(err.pages)}.
          </>
        );
      if (program !== 'managed' || mode === 'one') return <><code>use_data</code> reads every byte on the CPU. Nothing moves.</>;
      return (
        <>
          <code>use_data</code> reads every byte on the CPU.{' '}
          {count('fault')
            ? `Each page that lives in GPU memory faults back: ${plural(count('fault'), 'CPU fault')}, ${plural(count('move'), 'page')} moved.`
            : 'Every page is already in CPU memory.'}
        </>
      );
    case 'free':
      if (line.text.includes('cudaFree') && step?.events.some((x) => x.e === 'wait'))
        return (
          <>
            <code>cudaFree</code> "may perform implicit synchronization", so the kernel has finished by now in this model. Too
            late for <code>use_data</code>.
          </>
        );
      return line.text.includes('cudaFree') ? <><code>cudaFree</code> releases the memory.</> : <><code>free</code> releases the CPU buffer.</>;
    default:
      return null;
  }
}

/** The buffers of a view, drawn in the memory that holds each page. */
function Memories({ view, mode, hot }: { view: View; mode: Mode; hot: Set<string> }) {
  const bufs = Object.entries(view);
  const name = (b: string, kind: string) => `${b} (${kind === 'managed' ? (mode === 'one' ? 'shared buffer' : 'managed') : kind === 'device' ? 'cudaMalloc' : 'malloc'})`;
  const row = (box: 'cpu' | 'gpu' | 'one', b: string) => {
    const v = view[b];
    return (
      <div key={b} style={{ marginBottom: '0.5rem' }}>
        <div className="mx-label" style={{ fontWeight: 500 }}>{name(b, v.kind)}</div>
        <div className={'mx' + (box === 'gpu' ? ' global' : '')} style={{ gridTemplateColumns: `repeat(${v.pages.length}, auto)` }}>
          {v.pages.map((p, i) => {
            const here = box === 'one' ? p.loc === 'one' : p.loc === box || p.loc === 'both';
            const isHot = here && hot.has(`${b}:${i}`);
            return (
              <span key={i} data-box={box} data-buf={b} data-page={i} data-present={here ? 'yes' : 'no'}
                className={here ? (box === 'gpu' ? (isHot ? 'hot' : 'on') : '') : 'void'}
                style={here && box !== 'gpu' ? { borderColor: 'var(--ink-2)', fontWeight: 600, boxShadow: isHot ? 'inset 0 0 0 2px var(--ink)' : undefined } : undefined}>
                {here ? `p${i}` : ''}
              </span>
            );
          })}
        </div>
      </div>
    );
  };
  if (mode === 'one')
    return (
      <div className="panel" data-memory="one">
        <div className="mx-label">one system memory, used by CPU and GPU</div>
        {bufs.length ? bufs.map(([b]) => row('one', b)) : <p className="muted" style={{ margin: 0 }}>Nothing allocated yet.</p>}
      </div>
    );
  const where = (b: string, box: 'cpu' | 'gpu') => {
    const k = view[b].kind;
    return k === 'managed' || (k === 'host' ? box === 'cpu' : box === 'gpu');
  };
  return (
    <div className="cols" style={{ gridTemplateColumns: 'repeat(2, minmax(0, 1fr))' }}>
      {(['cpu', 'gpu'] as const).map((box) => (
        <div key={box} className="panel" data-memory={box}>
          <div className={'mx-label' + (box === 'gpu' ? ' t-global' : '')}>{box === 'cpu' ? 'CPU memory' : 'GPU memory'}</div>
          {bufs.filter(([b]) => where(b, box)).map(([b]) => row(box, b))}
          {!bufs.some(([b]) => where(b, box)) ? <p className="muted" style={{ margin: 0 }}>Nothing here.</p> : null}
        </div>
      ))}
    </div>
  );
}

function CounterRow({ c, mode, status }: { c: Counters; mode: Mode; status: ReactNode }) {
  const items: [string, number][] = [
    ['pages to GPU', c.h2d],
    ['pages to CPU', c.d2h],
    ['GPU faults', c.gpuFaults],
    ['CPU faults', c.cpuFaults],
  ];
  if (mode === 'one') items.push(['copied inside memory', c.copiedInMemory]);
  return (
    <div className="row" style={{ margin: '0.8rem 0 0' }}>
      {items.map(([label, v]) => (
        <span key={label} data-counter={label}>
          {label} <b data-value="">{v}</b>
        </span>
      ))}
      <span className="muted" data-status="">{status}</span>
    </div>
  );
}

function FigStepper() {
  const narrow = useNarrow();
  const [mode, setMode] = useState<Mode>('migrate');
  const [program, setProgram] = useState<Program>('managed');
  const [half, setHalf] = useState(false);
  const [noSync, setNoSync] = useState(false);
  const [n, setN] = useState(0);
  const runs = useMemo(
    () => Object.fromEntries(PROGRAMS.map((p) => [p, runFig(p, mode, { half, noSync })])) as Record<Program, FigRun>,
    [mode, half, noSync],
  );
  const fr = runs[program];
  const order = useMemo(() => steppable(fr.lines), [fr]);
  const total = order.length;
  const at = Math.min(n, total);
  useEffect(() => setN(0), [mode, program, half, noSync]);

  const curLine = at > 0 ? order[at - 1] : -1;
  const curStepIdx = curLine >= 0 ? fr.stepOf[curLine] : -1;
  const curStep = curStepIdx >= 0 ? fr.run.steps[curStepIdx] : null;
  // The last operation that has run so far decides what the memories look like.
  let lastIdx = -1;
  for (let k = 0; k < at; k++) if (fr.stepOf[order[k]] >= 0) lastIdx = fr.stepOf[order[k]];
  const last = lastIdx >= 0 ? fr.run.steps[lastIdx] : null;
  const view = last?.view ?? fr.run.initial;
  const counters = last?.counters ?? { h2d: 0, d2h: 0, gpuFaults: 0, cpuFaults: 0, remote: 0, invalidations: 0, evictions: 0, copiedInMemory: 0 };
  const hot = new Set((curStep?.events ?? []).filter((e) => e.page >= 0 && e.e !== 'local').map((e) => `${e.buf}:${e.page}`));
  const crashed = !!last?.crashed;
  const status = crashed ? 'program stopped' : last?.active ? 'kernel may be running' : at > 0 ? 'no kernel running' : '';

  const cell = (p: Program, idx: number | null) => {
    if (idx === null) return <span style={lnStyle(false, false, false)} />;
    const l = runs[p].lines[idx];
    const hl = p === program && idx === curLine;
    return (
      <span style={lnStyle(l.changed, l.deleted, hl)} data-line={idx} data-program={p} data-current={hl ? 'yes' : undefined}
        data-changed={l.changed ? 'yes' : undefined} data-deleted={l.deleted ? 'yes' : undefined}>
        {'  ' + (l.text.replace(/^ {4}/, '') || ' ')}
      </span>
    );
  };

  return (
    <Bench
      title="Fig. C.1, line by line"
      task="Step the managed port in full mode first. Then change one thing at a time."
      foot={
        <>
          Bold lines differ from the CPU original (Fig. C.1 prints them in red). Pages are numbered p0 to p{FIG_PAGES - 1}; a
          page drawn in a memory has its valid copy there. The listings are{' '}
          <code>labs/appC/snippets/{PROGRAM_FILE.cpu}</code>, <code>{PROGRAM_FILE.explicit}</code> and{' '}
          <code>{PROGRAM_FILE.managed}</code>, front-end-checked.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field" style={{ maxWidth: '100%' }}>
          System
          <select value={mode} onChange={(e) => setMode(e.target.value as Mode)} aria-label="System" style={{ maxWidth: '100%' }}>
            {MODES.map((m) => <option key={m} value={m}>{MODE_LABEL[m]}</option>)}
          </select>
        </label>
        <span className="seg" role="group" aria-label="Program to step">
          {PROGRAMS.map((p) => (
            <button key={p} aria-pressed={p === program} onClick={() => setProgram(p)}>{PROGRAM_NAME[p]}</button>
          ))}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="check"><input type="checkbox" checked={half} onChange={(e) => setHalf(e.target.checked)} /> kernel touches only the first half</label>
        <label className="check"><input type="checkbox" checked={noSync} onChange={(e) => setNoSync(e.target.checked)} /> delete <code>cudaDeviceSynchronize()</code></label>
      </div>

      <div role="group" aria-label="The three listings"
        style={{ display: 'grid', gridTemplateColumns: 'repeat(3, max-content)', gap: '0 0.6rem', fontSize: '0.71rem', lineHeight: 1.6, overflowX: 'auto' }}>
        {PROGRAMS.map((p, k) => (narrow && p !== program ? null : (
          <div key={p} data-listing={p} data-live={p === program ? 'yes' : 'no'}
            style={{ background: 'var(--cell)', border: `1px solid ${p === program ? 'var(--ink-3)' : 'var(--rule)'}`, borderRadius: 4, paddingBottom: '0.3rem', opacity: p === program ? 1 : 0.62 }}>
            <div style={{ fontSize: '0.78rem', fontWeight: 620, padding: '0.25rem 0.4rem', borderBottom: '1px solid var(--rule)', marginBottom: '0.2rem' }}>
              {PROGRAM_NAME[p]}{p === 'explicit' ? ' (ours)' : ''}
            </div>
            <span style={lnStyle(false, false, false)}>{FIG_HEAD}</span>
            {ROWS.map((r, i) => <div key={i}>{cell(p, r[k])}</div>)}
            <span style={lnStyle(false, false, false)}>{'}'}</span>
          </div>
        )))}
      </div>
      {narrow ? <p className="muted" style={{ margin: '0.4rem 0 0' }}>On a wider screen the three listings sit side by side.</p> : null}

      <div className="row" style={{ margin: '0.9rem 0 0.7rem' }}>
        <button onClick={() => setN(0)} disabled={at === 0}>Restart</button>
        <button onClick={() => setN(at - 1)} disabled={at === 0}>Back</button>
        <button className="primary" onClick={() => setN(at + 1)} disabled={at >= total}>Step</button>
        <button onClick={() => setN(total)} disabled={at >= total}>Run to the end</button>
        <span className="muted" aria-live="polite">line {at} of {total}</span>
      </div>
      <div className={'narrate' + (curStep?.errors.length ? ' bad' : '')} role="status" style={{ marginBottom: '0.9rem' }}>
        {curLine >= 0 ? (
          <>
            <code>{fr.lines[curLine].text.trim()}</code> {describe(program, mode, fr.lines[curLine], curStep, fr)}
          </>
        ) : (
          <>Nothing has run. Press Step to run the first line of the listing "{PROGRAM_NAME[program]}".</>
        )}
      </div>
      <Memories view={view} mode={mode} hot={hot} />
      <CounterRow c={counters} mode={mode} status={status} />

      <div className="mx-label" style={{ margin: '1.2rem 0 0.3rem' }}>
        Whole program, {MODE_SHORT[mode]}{half ? ', kernel touches half' : ''}{noSync ? ', synchronise deleted' : ''}
      </div>
      <table className="data" style={{ margin: 0, background: 'var(--page)' }} aria-label="Totals for the three programs">
        <thead>
          <tr>
            <th style={{ paddingLeft: '0.6rem' }}>Program</th>
            <th className="num">to GPU</th>
            <th className="num">to CPU</th>
            <th className="num">GPU faults</th>
            <th className="num">CPU faults</th>
            {mode === 'one' ? <th className="num">copied inside memory</th> : null}
            <th>Outcome</th>
          </tr>
        </thead>
        <tbody>
          {PROGRAMS.map((p) => {
            const r = runs[p].run;
            const e = r.errors[0];
            return (
              <tr key={p} data-program={p} data-moved={moved(r.counters)}>
                <td style={{ paddingLeft: '0.6rem' }}>{PROGRAM_NAME[p]}</td>
                <td className="num">{r.counters.h2d}</td>
                <td className="num">{r.counters.d2h}</td>
                <td className="num">{r.counters.gpuFaults}</td>
                <td className="num">{r.counters.cpuFaults}</td>
                {mode === 'one' ? <td className="num">{r.counters.copiedInMemory}</td> : null}
                <td data-outcome={e ? e.kind : 'ok'}>
                  {e ? <span className="chip fail">{e.kind === 'segfault' ? 'segmentation fault' : e.kind === 'race' ? 'race' : 'ordering error'}</span> : <span className="chip ok">correct</span>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </Bench>
  );
}

function ShapesTable() {
  return (
    <table className="data wide" style={{ maxWidth: '58rem' }} aria-label="Four shapes of unified memory">
      <thead>
        <tr><th>Shape</th><th>Examples</th><th>What moves</th><th>What you still owe</th></tr>
      </thead>
      <tbody>
        <tr>
          <td>Discrete GPU, migration in software</td>
          <td>CUDA Unified Memory; HIP managed memory</td>
          <td>Pages, between two memories, on fault (or all at launch when limited)</td>
          <td>Ordering; and on limited systems, no CPU touch while a kernel runs</td>
        </tr>
        <tr>
          <td>Two physical memories, one page table, coherent in hardware</td>
          <td>Grace Hopper (NVLink-C2C)</td>
          <td>Less faulting: coherent at cache-line granularity, pages moved by access counters</td>
          <td>Ordering</td>
        </tr>
        <tr>
          <td>One shared DRAM, caching rules per buffer type</td>
          <td>NVIDIA Tegra; Apple silicon</td>
          <td>Nothing between memories. On Apple, a shared buffer needs no copy</td>
          <td>Ordering, and choosing the buffer type. On some Tegra systems managed memory follows the limited rules</td>
        </tr>
        <tr>
          <td>One shared HBM</td>
          <td>AMD MI300A</td>
          <td>Nothing: "physical page migration is neither needed nor useful"</td>
          <td>Ordering; XNACK still on, for the mappings</td>
        </tr>
      </tbody>
    </table>
  );
}

function PagesReveal({ predicted }: { predicted: number }) {
  const m = moved(runFig('managed', 'migrate').run.counters);
  const e = moved(runFig('explicit', 'migrate').run.counters);
  return (
    <p>
      The model says <b>{m}</b>: {m / 2} pages over when the kernel touches them, {m / 2} back when <code>use_data</code> reads
      them. The explicit-copy port moves {e}.{' '}
      {predicted === m ? 'You had it.' : predicted < m ? 'Fewer than you might hope: one pointer does not mean one memory.' : 'Each page crosses once each way, no more.'}
    </p>
  );
}

export const questionsSC6: Record<string, Question> = {
  'appC/C.6/q-delete-sync': {
    kind: 'choice',
    id: 'appC/C.6/q-delete-sync',
    prompt: (
      <>
        Delete <code>cudaDeviceSynchronize()</code> from the managed <code>sortfile</code>. What happens on a Windows
        workstation, and on Linux with a compute capability 7.x GPU?
      </>
    ),
    options: [
      {
        label: 'Both work: Unified Memory keeps CPU and GPU coherent, so use_data sees the sorted buffer.',
        why: 'Coherent pages do not order anything. Nothing makes use_data wait for the sort, and on Windows the CPU must not touch managed memory at all while the GPU is active.',
      },
      {
        label: 'Windows: use_data segfaults, touching managed memory while the GPU is active. Linux: it may read unsorted data.',
        correct: true,
      },
      {
        label: 'Both crash with a segmentation fault: no system lets the CPU touch managed memory before a synchronise.',
        why: 'With compute capability 6.0 or later on Linux, the CPU may touch managed memory while a kernel runs. Allowed is not the same as correct.',
      },
      { label: 'Linux crashes, because pages fault back to the CPU during the sort, and Windows works.', why: 'The other way round: Windows is the limited mode.' },
    ],
    explain: (
      <p>
        Windows, WSL and some Tegra systems run Unified Memory in the limited mode, where any CPU access to managed memory
        while the GPU is active, that is, until it is synchronised, is a segmentation fault. NVIDIA's own example writes <code>y = 20</code> right after a launch:
        it runs on compute capability 6.x and fails before 6.0 and on Windows. On Linux with full support there is no crash,
        and that is the trap: <code>use_data</code> now races with the sort. The book says it plainly: the program still has
        to synchronise concurrent access.
      </p>
    ),
    source: 'book §C.6, pp. 626–627; notes/today/appC-facts.md A10',
  },
  'appC/C.6/q-oversub-windows': {
    kind: 'choice',
    id: 'appC/C.6/q-oversub-windows',
    prompt: (
      <>
        Windows. <code>cudaMallocManaged</code> of 1.5 times the GPU's memory, then a kernel over all of it. What happens?
      </>
    ),
    options: [
      { label: 'Pages fault in on demand, and the least recently used pages are evicted back to the CPU.', why: 'That is full Unified Memory on Linux. Windows is the limited mode: no GPU page faults.' },
      { label: 'It runs, at the speed of the interconnect.', why: 'It never gets as far as running.' },
      {
        label: "The driver's Sysmem Fallback spills the excess into system memory, so it runs, more slowly.",
        why: 'That Windows driver feature is for ordinary allocations that run out of GPU memory. The limit on managed memory is documented separately.',
      },
      { label: 'Not allowed: a limited system cannot allocate more managed memory than the GPU has.', correct: true },
    ],
    explain: (
      <p>
        NVIDIA's guide, on limited systems: "They cannot allocate more managed memory than the physical size of GPU memory."
        That is the book's third Kepler and Maxwell limit, still in force wherever <code>cudaDevAttrConcurrentManagedAccess</code>{' '}
        is 0. Oversubscription needs full Unified Memory, which section C.8 takes apart.
      </p>
    ),
    source: 'book §C.6, p. 627; notes/today/appC-facts.md A10, A18',
  },
  'appC/C.6/q-same-bytes': {
    kind: 'choice',
    id: 'appC/C.6/q-same-bytes',
    prompt: (
      <>
        Discrete GPU, full Unified Memory, and the kernel touches every byte of the file once. Compared with the explicit-copy
        port, how many bytes cross the link?
      </>
    ),
    options: [
      { label: 'About half: only the pages the kernel touches move, and the rest stay put.', why: 'It touches all of it, so all of it moves, and comes back when use_data reads it.' },
      { label: 'About the same: every page in once and out once.', correct: true },
      { label: 'None: the pointer is shared.', why: 'The pointer is shared; the memory is not. On a discrete GPU the bytes still have to reach GPU memory.' },
      { label: 'About double: every page faults twice.', why: 'Each page faults once in each direction: one crossing each way, as the two copies make.' },
    ],
    explain: (
      <p>
        In the stepper, full mode with the kernel touching everything: {FIG_PAGES} pages over and {FIG_PAGES} back, the same{' '}
        {2 * FIG_PAGES} the two <code>cudaMemcpy</code> calls move. Managed memory saved a second buffer and three lines, not
        traffic. NVIDIA said as much at launch: a carefully tuned program using streams and <code>cudaMemcpyAsync</code> "may
        very well perform better" than one that only uses Unified Memory. Where it does save traffic is when the kernel
        touches part of the data.
      </p>
    ),
    source: 'the stepper\'s model (src/sim/managedmem.ts); notes/today/appC-facts.md A9',
  },
  'appC/C.6/q-apple': {
    kind: 'choice',
    id: 'appC/C.6/q-apple',
    prompt: (
      <>
        You port <code>sortfile</code> to Metal on Apple silicon with shared buffers. Which of Fig. C.1's changes have a
        counterpart?
      </>
    ),
    options: [
      {
        label: 'All four, plus a didModifyRange call after fread so that the GPU sees what the CPU wrote.',
        why: 'didModifyRange belongs to managed storage, which is deprecated and "has no effect on Apple Silicon". And Metal has no free call.',
      },
      { label: 'A shared buffer, the launch, and a wait before use_data. Nothing migrates.', correct: true },
      { label: 'None: with unified memory you need no wait.', why: 'You still owe ordering: Apple asks you to make sure the GPU\'s changes complete before the CPU reads them.' },
    ],
    explain: (
      <p>
        Three of the four changes survive: the allocation (a buffer with shared storage, whose <code>contents</code> the CPU
        writes), the launch (encode and commit), and the wait (<code>waitUntilCompleted</code>). The fourth has none: Metal
        buffers are objects, released like any other, with no free call. There is no managed mode to migrate anything, and
        none is needed.
      </p>
    ),
    source: 'notes/today/ch05-facts.md C6; notes/today/appC-facts.md C2, C5 (citing ch02-facts C2)',
  },
  'appC/C.6/q-hip-xnack': {
    kind: 'choice',
    id: 'appC/C.6/q-hip-xnack',
    prompt: (
      <>
        <code>hipMallocManaged</code> on an AMD GPU with <code>HSA_XNACK</code> unset. Where does the kernel read the data from?
      </>
    ),
    options: [
      { label: 'From device memory, after each page migrates there on its first touch, as on NVIDIA.', why: 'Access-driven migration needs HSA_XNACK=1 and a driver with HMM; without them it is disabled.' },
      { label: 'Nowhere: without XNACK, hipMallocManaged returns an error and allocates nothing.', why: 'The call still works. AMD says managed memory then operates "in a degraded fashion".' },
      { label: 'From host memory, over the link, as if it were pinned host memory.', correct: true },
      { label: 'From device memory, after the runtime copies the whole allocation in bulk at the launch.', why: 'Not what AMD documents for this case: the header says the call then behaves the same as hipMallocHost, which allocates host memory.' },
    ],
    explain: (
      <p>
        AMD: without <code>HSA_XNACK=1</code> and a kernel driver with HMM, "access-driven memory migration will be disabled",
        and the header says <code>hipMallocManaged</code> then "behaves the same as hipMallocHost". The kernel reads pinned host
        memory over the link, which is C.2's zero-copy, not C.6's Unified Memory.
      </p>
    ),
    source: 'notes/today/appC-facts.md B5',
  },
};

export function SC6() {
  return (
    <>
      <p>
        CUDA 6 added <b>Unified Memory</b>: a pool of <b>managed</b> pages that the CPU and the GPU share through one pointer.
        A managed page can live in CPU memory, in GPU memory, or in both, and the CUDA runtime and the hardware move it and
        keep the copies coherent. To CPU code managed memory looks like CPU memory; to a kernel it looks like GPU memory.
        What it does not do is order your accesses: if the CPU and the GPU may touch the same data at the same time, the
        program still has to synchronise them.
      </p>
      <Source section="C.6" pages="626" />
      <p>
        The interface is one call. <code>cudaMallocManaged(&amp;ptr, bytes)</code> returns a pointer that host code and kernels
        may both dereference, and <code>cudaFree</code> releases it. No second buffer, no copy calls: the system decides when
        bytes move. This section is about what that buys and what it costs.
      </p>
      <Today checked="2026-11-20 (notes/today/appC-facts.md, A9)">
        <p>
          The book says 2013. NVIDIA announced Unified Memory in November 2013, the toolkit shipped in April 2014, and NVIDIA
          itself now says "introduced in 2014". Today's signature is{' '}
          <code>cudaMallocManaged(void **devPtr, size_t size, unsigned int flags = cudaMemAttachGlobal)</code>; the memory is
          not cleared. <code>__managed__</code> variables and managed memory pools are two other ways to get it.
        </p>
      </Today>

      <h2>Four changes to a CPU function</h2>
      <p>
        Fig. C.1 is a small CPU function: allocate a buffer, read a file into it, sort the bytes, hand them on. The managed
        port changes four lines. <code>malloc</code> becomes <code>cudaMallocManaged</code>, the sort becomes a kernel launch,
        a <code>cudaDeviceSynchronize()</code> appears where the CPU version had a blank line, and <code>free</code> becomes{' '}
        <code>cudaFree</code>. The <code>fread</code> and the <code>use_data</code> lines do not change: the CPU writes and
        reads through the same pointer the kernel sorts. The book's point is porting: the host side of the port is small. A
        parallel sort kernel still has to exist.
      </p>
      <Source section="C.6" pages="627" figs="Fig. C.1" />
      <p>
        The stepper adds a third column, the same port done Chapter 2's way: a second buffer from <code>cudaMalloc</code> and
        two <code>cudaMemcpy</code> calls. Then it asks, after every line, where each page of the buffer is.
      </p>
      <Beyond>
        <p>
          The explicit-copy column is ours. Fig. C.1 is NVIDIA's own CUDA 6 example, which compares only the CPU version and
          the managed one (appC-facts A9).
        </p>
      </Beyond>
      <Predict
        id="appC/C.6/p-pages"
        prompt={
          <>
            A discrete GPU with full Unified Memory. The file fills {FIG_PAGES} pages and the sort touches all of them. Over the
            whole managed <code>sortfile</code>, how many page crossings between CPU and GPU memory, both directions together?
          </>
        }
        unit="pages"
      >
        {(v) => <PagesReveal predicted={v} />}
      </Predict>
      <p>
        Step the managed port in full mode first, watching the two memories. The table under the stepper keeps the score for
        all three programs on the system you choose.
      </p>
      <Simplification>
        <p>
          Pages, not bytes or time, and {FIG_PAGES} of them. A page goes back to the CPU on the CPU's first touch after the
          synchronise; NVIDIA's guide says only that limited systems migrate managed memory back "when the GPU is
          synchronized" and that the CPU side can fault. First touch stands in for the driver's placement heuristics. A
          kernel makes all its accesses at its launch.
        </p>
      </Simplification>
      <FigStepper />
      <p>
        Three things to find. With the kernel touching everything, compare the managed port's total with the explicit
        copies'. Then tick <i>kernel touches only the first half</i> and compare again, in full mode and in limited mode:
        which of the three programs now moves less? Last, switch the system to Apple silicon and see what is left to move.
      </p>
      <Ask q={questionsSC6['appC/C.6/q-same-bytes']} />
      <p>
        The listings are files that pass the CUDA front-end check, so two things differ from the page. The figure prints{' '}
        <code>cudaDeviceSynchronize()</code> without a semicolon; as printed, the CUDA front-end check stops right there. And
        it elides the launch configuration as <code>&lt;&lt;&lt;...&gt;&gt;&gt;</code>; ours names it <code>grid</code> and{' '}
        <code>block</code>, with a comment saying they stand in for whatever a real sort would choose.
      </p>
      <p>
        Two other slips on the page do not touch the code. The text calls the replaced function <code>qsort()</code> while
        the figure calls it <code>qsort_char</code>; the files keep the figure's name. And the figure's heading reads "CUDA
        6&nbsp;&nbsp;code", with a double space.
      </p>
      <Source section="C.6" pages="627" figs="Fig. C.1" />

      <h2>What Kepler and Maxwell could not do</h2>
      <p>
        CUDA 6 ran on GPUs that could not keep CPU and GPU memory coherent in hardware, so most of the migration was done in
        software, with three limits. All managed memory the CPU had modified was flushed to the GPU before every launch. The
        CPU and the GPU could not access managed memory at the same time. And the managed pool could be no larger than the
        GPU's physical memory.
      </p>
      <Source section="C.6" pages="627" />
      <p>
        This is not history. NVIDIA documents exactly this behaviour today wherever{' '}
        <code>cudaDevAttrConcurrentManagedAccess</code> is 0, which means every Windows system, WSL, and some Tegra systems:
        managed memory moves to the GPU when a kernel starts, the CPU must not touch it while the GPU is active ("Concurrent
        CPU/GPU accesses, even to different managed memory allocations, will cause a segmentation fault"), it migrates back
        when the GPU is synchronised, and oversubscription is not allowed. That is the stepper's limited mode.
      </p>
      <Today checked="2026-11-20 (notes/today/appC-facts.md, A10)">
        <p>
          The book's architectures carry compute capability 3.x (Kepler) and 5.x (Maxwell); full Unified Memory needs 6.0
          (Pascal) or later, on Linux. Query <code>cudaDevAttrConcurrentManagedAccess</code> rather than guessing from the
          GPU.
        </p>
      </Today>
      <p>
        Now debug it. In the stepper, tick <i>delete <code>cudaDeviceSynchronize()</code></i> and run the managed port in each
        system. Predict first: which systems crash, and which give a wrong answer without crashing?
      </p>
      <Ask q={questionsSC6['appC/C.6/q-delete-sync']} />
      <Ask q={questionsSC6['appC/C.6/q-oversub-windows']} />

      <h2>Two things called unified memory</h2>
      <p>
        If you run models on a Mac you have heard "unified memory" for years, and it means something else. The difference
        is not in a picture of two boxes; it is in what moves, and when. Watch the same program on both.
      </p>
      <Simplification>
        <p>The animation runs the stepper's model and shares its simplifications: pages, not bytes or time.</p>
      </Simplification>
      <TwoUnifiedScene />
      <p>
        CUDA's term names a software-managed pool over separate memories: "a CUDA feature that enables managed memory which
        can be automatically migrated between CPU and GPUs". Apple's names hardware: "the CPU and the GPU share system
        memory". And these are only two of four shapes in use today, each with its own answer to "what moves, and what do I
        still owe?"
      </p>
      <Today checked="2026-11-20 (notes/today/appC-facts.md, A17, B8; ch05-facts C6)">
        <p>
          Grace Hopper: CPU and GPU "share a single per-process page table". Tegra: one DRAM, but each buffer type "has
          different accessing and caching behaviors". MI300A: XNACK "is still necessary". Apple silicon:{' '}
          <code>MTLStorageModeManaged</code> is deprecated, "Managed storage has no effect on Apple Silicon".
        </p>
      </Today>
      <ShapesTable />
      <p>
        One row is the whole of the book's appendix. The other three are where most of the confusion comes from, and the
        last column is the same everywhere: ordering is yours.
      </p>
      <Ask q={questionsSC6['appC/C.6/q-apple']} />

      <h2>Where you will meet it</h2>
      <p>
        llama.cpp has a switch for this. With <code>GGML_CUDA_ENABLE_UNIFIED_MEMORY=1</code> its CUDA backend allocates with{' '}
        <code>cudaMallocManaged</code> instead of <code>cudaMalloc</code>, which, in its own words, "allows swapping to system
        RAM instead of crashing when the GPU VRAM is exhausted", on Linux. The same source builds the HIP backend, where it
        also advises <code>hipMemAdviseSetCoarseGrain</code>, and its documentation warns that on HIP the switch "hurts
        performance for non-integrated GPUs". MLX uses managed memory for host-visible arrays on its CUDA backend and shared
        buffers on Metal: the same promise, one pointer for both sides, from two different mechanisms.
      </p>
      <Beyond>
        <p>
          Read from llama.cpp's <code>docs/build.md</code> and <code>ggml-cuda.cu</code>, and MLX's allocators, on
          2026-11-20 (appC-facts D3, D5; ch02-facts D3).
        </p>
      </Beyond>

      <h2>On HIP and Metal</h2>
      <p>
        HIP has the same calls, <code>hipMallocManaged</code> and <code>__managed__</code>, implemented on Linux and "under
        development" on Windows. Migration on access needs two things: <code>HSA_XNACK=1</code> in the environment and a
        kernel driver with HMM. Without them the call still succeeds, but the header says it "behaves the same as
        hipMallocHost": the kernel reads pinned host memory over the link. AMD also asks for sizes at a granularity of 4 KB
        ("it should be granularity of 4KB"). Metal has no managed allocation at all: a shared buffer on one memory has
        nothing to migrate.
      </p>
      <Beyond>
        <p>
          The book covers CUDA only. HIP from AMD's unified memory page and the ROCm 10.2 headers (appC-facts B5); Metal from
          Apple's storage-mode documentation (ch05-facts C6).
        </p>
      </Beyond>
      <Ask q={questionsSC6['appC/C.6/q-hip-xnack']} />
      <p>
        What makes the full mode possible is that a GPU can take a page fault and carry on. Section{' '}
        <a href={href('appC', 'C.8')}>C.8</a> takes one fault apart and asks when moving pages is the wrong answer. Lab C.1 on
        the <a href={href('appC', 'labs')}>labs page</a> feeds one kernel three ways, copies, mapped host memory and managed
        memory, on your own GPU.
      </p>
      <p>
        Back to <a href={href('appC', 'start')}>the map</a>: this step is C.6, Unified Memory in its limited form. The
        cells it changes are the managed-memory row, and on Windows, WSL and some Tegra systems they are still today's.
      </p>
    </>
  );
}
