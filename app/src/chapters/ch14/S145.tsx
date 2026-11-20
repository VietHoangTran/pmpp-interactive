import { useEffect, useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import type { Tier } from '../../components/notes';
import { Beyond, Bench, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import { useSolutionStatus } from './solutionStatus';
import { BOOK_KEYS, DEFECT_LABEL, bin, parallelPass, randomKeys, type Defect, type ParallelPass } from '../../sim/radixsort';

// One pass as the lesson shows it: the region "pass" of labs/ch14/snippets/s145_radix_iter.cu
// (Fig. 14.7's lines split at the scan into two kernels and a host function), and the region
// "scan" of the same file. The file is front-end-checked and run on the emulator against
// src/sim/radixsort.ts by S145.snippets.test.ts, which also fails if these copies drift or
// if the line numbers in STEPS stop pointing at the lines they name.
const PASS = `__global__ void radix_bits_kernel(unsigned int* input, unsigned int* bits,
        unsigned int N, unsigned int iter) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < N) {
        unsigned int key = input[i];
        unsigned int bit = (key >> iter) & 1;
        bits[i] = bit;
    }
}

__global__ void radix_scatter_kernel(unsigned int* input,
        unsigned int* output, unsigned int* bits, unsigned int N,
        unsigned int iter) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < N) {
        unsigned int key = input[i];
        unsigned int bit = (key >> iter) & 1;
        unsigned int numOnesBefore = bits[i];
        unsigned int numOnesTotal = bits[N];
        unsigned int dst = (bit == 0)
            ? (i - numOnesBefore)
            : (N - numOnesTotal + numOnesBefore);
        output[dst] = key;
    }
}

// bits has N + 1 entries.
void radix_sort_iter(unsigned int* input, unsigned int* output,
        unsigned int* bits, unsigned int N, unsigned int iter,
        unsigned int blockSize) {
    unsigned int numBlocks = (N + blockSize - 1) / blockSize;
    radix_bits_kernel<<<numBlocks, blockSize>>>(input, bits, N, iter);
    exclusive_scan_kernel<<<1, SCAN_THREADS>>>(bits, N);
    radix_scatter_kernel<<<numBlocks, blockSize>>>(input, output, bits,
                                                   N, iter);
}`;

const SCAN = `// Not the book's code: an exclusive scan of bits[0..N-1] followed by one zero, in one block,
// written in place. Each thread sums a contiguous chunk, the chunk sums are scanned in
// shared memory (Kogge-Stone, section 11.2), and each thread then rewrites its chunk as a
// running sum: chapter 11's coarsened scan (section 11.6) in the reduce-scan-scan order of
// section 11.9, with the chunks left in global memory. bits[N] receives the total.
__global__ void exclusive_scan_kernel(unsigned int* bits, unsigned int N) {
    __shared__ unsigned int sums[SCAN_THREADS];
    unsigned int t = threadIdx.x;
    unsigned int per = (N + blockDim.x - 1) / blockDim.x;
    unsigned int begin = (t * per < N) ? t * per : N;
    unsigned int end = (begin + per < N) ? begin + per : N;
    unsigned int mine = 0;
    for(unsigned int k = begin; k < end; ++k) mine += bits[k];
    sums[t] = mine;
    __syncthreads();
    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        unsigned int left = (t >= stride) ? sums[t - stride] : 0;
        __syncthreads();
        sums[t] += left;
        __syncthreads();
    }
    unsigned int run = sums[t] - mine;          // ones in the chunks before this one
    for(unsigned int k = begin; k < end; ++k) {
        unsigned int b = bits[k];
        bits[k] = run;
        run += b;
    }
    if(t == blockDim.x - 1) bits[N] = run;      // after the last chunk: the total
}`;

const SNIPPET_STATUS = 'front-end-checked; emulator-checked; unverified on CUDA and HIP hardware';
const TIERS: Record<string, Tier> = {
  input: 'global', output: 'global', bits: 'global',
  i: 'reg', key: 'reg', bit: 'reg', numOnesBefore: 'reg', numOnesTotal: 'reg', dst: 'reg',
  sums: 'shared', t: 'reg', per: 'reg', begin: 'reg', end: 'reg', mine: 'reg', stride: 'reg', left: 'reg', run: 'reg', b: 'reg', k: 'reg',
};

type Op = 'i' | 'load' | 'bit' | 'store-bit' | 'scan' | 'reload' | 'read' | 'dst' | 'store';
/** The stepper's steps and the lines of PASS each one highlights (pinned by S145.snippets.test.ts). */
export const STEPS: { op: Op; lines: number[]; label: string }[] = [
  { op: 'i', lines: [32, 3], label: 'kernel 1: i' },
  { op: 'load', lines: [5], label: 'load the key' },
  { op: 'bit', lines: [6], label: 'extract the bit' },
  { op: 'store-bit', lines: [7], label: 'store bits[i]' },
  { op: 'scan', lines: [33], label: 'scan' },
  { op: 'reload', lines: [34, 35, 14, 16, 17], label: 'kernel 2: i, key, bit' },
  { op: 'read', lines: [18, 19], label: 'read the scan' },
  { op: 'dst', lines: [20, 21, 22], label: 'destination' },
  { op: 'store', lines: [23], label: 'store the key' },
];
const PREDICT_THREADS = [2, 6, 11, 13];
type Col = 'i' | 'key' | 'bit' | 'before' | 'total' | 'dst';
const ROWS: { col: Col; label: string }[] = [
  { col: 'i', label: 'i' },
  { col: 'key', label: 'key' },
  { col: 'bit', label: 'bit' },
  { col: 'before', label: 'numOnesBefore' },
  { col: 'total', label: 'numOnesTotal' },
  { col: 'dst', label: 'dst' },
];
const LAB1 = 'ch14/lab1_twosorts/solution/';
const loaded = (t: string | null) => (t && !t.startsWith('// could not load') ? t : null);

// ---------------------------------------------------------------- the stepper
const LX = 92; // left label column
const CW = 38;
const KEY_Y = 30;
const BITS_Y = 112;
const OUT_Y = 222;

function narrate(pp: ParallelPass, s: number, f: number): string {
  const t = pp.threads[f];
  const n = pp.n;
  const who = `Thread ${t.thread} of block ${t.block}`;
  const kb = pp.keys.some((k) => k > 15) ? 8 : 4;
  const key = bin(t.key, kb);
  switch (s) {
    case 0:
      return `Nothing has run. ${n} keys, one thread each, in blocks of ${pp.blockDim}; this pass reads bit ${pp.bit}. Click a key or a column of the register table to follow another thread.`;
    case 1:
      return `The host launches kernel 1. ${who}: i = blockIdx.x·blockDim.x + threadIdx.x = ${t.block}·${pp.blockDim} + ${t.thread} = ${t.i}. (The page's line 03.)`;
    case 2:
      return `${who} loads its key: input[${t.i}] = ${key}. Every thread loads one key, so the whole input is read once.`;
    case 3:
      return `bit = (key >> ${pp.bit}) & 1 = ${t.bit}: shift the bit of interest to the right end, mask off the rest.`;
    case 4:
      return `${who} stores bits[${t.i}] = ${t.bit}. When every thread has done this, bits holds one bit per key. That is the end of kernel 1.`;
    case 5:
      if (pp.defect === 'scan-n' || pp.defect === 'last-as-total')
        return `The host launches the scan over the ${n} entries of a ${n}-entry bits array. Each entry now holds the ones before it, and no entry holds the total. (On the page this is line 10, gridExclusiveScan.)`;
      if (pp.defect === 'inclusive')
        return `The host launches an inclusive scan over ${n + 1} entries: entry k now counts the ones up to and including key k, not the ones before it.`;
      return `The host launches the scan, one launch over every block's bits: ${n} bits and a zero after them. Entry k now holds the ones before key k, and entry ${n} the total, ${pp.ones}. (On the page this is line 10, gridExclusiveScan.)`;
    case 6:
      return `The host launches kernel 2. ${who} computes i = ${t.i} again and reloads key ${key}, and recomputes its bit, ${t.bit}: the scan has overwritten bits[${t.i}].`;
    case 7:
      if (pp.defect === 'scan-n')
        return `numOnesBefore = bits[${t.i}] = ${t.onesBefore}. numOnesTotal = bits[${n}]: past the end of a ${n}-entry array. Out of range; what it returns is anyone's guess.`;
      if (pp.defect === 'last-as-total')
        return `numOnesBefore = bits[${t.i}] = ${t.onesBefore}. numOnesTotal = bits[${n - 1}] = ${t.total}, the last of the ${n} entries, as Fig. 14.6 suggests. The true total is ${pp.ones}.`;
      return `numOnesBefore = bits[${t.i}] = ${t.onesBefore}; numOnesTotal = bits[${n}] = ${t.total}.`;
    case 8:
      if (pp.defect === 'zero-formula') return `With the defect every key uses the zero formula: dst = i − numOnesBefore = ${t.formula}.`;
      return t.bit === 0
        ? `bit is 0, so dst = i − numOnesBefore = ${t.formula}: the zeros before it, since every key before it is a zero or a one.`
        : `bit is 1, so dst = N − numOnesTotal + numOnesBefore = ${t.formula}: all the zeros, then the ones before it.`;
    default: {
      const bad = pp.collisions.length + pp.holes.length + pp.oobWrites.length + pp.unknown.length;
      const own = t.dst == null ? `${who} cannot know where to write: its destination depends on the out-of-range read.` : t.oobWrite ? `${who} writes output[${t.dst}]: past the end of the output.` : `${who} writes output[${t.dst}] = ${key}.`;
      return bad
        ? `${own} Across the grid: ${pp.collisions.length} slots written twice, ${pp.holes.length} never written, ${pp.oobWrites.length} writes out of range.`
        : `${own} Every slot is written exactly once, and the output is the stable partition by bit ${pp.bit}.`;
    }
  }
}

function PassSvg({ pp, s, focus, setFocus, predict }: { pp: ParallelPass; s: number; focus: number; setFocus: (i: number) => void; predict: boolean }) {
  const n = pp.n;
  const kb = pp.keys.some((k) => k > 15) ? 8 : 4;
  const cx = (j: number) => LX + j * CW + (CW - 4) / 2;
  const totalX = LX + n * CW + 14;
  const w = totalX + CW + 10;
  const scanned = s >= 5;
  const wrote = s >= 9;
  const showDst = (i: number) => s >= 8 && !(predict && PREDICT_THREADS.includes(i));
  const cellCls = (slot: number) => (pp.writers[slot].length > 1 ? 'bad' : pp.writers[slot].length === 0 ? 'bad' : '');
  const sideCount = pp.scan.length > n ? n + 1 : n;
  return (
    <svg viewBox={`0 0 ${w} 278`} role="img" aria-label="One radix pass, thread by thread" style={{ width: '100%', maxWidth: w, minWidth: Math.min(w, 600), display: 'block' }}>
      <defs>
        <marker id="pp-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" style={{ fill: 'var(--ink-2)' }} />
        </marker>
        <marker id="pp-arrow-bad" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" style={{ fill: 'var(--fail)' }} />
        </marker>
      </defs>
      {Array.from({ length: Math.ceil(n / pp.blockDim) }, (_, b) => (
        <g key={b}>
          <line x1={LX + b * pp.blockDim * CW + 1} x2={LX + (b + 1) * pp.blockDim * CW - 5} y1={14} y2={14} stroke="var(--ink-2)" />
          <text x={LX + b * pp.blockDim * CW + (pp.blockDim * CW) / 2 - 2} y={10} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>block {b}</text>
        </g>
      ))}
      <text x={6} y={KEY_Y + 17} fontSize="11.5" fontWeight="600" style={{ fill: 'var(--global)' }}>input</text>
      {pp.keys.map((k, i) => (
        <g key={i} data-key-slot={i} onClick={() => setFocus(i)} style={{ cursor: 'pointer' }}>
          <rect x={LX + i * CW} y={KEY_Y} width={CW - 4} height={26} rx="2"
            fill={i === focus ? 'var(--global)' : s >= 2 ? 'var(--global-wash)' : 'var(--cell)'} stroke="var(--global)" />
          <text x={cx(i)} y={KEY_Y + 17} fontSize={kb === 8 ? 8.5 : 11} textAnchor="middle" style={{ fill: i === focus ? 'var(--page)' : 'var(--ink)' }}>{bin(k, kb)}</text>
          <text x={cx(i)} y={KEY_Y + 40} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{i}</text>
        </g>
      ))}
      <text x={6} y={BITS_Y + 17} fontSize="11.5" fontWeight="600" style={{ fill: 'var(--global)' }}>{scanned ? 'bits, scanned' : 'bits'}</text>
      {Array.from({ length: sideCount }, (_, k) => {
        const x = k < n ? LX + k * CW : totalX;
        const v = scanned ? pp.scan[k] : k < n ? (s >= 4 ? pp.bits[k] : null) : 0;
        const hidden = scanned && predict && k < n && PREDICT_THREADS.includes(k);
        const readTotal = s >= 7 && k === n && pp.defect !== 'scan-n' && pp.defect !== 'last-as-total';
        return (
          <g key={k} data-bits-slot={k}>
            <rect x={x} y={BITS_Y} width={CW - 4} height={26} rx="2" fill={readTotal ? 'var(--global-wash)' : 'var(--cell)'} stroke="var(--global)" strokeDasharray={k === n && !scanned ? '3 3' : undefined} />
            <text x={x + (CW - 4) / 2} y={BITS_Y + 17} fontSize="11.5" textAnchor="middle" data-v={hidden ? '' : v ?? ''}>{hidden ? '?' : v ?? ''}</text>
          </g>
        );
      })}
      {pp.scan.length === n ? (
        <g data-missing-total="">
          <rect x={totalX} y={BITS_Y} width={CW - 4} height={26} rx="2" fill={s >= 7 && pp.defect === 'scan-n' ? 'var(--fail-wash)' : 'none'} stroke="var(--fail)" strokeDasharray="3 3" />
          <text x={totalX + (CW - 4) / 2} y={BITS_Y + 17} fontSize="11" textAnchor="middle" style={{ fill: 'var(--fail)' }}>{s >= 7 && pp.defect === 'scan-n' ? '??' : ''}</text>
        </g>
      ) : null}
      <text x={totalX + (CW - 4) / 2} y={BITS_Y + 40} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>[{n}]</text>
      {s === 5 ? (
        <g data-wall="">
          <rect x={LX - 4} y={BITS_Y - 25} width={totalX + CW - LX} height={18} rx="2" fill="var(--ink)" />
          <text x={LX + 6} y={BITS_Y - 12} fontSize="11" style={{ fill: 'var(--page)' }}>exclusive scan: one launch over every block's bits; no thread goes on until it has finished</text>
        </g>
      ) : null}
      <text x={6} y={OUT_Y + 17} fontSize="11.5" fontWeight="600" style={{ fill: 'var(--global)' }}>output</text>
      {Array.from({ length: n }, (_, j) => {
        const v = wrote ? pp.out[j] : null;
        const cls = wrote ? cellCls(j) : '';
        return (
          <g key={j} data-out-slot={j} data-out-state={wrote ? (cls ? (pp.writers[j].length ? 'collision' : 'hole') : 'ok') : ''}>
            <rect x={LX + j * CW} y={OUT_Y} width={CW - 4} height={26} rx="2"
              fill={cls ? 'var(--fail-wash)' : 'var(--cell)'} stroke={cls ? 'var(--fail)' : 'var(--global)'} strokeDasharray={wrote && pp.writers[j].length === 0 ? '3 3' : undefined} />
            <text x={cx(j)} y={OUT_Y + 17} fontSize={kb === 8 ? 8.5 : 11} textAnchor="middle" style={{ fill: cls ? 'var(--fail)' : 'var(--ink)' }}>
              {wrote ? (pp.writers[j].length > 1 ? `×${pp.writers[j].length}` : v == null ? '' : bin(v, kb)) : ''}
            </text>
            <text x={cx(j)} y={OUT_Y + 40} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{j}</text>
          </g>
        );
      })}
      {s >= 8
        ? pp.threads.map((t) => {
            if (!showDst(t.i) || t.dst == null) return null;
            const out = t.oobWrite;
            const bad = out || (t.dst >= 0 && t.dst < n && pp.writers[t.dst].length > 1);
            const x2 = out ? totalX + 8 : cx(t.dst);
            return (
              <line key={t.i} data-arrow={t.i} data-to={t.dst} x1={cx(t.i)} y1={BITS_Y + 44} x2={x2} y2={OUT_Y - 3}
                stroke={bad ? 'var(--fail)' : t.i === focus ? 'var(--ink)' : 'var(--ink-2)'} strokeWidth={t.i === focus ? 2.2 : bad ? 1.5 : 1}
                strokeDasharray={wrote ? undefined : '4 3'} opacity={t.i === focus || bad ? 1 : 0.7}
                markerEnd={bad ? 'url(#pp-arrow-bad)' : 'url(#pp-arrow)'} />
            );
          })
        : null}
      {s >= 8 && pp.oobWrites.length ? (
        <text x={totalX + 8} y={OUT_Y + 17} fontSize="10.5" style={{ fill: 'var(--fail)' }}>out of range</text>
      ) : null}
    </svg>
  );
}

function PassStepper() {
  const [source, setSource] = useState<'book' | 'random'>('book');
  const [seed, setSeed] = useState(3);
  const [bit, setBit] = useState(0);
  const [defect, setDefect] = useState<Defect>('none');
  const [s, setS] = useState(0);
  const [focus, setFocus] = useState(6);
  const [predict, setPredict] = useState(false);
  const [guesses, setGuesses] = useState<Record<string, string>>({});
  const [checked, setChecked] = useState(false);

  const keys = useMemo(() => (source === 'book' ? BOOK_KEYS : randomKeys(16, 8, seed)), [source, seed]);
  const maxBit = source === 'book' ? 3 : 7;
  const b = Math.min(bit, maxBit);
  const pp = useMemo(() => parallelPass(keys, b, 4, defect), [keys, b, defect]);
  const truth = useMemo(() => parallelPass(keys, b, 4, 'none'), [keys, b]);
  useEffect(() => setS(0), [keys, b, defect]);
  useEffect(() => {
    setGuesses({});
    setChecked(false);
  }, [keys, b]);

  const hideFor = (i: number) => predict && !checked && PREDICT_THREADS.includes(i);
  const grade = (i: number, what: 'before' | 'dst') => {
    const g = guesses[`${i}:${what}`];
    if (g == null || g.trim() === '') return null;
    const want = what === 'before' ? truth.threads[i].onesBefore : truth.threads[i].dst;
    return Number(g) === want;
  };
  const allRight = PREDICT_THREADS.every((i) => grade(i, 'before') && grade(i, 'dst'));
  const step = STEPS[s - 1];
  const done = s === STEPS.length;
  const ft = pp.threads[focus];
  const badNow = s >= 7 && (ft.oobRead || (s >= 8 && (ft.dst == null || ft.oobWrite)));
  const kb = pp.keys.some((k) => k > 15) ? 8 : 4;
  const colStyle = (i: number) => (i === focus ? { background: 'var(--rule-soft)', fontWeight: 650 } : {});
  const guessBox = (i: number, what: 'before' | 'dst') => (
    <input type="number" aria-label={`thread ${i} ${what === 'before' ? 'ones before' : 'destination'}`}
      value={guesses[`${i}:${what}`] ?? ''} style={{ width: '2.9rem', padding: '0.05rem 0.2rem' }}
      onClick={(e) => e.stopPropagation()}
      onChange={(e) => setGuesses({ ...guesses, [`${i}:${what}`]: e.target.value })} />
  );
  const mark = (i: number, what: 'before' | 'dst') => {
    if (!(predict && checked && PREDICT_THREADS.includes(i))) return null;
    return grade(i, what) ? <span className="t-ok" data-grade="right"> ✓</span> : <span className="t-fail" data-grade="wrong"> ✗</span>;
  };
  const cell = (t: ParallelPass['threads'][number], col: Col) => {
    switch (col) {
      case 'i': return s >= 1 ? t.i : '';
      case 'key': return s >= 2 ? bin(t.key, kb) : '';
      case 'bit': return s >= 3 ? t.bit : '';
      case 'before':
        return hideFor(t.i) ? guessBox(t.i, 'before') : <>{s >= 7 ? t.onesBefore : ''}{mark(t.i, 'before')}</>;
      case 'total':
        return s >= 7 ? (t.total == null ? <span className="t-fail">??</span> : t.total) : '';
      default:
        return hideFor(t.i) ? guessBox(t.i, 'dst') : (
          <>
            {s >= 8 ? (t.dst == null ? <span className="t-fail">?</span> : t.oobWrite ? <span className="t-fail">{t.dst}</span> : t.dst) : ''}
            {mark(t.i, 'dst')}
          </>
        );
    }
  };

  return (
    <Bench
      title="One pass, thread by thread"
      task="Follow one thread through both kernels, predict four threads' numbers, then break the pass."
      foot={
        <>
          Amber is global memory, purple a thread's registers. Click a key or a column to follow that thread. Model:{' '}
          <code>src/sim/radixsort.ts</code>, <code>parallelPass</code> (its tests reproduce Fig. 14.6's rows).
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Keys
          <select value={source} onChange={(e) => setSource(e.target.value as 'book' | 'random')}>
            <option value="book">the book's sixteen (Fig. 14.5)</option>
            <option value="random">sixteen random 8-bit keys</option>
          </select>
        </label>
        {source === 'random' ? <button onClick={() => setSeed(seed + 1)}>Shuffle</button> : null}
        <label className="field">
          Bit
          <select value={b} onChange={(e) => setBit(Number(e.target.value))} aria-label="Bit">
            {Array.from({ length: maxBit + 1 }, (_, k) => <option key={k} value={k}>bit {k}</option>)}
          </select>
        </label>
        <label className="field">
          Kernel
          <select value={defect} onChange={(e) => setDefect(e.target.value as Defect)} aria-label="Kernel">
            {(Object.keys(DEFECT_LABEL) as Defect[]).map((d) => <option key={d} value={d}>{DEFECT_LABEL[d]}</option>)}
          </select>
        </label>
        <label className="check">
          <input type="checkbox" checked={predict} onChange={(e) => { setPredict(e.target.checked); setChecked(false); }} /> predict first
        </label>
      </div>

      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setS(0)} disabled={s === 0}>Restart</button>
        <button onClick={() => setS((x) => Math.max(0, x - 1))} disabled={s === 0}>Back</button>
        <button className="primary" onClick={() => setS((x) => Math.min(STEPS.length, x + 1))} disabled={done}>Step</button>
        <button onClick={() => setS(STEPS.length)} disabled={done}>Run to the end</button>
        <span className="muted" aria-live="polite">step {s} of {STEPS.length}{step ? `: ${step.label}` : ''}</span>
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 1fr)', gap: '1rem' }}>
        <Code tabs={[{ label: 's145_radix_iter.cu', code: PASS, tiers: TIERS }]} highlight={step ? step.lines : []} />
        <div className={'narrate' + (badNow ? ' bad' : '')} role="status" style={{ marginTop: '1.1rem' }}>
          {narrate(pp, s, focus)}
        </div>
      </div>

      <div className="mx-label t-global">Global memory</div>
      <PassSvg pp={pp} s={s} focus={focus} setFocus={setFocus} predict={hideFor(PREDICT_THREADS[0])} />

      <div className="mx-label t-reg" style={{ marginTop: '0.8rem' }}>Each thread's registers (click a column to follow that thread)</div>
      <div style={{ overflowX: 'auto' }}>
        <table className="threads" aria-label="Every thread's registers" data-part="registers" style={{ width: 'auto' }}>
          <thead>
            <tr>
              <th>block.thread</th>
              {pp.threads.map((t) => (
                <th key={t.i} data-thread={t.i} onClick={() => setFocus(t.i)} style={{ cursor: 'pointer', ...colStyle(t.i) }}>
                  {t.block}.{t.thread}{predict && PREDICT_THREADS.includes(t.i) ? '*' : ''}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {ROWS.map((row) => (
              <tr key={row.col}>
                <td>{row.label}</td>
                {pp.threads.map((t) => (
                  <td key={t.i} className="t-reg" data-thread={t.i} data-col={row.col} onClick={() => setFocus(t.i)} style={{ cursor: 'pointer', ...colStyle(t.i) }}>
                    {cell(t, row.col)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {predict ? (
        <div className="row" style={{ marginTop: '0.6rem' }} data-part="predict">
          <button className="primary" onClick={() => setChecked(true)}>Check my predictions</button>
          {checked ? (
            allRight ? <span className="chip ok">all eight numbers right</span> : <span className="chip fail">some numbers are off; the table now shows the model's</span>
          ) : (
            <span className="muted">Columns marked *: fill in ones before and destination for the corrected kernel, then check.</span>
          )}
        </div>
      ) : null}

      {done ? (
        <p style={{ margin: '0.8rem 0 0' }} data-part="outcome">
          {pp.correct ? (
            <span className="chip ok">every slot written once: the stable partition</span>
          ) : (
            <>
              <span className="chip fail">wrong output</span>{' '}
              {[
                pp.collisions.length ? `${pp.collisions.length} slot${pp.collisions.length > 1 ? 's' : ''} written twice (${pp.collisions.join(', ')})` : '',
                pp.holes.length ? `${pp.holes.length} never written (${pp.holes.join(', ')})` : '',
                pp.oobWrites.length ? `${pp.oobWrites.length} write${pp.oobWrites.length > 1 ? 's' : ''} out of range` : '',
                pp.oobReads.length ? `${pp.oobReads.length} reads of bits[${pp.n}] out of range` : '',
              ].filter(Boolean).join('; ')}.
            </>
          )}
          {defect === 'last-as-total' && pp.correct ? (
            <span className="muted"> Right here only because the last key's bit is 0. Try random keys until the last key's bit is 1.</span>
          ) : null}
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS145: Record<string, Question> = {
  'ch14/14.5/q-dst-one': {
    kind: 'numeric',
    id: 'ch14/14.5/q-dst-one',
    prompt: (
      <>
        A pass over n = 20 keys finds 9 ones in total. The key at index 13 has bit 1 and 6 ones before it. What is its
        destination?
      </>
    ),
    answer: 17,
    explain: (
      <p>
        All 20 − 9 = 11 zeros come first, then the 6 ones before it: 11 + 6 = 17, which is{' '}
        <code>N − numOnesTotal + numOnesBefore</code>.
      </p>
    ),
    source: 'ch. 14 §14.5, p. 337',
  },
  'ch14/14.5/q-dst-zero': {
    kind: 'numeric',
    id: 'ch14/14.5/q-dst-zero',
    prompt: <>The same pass; now the key at index 13 has bit 0, and 6 ones before it. What is its destination?</>,
    answer: 7,
    explain: (
      <p>
        Thirteen keys come before it and 6 of them are ones, so 7 are zeros: 13 − 6 = 7, which is{' '}
        <code>i − numOnesBefore</code>. The total does not enter into it.
      </p>
    ),
    source: 'ch. 14 §14.5, p. 337',
  },
  'ch14/14.5/q-total': {
    kind: 'choice',
    id: 'ch14/14.5/q-total',
    prompt: (
      <>
        Fig. 14.7 reads <code>bits[N]</code> for the total number of ones, and in Fig. 14.6 the last scan value shown,
        8, equals the total. Why is a scan of the N bits not enough?
      </>
    ),
    options: [
      { label: 'It is enough: the last entry of an exclusive scan is always the total.', why: 'It misses the last key\'s own bit. The book\'s keys end with 0000, which hides that.' },
      {
        label:
          'An exclusive scan of N entries ends with the ones before the last key, which is the total only when the last key\'s bit is 0. Scan N + 1 entries with a zero at the end, or add the last bit.',
        correct: true,
      },
      { label: 'Use an inclusive scan everywhere instead.', why: 'Then every one gets one more than its "ones before" and lands a slot too far: the stepper\'s inclusive variant shows it.' },
    ],
    explain: (
      <p>
        On the page, nothing says where <code>bits[N]</code> comes from; the figure's last value matches the total
        only because the last key is <code>0000</code>. The snippet allocates N + 1 entries and its scan writes the
        total into the last one. In the stepper, choose "total from the last of n entries" with random keys.
      </p>
    ),
    source: 'notes/ch14.md, "Unclear in the PDF" item 4; pp. 338–339',
  },
  'ch14/14.5/q-runs': {
    kind: 'numeric',
    id: 'ch14/14.5/q-runs',
    prompt: (
      <>
        One warp of 32 threads in a 1-bit pass holds z zeros and 32 − z ones, with 0 &lt; z &lt; 32. Into how many
        contiguous runs of output addresses does it write?
      </>
    ),
    answer: 2,
    unit: 'runs',
    explain: (
      <p>
        Each zero's destination is one more than the previous zero's, and the same holds for the ones, so the warp
        writes one run of z consecutive slots and one run of 32 − z, far apart. Compared with one run for a coalesced
        store, that is the cost section <a href={href('ch14', '14.6')}>14.6</a> sets out to remove; in NVIDIA's 32-byte
        sectors it is 4 to 7 sectors instead of 4.
      </p>
    ),
    source: 'our analysis of ch. 14 §14.6, p. 340 (notes/today/ch14-facts.md A10)',
  },
  'ch14/14.5/q-split': {
    kind: 'choice',
    id: 'ch14/14.5/q-split',
    prompt: <>Why is lab 14.1's pass three launches instead of Fig. 14.7's one kernel?</>,
    options: [
      {
        label:
          'The scan needs every block to have written its bits before any block reads the result, and an ordinary launch guarantees neither that all blocks are resident nor any order between them. Separate launches give that order on every backend.',
        correct: true,
      },
      { label: 'Kernels cannot call other functions.', why: 'They can call device functions. The problem is waiting for other blocks.' },
      { label: 'The scan needs more shared memory than one kernel may use.', why: 'The scan can live in global memory. What one kernel cannot do is wait for the whole grid.' },
    ],
    explain: (
      <p>
        On CUDA and HIP, launches on one stream run in order; on Metal, dispatches in one serial compute encoder run in
        order and see each other's writes. A cooperative launch with <code>grid.sync()</code> would also work on CUDA
        and HIP, at the price of a grid small enough to be resident all at once; Metal has no grid-wide barrier.
      </p>
    ),
    source: 'notes/ch14.md, "Unclear in the PDF" item 6; notes/today/ch04-facts.md A7, C3; ch02-facts.md A5, B3, C9',
  },
};

export function S145() {
  const q = questionsS145;
  const cu = loaded(useLabFile(LAB1 + 'kernel.cu'));
  const metal = loaded(useLabFile(LAB1 + 'kernel.metal'));
  const mm = loaded(useLabFile(LAB1 + 'launch.mm'));
  const lab1 = useSolutionStatus('ch14/lab1_twosorts');
  const tabs = [
    cu ? { label: 'CUDA (and HIP)', code: cu, tiers: TIERS, status: lab1.cu } : null,
    metal ? { label: 'Metal kernel', code: metal, tiers: TIERS, status: lab1.metal } : null,
    mm ? { label: 'Metal host', code: mm, status: lab1.metal } : null,
  ].filter((t) => t != null);
  return (
    <>
      <p>
        Each pass needs the whole result of the pass before it, so the passes run one after another. The parallelism is
        inside a pass. The book therefore writes a kernel for one pass and leaves the host to launch it once per bit.
      </p>
      <Source section="14.5" pages="336" />

      <h2>One thread per key</h2>
      <p>
        The straightforward scheme gives every key a thread, which works out where its key belongs in the output and
        writes it there. The book draws the sixteen keys of section 14.4 with four blocks of four threads; thread{' '}
        <code>i = blockIdx.x * blockDim.x + threadIdx.x</code> owns key <code>i</code>, as in every one-dimensional
        kernel since chapter 2. Real inputs are far longer and blocks far larger.
      </p>
      <Source section="14.5" pages="336–337" figs="Fig. 14.5" />

      <h2>Where each key goes</h2>
      <p>
        A key whose bit is 0 goes to the zero bucket at the front. Every zero before it also goes there, in order, so
        its destination is the number of zeros before it, which is its index minus the number of ones before it. A key
        whose bit is 1 goes after all the zeros, behind the ones before it.
      </p>
      <Source section="14.5" pages="337" />
      <table className="data" data-part="formulas">
        <thead>
          <tr><th>The key's bit</th><th>Destination</th></tr>
        </thead>
        <tbody>
          <tr><td>0</td><td><code>i − ones before</code></td></tr>
          <tr><td>1</td><td><code>n − ones in total + ones before</code></td></tr>
        </tbody>
      </table>
      <p>
        The sentence that introduces the second formula says "zero bucket"; it means the one bucket. Both formulas
        need one quantity a thread cannot know on its own: how many ones come before its key. That is an exclusive
        scan of the bits, and the total falls out of the same scan.
      </p>
      <Source section="14.5" pages="337–338" figs="Fig. 14.6" />
      <Ask q={q['ch14/14.5/q-dst-one']} label="Work it out" />
      <Ask q={q['ch14/14.5/q-dst-zero']} label="Work it out" />

      <h2>The kernel, step by step</h2>
      <p>
        Fig. 14.7 writes the pass as one kernel: compute <code>i</code>, load the key, extract the bit with{' '}
        <code>(key &gt;&gt; iter) &amp; 1</code>, store it in <code>bits</code>, call{' '}
        <code>gridExclusiveScan(bits, N)</code>, read "ones before" and the total back, compute the destination and
        store the key. Three things on the page need correcting before it runs.
      </p>
      <Source section="14.5" pages="338–339" figs="Fig. 14.7" />
      <ul>
        <li>
          <b><code>gridExclusiveScan</code> is pseudocode.</b> Every thread of the grid would have to wait in the middle
          of the kernel until every block had written its bits and the scan had finished. An ordinary launch promises
          neither that all blocks are resident at once nor any order between them, so no backend lets a kernel wait
          like that. The snippet below splits the pass at the scan into three launches, which is lab 14.1's design.
        </li>
        <li>
          <b>The total needs N + 1 entries.</b> The kernel reads <code>bits[N]</code>, which an exclusive scan of N
          bits never writes. The figure's last value equals the total only because the last key's bit is 0. Scan the N
          bits and a zero after them, so that entry N is the total.
        </li>
        <li>
          <b>A typo.</b> "indeed to right destination indices" on p. 339 should read "the right".
        </li>
      </ul>
      <p>
        One more change follows from the split: the second kernel recomputes each key's bit from the key, because the
        scan has overwritten <code>bits[i]</code> with the ones before it. The page keeps <code>bit</code> in a
        register across the scan; once the scan is a separate launch, registers do not survive it.
      </p>
      <Beyond title="Our corrections">
        <p>
          The split, the N + 1 entries and the recomputed bit are ours (notes/ch14.md, "Unclear in the PDF" items 4 to
          6). The kernels keep the page's names and order, so lines map across: the page's line 03 is our lines 3 and 14,
          its line 10 is our line 33.
        </p>
      </Beyond>
      <p>
        Step through the pass. Follow one thread through both kernels first; then tick "predict first" and fill in
        four threads' numbers before you run the scan; then pick a broken kernel and run it to the end.
      </p>
      <Simplification>
        <p>
          The grid-wide scan is one step that every thread sees complete at once. All threads advance together line by
          line, which a real launch does not promise (it does not matter here: no thread reads what another writes
          within a kernel). Blocks of four, as in the book; no timing.
        </p>
      </Simplification>
      <PassStepper />
      <Ask q={q['ch14/14.5/q-total']} />
      <Ask q={q['ch14/14.5/q-split']} />
      <p>
        Any correct exclusive scan will do between the two kernels. The snippet's is the shortest that is correct for
        any length: one block, each thread summing a chunk. For long inputs that single block is the bottleneck, and
        chapter 11's multi-kernel or single-pass scans are the real choice.
      </p>
      <Code
        tabs={[{ label: 'scan', code: SCAN, tiers: TIERS, status: SNIPPET_STATUS }]}
        caption={
          <>
            Both listings are from <code>labs/ch14/snippets/s145_radix_iter.cu</code>. Its unit test runs it on the
            course's CPU emulator under three thread orders: Fig. 14.6's rows on the book's keys, Fig. 14.4's four
            passes, every bit of random 8-bit keys against the model, and 1,000 random 32-bit keys sorted in 32 passes.
          </>
        }
      />

      <h2>Keys that carry token ids</h2>
      <p>
        A sampler sorts (probability, token id) pairs, not bare keys. The destination depends only on the key, so
        carrying a value costs one more load and one more store in the scatter: read <code>values[i]</code> beside the
        key and write it to <code>valuesOut[dst]</code>. Lab 14.3 does this for every row of a batch.
      </p>
      <Beyond>
        <p>
          The book defines key-value sorting (p. 330) but both of its kernels move bare keys; carrying the value is
          ours (notes/ch14.md, "Keys with payloads").
        </p>
      </Beyond>

      <h2>What the scatter does to memory</h2>
      <p>
        No two threads get the same destination: the two formulas number the zeros 0, 1, 2, … and the ones after them,
        so the scatter needs no atomics. But consecutive threads do not write consecutive slots. In the stepper, the
        four threads of block 0 write slots 0, 8, 1 and 9.
      </p>
      <Source section="14.6" pages="339–340" />
      <Ask q={q['ch14/14.5/q-runs']} />

      <h2>On each backend</h2>
      <p>
        On CUDA and HIP the book's single kernel could survive with a cooperative launch and <code>grid.sync()</code>,
        but every block must then be resident at once, which caps the size of the grid; the course's emulator allows it
        only up to 1,024 threads in all. Metal documents no grid-wide barrier at all. Separate launches work everywhere:
        launches on one CUDA or HIP stream run in the order they were issued, and dispatches in one serial Metal compute
        encoder run in order and see each other's writes.
      </p>
      <Today checked="2026-11-20 (notes/today/ch04-facts.md A7, C3; infra-facts.md A4; ch02-facts.md A5, B3, C9)">
        <p>
          Grid-wide sync needs <code>cudaLaunchCooperativeKernel</code> or <code>hipLaunchCooperativeKernel</code> and a
          grid that fits on the device at once. Metal's widest barrier is one threadgroup. The emulator's limit is in{' '}
          <code>labs/README.md</code>.
        </p>
      </Today>
      <p>
        The libraries go further. This per-pass structure, a scan of digit counts and then a scatter, is what CUB calls
        its multi-pass path; its default for 4-byte keys is a single-pass design that folds the scan into the scatter.
      </p>
      <Today checked="2026-11-20 (notes/today/ch14-facts.md, A1, B3, C4, D6)">
        <p>
          CUB's default from compute capability 6.0 for keys of 4 bytes or more is OneSweep, which uses chapter 11's
          decoupled look-back. rocPRIM implements OneSweep too. Metal documents no forward progress between
          threadgroups, so a look-back has no documented basis there; PyTorch's MPS radix sort uses separate count, scan
          and scatter dispatches.
        </p>
      </Today>
      {tabs.length ? (
        <Code
          tabs={tabs}
          caption={
            <>
              The reference solution of lab 14.1, loaded from <code>labs/{LAB1}</code>. Try the{' '}
              <a href={href('ch14', 'labs')}>lab</a> before you read it.
            </>
          }
        />
      ) : null}
    </>
  );
}
