import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source } from '../../components/notes';
import { benchRows, fmt, num, realBackends, useReports, type BenchRow } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  EXERCISE_1,
  FIG_11_3,
  FIG_11_5,
  KIND_LABEL,
  SCHEDULES,
  SCHEDULE_LABEL,
  barriersPerScan,
  iterations,
  kernelSource,
  resultBuffer,
  run,
  sharedBytes,
  swaps,
  type Config,
  type Schedule,
} from '../../sim/ksblock';
import { functionSource } from '../../sim/toolchain';
import { TraceView } from './KsGrid';

// ---------------------------------------------------------------- two buffers, one barrier
type VariantId = 'double' | 'no-copy' | 'single';
const VARIANTS: { id: VariantId; label: string; cfg: Omit<Config, 'threads'> }[] = [
  { id: 'double', label: 'two buffers, one barrier (Fig. 11.5)', cfg: FIG_11_5 },
  { id: 'no-copy', label: 'two buffers, copy branch left out', cfg: { ...FIG_11_5, copy: false } },
  { id: 'single', label: 'one buffer, barrier 1 only', cfg: { ...FIG_11_3, barrier2: false } },
];

function TwoBuffers() {
  const [variant, setVariant] = useState<VariantId>('double');
  const [schedule, setSchedule] = useState<Schedule>('in-order');
  const v = VARIANTS.find((x) => x.id === variant)!;
  const trace = useMemo(() => run({ ...v.cfg, threads: 8 }, EXERCISE_1, schedule), [v, schedule]);
  return (
    <Bench
      title="Two buffers, one barrier"
      task="Run each kernel to the end in each order. Which one is right in every order, and which is wrong in every order?"
      foot={
        <>
          Rows alternate between the two physical buffers, as in Fig. 11.4: buffer1_s holds the load, buffer2_s what
          iteration 0 wrote, and so on. Dashed arrows are copies. For one buffer, a row is buffer_s after that stride.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field" style={{ maxWidth: '100%' }}>
          Kernel
          <select value={variant} onChange={(e) => setVariant(e.target.value as VariantId)} aria-label="Kernel">
            {VARIANTS.map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
          </select>
        </label>
        <label className="field" style={{ maxWidth: '100%' }}>
          Thread order
          <select value={schedule} onChange={(e) => setSchedule(e.target.value as Schedule)} aria-label="Thread order">
            {SCHEDULES.map((s) => <option key={s} value={s}>{SCHEDULE_LABEL[s]}</option>)}
          </select>
        </label>
      </div>
      <TraceView trace={trace} />
    </Bench>
  );
}

function Outcomes() {
  return (
    <Bench title="Three kernels, every order" task="Which failures depend on the order, and which do not?"
      foot="Each cell runs the whole block of 8 threads in the model above on exercise 1's values.">
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ background: 'var(--page)', margin: 0 }} data-part="outcomes">
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}>Kernel</th>
              {SCHEDULES.map((s) => <th key={s}>{SCHEDULE_LABEL[s]}</th>)}
            </tr>
          </thead>
          <tbody>
            {VARIANTS.map((v) => (
              <tr key={v.id} data-variant={v.id}>
                <td style={{ paddingLeft: '0.6rem' }}>{v.label}</td>
                {SCHEDULES.map((s) => {
                  const r = run({ ...v.cfg, threads: 8 }, EXERCISE_1, s);
                  const kinds = [...new Set(r.violations.map((x) => x.kind))];
                  return (
                    <td key={s} data-outcome={r.correct ? 'correct' : 'wrong'} data-wrong={r.wrong.join(' ')}>
                      {r.correct ? <span className="chip ok">correct</span> : <span className="chip fail">{r.wrong.length} of 8 wrong</span>}
                      {kinds.map((k) => <div key={k} className="muted">{KIND_LABEL[k]}</div>)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- where is the answer
function SwapReplay({ committed }: { committed: number }) {
  const [B, setB] = useState(256);
  const sw = swaps(B);
  const res = resultBuffer(B);
  return (
    <div data-readout="buffer" data-result={res}>
      <p>
        A block of {B} threads runs {sw.length} iterations (strides 1 to {B / 2}). Each ends with a swap, so inBuffer_s
        names, in turn:{' '}
        <span className="mono">
          buffer1_s{sw.map((s) => ` → buffer${s.outBuf + 1}_s`).join('')}
        </span>
        . After the loop it names <b>buffer{res}_s</b>, which is where the answer is.{' '}
        {B === 256 ? (committed === res ? 'You had it.' : 'An even number of swaps brings it back to buffer1_s.') : ''}
      </p>
      <div className="row">
        <span className="muted">Try another block size:</span>
        <span className="seg" role="group" aria-label="Block size">
          {[8, 256, 512, 1024].map((b) => (
            <button key={b} aria-pressed={B === b} onClick={() => setB(b)}>{b}</button>
          ))}
        </span>
      </div>
      <p style={{ marginBottom: 0 }}>
        The kernel never needs to know: it always reads <code>inBuffer_s</code>, and the parity of log₂B decides which
        array that is.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- measured: two barriers against two buffers
// Lab 11.1 writes rows named "scan" with a "kernel" key (two_barrier, double_buffer, ...) and a "block" size.
const SIZE_KEYS = ['block', 'threads', 'seg', 'seg_size', 'block_size', 'B'];
const pickRow = (rows: BenchRow[], name: string) => {
  const named = rows.filter((r) => r.kernel === name || r.name === name);
  return named.find((r) => SIZE_KEYS.some((k) => r[k] === 1024)) ?? named[0];
};
/** How many times faster b ran than a: from times if both have one, else from a throughput. */
function speedup(a: BenchRow, b: BenchRow): number | undefined {
  for (const k of ['seconds', 'ms', 'us', 'time_ms', 'time_us']) {
    const x = num(a[k]);
    const y = num(b[k]);
    if (x && y) return x / y;
  }
  for (const k of Object.keys(a).filter((k) => /per_s$|gflops|gb_s/.test(k))) {
    const x = num(a[k]);
    const y = num(b[k]);
    if (x && y) return y / x;
  }
  return undefined;
}
function MeasuredDouble({ predicted }: { predicted: number }) {
  const { reports } = useReports();
  const found = realBackends(reports).flatMap((be) => {
    const labs = [...new Set(reports[be].results.map((r) => r.lab).filter((l) => l.startsWith('ch11/lab1')))];
    for (const lab of labs) {
      const rows = benchRows(reports[be], lab);
      const two = pickRow(rows, 'two_barrier');
      const dbl = pickRow(rows, 'double_buffer');
      const s = two && dbl ? speedup(two, dbl) : undefined;
      if (s) return [{ be, device: String(reports[be].device.name ?? be), s }];
    }
    return [];
  });
  if (!found.length)
    return (
      <p data-readout="measured" data-found="0">
        No measurement of lab 11.1 on this machine yet. Run it with <code>--bench</code> from the{' '}
        <a href={href('ch11', 'labs')}>labs page</a>, record it with <code>./verify.sh</code>, and come back: this box will
        compare your prediction with what your device did.
      </p>
    );
  return (
    <>
      {found.map((f) => (
        <p key={f.be} data-readout="measured" data-found="1">
          On <b>{f.device}</b> ({f.be}) the double-buffered kernel ran <b>{fmt(f.s)}×</b> as fast as the two-barrier
          kernel.{' '}
          {predicted > f.s * 1.5 ? 'You expected more.' : predicted < f.s / 1.5 ? 'You expected less.' : 'That is close to your prediction.'}
        </p>
      ))}
      <p className="muted">A measurement from your machine, not an expected value; other devices can differ either way.</p>
    </>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS113: Record<string, Question> = {
  'ch11/11.3/q-which-barrier': {
    kind: 'choice',
    id: 'ch11/11.3/q-which-barrier',
    prompt: <>Which barrier of Fig. 11.3 does double buffering remove?</>,
    options: [
      { label: 'Line 14, which guards only a write-after-read.', correct: true },
      {
        label: 'Line 10, at the top of the loop: it only keeps writers from overwriting values still being read.',
        why: "That describes line 14. Line 10 makes readers wait for the previous iteration's results, a true dependence, and two buffers do not change that.",
      },
      {
        label: 'Both: with two buffers no thread reads a slot that another thread writes.',
        why: 'Readers still need the values other threads wrote in the previous iteration, and one of the two barriers guards exactly that.',
      },
      { label: 'Neither; it only saves shared-memory traffic, not barriers.', why: 'It saves a barrier per iteration, and costs shared memory.' },
    ],
    explain: (
      <p>
        Line 14 enforces a write-after-read, or false, dependence: the store must wait for readers of the old value
        only because old and new share a slot. With separate input and output buffers they do not. Line 10 enforces a
        true (read-after-write) dependence: a reader needs the value another thread wrote in the previous iteration.
        Fig. 11.5 keeps that one, as its line 09.
      </p>
    ),
    source: 'ch. 11 §11.3, pp. 258–261',
  },
  'ch11/11.3/q-copy': {
    kind: 'choice',
    id: 'ch11/11.3/q-copy',
    prompt: <>In Fig. 11.5, the threads that are not adding skip the copy into outBuffer_s. What goes wrong?</>,
    options: [
      { label: 'Their output slots keep stale values, which later iterations read as final.', correct: true },
      {
        label: 'Nothing: those positions are final in the input buffer, so the result is unchanged.',
        why: 'They are final in the input buffer. The next iteration reads the other buffer.',
      },
      {
        label: 'A race: without the copy, readers of the next iteration are not ordered after the writers.',
        why: 'The result is wrong the same way in every order: no timing is involved.',
      },
      { label: 'Only the last output is wrong.', why: 'The positions that go idle first are the ones hit.' },
    ],
    explain: (
      <p>
        With one buffer, a finished value simply stays where it is. With two, each iteration writes a fresh buffer, and
        whatever is not written there is whatever that buffer held two iterations ago. Thread 0 in iteration 0 must copy
        x0 across, or slot 0 of buffer2_s is never written at all (p. 260). The stepper shows it: the result is wrong in
        every thread order, identically.
      </p>
    ),
    source: 'ch. 11 §11.3, pp. 259–260',
  },
  'ch11/11.3/q-buffer': {
    kind: 'choice',
    id: 'ch11/11.3/q-buffer',
    prompt: <>For a 512-thread block, which array holds the result of Fig. 11.5 after the loop?</>,
    options: [
      { label: 'buffer1_s', why: 'Count the swaps: that would need an even number of iterations.' },
      { label: 'buffer2_s', correct: true },
      { label: 'It depends on the thread order.', why: 'Every thread swaps its pointers identically, once per iteration.' },
      { label: 'Both hold it.', why: 'Only the last output buffer has the final values.' },
    ],
    explain: (
      <p>
        The loop runs 9 iterations, strides 1 to 256, each ending in a swap. After an odd number of swaps inBuffer_s
        points at buffer2_s. The kernel reads inBuffer_s, so it never has to know.
      </p>
    ),
    source: 'ch. 11 §11.3, p. 260 (the count is ours)',
  },
  'ch11/11.3/q-bytes': {
    kind: 'numeric',
    id: 'ch11/11.3/q-bytes',
    prompt: <>How many bytes of shared memory does one block of Fig. 11.5 use with SEG_SIZE = 1024 floats?</>,
    answer: sharedBytes(1024, 'double'),
    unit: 'bytes',
    explain: <p>Two arrays of 1024 four-byte floats: 8192 bytes, twice Fig. 11.3's 4096.</p>,
    source: 'ch. 11 Figs. 11.3 and 11.5, pp. 256, 260',
  },
  'ch11/11.3/q-fig119': {
    kind: 'choice',
    id: 'ch11/11.3/q-fig119',
    prompt: (
      <>
        Fig. 11.9, in the next section, has two barriers: one after the warps write their sums to shared memory, one after
        warp 0 has scanned those sums. Could double buffering remove either?
      </>
    ),
    options: [
      { label: 'No: both guard true dependences.', correct: true },
      {
        label: 'Yes, the first: it only protects the warp sums from being overwritten.',
        why: 'It makes warp 0 wait for the sums the other warps wrote. That is a read after a write.',
      },
      {
        label: 'Yes, the second: warp 0 writes back into the slots it has just read.',
        why: 'Each lane rewrites the slot it read itself, which needs no barrier. The barrier makes the other warps wait for those writes: a read after a write again.',
      },
      { label: 'Yes, both: two buffers remove every barrier between the stages.', why: 'Double buffering removes only write-after-read conflicts.' },
    ],
    explain: (
      <p>
        Each barrier there separates a write from the reads that need it: warp sums written before warp 0 reads them,
        scanned sums written before the other warps read them. Neither protects an old value from being overwritten, so
        a second buffer has nothing to fix.
      </p>
    ),
    source: 'ch. 11 §11.4, p. 264 (Fig. 11.9); §11.3, pp. 258–259',
  },
  'ch11/11.3/q-final-store': {
    kind: 'choice',
    id: 'ch11/11.3/q-final-store',
    prompt: <>Why does Fig. 11.5 need no barrier between the end of the loop and the final store?</>,
    options: [
      { label: 'Each thread stores the slot it wrote itself.', correct: true },
      {
        label: 'Global stores are ordered after shared-memory writes automatically.',
        why: 'The question is about the shared-memory read before the store.',
      },
      {
        label: 'The loop ends with a barrier that every thread passes before the store.',
        why: 'It starts with one; after the last write and swap there is none.',
      },
      { label: 'It does need one; the book forgot it, and the race rarely shows.', why: 'Check whose write the store reads.' },
    ],
    explain: (
      <p>
        A barrier is needed when a thread reads what another thread wrote. Here thread <i>t</i> reads slot <i>t</i>, its
        own write from the last iteration (an addition or a copy), so program order already guarantees it. That is our
        observation; the book does not say it.
      </p>
    ),
    source: 'ch. 11 Fig. 11.5, p. 260; notes/ch11.md, Fig. 11.5 "Care"',
  },
};

// ---------------------------------------------------------------- the section
export function S113() {
  const file = useLabFile('ch11/snippets/s112_kogge_stone.cu');
  const fn = file ? functionSource(file, 'double_buffer_scan_kernel') : null;
  const lines = kernelSource(FIG_11_5).line;
  const q = questionsS113;
  return (
    <>
      <p>
        Every iteration of Fig. 11.3 calls <code>__syncthreads()</code> twice, for one addition per thread. Barriers are
        one of the main costs of that kernel, so the next three sections remove them, one way after another.
      </p>
      <Source section="11.3" pages="258" />
      <p>
        The two barriers do not enforce the same thing. The first (line 10) makes readers wait until the previous
        iteration's values have been written: a <b>true</b>, or read-after-write, dependence. The second (line 14) makes
        writers wait until everyone has read the old values: a <b>false</b>, or write-after-read, dependence. It exists
        only because the old and the new value of a position share one slot. You met the same pair in the tiled matrix
        multiplication (<a href={href('ch05', '5.4')}>section 5.4</a>) and its cure in{' '}
        <a href={href('ch06', '6.7')}>section 6.7</a>.
      </p>
      <Source section="11.3" pages="258–259" />
      <Ask q={q['ch11/11.3/q-which-barrier']} />

      <h2>Read from one buffer, write the other</h2>
      <p>
        <b>Double buffering</b> puts each iteration's outputs somewhere other than its inputs. A fresh buffer per
        iteration would do it but waste shared memory. Once an iteration's outputs exist, its inputs are dead, so two
        buffers are enough: iteration 0 reads buffer1 and writes buffer2, iteration 1 reads buffer2 and writes buffer1,
        and so on. A writer can no longer spoil a value someone still reads, so the barrier against that goes.
      </p>
      <Source section="11.3" pages="259" figs="Fig. 11.4" />
      <p>
        Take iteration 0. Thread 2 reads x1 from slot 1 of buffer1 while thread 1 writes Σx0..x1 into slot 1 of buffer2.
        With one buffer, thread 1 would have to wait until thread 2 had read; with two, the order between them no longer
        matters.
      </p>
      <Aside>
        <p>
          A slip on p. 259: it gives thread 1's iteration-0 result as Σx1..x2 and has thread 1 read x1. Thread 1 writes
          Σx0..x1; thread 2 reads x1 and writes Σx1..x2.
        </p>
      </Aside>
      <p>
        One consequence is easy to miss. With one buffer, a finished value stays where it is. With two, the next
        iteration reads the other buffer, so a thread that would have gone idle, thread 0 in iteration 0 for one, must now
        copy its value across.
      </p>
      <Source section="11.3" pages="259–260" />
      <p>
        The stepper runs Fig. 11.5 thread by thread on exercise 1's values. Switch the copy off and look at what is left
        in the slots nobody wrote; then try one buffer with only barrier 1.
      </p>
      <Simplification>
        <p>
          As in section 11.2: one source-level operation per step, no warps, no timing, and shared memory nobody wrote
          reads as NaN. Each thread swaps its own pointers, as in the kernel.
        </p>
      </Simplification>
      <TwoBuffers />
      <Outcomes />
      <Ask q={q['ch11/11.3/q-copy']} />

      <h2>The kernel</h2>
      <p>
        Fig. 11.5 keeps one barrier, at the top of the loop (line 09). Active threads read <code>inBuffer_s</code> and
        write the sum to <code>outBuffer_s</code> (lines 11 to 12); the others copy their value across (line 14). Then
        every thread swaps its two pointers (lines 16 to 18). After the last swap the freshest values are in{' '}
        <code>inBuffer_s</code>, which is why the final store reads it.
      </p>
      <Source section="11.3" pages="260–261" figs="Fig. 11.5" />
      {fn ? (
        <Code
          tabs={[{ label: 'CUDA', code: fn.text, status: 'emulator-checked and front-end-checked; unverified on CUDA and HIP hardware' }]}
          highlight={[...(lines.barrier1 ?? []), ...(lines.copy ?? [])]}
          caption={
            <>
              Fig. 11.5 from labs/ch11/snippets/s112_kogge_stone.cu, numbered as on the page; the barrier and the copy are
              marked. The book calls it <code>scan_kernel</code>, like Fig. 11.3.
            </>
          }
        />
      ) : null}
      <p>
        No barrier is needed between the loop and the store either: each thread stores the slot it wrote itself in the
        last iteration. For a block of 1024 threads the kernel now executes {barriersPerScan(1024, 'double')} barriers
        instead of {barriersPerScan(1024, 'single')}.
      </p>
      <Aside>
        <p>The remark about the final store is ours (notes, Fig. 11.5); the book does not discuss it.</p>
      </Aside>
      <Ask q={q['ch11/11.3/q-final-store']} />

      <h2>Where is the answer?</h2>
      <p>
        The pointers swap at the end of every iteration, so which of the two arrays ends up holding the scan depends on
        how many iterations ran. Work it out before the replay does.
      </p>
      <Predict
        id="ch11/11.3/p-buffer"
        prompt={<>A block of 256 threads runs Fig. 11.5. After the loop, which array holds the result: 1 for buffer1_s or 2 for buffer2_s?</>}
      >
        {(v) => <SwapReplay committed={v} />}
      </Predict>
      <Ask q={q['ch11/11.3/q-buffer']} />

      <h2>What it costs</h2>
      <p>
        Two buffers need twice the shared memory: {sharedBytes(1024, 'double')} bytes per block of 1024 floats instead of{' '}
        {sharedBytes(1024, 'single')}. Shared memory is one of the budgets that cap how many blocks an SM keeps resident
        (<a href={href('ch05', '5.6')}>section 5.6</a>), so the saved barrier can be paid for with occupancy. Whether it is
        a gain on your device is a measurement, not a rule: lab 11.1 times both kernels at {iterations(1024)} iterations
        per block.
      </p>
      <Beyond>
        <p>The occupancy price is ours; the book only says the second buffer costs shared memory.</p>
      </Beyond>
      <Ask q={q['ch11/11.3/q-bytes']} label="Work it out" />
      <Predict
        id="ch11/11.3/p-double"
        prompt={
          <>
            At 1024 threads per block, how many times faster than the two-barrier kernel will the double-buffered kernel
            run on <em>your</em> device? Commit a number before running lab 11.1.
          </>
        }
        unit="× as fast"
      >
        {(v) => <MeasuredDouble predicted={v} />}
      </Predict>

      <h2>What is left</h2>
      <p>
        Double buffering still pays one barrier per iteration, and three shared-memory accesses (two reads, one write)
        for every single addition. The loop is dominated by synchronisation and shared memory. Section{' '}
        <a href={href('ch11', '11.4')}>11.4</a> moves most of the scan into registers and warp shuffles, and gets the
        whole block scan down to two barriers.
      </p>
      <Source section="11.4" pages="261" />
      <Ask q={q['ch11/11.3/q-fig119']} />
    </>
  );
}
