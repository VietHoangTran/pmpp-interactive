import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { benchRows, fmt, realBackends, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  C_CHOICES,
  LAB2,
  PROBLEMS,
  VOCAB,
  fastest,
  measuredLocal,
  sweepC,
  tableEntries,
  tileBytes,
  type Problem,
  type Row,
} from '../../sim/radixcost';
import { BOOK_KEYS, bin, localPass, summarise, warpStores } from '../../sim/radixlocal';

const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const dec = (v: number) => (Math.round(v * 10) / 10).toFixed(1);
const big = (v: number) => v.toLocaleString('en-US');
const kib = (b: number) => (b % 1024 === 0 ? `${big(b / 1024)} KiB` : `${big(b)} B`);

function Seg<T extends string | number>({ label, value, options, onPick, show }: {
  label: string; value: T; options: readonly T[]; onPick: (v: T) => void; show?: (v: T) => string;
}) {
  return (
    <span style={FIELD}>
      {label}
      <span className="seg" role="group" aria-label={label}>
        {options.map((o) => (
          <button key={String(o)} aria-pressed={value === o} onClick={() => onPick(o)}>
            {show ? show(o) : String(o)}
          </button>
        ))}
      </span>
    </span>
  );
}

// ---------------------------------------------------------------- Fig. 14.13 on the book's keys
const CW = 40;
const X0 = 20;
const cx = (i: number) => X0 + i * CW + CW / 2;

function Fig1413() {
  const [c, setC] = useState(2);
  const p = localPass(BOOK_KEYS, 0, 2, 4, c);
  const stores = warpStores(p, 32, 4, 32);
  const s = summarise(stores);
  const per = 4 * c;
  const local = p.blocks.flatMap((b) => b.sorted);
  const rowY = { in: 30, local: 112, out: 214 };
  return (
    <Bench
      title="Fig. 14.13, two ways"
      task="Switch between one and two keys per thread and count the arrows that travel in parallel."
      foot="Model: localPass() in src/sim/radixlocal.ts, a 2-bit digit, the first pass. Which thread copies which slot when it owns two keys is our choice (the book's figure does not say): in store instruction k, thread t copies local slot 4k + t."
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <Seg label="Keys per thread" value={c} options={[1, 2]} show={(v) => (v === 1 ? '1 (Fig. 14.11)' : '2 (Fig. 14.13)')} onPick={setC} />
      </div>
      <div style={{ overflowX: 'auto' }}>
        <svg viewBox="0 0 680 262" style={{ width: '100%', minWidth: 560, maxWidth: 760, display: 'block' }} role="img"
          aria-label={`Sixteen keys sorted locally by ${p.blocks.length} blocks and written out`} data-fig={c} data-runs={stores.map((w) => w.runs).join(' ')}>
          <defs>
            <marker id="s148-head" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
              <path d="M0,0 L8,4 L0,8 z" fill="var(--ink)" />
            </marker>
          </defs>
          <text x={X0} y={rowY.in - 7} fontSize="11" fontWeight="600" style={{ fill: 'var(--global)' }}>input (global memory), low two bits underlined</text>
          {BOOK_KEYS.map((k, i) => (
            <g key={i}>
              <rect x={X0 + i * CW + 1} y={rowY.in} width={CW - 2} height={24} rx="2" fill="var(--global-wash)" stroke="var(--global)" />
              <text x={cx(i)} y={rowY.in + 16} fontSize="11" textAnchor="middle">{bin(k, 4).slice(0, 2)}<tspan fontWeight="700" textDecoration="underline">{bin(k, 4).slice(2)}</tspan></text>
            </g>
          ))}
          {p.blocks.map((b) => (
            <g key={b.block}>
              <rect x={X0 + b.first * CW - 2} y={rowY.local - 30} width={per * CW + 4} height={64} rx="5" fill="none" stroke="var(--ink-3)" strokeDasharray="4 3" />
              <text x={X0 + b.first * CW + 2} y={rowY.local - 34} fontSize="10" style={{ fill: 'var(--ink-2)' }}>block {b.block}: {per} keys, 4 threads</text>
              {b.sorted.map((_, j) => (
                <text key={j} x={cx(b.first + j)} y={rowY.local - 8} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--reg)' }}>t{j % 4}</text>
              ))}
              {b.counts.map((n, d) => {
                if (!n) return null;
                const x1 = X0 + (b.first + b.offsets[d]) * CW + 4;
                const x2 = x1 + n * CW - 8;
                return <path key={d} d={`M${x1},${rowY.local + 28} v4 H${x2} v-4`} fill="none" stroke="var(--shared)" strokeWidth="1.3" />;
              })}
            </g>
          ))}
          {local.map((k, i) => (
            <g key={i} data-local={i}>
              <rect x={X0 + i * CW + 1} y={rowY.local} width={CW - 2} height={24} rx="2" fill="var(--shared-wash)" stroke="var(--shared)" />
              <text x={cx(i)} y={rowY.local + 16} fontSize="11" textAnchor="middle">{bin(k, 4)}</text>
            </g>
          ))}
          {p.writes.map((w) => (
            <line key={`${w.block}-${w.slot}`} x1={cx(p.blocks[w.block].first + w.slot)} y1={rowY.local + 36} x2={cx(w.addr)} y2={rowY.out - 3}
              stroke="var(--ink)" strokeWidth="1.3" markerEnd="url(#s148-head)" />
          ))}
          {p.out.map((k, i) => (
            <g key={i} data-out={i}>
              <rect x={X0 + i * CW + 1} y={rowY.out} width={CW - 2} height={24} rx="2" fill="var(--global-wash)" stroke="var(--global)" />
              <text x={cx(i)} y={rowY.out + 16} fontSize="11" textAnchor="middle">{bin(k, 4)}</text>
            </g>
          ))}
          <text x={X0} y={rowY.out + 40} fontSize="11" fontWeight="600" style={{ fill: 'var(--global)' }}>output after the pass (global memory)</text>
        </svg>
      </div>
      <table className="data" style={{ margin: '0.7rem 0 0', maxWidth: '36rem' }} data-fig-summary={c}>
        <tbody>
          <tr><td>blocks</td><td className="num" data-sum="blocks">{p.blocks.length}</td></tr>
          <tr><td>table entries (4 digit values × blocks)</td><td className="num" data-sum="table">{p.table.length}</td></tr>
          <tr><td>keys per local bucket</td><td className="num" data-sum="bucket">{fmt(per / 4)}</td></tr>
          <tr><td>store instructions, 4 threads each</td><td className="num" data-sum="stores">{s.stores}</td></tr>
          <tr><td>runs of consecutive addresses, all stores</td><td className="num" data-sum="runs"><b>{stores.reduce((a, w) => a + w.runs, 0)}</b></td></tr>
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- the explorer over c
const THREADS = [128, 256, 512, 1024];

interface Limit { key: string; label: string; bytes: number }
function useLimits(): Limit[] {
  const { reports } = useReports();
  const limits = realBackends(reports).flatMap((b) => {
    const d = reports[b].device;
    const field = 'maxThreadgroupMemoryLength' in d ? 'maxThreadgroupMemoryLength' : 'sharedMemPerBlock';
    const v = Number(d[field]);
    return isFinite(v) && v > 0 ? [{ key: b, label: `${String(d.name ?? b)} (${b}): ${field}`, bytes: v }] : [];
  });
  return limits;
}
function useMeasuredC(r: number) {
  const { reports } = useReports();
  return realBackends(reports)
    .map((b) => {
      const rows = benchRows(reports[b], LAB2);
      return { b, device: String(reports[b].device.name ?? b), byC: Object.fromEntries(C_CHOICES.map((c) => [c, measuredLocal(rows, r, c)])) as Record<number, number | undefined> };
    })
    .filter((d) => Object.values(d.byC).some((v) => v != null));
}

function KeysPerThread() {
  const [pid, setPid] = useState<Problem['id']>('n24');
  const [r, setR] = useState(8);
  const [threads, setThreads] = useState(256);
  const limits = useLimits();
  const [limitKey, setLimitKey] = useState<string>('');
  const [typed, setTyped] = useState('');
  const problem = PROBLEMS.find((p) => p.id === pid)!;
  const rows: Row[] = useMemo(() => sweepC(problem, 32, r, threads), [problem, r, threads]);
  const measured = useMeasuredC(r);
  const key = limitKey || limits[0]?.key || 'typed';
  const chosen = limits.find((l) => l.key === key);
  const typedBytes = Number(typed);
  const limit = key === 'typed' ? (typed.trim() !== '' && isFinite(typedBytes) && typedBytes > 0 ? typedBytes : null) : chosen?.bytes ?? null;
  const maxSectors = Math.max(...rows.map((x) => x.local.meanUnits));

  return (
    <Bench
      title="Keys per thread"
      task="Raise c and watch the runs shorten and the table shrink, until the tile no longer fits in shared memory."
      foot={
        <>
          Model: sweepC() in src/sim/radixcost.ts on src/sim/radixlocal.ts. 32-bit keys with 4-byte values; shared memory per
          block is the tile of c × threads pairs (8 bytes each) plus 2<sup>r</sup> four-byte counts, nothing else. Runs and
          sectors are averaged over every pass of a seeded sample of a few blocks, 32-lane warps, NVIDIA's 32-byte sectors.
          Registers and occupancy are not modelled.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <Seg label="Problem" value={pid} options={PROBLEMS.map((p) => p.id)} show={(id) => PROBLEMS.find((p) => p.id === id)!.label} onPick={setPid} />
        <Seg label="Digit width r" value={r} options={[1, 2, 4, 6, 8]} onPick={setR} />
        <Seg label="Threads per block" value={threads} options={THREADS} onPick={setThreads} />
      </div>
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <label className="field" style={{ maxWidth: '100%', minWidth: 0 }}>
          Shared memory a block may use
          <select value={key} aria-label="Shared memory limit" onChange={(e) => setLimitKey(e.target.value)} style={{ maxWidth: '100%' }}>
            {limits.map((l) => <option key={l.key} value={l.key}>{l.label} = {big(l.bytes)} bytes</option>)}
            <option value="typed">a number I type</option>
          </select>
        </label>
        {key === 'typed' ? (
          <label className="field">
            Bytes
            <input type="number" min={0} value={typed} aria-label="Your limit in bytes" placeholder="from doctor" onChange={(e) => setTyped(e.target.value)} />
          </label>
        ) : null}
      </div>
      {limit == null ? (
        <p className="ui" style={{ margin: '0 0 0.8rem' }} data-limit="none">
          {limits.length ? 'Type a limit in bytes to check each row against it.' : 'No device report on this machine.'} The limit is
          what <code>python3 labs/pmpp.py doctor</code> reports: <code>sharedMemPerBlock</code> on CUDA and HIP,{' '}
          <code>maxThreadgroupMemoryLength</code> on Metal.
        </p>
      ) : null}
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0, minWidth: '50rem', background: 'var(--page)' }} aria-label="Costs by keys per thread">
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.5rem' }}>c</th>
              <th className="num">keys per block</th>
              <th className="num">blocks per pass</th>
              <th className="num">table entries</th>
              <th className="num">keys per local bucket</th>
              <th className="num">runs per store</th>
              <th>sectors per store</th>
              <th className="num">shared memory per block</th>
              {measured.map((d) => <th key={d.b} className="num">{d.b}, ms</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((x) => {
              const over = limit != null && x.sharedBytes > limit;
              return (
                <tr key={x.c} data-c={x.c} data-over={over ? 1 : 0} style={over ? { background: 'var(--fail-wash)' } : undefined}>
                  <td style={{ paddingLeft: '0.5rem' }}><b>{x.c}</b></td>
                  <td className="num">{big(x.c * threads)}</td>
                  <td className="num" data-col="blocks">{big(x.blocks)}</td>
                  <td className="num" data-col="table">{big(x.tableEntries)}</td>
                  <td className="num" data-col="bucket">{fmt(x.keysPerBucket)}</td>
                  <td className="num" data-col="runs">{dec(x.local.meanRuns)}</td>
                  <td data-col="sectors" className="nowrap">
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, whiteSpace: 'nowrap' }}>
                      <i style={{ display: 'inline-block', height: 10, width: `${(x.local.meanUnits / maxSectors) * 6}rem`, background: 'var(--global-wash)', border: '1px solid var(--global)' }} />
                      {dec(x.local.meanUnits)}
                      <span className="muted">(min {dec(x.local.meanMin)})</span>
                    </span>
                  </td>
                  <td className="num nowrap" data-col="smem">
                    {over ? <span className="t-fail">{kib(x.sharedBytes)}, over by {kib(x.sharedBytes - limit!)}</span> : kib(x.sharedBytes)}
                  </td>
                  {measured.map((d) => <td key={d.b} className="num" data-measured={`${d.b}-${x.c}`}>{d.byC[x.c] != null ? fmt(d.byC[x.c]! * 1000) : <span className="muted">no row</span>}</td>)}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="ui" style={{ margin: '0.7rem 0 0' }} data-limit-line={limit ?? ''}>
        {limit != null ? (
          <>
            Limit in use: {big(limit)} bytes{chosen && key !== 'typed' ? `, as ${chosen.label} reported it` : ', as you typed it'}. Rows in red do
            not fit: the block could not be launched with that tile.
          </>
        ) : null}{' '}
        {measured.length
          ? 'The millisecond columns are lab 14.2 measured on this machine at its own problem size; the rest is the model.'
          : 'No lab 14.2 measurement on this machine yet; run python3 labs/pmpp.py run 14.2 --solution --bench and reload for a measured column.'}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
const N22 = { rows: 1, rowLength: 2 ** 22 };

export const questionsS148: Record<string, Question> = {
  'ch14/14.8/q-table': {
    kind: 'numeric',
    id: 'ch14/14.8/q-table',
    prompt: <>2<sup>22</sup> keys, 256 threads per block, r = 8, sixteen keys per thread. How many entries does the bucket table have?</>,
    answer: tableEntries(N22, 8, 256, 16),
    unit: 'entries',
    explain: (
      <p>
        A block now owns 16 × 256 = 4,096 keys, so there are 2<sup>22</sup> / 4,096 = 1,024 blocks, each with 256 buckets:{' '}
        {big(tableEntries(N22, 8, 256, 16))} entries. At one key per thread it was {big(tableEntries(N22, 8, 256, 1))}, sixteen
        times more.
      </p>
    ),
    source: 'ch. 14 §14.8, p. 345',
  },
  'ch14/14.8/q-smem': {
    kind: 'numeric',
    id: 'ch14/14.8/q-smem',
    prompt: <>Eight keys per thread, 256 threads, (uint32 key, uint32 value) pairs. How many bytes of shared memory does the tile alone need?</>,
    answer: tileBytes(256, 8),
    unit: 'bytes',
    explain: (
      <p>
        8 × 256 pairs of 8 bytes: {big(tileBytes(256, 8))} bytes, before the counts and any scan scratch. Compare it with the
        limit <code>python3 labs/pmpp.py doctor</code> reports for your device; then remember that a block using more shared
        memory also leaves room for fewer blocks beside it (section 5.6).
      </p>
    ),
    source: 'our arithmetic; book ch. 5 §5.6 on shared memory and occupancy',
  },
  'ch14/14.8/q-waves': {
    kind: 'choice',
    id: 'ch14/14.8/q-waves',
    prompt: (
      <>
        A device can run 40 of your blocks at a time, and the sort launches 16,384 blocks per pass. What does merging blocks
        by coarsening give up?
      </>
    ),
    options: [
      {
        label: 'Almost nothing: most blocks were waiting for a slot anyway, and larger blocks write longer runs.',
        correct: true,
      },
      { label: 'Parallelism, so it must be slower: fewer blocks means fewer keys being sorted at any one moment on the device.', why: 'Only 40 blocks run at once either way. As long as there are well over 40 blocks, the device stays as busy.' },
      { label: 'Correctness: coarsened blocks share buckets, and would need atomics to place their keys.', why: 'Each block still owns its keys and its table entries; the scan still gives every local bucket its own range.' },
    ],
    explain: (
      <p>
        The book's argument: many small blocks pay for parallelism with smaller local buckets, and the price is only worth
        paying if those blocks really run at the same time. With 16,384 blocks and room for 40, they run in waves of 40, so
        merging them gives up nothing until there are too few blocks to fill the device.
      </p>
    ),
    source: 'ch. 14 §14.8, p. 344',
  },
  'ch14/14.8/q-fig': {
    kind: 'numeric',
    id: 'ch14/14.8/q-fig',
    prompt: <>Fig. 14.13 splits sixteen keys between two blocks of four threads. How many keys per thread does that imply?</>,
    answer: 2,
    unit: 'keys',
    explain: (
      <p>
        16 keys / (2 blocks × 4 threads) = 2. The figure draws each block's four threads over only the first four of its eight
        keys and never says which thread handles which key; two per thread is the only reading that adds up.
      </p>
    ),
    source: 'ch. 14 §14.8, Fig. 14.13, p. 345; notes, "Unclear in the PDF" item 8',
  },
};

// ---------------------------------------------------------------- the section
export function S148() {
  const measured = useMeasuredC(8);
  const rowBlock = 256;
  const perThread = Math.ceil(VOCAB / rowBlock);
  return (
    <>
      <p>
        The local sort has a cost that has nothing to do with the digit. Every block writes its own local buckets, so the more
        blocks the keys are spread over, the fewer keys each block holds, the smaller its buckets, and the shorter its runs.
        That price buys parallelism, and it is worth paying only if the blocks really run at the same time. If the hardware
        runs them one wave after another anyway, the small buckets are paid for and nothing is bought.
      </p>
      <Source section="14.8" pages="344" />
      <p>
        So give each thread several keys. With c keys per thread a block of B threads owns c × B keys, its local buckets are
        on average c times larger, and consecutive threads are more likely to write consecutive addresses. The book's figure
        does this with a 2-bit digit: two blocks of four threads over the sixteen keys, each local bucket holding two keys,
        and the arrows to the output come in parallel pairs.
      </p>
      <Source section="14.8" pages="344–345" figs="Fig. 14.13" />
      <Aside>
        <p>
          Fig. 14.13 draws each block's threads over only half of its keys and does not say which thread owns which key. Eight
          keys per four-thread block means two keys per thread. The explorer below makes that explicit.
        </p>
      </Aside>
      <Fig1413 />
      <Ask q={questionsS148['ch14/14.8/q-fig']} />
      <p>
        There is a second benefit. The table has one entry per bucket per block, so c times fewer blocks make it c times
        smaller, and the global scan between the phases c times cheaper. At r = 8 that matters: with 256 threads per block and
        one key per thread, section 14.7 found the table as long as the data.
      </p>
      <Source section="14.8" pages="345" />
      <Ask q={questionsS148['ch14/14.8/q-table']} />

      <h2>What it costs</h2>
      <p>
        The tile lives in shared memory: c × B keys and as many values, 8 bytes a pair, plus the block's counts. Shared memory
        per block is limited, and the more a block takes, the fewer blocks fit on a multiprocessor beside it, which is chapter
        5's occupancy budget. So shared memory caps c, and occupancy can fall well before that cap. On Metal the limit is the
        device's <code>maxThreadgroupMemoryLength</code>; the explorer reads it, or <code>sharedMemPerBlock</code> on CUDA and
        HIP, from your doctor report.
      </p>
      <Beyond title="Ours">
        <p>
          The book names no cost of coarsening here; this one comes from chapters 5 and 6. Metal's threadgroup memory limit and
          how to size it at launch are in ch05-facts C1 and C8.
        </p>
      </Beyond>
      <Ask q={questionsS148['ch14/14.8/q-smem']} />
      <Predict
        id="ch14/14.8/p-best-c"
        prompt="At r = 8, lab 14.2 times its local-sort design with 1, 2, 4 and 8 keys per thread. Which will be fastest on your device?"
        unit="keys per thread"
      >
        {(v) =>
          measured.length ? (
            <>
              {measured.map((d) => {
                const best = fastest([1, 2, 4, 8].map((c) => ({ at: c, seconds: d.byC[c] })));
                return best ? (
                  <p key={d.b} data-best-c={`${d.b}-${best.at}`}>
                    <b>{d.device}</b> ({d.b}): fastest at <b>c = {best.at}</b>, {fmt(best.seconds * 1000)} ms
                    {best.at === v ? ', as you said' : `; you said ${v}`}.
                  </p>
                ) : null;
              })}
              <p className="muted">Compare it with the row where the explorer's sectors stop falling and the shared memory starts to climb.</p>
            </>
          ) : (
            <p data-measured="none">
              No measurement on this machine yet. Run <code>python3 labs/pmpp.py run 14.2 --solution --bench</code>, then
              reload this page.
            </p>
          )
        }
      </Predict>
      <p>
        The explorer sweeps c from 1 to 32 for a problem, a digit width and a block size of your choice, and checks each
        row's shared memory against the limit your device reported, or one you type.
      </p>
      <Simplification>
        <p>
          As in section 14.7: bytes and sectors are counted, not timed, on sampled keys. The shared-memory limit is the figure
          your device reported or the one you type. Registers, occupancy and the scan's own scratch space are not modelled.
        </p>
      </Simplification>
      <KeysPerThread />
      <p>
        Coarsening lengthens the runs and shrinks the table until the tile no longer fits. Exercise 3 asks for exactly this,
        and lab 14.2 exposes the keys per thread as a setting so that you can measure where your device stops gaining.
      </p>
      <Source section="14.11" pages="347" />
      <Today checked="2026-11-20">
        <p>
          CUB's <code>items_per_thread</code> is this factor: the tuning example in its documentation sets 21 for OneSweep and 25
          for the downsweep. Its <code>BlockRadixSort</code> documentation says efficiency grows with items per thread until
          register pressure or shared memory lowers occupancy too far (ch14-facts A4).
        </p>
      </Today>
      <Ask q={questionsS148['ch14/14.8/q-waves']} />

      <h2>Coarsening taken to its end</h2>
      <p>
        Lab 14.3, the sorted-candidates milestone, takes the idea as far as it goes: one block of {rowBlock} threads per
        vocabulary row, so a row of {big(VOCAB)} tokens gives each thread about {perThread} keys. That is far past any tile:{' '}
        {perThread} × {rowBlock} pairs of 8 bytes is {big(tileBytes(rowBlock, perThread))} bytes, so the block cannot hold its
        row in shared memory and has to work through it a piece at a time. What it gains is that a row's bucket counts never
        leave the block, so no scan across blocks and no grid-wide step between passes are needed. The price is the one this
        page started with, turned around: at batch 1 that is a single block for the whole device, the situation the explorer
        of <a href={href('ch10', '10.10')}>section 10.10</a> showed for a softmax with one block per row.
      </p>
      <Beyond title="Ours">
        <p>
          The book has no row-wise or batched sort; this design is lab 14.3's, and what it implies is our reading. The
          vocabulary size is one of those in ch09-facts D7.
        </p>
      </Beyond>
    </>
  );
}
