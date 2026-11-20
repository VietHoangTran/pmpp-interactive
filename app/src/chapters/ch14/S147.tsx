import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { benchRows, fmt, realBackends, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  LAB2,
  PROBLEMS,
  R_CHOICES,
  directTraffic,
  fastest,
  measuredLocal,
  passes,
  sweepR,
  tableEntries,
  type Problem,
  type Row,
} from '../../sim/radixcost';
import { BOOK_KEYS, bin, localPass, warpStores, summarise } from '../../sim/radixlocal';

const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const dec = (v: number) => (Math.round(v * 10) / 10).toFixed(1);
const big = (v: number) => v.toLocaleString('en-US');

function Seg<T extends string | number>({ label, value, options, onPick, show, disabled }: {
  label: string; value: T; options: readonly T[]; onPick: (v: T) => void; show?: (v: T) => string; disabled?: (v: T) => boolean;
}) {
  return (
    <span style={FIELD}>
      {label}
      <span className="seg" role="group" aria-label={label}>
        {options.map((o) => (
          <button key={String(o)} aria-pressed={value === o} disabled={disabled?.(o)} onClick={() => onPick(o)}>
            {show ? show(o) : String(o)}
          </button>
        ))}
      </span>
    </span>
  );
}

/** The digit widths lab 14.2's benchmark times at one key per thread (labs/ch14/lab2_multibit/main.cpp). */
const TIMED_R = [1, 2, 4, 6, 8];

/** Library choices of r (ch14-facts A4, B3, D6): tunings, not rules. */
const CHOSEN: Record<number, string> = {
  6: 'CUB, single tile',
  7: 'CUB, multi-pass',
  8: 'CUB OneSweep; rocPRIM; PyTorch MPS',
};

// ---------------------------------------------------------------- lab 14.2's measurements
function useMeasured() {
  const { reports } = useReports();
  return realBackends(reports)
    .map((b) => {
      const rows = benchRows(reports[b], LAB2);
      const byR = Object.fromEntries(R_CHOICES.map((r) => [r, measuredLocal(rows, r, 1)]));
      return { b, device: String(reports[b].device.name ?? b), byR };
    })
    .filter((d) => Object.values(d.byR).some((v) => v != null));
}

// ---------------------------------------------------------------- a small line chart over r
const XS = R_CHOICES;
function Chart({ title, series, log, unit, floor, id }: {
  title: string; id: string;
  /** `inTitle`: the title already names the series, so no label is drawn on the line. */
  series: { label: string; values: (number | undefined)[]; dashed?: boolean; tier?: string; inTitle?: boolean }[];
  log?: boolean; unit: string; floor?: { value: number; label: string; left?: boolean };
}) {
  const W = 300, H = 170, L = 44, R = 10, T = 16, B = 28;
  const all = series.flatMap((s) => s.values.filter((v): v is number => v != null && v > 0)).concat(floor ? [floor.value] : []);
  if (!all.length) return null;
  let lo = Math.min(...all), hi = Math.max(...all);
  if (!log) lo = 0;
  if (hi === lo) hi = lo + 1;
  const f = (v: number) => (log ? Math.log10(v) : v);
  const y = (v: number) => T + (H - T - B) * (1 - (f(v) - f(lo)) / (f(hi) - f(lo)));
  const x = (i: number) => L + ((W - L - R) * (i < 8 ? i : 8.6)) / 8.6;
  const ticks = log
    ? Array.from({ length: Math.floor(Math.log10(hi)) - Math.ceil(Math.log10(lo)) + 1 }, (_, k) => 10 ** (Math.ceil(Math.log10(lo)) + k))
    : [lo, (lo + hi) / 2, hi];
  return (
    <figure style={{ margin: 0, flex: '1 1 17rem', minWidth: 0, maxWidth: '24rem' }} data-chart={id}>
      <figcaption className="ui" style={{ fontSize: '0.84rem', marginBottom: 2 }}><b>{title}</b></figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', display: 'block' }} role="img" aria-label={title}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} stroke="var(--rule-soft)" />
            <text x={L - 4} y={y(t) + 3} fontSize="9" textAnchor="end" style={{ fill: 'var(--ink-3)' }}>{fmt(t, 2)}</text>
          </g>
        ))}
        {XS.map((r, i) => (
          <text key={r} x={x(i)} y={H - 12} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{r}</text>
        ))}
        <text x={(L + W) / 2} y={H - 1} fontSize="9" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>digit width r (bits); {unit}</text>
        <line x1={x(7) + 9} x2={x(8) - 9} y1={H - B} y2={H - B} stroke="var(--ink-3)" strokeDasharray="2 2" />
        {floor ? (
          <g>
            <line x1={L} x2={W - R} y1={y(floor.value)} y2={y(floor.value)} stroke="var(--ink)" strokeDasharray="4 3" />
            <text x={floor.left ? L + 3 : W - R} y={y(floor.value) - 3} fontSize="9" textAnchor={floor.left ? 'start' : 'end'} style={{ fill: 'var(--ink-2)' }}>{floor.label}</text>
          </g>
        ) : null}
        {series.map((s) => {
          const pts = s.values.map((v, i) => (v != null && v > 0 ? [x(i), y(v)] : null)).filter((p): p is number[] => !!p);
          const col = `var(--${s.tier ?? 'global'})`; // the keys and the table live in global memory; a measurement is drawn in ink
          return (
            <g key={s.label} data-series={s.label}>
              <polyline points={pts.map((p) => p.join(',')).join(' ')} fill="none" stroke={col} strokeWidth="1.8" strokeDasharray={s.dashed ? '5 3' : undefined} />
              {pts.map((p, k) => <circle key={k} cx={p[0]} cy={p[1]} r="2.6" fill={col} />)}
              {pts.length && !s.inTitle ? (
                <text x={pts[pts.length - 1][0] - 4} y={pts[pts.length - 1][1] - 6} fontSize="9" textAnchor="end" style={{ fill: col }}>{s.label}</text>
              ) : null}
            </g>
          );
        })}
      </svg>
    </figure>
  );
}

// ---------------------------------------------------------------- the explorer
const THREADS = [128, 256, 512, 1024];
type ProblemId = Problem['id'];

function BookExample() {
  const p = localPass(BOOK_KEYS, 0, 2, 4);
  const runs = summarise(warpStores(p, 32, 4, 32));
  const runs1 = summarise(warpStores(localPass(BOOK_KEYS, 0, 1, 4), 32, 4, 32));
  const ones = p.blocks.flatMap((b) => b.counts).filter((c) => c === 1).length;
  const words = ['none', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen'];
  const names = ['00', '01', '10', '11'];
  return (
    <div className="panel" style={{ overflowX: 'auto' }} data-book-example>
      <p style={{ margin: '0 0 0.5rem' }}>
        <b>The book's 2-bit example</b> (Figs. 14.11 and 14.12): sixteen 4-bit keys, four blocks of four threads, first pass.
      </p>
      <table className="data" style={{ margin: '0 0 0.6rem', minWidth: '30rem' }}>
        <thead>
          <tr>
            <th>block</th>
            {p.blocks.map((b) => <th key={b.block}>{b.block}</th>)}
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>sorted locally by the low two bits</td>
            {p.blocks.map((b) => <td key={b.block} className="nowrap" data-local={b.block}>{b.sorted.map((k) => bin(k, 4)).join(' ')}</td>)}
          </tr>
          {names.map((nm, d) => (
            <tr key={nm}>
              <td>#{nm}: count, then start</td>
              {p.blocks.map((b) => (
                <td key={b.block} className="num" data-cell={`${d}-${b.block}`}>
                  {b.counts[d]} <span className="muted">→</span> <b>{p.starts[b.block][d]}</b>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted" style={{ margin: 0, fontSize: '0.86rem' }} data-book-runs={runs.meanRuns}>
        {words[ones][0].toUpperCase() + words[ones].slice(1)} of the sixteen local buckets hold one key, so each block's store
        of four keys makes {dec(runs.meanRuns)} runs on average, where the 1-bit pass of Fig. 14.8 made {dec(runs1.meanRuns)}.
      </p>
    </div>
  );
}

function RadixWidth() {
  const [pid, setPid] = useState<ProblemId>('n20');
  const [bits, setBits] = useState<32 | 16>(32);
  const [threads, setThreads] = useState(256);
  const [book, setBook] = useState(false);
  const problem = PROBLEMS.find((p) => p.id === pid)!;
  const rows: Row[] = useMemo(() => sweepR(problem, bits, threads), [problem, bits, threads]);
  const ref = directTraffic(problem, bits);
  const measured = useMeasured();
  const minPerKey = Math.min(...rows.map((r) => r.traffic.perKey));
  const bestTraffic = rows.find((r) => r.traffic.perKey === minPerKey)!.r;

  return (
    <Bench
      title="Radix width"
      task="Read down the table as r grows: passes and bytes fall, runs climb, the table grows. Then change the problem, the key width and the block size."
      foot={
        <>
          Model: src/sim/radixcost.ts on src/sim/radixlocal.ts. One key per thread (section 14.8 varies it). Bytes are counted
          per key over the whole sort: per pass the count kernel reads the keys and the scatter reads and writes (key, 4-byte
          value) pairs, and each table entry is touched four times; ours, not measured. Runs and sectors are averaged over every
          pass of a seeded sample of {rows[0].local.n.toLocaleString('en-US')} keys in {rows[0].local.blocks} blocks, 32-lane warps,
          NVIDIA's 32-byte sectors.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <Seg label="Problem" value={pid} options={PROBLEMS.map((p) => p.id)} show={(id) => PROBLEMS.find((p) => p.id === id)!.label}
          onPick={(v) => { setPid(v); setBook(false); }} />
        <Seg label="Key width" value={bits} options={[32, 16] as (32 | 16)[]} show={(b) => (b === 32 ? '32 bits' : '16 bits (bf16)')} onPick={setBits} />
        <Seg label="Threads per block" value={threads} options={THREADS} onPick={setThreads} />
        <span style={FIELD}>
          Preset
          <span className="seg" role="group" aria-label="Preset">
            <button aria-pressed={book} onClick={() => setBook(!book)}>the book's 2-bit example</button>
          </span>
        </span>
      </div>

      {book ? <BookExample /> : null}

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', margin: book ? '1rem 0' : '0 0 1rem' }}>
        <Chart
          id="bytes"
          title="Bytes moved per key, whole sort"
          unit="log scale"
          log
          series={[{ label: 'local design', values: rows.map((r) => r.traffic.perKey) }]}
          floor={{ value: ref.perKey, label: "14.5's one-bit sort", left: true }}
        />
        <Chart
          id="table"
          title="Table entries per key"
          unit="log scale"
          log
          series={[{ label: 'table', values: rows.map((r) => r.tableToKeys) }]}
          floor={{ value: 1, label: 'as many entries as keys', left: true }}
        />
        <Chart
          id="sectors"
          title="Sectors per warp store (sampled)"
          unit="32-byte sectors"
          series={[{ label: 'local design', values: rows.map((r) => r.local.meanUnits) }]}
          floor={{ value: rows[0].local.meanMin, label: 'one coalesced store' }}
        />
        {measured.map((d) => (
          <Chart
            key={d.b}
            id={`measured-${d.b}`}
            title={`Measured on ${d.device} (${d.b}), lab 14.2: local sort, c = 1`}
            unit="milliseconds"
            series={[{ label: 'local, c = 1', inTitle: true, values: R_CHOICES.map((r) => (d.byR[r] != null ? d.byR[r]! * 1000 : undefined)), tier: 'ink' }]}
          />
        ))}
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0, minWidth: '46rem', background: 'var(--page)' }} aria-label="Costs by digit width">
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.5rem' }}>r</th>
              <th className="num">passes</th>
              <th className="num">launches</th>
              <th className="num">bytes per key</th>
              <th className="num">table entries</th>
              <th className="num">table ÷ keys</th>
              <th className="num">keys per local bucket</th>
              <th className="num">runs per store</th>
              <th className="num">sectors per store</th>
              {measured.map((d) => <th key={d.b} className="num">{d.b}, ms</th>)}
              <th>chosen by</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.r} data-r={r.r}>
                <td style={{ paddingLeft: '0.5rem' }}><b>{r.r}</b></td>
                <td className="num" data-col="passes">{r.passes}</td>
                <td className="num" data-col="launches">{r.launches}</td>
                <td className="num" data-col="bytes">{r.r === bestTraffic ? <b>{fmt(r.traffic.perKey)}</b> : fmt(r.traffic.perKey)}</td>
                <td className="num" data-col="table">{big(r.tableEntries)}</td>
                <td className="num" data-col="ratio" style={{ color: r.tableToKeys > 1 ? 'var(--fail)' : undefined }}>{fmt(r.tableToKeys)}</td>
                <td className="num" data-col="bucket">{fmt(r.keysPerBucket)}</td>
                <td className="num" data-col="runs">{dec(r.local.meanRuns)}</td>
                <td className="num" data-col="sectors">{dec(r.local.meanUnits)}</td>
                {measured.map((d) => <td key={d.b} className="num" data-measured={`${d.b}-${r.r}`}>{d.byR[r.r] != null ? fmt(d.byR[r.r]! * 1000) : <span className="muted">no row</span>}</td>)}
                <td className="muted" style={{ fontSize: '0.84rem' }}>{CHOSEN[r.r] ?? ''}</td>
              </tr>
            ))}
            <tr data-r="ref">
              <td style={{ paddingLeft: '0.5rem' }} className="muted" colSpan={3}>section 14.5's one-bit sort</td>
              <td className="num">{fmt(ref.perKey)}</td>
              <td colSpan={6 + measured.length} className="muted">a scan over every key in every pass instead of a table</td>
            </tr>
          </tbody>
        </table>
      </div>
      {!measured.length ? (
        <p className="ui" style={{ margin: '0.7rem 0 0' }} data-measured="none">
          No lab 14.2 measurement on this machine yet. Run <code>python3 labs/pmpp.py run 14.2 --solution --bench</code>, then
          reload: a column and a chart of measured times appear for each device.
        </p>
      ) : (
        <p className="ui" style={{ margin: '0.7rem 0 0' }}>
          The millisecond columns are measurements of lab 14.2 on this machine's devices, at its own problem size; everything
          else is the model.
        </p>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
const N22 = { rows: 1, rowLength: 2 ** 22 };
const BOOK2 = localPass(BOOK_KEYS, 0, 2, 4);

export const questionsS147: Record<string, Question> = {
  'ch14/14.7/q-passes': {
    kind: 'numeric',
    id: 'ch14/14.7/q-passes',
    prompt: <>You sort bf16 logits as 16-bit keys with a 6-bit digit. How many passes does the sort make?</>,
    answer: passes(16, 6),
    unit: 'passes',
    explain: (
      <p>
        ceil(16 / 6) = {passes(16, 6)}: bits 0 to 5, 6 to 11, and 12 to 15. The last pass has only four bits to work on, so
        it uses 16 of its 64 buckets. At r = 8 the same keys take {passes(16, 8)} passes; PyTorch's MPS sort picks 8 bits for
        16-bit keys (ch14-facts D6).
      </p>
    ),
    source: 'ch. 14 §14.7, p. 342; 16-bit float keys: ch14-facts A2',
  },
  'ch14/14.7/q-table': {
    kind: 'numeric',
    id: 'ch14/14.7/q-table',
    prompt: <>2<sup>22</sup> keys, 256 threads per block, one key per thread, r = 8. How many entries does the bucket table have?</>,
    answer: tableEntries(N22, 8, 256, 1),
    unit: 'entries',
    explain: (
      <p>
        2<sup>22</sup> / 256 = 16,384 blocks, each with 256 buckets: {big(tableEntries(N22, 8, 256, 1))} entries, exactly
        as many as there are keys. The scan of the table now costs about as much as moving the keys, in every pass. That is
        the book's second cost of a large radix.
      </p>
    ),
    source: 'ch. 14 §14.7, pp. 343–344',
  },
  'ch14/14.7/q-2bit': {
    kind: 'numeric',
    id: 'ch14/14.7/q-2bit',
    prompt: (
      <>
        The book's 2-bit example has these bucket counts for blocks 0 to 3: #00 = 1 1 0 2, #01 = 1 1 2 0, #10 = 1 1 1 1,
        #11 = 1 1 1 1. Where does block 2's 10 bucket start?
      </>
    ),
    answer: BOOK2.starts[2][2],
    explain: (
      <p>
        Everything with a smaller digit comes first: 4 keys in 00 and 4 in 01. Then the 10 keys of the blocks before block 2:
        one in block 0, one in block 1. 4 + 4 + 1 + 1 = {BOOK2.starts[2][2]}, the third entry of the row for 10 in Fig. 14.12.
      </p>
    ),
    source: 'ch. 14 §14.7, Fig. 14.12, pp. 343–344',
  },
  'ch14/14.7/q-sixteen': {
    kind: 'choice',
    id: 'ch14/14.7/q-sixteen',
    prompt: <>Why not use r = 16, and sort 32-bit keys in two passes?</>,
    options: [
      { label: 'Two passes cannot be stable.', why: 'Each pass is a stable partition whatever its width; two stable passes sort 32-bit keys correctly. The problem is cost, not correctness.' },
      { label: 'The hardware cannot shift a key by 16 bits.', why: 'A shift by 16 is as cheap as a shift by 1. The digit extraction is the same expression for any r.' },
      {
        label:
          'Each block would have 65,536 buckets for a few hundred keys: almost every local bucket holds one key or none, so nothing coalesces; the table has 65,536 entries per block, far more than the keys; and the block\'s counts alone would need 256 KiB of shared memory.',
        correct: true,
      },
    ],
    explain: (
      <p>
        Both of the book's costs go to their extreme. With 256 threads and one key each, the 65,536 buckets of a block hold
        about 0.004 keys on average, so every store is a scatter; and 65,536 four-byte counts are 256 KiB per block. Compare
        that with the shared-memory limit <code>python3 labs/pmpp.py doctor</code> reports for your device, and set r to 16 in
        the explorer.
      </p>
    ),
    source: 'ch. 14 §14.7, pp. 343–344',
  },
};

// ---------------------------------------------------------------- the section
export function S147() {
  const measured = useMeasured();
  return (
    <>
      <p>
        With one bit per pass, a sort of 32-bit keys makes 32 passes, each a full trip through global memory. A wider digit
        makes fewer. With an r-bit digit a pass distributes keys into 2<sup>r</sup> buckets, and keys of k bits need
        ceil(k / r) passes. The book's example sorts its 4-bit keys with a 2-bit digit: first by the low two bits into the
        buckets 00, 01, 10 and 11, then by the high two bits, each pass keeping the order inside every bucket.
      </p>
      <Source section="14.7" pages="342" figs="Fig. 14.10" />
      <p>
        Nothing else in the design changes: one thread per key, a local sort in shared memory, then bucket-by-bucket writes. The
        local sort of an r-bit digit is r one-bit splits in a row, each with its own scan over the block and no coordination
        with other blocks in between. After r splits the block's keys sit in 2<sup>r</sup> local buckets.
      </p>
      <Source section="14.7" pages="342" figs="Fig. 14.11" />
      <p>
        The table gets a row per digit value, 2<sup>r</sup> rows, each with one entry per block, and it is still laid out
        digit by digit. In the book's example the four rows scan to starts 0, 1, 2, 2 for 00; 4, 5, 6, 8 for 01; 8, 9, 10, 11
        for 10; and 12, 13, 14, 15 for 11.
      </p>
      <Source section="14.7" pages="342–343" figs="Fig. 14.12" />
      <Ask q={questionsS147['ch14/14.7/q-2bit']} />

      <h2>What a wider digit buys, and what it costs</h2>
      <p>
        Fewer passes mean fewer kernel launches, fewer trips of the keys through global memory, and fewer scans of a table.
        That is the whole benefit.
      </p>
      <Source section="14.7" pages="343" />
      <p>
        The costs are two. Each block now spreads its keys over more local buckets, so each bucket holds fewer keys and the
        block writes more, shorter runs: in Fig. 14.11 most local buckets hold a single key. And the table has 2<sup>r</sup>{' '}
        entries per block, so the scan between the phases grows with it. The book stops there. It says the radix must balance
        the two and gives no value.
      </p>
      <Source section="14.7" pages="343–344" />
      <Predict
        id="ch14/14.7/p-best-r"
        prompt="Lab 14.2 times its local-sort design at one key per thread for r = 1, 2, 4, 6 and 8. Which of those five will be fastest on your device?"
        unit="bits"
      >
        {(v) =>
          measured.length ? (
            <>
              {measured.map((d) => {
                const best = fastest(TIMED_R.map((r) => ({ at: r, seconds: d.byR[r] })));
                return best ? (
                  <p key={d.b} data-best-r={`${d.b}-${best.at}`}>
                    <b>{d.device}</b> ({d.b}): fastest at <b>r = {best.at}</b>, {fmt(best.seconds * 1000)} ms
                    {best.at === v ? ', as you said' : `; you said ${v}`}.
                  </p>
                ) : null;
              })}
              <p className="muted">The explorer's model has no clock. Compare where its bytes stop falling with where your times do.</p>
            </>
          ) : (
            <p data-measured="none">
              No measurement on this machine yet. Run <code>python3 labs/pmpp.py run 14.2 --solution --bench</code>, then
              reload this page. Meanwhile, the explorer below shows what each choice costs in bytes and sectors.
            </p>
          )
        }
      </Predict>
      <p>
        The explorer lays the costs side by side for every r from 1 to 8, and 16. The charts show the three trends; the table
        gives the numbers, with the digit widths that library sorts choose marked on the right.
      </p>
      <Simplification>
        <p>
          Bytes are counted from the model, not measured: no caches, no occupancy, no time. Runs and sectors come from sampled
          keys, a few blocks' worth. Only the columns marked as measured come from your device.
        </p>
      </Simplification>
      <RadixWidth />
      <p>
        Read the bytes column from the top. Bytes per key fall almost as 1/r while the keys dominate, then flatten and turn up
        once the table catches the data; at r = 16 they explode. Meanwhile the runs per store climb towards the warp width,
        because the local buckets shrink below a warp's 32 keys. The cheapest r in bytes is not the cheapest in sectors, and
        neither is a time. Somewhere in between is the best digit width, and only the device can say where.
      </p>
      <Today checked="2026-11-20">
        <p>
          CUB's tuned digit widths for keys wider than a byte are 8 bits in OneSweep, 7 in its multi-pass path and 6 for a
          single tile (ch14-facts A4). rocPRIM's tuned OneSweep configurations use 8 in 374 of 384 (B3). PyTorch's MPS sort uses
          8 for 16- and 32-bit keys; its source comment says that halves global traffic against 4 bits for the same block-local
          work (D6). hipCUB's <code>BlockRadixSort</code> accepts a <code>RADIX_BITS</code> parameter and ignores it (B3).
          These are tunings for particular devices, not rules.
        </p>
      </Today>
      <Ask q={questionsS147['ch14/14.7/q-passes']} />
      <Ask q={questionsS147['ch14/14.7/q-table']} />
      <Ask q={questionsS147['ch14/14.7/q-sixteen']} />

      <h2>Where this meets bf16 logits</h2>
      <p>
        A sampler that holds its logits in bf16 can sort them as 16-bit keys: section 14.4's float transform works on 16
        bits as it does on 32. With 8-bit digits that is two passes instead of four, each a per-block 256-bin count (the
        privatised histogram of <a href={href('ch09', '9.4')}>chapter 9</a>), a scan of the table and a scatter. Set the
        explorer's key width to 16 bits and read the r = 8 row: half the passes, and each pass still writes and scans a
        table of 256 entries per block while every key carries fewer bytes, so the table is a larger share of what is left.
      </p>
      <Beyond title="Ours">
        <p>
          The bf16 example is ours; the book's keys are unsigned integers. CUB lists <code>__nv_bfloat16</code> among its
          radix sort's key types (ch14-facts A2), and PyTorch's MPS sort uses 8-bit digits for 16-bit keys (D6).
        </p>
      </Beyond>
      <Aside>
        <p>
          Lab 14.2 works exercises 1 to 3 (p. 347); its digit width is the r of this page. See the{' '}
          <a href={href('ch14', 'labs')}>labs page</a>.
        </p>
      </Aside>
    </>
  );
}
