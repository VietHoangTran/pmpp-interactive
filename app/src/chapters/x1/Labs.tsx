import { useState, type ReactNode } from 'react';
import { Predict, type Question } from '../../components/Ask';
import { Beyond, Source, Today, type Tier } from '../../components/notes';
import {
  BackendsAndEmulator,
  LabFiles,
  LabPrediction,
  LabResults,
  LabsIntro,
  LabsPage,
  ReferenceSolution,
  RaceStatus,
  RunningLabs,
  VerificationStatus,
  benchRow,
  benchSets,
  defineLabs,
  measure,
  msCol,
  numCol,
  rawCol,
  useLabs,
  type BenchSet,
} from '../../components/LabKit';
import { fmt, num, type BenchRow, type Reports, type RunResult } from '../../lib/reports';
import { href } from '../../lib/router';

// The page is built from components/LabKit.tsx; what is here is module X1's own content.
// No question on this page: the labs are the exercise. Its six predictions are stored, not
// graded: x1/labs/p-lab1-sort, p-lab2-fused, p-lab2-sort, p-lab3-cpu, p-lab3-batch, p-lab4-emu.
// pmpp.py resolves "N.M" only for chapters, so the labs are numbered by path, as appC's are.
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('x1', [
  { num: 'x1/lab1', dir: 'lab1_topk', name: 'top-k by radix select (X1.1)' },
  { num: 'x1/lab2', dir: 'lab2_filters', name: 'top-p and min-p, composed and fused (X1.2)' },
  { num: 'x1/lab3', dir: 'lab3_sampler', name: 'the M5 sampler (X1.3)' },
  { num: 'x1/lab4', dir: 'lab4_filterhunt', name: 'filter hunt (X1.4)' },
]);
const [l1, l2, l3, l4] = LABS;

/** Identifier colours in the reference solutions: where each value lives. */
const tiers = (global: string[], shared: string[], reg: string[]): Record<string, Tier> => ({
  ...Object.fromEntries(global.map((k) => [k, 'global' as Tier])),
  ...Object.fromEntries(shared.map((k) => [k, 'shared' as Tier])),
  ...Object.fromEntries(reg.map((k) => [k, 'reg' as Tier])),
});
const TIERS_1 = tiers(
  ['logitBits', 'scratch', 'outIdx', 'outBits', 'outVal', 'logits', 'row', 'idxRow', 'valRow', 'thr'],
  ['hist_s', 'pick_s', 'warpTies_s', 'warpKeeps_s', 'hist', 'pick', 'group_ties', 'group_keeps'],
  ['prefix', 'mask', 'need', 'key', 'kStar', 'tiesBefore', 'keptBefore', 'slot', 'keep', 'above', 'tie'],
);
const TIERS_2 = tiers(
  ['sortedLogits', 'sortedProbs', 'candLogits', 'sortedPos', 'keptTopP', 'n', 'rowLogits', 'rowProbs'],
  ['stage_s', 'warpSums_s', 'warpCounts_s', 'warpMinP_s', 'warpTopP_s', 'simdSums_s', 'simdCounts_s', 'simdMinP_s', 'simdTopP_s', 'myStage'],
  ['l0', 'threshold', 'total', 'inv', 'carry', 'prob', 'keptMinP', 's'],
);
const TIERS_3 = tiers(
  ['logits', 'counts', 'seeds', 'streams', 'steps', 'tokens', 'uOut', 'uIn', 'workspace', 'sortedLogits', 'sortedPos', 'candIdx', 'n'],
  ['sums_s', 'chunk_s', 'hit_s', 'sums'],
  ['u', 'target', 'total', 'aggregate', 'winner', 'part', 'run', 'l0', 'w'],
);

// ---------------------------------------------------------------- reading the reports
const VOCAB = 151936;
const ms = (s: number | undefined) => (s ? `${fmt(s * 1000)} ms` : '?');
const big = (v: unknown) => (num(v) != null ? num(v)!.toLocaleString('en-US') : '');
const secs = (r: BenchRow | undefined) => num(r?.seconds);

/** Lab X1.1: the full sort's time over the select's, 64 × 151,936, k = 40; batch 1 alongside. */
const sortOverSelect = (reports: Reports) =>
  measure(reports, l1, (s: BenchSet) => {
    const at = (rows: number) => {
      const sel = secs(benchRow(s, 'topk_select', { rows, cols: VOCAB, k: 40 }));
      const sort = secs(benchRow(s, 'full_sort_then_first_k', { rows, cols: VOCAB, k: 40 }));
      return sel && sort ? { sel, sort } : null;
    };
    const m = at(64);
    if (!m) return null;
    const one = at(1);
    return {
      value: m.sort / m.sel,
      detail:
        `the select ${ms(m.sel)} against the sort ${ms(m.sort)} per call, 64 × 151,936, k = 40` +
        (one ? `; at 1 × 151,936 the ratio is ${fmt(one.sort / one.sel)}` : '; no batch-1 row'),
    };
  });

/** Lab X1.2: the composed chain after the sort over the fused kernel, 64 × 151,936. */
const fusedSpeedup = (reports: Reports) =>
  measure(reports, l2, (s: BenchSet) => {
    const a = benchRow(s, 'filters_unfused', { rows: 64, cols: VOCAB });
    const b = benchRow(s, 'filters_fused', { rows: 64, cols: VOCAB });
    const ta = secs(a);
    const tb = secs(b);
    if (!ta || !tb) return null;
    return {
      value: ta / tb,
      detail: `composed ${ms(ta)} at ${fmt(num(a!.gb_per_s) ?? NaN)} GB/s counted, fused ${ms(tb)} at ${fmt(num(b!.gb_per_s) ?? NaN)} GB/s counted, 64 × 151,936`,
    };
  });

/** Lab X1.2: the sort's share of the composed chain, sort included, 64 × 151,936; K = 40 alongside. */
const sortShare = (reports: Reports) =>
  measure(reports, l2, (s: BenchSet) => {
    const share = (cols: number) => num(benchRow(s, 'chain_composed', { rows: 64, cols })?.sort_share);
    const big = share(VOCAB);
    if (big == null) return null;
    const small = share(40);
    return {
      value: big * 100,
      detail: `64 rows of K = 151,936` + (small != null ? `; at K = 40 the sort's share is ${fmt(small * 100)} %` : '; no K = 40 row'),
    };
  });

/** Lab X1.3: GPU time per step over one CPU thread's, 1 × 151,936 at llama.cpp's defaults. */
const gpuOverCpu = (reports: Reports) =>
  measure(reports, l3, (s: BenchSet) => {
    const g = secs(benchRow(s, 'sample_step', { rows: 1, vocab: VOCAB }));
    const c = secs(benchRow(s, 'cpu_sample_step', { rows: 1, vocab: VOCAB }));
    if (!g || !c) return null;
    return { value: g / c, detail: `GPU ${ms(g)} per step against ${ms(c)} on one CPU thread, 1 × 151,936, top-k 40` };
  });

/** Lab X1.3: a step at batch 64 over a step at batch 1, both 151,936. */
const batchGrowth = (reports: Reports) =>
  measure(reports, l3, (s: BenchSet) => {
    const one = secs(benchRow(s, 'sample_step', { rows: 1, vocab: VOCAB }));
    const many = secs(benchRow(s, 'sample_step', { rows: 64, vocab: VOCAB }));
    if (!one || !many) return null;
    return { value: many / one, detail: `${ms(many)} per step at batch 64 against ${ms(one)} at batch 1, vocabulary 151,936` };
  });

const missingBench = (labNum: string) => (
  <>
    This needs the benchmark rows of lab {labNum} from a real device. Run{' '}
    <code>python3 labs/pmpp.py run {labNum} --bench</code> once your kernels pass (or with <code>--solution</code> to time
    the reference). The emulator is never benchmarked.
  </>
);

// ---------------------------------------------------------------- results tables
const WHAT: Record<string, string> = {
  topk_select: 'your select (both kernels)',
  full_sort_then_first_k: 'lab 14.3 sort, first k',
  cpu_partial_sort: 'std::partial_sort, one CPU thread',
  sort_rows: 'lab 14.3 sort alone',
  chain_composed: 'launchFilterRows, sort included',
  filters_unfused: 'composed chain after the sort',
  filters_fused: 'fused kernel after the sort',
  sample_step: 'launchSampleRows',
  cpu_sample_step: 'CPU reference chain, one thread',
  sample_step_whole_row: 'launchSampleRows, top-k 0',
  cpu_sample_step_whole_row: 'CPU chain, top-k 0, one thread',
};
const what = { head: 'What ran', cell: (r: BenchRow) => WHAT[r.name] ?? r.name };
const colsCol = (head: string, key: string) => ({ head, num: true, cell: (r: BenchRow) => big(r[key]) });
const pctCol = (head: string, key: string) => ({
  head,
  num: true,
  cell: (r: BenchRow) => (num(r[key]) != null ? fmt(num(r[key])! * 100) : ''),
});

/** Lab X1.3: each stage timed alone, from the stage_0_s .. stage_6_s fields of the step rows. */
const STAGES = ['penalties', 'top-k select', 'sort of k', 'softmax', 'top-p', 'min-p + combine', 'draw'];
function StageTable() {
  const { reports, loading } = useLabs();
  const sets = benchSets(reports, l3);
  const rows = sets.flatMap((s) =>
    s.rows.filter((r) => (r.name === 'sample_step' || r.name === 'sample_step_whole_row') && num(r.stage_0_s) != null).map((r) => ({ s, r })),
  );
  if (loading) return <p className="muted">Reading reports.</p>;
  if (!rows.length)
    return (
      <p className="muted" data-stages-empty="x1/lab3">
        No stage timings on this machine yet: they come with <code>python3 labs/pmpp.py run x1/lab3 --bench</code>.
      </p>
    );
  return (
    <div className="wide" style={{ maxWidth: '62rem', overflowX: 'auto' }}>
      <table className="data" data-table="stages">
        <thead>
          <tr>
            <th>Backend</th>
            <th>Kernel</th>
            <th className="num">Rows</th>
            <th className="num">Vocab</th>
            <th className="num">Top-k</th>
            {STAGES.map((h) => (
              <th key={h} className="num">
                {h}, ms
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(({ s, r }, i) => (
            <tr key={i}>
              <td>{s.backend}</td>
              <td>{s.variant === 'work' ? 'yours' : 'reference solution'}</td>
              <td className="num">{String(r.rows)}</td>
              <td className="num">{big(r.vocab)}</td>
              <td className="num">{String(r.top_k)}</td>
              {STAGES.map((_, k) => {
                const v = num(r[`stage_${k}_s`]);
                return (
                  <td key={k} className="num">
                    {v == null ? '' : fmt(v * 1000)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Lab X1.3: the tokens digest of the decisive draws, per backend and variant. */
// Devices write the full digest (tokens_digest, over decisive_draws of draws) and the core's
// (core_digest, over core_draws); the emulators write the core's only. A run from before the
// core existed wrote tokens_digest alone, over exactly the core's draws: it counts as a core.
type DigestTest = {
  case: string;
  pass: boolean;
  tokens_digest?: string;
  decisive_draws?: number;
  draws?: number;
  core_digest?: string;
  core_draws?: number;
};
type DigestRow = { b: string; variant: 'solution' | 'work'; full?: string; fullDraws?: number; core: string; coreDraws: number };
function digestRows(reports: Reports): DigestRow[] {
  return Object.keys(reports).flatMap((b) =>
    (['solution', 'work'] as const).flatMap((variant): DigestRow[] => {
      const t = (reports[b].results.filter((x) => x.lab === l3.id && x.variant === variant) as RunResult[])
        .flatMap((x) => x.tests as DigestTest[])
        .find((x) => typeof x.core_digest === 'string' || typeof x.tokens_digest === 'string');
      // Only a digest that means something: its case passed, and for your work, the work is not
      // the untouched starter.
      if (!t || !t.pass || (variant === 'work' && reports[b].work_state?.[l3.id] === 'untouched')) return [];
      const legacy = typeof t.core_digest !== 'string';
      const core = legacy ? t.tokens_digest! : t.core_digest!;
      const coreDraws = legacy ? t.decisive_draws ?? 0 : t.core_draws ?? 0;
      const full = !legacy && typeof t.tokens_digest === 'string' && (t.decisive_draws ?? 0) > 0 ? t.tokens_digest : undefined;
      if (coreDraws <= 0 && !full) return [];
      return [{ b, variant, full, fullDraws: full ? t.decisive_draws : undefined, core, coreDraws }];
    }),
  );
}
function DigestTable() {
  const { reports, loading } = useLabs();
  if (loading) return <p className="muted">Reading reports.</p>;
  const found = digestRows(reports);
  const fulls = found.filter((f) => f.full);
  const distinctFull = new Set(fulls.map((f) => f.full)).size;
  const distinctCore = new Set(found.map((f) => f.core)).size;
  const same = distinctFull <= 1 && distinctCore === 1;
  const names = (rows: DigestRow[]) => [...new Set(rows.map((f) => f.b))].join(', ');
  return (
    <>
      {found.length ? (
        <table className="data" data-table="digest">
          <thead>
            <tr>
              <th>Backend</th>
              <th>Kernel</th>
              <th>Full digest</th>
              <th className="num">Decisive draws</th>
              <th>Core digest</th>
              <th className="num">Decisive draws</th>
            </tr>
          </thead>
          <tbody>
            {found.map((f) => (
              <tr key={f.b + f.variant} data-digest-backend={f.b} data-digest-variant={f.variant} data-full={f.full ?? ''} data-core={f.core}>
                <td>{f.b}</td>
                <td>{f.variant === 'work' ? 'yours' : 'reference solution'}</td>
                <td>{f.full ? <code>{f.full}</code> : <span className="muted">core only</span>}</td>
                <td className="num">{f.fullDraws ?? ''}</td>
                <td>
                  <code>{f.core}</code>
                </td>
                <td className="num">{f.coreDraws}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
      <p data-digest={found.length ? (same ? 'same' : 'differ') : 'none'}>
        {!found.length
          ? 'No run of lab X1.3 on this machine has written a tokens digest yet.'
          : same
            ? found.length === 1
              ? 'One run has written a digest so far. Run another backend to have something to compare it with.'
              : `Every run listed wrote the same digests: the same decisive tokens in the core on ${names(found)}${fulls.length > 1 ? `, and in the full case on ${names(fulls)}` : ''}.`
            : `The runs wrote ${distinctFull > 1 ? `${distinctFull} different full digests` : ''}${distinctFull > 1 && distinctCore > 1 ? ' and ' : ''}${distinctCore > 1 ? `${distinctCore} different core digests` : ''}. Some backend drew a different token on a draw the CPU reference calls decisive: compare that backend's failures first.`}
      </p>
    </>
  );
}

/** Lab X1.4's prediction: how many of the three defects the emulator reports. Not measured from the reports. */
function HuntReveal({ committed }: { committed: number }): ReactNode {
  // The count and the reason come at once; which defect is which waits until the reader says
  // the hunt is over, so the reveal does not hand over the answers before the first run.
  const [fixed, setFixed] = useState(false);
  return (
    <div data-reveal="p-lab4-emu">
      <p>
        When this lab was built, <code>emu</code> reported <b>2</b> of the three, in all three of its thread orders.{' '}
        {committed === 2 ? 'That is your number.' : `You predicted ${committed}.`}
      </p>
      <p>
        The one it missed is a race between warps: a write that, on a GPU, may land before another warp has read the old
        value. The emulator runs every thread up to its next rendezvous before it lets any thread past one, so no warp is
        ever a stretch ahead of another. A read that comes before the first rendezvous and a write that comes several
        rendezvous later are then always in the safe order, in every thread order it tries. The race backend reports such a
        pair of lines; a GPU may pass it every run.
      </p>
      {fixed ? (
        <p data-reveal-detail="p-lab4-emu">
          Defect by defect: <code>emu</code> reported defect 1 (the inclusive mass) at every K from 31 up, and defect 2 (the
          carry added after the decision) at every K above 1,024. Defect 3, the missing barrier, passed every{' '}
          <code>emu</code> case: every thread reads the block sum before its first warp-level rendezvous, and the scan's
          warp-total write comes several rendezvous later.
        </p>
      ) : (
        <p>
          <button type="button" onClick={() => setFixed(true)}>
            I have fixed all three: show which <code>emu</code> saw
          </button>
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS}>
      <LabsIntro>
        Four labs, each in its own directory under <code>labs/x1/</code>. The first replaces lab 14.3's full sort with a
        selection for top-k. The second builds top-p and min-p twice, composed from earlier labs and then as one kernel.
        The third is milestone M5: the whole sampler, logits and counts in, one token per row out. The fourth hunts three
        defects in the second lab's fused kernel.
      </LabsIntro>
      <p>
        This module is not a chapter of the book, so <code>pmpp.py</code> does not know its labs by a number such as 5.2.
        Name them by path: <code>x1/lab1</code> is lab X1.1, and so on to <code>x1/lab4</code>.
      </p>
      <Beyond title="All of this is ours">
        <p>
          The book has no sampling. Every contract, tie rule, design and test on this page is ours; the parts they are
          built from are the book's, and each lab names its sections.
        </p>
      </Beyond>
      <RunningLabs
        example="x1/lab1"
        alsoShownIn={
          <>
            the measured panels of <a href={href('x1', 'X1.4')}>X1.4</a> and <a href={href('x1', 'X1.5')}>X1.5</a>
          </>
        }
      />
      <BackendsAndEmulator />
      <p>
        Every lab of this module is also built for the emulator's race flavour, because each has a shared update to get
        right: a histogram and per-warp counts in lab X1.1, a block sum and a scan sharing one array in labs X1.2 and
        X1.4, a total and chunk totals in lab X1.3. <code>python3 labs/pmpp.py run x1/lab1 --backend race</code> runs a
        block's threads at once under ThreadSanitizer, which reports two accesses to one location with nothing ordering
        them, with both source lines. A clean run is evidence, not proof: it sees only the schedules that happened.
      </p>
      <VerificationStatus />
      <p>
        Three labs reuse earlier labs' reference solutions, which you do not edit. On CUDA, HIP and the emulators they are
        copies under file names of their own, in <code>labs/x1/shared-g11/</code> (labs X1.1 and X1.3) and{' '}
        <code>labs/x1/shared-g12/</code> (lab X1.2). The build names each generated file after its source's name, and
        every lab's file is called <code>kernel.cu</code>, so the originals cannot sit side by side; and some of their
        helper functions are not <code>static</code>, so two of them in one emulator program would define the same name
        twice. Each copy differs from its original only in those helpers being <code>static</code>, and a unit test,{' '}
        <code>app/src/chapters/x1/copies-g11.test.ts</code> or <code>copies-g12.test.ts</code>, fails if anything else
        differs. On Metal the drivers load the earlier labs' own <code>solution/kernel.metal</code> files and link their{' '}
        <code>launch.mm</code>. Each lab tests the provided code on its own first, so a broken provided file shows up as
        itself.
      </p>
      <Beyond>
        <p>The copies, and the provided min-p count of lab X1.3, are ours.</p>
      </Beyond>

      {/* ------------------------------------------------------------ X1.1 */}
      <h2>Lab X1.1: top-k by radix select, measured against a full sort</h2>
      <p>
        One host function and two kernels:{' '}
        <code>launchTopKRows(logits, outIdx, outVal, scratch, rows, cols, k)</code>. For every row of a{' '}
        <code>[rows × cols]</code> float32 tensor, row-major with the batch first, write the min(k, cols) largest logits
        and their columns, in ascending column order, every tie at the k-th value going to the lowest columns, without
        sorting the row. Lab X1.3 calls a copy of this lab's reference solution as its top-k stage.
      </p>
      <Source section="9.4, 12.3, 12.5, 14.9" pages="211–214, 292–293, 297–298, 346" figs="Figs. 9.10 and 12.3" />
      <p>
        <code>topkThreshold</code> is section <a href={href('x1', 'X1.4')}>X1.4</a>'s narrowing in: up to four 8-bit
        passes in one launch, each counting the current byte of the keys that match the prefix so far in a 256-bin
        shared-memory histogram with atomics (chapter 9), then walking the bins from the top until the count reaches the
        keys still needed. It leaves the exact k-th key K* and the number of ties at K* still needed in{' '}
        <code>scratch</code>, and stops early when every key of the crossing bin is needed. <code>topkGather</code> keeps
        every key above K* and the first ties up to that budget, ranked by a ballot and a popcount within each warp and
        running counts across rounds of 256 columns: lab 12.4's stable filter with a tie budget. The kernels never do
        float arithmetic; the launcher hands them the logits as unsigned integers, as lab 14.3 does.
      </p>
      <LabFiles
        lab={l1}
        files={[
          ['kernel.cu', <><code>topkThreshold</code>, <code>topkGather</code> and <code>launchTopKRows</code>; lab 12.4's warp helpers with their typed lane mask, and lab 14.3's key, are provided in the file</>],
          ['kernel.metal', <><code>topk_threshold</code> and <code>topk_gather</code>, the histogram a <code>threadgroup atomic_uint</code> array</>],
          ['launch.mm', <><code>encodeTopKRows</code>, both kernels in the one compute encoder it is given</>],
        ]}
      />
      <Today checked="2026-11-20 (notes/today/infra-facts.md B3; ch12-facts.md B1, C2; ch04-facts.md C2)">
        <p>
          HIP's vote is 64 bits on every AMD target and its <code>_sync</code> functions take a 64-bit mask. Metal's{' '}
          <code>simd_ballot</code> leaves the bits above the SIMD-group width undefined, and how a threadgroup splits into
          SIMD-groups is the implementation's.
        </p>
      </Today>

      <h3>What the tests check</h3>
      <p>
        Run <code>python3 labs/pmpp.py run x1/lab1 --backend emu</code> first (a few minutes). On a device: rows 1, 4 and 7
        by columns 1, 2, 31, 255, 256, 257, 1,000, 4,096, 32,000, 50,257 and 151,936, with k = 1, 2, 40, 256, 1,000, the
        row length and the row length plus 5. The data are mixed within each batch so that a wrong row offset fails:
        uniform logits, peaked rows, every logit equal, exact ties at the k-th value planted across warp and round seams,
        interleaved −0.0 and +0.0, keys sharing their top 8, 16 or 24 bits (which force 2, 3 and 4 passes), crossing bins
        at either end of the walk, and negative-only rows. The comparison is exact: the columns equal the reference's, the
        values are the logits' bits, slots past min(k, cols) keep the driver's sentinel, and 64 guard entries past every
        output are untouched. The emulators run smaller shapes; the README lists them. Lab 14.3's sort, the baseline, is
        tested too.
      </p>
      <RaceStatus lab={l1} />

      <h3>Before you benchmark</h3>
      <p>
        Section X1.4 counts the bytes each design must move per key: 80 for the sort's four passes, 20 for the select plus
        8 for each kept pair, so about four times less traffic. Decide whether the time ratio will be above or below that,
        and why. Weigh three things: the sort's scatter goes to 256 buckets, so its writes are poorly coalesced; in the
        later passes the select's atomics all land on the few bins near the crossing; and both run one block per row.
      </p>
      <LabPrediction
        id="x1/labs/p-lab1-sort"
        prompt={<>At 64 rows of 151,936 logits and k = 40, how many times faster than lab 14.3's full sort will your select be?</>}
        unit="× the sort's speed"
        measure={sortOverSelect}
        missing={missingBench('x1/lab1')}
        reading={
          <>
            Both times are the device's, per call. The byte count says what each design must ask of memory; the ratio is
            what the device made of it. The table below has every shape, and the batch-1 ratio in the detail is the case
            where one block works and the rest of the device waits.
          </>
        }
      />
      <LabResults
        lab={l1}
        cols={[
          what,
          rawCol('Rows', 'rows'),
          colsCol('Cols', 'cols'),
          rawCol('k', 'k'),
          msCol,
          numCol('Rows/s', 'rows_per_s'),
          numCol('Bytes per key', 'bytes_per_key'),
          numCol('Sort / select', 'sort_over_select'),
          numCol('CPU / select', 'cpu_over_select'),
        ]}
      />
      <p>
        How the benchmark works: rows 1, 8 and 64 by 32,000 and 151,936 columns, k = 1, 40 and 1,024. Your select and lab
        14.3's sort are timed on the device, best of 5 runs, each repeating the call until it lasts at least 2 ms (20 ms
        for the sort). One CPU thread runs <code>std::partial_sort</code> per row, which is llama.cpp's host top-k for k up
        to 128, timed with a steady clock. Every timed result is checked before it is reported.
      </p>
      <Today checked="2026-11-20 (notes/today/ch14-facts.md D3)">
        <p>llama.cpp's host top-k is a <code>std::partial_sort</code> for k up to 128.</p>
      </Today>
      <ReferenceSolution lab={l1} tiers={TIERS_1} />

      {/* ------------------------------------------------------------ X1.2 */}
      <h2>Lab X1.2: top-p and min-p, composed from earlier labs, then fused</h2>
      <p>
        Part A composes. <code>launchFilterRows</code> sorts each row of K candidate logits largest first and stably with
        lab 14.3, takes the softmax at temperature 1 with lab 10.3, counts the top-p survivors with lab 11.6's keep rule
        (keep token i when i is 0, or p is at least 1, or the mass ranked before i is below p), and counts the min-p
        survivors with your <code>minPCount</code>: sorted logits at least the first plus ln(min-p), the logarithm computed
        once on the host. Then n = max(1, min(top-p's count, min-p's count)). That is llama.cpp's order, top-p then min-p,
        and taking the minimum works because both keep a prefix of the sorted row (section{' '}
        <a href={href('x1', 'X1.6')}>X1.6</a>).
      </p>
      <Source section="10.7–10.8, 11.1, 11.4" pages="239–244, 252–254, 261–265" figs="Figs. 10.13, 10.16, 11.8 and 11.9" />
      <p>
        Part B fuses. <code>launchFilterSortedFused</code> starts from rows that are already sorted, so the maximum is
        element 0. One block of 256 threads per row adds up exp(l − l₀) with chapter 10's block reduction, then walks the
        row in chunks of 1,024 with chapter 11's block scan and the mass of earlier chunks carried in a register. It
        computes each probability, decides top-p and min-p while the mass and the logit are in registers, counts the
        survivors, and writes the probabilities and n. It reads the logits twice and writes the probabilities once.
      </p>
      <Beyond>
        <p>
          The fusion and the prefix argument are ours. The filter order and both rules are llama.cpp's, checked in the
          facts pass (notes/today/x1-facts.md D1; ch11-facts.md D2; ch12-facts.md D3).
        </p>
      </Beyond>
      <LabFiles
        lab={l2}
        files={[
          ['kernel.cu', <><code>minPCount</code>, <code>launchMinPCount</code>, <code>launchFilterRows</code>, <code>filterSortedFused</code> and <code>launchFilterSortedFused</code>; the warp sum, warp scan and block sum are provided</>],
          ['kernel.metal', <><code>min_p_count</code> and <code>filter_sorted_fused</code></>],
          ['launch.mm', <><code>encodeMinPCount</code>, <code>encodeFilterRows</code> and <code>encodeFilterSortedFused</code></>],
        ]}
      />

      <h3>What the tests check</h3>
      <p>
        Both parts run on the same inputs. The exact rows hold groups of 2ʲ logits of 0 with the rest at −1000, so every
        probability is a power of two or 0 and every partial sum is exact in any order of addition. p sits exactly on the
        inclusive sum at one token: at token 0, inside a lane's run of four, at the last token of the first chunk of 1,024,
        and past it. That token must be kept and the next one dropped. A logit planted exactly at 0 + ln(min-p) as the
        host computes it must be kept. On softmax-shaped rows the probabilities must be within 1e-4 relative of a double
        reference, and n must equal the reference's wherever no mass and no logit lies within the band of a boundary;
        inside the band either neighbour is accepted. The README derives the band.
      </p>
      <p>
        The exact rows rely on exp(0) being exactly 1 and exp(−1000) exactly 0. Vendors' accuracy bounds of a few ulp do not
        promise either. Both held on Metal and on the emulators when this lab was built; on CUDA and HIP they are
        unverified.
      </p>
      <Today checked="2026-11-20 (notes/today/ch10-facts.md A11, B4, C4)">
        <p>
          CUDA's <code>expf</code> is within 2 ulp; Metal's fast <code>exp</code> within 3 + floor(|2x|) ulp; HIP publishes
          measured differences, not bounds.
        </p>
      </Today>
      <RaceStatus lab={l2} />

      <h3>Before you benchmark</h3>
      <p>
        Count the bytes per token first. After the sort, the composed chain's softmax reads the logits three times and
        writes the probabilities, lab 11.6 reads the probabilities and writes the mass and a keep byte, and min-p reads the
        logits again: 29 bytes. The fused kernel reads the logits twice and writes the probabilities once: 12. Then decide
        whether both run at the same bandwidth. For the second prediction, ask how much work lab 14.3 does per row whatever
        K is, and how much the filters do; section <a href={href('x1', 'X1.5')}>X1.5</a>'s explorer is the place to reason
        about it.
      </p>
      <LabPrediction
        id="x1/labs/p-lab2-fused"
        prompt={<>At 64 rows of 151,936, how many times faster than the composed chain after the sort will the fused kernel be?</>}
        unit="× the composed chain's speed"
        measure={fusedSpeedup}
        missing={missingBench('x1/lab2')}
        reading={
          <>
            The GB/s are counted, not measured: the minimum bytes each design must move, divided by its time. If both ran at
            the same counted bandwidth, the speed-up would be the ratio of the byte counts.
          </>
        }
      />
      <LabPrediction
        id="x1/labs/p-lab2-sort"
        prompt={<>At 64 rows of K = 151,936, what percentage of the composed chain's time, sort included, will the sort take?</>}
        unit="% of the composed chain's time"
        measure={sortShare}
        missing={missingBench('x1/lab2')}
        reading={
          <>
            The detail gives the same share at K = 40, llama.cpp's default top-k. Whether the share falls as K shrinks says
            how much of the sort's cost depends on K and how much is fixed per row.
          </>
        }
      />
      <LabResults
        lab={l2}
        cols={[
          what,
          rawCol('Rows', 'rows'),
          colsCol('K', 'cols'),
          msCol,
          numCol('Counted GB/s', 'gb_per_s'),
          rawCol('Bytes per token', 'bytes_per_token'),
          pctCol('Sort share, %', 'sort_share'),
          numCol('Fused speed-up', 'speedup_vs_unfused'),
        ]}
      />
      <p>
        How the benchmark works: rows 1 and 64 by K = 40, 1,024, 32,000 and 151,936, on softmax-shaped rows at llama.cpp's
        default top-p 0.95 and min-p 0.05. It times lab 14.3's sort alone, your whole <code>launchFilterRows</code>, the
        composed chain after the sort, and the fused kernel on the same sorted logits. Best of 5 timed runs, each repeating
        the call until it lasts at least 2 ms; every timed result is checked first. A 1-row shape is at most 600 KB per
        buffer, so read the 64-row shapes for bandwidth.
      </p>
      <ReferenceSolution lab={l2} tiers={TIERS_2} />

      {/* ------------------------------------------------------------ X1.3 */}
      <h2>Lab X1.3: milestone M5, a batched GPU sampler</h2>
      <p>
        <code>launchSampleRows</code> takes <code>[rows × vocab]</code> logits and token counts, a seed, a stream and a
        step per row, and the settings (top-k, top-p, min-p, temperature, and the repetition, frequency and presence
        penalties), and writes one token id per row. Every stage but three is an earlier lab's kernel. You write the three
        that none of them has, and the composition.
      </p>
      <ul>
        <li>
          <b>Philox4x32-10</b>, Random123's ten rounds, with the 32 × 32 to 64-bit product as an{' '}
          <code>unsigned long long</code> multiplication (<code>ulong</code> in Metal), and a kernel that writes the four
          words for given counters and keys, so the driver checks the published vectors on the device (section{' '}
          <a href={href('x1', 'X1.7')}>X1.7</a>).
        </li>
        <li>
          <b>u</b> = (w &gt;&gt; 8) × 2⁻²⁴, which is the same float on every backend.
        </li>
        <li>
          <b>The repetition penalty</b>, llama.cpp's rule: for a token already seen, multiply a logit at or below 0 by the
          penalty and divide a positive one (section <a href={href('x1', 'X1.3')}>X1.3</a>).
        </li>
        <li>
          <b>The draw</b>, one block per row over the n survivors: weights exp((lᵢ − l₀) / T), their total by a block
          reduction, target u × total, then chunks with a running sum; the first position whose inclusive sum is strictly
          greater than the target wins, and the last positive weight if rounding leaves none (section{' '}
          <a href={href('x1', 'X1.8')}>X1.8</a>).
        </li>
      </ul>
      <Today checked="2026-11-20 (notes/today/x1-facts.md A4, A5, B3, D1, D2, E2)">
        <p>
          Random123's three 10-round known-answer vectors; <code>__umulhi</code> exists on CUDA and HIP, and the lab uses
          the 64-bit product, which needs no intrinsic on any backend. llama.cpp's default chain and repetition rule. A
          24-bit conversion never returns 1.0.
        </p>
      </Today>
      <p>
        This course's emulator and front-end stub declare <code>__umulhi</code> too (see{' '}
        <code>labs/common/include/pmpp/emu/emu_device.h</code> and <code>pmpp/check/cuda_stub.h</code>), so either form
        builds wherever a <code>.cu</code> file does.
      </p>
      <p>
        The chain is llama.cpp's default restricted to the stages M5 builds: copy the logits into the workspace,
        repetition, then lab 9.3's frequency and presence pass; if the temperature is 0 or below, lab 10.3's argmax and
        stop; otherwise lab X1.1's select (the whole row when top-k is 0 or at least the vocabulary), lab 14.3's sort of the
        candidates, lab 10.3's softmax, lab 11.6's top-p count, the provided min-p count and combine, and the draw. Every
        tie at every stage goes to the lowest token id. Section <a href={href('x1', 'X1.1')}>X1.1</a> gives the order and
        its reasons.
      </p>
      <Beyond>
        <p>
          Our design: the per-row (seed, stream, step), the counter and key mapping, the u conversion, the draw rule and
          the workspace layout. By our reading of cuRAND's header the mapping equals the first <code>curand()</code> after{' '}
          <code>curand_init(seed, stream, 4 × step)</code>; that is unverified on a device.
        </p>
      </Beyond>
      <LabFiles
        lab={l3}
        files={[
          ['kernel.cu', <><code>philox4x32_10</code>, <code>philoxKatKernel</code>, <code>uniformFromBits</code>, <code>applyRepetition</code>, <code>drawRows</code>, <code>sampleWorkspaceBytes</code> and <code>launchSampleRows</code>; the other launchers and the workspace copy are provided</>],
          ['kernel.metal', <><code>philox4x32_10</code>, <code>philox_kat</code>, <code>uniform_from_bits</code>, <code>apply_repetition</code> and <code>draw_rows</code></>],
          ['launch.mm', <><code>encodePhiloxKat</code>, <code>encodeApplyRepetition</code>, <code>encodeDrawRows</code> and <code>encodeSampleRows</code>; on Metal the workspace is one buffer per intermediate, declared in <code>sampler_metal.h</code></>],
        ]}
      />

      <h3>What the tests check</h3>
      <p>
        Philox's known answers on the device first, then u bit for bit for 4,096 (seed, stream, step) triples, the first of
        which, all zero, must give the word <code>0x6627e8d5</code>. Then the draw alone with planted u: equal weights,
        where u = m × 2⁻²⁴ must give position floor(u × n) exactly, zero weights that must never win, u near 1, a single
        survivor, chunk seams, and the fallback. Then the repetition rule on logits of both signs and exact zeros. Then the
        whole sampler against the CPU reference, at llama.cpp's defaults (top-k 40, top-p 0.95, min-p 0.05, temperature 0.8)
        with penalties, and at top-k 1, top-k 0, top-p 1, min-p 0, temperature 0, temperature 2 and no penalties, 64 seeds by
        4 steps per shape. Last, a toy decode loop whose counts are kept on the device, and a digest of the tokens.
      </p>
      <p>
        Tokens are compared only where the draw is decisive. The CPU reference records how far every decision is from
        flipping: a penalised logit, the top-k cut, top-p's and min-p's counts, the order of neighbours, and the distance
        of u × total from a boundary of the running sum. A draw within a band of a boundary is counted and printed, never
        failed, because a device's float softmax and scan may land on the other side of it. Decisive draws must match
        exactly, and more than half of every case's draws must be decisive.
      </p>
      <Today checked="2026-11-20 (notes/today/x1-facts.md C5; ch05-facts.md C7)">
        <p>
          MSL: "Either round ties to even or round toward zero rounding mode may be supported", and fast math by default.
        </p>
      </Today>
      <RaceStatus lab={l3} />
      <h3>Same seed, same tokens, across backends</h3>
      <p>
        A digest is a hash of the decisive draws of one fixed case at llama.cpp's defaults, rows of 1,000. On a GPU the
        full case is 8 rows by 16 seeds by 8 steps, 1,024 draws, nearly all of them decisive: a backend that drew a
        different token once in a few hundred draws would change it. The emulators run one thread at a time, far too slowly
        for 1,024 draws, so they run only the core, the first 2 rows by 2 seeds by 2 steps, 8 draws; GPUs hash that core separately
        too, so every backend can be compared on it. Equal digests mean equal tokens on those draws, and the table gives the
        count beside each. It reads them from this machine's runs (<code>reports/runs/</code>); on a CUDA or ROCm machine
        the same run prints its <code>tokens_digest</code> and <code>core_digest</code>, to compare with these by hand.
      </p>
      <DigestTable />

      <h3>Before you benchmark</h3>
      <p>
        Reason from two facts. Every kernel runs one block per row, so at batch 1 one multiprocessor works and the rest
        wait (<a href={href('ch10', '10.10')}>section 10.10</a>). And only the copy, the penalties and the select read the
        whole vocabulary; after top-k every stage touches 40 values per row. llama.cpp samples on one host thread by
        default, which is what the CPU column measures against.
      </p>
      <Today checked="2026-11-20 (notes/today/x1-facts.md D1)">
        <p>
          llama.cpp: <code>bool backend_sampling = false;</code> for generation. Its GPU sampling path is experimental.
        </p>
      </Today>
      <LabPrediction
        id="x1/labs/p-lab3-cpu"
        prompt={
          <>
            At batch 1 and 151,936 tokens with llama.cpp's defaults, what is the GPU's time per sampling step divided by one
            CPU thread's? (Below 1 means the GPU wins.)
          </>
        }
        unit="× one CPU thread's time"
        measure={gpuOverCpu}
        missing={missingBench('x1/lab3')}
        reading={
          <>
            Metal's time is the command buffer's GPU time; CUDA and HIP times are host wall clock around the launches and a
            synchronise; the CPU's is a steady clock on the host. The stage table shows where the GPU's time went.
          </>
        }
      />
      <LabPrediction
        id="x1/labs/p-lab3-batch"
        prompt={<>How many times longer will a sampling step take at batch 64 than at batch 1, vocabulary 151,936?</>}
        unit="× the batch-1 step"
        measure={batchGrowth}
        missing={missingBench('x1/lab3')}
        reading={
          <>
            64 would mean every row costs as much as the first, as if rows ran one after another. A number near 1 means the
            extra rows ran on multiprocessors that batch 1 left idle.
          </>
        }
      />
      <LabResults
        lab={l3}
        cols={[
          what,
          rawCol('Rows', 'rows'),
          colsCol('Vocab', 'vocab'),
          rawCol('Top-k', 'top_k'),
          msCol,
          numCol('Tokens/s', 'tokens_per_s'),
          numCol('GPU / CPU', 'gpu_over_cpu'),
        ]}
      />
      <p>Each stage timed alone, on buffers prepared by running the stages once in order:</p>
      <StageTable />
      <p>
        How the benchmark works: rows 1, 8 and 64 by vocabularies of 32,000 and 151,936 at llama.cpp's defaults with the
        penalties on, and the whole-row path (top-k 0) at rows 1 and 8 by 151,936. One CPU thread runs the same chain in
        float, top-k by <code>std::partial_sort</code>. Best of 5 timed runs, each repeating the call until it lasts at
        least 2 ms; every result is checked against the CPU reference first.
      </p>

      <h3>The project milestone</h3>
      <p>
        <code>launchSampleRows</code> completes milestone M5: <code>[rows × vocab]</code> logits and token counts in, one
        token id per row out, the same token for the same (seed, stream, step) on every backend for every decisive draw.
        It follows the project's conventions so later milestones can call it: row-major float32 with the batch first, raw
        device pointers and explicit sizes, a <code>launch</code> host function beside the kernels, the CPU reference in{' '}
        <code>main.cpp</code>. Milestone M10 (lab X3.2) calls it on a GPU, after M7's decode step, when this lab's solution is
        present.
      </p>
      <ReferenceSolution lab={l3} tiers={TIERS_3} />

      {/* ------------------------------------------------------------ X1.4 */}
      <h2>Lab X1.4: filter hunt, three defects in a fused top-p and min-p kernel</h2>
      <p>
        You are handed lab X1.2's fused kernel, in CUDA and in Metal, with the same contract, names and launchers, and three
        independent defects planted in it. It compiles. Find and fix all three. This is the kernel you have just written,
        so the defects are the ones it invites: one in a boundary rule, one in the carry between chunks, one in
        synchronisation.
      </p>
      <Source section="10.7–10.8, 11.4" pages="239–244, 261–265" figs="Figs. 10.13, 10.16, 11.8 and 11.9" />
      <LabFiles
        lab={l4}
        files={[
          ['kernel.cu', <><code>filterSortedFused</code>, with the three defects; <code>launchFilterSortedFused</code> is correct</>],
          ['kernel.metal', <><code>filter_sorted_fused</code>, with the same three</>],
          ['launch.mm', <>provided and correct; nothing to change</>],
        ]}
      />
      <p>
        The README lists what is not a defect: the typed lane mask and <code>warpSize</code> where the book writes{' '}
        <code>0xffffffff</code> and 32, the threshold <code>-FLT_MAX</code> that keeps every token when min-p is off,
        ln(min-p) computed on the host, and no max pass on a sorted row.
      </p>
      <p>The method matters more than the fixes:</p>
      <ol>
        <li>
          Before running anything, read <code>work/kernel.cu</code> and write down the defects you can see, and for each the
          backend you expect to report it: <code>emu</code>, <code>race</code>, your GPU, or none.
        </li>
        <li>
          Run the emulator: <code>python3 labs/pmpp.py run x1/lab4 --backend emu</code>. Its thread orders are fixed, so
          what it reports, it reports every time. Use what the failing cases have in common (which K, which kind of row,
          which p) and the driver's hints.
        </li>
        <li>
          Fix one defect at a time and run again. Then run <code>python3 labs/pmpp.py run x1/lab4 --backend race</code>,
          which reports two threads touching the same memory with nothing ordering them, with both source lines, whether
          or not a value came out wrong.
        </li>
        <li>
          Make the same fixes in <code>kernel.metal</code> and run on the device: <code>python3 labs/pmpp.py run x1/lab4</code>.
          Note which defects it showed you and which it hid.
        </li>
      </ol>
      <Predict
        id="x1/labs/p-lab4-emu"
        unit="defects"
        prompt={<>After reading the kernel and before running it: of the three defects, how many will <code>emu</code> report?</>}
      >
        {(c) => <HuntReveal committed={c} />}
      </Predict>
      <p>
        Whatever your GPU showed, read it against what the vendors document. Different warps make independent progress
        between barriers, and reading and writing the same location without synchronisation is undefined behaviour. A case
        that passed ten times on your GPU showed you ten schedules of one device, not a correct kernel.
      </p>
      <Today checked="2026-11-20 (notes/today/ch04-facts.md A9; infra-facts.md C5; ch10-facts.md A1)">
        <p>
          NVIDIA: warps run independently. Apple: SIMD-groups "make independent forward progress ... in the absence of
          threadgroup barrier operations". CUDA: "Reading from or writing to the same memory location without memory fences
          or synchronization results in undefined behavior".
        </p>
      </Today>
      <p>
        The tests are lab X1.2's part B at smaller sizes, with rows longer than one chunk of 1,024 on every backend on
        purpose; on a GPU every batch runs ten times on fresh outputs. There is no benchmark.
      </p>
      <RaceStatus lab={l4} />
      <LabResults
        lab={l4}
        noBench="This lab has no benchmark. The result is the line above: every test passing on the emulator, the race backend and a device. As shipped, the kernel is meant to fail."
      />
      <ReferenceSolution lab={l4} tiers={TIERS_2} />
    </LabsPage>
  );
}
