import type { Question } from '../../components/Ask';
import { Ask } from '../../components/Ask';
import { Beyond, Source, Today, type Tier } from '../../components/notes';
import {
  BackendsAndEmulator,
  LabFiles,
  LabPrediction,
  LabResults,
  LabsIntro,
  LabsPage,
  ReferenceSolution,
  RunningLabs,
  VerificationStatus,
  benchRow,
  defineLabs,
  measure,
  msCol,
  numCol,
  preferredBench,
  rawCol,
  useLabs,
  type BenchSet,
  type Col,
  type Lab,
} from '../../components/LabKit';
import { benchRows, fmt, labStatus, num, type BenchRow, type Reports, type RunResult } from '../../lib/reports';
import { href } from '../../lib/router';

// The page is built from components/LabKit.tsx; what is here is chapter 20's own content.
// Predictions are stored, not graded; the one question is about lab 20.3's first defect.
export const questionsLabs: Record<string, Question> = {
  'ch20/labs/q-detector': {
    kind: 'choice',
    id: 'ch20/labs/q-detector',
    prompt: (
      <>
        Lab 20.3, defect 1: the <code>__syncwarp()</code> after the P write in <code>compute_P_and_update_D</code> is
        gone (on Metal, the <code>simdgroup_barrier</code>). Before you run anything: which backend do you expect to
        report it?
      </>
    ),
    options: [
      {
        label: 'emu: its fixed thread orders make every missing barrier fail.',
        why: 'emu switches threads only at synchronisation points, and a shuffle is one where the whole warp meets. Every lane has written its P before any lane leaves the sum butterfly, so the reads in compute_O see the writes.',
      },
      {
        label: 'race: ThreadSanitizer names the P write and the read.',
        correct: true,
      },
      {
        label: 'Your GPU: lanes that drift apart read stale P values from shared memory.',
        why: 'Nothing promises the lanes stay together, but nothing makes them run apart either. When the lab was written, Metal passed every test with this defect alone.',
      },
      {
        label: 'All three: a missing barrier is wrong on every backend.',
        why: 'Wrong, yes; visible, no. Only a tool that treats shuffles as not ordering memory, as the CUDA documentation says they do not, sees it every time.',
      },
    ],
    explain: (
      <p>
        Shuffles exchange registers; they do not order memory (ch20-facts A3). On <code>race</code> the threads run at
        once and only <code>__syncwarp()</code> orders their memory, so ThreadSanitizer reports the pair. On{' '}
        <code>emu</code> the shuffle rendezvous happens to line the lanes up, and on a GPU the hardware may keep a
        warp&rsquo;s lanes together whether or not the code asks. That is why the book&rsquo;s listing, which has no such
        barrier, can pass every test you give it. Section <a href={href('ch20', '20.5')}>20.5</a>&rsquo;s kernel stepper
        shows the schedule that breaks it.
      </p>
    ),
    source: 'Figs. 20.13 and 20.15, pp. 501–503; notes/today/ch20-facts.md A3; labs/ch20/lab3_flash_hunt/README.md',
  },
};

export const LABS = defineLabs('ch20', [
  { num: '20.1', dir: 'lab1_causal_softmax', name: 'causal softmax (M6, part one)' },
  { num: '20.2', dir: 'lab2_flash', name: 'flash attention forward (M6, part two)' },
  { num: '20.3', dir: 'lab3_flash_hunt', name: 'flash race hunt' },
  { num: '20.4', dir: 'lab4_decode', name: 'KV append and decode (M7)' },
  { num: '20.5', dir: 'lab5_gqa_decode', name: 'grouped-query decode' },
]);
const [l1, l2, l3, l4, l5] = LABS;

// Identifier colours, by the memory each lives in.
const TIERS_SOFTMAX: Record<string, Tier> = { S: 'global', P: 'global', L: 'global', scratch: 'shared', m: 'reg', s: 'reg' };
const TIERS_FLASH: Record<string, Tier> = {
  Q: 'global', K: 'global', V: 'global', O: 'global', L: 'global',
  KT_j: 'shared', S_i: 'shared', V_j: 'shared',
  Q_i: 'reg', O_i: 'reg', m_i: 'reg', D_i: 'reg', last_m: 'reg', S_ij: 'reg', O_ij: 'reg', P_ij: 'reg',
};
const TIERS_DECODE: Record<string, Tier> = {
  q: 'global', Kc: 'global', Vc: 'global', out: 'global', k_new: 'global', v_new: 'global',
  m_s: 'shared', D_s: 'shared', O_s: 'shared', qr: 'reg',
};

// ---------------------------------------------------------------- reading the benchmark rows
// Lab 20.1 records one row per N, 'unfused_attention', with s_seconds, softmax_seconds,
// o_seconds, total_seconds, softmax_share, softmax_gb_per_s and score_bytes (S and P, all heads).
// Lab 20.2 records 'attention' per n and b_c: flash_seconds, unfused_seconds, flash_over_unfused,
// flash_gflops, unfused_gflops, unfused_score_bytes. Lab 20.4 records 'decode' (batch, len, ms,
// gb_per_s, cache_bytes) and 'append' (batch, us, bytes_moved). Lab 20.5 records 'perhead' and
// 'grouped' per (batch, h_kv) with g_q, blocks, ms, gb_per_s and, on 'grouped', speedup_vs_perhead.
const mib = (v?: number) => (v == null ? '' : `${fmt(v / 2 ** 20)} MiB`);
const sec = (key: string, head: string): Col => ({
  head,
  num: true,
  cell: (r) => {
    const v = num(r[key]);
    return v == null ? '' : fmt(v * 1000);
  },
});

/** Lab 5.1's measured bandwidth on one backend, from the reference solution or the reader's own run. */
function bandwidth(reports: Reports, backend: string): number | undefined {
  const run = reports[backend];
  const rows = [...benchRows(run, 'ch05/lab1_roofline', 'work'), ...benchRows(run, 'ch05/lab1_roofline', 'solution')];
  return num(rows.find((r) => r.name === 'roofline')?.bandwidth_gb_per_s);
}

/** Lab 20.2's row at n, the book's B_c = 32 where a build has it, otherwise whatever B_c it ran. */
const flashRow = (s: BenchSet, n: number) => benchRow(s, 'attention', { n, b_c: 32 }) ?? benchRow(s, 'attention', { n });

const shareAt4096 = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const r = benchRow(s, 'unfused_attention', { n: 4096 });
    const share = num(r?.softmax_share);
    const sm = num(r?.softmax_seconds);
    const tot = num(r?.total_seconds);
    if (share == null || sm == null || tot == null) return null;
    return { value: 100 * share, detail: `softmax ${fmt(sm * 1000)} ms of ${fmt(tot * 1000)} ms in all, N = 4096, 8 heads` };
  });

const flashRatio4096 = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const r = flashRow(s, 4096);
    const f = num(r?.flash_seconds);
    const u = num(r?.unfused_seconds);
    if (!r || f == null || u == null) return null;
    return { value: f / u, detail: `flash ${fmt(f * 1000)} ms against unfused ${fmt(u * 1000)} ms, n = 4096, B_c = ${String(r.b_c)}` };
  });

const decodeMs = (reports: Reports) =>
  measure(reports, l4, (s) => {
    const r = benchRow(s, 'decode', { batch: 1, len: 8192 });
    const ms = num(r?.ms);
    const gb = num(r?.gb_per_s);
    if (ms == null || gb == null) return null;
    const bw = bandwidth(reports, s.backend);
    return {
      value: ms,
      detail: `${fmt(gb)} GB/s on the cache bytes${bw ? `, against lab 5.1's ${fmt(bw)} GB/s, ${fmt((100 * gb) / bw)}% of it` : '; no lab 5.1 bandwidth on this backend yet'}`,
    };
  });

const decodeBatch8 = (reports: Reports) =>
  measure(reports, l4, (s) => {
    const one = num(benchRow(s, 'decode', { batch: 1, len: 8192 })?.ms);
    const eight = num(benchRow(s, 'decode', { batch: 8, len: 8192 })?.ms);
    if (one == null || eight == null) return null;
    return { value: eight / one, detail: `${fmt(eight)} ms at batch 8 against ${fmt(one)} ms at batch 1, len 8192` };
  });

const groupedSpeedup = (reports: Reports) =>
  measure(reports, l5, (s) => {
    const g = benchRow(s, 'grouped', { batch: 1, h_kv: 8 });
    const p = benchRow(s, 'perhead', { batch: 1, h_kv: 8 });
    const gm = num(g?.ms);
    const pm = num(p?.ms);
    if (gm == null || pm == null) return null;
    return {
      value: pm / gm,
      detail: `per-head ${fmt(pm)} ms over ${String(p?.blocks)} blocks, grouped ${fmt(gm)} ms over ${String(g?.blocks)} blocks, len 8192`,
    };
  });

// ---------------------------------------------------------------- results columns
const softmaxCols: Col[] = [
  rawCol('N', 'n'),
  sec('s_seconds', 'S = Q Kᵀ, ms'),
  sec('softmax_seconds', 'Softmax, ms'),
  sec('o_seconds', 'O = P V, ms'),
  sec('total_seconds', 'Total, ms'),
  { head: 'Softmax share', num: true, cell: (r) => (num(r.softmax_share) == null ? '' : `${fmt(100 * num(r.softmax_share)!)}%`) },
  numCol('Softmax GB/s', 'softmax_gb_per_s'),
  { head: 'S and P', num: true, cell: (r) => mib(num(r.score_bytes)) },
];
const flashCols: Col[] = [
  rawCol('n', 'n'),
  rawCol('B_c', 'b_c'),
  sec('flash_seconds', 'Flash, ms'),
  sec('unfused_seconds', 'Unfused, ms'),
  numCol('Flash / unfused', 'flash_over_unfused'),
  numCol('Flash GFLOP/s', 'flash_gflops'),
  numCol('Unfused GFLOP/s', 'unfused_gflops'),
  { head: 'Unfused S and P', num: true, cell: (r) => mib(num(r.unfused_score_bytes)) },
];
const decodeCols: Col[] = [
  { head: 'Measured', cell: (r) => (r.name === 'append' ? 'append alone' : 'decode attention') },
  rawCol('Batch', 'batch'),
  rawCol('len', 'len'),
  msCol,
  numCol('GB/s', 'gb_per_s'),
  { head: 'Bytes', num: true, cell: (r) => mib(num(r.cache_bytes) ?? num(r.bytes_moved)) },
];
const groupedCols: Col[] = [
  { head: 'Kernel measured', cell: (r) => (r.name === 'perhead' ? 'per-head (lab 20.4)' : r.name === 'grouped' ? 'grouped (this lab)' : r.name) },
  rawCol('Batch', 'batch'),
  rawCol('h_kv', 'h_kv'),
  rawCol('g_q', 'g_q'),
  rawCol('Blocks', 'blocks'),
  msCol,
  numCol('GB/s, unique bytes', 'gb_per_s'),
  numCol('× per-head', 'speedup_vs_perhead'),
];

// ---------------------------------------------------------------- the race backend
type RaceResult = RunResult & { tsan?: unknown[] };
/** Every lab of the page on the race backend, which the status table does not show. */
function RaceTable() {
  const { reports, loading } = useLabs();
  const run = reports.race;
  const line = (l: Lab, variant: 'solution' | 'work') => {
    const rs = (run?.results.filter((x) => x.lab === l.id && x.variant === variant) ?? []) as RaceResult[];
    const tests = rs.flatMap((x) => x.tests);
    const races = rs.reduce((n, x) => n + (Array.isArray(x.tsan) ? x.tsan.length : 0), 0);
    const s = labStatus(run, l.id, variant);
    const text =
      s === 'none'
        ? variant === 'work'
          ? 'not started'
          : 'no run on this machine'
        : `${s === 'pass' ? 'passes' : 'not passing'}: ${tests.filter((t) => t.pass).length} of ${tests.length} tests, ${races ? `${races} ThreadSanitizer report${races === 1 ? '' : 's'}` : 'no race reported'}`;
    return { s, text };
  };
  if (loading) return <p className="muted">Reading reports.</p>;
  if (!run) return <p data-race="none">No run on the race backend on this machine yet.</p>;
  return (
    <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
      <table className="data" data-table="race">
        <thead>
          <tr>
            <th>On the race backend</th>
            <th>Reference solution</th>
            <th>Your work</th>
          </tr>
        </thead>
        <tbody>
          {LABS.map((l) => {
            const sol = line(l, 'solution');
            const mine = line(l, 'work');
            return (
              <tr key={l.id} data-race-lab={l.num} data-race-solution={sol.s} data-race-work={mine.s}>
                <td>
                  Lab {l.num}, {l.name}
                </td>
                <td>{sol.s === 'pass' ? `emulator-checked; ${sol.text}` : sol.text}</td>
                <td>{mine.text}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------- lab 20.5 beside Eq. 20.13
/** Measured grouped/per-head ratios per g_q and batch, beside the intensity ratio Eq. 20.13 gives. */
function GroupedVsIntensity() {
  const { reports, loading } = useLabs();
  const sets = preferredBench(reports, l5);
  const rows = sets.flatMap((s) =>
    s.rows
      .filter((r): r is BenchRow => r.name === 'grouped' && num(r.speedup_vs_perhead) != null)
      .map((r) => ({ s, r })),
  );
  if (loading) return <p className="muted">Reading reports.</p>;
  if (!rows.length)
    return (
      <p className="muted" data-gqa="none">
        No lab 20.5 benchmark on this machine yet. Run <code>python3 labs/pmpp.py run 20.5 --bench</code> (add{' '}
        <code>--solution</code> for the reference), then reload.
      </p>
    );
  return (
    <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
      <table className="data" data-table="gqa-intensity">
        <thead>
          <tr>
            <th>Backend</th>
            <th className="num">Batch</th>
            <th className="num">g_q</th>
            <th className="num">Blocks, per-head : grouped</th>
            <th className="num">Bytes requested, per-head : grouped</th>
            <th className="num">Measured speed-up, grouped over per-head</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ s, r }, i) => {
            const gq = num(r.g_q) ?? 1;
            const ph = benchRow(s, 'perhead', { batch: num(r.batch)!, h_kv: num(r.h_kv)! });
            return (
              <tr key={s.backend + i} data-gqa-row={`${String(r.batch)}-${gq}`}>
                <td>
                  {s.backend} <span className="muted">({s.device}, {s.variant === 'work' ? 'yours' : 'reference solution'})</span>
                </td>
                <td className="num">{String(r.batch)}</td>
                <td className="num">{gq}</td>
                <td className="num">
                  {String(ph?.blocks ?? '?')} : {String(r.blocks)}
                </td>
                <td className="num">{gq} : 1</td>
                <td className="num">{fmt(num(r.speedup_vs_perhead)!)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------- milestones M6 and M7
function MilestoneStatus() {
  const { reports, loading } = useLabs();
  const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
  const name = (b: string) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
  const passes = (l: Lab) => backends.filter((b) => labStatus(reports[b], l.id, 'work') === 'pass');
  const m6 = passes(l1).filter((b) => passes(l2).includes(b));
  const m7 = passes(l4);
  const gqa = passes(l5);
  const say = (bs: string[]) => bs.map(name).join(' and ');
  return (
    <p data-milestone={loading ? 'loading' : `m6:${m6.join(',') || 'none'};m7:${m7.join(',') || 'none'}`}>
      {loading ? (
        'Reading reports.'
      ) : !backends.length ? (
        'No lab results are recorded on this machine yet.'
      ) : (
        <>
          {m6.length
            ? `M6: your causal softmax and your flash kernel both pass on ${say(m6)} on this machine.`
            : 'M6: your labs 20.1 and 20.2 do not both pass on any backend of this machine yet.'}{' '}
          {m7.length
            ? `M7: your append and decode pass on ${say(m7)}${gqa.length ? `, and your grouped kernel on ${say(gqa)}` : ''}.`
            : 'M7: your lab 20.4 does not pass on any backend of this machine yet.'}
        </>
      )}
    </p>
  );
}

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS}>
      <LabsIntro>
        Five labs, each in its own directory under <code>labs/ch20/</code>. Labs 20.1 and 20.2 are milestone M6, causal
        attention for a prefill: first the unfused pipeline&rsquo;s middle stage, then the whole thing as one kernel. Lab
        20.3 hides three defects in that kernel, one of them the barrier the book&rsquo;s listing leaves out. Labs 20.4
        and 20.5 are milestone M7, one decode step with a KV cache: append and attend, then the same for grouped-query
        heads. Code you do not edit sits in <code>labs/ch20/shared-g11/</code> (the flash tests and the unfused
        pipeline), <code>labs/ch20/shared-g12/</code> (the decode tests) and lab 20.5&rsquo;s <code>provided/</code>{' '}
        folder.
      </LabsIntro>
      <RunningLabs
        example="20.4"
        alsoShownIn={
          <>
            sections <a href={href('ch20', '20.3')}>20.3</a>, <a href={href('ch20', '20.5')}>20.5</a> and{' '}
            <a href={href('ch20', '20.6')}>20.6</a>
          </>
        }
      />
      <BackendsAndEmulator />
      <p>
        Every lab here is also built for the emulator&rsquo;s second flavour, <code>race</code>, which runs one
        block&rsquo;s threads at the same time under ThreadSanitizer and names the two source lines of any
        unsynchronised pair of accesses. Ask for it with <code>--backend race</code>. Each lab has a reason to: two
        reductions sharing a scratch slot in 20.1, the warp and tile barriers of the flash kernel in 20.2 and 20.3, and
        the warps&rsquo; merge through shared memory in 20.4 and 20.5. Its sizes are small, because a 512-thread block
        under ThreadSanitizer is slow. The status table below shows the first flavour; this one shows the second.
      </p>
      <RaceTable />
      <p>
        One gap matters in this chapter more than in most: neither emulator models a warp running in lockstep, and{' '}
        <code>emu</code> treats a shuffle as a point where the whole warp meets. A missing warp barrier therefore passes
        on <code>emu</code> and is seen only by <code>race</code>. Lab 20.3 is built on that difference.
      </p>
      <VerificationStatus />

      {/* ------------------------------------------------------------ 20.1 */}
      <h2>Lab 20.1: causal mask and softmax (milestone M6, part one)</h2>
      <p>
        You write the middle stage of unfused attention: <code>launchCausalSoftmax(S, P, L, batch, nq, nk, scale)</code>{' '}
        turns scores <code>[batch, nq, nk]</code> into <code>P = softmax(scale · S + M)</code> and writes the logsumexp{' '}
        <code>L = m + log(sum)</code> of every row into <code>[batch, nq]</code>. The design is Fig. 20.4&rsquo;s: one
        block of 256 threads per row, strided columns, the mask as the loop bound so a masked score is never read, a max
        reduction, a barrier, a sum reduction, then the write of P. The two block reductions are lab 10.3&rsquo;s,
        milestone M4, provided and credited in the file; you write the masked passes, the barrier between the
        reductions, L and the launch.
      </p>
      <Source section="20.3" pages="486–488" figs="Fig. 20.4" />
      <p>
        Where the lab differs from the page: <code>gridDim.y</code> is 1, not 0, and here it walks the batch; the max
        starts at <code>-FLT_MAX</code>, since no CUDA header defines <code>NEG_INFINITY</code>; M4&rsquo;s reductions
        replace CUB, whose <code>cub::Max()</code> and <code>cub::Sum()</code> are gone; and L replaces the
        denominator D, which cannot rebuild P without the row max. Two choices are ours: the kernel applies the scale
        itself, and the mask is aligned to the lower right, so row r sees columns up to r + (nk − nq). With nq = nk that
        is the book&rsquo;s mask; with nq = 1 it is a decode step, which sees every key.
      </p>
      <Today checked="2026-11-20 (notes/today/ch20-facts.md A1, A5, A7, A8, D4, D5; infra-facts A11)">
        <p>
          FlashAttention since 2.1 and MLX align a short query&rsquo;s mask to the lower right; PyTorch&rsquo;s{' '}
          <code>is_causal</code> aligns it to the upper left.
        </p>
      </Today>
      <LabFiles
        lab={l1}
        files={[
          ['kernel.cu', <><code>causalSoftmaxKernel</code> and <code>launchCausalSoftmax</code>; the M4 reductions at the top are given</>],
          ['kernel.metal', <><code>causal_softmax</code>, with <code>simd_max</code>, <code>simd_sum</code> and threadgroup memory</>],
          ['launch.mm', <>the dispatch in <code>encodeCausalSoftmax</code>; the checks and bindings are given</>],
        ]}
      />
      <p>
        Run <code>python3 labs/pmpp.py run 20.1 --backend emu</code>, then <code>--backend race</code>, then the GPU.
        The shapes run from one column to 2,048 square and 4,097 columns in one row, with nq &lt; nk in one case, at batch
        1 and 3. Each row gets one recipe, mixed within a batch: uniform scores, the same offset by +90 and by −90 (exp(90)
        does not fit in a float, so a kernel that skips the max subtraction fails), the largest score at the last
        visible column, 10<sup>30</sup> in every masked column (a kernel that reads one gets zeros or NaN), and all
        scores equal. P must sum to 1 per row within a tolerance the README derives, masked P must be exactly 0, L must
        match a double reference, and canaries after P and L must survive. The README also says what <code>race</code>{' '}
        is here for: without the barrier between the reductions, warp 0 can overwrite <code>scratch[0]</code> before a
        slow warp has read the max.
      </p>
      <p>
        With <code>--bench</code> the driver runs the whole unfused pipeline at 8 heads, d = 128 and N = 1024, 2048 and
        4096: S = Q Kᵀ with your project&rsquo;s M3 (lab 15.6&rsquo;s reference), your softmax, then O = P V with M2 (lab
        5.3&rsquo;s reference). It times each stage and counts the softmax&rsquo;s GB/s on its minimum traffic.
      </p>
      <LabPrediction
        id="ch20/labs/p-lab1-share"
        prompt={
          <>
            At N = 4096, d = 128, what percentage of the three-stage pipeline&rsquo;s time will your mask-and-softmax
            kernel take? The S product does 2N²d FLOPs, masked half included. The softmax moves about 10N² bytes: the
            visible half of S read three times, P written once. Divide each by the rate that limits it, your compute
            ceiling and your bandwidth from <a href={href('ch05', 'labs')}>lab 5.1</a>.
          </>
        }
        unit="%"
        measure={shareAt4096}
        reading={
          <>
            If both kernels ran at their ceilings, a device whose ridge point is R FLOP/B would spend 2d / (10R) times as
            long in the product as in the softmax. Then ask how far M3 and M2 are from the compute ceiling, and your
            softmax from the bandwidth one. Section <a href={href('ch20', '20.3')}>20.3</a> counts the traffic; section{' '}
            <a href={href('ch20', '20.5')}>20.5</a> removes it.
          </>
        }
      />
      <LabResults lab={l1} filter={(r) => r.name === 'unfused_attention'} cols={softmaxCols} />
      <ReferenceSolution lab={l1} tiers={TIERS_SOFTMAX} />

      {/* ------------------------------------------------------------ 20.2 */}
      <h2>Lab 20.2: flash attention forward (milestone M6, part two)</h2>
      <p>
        You write causal attention as one kernel, so that the n × n scores of lab 20.1 never exist in device memory:{' '}
        <code>launchFlashAttention(Q, K, V, O, L, batch, heads, n, scale)</code> on <code>[batch, heads, n, 128]</code>,
        any n, writing the same L as lab 20.1. The design is the book&rsquo;s: a block of 512 threads, 16 warps of two
        rows each, owns a panel of 32 rows of Q; K, V and the S/P tile pass through shared memory one tile of B_c keys
        at a time; each thread keeps its slice of Q and O and the running max and denominator in registers. Grid y walks
        the (sequence, head) pairs.
      </p>
      <Source section="20.5" pages="492–504" figs="Figs. 20.8 to 20.16; Exercises 1 and 3, p. 511" />
      <p>
        Provided and already corrected: the constants, the lane-mask type and the 32-lane logical warp,{' '}
        <code>load_Q</code>, <code>load_KT_and_V</code> and <code>store_O</code>. Yours: <code>initialize</code>{' '}
        (Exercise 3), <code>compute_S_and_max</code>, <code>update_m_and_D</code>, <code>compute_P_and_update_D</code>{' '}
        with the warp barrier the page leaves out, <code>compute_O</code>, the kernel&rsquo;s two loops with both block
        barriers, and the launch (Exercise 1). Section <a href={href('ch20', '20.5')}>20.5</a> lists every place the
        lab departs from the page and why.
      </p>
      <p>
        On Metal, <code>launch.mm</code> is yours too. The book&rsquo;s tiles at B_c = 32 take 37,376 bytes, more than the
        32 KB of threadgroup memory Apple documents, so pipeline creation fails. Take the largest B_c in 32, 16, 8 whose
        tiles fit <code>maxThreadgroupMemoryLength</code>, compile the kernel with it through{' '}
        <code>preprocessorMacros</code>, and check that the pipeline runs SIMD-groups of 32 and 512 threads per
        threadgroup before dispatching.
      </p>
      <Today checked="2026-11-20 (notes/today/ch20-facts.md C1, C6; ch05-facts C3; ch03-facts C9)">
        <p>
          The 32 KB failure was observed on one Mac only. Apple documents no fixed SIMD-group width: read{' '}
          <code>threadExecutionWidth</code> from the pipeline, which Apple says returns the SIMD-group size.
        </p>
      </Today>
      <LabFiles
        lab={l2}
        files={[
          ['kernel.cu', <>the five device functions, <code>flashAttentionKernel</code>&rsquo;s loops and barriers, and <code>launchFlashAttention</code></>],
          ['kernel.metal', <>the same with <code>simd_broadcast</code>, <code>simd_max</code>, <code>simd_sum</code> and <code>simdgroup_barrier</code></>],
          ['launch.mm', <>the choice of B_c, the compile with it, the pipeline checks and the dispatch</>],
        ]}
      />
      <p>
        Run <code>python3 labs/pmpp.py run 20.2 --backend emu</code> first; it is slow, four cases under three thread
        orders. Then <code>--backend race</code>, then the GPU. n runs over 1, 2, 31, 32, 33, 63, 64, 65, 100, 257, 1000
        and 2048 with (batch, heads) = (1, 1) and (2, 3). Each head gets one recipe: uniform data; a late max, where the
        score grows along every row so the running max rises in every tile and every rescale matters; scores offset by
        about +90; and V with a distinct level per column. O and L are checked row by row against a double CPU reference
        with a tolerance the README derives from the exp and summation bounds, canaries after O and L must survive, and
        the driver checks that no n × n buffer was allocated.
      </p>
      <p>
        With <code>--bench</code>, your kernel runs against lab 20.1&rsquo;s unfused pipeline (M3, the reference causal
        softmax, M2) at 8 heads, d = 128, n = 512 to 4096. GFLOP/s counts 2n²d per head, half of the two full products,
        because half of S is masked; the unfused path computes the masked half anyway. On CUDA and HIP a B_c = 16 build
        runs beside the default 32.
      </p>
      <Beyond>
        <p>
          FlashAttention-2 runs in fp16 and bf16 only, cuDNN&rsquo;s SDPA lists no fp32 at all, and llama.cpp&rsquo;s Metal
          kernel does its two products with SIMD-group matrix instructions (facts A6, D2). This lab keeps the
          book&rsquo;s fp32 warp loops so that you can see every exchange.
        </p>
      </Beyond>
      <LabPrediction
        id="ch20/labs/p-lab2-ratio"
        prompt={
          <>
            At n = 4096, your flash time divided by the unfused pipeline&rsquo;s time. Below 1 means flash is faster.
            Weigh two forces. Flash never writes S and P: per head the unfused path writes and rereads at least 2n² × 4
            bytes, 128 MiB at n = 4096. But flash&rsquo;s two products are the book&rsquo;s simple warp loops, one shuffle
            per multiply-add for the scores, while the unfused path runs your tuned M3 and M2.
          </>
        }
        unit="× the unfused time"
        measure={flashRatio4096}
        reading={
          <>
            The book gives no timing, and nothing here says which force wins on your device. Footnote 8 (pp. 500–501)
            says the two products could use chapter 15&rsquo;s optimisations; production kernels run them on matrix
            hardware in 16-bit or 8-bit types, which this lab does not.
          </>
        }
      />
      <LabResults lab={l2} filter={(r) => r.name === 'attention'} cols={flashCols} />
      <ReferenceSolution lab={l2} tiers={TIERS_FLASH} />

      {/* ------------------------------------------------------------ 20.3 */}
      <h2>Lab 20.3: flash attention race hunt</h2>
      <p>
        <code>work/kernel.cu</code> and <code>work/kernel.metal</code> hold lab 20.2&rsquo;s reference kernel with three
        independent defects, the same three in both files. It compiles. You are told what kind each one is, not where it
        is: a missing barrier between the lanes of one warp, a missing barrier across the block, and a wrong value in the
        online softmax&rsquo;s bookkeeping. Find and fix all three, and record which backend showed you which.
      </p>
      <Source section="20.5" pages="492–504" figs="Figs. 20.9, 20.13 to 20.15" />
      <LabFiles
        lab={l3}
        files={[
          ['kernel.cu', <>lab 20.2&rsquo;s kernel with the three defects</>],
          ['kernel.metal', <>the same three, in Metal</>],
          ['launch.mm', <>lab 20.2&rsquo;s reference host side; it has no defect</>],
        ]}
      />
      <p>Before you run anything, predict. The README has a table for all three defects; this is the first.</p>
      <Ask q={questionsLabs['ch20/labs/q-detector']} label="Predict, then check" />
      <p>The method is lab 5.4&rsquo;s:</p>
      <ol>
        <li>
          <code>python3 labs/pmpp.py run 20.3 --backend emu</code>. Its thread orders are fixed, so a defect it can see
          fails every time. It is slow on this kernel: start it and read the code while it runs.
        </li>
        <li>Fix one thing and run again; the failures change as you go. The late-max data shows a rescale problem first.</li>
        <li>
          <code>python3 labs/pmpp.py run 20.3 --backend race</code>. ThreadSanitizer names the two lines that collide. An
          inlined function may show as <code>flashAttentionKernel</code> with no line number; the other side still names
          its line. Each pair is reported once per run, so a clean report is not a proof.
        </li>
        <li>Then your GPU. Note which defects it hid.</li>
      </ol>
      <p>
        The tests are lab 20.2&rsquo;s. When the lab was written each defect was checked alone: the warp barrier failed
        only on <code>race</code>; the block barrier failed on <code>emu</code> in every thread order, was reported by{' '}
        <code>race</code> and failed on Metal; the bookkeeping error failed everywhere.
      </p>
      <LabResults
        lab={l3}
        noBench="This lab has no benchmark: lab 20.2 times the same kernel. The result is the line above, every test passing on the emulators and on a device. As shipped, the kernel is meant to fail."
      />
      <ReferenceSolution lab={l3} tiers={TIERS_FLASH} />

      <h3>The project: milestone M6</h3>
      <p>
        M6 is causal attention for a prefill in the project&rsquo;s conventions: row-major <code>float</code> tensors,
        batch first, raw device pointers and explicit sizes, and a <code>launch</code> function beside each kernel. Two
        forms, both with the same L: the unfused pipeline (M3, <code>launchCausalSoftmax</code>, M2) and{' '}
        <code>launchFlashAttention</code>. Which one to call at a given n is your benchmark&rsquo;s answer.
      </p>

      {/* ------------------------------------------------------------ 20.4 */}
      <h2>Lab 20.4: KV cache append and single-token decode (milestone M7)</h2>
      <p>
        One decode step of one attention layer, for a batch of sequences with different lengths. Write two host
        functions and their kernels: <code>launchKvAppend</code> writes each sequence&rsquo;s new key and value at row{' '}
        <code>pos[b]</code> of caches laid out <code>[batch, max_seq, h_kv, 128]</code> and touches nothing else, and{' '}
        <code>launchDecodeAttention</code> lets each query head attend over the first <code>len[b]</code> cached rows of
        its K/V head, <code>h / (h_q / h_kv)</code>, so one kernel serves MHA, GQA and MQA. A single new query needs no
        mask.
      </p>
      <Source section="20.4 to 20.6" pages="488–492, 493–504, 506–507" figs="Figs. 20.5 to 20.7; Eqs. 20.3 to 20.6 and 20.9" />
      <p>
        The book shows no code for a cache or a decode kernel (p. 492); the design is ours. One block of 256 threads per
        (head, sequence). Each of the 8 warps takes every eighth key; the 32 lanes split the head dimension, 4 values
        each, and a butterfly sum gives every lane the score. Each warp keeps its own running max, denominator and
        unnormalised output, updated one key at a time as section 20.5&rsquo;s kernel does one tile at a time. At the
        end the 8 warps merge through shared memory with Eqs. 20.4 and 20.6, the split-K merge that FlashAttention-1
        used across warps and Flash-Decoding uses across blocks.
      </p>
      <Today checked="2026-11-20 (notes/today/ch20-facts.md A7, A8, D3, D4)">
        <p>
          The cache layout is Meta&rsquo;s reference Llama code&rsquo;s; the append is a scatter by index, as
          llama.cpp&rsquo;s <code>ggml_set_rows</code>. MLX&rsquo;s Metal decode kernel, <code>sdpa_vector</code>, has the
          same shape as this design.
        </p>
      </Today>
      <LabFiles
        lab={l4}
        files={[
          ['kernel.cu', <><code>warpSumAll</code>, <code>kvAppendKernel</code>, <code>decodeAttentionKernel</code> and the two launchers</>],
          ['kernel.metal', <><code>kv_append</code> and <code>decode_attention</code></>],
          ['launch.mm', <>the two dispatches; the bindings are given</>],
        ]}
      />
      <p>
        The append is checked bit for bit: the new rows land at <code>pos[b]</code>, every other element of both caches
        keeps its canary. Decode runs (h_q, h_kv) = (8, 8), (8, 2), (8, 1) and (6, 2), one to four sequences per call with
        lengths up to 4,097 on a GPU (smaller on the emulators), on random keys and on a late max that only one warp
        sees, so the merge must rescale the others. Rows at and past <code>len[b]</code> hold NaN. Then the milestone
        loop: 16 autoregressive steps where each step&rsquo;s output must equal the last row of a full causal attention
        recomputed from scratch over every token so far. That is section <a href={href('ch20', '20.4')}>20.4</a>&rsquo;s
        claim, tested.
      </p>
      <p>
        With <code>--bench</code>, decode attention alone at h_q = h_kv = 32, head dimension 128 (a 7B-class MHA layer,
        our choice), len 512 to 32,768, batch 1 and 8. GB/s counts the cache bytes one call must read, 2 × len × h_kv ×
        128 × 4 per sequence. Sizes that do not fit in half the device memory are skipped and listed. The append is timed
        separately.
      </p>
      <LabPrediction
        id="ch20/labs/p-lab4-ms"
        prompt={
          <>
            Using the bandwidth you measured in <a href={href('ch05', 'labs')}>lab 5.1</a>, the time in milliseconds of
            one decode call at len 8192, batch 1, 32 heads of 128 in fp32. Each 4-byte element of K and V is used in one
            multiply-add: about 0.5 FLOP/B, far below any ridge point, with 256 MiB of cache to read.
          </>
        }
        unit="ms"
        measure={decodeMs}
        reading={
          <>
            Eq. 20.9 gets &ldquo;about 1&rdquo; by counting one operation per one-byte element; the argument is the same.
            How close the measured GB/s comes to lab 5.1&rsquo;s is the fraction of your bandwidth this kernel reaches.
            Section <a href={href('ch20', '20.6')}>20.6</a> sets the same rows beside the model.
          </>
        }
      />
      <LabPrediction
        id="ch20/labs/p-lab4-batch8"
        prompt={
          <>
            At the same length, batch 8 as a multiple of batch 1&rsquo;s time. Each sequence has its own cache, so batch
            8 reads 8 times the bytes; but batch 1 launches 32 blocks and batch 8 launches 256.
          </>
        }
        unit="× batch 1"
        measure={decodeBatch8}
        reading={
          <>
            Fig. 20.17 says attention does not benefit from batching: the bytes per FLOP do not change. A ratio well below
            8 says batch 1 was not keeping enough requests in flight to reach your bandwidth, the point Flash-Decoding
            makes about small batches (facts A8).
          </>
        }
      />
      <LabResults lab={l4} filter={(r) => r.name === 'decode' || r.name === 'append'} cols={decodeCols} />
      <ReferenceSolution lab={l4} tiers={TIERS_DECODE} />

      {/* ------------------------------------------------------------ 20.5 */}
      <h2>Lab 20.5: grouped-query decode, loading each K/V row once</h2>
      <p>
        <b>Finish lab 20.4 first.</b> This lab&rsquo;s starter begins from lab 20.4&rsquo;s reference per-head kernel, so
        opening it gives away that lab&rsquo;s answer.
      </p>
      <p>
        In lab 20.4&rsquo;s kernel the g_q query heads that share a K/V head each get their own block, and each block reads
        the same K and V rows: the kernel asks memory for g_q times the bytes the cache holds. Write{' '}
        <code>launchDecodeAttentionGrouped</code>, with lab 20.4&rsquo;s arguments and contract, so that one block per
        (K/V head, sequence) loads each row once and uses it for all g_q queries, as llama.cpp&rsquo;s{' '}
        <code>gqa_ratio</code> path does. Each warp keeps g_q running states; the merge runs once per query head through
        the same shared buffer. The group size is a compile-time constant: a template on <code>GQ</code> in CUDA, a
        preprocessor macro the driver sets when it compiles the Metal library.
      </p>
      <Source section="20.7" pages="508–509" figs="Eqs. 20.12 and 20.13" />
      <p>
        Here g_q is the number of query heads per K/V head, the meaning Eqs. 20.12 and 20.13 need; the book&rsquo;s prose
        calls it the number of groups. Section <a href={href('ch20', '20.7')}>20.7</a> straightens this out. Eq.
        20.13&rsquo;s &ldquo;≈ g_q&rdquo; counts one-byte elements; in fp32 the kernel&rsquo;s intensity is about g_q / 2
        FLOP/B.
      </p>
      <LabFiles
        lab={l5}
        files={[
          ['kernel.cu', <><code>decodeAttentionGroupedKernel&lt;GQ&gt;</code>, starting from lab 20.4&rsquo;s per-head kernel, and <code>launchDecodeAttentionGrouped</code></>],
          ['kernel.metal', <><code>decode_attention_grouped</code>, the same starting point</>],
          ['launch.mm', <>the group-size check and the dispatch</>],
        ]}
      />
      <p>
        The tests are lab 20.4&rsquo;s with g_q = 1, 2, 3, 4 and 8, then the 16-step loop with lab 20.4&rsquo;s append and
        your kernel. They check the answers, not the grouping: a per-head kernel launched over (h_q, batch) passes them.
        Only the benchmark shows whether each row is read once. It times lab 20.4&rsquo;s reference kernel and yours at
        h_q = 32, len 8192, batch 1 and 4, and h_kv = 32, 8 and 4, with GB/s on the unique cache bytes for both.
      </p>
      <LabPrediction
        id="ch20/labs/p-lab5-speedup"
        prompt={
          <>
            At h_kv = 8 (g_q = 4), batch 1: how many times faster will your grouped kernel be than the per-head kernel?
            The per-head kernel requests 4 times the bytes, but a cache may serve the repeats if the 4 blocks of a group
            run at about the same time; and your kernel launches 8 blocks where the per-head one launches 32.
          </>
        }
        unit="× the per-head kernel"
        measure={groupedSpeedup}
        reading={
          <>
            Below 1 means the grouped kernel is slower. Neither CUDA nor Apple promises an order for blocks, so whether a
            group&rsquo;s blocks overlap is something to measure. Few blocks is the reason Flash-Decoding also splits a
            head&rsquo;s keys across blocks (facts A8).
          </>
        }
      />
      <p>Every measured ratio, beside the factor by which Eq. 20.13 says the bytes requested shrink:</p>
      <GroupedVsIntensity />
      <p>
        Where grouping costs blocks, the remedy is to split the keys as well, so that one K/V head gets several blocks
        again.
      </p>
      <Beyond>
        <p>
          The next step, not built here: give each (K/V head, sequence) several blocks over slices of the keys and merge
          them in a second kernel with the same Eqs. 20.4 and 20.6. Flash-Decoding does this (<code>num_splits</code>),
          llama.cpp merges with <code>flash_attn_combine_results</code>, and MLX switches to a two-pass kernel for long
          GQA caches (facts A8, D1, D4).
        </p>
      </Beyond>
      <LabResults lab={l5} filter={(r) => r.name === 'perhead' || r.name === 'grouped'} cols={groupedCols} />
      <ReferenceSolution lab={l5} tiers={TIERS_DECODE} />

      <h3>The project: milestone M7</h3>
      <p>
        M7 is one decode step of attention: <code>launchKvAppend</code> and <code>launchDecodeAttention</code> over a
        cache laid out <code>[batch, max_seq, h_kv, 128]</code>, with per-sequence lengths, and{' '}
        <code>launchDecodeAttentionGrouped</code> as its grouped-query form. Chapter 22&rsquo;s batching study, milestone
        M8, asks what batching such a decode step buys in throughput and costs in latency.
      </p>
      <MilestoneStatus />
    </LabsPage>
  );
}
