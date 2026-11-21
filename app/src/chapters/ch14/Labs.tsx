import type { Question } from '../../components/Ask';
import { Beyond, Source, Today, type Tier } from '../../components/notes';
import {
  BackendsAndEmulator,
  LabFiles,
  LabPrediction,
  LabResults,
  LabsIntro,
  LabsPage,
  RaceStatus,
  ReferenceSolution,
  RunningLabs,
  VerificationStatus,
  benchRow,
  defineLabs,
  measure,
  msCol,
  numCol,
  rawCol,
  type BenchSet,
} from '../../components/LabKit';
import { fmt, num, type BenchRow, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';

// The page is built from components/LabKit.tsx; what is here is chapter 14's own content.
// No question on this page: the labs are the exercise. Its four predictions are stored, not
// graded, under ch14/labs/p-lab1-oddeven, p-lab1-cpu, p-lab2-cpu and p-lab3-batch1. Lab 14.2
// has two more on the pages they belong to: ch14/14.6/p-local-gain and ch14/14.7/p-best-r.
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch14', [
  { num: '14.1', dir: 'lab1_twosorts', name: 'odd-even and one-bit radix' },
  { num: '14.2', dir: 'lab2_multibit', name: 'multi-bit, local sort' },
  { num: '14.3', dir: 'lab3_sortrows', name: 'sorted rows (M5)' },
]);
const [l1, l2, l3] = LABS;

/** Identifier colours in the reference solutions: where each value lives. */
const GLOBAL_1 = ['data', 'hasChanged', 'flag', 'keys', 'vals', 'tmpKeys', 'tmpVals', 'bits', 'scanScratch', 'keysIn', 'valsIn', 'keysOut', 'valsOut', 'onesBefore'];
const TIERS_1: Record<string, Tier> = {
  ...Object.fromEntries(GLOBAL_1.map((k) => [k, 'global' as Tier])),
  i: 'reg', tmp: 'reg', key: 'reg', bit: 'reg', dst: 'reg',
};
const TIERS_2: Record<string, Tier> = {
  ...Object.fromEntries([...GLOBAL_1, 'keysLocal', 'valsLocal', 'table', 'tableScanned', 'localStarts'].map((k) => [k, 'global' as Tier])),
  tile_s: 'shared', scan_s: 'shared', start_s: 'shared', offset_s: 'shared', keys_s: 'shared', vals_s: 'shared', sgTotals: 'shared',
  t: 'reg', c: 'reg', p: 'reg', tile: 'reg', base: 'reg', count: 'reg', digit: 'reg', dst: 'reg', myKeys: 'reg', myVals: 'reg', myOnes: 'reg',
};
const TIERS_3: Record<string, Tier> = {
  probs: 'global', probsBits: 'global', sortedProbs: 'global', sortedIdx: 'global', scratch: 'global',
  keys: 'global', idx: 'global', keysIn: 'global', idxIn: 'global', keysOut: 'global', idxOut: 'global', idxSorted: 'global',
  offset_s: 'shared', tileStart_s: 'shared', keys_s: 'shared', idx_s: 'shared', warpSums_s: 'shared',
  hist: 'shared', offset: 'shared', tile_start: 'shared', group_sums: 'shared',
  t: 'reg', u: 'reg', key: 'reg', digit: 'reg', dst: 'reg',
};

// ---------------------------------------------------------------- reading the reports
const N22 = 4194304;
const VOCAB = 151936;
const ms = (s: number | undefined) => (s ? `${fmt(s * 1000)} ms` : '?');
const big = (v: unknown) => (num(v) != null ? num(v)!.toLocaleString('en-US') : '');

/** Lab 14.1: odd-even sort's wall time over the one-bit radix sort's wall time, n = 4,096. */
const oddEvenSlowdown = (reports: Reports) =>
  measure(reports, l1, (s: BenchSet) => {
    const oe = benchRow(s, 'oddeven', { n: 4096 });
    const rx = benchRow(s, 'radix1', { n: 4096 });
    const a = num(oe?.seconds);
    const b = num(rx?.wall_seconds) ?? num(rx?.seconds);
    if (!a || !b) return null;
    return {
      value: a / b,
      detail: `odd-even ${ms(a)} in ${big(oe!.steps)} steps against ${ms(b)} for the radix sort, both host wall time, n = 4,096`,
    };
  });

/** Lab 14.1: std::stable_sort's time over the one-bit radix sort's, 2^22 pairs. */
const radixOverCpu = (reports: Reports) =>
  measure(reports, l1, (s: BenchSet) => {
    const cpu = num(benchRow(s, 'cpu_stable_sort', { n: N22 })?.seconds);
    const rx = num(benchRow(s, 'radix1', { n: N22 })?.seconds);
    if (!cpu || !rx) return null;
    return {
      value: cpu / rx,
      detail: `${ms(rx)} for the radix sort against ${ms(cpu)} for std::stable_sort on one CPU thread, ${N22.toLocaleString('en-US')} pairs`,
    };
  });

/** Lab 14.2: the fastest local-sort row against std::stable_sort, at the size the CPU row was timed. */
function bestLocal(rows: BenchRow[]): { best: BenchRow; cpu: BenchRow } | null {
  const cpu = rows.find((r) => r.name === 'cpu_stable_sort' && num(r.seconds));
  if (!cpu) return null;
  const local = rows.filter((r) => r.name === 'local' && num(r.seconds) && r.n === cpu.n);
  const best = local.reduce<BenchRow | undefined>((x, r) => (!x || num(r.seconds)! < num(x.seconds)! ? r : x), undefined);
  return best ? { best, cpu } : null;
}
const multiOverCpu = (reports: Reports) =>
  measure(reports, l2, (s: BenchSet) => {
    const m = bestLocal(s.rows);
    if (!m) return null;
    const a = num(m.cpu.seconds)!;
    const b = num(m.best.seconds)!;
    return {
      value: a / b,
      detail: `fastest row r = ${String(m.best.radix_bits)}, c = ${String(m.best.keys_per_thread)} at ${ms(b)}, against ${ms(a)} for std::stable_sort on one CPU thread, ${big(m.cpu.n)} pairs`,
    };
  });

/** Lab 14.3: std::stable_sort per row over the GPU sort, one row of 151,936; batch 64 alongside. */
const ratioAt = (s: BenchSet, rows: number) => {
  const gpu = num(benchRow(s, 'sort_rows', { rows, cols: VOCAB })?.seconds);
  const cpu = num(benchRow(s, 'cpu_stable_sort_rows', { rows, cols: VOCAB })?.seconds);
  return gpu && cpu ? { gpu, cpu, ratio: cpu / gpu } : null;
};
const rowsOverCpu = (reports: Reports) =>
  measure(reports, l3, (s: BenchSet) => {
    const one = ratioAt(s, 1);
    if (!one) return null;
    const many = ratioAt(s, 64);
    return {
      value: one.ratio,
      detail:
        `${ms(one.gpu)} per call on the device against ${ms(one.cpu)} on one CPU thread, 1 × 151,936` +
        (many ? `; at 64 × 151,936 the ratio is ${fmt(many.ratio)} (${ms(many.gpu)} against ${ms(many.cpu)})` : '; no batch-64 row'),
    };
  });

const missingBench = (labNum: string) => (
  <>
    This needs the benchmark rows of lab {labNum} from a real device. Run{' '}
    <code>python3 labs/pmpp.py run {labNum} --bench</code> once your kernels pass (or with <code>--solution</code> to time
    the reference). The emulator is never benchmarked.
  </>
);

// ---------------------------------------------------------------- results tables
const WHAT_1: Record<string, string> = {
  oddeven: 'odd-even sort, whole host loop',
  radix1: 'one-bit radix sort',
  cpu_stable_sort: 'std::stable_sort, one CPU thread',
};
const WHAT_2: Record<string, string> = {
  onebit_global: 'one bit, direct scatter (provided)',
  local: 'local sort',
  cpu_stable_sort: 'std::stable_sort, one CPU thread',
};
const WHAT_3: Record<string, string> = {
  sort_rows: 'GPU sort',
  cpu_stable_sort_rows: 'std::stable_sort, one CPU thread',
};
const what = (names: Record<string, string>) => ({ head: 'What ran', cell: (r: BenchRow) => names[r.name] ?? r.name });
const nCol = { head: 'n', num: true, cell: (r: BenchRow) => big(r.n) };
const wallCol = {
  head: 'Wall, ms',
  num: true,
  cell: (r: BenchRow) => (num(r.wall_seconds) != null ? fmt(num(r.wall_seconds)! * 1000) : r.name === 'oddeven' ? fmt(num(r.seconds)! * 1000) : ''),
};

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS}>
      <LabsIntro>
        Three labs, each in its own directory under <code>labs/ch14/</code>. The first writes the chapter's two listed
        kernels, odd-even sort and a one-bit radix pass, and times them against each other and against one CPU thread. The
        second does the book's exercises 1 to 3 in one multi-bit radix sort with local sorting. The third is the
        through-line project's milestone M5: every row of a batch of probabilities sorted largest first, stably, with the
        token ids carried along.
      </LabsIntro>
      <p>
        Lab 14.1 is optional. Its kernels are worked through in sections <a href={href('ch14', '14.2')}>14.2</a> and{' '}
        <a href={href('ch14', '14.5')}>14.5</a>, and lab 14.2 ships its one-bit sort ready-made as the baseline it
        measures against.
      </p>
      <Beyond title="Why three labs">
        <p>
          PLAN.md gives this chapter the Standard tier, one or two labs. Lab 14.3 is milestone M5 and lab 14.2 is the
          book's exercises 1 to 3, so those two are the chapter's labs; lab 14.1 is kept, as an optional first step, because
          it is the measured baseline that lab 14.2 compares against.
        </p>
      </Beyond>
      <RunningLabs
        example="14.1"
        alsoShownIn={
          <>
            the measured columns and predictions of <a href={href('ch14', '14.6')}>14.6</a>,{' '}
            <a href={href('ch14', '14.7')}>14.7</a> and <a href={href('ch14', '14.8')}>14.8</a>
          </>
        }
      />
      <BackendsAndEmulator />
      <p>
        Every lab of this chapter is also built for the emulator's race flavour, because each has a shared update to get
        right: a flag in lab 14.1, a tile and its scans in lab 14.2, a histogram, a tile and running offsets in lab 14.3.{' '}
        <code>python3 labs/pmpp.py run 14.1 --backend race</code> runs a block's threads at once under ThreadSanitizer,
        which reports two accesses to one location with nothing ordering them, with both source lines. A clean run is
        evidence, not proof: it sees only the schedules that happened (lab 14.2's tests show what that means for a
        missing barrier).
      </p>
      <VerificationStatus />
      <p>
        Two labs use provided code that you do not edit, in <code>labs/ch14/shared-g11/</code>: an exclusive scan of
        unsigned integers for any count, and lab 14.2's baseline. The scan is built the way section 11.9 builds one:
        block scans, a scan of the block totals, then an add, so no block ever waits for another. Each lab tests the
        provided code on its own first, so a broken provided file shows up as itself.
      </p>
      <Beyond>
        <p>The provided scan and baseline are ours; the book's chapter 14 calls a grid-wide scan it never defines.</p>
      </Beyond>

      {/* ------------------------------------------------------------ 14.1 */}
      <h2>Lab 14.1: two first sorts, odd-even and one-bit radix</h2>
      <p>
        Two sorts of 32-bit unsigned keys, written the way the chapter builds them. Part A is odd-even sort: the kernel of
        Fig. 14.2 and the host loop the book describes but does not show. Part B is a one-bit LSD radix sort of (key, value)
        pairs: Fig. 14.7, split where it calls a scan across the whole grid.
      </p>
      <Source section="14.2, 14.5" pages="331–333, 336–339" figs="Figs. 14.2, 14.6 and 14.7" />
      <p>
        <code>oddEvenStep(data, hasChanged, n, isOddStep)</code> gives thread t the pair (i, i + 1), with i = 2t on an
        even step and 2t + 1 on an odd one, puts the pair in order and, if it swapped, sets the flag with an atomic store.
        It returns at once when n is below 2 or its pair runs past the end. <code>launchOddEvenSort(data, flag, n)</code>{' '}
        alternates even and odd steps, starting even: clear the flag, launch an even step and the odd step after it, then
        read the flag, and stop only when that pair of steps made no swap. It returns the number of steps launched.
      </p>
      <p>
        For part B, Fig. 14.7 extracts the bits, calls <code>gridExclusiveScan</code>, then scatters. The book never
        defines that function, and a scan across blocks inside one kernel needs every block to wait for the others: a
        cooperative launch on CUDA and HIP, nothing documented on Metal. So a pass here is three launches.{' '}
        <code>radixExtractBits</code> writes <code>bits[i]</code> and <code>bits[n] = 0</code>; the provided{' '}
        <code>launchExclusiveScan(bits, n + 1, scratch)</code> turns that into ones-before, with the total at{' '}
        <code>bits[n]</code>; <code>radixScatter</code> applies the two destination formulas and moves each value with its
        key. <code>launchRadixSortOneBit</code> runs 32 passes, reading one pair of buffers and writing the other, and
        leaves the result in <code>keys</code> and <code>vals</code>.
      </p>
      <Today checked="2026-11-20 (notes/today/ch04-facts.md A7, C3; ch14-facts.md C4)">
        <p>
          A grid-wide barrier exists on CUDA and HIP only under a cooperative launch with every block resident. Metal has
          none, and documents no forward progress between threadgroups.
        </p>
      </Today>
      <p>
        The values are the keys' input positions; in lab 14.3 they are token ids. With positions as values the test sees
        stability directly: among equal keys, the positions must increase.
      </p>
      <LabFiles
        lab={l1}
        files={[
          [
            'kernel.cu',
            <>
              <code>oddEvenStep</code>, <code>launchOddEvenSort</code>, <code>radixExtractBits</code>,{' '}
              <code>radixScatter</code> and <code>launchRadixSortOneBit</code>
            </>,
          ],
          [
            'kernel.metal',
            <>
              <code>odd_even_step</code>, <code>radix_extract_bits</code> and <code>radix_scatter</code>; the flag is a{' '}
              <code>device atomic_uint*</code> set with <code>atomic_store_explicit</code>
            </>,
          ],
          [
            'launch.mm',
            <>
              <code>runOddEvenSort</code>, which commits one command buffer per even-odd pair of steps, waits for it and
              reads the flag, and <code>encodeRadixSortOneBit</code>, which puts all 32 passes, the provided scan's
              dispatches included, into the one compute encoder it is given
            </>,
          ],
        ]}
      />
      <Today checked="2026-11-20 (infra-facts.md C1, C12; ch05-facts.md C6; ch02-facts.md C9)">
        <p>
          Metal's atomic functions take only atomic-typed pointers. The CPU may read a shared buffer the GPU wrote only once
          the command buffer has completed. Dispatches in one plain compute encoder run in order and see each other's
          writes.
        </p>
      </Today>

      <h3>Five traps on the pages</h3>
      <p>
        Each turns a literal reading of the chapter into a wrong or fragile program. The lab's README names them without
        the code, and the reference solution says in its comments where it differs from the page.
      </p>
      <ul>
        <li>
          <b>The N + 1 scan.</b> Fig. 14.7 reads <code>bits[N]</code> as the total number of ones; that entry exists only
          if the scan covers N + 1 entries. Fig. 14.6 draws N.
        </li>
        <li>
          <b>The "zero bucket" slip.</b> On p. 337 the sentence that introduces the second formula says it is for keys of
          the zero bucket. The formula is the one bucket's.
        </li>
        <li>
          <b>The unsigned wrap.</b> Fig. 14.2 tests <code>i &lt; N - 1</code>; with an unsigned N of 0 that wraps and
          every thread passes. The driver never launches with n = 0, so write the guard so that it cannot wrap anyway.
        </li>
        <li>
          <b>The stopping rule.</b> One quiet step proves nothing: <code>[1, 3, 2, 4]</code> makes no swap on its first,
          even, step. Two steps in a row, one even and one odd in either order, both quiet, show that every pair is in
          order. Reading the flag once per even-odd pair, as this lab does, sees only the quiet pairs that start on an
          even step, so it may launch one step more than necessary, in exchange for half the reads.
        </li>
        <li>
          <b>The flag race.</b> Many threads may set <code>hasChanged</code> in one step. The book calls it benign and then
          warns that the C++ memory model makes it undefined behaviour anyway (p. 333). Use an atomic store.
        </li>
      </ul>
      <Source section="14.2, 14.5" pages="331–339" figs="Figs. 14.2, 14.6 and 14.7" />

      <h3>What the tests check</h3>
      <p>
        Run <code>python3 labs/pmpp.py run 14.1 --backend emu</code> first; it takes a few minutes, because every GPU
        thread runs on the CPU. The provided scan is tested on its own. Odd-even sort runs at n = 1 (with 2), 3, 4, 8, 33,
        256 and 1000 on random, reversed, sorted, all-equal and largest-first input and on the trap above; the result must be
        sorted and a permutation of the input, and the step count even, at least 2 for n of 2 or more, and at most n + 2
        rounded up to whole pairs. The radix sort runs at n = 1 (with 2), 31, 32, 33, 1000, 4097, 2^16 and 2^20 on
        full-range random keys (the top bit set about half the time), heavy ties, all equal, sorted, reversed and powers of
        two; keys must equal <code>std::stable_sort</code>'s and values its permutation exactly. The emulators stop at
        smaller sizes.
      </p>
      <p>
        Before every run the driver writes <code>0xCDCDCDCD</code> into <code>bits</code>, the scan's scratch, the
        temporary buffers and 64 guard slots past the end of every array, so a pass that writes past n is caught. On the
        race backend a plain store to the flag is reported as a data race with both source lines; the emulator cannot see
        it, because it never interrupts a thread between synchronisation points.
      </p>
      <RaceStatus lab={l1} />

      <h3>Before you benchmark</h3>
      <p>
        Write down why first, from two counts. Launches and host round trips: odd-even sort launches one kernel per step,
        about n of them, and waits for the GPU once per pair of steps to read the flag; the radix sort launches 32 × (2 +
        the scan's launches) kernels and never waits in the middle. Bytes per pass: one radix pass reads every pair and
        writes the bits, scans n + 1 entries, then reads keys, values and bits and writes keys and values, 32 times over,
        against <code>std::stable_sort</code>'s order of n log n comparisons and moves.
      </p>
      <LabPrediction
        id="ch14/labs/p-lab1-oddeven"
        prompt={<>At n = 4,096, how many times longer than your one-bit radix sort will your odd-even sort take, both timed on the host's clock?</>}
        unit="× the radix sort's time"
        measure={oddEvenSlowdown}
        missing={missingBench('14.1')}
        reading={
          <>
            Both sorts are timed by the host here, from the first launch to the result, so odd-even sort pays for every
            launch and every read of the flag. Compare the steps column with n before you explain the number.
          </>
        }
      />
      <LabPrediction
        id="ch14/labs/p-lab1-cpu"
        prompt={<>At 2^22 pairs, how many times faster than one CPU thread's <code>std::stable_sort</code> will your one-bit radix sort be? (Below 1 means slower.)</>}
        unit="× the speed of std::stable_sort"
        measure={radixOverCpu}
        missing={missingBench('14.1')}
        reading={
          <>
            The radix time is the device time of the work on Metal, and the host's time around the launches and the final
            synchronisation on CUDA and HIP; the CPU time is a host measurement of the same pairs. A one-bit sort makes 32
            passes over every pair; sections 14.6 to 14.8 are about making fewer, cheaper ones.
          </>
        }
      />
      <LabResults
        lab={l1}
        cols={[what(WHAT_1), nCol, { head: 'Steps', num: true, cell: (r) => big(r.steps) }, msCol, wallCol, numCol('Mkeys/s', 'mkeys_per_s')]}
      />
      <p>
        How the benchmark works: odd-even sort at n = 1,024 and 4,096, timed around the whole host loop; the radix sort at
        1,024, 4,096, 2^16, 2^20 and 2^22, on random full-range keys; <code>std::stable_sort</code> of the same pairs on one
        CPU thread. Each row is the best of 5 runs after a warm-up, and every timed result is checked before it is
        reported.
      </p>
      <ReferenceSolution lab={l1} tiers={TIERS_1} />

      {/* ------------------------------------------------------------ 14.2 */}
      <h2>Lab 14.2: multi-bit radix sort with local sorting</h2>
      <p>
        One stable LSD radix sort of (key, value) pairs that does the book's exercises 1, 2 and 3 at once: the local sort in
        shared memory of section 14.6, a digit of r bits as in 14.7, and c keys per thread as in 14.8. Both r and c run from 1
        to 8 and are chosen at run time. A block has 256 threads and owns a tile of 256c consecutive pairs; the last tile may
        be partial, and its padding is neither counted nor written. Lab 14.1's part B is this lab's baseline, provided
        ready-made, so you can start here.
      </p>
      <Source section="14.6 to 14.8" pages="339–345, 347" figs="Figs. 14.8, 14.9, 14.12, 14.13; exercises 1 to 3" />
      <p>
        One pass over the digit <code>(key &gt;&gt; shift) &amp; (2^r − 1)</code> is three launches, as section 14.6 lays it
        out. <code>radixLocalSort</code> loads the tile into shared memory, sorts it by the digit with r stable one-bit
        splits (each a block-level exclusive scan), writes it back to the same place, writes the block's 2^r bucket sizes
        into the digit-major table <code>table[d × numBlocks + block]</code> (all blocks' counts for digit 0, then digit 1,
        as in Figs. 14.9 and 14.12) and where each bucket starts inside the tile into <code>localStarts</code>. The provided
        scan turns the table into every local bucket's global start. <code>radixScatterLocal</code> then sends the pair at
        tile position p with digit d to that start plus (p − the bucket's local start), so neighbouring threads write
        neighbouring addresses within a bucket.
      </p>
      <p>
        <code>launchRadixSortMulti</code> runs ceil(32 / r) passes. When r does not divide 32 the last pass has only the
        bits that remain: a shift by 32 or more is undefined in C++, so the digit must shrink, not the mask alone.
      </p>
      <LabFiles
        lab={l2}
        files={[
          [
            'kernel.cu',
            <>
              <code>radixLocalSort</code>, <code>radixScatterLocal</code> and <code>launchRadixSortMulti</code>; both
              kernels take <code>keysPerThread</code> as their last argument
            </>,
          ],
          ['kernel.metal', <><code>radix_local_sort</code> and <code>radix_scatter_local</code></>],
          [
            'launch.mm',
            <>
              <code>encodeRadixSortMulti</code>: every pass, the provided scan included, in the one compute encoder it is
              given, with the threadgroup memory sized and checked against the device's limit before anything is encoded
            </>,
          ],
        ]}
      />
      <p>
        Do the block scan in shared memory with barriers. A warp-shuffle scan needs a lane mask, which on HIP must be a
        64-bit integer on every target; the shared-memory form has no mask to get wrong. In Metal the scan may use{' '}
        <code>simd_prefix_exclusive_sum</code> inside each SIMD-group and one threadgroup slot per SIMD-group for their
        totals, with the SIMD-group width read from the pipeline, never assumed. Such a scan adds up the thread totals in
        (SIMD-group, lane) order, so in the splits give each thread the tile positions of its rank in that order,{' '}
        <code>sg × width + lane</code>, not of its <code>thread_position_in_threadgroup</code>; the reference and the
        provided scan do. The tile is 256c × 8 bytes of threadgroup memory; whether it fits is your device's answer, from{' '}
        <code>python3 labs/pmpp.py doctor</code>.
      </p>
      <Today checked="2026-11-20 (infra-facts.md B3, C2, C5; ch04-facts.md C2; ch05-facts.md C1, C3, C8)">
        <p>
          HIP's warp intrinsics take a 64-bit mask. MSL's SIMD-group prefix sums need GPU family Apple7. The SIMD-group width
          is not a documented constant, and how a threadgroup is split into SIMD-groups is "implementation-defined".
          Threadgroup memory lengths are multiples of 16 bytes and must fit <code>maxThreadgroupMemoryLength</code>.
        </p>
      </Today>
      <p>
        Provided, not edited: the scan of lab 14.1, and <code>onebit_global.cu</code>, <code>.metal</code> and{' '}
        <code>_launch.mm</code>, which are lab 14.1's design, one bit per pass with every key written straight to its
        global place. That is the baseline for section 14.6's claim that local sorting pays.
      </p>

      <h3>What the tests check</h3>
      <p>
        On a GPU, every r in 1, 2, 3, 4, 5 and 8 (3 and 5 do not divide 32) with every c in 1, 2 and 4, at n = 1, 255, 256,
        257, 1000, 4097, 65537 and 2^20, with eight key patterns: lab 14.1's six, plus "every key shares its low byte" (on
        the first passes each block puts all its keys in one bucket, so the table is mostly zeros) and "keys below 256"
        (every higher digit is zero). Keys must equal <code>std::stable_sort</code>'s and values its permutation exactly.
        The driver fills the table, <code>localStarts</code>, the scratch and 64 guard slots with <code>0xCD</code> bytes
        before every run, so a count that is not written for this pass fails. A configuration the device cannot run is
        printed as skipped with the reason, never counted as a pass. The emulators pay for every thread of up to 96 launches
        per sort, so they run a smaller matrix whose key patterns rotate across the configurations; the README lists it.
      </p>
      <p>
        A missing barrier at the end of a split lets a thread read a half-written tile. We planted exactly that in a copy of
        the reference (2026-11-20). The emulator fails it every time: at r = 2 the first thread order already gives a
        wrong order, and AddressSanitizer then names the tile write that went out of bounds. The race backend also fails,
        but not by naming the barrier: the run stops with a crash at the tile write once a corrupted index is used, and the
        one race it reports is a later, consequential one on the bucket starts. At r = 1 both pass, because the next
        step's barrier covers the gap. So run the emulator first; a clean race run is not evidence that every barrier is
        there.
      </p>
      <Beyond>
        <p>
          When we planted it, the race flavour shared one ThreadSanitizer token between all of a block's{' '}
          <code>__syncthreads()</code>, which could hide a race between two barriers. It now alternates two tokens per
          barrier, and on 2026-11-21 it reported the wrap page's missing barrier on its source lines
          (<code>labs/ch14/snippets/README.md</code>); the planted defect of this lab has not been run again since.
        </p>
      </Beyond>
      <RaceStatus lab={l2} />

      <h3>Before you benchmark</h3>
      <p>
        This lab has three predictions. Two sit on the pages whose model they test:{' '}
        <a href={href('ch14', '14.6')}>14.6</a> asks whether local sorting at r = 1, c = 1 beats the provided direct
        scatter, and <a href={href('ch14', '14.7')}>14.7</a> which r is fastest at c = 1; <a href={href('ch14', '14.8')}>14.8</a>{' '}
        adds one on c. Make those there. The third is here. Count first: ceil(32 / r) passes, each reading and writing every
        pair twice; up to 2^r runs of addresses per warp in the scatter, each about 256c / 2^r keys long on random keys; a
        table of 2^r × ceil(n / 256c) entries written, scanned and read every pass.
      </p>
      <LabPrediction
        id="ch14/labs/p-lab2-cpu"
        prompt={<>At the size the benchmark settles on (2^22 pairs or more), how many times faster than one CPU thread's <code>std::stable_sort</code> will your fastest configuration be?</>}
        unit="× the speed of std::stable_sort"
        measure={multiOverCpu}
        missing={missingBench('14.2')}
        reading={
          <>
            Set it beside lab 14.1's second prediction: random pairs of the same kind, the same CPU baseline, a sort with
            fewer passes and coalesced writes. The detail says at which size it was timed. Which r and c won is part of
            the answer; the table shows every row.
          </>
        }
      />
      <LabResults
        lab={l2}
        cols={[
          what(WHAT_2),
          nCol,
          rawCol('r', 'radix_bits'),
          rawCol('c', 'keys_per_thread'),
          rawCol('Passes', 'passes'),
          msCol,
          numCol('Mkeys/s', 'mkeys_per_s'),
          numCol('Counted GB/s', 'counted_gbps'),
        ]}
      />
      <p>
        How the benchmark works: n starts at 2^22 and doubles while the baseline takes under 20 ms. It times the baseline,
        the local sort at r = 1, 2, 4, 6 and 8 with c = 1 and at c = 1, 2, 4 and 8 with r = 4 and with r = 8, skipping what
        the device cannot run, and <code>std::stable_sort</code> of the same pairs on one CPU thread. "Counted GB/s" is
        counted, not measured, traffic: 32 bytes per pair per pass divided by the time, with the table and the scan left
        out.
      </p>
      <ReferenceSolution lab={l2} tiers={TIERS_2} />

      {/* ------------------------------------------------------------ 14.3 */}
      <h2>Lab 14.3: sorted candidates, every row descending and stable (milestone M5)</h2>
      <p>
        One host function and the kernels behind it:{' '}
        <code>launchSortRowsDesc(probs, sortedProbs, sortedIdx, scratch, rows, cols)</code>. For every row of a{' '}
        <code>[rows × cols]</code> float32 tensor, row-major with the batch first, <code>sortedIdx</code> gets the column
        indices in descending order of value, equal values by increasing column, and <code>sortedProbs</code> gets the values
        in that order. <code>scratch</code> holds 4 × rows × cols unsigned integers for you to use. It is the input to top-p's
        cumulative sum: sort, then scan.
      </p>
      <Source section="14.4 to 14.8" pages="334–345" figs="Figs. 14.6, 14.7; the multi-bit local sort of exercises 1 to 3" />
      <p>
        The order is that of one key, sorted ascending. Take the 32 bits of the float; if they are −0.0 make them +0.0; set
        the sign bit of a non-negative value and invert every bit of a negative one; then invert the whole key, which makes
        the ascending sort descending without reversing the order of ties. On numbers that is descending numeric order with
        the two zeros equal. On NaN it is the bits: +NaN before everything and −NaN after everything. Section{' '}
        <a href={href('ch14', '14.4')}>14.4</a> builds this transform step by step.
      </p>
      <Today checked="2026-11-20 (notes/today/ch14-facts.md A3, E1)">
        <p>
          This is the transform CUB documents for its float keys, including the step that merges the two zeros, inverting
          the key for descending order, and sorting NaN by its bits.
        </p>
      </Today>
      <p>
        The kernels never perform a float operation. The launcher hands them <code>probs</code> and{' '}
        <code>sortedProbs</code> as unsigned views (a cast in CUDA; buffers bound as <code>device const uint*</code> and{' '}
        <code>device uint*</code> in Metal), so the −0.0 test, the sign test and the copy are integer work. On Metal both
        candidate default math modes allow "no signed zeros", so a float-side zero test may be rewritten. On HIP,{' '}
        <code>__float_as_uint</code> is in the header but not in AMD's math reference. The unsigned view needs no
        bit-cast intrinsic at all, which makes it the form that works on every backend.
      </p>
      <Today checked="2026-11-20 (notes/today/ch14-facts.md A8, B6, C3)">
        <p>
          CUDA documents <code>__float_as_uint</code>. HIP defines it in a device header that AMD's published math API
          does not list. Metal's fast math leaves NaN and INF undefined.
        </p>
      </Today>
      <p>
        The design, ours: one block of 256 threads per row, the shape of lab 10.3. <code>descKeys</code> writes each value's
        key and its column. <code>sortRowsPass</code>, launched once per digit of r bits (8 suggested, so four passes),
        counts the row's digits in a shared-memory histogram with atomics as in chapter 9, scans the 2^r counts to find where
        each bucket starts in the output row, then walks the row in tiles of 256c keys: it sorts each tile in shared memory
        by the digit with r stable one-bit splits and writes each key to its bucket's running offset, which then moves on by
        the bucket's size in that tile. <code>gatherRows</code> fills <code>sortedIdx</code> and copies the bits of{' '}
        <code>probs</code> at those indices into <code>sortedProbs</code>.
      </p>
      <p>
        Why it is stable across tiles: one block visits the tiles in order, and a bucket's offset only moves forward, so a
        key from an earlier tile always lands before an equal-digit key from a later tile. The running offset does the job
        of the book's table of bucket sizes per block: with one block per row there is one block to account for, and no
        scan across blocks between the passes.
      </p>
      <Beyond>
        <p>
          The float key, one block per row, the running offset per bucket and the gather are ours. The book has no batched
          or row-wise sort.
        </p>
      </Beyond>
      <LabFiles
        lab={l3}
        files={[
          [
            'kernel.cu',
            <>
              <code>descKeys</code>, <code>sortRowsPass</code>, <code>gatherRows</code> and{' '}
              <code>launchSortRowsDesc</code>; the block scan is provided in the file
            </>,
          ],
          ['kernel.metal', <><code>desc_keys</code>, <code>sort_rows_pass</code> and <code>gather_rows</code></>],
          ['launch.mm', <><code>encodeSortRowsDesc</code>, the whole sort in the one compute encoder it is given</>],
        ]}
      />

      <h3>What the tests check</h3>
      <p>
        Rows 1, 3, 4 and 7 by columns 1, 2, 31, 255, 256, 257, 1000, 4096, 32000, 50257 and 151936 (the last two are
        vocabulary sizes of real models), with the data mixed within a batch so that reading the wrong row fails: softmax
        probabilities; the same with planted exact ties, including runs of 40 equal values across every multiple of 256
        from 512 to 8192, so that for any c from 1 to 32 a run crosses a tile boundary of 256c keys once the row is long
        enough; every value equal; one-hot rows
        (the zeros must stay in column order); logits with −0.0 and +0.0 interleaved, infinities, <code>±FLT_MAX</code>{' '}
        and subnormals; and rows with NaNs of both signs.
      </p>
      <p>
        <code>sortedIdx</code> must equal the reference exactly, which is the stability check; <code>sortedProbs</code> must
        equal the gathered bits exactly; every output row must be a permutation; 64 guard entries past every output must be
        untouched, and <code>probs</code> unchanged. No tolerance: a sort does no arithmetic. The reference in{' '}
        <code>main.cpp</code> is <code>std::stable_sort</code> of the column indices on the integer key, cross-checked on
        every NaN-free row against <code>std::stable_sort</code> with <code>&gt;</code> on the floats. The emulator runs rows
        1 and 3 up to 4,099 columns, with the data recipes rotating as the size changes; the race backend runs two rows up to
        2,100 columns, two tiles even at 8 keys per thread.
      </p>
      <Today checked="2026-11-20 (notes/today/ch14-facts.md A11, C2)">
        <p>
          <code>std::stable_sort</code> keeps equivalent elements in order. Accelerate's <code>vDSP_vsorti</code> is
          documented as not stable, so it is not the reference.
        </p>
      </Today>
      <RaceStatus lab={l3} />

      <h3>The library route, per backend</h3>
      <p>
        The book's advice is to use a library sort unless you have a reason not to. On CUDA this operation is{' '}
        <code>cub::DeviceSegmentedRadixSort::SortPairsDescending</code> or{' '}
        <code>cub::DeviceSegmentedSort::StableSortPairsDescending</code>, but not the plain{' '}
        <code>DeviceSegmentedSort::SortPairsDescending</code>, which is documented as not stable. On HIP the same names exist
        under <code>hipcub::</code>. On Metal there is no kernel-level library sort; MPSGraph's <code>argSort</code> sorts
        along an axis and Apple states no stability or tie order for it, so if equal probabilities must come out by
        increasing token id, you write the kernel, as here. <a href={href('ch14', '14.9')}>Section 14.9</a> has the whole
        table.
      </p>
      <Today checked="2026-11-20 (notes/today/ch14-facts.md A5, B1, B2, C1)">
        <p>
          In hipCUB 4.7.0, <code>StableSortPairsDescending</code> calls <code>SortPairsDescending</code>, which runs
          rocPRIM's segmented radix sort; rocPRIM documents its radix sort as stable.
        </p>
      </Today>
      <p>
        One reason this kernel matters on HIP: llama.cpp's CUDA <code>argsort</code> sorts long rows with CUB, its HIP build
        has no CUB path, and its other path needs the whole row, padded to a power of two, in shared memory. A 151,936-column
        row pads to 262,144 integers, 1 MiB, so by our reading a vocabulary-sized <code>argsort</code> is reported
        unsupported on HIP and runs on another backend. A row sort whose shared memory does not grow with the row, like this
        one, is what that path lacks.
      </p>
      <Beyond>
        <p>
          Our reading of llama.cpp's source (notes/today/ch14-facts.md D1), not something we ran. This lab is not compiled or
          run on AMD hardware here.
        </p>
      </Beyond>

      <h3>Before you benchmark</h3>
      <p>
        One block per row means that at batch 1 a single block sorts the whole row while the rest of the device has nothing
        to do: the situation section <a href={href('ch10', '10.10')}>10.10</a> showed for a softmax, and the one the end of{' '}
        <a href={href('ch14', '14.8')}>14.8</a> describes for this lab. Write down a number, and then whether you expect the
        ratio to grow or shrink at batch 64; the reveal shows both.
      </p>
      <LabPrediction
        id="ch14/labs/p-lab3-batch1"
        prompt={
          <>
            At batch 1 and 151,936 columns, how many times faster than one CPU thread's <code>std::stable_sort</code> will
            your kernel be? (Below 1 means slower.)
          </>
        }
        unit="× the speed of std::stable_sort"
        measure={rowsOverCpu}
        missing={missingBench('14.3')}
        reading={
          <>
            The CPU sorts one row after another, so its time grows with the batch. The GPU gives each row its own block, so
            until the device is full more rows cost little extra time. Read the batch-64 ratio against your second answer.
          </>
        }
      />
      <LabResults
        lab={l3}
        cols={[
          what(WHAT_3),
          rawCol('Rows', 'rows'),
          { head: 'Cols', num: true, cell: (r) => big(r.cols) },
          msCol,
          numCol('Rows/s', 'rows_per_s'),
          numCol('Mkeys/s', 'mkeys_per_s'),
        ]}
      />
      <p>
        How the benchmark works: softmax probabilities at 1, 8 and 64 rows by 32,000 and 151,936 columns. Each timed run
        repeats the whole sort (keys, every pass, gather) until it lasts at least 20 ms, and the row is the best of 5 such
        runs, per call. The baseline is <code>std::stable_sort</code> with <code>&gt;</code> on each row in turn, plus the
        gather, on one CPU thread. Every timed result is checked before it is reported.
      </p>

      <h3>The project milestone</h3>
      <p>
        <code>launchSortRowsDesc</code> is milestone M5's sorted candidates. It follows the project's conventions so that
        later modules can compose it: row-major float32, batch first, raw device pointers and explicit sizes as arguments, a{' '}
        <code>launch</code> host function beside the kernels, the CPU reference in <code>main.cpp</code>. Its{' '}
        <code>sortedProbs</code> is exactly what lab 11.6's top-p keep mask takes, a row of probabilities sorted largest
        first, and <code>sortedIdx</code> says which token each kept slot holds. Module X1 joins the pieces and draws the
        sample; top-k it does with a selection, not this sort (<a href={href('ch14', '14.9')}>14.9</a> says why).
      </p>
      <ReferenceSolution lab={l3} tiers={TIERS_3} />
    </LabsPage>
  );
}
