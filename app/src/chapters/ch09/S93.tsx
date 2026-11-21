import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { benchRows, fmt, num, realBackends, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  BOOK_MEMORY,
  elementRate,
  histogramBound,
  peakBandwidth,
  perLocationRate,
  skewSlowdown,
  uniformShare,
} from '../../sim/atomicrate';
import { histogram, hottest, preset } from '../../sim/histogram';
import { AtomicWaitScene } from './AtomicWaitScene';

const LAB = 'ch09/lab1_racehunt';

/** "2.5 M/s", "640 M/s", "32 G/s". */
function rate(v: number): string {
  if (!isFinite(v)) return 'n/a';
  if (v >= 1e9) return `${fmt(v / 1e9)} G/s`;
  if (v >= 1e6) return `${fmt(v / 1e6)} M/s`;
  if (v >= 1e3) return `${fmt(v / 1e3)} k/s`;
  return `${fmt(v)} /s`;
}

// ---------------------------------------------------------------- explorer
type Place = 'dram' | 'llc' | 'shared';
const PLACES: { id: Place; label: string; min: number; max: number; start: number; words: string }[] = [
  { id: 'dram', label: 'DRAM', min: 100, max: 800, start: 200, words: "the book's example is 200" },
  { id: 'llc', label: 'last-level cache', min: 10, max: 90, start: 30, words: 'the book: "tens of cycles"; starting value illustrative' },
  { id: 'shared', label: 'shared memory', min: 1, max: 9, start: 4, words: 'the book: "a few cycles"; starting value illustrative' },
];
type Dist = 'uniform' | 'bright' | 'constant' | 'custom';
const DISTS: { id: Dist; label: string }[] = [
  { id: 'uniform', label: 'uniform' },
  { id: 'bright', label: 'a bright image' },
  { id: 'constant', label: 'constant' },
  { id: 'custom', label: 'set the busiest bin' },
];
const BIN_CHOICES = [1, 2, 4, 8, 16, 32, 64, 128, 256];
/** Share of the busiest bin in our sky picture, binned into `bins` equal intervals. */
const brightShare = (bins: number) => hottest(histogram(preset('sky').values, 256 / bins)).share;

function LogBars({ rows }: { rows: { label: string; v: number; tone: string }[] }) {
  const W = 440;
  const L = 118;
  const lo = 1e5;
  const hi = 1e11;
  const X = (v: number) => L + ((Math.log10(Math.max(lo, v)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * (W - L - 22);
  return (
    <svg viewBox={`0 0 ${W} ${rows.length * 26 + 26}`} role="img" aria-label="Rates on a log scale" style={{ width: '100%', maxWidth: W, display: 'block' }}>
      {[1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11].map((t) => (
        <g key={t}>
          <line x1={X(t)} x2={X(t)} y1={4} y2={rows.length * 26 + 6} stroke="var(--rule-soft)" />
          <text x={X(t)} y={rows.length * 26 + 20} fontSize="10.5" textAnchor="middle">{rate(t).replace('/s', '')}</text>
        </g>
      ))}
      {rows.map((r, i) => (
        <g key={r.label}>
          <text x={L - 6} y={i * 26 + 20} fontSize="11.5" textAnchor="end">{r.label}</text>
          <rect x={L} y={i * 26 + 8} width={Math.max(1, X(r.v) - L)} height={15} fill={r.tone} />
        </g>
      ))}
    </svg>
  );
}

function measuredSkew(reports: Reports) {
  return realBackends(reports).flatMap((b) => {
    const rows = benchRows(reports[b], LAB).filter((r) => r.name === 'basic');
    const uni = num(rows.find((r) => r.data === 'uniform')?.gpixels_per_s);
    const con = num(rows.find((r) => r.data === 'constant')?.gpixels_per_s);
    return uni && con ? [{ b, device: String(reports[b].device.name ?? b), uni, con, ratio: uni / con }] : [];
  });
}

// The explorer shows the model only. The reader's measured slowdown is the answer to the
// prediction further down (ch09/9.3/p-skew-slowdown), so it appears only in that
// prediction's reveal (MeasuredSkew), after the reader has committed a number.
function ThroughputExplorer() {
  const [place, setPlace] = useState<Place>('dram');
  const [lat, setLat] = useState<Record<Place, number>>({ dram: 200, llc: 30, shared: 4 });
  const [book, setBook] = useState(true);
  const [ghz, setGhz] = useState(1.5);
  const [bins, setBins] = useState(256);
  const [dist, setDist] = useState<Dist>('uniform');
  const [custom, setCustom] = useState(0.5);

  const P = PLACES.find((x) => x.id === place)!;
  const clockHz = book ? BOOK_MEMORY.clockHz : ghz * 1e9;
  const mem = { ...BOOK_MEMORY, clockHz };
  const cap = elementRate(mem);
  const one = perLocationRate(clockHz, lat[place]);
  const share =
    dist === 'uniform' ? uniformShare(bins) : dist === 'constant' ? 1 : dist === 'bright' ? brightShare(bins) : Math.max(custom, 1 / bins);
  const bound = histogramBound(one, share, cap);
  const reset = () => {
    setPlace('dram');
    setLat({ dram: 200, llc: 30, shared: 4 });
    setBook(true);
    setBins(256);
    setDist('uniform');
  };

  return (
    <Bench
      title="Atomic throughput"
      task="Start from the book's example. Then find what matters more: the number of bins, or the share of the busiest one?"
      foot={
        <>
          The memory system is the book's hypothetical one (8-byte channels, double data rate, 8 channels); its element
          rate scales with the clock you choose. Latencies are per access, the same for the load and the store.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={reset}>Book's example</button>
        <label className="field">
          Where the atomic is performed
          <span className="seg" role="group" aria-label="Where the atomic is performed">
            {PLACES.map((x) => (
              <button key={x.id} aria-pressed={place === x.id} onClick={() => setPlace(x.id)}>{x.label}</button>
            ))}
          </span>
        </label>
        <label className="field">
          Latency per access = {lat[place]} cycles ({P.words})
          <input type="range" min={P.min} max={P.max} step={place === 'dram' ? 10 : 1} value={lat[place]} aria-label="Latency per access"
            onChange={(e) => setLat({ ...lat, [place]: +e.target.value })} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Clock
          <span className="seg" role="group" aria-label="Clock">
            <button aria-pressed={book} onClick={() => setBook(true)}>the book's 1 GHz</button>
            <button aria-pressed={!book} onClick={() => setBook(false)}>custom</button>
          </span>
        </label>
        {!book ? (
          <label className="field">
            GHz
            <input type="number" min="0.1" max="5" step="0.1" value={ghz} onChange={(e) => setGhz(Math.max(0.1, +e.target.value || 0.1))} />
          </label>
        ) : null}
        <label className="field">
          Bins
          <select value={bins} onChange={(e) => setBins(+e.target.value)} aria-label="Bins">
            {BIN_CHOICES.map((b) => <option key={b} value={b}>{b}</option>)}
          </select>
        </label>
        <label className="field">
          Data
          <span className="seg" role="group" aria-label="Data">
            {DISTS.map((d) => (
              <button key={d.id} aria-pressed={dist === d.id} onClick={() => setDist(d.id)}>{d.label}</button>
            ))}
          </span>
        </label>
        <label className="field">
          Busiest bin's share = {fmt(100 * share)}%
          <input type="range" min="0" max="1" step="0.01" value={dist === 'custom' ? custom : share} disabled={dist !== 'custom'}
            aria-label="Busiest bin's share" onChange={(e) => setCustom(+e.target.value)} />
        </label>
      </div>

      <div className="cols">
        <div className="panel">
          <p style={{ margin: 0 }}>
            One location: {fmt(clockHz / 1e9)} GHz / ({lat[place]} + {lat[place]} cycles) ={' '}
            <b data-one={one}>{rate(one)}</b> atomics.
          </p>
          <p style={{ margin: '0.6rem 0 0' }}>
            The whole histogram: {rate(one)} / {fmt(share, 3)} ={' '}
            <b style={{ fontSize: '1.3rem' }} data-bound={bound.rate}>{rate(bound.rate)}</b>
            {bound.capped ? (
              <> <span className="chip global">capped by memory</span></>
            ) : null}
          </p>
          <p className="muted" style={{ margin: '0.6rem 0 0' }}>
            The memory system delivers {rate(cap).replace('/s', '')} four-byte elements per second ({fmt(peakBandwidth(mem) / 1e9)} GB/s).{' '}
            {bound.capped
              ? 'The atomics keep up with it here.'
              : `The atomics manage ${fmt(cap / bound.rate)} times fewer updates than that.`}
          </p>
          <p className="muted" style={{ margin: '0.4rem 0 0' }}>
            {dist === 'bright'
              ? `Our sky picture from section 9.1, in ${bins} bins: the busiest holds ${fmt(100 * share)}% of the pixels.`
              : dist === 'uniform'
                ? `Uniform data over ${bins} bins: each holds 1/${bins} of the updates.`
                : dist === 'constant'
                  ? 'Every pixel the same: one bin takes every update.'
                  : 'Your share for the busiest bin; the others share the rest.'}
          </p>
        </div>
        <div className="panel">
          <LogBars
            rows={[
              { label: 'one location', v: one, tone: 'var(--ink-3)' },
              { label: 'whole histogram', v: bound.rate, tone: 'var(--ink)' },
              { label: 'memory, elements', v: cap, tone: 'var(--global)' },
            ]}
          />
        </div>
      </div>
    </Bench>
  );
}

function MeasuredSkew({ predicted }: { predicted: number }) {
  const { reports } = useReports();
  const found = measuredSkew(reports);
  if (!found.length)
    return (
      <p data-part="skew-none">
        No measurement on this machine yet. Run <code>python3 labs/pmpp.py run 9.1 --solution --bench</code> and reload
        this page: it times the basic kernel on uniform bytes and on a constant image, and this box will compare your
        prediction with the result.
      </p>
    );
  return (
    <>
      {found.map((f) => (
        <p key={f.b} data-part="skew-measured">
          On <b>{f.device}</b> ({f.b}) the basic kernel ran <b>{fmt(f.ratio)}×</b> slower on the constant image ({fmt(f.con)}{' '}
          against {fmt(f.uni)} Gpixel/s).{' '}
          {f.ratio < 1
            ? 'The constant image ran faster here, the opposite of the model: something merged the same-address updates or kept the busy bin close (see the note above), and which of these applies is not documented for every backend.'
            : predicted > f.ratio * 2
            ? 'You expected considerably more. Something absorbed much of the contention the model charges for: caches, or the hardware or compiler merging same-address updates (see the note above). Which of these applies is not documented for every backend.'
            : predicted < f.ratio / 2
              ? 'You expected considerably less: one busy location costs more on this device than you allowed for.'
              : 'That is close to what you predicted.'}
        </p>
      ))}
      <p className="muted">These are measurements from your machine, not expected values. Another device can differ by a wide margin in either direction.</p>
    </>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS93: Record<string, Question> = {
  'ch09/9.3/q-ex1': {
    kind: 'numeric',
    id: 'ch09/9.3/q-ex1',
    prompt: <>Each atomic operation on a DRAM system takes 100 ns in total. What is the most atomics per second one global variable can take?</>,
    answer: 10,
    tol: 0.01,
    unit: 'M atomics/s',
    explain: <p>One at a time, 100 ns each: 1 / 100 ns = 10 million per second, however many threads are waiting.</p>,
    source: 'ch. 9 exercise 1, p. 220',
  },
  'ch09/9.3/q-ex2': {
    kind: 'numeric',
    id: 'ch09/9.3/q-ex2',
    prompt: (
      <>
        The GPU performs atomics in its L2 cache: 4 ns on a hit, 100 ns when the value has to come from DRAM, and 90% of
        the atomics hit. Roughly how many atomics per second can one global variable take?
      </>
    ),
    answer: 73.5,
    tol: 0.01,
    unit: 'M atomics/s',
    explain: (
      <p>
        The mean time is 0.9 × 4 + 0.1 × 100 = 3.6 + 10 = 13.6 ns, so about 73.5 million per second. The 10% of misses
        cost almost three times as much as all the hits together: the slow case dominates an average of times.
      </p>
    ),
    source: 'ch. 9 exercise 2, p. 220',
  },
  'ch09/9.3/q-ex3': {
    kind: 'numeric',
    id: 'ch09/9.3/q-ex3',
    prompt: (
      <>
        With exercise 1's atomics (100 ns each on one variable), a kernel performs 5 floating-point operations for every
        atomic. What is the highest floating-point rate the atomics allow?
      </>
    ),
    answer: 50,
    tol: 0.01,
    unit: 'M FLOP/s',
    explain: <p>5 × 10 million = 50 million operations per second, whatever the arithmetic units could do. The atomic is the bottleneck, and the arithmetic waits on it.</p>,
    source: 'ch. 9 exercise 3, p. 220',
  },
  'ch09/9.3/q-ex4': {
    kind: 'numeric',
    id: 'ch09/9.3/q-ex4',
    prompt: (
      <>
        Exercise 3 again, but the variable is privatised into shared memory, where one atomic takes 1 ns in total. All
        the original atomics become shared-memory atomics; the extra commits to the global variable add 10% to the run
        time. Still 5 floating-point operations per atomic. What is the highest floating-point rate now?
      </>
    ),
    answer: 4.55,
    tol: 0.02,
    unit: 'GFLOP/s',
    explain: (
      <>
        <p>
          Each atomic now costs 1 ns × 1.1 = 1.1 ns, about 909 million per second, and 5 × 909 million ≈ 4.5 GFLOP/s: 90
          times exercise 3.
        </p>
        <p>
          This wording is ours. The book says "shared memory access latency is 1 ns", which also allows counting a read
          and a write, 2 ns per atomic, as in the 200 + 200 cycle model above. That reading gives 2.2 ns per atomic and
          about 2.3 GFLOP/s. The book does not say which it means.
        </p>
      </>
    ),
    source: 'ch. 9 exercise 4, p. 220; the ambiguity is noted in notes/ch09.md, "Unclear in the PDF"',
  },
  'ch09/9.3/q-half': {
    kind: 'numeric',
    id: 'ch09/9.3/q-half',
    prompt: (
      <>
        The book's DRAM model: 2.5 M atomics per second on one location. A 256-bin histogram where half of the pixels
        fall in one bin and the rest are spread evenly over the other 255. Roughly how many atomics per second does the
        whole histogram get through?
      </>
    ),
    answer: 5,
    tol: 0.02,
    unit: 'M atomics/s',
    explain: (
      <p>
        The busy bin receives 0.5<i>N</i> of the <i>N</i> updates and takes them at 2.5 M/s, so it needs 0.5<i>N</i> / 2.5
        M seconds. Each other bin receives about 0.002<i>N</i> and finishes long before. <i>N</i> updates in 0.2<i>N</i>{' '}
        microseconds is 5 M/s: 128 times slower than the uniform 640 M/s, because of one bin.
      </p>
    ),
    source: 'our arithmetic on the §9.3 model, pp. 209–210',
  },
  'ch09/9.3/q-more-threads': {
    kind: 'choice',
    id: 'ch09/9.3/q-more-threads',
    prompt: <>Launching more threads made a memory-bound copy kernel faster. Will it speed up the basic histogram kernel on an all-white image?</>,
    options: [
      { label: 'Yes: more threads hide more latency, which is what sped up the copy kernel.', why: 'Latency hiding needs independent requests that can overlap. Atomics on one location cannot.' },
      { label: 'No: one atomic at a time on that bin, so more threads only make the queue longer.', correct: true },
      { label: 'Only if the extra threads are in different blocks, so their atomics start on other SMs.', why: 'Device-scope atomics on one location queue whichever block or SM they come from.' },
    ],
    explain: (
      <p>
        The copy kernel got faster because its requests were independent: more of them in flight meant more overlap.
        Every update to the white bin waits for the one before it, so the bin's rate is fixed by the latency, and adding
        threads adds waiting. The cures change the data's destination (private copies) or the latency (caches, shared
        memory), not the thread count.
      </p>
    ),
    source: 'ch. 9 §9.3, p. 209',
  },
};

export function S93() {
  const q = questionsS93;
  const book = useMemo(() => {
    const one = perLocationRate(1e9, 200);
    return { one, uni: histogramBound(one, 1 / 256).rate };
  }, []);
  return (
    <>
      <p>
        Any part of a massively parallel program that runs one step at a time can dominate its run time. Atomics on one
        location run one at a time by design, so the question is how much they cost.
      </p>
      <Source section="9.3" pages="209" />
      <p>
        Recall how a GPU copes with slow memory. A DRAM access takes hundreds of clock cycles (
        <a href={href('ch05', '5.1')}>chapter 5</a>). The GPU hides that by keeping many accesses in flight: while one
        warp waits, another runs (<a href={href('ch04', '4.6')}>chapter 4</a>), and the DRAM's channels and banks serve
        many requests at once (<a href={href('ch06', '6.2')}>chapter 6</a>). With enough independent requests, the speed
        is set by how many bytes per second the memory can deliver, not by how long one access takes.
      </p>
      <Source section="9.3" pages="209" />
      <p>
        Atomics on one location break that strategy. Only one can be in progress at a time, and each lasts about a load
        latency (to read the old value) plus a store latency (to write the new one). Nothing overlaps, so the location
        takes at most one atomic per load-plus-store latency: roughly 1 / (2 × latency).
      </p>
      <Source section="9.3" pages="209" figs="Fig. 9.7" />

      <h2>Why one busy bin sets the pace</h2>
      <p>
        Watch it as time. Each bar below is one request; its start and end are computed by the same model as the
        explorer further down, with the book's 200-cycle memory.
      </p>
      <Simplification>
        <p>
          One atomic per location at a time, different locations fully in parallel, no transfer delay between the load
          and the store (the book's arithmetic leaves it out too), no caches and no merging of atomics by the compiler.
          Eight threads stand in for a warp; a real warp has 32 lanes on NVIDIA and 32 or 64 on AMD.
          The cache, shared-memory and issue-slot figures are illustrative: the book gives only "tens" and "a few" cycles.
        </p>
      </Simplification>
      <AtomicWaitScene />

      <h2>The book's numbers</h2>
      <p>
        The book's example is a hypothetical memory system: 8 channels, each 8 bytes wide, double data rate, at 1 GHz,
        with an access latency of 200 cycles. Its peak is 8 bytes × 2 transfers × 1 G clocks × 8 channels = 128 GB/s,
        or 32 G four-byte elements per second. One location takes an atomic every 200 + 200 cycles: 1 GHz / 400 ={' '}
        {rate(book.one)}. With 256 bins hit perfectly evenly, the bins work in parallel and the histogram gets through
        256 × 2.5 M = {rate(book.uni)}, still fifty times below what the memory could deliver. And real images are not
        uniform: a bright picture puts most of its pixels in a few bins, and the gain falls far below the number of bins.
      </p>
      <Source section="9.3" pages="209–210" />
      <p>
        The explorer reproduces these numbers and lets you change each assumption: where the atomic is performed, its
        latency, the clock, the number of bins and how the data falls into them.
      </p>
      <Simplification>
        <p>
          The same model as the animation, as a formula: one location runs at clock / (load + store latency); the whole
          histogram at that rate divided by the busiest bin's share, but never faster than the memory delivers elements.
          No caches, no compiler aggregation.
        </p>
      </Simplification>
      <ThroughputExplorer />
      <Ask q={q['ch09/9.3/q-half']} label="Work it out" />
      <Ask q={q['ch09/9.3/q-more-threads']} />

      <h2>Remedy 1: shorten the latency</h2>
      <p>
        If the rate is one atomic per latency, the first remedy is a shorter latency. Modern GPUs perform device-scope
        atomics in the last-level cache, which all SMs share. A value that is in the cache is updated there; one that is
        not is brought in first. A heavily updated bin tends to stay in the cache.
      </p>
      <Source section="9.3" pages="210" />
      <p>
        The book puts a last-level cache access at "tens of cycles rather than hundreds", which improves atomic
        throughput "by at least an order of magnitude". In the explorer, switch to the last-level cache: the
        per-location and whole-histogram rates move by the same factor, until the histogram reaches what the memory can
        deliver.
      </p>
      <Today checked="2026-11-20 (notes/today/ch09-facts.md, A4, B2, C4)">
        <p>
          NVIDIA's Nsight Compute guide: global atomics "are serialized at the L2 atomic stage"; the metric{' '}
          <code>lts__t_sectors_srcunit_tex_op_atom</code> shows the volume. AMD: "Atomics execute directly in the L2
          cache". Apple documents neither where atomics run nor their latency. No vendor gives L2 or DRAM latency in
          cycles, so 200 and "tens" stay the book's example.
        </p>
      </Today>

      <h2>Exercises 1 to 4</h2>
      <p>
        The book's first four exercises are this section's arithmetic. Exercise 4 is given in our wording; its
        explanation says why.
      </p>
      <Ask q={q['ch09/9.3/q-ex1']} label="Exercise 1" />
      <Ask q={q['ch09/9.3/q-ex2']} label="Exercise 2" />
      <Ask q={q['ch09/9.3/q-ex3']} label="Exercise 3" />
      <Ask q={q['ch09/9.3/q-ex4']} label="Exercise 4" />

      <h2>Predict, then measure</h2>
      <p>
        In the book's model, an image whose pixels all have one value makes the basic kernel {fmt(skewSlowdown(book.one, 256, 32e9))}{' '}
        times slower than uniform bytes over 256 bins: every update joins one queue instead of 256. Real hardware may
        disagree, in either direction. Lab 9.1 times both inputs.
      </p>
      <Today checked="2026-11-20 (notes/today/ch09-facts.md, A9, B5)">
        <p>
          An NVIDIA blog post, last updated in 2017, says nvcc merges same-address atomics within a warp "in many cases",
          without saying which. LLVM's AMDGPU backend merges them only when the address is the same across the wavefront
          at compile time, which a histogram's data-dependent address is not. A measured penalty may be far below the
          model's.
        </p>
      </Today>
      <Predict
        id="ch09/9.3/p-skew-slowdown"
        prompt={<>On <em>your</em> GPU, how many times slower will the basic kernel run on a constant image than on uniform random bytes? Commit before you run lab 9.1.</>}
        unit="× slower"
      >
        {(v) => <MeasuredSkew predicted={v} />}
      </Predict>
      <p>
        One more detail decides which instruction the hardware runs. The basic kernel never uses the value{' '}
        <code>fetch_add</code> returns.
      </p>
      <Beyond>
        <p>
          PTX has two instructions for this: <code>atom</code>, which returns the old value, and <code>red</code>, which
          does not. An unused <code>fetch_add</code> result lets the compiler use <code>red</code>; PyTorch calls its
          helper <code>gpuAtomicAddNoReturn</code> for that reason. Which one nvcc emits is undocumented: read the PTX
          (notes/today/ch09-facts.md A9, D1).
        </p>
      </Beyond>

      <h2>The same trap in inference code</h2>
      <p>
        A single accumulator in global memory that every thread updates is the worst case of this section: one location,
        every thread in its queue. Summing the denominator of a softmax that way, one atomic add per thread, is exactly
        that. Chapter 10's multi-block reduction avoids it: each block sums its own segment privately and its leader
        commits once, so the queue on the global total has one entry per block (Fig. 10.18, pp. 244–245). That is the
        shape of the next section.
      </p>
      <Beyond>
        <p>The softmax example is ours; chapter 10 builds the kernel.</p>
      </Beyond>
      <p>
        Lab 9.1 is on the <a href={href('ch09', 'labs')}>labs page</a>; <a href={href('ch09', '9.4')}>section 9.4</a> takes
        the second remedy, fewer threads per location.
      </p>
    </>
  );
}
