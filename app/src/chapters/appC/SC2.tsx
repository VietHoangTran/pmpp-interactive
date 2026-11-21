import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, NoteGroup, Simplification, Source, Today } from '../../components/notes';
import { benchRows, fmt, realBackends, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  BOOK_RULE,
  H100,
  H100_RATIO,
  better,
  breakEven,
  copyFirstCost,
  lifetimeTables,
  mappedCost,
  mixedCurve,
  mixedTime,
  passCurve,
  ratioFromProbe,
} from '../../sim/zerocopy';

export const LAB = 'appC/lab1_feeding';

// ---------------------------------------------------------------- link ratios
interface RatioOption {
  key: string;
  label: string;
  rho: number;
  note: string;
}

export function ratioOptions(reports: Reports): RatioOption[] {
  const out: RatioOption[] = [
    { key: 'book', label: "the book's rule of thumb (10%)", rho: BOOK_RULE, note: 'The book: interconnect bandwidth is "typically less than 10%" of global-memory bandwidth (p. 625).' },
    {
      key: 'h100',
      label: "the book's example device (H100 SXM)",
      rho: H100_RATIO,
      note: `${H100.linkGBps} GB/s of PCIe Gen5 (NVIDIA's page) against ${H100.deviceGBps / 1000} TB/s of device memory (the book, ch. 5 p. 94): about ${fmt(100 * H100_RATIO, 2)}%, our arithmetic (appC-facts A4).`,
    },
  ];
  for (const b of realBackends(reports)) {
    if (b === 'metal' || b === 'race') continue;
    const row = benchRows(reports[b], LAB).find((r) => r.name === 'probe');
    const rho = ratioFromProbe(row);
    if (rho == null) continue;
    out.push({
      key: 'yours-' + b,
      label: `yours: ${String(reports[b].device.name ?? b)} (${b})`,
      rho,
      note: `Lab C.1's probe on ${reports[b].generated.slice(0, 10)}: ${fmt(Number(row!.link_gb_per_s))} GB/s for a pinned host-to-device copy against ${fmt(Number(row!.device_gb_per_s))} GB/s for a device-to-device copy. Copy bandwidths stand in for in-kernel reads.`,
    });
  }
  return out;
}

// ---------------------------------------------------------------- a small log-scale plot
interface Series {
  label: string;
  pts: [number, number][];
  dash?: boolean;
}
function Plot({
  series,
  xLog,
  xDomain,
  yDomain,
  xLabel,
  yLabel,
  marks,
  vline,
  aria,
}: {
  series: Series[];
  xLog: boolean;
  xDomain: [number, number];
  yDomain: [number, number];
  xLabel: string;
  yLabel: string;
  marks: { x: number; y: number; label: string }[];
  vline?: { x: number; label: string };
  aria: string;
}) {
  const W = 460;
  const H = 260;
  const pad = { l: 52, r: 12, t: 12, b: 40 };
  const lx = (v: number) => (xLog ? Math.log10(v) : v);
  const X = (v: number) => pad.l + ((lx(v) - lx(xDomain[0])) / (lx(xDomain[1]) - lx(xDomain[0]))) * (W - pad.l - pad.r);
  const Y = (v: number) =>
    H - pad.b - ((Math.log10(v) - Math.log10(yDomain[0])) / (Math.log10(yDomain[1]) - Math.log10(yDomain[0]))) * (H - pad.t - pad.b);
  const yt: number[] = [];
  for (let e = Math.floor(Math.log10(yDomain[0])); e <= Math.ceil(Math.log10(yDomain[1])); e++)
    for (const m of [1, 2, 5]) {
      const v = m * 10 ** e;
      if (v >= yDomain[0] * 0.999 && v <= yDomain[1] * 1.001) yt.push(v);
    }
  const xt = xLog ? [1, 10, 100, 1000].filter((v) => v >= xDomain[0] && v <= xDomain[1]) : [0, 0.25, 0.5, 0.75, 1];
  const inX = (v: number) => v >= xDomain[0] && v <= xDomain[1];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={aria} style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
      {yt.map((v) => (
        <g key={'y' + v}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          <text x={pad.l - 5} y={Y(v) + 4} fontSize="10" textAnchor="end">
            {fmt(v)}
          </text>
        </g>
      ))}
      {xt.map((v) => (
        <g key={'x' + v}>
          <line x1={X(v)} x2={X(v)} y1={pad.t} y2={H - pad.b} stroke="var(--rule-soft)" />
          <text x={X(v)} y={H - pad.b + 14} fontSize="10" textAnchor="middle">
            {xLog ? v : `${v * 100}%`}
          </text>
        </g>
      ))}
      <text x={(W + pad.l) / 2} y={H - 6} fontSize="11" textAnchor="middle">
        {xLabel}
      </text>
      <text x={12} y={(H - pad.b + pad.t) / 2} fontSize="11" textAnchor="middle" transform={`rotate(-90 12 ${(H - pad.b + pad.t) / 2})`}>
        {yLabel}
      </text>
      {vline && inX(vline.x) ? (
        <g>
          <line x1={X(vline.x)} x2={X(vline.x)} y1={pad.t} y2={H - pad.b} stroke="var(--ink-2)" strokeDasharray="2 3" />
          <text x={X(vline.x) + 4} y={pad.t + 12} fontSize="10" style={{ fill: 'var(--ink-2)' }}>
            {vline.label}
          </text>
        </g>
      ) : null}
      {series.map((s) => (
        <polyline
          key={s.label}
          fill="none"
          stroke={s.dash ? 'var(--ink-2)' : 'var(--ink)'}
          strokeWidth="2"
          strokeDasharray={s.dash ? '6 4' : undefined}
          points={s.pts
            .filter(([, y]) => y >= yDomain[0] * 0.999 && y <= yDomain[1] * 1.001)
            .map(([x, y]) => `${X(x)},${Y(y)}`)
            .join(' ')}
        />
      ))}
      {marks.map((m, i) => (
        <g key={i}>
          <circle cx={X(m.x)} cy={Y(m.y)} r="4.5" fill="var(--page)" stroke="var(--ink)" strokeWidth="2" />
          <text x={i ? X(m.x) + 7 : X(m.x) - 7} y={Y(m.y) - 7} fontSize="10.5" textAnchor={i ? 'start' : 'end'}>
            {m.label}
          </text>
        </g>
      ))}
    </svg>
  );
}

// ---------------------------------------------------------------- explorer
const pct = (v: number) => `${fmt(100 * v, 3)}%`;

function MappedCost() {
  const { reports } = useReports();
  const options = useMemo(() => ratioOptions(reports), [reports]);
  const [key, setKey] = useState('book');
  const [custom, setCustom] = useState(0.05);
  const [f, setF] = useState(0.1);
  const [sExp, setSExp] = useState(6); // s = 1 / 2^sExp
  const [kExp, setKExp] = useState(1); // k = 10^kExp
  const opt = options.find((o) => o.key === key);
  const rho = key === 'custom' ? custom : (opt?.rho ?? BOOK_RULE);
  const s = 1 / 2 ** sExp;
  const k = Math.max(1, Math.round(10 ** kExp));
  const none = mixedTime(f, rho, 'none');
  const perfect = mixedTime(f, rho, 'perfect');
  const top = Math.max(2, 1 / rho);
  const kStar = breakEven(s, rho);
  const m = mappedCost(k, s, rho);
  const c = copyFirstCost(k, s, rho);
  const win = better(k, s, rho);
  const curveB = passCurve(s, rho);
  const yMin = Math.min(...curveB.map((p) => Math.min(p.mapped, p.copy))) * 0.8;
  const yMax = Math.max(...curveB.map((p) => Math.max(p.mapped, p.copy))) * 1.25;
  const hasYours = options.some((o) => o.key.startsWith('yours-'));

  return (
    <Bench
      title="What a mapped read costs"
      task="Choose a link, then move the mapped fraction, the share touched per pass and the number of passes."
      foot={
        <>
          Bandwidth only. Device bandwidth is 1; the link has bandwidth ρ. First plot: a memory-bound kernel that takes a
          fraction f of its bytes over the link, relative to the same kernel with every byte in device memory. Second plot: a
          table read k times, a fraction s per pass, in units of "one table's bytes at device bandwidth".
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field" style={{ maxWidth: '100%' }}>
          Link bandwidth as a share of device bandwidth
          <select style={{ maxWidth: '100%' }} value={key} onChange={(e) => setKey(e.target.value)} aria-label="link">
            {options.map((o) => (
              <option key={o.key} value={o.key}>
                {o.label}
              </option>
            ))}
            <option value="custom">custom</option>
          </select>
        </label>
        {key === 'custom' ? (
          <label className="field">
            ρ = {pct(custom)}
            <input type="range" min={0.01} max={0.5} step={0.005} value={custom} aria-label="custom ratio" onChange={(e) => setCustom(+e.target.value)} />
          </label>
        ) : null}
        <span className="ui" data-rho={rho}>
          ρ = {pct(rho)}
        </span>
      </div>
      <p className="ui muted" style={{ marginTop: 0, fontSize: '0.82rem' }} data-rho-note>
        {key === 'custom' ? 'A ratio you chose.' : opt?.note}
        {hasYours ? null : (
          <>
            {' '}
            No "yours" yet: run lab C.1 with <code>--bench</code> on a CUDA or HIP machine (
            <code>python3 labs/pmpp.py run appC/lab1 --bench</code>) and reload. On Apple silicon there is no link to
            measure.
          </>
        )}
      </p>
      <div className="cols" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(19rem, 1fr))' }}>
        <div>
          <label className="field ui">
            Bytes from mapped host memory: f = {pct(f)}
            <input type="range" min={0} max={1} step={0.01} value={f} aria-label="mapped fraction" onChange={(e) => setF(+e.target.value)} />
          </label>
          <Plot
            aria="Kernel time against the mapped fraction"
            series={[
              { label: 'no overlap', pts: mixedCurve(rho, 'none').map((p) => [p.f, p.t]) },
              { label: 'perfect overlap', pts: mixedCurve(rho, 'perfect').map((p) => [p.f, p.t]), dash: true },
            ]}
            xLog={false}
            xDomain={[0, 1]}
            yDomain={[1, top]}
            xLabel="share of bytes from mapped memory (f)"
            yLabel="time vs all-device"
            marks={[
              { x: f, y: none, label: `${fmt(none)}×` },
              { x: f, y: Math.max(1, perfect), label: `${fmt(perfect)}×` },
            ]}
          />
          <p className="ui" style={{ fontSize: '0.85rem' }} data-mixed>
            Solid: the two streams one after the other, <b>{fmt(none)}×</b>. Dashed: perfectly overlapped,{' '}
            <b>{fmt(perfect)}×</b>. The link carries {pct(f)} of the bytes and takes{' '}
            <b>{pct(f / rho / none)}</b> of the time without overlap.
          </p>
        </div>
        <div>
          <div className="row" style={{ gap: '0.4rem 1rem' }}>
            <label className="field ui">
              Touched per pass: s = 1/{2 ** sExp}
              <input type="range" min={0} max={10} step={1} value={sExp} aria-label="share touched per pass" onChange={(e) => setSExp(+e.target.value)} />
            </label>
            <label className="field ui">
              Passes: k = {k}
              <input type="range" min={0} max={3} step={0.05} value={kExp} aria-label="passes" onChange={(e) => setKExp(+e.target.value)} />
            </label>
          </div>
          <Plot
            aria="Read in place against copy first, by number of passes"
            series={[
              { label: 'read in place', pts: curveB.map((p) => [p.k, p.mapped]), dash: true },
              { label: 'copy, then read', pts: curveB.map((p) => [p.k, p.copy]) },
            ]}
            xLog
            xDomain={[1, 1000]}
            yDomain={[yMin, yMax]}
            xLabel="passes over the table (k)"
            yLabel="cost"
            marks={[
              { x: k, y: m, label: 'in place' },
              { x: k, y: c, label: 'copy first' },
            ]}
            vline={{ x: kStar, label: `break-even ${fmt(kStar)}` }}
          />
          <p className="ui" style={{ fontSize: '0.85rem' }} data-passes data-winner={win}>
            Dashed: read in place, k·s/ρ = <b>{fmt(m)}</b>. Solid: copy once, then read, 1/ρ + k·s = <b>{fmt(c)}</b>.{' '}
            {win === 'mapped' ? 'Reading in place wins' : win === 'copy' ? 'Copying first wins' : 'A tie'}. Break-even at{' '}
            <b data-kstar>{fmt(kStar)}</b> passes. Over its life the kernel reads {fmt(k * s)} tables' worth; in place wins
            below {fmt(lifetimeTables(rho))}.
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSC2: Record<string, Question> = {
  'appC/C.2/q-mixed': {
    kind: 'numeric',
    id: 'appC/C.2/q-mixed',
    prompt: (
      <>
        A memory-bound kernel takes 90% of its bytes from device memory and 10% from mapped host memory. The link has
        10% of device bandwidth, and the two streams do not overlap. How many times longer does the kernel take than
        with every byte in device memory?
      </>
    ),
    answer: 1.9,
    tol: 0.02,
    unit: '×',
    explain: (
      <p>
        0.9 / 1 + 0.1 / 0.1 = 0.9 + 1 = 1.9. One tenth of the bytes takes more than half of the time. Even with perfect
        overlap the kernel cannot run faster than its slower stream, here 1.0: no gain over the all-device kernel at
        all.
      </p>
    ),
    source: 'book p. 625 for the 10% and the reasoning; the arithmetic is ours',
  },
  'appC/C.2/q-breakeven': {
    kind: 'numeric',
    id: 'appC/C.2/q-breakeven',
    prompt: (
      <>
        Each pass reads 1/64 of a table, and the link has 10% of device bandwidth. Above how many passes does copying
        the table first beat reading it in place?
      </>
    ),
    answer: 71.1,
    tol: 0.02,
    unit: 'passes',
    explain: (
      <p>
        In place costs k·s/ρ; copying costs 1/ρ + k·s. They meet at k = 1/(s(1 − ρ)) = 64 / 0.9 ≈ 71.1. From 72 passes
        on, copy first. Put differently: in place wins only while the kernel reads, over its whole life, less than
        about one table's worth.
      </p>
    ),
    source: 'our derivation from the book\'s rule, p. 625',
  },
  'appC/C.2/q-today-pointer': {
    kind: 'choice',
    id: 'appC/C.2/q-today-pointer',
    prompt: (
      <>
        Today, in a 64-bit process on a current CUDA device, you <code>cudaMallocHost</code> a buffer for a kernel to
        read in place. What do you pass to the kernel?
      </>
    ),
    options: [
      {
        label: 'The same pointer.',
        correct: true,
        why: 'Under unified addressing it works directly; cudaHostGetDevicePointer is needed only for registered or write-combined memory.',
      },
      { label: 'The value from cudaHostGetDevicePointer, as in 2009.', why: 'Still accepted, but no longer needed for this memory.' },
      { label: 'Nothing: kernels cannot read pinned memory.', why: 'They can, over the interconnect. That is what this section is about.' },
      { label: 'A cudaMalloc copy of it.', why: 'That is the copy model, which is what reading in place avoids.' },
    ],
    explain: (
      <p>
        NVIDIA: memory from <code>cudaMallocHost</code> and <code>cudaHostAlloc</code> is "always directly accessible
        from all devices that support unified addressing", with or without the Mapped flag.
      </p>
    ),
    source: 'appC-facts A3',
  },
  'appC/C.2/q-pinmemory': {
    kind: 'choice',
    id: 'appC/C.2/q-pinmemory',
    prompt: (
      <>
        Your training loop moves each batch with <code>batch.to("cuda", non_blocking=True)</code>, yet a profile shows
        the host stalling at every copy instead of moving on to the next step. The <code>DataLoader</code> was built with
        its defaults. What explains the stall, and what fixes it?
      </>
    ),
    options: [
      {
        label:
          'The batches are pageable, and a non_blocking copy from pageable memory may block. DataLoader(pin_memory=True) pins them, so the copy runs asynchronously.',
        correct: true,
      },
      {
        label: 'non_blocking=True affects only GPU-to-host copies, not host-to-GPU ones; nothing on the host side fixes the stall, so overlap it with more CUDA streams.',
        why: 'PyTorch: once a tensor is pinned, you can use asynchronous GPU copies by passing non_blocking=True. The flag works; the memory it copies from was the problem.',
      },
      {
        label: 'Have the model read each batch in place from mapped host memory, so there is no copy for the host to wait for at all.',
        why: 'That removes the copy and puts the link under every access the model makes. PyTorch pins memory to make the copy fast and asynchronous, not to read batches in place.',
      },
      {
        label: "Pin the model's weights as well, so every tensor the step touches is page-locked and the copies stop stalling.",
        why: 'The weights already live on the GPU. What crosses the link at each step is the batch, and the batch is what needs page-locked memory.',
      },
    ],
    explain: (
      <p>
        PyTorch's CUDA notes: copies from pinned memory are much faster, pinned tensors allow asynchronous copies with{' '}
        <code>non_blocking=True</code>, and "With non_blocking=True and pageable CPU memory, the call may block the host".
        Under the hood the caching host allocator calls <code>cudaHostAlloc(ptr, size, cudaHostAllocDefault)</code>:
        pinned, and under unified addressing readable by kernels, but used for the copy. Do not pin without limit: the
        same notes warn that overusing pinned memory causes problems when RAM runs low.
      </p>
    ),
    source: 'not in the book; appC-facts D1 (PyTorch source and docs, 2026-11-20)',
  },
};

// ---------------------------------------------------------------- the page
export function SC2() {
  return (
    <>
      <p>
        CUDA 2.2, in 2009, let a kernel read and write host memory directly, over the interconnect, with no{' '}
        <code>cudaMemcpy</code>. The book's recipe has three steps. Allocate with{' '}
        <code>cudaHostAlloc(&amp;h, bytes, cudaHostAllocMapped)</code>. Ask for the device-side value with{' '}
        <code>cudaHostGetDevicePointer(&amp;d, h, 0)</code>. Pass <code>d</code>, not <code>h</code>, to the kernel. Host
        code and device code use two different pointer values for the same bytes.
      </p>
      <Source section="C.2" pages="624" />
      <p>
        This zero-copy memory is pinned memory: the operating system must not page it out while the GPU reads it.
        Chapter 23 explains pinning (<a href={href('ch23', '23.3')}>section 23.3</a> here), and allocates its own pinned
        buffers with <code>cudaMallocHost</code> instead; the book never relates the two calls.
      </p>
      <NoteGroup>
        <Source section="C.2" pages="624" figs="pinning: ch. 23, pp. 562–563" />
        <Aside>
          <p>
            <b>In the PDF</b> the appendix says "we mentioned" that the flag's other values are for advanced use. We found
            no earlier mention in this edition (a search of the text finds <code>cudaHostAlloc</code> only in Appendix C).
          </p>
        </Aside>
      </NoteGroup>
      <p>
        Since then the recipe has shrunk. NVIDIA now calls this <b>mapped memory</b>. Under unified addressing, which
        every 64-bit process has (section C.3), memory from <code>cudaMallocHost</code> or <code>cudaHostAlloc</code>{' '}
        works in kernels through the host's own pointer, with or without the Mapped flag. The two-pointer step survives
        only for memory registered with <code>cudaHostRegister</code> and for write-combined memory.
      </p>
      <Today checked="2026-11-20 against the CUDA 13.5 runtime reference and guide (notes/today/appC-facts.md, A3)">
        <p>
          CUDA 2.2 is from May 2009, as the book says. NVIDIA adds a warning the book does not: atomics on mapped memory
          "are not atomic from the point of view of the host or other GPUs".
        </p>
      </Today>
      <Ask q={questionsSC2['appC/C.2/q-today-pointer']} />

      <h2>What a read over the link costs</h2>
      <p>
        Mapped memory pays the interconnect's long latency and limited bandwidth on every access. The book's rule of
        thumb: the interconnect has "typically less than 10%" of global-memory bandwidth. Chapter 5 showed that a kernel
        without enough reuse runs at the speed of its memory. If most of its bytes now come over the link, the link is
        its memory, and it runs at a tenth of the speed or less.
      </p>
      <NoteGroup>
        <Source section="C.2" pages="625" />
        <Today checked="2026-11-20 against NVIDIA's H100 and Grace Hopper pages (notes/today/appC-facts.md, A4)">
          <p>
            True for PCIe. On the book's example device, H100 SXM, 128 GB/s of PCIe Gen5 against 3.35 TB/s of device
            memory is about 3.8% (our arithmetic). NVIDIA's NVLink-C2C systems narrow the gap a lot. NVIDIA no longer
            states a ratio: measure yours.
          </p>
        </Today>
      </NoteGroup>
      <p>
        A mixed kernel is the realistic case: most bytes from device memory, a few from the host. Work this one out
        before you touch the explorer, then check it there with the book's 10%.
      </p>
      <Ask q={questionsSC2['appC/C.2/q-mixed']} />
      <p>
        The explorer runs the same arithmetic for any link. Try the book's example device and, once lab C.1 has run on
        your CUDA or HIP machine, your own link. On the second plot, watch where the two lines cross as you make each pass
        sparser.
      </p>
      <Simplification>
        <p>
          Bandwidth only, no latency. Bytes as moved, whereas a link moves fixed-size pieces, so sparse reads cost more
          than their bytes. No caching of mapped data on the GPU, and the copy is not overlapped with compute. "Yours"
          uses copy bandwidths as stand-ins for reads inside a kernel.
        </p>
      </Simplification>
      <MappedCost />
      <p>
        Two things to take from the first plot. A small mapped fraction already dominates a memory-bound kernel: at the
        book's 10%, five percent of the bytes adds almost half the run time again. And overlap does not rescue you once the
        link stream is the longer one.
      </p>

      <h2>"Occasionally and sparsely", made exact</h2>
      <p>
        The book's advice is to use mapped memory only for data a kernel touches occasionally and sparsely. The second
        plot turns that into a number. Reading in place costs the bytes you touch, at link speed, every time. Copying
        first costs the whole table once at link speed, then each touch at device speed. Reading in place wins exactly
        while the bytes the kernel reads over its whole life stay below 1/(1 − ρ) tables: a little more than one
        table's worth.
      </p>
      <NoteGroup>
        <Source section="C.2" pages="625" />
        <Beyond>
          <p>
            The break-even rule is our derivation from the book's advice, in the bandwidth-only model of the explorer.
          </p>
        </Beyond>
      </NoteGroup>
      <p>
        An embedding lookup, the book's description of the first step of a transformer (ch. 20, p. 480), is the
        textbook sparse read. Commit to a number:
      </p>
      <Predict
        id="appC/C.2/p-embedding"
        prompt={
          <>
            An embedding table of 1 GiB sits in host memory. One step looks up 64 rows of 16 KiB each. How many times fewer
            bytes cross the link if the kernel reads the rows in place than if you copy the table first?
          </>
        }
        unit="×"
      >
        {(v) => (
          <>
            <p>
              {v === 1024 ? 'Right: ' : 'The answer is 1024: '}2<sup>30</sup> bytes of table against 64 × 2
              <sup>14</sup> = 2<sup>20</sup> bytes of rows.
            </p>
            <p>
              One step is not the whole story. If later steps touch other rows, then after about a thousand steps the
              kernel has read a table's worth over the link, and a single copy would have been cheaper. The next
              question puts a number on that.
            </p>
          </>
        )}
      </Predict>
      <Ask q={questionsSC2['appC/C.2/q-breakeven']} />

      <h2>Where you have met pinned memory</h2>
      <p>
        <code>DataLoader(pin_memory=True)</code> and <code>tensor.pin_memory()</code> allocate exactly this kind of
        memory: PyTorch's caching host allocator calls <code>cudaHostAlloc</code> with <code>cudaHostAllocDefault</code>.
        What PyTorch wants from it is faster copies, and copies that can run asynchronously with{' '}
        <code>non_blocking=True</code>. It still copies each batch to the GPU; nothing reads the batch in place.
      </p>
      <Beyond>
        <p>PyTorch's source and CUDA notes, read on 2026-11-20 (appC-facts D1).</p>
      </Beyond>
      <Ask q={questionsSC2['appC/C.2/q-pinmemory']} />

      <h2>HIP and Metal</h2>
      <p>
        HIP has the same shape: <code>hipHostMalloc</code> with <code>hipHostMallocMapped</code>, and{' '}
        <code>hipHostGetDevicePointer</code> for the device-side value. AMD adds a choice CUDA does not have, coherence.
        Coarse-grained memory is up to date only after a synchronisation; fine-grained memory stays coherent while
        either side changes it, often by leaving it uncached on the GPU. The flags are{' '}
        <code>hipHostMallocCoherent</code> and <code>hipHostMallocNonCoherent</code>, or the environment variable{' '}
        <code>HIP_HOST_COHERENT</code>, and Mapped memory is always fine-grained. Which one you get by default, AMD's own
        page states both ways, so set it explicitly.
      </p>
      <Beyond>
        <p>AMD's HIP host-memory and coherence pages, ROCm 10.2 (appC-facts B2).</p>
      </Beyond>
      <p>
        On Apple silicon there is no interconnect between CPU memory and GPU memory to pay for, because there is one
        memory. A shared buffer, the default there, is read in place by every kernel, and Apple documents no distinction
        between pinned and pageable memory. To use memory you already have, <code>newBufferWithBytesNoCopy</code> wraps
        page-aligned memory as a buffer. llama.cpp does exactly that with memory-mapped model weights: zero copy, applied
        to the weights, on one memory.
      </p>
      <Beyond>
        <p>
          Apple's documentation (appC-facts C2; infra-facts C12; ch02-facts C2) and llama.cpp's Metal backend, read on
          2026-11-20 (appC-facts D4).
        </p>
      </Beyond>
      <p>
        <a href={href('appC', 'labs')}>Lab C.1</a> feeds one kernel three ways on each backend, mapped memory among them,
        and its probe row is where "yours" in the explorer comes from. Back to <a href={href('appC', 'start')}>the map</a>
        : this step is C.2.
      </p>
    </>
  );
}
