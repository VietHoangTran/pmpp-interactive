import type { ReactNode } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Beyond, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { fmt, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { PLAN, useAllChapters } from '../registry';
import { GOALS, gpuBackends, rollup } from '../../sim/booksteps';
import { BOOK, DEVICES, VENDOR, linearLayerRatio, marks, pcieRate, threshold, tileFor, type RowId } from '../../sim/hwtable';

// Chapter 24 has no code, so this page shows no kernel and needs no snippet file.
// Every number its questions state comes from the model the start page runs on
// (sim/hwtable.ts) or from a vendor figure quoted in notes/today/ch24-facts.md;
// Wrap.test.ts pins both.

const row = (id: RowId) => BOOK.rows.find((r) => r.id === id)!;
const device = (key: string) => DEVICES.find((d) => d.key === key)!;
const thr = (key: string) => threshold(device(key).peak, device(key).bw);
const THR_2007 = thr('book2007');
const THR_2025 = thr('book2025');
const THR_TC = thr('book2025tc');

// ---- q-gap: device memory against the host link, inside one column of Fig. 24.1.
export const GAP = {
  y2007: row('membw').from! / row('pcie').from!,
  y2025: row('membw').to / row('pcie').to,
  // The same quotient on NVIDIA's documents for one board each (facts A3, A5, A6, A10, A12).
  vendor2007: VENDOR.g80.membw.value / pcieRate(VENDOR.g80.pcieGen, 16),
  vendor2025: VENDOR.b200.membw.value / pcieRate(VENDOR.b200.pcieGen, 16),
};

// ---- q-perboard: NVIDIA's HGX B200 page gives FP32 for a board of eight (facts A8);
// the reference architecture gives bandwidth per GPU, "up to 8 TB/s" (facts A10).
export const HGX = { boardFp32Tflops: 600, gpus: 8, gpuBandwidthTBs: 8 };
const PER_GPU_THR = threshold((HGX.boardFp32Tflops / HGX.gpus) * 1000, HGX.gpuBandwidthTBs * 1000);
const MIXED_THR = threshold(HGX.boardFp32Tflops * 1000, HGX.gpuBandwidthTBs * 1000);
const OUT128 = marks(1).find((m) => m.id === 'out128')!.ratio;

// ---- q-mi355x: AMD's published peaks for the Instinct MI355X (facts B4, B1).
export const MI355X = { fp32Tflops: 157.3, fp32MatrixTflops: 157.3, fp16MatrixPflops: 2.5, bandwidthTBs: 8 };
const AMD_THR = threshold(MI355X.fp32Tflops * 1000, MI355X.bandwidthTBs * 1000);
const AMD_FP16_THR = threshold(MI355X.fp16MatrixPflops * 1e6, MI355X.bandwidthTBs * 1000);
const AMD_T = tileFor(AMD_THR);

// ---- q-next: the book's 16-bit linear layer at batch 1 (ch. 20 §20.6, p. 505), then 8 and 4 bits.
export const DECODE = {
  b16: linearLayerRatio(1, 1024, 4096, 2),
  b8: linearLayerRatio(1, 1024, 4096, 1),
  b4: linearLayerRatio(1, 1024, 4096, 0.5),
};

export const questionsWrap: Record<string, Question> = {
  'ch24/wrap/q-gap': {
    kind: 'numeric',
    id: 'ch24/wrap/q-gap',
    prompt: (
      <>
        In the 2025 column of the book's table, device memory delivers {fmt(row('membw').to)} GB/s and the host link{' '}
        {row('pcie').to} GB/s each way. A model too large for device memory can still be run by streaming its weights
        from host memory for every token. Per token, how many times longer does moving the weights take that way than
        reading them from device memory?
      </>
    ),
    answer: GAP.y2025,
    tol: 0.02,
    unit: 'times',
    explain: (
      <p>
        {fmt(row('membw').to)} / {row('pcie').to} = {fmt(GAP.y2025)}. In the 2007 column the same quotient is{' '}
        {row('membw').from} / {row('pcie').from} = {fmt(GAP.y2007)}. So the cost of crossing the host link instead of
        reading device memory went from about {Math.round(GAP.y2007)} times to over a hundred. On NVIDIA's own figures
        for one board each it went from {fmt(GAP.vendor2007)} to {fmt(GAP.vendor2025)}: different numbers, same
        direction. Both are peak rates, so this counts the bytes and nothing else. It is the arithmetic behind keeping
        weights, the KV cache and sampling on the device.
      </p>
    ),
    source: 'our analysis on the figures of Fig. 24.1, p. 578; vendor figures from notes/today/ch24-facts.md, A3 to A6, A10, A12',
  },
  'ch24/wrap/q-perboard': {
    kind: 'choice',
    id: 'ch24/wrap/q-perboard',
    prompt: (
      <>
        A colleague writes: "NVIDIA's HGX B200 page gives FP32 as {HGX.boardFp32Tflops} TFLOPS, and memory bandwidth is
        up to {HGX.gpuBandwidthTBs} TB/s per GPU. So the threshold is {HGX.boardFp32Tflops} / {HGX.gpuBandwidthTBs} ={' '}
        {fmt(MIXED_THR)} FLOP/B, and our float32 GEMM with 128 × 128 output tiles, at {fmt(OUT128)} FLOP/B, is
        memory-bound." What is wrong?
      </>
    ),
    options: [
      {
        label: 'Nothing. TFLOPS divided by TB/s is FLOP per byte, so the units check out.',
        why: 'The units do cancel, which is why this slip is easy to miss. Units say nothing about whether both numbers describe the same hardware.',
      },
      {
        label: `The ${HGX.boardFp32Tflops} TFLOPS is for the whole board of ${HGX.gpus} GPUs. Per GPU the threshold is about ${fmt(PER_GPU_THR, 2)} FLOP/B, and the kernel is above it.`,
        correct: true,
      },
      {
        label: 'A GEMM should be measured against the tensor-core figure, not the FP32 one.',
        why: 'Only if the kernel runs on the tensor cores, through a library, CUDA Tile, or TF32 switched on. A float32 GEMM written with ordinary arithmetic is limited by the FP32 figure.',
      },
      {
        label: 'The bandwidth should be the host link, because the matrices start in host memory.',
        why: "The threshold compares arithmetic with the traffic the kernel makes to device memory, where its operands live. The host link is crossed once, before the kernel runs.",
      },
    ],
    explain: (
      <p>
        NVIDIA's HGX page prints {HGX.boardFp32Tflops} TFLOPS for a board of {HGX.gpus}, which is{' '}
        {HGX.boardFp32Tflops / HGX.gpus} per GPU, and its reference architecture prints up to{' '}
        {HGX.gpuBandwidthTBs} TB/s per GPU. {fmt((HGX.boardFp32Tflops / HGX.gpus) * 1000)} GFLOP/s /{' '}
        {fmt(HGX.gpuBandwidthTBs * 1000)} GB/s = {fmt(PER_GPU_THR)} FLOP/B, so a {fmt(OUT128)} FLOP/B kernel is compute-bound by the section 5.1 model. Dividing
        a board's arithmetic by one GPU's bandwidth overstated the threshold {HGX.gpus} times. The book's own table
        makes the same kind of slip twice: its 2007 column takes bandwidth from one board and capacity from another,
        and its NVLink row counts both directions where the PCIe row counts one.
      </p>
    ),
    source: 'method of ch. 5 §5.1, pp. 94–97; output tile from ch. 15 §15.2, p. 351; NVIDIA figures from notes/today/ch24-facts.md, A8, A10',
  },
  'ch24/wrap/q-mi355x': {
    kind: 'numeric',
    id: 'ch24/wrap/q-mi355x',
    prompt: (
      <>
        AMD's page for its Instinct MI355X prints, among other rows: peak FP32 {MI355X.fp32Tflops} TFLOPs; peak FP32
        matrix {MI355X.fp32MatrixTflops} TFLOPs; peak FP16 matrix {MI355X.fp16MatrixPflops} PFLOPs; peak memory
        bandwidth {MI355X.bandwidthTBs} TB/s. What is the section 5.1 threshold for a float32 matmul on this part?
      </>
    ),
    answer: Number(AMD_THR.toFixed(2)),
    tol: 0.02,
    unit: 'FLOP/B',
    explain: (
      <>
        <p>
          {fmt(MI355X.fp32Tflops * 1000)} GFLOP/s / {fmt(MI355X.bandwidthTBs * 1000)} GB/s = {fmt(AMD_THR)} FLOP/B.
          The FP16 matrix figure would give {fmt(AMD_FP16_THR)}, but it is for 16-bit inputs, not your float32 data.
          On this part the FP32 matrix figure equals the plain one, and AMD footnotes TF32 as "support through software
          emulation", so there is no TF32-style jump to choose.
        </p>
        <p>
          A square shared-memory tile would need T = {AMD_T} to reach it. A 128 × 128 output tile ({fmt(OUT128)} FLOP/B)
          clears it; batch-1 generation, about {fmt(DECODE.b16, 2)} FLOP/B in 16 bits, does not. These are AMD's
          published peaks for one product, not your device.
        </p>
      </>
    ),
    source: "method of ch. 5 §5.1, pp. 94–97; AMD's product page as quoted in notes/today/ch24-facts.md, B4 and B1",
  },
  'ch24/wrap/q-next': {
    kind: 'choice',
    id: 'ch24/wrap/q-next',
    prompt: (
      <>
        Token-by-token generation of your model runs on Metal at batch 1 with 16-bit weights, and your measurements put
        its matrix-vector products close to the bandwidth ceiling. Which next step in this course raises the speed that
        ceiling allows it?
      </>
    ),
    options: [
      {
        label: "Chapter 15's register tiling: larger output tiles per thread.",
        why: 'Larger output tiles share fetched operands across more outputs. At batch 1 each weight is used for one output only, so there is nothing to share.',
      },
      {
        label: 'Turning on TF32 for matmul.',
        why: 'It raises a compute ceiling the kernel is far below, and TF32 is not among the types of Metal\'s matmul2d.',
      },
      { label: 'Module X2: store the weights in 8 or 4 bits and dequantise in the kernel.', correct: true },
      {
        label: 'Module X1: move sampling onto the GPU.',
        why: 'That saves copying a row of logits to the host for every token, which is a different cost. The weights are where the bytes are.',
      },
    ],
    explain: (
      <p>
        Two operations per weight and one fetch of it: the ratio is set by the bytes per weight. In the book's example
        layer it is about {fmt(DECODE.b16, 2)} FLOP/B at 16 bits; at 8 bits it is {fmt(DECODE.b8, 2)}, at 4 bits{' '}
        {fmt(DECODE.b4, 2)}. A memory-bound kernel speeds up in proportion, before any arithmetic gets cheaper. The
        other lever is batching, chapters 20 and 22, where one fetched weight serves several sequences. X2's milestone is
        M9.
      </p>
    ),
    source: 'ch. 20 §20.6, p. 505, for the 16-bit layer; the bit widths and the module are ours; notes/today/ch24-facts.md, C1, for matmul2d',
  },
};

// The modules and appendices after chapter 24, with the place in PyTorch's source that
// holds the same kind of code (notes/today/ch24-facts.md, D2 and D3).
const NEXT: { id: string; name: string; pages: string; adds: string; pytorch: ReactNode }[] = [
  {
    id: 'appA', name: 'Appendix A, numerical considerations', pages: 'pp. 583–600',
    adds: 'Why summation order changes a result, and the lower-precision formats that quantisation starts from. Feeds X2.',
    pytorch: <span className="muted">none named here</span>,
  },
  {
    id: 'appC', name: 'Appendix C, CUDA memories, address spaces and pointers', pages: 'pp. 623–629',
    adds: "The book's memory models, set against unified memory on Apple silicon.",
    pytorch: <><code>aten/src/ATen/mps/</code>: the Metal runtime side, among it <code>MPSAllocator.mm</code></>,
  },
  {
    id: 'x1', name: 'X1, sampling kernels', pages: 'not in the book',
    adds: 'Greedy, top-k and top-p sampling on the device, assembled from chapters 10, 11, 12 and 14. Milestone M5.',
    pytorch: <><code>Sort.metal</code>, <code>SortTopK.h</code>, <code>ScanKernel.metal</code> in <code>native/mps/kernels/</code></>,
  },
  {
    id: 'x2', name: 'X2, quantised kernels', pages: 'not in the book',
    adds: "int8 and 4-bit-weight GEMM, anchored on appendix A and chapter 15's intensity analysis. Milestone M9.",
    pytorch: <><code>Quantized.metal</code> in <code>native/mps/kernels/</code></>,
  },
  {
    id: 'x3', name: 'X3, reading real backends', pages: 'not in the book',
    adds: 'Find the kernel behind a given op in llama.cpp, MLX and PyTorch, and explain it.',
    pytorch: <><code>native/cuda/</code>, <code>native/mps/kernels/</code>, and <code>tools/amd_build/</code> for AMD</>,
  },
];
const NAME: Record<string, string> = { metal: 'Metal', cuda: 'CUDA', hip: 'HIP' };
const servedBy = (id: string) => GOALS.filter((g) => g.chapters.includes(id)).map((g) => g.label);

export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const all = useAllChapters();
  const built = new Map((all ?? []).filter((c) => c.sections.length).map((c) => [c.id, c]));
  // A link where the target has pages, plain text where it is still planned.
  const to = (id: string, text: ReactNode) => {
    const c = built.get(id);
    return c ? <a href={href(c.id, c.sections[0].id)}>{text}</a> : <>{text}</>;
  };

  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch24/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;

  const record = Object.values(rollup(PLAN, progress.sections, reports));
  const sectionsDone = record.reduce((n, r) => n + r.sectionsDone, 0);
  const sectionsWritten = (all ?? []).reduce((n, c) => n + c.sections.length, 0);
  const labsRun = record.reduce((n, r) => n + r.labs, 0);
  const labsPassing = record.reduce((n, r) => n + r.labsPassing, 0);
  const gpus = gpuBackends(reports);

  return (
    <>
      <h2 style={{ marginTop: 0 }}>What to carry forward</h2>
      <p>
        Chapter 24 teaches no new technique. What it leaves is one table and three habits for reading any table like
        it, which you will need every time a data sheet or a benchmark claim reaches you.
      </p>
      <Source section="24.2" pages="578" figs="Fig. 24.1" />
      <ul data-carry>
        <li>
          <b>Recompute before you quote.</b> Divide every ratio again, and check that both numbers of a quotient count
          the same thing: one board, one direction, one kind of arithmetic. The book's own table fails once on
          arithmetic and once on units, and mixes boards and directions besides.
        </li>
        <li>
          <b>Ask what a figure counts.</b> A published figure is a peak, not a measurement, and it comes with
          qualifiers: "up to", per GPU or per board, dense or with sparsity, TF32 or FP32. Numbers for your own device
          come from your own runs (lab 5.1, <code>python3 labs/pmpp.py doctor</code>), never from a data sheet.
        </li>
        <li>
          <b>Read the rows as limits on your kernel.</b> The top two give the threshold of section 5.1. On the book's
          figures it rose from about {fmt(THR_2007, 2)} FLOP/B to {fmt(THR_2025)} in FP32 and to {fmt(THR_TC)} against
          the tensor-core figure, which no shared-memory tile reaches: the reuse has to come from register tiles,
          batching or fewer bytes per value. The host link grew least, so cross it once: keep weights, the KV cache and
          sampling on the device. That last step is our reading; the book only prints the numbers.
        </li>
      </ul>

      <h2>A mixed check</h2>
      <p>
        These put the three habits to work on new numbers and new questions about the table.
        {asked ? ` So far in this chapter your latest answer is right on ${right} of the ${asked} questions you have tried.` : ''}
      </p>
      <Ask q={questionsWrap['ch24/wrap/q-gap']} />
      <Ask q={questionsWrap['ch24/wrap/q-perboard']} />
      <Ask q={questionsWrap['ch24/wrap/q-mi355x']} />
      <Ask q={questionsWrap['ch24/wrap/q-next']} />

      <h2>The book's exercises</h2>
      <p>
        There are none: chapter 24 has no code and no exercise section. What it has is a map of the book and one table,
        and each of their numbers is worked on the <a href={href('ch24', 'start')}>start page</a>.
      </p>
      <table className="data wide" style={{ maxWidth: '58rem' }} data-table="worked">
        <thead>
          <tr><th>The book's number</th><th>Page</th><th>Where it is worked here</th></tr>
        </thead>
        <tbody>
          <tr><td>Three steps: chapters 2 to 6, 7 to 15 and 16 to 23</td><td>577–578</td><td>"The book's three steps" and the first question</td></tr>
          <tr><td>232x and 638x for compute throughput</td><td>578</td><td>"Check the table", and the question on what would make 638x right</td></tr>
          <tr><td>93x for memory bandwidth, 128x for capacity</td><td>578</td><td>"Check the table", and its ratios recomputed on NVIDIA's figures</td></tr>
          <tr><td>8x for the host link</td><td>578</td><td>the first prediction, and the PCIe generation control of the recomputed ratios</td></tr>
          <tr><td>1,800 TB/s for NVLink</td><td>578</td><td>"Check the table", and its "NVLink counted" control</td></tr>
          <tr><td>638× and 93× in the running text</td><td>578</td><td>the same check: the text repeats the table</td></tr>
        </tbody>
      </table>

      <h2>Where the PDF looks wrong</h2>
      <p>
        Four pages, and most of the slips sit in the one table. Two cells contradict the table's own arithmetic or
        units (the first two items below). One label and the 2007 host-link generation are wrong against NVIDIA's
        documents (the next two). Four more items describe the hardware more loosely than the table looks, and the
        last two are wording and a loose grouping. Each correction comes from NVIDIA's documents or from arithmetic
        on them (PCIe rates from the Linux kernel's table, facts A5); nothing was measured. These are our readings of
        one PDF, so if your copy differs, trust your copy.
      </p>
      <Today checked="2026-11-20 (ch24-facts A1, A3 to A9, A11, A13, A15)">
        <p>
          Two points stay open: 86.4 GB/s rests on a forum post, and no NVIDIA page we read gives the 8800 GTX's PCI
          Express generation.
        </p>
      </Today>
      <ul data-errata>
        <li>
          Fig. 24.1 and the text beside it (p. 578) give the tensor-core ratio as 638x. On the printed inputs 2200 /
          0.345 = 6,377; against 0.3456, the peak NVIDIA's own factors give, it is 6,366. NVIDIA prints 2.2 petaFLOPS
          for that row, so the ratio is the wrong cell.
        </li>
        <li>NVLink is printed as "1,800 TB/s". NVIDIA's figure is 1,800 GB/s, and it counts both directions, where the PCIe row counts each way: 900 GB/s against 64.</li>
        <li>The row label "Tensor Core FP32" is not a precision NVIDIA lists. The figure is TF32 on Tensor Cores, counted with sparsity; dense is half.</li>
        <li>"Gen2 x16" for the 2007 board: NVIDIA's specification for the 1.5 GB Tesla C870 says generation 1, 4 GB/s each way, which makes the host-link growth about 16x, not 8x. No NVIDIA page we read gives the 8800 GTX's generation.</li>
        <li>The 2007 column mixes two boards. 1.5 GB is the Tesla C870's. NVIDIA's pages we read do not print 86.4 GB/s; it matches the 768 MB GeForce 8800 GTX's 384-bit bus at 900 MHz and a 2007 forum post quoting NVIDIA's table. The C870's 800 MHz memory gives 76.8 GB/s.</li>
        <li>The 2025 column's 80 TFLOPS matches NVIDIA's GB200 NVL72 rack figure divided by 72, and its 192 GB matches the "up to" of one technical brief. For shipped HGX B200 boards NVIDIA's documents give 75 TFLOPS and 180 GB per GPU.</li>
        <li>The column heads' years. The G80 reached shops in November 2006 and the first CUDA Toolkit followed in June 2007, so "2007" is the software date. The B200 was announced in March 2024, and NVIDIA's first quarter of Blackwell volume sales ended in January 2025. NVIDIA's pages already list newer parts.</li>
        <li>The table does not say what its figures are. All of them are published peaks, none a measurement.</li>
        <li>Wording, as printed: "a 18-year comparison" (caption, p. 578), "massively computing devices" (p. 578, presumably massively parallel), "the answer is a definitely no" (p. 579).</li>
        <li>Not an error, a loose grouping: chapters 22 and 23 count as "advanced patterns and applications" (pp. 577–578).</li>
      </ul>

      <h2>What this chapter left out</h2>
      <p>
        The start page keeps the three steps and the table. Everything else in the chapter is the authors' outlook,
        and it stays in the book.
      </p>
      <Source section="24.2" pages="578–580" />
      <ul data-left>
        <li>The impact claimed for HPC, AI, data analytics, finance, manufacturing and medicine, and chapters 19 and 20 as the deep-learning examples (p. 578).</li>
        <li>
          The growth in GPUs per job, hyperscalers and the networking that followed (p. 579). The authors write of tens
          or even hundreds of thousands of GPUs in each training job. NVIDIA confirms a training cluster of 100,000
          GPUs; that one job used all of them, and the start date of 2017, we could not verify (ch24-facts A17). NCCL
          itself is in {to('ch23', 'chapter 23')} (pp. 563–568), which this course condenses.
        </li>
        <li>How the range of problems broadened since the first edition of 2010, from dense matrices and Monte Carlo to sparse and graph methods (p. 579).</li>
        <li>
          The thesis that computing assisted by instruments now leads, with its three examples: GPS and self-driving,
          MRI, PET and individualised medicine, light sources and computed lithography masks (pp. 579–580), and its
          extension to content authoring, drug discovery and weather (p. 580).
        </li>
        <li>Parallel access to storage as the next opening, and the closing remarks (p. 580).</li>
        <li>Page 581 is blank and p. 582 is the divider for Part 4, the appendices.</li>
      </ul>
      <p>
        Not built here, on purpose: a lab, because nothing in the chapter is a claim a kernel can test (host-to-device
        copies are measured in {to('ch02', "chapter 2's")} labs and come back in {to('ch23', 'chapter 23')}); a narrated
        animation, because nothing in it unfolds in time; and any 2007-against-today column for AMD or Apple. Apple
        publishes no FLOPS figure, so for Apple it cannot be built.
      </p>

      <h2>The project so far</h2>
      <p data-project>
        Chapter 24 has no lab and adds no milestone. Across the course you have marked {sectionsDone} of{' '}
        {sectionsWritten} written sections done.{' '}
        {gpus.length === 0
          ? 'No lab results from a GPU are recorded on this machine yet.'
          : `Of the ${labsRun} labs run on this machine, ${labsPassing} pass in your own work/ copy on ${gpus.map((b) => NAME[b] ?? b).join(' and ')}; a lab you have not started counts as not passing.`}{' '}
        The <a href={href('ch24', 'start')}>steps map on the start page</a> shows the same record chapter by chapter,
        with each goal's milestones.
      </p>

      <h2>Where next</h2>
      <p>
        The book continues with its appendices. Appendix B, deep learning basics (pp. 601–622), is skipped here because
        you already know it. Appendices A and C get their own pages in this course.
      </p>
      <Source pages="582" />
      <p>
        After them come three modules the book does not have, each built from chapters you will have read. Pick by the
        goal you are working on.
      </p>
      <Beyond>
        <p>The modules and their milestones are this course's plan, not the book's.</p>
      </Beyond>
      {/* On a phone the table scrolls inside its own box; the page does not. */}
      <div className="wide" style={{ overflowX: 'auto', maxWidth: '62rem', marginBottom: '1.4rem' }}>
        <table className="data" style={{ margin: 0, minWidth: '34rem' }} data-table="next">
          <thead>
            <tr><th>Next</th><th>What it adds</th><th>Same kind of code in PyTorch</th></tr>
          </thead>
          <tbody>
            {NEXT.map((n) => (
              <tr key={n.id} data-next={n.id}>
                <td style={{ minWidth: '10rem' }}>
                  <b>{to(n.id, n.name)}</b>
                  <div className="muted">{n.pages}</div>
                  <div data-goals>for: {servedBy(n.id).join(', ')}</div>
                </td>
                <td>{n.adds}</td>
                <td>{n.pytorch}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p>
        The last column is where PyTorch keeps the same kind of kernel. CUDA kernels are hand-written in{' '}
        <code>aten/src/ATen/native/cuda/</code> and Metal kernels in <code>aten/src/ATen/native/mps/kernels/</code>. For
        AMD the build translates the CUDA sources with hipify (<code>tools/amd_build/build_amd.py</code>): the same idea
        as this course's one <code>.cu</code> file for CUDA and HIP, done by a translation step instead of a header.
      </p>
      <Today checked="2026-11-20 against the PyTorch repository (ch24-facts D2, D3)">
        <p>
          File names only. Which operators run through these <code>.metal</code> files, and which through MPSGraph in{' '}
          <code>operations/</code>, was not checked; that is module X3's work.
        </p>
      </Today>
    </>
  );
}
