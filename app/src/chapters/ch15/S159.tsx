import { useEffect, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { LabsPage, Reveal, benchRow, defineLabs, measure, type BenchSet } from '../../components/LabKit';
import { Bench, Beyond, Source, Today } from '../../components/notes';
import { fmt, num, type BenchRow, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { FEATURES, TARGETS, summarize, type Cell, type FeatureId, type Status } from '../../sim/gemmhw';

// ---------------------------------------------------------------- questions
export const questionsS159: Record<string, Question> = {
  'ch15/15.9/q-wgmma': {
    kind: 'choice',
    id: 'ch15/15.9/q-wgmma',
    prompt: <>Does a kernel built for sm_90a with WGMMA run on a compute capability 10.0 GPU?</>,
    options: [
      { label: 'No. Code built for an "a" target cannot run on later generations; Blackwell\'s counterpart is tcgen05 with tensor memory.', correct: true },
      { label: 'Yes: newer GPUs run code built for older ones.', why: 'That is the usual rule, and "a" targets are the exception: PTX for them "cannot be run on later generation devices".' },
      { label: 'Yes, but more slowly.', why: 'The rule for "a" targets is that such code cannot be run on later generations at all.' },
      { label: 'Only if it was also built for sm_90.', why: 'An sm_90 build cannot contain WGMMA at all: it "Requires sm_90a".' },
    ],
    explain: (
      <p>
        Architecture-specific targets such as sm_90a "do not follow the onion layer model". A kernel that wants tensor
        cores on both Hopper and data-centre Blackwell needs two code paths, or a library that has them.
      </p>
    ),
    source: 'ch15-facts A6 (PTX ISA 9.5)',
  },
  'ch15/15.9/q-tmem-rtx': {
    kind: 'choice',
    id: 'ch15/15.9/q-tmem-rtx',
    prompt: (
      <>
        One CUDA binary must run its GEMM on the largest tensor-core multiply each GPU offers, on compute capability
        9.0, 10.0 and 12.x (RTX 50). Which code paths does it have to carry?
      </>
    ),
    options: [
      {
        label: 'Three: WGMMA in an sm_90a build, tcgen05 with TMEM in an sm_100a (or sm_100f) build, and warp-level mma for 12.x, chosen by compute capability at run time.',
        correct: true,
      },
      { label: 'One sm_90a build with WGMMA: newer GPUs run code built for older ones.', why: 'That is the usual rule, and "a" targets are its exception: their code "cannot be run on later generation devices".' },
      { label: 'Two: WGMMA for 9.0, and tcgen05 with TMEM for both Blackwell families.', why: '12.x is Blackwell too, but sm_120 is not among tcgen05\'s targets. It has no TMEM.' },
      { label: 'One warp-level mma path, since that runs on all three and uses their tensor cores.', why: 'It does run everywhere, which makes it the fallback. But it keeps every operand and the accumulator in registers, so on 9.0 and 10.0 it leaves the larger multiply unused.' },
    ],
    explain: (
      <p>
        WGMMA exists only in the sm_90a build, and code built for an "a" target does not run on later generations.
        tcgen05 and TMEM belong to the sm_100 and sm_110 families, not to 12.x. So 12.x falls back to warp-level mma.
        Libraries carry all of these and choose at run time, which is part of why calling them is the default.
      </p>
    ),
    source: 'ch15-facts A6 (PTX ISA 9.5 target notes); compute capabilities from ch04-facts A12',
  },
  'ch15/15.9/q-ldgsts': {
    kind: 'choice',
    id: 'ch15/15.9/q-ldgsts',
    prompt: (
      <>
        You rewrote <code>loadTile</code> with <code>cuda::memcpy_async</code>, and Nsight Compute&rsquo;s memory chart
        still shows the shared-memory fills going through the register file. What do you do next?
      </>
    ),
    options: [
      {
        label: 'Check that the build targets compute capability 8.0 or later and look for LDGSTS in the SASS; if it is missing, switch to the __pipeline_memcpy_async primitives, which guarantee it, or to CUTLASS.',
        correct: true,
      },
      { label: 'Nothing: with optimisation on, nvcc fuses the load and the store into LDGSTS.', why: 'The book says so, but no NVIDIA page we read does, and NVIDIA\'s guide says the compiler may emit LDG and STS pairs.' },
      { label: 'Add a __syncthreads() after each copy.', why: 'A barrier after the wait is needed for correctness (ch06-facts A16), but it does not change which instructions do the copy.' },
      { label: 'Nothing: the chart counts every shared-memory store as register-file traffic.', why: 'NVIDIA\'s triage guide reads exactly this picture as "the async-copy path is not actually being exercised".' },
    ],
    explain: (
      <p>
        LDGSTS needs compute capability 8.0. <code>cuda::memcpy_async</code> may produce it but is not guaranteed to;
        the guide says the lower-level primitives ensure it. The chart and the SASS are how you know which you got.
      </p>
    ),
    source: 'ch15-facts A7; ch06-facts A16',
  },
  'ch15/15.9/q-tma-2d': {
    kind: 'choice',
    id: 'ch15/15.9/q-tma-2d',
    prompt: <>Can <code>cuda::memcpy_async</code> fetch the 128 × 8 tile of A through TMA?</>,
    options: [
      { label: 'No. It reaches TMA only for 1D bulk copies; a 2D tile needs a tensor map from cuTensorMapEncodeTiled and cuda::ptx::cp_async_bulk_tensor.', correct: true },
      { label: 'Yes, on compute capability 9.0 and later.', why: 'The hardware is there; the API is not the one the book names. A 128 × 8 tile of a wider matrix is not one contiguous run.' },
      { label: 'Yes, if the tile is padded to a power of two.', why: 'Padding does not make a strided 2D tile into a 1D contiguous copy.' },
    ],
    explain: (
      <p>
        The tile's eight floats per row sit K floats apart in global memory. A bulk tensor copy needs a description of
        that layout, the tensor map, built on the host and passed to the kernel.
      </p>
    ),
    source: 'ch15-facts A8',
  },
  'ch15/15.9/q-progression': {
    kind: 'choice',
    id: 'ch15/15.9/q-progression',
    prompt: <>What does the progression from WMMA to WGMMA to TMEM attack?</>,
    options: [
      { label: 'Register pressure, the limit of section 15.7: the inputs move to shared memory, then the accumulators leave the register file.', correct: true },
      { label: 'Global-memory bandwidth.', why: 'That is what larger tiles, and TMA, are for. These change where the operands of the multiply live.' },
      { label: 'Bank conflicts in shared memory.', why: 'Padding (section 15.6) handles those; these interfaces are about registers.' },
      { label: 'Warp divergence.', why: 'None of the three changes how branches are handled. They change where the operands and results live.' },
    ],
    explain: (
      <p>
        Warp-level MMA keeps every input and output in registers. WGMMA loads its operand tiles from shared memory. With
        tcgen05 the result is written to TMEM. Each step frees registers that section 15.7 found to be the scarce resource.
      </p>
    ),
    source: 'ch. 15 §15.9, p. 369; ch15-facts A6 (NVIDIA\'s triage guide)',
  },
  'ch15/15.9/q-fp32-default': {
    kind: 'choice',
    id: 'ch15/15.9/q-fp32-default',
    prompt: (
      <>
        On a GPU with tensor cores your FP32 M3 kernel runs at a third of <code>cublasSgemm</code>&rsquo;s speed at
        batch 256, the library in its default math mode. Which explanation can you rule out without measuring?
      </>
    ),
    options: [
      {
        label: 'That cuBLAS quietly computes in TF32 on its tensor cores.',
        correct: true,
      },
      { label: 'That cuBLAS picked a kernel tuned for this shape.', why: 'Libraries choose a kernel per shape. That is a live candidate, and a profile of which kernel ran would tell you.' },
      { label: 'That cuBLAS overlaps its tile loads with arithmetic through asynchronous copies.', why: 'Where the hardware has LDGSTS or TMA, a library can use them. A candidate.' },
      { label: 'That cuBLAS tunes tile shapes and pipeline depth beyond the book\'s kernel.', why: 'Also a candidate: the book\'s kernel is one fixed design.' },
    ],
    explain: (
      <p>
        cuBLAS&rsquo;s default mode keeps at least the precision you asked for; TF32 and the BF16x9 emulation are
        opt-in, the latter also through environment variables on compute capability 10.x, so check the math mode lab
        15.6 prints. hipBLAS&rsquo;s XF32 and MPS&rsquo;s reduced precision are opt-in too. What is left is technique.
      </p>
    ),
    source: 'ch15-facts A10, B6, C6',
  },
  'ch15/15.9/q-rdna': {
    kind: 'choice',
    id: 'ch15/15.9/q-rdna',
    prompt: (
      <>
        Your HIP GEMM must use matrix instructions on a CDNA GPU and on an RDNA3 one. The HIP features table shows no
        Matrix Cores on RDNA. What do you build?
      </>
    ),
    options: [
      {
        label: 'Two paths chosen per target at compile time: MFMA builtins for CDNA (gfx9), WMMA builtins (__builtin_amdgcn_wmma_*) for RDNA3 (gfx11).',
        correct: true,
      },
      { label: 'MFMA builtins everywhere: they are AMD\'s matrix instructions.', why: 'MFMA belongs to the CDNA (gfx9) feature sets. RDNA3 has a different family, the WMMA builtins.' },
      { label: 'A matrix path for CDNA only, and plain FMAs on RDNA3, as the table says.', why: 'The table and the compiler disagree. The compiler\'s gfx11 and gfx12 feature sets include WMMA, and llama.cpp calls those builtins.' },
      { label: 'One path written with nvcuda::wmma, compiled through the course\'s HIP header.', why: 'HIP has no nvcuda::wmma. The C++ route on AMD is rocWMMA, spelled differently, so matrix code stays out of the shared .cu.' },
    ],
    explain: <p>Where a vendor&rsquo;s table and its compiler disagree, the compiler decides what your code can use. llama.cpp&rsquo;s <code>mma.cuh</code> makes exactly this split.</p>,
    source: 'ch15-facts B4, D1; infra-facts B12',
  },
  'ch15/15.9/q-handwrite': {
    kind: 'choice',
    id: 'ch15/15.9/q-handwrite',
    prompt: <>When should you write your own GEMM rather than call the library?</>,
    options: [
      { label: 'When the library lacks the operation: a quantised weight format expanded into the register strips, a fused epilogue, or shapes it serves badly. Otherwise call it.', correct: true },
      { label: 'Always, since a hand-written kernel can be specialised to the problem.', why: 'The libraries already carry every optimisation in this chapter and more (tensor cores, TMEM, asynchronous copies).' },
      { label: 'Never; the techniques are only worth knowing for other kernels.', why: 'The book says programmers rarely write GEMM from scratch, not never: a library cannot run an operation it does not have.' },
    ],
    explain: (
      <p>
        The book's point is that the libraries embody this chapter (p. 368). Ours, for your goals: the reason to hand-write
        is a format or fusion the library does not offer, which is where module X2 goes.
      </p>
    ),
    source: 'ch. 15 §15.9, p. 368; when to hand-write is our reading',
  },
};

// ---------------------------------------------------------------- explorer: what can this target use?
const STATUS_TEXT: Record<Status, string> = { yes: 'yes', no: 'no', undocumented: 'not documented' };
const STATUS_CHIP: Record<Status, string> = { yes: 'ok', no: 'fail', undocumented: 'global' };

function CellView({ c }: { c: Cell }) {
  return (
    <>
      <span className={'chip ' + STATUS_CHIP[c.status]} data-status={c.status}>{STATUS_TEXT[c.status]}</span>{' '}
      <span>{c.what}</span> <span className="muted" style={{ fontSize: '0.8rem' }}>({c.items.join(', ')})</span>
    </>
  );
}

function HardwareBench() {
  const [a, setA] = useState('cuda-9.0a');
  const [b, setB] = useState('');
  const A = summarize(a);
  const B = b ? summarize(b) : null;
  const narrow = useNarrow();
  const differs = (f: FeatureId) => !!B && B.cells[f].status !== A.cells[f].status;
  const pick = (v: string, set: (x: string) => void, label: string, allowNone: boolean) => (
    <label className="field" style={{ maxWidth: '100%' }}>
      {label}
      <select value={v} aria-label={label} onChange={(e) => set(e.target.value)} style={{ maxWidth: '100%' }}>
        {allowNone ? <option value="">nothing</option> : null}
        {(['CUDA', 'HIP', 'Metal'] as const).map((bk) => (
          <optgroup key={bk} label={bk}>
            {TARGETS.filter((t) => t.backend === bk).map((t) => (
              <option key={t.id} value={t.id}>{t.label}</option>
            ))}
          </optgroup>
        ))}
      </select>
    </label>
  );
  return (
    <Bench
      title="What can this target use?"
      task="Find a target where WGMMA code cannot run, one without TMEM, one with undocumented copies."
      foot={
        <>
          Every cell comes from a tested table (src/sim/gemmhw.ts) built from notes/today/ch15-facts.md. A bare item such
          as A6 is in ch15-facts; "infra", "ch03" and "ch06" name the other facts files; "ch15 table" is that file's
          backend table. "Not documented" describes the documentation, not the silicon.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        {pick(a, setA, 'Target', false)}
        {pick(b, setB, 'Compare with', true)}
      </div>
      {A.toolkit || B?.toolkit ? (
        <p style={{ margin: '0 0 0.7rem' }} data-toolkit>
          {A.toolkit ? <><b>{A.target.label}:</b> <CellView c={A.toolkit} /></> : null}
          {B?.toolkit ? <><br /><b>{B.target.label}:</b> <CellView c={B.toolkit} /></> : null}
        </p>
      ) : null}
      {narrow ? (
        // On a phone: one card per feature, the targets stacked inside it.
        <div data-hw-cards>
          {FEATURES.map((f) => (
            <div key={f.id} className="panel" data-feature={f.id} style={{ marginBottom: '0.6rem' }}
              data-differs={B && differs(f.id) ? 1 : undefined}>
              <p style={{ margin: '0 0 0.4rem' }}>
                <b>{f.label}</b> <span className="muted">({f.book})</span>
                {B && differs(f.id) ? <b className="ui" style={{ fontSize: '0.8rem' }}> differs</b> : null}
              </p>
              <p style={{ margin: 0 }} data-col="a">
                {B ? <span className="muted">{A.target.label}: </span> : null}
                <CellView c={A.cells[f.id]} />
              </p>
              {B ? (
                <p style={{ margin: '0.4rem 0 0' }} data-col="b">
                  <span className="muted">{B.target.label}: </span>
                  <CellView c={B.cells[f.id]} />
                </p>
              ) : null}
            </div>
          ))}
        </div>
      ) : (
        <table className="data" style={{ margin: 0 }} data-hw-table>
          <thead>
            <tr>
              <th>Feature (the book's name)</th>
              <th>{A.target.label}</th>
              {B ? <th>{B.target.label}</th> : null}
            </tr>
          </thead>
          <tbody>
            {FEATURES.map((f) => (
              <tr key={f.id} data-feature={f.id} data-differs={B && differs(f.id) ? 1 : undefined}>
                <td style={{ minWidth: '9rem' }}>
                  {f.label} <span className="muted">({f.book})</span>
                  {B && differs(f.id) ? <><br /><b className="ui" style={{ fontSize: '0.8rem' }}>differs</b></> : null}
                </td>
                <td data-col="a"><CellView c={A.cells[f.id]} /></td>
                {B ? <td data-col="b"><CellView c={B.cells[f.id]} /></td> : null}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Bench>
  );
}

/** True below the width at which the app's own layout switches to one column (52rem). */
function useNarrow() {
  const query = '(max-width: 52rem)';
  const [narrow, setNarrow] = useState(() => typeof window !== 'undefined' && window.matchMedia(query).matches);
  useEffect(() => {
    const mq = window.matchMedia(query);
    const on = () => setNarrow(mq.matches);
    mq.addEventListener('change', on);
    return () => mq.removeEventListener('change', on);
  }, []);
  return narrow;
}

// ---------------------------------------------------------------- your M3 against the library
// Lab 15.6 (milestone M3, labs/ch15/lab6_linear) records rows "m2", "m3" and "vendor" with
// batch, in and out, on the 1024 -> 4096 layer and on a 4096 -> 4096 one; this page reads the
// first. The m, k, n form is how labs 15.1 to 15.5 report the same layer.
const LABS = defineLabs('ch15', [{ num: '15.6', dir: 'lab6_linear', name: 'M3 fast linear layer' }] as const);
const [M3] = LABS;
const BATCH = 256;
const atBatch = (r: BenchRow) => num(r.m) === BATCH && num(r.k) === 1024 && num(r.n) === 4096;
const rowAt = (s: BenchSet, name: string) =>
  benchRow(s, name, { batch: BATCH, in: 1024, out: 4096 }) ?? s.rows.find((r) => r.name === name && atBatch(r));

/** Speed of a row: GFLOP/s if reported, else 1 / seconds (same problem, so the ratio is the same). */
const speed = (r: BenchRow | undefined) => num(r?.gflops) ?? (num(r?.seconds) ? 1 / num(r?.seconds)! : undefined);

function percentOfVendor(reports: Reports) {
  return measure(reports, M3, (s: BenchSet) => {
    const mine = rowAt(s, 'm3');
    const lib = rowAt(s, 'vendor');
    const a = speed(mine);
    const v = speed(lib);
    if (!a || !v) return null;
    const unit = num(mine?.gflops) != null && num(lib?.gflops) != null ? 'GFLOP/s' : 'runs per second';
    return { value: (100 * a) / v, detail: `${fmt(a)} against ${fmt(v)} ${unit} at batch ${BATCH}` };
  });
}

export function S159() {
  return (
    <>
      <p>
        The book ends the chapter by pointing out that programmers rarely write this kernel from scratch: vendors ship
        libraries with all of these optimisations and more, on hardware features the book only names. This page sorts
        out what the libraries and the hardware add, which of it each backend lets your own kernel use today, and how far
        your kernel is from the library.
      </p>

      <h2>Libraries</h2>
      <p>
        NVIDIA's are cuBLAS, its implementation of the standard BLAS interface, matrix multiply included; cuDNN, routines
        for deep neural networks that multiply matrices underneath; CUTLASS, an open-source library for building matrix
        multiplication into larger kernels at several levels; and cuTile, where you write array code and the compiler does
        the optimising.
      </p>
      <Source section="15.9" pages="368" />
      <p>
        With all that available, programmers rarely write a GEMM from scratch. The techniques are worth having anyway,
        because they apply far beyond GEMM.
      </p>
      <Today checked="2026-11-20 (ch15-facts A9, B6, B7, C5, C6; infra-facts A12)">
        <p>
          cuTile is now CUDA Tile (Python; C++ from toolkit 13.3). AMD: hipBLAS, rocBLAS, Composable Kernel, rocWMMA.
          Apple: MPS and Metal Performance Primitives, no template GEMM library.
        </p>
      </Today>
      <p>
        There is a reason to write your own that the book leaves implicit, and it is the one that matters for your work:
        the library does not have the operation. A quantised weight format that has to be expanded into the register
        strips of section 15.4, an epilogue fused into the store, a shape the library serves badly. Otherwise, call the
        library. Module X2 writes the first kind.
      </p>
      <Beyond>
        <p>When to hand-write is our reading, built on the book's remark that programmers rarely write GEMM from scratch.</p>
      </Beyond>
      <Ask q={questionsS159['ch15/15.9/q-handwrite']} />

      <h2>Tensor cores</h2>
      <p>
        From Volta on, some NVIDIA GPUs have tensor cores: units that each perform a small matrix multiplication in one
        instruction. You reach them through the compiler, through a special interface or, most often, through cuBLAS,
        cuDNN and CUTLASS.
      </p>
      <Today checked="2026-11-20 against the cuBLAS 13.5 docs and nvcc 13.5 (ch15-facts A5)">
        <p>
          Volta is compute capability 7.0. nvcc 13.5 lists sm_75 and later, no sm_70, though no single NVIDIA sentence
          says the compiler dropped it.
        </p>
      </Today>
      <p>
        The book names two gains. One instruction now does the work of many, which relieves the limit on decoding and
        issuing instructions. And the units have very high throughput, helped by warp-level hardware for exchanging
        register contents and by lower-precision arithmetic.
      </p>
      <Source section="15.9" pages="368" />

      <h2>WMMA, WGMMA and tensor memory</h2>
      <p>
        The book traces three generations of the interface. WMMA (warp matrix multiply and accumulate) works one warp at
        a time. WGMMA (warp group matrix multiply accumulate), from Hopper, has a group of warps cooperate on a larger
        multiplication. It can take its inputs straight from shared memory, which saves registers, and it runs
        asynchronously with other instructions, so overlapping long waits depends less on the compiler. Blackwell adds
        tensor memory, TMEM: on-chip storage for the inputs and outputs of tensor-core operations, which takes still more
        pressure off the registers.
      </p>
      <Source section="15.9" pages="369" />
      <p>
        Read the three as one idea. Section 15.7 finds that registers limit this chapter's kernel: the accumulator tile
        alone takes 64 per thread. Each generation moves something out of the register file. First the inputs go to
        shared memory, then the accumulator goes to TMEM.
      </p>
      <Today checked="2026-11-20 against PTX ISA 9.5 and the Nsight Compute triage guide (ch15-facts A6; infra-facts A12)">
        <p>
          WMMA is still documented for compute capability 7.0 and up, not "pre-Hopper only". WGMMA "Requires sm_90a", and
          code built for an "a" target "cannot be run on later generation devices". tcgen05, with TMEM, targets the sm_100
          and sm_110 families only: not compute capability 12.x (RTX 50). NVIDIA's own summary: warp MMA keeps inputs and
          outputs in registers, WGMMA loads operand tiles from shared memory, and the fifth generation writes its result
          to TMEM.
        </p>
      </Today>
      <Ask q={questionsS159['ch15/15.9/q-progression']} />
      <Ask q={questionsS159['ch15/15.9/q-wgmma']} />
      <Ask q={questionsS159['ch15/15.9/q-tmem-rtx']} />

      <h2>Moving data without the registers</h2>
      <p>
        Once tensor cores make the arithmetic fast, the limit can move to getting data between global and shared memory:
        the bandwidth, the instructions spent moving it, and the registers it occupies on the way. Vector stores (section
        15.5) help: each thread stores more per instruction, with less overhead and better coalescing. But copying from
        global to shared memory with a vector load and a vector store still takes two steps through registers.
      </p>
      <Source section="15.9" pages="369" />
      <p>
        LDGSTS is one instruction that loads from global memory straight into shared memory, without registers, and the
        hardware does it in the background, so overlapping it with arithmetic no longer depends on the compiler's
        scheduling. The book says nvcc recognises a vector load followed by a shared-memory store and emits LDGSTS on its
        own, and that <code>cuda::memcpy_async</code> from libcu++ uses it. TMA, from Hopper, does the same for whole
        multi-dimensional tensors, again without registers, through <code>cuda::memcpy_async</code> or the libraries.
      </p>
      <p>
        Two of those claims do not hold as written. NVIDIA documents no such fusion: its guide says the compiler may
        emit LDG and STS pairs for ordinary code, and its triage guide says to convert the code yourself or use CUTLASS.{' '}
        <code>cuda::memcpy_async</code> may produce LDGSTS but is not guaranteed to; the{' '}
        <code>__pipeline_memcpy_async</code> primitives are. And <code>cuda::memcpy_async</code> reaches TMA only for
        one-dimensional bulk copies into the block's own shared memory. A two-dimensional GEMM tile needs a tensor map
        built on the host with <code>cuTensorMapEncodeTiled</code> and <code>cuda::ptx::cp_async_bulk_tensor</code>. To
        know what you got, look: LDGSTS in the compiled SASS, or Nsight Compute's memory chart.
      </p>
      <Today checked="2026-11-20 against the CUDA 13.5 programming guide, PTX ISA 9.5 and the CCCL docs (ch15-facts A7, A8, A11)">
        <p>
          LDGSTS: compute capability 8.0 and later. TMA: 9.0 and later. The compiler is nvcc and the library is libcu++
          (libcudacxx, now part of CCCL); the book's "multidimensional tenors" means tensors.
        </p>
      </Today>
      <Ask q={questionsS159['ch15/15.9/q-ldgsts']} />
      <Ask q={questionsS159['ch15/15.9/q-tma-2d']} />

      <h2>On HIP and Metal</h2>
      <p>
        The book covers NVIDIA only. On AMD the matrix instructions are compiler builtins. CDNA
        GPUs have MFMA instructions, including one with FP32 inputs and outputs; CDNA 4 (gfx950) lacks the
        reduced-precision xf32 form. RDNA3 and RDNA4 have WMMA builtins, even though the HIP features table shows no
        Matrix Cores on RDNA. rocWMMA is the C++ library route. For asynchronous copies, the intrinsic AMD's guide names
        does not exist in ROCm 10.2, and the global-to-LDS builtin that does exist is not in the HIP guide, so this course
        does not teach it. On HIP, section 15.8's software pipeline stays in its software form.
      </p>
      <Today checked="2026-11-20 against the ROCm 10.2 compiler sources and HIP docs (ch15-facts B4, B5; infra-facts B12)">
        <p>
          <code>__builtin_amdgcn_mfma_f32_16x16x4f32</code> takes and returns FP32; the RDNA forms are{' '}
          <code>__builtin_amdgcn_wmma_*_w32</code> and <code>_w32_gfx12</code>. The guide's{' '}
          <code>__builtin_amdgcn_async_work_group_copy</code> occurs nowhere in the compiler.
        </p>
      </Today>
      <p>
        Metal has <code>simdgroup_float8x8</code> with <code>simdgroup_multiply_accumulate</code> on Apple7 and later and,
        with Metal 4, Metal Performance Primitives' <code>matmul2d</code>, which supports float × float into float and,
        from OS 26.4, 4-bit integer formats. Apple documents no matrix unit behind either, so this course never calls
        them tensor cores. Metal documents no asynchronous copy at all.
      </p>
      <Today checked="2026-11-20 against the MSL 4.1 specification (ch15-facts C5; infra-facts C10; ch06-facts C7)">
        <p>Apple recommends <code>matmul2d</code> over <code>simdgroup_matrix</code>. The MSL 4.1 specification has no "async".</p>
      </Today>
      <p>
        For your own code the rule is the one in this course's list of portability traps: tensor and matrix-core
        intrinsics come from different libraries on each vendor, with no shared spelling, so they stay out of the shared{' '}
        <code>.cu</code> file that serves CUDA, HIP and the emulators.
      </p>
      <HardwareBench />
      <p>
        Things to find: WGMMA code built for an H100 does not run on a compute capability 10.0 GPU; an RTX 50 has no
        TMEM; an RDNA3 card does have matrix instructions; and nothing on Metal or HIP documents an asynchronous copy you
        can call.
      </p>
      <Ask q={questionsS159['ch15/15.9/q-rdna']} />

      <h2>How real backends reach the hardware</h2>
      <p>
        llama.cpp's CUDA backend does not use WMMA. Its <code>mma.cuh</code> issues PTX <code>mma.sync</code> and{' '}
        <code>ldmatrix</code> instructions directly on NVIDIA, and the MFMA and WMMA builtins on AMD, from the same file.
        PyTorch sends a float matmul on CUDA to cuBLASLt or cuBLAS, and on ROCm can send it to Composable Kernel.
      </p>
      <Beyond>
        <p>Read from the two projects' sources on 2026-11-20 (ch15-facts D1, D4).</p>
      </Beyond>

      <h2>Your kernel against the library</h2>
      <p>
        Lab 15.6, milestone M3, times your fast linear layer and the vendor library side by side on the 1024 → 4096 layer
        (<a href={href('ch15', 'labs')}>labs</a>). All three vendor baselines compute FP32 by default, so whatever gap
        you see is technique, not precision. Commit a number first.
      </p>
      <Today checked="2026-11-20 (ch15-facts A10, B6, C6; not in the book)">
        <p>
          cuBLAS: TF32 and BF16x9 are opt-in. hipBLAS: XF32 is opt-in. MPS: reduced precision is an option you set.
        </p>
      </Today>
      <LabsPage labs={LABS}>
        <Predict
          id="ch15/15.9/p-m3-vs-vendor"
          prompt={<>At batch {BATCH}, what percentage of the vendor library's speed will your M3 kernel reach on your device?</>}
          unit="% of the library"
        >
          {(v) => (
            <Reveal
              committed={v}
              measure={percentOfVendor}
              unit="% of the library"
              reading={
                <>
                  100% would be the library's speed. These are measurements on your machine with whichever kernel passed
                  its tests, not expected values; the library uses techniques this chapter only names.
                </>
              }
              missing={
                <>
                  Run <code>python3 labs/pmpp.py run 15.6 --solution --bench</code> on a machine with a GPU for the
                  reference solution (or <code>run 15.6 --bench</code> once your own kernel passes), then reload.
                </>
              }
            />
          )}
        </Predict>
      </LabsPage>
      <Ask q={questionsS159['ch15/15.9/q-fp32-default']} />
    </>
  );
}
