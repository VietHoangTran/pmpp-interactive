import { useEffect, useMemo, useState, type CSSProperties } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabExcerpt } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Today } from '../../components/notes';
import { href } from '../../lib/router';
import { MECHANISMS, PROJECT_NAME, BACKEND_NAME, VARIANT_MECHANISMS, gradeSort, type Mechanism } from '../../sim/backendmap';
import {
  BLOCK_CHOICES,
  N_READS_CHOICES,
  PATH_LABEL,
  STRIDE_GROUPS,
  STYLES,
  WARP,
  divergentWarps,
  footprint,
  launch,
  pathCounts,
  tailSummary,
  type Launch,
  type LaunchCfg,
  type Path,
  type Style,
} from '../../sim/elemshapes';
import type { Excerpt as ExcerptData } from '../../sim/x3sources';
import { Excerpt } from './Excerpt';
import { OpMap } from './OpMap';

// Section X3.2: elementwise and gather ops in the three projects, read as chapter 2's kernel
// with chapter 6's coarsening and vector loads, and how each project picks a variant. The
// explorer runs on sim/elemshapes.ts, the sort on VARIANT_MECHANISMS in sim/backendmap.ts. The
// excerpts are checked against notes/today/x3-facts.md by excerpts-g1.test.ts.

const FACTS = 'notes/today/x3-facts.md';
const CHECKED = (items: string) => `2026-11-20 in the facts pass (${FACTS}, ${items})`;

export const EXCERPTS_SX32: Record<string, ExcerptData> = {
  unary: {
    id: 'x32-unary', project: 'llama.cpp', path: 'ggml/src/ggml-cuda/unary.cu', lines: [116, 127], factsItem: 'B1', lang: 'cuda',
    code: `template <float (*op)(float), typename T>
static __global__ void unary_op_kernel(const T * x, T * dst, const int k) {
    ggml_cuda_pdl_lc();
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    ggml_cuda_pdl_sync();
    dst[i] = ggml_cuda_cast<T>(op(ggml_cuda_cast<float>(x[i])));
}`,
  },
  unaryMetal: {
    id: 'x32-unary-metal', project: 'llama.cpp', path: 'ggml/src/ggml-metal/kernels/unary.metal', lines: [3, 134], factsItem: 'B1', lang: 'metal',
    code: `constant short FC_unary_op [[function_constant(FC_UNARY + 0)]];
...
if (FC_OP == OP_UNARY_NUM_SILU) {
    dst_ptr[i0] = ggml_metal_unary_cast<T>(x / (1 + exp(-x)));
}`,
  },
  swiglu: {
    id: 'x32-swiglu-loop', project: 'llama.cpp', path: 'ggml/src/ggml-metal/kernels/unary.metal', lines: [290, 309], factsItem: 'B1', lang: 'metal',
    code: `for (int i0 = tgpig.x*ntg.x + tpitg.x; i0 < args.ne0; i0 += ntg.x*ntgs.x)`,
  },
  binaryVv: {
    id: 'x32-binary-vv', project: 'mlx', path: 'mlx/backend/cuda/binary/binary.cuh', lines: [93, 118], factsItem: 'B1', lang: 'cuda',
    code: `template <typename Op, typename In, typename Out, typename IdxT, int N_READS>
__global__ __launch_bounds__(BINARY_MAX_BLOCK_DIM) void binary_vv(
    const In* a, const In* b, Out* out, IdxT size) {
  IdxT index = cg::this_grid().thread_rank();
  if ((index + 1) * N_READS > size) {
    for (IdxT i = index * N_READS; i < size; ++i) {
      out[i] = Op{}(a[i], b[i]);
    }
  } else {
    auto a_vec = load_vector<N_READS>(a, index);
    auto b_vec = load_vector<N_READS>(b, index);
...
    store_vector<N_READS>(out, index, out_vec);
  }
}`,
  },
  loadVector: {
    id: 'x32-load-vector', project: 'mlx', path: 'mlx/backend/cuda/device/utils.cuh', lines: [33, 70], factsItem: 'B1', lang: 'cuda',
    code: `struct alignas(sizeof(T) * N) AlignedVector
...
if (is_aligned<N>(ptr)) {
  auto* from = reinterpret_cast<const AlignedVector<T, N>*>(ptr);
  return from[offset];
} else {
...
}`,
  },
  siluTorch: {
    id: 'x32-silu-torch', project: 'pytorch', path: 'aten/src/ATen/native/cuda/ActivationSiluKernel.cu', lines: [30, 33], factsItem: 'A3', lang: 'cuda',
    code: `gpu_kernel(iter, [] GPU_LAMBDA(scalar_t x) -> scalar_t {
  using opmath_t = at::opmath_type<scalar_t>;
  const opmath_t x_acc = static_cast<opmath_t>(x);
  return x_acc / (opmath_t(1) + ::exp(-x_acc));
});`,
  },
  embedTorch: {
    id: 'x32-embedding-torch', project: 'pytorch', path: 'aten/src/ATen/native/Embedding.cpp', lines: [35, 55], factsItem: 'B7', lang: 'cpp',
    code: `return weight.index_select(0, indices.reshape(-1)).view_symint(size);`,
  },
};

export const questionsSX32: Record<string, Question> = {
  'x3/X3.2/q-read-unary': {
    kind: 'choice',
    id: 'x3/X3.2/q-read-unary',
    prompt: <>Which course kernel is <code>unary_op_kernel</code>, and what does it do differently?</>,
    options: [
      { label: 'Lab 2.3’s elementwise kernel, with the op as a template parameter: one instantiation per unary op.', correct: true },
      { label: 'Lab 6.2’s coarsened kernel: each thread handles several elements.', why: 'Each thread computes i once and writes dst[i]: one element, as in lab 2.3.' },
      { label: 'Lab 10.3’s row softmax, with the reduction removed.', why: 'There is no row and no reduction: it is a flat index with a guard.' },
      { label: 'Lab 2.3’s kernel, but with a grid-stride loop so that a fixed number of blocks covers any number of elements.', why: 'There is no loop. The launch computes enough blocks for k elements, as lab 2.3 does.' },
    ],
    explain: (
      <p>
        Global index, guard, one element: lab 2.3’s three lines. The op is <code>template &lt;float (*op)(float), typename T&gt;</code>,
        so <code>ggml_cuda_op_silu</code> is <code>ggml_cuda_op_unary&lt;op_silu&gt;</code>, and every unary op is a separate
        instantiation compiled into the library. The two <code>pdl</code> calls are the other difference; they do nothing unless
        programmatic dependent launch is on.
      </p>
    ),
    source: `${FACTS} B1; lab 2.3`,
  },
  'x3/X3.2/q-read-binvv': {
    kind: 'choice',
    id: 'x3/X3.2/q-read-binvv',
    prompt: <>Lab 6.2’s <code>vecAdd4Kernel</code> and MLX’s <code>binary_vv</code> look alike. What does <code>binary_vv</code> add?</>,
    options: [
      { label: 'A run-time alignment test: the vector load is taken only when the pointer is aligned for it.', correct: true },
      { label: 'A second launch for the tail, so that no thread of the main launch has to branch on the end of the array.', why: 'Neither launches twice: both handle the partial chunk in the last thread with a scalar loop.' },
      { label: 'Shared memory: each block stages its vectors there first, so that its global loads coalesce.', why: 'There is no shared memory in either: each thread reads its own chunk straight from global memory.' },
      { label: 'Nothing: it is the same kernel with a template for the op.', why: 'The template is new, but so is the alignment test, which lab 6.2 does not have.' },
    ],
    explain: (
      <p>
        Both give each thread N consecutive elements, take a vector load for a whole chunk, and finish a partial last chunk one
        element at a time in the same thread. Lab 6.2 assumes its pointers are 16-byte aligned, which holds for a fresh{' '}
        <code>cudaMalloc</code> buffer. MLX does not assume it:{' '}
        <code>load_vector</code> tests <code>is_aligned&lt;N&gt;(ptr)</code> first and falls back to scalar loads.
      </p>
    ),
    source: `${FACTS} B1; notes/today/infra-facts.md A6; lab 6.2`,
  },
  'x3/X3.2/q-read-fconst': {
    kind: 'choice',
    id: 'x3/X3.2/q-read-fconst',
    prompt: <>In llama.cpp’s <code>unary.metal</code>, what plays the part that the template parameter <code>op</code> plays in <code>unary.cu</code>?</>,
    options: [
      { label: 'The function constant FC_unary_op, tested inside the one kernel body.', correct: true },
      { label: 'A Metal template parameter, exactly as in CUDA.', why: 'MSL has templates, and llama.cpp uses them for types, but the op itself is a function constant here.' },
      { label: 'A kernel argument read from the args buffer at run time.', why: 'An argument could change per dispatch; a function constant is fixed when the pipeline is made.' },
      { label: 'A preprocessor macro set when the library is compiled.', why: 'That is the ubershader approach that function constants replace, in the spec’s own words.' },
    ],
    explain: (
      <p>
        One template, <code>kernel_unary_impl</code>, serves every unary op, and <code>if (FC_OP == OP_UNARY_NUM_SILU)</code>{' '}
        chooses the body. The value is set when the pipeline is created, so each pipeline is a variant made for one op.
      </p>
    ),
    source: `${FACTS} B1, C1`,
  },
  'x3/X3.2/q-tail': {
    kind: 'numeric',
    id: 'x3/X3.2/q-tail',
    prompt: (
      <>
        MLX’s <code>binary_vv</code> with N_READS = 4 runs on 1002 elements. Assume the launch has ceil(1002 / 4) = 251
        threads (the facts did not read MLX’s launch computation). How many elements go through the scalar tail loop?
      </>
    ),
    answer: 2,
    unit: 'elements',
    explain: (
      <p>
        Thread 250 has (250 + 1) × 4 = 1004 &gt; 1002, so it takes the tail branch and loops over elements 1000 and 1001.
        Threads 0 to 249 have (index + 1) × 4 ≤ 1000 and take the vector path. The tail always holds n mod N_READS elements, in
        one thread. The explorer below shows it for any n.
      </p>
    ),
    source: `${FACTS} B1; sim/elemshapes.ts`,
  },
  'x3/X3.2/q-misaligned': {
    kind: 'choice',
    id: 'x3/X3.2/q-misaligned',
    prompt: <>The same call, N_READS = 4, but on a view that starts one float into its buffer. What changes?</>,
    options: [
      { label: 'is_aligned fails in every thread, so each loads its four elements one at a time; the result is unchanged.', correct: true },
      { label: 'The results are wrong: the vector loads read across the view’s start.', why: 'The test exists to stop that. No misaligned vector load is issued.' },
      { label: 'The kernel faults on the first misaligned load.', why: 'It never makes one: the scalar loop runs instead.' },
      { label: 'Only the last thread changes path, because an offset only moves where the tail starts and the other chunks stay whole.', why: 'The offset moves every chunk, so every vector load would be misaligned.' },
    ],
    explain: (
      <p>
        <code>load_vector</code> reinterprets the pointer as an <code>AlignedVector</code> only after <code>is_aligned&lt;N&gt;(ptr)</code>.
        Shifted by 4 bytes, no chunk start is 16-byte aligned, so every thread takes the scalar loop. Each thread still owns the
        same four elements, so within one load instruction a warp now reads every fourth float rather than one unbroken run. Try
        the offset in the explorer.
      </p>
    ),
    source: `${FACTS} B1; sim/elemshapes.ts`,
  },
  'x3/X3.2/q-pdl': {
    kind: 'choice',
    id: 'x3/X3.2/q-pdl',
    prompt: <>What do <code>ggml_cuda_pdl_lc()</code> and <code>ggml_cuda_pdl_sync()</code> do in a HIP build of llama.cpp?</>,
    options: [
      { label: 'Nothing: programmatic dependent launch is CUDA-only, and the calls are no-ops unless it is enabled.', correct: true },
      { label: 'They become HIP’s own dependent-launch calls through vendors/hip.h.', why: 'The PDL macro is defined only when GGML_USE_HIP is not (ch18 D3).' },
      { label: 'They are barriers that every thread of the block must reach.', why: 'They are about the order between consecutive kernels, not between threads.' },
      { label: 'They fail to compile, so the HIP build leaves out unary.cu.', why: 'The HIP build compiles every ggml-cuda/*.cu, unary.cu included.' },
    ],
    explain: (
      <p>
        PDL lets a dependent kernel launch before the kernel it depends on, in the same stream, has finished. llama.cpp defines{' '}
        <code>GGML_CUDA_USE_PDL</code> only <code>#if !defined(GGML_USE_HIP) &amp;&amp; …</code>, and the device helpers act only on
        Hopper or later. On HIP the two lines compile to nothing, and the kernel is lab 2.3’s.
      </p>
    ),
    source: `${FACTS} B1; notes/today/ch18-facts.md D3`,
  },
  'x3/X3.2/q-fconst': {
    kind: 'choice',
    id: 'x3/X3.2/q-fconst',
    prompt: (
      <>
        llama.cpp’s CUDA unary kernel makes the op a template parameter; its Metal kernel makes it a function constant. What
        does the function constant buy on Metal?
      </>
    ),
    options: [
      { label: 'One compiled library; each variant is fixed when its pipeline state is created, not compiled offline.', correct: true },
      { label: 'The op can change from thread to thread at run time, so one dispatch can mix several ops in one grid.', why: 'A function constant has one value per pipeline, the same for every thread.' },
      { label: 'Faster math: function constants enable fast::exp.', why: 'The math mode is a compile option of its own (ch05 C7).' },
      { label: 'It lets the CUDA and Metal kernels share one source file, with the op chosen by a constant on each side.', why: 'They are separate files: unary.cu and unary.metal.' },
    ],
    explain: (
      <p>
        MSL 4.1, section 5.8: function constants “provide the same ease of use as preprocessor macros but moves the generation of
        the specific variants to the creation of the pipeline state, so you don’t have to compile the variants offline.” The
        value comes from an <code>MTLFunctionDescriptor</code>, not from the source.
      </p>
    ),
    source: `${FACTS} B1, C1 (MSL 4.1 §5.8, p. 190)`,
  },
  'x3/X3.2/q-silu-bits': {
    kind: 'choice',
    id: 'x3/X3.2/q-silu-bits',
    prompt: <>Will llama.cpp’s Metal SiLU and MLX’s Metal SiLU give bit-identical outputs for the same float input?</>,
    options: [
      { label: 'Not necessarily: x / (1 + exp(−x)) and x · sigmoid(x) can round differently, and Metal’s math mode adds more freedom.', correct: true },
      { label: 'Yes: both compute SiLU, and float SiLU has one correctly rounded answer, which any careful float kernel must produce.', why: 'The exact value has one answer; two float formulas can round to different neighbours of it.' },
      { label: 'Yes, because both run on the same GPU, through the same Metal compiler, in the same math mode.', why: 'Same hardware, different operations: a division against a multiply by a sigmoid.' },
      { label: 'No: the two formulas always differ in the last bit, whatever the input.', why: '“Need not agree” is not “always differ”. Often they will match. Nobody measured it here.' },
    ],
    explain: (
      <p>
        The two are equal in exact arithmetic. In float, a division of x by 1 + exp(−x) and a product of x with a separately
        rounded sigmoid are different sequences of rounded operations. On Metal the default math mode allows the compiler
        further liberties. We have not measured how often they differ.
      </p>
    ),
    source: `${FACTS} B1; notes/today/ch05-facts.md C7`,
  },
  'x3/X3.2/q-getrows-quant': {
    kind: 'choice',
    id: 'x3/X3.2/q-getrows-quant',
    prompt: <>Why can llama.cpp’s GET_ROWS on a quantised embedding table not be a plain row copy?</>,
    options: [
      { label: 'The stored rows are quantised blocks, so the kernel dequantises each row to float as it gathers it.', correct: true },
      { label: 'Quantised rows are stored column-major, so a row is not contiguous.', why: 'Nothing in the facts says so. The difference is the encoding, not the layout.' },
      { label: 'The token ids must be sorted first, so that rows are read in order.', why: 'A gather reads rows in any order; one output row per id.' },
      { label: 'It can: the output stays quantised and later kernels dequantise it.', why: 'The output of GET_ROWS feeds float kernels, and the facts say rows are dequantised on the fly.' },
    ],
    explain: (
      <p>
        <code>getrows.cu</code> has <code>k_get_rows</code> for quantised rows next to <code>k_get_rows_float</code> and{' '}
        <code>k_get_rows_float_vec</code>; on Metal <code>kernel_get_rows_q</code> sits next to <code>kernel_get_rows_f</code>.
        The quantised one decodes each block as it copies. Module <a href={href('x2', 'start')}>X2</a> builds those block formats.
      </p>
    ),
    source: `${FACTS} B7`,
  },
};

// ---------------------------------------------------------------- course kernels, loaded from the labs
// ---------------------------------------------------------------- the launch-shapes explorer
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)', fontFamily: 'var(--sans)', fontSize: '0.84rem' };
const CW = 13;
/** Fewer elements per row below 600 px, so the cells stay large enough to read. */
function usePerRow(): number {
  const q = '(max-width: 600px)';
  const [narrow, setNarrow] = useState(() => typeof window !== 'undefined' && window.matchMedia(q).matches);
  useEffect(() => {
    const m = window.matchMedia(q);
    const on = () => setNarrow(m.matches);
    m.addEventListener('change', on);
    return () => m.removeEventListener('change', on);
  }, []);
  return narrow ? 20 : 50;
}

function fillOf(path: Path | null): { fill: string; stroke: string; dash?: string; width?: number } {
  switch (path) {
    case 'vector':
      return { fill: 'var(--global)', stroke: 'var(--global)' };
    case 'scalar':
      return { fill: 'var(--global-wash)', stroke: 'var(--global)' };
    case 'tail':
      return { fill: 'var(--global-wash)', stroke: 'var(--ink)', width: 2 };
    case 'misaligned':
      return { fill: 'var(--global-wash)', stroke: 'var(--global)', dash: '2 2' };
    default:
      return { fill: 'var(--cell)', stroke: 'var(--rule)' };
  }
}

function Swatch({ path }: { path: Path }) {
  const f = fillOf(path);
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem', marginRight: '0.9rem' }}>
      <svg width="14" height="14" aria-hidden="true">
        <rect x="1" y="1" width="12" height="12" rx="1" fill={f.fill} stroke={f.stroke} strokeWidth={f.width ?? 1} strokeDasharray={f.dash} />
      </svg>
      {PATH_LABEL[path]}
    </span>
  );
}

function Elements({ l }: { l: Launch }) {
  const PER_ROW = usePerRow();
  const n = l.cfg.n;
  const rows = Math.ceil(n / PER_ROW);
  return (
    <svg viewBox={`0 0 ${PER_ROW * CW + 40} ${rows * (CW + 4) + 18}`} style={{ width: '100%', maxWidth: 720, display: 'block' }} role="img"
      aria-label="Elements, coloured by the path that wrote them" data-part="elements">
      <text x={0} y={11} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>elements 0 to {n - 1}, {PER_ROW} per row</text>
      {l.pathOf.map((p, e) => {
        const r = Math.floor(e / PER_ROW);
        const c = e % PER_ROW;
        const f = fillOf(p);
        const newOwner = e > 0 && l.writer[e] !== l.writer[e - 1] && c !== 0;
        return (
          <rect key={e} x={36 + c * CW + (newOwner ? 1 : 0)} y={18 + r * (CW + 4)} width={CW - 2 - (newOwner ? 1 : 0)} height={CW} rx="1"
            fill={f.fill} stroke={f.stroke} strokeWidth={f.width ?? 0.8} strokeDasharray={f.dash} data-path={p ?? 'none'} data-writer={l.writer[e]} />
        );
      })}
      {Array.from({ length: rows }, (_, r) => (
        <text key={r} x={0} y={18 + r * (CW + 4) + 10} fontSize="10" style={{ fill: 'var(--ink-3)' }}>{r * PER_ROW}</text>
      ))}
    </svg>
  );
}

function Warps({ l }: { l: Launch }) {
  const TW = 18;
  const div = new Set(divergentWarps(l));
  return (
    <svg viewBox={`0 0 ${WARP * TW + 150} ${l.warps.length * 22 + 18}`} style={{ width: '100%', maxWidth: 720, display: 'block' }} role="img"
      aria-label="Threads by warp, coloured by path" data-part="warps">
      <text x={0} y={11} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>threads, {WARP} per row: a CUDA warp</text>
      {l.warps.map((w) => (
        <g key={w.w} data-warp={w.w} data-diverges={w.diverges ? '1' : '0'}>
          <text x={0} y={18 + w.w * 22 + 13} fontSize="10" style={{ fill: 'var(--ink-3)' }}>warp {w.w}</text>
          {l.plans.slice(w.threads[0], w.threads[1] + 1).map((p) => {
            const f = fillOf(p.path === 'idle' ? null : p.path);
            return <rect key={p.t} x={50 + (p.t % WARP) * TW} y={18 + w.w * 22} width={TW - 3} height={17} rx="1.5" fill={f.fill} stroke={f.stroke}
              strokeWidth={f.width ?? 0.8} strokeDasharray={f.dash} />;
          })}
          {div.has(w.w) ? <text x={50 + WARP * TW + 6} y={18 + w.w * 22 + 13} fontSize="10.5">branch splits</text> : null}
        </g>
      ))}
    </svg>
  );
}

function LaunchShapes() {
  const [style, setStyle] = useState<Style>('binary_vv');
  const [n, setN] = useState(150);
  const [block, setBlock] = useState(32);
  const [nReads, setNReads] = useState(4);
  const [offset, setOffset] = useState(0);
  const cfg: LaunchCfg = { style, n, block, nReads, offset };
  const l = useMemo(() => launch(cfg), [style, n, block, nReads, offset]); // eslint-disable-line react-hooks/exhaustive-deps
  const counts = pathCounts(l);
  const tail = tailSummary(l);
  const div = divergentWarps(l);
  const vv = style === 'binary_vv';
  const paths: Path[] = vv ? ['vector', 'tail', 'misaligned', 'idle'] : ['scalar', 'idle'];
  const shownWarps = l.warps.slice(0, 3);

  return (
    <Bench
      title="Launch shapes"
      task="Pick a kernel style, then move n, the block size and the pointer offset. Watch which thread writes what, and by which path."
      foot={
        <>
          Drawn from sim/elemshapes.ts. A gap separates one thread’s elements from the next. A warp is drawn 32 wide, as on
          CUDA; Metal reports its SIMD-group width per pipeline. Only loads of the first input are listed. The binary_vv launch here is ours: ceil(n /
          N_READS) threads rounded up to whole blocks. The grid-stride style uses {STRIDE_GROUPS} groups, also our choice.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          Kernel style
          <span className="seg" role="group" aria-label="Kernel style" style={{ flexWrap: 'wrap' }}>
            {STYLES.map((s) => (
              <button key={s.id} aria-pressed={style === s.id} onClick={() => setStyle(s.id)} title={s.who} data-style={s.id}>{s.label}</button>
            ))}
          </span>
        </span>
        <span style={FIELD}>
          Block size
          <span className="seg" role="group" aria-label="Block size">
            {BLOCK_CHOICES.map((b) => <button key={b} aria-pressed={block === b} onClick={() => setBlock(b)}>{b}</button>)}
          </span>
        </span>
        {vv ? (
          <span style={FIELD}>
            N_READS
            <span className="seg" role="group" aria-label="N_READS">
              {N_READS_CHOICES.map((r) => <button key={r} aria-pressed={nReads === r} onClick={() => setNReads(r)}>{r}</button>)}
            </span>
          </span>
        ) : null}
      </div>
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        <label className="field" style={{ flex: '1 1 14rem' }}>
          n = {n}
          <input type="range" min={1} max={300} value={n} aria-label="n" onChange={(e) => setN(Number(e.target.value))} style={{ width: '100%' }} />
        </label>
        {vv ? (
          <label className="field" style={{ flex: '1 1 10rem' }}>
            pointer offset = {offset} float{offset === 1 ? '' : 's'} ({offset * 4} bytes)
            <input type="range" min={0} max={3} value={offset} aria-label="pointer offset" onChange={(e) => setOffset(Number(e.target.value))} style={{ width: '100%' }} />
          </label>
        ) : null}
      </div>
      <p className="ui muted" style={{ margin: '0 0 0.4rem', fontSize: '0.8rem' }}>
        {STYLES.find((s) => s.id === style)!.who}. {paths.map((p) => <Swatch key={p} path={p} />)}
      </p>
      <Elements l={l} />
      <div style={{ height: '0.6rem' }} />
      <Warps l={l} />
      <p style={{ margin: '0.6rem 0 0.3rem' }} data-readout="summary" data-threads={l.threads} data-tail-elements={tail.elements} data-tail-threads={tail.threads}
        data-divergent={div.join(',')} data-vector={counts.vector.threads} data-misaligned={counts.misaligned.threads} data-idle={counts.idle.threads}
        data-aligned={l.aligned ? '1' : '0'}>
        <b>{l.threads}</b> threads in <b>{l.blocks}</b> block{l.blocks === 1 ? '' : 's'} for {n} elements.{' '}
        {vv ? (
          <>
            {counts.vector.threads} take the vector load, {counts.misaligned.threads} the scalar loop because the pointer is
            not aligned, {tail.threads} the tail ({tail.elements} element{tail.elements === 1 ? '' : 's'}), {counts.idle.threads} have
            nothing to do.{' '}
          </>
        ) : style === 'per-element' ? (
          <>{counts.scalar.threads} write one element each, {counts.idle.threads} stop at the guard. </>
        ) : (
          <>Each thread loops with a stride of {l.threads}: up to {Math.ceil(n / l.threads)} trips. </>
        )}
        {div.length ? <>The branch splits in warp {div.join(', ')} only.</> : <>No warp splits: every warp takes one branch.</>}
      </p>
      <div className="ui" style={{ fontSize: '0.82rem' }} data-readout="loads">
        {shownWarps.map((w) => (
          <p key={w.w} style={{ margin: '0 0 0.2rem' }} data-warp-loads={w.w}>
            <span className="muted">warp {w.w}, loads of a:</span>{' '}
            {w.loads.length === 0
              ? 'none'
              : w.loads.slice(0, 4).map((ld, k) => {
                  const f = footprint(ld, vv ? offset : 0);
                  return (
                    <span key={k} data-contiguous={f.contiguous ? '1' : '0'}>
                      {k ? '; ' : ''}
                      {ld.lanes[0].width > 1 ? `one ${ld.lanes[0].width}-wide load per lane` : `load ${ld.k + 1}`}: {f.read} float{f.read === 1 ? '' : 's'}{' '}
                      {f.contiguous ? `in one run, ${f.lo} to ${f.hi}` : `spread over ${f.lo} to ${f.hi}`}
                    </span>
                  );
                })}
            {w.loads.length > 4 ? '; …' : ''}
          </p>
        ))}
        {l.warps.length > 3 ? <p className="muted" style={{ margin: 0 }}>Warps 3 and up are not listed.</p> : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the variant sort
function VariantSort() {
  const [picks, setPicks] = useState<Record<string, Mechanism | undefined>>({});
  const [checked, setChecked] = useState(false);
  const g = gradeSort(picks);
  const all = VARIANT_MECHANISMS.every((v) => picks[v.id]);
  return (
    <Bench
      title="How each project picks a variant"
      task="Sort each line by the moment its variant is chosen. Then check."
      foot={<>Every line is quoted in notes/today/x3-facts.md. The sort and its answers are VARIANT_MECHANISMS in sim/backendmap.ts.</>}
    >
      <div style={{ display: 'grid', gap: '0.5rem' }}>
        {VARIANT_MECHANISMS.map((v) => {
          const ok = checked && picks[v.id] === v.mechanism;
          const bad = checked && picks[v.id] !== v.mechanism;
          return (
            <div key={v.id} className="panel" data-item={v.id} data-result={checked ? (ok ? 'right' : 'wrong') : ''}
              style={{ padding: '0.5rem 0.7rem', borderColor: ok ? 'var(--ok)' : bad ? 'var(--fail)' : undefined }}>
              <div className="row" style={{ justifyContent: 'space-between', gap: '0.4rem 1rem' }}>
                <code style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', flex: '1 1 18rem' }}>{v.code}</code>
                <span className="muted ui" style={{ fontSize: '0.78rem' }}>{PROJECT_NAME[v.project]}, {BACKEND_NAME[v.backend]}</span>
                <select aria-label={'mechanism for ' + v.id} value={picks[v.id] ?? ''} onChange={(e) => { setPicks({ ...picks, [v.id]: e.target.value as Mechanism }); setChecked(false); }}>
                  <option value="" disabled>chosen when…</option>
                  {MECHANISMS.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
                </select>
              </div>
              {checked ? (
                <p className="ui" style={{ margin: '0.35rem 0 0', fontSize: '0.82rem' }}>
                  <span className={ok ? 't-ok' : 't-fail'}>{ok ? 'Right.' : `It is: ${MECHANISMS.find((m) => m.id === v.mechanism)!.label}.`}</span>{' '}
                  {v.why} <span className="muted">({v.facts})</span>
                </p>
              ) : null}
            </div>
          );
        })}
      </div>
      <div className="row" style={{ marginTop: '0.7rem' }}>
        <button className="primary" disabled={!all} onClick={() => setChecked(true)}>Check</button>
        {checked ? <span data-readout="sort-score"><b>{g.right} of {g.total}</b> right.</span> : <span className="muted ui">{all ? '' : 'Choose for every line first.'}</span>}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the section
export function SX32() {
  const q = questionsSX32;
  const x = EXCERPTS_SX32;
  // Each kernel without its comment, numbered as in its file.
  const lab23 = useLabExcerpt('ch02/lab3_elementwise/solution/kernel.cu', 'addResidualKernel', { comment: false });
  const lab62 = useLabExcerpt('ch06/lab2_vector/solution/kernel.cu', 'vecAdd4Kernel', { comment: false });
  return (
    <>
      <p>
        <code>torch.nn.functional.silu</code> is the gate of every Llama MLP: Hugging Face’s <code>LlamaMLP</code> computes{' '}
        <code>down_proj(act_fn(gate_proj(x)) * up_proj(x))</code>, and the SiLU and the product between the two projections are
        elementwise ops. So is every residual add. This section reads the kernels the three projects run for them, and for the
        embedding lookup, which is a gather. You have written all of them: they are chapter 2’s kernel, sometimes with chapter
        6’s coarsening and vector loads.
      </p>
      <Beyond title="Ours, not the book's">
        <p>
          The course mapping (which lab each kernel is) is ours. The kernels were read in the facts pass on 2026-11-20 at the
          pinned commits; none was run.
        </p>
      </Beyond>

      <h2>llama.cpp on CUDA: lab 2.3 with the op as a template</h2>
      <p>
        Here is the whole kernel behind SiLU, GELU and llama.cpp’s other unary ops on CUDA. Read it next to lab 2.3’s residual add
        before you read on.
      </p>
      <Excerpt x={x.unary} caption="llama.cpp’s unary kernel." />
      <Code tabs={[{ label: 'lab 2.3, addResidualKernel', code: lab23?.text ?? '// loading…', firstLine: lab23?.firstLine }]}
        caption={<>The course’s kernel, loaded from <code>labs/ch02/lab3_elementwise/solution/kernel.cu</code>.</>} />
      <Ask q={q['x3/X3.2/q-read-unary']} label="Read the excerpt" />
      <p>
        The launch is lab 2.3’s too: <code>(k + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE</code> blocks. SiLU itself is{' '}
        <code>x / (1.0f + expf(-x))</code>. SwiGLU is a second kernel, <code>unary_gated_op_kernel</code>, still one thread per
        output, computing <code>op(x[j0]) * g[j1]</code>. Residual adds go through <code>k_bin_bcast</code>, and{' '}
        <code>ggml_cuda_op_fused_add</code> fuses 2 to 8 consecutive adds into one launch.
      </p>
      <Today checked={CHECKED('B1; ch18 D3')}>
        <p>
          <code>unary.cu</code> lines 116 to 127, launch at line 131; <code>ggml_cuda_op_silu_single</code> in{' '}
          <code>unary.cuh</code>. PDL: defined only when not building for HIP, active from Hopper on.
        </p>
      </Today>
      <Ask q={q['x3/X3.2/q-pdl']} />

      <h2>llama.cpp on Metal: one kernel, a function constant</h2>
      <p>
        The Metal side has one template, <code>kernel_unary_impl</code>, for every unary op. Where CUDA makes the op a template
        parameter, Metal makes it a function constant, declared in the source without a value and tested inside the body:
      </p>
      <Excerpt x={x.unaryMetal} caption="The op selector and the SiLU branch of unary.metal." />
      <Ask q={q['x3/X3.2/q-read-fconst']} label="Read the excerpt" />
      <p>
        The value is supplied when the host creates the pipeline, so one compiled library holds every op and each pipeline is
        specialised for one of them. A <code>float4</code> instantiation, <code>kernel_unary_f32_f32_4</code>, does
        four elements per thread. SwiGLU on Metal is written differently again, as a grid-stride loop over a fixed grid:
      </p>
      <Today checked={CHECKED('C1')}>
        <p>
          MSL 4.1 §5.8: “Metal 1.2 and later support function constants.” They “provide the same ease of use as preprocessor
          macros but moves the generation of the specific variants to the creation of the pipeline state, so you don’t have to
          compile the variants offline.”
        </p>
      </Today>
      <Excerpt x={x.swiglu} caption="kernel_swiglu’s loop: start at the global index, step by the whole grid." />
      <Ask q={q['x3/X3.2/q-fconst']} />

      <h2>MLX on CUDA: chapter 6’s coarsening, with a guarded vector load</h2>
      <p>
        MLX’s CUDA binary kernel gives each thread <code>N_READS</code> consecutive elements. A thread whose chunk would run past
        the end takes a scalar loop over what is left; every other thread loads its chunk as one vector. There is no second
        launch for the tail.
      </p>
      <Excerpt x={x.binaryVv} caption="MLX’s binary_vv. The lines that apply Op to the vectors are left out." />
      <p>
        Lab 6.2 asked you for the same kernel. Compare its body:
      </p>
      <Code tabs={[{ label: 'lab 6.2, vecAdd4Kernel', code: lab62?.text ?? '// loading…', firstLine: lab62?.firstLine }]}
        caption={<>Loaded from <code>labs/ch06/lab2_vector/solution/kernel.cu</code>.</>} />
      <p>
        The difference is inside <code>load_vector</code>. It does the book’s pointer cast, but only after testing that the
        pointer is aligned to the vector’s size, and falls back to a scalar loop if not:
      </p>
      <Today checked={CHECKED('B1; infra A6; ch06 D4')}>
        <p>
          No vendor documents that casting <code>float*</code> to a vector pointer gives one wide load; the course measures it
          instead (infra A6). PyTorch’s <code>MemoryAccess.cuh</code> makes the same run-time choice from pointer alignment.
        </p>
      </Today>
      <Excerpt x={x.loadVector} caption="MLX’s load_vector: the cast, guarded at run time." />
      <Ask q={q['x3/X3.2/q-read-binvv']} label="Read the excerpt" />

      <h3>Explore the shapes</h3>
      <p>
        Predict before you touch the controls. With N_READS = 4 and n = 150, how many threads take the tail, and how many
        elements does it hold? Then shift the pointer by one float. Which threads change path, and does any element change
        owner?
      </p>
      <Simplification>
        <p>
          No timing, no caches, no memory transactions: the model shows which accesses each thread makes, not what they cost.
          Only loads of the first input are drawn. MLX’s launch computation was not read, so binary_vv’s grid here is ours.
        </p>
      </Simplification>
      <LaunchShapes />
      <p>
        What you should have seen. The tail holds n mod N_READS elements in one thread, and the branch splits in at most one
        warp. A one-float offset sends every thread down the scalar path; each thread keeps its elements, so the result is the
        same, but in each load instruction the lanes of a warp now read every fourth float instead of one unbroken run (with
        N_READS = 2, a two-float offset is still aligned). The grid-stride style covers any n with the same fixed grid.
      </p>
      <Aside>
        <p>
          MLX’s Metal kernels do the same coarsening in <code>binary.h</code>: <code>WorkPerThread&lt;T&gt;::n</code> elements per
          thread, with a guard <code>index + i &lt; size</code> (ch02 D3).
        </p>
      </Aside>
      <Ask q={q['x3/X3.2/q-tail']} />
      <Ask q={q['x3/X3.2/q-misaligned']} />

      <h2>PyTorch: a lambda over one elementwise loop</h2>
      <p>
        PyTorch’s CUDA elementwise ops are lambdas passed to one launcher, <code>gpu_kernel</code>, which runs{' '}
        <code>elementwise_kernel</code>: a 1D grid whatever the tensor’s shape, several elements per thread, a block size fixed in
        source (128 threads on CUDA, 256 under ROCm), and a vector width chosen from each pointer’s alignment at run time. SiLU’s
        lambda computes in <code>at::opmath_type</code>, the accumulate type, and casts back to the tensor’s type at the end.
      </p>
      <Today checked={CHECKED('A3; ch03 D6, ch06 D4')}>
        <p>
          <code>num_threads()</code> is <code>C10_WARP_SIZE * 4</code> on CUDA and 256 under <code>USE_ROCM</code>.
          {' '}<code>can_vectorize_up_to</code> returns 8, 4, 2 or 1 from the pointer’s address.
        </p>
      </Today>
      <Excerpt x={x.siluTorch} caption="PyTorch’s SiLU on CUDA, registered with REGISTER_DISPATCH(silu_stub, &silu_kernel)." />
      <p>
        On MPS, PyTorch names the kernel instead: <code>lib.exec_binary_kernel(iter, "add")</code> picks{' '}
        <code>add_functor</code> or <code>add_alpha_functor</code> in <code>kernels/BinaryKernel.metal</code>.
      </p>

      <h3>The same function, three formulas</h3>
      <p>
        llama.cpp on CUDA and Metal, PyTorch on CUDA and llama2.c compute SiLU as x / (1 + exp(−x)). MLX computes x ·
        sigmoid(x). The two are equal in exact arithmetic and need not round alike in float. When you compare your kernel with
        one of these, compare with a tolerance; section <a href={href('x3', 'X3.9')}>X3.9</a> says how the projects do it.
      </p>
      <Beyond title="Ours, not measured">
        <p>That the two formulas can round differently is our reading. Nobody in this course has measured how often they do.</p>
      </Beyond>
      <Ask q={q['x3/X3.2/q-silu-bits']} />

      <h2>Embedding is a gather everywhere</h2>
      <p>
        Every project looks up embeddings as a row gather: one output row per token id. llama.cpp has a graph op for it,{' '}
        <code>GGML_OP_GET_ROWS</code>, with <code>k_get_rows_float</code> and <code>k_get_rows_float_vec</code> for float tables
        and <code>k_get_rows</code> for quantised ones, which dequantises each row as it copies; on Metal the host asks for{' '}
        <code>kernel_get_rows_</code> plus the type’s name. PyTorch has no embedding kernel per backend at all: the op is a
        composite of <code>index_select</code>.
      </p>
      <Excerpt x={x.embedTorch} caption="PyTorch’s embedding_symint, the whole forward." />
      <p>
        MLX’s <code>nn.Embedding</code> is <code>self.weight[x]</code>. All three do what lab 22.2’s <code>launchEmbed</code>{' '}
        does, a row gather with one output row per token id. How each maps threads to outputs was not read.
      </p>
      <Today checked={CHECKED('B7')}>
        <p>
          <code>getrows.cu</code> lines 6, 73, 105; <code>quantize.metal</code> lines 266 and 297; PyTorch{' '}
          <code>native_functions.yaml</code> lines 2261 to 2263 (<code>CompositeExplicitAutograd: embedding_symint</code>).
        </p>
      </Today>
      <Ask q={q['x3/X3.2/q-getrows-quant']} />
      <OpMap op="embedding" project="llama.cpp" title="Op map: embedding" />

      <h2>How each project picks a variant</h2>
      <p>
        Every kernel above exists in several variants: per op, per type, per fusion level, per vector width. The projects choose
        among them at different moments. On CUDA and HIP the choice is usually made when the library is compiled; on Metal it
        is made when a pipeline is created, or by the name the host asks for; MLX and PyTorch also generate code, at run time
        and at build time.
      </p>
      <VariantSort />
      <p>
        The practical rule for reading: a CUDA or HIP variant is a template instantiation you can find by its template’s name.
        A Metal variant is a string or a function constant’s value, set by host code you have to read first. A generated
        variant is in no file at all.
      </p>
    </>
  );
}
