import { useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Today } from '../../components/notes';
import { href } from '../../lib/router';
import {
  DIFFER_ROWS,
  ENTRY_POINTS,
  PROJECT_NAME,
  ROUTE_CHOICES,
  hopOrder,
  route,
  scoreWalk,
  type Route,
} from '../../sim/backendmap';
import { linesText, permalink, shortCommit, type Excerpt as ExcerptData } from '../../sim/x3sources';
import { Excerpt } from './Excerpt';
import { OneCallScene } from './OneCallScene';
import { OpMap } from './OpMap';

// Section X3.1: the reading method. Each project has one place where an op becomes a backend
// call; start there, follow one name, and on Metal follow a string. The op map, the
// "Follow the call" stepper and the one-call scene all run on sim/backendmap.ts. The
// excerpts below are checked against notes/today/x3-facts.md by excerpts-g1.test.ts.

const FACTS = 'notes/today/x3-facts.md';
const CHECKED = (items: string) => `2026-11-20 in the facts pass (${FACTS}, ${items})`;

export const EXCERPTS_SX31: Record<string, ExcerptData> = {
  switch: {
    id: 'x31-switch', project: 'llama.cpp', path: 'ggml/src/ggml-cuda/ggml-cuda.cu', lines: [2812, 3042], factsItem: 'A1', lang: 'cuda',
    code: `static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    switch (dst->op) {
...
        case GGML_OP_GET_ROWS:
            ggml_cuda_op_get_rows(ctx, dst);
...
        case GGML_OP_RMS_NORM:
            ggml_cuda_op_rms_norm(ctx, dst);
...
        case GGML_OP_SOFT_MAX:
            ggml_cuda_op_soft_max(ctx, dst);
...`,
  },
  fusion: {
    id: 'x31-fusion', project: 'llama.cpp', path: 'ggml/src/ggml-cuda/ggml-cuda.cu', lines: [5268, 5271], factsItem: 'A1', lang: 'cuda',
    code: `if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
    ggml_cuda_op_rms_norm_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
    return 1;
}`,
  },
  snprintf: {
    id: 'x31-snprintf', project: 'llama.cpp', path: 'ggml/src/ggml-metal/ggml-metal-device.cpp', lines: 154, factsItem: 'A1', lang: 'cpp',
    code: `snprintf(base, 256, "kernel_get_rows_%s", ggml_type_name(tsrc));`,
  },
  hostName: {
    id: 'x31-host-name', project: 'llama.cpp', path: 'ggml/src/ggml-metal/kernels/norm.metal', lines: 179, factsItem: 'C2', lang: 'metal',
    code: `template [[host_name("kernel_rms_norm_mul_f32")]] kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float, 2>;`,
  },
  rmsNormPy: {
    id: 'x31-mlx-rmsnorm', project: 'mlx', path: 'python/mlx/nn/layers/normalization.py', lines: 166, factsItem: 'A2', lang: 'python',
    code: `return mx.fast.rms_norm(x, self["weight"], self.eps)`,
  },
  siluPy: {
    id: 'x31-mlx-silu', project: 'mlx', path: 'python/mlx/nn/layers/activations.py', lines: [136, 145], factsItem: 'A2', lang: 'python',
    code: `@partial(mx.compile, shapeless=True)
def silu(x):
...
    return x * mx.sigmoid(x)`,
  },
  addH: {
    id: 'x31-ufunc-add', project: 'pytorch', path: 'aten/src/ATen/native/ufunc/add.h', lines: null, factsItem: 'A3', lang: 'cpp',
    code: `return self + alpha * other;`,
  },
};

export const questionsSX31: Record<string, Question> = {
  'x3/X3.1/q-no-add-file': {
    kind: 'choice',
    id: 'x3/X3.1/q-no-add-file',
    prompt: (
      <>
        You grep PyTorch’s <code>aten/src/ATen/native/cuda</code> for the CUDA kernel behind <code>add.out</code> and find
        nothing. Why?
      </>
    ),
    options: [
      {
        label: 'add.out is a ufunc op: torchgen writes its CUDA file, UfuncCUDA_add.cu, at build time from ufunc/add.h.',
        correct: true,
      },
      { label: 'Adds go to cuBLAS, so PyTorch has no add kernel of its own.', why: 'No library is involved. The generated file calls PyTorch’s own gpu_kernel loop.' },
      { label: 'The kernel moved to the MPS tree, which serves every GPU now.', why: 'MPS is the Apple backend only, and it has its own hand-written add_out_mps for exactly this reason.' },
      { label: 'add is implemented in Python and traced into a kernel at run time.', why: 'That describes no PyTorch op here. The scalar body is C++ in native/ufunc/add.h.' },
    ],
    explain: (
      <p>
        <code>native_functions.yaml</code> gives <code>add.out</code> a <code>ufunc_inner_loop</code> and no <code>CUDA:</code>{' '}
        key. torchgen generates <code>UfuncCUDA_add.cu</code> during the build, containing <code>gpu_kernel(iter, ufunctor);</code>{' '}
        and the <code>REGISTER_DISPATCH</code> that wires it up. The file exists only in a build tree.
      </p>
    ),
    source: `${FACTS} A3`,
  },
  'x3/X3.1/q-fusion-breakpoint': {
    kind: 'choice',
    id: 'x3/X3.1/q-fusion-breakpoint',
    prompt: (
      <>
        You run a Llama model on llama.cpp’s CUDA backend with a breakpoint in <code>ggml_cuda_op_rms_norm</code>. It never
        hits, though the model plainly normalises. What is the likeliest reason, and how do you confirm it?
      </>
    ),
    options: [
      {
        label: 'The fusion pass sent RMS_NORM and the MUL after it to ggml_cuda_op_rms_norm_fused. Rerun with GGML_CUDA_DISABLE_FUSION set.',
        correct: true,
      },
      { label: 'The op ran on the CPU backend. Check ggml_backend_cuda_device_supports_op.', why: 'The CUDA backend accepts RMS_NORM. supports_op would tell you if it did not, but this is not the likely cause.' },
      { label: 'The model file stores pre-normalised weights, so no norm runs.', why: 'RMSNorm divides each activation row by its own root mean square, which depends on the input. Weights cannot do that ahead of time.' },
      { label: 'The HIP build was loaded instead. Check which library the process opened.', why: 'The HIP build compiles the same norm.cu and runs the same fusion pass, so this would not explain it either.' },
    ],
    explain: (
      <p>
        Before the switch in <code>ggml_cuda_compute_forward</code>, a fusion pass looks at runs of nodes. RMS_NORM followed by
        MUL becomes one call to <code>ggml_cuda_op_rms_norm_fused</code>, and RMS_NORM, MUL, ADD, tried one step earlier,
        one call to <code>ggml_cuda_op_rms_norm_fused_add</code>. The environment variable{' '}
        <code>GGML_CUDA_DISABLE_FUSION</code> turns the pass off, and then the plain case runs.
      </p>
    ),
    source: `${FACTS} A1`,
  },
  'x3/X3.1/q-metal-string': {
    kind: 'choice',
    id: 'x3/X3.1/q-metal-string',
    prompt: (
      <>
        llama.cpp’s Metal host wants the RMSNorm kernel fused with the weight multiply, for float rows. To find its source,
        what do you search the <code>.metal</code> files for?
      </>
    ),
    options: [
      { label: 'The string the host asks for, kernel_rms_norm_mul_f32, which a [[host_name(...)]] gives to one instantiation.', correct: true },
      { label: 'The template’s C++ name, kernel_rms_norm_fuse_impl, which is the kernel the pipeline runs.', why: 'That template serves every fusion level and type. Its name does not tell you which instantiation runs.' },
      { label: 'The host function’s name, ggml_metal_op_norm, which the .metal file defines.', why: 'ggml_metal_op_norm is C++ host code in ggml-metal-ops.cpp; it is where you came from.' },
      { label: 'The op enum GGML_OP_RMS_NORM, which the .metal file switches on.', why: 'The switch is on the host. Kernel source never sees the op enum.' },
    ],
    explain: (
      <p>
        Metal finds a kernel by a name string. In <code>norm.metal</code> line 179,{' '}
        <code>template [[host_name("kernel_rms_norm_mul_f32")]] … kernel_rms_norm_fuse_impl&lt;float, 2&gt;;</code> gives the
        instantiation with fusion level 2 the name the host builds. Templates need it: each specialisation gets its own name.
      </p>
    ),
    source: `${FACTS} A1, C2; MSL 4.1 on [[host_name]]`,
  },
  'x3/X3.1/q-silu-file': {
    kind: 'choice',
    id: 'x3/X3.1/q-silu-file',
    prompt: <>Where is the Metal source of MLX’s <code>nn.silu</code>?</>,
    options: [
      {
        label: 'Nowhere in the tree: nn.silu is under mx.compile, so Compiled::eval_gpu writes its source at run time and compiles it.',
        correct: true,
      },
      { label: 'In mlx/backend/metal/kernels/unary.h, beside the other unary ops.', why: 'A plausible place, but the facts show nn.silu is an mx.compile function, which has no kernel file.' },
      { label: 'In a .metal file generated when MLX is built, like PyTorch’s ufunc files.', why: 'Generated, yes, but at run time on first call, not at build time.' },
      { label: 'In MPS: MLX calls Apple’s Metal Performance Shaders for activations.', why: 'Nothing in the facts routes MLX activations to MPS.' },
    ],
    explain: (
      <p>
        <code>nn.silu</code> is <code>@partial(mx.compile, shapeless=True)</code>. <code>Compiled::eval_gpu</code> in{' '}
        <code>metal/compiled.cpp</code> writes <code>[[host_name("…")]] [[kernel]] void …(</code> into a source string and compiles
        it with <code>d.get_library(...)</code>. To read it, you would print the generated source; there is no file to open.
      </p>
    ),
    source: `${FACTS} A2`,
  },
  'x3/X3.1/q-mlx-hip': {
    kind: 'choice',
    id: 'x3/X3.1/q-mlx-hip',
    prompt: (
      <>
        Could you build MLX’s CUDA backend for an AMD GPU by running it through a hipify step, as PyTorch does with its CUDA
        sources?
      </>
    ),
    options: [
      {
        label: 'Not as is: it relies on cuBLASLt, cuDNN’s frontend and cooperative groups, none of them a HIP-compatible spelling, and MLX has no AMD backend.',
        correct: true,
      },
      { label: 'Yes: hipify renames every CUDA call, so any CUDA source builds for HIP.', why: 'Libraries and cooperative groups are where the simple claim stops being true (the course’s portability notes, infra B4).' },
      { label: 'Yes: MLX already ships that step under mlx/backend/rocm.', why: 'There is no such directory at the pinned commit; an experimental ROCm pull request is open and unmerged.' },
      { label: 'No, because AMD GPUs have no warps.', why: 'They do: 64 or 32 wide. That is not the obstacle.' },
    ],
    explain: (
      <p>
        PyTorch can hipify because its CUDA tree was written with that step in mind and pairs it with AMD libraries. MLX’s CUDA
        backend calls cuBLASLt for GEMM, cuDNN’s frontend for attention, and cooperative groups in its reductions. Each needs a
        port, not a rename. <code>mlx/backend</code> has no HIP directory, and pull request #2300 is still open.
      </p>
    ),
    source: `${FACTS} B9; notes/today/infra-facts.md B4`,
  },
};

// ---------------------------------------------------------------- Follow the call
function FollowTheCall() {
  const [choice, setChoice] = useState(ROUTE_CHOICES[0].id);
  const [fusion, setFusion] = useState(true);
  const c = ROUTE_CHOICES.find((x) => x.id === choice)!;
  const r: Route = route(c.variants ? (fusion ? c.variants.on : c.variants.off) : c.id);
  const [hop, setHop] = useState(0);
  const [first, setFirst] = useState<(number | null)[]>([]);
  const [pick, setPick] = useState<number | null>(null);

  const reset = (id: string, f = fusion) => {
    setChoice(id);
    setFusion(f);
    setHop(0);
    setFirst([]);
    setPick(null);
  };
  const done = hop >= r.hops.length;
  const h = done ? null : r.hops[hop];
  const order = h ? hopOrder(r.id, hop, h.options.length) : [];
  const right = h && pick != null ? !!h.options[pick].correct : false;
  const choose = (i: number) => {
    setPick(i);
    setFirst((f) => {
      if (f[hop] != null) return f;
      const n = [...f];
      n[hop] = i;
      return n;
    });
  };
  const next = () => {
    setHop(hop + 1);
    setPick(null);
  };
  const score = scoreWalk(r, first);

  return (
    <Bench
      title="Follow the call"
      task="Choose what to search for at each hop. Then the hop is revealed, with its file, lines and permalink."
      foot={<>Each hop is one search in the real tree at the pinned commit. The routes and their facts items are data in sim/backendmap.ts (ROUTES).</>}
    >
      <div className="row" style={{ marginBottom: '0.6rem', alignItems: 'flex-end' }}>
        <span className="seg" role="group" aria-label="Route" style={{ flexWrap: 'wrap' }}>
          {ROUTE_CHOICES.map((x) => (
            <button key={x.id} aria-pressed={choice === x.id} onClick={() => reset(x.id)} data-route={x.id}>{x.label}</button>
          ))}
        </span>
        {c.variants ? (
          <label className="check">
            <input type="checkbox" checked={fusion} onChange={(e) => reset(choice, e.target.checked)} /> fusion on (the default)
          </label>
        ) : null}
      </div>
      <p className="ui" style={{ margin: '0 0 0.6rem', fontSize: '0.88rem' }} data-readout="start">
        <b>Start.</b> {r.start}
      </p>
      <ol data-part="trail" style={{ margin: '0 0 0.6rem', paddingLeft: '1.3em', fontFamily: 'var(--sans)', fontSize: '0.84rem' }}>
        {r.hops.slice(0, hop).map((x, i) => (
          <li key={i} style={{ marginBottom: '0.3rem' }} data-hop={i}>
            <span className={first[i] != null && x.options[first[i]!].correct ? 't-ok' : 't-fail'}>
              {first[i] != null && x.options[first[i]!].correct ? 'right first time' : 'found on a later try'}
            </span>
            {': '}
            {x.found}{' '}
            <a href={permalink(x.project, x.path, x.lines ?? null, !/\.\w+$/.test(x.path))} target="_blank" rel="noreferrer">
              {x.path.split('/').pop()}{x.lines != null ? ', ' + linesText(x.lines) : ''}
            </a>
          </li>
        ))}
      </ol>
      {h ? (
        <div className="panel" data-part="hop" data-hop={hop}>
          <p style={{ margin: '0 0 0.5rem' }}>
            <b>Hop {hop + 1} of {r.hops.length}.</b> {h.prompt}
          </p>
          <div style={{ display: 'grid', gap: '0.35rem' }}>
            {order.map((i) => {
              const o = h.options[i];
              const state = pick === i ? (o.correct ? 'right' : 'wrong') : '';
              return (
                <button key={i} className="quiet" onClick={() => choose(i)} disabled={right} data-option={o.label}
                  style={{
                    textAlign: 'left', fontWeight: 450,
                    borderColor: state === 'right' ? 'var(--ok)' : state === 'wrong' ? 'var(--fail)' : undefined,
                    background: state === 'right' ? 'var(--ok-wash)' : state === 'wrong' ? 'var(--fail-wash)' : undefined,
                    color: 'var(--ink)',
                  }}>
                  {o.label}
                  {o.kind ? <span className="muted"> ({o.kind})</span> : null}
                </button>
              );
            })}
          </div>
          {pick != null && !right ? (
            <p className="t-fail" style={{ margin: '0.5rem 0 0' }} data-readout="why">{h.options[pick].why} Try another.</p>
          ) : null}
          {right ? (
            <div style={{ marginTop: '0.6rem' }} data-readout="found">
              <p style={{ margin: '0 0 0.4rem' }}>
                <span className="t-ok"><b>Found.</b></span> {h.found}
              </p>
              <p className="muted" style={{ margin: '0 0 0.5rem', fontSize: '0.84rem' }}>
                <a href={permalink(h.project, h.path, h.lines ?? null, !/\.\w+$/.test(h.path))} target="_blank" rel="noreferrer" data-part="permalink">
                  {h.path}{h.lines != null ? ', ' + linesText(h.lines) : ''}
                </a>{' '}
                at {shortCommit(h.project)}; {h.facts.map((f) => (f.includes(' ') ? f : 'x3 ' + f)).join(', ')}.
              </p>
              <button className="primary" onClick={next}>{hop + 1 < r.hops.length ? 'Next hop' : 'Finish'}</button>
            </div>
          ) : null}
        </div>
      ) : (
        <div className="panel" data-readout="end">
          <p style={{ margin: '0 0 0.4rem' }}>
            <b>Done: {score.right} of {score.answered} hops right on the first choice.</b> {r.lesson}
          </p>
          <button onClick={() => reset(choice)}>Walk it again</button>
        </div>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- the section
export function SX31() {
  const q = questionsSX31;
  const x = EXCERPTS_SX31;
  return (
    <>
      <p>
        Every framework you use has a place where an op you called becomes a call into one backend: a CUDA kernel, a HIP
        kernel, a Metal pipeline. Find that place, and each op is a short search away. This section shows where it is in
        llama.cpp, MLX and PyTorch, and the two things that break the simple search: fusion and code that is generated.
      </p>
      <Beyond title="Ours, not the book's">
        <p>
          Module X3 is not in the book. Every claim about a codebase on this page was read in the facts pass on 2026-11-20 at
          the commits on the <a href={href('x3', 'start')}>start page</a>. Nothing was built or run. The reading rules are ours.
        </p>
      </Beyond>

      <h2>llama.cpp: one switch per backend</h2>
      <p>
        In llama.cpp a model is a graph of tensors, and each node’s op is a <code>GGML_OP_*</code> enum value. The CUDA
        backend turns a node into work in one function, <code>ggml_cuda_compute_forward</code>, which is a single switch on
        that value. Each case calls a host function named after the op, and each host function lives in the{' '}
        <code>.cu</code> file of the same name: <code>getrows.cu</code>, <code>norm.cu</code>, <code>softmax.cu</code>,{' '}
        <code>rope.cu</code>, <code>fattn.cu</code>. The HIP backend is the same switch, because it is the same file compiled
        as HIP.
      </p>
      <Today checked={CHECKED('A1, B9')}>
        <p>
          <code>ggml_cuda_compute_forward</code> at line 2812; <code>ggml_backend_cuda_device_supports_op</code> (line 6336)
          says whether the backend accepts an op at all. <code>ggml-hip/CMakeLists.txt</code> compiles{' '}
          <code>../ggml-cuda/*.cu</code> as language HIP.
        </p>
      </Today>
      <Excerpt x={x.switch} caption="The CUDA dispatch: one switch over the op enum, three of its cases." />
      <p>
        There is a catch. Before that switch sees a node, a fusion pass looks at runs of nodes. If an RMS_NORM is followed by
        a MUL (the norm’s weight), the pair goes to one fused call, and the switch’s own case for RMS_NORM never runs. In a
        Llama graph the kernel that runs for RMS_NORM is usually the fused one, so the host function you find first is not
        the one that runs.
      </p>
      <Aside>
        <p>
          <code>GGML_CUDA_DISABLE_FUSION</code>, read at line 181, switches the pass off. Use it when you want the plain path:
          to set a breakpoint, or to compare the two.
        </p>
      </Aside>
      <Excerpt x={x.fusion} caption="The fusion pass, before the switch. RMS_NORM, MUL, ADD is tried one step earlier." />

      <h3>Metal finds kernels by string</h3>
      <p>
        The Metal backend has its own switch, <code>ggml_metal_op_encode_impl</code>, over the same enum. Each case encodes
        the work and returns <code>n_fuse</code>, the number of graph nodes it consumed. To get a pipeline, the host builds
        the kernel’s name as a string and asks for it by that name:
      </p>
      <Excerpt x={x.snprintf} caption="The host builds the name of the kernel it wants." />
      <p>
        In the <code>.metal</code> source, the attribute <code>[[host_name(...)]]</code> gives a function, or one
        instantiation of a template, the name the host will ask for. So the C++ template name is shared by every variant,
        and the string tells you which one runs.
      </p>
      <Today checked={CHECKED('C2')}>
        <p>
          MSL 4.1: “In Metal 2.2 and later, you can override the default name that the Metal Framework API uses to refer to a
          qualified function. Add the [[host_name(name)]] attribute…” Templates need “a different name for each
          specialization.”
        </p>
      </Today>
      <Beyond title="Reading rule (ours)">
        <p>To find the Metal kernel behind a host call, search the .metal files for the string the host formats, not for a C++ function name.</p>
      </Beyond>
      <Excerpt x={x.hostName} caption="One instantiation of the RMSNorm template, fusion level 2, named for the host." />

      <h2>MLX: one method per primitive, per backend</h2>
      <p>
        MLX’s Python layers call core ops or <code>mx.fast.*</code>. <code>nn.RMSNorm</code> is one line,{' '}
        <code>mx.fast.rms_norm</code>; <code>nn.Embedding</code> is <code>self.weight[x]</code>, a gather. Each op becomes a{' '}
        <code>Primitive</code> subclass in the graph, and each backend directory implements that primitive’s{' '}
        <code>eval_gpu</code> once: <code>RMSNorm::eval_gpu</code> is in <code>metal/normalization.cpp</code> and in{' '}
        <code>cuda/rms_norm.cu</code>, <code>Softmax::eval_gpu</code> in <code>cuda/softmax.cu</code> and{' '}
        <code>metal/softmax.cpp</code>. So the search is for <code>Name::eval_gpu</code> in{' '}
        <code>mlx/backend/cuda</code> or <code>mlx/backend/metal</code>. Elementwise ops are stamped out by a macro,{' '}
        <code>BINARY_GPU(Add)</code>, so you search for the macro’s use instead.
      </p>
      <Excerpt x={x.rmsNormPy} caption="nn.RMSNorm.__call__: the whole layer." />
      <p>
        The exception is anything under <code>mx.compile</code>. <code>nn.silu</code> is such a function, and so is mlx-lm’s{' '}
        <code>swiglu</code>. Its kernel is written as source text the first time it runs, by <code>Compiled::eval_gpu</code>,
        with <code>[[host_name]]</code> on Metal and <code>__global__</code> on CUDA, and compiled then. There is no file to
        open.
      </p>
      <Today checked={CHECKED('A2, B9')}>
        <p>
          <code>metal/compiled.cpp</code> line 257 writes <code>[[host_name("{'{0}'}")]]</code>; <code>cuda/compiled.cpp</code> line
          225 writes <code>__global__ void {'{}'}(</code> and builds through <code>cu::get_jit_module</code>. MLX has no HIP
          backend: <code>mlx/backend</code> holds common, cpu, cuda, gpu, metal, no_cpu, no_gpu; ROCm pull request #2300 is open.
        </p>
      </Today>
      <Excerpt x={x.siluPy} caption="nn.silu: compiled, so its kernel is generated at run time." />

      <h2>PyTorch: the yaml is the map</h2>
      <p>
        In PyTorch the map from op to kernel is one file, <code>aten/src/ATen/native/native_functions.yaml</code>. Each op has
        a <code>- func:</code> entry, and its <code>dispatch:</code> keys name a function per backend:{' '}
        <code>CUDA: softmax_cuda_out</code>, <code>MPS: softmax_mps_out</code>. ROCm has no key of its own: the same{' '}
        <code>native/cuda</code> sources are hipified when PyTorch is built for AMD.
      </p>
      <Beyond title="Reading rule (ours)">
        <p>
          Find the op’s <code>- func:</code> line and read <code>dispatch:</code>. With no key for your backend, follow{' '}
          <code>structured_delegate</code>, <code>ufunc_inner_loop</code>, or the <code>_stub</code> to its{' '}
          <code>REGISTER_DISPATCH</code>.
        </p>
      </Beyond>
      <p>
        You know <code>torch.nn.functional.rms_norm</code> from a Hugging Face Llama layer. It calls{' '}
        <code>rms_norm_symint</code>, which calls the operator <code>at::_fused_rms_norm</code>, whose entry reads{' '}
        <code>CUDA: _fused_rms_norm_cuda</code>, <code>MPS: _fused_rms_norm_mps</code>. Two more searches land in{' '}
        <code>layer_norm_kernel.cu</code> on CUDA and <code>RMSNorm.metal</code> on MPS. The stepper below walks it.
      </p>
      <p>
        Not every op has a backend key. The most common one, <code>add.out</code>, is a ufunc op: its entry has a{' '}
        <code>ufunc_inner_loop</code> and no <code>CUDA:</code> line. Its scalar body is one line of C++, and its CUDA kernel
        is a file that torchgen writes during the build, <code>UfuncCUDA_add.cu</code>, calling{' '}
        <code>gpu_kernel(iter, ufunctor);</code>. It is in no source file you can grep.
      </p>
      <Today checked={CHECKED('A3, B2; ch01 D4, ch24 D3')}>
        <p>
          <code>add.out</code>: <code>structured: True</code>, <code>ufunc_inner_loop: Generic: add (...)</code> (lines 565 to
          571). MPS: “add.out is a ufunc op, and ufunc codegen only emits CPU/CUDA impls, so MPS needs this entry.” ROCm:{' '}
          <code>tools/amd_build/build_amd.py</code> hipifies <code>native/cuda/*</code>.
        </p>
      </Today>
      <Excerpt x={x.addH} caption="The only op-specific source of the CUDA add kernel: its scalar body in native/ufunc/add.h. torchgen generates the rest at build time." />

      <h2>The three entry points side by side</h2>
      <table className="data wide" data-part="entry-points" style={{ maxWidth: '62rem' }}>
        <thead>
          <tr><th>Project</th><th>An op is</th><th>CUDA</th><th>HIP</th><th>Metal</th></tr>
        </thead>
        <tbody>
          {ENTRY_POINTS.map((e) => (
            <tr key={e.project} data-project={e.project}>
              <td><b>{PROJECT_NAME[e.project]}</b></td>
              <td>{e.what}</td>
              <td>{e.cuda}</td>
              <td>{e.hip}</td>
              <td>{e.metal}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p>
        Three rows of the facts file’s table “Where the backends really differ” say the same thing across projects. On
        CUDA and HIP the host names a kernel by its symbol and variants are template parameters; on Metal the host names it
        by a string, and variants are function constants or named instantiations. Section{' '}
        <a href={href('x3', 'X3.2')}>X3.2</a> reads both mechanisms in real kernels.
      </p>
      <table className="data wide" data-part="differ" style={{ maxWidth: '62rem' }}>
        <thead>
          <tr><th>Feature</th><th>CUDA</th><th>HIP (ROCm)</th><th>Metal</th></tr>
        </thead>
        <tbody>
          {DIFFER_ROWS.map((r) => (
            <tr key={r.feature}>
              <td>{r.feature}</td>
              {[r.cuda, r.hip, r.metal].map((t, i) => (
                <td key={i}>{t.split('`').map((s, k) => (k % 2 ? <code key={k}>{s}</code> : s))}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>

      <h2>Find it yourself</h2>
      <p>
        The op map holds the result of following every op in the tour in every project and backend. Before you open it,
        predict two columns: what HIP holds for llama.cpp and for MLX, and which cells have no file at all.
      </p>
      <OpMap />
      <p>
        You should have found that HIP is the CUDA file in llama.cpp, the hipified CUDA source in PyTorch (except where it
        calls AMD libraries), and empty in MLX; and that some cells have no file: PyTorch’s CUDA add, MLX’s SiLU, PyTorch’s
        RoPE, which is not an operator at all.
      </p>
      <Ask q={q['x3/X3.1/q-no-add-file']} />
      <Ask q={q['x3/X3.1/q-silu-file']} />
      <p>
        The op map shows where you end up. The stepper makes you take the hops: at each one you choose what to search for
        before it is revealed.
      </p>
      <FollowTheCall />
      <Ask q={q['x3/X3.1/q-fusion-breakpoint']} />
      <Ask q={q['x3/X3.1/q-metal-string']} />

      <h2>One call, watched</h2>
      <p>
        The animation follows one RMSNorm through llama.cpp’s CUDA and Metal backends, in order: the fusion pass consuming two
        nodes, the kernel it launches, and the string Metal looks up.
      </p>
      <Aside>
        <p>
          The graph is the start of a Llama layer as llama.cpp would hold it: the order of the four nodes is ours. The kernel
          names, lines and fusion rule are the facts file’s.
        </p>
      </Aside>
      <OneCallScene />

      <h2>HIP and Metal, in one paragraph</h2>
      <p>
        For HIP there is little to find. In llama.cpp the HIP kernel is the CUDA file. In PyTorch it is the CUDA file after a
        translation step, except for the matrix and attention libraries, which differ on AMD. MLX has no HIP backend. For
        Metal there is always a string in the middle: llama.cpp formats one, MLX writes one into generated source, and
        PyTorch’s MPS code builds one with <code>fmt::format</code> (<code>RMSNorm.mm</code>) before asking for the pipeline.
      </p>
      <Ask q={q['x3/X3.1/q-mlx-hip']} />
    </>
  );
}
