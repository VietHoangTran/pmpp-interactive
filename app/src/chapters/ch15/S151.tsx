import { useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import {
  BLAS_EDITS, OPERATIONS, agreement, blasCall, editBlas, editMps, exampleBuffers, mapOperation, mpsCall, reference, runBlas, runMps,
  type BlasCall, type BlasEdit, type Mapping, type MpsCall, type OpId, type RunResult,
} from '../../sim/gemmlayout';

// ---------------------------------------------------------------- questions
export const questionsS151: Record<string, Question> = {
  'ch15/15.1/q-linear': {
    kind: 'choice',
    id: 'ch15/15.1/q-linear',
    prompt: (
      <>
        <code>nn.Linear</code> computes y = x Wᵀ + b, with W stored as (out_features, in_features). Which mapping onto the
        book's D = αA × B + βC is right?
      </>
    ),
    options: [
      {
        label: 'A = x, row-major; B = Wᵀ, which is W\'s memory read column-major; C = b on every row; α = β = 1.',
        correct: true,
      },
      {
        label: 'A = x; B = W exactly as it is stored, row-major, out × in; C = b repeated on every row; α = β = 1.',
        why: 'W is out × in, and the book\'s B has to be K × N, in × out. Used as stored you would compute x W, which only even has matching sizes when in = out.',
      },
      {
        label: 'A = x; B = Wᵀ stored column-major; C = y, the layer\'s output; α = β = 1.',
        why: 'C is the matrix added to the product. The result is D.',
      },
      {
        label: 'A = x; B = Wᵀ; C = b repeated on every row; α = 1/batch to average over the batch, β = 1.',
        why: 'Nothing in a linear layer averages over the batch. Each row is transformed on its own, with α = 1.',
      },
    ],
    explain: (
      <p>
        The book stacks the vectors to transform as the rows of A, so A = x. Column j of B must hold the coefficients
        for output j, and those are row j of W: B = Wᵀ. A row-major W, read column by column, is exactly that B. The bias
        is the same for every input, so C is b copied onto each of the batch rows.
      </p>
    ),
    source: 'ch. 15 §15.1, pp. 349–350; the shapes of nn.Linear from ch03-facts D6; the mapping is ours',
  },
  'ch15/15.1/q-rowmajor-trick': {
    kind: 'choice',
    id: 'ch15/15.1/q-rowmajor-trick',
    prompt: (
      <>
        A (M × K), B (K × N) and C (M × N) are row-major. Why does{' '}
        <code>cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &amp;one, B, N, A, K, &amp;zero, C, N)</code> leave
        C = A B in C?
      </>
    ),
    options: [
      {
        label: 'Each row-major array, read column-major, is its transpose, so cuBLAS computes Cᵀ = Bᵀ Aᵀ, and Cᵀ column-major is C row-major.',
        correct: true,
      },
      {
        label: 'cuBLAS sees leading dimensions equal to the row lengths, recognises row-major input, and transposes it into a hidden copy first.',
        why: 'Nothing is copied. The arguments only describe how to read memory that is already there.',
      },
      {
        label: 'It works only for square matrices; other shapes must be padded to square before the call.',
        why: 'Run a 3 × 4 example in the explorer above: it agrees cell for cell.',
      },
      {
        label: 'Passing B first makes cuBLAS compute B A, and B A equals A B for matrices of matching sizes.',
        why: 'Matrix products do not commute. The order swaps because transposing a product reverses it: (A B)ᵀ = Bᵀ Aᵀ.',
      },
    ],
    explain: (
      <p>
        A row-major M × K array occupies memory exactly like a column-major K × M array, its transpose. So the library
        sees Bᵀ (N × K) and Aᵀ (K × M), multiplies them into an N × M column-major result, and that memory read
        row-major is the M × N matrix C. The leading dimensions are the row lengths of the row-major arrays: N, K and N.
      </p>
    ),
    source: 'ch15-facts A10 (our derivation of the standard idiom); llama.cpp calls cuBLAS the same way (ch03-facts D2)',
  },
  'ch15/15.1/q-qkt': {
    kind: 'choice',
    id: 'ch15/15.1/q-qkt',
    prompt: (
      <>
        Attention scores S = Q Kᵀ, with Q (n_q × d) and K (n_kv × d), both row-major. Which operand plays the role of{' '}
        <code>nn.Linear</code>'s weight?
      </>
    ),
    options: [
      { label: 'K: its rows run along d, and it is used transposed, like W (out, in).', correct: true },
      { label: 'Q, because the left operand of a product is the one a library treats as the weight.', why: 'Q is the book\'s A, stored as the book stores A: one vector per row.' },
      { label: 'Both: in Q Kᵀ each operand is read transposed, as W is in x Wᵀ.', why: 'The left operand is stored as the book expects. Only the right one is turned around.' },
      { label: 'Neither: attention needs a layout of its own, with K stored column by column.', why: 'Pick Q Kᵀ in the explorer. Its call has the same shape as nn.Linear\'s, with transa = T.' },
    ],
    explain: (
      <p>
        In the book's terms M = n_q, N = n_kv and K = d. The book's B is d × n_kv, and K's memory, one token per row, is
        that B stored column-major: the same situation as a weight stored (out, in). The layout milestone M3 handles
        therefore covers Q Kᵀ as well, and chapter 20 builds attention on it.
      </p>
    ),
    source: 'ch. 15 §15.1, p. 350 (attention as large products, ch. 20); the mapping is ours',
  },
};

// ---------------------------------------------------------------- the explorer
function Matrix({ title, rows, cols, ld, data, ok, tier, testid }: {
  title: string;
  rows: number;
  cols: number;
  ld: number;
  data: number[];
  ok?: boolean[];
  tier: 'global' | 'reg';
  testid: string;
}) {
  return (
    <div data-matrix={testid} style={{ minWidth: 0 }}>
      <div className="ui muted" style={{ fontSize: '0.8rem', marginBottom: '0.25rem' }}>{title}</div>
      <div style={{ display: 'grid', gridTemplateColumns: `repeat(${cols}, 2.1rem)`, gap: 2 }}>
        {Array.from({ length: rows * cols }, (_, i) => {
          const r = Math.floor(i / cols);
          const c = i % cols;
          const v = data[r * ld + c];
          const good = ok ? ok[i] : undefined;
          const bg = good === false ? 'var(--fail-wash)' : good ? 'var(--ok-wash)' : `var(--${tier}-wash)`;
          const bd = good === false ? 'var(--fail)' : good ? 'var(--ok)' : `var(--${tier})`;
          return (
            <span key={i} data-cell={`${r},${c}`} data-ok={good == null ? undefined : good ? 1 : 0}
              className="mono"
              style={{ background: bg, border: `1px solid ${bd}`, borderRadius: 2, textAlign: 'center', fontSize: '0.78rem', padding: '0.15rem 0', color: good === false ? 'var(--fail)' : 'var(--ink)' }}>
              {Number.isFinite(v) ? v : '?'}
            </span>
          );
        })}
      </div>
    </div>
  );
}

const opName = (t: 'N' | 'T') => (t === 'N' ? 'CUBLAS_OP_N' : 'CUBLAS_OP_T');

function blasText(c: BlasCall, names: Mapping['op']['names'], numeric: boolean) {
  const v = (n: number, s: string) => (numeric ? String(n) : s);
  const nm = (role: 'A' | 'B' | 'C') => names[role];
  const lines: [string, string][] = [
    [`cublasSgemm(handle,`, ''],
    [`    ${opName(c.transa)}, ${opName(c.transb)},`, '// transa, transb'],
    [`    ${v(c.m, c.symbolic.m)}, ${v(c.n, c.symbolic.n)}, ${v(c.k, c.symbolic.k)},`, '// m, n, k'],
    [`    &alpha, ${nm(c.a)}, ${v(c.lda, c.symbolic.lda)},`, '// A, lda'],
    [`    ${nm(c.b)}, ${v(c.ldb, c.symbolic.ldb)},`, '// B, ldb'],
    [`    &beta, ${nm('C')}, ${v(c.ldc, c.symbolic.ldc)});`, '// C, ldc'],
  ];
  const w = Math.max(...lines.slice(1).map(([a]) => a.length)) + 2;
  return lines.map(([a, b]) => (b ? a.padEnd(w) + b : a)).join('\n');
}

function mpsText(c: MpsCall, numeric: boolean) {
  const v = (n: number, s: string) => (numeric ? String(n) : s);
  return (
    `[[MPSMatrixMultiplication alloc]\n` +
    `    initWithDevice: device\n` +
    `    transposeLeft: ${c.transposeLeft ? 'YES' : 'NO'}\n` +
    `    transposeRight: ${c.transposeRight ? 'YES' : 'NO'}\n` +
    `    resultRows: ${v(c.resultRows, c.symbolic.resultRows)}\n` +
    `    resultColumns: ${v(c.resultColumns, c.symbolic.resultColumns)}\n` +
    `    interiorColumns: ${v(c.interiorColumns, c.symbolic.interiorColumns)}\n` +
    `    alpha: 1.0 beta: 0.0]`
  );
}

function verdict(map: Mapping, r: RunResult, ok: boolean[]) {
  const good = ok.filter(Boolean).length;
  if (r.readOutside || r.writeOutside)
    return (
      <p data-verdict="outside" style={{ margin: '0.4rem 0 0' }}>
        <span className="chip fail">{r.readOutside ? 'reads past the end of a buffer' : 'writes past the end of C'}</span> The
        call compiles and runs; on a GPU it would read or write memory that is not this matrix.
      </p>
    );
  return (
    <p data-verdict={good === ok.length ? 'agree' : 'wrong'} style={{ margin: '0.4rem 0 0' }}>
      {good === ok.length ? (
        <span className="chip ok">agrees with the row-major product</span>
      ) : (
        <span className="chip fail">wrong values</span>
      )}{' '}
      {good} of {ok.length} cells of {map.op.names.C} agree.
    </p>
  );
}

const DIM_MAX = 6;

function SpellingsBench() {
  const [id, setId] = useState<OpId>('linear');
  const [shape, setShape] = useState({ M: 3, N: 4, K: 5 });
  const [edits, setEdits] = useState<Partial<Record<BlasEdit, boolean>>>({});
  const [flipRight, setFlipRight] = useState(false);
  const [numeric, setNumeric] = useState(false);

  const map = useMemo(() => mapOperation(id, shape), [id, shape]);
  const bufs = useMemo(() => exampleBuffers(map), [map]);
  const ref = useMemo(() => reference(map, bufs), [map, bufs]);
  const call = editBlas(blasCall(map), edits);
  const blas = runBlas(call, bufs);
  const blasOk = agreement(map, blas.C, ref);
  const mps = editMps(mpsCall(map), flipRight);
  const mpsRun = runMps(mps, bufs);
  const mpsOk = agreement(map, mpsRun.C, ref);
  const { op } = map;
  const d = op.dims;
  const n = op.names;

  const slider = (key: 'M' | 'N' | 'K') => (
    <label className="field" key={key}>
      {d[key]} (the book's {key}) = {shape[key]}
      <input type="range" min="1" max={DIM_MAX} value={shape[key]} aria-label={`book ${key}`}
        onChange={(e) => setShape({ ...shape, [key]: +e.target.value })} />
    </label>
  );

  return (
    <Bench
      title="One product, three spellings"
      task="Pick an operation. Find where its transposes went, then break the call on purpose."
      foot={
        <>
          The inputs are small integers fixed by position, so every product is exact. The calls are executed by a model of
          how each library reads memory (src/sim/gemmlayout.ts): column-major for the BLAS call, row-major through rowBytes
          for MPS. Nothing here ran on a GPU.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          Operation
          <select value={id} aria-label="operation" onChange={(e) => setId(e.target.value as OpId)}>
            {OPERATIONS.map((o) => (
              <option key={o.id} value={o.id}>{o.label}</option>
            ))}
          </select>
        </label>
        {slider('M')}
        {slider('N')}
        {slider('K')}
        <label className="check">
          <input type="checkbox" checked={numeric} onChange={(e) => setNumeric(e.target.checked)} /> show the calls with numbers
        </label>
      </div>
      <div className="cols">
        <div className="panel" data-panel="book">
          <p style={{ margin: '0 0 0.4rem' }}>
            <b>{op.formula}</b> in the book's terms
          </p>
          <table className="data" style={{ margin: 0 }}>
            <tbody>
              <tr><td className="nowrap">M, N, K</td><td data-dims>{d.M} = {shape.M}, {d.N} = {shape.N}, {d.K} = {shape.K}</td></tr>
              <tr><td>A</td><td>{n.A}: {d.M} × {d.K}, row-major, as the book stores A</td></tr>
              <tr>
                <td>B</td>
                <td data-bstore={op.bStoredNK ? 'column' : 'row'}>
                  {op.bStoredNK ? (
                    <>{n.B} is stored {d.N} × {d.K}: the book's B ({d.K} × {d.N}) <b>stored column-major</b></>
                  ) : (
                    <>{n.B}: {d.K} × {d.N}, row-major, as the book stores B</>
                  )}
                </td>
              </tr>
              <tr><td>C</td><td>{n.C}: {d.M} × {d.N}, row-major</td></tr>
            </tbody>
          </table>
          <p className="muted" style={{ margin: '0.5rem 0 0', fontSize: '0.85rem' }}>{op.note}</p>
        </div>
        <div className="panel" data-panel="ref">
          <p style={{ margin: '0 0 0.5rem' }}>
            <b>The data</b>, and the row-major reference computed straight from {op.formula}
          </p>
          <div className="row" style={{ alignItems: 'flex-start' }}>
            <Matrix title={`${n.A} as stored`} rows={map.A.rows} cols={map.A.cols} ld={map.A.ld} data={bufs.A} tier="global" testid="A" />
            <Matrix title={`${n.B} as stored`} rows={map.B.rows} cols={map.B.cols} ld={map.B.ld} data={bufs.B} tier="global" testid="B" />
            <Matrix title={`${n.C} = ${op.formula.split(' = ')[1]}`} rows={shape.M} cols={shape.N} ld={shape.N} data={ref} tier="reg" testid="ref" />
          </div>
        </div>
      </div>

      <div className="cols" style={{ marginTop: '1rem' }}>
        <div className="panel" data-panel="blas">
          <p style={{ margin: '0 0 0.4rem' }}>
            <b>Column-major BLAS:</b> cuBLAS, and hipBLAS with the same arguments (<code>hipblasSgemm</code>,{' '}
            <code>HIPBLAS_OP_N</code>, <code>HIPBLAS_OP_T</code>)
          </p>
          <pre className="mono" data-call="blas" style={{ fontSize: '0.78rem', overflowX: 'auto', margin: '0 0 0.6rem', padding: '0.5rem', background: 'var(--cell)', border: '1px solid var(--rule)', borderRadius: 3 }}>
            {blasText(call, n, numeric)}
          </pre>
          <div className="row" style={{ gap: '0.3rem 1rem', marginBottom: '0.6rem' }}>
            {(Object.keys(BLAS_EDITS) as BlasEdit[]).map((k) => (
              <label key={k} className="check">
                <input type="checkbox" checked={!!edits[k]} data-edit={k}
                  onChange={(e) => setEdits({ ...edits, [k]: e.target.checked })} /> {BLAS_EDITS[k]}
              </label>
            ))}
          </div>
          <Matrix title={`${n.C} after the call, read row-major`} rows={shape.M} cols={shape.N} ld={shape.N} data={blas.C} ok={blasOk} tier="reg" testid="blas" />
          {verdict(map, blas, blasOk)}
        </div>
        <div className="panel" data-panel="mps">
          <p style={{ margin: '0 0 0.4rem' }}>
            <b>Row-major MPS:</b> <code>MPSMatrixMultiplication</code> with transpose flags
          </p>
          <pre className="mono" data-call="mps" style={{ fontSize: '0.78rem', overflowX: 'auto', margin: '0 0 0.6rem', padding: '0.5rem', background: 'var(--cell)', border: '1px solid var(--rule)', borderRadius: 3 }}>
            {mpsText(mps, numeric)}
          </pre>
          <table className="data" style={{ margin: '0 0 0.6rem' }}>
            <thead>
              <tr><th>Matrix</th><th className="num">rows</th><th className="num">columns</th><th className="num">rowBytes</th></tr>
            </thead>
            <tbody>
              {([['left', mps.left], ['right', mps.right], ['result', mps.result]] as const).map(([k, m]) => (
                <tr key={k}>
                  <td>{k}: {n[m.role]}</td>
                  <td className="num">{m.rows}</td>
                  <td className="num">{m.columns}</td>
                  <td className="num">{m.rowBytes}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <label className="check" style={{ marginBottom: '0.6rem' }}>
            <input type="checkbox" checked={flipRight} data-edit="flipRight" onChange={(e) => setFlipRight(e.target.checked)} /> flip transposeRight
          </label>
          <Matrix title={`${n.C} after the call`} rows={shape.M} cols={shape.N} ld={shape.N} data={mpsRun.C} ok={mpsOk} tier="reg" testid="mps" />
          {verdict(map, mpsRun, mpsOk)}
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the page
export function S151() {
  return (
    <>
      <p>
        This section says what GEMM is and how the products in a transformer map onto it.
      </p>

      <h2>From one vector to GEMM</h2>
      <p>
        Start with a single row vector v times a matrix B: u = v × B. Every element of u mixes all the elements of v,
        and the mix for element j is column j of B. So column j holds the coefficients that produce output j. The book's
        example is graphics: rotate a point by multiplying its 3D coordinates with a 3 × 3 matrix, then add a vector to
        shift it and multiply by a factor to scale it.
      </p>
      <Source section="15.1" pages="349–350" />
      <p>
        To transform many vectors with the same matrix, stack them as the rows of A. Then D = A × B, and row i of D is
        vector i transformed. Add a scale and a shift and you have the general form D = αA × B + βC, which linear algebra
        libraries call GEMM, general matrix multiply. The rows of C are what gets added to the matching rows of the
        product.
      </p>
      <p>
        Only A × B is expensive. It is the only part the book optimises, and the only part this chapter's labs compute.
      </p>
      <Source section="15.1" pages="350" />

      <h2>Which way each matrix is stored</h2>
      <p>
        A matrix in memory can be row-major or column-major depending on where it came from, for instance whether the
        code works on the matrix or on its transpose (chapter 6). The chapter assumes both inputs are row-major. Its
        techniques carry over to the other combinations, and GEMM libraries accept all of them.
      </p>
      <p>
        Size decides which optimisations pay. Before deep learning, most products in applications were modest, because
        the vectors being transformed were short. Chapters 19 and 20 write convolution layers and attention layers as
        products of large matrices, and much of this chapter is aimed at that large case.
      </p>
      <Source section="15.1" pages="350" />

      <h2>Underneath nn.Linear</h2>
      <p>
        You already know the most common GEMM in a model. <code>nn.Linear</code> computes y = x Wᵀ + b, with x of shape
        (batch, in) and the weight W stored as (out_features, in_features). In the book's terms: A = x, batch × in,
        row-major, so M = batch and K = in. B = Wᵀ, in × out, so N = out; in memory W is out rows of in floats, which is
        the book's B stored column-major. C is the bias b repeated on every row, and α = β = 1.
      </p>
      <Beyond>
        <p>
          The mapping is ours. The shapes are PyTorch's; ggml stores weights the same way (ch03-facts D2, D6).
        </p>
      </Beyond>
      <p>
        Lab 6.4 (milestone M2) took this layout by turning the corner in shared memory. Lab 15.6 (milestone M3) takes it
        for the fast kernel this chapter builds (<a href={href('ch15', 'labs')}>labs</a>).
      </p>
      <Ask q={questionsS151['ch15/15.1/q-linear']} />

      <h2>One product, three spellings</h2>
      <p>
        The libraries you will measure against do not share a layout. cuBLAS and hipBLAS are column-major; Apple's
        MPSMatrix is row-major. A row-major program calling a column-major library copies nothing. It describes its own
        arrays the way the library will read them: a row-major M × N array, read column-major, is the N × M transpose. So
        instead of C = A B it asks for Cᵀ = Bᵀ Aᵀ, with B passed first. MPS needs no trick, only a flag for an operand
        stored the other way.
      </p>
      <Today checked="2026-11-20 against the cuBLAS 13.5 and rocBLAS docs, hipblas.h at ROCm 10.2 and the macOS 27.0 SDK headers (ch03-facts A7, B6, C7; ch15-facts A10, B6, C6)">
        <p>
          cuBLAS is column-major "for maximum compatibility with existing Fortran environments"; rocBLAS is "the same as
          legacy BLAS and cuBLAS"; <code>hipblasSgemm</code> has the same arguments as <code>cublasSgemm</code>.
          MPSMatrix data "is assumed to be stored in row-major order", and <code>rowBytes</code> must be a multiple of the
          element size.
        </p>
      </Today>
      <SpellingsBench />
      <p>
        Three things to find. Compare <code>nn.Linear</code> with x @ W: the only difference is transa, T instead of N, and
        transposeRight on the MPS side. Compare Q Kᵀ with <code>nn.Linear</code>: the same call, because K is stored like
        a weight. Then switch on any edit. A wrong argument does not crash anything: it gives numbers, wrong ones, or
        reads past the end of a buffer. Only a comparison with a reference catches it, which is why every lab checks
        against one on the CPU.
      </p>
      <Simplification>
        <p>
          The explorer executes each call as the library defines it, with α = 1 and β = 0, on small integers. It does not
          reproduce a library's checks on its arguments or its error codes, and it computes exactly rather than in FP32.
        </p>
      </Simplification>
      <Ask q={questionsS151['ch15/15.1/q-rowmajor-trick']} />
      <Ask q={questionsS151['ch15/15.1/q-qkt']} />

      <h2>What the baseline computes</h2>
      <p>
        Labs 15.1 to 15.5 time your kernel beside the vendor's, with the call from the explorer's first row, the book's
        product (it is in <code>labs/ch15/shared/vendor_blas.cu</code>, and checked against the CPU reference before
        anything is timed). Lab 15.6 does the same for the linear layer. The comparison is fair only if both sides do
        the same arithmetic, and by default they do. A plain{' '}
        <code>cublasSgemm</code> keeps FP32 precision: TF32 tensor cores and the BF16x9 emulation are modes you have to
        ask for. hipBLAS has an XF32 mode you have to ask for. MPS computes with higher internal precision unless you allow
        it to reduce it. So the gap you will measure is technique, not precision.
      </p>
      <Today checked="2026-11-20 (ch15-facts A10, B6, C6, D4)">
        <p>
          cuBLAS's default math mode uses "precisions with at least the same number of mantissa and exponent bits as
          requested"; <code>NVIDIA_TF32_OVERRIDE=0</code> turns TF32 off everywhere; BF16x9 exists on compute capability
          10.0 and 10.3 only. hipBLAS: <code>HIPBLAS_XF32_XDL_MATH</code> is opt-in. MPS:{' '}
          <code>MPSKernelOptionsAllowReducedPrecision</code> is opt-in. PyTorch sends a float matmul to cuBLASLt or
          cuBLAS, and on ROCm can send it to Composable Kernel.
        </p>
      </Today>
    </>
  );
}
