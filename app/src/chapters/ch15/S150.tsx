import { Ask, Predict, type Question } from '../../components/Ask';
import { LabsPage, Reveal, benchRow, defineLabs, measure, preferredBench, useLabs, type BenchSet } from '../../components/LabKit';
import { Beyond, Bench, Source } from '../../components/notes';
import { fmt, num, type BenchRow, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';

// Prerequisite questions drawn from the notes of chapters 4 to 6. Each explanation names
// where the idea was taught and which section of this chapter builds on it.
export const questionsS150: Record<string, Question> = {
  'ch15/start/q-tile-ratio': {
    kind: 'numeric',
    id: 'ch15/start/q-tile-ratio',
    prompt: (
      <>
        The naive matmul kernel does 0.25 FLOP per byte it asks global memory for, and tiling with T × T tiles divides
        that traffic by T. Lab 5.3&rsquo;s kernel uses 16 × 16 tiles. What ratio does it reach in that model?
      </>
    ),
    answer: 4,
    tol: 0.01,
    unit: 'FLOP/B',
    explain: (
      <p>
        0.25 × 16 = 4 FLOP/B. The book&rsquo;s 32 × 32 tiles give 8, still under its example device&rsquo;s 20, and
        chapter 5 leaves going further to this chapter. Section <a href={href('ch15', '15.2')}>15.2</a> redoes the count
        for tiles of any shape and finds that only the output tile matters: 128 × 128 gives 32. Taught in{' '}
        <a href={href('ch05', '5.4')}>ch. 5 §5.4</a>.
      </p>
    ),
    source: 'ch. 5 §5.1, p. 97, and §5.4, p. 111',
  },
  'ch15/start/q-local-array': {
    kind: 'choice',
    id: 'ch15/start/q-local-array',
    prompt: (
      <>
        A thread keeps eight partial sums in <code>float acc[8]</code> and updates them in{' '}
        <code>{'for (int j = 0; j < 8; ++j) acc[j] += ...;'}</code>, a loop the compiler does not unroll. Where does{' '}
        <code>acc</code> live?
      </>
    ),
    options: [
      {
        label: 'In registers, like any automatic variable.',
        why: 'Registers cannot be indexed by a value computed at run time. acc[j] with j a loop variable needs an address.',
      },
      {
        label: 'In local memory: private to the thread, but physically in device memory.',
        correct: true,
      },
      {
        label: 'In shared memory, because it is an array.',
        why: 'Shared memory is what you declare __shared__, and it is visible to the whole block. acc is private to one thread.',
      },
      {
        label: 'In constant memory, because its size is a constant.',
        why: 'Constant memory is read-only for the device and declared __constant__. A constant size does not put an array there.',
      },
    ],
    explain: (
      <p>
        An automatic array goes to local memory unless every access uses a constant index. Fully unroll the loop (its
        trip count is a constant) and <code>acc[j]</code> becomes <code>acc[0]</code> to <code>acc[7]</code>, which the
        compiler can keep in registers. This chapter&rsquo;s kernel keeps 64 accumulators per thread that way, and every
        loop that touches them carries <code>#pragma unroll</code>: section{' '}
        <a href={href('ch15', '15.3')}>15.3</a>. Taught in <a href={href('ch05', '5.2')}>ch. 5 §5.2</a> and{' '}
        <a href={href('ch06', '6.6')}>ch. 6 §6.6</a>.
      </p>
    ),
    source: 'ch. 5 §5.2, pp. 98 and 103; ch. 6 §6.6, pp. 145–146',
  },
  'ch15/start/q-bank-stride': {
    kind: 'numeric',
    id: 'ch15/start/q-bank-stride',
    prompt: (
      <>
        A block declares <code>__shared__ float t[32][16];</code> and the 32 lanes of a warp each read{' '}
        <code>t[lane][0]</code>. With 32 banks of 4 bytes, how many lanes land in the busiest bank?
      </>
    ),
    answer: 16,
    unit: 'lanes',
    explain: (
      <p>
        Lane l reads word 16·l, and 16·l mod 32 is 0 for even lanes and 16 for odd ones: two banks, sixteen lanes on
        each, a 16-way conflict. A row of 17 floats would put lane l in bank 17·l mod 32, a different bank for every
        lane. Section <a href={href('ch15', '15.6')}>15.6</a> meets the same problem in the A tile, which is 8 floats
        wide and read by groups of lanes 4 rows apart. Taught in <a href={href('ch06', '6.4')}>ch. 6 §6.4</a>.
      </p>
    ),
    source: 'ch. 6 §6.4, pp. 139–141, and exercise 5, p. 155',
  },
  'ch15/start/q-registers': {
    kind: 'numeric',
    id: 'ch15/start/q-registers',
    prompt: (
      <>
        On the book&rsquo;s example SM, with 65,536 registers and room for 2048 threads, a kernel needs 128 registers per
        thread. At most what percentage of the SM&rsquo;s thread slots can be filled?
      </>
    ),
    answer: 25,
    tol: 0.01,
    unit: '%',
    explain: (
      <p>
        65,536 / 128 = 512 threads, and 512 / 2048 = 25%. The book&rsquo;s own example is 64 registers for 50%. This
        chapter&rsquo;s kernel goes further: close to the 255-register maximum, one block of 256 threads per SM, 12.5%,
        and section <a href={href('ch15', '15.7')}>15.7</a> argues that this is the right trade. Taught in{' '}
        <a href={href('ch04', '4.7')}>ch. 4 §4.7</a>.
      </p>
    ),
    source: "ch. 4 §4.7, p. 86 (the device figures are the book's Hopper H100 example)",
  },
};

// ---------------------------------------------------------------- your starting point
// Lab 15.1's benchmark times lab 5.3's reference kernel (row "tiled_5_3") and the vendor's
// GEMM (row "vendor") beside the stage itself, at the same shapes (labs/ch15/shared/stage_driver.hpp).
const [L151] = defineLabs('ch15', [{ num: '15.1', dir: 'lab1_coarse', name: 'coarsened' }]);
const SIDE = 2048;
const CUBE = { m: SIDE, k: SIDE, n: SIDE };
const WHOSE = "lab 15.1's benchmark (lab 5.3's reference kernel, T = 16, and the vendor library)";
const gflops = (r?: BenchRow) => num(r?.gflops);

/** Lab 5.3's speed as a percentage of the library's, at 2048 cubed. */
export const gapAt2048 = (reports: Reports) =>
  measure(reports, L151, (s: BenchSet) => {
    const mine = gflops(benchRow(s, 'tiled_5_3', CUBE));
    const lib = gflops(benchRow(s, 'vendor', CUBE));
    if (!mine || !lib) return null;
    return { value: (100 * mine) / lib, detail: `${fmt(mine)} against ${fmt(lib)} GFLOP/s`, whose: WHOSE };
  });

/** Every shape lab 15.1 timed: lab 5.3's kernel against the library, one table per backend.
 *  Shown once a prediction is committed. */
function GapTable() {
  const { reports } = useLabs();
  const sets = preferredBench(reports, L151);
  const tables = sets
    .map((s) => ({
      s,
      rows: s.rows
        .filter((r) => r.name === 'tiled_5_3')
        .map((r) => ({ r, mine: gflops(r), lib: gflops(benchRow(s, 'vendor', { m: num(r.m) ?? -1, k: num(r.k) ?? -1, n: num(r.n) ?? -1 })) })),
    }))
    .filter((t) => t.rows.length);
  if (!tables.length) return null;
  return (
    <div className="panel" style={{ overflowX: 'auto', marginTop: '0.6rem' }}>
      {tables.map(({ s, rows }) => (
        <table className="data" style={{ margin: 0 }} data-gap-table={s.backend} key={s.backend}>
          <thead>
            <tr>
              <th>
                {s.backend} <span className="muted">({s.device})</span>, M × K × N
              </th>
              <th className="num">Lab 5.3, GFLOP/s</th>
              <th className="num">Library, GFLOP/s</th>
              <th className="num">Lab 5.3 as % of the library</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ r, mine, lib }, i) => (
              <tr key={i}>
                <td style={{ whiteSpace: 'nowrap' }}>
                  {String(r.m)} × {String(r.k)} × {String(r.n)}
                </td>
                <td className="num">{mine ? fmt(mine) : ''}</td>
                <td className="num">{lib ? fmt(lib) : ''}</td>
                <td className="num">{mine && lib ? fmt((100 * mine) / lib) : ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ))}
    </div>
  );
}

function StartingPoint() {
  return (
    <LabsPage labs={[L151]}>
      <Bench
        title="Your starting point"
        task="Commit a number, then see this machine's measurement."
        foot={<>Read from this machine&rsquo;s recorded lab runs (<code>reports/runs/</code>). Nothing here is an expected value.</>}
      >
        <Predict
          id="ch15/start/p-gap"
          prompt={
            <>
              At {SIDE} × {SIDE} × {SIDE}, what percentage of the vendor library&rsquo;s speed does chapter 5&rsquo;s tiled
              kernel reach on your device?
            </>
          }
          unit="% of the library"
        >
          {(v) => (
            <>
              <Reveal
                committed={v}
                measure={gapAt2048}
                unit="% of the library"
                reading={
                  <>
                    100% would be the library&rsquo;s speed. Both sides compute in FP32 (section{' '}
                    <a href={href('ch15', '15.1')}>15.1</a>), so the gap is technique, not precision. The table has every
                    shape the benchmark times; look at the batch-1 layer before you read section{' '}
                    <a href={href('ch15', '15.2')}>15.2</a>.
                  </>
                }
                missing={
                  <>
                    Run <code>python3 labs/pmpp.py run 15.1 --solution --bench</code> on a machine with a GPU. It times
                    lab 5.3&rsquo;s reference kernel and the library whether or not you have written lab 15.1 yet.
                  </>
                }
              />
              <GapTable />
            </>
          )}
        </Predict>
      </Bench>
    </LabsPage>
  );
}

// ---------------------------------------------------------------- the chapter at a glance
const MAP: { part: string; rows: [string, string, string][] }[] = [
  {
    part: 'The model',
    rows: [
      ['15.1', 'Background', 'Map a linear layer or an attention product onto GEMM and onto a library call'],
      ['15.2', 'Data reuse analysis', 'Compute the intensity of any output tile; see why only its size matters'],
    ],
  },
  {
    part: 'The kernel, one bottleneck at a time',
    rows: [
      ['15.3', 'Larger tiles with thread coarsening', 'A 128 × 128 tile per block, an 8 × 8 patch per thread, in registers'],
      ['15.4', 'Register tiling of the input tiles', 'Read each shared value once per thread, by swapping two loops'],
      ['15.5', 'Coalesced storing of the output tile', 'Re-map threads so that a warp’s stores land side by side'],
      ['15.6', 'Eliminating bank conflicts', 'One padding column, and the line of code the book leaves out'],
    ],
  },
  {
    part: 'Paying for it',
    rows: [
      ['15.7', 'Occupancy considerations', 'Why one block per SM is the right trade here'],
      ['15.8', 'Software pipelining', 'Load the next tiles while computing on these, with one barrier'],
    ],
  },
  {
    part: 'Beyond your kernel',
    rows: [['15.9', 'Specialized software and hardware', 'Libraries, tensor cores, asynchronous copies, and what each backend has']],
  },
];
const LADDER: [string, string, string][] = [
  ['15.1', 'The 128 × 128 coarsened kernel', '15.3'],
  ['15.2', 'Register tiling: the loops interchanged', '15.4'],
  ['15.3', 'Coalesced stores: four 4 × 4 sub-tiles and float4 stores', '15.5'],
  ['15.4', 'Padding and vector loads', '15.6, 15.7'],
  ['15.5', 'Double buffering and software pipelining', '15.8'],
  ['15.6', 'The fast linear layer (milestone M3)', '15.3 to 15.8'],
  ['15.7', 'Bug hunt: three defects in a fast linear layer', '15.6, 15.8'],
];

export function S150() {
  const q = questionsS150;
  return (
    <>
      <p>
        Matrix multiplication does a great deal of arithmetic on comparatively little data, so every element is used many
        times, and there is a lot to gain from deciding who loads what. Chapter 3 used it to map threads onto a grid,
        chapter 5 to introduce tiling. This chapter goes further than either, with optimisations that are specific to
        GPUs.
      </p>
      <Source pages="349" />
      <p>
        Section 5.4 closed with a remark: on the book&rsquo;s example GPU, 32 × 32 tiles still leave the kernel short of
        data, at 40% of the arithmetic the device can do. Going further was left to this chapter and to libraries such
        as cuBLAS and CUTLASS.
      </p>
      <Source section="5.4" pages="111" />
      <p>
        So the chapter takes the tiled kernel apart and rebuilds it, one bottleneck at a time, into the skeleton of a
        library GEMM: a 128 × 128 output tile per block, an 8 × 8 patch of it in each thread&rsquo;s registers, input
        strips in registers too, stores arranged so that a warp&rsquo;s writes land side by side, a padded shared tile, a
        block that runs alone on its SM on purpose, and a loop that loads the next tiles while it computes on the
        current ones. The book argues every step on paper and gives no timings. The labs supply them: you build the
        kernel stage by stage, time each stage against chapter 5&rsquo;s kernel and the vendor&rsquo;s library on your
        own GPU, and finish with the project&rsquo;s linear layer, milestone M3.
      </p>

      <h2>Underneath your framework</h2>
      <p>
        When PyTorch multiplies two float tensors on an NVIDIA GPU it calls cuBLAS or cuBLASLt, and on ROCm it can call
        Composable Kernel. Those libraries are built from this chapter&rsquo;s ideas plus hardware the chapter only
        surveys. llama.cpp shows where the ideas stop paying: its CUDA backend switches to matrix-vector kernels when a
        weight matrix serves 8 tokens or fewer, which is what generating one token at a time does, and uses tiled matrix
        kernels above that. Section <a href={href('ch15', '15.2')}>15.2</a> gives the reason in one line of arithmetic.
      </p>
      <Beyond>
        <p>
          Read from PyTorch&rsquo;s <code>CUDABlas.cpp</code> and llama.cpp&rsquo;s <code>mmvq.cuh</code> and{' '}
          <code>mmvf.cuh</code> on 2026-11-20 (<code>notes/today/ch15-facts.md</code> D1, D4; ch06-facts D1).
        </p>
      </Beyond>

      <h2>Your starting point</h2>
      <p>
        Lab 15.1&rsquo;s benchmark times three kernels side by side: lab 5.3&rsquo;s reference solution (chapter 5&rsquo;s
        tiled kernel with T = 16), the stage you are building, and the vendor&rsquo;s single-precision GEMM (cuBLAS,
        hipBLAS or Metal Performance Shaders). Before you look at the first and the last, commit a number.
      </p>
      <StartingPoint />

      <h2>The chapter at a glance</h2>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" data-map>
          <thead>
            <tr>
              <th>About</th>
              <th>Section</th>
              <th>What you will be able to do</th>
            </tr>
          </thead>
          <tbody>
            {MAP.flatMap((g) =>
              g.rows.map(([id, title, what], i) => (
                <tr key={id}>
                  <td>{i === 0 ? g.part : ''}</td>
                  <td>
                    <a href={href('ch15', id)}>
                      {id} {title}
                    </a>
                  </td>
                  <td>{what}</td>
                </tr>
              )),
            )}
          </tbody>
        </table>
      </div>
      <p>
        Seven labs, on the <a href={href('ch15', 'labs')}>labs page</a>. The first five are one ladder: each starts from
        the previous lab&rsquo;s solution with one part removed, and each benchmark puts your stage beside the one
        before it.
      </p>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" data-labs>
          <thead>
            <tr>
              <th>Lab</th>
              <th>What you build</th>
              <th>Sections</th>
            </tr>
          </thead>
          <tbody>
            {LADDER.map(([n, t, s]) => (
              <tr key={n} data-lab={n}>
                <td>{n}</td>
                <td>{t}</td>
                <td>{s}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>Before you start</h2>
      <p>
        Four questions on what this chapter assumes from chapters 4 to 6. Each explanation points to where the idea was
        taught and to the section here that builds on it. They come back in <a href={href('review')}>review</a>.
      </p>
      <Ask q={q['ch15/start/q-tile-ratio']} label="From chapter 5" />
      <Ask q={q['ch15/start/q-local-array']} label="From chapters 5 and 6" />
      <Ask q={q['ch15/start/q-bank-stride']} label="From chapter 6" />
      <Ask q={q['ch15/start/q-registers']} label="From chapter 4" />
    </>
  );
}
