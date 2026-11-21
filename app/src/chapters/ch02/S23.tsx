import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { benchRows, fmt, num, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  FLOAT_BYTES,
  boundaryTraffic,
  chainOps,
  chainTimes,
  program,
  type Memory,
  type Strategy,
  type Traffic,
} from '../../sim/offload';
import { RoundTripScene } from './RoundTripScene';

const SEQUENTIAL = 'ch02/snippets/vecadd_sequential.cu';
const STARTER = 'ch02/lab1_vecadd/starter/kernel.cu';
const LAB3 = 'ch02/lab3_elementwise';

/** The host function of lab 2.1's kernel.cu, with the line of the file it starts on. */
export function hostFunction(src: string): { code: string; firstLine: number } | null {
  const lines = src.replace(/\n$/, '').split('\n');
  const at = lines.findIndex((l) => /^int vecAdd\(/.test(l));
  return at < 0 ? null : { code: lines.slice(at).join('\n'), firstLine: at + 1 };
}

// The corrected declaration line of Fig. 2.5, as the tested stub has it (sim/offload.test.ts ties it to the file).
const DECLARATION = program('cuda').head[2].trim();

// ---------------------------------------------------------------- interactive A: boundary traffic
const STRATEGY_LABEL: Record<Strategy, string> = {
  transparent: 'every operation is a transparent offload',
  resident: 'data stays on the device between operations',
};
const MEMORY_LABEL: Record<Memory, string> = {
  separate: 'separate device memory (CUDA, HIP)',
  shared: 'one shared memory (Metal on Apple silicon, data created in shared buffers)',
};

export function bytesText(b: number): string {
  if (b === 0) return '0 bytes';
  const units = ['bytes', 'KiB', 'MiB', 'GiB', 'TiB'];
  let v = b;
  let u = 0;
  while (v >= 1024 && u < units.length - 1) {
    v /= 1024;
    u++;
  }
  return `${fmt(v)} ${units[u]}`;
}
const count = (v: number) => v.toLocaleString('en-US');

// The picture is drawn narrow on purpose, so that its labels stay legible in a
// half-width column and on a phone.
const PW = 380;
const ROW = 54; // one operation: up to three arrows
const TOP = 46;
const HOST_W = 104;
const DEV_W = 132;
const SLOT = 15;

/** A host box, a device box, and one arrow per tensor-sized transfer. */
function TrafficPicture({ t, memory }: { t: Traffic; memory: Memory }) {
  const k = t.ops.length;
  const h = TOP + k * ROW + 12;
  const hostX = 1;
  const devX = PW - 1 - DEV_W;
  const x0 = hostX + HOST_W + 4; // where an arrow meets host memory
  const x1 = devX - 4; // where it meets device memory
  const style = { width: '100%', maxWidth: 460, display: 'block' } as const;
  const ops = t.ops.map((name, i) => (
    <g key={i} data-op={name}>
      <rect x={devX + 8} y={TOP + i * ROW + 6} width={DEV_W - 16} height={ROW - 12} rx="3" fill="var(--global-wash)" stroke="var(--global)" />
      <text x={devX + DEV_W / 2} y={TOP + i * ROW + ROW / 2 + 4} fontSize="12" textAnchor="middle">{name}</text>
    </g>
  ));
  if (memory === 'shared')
    return (
      <svg viewBox={`0 0 ${PW} ${h}`} role="img" aria-label="One shared memory: no transfers" data-picture="shared" style={style}>
        <rect x={hostX} y={10} width={PW - 2} height={h - 14} rx="4" fill="var(--page)" stroke="var(--rule)" strokeWidth="1.4" />
        <text x={hostX + 12} y={30} fontSize="12.5" fontWeight="600">One memory, used by host code and kernels</text>
        <text x={hostX + 12} y={TOP + 24} fontSize="12.5">Nothing crosses:</text>
        <text x={hostX + 12} y={TOP + 41} fontSize="12.5">there is no boundary.</text>
        <text x={hostX + 12} y={TOP + 66} fontSize="12" style={{ fill: 'var(--ink-2)' }}>The kernels still read</text>
        <text x={hostX + 12} y={TOP + 82} fontSize="12" style={{ fill: 'var(--ink-2)' }}>and write every byte.</text>
        {ops}
      </svg>
    );
  const slot = new Map<number, number>(); // how many arrows this operation's row already holds
  return (
    <svg viewBox={`0 0 ${PW} ${h}`} role="img" aria-label="Transfers between host memory and device memory" data-picture="separate" style={style}>
      <rect x={hostX} y={10} width={HOST_W} height={h - 14} rx="4" fill="var(--page)" stroke="var(--rule)" strokeWidth="1.4" />
      <text x={hostX + 10} y={30} fontSize="12.5" fontWeight="600">Host memory</text>
      <rect x={devX} y={10} width={DEV_W} height={h - 14} rx="4" fill="var(--page)" stroke="var(--rule)" strokeWidth="1.4" />
      <text x={devX + 10} y={30} fontSize="12.5" fontWeight="600">Device memory</text>
      <line x1={(x0 + x1) / 2} y1={14} x2={(x0 + x1) / 2} y2={h - 6} stroke="var(--ink-3)" strokeDasharray="4 4" />
      {ops}
      {t.transfers.map((tr, j) => {
        const s = slot.get(tr.op) ?? 0;
        slot.set(tr.op, s + 1);
        const y = TOP + tr.op * ROW + ROW / 2 - SLOT + s * SLOT;
        const into = tr.dir === 'in';
        const tip = into ? x1 : x0;
        const d = into ? -7 : 7;
        return (
          <g key={j} data-transfer={tr.dir}>
            <line x1={x0} y1={y} x2={x1} y2={y} stroke="var(--ink-2)" strokeWidth="1.4" />
            <polygon points={`${tip},${y} ${tip + d},${y - 3.5} ${tip + d},${y + 3.5}`} fill="var(--ink-2)" />
            <text x={hostX + HOST_W - 8} y={y + 4} fontSize="12" textAnchor="end" style={into ? undefined : { fontWeight: 650 }}>
              {into ? tr.what : `${tr.what} back`}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function ChainMeasured() {
  const { reports, loading } = useReports();
  const found = Object.keys(reports)
    .filter((b) => b !== 'emu' && b !== 'race')
    .flatMap((b) =>
      benchRows(reports[b], LAB3)
        .filter((r) => r.name === 'chain')
        .flatMap((r) => {
          const t = chainTimes(r);
          const rows = num(r.rows);
          const d = num(r.d);
          const shape = rows && d ? `${count(rows)} rows of ${count(d)} floats` : num(r.n) ? `n = ${count(num(r.n)!)}` : '';
          // Lab 2.3 says in words what one round trip costs on this backend (its 'copies' column).
          const trip = typeof r.copies === 'string' ? r.copies : typeof r.roundtrip === 'string' ? r.roundtrip : '';
          return t ? [{ b, device: String(reports[b].device.name ?? b), shape, trip, ...t }] : [];
        }),
    );
  if (loading) return null;
  if (!found.length)
    return (
      <p className="ui" data-out="chain-measured" data-found="0" style={{ margin: '0.9rem 0 0' }}>
        No measurement on this machine yet, so no time is shown: this page does not model one. Run{' '}
        <code>python3 labs/pmpp.py run 2.3 --solution --bench</code> and reload this page: the run records its
        times, and those for the resident chain and for three round trips appear here.
      </p>
    );
  return (
    <div data-out="chain-measured" data-found={found.length} style={{ margin: '0.9rem 0 0' }}>
      {found.map((f, i) => (
        <p className="ui" key={i} style={{ margin: '0 0 0.3rem' }}>
          Lab 2.3 on <b>{f.device}</b> ({f.b}){f.shape ? `, ${f.shape}` : ''}: its chain of three operations took{' '}
          <b>{fmt(f.resident * 1e3)} ms</b> as one call that keeps the data where the kernels run and{' '}
          <b>{fmt(f.roundTrips * 1e3)} ms</b> as three round trips{f.trip ? ` (each round trip: ${f.trip})` : ''}.
          {f.b === 'metal'
            ? ' On Metal nothing crosses a boundary: what the round trips pay for is copying host arrays into shared buffers and back, which an interface that takes host arrays forces.'
            : ''}
        </p>
      ))}
      <p className="muted" style={{ margin: 0 }}>Measured on your machine, not predicted. Another device can differ widely in either direction.</p>
    </div>
  );
}

function BoundaryTraffic() {
  const [k, setK] = useState(3);
  const [exp, setExp] = useState(20);
  const [strategy, setStrategy] = useState<Strategy>('transparent');
  const [memory, setMemory] = useState<Memory>('separate');
  const n = 2 ** exp;
  const both = useMemo(
    () => ({
      transparent: boundaryTraffic(k, 'transparent', memory, n),
      resident: boundaryTraffic(k, 'resident', memory, n),
    }),
    [k, memory, n],
  );
  const t = both[strategy];
  const ops = chainOps(k);
  const cols: Strategy[] = ['transparent', 'resident'];
  const row = (label: string, key: string, f: (x: Traffic) => string) => (
    <tr>
      <td style={{ paddingLeft: '0.6rem' }}>{label}</td>
      {cols.map((s) => (
        <td key={s} data-cell={`${key}-${s}`} style={{ fontWeight: s === strategy ? 650 : 400, fontVariantNumeric: 'tabular-nums' }}>
          {f(both[s])}
        </td>
      ))}
    </tr>
  );
  return (
    <Bench
      title="Boundary traffic"
      task="Lengthen the chain and watch the two columns of the table. Then switch to one shared memory."
      foot={
        <>
          Each operation combines the running result <code>x</code> with one fresh tensor of n floats ({FLOAT_BYTES}{' '}
          bytes each). An arrow is one tensor-sized transfer. The counts are exact; there are no times in this picture.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <span>Operations in the chain: {k}</span>
          <input type="range" min="1" max="8" value={k} aria-label="Operations in the chain" onChange={(e) => setK(+e.target.value)} />
        </label>
        <label className="field">
          <span>Elements per tensor: n = 2<sup>{exp}</sup> ({bytesText(n * FLOAT_BYTES)})</span>
          <input type="range" min="10" max="28" value={exp} aria-label="Elements per tensor, as a power of two" onChange={(e) => setExp(+e.target.value)} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.5rem' }}>
        <span className="seg" role="group" aria-label="Strategy" style={{ flexWrap: 'wrap' }}>
          {cols.map((s) => (
            <button key={s} aria-pressed={strategy === s} onClick={() => setStrategy(s)}>{STRATEGY_LABEL[s]}</button>
          ))}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <span className="seg" role="group" aria-label="Memory" style={{ flexWrap: 'wrap' }}>
          {(['separate', 'shared'] as Memory[]).map((m) => (
            <button key={m} aria-pressed={memory === m} onClick={() => setMemory(m)}>{MEMORY_LABEL[m]}</button>
          ))}
        </span>
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)' }}>
        <div className="panel" style={{ background: 'var(--bench)' }}>
          <TrafficPicture t={t} memory={memory} />
        </div>
        <div>
          <p className="ui" data-out="traffic-summary" style={{ margin: '0 0 0.6rem' }}>
            {ops.map((o) => o.name).join(', ')}:{' '}
            <b>
              {t.transfers.length} tensor-sized {t.transfers.length === 1 ? 'transfer' : 'transfers'}, {bytesText(t.bytesAcross)}
            </b>{' '}
            across the boundary.
          </p>
          <table className="data" style={{ background: 'var(--page)', margin: 0, width: '100%' }}>
            <thead>
              <tr>
                <th style={{ paddingLeft: '0.6rem' }}>For this chain</th>
                <th>each operation a round trip</th>
                <th>data kept on the device</th>
              </tr>
            </thead>
            <tbody>
              {row('tensor-sized transfers in', 'in', (x) => String(x.transfersIn))}
              {row('tensor-sized transfers out', 'out', (x) => String(x.transfersOut))}
              {row('bytes across the boundary', 'across', (x) => bytesText(x.bytesAcross))}
              {row('bytes the kernels read and write', 'kernel', (x) => bytesText(x.kernelBytes))}
              {row('bytes across per byte the kernels touch', 'ratio', (x) => fmt(x.ratio))}
            </tbody>
          </table>
          <ChainMeasured />
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
const FOUR = {
  transparent: boundaryTraffic(4, 'transparent', 'separate', 1).transfers.length,
  resident: boundaryTraffic(4, 'resident', 'separate', 1).transfers.length,
};

export const questionsS23: Record<string, Question> = {
  'ch02/2.3/q-output-only': {
    kind: 'choice',
    id: 'ch02/2.3/q-output-only',
    prompt: (
      <>
        For <code>C = A + B</code>, the stub allocates three device arrays. Which of them must be filled by a copy
        before the launch?
      </>
    ),
    options: [
      {
        label: 'All three.',
        why: 'The result is still right. You have paid for one more transfer of n floats, and the kernel overwrites every one of them.',
      },
      { label: <><code>A_d</code> and <code>B_d</code> only.</>, correct: true },
      {
        label: <>None: the kernel reads <code>A_h</code> and <code>B_h</code> from host memory.</>,
        why: 'The kernel is handed device addresses. In this chapter kernels do not touch host memory at all.',
      },
      { label: <><code>C_d</code> only.</>, why: 'C_d is the one array whose old contents nobody reads.' },
    ],
    explain: (
      <p>
        <code>C</code> is written and never read, so it needs an allocation and not its old contents. The book's more
        complete listing does exactly this: three allocations, two copies in, one copy out. The rule carries to every
        stub you will write: copy in what the kernel reads, copy out what the caller needs, and allocate everything
        the kernel touches.
      </p>
    ),
    source: 'ch. 2 §2.4, p. 31, Fig. 2.8',
  },
  'ch02/2.3/q-pays': {
    kind: 'choice',
    id: 'ch02/2.3/q-pays',
    prompt: (
      <>
        You may offload exactly one of two host functions transparently (copy in, compute, copy out): adding two
        vectors of n·n floats, or multiplying two n × n matrices of floats. Which is more likely to repay its copies as
        n grows?
      </>
    ),
    options: [
      {
        label: 'The addition: it is the simpler kernel, so the device finishes it sooner.',
        why: 'Finishing sooner is the trouble. It does one addition for every 12 bytes it moves, at every n.',
      },
      { label: 'The matrix product: its arithmetic grows faster than the bytes it has to move.', correct: true },
      {
        label: 'Neither can be told apart: both move the same number of bytes.',
        why: 'They do move the same bytes, which is exactly why the one that does more arithmetic per byte moved is the better bet.',
      },
    ],
    explain: (
      <>
        <p>
          Both calls copy two inputs of n² floats in and one result out: 12n² bytes across the boundary. The addition
          does n² additions for that, one per 12 bytes, and the ratio never improves. The product does about n³
          multiplications and n³ additions, so the arithmetic bought per byte moved grows in proportion to n.
        </p>
        <p>
          This is reasoning about counts, not a measurement. Counting operations per byte is the method of{' '}
          <a href={href('ch05', '5.1')}>chapter 5, section 5.1</a>, applied there to the bytes a kernel fetches and
          here to the bytes a stub copies.
        </p>
      </>
    ),
    source: 'our reasoning from ch. 2 §2.3, p. 28, with the counting method of ch. 5 §5.1',
  },
  'ch02/2.3/q-interface': {
    kind: 'choice',
    id: 'ch02/2.3/q-interface',
    prompt: (
      <>
        On Apple silicon you keep the signature{' '}
        <code>vecAdd(const float* A_h, const float* B_h, float* C_h, int n)</code>, and the caller passes ordinary heap
        arrays. The CUDA stub copies data three times. How many of those three copies can a Metal version with
        shared-storage buffers avoid?
      </>
    ),
    options: [
      {
        label: 'All three: with shared storage the GPU uses the caller\'s arrays as they are.',
        why: 'Shared storage lets the CPU and the GPU use the same bytes. It does not put the caller\'s heap arrays inside a buffer.',
      },
      {
        label: 'None, unless the arrays are page-aligned and wrapped without copying, or the caller is changed to create its data in buffers.',
        correct: true,
      },
      {
        label: 'Only the copy back: the result buffer can be read in place.',
        why: 'It can be read in place, but this signature promises the result in C_h, which is a different allocation. Getting it there is a copy.',
      },
    ],
    explain: (
      <p>
        A Metal kernel's array arguments are bound to buffers. <code>newBufferWithBytes:length:options:</code> makes one by copying the
        caller's array; <code>newBufferWithBytesNoCopy:</code> wraps existing memory, and only when the pointer and the
        length are page-aligned. Shared memory removes the need for a copy. It does not remove the copies an interface
        forces.
      </p>
    ),
    source: 'not in the book; Apple documentation as recorded in notes/today/ch02-facts.md, C2 and C3 (read 2026-11-20)',
  },
};

// ---------------------------------------------------------------- the section
export function S23() {
  const sequential = useLabFile(SEQUENTIAL);
  const starterFile = useLabFile(STARTER);
  const starter = starterFile ? hostFunction(starterFile) : null;
  return (
    <>
      <p>
        The book's first program adds two vectors: <code>C[i] = A[i] + B[i]</code> for every <code>i</code>. It calls
        this the hello world of data-parallel programming, and by the test of{' '}
        <a href={href('ch02', '2.1')}>section 2.1</a> it is the easiest case there is: each output reads one element of
        each input and nothing that another iteration writes. Here it is with no GPU involved.
      </p>
      <Source section="2.3" pages="26" figs="Fig. 2.4" />
      <Code
        tabs={[{ label: 'host only', code: sequential ?? '// loading labs/' + SEQUENTIAL, status: 'front-end-checked' }]}
        caption={
          <>
            <code>labs/{SEQUENTIAL}</code>: the book's Fig. 2.4 with <code>const</code> on the inputs and real data in
            place of its comments. There is no device code in this file, and it is still a CUDA C++ program: any
            ordinary C++ program is one that happens to have only host code (p. 24).
          </>
        }
      />
      <p>
        The suffix on the names is a convention the book introduces here: <code>_h</code> marks data that lives in
        host memory and <code>_d</code> data that lives on the device. From now on a function can hold two pointers to
        the same vector in two different memories, and the suffix is the only thing that tells you which is which.
        This course applies it everywhere. The book's own listings do not; the{' '}
        <a href={href('ch02', 'wrap')}>summary page</a> lists where they drop it.
      </p>
      <Source section="2.3" pages="26" />

      <h2>Moving the loop to the device: three parts</h2>
      <p>
        To run the loop on a GPU, <code>vecAdd</code> keeps its name and its arguments and changes its body. The body
        now has three parts. Part 1 allocates memory on the device for A, B and C and copies A and B into it. Part 2
        launches a kernel, which is where the additions happen. Part 3 copies C back to the host and frees the three
        device arrays. This is the function you fill in for lab 2.1, as the lab hands it to you:
      </p>
      <Source section="2.3" pages="27" figs="Fig. 2.5" />
      <Code
        tabs={[{ label: 'lab 2.1 starter', code: starter?.code ?? starterFile ?? '// loading labs/' + STARTER }]}
        firstLine={starter?.firstLine ?? 1}
        caption={
          <>
            The host function in <code>labs/{STARTER}</code>, the file lab 2.1 starts from (the kernel above it is left
            out here). The three parts are the book's; the lab adds a return value, a block size chosen by the caller,
            and a check after the launch that <a href={href('ch02', '2.6')}>section 2.6</a> explains.
          </>
        }
      />
      <p>
        One line of the book's skeleton needs a warning before you type it. Line 03 of Fig. 2.5 is printed as{' '}
        <code>float *d_A *d_B, *d_C;</code>. A comma is missing, so it does not compile, and it uses a{' '}
        <code>d_</code> prefix where the text and every later figure use the suffix. The line the course uses, and
        tests, is <code>{DECLARATION}</code>. The hunt under Error checking in{' '}
        <a href={href('ch02', '2.4')}>section 2.4</a> shows you the printed line again, beside the slip in the book's
        error-check snippet.
      </p>
      <Source section="2.3" pages="27" figs="Fig. 2.5, line 03" />

      <h2>Three parts on two lanes</h2>
      <p>
        The book's figure for the three parts is two boxes and two arrows: host memory, device memory, Part 1 across,
        Part 3 back. What a still picture cannot show is who is waiting for whom. There are two lanes of activity, a
        host thread and a device, and two memories, and at most moments one lane is idle or blocked. The animation
        follows 8 small numbers through the whole function, and then plays the same steps again on a machine where the
        boundary between the two memories does not exist.
      </p>
      <Simplification>
        <p>
          The model behind the animation has no clock. Step widths are not durations, and how many results the kernel
          has produced when the host reaches the copy back is the drawing's choice. What is exact: which values are
          where after each call, how many bytes have crossed, and in which calls the host thread waits. Results appear
          out of order on purpose, because nothing promises an order among threads.
        </p>
      </Simplification>
      <RoundTripScene />
      <p>
        Two things in the animation are not on the book's pages. The chapter never says whether a launch waits for
        the kernel, and its listings never wait explicitly. They work because of what the copy back does:{' '}
        <a href={href('ch02', '2.2')}>section 2.2</a> has the launch returning at once, and here you saw the copy
        refuse to return until the kernel ahead of it was done.
      </p>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md, A4 and A5)">
        <p>
          NVIDIA's guide says three things: a launch is asynchronous with respect to the host thread,{' '}
          <code>cudaMemcpy</code> does not return until the copy has completed, and operations in the default stream
          run in the order they were issued. That the copy back therefore waits for the kernel is our reading of those
          three statements together. No single sentence of the guide says it.
        </p>
      </Today>
      <Ask q={questionsS23['ch02/2.3/q-output-only']} />

      <h2>An outsourcing agent, and what it charges</h2>
      <p>
        Seen from <code>main</code>, nothing changed. It calls <code>vecAdd(A, B, C, N)</code> with host arrays and
        finds the sums in <code>C</code>. The function has become an agent that ships the inputs out, has the work done
        elsewhere, and ships the result home, and the caller cannot tell. The book calls this transparent outsourcing
        and warns about it in the same breath: all that copying back and forth can make it very inefficient. Real code
        keeps its large data structures on the device and calls device functions on them from the host. The
        transparent version is in this chapter because it is the simplest thing that is complete.
      </p>
      <Source section="2.3" pages="28" />
      <p>
        The animation put a number on one call: 12 bytes moved for every element added. The cost that matters in
        practice is what happens when calls are chained, because the output of one transparent call is the input of
        the next and it travels home and back out in between.
      </p>
      <Predict
        id="ch02/2.3/p-crossings"
        prompt={
          <>
            Four elementwise operations in a chain. Each combines the running result with one new tensor of n floats,
            and each is written as its own transparent offload, like <code>vecAdd</code>. In total, how many
            tensor-sized transfers cross the boundary between host and device?
          </>
        }
        unit="transfers"
      >
        {(v) => (
          <>
            <p>
              It is {FOUR.transparent}: every call copies two tensors in and one out.{' '}
              {v === FOUR.transparent
                ? 'That is what you counted.'
                : v === FOUR.resident
                  ? 'Your count is what the chain needs when the data stays on the device, which is the point of the next step.'
                  : 'Count per call, not per tensor: the running result goes out and comes home at every step.'}
            </p>
            <p>
              Keep the data on the device between the operations and it is {FOUR.resident}: the start value and the
              four operands in, one result out. Set the explorer below to 4 operations and switch between the two.
            </p>
          </>
        )}
      </Predict>
      <p>
        The explorer counts this for any chain. Its default is a chain of three elementwise operations: add a bias,
        scale, add a residual. Find the rule for each column of the table as the chain grows, then find the row that
        is the same in both columns.
      </p>
      <Simplification>
        <p>
          The explorer counts bytes that cross the boundary, which is exact. It does not model how long a byte takes to
          cross, page-locked host memory, or allocators that cache device memory. The only times it shows are your own
          measurements from lab 2.3.
        </p>
      </Simplification>
      <BoundaryTraffic />
      <p>
        A transparent chain moves three tensors per operation. A resident chain moves each input once and the result
        once, k + 2 tensors for k operations, and the gap widens with every operation you add. The row that does not
        move is the kernels' own work: they read and write the same bytes either way. Keeping data resident does not
        make any kernel faster. It removes work that was never arithmetic.
      </p>

      <h2>Where this sits under a framework call</h2>
      <p>
        You already write code that obeys this rule. You call <code>.to(device)</code> on a model once and leave the
        weights there. You keep activations on the device from one layer to the next, and a decoder's cache of keys
        and values stays there from one token to the next. Each of those habits is the right-hand column of the table.
        And the call itself is this section's Part 1: in PyTorch's CUDA backend, moving a tensor between the host and
        the device ends in a host-to-device or device-to-host <code>cudaMemcpy</code>, the asynchronous variant when
        you ask for <code>non_blocking</code> and a copy followed by a wait when you do not.
      </p>
      <Beyond>
        <p>
          PyTorch is not in the book. The copy is in <code>aten/src/ATen/native/cuda/Copy.cu</code>, read on
          2026-11-20 (notes/today/ch02-facts.md, D4). That weights and caches are kept resident for this reason is our
          account of common practice, not a quotation.
        </p>
      </Beyond>

      <h2>The same stub on AMD and Apple GPUs</h2>
      <p>
        On AMD the stub is this one with other names. AMD's documentation describes the typical sequence of operations
        for a HIP host program as: initialise, allocate memory on the host and the device, transfer the input data to
        the device, launch the kernel, synchronise, handle errors, clean up. The calls are <code>hipMalloc</code>,{' '}
        <code>hipMemcpy</code> and <code>hipFree</code>, with the same parameters in the same order.
      </p>
      <Beyond>
        <p>
          The book uses CUDA only. The HIP statements are from AMD's programming-model page and runtime header
          (notes/today/ch02-facts.md, B1 and B3), read on 2026-11-20. No lab here has run on a HIP device.
        </p>
      </Beyond>
      <p>
        On Apple silicon the picture changes at Part 1 and Part 3, as the last part of the animation showed. A buffer
        created with shared storage is one piece of memory that both the CPU and the GPU use. If your data is created
        in such a buffer, nothing crosses anything. But a stub that keeps <em>this</em> interface, ordinary host arrays
        in and out, still has to get those arrays into buffers: by one copy each, or by wrapping memory that happens to
        be page-aligned. So the natural interface on Metal is buffers, not host arrays.{' '}
        <a href={href('ch02', '2.4')}>Section 2.4</a> steps through both versions call by call.
      </p>
      <Beyond>
        <p>
          Metal is not in the book. Shared storage, <code>newBufferWithBytes</code> and{' '}
          <code>newBufferWithBytesNoCopy</code> are as Apple documents them (notes/today/ch02-facts.md, C2 and C3;
          ch05-facts.md, C6).
        </p>
      </Beyond>
      <Ask q={questionsS23['ch02/2.3/q-interface']} />

      <h2>Which operations repay a round trip</h2>
      <p>
        Sometimes a round trip is the only option: the data starts in a file or a tokenizer, and the answer has to
        reach a CPU. Whether a kernel is worth that trip depends on how much arithmetic it does for each byte moved,
        and <a href={href('ch05', '5.1')}>chapter 5</a> turns that into a habit of counting operations per byte. One
        question uses it now.
      </p>
      <Ask q={questionsS23['ch02/2.3/q-pays']} />
      <p>
        The three parts are still only comments. <a href={href('ch02', '2.4')}>Section 2.4</a> fills in Parts 1 and 3
        with real calls, and <a href={href('ch02', 'labs')}>lab 2.1</a> has you write all three.
      </p>
    </>
  );
}
