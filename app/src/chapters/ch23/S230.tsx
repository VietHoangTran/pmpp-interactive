import { useState, type CSSProperties, type KeyboardEvent, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source } from '../../components/notes';
import { href } from '../../lib/router';
import {
  EXERCISE1,
  bestTiles,
  coverage,
  factorisations,
  firstRows,
  launches,
  partition,
  pointerExpr,
  rowKind,
  rowShares,
  tileOf,
  tileSplit,
  visibleRows,
  type Launch,
  type Partition,
  type TileSplit,
} from '../../sim/halopart';

// Prerequisite questions drawn from chapters 2, 4, 8, 10 and 22 and appendix C (their notes
// and facts files), plus one of this chapter's exercises. Each explanation says where the idea
// was taught and where this chapter uses it.
export const questionsS230: Record<string, Question> = {
  'ch23/start/q-queued': {
    kind: 'choice',
    id: 'ch23/start/q-queued',
    prompt: (
      <>
        <pre className="mono" style={{ margin: '0 0 0.6rem', whiteSpace: 'pre-wrap' }}>
          {'t0 = now();\nscale<<<g, b>>>(d, n);\nt1 = now();\ncudaDeviceSynchronize();\nt2 = now();'}
        </pre>
        The kernel runs for far longer than a launch takes. What does <code>t1 − t0</code> measure?
      </>
    ),
    options: [
      {
        label: 'Only the time to queue the launch; the kernel may not even have started.',
        correct: true,
      },
      {
        label: 'The kernel’s run time, because the call returns once the last block has finished.',
        why: 'That is t2 − t0, plus the queuing. A launch returns as soon as the kernel is queued (p. 556).',
      },
      {
        label: 'Nothing useful: a launch waits until the GPU is idle, so it is the previous kernel.',
        why: 'A launch does not wait for the GPU. Work queued earlier in the same stream runs first, on the GPU, while the host moves on.',
      },
      {
        label: (
          <>
            The copy of <code>d</code> to the GPU that the launch makes before the kernel starts.
          </>
        ),
        why: 'A launch copies nothing. d is a device pointer; its data was copied earlier, with cudaMemcpy.',
      },
    ],
    explain: (
      <p>
        Returning from a launch means the kernel has been put in a stream&rsquo;s queue, and nothing more: it may not
        even have started (p. 556). The host is free to queue more work, and section{' '}
        <a href={href('ch23', '23.3')}>23.3</a> is built on that: a copy or another kernel can be queued while this one
        runs. To time a kernel, wait for it inside the timed region, as <code>t2</code> does, or record events around it
        and read the time between them (page <a href={href('ch23', '23.4')}>23.3–23.5</a> uses events only for ordering). Taught in <a href={href('ch02', '2.6')}>ch. 2 §2.6</a>.
      </p>
    ),
    source: 'ch. 23 §23.3, p. 556; notes/today/ch02-facts.md A5',
  },
  'ch23/start/q-halo-rows': {
    kind: 'numeric',
    id: 'ch23/start/q-halo-rows',
    prompt: (
      <>
        A stencil that reads the four neighbours of each point runs on a grid of 128 rows, split by rows over 4 GPUs.
        GPU 1 owns global rows 32 to 63. How many rows must its local array hold?
      </>
    ),
    answer: 34,
    unit: 'rows',
    explain: (
      <p>
        Its own 32 rows, plus row 31 above (the last row of GPU 0) and row 64 below (the first row of GPU 2): 34. That
        is the book&rsquo;s <code>ny</code> for this GPU, which counts the two halo rows. The stencil reaches one row up
        and one row down, so one halo row on each side is enough. The explorer below starts on this split. Halo cells
        were taught in <a href={href('ch08', 'start')}>chapter 8</a>.
      </p>
    ),
    source: 'ch. 23 §23.1, p. 544; halo cells: ch. 8, p. 183',
  },
  'ch23/start/q-blockreduce': {
    kind: 'choice',
    id: 'ch23/start/q-blockreduce',
    prompt: (
      <>
        In the kernel of Fig. 23.1 every thread, including those that fail the bounds test, calls{' '}
        <code>blockReduce(residue*residue)</code>, and only thread (0,0) of each block then calls{' '}
        <code>atomicAdd</code>. Why must the <code>blockReduce</code> call sit outside the bounds test?
      </>
    ),
    options: [
      {
        label: 'Its barriers must be reached by every thread of the block; a thread with no point brings 0.',
        correct: true,
      },
      {
        label: 'So that each thread can make its own atomicAdd to the norm, once the block’s reduction is done.',
        why: 'Only thread (0,0) adds, once per block. Reducing first is what saves the other atomics.',
      },
      {
        label: 'To save the cost of divergence: a reduction runs faster when every thread takes one path.',
        why: 'Divergence would cost time. A barrier that some threads skip is worse: wrong results or a hang (ch. 4, p. 72).',
      },
      {
        label: 'Because the compiler rejects a call to a device function inside an if statement.',
        why: 'It compiles. Keeping every barrier reachable by the whole block is the programmer’s job (ch. 4, p. 72).',
      },
    ],
    explain: (
      <p>
        The book does not show <code>blockReduce</code>; it points to chapter 10 or to CUB for one (p. 543). A
        block-wide reduction from chapter 10 synchronises its threads with <code>__syncthreads</code>, and a barrier must
        be reached by every thread of the block (<a href={href('ch04', '4.3')}>ch. 4 §4.3</a>). An edge block has
        threads past the last row or column, so the reduction stays outside the <code>if</code> and those threads bring a
        residue of 0. The one <code>atomicAdd</code> per block is chapter 10&rsquo;s multi-block commit (
        <a href={href('ch10', '10.9')}>§10.9</a>).
      </p>
    ),
    source: 'ch. 23 §23.1, pp. 542–543; ch. 4 §4.3, p. 72; ch. 10 §10.9, pp. 244–245',
  },
  'ch23/start/q-amdahl': {
    kind: 'numeric',
    id: 'ch23/start/q-amdahl',
    prompt: (
      <>
        Chapter 22&rsquo;s example: 95% of a program&rsquo;s run time is made 100 times faster on the GPU and 5% stays
        on the CPU. If the CPU part can run while the GPU works, what speed-up does the whole program get?
      </>
    ),
    answer: 20,
    tol: 0.01,
    unit: '×',
    explain: (
      <p>
        With overlap the run lasts as long as the longer part: max(5%, 0.95%) = 5% of the original, so 1 / 0.05 = 20×.
        Without overlap the parts add: 1 / (5% + 0.95%) ≈ 16.8×, which the book prints as 17×. Section{' '}
        <a href={href('ch23', '23.3')}>23.3</a> makes the same move inside one step of the stencil: send the boundary
        rows while the internal rows are computed, so a step costs the longer of the two instead of their sum. Taught
        in <a href={href('ch22', '22.3')}>ch. 22 §22.3</a>.
      </p>
    ),
    source: 'ch. 22 §22.3, pp. 537–538; ch. 23 §23.3, p. 554',
  },
  'ch23/start/q-pinned': {
    kind: 'choice',
    id: 'ch23/start/q-pinned',
    prompt: (
      <>
        Appendix C says zero-copy memory, host memory that a kernel reads and writes directly, must be pinned. Why?
      </>
    ),
    options: [
      {
        label: 'A kernel reaches it over the interconnect at any moment, so the system must not move it.',
        correct: true,
      },
      {
        label: 'Pinned memory is faster to read, and a kernel cannot afford slow reads from the host.',
        why: 'Speed is not the reason. Zero-copy reads are slow whatever you do, limited by the interconnect (p. 625).',
      },
      {
        label: 'The CPU cannot write pinned memory, so the GPU never sees a half-written value.',
        why: 'The host reads and writes it through its own pointer; sharing it is the point (p. 624).',
      },
      {
        label: 'Pinning is what gives the buffer an address that a kernel is able to use.',
        why: 'That address comes from a separate call, cudaHostGetDevicePointer (p. 624). Pinning keeps the memory behind it in place.',
      },
    ],
    explain: (
      <p>
        The operating system may move a page of ordinary memory, or write it out to disk, at any time. A GPU using the
        memory at that moment would read or write whatever sits there now. Chapter 23 gives the same reason for copies:
        the DMA hardware behind <code>cudaMemcpy</code> works on physical addresses (pp. 562–563). Section{' '}
        <a href={href('ch23', '23.3')}>23.3</a> takes it from there: why an asynchronous copy wants its host buffer
        pinned. Taught in <a href={href('appC', 'C.2')}>appendix C §C.2</a>.
      </p>
    ),
    source: 'appendix C §C.2, pp. 624–625; ch. 23 §23.3, pp. 562–563',
  },
  'ch23/start/q-ex1-bytes': {
    kind: 'numeric',
    id: 'ch23/start/q-ex1-bytes',
    prompt: (
      <>
        This chapter&rsquo;s stencil runs on a grid 64 points wide and 512 tall, split by rows over 16 GPUs, with
        4-byte floats. How many bytes does each GPU send to its neighbours per step?
      </>
    ),
    answer: 512,
    unit: 'bytes',
    explain: (
      <p>
        512 / 16 = 32 rows each. Every step a GPU sends its top row up and its bottom row down: 2 × 64 × 4 = 512 bytes,
        256 to each neighbour. Neither the 16 nor the 512 rows appear in that product. The explorer&rsquo;s exercise
        preset shows parts 1a to 1d, with both readings of 1a, 1c and 1d: the exercise does not say whether the two edge
        columns the kernel skips count as computed.
      </p>
    ),
    source: 'ch. 23 §23.7, exercise 1, p. 576',
  },
};

// ---------------------------------------------------------------- formatting

const n = (v: number) => v.toLocaleString('en-US');
const pct = (v: number) => `${(100 * v).toLocaleString('en-US', { maximumFractionDigits: 1 })}%`;
const range = (a: number, b: number) => (a === b ? `${a}` : `${a} to ${b}`);

// Hatching for the two edge columns the kernel skips, and for stale halo rows. Ink only.
const HATCH = 'repeating-linear-gradient(135deg, var(--rule) 0 1.5px, transparent 1.5px 5px)';

// ---------------------------------------------------------------- the domain, one band per GPU

function DomainPicture({ p, onPick }: { p: Partition; onPick: (g: number) => void }) {
  const shares = rowShares(p.rows, p.gpus);
  const firsts = firstRows(shares);
  const X = 58;
  const Y = 26;
  const W = 120;
  const H = 300;
  const y = (r: number) => Y + (r / p.rows) * H;
  // An edge row drawn at its true height would vanish; give it at least 4 px, kept
  // inside its owner's band (above the band's bottom for a last row, below its top for a first).
  const rowStrip = (r: number, last: boolean) => {
    const t = Math.max(4, y(r + 1) - y(r));
    const y0 = last ? y(r + 1) - t : y(r);
    return { y0: Math.max(Y, Math.min(y0, Y + H - t)), h: t };
  };
  const order = shares.map((_, g) => g).sort((a, b) => (a === p.gpu ? 1 : b === p.gpu ? -1 : a - b));
  const key = (g: number) => (e: KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onPick(g);
    }
  };
  const topRow = firsts[p.top] + shares[p.top] - 1;
  const bottomRow = firsts[p.bottom];
  const halo = [rowStrip(topRow, true), rowStrip(bottomRow, false)];
  const edge = [rowStrip(p.first, false), rowStrip(p.last, true)];
  return (
    <svg viewBox="0 0 200 346" width="100%" style={{ maxWidth: 250, display: 'block', margin: '0 auto' }} data-domain>
      <title>The grid cut into one band of rows per GPU</title>
      <text x={X + W / 2} y={15} fontSize="11" textAnchor="middle">
        nx = {n(p.nx)}
      </text>
      <text x={X + W + 10} y={Y + H / 2} fontSize="11" textAnchor="middle" transform={`rotate(90 ${X + W + 10} ${Y + H / 2})`}>
        {n(p.rows)} rows
      </text>
      {order.map((g) => {
        const y0 = y(firsts[g]);
        const h = y(firsts[g] + shares[g]) - y0;
        const sel = g === p.gpu;
        return (
          <g
            key={g}
            data-gpu={g}
            role="button"
            tabIndex={0}
            aria-pressed={sel}
            aria-label={`GPU ${g}: rows ${range(firsts[g], firsts[g] + shares[g] - 1)}`}
            onClick={() => onPick(g)}
            onKeyDown={key(g)}
            style={{ cursor: 'pointer' }}
          >
            <rect
              x={X}
              y={y0}
              width={W}
              height={h}
              fill={sel ? 'var(--global-wash)' : g % 2 ? 'var(--rule-soft)' : 'var(--cell)'}
              stroke={sel ? 'var(--global)' : 'var(--rule)'}
              strokeWidth={sel ? 1.5 : 0.6}
            />
            {h >= 13 || sel ? (
              <text x={X - 6} y={Math.min(Math.max(y0 + h / 2 + 4, Y + 9), Y + H)} fontSize="10.5" textAnchor="end" fontWeight={sel ? 650 : 400}>
                GPU {g}
              </text>
            ) : null}
          </g>
        );
      })}
      {halo.map((s, i) => (
        <rect key={'h' + i} x={X + 1} y={s.y0} width={W - 2} height={s.h} fill="var(--ink-2)" pointerEvents="none" />
      ))}
      {edge.map((s, i) => (
        <rect key={'e' + i} x={X + 1} y={s.y0} width={W - 2} height={s.h} fill="var(--global)" pointerEvents="none" />
      ))}
      <rect x={X} y={Y} width={W} height={H} fill="none" stroke="var(--ink-3)" strokeWidth={1} pointerEvents="none" />
    </svg>
  );
}

// ---------------------------------------------------------------- one GPU's local array

function Strip({ kind, stale }: { kind: 'halo' | 'boundary' | 'internal'; stale: boolean }) {
  const body: CSSProperties =
    kind === 'halo'
      ? stale
        ? { background: HATCH, border: '1px dashed var(--ink-3)' }
        : { background: 'var(--rule-soft)', border: '1px dashed var(--ink-3)' }
      : kind === 'boundary'
        ? { background: 'var(--global)', opacity: 0.75 }
        : { background: 'var(--global-wash)', border: '1px solid var(--rule)' };
  const ends: CSSProperties = { width: '8%', background: HATCH, border: '1px solid var(--rule)' };
  return (
    <div style={{ display: 'flex', height: 16, minWidth: 56 }} aria-hidden>
      {kind === 'halo' ? (
        <div style={{ flex: 1, ...body }} />
      ) : (
        <>
          <div style={ends} />
          <div style={{ flex: 1, ...body }} />
          <div style={ends} />
        </>
      )}
    </div>
  );
}

type Phase = 'before' | 'after';

/** What row r of the chosen GPU's array is, in words. */
function rowRole(p: Partition, r: number, phase: Phase): ReactNode {
  const g = (r - 1 + p.first + p.rows) % p.rows; // the global row this local row holds
  const kind = rowKind(p.ny, r);
  if (kind === 'halo') {
    const owner = r === 0 ? p.top : p.bottom;
    const side = r === 0 ? 'Top' : 'Bottom';
    const wraps = r === 0 ? p.first === 0 : p.last === p.rows - 1;
    const note = p.self ? ' (itself, by the wrap-around)' : wraps ? ' (by the wrap-around)' : '';
    return phase === 'before' ? (
      <>
        {side} halo: a copy of global row {g}, from GPU {owner}
        {note}
      </>
    ) : (
      <>
        {side} halo, <b>stale</b>: GPU {owner} has just computed row {g} again
      </>
    );
  }
  if (kind === 'boundary') {
    const both = r === 1 && r === p.ny - 2;
    const to = both ? `GPUs ${p.top} and ${p.bottom}` : r === 1 ? `GPU ${p.top}` : `GPU ${p.bottom}`;
    const side = both ? 'Top and bottom' : r === 1 ? 'Top' : 'Bottom';
    return phase === 'before' ? (
      <>
        {side} boundary: global row {g}; after the step it goes to {to}
      </>
    ) : (
      <>
        {side} boundary, <b>new</b>: must reach {to} before the next step
      </>
    );
  }
  return <>Internal: global row {g}</>;
}

function LaunchCell({ l, r, gap, twice }: { l: Launch; r: number; gap?: { from: number; to: number }; twice: number[] }) {
  if (gap) {
    const w = l.writes.filter((x) => x >= gap.from && x <= gap.to);
    if (w.length === 0) return <td data-launch={l.name} />;
    const t0 = w[0] - l.base - 1;
    const t1 = w[w.length - 1] - l.base - 1;
    return (
      <td data-launch={l.name} data-cell="write">
        <span className="chip global">
          t{t0}–t{t1}
        </span>
      </td>
    );
  }
  if (l.writes.includes(r))
    return (
      <td data-launch={l.name} data-cell="write">
        <span className={'chip ' + (twice.includes(r) ? 'fail' : 'global')}>t{r - l.base - 1}</span>
      </td>
    );
  if (l.reads.includes(r))
    return (
      <td data-launch={l.name} data-cell="read" className="muted">
        read
      </td>
    );
  return <td data-launch={l.name} />;
}

function LocalArray({ p, phase, showLaunches }: { p: Partition; phase: Phase; showLaunches: boolean }) {
  const ls = launches(p.ny, p.nx);
  const cov = coverage(p.ny);
  return (
    <table className="data" data-local style={{ margin: 0, minWidth: showLaunches ? '27rem' : '17rem' }}>
      <thead>
        <tr>
          <th className="num">row</th>
          <th style={{ width: '22%', minWidth: 64 }}>GPU {p.gpu}&rsquo;s array</th>
          <th>pointer, and what the row holds</th>
          {showLaunches ? ls.map((l) => <th key={l.name}>{l.name}</th>) : null}
        </tr>
      </thead>
      <tbody>
        {visibleRows(p.ny).map((it) =>
          it.kind === 'gap' ? (
            <tr key="gap" data-gap>
              <td className="num muted">⋮</td>
              <td>
                <div style={{ height: 22, background: 'var(--global-wash)', border: '1px dashed var(--rule)' }} aria-hidden />
              </td>
              <td className="muted" data-role>
                Rows {range(it.from, it.to)}: {n(it.to - it.from + 1)} more internal {it.to === it.from ? 'row' : 'rows'}
              </td>
              {showLaunches ? ls.map((l) => <LaunchCell key={l.name} l={l} r={-1} gap={it} twice={cov.twice} />) : null}
            </tr>
          ) : (
            <tr key={it.row} data-row={it.row} data-kind={rowKind(p.ny, it.row)}>
              <td className="num">{it.row}</td>
              <td>
                <Strip kind={rowKind(p.ny, it.row)} stale={phase === 'after'} />
              </td>
              <td>
                <code style={{ whiteSpace: 'nowrap' }}>{pointerExpr(p.ny, it.row)}</code>
                <div data-role style={{ fontSize: '0.78rem', marginTop: '0.15rem' }}>
                  {rowRole(p, it.row, phase)}
                </div>
              </td>
              {showLaunches ? ls.map((l) => <LaunchCell key={l.name} l={l} r={it.row} twice={cov.twice} />) : null}
            </tr>
          ),
        )}
      </tbody>
    </table>
  );
}

function LaunchTable({ p }: { p: Partition }) {
  const ls = launches(p.ny, p.nx);
  const cov = coverage(p.ny);
  return (
    <>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" data-launches style={{ marginTop: '0.8rem', minWidth: '34rem' }}>
          <thead>
            <tr>
              <th>launch</th>
              <th>stream</th>
              <th>pointer it gets</th>
              <th className="num">element</th>
              <th>rows it is told</th>
              <th>rows it writes</th>
            </tr>
          </thead>
          <tbody>
            {ls.map((l) => (
              <tr key={l.name} data-launch={l.name}>
                <td>{l.name}</td>
                <td>
                  <code style={{ whiteSpace: 'nowrap' }}>{l.stream}</code>
                </td>
                <td>
                  <code style={{ whiteSpace: 'nowrap' }}>{l.pointer}</code>
                </td>
                <td className="num" data-offset={l.offset}>
                  {n(l.offset)}
                </td>
                <td data-rows={l.rows}>
                  {l.rowsExpr === String(l.rows) ? (
                    l.rows
                  ) : (
                    <>
                      <code>{l.rowsExpr}</code> = {l.rows}
                    </>
                  )}
                </td>
                <td data-writes={l.writes.join(',')}>{l.writes.length ? range(l.writes[0], l.writes[l.writes.length - 1]) : 'none'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className={cov.exact ? '' : 'narrate bad'} data-coverage={cov.exact ? 'exact' : 'overlap'} style={{ margin: '0 0 0.4rem' }}>
        {cov.exact ? (
          <>
            <span className="chip ok">covered once</span> Together the three launches write rows 1 to {p.ny - 2} once
            each and never touch the halo rows 0 and {p.ny - 1}. Thread row t of a launch handed{' '}
            <code>output + base*nx</code> writes row base + t + 1: the kernel&rsquo;s <code>+ 1</code> and its guard{' '}
            <code>y &lt; ny - 1</code> do the rest.
          </>
        ) : (
          <>
            <span className="chip fail">row {cov.twice.join(', ')} twice</span> With one row per GPU, ny = 3: the top
            and the bottom launch both compute row 1, so its residue is added to the norm twice. Section 23.3&rsquo;s
            split needs at least two rows per GPU (our reading of the kernel&rsquo;s guard).
          </>
        )}
      </p>
    </>
  );
}

// ---------------------------------------------------------------- the 2D alternative (ours)

function TilesPicture({ nx, rows, t, gpu, onPick }: { nx: number; rows: number; t: TileSplit; gpu: number; onPick: (g: number) => void }) {
  const ratio = nx / rows;
  const S = 170;
  const bw = ratio >= 1 ? S : Math.max(56, S * ratio);
  const bh = ratio >= 1 ? Math.max(56, S / ratio) : S;
  const X = 10;
  const Y = 10;
  const hs = rowShares(rows, t.p);
  const ws = rowShares(nx, t.q);
  const hy = firstRows(hs).map((r) => Y + (r / rows) * bh);
  const wx = firstRows(ws).map((c) => X + (c / nx) * bw);
  const at = tileOf(gpu, t.q);
  const cells = [];
  for (let i = 0; i < t.p; i++)
    for (let j = 0; j < t.q; j++) {
      const g = i * t.q + j;
      const x0 = wx[j];
      const y0 = hy[i];
      const w = (ws[j] / nx) * bw;
      const h = (hs[i] / rows) * bh;
      cells.push(
        <rect
          key={g}
          data-tile={g}
          x={x0}
          y={y0}
          width={w}
          height={h}
          fill={g === gpu ? 'var(--global-wash)' : 'var(--cell)'}
          stroke="var(--rule)"
          strokeWidth={0.6}
          style={{ cursor: 'pointer' }}
          onClick={() => onPick(g)}
        />,
      );
    }
  const sx = wx[at.col];
  const sy = hy[at.row];
  const sw = (ws[at.col] / nx) * bw;
  const sh = (hs[at.row] / rows) * bh;
  return (
    <svg viewBox={`0 0 ${bw + 2 * X} ${bh + 2 * Y}`} width="100%" style={{ maxWidth: 200, display: 'block', margin: '0 auto' }} data-tilespic>
      <title>
        The grid cut into {t.p} by {t.q} tiles
      </title>
      {cells}
      <rect x={sx} y={sy} width={sw} height={sh} fill="none" stroke="var(--global)" strokeWidth={1.5} pointerEvents="none" />
      {t.p > 1 ? (
        <>
          <line x1={sx} x2={sx + sw} y1={sy} y2={sy} stroke="var(--ink)" strokeWidth={3} />
          <line x1={sx} x2={sx + sw} y1={sy + sh} y2={sy + sh} stroke="var(--ink)" strokeWidth={3} />
        </>
      ) : null}
      {t.q > 1 ? (
        <>
          <line x1={sx} x2={sx} y1={sy} y2={sy + sh} stroke="var(--ink)" strokeWidth={3} strokeDasharray="2 3" />
          <line x1={sx + sw} x2={sx + sw} y1={sy} y2={sy + sh} stroke="var(--ink)" strokeWidth={3} strokeDasharray="2 3" />
        </>
      ) : null}
      <rect x={X} y={Y} width={bw} height={bh} fill="none" stroke="var(--ink-3)" pointerEvents="none" />
    </svg>
  );
}

function TilesPanel({ p, pick, setPick, onGpu }: { p: Partition; pick: string | null; setPick: (k: string) => void; onGpu: (g: number) => void }) {
  if (p.gpus === 1)
    return (
      <div className="panel" data-tiles="none" style={{ marginTop: '1rem' }}>
        <p style={{ margin: 0 }}>One GPU: there is nothing to cut. Add GPUs to compare rows with tiles.</p>
      </div>
    );
  const options = factorisations(p.gpus).filter(({ p: tp, q }) => p.rows >= tp && p.nx >= q);
  const best = bestTiles(p.nx, p.rows, p.gpus);
  const chosen = options.find((o) => `${o.p}x${o.q}` === pick) ?? best;
  const t = tileSplit(p.nx, p.rows, chosen.p, chosen.q);
  const rowsT = tileSplit(p.nx, p.rows, p.gpus, 1);
  const maxShare = Math.max(...rowShares(p.rows, p.gpus));
  const memory = (s: TileSplit) => {
    const parts: string[] = [];
    if (s.rowMessages) parts.push(`${s.rowMessages} row pieces of ${n(s.w)} values, each one contiguous run`);
    if (s.columnMessages) parts.push(`${s.columnMessages} columns of ${n(s.h)} single values, ${n(p.nx)} apart in memory`);
    return parts.join('; ');
  };
  const saving = 1 - t.bytes / rowsT.bytes;
  return (
    <div className="panel" data-tiles={`${t.p}x${t.q}`} style={{ marginTop: '1rem' }}>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span>
          Arrange the {p.gpus} GPUs as <i>P</i> tile rows × <i>Q</i> tile columns:
        </span>
        <span className="seg" style={{ flexWrap: 'wrap', rowGap: 4 }}>
          {options.map((o) => {
            const k = `${o.p}x${o.q}`;
            return (
              <button key={k} data-pq={k} aria-pressed={o.p === t.p && o.q === t.q} onClick={() => setPick(k)}>
                {o.p} × {o.q}
              </button>
            );
          })}
        </span>
        <span className="muted" data-best={`${best.p}x${best.q}`}>
          Sends least here: {best.p} × {best.q}.
        </span>
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 12rem) minmax(0, 1fr)' }}>
        <div>
          <TilesPicture nx={p.nx} rows={p.rows} t={t} gpu={p.gpu} onPick={onGpu} />
          <p className="muted" style={{ fontSize: '0.74rem', margin: '0.4rem 0 0' }}>
            GPU {p.gpu}&rsquo;s tile. Solid edges: row pieces it sends up and down. Dotted edges: columns it sends left and
            right.
          </p>
        </div>
        <table className="data" data-tilecompare style={{ margin: 0 }}>
          <thead>
            <tr>
              <th>per GPU, each step</th>
              <th>rows ({p.gpus} × 1)</th>
              <th>tiles ({t.p} × {t.q})</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>largest piece</td>
              <td>
                {n(maxShare)} × {n(p.nx)}
              </td>
              <td data-t="piece">
                {n(t.h)} × {n(t.w)}
              </td>
            </tr>
            <tr>
              <td>neighbours, one per cut side</td>
              <td data-t="rows-neighbours">{rowsT.neighbours}</td>
              <td data-t="tiles-neighbours">{t.neighbours}</td>
            </tr>
            <tr>
              <td>values sent</td>
              <td data-t="rows-points">{n(rowsT.points)}</td>
              <td data-t="tiles-points">{n(t.points)}</td>
            </tr>
            <tr>
              <td>bytes sent</td>
              <td data-t="rows-bytes">{n(rowsT.bytes)}</td>
              <td data-t="tiles-bytes">{n(t.bytes)}</td>
            </tr>
            <tr>
              <td>how it sits in memory</td>
              <td>{memory(rowsT)}</td>
              <td>{memory(t)}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p data-tileverdict style={{ margin: '0.7rem 0 0' }}>
        {t.q === 1 ? (
          <>This is the row split itself.</>
        ) : saving > 0 ? (
          <>
            {t.p} × {t.q} sends {pct(saving)} less than rows, and pays for it with {t.neighbours} neighbours and{' '}
            {t.columnMessages} strided {t.columnMessages === 1 ? 'column' : 'columns'}: a column has to be gathered from{' '}
            {n(t.h)} places before it can travel, or sent as many small pieces.
          </>
        ) : (
          <>
            Rows send less on this grid: {n(rowsT.points)} values against {n(t.points)}, all of them in contiguous rows.
          </>
        )}
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- the explorer

const NX = { min: 8, max: 1024, step: 8 };
const ROWS = { min: 32, max: 1024, step: 8 };
const GPUS = { min: 1, max: 32 };

function HaloBench() {
  const [nx, setNx] = useState(64);
  const [rows, setRows] = useState(128);
  const [gpus, setGpus] = useState(4);
  const [gpu, setGpu] = useState(1);
  const [phase, setPhase] = useState<Phase>('before');
  const [showLaunches, setShowLaunches] = useState(false);
  const [showTiles, setShowTiles] = useState(false);
  const [tilePick, setTilePick] = useState<string | null>(null);

  const g = Math.min(gpu, gpus - 1);
  const p = partition(nx, rows, gpus, g);
  const isExercise = nx === EXERCISE1.nx && rows === EXERCISE1.rows && gpus === EXERCISE1.gpus;
  const slider = (label: ReactNode, value: number, set: (v: number) => void, r: { min: number; max: number; step?: number }, name: string) => (
    <label className="field" style={{ minWidth: '11rem' }}>
      <span>
        {label} <b data-value={name}>{n(value)}</b>
      </span>
      <input type="range" min={r.min} max={r.max} step={r.step ?? 1} value={value} aria-label={name} data-slider={name} onChange={(e) => set(+e.target.value)} />
    </label>
  );

  return (
    <Bench
      title="Rows, halos and launches"
      task="Load the exercise, then drag the GPU count: watch what each GPU sends against what it computes."
      foot={
        <>
          Counts follow the kernel of Fig. 23.1 and the exchange of Fig. 23.7, with 4-byte floats. Neighbours wrap
          around as in the book&rsquo;s code. The tile comparison is ours.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        {slider('grid width nx', nx, setNx, NX, 'nx')}
        {slider('rows in the grid', rows, setRows, ROWS, 'rows')}
        {slider('GPUs', gpus, setGpus, GPUS, 'gpus')}
        <button
          data-preset="exercise1"
          aria-pressed={isExercise}
          onClick={() => {
            setNx(EXERCISE1.nx);
            setRows(EXERCISE1.rows);
            setGpus(EXERCISE1.gpus);
          }}
        >
          Exercise 1 (p. 576)
        </button>
      </div>
      <div className="row" style={{ marginBottom: '1rem' }}>
        <span className="seg" role="group" aria-label="When">
          <button data-phase="before" aria-pressed={phase === 'before'} onClick={() => setPhase('before')}>
            before the step
          </button>
          <button data-phase="after" aria-pressed={phase === 'after'} onClick={() => setPhase('after')}>
            after the step
          </button>
        </span>
        <label className="check">
          <input type="checkbox" data-toggle="launches" checked={showLaunches} onChange={(e) => setShowLaunches(e.target.checked)} />
          the three launches of section 23.3
        </label>
        <label className="check">
          <input type="checkbox" data-toggle="tiles" checked={showTiles} onChange={(e) => setShowTiles(e.target.checked)} />
          tiles (2D split)
        </label>
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 13rem) minmax(0, 1fr)' }}>
        <div>
          <DomainPicture p={p} onPick={setGpu} />
          <div className="legend" style={{ marginTop: '0.5rem', fontSize: '0.74rem' }}>
            <span>
              <i style={{ background: 'var(--global)' }} />
              GPU {p.gpu}&rsquo;s edge rows: it sends these
            </span>
            <span>
              <i style={{ background: 'var(--ink-2)' }} />
              neighbours&rsquo; rows it copies in as halos
            </span>
          </div>
          <p className="muted" style={{ fontSize: '0.74rem', margin: '0.4rem 0 0' }}>
            Click a band to pick a GPU.
          </p>
        </div>
        <div style={{ minWidth: 0 }}>
          <div style={{ overflowX: 'auto' }}>
            <LocalArray p={p} phase={phase} showLaunches={showLaunches} />
          </div>
          <p className="muted" style={{ fontSize: '0.74rem', margin: '0.4rem 0 0' }}>
            Hatched ends: columns 0 and nx − 1, which the kernel never computes.
            {showLaunches ? ' t k: written by thread row k of that launch; read: in its slice, not written.' : ''}
          </p>
        </div>
      </div>

      {showLaunches ? <LaunchTable p={p} /> : null}

      <div className="cols plot" style={{ marginTop: '1rem' }}>
        <table className="data" data-readouts style={{ margin: 0 }}>
          <thead>
            <tr>
              <th>GPU {p.gpu}, every step</th>
              <th className="num">all nx columns</th>
              <th className="num">the kernel&rsquo;s nx − 2</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>rows it owns (global {range(p.first, p.last)})</td>
              <td className="num" data-readout="owned">
                {n(p.owned)}
              </td>
              <td />
            </tr>
            <tr>
              <td>
                <code>ny</code>: those rows plus two halo rows
              </td>
              <td className="num" data-readout="ny">
                {n(p.ny)}
              </td>
              <td />
            </tr>
            <tr>
              <td>points it computes</td>
              <td className="num" data-readout="output-all">
                {n(p.output.all)}
              </td>
              <td className="num" data-readout="output-kernel">
                {n(p.output.kernel)}
              </td>
            </tr>
            <tr>
              <td>halo points it receives (whole rows)</td>
              <td className="num" data-readout="halo">
                {n(p.halo)}
              </td>
              <td />
            </tr>
            <tr>
              <td>boundary points (rows 1 and ny − 2)</td>
              <td className="num" data-readout="boundary-all">
                {n(p.boundary.all)}
              </td>
              <td className="num" data-readout="boundary-kernel">
                {n(p.boundary.kernel)}
              </td>
            </tr>
            <tr>
              <td>internal points (rows 2 to ny − 3)</td>
              <td className="num" data-readout="internal-all">
                {n(p.internal.all)}
              </td>
              <td className="num" data-readout="internal-kernel">
                {n(p.internal.kernel)}
              </td>
            </tr>
            <tr>
              <td>bytes it sends</td>
              <td className="num" data-readout="bytes">
                {n(p.bytesSent)}
              </td>
              <td className="muted">{p.self ? 'to itself' : `${n(p.bytesPerNeighbour)} to each neighbour`}</td>
            </tr>
          </tbody>
        </table>
        <div className="narrate" data-share={p.exchangeShare}>
          {p.self ? (
            <p style={{ margin: 0 }}>
              One GPU. The book&rsquo;s wrap-around makes GPU 0 its own top and bottom neighbour, so its two exchanges
              copy its own edge rows into its own halo rows and nothing crosses a link.
            </p>
          ) : (
            <p style={{ margin: 0 }}>
              GPU {p.gpu} computes {n(p.owned)} {p.owned === 1 ? 'row' : 'rows'} and sends 2 every step: the exchange is{' '}
              <b>{pct(p.exchangeShare)}</b> of the rows it computes. Double the GPUs and its rows halve, but it still
              sends one row up and one down.
            </p>
          )}
        </div>
      </div>

      {showTiles ? <TilesPanel p={p} pick={tilePick} setPick={setTilePick} onGpu={setGpu} /> : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- the page

export function S230() {
  const q = questionsS230;
  const at16 = partition(EXERCISE1.nx, EXERCISE1.rows, 16, 0);
  const at32 = partition(EXERCISE1.nx, EXERCISE1.rows, 32, 0);
  return (
    <>
      <p>
        Chapter 23 takes one stencil from a single GPU to a cluster of GPUs, and uses it to introduce three ways of
        moving data between them: MPI, NCCL and NVSHMEM. Those cluster libraries are not what this course keeps. What it
        keeps is the machinery the chapter needs on the way, which you will use on one GPU:
      </p>
      <Source pages="541" />
      <ul>
        <li>
          a launch or an asynchronous copy is <b>queued, not done</b>: when the call returns, the work may not even have
          started;
        </li>
        <li>
          <b>streams</b>, ordered queues of GPU work that may run alongside each other;
        </li>
        <li>
          <b>events</b>, which let one queue wait for a point in another without stopping the host;
        </li>
        <li>
          <b>asynchronous copies</b> and the <b>pinned</b> host memory they depend on;
        </li>
        <li>
          and the pattern that ties them together: compute first what someone else is waiting for, then overlap its
          transfer with the rest of the work.
        </li>
      </ul>
      <Aside>
        <p>
          The chapter names no GPU and no library version. Every number on this page is rows, points or bytes of a grid
          you choose, so none of it depends on a device.
        </p>
      </Aside>
      <p>
        That is the book&rsquo;s section 23.3, split over two pages here. Page <a href={href('ch23', '23.3')}>23.3</a>{' '}
        covers its first half (overlap, streams, asynchronous copies, pinned memory, pp. 553–556 and 560–563); page{' '}
        <a href={href('ch23', '23.4')}>23.3–23.5</a> covers its second half (the three launches, priorities, events, pp.
        556–562). MPI, NCCL and NVSHMEM, the book&rsquo;s sections 23.2, 23.4 and 23.5, get one paragraph each on that
        page and a single table that compares where their loops wait, with their pages in the book for when you need them.
        The explorers model only the two MPI loops, and no question on NCCL or NVSHMEM enters your review. The chapter has
        no project milestone. Its one lab,{' '}
        <a href={href('ch23', 'labs')}>Hide the copies behind the kernel</a>, measures the overlap on your own GPU.
      </p>
      <Source section="23.3" pages="553–563" />

      <h2>The running example</h2>
      <p>
        The book&rsquo;s example is the Jacobi method on a 2D grid, a stencil like those of chapter 8. Each step computes
        every interior point&rsquo;s new value as the average of its four neighbours&rsquo; old values, reading one array
        and writing another. Steps repeat until the values stop changing much. To decide when, the kernel also measures
        how far each point moved, its residue (new minus old). The squared residues are summed inside each block, and one
        thread per block adds the block&rsquo;s sum to a single total with <code>atomicAdd</code>: chapter 10&rsquo;s
        multi-block commit. The total is the square of the L2 norm; the host takes the root.
      </p>
      <Source section="23.1" pages="542–543" figs="Fig. 23.1" />
      <p>
        The grid is stored row-major with x fastest: all of row 0, then all of row 1, and so on. Cut it along y and each
        GPU gets a run of whole rows, contiguous in memory. To update its first and last rows a GPU needs the row just
        above its share and the row just below, and those belong to its neighbours. So each GPU&rsquo;s arrays carry two
        extra rows, the <b>halo rows</b>, each a copy of a neighbour&rsquo;s edge row. The book&rsquo;s{' '}
        <code>ny</code> is the height of those arrays with the halo rows included.
      </p>
      <Source section="23.1" pages="543–544" figs="Figs. 23.2, 23.3" />
      <p>
        A GPU reads its halo rows and never writes them. After one step its neighbours have new values for those rows,
        and the copies are stale. Before the next step every GPU sends its own top and bottom rows (its{' '}
        <b>boundary rows</b>) to its neighbours and receives fresh halo rows in return: the <b>halo exchange</b>. So
        every step has two phases, compute and then communicate, and the communicate phase also sums the GPUs&rsquo;
        norms into one.
      </p>
      <Source section="23.1" pages="544–545" />
      <Ask q={q['ch23/start/q-halo-rows']} label="From chapter 8" />

      <h2>One GPU&rsquo;s share of the grid</h2>
      <p>
        The explorer below is the book&rsquo;s Figs. 23.2 and 23.3 with numbers attached. Pick a grid, a GPU count and
        one GPU, and compare two quantities: the rows it computes and the rows it sends. One toggle previews section
        23.3, where the same kernel is launched three times on different slices of the array; another compares the
        book&rsquo;s split by rows with a split into tiles.
      </p>
      <Simplification>
        <p>
          One row-major grid of 4-byte floats, cut into whole rows; when they do not divide evenly the last GPU takes the
          remainder (the book&rsquo;s examples divide evenly). Neighbours wrap around as in the book&rsquo;s code (p.
          551), although its figures draw GPU 0 with no neighbour above. Points and bytes only, no time. The tile
          comparison is ours and gives every tile a neighbour on each side that is cut.
        </p>
      </Simplification>
      <Ask q={q['ch23/start/q-ex1-bytes']} label="Book exercise 1e" />
      <Predict
        id="ch23/start/p-32gpus"
        prompt={
          <>
            The exercise&rsquo;s grid, 64 wide and 512 tall, now runs on 32 GPUs instead of 16. How many bytes does each
            GPU send per step?
          </>
        }
        unit="bytes"
      >
        {(v) => (
          <p>
            <b>{n(at32.bytesSent)} bytes</b>, the same as on 16 GPUs. Each GPU still sends one row of {EXERCISE1.nx}{' '}
            floats up and one down. What halved is its work, from {at16.owned} rows to {at32.owned}, so the exchange went
            from {pct(at16.exchangeShare)} to {pct(at32.exchangeShare)} of the rows it computes.{' '}
            {v === at32.bytesSent
              ? 'That is what you had.'
              : v < at32.bytesSent
                ? 'Halving is the natural guess, and the one to unlearn: the bytes follow the width of a row, not the number of rows.'
                : 'More GPUs do not add messages per GPU in a row split: each still has exactly two neighbours.'}
          </p>
        )}
      </Predict>
      <HaloBench />
      <p>
        Three things to find. First, load the exercise and drag the GPU count: the bytes each GPU sends stay put while
        the rows it computes shrink, so communication takes a growing share of every step. Second, turn on the three
        launches: each gets a pointer into the same array and a row count, the kernel&rsquo;s guard turns that into
        the rows it writes, and together they write rows 1 to <code>ny − 2</code> exactly once. Then drag the rows
        down until each GPU owns a single row, and watch that break. Third, make the grid square, or wide, and turn on
        tiles: a 2D split sends less, but each GPU talks to more neighbours and some of what it sends are columns, one
        value per row, scattered through memory.
      </p>
      <Beyond>
        <p>
          The tile numbers are ours. The book states the trade-off in one paragraph: less data, four neighbours instead
          of two, data that is not contiguous (p. 545). The double count at one row per GPU is also ours, read off the
          kernel&rsquo;s guard.
        </p>
      </Beyond>

      <h2>The book&rsquo;s first loop, in words</h2>
      <p>
        Section 23.2 builds the multi-GPU version with MPI, one process per GPU. Leave the MPI calls for page{' '}
        <a href={href('ch23', '23.4')}>23.3–23.5</a>; the shape of the loop is what matters here. Every iteration, each GPU:
      </p>
      <ol>
        <li>zeroes the norm on the device;</li>
        <li>
          runs the Jacobi kernel over its <code>ny</code> rows;
        </li>
        <li>waits for the kernel to finish;</li>
        <li>
          swaps edge rows with its two neighbours: it sends row 1 up while receiving row <code>ny − 1</code> from below,
          then sends row <code>ny − 2</code> down while receiving row 0 from above;
        </li>
        <li>copies its norm to the host;</li>
        <li>sums the norms of all GPUs and takes the square root, to decide whether to go on;</li>
        <li>swaps its input and output pointers, so this step&rsquo;s output is the next step&rsquo;s input.</li>
      </ol>
      <Source section="23.2" pages="548–549" figs="Fig. 23.7" />
      <p>
        Step 3 keeps steps 2 and 4 apart: the exchange cannot start until the whole kernel is done. So while the GPUs
        compute, the links between them carry nothing, and while the rows travel, the GPUs have nothing to do. Section{' '}
        <a href={href('ch23', '23.3')}>23.3</a> starts from the explorer&rsquo;s three launches: compute the two boundary
        rows first, start sending them, and compute the internal rows while they travel. That needs launches that do not
        wait for each other and copies that do not stop the host, which is what streams, events and pinned memory
        provide.
      </p>
      <Source section="23.3" pages="553–555" figs="Figs. 23.12, 23.13" />

      <h2>Before you start</h2>
      <p>
        Four questions on what this chapter leans on from chapters 2, 4, 10 and 22 and appendix C. Each explanation
        points to where the idea was taught and to the page here that uses it. They come back in{' '}
        <a href={href('review')}>review</a>.
      </p>
      <Ask q={q['ch23/start/q-queued']} label="From chapter 2" />
      <Ask q={q['ch23/start/q-blockreduce']} label="From chapters 4 and 10" />
      <Ask q={q['ch23/start/q-amdahl']} label="From chapter 22" />
      <Ask q={q['ch23/start/q-pinned']} label="From appendix C" />
    </>
  );
}
