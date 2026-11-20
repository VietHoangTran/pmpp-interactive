import { useState, type CSSProperties } from 'react';
import { Bench } from '../../components/notes';
import { href } from '../../lib/router';
import {
  BACKENDS,
  BACKEND_NAME,
  OPS,
  PROJECTS,
  PROJECT_NAME,
  cellLink,
  op as opRow,
  whereLink,
  type Backend,
  type Cell,
  type OpId,
  type Project,
} from '../../sim/backendmap';
import { linesText, shortCommit } from '../../sim/x3sources';

// The op map: for one op and one project, where the op enters each backend, the host function,
// the kernel's file and function with a permalink at the pinned commit, the facts items, and a
// status tag. All of it is data in sim/backendmap.ts. Later sections embed it preselected:
// <OpMap op="softmax" project="mlx" />.

const CARD: CSSProperties = {
  border: '1px solid var(--rule)',
  borderRadius: 4,
  background: 'var(--page)',
  padding: '0.7rem 0.8rem',
  fontFamily: 'var(--sans)',
  fontSize: '0.84rem',
  lineHeight: 1.42,
  minWidth: 0,
};
const DT: CSSProperties = { color: 'var(--ink-2)', fontSize: '0.74rem', margin: '0.45rem 0 0.05rem', textTransform: 'lowercase' };
const DD: CSSProperties = { margin: 0, overflowWrap: 'anywhere' };
const TAG: CSSProperties = {
  display: 'inline-block',
  border: '1px solid var(--rule)',
  borderRadius: 3,
  padding: '0 0.4em',
  margin: '0 0.3rem 0.25rem 0',
  fontSize: '0.74rem',
  whiteSpace: 'nowrap',
};

function Tags({ c }: { c: Cell }) {
  return (
    <div data-part="tags" data-tags={c.tags.join('|')}>
      {c.tags.map((t) => (
        <span key={t} style={{ ...TAG, fontStyle: t === 'not read in the facts pass' || t === 'no backend' ? 'italic' : 'normal', color: t === 'no backend' ? 'var(--ink-3)' : 'var(--ink)' }}>
          {t}
        </span>
      ))}
    </div>
  );
}

function CellCard({ project, backend, c }: { project: Project; backend: Backend; c: Cell }) {
  const link = cellLink(project, c);
  const none = c.tags.includes('no backend');
  return (
    <section style={{ ...CARD, opacity: none ? 0.75 : 1 }} data-backend={backend} data-file={c.file ?? ''} aria-label={`${BACKEND_NAME[backend]} cell`}>
      <div className="row" style={{ justifyContent: 'space-between', gap: '0.3rem 0.8rem', alignItems: 'baseline', marginBottom: '0.3rem' }}>
        <b style={{ fontSize: '0.95rem' }}>{BACKEND_NAME[backend]}</b>
        <span className="muted" style={{ fontSize: '0.74rem' }}>{c.facts.map((f) => (f.includes(' ') ? f : 'x3 ' + f)).join(', ')}</span>
      </div>
      <Tags c={c} />
      {none ? (
        <p style={{ margin: '0.3rem 0 0' }} data-part="note">{c.note}</p>
      ) : (
        <dl style={{ margin: 0 }}>
          <dt style={DT}>Entry point</dt>
          <dd style={DD} data-part="entry">{c.entry}</dd>
          {c.host ? (
            <>
              <dt style={DT}>Host function</dt>
              <dd style={DD} data-part="host"><code>{c.host}</code></dd>
            </>
          ) : null}
          <dt style={DT}>Kernel</dt>
          <dd style={DD} data-part="kernel">
            {c.kernel ? <>{c.kernel}</> : <span className="muted">none</span>}
            {c.file ? (
              <>
                <br />
                {link ? (
                  <a href={link} target="_blank" rel="noreferrer" data-part="permalink">
                    <code>{c.file}</code>
                  </a>
                ) : (
                  <code>{c.file}</code>
                )}
                {c.lines != null ? <span className="muted">, {linesText(c.lines)}</span> : null}
                <span className="muted">
                  {c.readAt === 'master' ? ' (read at master in another chapter’s facts pass; no pinned lines)' : ` at ${shortCommit(project)}`}
                </span>
              </>
            ) : (
              <>
                <br />
                <span className="muted" data-part="nofile">no file to open</span>
              </>
            )}
          </dd>
          {c.also?.length ? (
            <>
              <dt style={DT}>Also at the pinned commit</dt>
              <dd style={DD}>
                {c.also.map((w, i) => (
                  <span key={w.label}>
                    {i ? '; ' : ''}
                    <a href={whereLink(project, w)} target="_blank" rel="noreferrer">{w.label}</a>
                  </span>
                ))}
              </dd>
            </>
          ) : null}
          {c.note ? <p style={{ margin: '0.5rem 0 0', color: 'var(--ink-2)' }} data-part="note">{c.note}</p> : null}
        </dl>
      )}
    </section>
  );
}

/** One plain sentence on what the three backends of this project have in common for this op. */
function hipLine(p: Project, cells: Record<Backend, Cell>): string {
  const hip = cells.hip;
  if (hip.tags.includes('no backend')) return 'HIP: MLX has no AMD backend, so the column is empty.';
  if (hip.tags.includes('same source as CUDA')) return 'HIP: the CUDA file itself, compiled as HIP.';
  if (hip.tags.includes('not read in the facts pass')) return 'HIP: not read in the facts pass.';
  if (hip.tags.includes('vendor library')) return `HIP: ${PROJECT_NAME[p]} calls a different library on AMD.`;
  if (hip.tags.includes('hipified at build time')) return 'HIP: the CUDA source, translated by hipify at build time.';
  return 'HIP: no operator of its own; ordinary tensor ops.';
}

export function OpMap({ op = 'rmsnorm', project = 'llama.cpp', title = 'Op map' }: { op?: OpId; project?: Project; title?: string }) {
  const [o, setO] = useState<OpId>(op);
  const [p, setP] = useState<Project>(project);
  const row = opRow(o);
  const cells = row.cells[p];
  return (
    <Bench
      title={title}
      task="Pick an op and a project. Find where it lands on CUDA, HIP and Metal."
      foot={
        <>
          Every cell was read in the facts pass of 2026-11-20 (notes/today/x3-facts.md and the items named). Links go to the
          pinned commit, except where a cell rests on another chapter’s item read at master. Nothing here was built or run.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <span className="field" style={{ display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)', fontFamily: 'var(--sans)', fontSize: '0.84rem' }}>
          Op
          <span className="seg" role="group" aria-label="Op" style={{ flexWrap: 'wrap' }}>
            {OPS.map((r) => (
              <button key={r.id} aria-pressed={o === r.id} onClick={() => setO(r.id)} data-op={r.id}>{r.label}</button>
            ))}
          </span>
        </span>
        <span className="field" style={{ display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)', fontFamily: 'var(--sans)', fontSize: '0.84rem' }}>
          Project
          <span className="seg" role="group" aria-label="Project">
            {PROJECTS.map((x) => (
              <button key={x} aria-pressed={p === x} onClick={() => setP(x)} data-project={x}>{PROJECT_NAME[x]}</button>
            ))}
          </span>
        </span>
      </div>
      <p className="ui" style={{ margin: '0 0 0.6rem', fontSize: '0.86rem' }} data-readout="course">
        <b>{PROJECT_NAME[p]}, {row.label}.</b>{' '}
        <span className="muted">Where you met the technique:</span>{' '}
        {row.course.map((l, i) => (
          <span key={l.num}>
            {i ? '; ' : ''}
            <a href={href(l.chapter, 'labs')}>lab {l.num}</a> ({l.what}{l.planned ? ', in this module' : ''})
          </span>
        ))}
        . <span className="muted">Read in section</span> <a href={href('x3', row.section)}>{row.section}</a>.
      </p>
      <div data-part="cells" data-op={o} data-project={p}
        style={{ display: 'grid', gap: '0.6rem', gridTemplateColumns: 'repeat(auto-fit, minmax(15.5rem, 1fr))' }}>
        {BACKENDS.map((b) => (
          <CellCard key={b} project={p} backend={b} c={cells[b]} />
        ))}
      </div>
      <p className="ui muted" style={{ margin: '0.6rem 0 0', fontSize: '0.84rem' }} data-readout="hip">{hipLine(p, cells)}</p>
    </Bench>
  );
}
