import { PLAN, useAllChapters } from '../chapters/registry';
import { href } from '../lib/router';
import { useProgress } from '../lib/progress';
import { dueCount } from '../lib/review';
import { useReports } from '../lib/reports';

export function Home() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const chapters = useAllChapters();
  const due = dueCount(progress.review);
  const backends = Object.entries(reports);
  const built = new Map((chapters ?? []).filter((c) => c.sections.length).map((c) => [c.id, c]));
  return (
    <article className="page" data-ready={chapters ? '' : undefined}>
      <header className="section-head">
        <span className="num" />
        <h1>From tensors to kernels</h1>
        <p className="lede">
          A working copy of <i>Programming Massively Parallel Processors</i> (5th edition) for one reader: you know
          what <code>torch.matmul</code> computes, and this is about what the GPU does to compute it, and how to write
          that yourself on CUDA, HIP and Metal.
        </p>
      </header>
      <div className="prose">
        <h2 style={{ marginTop: 0 }}>Where you are</h2>
        <ul>
          {PLAN.filter((r) => r.id && built.has(r.id)).map((r) => {
            const ch = built.get(r.id!)!;
            const done = ch.sections.filter((s) => progress.sections[`${ch.id}/${s.id}`]?.done).length;
            return (
              <li key={ch.id}>
                <a href={href(ch.id, ch.sections[0].id)}>
                  {/^\d/.test(ch.num) ? 'Chapter ' : ch.num.startsWith('X') ? 'Module ' : 'Appendix '}
                  {ch.num}: {ch.title}
                </a>{' '}
                <span className="ui muted">
                  {done} of {ch.sections.length} sections done
                </span>
              </li>
            );
          })}
          <li>
            <a href={href('review')}>Review</a>{' '}
            <span className="ui muted">{due ? `${due} due today` : 'nothing due today'}</span>
          </li>
        </ul>

        <h2>Your devices</h2>
        {backends.length ? (
          <table className="data wide" style={{ maxWidth: '52rem' }}>
            <thead>
              <tr>
                <th>Backend</th>
                <th>Device, as it reports itself</th>
                <th>Last run</th>
              </tr>
            </thead>
            <tbody>
              {backends.map(([name, run]) => (
                <tr key={name}>
                  <td>{name}</td>
                  <td>{String(run.device.name ?? 'unknown')}</td>
                  <td>{run.generated.replace('T', ' ')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p>
            No lab results on this machine yet. Run <code>./verify.sh</code> in the repository root; lessons that compare
            against your own measurements will pick them up.
          </p>
        )}
        <p className="status-line">
          Nothing in the lessons assumes a particular GPU. Where the book works an example on a specific device, the
          lesson says so and puts your own measured numbers beside it.
        </p>

        <h2>The whole course</h2>
        <p>
          Depth follows your goals: matmul, attention, quantised operations and sampling for LLM inference, on more than
          one backend. Three modules are ours because the book does not cover them.
        </p>
        <table className="data wide">
          <thead>
            <tr>
              <th>Ch.</th>
              <th>Title</th>
              <th>Book pp.</th>
              <th>Depth</th>
              <th>Why</th>
              <th>Project step</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {PLAN.map((r) => {
              const ch = r.id ? built.get(r.id) : undefined;
              return (
                <tr key={r.num}>
                  <td>{r.num}</td>
                  <td>{ch ? <a href={href(ch.id, ch.sections[0].id)}>{r.title}</a> : r.title}</td>
                  <td className="nowrap">{r.pages}</td>
                  <td>{r.tier}</td>
                  <td>{r.why}</td>
                  <td>{r.milestone ?? ''}</td>
                  <td className="nowrap">{ch ? 'Ready' : r.tier === 'Skipped' ? 'Not planned' : 'Planned'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </article>
  );
}
