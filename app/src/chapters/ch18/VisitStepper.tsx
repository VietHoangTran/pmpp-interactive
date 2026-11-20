import { useMemo, useState } from 'react';
import { Code } from '../../components/Code';
import { Bench } from '../../components/notes';
import { BOOK, makeGraph } from '../../sim/bfs';
import {
  FORMS,
  FORM_IDS,
  OUTCOME_NAME,
  SCENARIOS,
  UNVISITED,
  canMove,
  classify,
  enumerate,
  isDone,
  replay,
  runOrder,
  type Enumeration,
  type FormId,
  type Move,
  type OutcomeKind,
} from '../../sim/bfsvisit';
import { BFS_TIERS, BfsGraph } from './BfsGraph';

// "Two threads, one neighbour" (section 18.3): step the threads of one level through a visit
// form, one statement at a time, or try every order. The model is src/sim/bfsvisit.ts; each
// form's listing is its region of labs/ch18/snippets/s183_variants.cu.

const TRAP_GRAPH = makeGraph('trap', 'the trap', 5, SCENARIOS.trap.edges, [
  { x: 0.08, y: 0.5 }, { x: 0.4, y: 0.12 }, { x: 0.4, y: 0.88 }, { x: 0.82, y: 0.3 }, { x: 0.82, y: 0.88 },
]);
const BOUND = 12; // steps of one thread after which the stepper calls it a loop that does not end

const badge = (k: OutcomeKind) => <span className={'chip ' + (k === 'correct' ? 'ok' : 'fail')} data-outcome={k}>{OUTCOME_NAME[k]}</span>;

export function VisitStepper() {
  const [scId, setScId] = useState<'book' | 'trap'>('trap');
  const [formId, setFormId] = useState<FormId>('plain-append');
  const [moves, setMoves] = useState<Move[]>([]);
  const [spurious, setSpurious] = useState(1);
  const [all, setAll] = useState<Enumeration | null>(null);
  const sc = SCENARIOS[scId];
  const form = FORMS[formId];
  const g = scId === 'book' ? BOOK : TRAP_GRAPH;
  const { states, events } = useMemo(() => replay(form, sc, moves), [form, sc, moves]);
  const st = states[states.length - 1];
  const last = events[events.length - 1];
  const spinning = st.threads.findIndex((t) => t.pc !== 'done' && t.steps >= BOUND);
  const done = isDone(st);
  const outcome = done ? classify(sc, st) : null;
  const restart = (keepAll = false) => {
    setMoves([]);
    if (!keepAll) setAll(null);
  };
  const runIn = (order: 'forward' | 'reverse') => {
    const r = runOrder(form, sc, order, BOUND);
    const ids = sc.threads.map((_, i) => i);
    if (order === 'reverse') ids.reverse();
    // The same moves runOrder made: each thread to its end (or to the bound), in turn.
    const mv: Move[] = [];
    for (const id of ids) {
      const n = r.events.filter((e) => e.thread === id).length;
      for (let j = 0; j < n; j++) mv.push({ thread: id });
    }
    setMoves(mv);
  };
  const relevant = new Set([...sc.threads.flatMap((t) => [t.vertex, ...t.nbs])]);
  const vertices = [...relevant].sort((a, b) => a - b);
  return (
    <Bench
      title="Two threads, one neighbour"
      task="Pick a visit code, then step the threads in an order of your choosing. Can you make it go wrong?"
      foot={
        <>
          Each step runs one statement of one thread. The neighbour's level is read, written or swapped in global
          memory; <code>e</code> is each thread's own register.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field" style={{ maxWidth: '100%' }}>
          Scenario
          <select value={scId} onChange={(e) => { setScId(e.target.value as 'book' | 'trap'); restart(); }} aria-label="Scenario" style={{ maxWidth: '100%' }}>
            {Object.values(SCENARIOS).map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
          </select>
        </label>
        <label className="field" style={{ maxWidth: '100%' }}>
          Visit code
          <select value={formId} onChange={(e) => { setFormId(e.target.value as FormId); restart(); }} aria-label="Visit code" style={{ maxWidth: '100%' }}>
            {FORM_IDS.map((id) => <option key={id} value={id}>{FORMS[id].label}</option>)}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <button onClick={() => restart(true)} disabled={!moves.length}>Restart</button>
        <button onClick={() => setMoves(moves.slice(0, -1))} disabled={!moves.length}>Back</button>
        <button onClick={() => runIn('forward')}>Run forward order</button>
        <button onClick={() => runIn('reverse')}>Run reverse order</button>
        <span className="muted">{moves.length} step{moves.length === 1 ? '' : 's'}</span>
      </div>
      <div className={'narrate' + (outcome && outcome.kind !== 'correct' ? ' bad' : spinning >= 0 ? ' bad' : '')} role="status" style={{ marginBottom: '0.8rem' }} data-narrate>
        {last ? last.text : <>Nothing has run. Every thread is about to take its first neighbour. Step any thread.</>}
        {spinning >= 0 ? (
          <div style={{ marginTop: '0.3rem' }} data-final="never-terminates">
            {badge('never-terminates')} The thread on {sc.threads[spinning].vertex} has taken {st.threads[spinning].steps} steps and is back where it was: it will go round for ever.
          </div>
        ) : outcome ? (
          <div style={{ marginTop: '0.3rem' }} data-final={outcome.kind}>
            {badge(outcome.kind)} {outcome.detail}.
          </div>
        ) : null}
      </div>

      {/* The listing takes the bench's full width, so its longest lines (the signature, the loop
          header) show whole and the stepped line is never the part cut off. */}
      <div data-visit-code>
        <Code tabs={[{ label: form.region, code: form.code.join('\n'), tiers: BFS_TIERS, status: 'emulator-checked in forward and reverse order' }]} highlight={last ? [last.line + 1] : []} />
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 6fr) minmax(0, 5fr)' }}>
        <div style={{ minWidth: 0 }}>
          <div style={{ overflowX: 'auto' }}>
          <table className="threads" aria-label="Threads">
            <thead>
              <tr><th>thread on</th><th>neighbours</th><th>now at</th><th>e</th><th /></tr>
            </thead>
            <tbody>
              {sc.threads.map((t, i) => {
                const ts = st.threads[i];
                const at = ts.pc === 'done' ? 'returned' : ts.pc === 'next' ? `next neighbour (${ts.k + 1} of ${t.nbs.length})` : `${ts.pc} on ${ts.nb}`;
                const canSp = canMove(form, st, { thread: i, spurious: true });
                return (
                  <tr key={i} className={last?.thread === i ? 'cur' : ''} data-thread={i}>
                    <td>{t.vertex}</td>
                    <td>{t.nbs.join(', ')}</td>
                    <td>{at}</td>
                    <td className="t-reg">{ts.e === UNVISITED ? 'UNVISITED' : ts.e}</td>
                    <td>
                      <span className="row" style={{ gap: '0.3rem', flexWrap: 'nowrap' }}>
                        <button onClick={() => setMoves([...moves, { thread: i }])} disabled={ts.pc === 'done'}>Step {t.vertex}</button>
                        {form.weak ? (
                          <button className="quiet" onClick={() => setMoves([...moves, { thread: i, spurious: true }])} disabled={!canSp}
                            title="Make this weak compare-and-swap fail although the values are equal">
                            fail spuriously
                          </button>
                        ) : null}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          </div>
        </div>
        <div style={{ minWidth: 0 }}>
          <BfsGraph
            g={g}
            label="The scenario's graph with the current levels"
            height={200}
            look={(v) => (!relevant.has(v) && scId === 'book' ? 'dim' : st.level[v] === UNVISITED ? 'unvisited' : sc.level0[v] === UNVISITED ? 'new' : 'visited')}
            tag={(v) => (st.level[v] === UNVISITED ? null : String(st.level[v]))}
            ring={(v) => (st.writes[v] > 1 || (sc.level0[v] !== UNVISITED && st.level[v] !== sc.level0[v]) ? 'fail' : null)}
            edgeLook={(_, a) => (sc.threads.some((t) => t.vertex === a) ? 'plain' : 'dim')}
          />
          <div className="mx-label t-global" style={{ marginTop: '0.5rem' }}>level[] (global memory)</div>
          <div className="mx global" style={{ gridTemplateColumns: `repeat(${vertices.length}, auto)` }} data-levels>
            {vertices.map((v) => {
              const changed = st.level[v] !== sc.level0[v];
              const bad = st.writes[v] > 1 || (sc.level0[v] !== UNVISITED && changed);
              return (
                <span key={v} data-v={v} className={bad ? 'bad' : changed ? 'hot' : ''} title={`level[${v}], written ${st.writes[v]} times`}>
                  {st.level[v] === UNVISITED ? '·' : st.level[v]}
                  <sub>{v}</sub>
                </span>
              );
            })}
          </div>
          <p className="muted" style={{ margin: '0.3rem 0 0.6rem' }}>Small numbers are vertex ids; · is UNVISITED. Red: written twice, or an older level overwritten.</p>
          {form.appends ? (
            <>
              <div className="mx-label t-global">currFrontier (global memory), counter = {st.counter}</div>
              <div className="mx global" style={{ gridTemplateColumns: `repeat(${Math.max(1, st.frontier.length)}, auto)` }} data-frontier={st.frontier.join(',')}>
                {st.frontier.length ? st.frontier.map((v, i) => (
                  <span key={i} className={st.frontier.indexOf(v) !== i ? 'bad' : 'hot'}>{v}<sub>{i}</sub></span>
                )) : <span className="void">empty</span>}
              </div>
            </>
          ) : (
            <p className="muted" style={{ margin: 0 }}>No frontier: this code only labels.</p>
          )}
        </div>
      </div>

      <div className="panel" style={{ marginTop: '0.9rem' }}>
        <div className="row">
          <button className="primary" onClick={() => setAll(enumerate(form, sc, { spurious: form.weak ? spurious : 0 }))}>Try every order</button>
          {form.weak ? (
            <label className="field">
              Spurious failures allowed
              <span className="seg" role="group" aria-label="Spurious failures allowed">
                {[0, 1, 2].map((k) => <button key={k} aria-pressed={spurious === k} onClick={() => { setSpurious(k); setAll(null); }}>{k}</button>)}
              </span>
            </label>
          ) : null}
        </div>
        {all ? (
          <div style={{ marginTop: '0.6rem' }} data-every={all.outcomes.map((o) => o.kind).join(',')}>
            <p style={{ margin: '0 0 0.4rem' }}>
              {all.outcomes.length === 1 && all.outcomes[0].kind === 'correct'
                ? <>Every order is correct ({all.states} distinct states explored).</>
                : <>Over every order ({all.states} distinct states), the outcomes are:</>}
            </p>
            {all.outcomes.length > 1 || all.outcomes[0].kind !== 'correct' ? (
              <ul style={{ margin: 0, paddingLeft: '1.2rem' }}>
                {all.outcomes.map((o) => (
                  <li key={o.kind} style={{ marginBottom: '0.3rem' }}>
                    {badge(o.kind)} {o.detail}.{' '}
                    {o.example.length ? (
                      <button className="quiet" onClick={() => setMoves(o.example)}>Load this order ({o.example.length} steps)</button>
                    ) : null}
                  </li>
                ))}
              </ul>
            ) : null}
          </div>
        ) : null}
      </div>
    </Bench>
  );
}
