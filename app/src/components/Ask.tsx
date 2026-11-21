import { useState, type ReactNode } from 'react';
import { useProgress } from '../lib/progress';

// Questions. Each has a stable id; the first answer is stored in the progress
// file and enters the spaced-review pool, so it comes back later.
export interface ChoiceQ {
  kind: 'choice';
  id: string;
  prompt: ReactNode;
  options: { label: ReactNode; correct?: boolean; why?: ReactNode }[];
  /** Keep the written order (a scale, or "none of these" last). Otherwise the order is shuffled. */
  fixedOrder?: boolean;
  explain?: ReactNode;
  /** Where the answer comes from, e.g. "ch. 5 §5.4, p. 110". */
  source?: string;
}
export interface NumericQ {
  kind: 'numeric';
  id: string;
  prompt: ReactNode;
  answer: number;
  /** Relative tolerance; 0 means exact. */
  tol?: number;
  unit?: string;
  explain?: ReactNode;
  source?: string;
}
export type Question = ChoiceQ | NumericQ;

/**
 * The order the options are shown in: a shuffle seeded by the question's id, so it is the
 * same on every visit and machine, and the correct option is not always in the place the
 * author happened to write it. Answers are stored by the written index, so changing the
 * shuffle never invalidates progress. Browser checks set window.__PMPP_WRITTEN_ORDER__ to
 * see the written order, which is what they click by index.
 */
export function displayOrder(q: ChoiceQ): number[] {
  const idx = q.options.map((_, i) => i);
  const written =
    q.fixedOrder || (typeof window !== 'undefined' && (window as { __PMPP_WRITTEN_ORDER__?: boolean }).__PMPP_WRITTEN_ORDER__);
  if (written) return idx;
  let h = 2166136261; // FNV-1a over the id seeds mulberry32, a well-mixed 32-bit generator
  for (let i = 0; i < q.id.length; i++) h = Math.imul(h ^ q.id.charCodeAt(i), 16777619) >>> 0;
  const next = () => {
    h = (h + 0x6d2b79f5) >>> 0;
    let t = Math.imul(h ^ (h >>> 15), h | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(next() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  return idx;
}

export const numericOk = (q: NumericQ, v: number) =>
  q.tol ? Math.abs(v - q.answer) <= Math.abs(q.answer) * q.tol + 1e-12 : v === q.answer;

export function Ask({
  q,
  label = 'Check your reasoning',
  review = false,
  onResult,
}: {
  q: Question;
  label?: string;
  /** In review mode nothing is prefilled and the result only reschedules the item. */
  review?: boolean;
  onResult?: (correct: boolean) => void;
}) {
  const { progress, answer, recordReview } = useProgress();
  const prior = review ? undefined : progress.answers[q.id];
  const [pick, setPick] = useState<number | null>(
    q.kind === 'choice' && prior && typeof prior.value === 'number' ? prior.value : null,
  );
  const [text, setText] = useState(q.kind === 'numeric' && prior && prior.value != null ? String(prior.value) : '');
  const [checked, setChecked] = useState<boolean>(!!prior);

  const value = q.kind === 'choice' ? pick : text.trim() === '' ? null : Number(text);
  const correct =
    value == null || (typeof value === 'number' && isNaN(value))
      ? false
      : q.kind === 'choice'
        ? !!q.options[value as number]?.correct
        : numericOk(q, value as number);

  const check = () => {
    if (value == null) return;
    setChecked(true);
    if (review) recordReview(q.id, correct);
    else {
      if (!progress.answers[q.id]) recordReview(q.id, correct); // first answer schedules it
      answer(q.id, value, correct);
    }
    onResult?.(correct);
  };

  return (
    <div className="q" data-question={q.id} data-question-kind={q.kind}>
      <div className="kind">{label}</div>
      <div className="prompt">{q.prompt}</div>
      {q.kind === 'choice' ? (
        <div className="opts" role="radiogroup">
          {displayOrder(q).map((i) => [q.options[i], i] as const).map(([o, i]) => (
            <label
              key={i}
              className={checked && pick === i ? (o.correct ? 'right' : 'wrong') : checked && o.correct ? 'right' : ''}
            >
              <input
                type="radio"
                data-written-index={i}
                name={q.id + (review ? '-r' : '')}
                checked={pick === i}
                onChange={() => {
                  setPick(i);
                  setChecked(false);
                }}
              />
              <span>
                {o.label}
                {checked && pick === i && o.why ? <span className="muted"> {o.why}</span> : null}
              </span>
            </label>
          ))}
        </div>
      ) : (
        <div className="row" style={{ marginBottom: '0.7rem' }}>
          <input
            type="number"
            step="any"
            value={text}
            aria-label="your answer"
            onChange={(e) => {
              setText(e.target.value);
              setChecked(false);
            }}
            onKeyDown={(e) => e.key === 'Enter' && check()}
          />
          {q.unit ? <span className="ui muted">{q.unit}</span> : null}
        </div>
      )}
      <button className="primary" onClick={check} disabled={value == null}>
        Check
      </button>
      {checked ? (
        <div className={'verdict ' + (correct ? 'right' : 'wrong')}>
          <p>
            <b>{correct ? 'Correct.' : q.kind === 'numeric' ? `Not quite. The answer is ${q.answer}${q.unit ? ' ' + q.unit : ''}.` : 'Not quite.'}</b>
          </p>
          {q.explain ? <div>{q.explain}</div> : null}
          {q.source ? <p className="muted">Source: {q.source}</p> : null}
        </div>
      ) : null}
    </div>
  );
}

/**
 * Commit to a prediction before seeing the answer. The reveal gets what you committed.
 *
 * By default the prediction is a number. With `options` it is a choice among them instead:
 * the reader picks one, the written index of the pick is stored under `id` and passed to
 * `children` (with the option itself as the second argument). Options are shown in the
 * written order, since a prediction has no right answer to hide.
 */
export function Predict({
  id,
  prompt,
  unit,
  options,
  children,
}: {
  id: string;
  prompt: ReactNode;
  unit?: string;
  /** Make this a choice: the options, in the order shown. */
  options?: ReactNode[];
  children: (committed: number, option?: ReactNode) => ReactNode;
}) {
  const { progress, answer } = useProgress();
  const prior = progress.answers[id];
  const [text, setText] = useState('');
  const [pick, setPick] = useState<number | null>(null);
  const stored = typeof prior?.value === 'number' ? prior.value : null;
  // A stored index outside the options (an option removed since) does not count as committed.
  const committed = stored != null && options && !(Number.isInteger(stored) && stored >= 0 && stored < options.length) ? null : stored;
  const commit = () => {
    if (options) {
      if (pick != null) answer(id, pick);
      return;
    }
    const v = Number(text);
    if (text.trim() !== '' && !isNaN(v)) answer(id, v);
  };
  return (
    <div className="q" data-question={id} data-question-kind={options ? 'predict-choice' : 'predict'}>
      <div className="kind">Predict first</div>
      <div className="prompt">{prompt}</div>
      {committed == null ? (
        options ? (
          <>
            <div className="opts" role="radiogroup">
              {options.map((o, i) => (
                <label key={i}>
                  <input type="radio" data-written-index={i} name={id} checked={pick === i} onChange={() => setPick(i)} />
                  <span>{o}</span>
                </label>
              ))}
            </div>
            <button className="primary" onClick={commit} disabled={pick == null}>
              Commit prediction
            </button>
          </>
        ) : (
          <div className="row">
            <input
              type="number"
              step="any"
              value={text}
              aria-label="your prediction"
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && commit()}
            />
            {unit ? <span className="ui muted">{unit}</span> : null}
            <button className="primary" onClick={commit} disabled={text.trim() === ''}>
              Commit prediction
            </button>
          </div>
        )
      ) : (
        <div className="verdict info">
          <p>
            <b>
              {options ? (
                <>You predicted: {options[committed]}</>
              ) : (
                <>
                  You predicted {committed}
                  {unit ? ' ' + unit : ''}.
                </>
              )}
            </b>
          </p>
          {children(committed, options?.[committed])}
        </div>
      )}
    </div>
  );
}
