import { useState, type ReactNode } from 'react';
import { useProgress } from '../lib/progress';

// Questions. Each has a stable id; the first answer is stored in the progress
// file and enters the spaced-review pool, so it comes back later.
export interface ChoiceQ {
  kind: 'choice';
  id: string;
  prompt: ReactNode;
  options: { label: ReactNode; correct?: boolean; why?: ReactNode }[];
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
    <div className="q">
      <div className="kind">{label}</div>
      <div className="prompt">{q.prompt}</div>
      {q.kind === 'choice' ? (
        <div className="opts" role="radiogroup">
          {q.options.map((o, i) => (
            <label
              key={i}
              className={checked && pick === i ? (o.correct ? 'right' : 'wrong') : checked && o.correct ? 'right' : ''}
            >
              <input
                type="radio"
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

/** Commit to a number before seeing the answer. The reveal gets what you committed. */
export function Predict({
  id,
  prompt,
  unit,
  children,
}: {
  id: string;
  prompt: ReactNode;
  unit?: string;
  children: (committed: number) => ReactNode;
}) {
  const { progress, answer } = useProgress();
  const prior = progress.answers[id];
  const [text, setText] = useState('');
  const committed = typeof prior?.value === 'number' ? prior.value : null;
  const commit = () => {
    const v = Number(text);
    if (text.trim() !== '' && !isNaN(v)) answer(id, v);
  };
  return (
    <div className="q">
      <div className="kind">Predict first</div>
      <div className="prompt">{prompt}</div>
      {committed == null ? (
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
      ) : (
        <div className="verdict info">
          <p>
            <b>
              You predicted {committed}
              {unit ? ' ' + unit : ''}.
            </b>
          </p>
          {children(committed)}
        </div>
      )}
    </div>
  );
}
