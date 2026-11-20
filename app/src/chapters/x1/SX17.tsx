import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Today } from '../../components/notes';
import { href } from '../../lib/router';
import {
  E3,
  FIRST_ONE,
  JUMPS,
  KATS,
  M0,
  M1,
  MPS_ROUNDS,
  ROUNDS,
  W0,
  W1,
  avalanche,
  binadeProfile,
  curandUniform,
  differingInputs,
  f32bits,
  f32short,
  hex,
  hipUniformAMD,
  m5Input,
  m5Uniform,
  mpsInput,
  onesCount,
  parse64,
  philoxTrace,
  roundsUntilAllWordsDiffer,
  sameWords,
  sampleConversions,
} from '../../sim/philox';
import { PhiloxRoundsScene } from './PhiloxRoundsScene';

// Section X1.7: counter-based random numbers. Everything the interactives show is computed by
// src/sim/philox.ts (philoxTrace, avalanche, the three conversions and the binade analysis),
// pinned in philox.test.ts, which also runs labs/x1/snippets/philox.cu on the emulator and
// philox.metal on this Mac's GPU against the same numbers.

const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const MONO: CSSProperties = { fontFamily: 'var(--sans)', fontVariationSettings: "'MONO' 1" };
const loaded = (t: string | null) => (t && !t.startsWith('// could not load') ? t : null);
const big = (v: bigint) => '0x' + v.toString(16);
const MAX = (1n << 64n) - 1n;

// ---------------------------------------------------------------- the stepper
const PRESETS: { id: string; label: string; seed: bigint; stream: bigint; step: bigint }[] = [
  { id: 'zero', label: 'all zero', seed: 0n, stream: 0n, step: 0n },
  { id: 'ones', label: 'all ones', seed: MAX, stream: MAX, step: MAX },
  // Random123's "digits of pi" vector, expressed through M5's mapping (counter words are step lo, step hi, stream lo, stream hi)
  { id: 'pi', label: 'digits of pi', seed: 0x299f31d0a4093822n, stream: 0x0370734413198a2en, step: 0x85a308d3243f6a88n },
];

function WordCell({ v, diff, ok, id }: { v: number; diff?: boolean; ok?: boolean; id?: string }) {
  return (
    <span data-w={id} data-value={hex(v)} data-diff={diff ? 1 : 0}
      style={{
        ...MONO, display: 'inline-block', padding: '0 0.3rem', marginRight: 3, borderRadius: 3, fontSize: '0.78rem',
        border: `1px solid ${ok ? 'var(--ok)' : diff ? 'var(--ink)' : 'var(--rule)'}`,
        background: ok ? 'var(--ok-wash)' : diff ? 'var(--rule)' : 'var(--cell)', fontWeight: diff ? 700 : 400,
      }}>
      {hex(v)}
    </span>
  );
}

function PhiloxStepper() {
  const [seedT, setSeedT] = useState('0');
  const [streamT, setStreamT] = useState('0');
  const [stepT, setStepT] = useState('0');
  const [mps, setMps] = useState(false);
  const [at, setAt] = useState(0);
  const seed = parse64(seedT);
  const stream = parse64(streamT);
  const step = parse64(stepT);
  const valid = seed !== null && stream !== null && step !== null;
  const R = mps ? MPS_ROUNDS : ROUNDS;

  const run = useMemo(() => {
    if (!valid) return null;
    const s1 = (stream! + 1n) & MAX;
    const A = mps ? mpsInput(seed!, stream!) : m5Input(seed!, stream!, step!);
    const B = mps ? mpsInput(seed!, s1) : m5Input(seed!, s1, step!);
    const ta = philoxTrace(A.ctr, A.key, R);
    const tb = philoxTrace(B.ctr, B.key, R);
    const av = avalanche(A.ctr, B.ctr, A.key, R);
    return { A, B, ta, tb, av, s1 };
  }, [seed, stream, step, mps, valid, R]);

  const set = (f: (s: string) => void) => (s: string) => {
    f(s);
    setAt(0);
  };
  const preset = (id: string) => {
    const p = PRESETS.find((x) => x.id === id)!;
    setSeedT(big(p.seed));
    setStreamT(big(p.stream));
    setStepT(big(p.step));
    setAt(0);
  };
  const activePreset = valid ? PRESETS.find((p) => p.seed === seed && p.stream === stream && p.step === step)?.id : undefined;

  if (!run)
    return (
      <Bench title="Philox round by round" task="Enter a seed, a stream and a step, or pick a known answer.">
        <PhiloxInputs {...{ seedT, streamT, stepT, setSeedT: set(setSeedT), setStreamT: set(setStreamT), setStepT: set(setStepT), preset, activePreset, mps, setMps: (v: boolean) => { setMps(v); setAt(0); } }} />
        <p className="t-fail" data-invalid="">Each value must be a whole number from 0 to 2⁶⁴ − 1, in decimal or with 0x.</p>
      </Bench>
    );

  const { A, ta, tb, av, s1 } = run;
  const cur = at > 0 ? ta[at - 1] : null;
  const done = at === R;
  const kat = !mps ? KATS.find((k) => sameWords(k.ctr, A.ctr) && sameWords(k.key, A.key)) : undefined;
  const e3 = mps ? E3.find((e) => e.seed === seed && e.index === stream) : undefined;
  const out = ta[R - 1].out;
  const katOk = done && kat ? sameWords(out, kat.expected) : false;
  const mpsOk = done && e3 ? sameWords(out, e3.mps) : false;

  const narrate = !cur
    ? `Nothing has run yet. The counter is (${A.ctr.map(hex).join(', ')}) and the key (${A.key.map(hex).join(', ')}). ${mps ? 'In PyTorch\'s Metal layout as we read it, the key is (seed high, seed low) and the index sits in words z and w, high word first.' : 'In M5\'s layout the counter is (step low, step high, stream low, stream high) and the key is (seed low, seed high).'} Press Step.`
    : `Round ${cur.round}${cur.round > 1 ? `, with the key bumped by (${hex(W0)}, ${hex(W1)}) to (${cur.key.map(hex).join(', ')})` : ', with the key as given'}: ${hex(M0)} × ${hex(cur.p0.a)} = ${hex(cur.p0.hi)} ${hex(cur.p0.lo)} and ${hex(M1)} × ${hex(cur.p1.a)} = ${hex(cur.p1.hi)} ${hex(cur.p1.lo)}. The high halves are XORed with y and w and the key; the low halves move into y and w. ${av.differing[cur.round - 1]} of the 4 words now differ from stream ${s1}'s.`;

  return (
    <Bench
      title="Philox round by round"
      task="Step through the all-zero vector, then change the stream by one and watch the right column."
      foot={
        <>
          Exact 32-bit arithmetic. Left: your inputs. Right: the same seed and step with stream + 1; a word in bold with a
          heavier border differs from the one beside it on the left. A green word matches the published vector.
        </>
      }
    >
      <PhiloxInputs {...{ seedT, streamT, stepT, setSeedT: set(setSeedT), setStreamT: set(setStreamT), setStepT: set(setStepT), preset, activePreset, mps, setMps: (v: boolean) => { setMps(v); setAt(0); } }} />
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setAt(0)} disabled={at === 0}>Restart</button>
        <button onClick={() => setAt(at - 1)} disabled={at === 0}>Back</button>
        <button className="primary" onClick={() => setAt(at + 1)} disabled={done}>Step</button>
        <button onClick={() => setAt(R)} disabled={done}>Run to the end</button>
        <span className="muted" aria-live="polite" data-step={at}>round {at} of {R}</span>
      </div>
      <div className="narrate" role="status" style={{ marginBottom: '0.9rem' }}>{narrate}</div>

      <div style={{ overflowX: 'auto' }}>
        <table className="threads" style={{ minWidth: '44rem' }} data-part="rounds">
          <thead>
            <tr>
              <th>round</th>
              <th>key</th>
              <th>stream {stream!.toString()}</th>
              <th>stream {s1.toString()}</th>
              <th>differ</th>
            </tr>
          </thead>
          <tbody>
            <tr data-round={0}>
              <td>in</td>
              <td>{A.key.map((k, i) => <WordCell key={i} v={k} />)}</td>
              <td>{A.ctr.map((v, i) => <WordCell key={i} v={v} id={`in-a-${i}`} />)}</td>
              <td>{run.B.ctr.map((v, i) => <WordCell key={i} v={v} id={`in-b-${i}`} diff={v !== A.ctr[i]} />)}</td>
              <td>{A.ctr.filter((v, i) => v !== run.B.ctr[i]).length}</td>
            </tr>
            {ta.slice(0, at).map((r, k) => (
              <tr key={k} data-round={r.round} data-differing={av.differing[k]} className={k === at - 1 ? 'cur' : ''}>
                <td>{r.round}</td>
                <td>{r.key.map((v, i) => <WordCell key={i} v={v} />)}</td>
                <td>{r.out.map((v, i) => <WordCell key={i} v={v} id={`a-${i}`} ok={k === R - 1 && (katOk || mpsOk)} />)}</td>
                <td>{tb[k].out.map((v, i) => <WordCell key={i} v={v} id={`b-${i}`} diff={av.masks[k][i]} />)}</td>
                <td>{av.differing[k]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {cur ? (
        <div className="panel" style={{ marginTop: '0.8rem' }} data-part="detail">
          <div className="mx-label">Round {cur.round} of the left column in detail</div>
          <div style={{ ...MONO, fontSize: '0.8rem', lineHeight: 1.7 }}>
            <div>{hex(M0)} × x = {hex(cur.p0.a)} → high {hex(cur.p0.hi)}, low {hex(cur.p0.lo)}</div>
            <div>{hex(M1)} × z = {hex(cur.p1.a)} → high {hex(cur.p1.hi)}, low {hex(cur.p1.lo)}</div>
            <div>new x = {hex(cur.p1.hi)} ⊕ y {hex(cur.ctrIn[1])} ⊕ key {hex(cur.key[0])} = <b>{hex(cur.out[0])}</b>; new y = low 1 = <b>{hex(cur.out[1])}</b></div>
            <div>new z = {hex(cur.p0.hi)} ⊕ w {hex(cur.ctrIn[3])} ⊕ key {hex(cur.key[1])} = <b>{hex(cur.out[2])}</b>; new w = low 0 = <b>{hex(cur.out[3])}</b></div>
          </div>
        </div>
      ) : null}

      {done ? (
        <div style={{ marginTop: '0.8rem' }} data-part="result" data-kat={kat?.id ?? ''} data-kat-ok={katOk ? 1 : 0} data-mps-ok={mpsOk ? 1 : 0} data-all-by={av.allBy ?? ''}>
          {kat ? (
            <p style={{ margin: 0 }}>
              {katOk ? <b className="t-ok">✓ </b> : <b className="t-fail">✗ </b>}
              Random123's vector "{kat.label}": {kat.expected.map(hex).join(' ')}. {katOk ? 'Exactly these words.' : 'The words do not match.'}
            </p>
          ) : mps ? (
            <p style={{ margin: 0 }}>
              {e3 ? (
                <>
                  {mpsOk ? <b className="t-ok">✓ </b> : <b className="t-fail">✗ </b>}
                  E3's transcription of PyTorch's Metal Philox for seed {e3.seed.toString()}, index {e3.index.toString()}:{' '}
                  {e3.mps.map(hex).join(' ')}. Philox4x32-10 in M5's layout gives {e3.ten.map(hex).join(' ')} for the same seed
                  and index: a different stream.
                </>
              ) : (
                <>Nine rounds in PyTorch's word order. Seeds 0 and 42 with stream 0 have transcribed vectors (E3) to compare with.</>
              )}
            </p>
          ) : (
            <p style={{ margin: 0 }} className="muted">
              No published vector for these inputs; the presets have one each. These words are what any correct Philox4x32-10 must
              produce for them.
            </p>
          )}
          {!mps ? (
            <p style={{ margin: '0.4rem 0 0' }} data-u={m5Uniform(out[0])}>
              M5's u for this row and step: word 0 is {hex(out[0])}, its top 24 bits are {(out[0] >>> 8).toLocaleString('en-US')},
              and u = {(out[0] >>> 8).toLocaleString('en-US')} × 2⁻²⁴ = <b>{f32short(m5Uniform(out[0]))}</b>.
            </p>
          ) : null}
          <p style={{ margin: '0.4rem 0 0' }}>
            Streams {stream!.toString()} and {s1.toString()}: all four words differ from round <b data-allby="">{av.allBy ?? 'never'}</b> on.
          </p>
        </div>
      ) : null}
    </Bench>
  );
}

function PhiloxInputs(p: {
  seedT: string; streamT: string; stepT: string;
  setSeedT: (s: string) => void; setStreamT: (s: string) => void; setStepT: (s: string) => void;
  preset: (id: string) => void; activePreset?: string; mps: boolean; setMps: (v: boolean) => void;
}) {
  return (
    <>
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <label className="field">
          seed (64-bit)
          <input type="text" value={p.seedT} aria-label="seed" onChange={(e) => p.setSeedT(e.target.value)} style={{ width: '12.5rem' }} />
        </label>
        <label className="field">
          stream (the row's request)
          <input type="text" value={p.streamT} aria-label="stream" onChange={(e) => p.setStreamT(e.target.value)} style={{ width: '12.5rem' }} />
        </label>
        <label className="field">
          step
          <input type="text" value={p.stepT} aria-label="step" disabled={p.mps} onChange={(e) => p.setStepT(e.target.value)} style={{ width: '12.5rem' }} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          Known answers
          <span className="seg" role="group" aria-label="Known answers">
            {PRESETS.map((x) => (
              <button key={x.id} aria-pressed={p.activePreset === x.id} onClick={() => p.preset(x.id)}>{x.label}</button>
            ))}
          </span>
        </span>
        <label className="check">
          <input type="checkbox" checked={p.mps} aria-label="PyTorch MPS as we read its source" onChange={(e) => p.setMps(e.target.checked)} />
          PyTorch MPS as we read its source (9 rounds, high word first; not run)
        </label>
      </div>
    </>
  );
}

// ---------------------------------------------------------------- bits to u
function BitsExplorer() {
  const [x, setX] = useState(FIRST_ONE);
  const [text, setText] = useState('0x' + FIRST_ONE.toString(16));
  const profile = useMemo(() => binadeProfile(), []);
  const sample = useMemo(() => sampleConversions(200000, 1), []);
  const setFrom = (s: string) => {
    setText(s);
    const v = parse64(s);
    if (v !== null && v <= 0xffffffffn) setX(Number(v));
  };
  const jump = (v: number) => {
    setX(v);
    setText(v < 4096 ? String(v) : '0x' + v.toString(16));
  };
  const c = curandUniform(x);
  const h = hipUniformAMD(x);
  const m = m5Uniform(x);
  const agree = c === h;
  const binade = x === 0 ? -1 : Math.floor(Math.log2(x));
  const rows: { id: string; label: string; v: number; note: string }[] = [
    { id: 'curand', label: 'curand_uniform (NVIDIA): x·2⁻³² + 2⁻³³', v: c, note: '(0, 1]' },
    { id: 'hip', label: 'hiprand_uniform on AMD (rocRAND): 2⁻³² + x·2⁻³²', v: h, note: '(0, 1]' },
    { id: 'm5', label: 'M5: (x >> 8) × 2⁻²⁴', v: m, note: '[0, 1 − 2⁻²⁴]' },
  ];
  const differ = differingInputs();
  return (
    <Bench
      title="Bits to u"
      task="Jump to 0, then 2²³ and 2²⁴ + 1: where do the vendors disagree? Then 0xFFFFFF80."
      foot={<>Each conversion rounded to float32, round to nearest even, once per operation, as every backend does for these exact operations.</>}
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <label className="field">
          x (32 bits, decimal or 0x)
          <input type="text" value={text} aria-label="x" onChange={(e) => setFrom(e.target.value)} style={{ width: '10rem' }} />
        </label>
        <span style={FIELD}>
          Jump to
          <span className="seg" role="group" aria-label="Jump to" style={{ flexWrap: 'wrap' }}>
            {JUMPS.map((j) => (
              <button key={j.label} aria-pressed={x === j.x} onClick={() => jump(j.x)}>{j.label}</button>
            ))}
          </span>
        </span>
      </div>
      <input type="range" min={0} max={4294967295} step={1} value={x} aria-label="x slider" style={{ width: '100%', marginBottom: '0.6rem' }}
        onChange={(e) => jump(Number(e.target.value))} />
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0, background: 'var(--page)', minWidth: '36rem' }} data-part="conversions" data-x={x}>
          <thead>
            <tr><th style={{ paddingLeft: '0.6rem' }}>Conversion</th><th className="num">u</th><th>float32 bits</th><th>range</th><th>1.0?</th></tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} data-conv={r.id} data-u={r.v} data-bits={hex(f32bits(r.v))}>
                <td style={{ paddingLeft: '0.6rem' }}>{r.label}</td>
                <td className="num" style={MONO}>{f32short(r.v)}</td>
                <td style={MONO}>{hex(f32bits(r.v))}</td>
                <td>{r.note}</td>
                <td>{r.v === 1 ? <b className="t-fail">exactly 1.0</b> : 'no'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p style={{ margin: '0.7rem 0 0' }} data-agree={agree ? 1 : 0}>
        For x = {x.toLocaleString('en-US')}: NVIDIA and AMD {agree ? <b className="t-ok">agree</b> : <b className="t-fail">give different floats</b>}
        {agree ? '' : ` (${f32short(c)} and ${f32short(h)})`}.
      </p>

      <div className="panel" style={{ marginTop: '0.8rem' }} data-part="strip">
        <div className="mx-label">Where the vendors disagree, binade by binade: x = 0, then x in [2ᵏ, 2ᵏ⁺¹) for k = 0 to 31</div>
        <div style={{ display: 'flex', gap: 2, alignItems: 'flex-end', overflowX: 'auto' }}>
          {profile.map((r) => {
            const share = r.differ / r.size;
            return (
              <div key={r.k} data-binade={r.k} data-share={share} title={`k = ${r.k}: ${r.differ.toLocaleString('en-US')} of ${r.size.toLocaleString('en-US')} differ`}
                style={{ flex: '1 1 0', minWidth: 10, textAlign: 'center', fontSize: '0.6rem', color: 'var(--ink-2)' }}>
                <div style={{ height: 48, display: 'flex', alignItems: 'flex-end', border: `1px solid ${binade === r.k ? 'var(--ink)' : 'var(--rule)'}` }}>
                  <div style={{ width: '100%', height: `${share * 100}%`, background: 'var(--fail)' }} />
                </div>
                <div style={{ height: 10 }}>{r.onesCurand ? <b style={{ color: 'var(--fail)' }}>1.0</b> : ''}</div>
                {r.k >= 0 && r.k % 4 === 0 ? r.k : '\u00a0'}
              </div>
            );
          })}
        </div>
        <p style={{ margin: '0.5rem 0 0', fontSize: '0.85rem' }} data-differ={differ} data-ones={onesCount('curand')} data-ones-hip={onesCount('hip')}>
          Bar height: the share of the binade where the two vendor conversions differ. Every x below 2²³ differs; half of [2²³,
          2²⁴); a quarter of [2²⁴, 2²⁵), where AMD's added 2⁻³² is exactly half a float step and the tie rounds to even; none
          above. In all <b>{differ.toLocaleString('en-US')}</b> of 2³² inputs differ, one in {2 ** 32 / differ}. Both conversions
          return exactly 1.0 for the top <b>{onesCount('curand')}</b> inputs (x ≥ 0xFFFFFF80). M5's never does.
        </p>
        <p className="muted" style={{ margin: '0.4rem 0 0', fontSize: '0.8rem' }} data-sample={sample.differ}>
          Check on a seeded sample of {sample.n.toLocaleString('en-US')} Philox words: {sample.differ.toLocaleString('en-US')} differ
          ({((sample.differ / sample.n) * 1e6).toFixed(0)} per million; the analysis says {(1e6 / 256).toFixed(0)}).{' '}
          {sample.m5AtOne ? `M5's u reached 1.0 ${sample.m5AtOne} times.` : "M5's u never reached 1.0."}
        </p>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSX17: Record<string, Question> = {
  'x1/X1.7/q-stateless': {
    kind: 'choice',
    id: 'x1/X1.7/q-stateless',
    prompt: <>Why does a counter-based generator suit a batched sampler better than one <code>std::mt19937</code> shared by the batch?</>,
    options: [
      {
        label: "Each (seed, stream, step) maps straight to its numbers, so each row's thread computes its own u in any order, and a row's draw does not depend on how many rows were drawn before it",
        correct: true,
      },
      { label: 'Philox has a longer period than mt19937, so a long generation never repeats', why: "mt19937's period is far longer than any generation; the period is not the problem, the shared sequential state is." },
      { label: 'mt19937 cannot run on a GPU, so the host would have to draw every u', why: 'It can be ported; the trouble is that its next value depends on every value before it, so rows would wait on each other and on the batch order.' },
    ],
    explain: (
      <p>
        A sequential generator hands out values in order: row 5's u depends on how many values rows 0 to 4 took this step,
        which changes with the batch. A counter-based generator is a pure function of (counter, key): no thread waits, and
        any backend that implements the same function gets the same bits.
      </p>
    ),
    source: 'notes/today/x1-facts.md A4, D3 (checked 2026-11-20)',
  },
  'x1/X1.7/q-mapping': {
    kind: 'choice',
    id: 'x1/X1.7/q-mapping',
    prompt: <>By our reading of cuRAND's header, which Philox call produces the first <code>curand()</code> after <code>curand_init(seed, stream, 4 × step, &s)</code>?</>,
    options: [
      { label: 'Counter (step low, step high, stream low, stream high), key (seed low, seed high), output word 0', correct: true },
      { label: 'Counter (stream low, stream high, step low, step high), key (seed low, seed high), output word 0', why: 'The subsequence goes into words z and w, the offset / 4 into x and y.' },
      { label: 'Counter (step low, step high, stream low, stream high), key (seed high, seed low), output word 3', why: "The key's low word comes first, and the state starts at offset mod 4 = 0, so word 0." },
    ],
    explain: (
      <p>
        <code>curand_init</code> puts the seed in the key (low, high), adds the subsequence to counter words z and w, adds
        offset / 4 to x and y, and starts at word offset mod 4. With offset 4 × step that is word 0 of M5's counter. This is
        our derivation from the header, unverified on a device until a CUDA verify report shows lab X1.3's known-answer test
        passing.
      </p>
    ),
    source: 'notes/today/x1-facts.md A3, E1 (our derivation; unverified on a device)',
  },
  'x1/X1.7/q-one': {
    kind: 'numeric',
    id: 'x1/X1.7/q-one',
    prompt: <>For how many of the 2³² possible inputs does <code>curand_uniform</code> return exactly 1.0?</>,
    answer: 128,
    unit: 'inputs',
    explain: (
      <p>
        Every x from 0xFFFFFF80 up converts to the float 2³² (the nearest float32), so x·2⁻³² is 1.0 and adding 2⁻³³ leaves
        it there: 2³² − 0xFFFFFF80 = 128 inputs. AMD's conversion returns 1.0 for the same 128.
      </p>
    ),
    source: 'notes/today/x1-facts.md A2, E2; our arithmetic, src/sim/philox.ts',
  },
  'x1/X1.7/q-differ': {
    kind: 'numeric',
    id: 'x1/X1.7/q-differ',
    prompt: <>For identical raw bits, <code>curand_uniform</code> on NVIDIA and <code>hiprand_uniform</code> on AMD return different floats for 1 in how many inputs?</>,
    answer: 256,
    explain: (
      <p>
        Exactly 2²⁴ of 2³² inputs, by our binade analysis: all x below 2²³, the even x in [2²³, 2²⁴), and a quarter of [2²⁴,
        2²⁵) where AMD's 2⁻³² is half a float step and ties round to even. The facts pass sampled 3,903 per million, which
        agrees.
      </p>
    ),
    source: 'notes/today/x1-facts.md B2, E2; our arithmetic, src/sim/philox.ts',
  },
  'x1/X1.7/q-metal': {
    kind: 'choice',
    id: 'x1/X1.7/q-metal',
    prompt: <>What does the Metal Shading Language give a kernel for random numbers?</>,
    options: [
      { label: 'Nothing: MSL 4.1 has no random function, so M5 writes Philox in MSL with mulhi or a ulong product', correct: true },
      { label: 'A Philox state type and a uniform function, like cuRAND\'s device API', why: "Apple's Philox is MPSMatrixRandomPhilox, encoded from the host, and MPSGraph ops; neither is callable from a shader." },
      { label: 'A hash-based noise function meant for graphics, which a sampler can use', why: 'The spec has no random, noise or hash function at all; "random" appears once, about iterators.' },
    ],
    explain: (
      <p>
        The 4.1 spec's only "random" is "random access iterator". Apple's documented Philox is a host-encoded fill kernel
        and MPSGraph ops whose rounds, seeding and conversion are not documented. MSL does have <code>mulhi</code> and{' '}
        <code>ulong</code>, which is all Philox needs.
      </p>
    ),
    source: 'notes/today/x1-facts.md C1, C2, C3, C4 (checked 2026-11-20)',
  },
  'x1/X1.7/q-raw-bits': {
    kind: 'choice',
    id: 'x1/X1.7/q-raw-bits',
    prompt: <>Why does M5 take raw 32-bit words and convert them itself instead of calling <code>curand_uniform</code>?</>,
    options: [
      {
        label: "cuRAND does not promise how many raw values a distribution consumes, the vendors' conversions differ and can return 1.0, and Metal has none; the 24-bit conversion is exact everywhere",
        correct: true,
      },
      { label: 'curand_uniform is slower than a shift and a multiply', why: 'Speed is not the reason; agreement across backends is.' },
      { label: "curand_uniform returns a double, and Metal has no double", why: 'It returns a float in (0, 1]. The problem is which float, and on which backend.' },
    ],
    explain: (
      <p>
        "The number of values consumed is not guaranteed to be fixed" (cuRAND's docs). NVIDIA's and AMD's formulas differ for
        1 in 256 inputs and both can return 1.0. <code>(x &gt;&gt; 8) × 2⁻²⁴</code> is an integer shift, an exact conversion
        and a power-of-two scale: the same float on every backend under any math mode.
      </p>
    ),
    source: 'notes/today/x1-facts.md A1, A2, B2, C1, E2',
  },
  'x1/X1.7/q-slot': {
    kind: 'choice',
    id: 'x1/X1.7/q-slot',
    prompt: (
      <>
        Under continuous batching a request sits in row 3 at one step and row 1 at the next. If the generator's subsequence
        were the batch row, as in FlashInfer's kernel, what happens to that request's reproducibility?
      </>
    ),
    options: [
      { label: "Its u depends on its slot, so the same seed with a different batch mix gives different tokens; M5 takes a per-row stream id (the request's) and a per-row step", correct: true },
      { label: 'Nothing: the seed alone fixes the stream', why: 'The seed is the key; the row picks the subsequence, so the slot changes the bits.' },
      { label: 'Only the first token changes; later steps realign', why: 'Every step whose slot differs draws from a different subsequence.' },
    ],
    explain: (
      <p>
        FlashInfer calls <code>curand_init(philox_seed, bx, philox_offset, &state)</code> with the block index as the
        subsequence. That is reproducible for a fixed batch. M5 makes the stream and the step inputs per row, so a request's
        bits follow the request, not the slot (our design choice).
      </p>
    ),
    source: 'notes/today/x1-facts.md A6; ours',
  },
};

// ---------------------------------------------------------------- the section
export function SX17() {
  const cu = loaded(useLabFile('x1/snippets/philox.cu'));
  const metal = loaded(useLabFile('x1/snippets/philox.metal'));
  return (
    <>
      <p>
        The last stage of the sampler needs one uniform number u per row and per step. Every backend must produce the same u
        from the same inputs, or the same request will draw different tokens on CUDA, HIP and Metal. This page builds that
        number from three inputs per row: a seed, a stream and a step.
      </p>
      <Beyond>
        <p>
          Nothing here is in the book. Random123's Philox is BSD-licensed and is shown with its notice. cuRAND's headers are
          under NVIDIA's restrictive licence, so they are described here, never shown (x1-facts, licence note).
        </p>
      </Beyond>

      <h2>Why not one generator for the batch</h2>
      <p>
        llama.cpp's host sampler draws from a <code>std::mt19937</code>. That is a sequential state: each value depends on
        every value before it. One generator shared by a batch makes row 5's u depend on how many values rows 0 to 4 took, so
        the batch order leaks into every token, and no row can draw until the rows before it have.
      </p>
      <p>
        A counter-based generator has no state at all. It is a function: result = f(counter, key). Put the step and the
        row's stream in the counter and the seed in the key, and every thread computes its own row's numbers directly, in
        any order, on any device. You have used one already: <code>torch.rand</code> on a CUDA tensor initialises cuRAND's
        Philox per thread with <code>curand_init(seed, idx, offset, &state)</code>.
      </p>
      <Today checked="2026-11-20 (notes/today/x1-facts.md A3, A4, D3)">
        <p>
          llama.cpp <code>dist</code>: <code>std::mt19937 rng</code>. Random123: counter-based generators are "stateless
          functions"; "a unique (counter, key) tuple will always produce the same result."
        </p>
      </Today>
      <Ask q={questionsSX17['x1/X1.7/q-stateless']} />

      <h2>Philox4x32-10, round by round</h2>
      <p>
        Philox's state is a 128-bit counter of four 32-bit words and a 64-bit key of two. One round multiplies word 0 by
        0xD2511F53 and word 2 by 0xCD9E8D57. Each 64-bit product splits into a high and a low half. The high halves are XORed
        with the other two words and the key, the low halves move into the gaps, and the result is the next round's counter.
        Between rounds the key grows by 0x9E3779B9 and 0xBB67AE85. Ten rounds make Philox4x32-10. Each round is two integer
        multiplies and a few XORs, which is why it is cheap enough to run once per row per step.
      </p>
      <Today checked="2026-11-20 (notes/today/x1-facts.md A4)">
        <p>
          From Random123's <code>philox.h</code>: the round's output is <code>{'{hi1^ctr[1]^key[0], lo1, hi0^ctr[3]^key[1], lo0}'}</code>;
          no key bump before the first round; 10 rounds by default.
        </p>
      </Today>
      <Predict
        id="x1/X1.7/p-avalanche"
        prompt={
          <>
            Seed 0 and step 0. Rows 0 and 1 differ in one bit of the counter (the stream word). After how many rounds do
            all four output words differ?
          </>
        }
        unit="rounds"
      >
        {(v) => {
          const r = roundsUntilAllWordsDiffer(0n, 0n, 0n);
          const a = avalanche(m5Input(0n, 0n, 0n).ctr, m5Input(0n, 1n, 0n).ctr, [0, 0]);
          return (
            <p data-reveal={r ?? ''}>
              <b>{r}.</b> One word differs after rounds 1 and 2 (a different one each time), then {a.differing[2]} after round
              3, {a.differing[3]} after round 4, and all four after round 5. {v === r ? 'Just as you said.' : v < (r ?? 0) ? 'Slower than you expected: each round only moves a difference through the two multiplies.' : 'Faster than you expected: the multiplies spread a difference into whole words each round.'}{' '}
              The other five rounds are margin: Random123 also publishes 7-round vectors, and a Philox with fewer rounds is a
              different stream.
            </p>
          );
        }}
      </Predict>
      <p>
        The stepper runs the rounds on your inputs, with a second row beside it whose stream is one higher. The presets are
        Random123's three published vectors.
      </p>
      <Simplification>
        <p>
          The stepper's arithmetic is exact 32-bit integer arithmetic, checked against BigInt. The PyTorch MPS variant is our
          transcription of its source (nine rounds, high word first), not run on a device.
        </p>
      </Simplification>
      <PhiloxStepper />
      <p>
        Three things to find. Any (seed, stream, step) is computed directly: type a stream of 10¹⁸ and nothing else runs. A
        one-bit change in the stream reaches all four words within five rounds. And a "Philox" with nine rounds is a
        different stream. Look at round 9 of the all-zero vector: 7028b1c8 d6594c40 4f2051bb 76d6628e. That is what our
        reading of PyTorch's Metal Philox returns for seed 0 (E3): it stops one round early. Switch on the PyTorch MPS reading
        with seed 42 and compare its words with the ten-round ones.
      </p>
      <PhiloxRoundsScene />

      <h2>M5's mapping</h2>
      <p>
        Milestone M5 sets the counter to (step low, step high, stream low, stream high) and the key to (seed low, seed high),
        and turns output word 0 into u. Seed, stream and step are inputs per row. The stream is the request's id, not the
        batch row: under continuous batching a request moves between rows from one step to the next, and its numbers must not
        move with it.
      </p>
      <p>
        By our reading of cuRAND's header, that word is exactly the first <code>curand()</code> after{' '}
        <code>curand_init(seed, stream, 4 × step, &s)</code>. AMD's rocRAND device header initialises the same way by our
        reading, while AMD's compatibility table says its Philox does not reproduce cuRAND's sequence.
      </p>
      <Today checked="2026-11-20 (notes/today/x1-facts.md A3, B1, E1)">
        <p>
          cuRAND: key from the seed, subsequence into counter words z and w, offset / 4 into x and y, state at offset mod 4.
          AMD's table: "Philox 32x4-10 | No" (host API, legacy ordering). Unresolved and unverified on any device.
        </p>
      </Today>
      <Ask q={questionsSX17['x1/X1.7/q-mapping']} />
      <Ask q={questionsSX17['x1/X1.7/q-slot']} />

      <h2>What each backend gives you</h2>
      <p>
        CUDA has cuRAND's device API: <code>curandStatePhilox4_32_10_t</code> and{' '}
        <code>curand_init(seed, subsequence, offset, &state)</code>. Its distribution functions "may use any number of
        unsigned integer values" from the generator, so a reproducible draw starts from raw bits. HIP has hipRAND and rocRAND
        with the same arguments, and HIPIFY maps the names, but the uniform conversion differs (below). Metal has nothing: the
        shading language has no random function.
      </p>
      <Today checked="2026-11-20 (notes/today/x1-facts.md A1, B1, B3, C1 to C4)">
        <p>
          MSL 4.1: no random, noise or hash function. <code>MPSMatrixRandomPhilox</code> is a host-encoded fill kernel;
          MPSGraph's Philox ops do not document rounds, seeding or conversion. MSL has <code>mulhi</code> and{' '}
          <code>ulong</code>; Random123 has run as MSL; PyTorch ships its own Metal Philox.
        </p>
      </Today>
      <p>
        So M5 writes Philox itself, once in the <code>.cu</code> file and once in the <code>.metal</code> file. It forms the
        64-bit product in <code>unsigned long long</code> (<code>ulong</code> on Metal) instead of calling{' '}
        <code>__umulhi</code>. Both are correct; the product form also compiles on the course's emulator and front-end stub,
        which do not declare <code>__umulhi</code>.
      </p>
      <Ask q={questionsSX17['x1/X1.7/q-metal']} />

      <h2>Known answers are the portability test</h2>
      <p>
        Random123 publishes test vectors: for a given counter and key, the four words a correct Philox4x32-10 must produce.
        Three of them are the stepper's presets. A Philox that prints those words on a backend is the same function there, bit
        for bit, so lab X1.3 checks them on every backend before it trusts a single token.
      </p>
      <Today checked="2026-11-20 (notes/today/x1-facts.md A4, E1)">
        <p>
          <code>tests/kat_vectors</code>: counter 0, key 0 gives <code>6627e8d5 e169c58d bc57ac4c 9b00dbd8</code>. Our
          transcription reproduces all three 10-round vectors (E1); so does this page's model, and so do the snippets below.
        </p>
      </Today>

      <h2>From bits to u</h2>
      <p>
        A word of 32 random bits still has to become a float. The vendors disagree on how. <code>curand_uniform</code>{' '}
        computes x·2⁻³² + 2⁻³³ in float, with range (0, 1]. rocRAND, and so <code>hiprand_uniform</code> on AMD, computes
        2⁻³² + x·2⁻³². Both include 1.0. M5 keeps the top 24 bits: <code>(x &gt;&gt; 8) × 2⁻²⁴</code>. That is exact,
        because every value below 2²⁴ is a float and the scale is a power of two, so it is bit-identical on every backend
        under any math mode, with range [0, 1 − 2⁻²⁴].
      </p>
      <Today checked="2026-11-20 (notes/today/x1-facts.md A2, B2, E2)">
        <p>
          cuRAND: "1.0 is included and 0.0 is excluded." rocRAND: <code>ROCRAND_2POW32_INV + (v * ROCRAND_2POW32_INV)</code>.
          The facts pass found 3,903 differences in a million inputs.
        </p>
      </Today>
      <BitsExplorer />
      <Ask q={questionsSX17['x1/X1.7/q-one']} label="Work it out" />
      <Ask q={questionsSX17['x1/X1.7/q-differ']} label="Work it out" />
      <p>
        M5's u can be exactly 0, for 1 in 2²⁴ words. That is why the draw on the{' '}
        <a href={href('x1', 'X1.8')}>next page</a> uses "strictly greater": a token of zero weight at the start of the row
        must never be picked.
      </p>
      <Ask q={questionsSX17['x1/X1.7/q-raw-bits']} />

      <h2>What the frameworks do today</h2>
      <p>
        No two of them share a stream. PyTorch uses mt19937 on the CPU and cuRAND's Philox per thread on CUDA; on MPS it
        runs its own Metal Philox, which by our reading of <code>c10/metal/random.h</code> has 9 rounds and puts the high
        word first. MLX uses a splittable Threefry-2x32, with 20 rounds by our count of its Metal kernel. PyTorch's own documentation says results may differ
        across platforms and between CPU and GPU, even with identical seeds.
      </p>
      <Today checked="2026-11-20 (notes/today/x1-facts.md D5, D6, D7, E3)">
        <p>
          PyTorch MPS, our transcription (not run): seed 0, index 0 gives <code>7028b1c8 …</code> where Philox4x32-10 gives{' '}
          <code>6627e8d5 …</code>. MLX: "a splittable version of Threefry, which is a counter-based PRNG." The 20 rounds are our count of the
          loop in <code>random.metal</code> (D7), five passes of four rounds; MLX does not document the count.
        </p>
      </Today>

      <h2>The code</h2>
      <p>
        The round, the key schedule, M5's mapping and the 24-bit conversion, with Random123's notice. The CUDA file runs
        unchanged on HIP and the emulators; the Metal file is the same function in MSL with <code>mulhi</code> for the high
        half and the wrapping 32-bit product for the low half. Lab X1.3's Metal kernel uses the other form, a{' '}
        <code>ulong</code> product, so you see both; they pass the same known-answer vectors.
      </p>
      {cu && metal ? (
        <Code
          tabs={[
            { label: 'CUDA and HIP', code: cu, status: 'emulator-checked and front-end-checked; unverified on CUDA and HIP hardware' },
            { label: 'Metal', code: metal, status: 'verified on Metal' },
          ]}
          caption={
            <>
              <code>labs/x1/snippets/philox.cu</code> and <code>philox.metal</code>. <code>app/src/sim/philox.test.ts</code> runs
              both, prints the known-answer vectors and 64 rows of u, and compares them with this page's model.
            </>
          }
        />
      ) : (
        <p className="muted">The snippets (<code>labs/x1/snippets/philox.cu</code>, <code>philox.metal</code>) appear here once loaded.</p>
      )}
      <p>
        Lab X1.3 on the <a href={href('x1', 'labs')}>labs page</a> builds this into milestone M5 and checks the known answers
        on every backend first.
      </p>
    </>
  );
}
