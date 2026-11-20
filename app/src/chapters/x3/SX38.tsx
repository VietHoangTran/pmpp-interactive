import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Today } from '../../components/notes';
import { href } from '../../lib/router';
import {
  CHECKPOINTS,
  HEADER_FIELDS,
  LLAMA2C_EPS,
  LLAMA2C_THETA,
  M7_HEAD_DIM,
  SMOL_CONFIG,
  SMOL_HEADER_LEN,
  STORIES_15M,
  archFromHeader,
  bf16ToF32,
  bf16ToF32Bits,
  checkSize,
  checklist,
  encodeHeader,
  fileBytes,
  fitOf,
  floatCount,
  hex,
  hex16,
  hex32,
  layout,
  lengthBytes,
  numel,
  safetensorsBytes,
  shapeOf,
  type Arch,
  type Llama2cHeader,
  type TensorSpec,
} from '../../sim/ckptformat';

// Section X3.8: real checkpoint formats. Every number shown is computed by
// src/sim/ckptformat.ts and pinned in ckptformat.test.ts against notes/today/x3-facts.md
// (D1 to D6, E1, E2; appA A2). The C++ check is labs/x3/snippets/ckpt_check.cpp, which the
// same test compiles and runs on files built from the model.

const MONO: CSSProperties = { fontFamily: 'var(--sans)', fontVariationSettings: "'MONO' 1" };
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const n0 = (x: number) => x.toLocaleString('en-US');
const loaded = (t: string | null) => (t && !t.startsWith('// could not load') ? t : null);

// ---------------------------------------------------------------- decode a header (stepper)

function HeaderDecoder() {
  const h = STORIES_15M;
  const bytes = hex(encodeHeader(h));
  const [k, setK] = useState(0); // field being decoded; 7 = all done
  const [guess, setGuess] = useState('');
  const [shown, setShown] = useState<(number | null)[]>(() => HEADER_FIELDS.map(() => null));
  const [verdicts, setVerdicts] = useState<(boolean | null)[]>(() => HEADER_FIELDS.map(() => null));
  const done = k >= HEADER_FIELDS.length;
  const revealedHere = !done && shown[k] != null;

  const reveal = () => {
    const v = h[HEADER_FIELDS[k]];
    const g = guess.trim() === '' ? null : Number(guess);
    setShown((s) => s.map((x, i) => (i === k ? v : x)));
    setVerdicts((s) => s.map((x, i) => (i === k ? (g == null ? null : g === v) : x)));
  };
  const next = () => {
    setK(k + 1);
    setGuess('');
  };
  const restart = () => {
    setK(0);
    setGuess('');
    setShown(HEADER_FIELDS.map(() => null));
    setVerdicts(HEADER_FIELDS.map(() => null));
  };
  const right = verdicts.filter((v) => v === true).length;
  const sh = shapeOf(h);

  return (
    <Bench
      title="Decode a header"
      task="Read each int32 from its four bytes before you reveal it."
      foot={<>The first 28 bytes of stories15M.bin, rebuilt from the header the facts pass read from the file (x3-facts D2). Each field is a little-endian int32: the first byte is the lowest.</>}
    >
      <div style={{ ...MONO, display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '0.9rem' }} data-part="dump">
        {HEADER_FIELDS.map((f, i) => {
          const cur = i === k;
          return (
            <div key={f} data-field={f} data-shown={shown[i] ?? ''}
              style={{ border: `1px solid ${cur ? 'var(--ink)' : 'var(--rule)'}`, borderRadius: 4, padding: '0.35rem 0.5rem', background: cur ? 'var(--page)' : 'transparent', minWidth: '7.4rem' }}>
              <div style={{ fontSize: 'var(--step--1)', color: 'var(--ink-3)' }}>bytes {4 * i} to {4 * i + 3}</div>
              <div style={{ fontSize: '1.02rem', letterSpacing: '0.04em' }}>{bytes.slice(4 * i, 4 * i + 4).join(' ')}</div>
              <div style={{ fontSize: 'var(--step--1)', color: 'var(--ink-2)' }}>
                {f}
                {shown[i] != null ? <> = <b style={{ color: 'var(--ink)' }}>{n0(shown[i]!)}</b></> : null}
                {verdicts[i] === true ? <span className="t-ok"> ✓</span> : verdicts[i] === false ? <span className="t-fail"> ✗</span> : null}
              </div>
            </div>
          );
        })}
      </div>

      {!done ? (
        <>
          <div className="row" style={{ alignItems: 'flex-end', marginBottom: '0.7rem' }}>
            <label style={FIELD}>
              <span>
                Field {k + 1} of 7, <code>{HEADER_FIELDS[k]}</code>: bytes <span style={MONO}>{bytes.slice(4 * k, 4 * k + 4).join(' ')}</span>
              </span>
              <input type="number" step="1" value={guess} aria-label="your int32" disabled={revealedHere}
                onChange={(e) => setGuess(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && !revealedHere && reveal()} />
            </label>
            {!revealedHere ? (
              <button className="primary" onClick={reveal}>Reveal</button>
            ) : (
              <button className="primary" onClick={next}>{k + 1 < HEADER_FIELDS.length ? 'Next field' : 'Finish'}</button>
            )}
            {k > 0 || revealedHere ? <button onClick={restart}>Restart</button> : null}
          </div>
          <div className="narrate" role="status" data-part="decode-says">
            {revealedHere ? (
              <>
                Reversed, the bytes read 0x{bytes.slice(4 * k, 4 * k + 4).reverse().join('')} = <b>{n0(shown[k]!)}</b>.{' '}
                {verdicts[k] === true ? 'You had it.' : verdicts[k] === false ? `You wrote ${guess}.` : ''}
                {k === 0 && verdicts[k] === false && Number(guess) === 0x20010000 ? ' That is the big-endian reading.' : ''}
              </>
            ) : k === 0 ? (
              <>Little-endian means the bytes are stored lowest first. Reverse them and read one hex number: 20 01 00 00 becomes 0x00000120.</>
            ) : (
              <>Same rule: reverse the four bytes, then read them as one hexadecimal number.</>
            )}
          </div>
        </>
      ) : (
        <div className="narrate" role="status" data-part="decode-done">
          All seven decoded{verdicts.some((v) => v != null) ? `, ${right} of 7 by you` : ''}. vocab_size is positive, so the classifier is the
          embedding. Head size {h.dim} / {h.n_heads} = <b>{sh.headSize}</b>; {h.n_kv_heads} KV heads, so every query head has its own.{' '}
          <button onClick={restart} style={{ marginLeft: '0.4rem' }}>Restart</button>
        </div>
      )}
    </Bench>
  );
}

function SizeReveal({ predicted }: { predicted: number }) {
  const h = STORIES_15M;
  const exp = fileBytes(h);
  return (
    <>
      <p data-reveal={exp}>
        <b>{n0(exp)} bytes</b>: {n0(floatCount(h))} floats times 4, plus the 28-byte header. The Hub lists stories15M.bin at{' '}
        {n0(CHECKPOINTS.find((c) => c.id === 'stories15M')!.hubBytes!)} bytes.{' '}
        {predicted === exp
          ? 'Exactly right.'
          : Math.abs(predicted - exp) === 4 * h.seq_len * (h.dim / h.n_heads)
            ? 'You left out the two RoPE tables: run.c skips them, but they are in the file.'
            : Math.abs(predicted - exp) === 28
              ? 'Off by the header.'
              : Math.abs(predicted - exp) / exp < 0.02
                ? 'Close: compare your terms with the explorer below.'
                : 'Compare your terms with the block table in the explorer below.'}
      </p>
      <p className="muted" style={{ marginBottom: 0 }}>
        By far the largest block is the embedding, {n0(h.vocab_size)} × {h.dim} = {n0(h.vocab_size * h.dim)} floats, about{' '}
        {Math.round((100 * h.vocab_size * h.dim) / floatCount(h))}% of the file.
      </p>
    </>
  );
}

// ---------------------------------------------------------------- checkpoint explorer

type Choice = string; // a checkpoint id or 'custom'

const SHAPE_ROWS = (t: TensorSpec[]) => {
  // Group the 272 tensors by kind: embedding, the nine per-layer tensors, final norm.
  const kinds = new Map<string, { kind: string; shape: number[]; count: number; each: number }>();
  for (const x of t) {
    const kind = x.name.replace(/^model\.layers\.\d+\./, 'model.layers.N.');
    const k = kinds.get(kind);
    if (k) k.count += 1;
    else kinds.set(kind, { kind, shape: x.shape, count: 1, each: numel(x.shape) });
  }
  return [...kinds.values()];
};

function HeaderInputs({ h, set }: { h: Record<string, string>; set: (f: string, v: string) => void }) {
  return (
    <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }} data-part="custom">
      {HEADER_FIELDS.map((f) => (
        <label key={f} style={FIELD}>
          {f}
          <input type="number" step="1" value={h[f]} aria-label={f} style={{ width: '6.6rem' }} onChange={(e) => set(f, e.target.value)} />
        </label>
      ))}
    </div>
  );
}

function FitRow({ arch }: { arch: Arch }) {
  const f = fitOf(arch);
  const hs = Number.isInteger(f.headSize) ? String(f.headSize) : f.headSize.toFixed(2);
  return (
    <p style={{ margin: '0.7rem 0' }} data-part="fit" data-head={f.headSize} data-m7={f.m7 ? 'yes' : 'no'} data-lanes={f.perLane ?? ''}>
      Head size <b>{hs}</b> ({arch.dim} / {arch.heads}); <b>{Number.isInteger(f.qPerKv) ? f.qPerKv : f.qPerKv.toFixed(2)}</b> query head
      {f.qPerKv === 1 ? '' : 's'} per KV head.{' '}
      {f.m7 ? <span className="chip ok">fits M7's HEAD_DIM {M7_HEAD_DIM}</span> : <span className="chip fail">not M7's HEAD_DIM {M7_HEAD_DIM}</span>}{' '}
      {f.perLane != null ? (
        <span className="chip ok">lab 20.4's 32 lanes: {f.perLane} per lane</span>
      ) : (
        <span className="chip fail">not a multiple of 32</span>
      )}
    </p>
  );
}

function Checklist({ arch }: { arch: Arch }) {
  const items = checklist(arch);
  return (
    <div data-part="checklist" style={{ marginTop: '0.9rem' }}>
      <div className="mx-label" style={{ marginBottom: '0.3rem' }}>Before the course's kernels can run it</div>
      <ul style={{ margin: 0, paddingLeft: '1.1rem' }}>
        {items.map((i) => (
          <li key={i.id} data-item={i.id} data-todo={i.todo ? 'yes' : 'no'} style={{ marginBottom: '0.3rem' }}>
            {i.todo ? <span className="chip fail">to do</span> : <span className="chip ok">fine</span>} <b>{i.label}.</b>{' '}
            <span className="muted">{i.detail}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Explorer() {
  const [pick, setPick] = useState<Choice>('stories260K');
  const [custom, setCustom] = useState<Record<string, string>>(() =>
    Object.fromEntries(HEADER_FIELDS.map((f) => [f, String(STORIES_15M[f])])),
  );
  const [download, setDownload] = useState('');
  const ck = CHECKPOINTS.find((c) => c.id === pick);
  const customHeader: Llama2cHeader | null = useMemo(() => {
    if (HEADER_FIELDS.some((f) => custom[f].trim() === '' || !Number.isInteger(Number(custom[f])))) return null;
    return Object.fromEntries(HEADER_FIELDS.map((f) => [f, Number(custom[f])])) as unknown as Llama2cHeader;
  }, [custom]);
  const header = ck ? ck.header ?? null : customHeader;
  const problems = header ? shapeOf(header).problems : ['every field must be an integer'];
  const usable = !!header && problems.length === 0 && header.n_layers > 0 && header.hidden_dim > 0 && header.seq_len > 0 && header.vocab_size !== 0;
  const arch: Arch | null = ck ? ck.arch : usable ? archFromHeader(header!) : null;
  const dl = download.trim() === '' ? null : Number(download);

  return (
    <Bench
      title="Checkpoint explorer"
      task="Pick a checkpoint, or type a header. Which small real checkpoint can M7's decode kernel run as written?"
      foot={
        <>
          Sizes and shapes from x3-facts D2 and D4; the formulas are ours (E1). The two M10 models are the shapes lab X3.2 generates
          (a unit test ties them to its CMake file).
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label="Checkpoint" style={{ flexWrap: 'wrap' }}>
          {CHECKPOINTS.map((c) => (
            <button key={c.id} aria-pressed={pick === c.id} onClick={() => { setPick(c.id); setDownload(''); }}>{c.label}</button>
          ))}
          <button aria-pressed={pick === 'custom'} onClick={() => { setPick('custom'); setDownload(''); }}>your own header</button>
        </span>
      </div>
      {pick === 'custom' ? (
        <HeaderInputs h={custom} set={(f, v) => setCustom((c) => ({ ...c, [f]: v }))} />
      ) : null}

      {ck ? (
        <p className="muted" style={{ marginTop: 0 }} data-part="file">
          {ck.file} · {ck.licence} · {ck.format === 'llama2c' ? 'llama2.c version 0 .bin' : 'safetensors plus config.json'} · facts {ck.facts}
        </p>
      ) : null}

      {pick === 'custom' && !usable ? (
        <p data-part="problem">
          <span className="chip fail">not a usable header</span>{' '}
          {problems.length ? problems.join('; ') : 'n_layers, hidden_dim and seq_len must be positive and vocab_size non-zero'}.
        </p>
      ) : null}

      {(ck?.format === 'llama2c' || (!ck && usable)) && header && usable ? (
        <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 6fr) minmax(0, 5fr)', alignItems: 'start' }}>
          <div style={{ overflowX: 'auto' }}>
            <table className="data" style={{ margin: 0, background: 'var(--page)', width: '100%' }} data-part="blocks">
              <thead>
                <tr><th style={{ paddingLeft: '0.6rem' }}>Block, in file order</th><th className="num">Floats</th></tr>
              </thead>
              <tbody>
                <tr>
                  <td style={{ paddingLeft: '0.6rem' }}>header <span className="muted" style={{ fontSize: 'var(--step--1)' }}>7 × int32</span></td>
                  <td className="num muted">28 bytes</td>
                </tr>
                {layout(header).map((b) => (
                  <tr key={b.name} data-block={b.name}>
                    <td style={{ paddingLeft: '0.6rem' }}>
                      <span style={MONO}>{b.name}</span>{b.skipped ? <span className="muted"> (skipped)</span> : null}{' '}
                      <span className="muted" style={{ fontSize: 'var(--step--1)', whiteSpace: 'nowrap' }}>{b.what}</span>
                    </td>
                    <td className="num">{n0(b.floats)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div>
            <p style={{ marginTop: 0 }} data-part="size" data-floats={floatCount(header)} data-bytes={fileBytes(header)}>
              {n0(floatCount(header))} floats × 4 + 28 = <b>{n0(fileBytes(header))} bytes</b>.
              {ck && ck.hubBytes != null ? (
                fileBytes(header) === ck.hubBytes ? (
                  <> <span className="chip ok">the Hub's size, to the byte</span></>
                ) : (
                  <> <span className="chip fail">the Hub says {n0(ck.hubBytes)}</span></>
                )
              ) : ck ? (
                <> <span className="muted">make_model.py checks the file it writes against the same formula.</span></>
              ) : null}
            </p>
            <label style={FIELD}>
              Your download's size in bytes
              <input type="number" step="1" value={download} aria-label="download size" onChange={(e) => setDownload(e.target.value)} />
            </label>
            {dl != null && Number.isFinite(dl) ? (
              (() => {
                const c = checkSize(header, dl);
                return (
                  <p data-part="download" data-ok={c.ok ? 'yes' : 'no'} style={{ margin: '0.5rem 0 0' }}>
                    {c.ok ? <span className="chip ok">matches the header</span> : <span className="chip fail">does not match</span>}{' '}
                    {c.hint ?? 'Safe to load as a version 0 file.'}
                  </p>
                );
              })()
            ) : null}
            <FitRow arch={arch!} />
          </div>
        </div>
      ) : null}

      {ck?.format === 'safetensors' && ck.st ? (
        <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 6fr) minmax(0, 5fr)', alignItems: 'start' }}>
          <div style={{ overflowX: 'auto' }}>
            <table className="data" style={{ margin: 0, background: 'var(--page)', minWidth: '24rem' }} data-part="tensors">
              <thead>
                <tr><th style={{ paddingLeft: '0.6rem' }}>Tensor (BF16)</th><th>Shape</th><th className="num">Count</th></tr>
              </thead>
              <tbody>
                {SHAPE_ROWS(ck.st.tensors).map((r) => (
                  <tr key={r.kind}>
                    <td style={{ paddingLeft: '0.6rem', ...MONO, fontSize: 'var(--step--1)' }}>{r.kind}</td>
                    <td className="muted" style={{ whiteSpace: 'nowrap' }}>[{r.shape.join(', ')}]</td>
                    <td className="num">{r.count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div>
            {(() => {
              const elems = ck.st.tensors.reduce((s, t) => s + numel(t.shape), 0);
              const bytes = safetensorsBytes(ck.st.headerLen, ck.st.tensors);
              return (
                <p style={{ marginTop: 0 }} data-part="size" data-bytes={bytes} data-tensors={ck.st.tensors.length}>
                  {ck.st.tensors.length} tensors, {n0(elems)} values × 2 bytes, plus 8 + N = 8 + {n0(ck.st.headerLen)} header bytes:{' '}
                  <b>{n0(bytes)} bytes</b>.{' '}
                  {bytes === ck.hubBytes ? <span className="chip ok">the Hub's size, to the byte</span> : <span className="chip fail">the Hub says {n0(ck.hubBytes ?? 0)}</span>}
                </p>
              );
            })()}
            <p style={{ margin: '0 0 0.4rem' }} data-part="first-bytes">
              First bytes: <span style={MONO}>{lengthBytes(SMOL_HEADER_LEN).join(' ')} 7B</span>: N = {n0(SMOL_HEADER_LEN)}, then the
              header's <code>{'{'}</code>.
            </p>
            <p className="muted" style={{ margin: 0 }}>
              config.json: hidden {SMOL_CONFIG.hidden_size}, intermediate {SMOL_CONFIG.intermediate_size}, {SMOL_CONFIG.num_hidden_layers} layers,{' '}
              {SMOL_CONFIG.num_attention_heads} heads over {SMOL_CONFIG.num_key_value_heads} KV heads, vocab {n0(SMOL_CONFIG.vocab_size)}, rope_theta{' '}
              {n0(SMOL_CONFIG.rope_theta)}, eps {SMOL_CONFIG.rms_norm_eps.toExponential(0)}, tied embeddings.
            </p>
            <FitRow arch={ck.arch} />
          </div>
        </div>
      ) : null}

      {arch ? <Checklist arch={arch} /> : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- questions

export const questionsSX38: Record<string, Question> = {
  'x3/X3.8/q-size-260k': {
    kind: 'numeric',
    id: 'x3/X3.8/q-size-260k',
    prompt: (
      <>
        stories260K's header is dim 64, hidden_dim 172, 5 layers, 8 heads, 4 KV heads, vocab_size 512, seq_len 512, so the classifier is
        shared. How many bytes should the file have?
      </>
    ),
    answer: 1056540,
    unit: 'bytes',
    explain: (
      <p>
        Head size 64 / 8 = 8 and kv_dim = 64 × 4 / 8 = 32. Floats: embedding 512 × 64 = 32,768; two norms per layer, 2 × 5 × 64 = 640; wq
        and wo 2 × 5 × 64 × 64 = 40,960; wk and wv 2 × 5 × 32 × 64 = 20,480; w1, w2, w3 3 × 5 × 64 × 172 = 165,120; final norm 64; the two
        RoPE tables 512 × 8 = 4,096. That is 264,128 floats, × 4 + 28 = 1,056,540 bytes, the size the Hub lists.
      </p>
    ),
    source: 'ours (x3-facts E1), from the header in D2',
  },
  'x3/X3.8/q-negative-vocab': {
    kind: 'choice',
    id: 'x3/X3.8/q-negative-vocab',
    prompt: <>A llama2.c header's vocab_size is negative. What does the loader do differently?</>,
    options: [
      { label: 'The classifier is not the embedding: a separate vocab × dim block follows the two skipped RoPE tables.', correct: true },
      { label: 'The file is version 1 or later; run.c refuses it.', why: 'Later versions are a different header. The sign is run.c’s own flag inside version 0.' },
      { label: 'The tokens are stored in reverse order.', why: 'Nothing in run.c reverses anything. The sign carries one bit of information.' },
      { label: 'The weights are float16 instead of float32.', why: 'Version 0 is float32 throughout; run.c maps the floats directly.' },
    ],
    explain: (
      <p>
        run.c reads <code>shared_weights = vocab_size &gt; 0</code> and then uses |vocab_size|. With shared weights the LM head reuses
        token_embedding_table; without them, <code>wcls</code> is the next vocab × dim floats after freq_cis_imag. A loader that forgets
        the sign reads a negative size and fails; one that takes the absolute value but ignores the flag runs the wrong classifier. The
        size check catches the second: the file is vocab × dim × 4 bytes longer than a shared header predicts.
      </p>
    ),
    source: 'x3-facts D1 (run.c lines 142 to 150, memory_map_weights)',
  },
  'x3/X3.8/q-st-n': {
    kind: 'numeric',
    id: 'x3/X3.8/q-st-n',
    prompt: (
      <>
        A safetensors file starts <code>40 77 00 00 00 00 00 00 7B</code>. A tensor's header entry says{' '}
        <code>"data_offsets": [0, 56623104]</code>. At what byte offset in the file does that tensor's data start?
      </>
    ),
    answer: 30536,
    explain: (
      <p>
        The first 8 bytes are N as a little-endian 64-bit integer: 0x7740 = 30,528. The header occupies bytes 8 to 30,535, and the byte
        buffer starts after it. data_offsets are relative to that buffer, not to the file, so the tensor starts at 8 + 30,528 + 0 = 30,536.
        (These are SmolLM2-135M's first bytes; 56,623,104 bytes is its embedding, 49,152 × 576 values × 2.)
      </p>
    ),
    source: 'x3-facts D3 (format), D4 (N and the embedding shape)',
  },
  'x3/X3.8/q-bf16': {
    kind: 'choice',
    id: 'x3/X3.8/q-bf16',
    prompt: <>A BF16 tensor stores the 16 bits 0x3F80. What float32 bit pattern do you write, and is the conversion exact?</>,
    options: [
      { label: `${hex32(bf16ToF32Bits(0x3f80))}, which is ${bf16ToF32(0x3f80)}.0; exact, because bf16 is the top half of a float32.`, correct: true },
      { label: '0x00003F80; exact.', why: 'That puts the bits in the bottom half: the result is a tiny subnormal, not 1.0.' },
      { label: '0x3F800000, but it must be rounded, so it is only approximately 1.0.', why: 'Widening adds sixteen zero mantissa bits. Nothing is lost, so nothing is rounded.' },
      { label: 'It depends on the exponent bias, which bf16 and float32 do not share.', why: 'They share it: bf16 has float32’s 8-bit exponent.' },
    ],
    explain: (
      <p>
        bfloat16 keeps float32's sign bit, its 8 exponent bits and the top 7 mantissa bits. Shifting the 16 stored bits left by 16 gives the
        float32 with those fields and a zero tail, the exact same value. The other direction, float32 to bf16, is the one that rounds.
      </p>
    ),
    source: 'appA-facts A2 (layout); x3-facts D4 (every SmolLM2 tensor is BF16)',
  },
  'x3/X3.8/q-theta': {
    kind: 'choice',
    id: 'x3/X3.8/q-theta',
    prompt: (
      <>
        You run SmolLM2-135M from BOS with llama2.c's hard-coded RoPE base 10,000 instead of the config's 100,000. The first generated token
        matches the reference. Is the wrong base harmless?
      </>
    ),
    options: [
      {
        label: 'No. At position 0 every rotation angle is 0 whatever the base, so the first token cannot reveal it; later positions rotate by different angles and the tokens drift.',
        correct: true,
      },
      { label: 'Yes: RoPE angles are small, so a factor of 10 in the base changes little.', why: 'Low-frequency pairs change by a lot: the base sets how fast every pair after the first rotates.' },
      { label: 'Yes, as long as epsilon is right: RMSNorm absorbs the difference.', why: 'RMSNorm rescales a vector; it cannot undo a rotation of Q against K.' },
      { label: 'No, and the first token should already differ; a match means the base is not used.', why: 'At position 0, pos × frequency is 0 for every pair and every base.' },
    ],
    explain: (
      <p>
        The angle for a pair is position × frequency, and the frequency is base<sup>−2i/d</sup>. At position 0 the product is 0 for any
        base, so Q and K are not rotated at all and a wrong base is invisible. From position 1 on, every pair except the first rotates by
        a different angle, attention scores change, and a greedy run drifts. A test that checks only the first token, or only short
        prompts, misses this; teacher-forced logits over a long sequence do not (section <a href={href('x3', 'X3.9')}>X3.9</a>).
      </p>
    ),
    source: 'ours; the constants from x3-facts D1 (run.c) and D4 (config.json)',
  },
  'x3/X3.8/q-head64': {
    kind: 'choice',
    id: 'x3/X3.8/q-head64',
    prompt: <>Which of the three small checkpoints could lab 20.4's decode kernel serve by changing only HEAD_DIM?</>,
    options: [
      { label: 'SmolLM2-135M: head size 64 = 2 × 32, so 2 values per lane.', correct: true },
      { label: 'stories15M: head size 48.', why: '48 is not a multiple of 32. HEAD_DIM / 32 rounds down to 1, so each lane takes one dimension and 16 of the 48 are never read.' },
      { label: 'stories260K: head size 8.', why: 'Fewer dimensions than lanes: HEAD_DIM / 32 is 0.' },
      { label: 'All three: HEAD_DIM is just a constant.', why: 'The kernel also divides the head over 32 lanes, which constrains the constant.' },
    ],
    explain: (
      <p>
        Lab 20.4 gives each lane HEAD_DIM / 32 dimensions, interleaved (lane L holds L, L + 32, ...). That needs a multiple of 32. Only
        SmolLM2's 64 qualifies, and its 9 query heads over 3 KV heads fit the kernel's GQA rule (h_q a multiple of h_kv). Our arithmetic from
        the lab's source; not run on a device with this checkpoint.
      </p>
    ),
    source: 'x3-facts E2; labs/ch20/lab4_decode/solution/kernel.cu (D_PER_LANE, contract); unverified on a device',
  },
  'x3/X3.8/q-gguf': {
    kind: 'choice',
    id: 'x3/X3.8/q-gguf',
    prompt: <>llama.cpp loads GGUF. Why does M10's real-checkpoint path not start there?</>,
    options: [
      {
        label: 'GGUF has typed metadata, 32-byte default alignment and usually quantised tensors (M9’s subject); llama2.c .bin and safetensors need far less parsing.',
        correct: true,
      },
      { label: 'GGUF is undocumented.', why: 'It has a written spec, ggml/docs/gguf.md, version 3.' },
      { label: 'GGUF files are big-endian.', why: 'The spec says models are little-endian by default.' },
      { label: 'GGUF cannot hold a Llama-architecture model.', why: 'It is the format llama.cpp runs Llama models from.' },
    ],
    explain: (
      <p>
        Nothing is wrong with GGUF; it does more than M10 needs. A version 0 .bin is seven integers and float32. safetensors is a length,
        JSON and row-major bytes. GGUF adds a typed key-value section, tensor offsets padded to an alignment (32 unless the file says
        otherwise), and, in most published files, block-quantised types that M10's FP32 kernels cannot read without M9's dequantisation.
      </p>
    ),
    source: 'x3-facts D5 (ggml/docs/gguf.md)',
  },
  'x3/X3.8/q-export-gqa': {
    kind: 'choice',
    id: 'x3/X3.8/q-export-gqa',
    prompt: <>You convert SmolLM2-135M to a .bin with llama2.c's export.py. What goes wrong?</>,
    options: [
      {
        label: 'permute_reverse reshapes k_proj (192 × 576) as if it had n_heads × head_size = 576 rows, so it fails or mis-permutes; the K permutation must use the KV head count, as llama.cpp’s does.',
        correct: true,
      },
      { label: 'Nothing: export.py reads config.json and handles any Llama.', why: 'Its permute uses n_heads and dim by default, also for K.' },
      { label: 'It writes bf16 values into a float32 file.', why: 'That is not the failure the source shows; the shape of K is.' },
      { label: 'It drops the tied embedding.', why: 'A missing lm_head is the shared-classifier case the format already has.' },
    ],
    explain: (
      <p>
        <code>permute_reverse(w, n_heads=config.n_heads, dim1=config.dim, dim2=config.dim)</code> views w as (n_heads, 2, dim / n_heads /
        2, dim). With 3 KV heads, k_proj has 3 × 64 = 192 rows, not 576, so the view does not match. llama.cpp's converter switches to
        n_head_kv for K. This is our reading of the source; the facts pass did not run it.
      </p>
    ),
    source: 'x3-facts D6 (export.py lines 468 to 475), B6 (llama.cpp permute); by reading, not run',
  },
};

// ---------------------------------------------------------------- the section

export function SX38() {
  const cpp = loaded(useLabFile('x3/snippets/ckpt_check.cpp'));
  return (
    <>
      <p>
        Every model in this course so far has been weights you generated. A real checkpoint is a file someone else wrote: a format you
        must parse exactly, and conventions that are not in the file at all. This page reads two formats by hand, checks a download before
        trusting it, and lists what stands between a Hugging Face Llama and the course's kernels.
      </p>
      <Beyond>
        <p>
          Module X3 is ours. Formats, files and sizes are from notes/today/x3-facts.md (D1 to D6, E1, E2), read at pinned commits on
          2026-11-20.
        </p>
      </Beyond>
      <p>
        You already use this step every day. A Hugging Face checkpoint is a <code>config.json</code> beside a{' '}
        <code>model.safetensors</code>; <code>from_pretrained</code> turns the two into tensors before any kernel runs. Here you do that
        step yourself, for the simplest real format first.
      </p>

      <h2>llama2.c's .bin: seven integers, then floats</h2>
      <p>
        llama2.c's <code>run.c</code> is a whole Llama decoder step in plain C, and its file format is as small as a format can be. The
        first 28 bytes are seven int32: <code>dim</code>, <code>hidden_dim</code>, <code>n_layers</code>, <code>n_heads</code>,{' '}
        <code>n_kv_heads</code>, <code>vocab_size</code>, <code>seq_len</code>. Then come float32 weights in a fixed order: the token
        embedding, the attention RMSNorm weights, <code>wq</code>, <code>wk</code>, <code>wv</code>, <code>wo</code>, the feed-forward
        norms, <code>w1</code>, <code>w2</code>, <code>w3</code>, the final norm, two old RoPE tables that run.c skips, and a classifier
        only if the header says so. The sign of <code>vocab_size</code> says so: negative means the classifier is its own matrix.
      </p>
      <p>
        Its matmul is <code>W (d, n) @ x</code> with <code>w[i * n + j]</code>: row-major, (out, in), the <code>nn.Linear</code> layout and
        this project's convention. Your GEMV reads these weights without a transpose.
      </p>
      <Today checked="2026-11-20 (notes/today/x3-facts.md D1)">
        <p>
          Not in the header, hard-coded in run.c: RMSNorm epsilon {LLAMA2C_EPS.toExponential(0)} and RoPE base {n0(LLAMA2C_THETA)}. Greedy keeps the first
          maximum; generation starts from BOS = 1. run.c reads only this "version 0" layout.
        </p>
      </Today>
      <HeaderDecoder />
      <Predict
        id="x3/X3.8/p-size-15m"
        prompt={<>From the seven values you just decoded and the block order above, how many bytes long is stories15M.bin?</>}
        unit="bytes"
      >
        {(v) => <SizeReveal predicted={v} />}
      </Predict>
      <Ask q={questionsSX38['x3/X3.8/q-negative-vocab']} />

      <h2>Check a download by its size</h2>
      <p>
        Two small models in this format are on the Hugging Face Hub under the MIT licence, in <code>karpathy/tinyllamas</code>:
        stories260K (1,056,540 bytes: dim 64, 5 layers, 8 heads over 4 KV heads, a 512-token vocabulary) and stories15M (60,816,028 bytes).
        Both sizes follow exactly from their headers. Count the floats block by block, multiply by 4, add 28. The K and V projections are{' '}
        <code>kv_dim = dim × n_kv_heads / n_heads</code> rows tall, and the two skipped RoPE tables are{' '}
        <code>seq_len × head_size / 2</code> floats each.
      </p>
      <Ask q={questionsSX38['x3/X3.8/q-size-260k']} label="Work it out" />
      <p>
        So a loader can check a file before it maps a single weight: an interrupted download, a header read with the wrong byte order, a
        flipped classifier flag or a file in another layout all fail in one comparison instead of as nonsense logits. Here is that check in
        C++.
      </p>
      {cpp ? (
        <Code
          tabs={[{ label: 'C++ (host)', code: cpp, status: 'host code, compiled and run by app/src/sim/ckptformat.test.ts on files built from the model' }]}
          caption={
            <>
              labs/x3/snippets/ckpt_check.cpp. The test writes files with the published headers at their predicted length (and one 4,096
              bytes short), runs this program on each and compares its output with the explorer's model.
            </>
          }
        />
      ) : null}

      <h2>safetensors: a length, JSON, then bytes</h2>
      <p>
        Hugging Face checkpoints use safetensors. The first 8 bytes are N, an unsigned little-endian 64-bit integer. The next N bytes are a
        UTF-8 JSON header that must start with <code>{'{'}</code>, mapping each tensor name to its <code>dtype</code>,{' '}
        <code>shape</code> and <code>data_offsets</code> <code>[begin, end]</code>. Everything after that is the byte buffer: row-major,
        little-endian. A key <code>__metadata__</code> may hold free-form strings.
      </p>
      <p>
        Two details trip loaders. The offsets are relative to the start of the byte buffer, so a tensor starts at 8 + N + begin in the file.
        And the format does not check the values: NaN and infinity may be in the file, so check for them after converting, not before.
        Python's standard library is enough to read it: a length, <code>json.loads</code> and slices.
      </p>
      <Ask q={questionsSX38['x3/X3.8/q-st-n']} label="Work it out" />

      <h2>A Hugging Face Llama: SmolLM2-135M</h2>
      <p>
        <code>HuggingFaceTB/SmolLM2-135M</code> (Apache-2.0) is a <code>LlamaForCausalLM</code>: hidden size 576, intermediate 1536, 30
        layers, 9 query heads over 3 KV heads, so head size 64. Its <code>model.safetensors</code> holds 272 tensors, every one BF16. There
        is no <code>lm_head.weight</code>: the embeddings are tied, the same idea as llama2.c's shared classifier. The architecture is the
        step run.c computes (RMSNorm, grouped-query attention with RoPE, SwiGLU), but four things must change before your kernels can run
        it.
      </p>
      <ol>
        <li>
          <b>bf16 to float32.</b> Put the 16 stored bits in the top half of a 32-bit word. bfloat16 is the top half of a float32, so this is
          exact: {hex16(0x3f80)} becomes {hex32(bf16ToF32Bits(0x3f80))}, which is {bf16ToF32(0x3f80)}.
        </li>
        <li>
          <b>The RoPE pairing.</b> Hugging Face rotates element i with element i + head_size / 2; run.c rotates consecutive pairs (section{' '}
          <a href={href('x3', 'X3.7')}>X3.7</a>). Either use the half-split pairing, or permute the rows of Q and K as llama.cpp's converter
          does, with the KV head count for K.
        </li>
        <li>
          <b>Grouped-query attention, 9 over 3.</b> Query head h reads KV head h / 3, and k_proj and v_proj are 192 × 576, not 576 × 576.
        </li>
        <li>
          <b>The constants.</b> RoPE base 100,000 and epsilon 1e-5 come from <code>config.json</code>, not from llama2.c's 10,000.
        </li>
      </ol>
      <Today checked="2026-11-20 (notes/today/x3-facts.md D4, B6; appA-facts A2)">
        <p>
          config.json says <code>"rope_interleaved": false</code> and <code>"tie_word_embeddings": true</code>. transformers moves an old
          top-level <code>rope_theta</code> into <code>rope_parameters</code> on load, with 10,000 as the default when none is given.
        </p>
      </Today>
      <Ask q={questionsSX38['x3/X3.8/q-bf16']} />
      <Ask q={questionsSX38['x3/X3.8/q-theta']} />
      <p>
        The explorer puts the three checkpoints side by side. The safetensors size works the same way as the .bin's: values × 2 bytes, plus
        8 + N, and with SmolLM2's N of 30,528 it lands on the Hub's size to the byte. Then look at the head sizes.
      </p>
      <Simplification>
        <p>
          The explorer computes sizes and shapes; it reads no real file and runs no model. It knows the header fields and tensor shapes the
          facts pass recorded, and nothing about the weights' values or the tokenizer: M10 compares token ids, so it needs none.
        </p>
      </Simplification>
      <Explorer />
      <Ask q={questionsSX38['x3/X3.8/q-head64']} />

      <h2>Two traps</h2>
      <p>
        <b>GGUF is not the place to start.</b> It is what llama.cpp loads, and its spec is clear, but it brings typed metadata, tensor data
        aligned to 32 bytes unless the file says otherwise, and, in most published files, quantised tensor types. Those are M9's subject;
        M10 runs FP32.
      </p>
      <Today checked="2026-11-20 (notes/today/x3-facts.md D5)">
        <p>
          ggml/docs/gguf.md: magic <code>GGUF</code>, version 3, tensor offsets relative to the data section and a multiple of{' '}
          <code>general.alignment</code>, 32 if unset; little-endian by default.
        </p>
      </Today>
      <Ask q={questionsSX38['x3/X3.8/q-gguf']} />
      <p>
        <b>llama2.c's own Hugging Face import does not handle grouped-query K.</b> Its <code>export.py</code> undoes Hugging Face's RoPE
        permutation with <code>n_heads</code> and <code>dim</code> for both Q and K. A GQA model's K projection is smaller than that, so the
        reshape cannot match. llama.cpp's converter uses the KV head count for K. This is a reading of the source, not a run.
      </p>
      <Ask q={questionsSX38['x3/X3.8/q-export-gqa']} />

      <h2>What this means for M10</h2>
      <p>
        Milestone M10 runs a Llama-style model end to end on the course's kernels. Its model is synthetic, generated in the repository, and
        written in llama2.c's version 0 format, so the same loader reads a stories file with no change. Its head size is 128 by
        construction: the lab's generator requires dim = heads × 128. What stops a real file is exactly that: none of the three small
        checkpoints has M7's 128, so the decode kernel needs HEAD_DIM as a parameter, or one build per model, and the lane layout limits
        which values work.
      </p>
      <p>
        Nothing in the repository downloads a checkpoint. Lab X3.2's README, on the <a href={href('x3', 'labs')}>labs page</a>, describes
        the manual path for a file you download yourself. In outline: fetch stories260K, check its size with the program above, point the
        loader at it, and compare token ids with run.c's greedy run (<code>-t 0.0</code>). That path is unverified end to end. The next
        section is about making that comparison mean something.
      </p>
    </>
  );
}
