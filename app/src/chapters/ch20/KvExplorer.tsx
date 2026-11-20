import { useMemo, useState } from 'react';
import { Bench } from '../../components/notes';
import { benchRows, num, realBackends, useReports, type Reports } from '../../lib/reports';
import {
  ELEMENT_SIZES,
  PRESETS,
  aiMLAPrinted,
  asVariant,
  bytesPerToken,
  bytesPerTokenPrinted,
  cacheTotal,
  decodeAI,
  maxBatch,
  presetByKey,
  stepFloorSeconds,
  toFlopPerByte,
  valuesPerTokenLayer,
  variantOf,
  type AttnShape,
  type Variant,
} from '../../sim/kvmath';

// The KV-cache explorer of sections 20.6 (mode "size": one model, a batch, your device's
// memory) and 20.7 (mode "compare": MHA, GQA, MQA and MLA side by side). Every number is
// computed by src/sim/kvmath.ts; presets come from the book (p. 506) and the facts file
// (E2, E5, E7) only.

/** Bytes in binary units, three significant digits. */
export function fmtBytes(b: number): string {
  if (!isFinite(b)) return 'n/a';
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let v = b;
  let u = 0;
  while (Math.abs(v) >= 1024 && u < units.length - 1) {
    v /= 1024;
    u++;
  }
  const s = u === 0 ? Math.round(v).toLocaleString('en-US') : Number(v.toPrecision(3)).toLocaleString('en-US');
  return `${s} ${units[u]}`;
}

const fmtNum = (v: number, digits = 3) => (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString('en-US') : Number(v.toPrecision(digits)).toString());

export interface DeviceMem {
  key: string;
  label: string;
  bytes: number;
  field: string;
  /** Lab 5.1's measured bandwidth on this backend, bytes per second, if any. */
  bw?: number;
}

/** Devices with a memory figure in the reports: totalGlobalMem on CUDA and HIP, recommendedMaxWorkingSetSize on Metal. */
export function deviceMemories(reports: Reports): DeviceMem[] {
  return realBackends(reports).flatMap((b) => {
    const d = reports[b].device ?? {};
    const field = b === 'metal' ? 'recommendedMaxWorkingSetSize' : 'totalGlobalMem';
    const bytes = Number(d[field]);
    if (!(bytes > 0)) return [];
    const roof = benchRows(reports[b], 'ch05/lab1_roofline').find((r) => r.name === 'roofline');
    const gbs = num(roof?.bandwidth_gb_per_s);
    return [{ key: b, label: `${String(d.name ?? b)} (${b})`, bytes, field, bw: gbs ? gbs * 1e9 : undefined }];
  });
}

interface Custom {
  l: number;
  hq: number;
  hkv: number;
  d: number;
  mla: boolean;
  dc: number;
  dr: number;
}

function NumField({ label, value, onChange, min = 1, step = 1, width = '6rem' }: { label: string; value: number; onChange: (v: number) => void; min?: number; step?: number; width?: string }) {
  return (
    <label className="field">
      {label}
      <input type="number" min={min} step={step} value={value} style={{ width }}
        onChange={(e) => { const v = Number(e.target.value); if (isFinite(v) && v >= min) onChange(v); }} />
    </label>
  );
}

function ElementSelect({ label, value, onChange }: { label: string; value: string; onChange: (k: string) => void }) {
  return (
    <label className="field" style={FIT}>
      {label}
      <select style={FIT} value={value} onChange={(e) => onChange(e.target.value)} aria-label={label}>
        {ELEMENT_SIZES.map((e) => <option key={e.key} value={e.key}>{e.label}</option>)}
      </select>
    </label>
  );
}

/** Lets a long select shrink with the bench on a narrow screen. */
const FIT = { maxWidth: '100%', minWidth: 0 } as const;

const divisors = (n: number) => Array.from({ length: n }, (_, i) => i + 1).filter((k) => n % k === 0);

// ------------------------------------------------------------------ size mode

function SizeMode() {
  const { reports } = useReports();
  const devices = useMemo(() => deviceMemories(reports), [reports]);
  const [key, setKey] = useState('llama2-7b');
  const [custom, setCustom] = useState<Custom>({ l: 32, hq: 32, hkv: 8, d: 128, mla: false, dc: 512, dr: 64 });
  const preset = presetByKey(key);
  const shape: AttnShape = preset
    ? preset.shape
    : { l: custom.l, hq: custom.hq, hkv: custom.mla ? custom.hq : Math.min(custom.hkv, custom.hq), d: custom.d, mla: custom.mla ? { dc: custom.dc, dr: custom.dr } : undefined };
  const [N, setN] = useState(4096);
  const [b, setB] = useState(1);
  const [elem, setElem] = useState('f16');
  const [params, setParams] = useState<number | null>(7);
  const [welem, setWelem] = useState('f16');
  const [dev, setDev] = useState<string>('');
  const [customGiB, setCustomGiB] = useState<number | null>(null);

  const choose = (k: string) => {
    setKey(k);
    const p = presetByKey(k);
    if (p) {
      setParams(p.params ? p.params / 1e9 : null);
      if (p.context) setN(p.context);
    }
  };

  const p = ELEMENT_SIZES.find((e) => e.key === elem)!.bytes;
  const wp = ELEMENT_SIZES.find((e) => e.key === welem)!.bytes;
  const device = devices.find((d) => d.key === dev) ?? (dev === '' && devices.length ? devices[0] : undefined);
  const useCustom = dev === 'custom' || !device;
  const memBytes = useCustom ? (customGiB != null ? customGiB * 2 ** 30 : null) : device.bytes;
  const weights = params != null ? params * 1e9 * wp : 0;
  const perLayer = valuesPerTokenLayer(shape) * p;
  const perToken = bytesPerToken(shape, p);
  const printed = bytesPerTokenPrinted(shape, p);
  const total = cacheTotal(shape, b, N, p);
  const fits = memBytes != null ? maxBatch(memBytes, weights, shape, N, p) : null;
  const ai = decodeAI(shape, N);
  const variant = variantOf(shape);
  const bw = useCustom ? undefined : device.bw;
  const floor = bw ? stepFloorSeconds(weights, total, bw) : null;
  const over = memBytes != null && weights + total > memBytes;

  return (
    <Bench
      title="Size the cache"
      task="Pick a model, then find the largest batch that fits beside its weights on your device."
      foot={
        <>
          Cache sizes are the book’s Eqs. 20.7 and 20.8 with the corrections of the facts file: K/V heads for GQA and MQA,
          and MLA’s own latent width. Units are binary (1 GiB = 2³⁰ bytes), as the book’s “GB” are. Device memory is what
          your lab runs reported, not a vendor figure.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field" style={FIT}>
          Model
          <select style={FIT} value={key} onChange={(e) => choose(e.target.value)} aria-label="Model">
            {PRESETS.map((x) => <option key={x.key} value={x.key}>{x.label}</option>)}
            <option value="custom">Custom shape</option>
          </select>
        </label>
        {preset ? (
          <span className="muted" data-shape="">
            {shape.l} layers, {shape.hq} query heads of {shape.d}, {variant === 'MLA' ? `MLA latent ${shape.mla!.dc} + ${shape.mla!.dr}` : `${shape.hkv} K/V ${shape.hkv === 1 ? 'head' : 'heads'}`}
            {preset.context ? `, context ${preset.context.toLocaleString('en-US')}` : ''}
          </span>
        ) : null}
      </div>
      {preset ? (
        <p className="muted" style={{ marginTop: 0 }}>
          Source: {preset.source}.{preset.note ? ' ' + preset.note : ''}
        </p>
      ) : (
        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <NumField label="layers l" value={custom.l} onChange={(v) => setCustom({ ...custom, l: v })} />
          <NumField label="query heads h_q" value={custom.hq} onChange={(v) => setCustom({ ...custom, hq: v })} />
          {!custom.mla ? <NumField label="K/V heads h_kv" value={custom.hkv} onChange={(v) => setCustom({ ...custom, hkv: v })} /> : null}
          <NumField label="head width d" value={custom.d} onChange={(v) => setCustom({ ...custom, d: v })} />
          <label className="check">
            <input type="checkbox" checked={custom.mla} onChange={(e) => setCustom({ ...custom, mla: e.target.checked })} /> MLA latent
          </label>
          {custom.mla ? (
            <>
              <NumField label="d_c" value={custom.dc} onChange={(v) => setCustom({ ...custom, dc: v })} />
              <NumField label="d_rope" value={custom.dr} min={0} onChange={(v) => setCustom({ ...custom, dr: v })} />
            </>
          ) : null}
        </div>
      )}
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Tokens per sequence N = {N.toLocaleString('en-US')}
          <input type="range" min="0" max="17" step="0.25" value={Math.log2(N)} aria-label="Tokens per sequence"
            onChange={(e) => setN(Math.max(1, Math.round(2 ** Number(e.target.value))))} />
        </label>
        {key === 'gpt3' ? (
          <span className="seg" role="group" aria-label="GPT-3 sequence length">
            <button aria-pressed={N === 2048} onClick={() => setN(2048)}>its window, 2,048</button>
            <button aria-pressed={N === 4096} onClick={() => setN(4096)}>the book’s 4,096</button>
          </span>
        ) : null}
        <label className="field">
          Batch b = {b}
          <input type="range" min="0" max="10" step="1" value={Math.log2(b)} aria-label="Batch"
            onChange={(e) => setB(2 ** Number(e.target.value))} />
        </label>
        <ElementSelect label="Cache element" value={elem} onChange={setElem} />
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          Weights, billions of parameters
          <input type="number" min="0" step="0.1" value={params ?? ''} placeholder="not in our facts" aria-label="Parameters in billions" style={{ width: '9rem' }}
            onChange={(e) => setParams(e.target.value === '' ? null : Math.max(0, Number(e.target.value)))} />
        </label>
        <ElementSelect label="Weight element" value={welem} onChange={setWelem} />
        <label className="field" style={FIT}>
          Memory to fit in
          <select style={FIT} value={useCustom ? 'custom' : device.key} onChange={(e) => setDev(e.target.value)} aria-label="Device memory">
            {devices.map((d) => <option key={d.key} value={d.key}>{d.label}: {fmtBytes(d.bytes)}</option>)}
            <option value="custom">A size you type</option>
          </select>
        </label>
        {useCustom ? (
          <label className="field">
            GiB
            <input type="number" min="1" step="1" value={customGiB ?? ''} placeholder="your device's" style={{ width: '8rem' }}
              onChange={(e) => { const v = Number(e.target.value); setCustomGiB(e.target.value === '' || !isFinite(v) || v <= 0 ? null : v); }} />
          </label>
        ) : null}
      </div>

      <table className="data" style={{ background: 'var(--page)', margin: 0 }} data-kv-size="">
        <tbody>
          <tr>
            <td>Cache per token, one layer</td>
            <td className="num" data-per-layer={perLayer}>{fmtBytes(perLayer)}</td>
            <td className="muted">{variant === 'MLA' ? `(d_c + d_rope) × p` : `2 × ${shape.hkv} K/V heads × ${shape.d} × p`}</td>
          </tr>
          <tr>
            <td>Cache per token, all {shape.l} layers</td>
            <td className="num" data-per-token={perToken}>{fmtBytes(perToken)}</td>
            <td className="muted">
              {printed !== perToken ? <>the book’s printed equation gives {fmtBytes(printed)}</> : 'the book’s equation gives the same'}
            </td>
          </tr>
          <tr>
            <td>Cache for b × N = {(b * N).toLocaleString('en-US')} tokens</td>
            <td className="num" data-total={total}>{fmtBytes(total)}</td>
            <td className="muted">Eq. 20.8: b × N × per-token bytes</td>
          </tr>
          <tr>
            <td>Weights</td>
            <td className="num" data-weights={weights}>{params != null ? fmtBytes(weights) : 'not counted'}</td>
            <td className="muted">
              {params != null
                ? `${params}B parameters × ${fmtNum(wp)} bytes = ${fmtNum(weights / 1e9)} GB in decimal units${preset?.paramsNote ? '; ' + preset.paramsNote : ''}`
                : 'type a parameter count to include them'}
            </td>
          </tr>
          <tr>
            <td>Weights + cache</td>
            <td className="num">{fmtBytes(weights + total)}</td>
            <td>
              {memBytes == null ? (
                <span className="muted" data-fits="">type your device&rsquo;s memory to compare</span>
              ) : over ? (
                <span className="chip fail" data-fits="no">does not fit in {fmtBytes(memBytes)}</span>
              ) : (
                <span className="chip ok" data-fits="yes">fits in {fmtBytes(memBytes)}</span>
              )}
              {useCustom && memBytes != null && !devices.length ? <span className="muted"> (a size you typed: no device has reported its memory yet)</span> : null}
            </td>
          </tr>
          <tr>
            <td>Largest batch of N-token sequences that fits</td>
            <td className="num" data-max-batch={fits ?? ''}>{fits != null ? fits.toLocaleString('en-US') : '–'}</td>
            <td className="muted">{fits == null ? 'needs a memory size' : fits === 0 ? 'not even one: the weights leave too little, or one sequence is too long' : 'memory minus weights, divided by one sequence’s cache'}</td>
          </tr>
          <tr>
            <td>Decode attention intensity</td>
            <td className="num" data-ai={ai.toFixed(3)}>{fmtNum(ai)}</td>
            <td className="muted">multiply-adds per element read ({variant}, [13]’s formula) = {fmtNum(toFlopPerByte(ai, p))} FLOP/B at this cache element</td>
          </tr>
          <tr>
            <td>Floor on one decode step (ours)</td>
            <td className="num" data-floor={floor ?? ''}>{floor != null ? `${fmtNum(floor * 1000)} ms` : '–'}</td>
            <td className="muted">
              {floor != null
                ? <>every weight and cached byte read once at lab 5.1’s {fmtNum(bw! / 1e9)} GB/s: at most {fmtNum(b / floor)} tokens per second for this batch</>
                : 'needs lab 5.1’s bandwidth on the chosen device'}
            </td>
          </tr>
        </tbody>
      </table>
      <MemoryBar weights={weights} cache={total} mem={memBytes} />
      {!useCustom ? (
        <p className="muted" style={{ marginBottom: 0 }}>
          {device.label}: <code>{device.field}</code> = {fmtBytes(device.bytes)}, from your reports.
        </p>
      ) : null}
    </Bench>
  );
}

function MemoryBar({ weights, cache, mem }: { weights: number; cache: number; mem: number | null }) {
  const total = Math.max(mem ?? 0, weights + cache) || 1;
  const s = 100 / total;
  return (
    <div style={{ marginTop: '0.8rem' }} aria-label="Memory use" data-memory-bar="">
      <div style={{ position: 'relative', height: '1.1rem', background: 'var(--cell)', border: '1px solid var(--rule)', borderRadius: 2 }}>
        <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: `${weights * s}%`, background: 'var(--global-wash)', borderRight: '1px solid var(--global)' }} />
        <div style={{ position: 'absolute', left: `${weights * s}%`, top: 0, bottom: 0, width: `${cache * s}%`, background: mem != null && weights + cache > mem ? 'var(--fail)' : 'var(--global)' }} />
        {mem != null ? <div style={{ position: 'absolute', left: `${mem * s}%`, top: -4, bottom: -4, borderLeft: '2px solid var(--ink)' }} /> : null}
      </div>
      <div className="legend" style={{ marginTop: '0.3rem', fontSize: '0.8rem' }}>
        <span><i style={{ background: 'var(--global-wash)', border: '1px solid var(--global)' }} />weights</span>
        <span><i style={{ background: 'var(--global)' }} />KV cache</span>
        {mem != null ? <span>black line: the memory you chose</span> : null}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ compare mode

const ORDER: Variant[] = ['MHA', 'GQA', 'MQA', 'MLA'];

function CompareMode() {
  const [key, setKey] = useState('deepseek-v2');
  const preset = presetByKey(key)!;
  const base = preset.shape;
  const choices = divisors(base.hq).filter((k) => k > 1 && k < base.hq);
  const [hkvPick, setHkv] = useState(8);
  const hkv = choices.includes(hkvPick) ? hkvPick : choices[Math.floor(choices.length / 2)] ?? 1;
  const [N, setN] = useState(4096);
  const [elem, setElem] = useState('f16');
  const p = ELEMENT_SIZES.find((e) => e.key === elem)!.bytes;
  const latent = base.mla ?? { dc: 4 * base.d, dr: base.d / 2 };
  const rows = ORDER.map((v) => {
    const s = asVariant(base, v, hkv, latent);
    const ai = decodeAI(s, N);
    return {
      v,
      s,
      bytes: bytesPerToken(s, p),
      printed: bytesPerTokenPrinted(s, p),
      ai,
      aiPrinted: v === 'MLA' ? aiMLAPrinted(N, s.hq) : ai,
    };
  });
  const maxBytes = Math.max(...rows.map((r) => r.bytes));
  const mha = rows[0].bytes;
  return (
    <Bench
      title="Four ways to cache K and V"
      task="Pick a model and compare what each variant caches per token and what its decode step does per byte."
      foot={
        <>
          Bars are the corrected cache per token, all layers. “As printed” is the book’s Eq. 20.8, 20.12, 20.10 or 20.14 and,
          for MLA, Eq. 20.15. Intensity is [13]’s Table 1 in multiply-adds per element, then FLOP per byte at the chosen
          element size. {preset.shape.mla ? '' : 'For a model without MLA, the latent uses DeepSeek-V2’s proportions (d_c = 4d, d_rope = d/2; facts E5): our choice, for comparison only.'}
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field" style={FIT}>
          Model
          <select style={FIT} value={key} onChange={(e) => setKey(e.target.value)} aria-label="Model to compare">
            {PRESETS.filter((x) => x.key !== 'palm2-printed').map((x) => <option key={x.key} value={x.key}>{x.label}</option>)}
          </select>
        </label>
        <label className="field" style={FIT}>
          GQA K/V heads
          <select style={FIT} value={hkv} onChange={(e) => setHkv(Number(e.target.value))} aria-label="GQA K/V heads">
            {choices.map((k) => <option key={k} value={k}>{k} (g_q = {base.hq / k})</option>)}
          </select>
        </label>
        <label className="field">
          Cached tokens N = {N.toLocaleString('en-US')}
          <input type="range" min="6" max="17" step="0.5" value={Math.log2(N)} aria-label="Cached tokens"
            onChange={(e) => setN(Math.round(2 ** Number(e.target.value)))} />
        </label>
        <ElementSelect label="Element" value={elem} onChange={setElem} />
      </div>
      <table className="data" style={{ background: 'var(--page)', margin: 0 }} data-kv-compare="">
        <thead>
          <tr>
            <th>Variant</th>
            <th>What is cached per token per layer</th>
            <th className="num">Per token</th>
            <th style={{ minWidth: '9rem' }} />
            <th className="num">As printed</th>
            <th className="num">Intensity</th>
            <th className="num">As printed</th>
            <th className="num">FLOP/B</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.v} data-variant={r.v} data-bytes={r.bytes} data-printed={r.printed} data-ai={r.ai.toFixed(3)} data-ai-printed={r.aiPrinted.toFixed(3)}>
              <td><b>{r.v}</b>{r.v === 'GQA' ? ` (${hkv} K/V heads)` : ''}</td>
              <td className="muted">
                {r.v === 'MLA' ? `one latent of ${latent.dc} + ${latent.dr}` : `K and V for ${r.s.hkv} ${r.s.hkv === 1 ? 'head' : 'heads'} of ${base.d}`}
              </td>
              <td className="num">{fmtBytes(r.bytes)}</td>
              <td>
                <div style={{ height: '0.7rem', width: `${Math.max(1, (r.bytes / maxBytes) * 100)}%`, background: 'var(--global)' }} />
                <span className="muted" style={{ fontSize: '0.78rem' }}>{r.v === 'MHA' ? '' : `${fmtNum(mha / r.bytes)}× smaller than MHA`}</span>
              </td>
              <td className="num">{r.printed === r.bytes ? 'same' : fmtBytes(r.printed)}</td>
              <td className="num">{fmtNum(r.ai)}</td>
              <td className="num">{r.aiPrinted === r.ai ? 'same' : fmtNum(r.aiPrinted)}</td>
              <td className="num">{fmtNum(toFlopPerByte(r.ai, p))}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Bench>
  );
}

export function KvExplorer({ mode }: { mode: 'size' | 'compare' }) {
  return mode === 'size' ? <SizeMode /> : <CompareMode />;
}
