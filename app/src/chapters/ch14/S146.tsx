import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { benchRows, fmt, realBackends, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { LAB2, measuredLocal, measuredOneBit } from '../../sim/radixcost';
import {
  BOOK_KEYS,
  BOOK_KEY_BITS,
  UNITS,
  bin,
  directPass,
  exclusiveScan,
  bucketTable,
  localPass,
  localSortBlocks,
  passCount,
  passInput,
  placeFromCounts,
  softmaxKeys,
  starts,
  summarise,
  uniformKeys,
  unitsOf,
  warpStores,
  type Design,
  type Layout,
  type WarpStore,
} from '../../sim/radixlocal';
import { LocalSortScene } from './LocalSortScene';
import { useSolutionStatus } from './solutionStatus';

// ---------------------------------------------------------------- shared bits
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const plural = (n: number, one: string, many = one + 's') => `${n} ${n === 1 ? one : many}`;
const dec = (v: number) => (Math.round(v * 10) / 10).toFixed(1);

function Seg<T extends string | number>({ label, value, options, onPick, show, disabled }: {
  label: string; value: T; options: readonly T[]; onPick: (v: T) => void; show?: (v: T) => string; disabled?: (v: T) => boolean;
}) {
  return (
    <span style={FIELD}>
      {label}
      <span className="seg" role="group" aria-label={label}>
        {options.map((o) => (
          <button key={String(o)} aria-pressed={value === o} disabled={disabled?.(o)} onClick={() => onPick(o)}>
            {show ? show(o) : String(o)}
          </button>
        ))}
      </span>
    </span>
  );
}

// ---------------------------------------------------------------- interactive 1: where a warp's writes land
type KeySet = 'book' | 'uniform' | 'softmax';
const KEYSET_LABEL: Record<KeySet, string> = {
  book: "the book's sixteen",
  uniform: '4,096 random keys',
  softmax: 'a 4,096-token softmax row',
};
const N_SAMPLE = 4096;
const KEYS: Record<KeySet, readonly number[]> = {
  book: BOOK_KEYS,
  uniform: uniformKeys(N_SAMPLE, 1),
  softmax: softmaxKeys(N_SAMPLE, 1),
};
const ELEM = 4; // every key is stored as a 4-byte unsigned int
const METAL_SIZES = [16, 32, 64, 128];

function BookStrip({ addrs, bounds, out, source, local, lanesFrom }: {
  addrs: number[]; bounds: number[]; out: number[]; source: number[]; local: boolean; lanesFrom: number[];
}) {
  const cw = 680 / 16;
  const hit = new Set(addrs);
  const from = new Set(lanesFrom);
  const tier = local ? 'shared' : 'global';
  return (
    <>
      <div style={{ overflowX: 'auto' }}>
        <svg viewBox="0 0 680 140" style={{ width: '100%', minWidth: 560, maxWidth: 820, display: 'block' }} role="img"
          aria-label="This block's keys and where they land in the output" data-strip="book">
          <defs>
            <marker id="s146-head" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
              <path d="M0,0 L8,4 L0,8 z" fill="var(--ink)" />
            </marker>
          </defs>
          {lanesFrom.map((src, l) => (
            <text key={l} x={src * cw + cw / 2} y={12} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>t{l}</text>
          ))}
          {source.map((k, i) => (
            <g key={i} data-src={i}>
              <rect x={i * cw + 1} y={18} width={cw - 2} height={24} rx="2" fill={from.has(i) ? `var(--${tier})` : `var(--${tier}-wash)`} stroke={`var(--${tier})`} />
              <text x={i * cw + cw / 2} y={34} fontSize="11" textAnchor="middle" style={{ fill: from.has(i) ? 'var(--page)' : 'var(--ink)' }}>{bin(k, 4)}</text>
            </g>
          ))}
          {[1, 2, 3].map((b) => <line key={b} x1={b * 4 * cw} x2={b * 4 * cw} y1={14} y2={46} stroke="var(--ink-3)" strokeDasharray="3 3" />)}
          {lanesFrom.map((src, l) => (
            <line key={l} x1={src * cw + cw / 2} y1={43} x2={addrs[l] * cw + cw / 2} y2={100} stroke="var(--ink)" strokeWidth="1.6" markerEnd="url(#s146-head)" />
          ))}
          {out.map((k, i) => (
            <g key={i} data-out={i} data-hit={hit.has(i) ? 1 : 0}>
              <rect x={i * cw + 1} y={104} width={cw - 2} height={24} rx="2" fill={hit.has(i) ? 'var(--global)' : 'var(--global-wash)'} stroke="var(--global)" />
              <text x={i * cw + cw / 2} y={120} fontSize="11" textAnchor="middle" style={{ fill: hit.has(i) ? 'var(--page)' : 'var(--ink)' }}>{bin(k, 4)}</text>
            </g>
          ))}
          {bounds.slice(1).map((b, i) => (b > 0 && b < 16 ? <line key={i} x1={b * cw} x2={b * cw} y1={100} y2={134} stroke="var(--ink)" strokeWidth="2.4" /> : null))}
        </svg>
      </div>
      <p className="muted" style={{ margin: '0.3rem 0 0', fontSize: '0.84rem' }}>
        Top: {local ? "each block's keys after its local sort, in shared memory" : 'the keys as this pass reads them, in global memory'}, dashed lines
        between blocks; the darker cells are this store's. Bottom: the output in global memory; thick lines separate its buckets.
      </p>
    </>
  );
}

function WideStrip({ n, addrs, bounds }: { n: number; addrs: number[]; bounds: number[] }) {
  const W = 680;
  const x = (a: number) => (a / n) * W;
  // A marker above the start of every run, so a run of one key is still visible at this scale.
  const sorted = [...addrs].sort((a, b) => a - b);
  const runStarts = sorted.filter((a, i) => i === 0 || a !== sorted[i - 1] + 1);
  return (
    <>
      <svg viewBox="0 0 680 44" style={{ width: '100%', display: 'block' }} role="img" aria-label="Output of the pass with this warp's writes marked" data-strip="wide" data-runs={runStarts.length}>
        {runStarts.map((a) => (
          <path key={a} d={`M${x(a) - 4},2 h8 l-4,7 z`} fill="var(--ink)" data-run-start={a} />
        ))}
        <rect x="0" y="12" width={W} height="26" fill="var(--global-wash)" stroke="var(--global)" />
        {bounds.slice(1, -1).map((b, i) => <line key={i} x1={x(b)} x2={x(b)} y1={12} y2={38} stroke="var(--ink-3)" strokeWidth={bounds.length > 20 ? 0.4 : 1} />)}
        {addrs.map((a, i) => (
          <rect key={i} x={x(a)} y={8} width={Math.max(1.5, W / n)} height={34} fill="var(--global)" />
        ))}
      </svg>
      <p className="muted" style={{ margin: '0.2rem 0 0', fontSize: '0.84rem' }}>
        The whole output, {n.toLocaleString('en-US')} keys, in global memory. Dark: what this store writes, with a triangle over each
        run. Thin lines: bucket boundaries.
      </p>
    </>
  );
}

function Units({ w, unitBytes, noun }: { w: WarpStore; unitBytes: number; noun: string }) {
  const per = Math.max(1, unitBytes / ELEM);
  const written = new Set(w.addrs);
  const shown = w.units.slice(0, 48);
  return (
    <div data-units={w.units.length} style={{ display: 'flex', flexWrap: 'wrap', gap: '6px 4px', alignItems: 'center' }} aria-label={`${noun}s touched`}>
      {shown.map((u, k) => (
        <span key={u} style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
          {k > 0 && u !== shown[k - 1] + 1 ? <span className="muted" style={{ fontSize: '0.8rem' }}>…</span> : null}
          <span data-unit={u} title={`${noun} ${u}: bytes ${u * unitBytes} to ${(u + 1) * unitBytes - 1}`}
            style={{ display: 'inline-flex', border: '1px solid var(--global)', background: 'var(--cell)' }}>
            {Array.from({ length: per }, (_, e) => (
              <i key={e} style={{ width: per > 16 ? 4 : 6, height: 14, background: written.has(u * per + e) ? 'var(--global)' : 'transparent', borderLeft: e ? '1px solid var(--rule-soft)' : 'none' }} />
            ))}
          </span>
        </span>
      ))}
      {w.units.length > shown.length ? <span className="muted">and {w.units.length - shown.length} more</span> : null}
    </div>
  );
}

function WarpWrites() {
  const [ks, setKs] = useState<KeySet>('uniform');
  const [B, setB] = useState(256);
  const [W, setW] = useState(32);
  const [r, setR] = useState(1);
  const [pass, setPass] = useState(0);
  const [unitId, setUnitId] = useState('nv-sector');
  const [metal, setMetal] = useState(32);
  const [design, setDesign] = useState<Design>('direct');
  const [sel, setSel] = useState(0);

  const book = ks === 'book';
  const bits = book ? BOOK_KEY_BITS : 32;
  const P = passCount(bits, r);
  const p = Math.min(pass, P - 1);
  const bitsLabel = (sh: number) => (r === 1 ? `bit ${sh}` : `bits ${sh} to ${sh + r - 1}`);
  const block = book ? 4 : B;
  const unit = UNITS.find((u) => u.id === unitId)!;
  const unitBytes = unit.bytes ?? metal;

  const m = useMemo(() => {
    const input = passInput(KEYS[ks], r, p);
    const shift = p * r;
    const plans = { direct: directPass(input, shift, r, block), local: localPass(input, shift, r, block) };
    const stores = { direct: warpStores(plans.direct, W, ELEM, unitBytes), local: warpStores(plans.local, W, ELEM, unitBytes) };
    const counts = new Array<number>(2 ** r).fill(0);
    for (const k of input) counts[(k >>> shift) & (2 ** r - 1)]++;
    const bounds = [...exclusiveScan(counts), input.length];
    return { input, shift, plans, stores, sum: { direct: summarise(stores.direct), local: summarise(stores.local) }, bounds };
  }, [ks, r, p, block, W, unitBytes]);

  const list = m.stores[design];
  const s = Math.min(sel, list.length - 1);
  const w = list[s];
  const lanes = Math.min(W, block);
  const plan = m.plans[design];
  // Where each lane's key sits before the store: its input slot (direct) or its local slot (local).
  const lanesFrom = list[s].addrs.map((_, l) => w.block * block + w.store * block + w.warp * W + l);
  const firstThread = w.warp * W;
  const noun = unit.noun;
  const pick = (k: KeySet) => {
    setKs(k);
    setPass(0);
    setSel(0);
    if (k === 'book' && r > 4) setR(1);
  };

  return (
    <Bench
      title="Where a warp's writes land"
      task="Start with the direct scatter and raise r. Then switch the design to the local sort, and try a block of 4 threads against one of 256."
      foot={
        <>
          Model: src/sim/radixlocal.ts. One warp-wide store at a time; 4-byte keys; the output starts on a {noun} boundary.
          Each pass sees the keys as the earlier passes left them{p ? ` (here, after passes 0 to ${p - 1})` : ''}. The
          softmax row is synthetic (normal logits, standard deviation 2, ours), its keys transformed for a descending sort
          as lab 14.3 does (ch14-facts E1).
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <Seg label="Keys" value={ks} options={['book', 'uniform', 'softmax'] as KeySet[]} show={(k) => KEYSET_LABEL[k]} onPick={pick} />
        <Seg label={book ? 'Threads per block (the figure has 4)' : 'Threads per block'} value={block} options={[4, 128, 256]}
          disabled={(v) => book && v !== 4} onPick={(v) => { setB(v); setSel(0); }} />
        <Seg label="Warp width" value={W} options={[32, 64]} show={(v) => (v === 64 ? '64 (AMD gfx9)' : '32')} onPick={(v) => { setW(v); setSel(0); }} />
      </div>
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <Seg label="Digit width r" value={r} options={[1, 2, 4, 8]} disabled={(v) => book && v > 4}
          onPick={(v) => { setR(v); setPass(0); setSel(0); }} />
        {P <= 4 ? (
          <Seg label={`Pass (${bitsLabel(m.shift)})`} value={p} options={Array.from({ length: P }, (_, i) => i)}
            onPick={(v) => { setPass(v); setSel(0); }} />
        ) : (
          <label className="field">
            Pass {p} of passes 0 to {P - 1} ({bitsLabel(m.shift)})
            <input type="range" min={0} max={P - 1} value={p} aria-label="Pass" onChange={(e) => { setPass(Number(e.target.value)); setSel(0); }} />
          </label>
        )}
        <Seg label="Design" value={design} options={['direct', 'local'] as Design[]}
          show={(d) => (d === 'direct' ? 'direct scatter' : 'local sort, then write')} onPick={setDesign} />
      </div>
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <label className="field" style={{ maxWidth: '100%', minWidth: 0 }}>
          Memory unit
          <select value={unitId} aria-label="Memory unit" onChange={(e) => setUnitId(e.target.value)} style={{ maxWidth: '100%' }}>
            {UNITS.map((u) => <option key={u.id} value={u.id}>{u.label}</option>)}
          </select>
        </label>
        {unit.bytes == null ? (
          <Seg label="Your segment size (bytes)" value={metal} options={METAL_SIZES} onPick={setMetal} />
        ) : null}
        <label className="field" style={{ flex: '1 1 14rem' }}>
          Store {s + 1} of {list.length}: block {w.block}, {book || block <= W ? `its ${lanes} threads` : `warp ${w.warp} (threads ${firstThread} to ${firstThread + w.addrs.length - 1})`}
          <input type="range" min={0} max={list.length - 1} value={s} aria-label="Which warp store" onChange={(e) => setSel(Number(e.target.value))} />
        </label>
      </div>
      {unit.bytes == null ? (
        <p className="ui" style={{ margin: '0 0 0.8rem' }} data-metal-note>
          Apple documents no coalescing rule, line size or sector size (ch06-facts C1). The counts below use the {metal}-byte
          size you picked; only a measurement says what a Mac does.
        </p>
      ) : null}

      {book ? (
        <BookStrip addrs={w.addrs} bounds={m.bounds} out={plan.out} lanesFrom={lanesFrom} local={design === 'local'}
          source={design === 'local' ? plan.blocks.flatMap((b) => b.sorted) : m.input} />
      ) : (
        <WideStrip n={plan.n} addrs={w.addrs} bounds={m.bounds} />
      )}

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginTop: '0.9rem', alignItems: 'flex-start' }}>
        <div className="panel" style={{ flex: '1 1 18rem', minWidth: 0 }} aria-live="polite">
          <p style={{ margin: '0 0 0.5rem' }}>
            <b>This store.</b>{' '}
            <span data-readout="store">
              {plural(w.addrs.length, 'lane')} write{w.addrs.length === 1 ? 's' : ''} {plural(w.runs, 'run')} of consecutive addresses, touching{' '}
              <b data-readout="units">{plural(w.units.length, noun)}</b> of {unitBytes} bytes; one aligned run of the same keys would need{' '}
              <span data-readout="min">{w.minUnits}</span>.
            </span>
          </p>
          <p className="muted" style={{ margin: '0 0 0.5rem', fontSize: '0.86rem' }} data-readout="digits">
            Lane digits: {w.digits.join(' ')}
          </p>
          <Units w={w} unitBytes={unitBytes} noun={noun} />
        </div>
        <div className="panel" style={{ flex: '1 1 18rem', minWidth: 0, overflowX: 'auto' }}>
          <p style={{ margin: '0 0 0.4rem' }}>
            <b>Every store of this pass</b>, on average
          </p>
          <table className="data" style={{ margin: 0 }}>
            <thead>
              <tr>
                <th />
                <th className="num">direct scatter</th>
                <th className="num">local sort</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>warp stores</td>
                <td className="num">{m.sum.direct.stores}</td>
                <td className="num">{m.sum.local.stores}</td>
              </tr>
              <tr>
                <td>runs per store</td>
                <td className="num" data-avg="direct-runs">{dec(m.sum.direct.meanRuns)}</td>
                <td className="num" data-avg="local-runs">{dec(m.sum.local.meanRuns)}</td>
              </tr>
              <tr>
                <td>{noun}s per store</td>
                <td className="num" data-avg="direct-units">{design === 'direct' ? <b>{dec(m.sum.direct.meanUnits)}</b> : dec(m.sum.direct.meanUnits)}</td>
                <td className="num" data-avg="local-units">{design === 'local' ? <b>{dec(m.sum.local.meanUnits)}</b> : dec(m.sum.local.meanUnits)}</td>
              </tr>
              <tr>
                <td>coalesced minimum</td>
                <td className="num">{dec(m.sum.direct.meanMin)}</td>
                <td className="num">{dec(m.sum.local.meanMin)}</td>
              </tr>
              <tr>
                <td>worst store</td>
                <td className="num">{m.sum.direct.maxUnits}</td>
                <td className="num">{m.sum.local.maxUnits}</td>
              </tr>
            </tbody>
          </table>
          <p className="muted" style={{ margin: '0.5rem 0 0', fontSize: '0.86rem' }}>
            The direct scatter makes at most {Math.min(lanes, 2 ** r)} runs per store here: the smaller of the lanes ({lanes}) and the
            buckets (2<sup>{r}</sup> = {2 ** r}).
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- interactive 2: the bucket table
const BOOK_ZEROS = localSortBlocks(BOOK_KEYS, 0, 1, 4).map((b) => b.counts[0]);
const BOOK_LOCAL = localSortBlocks(BOOK_KEYS, 0, 1, 4);

function BucketTable() {
  const [zeros, setZeros] = useState<number[]>(BOOK_ZEROS);
  const [layout, setLayout] = useState<Layout>('digit-major');
  const [sel, setSel] = useState<{ b: number; t: number }>({ b: 2, t: 0 });
  const counts = zeros.map((z) => [z, 4 - z]);
  const isBook = zeros.every((z, i) => z === BOOK_ZEROS[i]);
  const table = bucketTable(counts, layout);
  const scanned = exclusiveScan(table);
  const st = starts(counts, layout);
  const placed = placeFromCounts(counts, layout);
  const byAddr = [...placed].sort((a, b) => a.addr - b.addr);
  const totalZeros = zeros.reduce((a, b) => a + b, 0);
  // A partitioned output holds zeros in its first totalZeros positions and ones after them.
  const misplaced = (i: number) => byAddr[i].digit !== (i < totalZeros ? 0 : 1);
  const firstBad = byAddr.findIndex((_, i) => misplaced(i));
  const ok = firstBad < 0;
  const labels = counts.flatMap((_, b) => [0, 1].map((d) => ({ b, d }))).sort((x, y) =>
    layout === 'digit-major' ? x.d - y.d || x.b - y.b : x.b - y.b || x.d - y.d,
  );
  const me = placed.find((x) => x.block === sel.b && x.slot === sel.t)!;
  const z = zeros[sel.b];
  const keyAt = (b: number, t: number) => (isBook ? bin(BOOK_LOCAL[b].sorted[t], 4) : null);
  const preset = (zs: number[], s: { b: number; t: number }) => {
    setZeros(zs);
    setLayout('digit-major');
    setSel(s);
  };
  const setZ = (b: number, v: number) => setZeros(zeros.map((x, i) => (i === b ? Math.max(0, Math.min(4, v)) : x)));

  return (
    <Bench
      title="The bucket table"
      task="Change a block's split between zeros and ones and watch every start move. Click a thread to see where it writes. Then scan the table in the wrong order."
      foot="Model: bucketTable(), starts() and placeFromCounts() in src/sim/radixlocal.ts. Four blocks of four keys and a 1-bit digit, as in Fig. 14.9. Your choices here are not recorded."
    >
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          Presets
          <span className="seg" role="group" aria-label="Presets">
            <button onClick={() => preset(BOOK_ZEROS, { b: 0, t: 0 })}>Fig. 14.9</button>
            <button onClick={() => preset(BOOK_ZEROS, { b: 2, t: 1 })}>p. 342: block 2</button>
            <button onClick={() => preset(BOOK_ZEROS, { b: 3, t: 3 })}>p. 342: block 3</button>
          </span>
        </span>
        <Seg label="Table layout before the scan" value={layout} options={['digit-major', 'block-major'] as Layout[]}
          show={(l) => (l === 'digit-major' ? 'all zero counts, then all one counts' : 'block by block (wrong)')} onPick={setLayout} />
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.7rem', marginBottom: '1rem' }}>
        {counts.map(([zc, oc], b) => (
          <div key={b} className="panel" style={{ flex: '1 1 9.5rem', minWidth: '9rem', padding: '0.6rem' }} data-block={b}>
            <p style={{ margin: '0 0 0.4rem' }}><b>Block {b}</b></p>
            <div style={{ display: 'flex', flexWrap: 'nowrap', alignItems: 'center', gap: '0.3rem', marginBottom: '0.45rem' }}>
              <button aria-label={`fewer zeros in block ${b}`} onClick={() => setZ(b, zc - 1)} disabled={zc === 0} style={{ padding: '0.15rem 0.5rem' }}>−</button>
              <span className="ui" data-counts={`${zc} ${oc}`} style={{ flex: '1 1 auto', minWidth: 0, textAlign: 'center', fontSize: '0.8rem', lineHeight: 1.2 }}>
                {zc} zero{zc === 1 ? '' : 's'}, {oc} one{oc === 1 ? '' : 's'}
              </span>
              <button aria-label={`more zeros in block ${b}`} onClick={() => setZ(b, zc + 1)} disabled={zc === 4} style={{ padding: '0.15rem 0.5rem' }}>+</button>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 3 }} role="group" aria-label={`threads of block ${b}`}>
              {[0, 1, 2, 3].map((t) => {
                const d = t < zc ? 0 : 1;
                const on = sel.b === b && sel.t === t;
                return (
                  <button key={t} aria-pressed={on} aria-label={`block ${b} thread ${t}`} onClick={() => setSel({ b, t })}
                    style={{ padding: '0.2rem 0', display: 'flex', flexDirection: 'column', alignItems: 'center', lineHeight: 1.2,
                      background: on ? 'var(--shared-wash)' : undefined, borderColor: on ? 'var(--shared)' : undefined, color: on ? 'var(--ink)' : undefined }}>
                    <span className="muted" style={{ fontSize: '0.72rem' }}>t{t}</span>
                    <b style={{ fontSize: '0.86rem' }}>{keyAt(b, t) ?? d}</b>
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      <div className="panel" style={{ overflowX: 'auto', marginBottom: '0.9rem' }}>
        <table className="data" style={{ margin: 0, minWidth: '34rem' }} aria-label="The table and its scan">
          <thead>
            <tr>
              <th>table entry</th>
              {labels.map(({ b, d }) => <th key={`${b}-${d}`} className="num">{d ? 'ones' : 'zeros'} b{b}</th>)}
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>count</td>
              {table.map((v, i) => <td key={i} className="num" data-table={i}>{v}</td>)}
            </tr>
            <tr>
              <td>exclusive scan</td>
              {scanned.map((v, i) => <td key={i} className="num" data-scan={i}><b>{v}</b></td>)}
            </tr>
          </tbody>
        </table>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', alignItems: 'flex-start' }}>
        <div className="panel" style={{ flex: '1 1 16rem' }} aria-live="polite">
          <p style={{ margin: 0 }} data-readout="thread">
            <b>Block {sel.b}, thread {sel.t}.</b>{' '}
            {me.digit === 0 ? (
              <>
                Its local slot {sel.t} is below the block's zero count ({z}), so it holds a zero. It writes to the zero bucket's
                start plus its slot: {st[sel.b][0]} + {sel.t} = <b>{me.addr}</b>.
              </>
            ) : (
              <>
                Its local slot {sel.t} is not below the block's zero count ({z}), so it holds a one, the{' '}
                {['first', 'second', 'third', 'fourth'][sel.t - z]} of the block's ones. It writes to the one bucket's start
                plus {sel.t} − {z}: {st[sel.b][1]} + {sel.t - z} = <b>{me.addr}</b>.
              </>
            )}
          </p>
          <table className="data" style={{ margin: '0.6rem 0 0' }}>
            <thead>
              <tr>
                <th>block</th>
                {counts.map((_, b) => <th key={b} className="num">{b}</th>)}
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>zero bucket starts at</td>
                {st.map((x, b) => <td key={b} className="num" data-start={`${b}-0`}>{x[0]}</td>)}
              </tr>
              <tr>
                <td>one bucket starts at</td>
                {st.map((x, b) => <td key={b} className="num" data-start={`${b}-1`}>{x[1]}</td>)}
              </tr>
            </tbody>
          </table>
        </div>
        <div className="panel" style={{ flex: '1 1 20rem', minWidth: 0 }}>
          <p style={{ margin: '0 0 0.5rem' }}><b>The output</b> (digit, and which block and thread wrote it)</p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(8, minmax(0, 1fr))', gap: 3 }} data-output={byAddr.map((x) => x.digit).join('')}>
            {byAddr.map((x, i) => {
              const bad = misplaced(i);
              const mine = x.block === sel.b && x.slot === sel.t;
              return (
                <div key={i} data-addr={i} data-bad={bad ? 1 : 0}
                  style={{ border: `1px solid ${bad ? 'var(--fail)' : 'var(--global)'}`, background: bad ? 'var(--fail-wash)' : mine ? 'var(--global)' : 'var(--global-wash)',
                    color: mine && !bad ? 'var(--page)' : 'var(--ink)', borderRadius: 2, textAlign: 'center', fontSize: '0.74rem', lineHeight: 1.25, padding: '0.15rem 0' }}>
                  <span style={{ opacity: 0.75 }}>{i}</span>
                  <br />
                  <b>{x.digit}</b>
                  <br />
                  <span style={{ opacity: 0.8 }}>b{x.block}t{x.slot}</span>
                </div>
              );
            })}
          </div>
          <p style={{ margin: '0.6rem 0 0' }} data-verdict={ok ? 'partitioned' : 'not-partitioned'}>
            {ok ? (
              <span className="t-ok">Partitioned: every zero before every one, and each block's keys in block order.</span>
            ) : (
              <span className="t-fail">
                Not partitioned: position {firstBad} holds a one, written by block {byAddr[firstBad].block}, where only zeros belong.
                Scanned block by block, a block's ones are placed ahead of the next block's zeros.
              </span>
            )}
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- lab 14.2's measurement
function useLab2() {
  const { reports } = useReports();
  const devs = realBackends(reports).flatMap((b) => {
    const rows = benchRows(reports[b], LAB2);
    const local = measuredLocal(rows, 1, 1);
    const onebit = measuredOneBit(rows);
    return local != null && onebit != null ? [{ b, device: String(reports[b].device.name ?? b), local, onebit }] : [];
  });
  return { devs };
}

function LabListing() {
  const cu = useLabFile('ch14/lab2_multibit/solution/kernel.cu');
  const metal = useLabFile('ch14/lab2_multibit/solution/kernel.metal');
  const st = useSolutionStatus(LAB2);
  if (!cu || !metal) return null;
  return (
    <Code
      tabs={[
        { label: 'CUDA and HIP', code: cu, status: st.cu },
        { label: 'Metal', code: metal, status: st.metal },
      ]}
      caption={
        <>
          The reference solution of <a href={href('ch14', 'labs')}>lab 14.2</a>, loaded from{' '}
          <code>labs/ch14/lab2_multibit/solution/</code>. Try it in the lab first: write the three kernels yourself and keep
          this for when you are stuck.
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- questions
const Q_COUNTS = [[3, 1], [1, 3], [2, 2], [2, 2]];
const Q_START = starts(Q_COUNTS, 'digit-major')[2][1];
const Q_THREAD = placeFromCounts(Q_COUNTS, 'digit-major').find((x) => x.block === 1 && x.slot === 2)!.addr;
const Q_SECTORS = unitsOf([...Array.from({ length: 12 }, (_, i) => 4 + i), ...Array.from({ length: 20 }, (_, i) => 64 + i)], 4, 32).length;
const P_R8 = summarise(warpStores(directPass(KEYS.uniform, 0, 8, 256), 32, 4, 32));
const P_R1 = summarise(warpStores(directPass(KEYS.uniform, 0, 1, 256), 32, 4, 32));

export const questionsS146: Record<string, Question> = {
  'ch14/14.6/q-start': {
    kind: 'numeric',
    id: 'ch14/14.6/q-start',
    prompt: (
      <>
        Four blocks finish their local sort with zero counts [3, 1, 2, 2] and one counts [1, 3, 2, 2]. At which output
        position does block 2's one bucket start?
      </>
    ),
    answer: Q_START,
    explain: (
      <p>
        All 8 zeros of all four blocks come first. After them come the ones of the blocks before block 2: block 0's 1 and
        block 1's 3. So 8 + 1 + 3 = {Q_START}. In the table laid out as [3, 1, 2, 2, 1, 3, 2, 2], that is the exclusive
        scan at the entry "ones of block 2".
      </p>
    ),
    source: 'ch. 14 §14.6, Fig. 14.9, pp. 340–341',
  },
  'ch14/14.6/q-thread': {
    kind: 'numeric',
    id: 'ch14/14.6/q-thread',
    prompt: (
      <>
        The same four blocks. Block 1 holds 1 zero and 3 ones, and its one bucket starts at 9. Where does block 1's thread 2
        write its key?
      </>
    ),
    answer: Q_THREAD,
    explain: (
      <p>
        Thread 2's local slot is 2, which is not below the block's zero count of 1, so it holds a one: the second of the
        block's ones, at offset 2 − 1 = 1 inside the bucket. 9 + 1 = {Q_THREAD}.
      </p>
    ),
    source: 'ch. 14 §14.6, pp. 341–342',
  },
  'ch14/14.6/q-sectors': {
    kind: 'numeric',
    id: 'ch14/14.6/q-sectors',
    prompt: (
      <>
        A warp of 32 lanes stores 4-byte keys in two runs: 12 zeros starting 16 bytes into a 32-byte sector, and 20 ones
        starting on a sector boundary somewhere else. How many 32-byte sectors does the store touch?
      </>
    ),
    answer: Q_SECTORS,
    unit: 'sectors',
    explain: (
      <p>
        The zeros cover 48 bytes from offset 16, so bytes 16 to 63: two sectors. The ones cover 80 bytes from a boundary:
        three sectors, the last one half used. {Q_SECTORS} in all, where one aligned run of 32 keys needs 4.
      </p>
    ),
    source: 'ch14-facts A10; ch06-facts A2 (NVIDIA counts a warp\'s access in 32-byte sectors)',
  },
  'ch14/14.6/q-layout': {
    kind: 'choice',
    id: 'ch14/14.6/q-layout',
    prompt: <>Why must the table list every block's zero count before any block's one count?</>,
    options: [
      { label: 'It makes the scan of the table coalesced.', why: 'Either layout is one contiguous array; the scan reads it the same way. The order changes what the scan adds up, not how it reads.' },
      {
        label:
          "Every one must land after every zero of every block, and an exclusive scan in that order adds up exactly the counts that come before each bucket. Block by block, a block's ones would land before the next block's zeros.",
        correct: true,
      },
      { label: 'Any order works: addition is commutative, so the scan gives the same totals.', why: 'The final total is the same, but an exclusive scan gives a different prefix at each entry. Try the block-by-block layout in the explorer.' },
    ],
    explain: (
      <p>
        An exclusive scan gives each entry the sum of everything listed before it. The start of block b's one bucket must be
        the number of keys that belong in front of it: all zeros, of every block, plus the ones of blocks 0 to b − 1. Only a
        table with all zero counts first, then the one counts in block order, puts exactly those entries before it.
      </p>
    ),
    source: 'ch. 14 §14.6, p. 340',
  },
};

// ---------------------------------------------------------------- the section
export function S146() {
  const { devs } = useLab2();
  return (
    <>
      <p>
        The pass of section 14.5 is correct and stable. Its stores are the problem. Every thread computes where its key goes
        and stores it there, and the destination depends on the key's bit, not on the thread's position. In block 0 of the
        book's example the four threads write to the zero bucket, the one bucket, the zero bucket and the one bucket in
        turn: neighbouring threads, addresses far apart.
      </p>
      <Source section="14.6" pages="339–340" figs="Fig. 14.5" />
      <p>
        How much does that cost? Look inside one warp. Each zero's destination is one more than the previous zero's, because
        the scan counted exactly the keys in between, and the same holds for the ones. So a warp's store is at most two runs
        of consecutive addresses. The book's estimate is that the warp touches two cache lines, or a little more depending on
        alignment.
      </p>
      <Source section="14.6" pages="340" />
      <p>
        In NVIDIA's own unit, the 32-byte sector, the two runs cost 4 to 7 sectors, where one coalesced store of 32 four-byte
        keys costs 4. At one bit that is a modest loss. It grows with the digit: with an r-bit digit a warp writes up to
        2<sup>r</sup> runs, and with 8 bits on random keys almost every lane has a bucket to itself, which approaches one
        sector per key.
      </p>
      <Today checked="2026-11-20">
        <p>
          NVIDIA serves a warp's access in 32-byte sectors, four to a 128-byte line, in any lane order (ch06-facts A2, A4).
          AMD groups a wavefront's accesses into 64- or 128-byte lines (B1), and its wavefronts are 64 wide on gfx9
          (ch05-facts B5). Apple documents no rule at all (C1): measure. The sector arithmetic is ours (ch14-facts A10).
        </p>
      </Today>
      <Predict
        id="ch14/14.6/p-direct-r8"
        prompt="Before you open the explorer: random 32-bit keys, blocks of 256 threads, an 8-bit digit, the direct scatter. How many 32-byte sectors does an average warp store touch?"
        unit="sectors"
      >
        {(v) => (
          <p>
            The model gives <b>{dec(P_R8.meanUnits)}</b> sectors per store, for {dec(P_R8.meanRuns)} runs among 32 lanes; at
            one bit it is {dec(P_R1.meanUnits)}.{' '}
            {Math.abs(v - P_R8.meanUnits) <= 3
              ? 'Close to yours.'
              : v < P_R8.meanUnits
                ? 'More than you said: with 256 buckets, two lanes of a warp rarely share one, so nearly every lane is a run of its own.'
                : 'Fewer than you said: a run of one 4-byte key still costs only one sector, and a few lanes do share a bucket.'}{' '}
            Set the explorer to r = 8 and check it.
          </p>
        )}
      </Predict>
      <p>
        The explorer runs one pass of a sort and follows its stores one warp at a time. Pick a store with the slider: the left
        panel counts its runs and the memory units it touches, and the table on the right averages every store of the pass in
        both designs, so you can compare them without switching.
      </p>
      <Simplification>
        <p>
          One warp-wide store at a time, no cache, no timing: a sector two stores touch counts twice. Segment sizes are the
          vendors' documented units; for Metal the size is yours. The output array starts on a unit boundary.
        </p>
      </Simplification>
      <WarpWrites />
      <p>
        Three things to find. First, the direct scatter never makes more runs per store than the smaller of the warp width
        and 2<sup>r</sup>, and at r = 8 it comes close to the warp width. Second, the local sort changes nothing with 4
        threads per block and cuts the runs with 256; the next part says why. Third, on the softmax row the top digits take
        only a handful of values, because they hold the float's exponent, so the last passes of a sort of probabilities
        write few, long runs even without a local sort.
      </p>
      <Ask q={questionsS146['ch14/14.6/q-sectors']} />

      <h2>Do the scattering in shared memory</h2>
      <p>
        Chapter 6 offered three ways out of an access pattern that does not coalesce: change which thread handles which data,
        change the layout of the data, or do the irregular part in shared memory and move data between shared and global
        memory in an orderly way. A sort cannot choose its output layout, so this chapter takes the third, as{' '}
        <a href={href('ch12', '12.6')}>section 12.6</a> did for the stable filter.
      </p>
      <Source section="14.6" pages="340" />
      <Aside>
        <p>
          The three options are in <a href={href('ch06', '6.1')}>section 6.1</a> (book §6.1, p. 129).
        </p>
      </Aside>
      <p>
        Each block first runs the same pass on its own keys only, in shared memory, with a scan over the block instead of over
        the whole input. The block's keys come out sorted by the digit into local buckets. Then the block copies its local
        buckets to global memory, each one to its place inside the matching global bucket. Inside a bucket, neighbouring
        threads now write neighbouring addresses.
      </p>
      <Source section="14.6" pages="340" figs="Fig. 14.8" />
      <p>
        Look at what that changes and what it does not. In the book's example, block 0 writes to positions 0, 1, 8 and 9 in
        both designs; only which thread writes which has changed. A warp's store costs the sectors its addresses touch,
        whatever the lane order, so for a block no wider than a warp the local sort buys nothing. It pays when a block holds
        several warps: after the local sort each warp copies a stretch of consecutive local slots, and those mostly lie in one
        local bucket. In the explorer, compare 4 threads per block with 256.
      </p>
      <Beyond title="Ours">
        <p>
          The point about lane order follows from NVIDIA's rule that a warp's access costs the sectors it touches "in some
          linear or permuted way" (ch06-facts A2). The book does not make it.
        </p>
      </Beyond>
      <LocalSortScene />

      <h2>Where each local bucket goes</h2>
      <p>
        A block can only write its buckets once it knows where they start, and that depends on every other block. Block b's
        zero bucket goes after the zero buckets of blocks 0 to b − 1. Its one bucket goes after the zero buckets of all the
        blocks and after the one buckets of blocks 0 to b − 1.
      </p>
      <Source section="14.6" pages="340" />
      <p>
        One scan answers all of these at once. After its local sort every block writes its bucket sizes into a table, and the
        table is laid out digit by digit: the zero counts of all blocks, then the one counts of all blocks. An exclusive scan of
        that array gives, at each entry, the number of keys that belong in front of that bucket, which is its start. In the
        book's example the counts are [2, 2, 1, 3] and [2, 2, 3, 1], and the scan gives zero-bucket starts [0, 2, 4, 5] and
        one-bucket starts [8, 10, 12, 15].
      </p>
      <Source section="14.6" pages="340–341" figs="Fig. 14.9" />
      <BucketTable />
      <Ask q={questionsS146['ch14/14.6/q-start']} />
      <Ask q={questionsS146['ch14/14.6/q-layout']} />

      <h2>The write phase</h2>
      <p>
        Each block keeps its own two counts, so every thread can tell from its index which bucket its local slot is in.
        Thread t copies local slot t: if t is below the block's zero count it writes to the block's zero start plus t,
        otherwise to its one start plus t minus the zero count. In block 2, which holds one zero, thread 0 writes the zero and
        threads 1 to 3 write the three ones side by side. In block 3 it is the other way round: threads 0 to 2 write zeros,
        thread 3 the one.
      </p>
      <Source section="14.6" pages="341–342" />
      <Ask q={questionsS146['ch14/14.6/q-thread']} />

      <h2>Three kernels per pass</h2>
      <p>
        The table creates a dependence across the whole grid: no block may write until every block has counted and the table
        has been scanned. Fig. 14.7 hid the same problem inside its <code>gridExclusiveScan</code>. The plain way out is to
        end the kernel at each such point. One pass becomes three launches: count each block's digits, scan the table, then
        let every block sort its keys locally and scatter them. The keys are read twice, once to count and once to move, and
        nothing has to wait inside a kernel. The book leaves the code as exercise 1; lab 14.2 is that exercise.
      </p>
      <Source section="14.11" pages="347" />
      <Today checked="2026-11-20">
        <p>
          PyTorch's radix sort for Apple GPUs is this design written in Metal: kernels <code>radix_count</code>,{' '}
          <code>radix_scan</code> and <code>radix_scatter</code>, the table laid out digit by digit across blocks as in Fig.
          14.9, and a per-bit split for the local sort (ch14-facts D6). Metal documents no barrier across threadgroups, so
          separate dispatches are the documented route there (C4).
        </p>
      </Today>
      <LabListing />
      <Predict
        id="ch14/14.6/p-local-gain"
        prompt="Lab 14.2 provides the one-bit direct-scatter sort of section 14.5 and asks you for the local-sort design. At r = 1 and one key per thread, how many times faster will the local-sort design be on your device?"
        unit="×"
      >
        {(v) =>
          devs.length ? (
            <>
              {devs.map((d) => {
                const x = d.onebit / d.local;
                return (
                  <p key={d.b} data-measured={d.b}>
                    <b>{d.device}</b> ({d.b}): the direct one-bit sort took {fmt(d.onebit * 1000)} ms and the local sort at r = 1,
                    one key per thread, {fmt(d.local * 1000)} ms: <b>{fmt(x)}×</b>
                    {x < 1 ? ', so the local sort was slower' : ''}. You predicted {v}×.
                  </p>
                );
              })}
              <p className="muted">
                At one bit the explorer's sector counts fall only a little; whatever else differs between the two (the
                table scan instead of a scan over every key, the second read of the keys) is in the measurement too.
              </p>
            </>
          ) : (
            <p data-measured="none">
              No measurement on this machine yet. Run <code>python3 labs/pmpp.py run 14.2 --solution --bench</code>, then
              reload this page to compare.
            </p>
          )
        }
      </Predict>
    </>
  );
}
