import { useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import {
  CALLS,
  LLAMA2_7B_BYTES_PER_TOKEN,
  MiB,
  STRATEGIES,
  apply,
  bookExample,
  bookThreeCalls,
  clean,
  grow,
  initial,
  run,
  slots,
  type Dev,
  type Op,
  type Outcome,
  type Strategy,
  type VmmState,
} from '../../sim/vmmgrow';
import { binary } from '../../sim/addrwidth';
import { GrowBufferScene } from './GrowBufferScene';

/** A long identifier that may break anywhere, for the narrow margin notes. */
function Id({ children }: { children: ReactNode }) {
  return <code style={{ whiteSpace: 'normal', overflowWrap: 'anywhere' }}>{children}</code>;
}

// ---------------------------------------------------------------- quoted signatures
// Lesson copies of labs/appC/snippets/c9_vmm_decls.h and c9_vmm_decls_hip.h, quoted from
// the vendors' references as recorded in notes/today/appC-facts.md (A15, B7). SC9.test.ts
// fails if a copy drifts from its file, and checks that every declaration appears in the
// facts file. Not compiled anywhere.
export const SIGS_CUDA = `// 1. Reserve a range of virtual addresses.
CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment,
                             CUdeviceptr addr, unsigned long long flags);
// 2. Allocate physical memory on a device; you get a handle, not an address.
CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size,
                     const CUmemAllocationProp *prop, unsigned long long flags);
// 3. Put that memory behind part of the range.
CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                  CUmemGenericAllocationHandle handle, unsigned long long flags);
// 4. Let devices read and write it. The book's list leaves this one out.
CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc *desc, size_t count);
// Teardown, in this order: cuMemUnmap, cuMemRelease, cuMemAddressFree.`;

export const SIGS_HIP = `// The same steps: hipMemAddressReserve, hipMemCreate, hipMemMap, hipMemSetAccess.
hipError_t hipMemAddressReserve(void** ptr, size_t size, size_t alignment, void* addr,
                                unsigned long long flags);
hipError_t hipMemSetAccess(void* ptr, size_t size, const hipMemAccessDesc* desc, size_t count);
// Our facts file quotes these two declarations; hipMemCreate and hipMemMap are named, not quoted.`;

// ---------------------------------------------------------------- the stepper
const G = 2 * MiB; // one granule; a parameter of the model, not a device value
const N = 8;
const SLOT = 62;
const SIZES = [1, 2, 4, 1.5];

const KIND_CHIP: Record<Outcome['kind'], string> = { ok: 'ok', error: 'fail', crash: 'fail', order: 'fail' };
const KIND_WORD: Record<Outcome['kind'], string> = { ok: 'done', error: 'refused', crash: 'crash', order: 'out of order' };

function opText(op: Op): string {
  const g = (b: number) => `${Number((b / G).toFixed(2))}`;
  switch (op.op) {
    case 'reserve':
      return `reserve ${g(op.size)} granules`;
    case 'create':
      return `create ${g(op.size)} granule${op.size === G ? '' : 's'} on GPU ${op.device}`;
    case 'map':
      return `map h${op.chunk} at granule ${g(op.offset)}`;
    case 'setAccess':
      return `grant GPU ${op.device} granules ${g(op.offset)} to ${Number(g(op.offset + op.size)) - 1}`;
    case 'read':
      return `kernel on GPU ${op.device} reads granules ${g(op.offset)} to ${Number(g(op.offset + op.size)) - 1}`;
    case 'unmap':
      return `unmap h${op.chunk}`;
    case 'release':
      return `release h${op.chunk}`;
    case 'free':
      return 'free the range';
  }
}

function Seg<T extends string | number>({ label, value, options, set }: { label: string; value: T; options: { v: T; t: string }[]; set: (v: T) => void }) {
  return (
    <span className="seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={String(o.v)} aria-pressed={value === o.v} onClick={() => set(o.v)}>
          {o.t}
        </button>
      ))}
    </span>
  );
}

function SlotSelect({ label, value, set, from = 0 }: { label: string; value: number; set: (v: number) => void; from?: number }) {
  return (
    <select aria-label={label} value={value} onChange={(e) => set(Number(e.target.value))}>
      {Array.from({ length: N - from }, (_, k) => k + from).map((k) => (
        <option key={k} value={k}>
          {k}
        </option>
      ))}
    </select>
  );
}

function VmmStepper() {
  const [state, setState] = useState<VmmState>(() => initial(G));
  const [log, setLog] = useState<Outcome[]>([]);
  const [createDev, setCreateDev] = useState<Dev>(0);
  const [createSize, setCreateSize] = useState(4);
  const [mapChunk, setMapChunk] = useState(1);
  const [mapAt, setMapAt] = useState(0);
  const [accDev, setAccDev] = useState<Dev>(0);
  const [accFrom, setAccFrom] = useState(0);
  const [accTo, setAccTo] = useState(N - 1);
  const [readDev, setReadDev] = useState<Dev>(0);
  const [readFrom, setReadFrom] = useState(0);
  const [readTo, setReadTo] = useState(3);
  const [tdChunk, setTdChunk] = useState(1);

  const doOp = (op: Op) => {
    const r = apply(state, op);
    setState(r.state);
    setLog([...log, r.outcome]);
    if (op.op === 'create' && r.outcome.kind === 'ok') {
      setMapChunk(state.nextId);
      setTdChunk(state.nextId);
    }
  };
  const preset = (ops: Op[]) => {
    const r = run(G, ops);
    setState(r.state);
    setLog(r.log);
  };
  const last = log[log.length - 1];
  const lastRead = [...log].reverse().find((o) => o.op.op === 'read');
  const sl = slots(state);
  const live = state.chunks.filter((c) => !c.released);
  const W = 30 + N * SLOT + 10;
  const pieceAt = (k: number) => (lastRead && last === lastRead ? lastRead.pieces?.find((p) => k * G >= p.offset && k * G < p.offset + p.size) : undefined);

  return (
    <Bench
      title="Reserve, create, map, grant"
      task="Run the book's three calls, find why the kernel crashes, fix it, then tear everything down in the documented order."
      foot={
        <>
          Rules from NVIDIA's guide §4.18 and Driver API reference, and AMD's HIP page (appC-facts.md, A15, B7). Mapping
          over mapped addresses and granting access over unmapped ones are refused by this model; our sources do not say
          what the driver does.
        </>
      }
    >
      <div className="row" role="group" aria-label="Presets" style={{ marginBottom: '0.9rem' }}>
        <span className="muted">Start from</span>
        <button className="quiet" data-preset="three" onClick={() => preset(bookThreeCalls(G))}>
          The book's three calls, then a kernel
        </button>
        <button className="quiet" data-preset="book" onClick={() => preset(bookExample(G))}>
          The book's example: one pointer over two GPUs
        </button>
        <button
          className="quiet"
          data-preset="empty"
          onClick={() => {
            setState(initial(G));
            setLog([]);
          }}
        >
          Nothing
        </button>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${W} 148`} role="img" aria-label="The reserved address range, what is mapped behind it, and which GPU may access it" style={{ width: '100%', maxWidth: W, minWidth: 460, display: 'block' }} data-part="range" data-reserved={state.reserved == null ? 0 : state.reserved / G}>
          <text x={30} y={14} fontSize="11" style={{ fill: 'var(--ink-2)' }}>
            {state.reserved == null ? (state.freed ? 'the range has been freed' : 'no addresses reserved yet') : `one pointer: base + 0 … base + ${state.reserved / G} granules`}
          </text>
          {Array.from({ length: N }, (_, k) => {
            const s = sl[k];
            const p = pieceAt(k);
            const reserved = s != null;
            return (
              <g key={k} data-slot={k} data-device={s?.device ?? ''} data-read={p?.status ?? ''}>
                <rect
                  x={30 + k * SLOT}
                  y={22}
                  width={SLOT - 4}
                  height={40}
                  rx="2"
                  fill={!reserved ? 'transparent' : s.device === 0 ? 'var(--global)' : s.device === 1 ? 'var(--global-wash)' : 'var(--cell)'}
                  stroke={p && (p.status === 'unmapped' || p.status === 'no-access') ? 'var(--fail)' : reserved ? 'var(--global)' : 'var(--rule)'}
                  strokeWidth={p ? 2.5 : 1}
                  strokeDasharray={!reserved ? '2 4' : s.device == null ? '4 3' : undefined}
                  opacity={reserved ? 1 : 0.6}
                />
                <text x={30 + k * SLOT + (SLOT - 4) / 2} y={47} fontSize="11" textAnchor="middle" style={{ fill: s?.device === 0 ? 'var(--page)' : 'var(--ink)' }}>
                  {reserved ? (s.chunk != null ? `h${s.chunk}` : 'empty') : ''}
                </text>
                <text x={30 + k * SLOT + (SLOT - 4) / 2} y={76} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>
                  {k}
                </text>
                {([0, 1] as Dev[]).map((d) => (
                  <text key={d} x={30 + k * SLOT + (SLOT - 4) / 2} y={100 + d * 18} fontSize="12" textAnchor="middle" data-access={`${d}`} data-on={s?.access[d] ? 1 : 0} style={{ fill: s?.access[d] ? 'var(--ok)' : 'var(--ink-3)' }}>
                    {reserved ? (s.access[d] ? '✓' : '·') : ''}
                  </text>
                ))}
                {p ? (
                  <text x={30 + k * SLOT + (SLOT - 4) / 2} y={142} fontSize="10" textAnchor="middle" style={{ fill: p.status === 'unmapped' || p.status === 'no-access' ? 'var(--fail)' : 'var(--ok)' }}>
                    {p.status === 'no-access' ? 'no access' : p.status}
                  </text>
                ) : null}
              </g>
            );
          })}
          <text x={2} y={47} fontSize="10" style={{ fill: 'var(--ink-3)' }}>
            VA
          </text>
          <text x={2} y={104} fontSize="10" style={{ fill: 'var(--ink-3)' }}>
            G0
          </text>
          <text x={2} y={122} fontSize="10" style={{ fill: 'var(--ink-3)' }}>
            G1
          </text>
        </svg>
      </div>
      <p className="muted" style={{ margin: '0.3rem 0 0.8rem' }}>
        Top row: the reserved addresses, one box per granule, filled where a chunk is mapped (solid: GPU 0's memory; pale:
        GPU 1's). Rows G0 and G1: a tick where that GPU has been granted access.
      </p>

      <p className={'narrate' + (last && last.kind !== 'ok' ? ' bad' : '')} data-out="last" data-kind={last?.kind ?? ''} data-code={last?.code ?? ''} aria-live="polite" style={{ margin: '0 0 0.8rem' }}>
        {last ? (
          <>
            <b>{last.call === 'kernel' ? 'Kernel' : <code>{last.call}</code>}:</b> {last.text}
          </>
        ) : (
          'Nothing yet. Reserve a range, or start from one of the presets.'
        )}
      </p>
      <div className="row" style={{ marginBottom: '0.8rem' }} data-part="chunks">
        <span className="muted">Chunks</span>
        {state.chunks.length ? (
          state.chunks.map((c) => {
            const m = state.maps.find((x) => x.chunk === c.id);
            return (
              <span key={c.id} className={'chip ' + (c.released ? '' : 'global')} data-chunk={c.id}>
                h{c.id}: {c.size / G} on GPU {c.device}
                {c.released ? ', released' : m ? `, at ${m.offset / G}` : ', not mapped'}
              </span>
            );
          })
        ) : (
          <span className="muted">none</span>
        )}
      </div>

      <div className="panel" style={{ display: 'grid', gap: '0.55rem', marginBottom: '1rem' }} data-part="controls">
        <div className="row" style={{ gap: '0.4rem 0.6rem' }}>
          <code style={{ width: '11rem' }}>{CALLS.reserve.cuda}</code>
          <button data-act="reserve" onClick={() => doOp({ op: 'reserve', size: N * G })}>
            Reserve {N} granules
          </button>
        </div>
        <div className="row" style={{ gap: '0.4rem 0.6rem' }}>
          <code style={{ width: '11rem' }}>{CALLS.create.cuda}</code>
          <Seg label="Create on" value={createDev} options={[{ v: 0 as Dev, t: 'GPU 0' }, { v: 1 as Dev, t: 'GPU 1' }]} set={setCreateDev} />
          <select aria-label="Chunk size" value={createSize} onChange={(e) => setCreateSize(Number(e.target.value))}>
            {SIZES.map((s) => (
              <option key={s} value={s}>
                {s} granule{s === 1 ? '' : 's'}
              </option>
            ))}
          </select>
          <button data-act="create" onClick={() => doOp({ op: 'create', device: createDev, size: createSize * G })}>
            Create
          </button>
        </div>
        <div className="row" style={{ gap: '0.4rem 0.6rem' }}>
          <code style={{ width: '11rem' }}>{CALLS.map.cuda}</code>
          <select aria-label="Chunk to map" value={mapChunk} onChange={(e) => setMapChunk(Number(e.target.value))}>
            {(live.length ? live : [{ id: 1 }]).map((c) => (
              <option key={c.id} value={c.id}>
                h{c.id}
              </option>
            ))}
          </select>
          <span className="muted">at granule</span>
          <SlotSelect label="Map at granule" value={mapAt} set={setMapAt} />
          <button data-act="map" onClick={() => doOp({ op: 'map', chunk: mapChunk, offset: mapAt * G })}>
            Map
          </button>
        </div>
        <div className="row" style={{ gap: '0.4rem 0.6rem' }}>
          <code style={{ width: '11rem' }}>{CALLS.setAccess.cuda}</code>
          <Seg label="Grant to" value={accDev} options={[{ v: 0 as Dev, t: 'GPU 0' }, { v: 1 as Dev, t: 'GPU 1' }]} set={setAccDev} />
          <SlotSelect label="Grant from granule" value={accFrom} set={setAccFrom} />
          <span className="muted">to</span>
          <SlotSelect label="Grant to granule" value={accTo} set={setAccTo} />
          <button data-act="access" onClick={() => doOp({ op: 'setAccess', device: accDev, offset: accFrom * G, size: Math.max(1, accTo - accFrom + 1) * G })}>
            Grant
          </button>
        </div>
        <div className="row" style={{ gap: '0.4rem 0.6rem' }}>
          <code style={{ width: '11rem' }}>kernel</code>
          <Seg label="Kernel on" value={readDev} options={[{ v: 0 as Dev, t: 'GPU 0' }, { v: 1 as Dev, t: 'GPU 1' }]} set={setReadDev} />
          <span className="muted">reads</span>
          <SlotSelect label="Read from granule" value={readFrom} set={setReadFrom} />
          <span className="muted">to</span>
          <SlotSelect label="Read to granule" value={readTo} set={setReadTo} />
          <button data-act="read" onClick={() => doOp({ op: 'read', device: readDev, offset: readFrom * G, size: Math.max(1, readTo - readFrom + 1) * G })}>
            Launch
          </button>
        </div>
        <div className="row" style={{ gap: '0.4rem 0.6rem' }}>
          <span className="muted" style={{ width: '11rem' }}>
            teardown
          </span>
          <select aria-label="Chunk to tear down" value={tdChunk} onChange={(e) => setTdChunk(Number(e.target.value))}>
            {(state.chunks.length ? state.chunks : [{ id: 1 }]).map((c) => (
              <option key={c.id} value={c.id}>
                h{c.id}
              </option>
            ))}
          </select>
          <button data-act="unmap" onClick={() => doOp({ op: 'unmap', chunk: tdChunk })}>
            {CALLS.unmap.cuda}
          </button>
          <button data-act="release" onClick={() => doOp({ op: 'release', chunk: tdChunk })}>
            {CALLS.release.cuda}
          </button>
          <button data-act="free" onClick={() => doOp({ op: 'free' })}>
            {CALLS.free.cuda}
          </button>
        </div>
      </div>
      <div className="panel">
        <p className="muted" style={{ margin: '0 0 0.4rem' }}>
          Calls so far
        </p>
        {log.length ? (
          <ol style={{ margin: 0, paddingLeft: '1.4rem', maxHeight: '15rem', overflowY: 'auto' }} data-part="log">
            {log.map((o, k) => (
              <li key={k} data-kind={o.kind}>
                <span className={'chip ' + KIND_CHIP[o.kind]}>{KIND_WORD[o.kind]}</span> {opText(o.op)}
              </li>
            ))}
          </ol>
        ) : (
          <p className="muted" style={{ margin: 0 }}>
            none
          </p>
        )}
        {log.length && clean(state) && state.freed ? (
          <p style={{ margin: '0.7rem 0 0' }} data-out="clean">
            <span className="chip ok">clean</span> Everything is unmapped, released and freed.
          </p>
        ) : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- growing a KV cache
const MAX_TOKENS = 4096;
const CHUNKS_MIB = [2, 8, 32, 128];
const CW = 720;
const CH = 210;

function GrowKV() {
  const [tokens, setTokens] = useState(1500);
  const [kib, setKib] = useState(LLAMA2_7B_BYTES_PER_TOKEN / 1024);
  const [chunkMiB, setChunkMiB] = useState(2);
  const [strategy, setStrategy] = useState<Strategy>('double');

  const bytesPerToken = Math.max(1, kib) * 1024;
  const chunk = chunkMiB * MiB;
  const reserve = Math.ceil((MAX_TOKENS * bytesPerToken) / chunk) * chunk;
  const results = useMemo(() => {
    const o = { tokens, bytesPerToken, chunk, reserve };
    return Object.fromEntries(STRATEGIES.map((s) => [s.key, grow(s.key, o)])) as Record<Strategy, ReturnType<typeof grow>>;
  }, [tokens, bytesPerToken, chunk, reserve]);
  const r = results[strategy];
  const top = Math.max(...STRATEGIES.map((s) => results[s.key].peak));
  const X = (t: number) => 56 + ((t - 1) / Math.max(1, tokens - 1)) * (CW - 72);
  const Y = (b: number) => CH - 22 - (b / top) * (CH - 66);
  const stride = Math.max(1, Math.floor(r.steps.length / 600));
  const sampled = r.steps.filter((s, k) => k % stride === 0 || s.moved || k === r.steps.length - 1);
  const allocPath = sampled.map((s, k) => `${k ? 'L' : 'M'} ${X(s.token).toFixed(1)} ${Y(s.allocated).toFixed(1)}`).join(' ');
  const livePath = `M ${X(1)} ${Y(r.steps[0].live)} L ${X(tokens)} ${Y(r.finalLive)}`;
  const spikes = r.steps.filter((s) => s.moved);
  const showSpikes = spikes.length <= 200 ? spikes : spikes.filter((_, k) => k % Math.ceil(spikes.length / 200) === 0);

  return (
    <Bench
      title="Growing a KV cache"
      task="Compare the three strategies at 4096 tokens: what is copied, the most memory held at once, and whether the pointer moves."
      foot={
        <>
          The default is the book's Llama 2 7B example at 16 bits: 2 × 32 layers × hidden dimension 4096 × 2 bytes = 512 KiB per token (Eq.
          20.7, ch. 20 §20.6, pp. 505–506). The 2 MiB chunk is the size PyTorch's allocator comment gives for its
          mappings (appC-facts.md, D2). The reservation covers {MAX_TOKENS} tokens.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field" style={{ flex: '1 1 16rem' }}>
          Tokens generated: {tokens.toLocaleString('en-US')}
          <input type="range" min={1} max={MAX_TOKENS} value={tokens} aria-label="Tokens generated" onChange={(e) => setTokens(Number(e.target.value))} />
        </label>
        <label className="field">
          KV cache per token (KiB)
          <input type="number" min={1} step={1} value={kib} aria-label="KV cache per token in KiB" onChange={(e) => setKib(Math.max(1, Math.floor(Number(e.target.value) || 1)))} />
        </label>
        <label className="field">
          Chunk
          <select value={chunkMiB} aria-label="Chunk size" onChange={(e) => setChunkMiB(Number(e.target.value))}>
            {CHUNKS_MIB.map((c) => (
              <option key={c} value={c}>
                {c} MiB
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <Seg label="Strategy" value={strategy} options={STRATEGIES.map((s) => ({ v: s.key, t: s.label }))} set={setStrategy} />
      </div>

      <div className="panel" style={{ overflowX: 'auto', marginBottom: '1rem' }}>
        <svg viewBox={`0 0 ${CW} ${CH}`} role="img" aria-label="Memory allocated and data held as tokens are generated" style={{ width: '100%', minWidth: 460, display: 'block' }} data-part="chart" data-strategy={strategy}>
          <line x1={56} x2={CW - 16} y1={Y(0)} y2={Y(0)} stroke="var(--rule)" />
          <line x1={56} x2={56} y1={Y(0)} y2={Y(top)} stroke="var(--rule)" />
          <line x1={56} x2={CW - 16} y1={Y(top)} y2={Y(top)} stroke="var(--rule-soft)" />
          <text x={52} y={Y(top) + 4} fontSize="12" textAnchor="end" style={{ fill: 'var(--ink-3)' }}>
            {binary(top).text}
          </text>
          <text x={52} y={Y(0) + 4} fontSize="12" textAnchor="end" style={{ fill: 'var(--ink-3)' }}>
            0
          </text>
          <text x={CW - 16} y={CH - 4} fontSize="12" textAnchor="end" style={{ fill: 'var(--ink-3)' }}>
            tokens: 1 to {tokens.toLocaleString('en-US')}
          </text>
          {showSpikes.map((s) => (
            <line key={s.token} x1={X(s.token)} x2={X(s.token)} y1={Y(s.allocated)} y2={Y(s.during)} stroke="var(--global)" strokeDasharray="3 2" strokeWidth="1.5" />
          ))}
          <path d={livePath} fill="none" stroke="var(--ink-2)" strokeWidth="1.5" />
          <path d={allocPath} fill="none" stroke="var(--global)" strokeWidth="2.2" />
          <text x={64} y={16} fontSize="12.5" fontWeight="600" style={{ fill: 'var(--global)' }}>
            {strategy === 'vmm' ? 'mapped' : 'allocated'}
            {strategy === 'vmm' ? '' : '; dashed: old and new together, during a copy'}
          </text>
          <text x={64} y={33} fontSize="12.5" style={{ fill: 'var(--ink-2)' }}>
            data held
          </text>
        </svg>
        <p className="ui muted" style={{ margin: '0.5rem 0 0' }} data-out="note">
          {strategy === 'double'
            ? r.peakRatio
              ? `During each copy, old and new together are up to ${Number(r.peakRatio.toPrecision(3))} times the data being copied.`
              : 'No copy yet: the first buffer still has room.'
            : strategy === 'exact'
              ? 'Every token copies everything held so far, so the total copied grows with the square of the length.'
              : `Reserved addresses: ${binary(r.reserved ?? 0).text}, with memory behind ${binary(r.finalAllocated).text} of them.`}
        </p>
      </div>
      <div style={{ overflowX: 'auto', marginTop: '1rem' }}>
        <table className="data" style={{ margin: 0 }} data-part="compare">
          <thead>
            <tr>
              <th>
                All three, at {tokens.toLocaleString('en-US')} tokens (<span data-out="live">{binary(r.finalLive).text}</span> of data)
              </th>
              <th className="num">Copied (× the data)</th>
              <th className="num">Most at once</th>
              <th className="num">Unused now</th>
              <th className="num">Pointer changes</th>
            </tr>
          </thead>
          <tbody>
            {STRATEGIES.map((s) => {
              const x = results[s.key];
              return (
                <tr
                  key={s.key}
                  data-strategy={s.key}
                  data-copied={x.copiedTotal}
                  data-peak={x.peak}
                  data-unused={x.unused}
                  data-moves={x.pointerChanges}
                  data-selected={s.key === strategy ? 1 : 0}
                  onClick={() => setStrategy(s.key)}
                  style={{ cursor: 'pointer', background: s.key === strategy ? 'var(--rule-soft)' : undefined }}
                >
                  <td style={{ fontWeight: s.key === strategy ? 600 : 400 }}>{s.label}</td>
                  <td className="num">
                    {binary(x.copiedTotal).text}
                    {x.copiedTotal ? <span className="muted"> ({Number(x.copyRatio.toPrecision(3))} ×)</span> : null}
                  </td>
                  <td className="num">{binary(x.peak).text}</td>
                  <td className="num">{binary(x.unused).text}</td>
                  <td className="num">{x.pointerChanges.toLocaleString('en-US')}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSC9: Record<string, Question> = {
  'appC/C.9/q-setaccess': {
    kind: 'choice',
    id: 'appC/C.9/q-setaccess',
    prompt: (
      <>
        You reserved a range, created memory on GPU 0 and mapped it at offset 0. The first kernel on GPU 0 that reads it
        crashes. What is missing?
      </>
    ),
    options: [
      { label: 'cudaDeviceSynchronize after the map.', why: 'Nothing here is asynchronous. The mapping is in place; the device just may not use it.' },
      { label: 'A second cuMemMap.', why: 'The memory is already behind the addresses. Mapping it again changes nothing about who may use it.' },
      {
        label: 'cuMemSetAccess, granting GPU 0 read and write access to that range.',
        correct: true,
      },
      { label: 'cudaDeviceEnablePeerAccess.', why: 'That is for loads into another GPU\'s ordinary allocations, and the memory here is on GPU 0 itself.' },
    ],
    explain: (
      <p>
        NVIDIA's guide: mapping "does not make the address accessible, and would result in a program crash if accessed by
        a CUDA kernel". The book's list of three calls leaves the fourth out. Address, memory and access are three
        separate things, each with its own call.
      </p>
    ),
    source: 'CUDA guide §4.18.3.4, checked 2026-11-20 (notes/today/appC-facts.md, A15); the book, p. 629',
  },
  'appC/C.9/q-copied': {
    kind: 'numeric',
    id: 'appC/C.9/q-copied',
    prompt: (
      <>
        A buffer grows one chunk at a time from 1 chunk to 16 chunks. It starts with a capacity of 1 chunk and doubles its
        capacity whenever it is full, copying its contents into the new buffer each time. How many chunks are copied in
        all?
      </>
    ),
    answer: 15,
    unit: 'chunks',
    explain: (
      <p>
        It moves when the 2nd, 3rd, 5th and 9th chunk arrive, copying 1, 2, 4 and 8 chunks: 15, one less than the final
        size. With a reserved range and one chunk mapped at a time, the answer is 0. To see it in the explorer, set 2048
        KiB per token, a 2 MiB chunk and 16 tokens: 30 MiB copied, which is 15 chunks.
      </p>
    ),
    source: 'Ours: the arithmetic of doubling; src/sim/vmmgrow.ts',
  },
  'appC/C.9/q-pointer': {
    kind: 'choice',
    id: 'appC/C.9/q-pointer',
    prompt: (
      <>
        A buffer lives in a reserved range. You map a new chunk right after the last one. Do kernels that hold the
        buffer's base pointer need a new pointer?
      </>
    ),
    options: [
      { label: 'Yes, after every growth.', why: 'That is the copying strategy, where the data moves to a new allocation.' },
      {
        label: 'No. The base address does not change; only more of the range now has memory behind it.',
        correct: true,
      },
      { label: 'Only when the new chunk is on another GPU.', why: 'The address is still inside the same reserved range. One pointer can span chunks on different devices, which is the book\'s example.' },
    ],
    explain: (
      <p>
        Reserving first is what makes this work. NVIDIA describes the use as growing an allocation the way realloc or{' '}
        <code>std::vector</code> grows CPU memory, without moving data. The new chunk does need its own access grant.
      </p>
    ),
    source: 'NVIDIA, "Introducing Low-Level GPU Virtual Memory Management" and guide §4.18, checked 2026-11-20 (appC-facts.md, A15)',
  },
  'appC/C.9/q-peer': {
    kind: 'choice',
    id: 'appC/C.9/q-peer',
    prompt: (
      <>
        PyTorch runs with <code>expandable_segments</code> on a two-GPU machine, and the program relies on{' '}
        <code>cudaDeviceEnablePeerAccess</code> for direct loads between the GPUs. What does PyTorch's own source warn?
      </>
    ),
    options: [
      { label: 'Nothing changes: peer access covers every allocation on the peer.', why: 'Memory mapped with cuMemMap is not ordinary device memory.' },
      { label: 'expandable_segments turns peer access on by itself.', why: 'The comment says the opposite kind of thing: the runtime peer call does not reach this memory.' },
      {
        label: 'cudaDeviceEnablePeerAccess does not work for memory allocated with cuMemMap; a peer must be granted access through VMM, with cuMemSetAccess.',
        correct: true,
      },
      { label: 'Peer loads are faster with VMM memory.', why: 'The comment is about whether access works at all, not speed.' },
    ],
    explain: (
      <p>
        The allocator's design comment lists it as a limit: runtime APIs for sharing memory such as{' '}
        <code>cudaDeviceEnablePeerAccess</code> "do not work for memory allocated with cuMemMap". It is the other half of{' '}
        <a href={href('appC', 'C.5')}>§C.5</a>: with VMM, access is granted per range with <code>cuMemSetAccess</code>.
      </p>
    ),
    source: 'PyTorch c10/cuda/CUDACachingAllocator.cpp, checked 2026-11-20 (appC-facts.md, D2, A8)',
  },
};

// ---------------------------------------------------------------- the section
export function SC9() {
  return (
    <>
      <p>
        The appendix ends with control over the address space itself. In the book's account, CUDA 11 added three
        low-level calls. <code>cuMemAddressReserve</code> reserves a range of virtual addresses. <code>cuMemCreate</code>{' '}
        allocates physical memory on any device. <code>cuMemMap</code> puts that memory behind any position in the
        reserved range. Together they allow layouts that a single <code>cudaMalloc</code> cannot give: for example, one 3D
        volume whose parts live on several devices but which a kernel reaches through one pointer. The book gives no year
        and no code.
      </p>
      <Source section="C.9" pages="629" />
      <p>
        Two corrections. The calls arrived in CUDA 10.2, in November 2019, not CUDA 11. And the list is one call short:
        mapping memory into a range does not make it usable. A kernel that touches it crashes until{' '}
        <code>cuMemSetAccess</code> grants the device access. Teardown runs in reverse: unmap, release the memory, free the
        addresses. NVIDIA says these calls require the Driver API directly, and a device says whether it supports them
        through <code>CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED</code>.
      </p>
      <Today checked="2026-11-20 against NVIDIA's VMM post, the CUDA 13.5 guide §4.18 and the Driver API reference (appC-facts.md, A15)">
        <p>CUDA 10.2 (November 2019), not CUDA 11.</p>
        <p>"cuMemMap does not make the address accessible": <Id>cuMemSetAccess</Id> is required.</p>
      </Today>
      <Code
        tabs={[
          { label: 'CUDA Driver API', code: SIGS_CUDA, status: 'quoted from the reference; not compiled here: unverified' },
          { label: 'HIP', code: SIGS_HIP, status: 'quoted from the ROCm 10.2 header; not compiled here: unverified' },
        ]}
        caption={
          <>
            labs/appC/snippets/c9_vmm_decls.h and c9_vmm_decls_hip.h: declarations as NVIDIA's Driver API reference
            (v13.5) and AMD's ROCm 10.2 header give them, recorded in notes/today/appC-facts.md (A15, B7); the comments
            are ours. Nothing on this page calls them: the course's CUDA front-end check has no Driver API, and no lab
            uses these calls.
          </>
        }
      />

      <h2>Reserve, create, map, grant</h2>
      <p>
        The stepper gives you a range of {N} granules and the memory of two GPUs. A granule is the unit these calls work
        in; here it is a parameter, not a value from your device. Start from the book's three calls: they reserve, create
        and map, and then a kernel reads. Find out why it crashes and make it work. Then load the book's own example, a
        range whose first half is backed by GPU 0 and second half by GPU 1, and launch a kernel on each GPU.
      </p>
      <Simplification>
        <p>
          The granule is a parameter, not a device value. Only the documented order is enforced, not what a driver does
          out of it. No timing. A chunk is mapped at one place only.
        </p>
      </Simplification>
      <VmmStepper />
      <Ask q={questionsSC9['appC/C.9/q-setaccess']} />
      <p>
        Three resources, then, each with its own call: addresses, memory and access. Keeping them apart is what lets one
        pointer span two devices, and also what lets a buffer grow without moving.
      </p>

      <h2>A buffer that grows without moving</h2>
      <p>
        NVIDIA now names the use that matters most to you: allocators and caches "such as dynamically managing key-value
        caches in large language models". A KV cache gains one entry per generated token, and its final size is not known
        when generation starts. With ordinary allocations, growth means a new, larger buffer and a copy. With a reserved
        range, it means mapping one more chunk after the last.
      </p>
      <Today checked="2026-11-20 against NVIDIA's VMM post and guide §4.18 (appC-facts.md, A15)">
        <p>NVIDIA compares the growth to realloc or std::vector, without moving data, and says NCCL and NVSHMEM use VMM.</p>
      </Today>
      <GrowBufferScene />
      <p>
        The explorer runs all three strategies on a real cache size: the book's Llama 2 7B example, 512 KiB of keys and
        values per token at 16 bits, so 4096 tokens hold 2 GiB. Commit to a number before you open it.
      </p>
      <Predict
        id="appC/C.9/p-double-copy"
        prompt={
          <>
            The cache grows to 4096 tokens of 512 KiB each, 2 GiB in all. It starts as one 2 MiB buffer and doubles
            whenever it is full, copying everything it holds each time. How many GiB does it copy over the whole run?
          </>
        }
        unit="GiB"
      >
        {(v) => {
          const d = grow('double', { tokens: 4096, bytesPerToken: LLAMA2_7B_BYTES_PER_TOKEN, chunk: 2 * MiB });
          const gib = d.copiedTotal / 1024 ** 3;
          return (
            <p>
              The model copies {d.copiedTotal / MiB} MiB, which is {Number(gib.toFixed(3))} GiB: 2 + 4 + … + 1024 MiB,
              one chunk short of the final size.{' '}
              {Math.abs(v - gib) < 0.25
                ? 'That is what you had.'
                : v > 4
                  ? 'Each copy moves only what the buffer held at that moment, and those amounts halve going back in time, so they add up to less than the final size.'
                  : 'Each doubling copies as much as all the earlier copies together, plus one chunk, so the total lands just under the final size.'}{' '}
              Exact fit would copy about 4 TiB; reserve-and-map copies nothing.
            </p>
          );
        }}
      </Predict>
      <Simplification>
        <p>
          One sequence on one device. Allocation and mapping are not timed, and there are no other allocations to
          fragment memory. Addresses for the copying strategies come from an allocator that never reuses a freed hole.
        </p>
      </Simplification>
      <GrowKV />
      <Ask q={questionsSC9['appC/C.9/q-copied']} />
      <Ask q={questionsSC9['appC/C.9/q-pointer']} />

      <h2>Where you meet it</h2>
      <p>
        PyTorch's CUDA caching allocator has an option that does this. With <code>expandable_segments</code> set in{' '}
        <code>PYTORCH_CUDA_ALLOC_CONF</code> (experimental, off by default), it reserves address space for about the whole
        GPU's memory and maps physical memory into it as the program needs it, through <code>cuMemCreate</code>,{' '}
        <code>cuMemAddressReserve</code>, <code>cuMemMap</code> and <code>cuMemSetAccess</code>, at a 2 MiB granularity
        by its own comment. The same comment states the price: the runtime's peer-access call does not reach memory mapped
        this way.
      </p>
      <Beyond>
        <p>PyTorch: c10/cuda/CUDACachingAllocator.cpp, design comment (appC-facts.md, D2).</p>
        <p>
          llama.cpp's CUDA backend keeps a VMM pool that reserves 32 GB of addresses, on by default for CUDA and off by
          default for HIP (<Id>GGML_HIP_NO_VMM</Id>, D3).
        </p>
      </Beyond>
      <Ask q={questionsSC9['appC/C.9/q-peer']} />
      <p>
        The book itself takes the same problem on inside the kernel. Its chapter on attention describes PagedAttention,
        which splits the KV cache into equal-size blocks because a cache reserved for the longest possible input leaves
        memory fragmented and wasted. VMM keeps one contiguous range of addresses and fills it in; PagedAttention, inspired
        by an operating system's paging, works in fixed-size blocks that are brought into GPU memory when the
        computation needs them. See{' '}
        <a href={href('ch20', '20.7')}>chapter 20, §20.7</a>.
      </p>
      <Source section="20.7" pages="509" />

      <h2>On AMD and Apple GPUs</h2>
      <p>
        HIP has the same four steps under runtime-API names, <code>hipMemAddressReserve</code>, <code>hipMemCreate</code>,{' '}
        <code>hipMemMap</code> and <code>hipMemSetAccess</code>, built on HSA, on Linux (AMD lists Windows as under
        development). A size that is not aligned makes <code>hipMemCreate</code> return an out-of-memory error.
        llama.cpp's HIP build maps each <code>cuMem</code> call to its <code>hipMem</code> name.
      </p>
      <Beyond>
        <p>Checked 2026-11-20 against AMD's HIP virtual memory page, the ROCm 10.2 header and llama.cpp's vendors/hip.h (appC-facts.md, B7, D3).</p>
      </Beyond>
      <p>
        Metal 4 has a close analogue for one device, though nothing that spans devices. A placement sparse buffer plays
        the reserved range, a placement heap plays the memory from <code>cuMemCreate</code>, and{' '}
        <code>updateBufferMappings</code> on an <code>MTL4CommandQueue</code> plays <code>cuMemMap</code>. There is no
        access step. The mapping is ours and we have not run it. One trap: the header gives the buffer constructor and
        the queue method as macOS 26.0, but <code>supportsPlacementSparse</code> as 26.4, so check that property at run
        time before relying on any of it.
      </p>
      <Beyond>
        <p>
          Metal: <Id>newBufferWithLength:options:placementSparsePageSize:</Id>, placement <Id>MTLHeap</Id>,{' '}
          <Id>updateBufferMappings:heap:operations:count:</Id>. Checked 2026-11-20 against the macOS 27.0 SDK headers
          (appC-facts.md, C6).
        </p>
      </Beyond>
      <p>
        Back to <a href={href('appC', 'start')}>the map</a>: this step is C.9. A kernel can use a mapped range only once
        access is granted, and our sources do not say whether host code may dereference one.
      </p>
    </>
  );
}
