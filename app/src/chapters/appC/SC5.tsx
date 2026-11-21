import { useState, type ReactNode } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import {
  MAX_PEERS_WITHOUT_NVSWITCH,
  allToAll,
  canEnable,
  deref,
  emptyGrants,
  enable,
  enableAll,
  haloExchange,
  outgoing,
  revoke,
  type Grants,
  type PeerSystem,
} from '../../sim/peeraccess';

/** A long identifier that may break anywhere, for the narrow margin notes. */
function Id({ children }: { children: ReactNode }) {
  return <code style={{ whiteSpace: 'normal', overflowWrap: 'anywhere' }}>{children}</code>;
}

const COUNTS = [2, 4, 8, 9, 16];

// ---------------------------------------------------------------- the explorer
function PeerMatrix() {
  const [sys, setSys] = useState<PeerSystem>({ n: 4, nvswitch: false, iommu: false });
  const [g, setG] = useState<Grants>(() => emptyGrants(4));
  // 'neutral' for a grant the model leaves out because the rule is not documented.
  const [status, setStatus] = useState<{ tone: 'ok' | 'bad' | 'neutral'; text: string } | null>(null);
  const [reader, setReader] = useState(0);
  const [owner, setOwner] = useState(1);
  const [approach, setApproach] = useState<'copy' | 'peer'>('copy');

  const change = (next: PeerSystem) => {
    setSys(next);
    setG(emptyGrants(next.n));
    setStatus({ tone: 'ok', text: 'The system changed, so every grant was cleared.' });
    if (reader >= next.n) setReader(0);
    if (owner >= next.n) setOwner(1);
  };
  const click = (i: number, j: number) => {
    if (i === j) return;
    if (g[i][j]) {
      setG(revoke(g, i, j));
      setStatus({ tone: 'ok', text: `Taken back: GPU ${i} no longer loads from GPU ${j}.` });
      return;
    }
    const r = enable(sys, g, i, j);
    setG(r.grants);
    setStatus(
      r.result.ok
        ? { tone: 'ok', text: `${r.result.call}: GPU ${i} may now load from GPU ${j}'s memory. GPU ${j} may not load from GPU ${i}'s.` }
        : { tone: r.result.reason === 'undocumented' ? 'neutral' : 'bad', text: r.result.text },
    );
  };
  const all = () => {
    const r = enableAll(sys);
    setG(r.grants);
    setStatus(
      r.unknown
        ? {
            tone: 'neutral',
            text: `Tried every pair: ${r.granted} grants succeeded; for ${r.unknown} the rule on this system is not documented, so the model granted none of them.`,
          }
        : r.refused
          ? { tone: 'bad', text: `Tried every pair: ${r.granted} grants succeeded and ${r.refused} were refused.` }
          : { tone: 'ok', text: `Tried every pair: all ${r.granted} grants succeeded.` },
    );
  };
  const d = deref(g, reader, owner);
  const whole = allToAll(sys);
  const halo = haloExchange(approach, g);
  const cell = sys.n > 9 ? 26 : 34;

  return (
    <Bench
      title="Peer access matrix"
      task="Make every GPU able to load from every other, then find the system where you cannot."
      foot={
        <>
          The rules are NVIDIA's, from the CUDA guide §3.4.2 and the runtime reference (appC-facts.md, A8): grants are one
          way, at most {MAX_PEERS_WITHOUT_NVSWITCH} peers per device without NVSwitch, no PCIe peer-to-peer on bare-metal
          Linux with the IOMMU on. The halo steps are our reading of the book's comparison (p. 626).
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          GPUs
          <select value={sys.n} aria-label="Number of GPUs" onChange={(e) => change({ ...sys, n: Number(e.target.value) })}>
            {COUNTS.map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>
        <label className="check">
          <input type="checkbox" checked={sys.nvswitch} onChange={(e) => change({ ...sys, nvswitch: e.target.checked })} /> NVSwitch
        </label>
        <label className="check">
          <input type="checkbox" checked={sys.iommu} onChange={(e) => change({ ...sys, iommu: e.target.checked })} /> bare-metal Linux,
          IOMMU on
        </label>
        <button className="quiet" onClick={all}>
          Enable every pair
        </button>
        <button
          className="quiet"
          onClick={() => {
            setG(emptyGrants(sys.n));
            setStatus(null);
          }}
        >
          Clear
        </button>
      </div>

      <div className="cols plot">
        <div className="panel" style={{ overflowX: 'auto' }}>
          <p className="muted" style={{ margin: '0 0 0.5rem' }}>
            Row: the GPU whose kernels load. Column: the GPU whose memory they load from. Click a cell to call{' '}
            <code>cudaDeviceEnablePeerAccess</code> on the row's GPU; click again to take the grant back.
          </p>
          <table style={{ borderCollapse: 'collapse', fontFamily: 'var(--sans)', fontSize: '0.74rem' }} data-part="matrix" data-n={sys.n}>
            <thead>
              <tr>
                <th style={{ padding: '0 0.4rem 0.2rem 0', textAlign: 'right', fontWeight: 500 }} className="muted">
                  from \ to
                </th>
                {g.map((_, j) => (
                  <th key={j} style={{ width: cell, textAlign: 'center', fontWeight: 600 }}>
                    {j}
                  </th>
                ))}
                <th style={{ paddingLeft: '0.5rem', textAlign: 'left', fontWeight: 500 }} className="muted">
                  peers
                </th>
              </tr>
            </thead>
            <tbody>
              {g.map((row, i) => (
                <tr key={i}>
                  <th style={{ padding: '0 0.4rem 0 0', textAlign: 'right', fontWeight: 600 }}>{i}</th>
                  {row.map((on, j) => {
                    const self = i === j;
                    const res = self || on ? null : canEnable(sys, g, i, j);
                    const unknown = !!res && !res.ok && res.reason === 'undocumented';
                    const blocked = !!res && !res.ok && !unknown;
                    return (
                      <td key={j} style={{ padding: 1 }}>
                        <button
                          data-cell={`${i}-${j}`}
                          data-on={on ? 1 : 0}
                          disabled={self}
                          aria-label={self ? `GPU ${i}, own memory` : `GPU ${i} loads from GPU ${j}: ${on ? 'enabled' : unknown ? 'not documented' : 'not enabled'}`}
                          onClick={() => click(i, j)}
                          style={{
                            width: cell,
                            height: cell - 4,
                            padding: 0,
                            borderRadius: 2,
                            fontWeight: 600,
                            border: '1px solid ' + (on ? 'var(--ok)' : blocked ? 'var(--fail)' : 'var(--rule)'),
                            background: self ? 'var(--rule-soft)' : on ? 'var(--ok-wash)' : 'var(--cell)',
                            color: on ? 'var(--ok)' : blocked ? 'var(--fail)' : 'var(--ink-3)',
                            opacity: 1,
                          }}
                        >
                          {self ? 'own' : on ? '✓' : blocked ? '×' : unknown ? '?' : ''}
                        </button>
                      </td>
                    );
                  })}
                  <td style={{ paddingLeft: '0.5rem' }} data-peers={outgoing(g, i)}>
                    {outgoing(g, i)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p data-out="status" className={status?.tone === 'bad' ? 'narrate bad' : 'narrate'} style={{ margin: '0.7rem 0 0' }} aria-live="polite">
            {status ? status.text : 'No grants yet. Every GPU can load from its own memory only.'}
          </p>
          <p className="muted" style={{ margin: '0.5rem 0 0' }} data-out="all-to-all" data-possible={whole === 'possible' ? 1 : 0} data-all={whole}>
            Every GPU loading from every other needs {sys.n - 1} peer{sys.n === 2 ? '' : 's'} each:{' '}
            {whole === 'unknown'
              ? 'not documented here. NVIDIA\'s IOMMU rule is about PCIe peer-to-peer, and our sources say nothing about NVSwitch with the IOMMU on.'
              : sys.iommu
                ? 'impossible here, since no PCIe peer grant is allowed at all.'
                : whole === 'possible'
                  ? sys.nvswitch
                    ? 'possible, since NVSwitch lifts the limit.'
                    : `possible, within the limit of ${MAX_PEERS_WITHOUT_NVSWITCH}.`
                  : `impossible without NVSwitch, which allows ${MAX_PEERS_WITHOUT_NVSWITCH}.`}
          </p>
        </div>

        <div>
          <div className="panel" style={{ marginBottom: '1rem' }}>
            <p style={{ margin: '0 0 0.5rem' }}>
              <b>A kernel follows a pointer</b>
            </p>
            <div className="row" style={{ marginBottom: '0.5rem' }}>
              <label className="field">
                Kernel on GPU
                <select value={reader} aria-label="Kernel on GPU" onChange={(e) => setReader(Number(e.target.value))}>
                  {g.map((_, i) => (
                    <option key={i} value={i}>
                      {i}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                reads memory of GPU
                <select value={owner} aria-label="Memory of GPU" onChange={(e) => setOwner(Number(e.target.value))}>
                  {g.map((_, i) => (
                    <option key={i} value={i}>
                      {i}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <p data-out="deref" data-how={d.how} style={{ margin: 0 }}>
              <span className={'chip ' + (d.ok ? 'ok' : 'fail')}>{d.how === 'local' ? 'local' : d.how === 'peer' ? 'peer load' : 'error'}</span>{' '}
              {d.text}
            </p>
          </div>

          <div className="panel">
            <p style={{ margin: '0 0 0.5rem' }}>
              <b>One halo exchange, GPU 0 and GPU 1</b>
            </p>
            <span className="seg" role="group" aria-label="Halo exchange">
              <button aria-pressed={approach === 'copy'} onClick={() => setApproach('copy')}>
                host-driven copies
              </button>
              <button aria-pressed={approach === 'peer'} onClick={() => setApproach('peer')}>
                peer loads
              </button>
            </span>
            <ol style={{ margin: '0.7rem 0 0.6rem', paddingLeft: '1.3rem' }} data-part="halo-steps">
              {halo.steps.map((s, k) => (
                <li key={k} data-kind={s.kind}>
                  {s.text}
                  {s.kind === 'order' ? <span className="muted"> (ours: the book does not discuss it)</span> : null}
                </li>
              ))}
            </ol>
            <table className="data" style={{ margin: 0 }} data-out="halo" data-staging={halo.staging} data-waits={halo.hostWaits} data-copies={halo.hostCopies}>
              <tbody>
                <tr>
                  <td>Staging buffers for copied rows</td>
                  <td className="num">{halo.staging}</td>
                </tr>
                <tr>
                  <td>Copies the host sets up</td>
                  <td className="num">{halo.hostCopies}</td>
                </tr>
                <tr>
                  <td>Times the host waits for a GPU</td>
                  <td className="num">{halo.hostWaits}</td>
                </tr>
                <tr>
                  <td>Ordering points between the GPUs</td>
                  <td className="num">{halo.ordering}</td>
                </tr>
              </tbody>
            </table>
            {approach === 'peer' ? (
              <p style={{ margin: '0.6rem 0 0' }} data-out="halo-ready" data-ready={halo.ready ? 1 : 0}>
                {halo.ready ? (
                  <>
                    <span className="chip ok">ready</span> GPU 0 and GPU 1 each have a grant to the other.
                  </>
                ) : (
                  <>
                    <span className="chip fail">not ready</span> Missing in the matrix:{' '}
                    {halo.missing.map(([a, b]) => `GPU ${a} → GPU ${b}`).join(' and ')}. A halo exchange needs a grant in each
                    direction.
                  </>
                )}
              </p>
            ) : null}
          </div>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSC5: Record<string, Question> = {
  'appC/C.5/q-oneway': {
    kind: 'choice',
    id: 'appC/C.5/q-oneway',
    prompt: (
      <>
        With GPU 0 as the current device, a program calls <code>cudaDeviceEnablePeerAccess(1, 0)</code>. Which loads
        now work?
      </>
    ),
    options: [
      {
        label: 'Kernels on GPU 0 may load from GPU 1, and kernels on GPU 1 may load from GPU 0: peer access is symmetric.',
        why: 'The grant runs one way only. The reverse needs its own call, made on GPU 1.',
      },
      {
        label: "GPU 0's kernels may load from GPU 1's memory; the reverse needs its own call with GPU 1 current.",
        correct: true,
      },
      {
        label: "Kernels on GPU 1 may load from GPU 0's memory: the argument names the GPU being given access.",
        why: 'The argument names the peer whose memory becomes reachable. The device being given access is the current one, GPU 0.',
      },
      {
        label: 'Any GPU in the system may now load from any other.',
        why: 'One call grants one direction between two devices, nothing more.',
      },
    ],
    explain: (
      <p>
        NVIDIA's reference says the access granted by this call "is unidirectional". The pointer value through which a
        kernel reaches the peer's memory is the same on both devices, so nothing else about the code changes.
      </p>
    ),
    source: 'CUDA runtime reference, checked 2026-11-20 (notes/today/appC-facts.md, A8)',
  },
  'appC/C.5/q-eight': {
    kind: 'choice',
    id: 'appC/C.5/q-eight',
    prompt: (
      <>
        A system has sixteen GPUs and no NVSwitch. Every GPU should be able to load directly from every other. Can{' '}
        <code>cudaDeviceEnablePeerAccess</code> set that up?
      </>
    ),
    options: [
      { label: 'Yes: one call per ordered pair, 240 calls in all, each made with the loading GPU current.', why: 'Each GPU would need 15 peers, and the limit without NVSwitch is eight.' },
      { label: 'Yes, if each pair is enabled in both directions, with two calls per pair.', why: 'Both directions are needed for two-way loads, but that does not lift the limit of eight peers per device.' },
      {
        label: 'No. Each GPU would need 15 peers, and without NVSwitch a device supports at most eight. Use NVSwitch, VMM grants or copies.',
        correct: true,
      },
      { label: 'Yes on Windows, where the driver lifts the peer limit; not on Linux.', why: 'The guide states the limit for every system without NVSwitch, whatever the operating system. The one platform rule it documents is of another kind: no PCIe peer-to-peer on bare-metal Linux with the IOMMU on.' },
    ],
    explain: (
      <p>
        Try it in the matrix: set sixteen GPUs and press "Enable every pair". Each row stops at eight grants. With
        NVSwitch on, all 240 go through. NVIDIA also suggests the virtual memory management calls of{' '}
        <a href={href('appC', 'C.9')}>§C.9</a> as the more scalable way to grant peers access to particular memory.
      </p>
    ),
    source: 'CUDA guide §3.4.2.2, checked 2026-11-20 (notes/today/appC-facts.md, A8)',
  },
  'appC/C.5/q-removes': {
    kind: 'choice',
    id: 'appC/C.5/q-removes',
    prompt: (
      <>
        Compared with host-triggered device-to-device copies, what does loading the neighbour's halo directly in the
        kernel remove?
      </>
    ),
    options: [
      {
        label: "The staging copy and the host's copy set-up and synchronisation.",
        correct: true,
      },
      {
        label: "All synchronisation between the GPUs, since each kernel reads its neighbour's rows when it needs them.",
        why: 'A GPU must still not read its neighbour\'s rows before the neighbour has written them. Some ordering between the devices remains.',
      },
      {
        label: 'The traffic over the interconnect, because the kernel loads the halo instead of copying it.',
        why: 'The halo still crosses the link; each peer load goes over it. What disappears is the extra copy and the host\'s part in it.',
      },
    ],
    explain: (
      <p>
        The book lists exactly these costs of the copy approach: extra memory to hold the copied data, and the overhead of
        host-device synchronisation and copy set-up. That ordering between the devices is still needed is our note; the
        book does not discuss it.
      </p>
    ),
    source: 'App. C §C.5, p. 626',
  },
  'appC/C.5/q-metal': {
    kind: 'choice',
    id: 'appC/C.5/q-metal',
    prompt: (
      <>
        You are porting a CUDA inference server that splits a model's layers between two GPUs: after{' '}
        <code>cudaDeviceEnablePeerAccess</code>, GPU 1's kernels read GPU 0's activations through a peer pointer. The
        target is a Mac with Apple silicon. What becomes of the peer-access path?
      </>
    ),
    options: [
      {
        label:
          'No Metal counterpart: peer groups are deprecated, "Not applicable on Apple Silicon". Design the Metal build without a GPU-to-GPU split.',
        correct: true,
      },
      {
        label: 'Swap cudaDeviceEnablePeerAccess for Metal\'s peer-access call; the kernels stay the same.',
        why: 'Metal has no such call on Apple silicon. Nothing grants one GPU access to another GPU\'s buffer there.',
      },
      {
        label: "Bind the other GPU's buffer through newRemoteBufferViewForDevice, which gives a view of a peer's buffer, and read that view in the kernel.",
        why: 'That method made a view for blit copies between GPUs in a peer group, on Intel Macs. It never served kernel loads, and it is deprecated.',
      },
      {
        label: 'Keep the design: on one physical memory every buffer is visible to every GPU, so the peer pointer just works.',
        why: 'One memory is the CPU and the GPU sharing DRAM. It says nothing about a second GPU, and Metal documents no peers on Apple silicon.',
      },
    ],
    explain: (
      <p>
        <code>MTLDevice.peerGroupID</code> and <code>newRemoteBufferViewForDevice:</code> are both marked "Not applicable on
        Apple Silicon" in the macOS 27.0 SDK, and the remote view was a copy path, not a dereference. A design built on peer
        loads is a CUDA and HIP design; on a Mac you plan the Metal build without it rather than translate the call.
      </p>
    ),
    source: "Apple's documentation and the macOS 27.0 SDK headers, checked 2026-11-20 (notes/today/appC-facts.md, C4)",
  },
};

// ---------------------------------------------------------------- the section
export function SC5() {
  return (
    <>
      <p>
        A wide physical address also has room for more than one GPU. The book's point here is that system software can
        place every GPU's memory in one unified physical address space. A kernel on one GPU can then load from another
        GPU's memory, on the same PCIe bus or NVLink domain, by dereferencing a pointer into it.
      </p>
      <Source section="C.5" pages="626" />
      <p>
        That replaces something clumsy. Take the halo exchange of chapter 23's stencil (
        <a href={href('ch23', '23.4')}>section 23.4</a> here), where each GPU owns half a grid
        and needs a few boundary rows of the other half before its next step. Without direct access, the host has to
        trigger device-to-device copies: wait for both GPUs, set up a copy each way into a spare buffer, then launch the
        next step. The book names the costs: extra memory to hold the copied rows, and the host's synchronisation and
        copy set-up. With direct access, each kernel takes its neighbour's pointer as an argument and loads the rows
        itself.
      </p>
      <Source section="C.5" pages="626" />
      <p>
        The book says this became possible with Kepler. NVIDIA's own account does not support that boundary: its CUDA 6
        announcement says unified addressing, since CUDA 4 in 2011, let GPU code reach pointers on "a different GPU", and
        2011 is before Kepler's launch. We found no NVIDIA page that names the first architecture with peer loads, so
        this course teaches no boundary.
      </p>
      <Today checked="2026-11-20 against NVIDIA's CUDA 6 Unified Memory post (appC-facts.md, A8)">
        <p>"Before Kepler" is not supported by NVIDIA's account. The first architecture with peer loads: unverified.</p>
      </Today>
      <Ask q={questionsSC5['appC/C.5/q-removes']} />

      <h2>It has to be switched on</h2>
      <p>
        The book's paragraph reads as if any GPU could load from any other. In CUDA today a program must ask first.{' '}
        <code>cudaDeviceCanAccessPeer</code> says whether two devices can be peers at all. Then, with the device that will
        do the loading as the current device, the program calls <code>cudaDeviceEnablePeerAccess(peerDevice, 0)</code>.
        After that, kernels on the current device can load through a pointer into the peer's memory, and the pointer has
        the same value on both devices. Without a grant, data still moves between GPUs by copying, with{' '}
        <code>cudaMemcpyPeer</code>.
      </p>
      <Today checked="2026-11-20 against the CUDA 13.5 guide §3.4.2 and runtime reference (appC-facts.md, A8)">
        <p>Access granted by one call "is unidirectional".</p>
        <p>
          Without NVSwitch, at most eight peer connections per device. No PCIe peer-to-peer on bare-metal Linux with the
          IOMMU on.
        </p>
      </Today>
      <Ask q={questionsSC5['appC/C.5/q-oneway']} />
      <p>
        Two limits come with it. Without NVSwitch, a device supports at most eight peer connections, so a large box cannot
        connect everything to everything. And on bare-metal Linux with the IOMMU switched on, CUDA does not support PCIe
        peer-to-peer at all. There is also a cost that grows with use: enabling peer access makes every later allocation
        on the peer more expensive, and NVIDIA points to the virtual memory calls of{' '}
        <a href={href('appC', 'C.9')}>§C.9</a>, which grant access only to the memory that needs it, as the scalable
        alternative.
      </p>
      <p>
        The explorer applies these rules. Grant access cell by cell and watch the direction; then ask for every pair on
        nine GPUs and on sixteen. Beside it, a kernel follows a pointer, and one halo exchange is laid out both ways, with
        the costs the book names counted.
      </p>
      <Simplification>
        <p>
          Whether two GPUs can be peers at all is taken as given; on a real machine <Id>cudaDeviceCanAccessPeer</Id>{' '}
          decides, from the topology.
        </p>
        <p>
          "Eight peer connections" is modelled as eight outgoing grants; the guide does not say how connections are
          counted. No bandwidth or latency.
        </p>
        <p>
          Without NVSwitch the GPUs are taken to be PCIe peers, which is what the IOMMU rule covers. With NVSwitch and the
          IOMMU on, our sources say nothing, and the explorer says so instead of guessing.
        </p>
      </Simplification>
      <PeerMatrix />
      <Ask q={questionsSC5['appC/C.5/q-eight']} />

      <h2>Where you meet it</h2>
      <p>
        Splitting one model across several GPUs means each GPU computes part of a layer and needs partial results that
        another GPU holds. Whether those arrive by peer loads or by copies is exactly this section's choice. llama.cpp,
        for one, leaves it to you: its CUDA backend has an environment variable that turns peer access between GPUs
        on.
      </p>
      <Beyond>
        <p>
          llama.cpp's build notes: "The environment variable <Id>GGML_CUDA_P2P</Id> can be set to enable peer-to-peer
          access between multiple GPUs".
        </p>
        <p className="muted">Checked 2026-11-20 against llama.cpp master (appC-facts.md, D3).</p>
      </Beyond>

      <h2>On AMD and Apple GPUs</h2>
      <p>
        HIP has the same model under HIP names, <code>hipDeviceCanAccessPeer</code> and{' '}
        <code>hipDeviceEnablePeerAccess</code>. Without a grant, <code>hipMemcpy</code> between GPUs still works, but AMD
        says it stages the data through host memory. Metal has nothing comparable on Apple silicon: the peer-group calls
        of Intel-based Macs are deprecated there.
      </p>
      <Beyond>
        <p>
          HIP: checked against the ROCm 10.2 header and AMD's multi-device page (appC-facts.md, B4).
        </p>
        <p>
          Metal: <Id>peerGroupID</Id> and <Id>newRemoteBufferViewForDevice:</Id> are deprecated, "Not applicable on Apple
          Silicon" (C4).
        </p>
      </Beyond>
      <Ask q={questionsSC5['appC/C.5/q-metal']} />
      <p>
        Back to <a href={href('appC', 'start')}>the map</a>: this is the step marked C.4–C.5. The cells it changes are the
        row for a peer GPU's <code>cudaMalloc</code> memory, from "no" to "only after access is enabled".
      </p>
    </>
  );
}
