import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { fmt, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  BOOK_DEVICE,
  BOOK_DEVICE_TODAY,
  TERMS,
  deviceFromReport,
  isGpuBackend,
  levels,
  noReport,
  residentThreads,
  threadsPerCore,
  totalCores,
  type Device,
  type Level,
  type LevelId,
  type Vocabulary,
} from '../../sim/blocksched';

const VOCABS: Vocabulary[] = ['cuda', 'hip', 'metal'];
const PROVENANCE: Record<Level['provenance'], string> = {
  book: 'the book states it',
  reported: 'your device reported it',
  'vendor-doc': "AMD's documentation, CDNA family",
  none: 'nobody says',
};
const where = (w: string) => (w.startsWith('4.') ? href('ch04', w) : href('ch05', 'start'));

/** The count cell of a level row, in words. */
function countText(l: Level): string {
  if (l.count != null) return l.id === 'device' ? '1' : `${fmt(l.count, 6)} ${l.per}`;
  if (l.id === 'onchip' || l.id === 'global') return l.per;
  return l.name == null ? 'not documented' : 'not reported';
}

// Fig. 4.1 redrawn. The picture is a schematic: it always draws two clusters of
// four units with sixteen cores each, and the labels say how many there really are.
function Picture({ ls, sel, onPick, known }: { ls: Level[]; sel: LevelId; onPick: (id: LevelId) => void; known: boolean }) {
  const L = Object.fromEntries(ls.map((l) => [l.id, l])) as Record<LevelId, Level>;
  const stroke = (id: LevelId) => ({
    stroke: sel === id ? 'var(--ink)' : 'var(--ink-3)',
    strokeWidth: sel === id ? 2.4 : 1,
    strokeDasharray: L[id].name == null ? '4 3' : undefined,
  });
  const pick = (id: LevelId) => ({ onClick: () => onPick(id), style: { cursor: 'pointer' } });
  const units = [0, 1, 2, 3];
  const ghost = known ? 1 : 0.45;
  return (
    <svg viewBox="0 0 560 318" role="img" aria-label="Schematic of the device, redrawn from the book's Fig. 4.1" style={{ width: '100%', maxWidth: 560, display: 'block' }}>
      <g {...pick('device')}>
        <rect x="4" y="20" width="552" height="292" rx="5" fill="var(--page)" {...stroke('device')} />
        <text x="12" y="14" fontSize="12" fontWeight="600">{L.device.name}</text>
      </g>
      {[0, 1].map((c) => {
        const cx = 16 + c * 268;
        return (
          <g key={c} opacity={ghost}>
            <g {...pick('cluster')}>
              <rect x={cx} y="42" width="260" height="150" rx="4" fill="none" {...stroke('cluster')} strokeDasharray={L.cluster.name == null ? '4 3' : '6 3'} />
              <text x={cx + 8} y="56" fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>{L.cluster.name ?? ''}</text>
            </g>
            {units.map((u) => {
              const ux = cx + 10 + u * 61;
              return (
                <g key={u}>
                  <rect x={ux} y="64" width="56" height="120" rx="3" fill="var(--cell)" {...stroke('unit')} {...pick('unit')} />
                  <rect x={ux + 5} y="69" width="46" height="11" rx="1.5" fill="var(--rule)" stroke="var(--ink-3)" strokeWidth="0.6" {...pick('unit')} />
                  <text x={ux + 28} y="77.5" fontSize="7.5" textAnchor="middle" style={{ fill: 'var(--ink-2)', pointerEvents: 'none' }}>control</text>
                  {[0, 1].map((g) => (
                    <g key={g} {...pick('group')}>
                      <rect x={ux + 4 + g * 25} y="84" width="23" height="48" rx="2" fill="none" {...stroke('group')} />
                      {Array.from({ length: 8 }, (_, i) => (
                        <rect key={i} x={ux + 6.5 + g * 25 + (i % 2) * 9.5} y={86.5 + Math.floor(i / 2) * 11} width="8" height="9.5" rx="1"
                          fill={sel === 'core' ? 'var(--ink-2)' : 'var(--bench)'} stroke={sel === 'core' ? 'var(--ink)' : 'var(--ink-3)'} strokeWidth={sel === 'core' ? 1.2 : 0.7}
                          onClick={(e) => { e.stopPropagation(); onPick('core'); }} style={{ cursor: 'pointer' }} />
                      ))}
                    </g>
                  ))}
                  <g {...pick('onchip')}>
                    <rect x={ux + 5} y="137" width="46" height="42" rx="2" fill="var(--page)" {...stroke('onchip')} />
                    <rect x={ux + 8} y="141" width="40" height="15" rx="1.5" fill="var(--reg-wash)" stroke="var(--reg)" strokeWidth="0.8" />
                    <rect x={ux + 8} y="160" width="40" height="15" rx="1.5" fill="var(--shared-wash)" stroke="var(--shared)" strokeWidth="0.8" />
                  </g>
                </g>
              );
            })}
          </g>
        );
      })}
      {!known ? (
        <g>
          <rect x="150" y="100" width="260" height="34" rx="3" fill="var(--page)" stroke="var(--ink-3)" />
          <text x="280" y="121" fontSize="12" textAnchor="middle">nothing inside the device is documented</text>
        </g>
      ) : null}
      {/* memory controllers and global memory */}
      <line x1="30" x2="530" y1="204" y2="204" stroke="var(--ink-3)" />
      {[0, 1, 2, 3, 4, 5].map((k) => (
        <g key={k}>
          <line x1={63 + k * 87} x2={63 + k * 87} y1="204" y2="214" stroke="var(--ink-3)" />
          <rect x={30 + k * 87} y="214" width="66" height="20" rx="2" fill="var(--cell)" stroke="var(--ink-3)" />
          <text x={63 + k * 87} y="227.5" fontSize="8" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>controller</text>
          <line x1={63 + k * 87} x2={63 + k * 87} y1="234" y2="254" stroke="var(--global)" strokeWidth="1.4" />
        </g>
      ))}
      <g {...pick('global')}>
        <rect x="16" y="254" width="528" height="48" rx="3" fill="var(--global-wash)" stroke="var(--global)" strokeWidth={sel === 'global' ? 2.4 : 1.2} />
        <text x="280" y="283" fontSize="12.5" fontWeight="600" textAnchor="middle" style={{ fill: 'var(--global)' }}>{L.global.name}</text>
      </g>
    </svg>
  );
}

function LevelExplorer() {
  const { reports } = useReports();
  const [vocab, setVocab] = useState<Vocabulary>('cuda');
  const [pickDevice, setPickDevice] = useState(0);
  const [sel, setSel] = useState<LevelId>('unit');

  const devices: Device[] = useMemo(() => {
    const mine = Object.keys(reports).filter(isGpuBackend)
      .map((b) => deviceFromReport(b, reports[b].device))
      .filter((d) => d.vocabulary === vocab);
    if (vocab === 'cuda') return [BOOK_DEVICE, ...mine];
    return mine.length ? mine : [noReport(vocab)];
  }, [reports, vocab]);
  const d = devices[Math.min(pickDevice, devices.length - 1)];
  const ls = useMemo(() => levels(d), [d]);
  const cur = ls.find((l) => l.id === sel)!;
  const hardware = ls.filter((l) => ['cluster', 'unit', 'group', 'core'].includes(l.id));
  const askable = hardware.filter((l) => l.ask != null).length;
  const known = hardware.some((l) => l.name != null);
  const t = TERMS[vocab];
  const optionLabel = (x: Device) =>
    x.source === 'book' ? "the book's example" : x.source === 'report' ? `your device: ${x.name}` : `no ${t.api} device report on this machine`;

  return (
    <Bench
      title="The device, level by level"
      task="Switch the backend and the device. Which of these sizes can a program ask for?"
      foot={
        <>
          The picture is the book's Fig. 4.1 as a schematic with a fixed number of boxes, greyed where a vendor documents
          no such level; the table says how many there are. In the book's picture,{' '}
          <span className="chip reg">registers</span> and <span className="chip shared">shared memory</span> are on the
          chip and <span className="chip global">global memory</span> is off it.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <span className="seg" role="group" aria-label="Backend">
          {VOCABS.map((v) => (
            <button key={v} aria-pressed={v === vocab} onClick={() => { setVocab(v); setPickDevice(0); }}>
              {TERMS[v].api}
            </button>
          ))}
        </span>
        <label className="field">
          Device
          <select value={Math.min(pickDevice, devices.length - 1)} onChange={(e) => setPickDevice(+e.target.value)} aria-label="Device">
            {devices.map((x, i) => <option key={i} value={i}>{optionLabel(x)}</option>)}
          </select>
        </label>
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)' }}>
        <div>
          <Picture ls={ls} sel={sel} onPick={setSel} known={known} />
        </div>
        <div>
          <table className="data levels" style={{ margin: 0, background: 'var(--page)', width: '100%' }} aria-label="Levels of the device">
            <thead>
              <tr>
                <th style={{ paddingLeft: '0.5rem', width: '36%' }}>Level</th>
                <th style={{ width: '27%' }}>How many</th>
                <th>A program asks with</th>
              </tr>
            </thead>
            <tbody>
              {ls.map((l) => (
                <tr key={l.id} data-level={l.id} data-count={l.count ?? ''} style={l.id === sel ? { background: 'var(--rule-soft)' } : undefined}>
                  <td style={{ paddingLeft: '0.5rem' }}>
                    <button className="quiet" aria-pressed={l.id === sel} onClick={() => setSel(l.id)} style={{ whiteSpace: 'normal', textAlign: 'left' }}>
                      {l.name ?? `no counterpart of the ${l.bookName} documented`}
                    </button>
                  </td>
                  <td>
                    {countText(l)}
                    {l.count != null ? <div className="muted says">{PROVENANCE[l.provenance]}</div> : null}
                  </td>
                  <td>{l.ask ? <code style={{ whiteSpace: 'normal', overflowWrap: 'anywhere' }}>{l.ask}</code> : <span className="muted">no query</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="summary" role="status" style={{ margin: '0.8rem 0 0' }}>
            Of the four hardware levels under the device, a {t.api} program can ask for the size of <b>{askable}</b>.
            {d.source === 'report' && d.threadsPerUnit != null ? (
              <> Your device also reports {fmt(d.threadsPerUnit, 6)} threads in progress per {t.unit}{d.blocksPerUnit != null ? <> and {fmt(d.blocksPerUnit, 6)} blocks</> : null}.</>
            ) : null}
          </p>
        </div>
      </div>
      <div className="panel" style={{ marginTop: '1rem' }} data-selected={cur.id}>
        <p style={{ margin: '0 0 0.4rem' }}>
          <b>{cur.name ?? `No counterpart of the ${cur.bookName} documented`}</b>
          {cur.holds.length ? ': what the rest of the chapter places here' : null}
        </p>
        {cur.holds.length ? (
          <ul style={{ margin: '0 0 0.5rem', paddingLeft: '1.2em' }}>
            {cur.holds.map((h) => (
              <li key={h.what}>
                {h.what} (<a href={where(h.where)}>{h.where.startsWith('4.') ? 'section ' + h.where : h.where}</a>)
              </li>
            ))}
          </ul>
        ) : null}
        <p className="muted" style={{ margin: 0 }}>{cur.note}</p>
      </div>
    </Bench>
  );
}

export const questionsS41: Record<string, Question> = {
  'ch04/4.1/q-cores': {
    kind: 'choice',
    id: 'ch04/4.1/q-cores',
    prompt: (
      <>
        A colleague reads the book's example and says: "The device has 16,896 cores, so at most 16,896 threads of my
        kernel are in progress at any moment." What is the correction?
      </>
    ),
    options: [
      {
        label: 'Cores bound what executes in one cycle, not what is in progress: each SM keeps 2048 threads resident.',
        correct: true,
      },
      { label: 'No correction: one thread per core, and the rest of the grid waits on the runtime\'s list.', why: 'That is how you would size a CPU thread pool. An SM holds sixteen times more threads than it has cores in the example.' },
      { label: 'Too high: an SM runs one thread at a time, so at most 132 threads are in progress.', why: 'An SM is not a core. It has 128 of them in the example.' },
      { label: 'Too high: a block holds at most 1024 threads, and the device runs one block at a time.', why: 'That limit is per block. Many blocks are in progress at once.' },
    ],
    explain: (
      <p>
        Two separate numbers. What can execute in the same clock cycle is bounded by the cores: 132 × 128 = 16,896. What
        is in progress is bounded by what the SMs keep resident: 132 × 2048 = 270,336 threads in the example. The gap is
        deliberate (section 4.6), and the cores are not independent processors either: they share control logic, so
        they run in groups that execute one instruction together (section 4.4).
      </p>
    ),
    source: 'ch. 4 §4.1, p. 68 and §4.6, p. 85 (figures are the book\'s example device)',
  },
  'ch04/4.1/q-mpcount': {
    kind: 'choice',
    id: 'ch04/4.1/q-mpcount',
    prompt: (
      <>
        The same program prints <code>multiProcessorCount</code> on an NVIDIA device through CUDA and on an AMD device
        through HIP. What do the two numbers count?
      </>
    ),
    options: [
      { label: 'Cores on both: one field name, so the larger number is the larger device.', why: 'Neither is a core count.' },
      {
        label: 'SMs on one; compute units, or half of them, on the other. The two are not comparable.',
        correct: true,
      },
      { label: 'SMs on both: HIP translates the AMD hardware into NVIDIA\'s units for this field.', why: 'HIP keeps the field name. It fills it with AMD\'s own unit.' },
      { label: 'The number of GPUs in the machine, which both runtimes report the same way.', why: 'That is what cudaGetDeviceCount and hipGetDeviceCount return.' },
    ],
    explain: (
      <p>
        The name carries over; the unit does not. On CUDA it is the number of SMs. HIP's header says it equals the
        number of compute units, or half that number when the device runs in workgroup-processor mode, where one
        processor holds two CUs. An SM and a CU are built differently inside, so a ratio of the two counts says nothing
        about relative speed. Use the number for what it is good for on either vendor: sizing a launch against the
        device you are on.
      </p>
    ),
    source: 'book p. 89 for CUDA; notes/today/ch04-facts.md B1 and ch05-facts.md B2 for HIP',
  },
  'ch04/4.1/q-global': {
    kind: 'choice',
    id: 'ch04/4.1/q-global',
    prompt: (
      <>
        A kernel takes an argument <code>float* M</code>. In the picture of the device, where does the data behind{' '}
        <code>M</code> live, and what does every access to it cross?
      </>
    ),
    options: [
      { label: 'In the on-chip memory of the SM the thread runs on; nothing is crossed.', why: 'On-chip memory holds registers and shared memory. A pointer argument does not point there.' },
      { label: 'In host memory; every access crosses the bus between CPU and GPU.', why: 'The host copied it to the device once, before the launch. The kernel never touches host memory.' },
      {
        label: 'In global memory, off the chip; every access goes out through the memory controllers.',
        correct: true,
      },
      { label: 'In the registers of the threads that read it, filled when the kernel starts.', why:'A value lands in a register after it has been fetched. The array itself is elsewhere.' },
    ],
    explain: (
      <p>
        Global memory is the wide bar at the bottom of the picture: one pool for the whole device, reached through the
        memory controllers. Compared with arithmetic on a register, an access to it takes a long time, and it is the
        operation section 4.6 is built around: what should an SM do with a thread that is waiting for one?
      </p>
    ),
    source: 'ch. 4 §4.1, p. 68 and §4.6, p. 83',
  },
};

export function S41() {
  const q = questionsS41;
  const book = BOOK_DEVICE;
  return (
    <>
      <p>
        Seen from a kernel, a GPU is an array of <b>streaming multiprocessors</b>, SMs for short. Each SM is a set of
        simple cores that share one piece of control logic and one pool of memory on the chip. The book calls the cores{' '}
        <b>streaming processors</b>. You will also meet them as "CUDA cores"; the book avoids that name so nobody takes
        them for CPU cores, and section 4.4 shows how different they are.
      </p>
      <Source section="4.1" pages="68" figs="Fig. 4.1, redrawn in the explorer below" />
      <p>
        Outside the SMs sits the device's <b>global memory</b>: gigabytes of DRAM that every thread of every grid can
        reach, through several memory controllers that together give it its bandwidth. Older GPUs used GDDR memory;
        since NVIDIA's Pascal generation a GPU may use one of the HBM variants, packaged with the GPU chip. The book
        calls all of it DRAM from here on, and so does this course. Using it well is chapter 6.
      </p>
      <Today checked="2026-11-20 (notes/today/ch04-facts.md, A4)">
        <p>On the book's example device, the H100, HBM3 is on the SXM5 variant only; the PCIe variant has HBM2e.</p>
      </Today>

      <h2>The book's example</h2>
      <p>
        The book puts numbers on the picture with one device: {book.units} SMs of {book.coresPerUnit} streaming
        processors each, {fmt(totalCores(book)!, 6)} in all. Since the Hopper generation the SMs are also grouped into{' '}
        <b>GPU processing clusters</b> (GPCs), {book.clusters} on this device. Treat every one of these as a property of
        that one device. Yours has other numbers, and the explorer below shows which of them it will tell you.
      </p>
      <Today checked="2026-11-20 (notes/today/ch04-facts.md, A1, A2)">
        <p>
          The 132 is the H100 SXM5; the PCIe variant has {BOOK_DEVICE_TODAY.unitsOtherVariant}. A level the book leaves
          out, the TPC of {BOOK_DEVICE_TODAY.unitsPerTpc} SMs: {BOOK_DEVICE_TODAY.tpcs} TPCs make the 132. Their split
          over 8 GPCs is not published (unverified).
        </p>
      </Today>
      <Predict
        id="ch04/4.1/p-resident"
        prompt={
          <>
            Each SM of the book's example has {book.coresPerUnit} cores. How many threads do you expect one SM can have
            in progress at once?
          </>
        }
        unit="threads"
      >
        {(v) => (
          <>
            <p>
              The book's example: <b>{fmt(book.threadsPerUnit!, 6)}</b> threads per SM, {threadsPerCore(book)} for every
              core (p. 85).{' '}
              {v === book.coresPerUnit
                ? 'One thread per core is the natural guess, and it is how you would size a thread pool on a CPU.'
                : v < book.threadsPerUnit!
                  ? 'You expected fewer.'
                  : v === book.threadsPerUnit
                    ? 'Exactly.'
                    : 'You expected even more.'}
            </p>
            <p>
              An SM cannot execute {fmt(book.threadsPerUnit!, 6)} threads in the same cycle: it has {book.coresPerUnit}{' '}
              cores. It keeps them all in progress and executes some of them at a time, so that a thread waiting for
              memory holds nothing up. Across the device that is {fmt(residentThreads(book)!, 6)} threads in progress on{' '}
              {fmt(totalCores(book)!, 6)} cores. <a href={href('ch04', '4.6')}>Section 4.6</a> explains why it is built
              this way.
            </p>
          </>
        )}
      </Predict>

      <h2>What a device will tell you</h2>
      <p>
        A kernel that is to run well on more than one device has to size itself from what the device reports. So it
        matters which sizes of this picture are reported at all. Pick a backend and a device, then select a level to see
        what the rest of the chapter places there.
      </p>
      <Simplification>
        <p>
          This is a table of names and counts, not a model of hardware. It knows only what the book states about its
          example, what your device's report contains, and one description from AMD's documentation.
        </p>
      </Simplification>
      <LevelExplorer />

      <h2>Three consequences</h2>
      <p>The rest of the chapter unpacks three things this picture already implies.</p>
      <ul>
        <li>
          <b>A block goes whole to one SM.</b> The on-chip memory and the control logic belong to an SM, so threads
          that are to share memory or wait for each other have to be on the same one (
          <a href={href('ch04', '4.2')}>4.2</a>, <a href={href('ch04', '4.3')}>4.3</a>).
        </li>
        <li>
          <b>Cores share control, so threads run in groups.</b> One fetched instruction drives many cores at once. The
          groups are warps (<a href={href('ch04', '4.4')}>4.4</a>), and a group whose threads want different
          instructions pays for it (<a href={href('ch04', '4.5')}>4.5</a>).
        </li>
        <li>
          <b>An SM holds more threads than it has cores.</b> That is what lets it keep computing while threads wait on
          global memory (<a href={href('ch04', '4.6')}>4.6</a>), and how many it can hold is a budget your kernel
          spends (<a href={href('ch04', '4.7')}>4.7</a>).
        </li>
      </ul>
      <Ask q={q['ch04/4.1/q-cores']} />
      <Ask q={q['ch04/4.1/q-global']} />

      <h2>On other backends</h2>
      <p>
        AMD documents a picture of the same shape with other names: shader engines contain shader arrays, which contain{' '}
        <b>compute units</b> (CUs), the units a block runs on. For its CDNA family AMD describes a CU as four SIMD
        processors of 16 lanes each. Under HIP the field <code>multiProcessorCount</code> exists with the same spelling
        and counts CUs, or half of them when the device runs in workgroup-processor mode.
      </p>
      <Beyond>
        <p>
          The book covers NVIDIA hardware only. AMD's and Apple's terms are from their documentation
          (notes/today/ch04-facts.md, B1, C1). Setting SM beside CU, and processing block beside SIMD, is our
          comparison, and it is loose: a 16-lane SIMD takes four cycles for one instruction of a 64-thread wavefront.
        </p>
      </Beyond>
      <p>
        Metal documents no such unit. An <code>MTLDevice</code> has a name, memory limits and a maximum threadgroup
        size, and no count of cores or compute units. On Metal you size a launch from properties of the compiled
        kernel, which is <a href={href('ch04', '4.8')}>section 4.8</a>.
      </p>
      <Ask q={q['ch04/4.1/q-mpcount']} />

      <h2>From a Python prompt</h2>
      <p>
        You do not need C++ to see your own device's side of this. On a CUDA build of PyTorch,{' '}
        <code>torch.cuda.get_device_properties(0).multi_processor_count</code> prints the SM count, and the same object
        has <code>max_threads_per_multi_processor</code> and <code>warp_size</code>. It is a direct binding of the
        structure section 4.8 queries from C++.
      </p>
      <Beyond>
        <p>
          Ours, read in PyTorch's source on 2026-11-20 (notes/today/ch04-facts.md, D4). Not run on a CUDA machine by
          this course: unverified.
        </p>
      </Beyond>
    </>
  );
}
