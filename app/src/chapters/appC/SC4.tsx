import { useMemo, useState, type ReactNode } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  ELEMENTS,
  INDEX,
  MARKS,
  arrayLimit,
  binary,
  bitsOf,
  memoryFigures,
  reach,
  type IndexType,
  type MemoryFigure,
} from '../../sim/addrwidth';

/** A long identifier that may break anywhere, for the narrow margin notes. */
function Id({ children }: { children: ReactNode }) {
  return <code style={{ whiteSpace: 'normal', overflowWrap: 'anywhere' }}>{children}</code>;
}

const INDEX_LABEL: Record<IndexType, string> = { int32: 'int', uint32: 'unsigned int', int64: 'int64_t' };
const FIELD_SHORT: Record<MemoryFigure['field'], string> = {
  totalGlobalMem: 'global memory',
  maxBufferLength: 'largest buffer',
  recommendedMaxWorkingSetSize: 'working set',
};
const pow = (e: number) => (
  <>
    2<sup>{e}</sup>
  </>
);

// ---------------------------------------------------------------- your devices
function YourMemory() {
  const { reports, loading } = useReports();
  const figs = useMemo(() => memoryFigures(reports), [reports]);
  const intFloats = arrayLimit('int32', 4).bytes;
  return (
    <div className="panel" data-your-memory={figs.length} style={{ margin: '0.4rem 0 1.4rem' }}>
      <p className="ui" style={{ margin: '0 0 0.4rem' }}>
        <b>What your devices reported</b>
      </p>
      {figs.length ? (
        <>
          <div style={{ overflowX: 'auto' }}>
            <table className="data" style={{ margin: 0 }}>
              <thead>
                <tr>
                  <th>Device</th>
                  <th>Property</th>
                  <th className="num">Reported</th>
                  <th className="num">Against 8 GiB</th>
                </tr>
              </thead>
              <tbody>
                {figs.map((f) => (
                  <tr key={f.backend + f.field} data-figure={f.field} data-bytes={f.bytes}>
                    <td>
                      {f.name} <span className="muted">({f.backend})</span>
                    </td>
                    <td>
                      <code>{f.field}</code>
                      <br />
                      <span className="muted">{f.meaning}</span>
                    </td>
                    <td className="num">{binary(f.bytes).text}</td>
                    <td className="num">
                      {f.bytes > intFloats ? (
                        <span className="chip fail">{Number((f.bytes / intFloats).toFixed(2))} times</span>
                      ) : (
                        <span className="chip ok">within</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="ui muted" style={{ margin: '0.6rem 0 0' }}>
            The last column compares each figure with the 8 GiB of floats an <code>int</code> index reaches. A figure
            above it is memory an <code>int</code>-indexed kernel over one float array could not reach the end of. On
            Metal the figure to compare is <code>maxBufferLength</code>, since one array lives in one buffer.
          </p>
        </>
      ) : (
        <p className="ui muted" style={{ margin: 0 }}>
          {loading ? (
            'Reading reports.'
          ) : (
            <>
              No GPU is recorded on this machine yet. Run <code>python3 labs/pmpp.py run appC/lab1 --bench</code> and
              reload: the first run on a backend records what the device reports.
            </>
          )}
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- the explorer
const LO = 16;
const HI = 64;
const W = 640;
const PAD = 26;
const x = (bits: number) => PAD + ((Math.min(HI, Math.max(LO, bits)) - LO) / (HI - LO)) * (W - 2 * PAD);

function BitsToBytes() {
  const { reports } = useReports();
  const figs = useMemo(() => memoryFigures(reports), [reports]);
  const [bits, setBits] = useState(32);
  const [elem, setElem] = useState(4);
  const [index, setIndex] = useState<IndexType>('int32');

  const r = reach(bits);
  const lim = arrayLimit(index, elem, bits);
  const elemLabel = ELEMENTS.find((e) => e.bytes === elem)?.label ?? `${elem} bytes`;
  const over = figs.filter((f) => f.bytes > r);
  const limBits = bitsOf(lim.indexBytes);

  return (
    <Bench
      title="Bits to bytes"
      task="Find the smallest width that names all of your device's memory, then the index type that reaches all of it."
      foot={
        <>
          Reach is 2<sup>bits</sup> bytes, in binary units (1 GiB = 2<sup>30</sup> bytes). The marks come from the book
          (32, 48 and 49 bits), the Fermi whitepaper (40) and the Linux documentation (48 and 57); see the list below the
          bar.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        <label className="field" style={{ flex: '1 1 18rem' }}>
          Address width: {bits} bits
          <input
            type="range"
            min={LO}
            max={HI}
            step={1}
            value={bits}
            aria-label="Address width in bits"
            onChange={(e) => setBits(Number(e.target.value))}
          />
        </label>
        <div aria-live="polite">
          <p style={{ margin: 0, fontSize: '1.5rem', fontWeight: 600, lineHeight: 1.2 }}>
            <output data-out="reach" data-bits={bits} data-bytes={r}>
              {binary(r).text}
            </output>
          </p>
          <p className="muted" style={{ margin: 0 }}>
            {pow(bits)} bytes. One bit fewer: {binary(reach(bits - 1)).text}. One more: {binary(reach(bits + 1)).text}.
          </p>
        </div>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <svg
          viewBox={`0 0 ${W} 150`}
          role="img"
          aria-label={`Log scale of reach from ${LO} to ${HI} bits, filled to ${bits} bits`}
          style={{ width: '100%', minWidth: 520, display: 'block' }}
          data-part="bar"
        >
          <rect x={x(LO)} y={52} width={x(HI) - x(LO)} height={20} fill="var(--cell)" stroke="var(--rule)" />
          <rect x={x(LO)} y={52} width={x(bits) - x(LO)} height={20} fill="var(--global-wash)" stroke="var(--global)" data-part="fill" />
          {[16, 24, 32, 40, 48, 56, 64].map((b) => (
            <g key={b}>
              <line x1={x(b)} x2={x(b)} y1={72} y2={78} stroke="var(--ink-3)" />
              <text x={x(b)} y={90} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>
                {b}
              </text>
            </g>
          ))}
          {MARKS.map((m, i) => (
            <g key={m.bits} data-mark={m.bits}>
              <line x1={x(m.bits)} x2={x(m.bits)} y1={i % 2 ? 30 : 16} y2={72} stroke="var(--ink-2)" strokeDasharray="2 2" />
              <text x={x(m.bits)} y={i % 2 ? 26 : 12} fontSize="10.5" textAnchor={i === 3 ? 'start' : i === 2 ? 'end' : 'middle'} style={{ fill: 'var(--ink-2)' }}>
                {m.bits}: {m.label}
              </text>
            </g>
          ))}
          {figs.map((f, i) => {
            const fx = x(bitsOf(f.bytes));
            const y = 102 + (i % 3) * 15;
            return (
              <g key={f.backend + f.field} data-part="device" data-bits={bitsOf(f.bytes).toFixed(3)}>
                <path d={`M ${fx} 74 l -4 7 h 8 z`} fill={f.bytes > r ? 'var(--fail)' : 'var(--ok)'} />
                <line x1={fx} x2={fx} y1={81} y2={y - 8} stroke="var(--rule)" />
                <text x={fx} y={y} fontSize="10" textAnchor={fx > W / 2 ? 'end' : 'start'}>
                  {f.backend}: {FIELD_SHORT[f.field]}, {binary(f.bytes).text}
                </text>
              </g>
            );
          })}
          <g data-part="limit" data-bits={limBits.toFixed(3)}>
            <rect x={x(limBits) - 4} y={56} width={8} height={12} transform={`rotate(45 ${x(limBits)} 62)`} fill="var(--reg)" />
          </g>
          <text x={x(LO)} y={146} fontSize="10.5" style={{ fill: 'var(--reg)' }} data-part="limit-label">
            ◆ how far an {INDEX_LABEL[index]} index over {elemLabel.split(',')[0]} reaches: {binary(lim.indexBytes).text}
            {limBits > HI ? ' (past the end of this bar)' : ''}
          </text>
        </svg>
      </div>

      <div className="row" role="group" aria-label="Marks" style={{ margin: '0.6rem 0 1rem' }}>
        {MARKS.map((m) => (
          <button key={m.bits} className="quiet" data-goto={m.bits} aria-pressed={bits === m.bits} onClick={() => setBits(m.bits)} title={m.source}>
            {m.bits} bits, {m.label}: {m.what}
          </button>
        ))}
      </div>

      {over.length ? (
        <p data-out="wall" data-over={over.length} style={{ margin: '0 0 1rem' }}>
          <span className="chip fail">wall</span> {bits}-bit addresses cannot name all of what your devices reported:{' '}
          {over.map((f) => `${f.name} (${f.backend}), ${FIELD_SHORT[f.field]} ${binary(f.bytes).text}`).join('; ')}.
        </p>
      ) : figs.length ? (
        <p data-out="wall" data-over={0} style={{ margin: '0 0 1rem' }}>
          <span className="chip ok">fits</span> {bits}-bit addresses name every memory figure your devices reported.
        </p>
      ) : null}

      <div className="cols plot">
        <div className="panel">
          <div className="row" style={{ alignItems: 'flex-start' }}>
            <label className="field">
              Element type
              <select value={elem} onChange={(e) => setElem(Number(e.target.value))} aria-label="Element type">
                {ELEMENTS.map((e) => (
                  <option key={e.bytes} value={e.bytes}>
                    {e.label}
                  </option>
                ))}
              </select>
            </label>
            <div className="field" style={{ display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' }}>
              Index type
              <span className="seg" role="group" aria-label="Index type">
                {(Object.keys(INDEX) as IndexType[]).map((k) => (
                  <button key={k} aria-pressed={index === k} onClick={() => setIndex(k)}>
                    {INDEX_LABEL[k]}
                  </button>
                ))}
              </span>
            </div>
          </div>
        </div>
        <div className="panel" aria-live="polite">
          <p className="muted" style={{ margin: 0 }}>
            Largest array this kernel can index
          </p>
          <p style={{ fontSize: '1.5rem', fontWeight: 600, margin: '0 0 0.2rem', lineHeight: 1.2 }}>
            <output data-out="limit" data-bytes={lim.bytes} data-binds={lim.binds}>
              {binary(lim.bytes).text}
            </output>
          </p>
          <p style={{ margin: 0 }} data-out="limit-why">
            {lim.binds === 'index' ? (
              <>
                The index runs out first: <code>{INDEX_LABEL[index]}</code> reaches {pow(INDEX[index].log2Count)}{' '}
                elements, and at {elem === 0.5 ? 'half a byte' : `${elem} byte${elem === 1 ? '' : 's'}`} each that is{' '}
                {binary(lim.indexBytes).text}.
              </>
            ) : (
              <>
                The address runs out first: {bits}-bit addresses name {binary(r).text}, less than the{' '}
                {binary(lim.indexBytes).text} a <code>{INDEX_LABEL[index]}</code> index could reach.
              </>
            )}
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSC4: Record<string, Question> = {
  'appC/C.4/q-4gib': {
    kind: 'numeric',
    id: 'appC/C.4/q-4gib',
    prompt: (
      <>
        A byte address is 32 bits wide. How many GiB of memory can it name? (1 GiB is 2<sup>30</sup> bytes.)
      </>
    ),
    answer: 4,
    unit: 'GiB',
    explain: (
      <p>
        2<sup>32</sup> = 2<sup>2</sup> × 2<sup>30</sup>, so 4 GiB. The book gives this as the ceiling on two things at
        once: how much memory an early GPU could carry, and how much data one kernel could work on.
      </p>
    ),
    source: 'App. C §C.4, pp. 625–626',
  },
  'appC/C.4/q-why-special': {
    kind: 'choice',
    id: 'appC/C.4/q-why-special',
    prompt: (
      <>
        With 32-bit GPU addresses, zero-copy access needed a special allocation (<code>cudaHostAlloc</code>) and a
        second pointer for the kernel, instead of letting a kernel use any pointer from <code>malloc</code>. Which reason
        does this section give?
      </>
    ),
    options: [
      {
        label: 'Pinned memory is faster than pageable memory, so the GPU insists on it.',
        why: 'Pinning keeps the pages in place while the GPU reads them (§C.2). It does not explain why the kernel needed a different pointer value.',
      },
      {
        label: 'Memory from malloc is not contiguous, so the GPU could not walk it.',
        why: 'A virtual range from malloc is contiguous by definition. The question is whether the GPU can even name its address.',
      },
      {
        label:
          'A host virtual address uses up to 48 bits and cannot be written in a 32-bit GPU pointer, so the GPU had to be given a different address for the same buffer, one it could form.',
        correct: true,
      },
      {
        label: 'A GPU cannot read DRAM attached to the CPU.',
        why: 'Zero-copy memory exists precisely so that a kernel can, over the interconnect (§C.2).',
      },
    ],
    explain: (
      <p>
        A 48-bit host address does not fit in 32 bits. So the book counts the narrow GPU pointer among the reasons for
        the special allocation, and for the two pointer values, one for the host and one for the kernel, that §C.2
        describes.
      </p>
    ),
    source: 'App. C §C.4, p. 626, with §C.2, p. 624',
  },
  'appC/C.4/q-fermi-tib': {
    kind: 'numeric',
    id: 'appC/C.4/q-fermi-tib',
    prompt: <>Fermi's unified address space was 40 bits wide. How many TiB is that?</>,
    answer: 1,
    unit: 'TiB',
    explain: (
      <>
        <p>
          2<sup>40</sup> bytes is exactly 1 TiB, which NVIDIA's Fermi whitepaper calls "a Terabyte". That is 2
          <sup>8</sup> = 256 times the 4 GiB wall.
        </p>
        <p>
          It is also why "Kepler was first" does not hold: the generation before Kepler had already left 32 bits behind.
        </p>
      </>
    ),
    source: 'Fermi whitepaper, checked 2026-11-20 (notes/today/appC-facts.md, A6)',
  },
  'appC/C.4/q-int-index': {
    kind: 'numeric',
    id: 'appC/C.4/q-int-index',
    prompt: (
      <>
        A kernel computes <code>int i = blockIdx.x * blockDim.x + threadIdx.x</code> and uses <code>i</code> to index a
        float array. About how many GiB can the array hold before <code>i</code> overflows?
      </>
    ),
    answer: 8,
    tol: 0.01,
    unit: 'GiB',
    explain: (
      <>
        <p>
          <code>i</code> runs from 0 to 2<sup>31</sup> − 1, so it reaches 2<sup>31</sup> elements. At 4 bytes each that
          is 2<sup>33</sup> bytes: 8 GiB. In half precision, 4 GiB. The width of the pointer has nothing to do with it;
          the index is the narrow part.
        </p>
        <p>
          Real code either computes the offset in 64 bits or checks the size before launching.{' '}
          <a href={href('ch03', '3.4')}>Chapter 3, §3.4</a> shows how PyTorch, llama.cpp and MLX do it.
        </p>
      </>
    ),
    source: 'Ours: C++ integer limits; see ch. 3 §3.4 and notes/today/ch03-facts.md',
  },
};

// ---------------------------------------------------------------- the section
export function SC4() {
  return (
    <>
      <p>
        Every pointer is a number, and the width of that number decides how much memory a program can name. The first
        CUDA GPUs used 32-bit virtual addresses, and their physical addresses were at most 32 bits wide. Before reading
        on, work out what that allowed.
      </p>
      <Source section="C.4" pages="625–626" />
      <Ask q={questionsSC4['appC/C.4/q-4gib']} label="Work it out" />

      <h2>A wall at 4 GiB</h2>
      <p>
        Four gibibytes was a wall on two sides. The physical width capped how much memory a GPU could carry. The width
        of a pointer capped how much data one kernel could work on, wherever the data lived, in device memory or in host
        memory.
      </p>
      <p>
        It also kept the host and the device apart. The book notes that a 64-bit CPU uses 48 of its 64 address bits. A
        host address therefore did not fit in a 32-bit GPU pointer at all, and the book counts this among the reasons
        that zero-copy memory, in <a href={href('appC', 'C.2')}>§C.2</a>, needed its own allocation call and a separate
        pointer for the kernel.
      </p>
      <Source section="C.4" pages="626" />
      <Ask q={questionsSC4['appC/C.4/q-why-special']} />

      <h2>Sixty-four bits, and a date to correct</h2>
      <p>
        The book's next step is the Kepler architecture: from Kepler on, it says, GPUs used 64-bit virtual addresses and
        physical addresses of at least 40 bits. A GPU could carry more than 4 GiB, a kernel could work on larger data,
        and, the part the rest of this appendix builds on, the host and the device could use exactly the same pointer
        value for a piece of data wherever it lived.
      </p>
      <Source section="C.4" pages="626" />
      <p>
        Two details of that paragraph do not survive a check against NVIDIA's own documents. The book dates Kepler to
        2013 and presents it as the generation that left 32 bits behind. NVIDIA launched the first Kepler GPUs in March
        2012. And Fermi, the generation before, already had 64-bit load and store addressing, a 40-bit unified address
        space, up to 6 GB of memory, and one address space for the host and every device of compute capability 2.0 or
        later in a 64-bit process.
      </p>
      <Today checked="2026-11-20 against NVIDIA's Kepler launch release, the Fermi whitepaper and the archived CUDA 11.0 guide (appC-facts.md, A6)">
        <p>Kepler: March 2012. Fermi (whitepaper 2009, products 2010) was first past 32 bits.</p>
        <p>"At least 40-bit physical" is the book's figure; no NVIDIA page we read states one.</p>
      </Today>
      <Ask q={questionsSC4['appC/C.4/q-fermi-tib']} />
      <p>
        The book's 48 bits for a CPU is the classic x86-64 layout: four levels of page tables, 256 TiB of virtual
        addresses. Newer CPUs and kernels can add a fifth level and 57-bit addresses, but Linux still places a program's
        memory below the 47-bit line unless the program asks for more. Whether CUDA copes with a host pointer above that
        line we could not find out, so this course says nothing either way.
      </p>
      <Today checked="2026-11-20 against the Linux kernel's 5-level paging documentation (appC-facts.md, A7)">
        <p>4-level paging: 256 TiB. 5-level: 128 PiB, and 56-bit user space on request.</p>
        <p>CUDA's handling of pointers above 47 bits: unverified.</p>
      </Today>

      <h2>The 32-bit limit that is still with you</h2>
      <p>
        Sixty-four-bit pointers did not remove every 32-bit limit from kernels. They moved it into code you write. The
        kernels of the early chapters compute their index as an <code>int</code>, and an <code>int</code> holds at most 2
        <sup>31</sup> − 1. However much memory the device has, that index stops at about two billion elements.
      </p>
      <Beyond>
        <p>
          The cap is plain C++, not a GPU fact. <a href={href('ch03', '3.4')}>Chapter 3, §3.4</a> shows how PyTorch,
          llama.cpp and MLX deal with it.
        </p>
      </Beyond>
      <Ask q={questionsSC4['appC/C.4/q-int-index']} />
      <p>
        Whether that matters depends on the device. Here is what yours reported, set against the 8 GiB of floats an{' '}
        <code>int</code> reaches.
      </p>
      <YourMemory />
      <p>
        The explorer puts the two limits side by side. Drag the address width and watch the reach double with each bit;
        the marked widths are the ones this appendix mentions, and your devices' figures sit under the bar. Then choose an
        element type and an index type and see which runs out first, the index or the address. Try an{' '}
        <code>int</code> over floats at 32 bits, then at 40.
      </p>
      <Simplification>
        <p>
          Reach counts addressable bytes only. Real devices and operating systems reserve parts of an address space, and
          physical and virtual widths are not modelled separately.
        </p>
      </Simplification>
      <BitsToBytes />

      <h2>On AMD and Apple GPUs</h2>
      <p>
        The book covers only NVIDIA. On AMD GPUs the compiler supports only 64-bit process address spaces, and shared
        memory (LDS) and per-thread scratch are 32-bit windows inside one flat 64-bit space. On Metal, every buffer has a
        GPU virtual address, <code>gpuAddress</code>, of type <code>MTLGPUAddress</code>, a 64-bit value.
      </p>
      <Beyond>
        <p>
          Checked 2026-11-20 against LLVM's AMDGPU documentation and the macOS 27.0 SDK headers (appC-facts.md, B1, C3).
        </p>
        <p>
          Whether a Metal buffer's <Id>gpuAddress</Id> equals its CPU pointer is not documented.
        </p>
      </Beyond>
      <p>
        Back to <a href={href('appC', 'start')}>the map</a>: C.4 and C.5 share the step marked C.4–C.5. Wider addresses
        change no cell on their own; the only cells that step changes are in the row for a peer GPU, which is section C.5.
      </p>
    </>
  );
}
