import { useMemo, useState, type ReactNode } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { CPU_WIDTHS, GPU_WIDTHS, GiB, binary, coverage, isEmulator, memoryFigures, smallestWidth } from '../../sim/addrwidth';
import { ROWS, checklist, granularityText, paradigm, paradigmFromRun, type Attrs } from '../../sim/umparadigm';

/** A long identifier that may break anywhere, for the narrow margin notes. */
function Id({ children }: { children: ReactNode }) {
  return <code style={{ whiteSpace: 'normal', overflowWrap: 'anywhere' }}>{children}</code>;
}

// ---------------------------------------------------------------- does it all fit
const GIB_STEPS = Array.from({ length: 20 }, (_, k) => 2 ** k); // 1 GiB .. 512 TiB
const SW = 640;

function DoesItFit() {
  const { reports } = useReports();
  const yours = useMemo(() => memoryFigures(reports).filter((f) => f.field === 'totalGlobalMem'), [reports]);
  const [gpuBits, setGpuBits] = useState(49);
  const [cpuBits, setCpuBits] = useState(48);
  const [gib, setGib] = useState(64);

  const bytes = gib * GiB;
  const c = coverage(gpuBits, cpuBits, bytes);
  const need = smallestWidth(cpuBits, bytes);
  const scale = Math.max(c.capacity, c.needed);
  const px = (b: number) => (b / scale) * (SW - 20);
  const cpuW = px(c.cpu);
  const gpuW = Math.max(6, px(c.gpu));
  const capW = px(c.capacity);
  const step = GIB_STEPS.reduce((best, v, k) => (Math.abs(Math.log2(v) - Math.log2(gib)) < Math.abs(Math.log2(GIB_STEPS[best]) - Math.log2(gib)) ? k : best), 0);

  return (
    <Bench
      title="Does it all fit?"
      task="Find the smallest GPU width that holds a 48-bit CPU space and your GPU's memory, then try Linux's default 47 bits."
      foot="Widths: the book (48-bit CPUs, 49-bit Pascal, pp. 626–628) and the Linux 5-level paging documentation (47 and 56 bits, appC-facts.md A7)."
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-start' }}>
        <div className="field" style={{ display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' }}>
          GPU virtual address width
          <span className="seg" role="group" aria-label="GPU virtual address width">
            {GPU_WIDTHS.map((b) => (
              <button key={b} aria-pressed={gpuBits === b} onClick={() => setGpuBits(b)}>
                {b}
              </button>
            ))}
          </span>
        </div>
        <label className="field" style={{ maxWidth: '100%' }}>
          CPU user space
          <select style={{ maxWidth: '100%' }} value={cpuBits} aria-label="CPU user space" onChange={(e) => setCpuBits(Number(e.target.value))}>
            {CPU_WIDTHS.map((w) => (
              <option key={w.bits} value={w.bits}>
                {w.label}
              </option>
            ))}
          </select>
        </label>
        <label className="field" style={{ minWidth: '14rem' }}>
          GPU memory to place: {binary(bytes).text}
          <input
            type="range"
            min={0}
            max={GIB_STEPS.length - 1}
            value={step}
            aria-label="GPU memory to place"
            onChange={(e) => setGib(GIB_STEPS[Number(e.target.value)])}
          />
        </label>
      </div>
      {yours.length ? (
        <div className="row" role="group" aria-label="Your devices" style={{ marginBottom: '0.8rem' }}>
          <span className="muted">Yours</span>
          {yours.map((f) => (
            <button key={f.backend} className="quiet" data-yours={f.backend} onClick={() => setGib(f.bytes / GiB)}>
              {f.name}: {binary(f.bytes).text}
            </button>
          ))}
        </div>
      ) : (
        <p className="muted" style={{ margin: '0 0 0.8rem' }}>
          No CUDA or HIP device is recorded on this machine, so there is no preset for your GPU's memory. On Apple silicon
          the GPU has no memory of its own to place: it uses the system's.
        </p>
      )}

      <div style={{ overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${SW} 118`} role="img" aria-label="The GPU's virtual address space with the CPU range and the GPU range packed into it" style={{ width: '100%', minWidth: 480, display: 'block' }} data-part="strip" data-fits={c.fits ? 1 : 0}>
          <text x={10} y={14} fontSize="11" style={{ fill: 'var(--ink-2)' }}>
            {gpuBits}-bit GPU virtual addresses: {binary(c.capacity).text}
          </text>
          <rect x={10} y={22} width={capW} height={44} fill="var(--cell)" stroke="var(--ink-2)" strokeWidth="1.5" data-part="capacity" />
          <rect x={10} y={30} width={Math.min(cpuW, capW)} height={28} fill="var(--rule-soft)" stroke="var(--ink-3)" />
          {cpuW > capW ? <rect x={10 + capW} y={30} width={cpuW - capW} height={28} fill="var(--fail-wash)" stroke="var(--fail)" /> : null}
          <rect x={10 + cpuW} y={30} width={gpuW} height={28} fill={c.gpuOutside > 0 ? 'var(--fail)' : 'var(--global)'} data-part="gpu" />
          <text x={14} y={48} fontSize="11">
            CPU user space, {binary(c.cpu).text}
          </text>
          <text x={Math.min(SW - 10, 10 + cpuW + gpuW + 4)} y={82} fontSize="11" textAnchor={10 + cpuW + gpuW > SW - 160 ? 'end' : 'start'} style={{ fill: c.gpuOutside > 0 ? 'var(--fail)' : 'var(--global)' }}>
            GPU memory, {binary(c.gpu).text}
            {c.gpu / scale < 0.01 ? ' (drawn wider than to scale)' : ''}
          </text>
          <text x={10} y={108} fontSize="12" fontWeight="600" data-part="verdict" style={{ fill: c.fits ? 'var(--ok)' : 'var(--fail)' }}>
            {c.fits
              ? `Fits, with ${binary(c.spare).text} to spare.`
              : c.cpuOutside > 0
                ? `Does not fit: the CPU range alone is ${Math.round(c.cpu / c.capacity).toLocaleString('en-US')} times the GPU's address space.`
                : `Does not fit: ${binary(c.over).text} of GPU memory falls past the end.`}
          </text>
        </svg>
      </div>
      <p style={{ margin: '0.6rem 0 0' }} data-out="smallest" data-bits={need}>
        Smallest width that holds this CPU range plus {binary(bytes).text}: <b>{need} bits</b>.
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- which paradigm
const ATTR_NAMES: { key: keyof Attrs; cuda: string }[] = [
  { key: 'concurrentManagedAccess', cuda: 'cudaDevAttrConcurrentManagedAccess' },
  { key: 'pageableMemoryAccess', cuda: 'cudaDevAttrPageableMemoryAccess' },
  { key: 'usesHostPageTables', cuda: 'cudaDevAttrPageableMemoryAccessUsesHostPageTables' },
];
const PRESET: Record<string, Attrs> = {
  limited: { concurrentManagedAccess: false, pageableMemoryAccess: false, usesHostPageTables: false },
  managed: { concurrentManagedAccess: true, pageableMemoryAccess: false, usesHostPageTables: false },
  hmm: { concurrentManagedAccess: true, pageableMemoryAccess: true, usesHostPageTables: false },
  ats: { concurrentManagedAccess: true, pageableMemoryAccess: true, usesHostPageTables: true },
};

function WhichParadigm() {
  const { reports, loading } = useReports();
  const [a, setA] = useState<Attrs>(PRESET.managed);
  const [note, setNote] = useState<ReactNode>(null);
  const p = paradigm(a);
  const checks = checklist(a);

  const yours = () => {
    const found = Object.keys(reports)
      .filter((b) => !isEmulator(b) && b !== 'metal')
      .map((b) => ({ b, attrs: paradigmFromRun(reports[b]), name: String(reports[b].device.name ?? b), when: reports[b].generated }))
      .filter((x) => x.attrs);
    if (found.length) {
      const f = found[0];
      setA(f.attrs as Attrs);
      setNote(
        <>
          Yours: {f.name} ({f.b}), as lab C.1 reported it on {f.when.slice(0, 10)}.
          {found.length > 1 ? ` Other backends with a report: ${found.slice(1).map((x) => x.b).join(', ')}.` : ''}
        </>,
      );
      return;
    }
    const metal = Object.keys(reports).includes('metal');
    setNote(
      <>
        No lab C.1 result from a CUDA or HIP machine yet. Run{' '}
        <code>python3 labs/pmpp.py run appC/lab1 --bench</code> on one, then come back.
        {metal
          ? ' Your Metal device has no such attributes: Metal has no managed memory on Apple silicon, and CPU memory reaches the GPU only when it is wrapped in a buffer.'
          : ''}
      </>,
    );
  };

  return (
    <Bench
      title="Which paradigm is this machine?"
      task="Set the switches for Linux without HMM, for WSL and for Grace Hopper, and compare what works."
      foot="The table and the attribute names are NVIDIA's (CUDA guide 2.6.2.1); what works on each row comes from guide sections 2.6.2, 4.1.2 and 4.1.3 (appC-facts.md, A10 to A14). HIP has attributes of the same names and meanings (B5)."
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-start' }}>
        {ATTR_NAMES.map((n) => (
          <label key={n.key} className="check" style={{ alignItems: 'flex-start' }}>
            <input
              type="checkbox"
              data-attr={n.key}
              checked={a[n.key]}
              onChange={(e) => {
                setA({ ...a, [n.key]: e.target.checked });
                setNote(null);
              }}
            />
            <span>
              <Id>{n.cuda}</Id> = {a[n.key] ? 1 : 0}
            </span>
          </label>
        ))}
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <button className="primary" onClick={yours} disabled={loading}>
          Yours
        </button>
        {note ? (
          <span data-out="yours" style={{ flex: '1 1 20rem' }}>
            {note}
          </span>
        ) : null}
      </div>

      <div className="cols plot">
        <div className="panel" style={{ overflowX: 'auto' }}>
          <p className="muted" style={{ margin: '0 0 0.2rem' }}>
            Paradigm
          </p>
          <p style={{ fontSize: '1.25rem', fontWeight: 600, margin: '0 0 0.6rem', lineHeight: 1.25 }}>
            <output data-out="paradigm" data-key={p.row.key} data-listed={p.listed ? 1 : 0}>
              {p.row.name}
            </output>
          </p>
          {!p.listed ? (
            <p style={{ margin: '0 0 0.6rem' }}>
              <span className="chip fail">not in the table</span> {p.note}
            </p>
          ) : null}
          <table className="data" style={{ margin: 0 }}>
            <thead>
              <tr>
                <th style={{ paddingLeft: '0.4rem' }}>Paradigm, and where NVIDIA says you find it</th>
                <th>Attributes</th>
              </tr>
            </thead>
            <tbody>
              {ROWS.map((r) => (
                <tr
                  key={r.key}
                  data-row={r.key}
                  data-on={r.key === p.row.key ? 1 : 0}
                  onClick={() => {
                    setA(PRESET[r.key]);
                    setNote(null);
                  }}
                  style={{ cursor: 'pointer', background: r.key === p.row.key ? 'var(--rule-soft)' : undefined }}
                >
                  <td style={{ paddingLeft: '0.4rem' }}>
                    <span style={{ fontWeight: r.key === p.row.key ? 650 : 500 }}>{r.name}</span>
                    <br />
                    <span className="muted">{r.where}</span>
                  </td>
                  <td style={{ minWidth: '9rem' }}>{r.condition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="panel">
          <p className="muted" style={{ margin: '0 0 0.4rem' }}>
            What works here
          </p>
          <ul style={{ listStyle: 'none', padding: 0, margin: 0 }} data-part="checks">
            {checks.map((k) => (
              <li key={k.key} data-check={k.key} data-answer={k.answer} style={{ marginBottom: '0.55rem' }}>
                <span className={'chip ' + (k.answer === 'yes' ? 'ok' : k.answer === 'no' ? 'fail' : '')}>{k.answer}</span> {k.what}
                <br />
                <span className="muted">{k.why}</span>
              </li>
            ))}
            <li data-check="granularity">
              <span className="chip">unit</span> Managed data moves or stays coherent in: {granularityText(a)}.
            </li>
          </ul>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSC7: Record<string, Question> = {
  'appC/C.7/q-49': {
    kind: 'numeric',
    id: 'appC/C.7/q-49',
    prompt: (
      <>
        What is the smallest address width, in bits, that holds two separate ranges of 2<sup>48</sup> bytes each?
      </>
    ),
    answer: 49,
    unit: 'bits',
    explain: (
      <p>
        2 × 2<sup>48</sup> = 2<sup>49</sup>. One range is a 48-bit CPU address space; the other has room for every GPU's
        memory with plenty to spare. That is the book's reasoning for Pascal's 49 bits.
      </p>
    ),
    source: 'App. C §C.7, pp. 627–628',
  },
  'appC/C.7/q-paradigm': {
    kind: 'choice',
    id: 'appC/C.7/q-paradigm',
    prompt: (
      <>
        A device reports <code>cudaDevAttrConcurrentManagedAccess</code> = 1 and{' '}
        <code>cudaDevAttrPageableMemoryAccess</code> = 0. Which holds?
      </>
    ),
    options: [
      {
        label: 'Everything works, malloc and mmap pointers included: concurrent managed access means the top paradigm.',
        why: 'Full support covers managed allocations. Memory from malloc needs PageableMemoryAccess = 1.',
      },
      {
        label: 'It is the limited paradigm: managed memory moves in bulk at each launch.',
        why: 'That is ConcurrentManagedAccess = 0.',
      },
      {
        label:
          'Full Unified Memory for managed allocations only: concurrent CPU access and migration on fault, but no malloc pointers in kernels.',
        correct: true,
      },
      { label: 'Managed memory is not supported, because PageableMemoryAccess = 0 rules it out.', why: 'Both of these attributes describe how managed memory is supported; neither says it is missing.' },
    ],
    explain: (
      <p>
        NVIDIA's table puts this combination in its second row. Pascal-class page faulting covers memory allocated as
        managed; system-allocated memory needs pageable memory access, which means HMM or ATS.
      </p>
    ),
    source: 'CUDA guide 2.6.2.1 and 4.1.2, checked 2026-11-20 (notes/today/appC-facts.md, A12, A14)',
  },
  'appC/C.7/q-pascal-malloc': {
    kind: 'choice',
    id: 'appC/C.7/q-pascal-malloc',
    prompt: (
      <>
        A Pascal GPU in a Linux machine without HMM. A kernel follows a linked list that the host built with{' '}
        <code>new</code>. What happens?
      </>
    ),
    options: [
      { label: "It works: the GPU faults on each node, and the driver migrates that node's page to GPU memory.", why: 'Page faulting on this machine covers managed allocations only. Memory from new is not managed.' },
      { label: 'It works, more slowly, because each node is read in place over the interconnect.', why: 'That would be mapped (zero-copy) memory, which new does not give you either.' },
      {
        label: "It fails: Pascal's faulting covers managed memory; new needs HMM or ATS.",
        correct: true,
      },
      { label: 'It works while the whole list fits in GPU memory, and fails once it outgrows it.', why: 'Size is not the issue. The GPU has no access to that memory at all.' },
    ],
    explain: (
      <p>
        The book says from Pascal on a GPU can traverse a host-built linked structure, and leaves open what allocated it.
        NVIDIA's guide answers: "the GPU cannot access system-allocated memory" on such a machine. Build the list in
        managed memory, or run where <code>PageableMemoryAccess</code> is 1.
      </p>
    ),
    source: 'App. C §C.8, pp. 628–629, against the CUDA guide 4.1.2 (notes/today/appC-facts.md, A12)',
  },
  'appC/C.7/q-wsl': {
    kind: 'choice',
    id: 'appC/C.7/q-wsl',
    prompt: <>The same program runs under WSL on the same GPU. Which managed-memory behaviour applies?</>,
    options: [
      { label: 'The same as native Linux on that GPU, since the paradigm is a property of the GPU.', why: 'The paradigm depends on the platform as well as the GPU. NVIDIA lists Windows and WSL among the limited systems.' },
      { label: 'HMM, or at least full Unified Memory: WSL runs a real Linux kernel, so the Linux rules apply.', why: 'NVIDIA lists WSL with Windows, in the limited paradigm.' },
      {
        label: 'The limited one: bulk moves at launch, no CPU access while the GPU runs, no oversubscription.',
        correct: true,
      },
      { label: 'No managed memory at all.', why: 'Managed memory works there, with the limited rules.' },
    ],
    explain: (
      <p>
        <code>ConcurrentManagedAccess</code> is 0 on Windows and WSL. These are the rules the book describes for Kepler
        and Maxwell, still in force today wherever that attribute is 0.
      </p>
    ),
    source: 'CUDA guide 2.6.2.3 and 4.1.3, checked 2026-11-20 (notes/today/appC-facts.md, A10, A14)',
  },
};

// ---------------------------------------------------------------- the section
export function SC7() {
  return (
    <>
      <p>
        By the book's account, the next widening came with the Pascal architecture in 2016. Its address translation grew
        to 49-bit virtual addresses: enough for the 48-bit address space of a CPU and, beside it, the GPU's own memory.
        CPUs and GPUs could then, in the book's words, "truly share the pointer values", and a GPU could walk
        pointer-based data structures that live in host memory.
      </p>
      <Source section="C.7" pages="627–628" />
      <p>
        NVIDIA's own account agrees on the width and the year, and ties it to CUDA 8: 49 bits are, in its words,
        "sufficient to enable GPUs to access the entire system memory plus the memory of all GPUs in the system".
      </p>
      <Today checked="2026-11-20 against NVIDIA's Pascal Unified Memory post and the toolkit archive (appC-facts.md, A11)">
        <p>
          CUDA 8: September 2016 (toolkit archive). The book's 2016 for Pascal stands (P100 overview, © 2016).
        </p>
      </Today>
      <Ask q={questionsSC7['appC/C.7/q-49']} label="Work it out" />
      <p>
        The explorer packs a CPU's user range and some GPU memory end to end into a GPU's virtual address space. Try 48
        bits against a 48-bit CPU, then 49. Then pick Linux's default user space: on a machine with five levels of page
        tables, Linux still keeps a program below the 47-bit line unless it asks for more, and 47 plus some GPU memory
        fits in 48. Whether CUDA handles a host pointer above 47 bits we could not verify.
      </p>
      <Simplification>
        <p>The ranges are packed end to end. Real layouts reserve regions, and our sources do not document them.</p>
      </Simplification>
      <DoesItFit />

      <h2>Naming an address is not permission to use it</h2>
      <p>
        "Share pointer values" invites a reading the book never commits to: that a kernel can follow any host pointer.
        The book says a Pascal GPU can traverse a host-built linked structure, and leaves open what allocated it. NVIDIA's
        documentation settles it. Pascal's page faulting covers memory allocated as managed. A kernel reaches plain{' '}
        <code>malloc</code>, <code>new</code> and <code>mmap</code> memory only on systems with HMM, which is a software
        mechanism in Linux, or ATS, which is hardware, on Grace Hopper and Grace Blackwell. And in the other direction,
        memory from <code>cudaMalloc</code> stays out of host code's reach even with hardware coherence.
      </p>
      <Today checked="2026-11-20 against the CUDA 13.5 guide 2.6.2 and 4.1 (appC-facts.md, A12)">
        <p>HMM needs Linux 6.1.24+, 6.2.11+ or 6.3+, compute capability 7.5+, and driver 535+ with the open kernel modules.</p>
        <p>
          Check: <Id>nvidia-smi -q | grep Addressing</Id> prints "Addressing Mode : HMM".
        </p>
      </Today>
      <Ask q={questionsSC7['appC/C.7/q-pascal-malloc']} />

      <h2>Which machine do you have?</h2>
      <p>
        You do not need the hardware history to tell which rules apply where your code runs. Three device attributes
        answer it: <code>cudaDevAttrConcurrentManagedAccess</code>, <code>cudaDevAttrPageableMemoryAccess</code> and{' '}
        <code>cudaDevAttrPageableMemoryAccessUsesHostPageTables</code>. NVIDIA's guide sorts machines into four
        paradigms by their values. The first is the book's Kepler and Maxwell picture, which survives today on Windows,
        WSL and some Tegra systems. The other three differ in which memory the GPU can use and in how coherence is kept.
      </p>
      <Today checked="2026-11-20 against the CUDA 13.5 guide 2.6.2.1 (appC-facts.md, A14)">
        <p>Query each with <Id>cudaDeviceGetAttribute</Id>. HIP has the same three under <Id>hipDeviceAttribute…</Id> names (B5).</p>
      </Today>
      <p>
        Set the switches and read what works. The "Yours" button loads what <a href={href('appC', 'labs')}>lab C.1</a>{' '}
        reported on a CUDA or HIP machine, if you have run it there.
      </p>
      <WhichParadigm />
      <Ask q={questionsSC7['appC/C.7/q-paradigm']} />
      <Ask q={questionsSC7['appC/C.7/q-wsl']} />
      <p>
        So one CUDA source can behave three ways on the same GPU: under WSL with the limited rules, on Linux without HMM
        with managed memory only, and on a newer Linux with HMM using any pointer. A program that relies on more than the
        limited rules should check these attributes at start-up instead of assuming.
      </p>

      <h2>Where you meet it</h2>
      <p>
        On HMM and ATS systems, file-backed memory from <code>mmap</code> counts as unified memory too. A kernel could
        then read the weights of a memory-mapped model file in place, with no load step in between. That is what
        llama.cpp's Metal backend does on a Mac: it wraps the mmapped weight file in Metal buffers instead of copying it.
        On a PCIe machine without HMM, a CUDA kernel cannot follow a pointer into the mapped file, because
        system-allocated memory is not GPU-accessible there.
      </p>
      <Beyond>
        <p>
          NVIDIA's guide: with pageable memory access, "This includes file-backed memory allocations created with mmap"
          (A12). NVIDIA's HMM post reruns the book's Fig. C.1 with plain <Id>malloc</Id> (A9).
        </p>
        <p>llama.cpp wraps the mapped range with <Id>newBufferWithBytesNoCopy</Id> (D4).</p>
      </Beyond>

      <h2>On AMD and Apple GPUs</h2>
      <p>
        AMD's GPUs follow the same idea under other switches. A kernel can use memory from <code>new</code> or{' '}
        <code>malloc</code> only on CDNA2 or later, with <code>HSA_XNACK=1</code> set and a kernel driver with HMM support,
        and the first GPU access to such a page is a recoverable page fault. The three attributes exist in HIP with
        CUDA's meanings. On Metal there are no such attributes: a kernel uses memory the CPU allocated in place only once
        it is wrapped in a buffer with <code>newBufferWithBytesNoCopy</code>, which needs page-aligned memory (see{' '}
        <a href={href('ch02', '2.4')}>chapter 2, §2.4</a>).
      </p>
      <Beyond>
        <p>Checked 2026-11-20 against AMD's HIP unified memory page and the ROCm 10.2 header (appC-facts.md, B5), and Apple's MTLDevice header (ch02-facts.md, C2).</p>
      </Beyond>
      <p>
        Back to <a href={href('appC', 'start')}>the map</a>: C.7 and C.8 share the step marked C.7–C.8, which changes the
        managed-memory row. Plain <code>malloc</code> memory has to wait for the step beyond the book, HMM and ATS.
      </p>
    </>
  );
}
