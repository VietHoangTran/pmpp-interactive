import { Ask, Predict, type Question } from '../../components/Ask';
import { Beyond, Simplification, Source, Today } from '../../components/notes';
import { useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { DerefMap } from './DerefMap';

const P_WEIGHTS = 'appC/start/p-weights';
/** 8 billion parameters, 2 bytes each. */
export const WEIGHT_BYTES = 8e9 * 2;

// ---------------------------------------------------------------- your device memory
interface MemRow {
  backend: string;
  name: string;
  /** [label, bytes] pairs, as the device reported them. */
  figures: [string, number][];
  /** The figure the weights are compared against. */
  budget: number;
  budgetLabel: string;
}

export function memoryRows(reports: Reports): MemRow[] {
  return Object.keys(reports)
    .filter((b) => b !== 'emu' && b !== 'race')
    .flatMap((b): MemRow[] => {
      const d = reports[b].device ?? {};
      const n = (k: string) => {
        const v = Number(d[k]);
        return isFinite(v) && v > 0 ? v : undefined;
      };
      const name = String(d.name ?? b);
      if (b === 'metal') {
        const ws = n('recommendedMaxWorkingSetSize');
        const phys = n('physicalMemory');
        if (!ws) return [];
        return [
          {
            backend: b,
            name,
            figures: [['recommendedMaxWorkingSetSize', ws], ...(phys ? [['physicalMemory', phys] as [string, number]] : [])],
            budget: ws,
            budgetLabel: 'the recommended working set',
          },
        ];
      }
      const total = n('totalGlobalMem');
      if (!total) return [];
      return [{ backend: b, name, figures: [['totalGlobalMem', total]], budget: total, budgetLabel: 'device memory' }];
    });
}

const gb = (bytes: number) => (bytes / 1e9).toFixed(1) + ' GB';

function YourMemory() {
  const { reports, loading } = useReports();
  if (loading) return <p className="muted">Reading your device reports…</p>;
  const rows = memoryRows(reports);
  if (!rows.length)
    return (
      <p data-yours="none" style={{ marginBottom: '1em' }}>
        <b>No device recorded on this machine.</b> Run <code>python3 labs/pmpp.py run appC/lab1 --bench</code> and
        reload this page: the first run on a backend records what the device reports.
      </p>
    );
  return (
    <div data-yours="some" style={{ marginBottom: '1em' }}>
      {rows.map((r) => {
        const fits = WEIGHT_BYTES <= r.budget;
        return (
          <p key={r.backend} data-backend={r.backend} data-fits={fits ? 'yes' : 'no'}>
            <b>
              {r.name} ({r.backend})
            </b>{' '}
            reports{' '}
            {r.figures.map(([k, v], i) => (
              <span key={k}>
                {i ? ' and ' : ''}
                <code>{k}</code> = {gb(v)}
              </span>
            ))}
            . The 16 GB of weights {fits ? 'fit under' : 'do not fit in'} {r.budgetLabel}
            {fits ? ', before activations and the KV cache ask for their share.' : '.'}
          </p>
        );
      })}
      {rows.some((r) => r.backend === 'metal') ? (
        <p className="muted">
          On Apple silicon the weights sit in the one memory both processors use. Apple calls{' '}
          <code>recommendedMaxWorkingSetSize</code> the amount the GPU can use "without affecting its runtime
          performance": advice, not a hard limit, and Apple documents no way to oversubscribe it (appC-facts C5).
        </p>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSC0: Record<string, Question> = {
  'appC/start/q-hostderef': {
    kind: 'choice',
    id: 'appC/start/q-hostderef',
    prompt: (
      <>
        CUDA host code. After <code>cudaMalloc((void**)&amp;A_d, n*sizeof(float))</code> and a <code>cudaMemcpy</code>{' '}
        into <code>A_d</code>, the host runs <code>printf("%f", A_d[0])</code>. What happens?
      </>
    ),
    options: [
      {
        label: 'It is not allowed: host code may not dereference cudaMalloc memory. The result is undefined, typically a crash.',
        correct: true,
      },
      {
        label: 'It prints the value, slowly, because the read crosses the interconnect.',
        why: 'That is what AMD documents for hipMalloc memory under HIP (appC-facts B3). CUDA does not allow it.',
      },
      {
        label: 'It prints the value, because unified addressing gives host and device one address space.',
        why: 'One address space is not one access right. Section C.3 is about exactly this difference.',
      },
      {
        label: 'It works once cudaDeviceSynchronize() has been called.',
        why: 'Synchronising orders the work. It does not make device memory readable from host code.',
      },
    ],
    explain: (
      <p>
        <code>A_d</code> holds an address in device memory. Host code may pass it to API calls and kernels, and nothing
        else (<a href={href('ch02', '2.4')}>chapter 2, section 2.4</a>). No era of this appendix changes that: even on
        systems with hardware coherence, NVIDIA says host code cannot reach <code>cudaMalloc</code> memory.
      </p>
    ),
    source: 'ch. 2 §2.4, p. 30; notes/today/ch02-facts.md A17; appC-facts A12, B3',
  },
  'appC/start/q-membound': {
    kind: 'numeric',
    id: 'appC/start/q-membound',
    prompt: (
      <>
        A memory-bound kernel reads 1 GiB once and takes 1 ms reading it from device memory. The same bytes come
        instead over a link with one tenth of that bandwidth. How many milliseconds now?
      </>
    ),
    answer: 10,
    tol: 0.01,
    unit: 'ms',
    explain: (
      <p>
        For a memory-bound kernel, time is bytes divided by bandwidth (
        <a href={href('ch05', '5.1')}>chapter 5, section 5.1</a>). Same bytes, a tenth of the bandwidth: ten times the
        time. One tenth is the book's own rule of thumb for PCIe against device memory (p. 625).{' '}
        <a href={href('appC', 'C.2')}>Section C.2</a> asks what that does to a kernel that reads only some of its bytes
        over the link.
      </p>
    ),
    source: 'ch. 5 §5.1, pp. 93–97; appendix C, p. 625',
  },
  'appC/start/q-sharedscope': {
    kind: 'choice',
    id: 'appC/start/q-sharedscope',
    prompt: (
      <>
        A kernel declares <code>__shared__ float tile[256]</code>. Thread 5 of block 2 writes <code>tile[5]</code>. Who
        can read that value?
      </>
    ),
    options: [
      { label: 'The threads of block 2, after a barrier.', correct: true },
      { label: 'Every thread of the grid.', why: 'Shared memory has block scope: every block has its own tile.' },
      { label: 'Only thread 5 of block 2.', why: 'That describes a register or a local array. A __shared__ variable is visible to its whole block.' },
      { label: 'The host, after the kernel finishes.', why: 'Shared memory lives only while its block runs. To keep a value, write it to global memory.' },
    ],
    explain: (
      <p>
        One copy per block, visible to that block's threads for the block's lifetime (
        <a href={href('ch05', '5.2')}>chapter 5, section 5.2</a>). Keep this in mind for{' '}
        <a href={href('appC', 'C.1')}>section C.1</a>: a pointer to <code>tile</code> means something only inside the
        block that owns it.
      </p>
    ),
    source: 'ch. 5 §5.2, pp. 98–104',
  },
  'appC/start/q-ordering': {
    kind: 'choice',
    id: 'appC/start/q-ordering',
    prompt: (
      <>
        Metal on Apple silicon, shared buffers. The CPU fills <code>input.contents</code>, commits a command buffer whose
        kernel reads <code>input</code>, then immediately writes the next batch into the same buffer. What is wrong?
      </>
    ),
    options: [
      { label: 'The GPU may not have read the first batch yet, so the CPU can overwrite it under the kernel.', correct: true },
      {
        label: 'Nothing: unified memory is coherent, so the kernel sees the right data.',
        why: 'Shared storage removes the copy, not the ordering. Apple makes you responsible for the ordering.',
      },
      {
        label: 'The write fails, because a buffer is read-only while a command buffer that uses it is in flight.',
        why: 'Nothing stops the write. Keeping the two sides apart is your job.',
      },
      {
        label: 'The kernel reads stale data from a GPU cache, so call didModifyRange first.',
        why: 'didModifyRange belongs to managed storage, which has no effect on Apple silicon.',
      },
    ],
    explain: (
      <p>
        <code>commit</code> returns while the work is still queued or running. With shared storage there is no copy to
        hide behind: wait for the command buffer to complete before you reuse the buffer, or write the next batch into
        a second buffer. You met the same rule in <a href={href('ch02', '2.4')}>chapter 2, section 2.4</a>.
      </p>
    ),
    source: 'not in the book; notes/today/ch05-facts.md C6',
  },
};

// ---------------------------------------------------------------- the page
export function SC0() {
  return (
    <>
      <p>
        In chapter 2 you wrote host code in three parts: allocate device memory and copy the inputs in, launch the
        kernel, then copy the result out and free. Behind it sits a simple picture of the machine. The CPU has its
        memory, the GPU has its own, and nothing crosses between them except what <code>cudaMemcpy</code> carries. The
        book says early GPUs supported nothing else.
      </p>
      <Source pages="623" />
      <p>
        You have used this model without writing it. In PyTorch, <code>tensor.to('cuda')</code> ends in a
        host-to-device <code>cudaMemcpyAsync</code>. On a Mac, <code>tensor.to('mps')</code> still copies, although the
        CPU and the GPU share one memory there.
      </p>
      <Beyond>
        <p>
          The PyTorch connection is ours, read from PyTorch's source on 2026-11-20 (notes/today/ch02-facts.md D4, as
          cited in appC-facts).
        </p>
      </Beyond>
      <p>
        This appendix is the book's history of how CUDA loosened that picture, one step at a time. It is seven pages
        with no exercises, and it is the background to a question you will keep meeting in inference code: where does
        this tensor live, and who can read it from there?
      </p>
      <Today checked="2026-11-20 against NVIDIA's and AMD's documentation (notes/today/appC-facts.md, A1)">
        <p>
          Still the default on discrete GPUs. NVIDIA's guide opens its memory chapter with separate physical memories,
          and NVIDIA and AMD both still describe GPU memory as faster but smaller than the CPU's.
        </p>
      </Today>

      <h2>Why copying everything stops being enough</h2>
      <p>The book names two problems with the copy model.</p>
      <Source pages="623" />
      <p>
        <b>Input and output devices.</b> Disk controllers and network cards are built to read and write host memory.
        Input from a disk lands in host memory and then has to be copied to the GPU, and output makes the same trip back.
        Each trip adds latency and costs throughput. Letting those devices work on GPU memory directly would be faster
        and simpler.
      </p>
      <Beyond>
        <p>
          NVIDIA's answer today is the GPUDirect family: storage drives and network adapters that read and write GPU
          memory directly (appC-facts A19). This course does not use it.
        </p>
      </Beyond>
      <p>
        <b>Data larger than the GPU.</b> Programs keep their data in host memory, and GPU memory tends to be faster but
        smaller. Data that does not fit has to be cut into pieces that do and moved through piece by piece. Chapter 21
        does exactly this, pushing a 3D grid through the GPU in 2D slices. Some structures have no good way to be cut
        at all. The book's better answers: let the GPU read host memory directly, or let the runtime move only the data
        a kernel actually touches.
      </p>
      <Source pages="623" figs="chapter 21's slices are on p. 517" />
      <p>
        The second problem is the one you will meet first, as a model that does not fit. Commit to a number before you
        look at your own device.
      </p>
      <Predict
        id={P_WEIGHTS}
        prompt={
          <>
            An 8-billion-parameter model stored in 16-bit values: how many GB (10<sup>9</sup> bytes) of weights?
          </>
        }
        unit="GB"
      >
        {(v) => (
          <>
            <p>
              {Math.abs(v - 16) < 0.05 ? 'Right: ' : 'The answer is 16 GB: '}8 × 10<sup>9</sup> parameters × 2 bytes
              each = 1.6 × 10<sup>10</sup> bytes. Here is what your devices report.
            </p>
            <YourMemory />
            <p>
              When a model does not fit, the appendix offers three answers. Cut it into pieces that fit and stream them
              through, as chapter 21 does. Read it in place from host memory over the interconnect (
              <a href={href('appC', 'C.2')}>section C.2</a>). Or let the runtime migrate pages on demand and
              oversubscribe GPU memory (<a href={href('appC', 'C.8')}>section C.8</a>).
            </p>
          </>
        )}
      </Predict>

      <h2>The map</h2>
      <p>
        The rest of the appendix, sections C.1 to C.9, is the history in seven steps; two of them take two sections
        each. Each step changes the answer to one question: given a pointer, who may dereference it, and what does it
        cost? The map below is that answer, era by era, plus one step the book does not reach. Step through it with Previous and Next, and look at the cells marked
        "changed". Three things to find out:
      </p>
      <ul>
        <li>when the addresses became one (C.3), and how many steps later access followed;</li>
        <li>
          whether a kernel can read plain <code>malloc</code> memory once Pascal brings page faults;
        </li>
        <li>
          whether <code>cudaMalloc</code> memory ever becomes readable from host code, on CUDA and on HIP.
        </li>
      </ul>
      <Simplification>
        <p>
          One cell is one documented rule; costs are classes, not numbers. Windows, WSL and some Tegra systems today
          share the cells of the CUDA 6 limited era, and that era is labelled so. "Not documented" means our sources do
          not say.
        </p>
      </Simplification>
      <p>
        Where our sources date a step differently from the book, the map shows both. Fermi's whitepaper is from 2009,
        but its products reached the market in March 2010. Kepler launched in 2012, not 2013, and it was not the first
        GPU with 64-bit addresses. CUDA 6 was announced in November 2013 and shipped in April 2014. The reserve-and-map
        calls of C.9 arrived in CUDA 10.2, not CUDA 11.
      </p>
      <Today checked="2026-11-20 against NVIDIA's documents and press releases (notes/today/appC-facts.md, A2, A6, A9, A15)">
        <p>Four of the book's dates need a correction. The map labels each one with its facts item.</p>
      </Today>
      <DerefMap />
      <p>
        Each section page of this appendix goes deeper into one era and links back to this map. The platform switch
        shows what the same rows look like on HIP and on Metal today. HIP turns out to have the book's eras too, chosen
        per GPU and per process instead of by date. Metal mostly has one answer, because it has one memory.
      </p>

      <h2>One name, two things</h2>
      <p>
        A warning before you go on, because the word will mislead you otherwise. CUDA's <b>Unified Memory</b> (section
        C.6) is a pool of pages that the CUDA runtime moves between two physical memories. Apple's <b>unified memory</b>{' '}
        is one physical memory that the CPU and the GPU share, with nothing to move. They share a name and little else.
        In this appendix the capital U always means CUDA's feature, and <a href={href('appC', 'C.6')}>section C.6</a>{' '}
        sets the two side by side.
      </p>
      <Beyond title="The two definitions">
        <p>
          NVIDIA: "A CUDA feature that enables managed memory which can be automatically migrated between CPU and
          GPUs." Apple: "the CPU and the GPU share system memory" (appC-facts, terminology; ch05-facts C6).
        </p>
      </Beyond>

      <h2>Before you start</h2>
      <p>
        Four questions on what this appendix assumes from chapters 2 and 5, and from the Metal side of chapter 2. Each
        explanation links back to where the idea comes from, and each question returns in{' '}
        <a href={href('review')}>review</a>.
      </p>
      <Ask q={questionsSC0['appC/start/q-hostderef']} label="From chapter 2" />
      <Ask q={questionsSC0['appC/start/q-membound']} label="From chapter 5" />
      <Ask q={questionsSC0['appC/start/q-sharedscope']} label="From chapter 5" />
      <Ask q={questionsSC0['appC/start/q-ordering']} label="From chapter 2, Metal" />
    </>
  );
}
