import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Aside, Source } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { INDEX } from '../../sim/addrwidth';
import { breakEven, BOOK_RULE } from '../../sim/zerocopy';
import { DerefMap } from './DerefMap';

// A program of our own, for one question: a managed table of layers that holds device
// pointers. Checked copy: labs/appC/snippets/wrap_layer_table.cu (Wrap.snippets.test.ts
// ties the two; ./verify.sh front-end-checks the file).
export const LAYER_TABLE = String.raw`#include <cstdio>

// Where one layer's keys and values live: two device pointers.
struct LayerKV {
    float* k;
    float* v;
};

// Block b copies the first key of layer b. The table of layers is read from
// memory, and then the pointer found in it is followed.
__global__ void firstKeys(const LayerKV* layers, float* out) {
    if (threadIdx.x == 0) out[blockIdx.x] = layers[blockIdx.x].k[0];
}

void checkLayers(int nLayers, size_t perLayer) {
    LayerKV* layers;
    float* out;
    cudaMallocManaged((void**)&layers, nLayers * sizeof(LayerKV));   // the table: managed
    cudaMallocManaged((void**)&out, nLayers * sizeof(float));
    for (int l = 0; l < nLayers; ++l) {
        cudaMalloc((void**)&layers[l].k, perLayer * sizeof(float));   // the keys: device memory
        cudaMalloc((void**)&layers[l].v, perLayer * sizeof(float));
        cudaMemset(layers[l].k, 0, perLayer * sizeof(float));
    }
    firstKeys<<<nLayers, 32>>>(layers, out);
    cudaDeviceSynchronize();
    printf("%f\n", out[0]);            // line A
    printf("%f\n", layers[0].k[0]);    // line B
}`;

// The numbers the questions ask for.
/** A KV cache laid out [layer][token][head][dim]: elements per layer. */
export const KV = { layers: 32, tokens: 32768, heads: 32, dim: 128 };
export const KV_PER_LAYER = KV.tokens * KV.heads * KV.dim;
/** An int offset reaches 2^31 − 1, so the smallest value of `layer` whose offset does not fit. */
export const KV_FIRST_BAD = Math.ceil(2 ** INDEX.int32.log2Count / KV_PER_LAYER);
/** The embedding question: one row of the table read per generated token. */
export const VOCAB = 128000;
export const EMBED_BREAK_EVEN = breakEven(1 / VOCAB, BOOK_RULE);

const n = (v: number) => v.toLocaleString('en-US');

export const questionsWrap: Record<string, Question> = {
  'appC/wrap/q-layer-table': {
    kind: 'choice',
    id: 'appC/wrap/q-layer-table',
    prompt: (
      <>
        <code>checkLayers</code> above runs on Linux with full Unified Memory (<code>ConcurrentManagedAccess</code> = 1).
        What happens at lines A and B?
      </>
    ),
    options: [
      {
        label: 'Both print 0.',
        why: 'Line A is fine: out is managed and the kernel has finished. Line B reads layers[0].k, which is fine, and then dereferences it on the host. That pointer came from cudaMalloc.',
      },
      {
        label: 'Line A prints 0. Line B is not allowed: host code dereferences cudaMalloc memory, although the table that holds the pointer is managed.',
        correct: true,
      },
      {
        label: 'The kernel fails: a kernel cannot follow a pointer that it read from memory.',
        why: 'It can, as long as the memory at the other end is memory the kernel may dereference. The table is managed and the keys are device memory; both qualify.',
      },
      {
        label: 'Both fail: managed memory cannot hold device pointers.',
        why: 'A pointer stored in memory is just a number, and any memory can hold it. What matters is who dereferences it, and where it points.',
      },
    ],
    explain: (
      <>
        <p>
          Follow each pointer to where it points. <code>layers</code> is managed, so the host may write the table and
          the kernel may read it. Each <code>k</code> is <code>cudaMalloc</code> memory: the kernel may follow it, the
          host may only hand it to API calls. Line B reads the pointer from managed memory without trouble and then
          dereferences device memory on the host, which no era of this appendix allows, not even hardware coherence.
          Copy the value out with <code>cudaMemcpy</code>, or put the keys in managed memory too.
        </p>
        <p>
          Two places where this does not show. Under HIP, AMD documents host reads of <code>hipMalloc</code> memory as
          allowed, over the interconnect, so line B prints. On the course's CPU emulator host and device memory are one
          memory, so the mistake goes unnoticed there as well.
        </p>
      </>
    ),
    source: 'book §C.3, p. 625 (pointers read from memory) and §C.6, pp. 626–627; appC-facts A12, B3; ch02-facts A17',
  },
  'appC/wrap/q-kv-offset': {
    kind: 'numeric',
    id: 'appC/wrap/q-kv-offset',
    prompt: (
      <>
        A KV cache holds the keys of {KV.layers} layers in one array of half-precision values, laid out [layer][token]
        [head][dim] with {n(KV.tokens)} tokens, {KV.heads} heads and a head size of {KV.dim}. A kernel finds its layer
        with <code>const __half* k = cache + layer * tokens * heads * dim</code>, where every variable on the right is
        an <code>int</code>. Pointers are 64 bits wide. What is the smallest value of <code>layer</code> for which the
        pointer comes out wrong?
      </>
    ),
    answer: KV_FIRST_BAD,
    explain: (
      <>
        <p>
          The product is computed in <code>int</code> before it is added to the pointer. One layer is{' '}
          {n(KV.tokens)} × {KV.heads} × {KV.dim} = 2<sup>27</sup> elements, and an <code>int</code> holds at most 2
          <sup>31</sup> − 1. Layer 15 starts at 15 × 2<sup>27</sup>, which fits; layer {KV_FIRST_BAD} starts at exactly
          2<sup>31</sup>, which does not. Signed overflow is undefined in C++, and in practice the offset wraps negative
          and the pointer lands before the cache.
        </p>
        <p>
          The 64-bit pointer does not help, because the narrow part is the arithmetic. The whole array is 8 GiB, nowhere
          near any address limit of <a href={href('appC', 'C.4')}>section C.4</a>. Write{' '}
          <code>(size_t)layer * tokens * heads * dim</code> so that every multiplication happens in 64 bits.
        </p>
      </>
    ),
    source: 'ours: C++ integer arithmetic; the address widths are §C.4, pp. 625–626; index width in ch. 3 §3.4',
  },
  'appC/wrap/q-private-logits': {
    kind: 'choice',
    id: 'appC/wrap/q-private-logits',
    prompt: (
      <>
        A Metal decode loop on Apple silicon writes its logits into a buffer with private storage, because private is
        for data only the GPU touches. Sampling runs on the CPU. What does the CPU have to do to read the logits after
        each token?
      </>
    ),
    options: [
      {
        label: 'Read logits.contents after waitUntilCompleted: the CPU and the GPU share one memory.',
        why: 'One physical memory does not make every buffer CPU-visible. contents is NULL for a private buffer.',
      },
      {
        label: 'Encode a blit that copies the logits into a shared buffer, wait for that command buffer, then read the shared buffer. Or keep the logits in a shared buffer from the start.',
        correct: true,
      },
      {
        label: 'Call didModifyRange on the buffer, then read it.',
        why: 'didModifyRange belongs to managed storage, which has no effect on Apple silicon.',
      },
      {
        label: 'Nothing can make them readable: sampling has to move to the GPU.',
        why: 'A blit copy between buffers is exactly how data gets into or out of a private buffer.',
      },
    ],
    explain: (
      <p>
        Apple: private resources are not CPU-accessible, and <code>contents</code> returns NULL for them. Data reaches
        or leaves a private buffer through a blit copy into or from a shared one, which is what lab C.1's{' '}
        <code>feedPrivateCopy</code> does at both ends. Logits that the CPU reads every token are not "data only the GPU
        touches", so a shared buffer, read after the wait, is the simpler choice. On one memory the question is never
        whether bytes must cross a link; it is which buffer type you chose.
      </p>
    ),
    source: 'not in the book; ch02-facts C2, C3; ch05-facts C6; appC-facts C2',
  },
  'appC/wrap/q-mapped-atomics': {
    kind: 'choice',
    id: 'appC/wrap/q-mapped-atomics',
    prompt: (
      <>
        A sampling kernel counts accepted tokens with <code>atomicAdd</code> on a counter in mapped host memory, so that
        the host can read it without a copy. While the kernel runs, a host thread also adds to the same counter with{' '}
        <code>std::atomic</code>. What does NVIDIA's documentation say?
      </>
    ),
    options: [
      {
        label: 'It is safe: both sides use atomic operations on the same physical memory.',
        why: 'NVIDIA says the GPU\'s atomics on mapped memory are not atomic from the host\'s point of view.',
      },
      {
        label: 'Updates can be lost: atomics on mapped memory are not atomic from the host\'s point of view, and are not guaranteed to be supported on every system.',
        correct: true,
      },
      {
        label: 'It is safe as long as the counter is pinned.',
        why: 'Mapped memory is pinned already. Pinning keeps the pages from being paged out; it says nothing about atomicity.',
      },
      {
        label: 'It does not compile: atomicAdd accepts only pointers to device memory.',
        why: 'Under unified addressing the kernel receives an ordinary pointer, and atomicAdd compiles. The problem is at run time.',
      },
    ],
    explain: (
      <p>
        Mapped memory makes host memory reachable, not shared in every sense. Keep a counter that both sides update in
        two parts: the GPU counts in device memory and the host reads it after the kernel, or each side keeps its own
        count and you add them at the end. The book's own rule from <a href={href('appC', 'C.6')}>section C.6</a>{' '}
        applies here too: concurrent access is still yours to order.
      </p>
    ),
    source: 'vendor documentation, not in the book: appC-facts A3; book §C.6, p. 626 for synchronising concurrent access',
  },
  'appC/wrap/q-embed-mapped': {
    kind: 'numeric',
    id: 'appC/wrap/q-embed-mapped',
    prompt: (
      <>
        A model's token-embedding table has {n(VOCAB)} rows and sits in mapped host memory. Each generated token reads
        one row, a different one every time. In section C.2's bandwidth-only model, with the book's link ratio of 10%,
        after how many generated tokens would copying the whole table to device memory first have been cheaper?
      </>
    ),
    answer: Math.round(EMBED_BREAK_EVEN),
    tol: 0.02,
    unit: 'tokens',
    explain: (
      <>
        <p>
          Each token reads 1/{n(VOCAB)} of the table, and reading in place wins while the reads over the table's life
          add up to less than 1 / (1 − 0.1) ≈ 1.11 tables. That is {n(VOCAB)} / 0.9 ≈ {n(Math.round(EMBED_BREAK_EVEN))}{' '}
          tokens, far more than one generation produces. The embedding table is the "occasionally and sparsely" read
          data the book has in mind: in this model it can stay in host memory and free its space in device memory.
        </p>
        <p>
          The model leaves out latency, which a single row per token pays in full, and the prompt, whose many tokens
          read many rows at once. Lab C.1's <code>sparse1</code> workload is this lookup at one row in 64; measure it
          before you decide.
        </p>
      </>
    ),
    source: 'book p. 625 ("occasionally and sparsely", the 10%); the break-even rule is ours (§C.2 page); the lookup is ch. 20, p. 480',
  },
};

const LAB = 'appC/lab1_feeding';
// Backends that are not a GPU: the emulator and its race-detection flavour.
const EMULATED = new Set(['emu', 'race']);

export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const devices = Object.keys(reports).filter((b) => !EMULATED.has(b));
  const mine = devices.filter((b) => labStatus(reports[b], LAB, 'work') === 'pass');
  const mineEmu = labStatus(reports.emu, LAB, 'work') === 'pass';
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('appC/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>What to carry forward</h2>
      <ul>
        <li>
          Every pointer question has two halves. Can the value name the memory, and may this code dereference it there?
          The appendix's history is mostly the first half arriving (one address space on the GPU, then one for host and
          devices, then addresses wide enough for all of it) with the second following later, or not at all.
        </li>
        <li>
          In CUDA and HIP one helper with a plain pointer serves shared, local and global memory: the address decides
          which memory a load touches, and the speed depends on which. MSL as specified makes every pointer name its
          address space, so a helper is written once per memory it is called on.
        </li>
        <li>
          Reading host memory in place pays the interconnect on every access. It wins only while the bytes a kernel
          reads over its whole life stay below about one table's worth; past that, copy first. On Apple silicon there is
          no link to pay, and the choice becomes which buffer type you use.
        </li>
        <li>
          One virtual address space tells the runtime where a pointer lives, which is why <code>cudaMemcpyDefault</code>{' '}
          works. It does not say who may use the pointer. In CUDA, host code never dereferences{' '}
          <code>cudaMalloc</code> memory, on any system; HIP allows it at link speed.
        </li>
        <li>
          Unified Memory, with a capital U, is a pool of pages the CUDA runtime moves between two memories. Three device
          attributes say which kind you have: limited (managed memory moves in bulk at a launch, and the CPU keeps out
          while the GPU runs), full (pages move on a fault, and you may oversubscribe), and full for all host memory (HMM or ATS).
          Apple's unified memory is one physical memory with nothing to move.
        </li>
        <li>
          A GPU page fault lets the runtime migrate a page or map it. Migrating pays for every page moved, in work
          proportional to the page size; mapping pays on every access. Hints change the cost, never the result.
        </li>
        <li>
          Reaching another GPU's memory, and memory you mapped yourself, takes an explicit grant:{' '}
          <code>cudaDeviceEnablePeerAccess</code>, one direction at a time, and <code>cuMemSetAccess</code> after{' '}
          <code>cuMemMap</code>. Reserving addresses first lets a buffer such as a KV cache grow without moving.
        </li>
        <li>
          None of this orders anything. One pointer, coherent pages or one physical memory each remove a copy, and every
          one of them still leaves you to make the CPU and the GPU take turns.
        </li>
      </ul>
      <Source section="C.1–C.9" pages="623–629" />
      <p>
        The map from the <a href={href('appC', 'start')}>start page</a>, compact, opened at the step beyond the book.
        Switch to HIP or Metal to see the same rows today, and click a cell for its rule and source.
      </p>
      <DerefMap compact initialEra="hmm" />

      <h2>A mixed check</h2>
      <p>
        These draw on the whole appendix and on situations the sections did not show.
        {asked ? ` So far in this appendix you have answered ${right} of ${asked} questions correctly on the first recorded try.` : ''}
      </p>
      <Code
        tabs={[
          {
            label: 'CUDA (not a lab file: for this question only)',
            code: LAYER_TABLE,
            tiers: { layers: 'global', out: 'global' },
            status: 'front-end-checked; has a defect on purpose',
          },
        ]}
        caption={
          <>
            Ours, not the book's. The checked copy is <code>labs/appC/snippets/wrap_layer_table.cu</code>. Error checks are
            left out to keep the question short.
          </>
        }
      />
      <Ask q={questionsWrap['appC/wrap/q-layer-table']} label="Find the defect" />
      <Ask q={questionsWrap['appC/wrap/q-kv-offset']} />
      <Ask q={questionsWrap['appC/wrap/q-private-logits']} />
      <Ask q={questionsWrap['appC/wrap/q-mapped-atomics']} />
      <Ask q={questionsWrap['appC/wrap/q-embed-mapped']} />

      <h2>The book's exercises</h2>
      <p>
        The appendix has none: its seven pages end with a single reference. In their place, here is where each part of
        it is worked in this course, so that you can check you have met all of it.
      </p>
      <table className="data wide" style={{ maxWidth: '58rem' }} data-table="exercises">
        <thead>
          <tr><th>Book</th><th>Pages</th><th>Where it is worked here</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>The copy model and its two problems</td>
            <td>623</td>
            <td>the weights prediction and the map on the <a href={href('appC', 'start')}>start page</a></td>
          </tr>
          <tr>
            <td>C.1</td>
            <td>624</td>
            <td>the address-window explorer and the compile cards in <a href={href('appC', 'C.1')}>C.1</a></td>
          </tr>
          <tr>
            <td>C.2</td>
            <td>624–625</td>
            <td>the cost explorer in <a href={href('appC', 'C.2')}>C.2</a>; the mapped path of lab C.1</td>
          </tr>
          <tr>
            <td>C.3</td>
            <td>625</td>
            <td>the copy-direction explorer and the list stepper in <a href={href('appC', 'C.3')}>C.3</a></td>
          </tr>
          <tr>
            <td>C.4</td>
            <td>625–626</td>
            <td>the bits-to-bytes explorer in <a href={href('appC', 'C.4')}>C.4</a>; the KV offset question above</td>
          </tr>
          <tr>
            <td>C.5</td>
            <td>626</td>
            <td>the peer access matrix in <a href={href('appC', 'C.5')}>C.5</a></td>
          </tr>
          <tr>
            <td>C.6, Fig. C.1</td>
            <td>626–627</td>
            <td>the Fig. C.1 stepper and the animation of the two unified memories in <a href={href('appC', 'C.6')}>C.6</a>; the managed path of lab C.1</td>
          </tr>
          <tr>
            <td>C.7</td>
            <td>627–628</td>
            <td>the fit check and the paradigm table in <a href={href('appC', 'C.7')}>C.7</a></td>
          </tr>
          <tr>
            <td>C.8</td>
            <td>628–629</td>
            <td>the coherence explorer, the page-fault animation and the oversubscription explorer in <a href={href('appC', 'C.8')}>C.8</a></td>
          </tr>
          <tr>
            <td>C.9</td>
            <td>629</td>
            <td>the reserve-map-grant stepper and the growing-buffer animation in <a href={href('appC', 'C.9')}>C.9</a></td>
          </tr>
        </tbody>
      </table>

      <h2>Where the PDF looks wrong</h2>
      <p>
        First the slips in the printed page of this new edition, listed so that you are not thrown by them when you read
        the book alongside.
      </p>
      <ul data-list="errata">
        <li>
          Fig. C.1 (p. 627): <code>cudaDeviceSynchronize()</code> has no semicolon, so the right-hand box does not
          compile as printed; the launch configuration is elided as <code>&lt;&lt;&lt;...&gt;&gt;&gt;</code>; the prose
          says the kernel replaces <code>qsort()</code>, while the figure calls it <code>qsort_char</code>; the heading
          has two spaces between "CUDA 6" and "code". The checked copies are in <code>labs/appC/snippets/</code>, and{' '}
          <a href={href('appC', 'C.6')}>section C.6</a> steps through them.
        </li>
        <li>
          p. 624: "We mentioned that the other values of the flag argument are for more advanced usage." We found no
          earlier mention in this edition (a search of the text finds <code>cudaHostAlloc</code> only in Appendix C).
          Chapter 23 (p. 562) allocates pinned memory with <code>cudaMallocHost</code>{' '}
          instead, and the book never relates the two. They are one allocation, with flags and without.
        </li>
        <li>p. 624: "shared, memory, local memory, and global memory" has a stray comma.</li>
        <li>p. 623 calls the appendix "this section".</li>
        <li>
          The appendix says "Unified Virtual Address Space (UVAS)" (p. 625), chapter 2's footnote says "Unified Virtual
          Address (UVA)" (p. 35). Same feature; NVIDIA uses both names.
        </li>
        <li>
          Reference [1] (p. 629) lists the authors in a different order from the published paper, which is Gelado,
          Cabezas, Navarro, Stone, Patel, Hwu.
        </li>
      </ul>
      <Aside>
        <p>These are our readings of one PDF. If your copy differs, trust your copy.</p>
      </Aside>
      <p>
        Then the statements that NVIDIA's own documents, checked on 2026-11-20, correct or leave open. The sections
        teach the corrected form; this is the list in one place. Each item names its entry in{' '}
        <code>notes/today/appC-facts.md</code>, where the vendor's own words are quoted.
      </p>
      <ul data-list="corrections">
        <li>
          Fermi "introduced 2009" (p. 624): the whitepaper is from 2009; NVIDIA says Fermi reached the market in March
          2010 (A2).
        </li>
        <li>
          Kepler "introduced in 2013" (p. 626): the first Kepler GPUs launched in March 2012. Nor was Kepler first:
          Fermi already had 64-bit load and store addressing, a 40-bit unified address space, up to 6 GB of memory and
          unified addressing. Kepler's "at least 40-bit" physical addresses could not be checked (A6).
        </li>
        <li>
          Peer access "before Kepler" needed host-driven copies (p. 626): NVIDIA's own account has unified addressing
          reaching another GPU's memory since CUDA 4, and no source we read names the first architecture with peer
          loads. Peer access must also be enabled, one direction at a time, with at most eight peers per GPU without
          NVSwitch (A8).
        </li>
        <li>
          CUDA 6 and Unified Memory "2013" (p. 626): announced in November 2013, shipped in April 2014 (A9).
        </li>
        <li>
          The reserve-and-map calls "CUDA 11" (p. 629): they arrived in CUDA 10.2, in November 2019, and{' '}
          <code>cuMemMap</code> alone does not make memory usable: <code>cuMemSetAccess</code> must grant access (A15).
        </li>
        <li>
          Mapped memory needs a second pointer from <code>cudaHostGetDevicePointer()</code> (p. 624): true in 2009. Under
          unified addressing <code>cudaHostAlloc</code> and <code>cudaMallocHost</code> memory works in kernels through
          the host's own pointer; only registered and write-combined memory still has a distinct device address (A3).
        </li>
        <li>
          The link has "less than 10%" of device bandwidth (p. 625): true of PCIe on NVIDIA's current figures, not of
          NVLink-C2C systems. NVIDIA no longer states a ratio (A4).
        </li>
        <li>
          "Each physical memory address maps to only one virtual location" (p. 625): registered and write-combined host
          memory are documented exceptions (A5).
        </li>
        <li>
          UVAS "originated from" GMAC (p. 625): no NVIDIA source we read mentions GMAC, so this stays the book's account
          (A5).
        </li>
        <li>
          Left open by the book, settled by NVIDIA: from Pascal on a kernel can follow host-built structures in managed
          memory, but reaches plain <code>malloc</code> memory only with HMM or ATS (A12); and the driver decides between
          migrating and mapping by heuristics, which <code>cudaMemAdvise</code> and <code>cudaMemPrefetchAsync</code>{' '}
          can steer (A13).
        </li>
      </ul>

      <h2>What this appendix left out</h2>
      <ul>
        <li>
          The opening's problem with disks and network cards (p. 623) gets one note on the start page; the book never
          returns to it.
        </li>
        <li>
          Chapter 21's grid cut into 2D slices (p. 623) is named as the book's example of chunking, not reworked here; it
          is on p. 517.
        </li>
        <li>
          Chapter 23's halo exchange (p. 626) appears only in <a href={href('appC', 'C.5')}>section C.5</a>'s step list.
          The multi-GPU stencil, MPI, NCCL and NVSHMEM stay in chapter 23 (pp. 541–576).
        </li>
        <li>
          The large in-core data sets of CAD (p. 629) and the cheaper CUDA libraries (p. 624) are one sentence each in
          the book, and are not expanded.
        </li>
        <li>The GMAC paper, the book's reference [1], is cited, with its published author order, but not summarised.</li>
        <li>
          Fig. C.1's sort kernel is not in the book, and none is written here: lab C.1 uses a gather kernel instead.
        </li>
      </ul>

      <h2>The project so far</h2>
      <p>
        The appendix adds no milestone. Lab C.1's <code>gatherAddKernel</code> follows the project's conventions and can
        serve as <code>nanoinfer</code>'s token-embedding lookup.{' '}
        <span data-c1>
          {devices.length === 0 && !reports.emu
            ? 'No lab results are recorded on this machine yet.'
            : mine.length
              ? `Your own feed functions pass on ${mine.join(' and ')} on this machine.`
              : mineEmu
                ? 'Your own feed functions pass on the emulator; they have not passed on a device on this machine yet.'
                : 'Your own feed functions have not passed on this machine yet.'}
        </span>{' '}
        See the <a href={href('appC', 'labs')}>labs</a>. What the appendix gives the project is a way to decide where each
        tensor lives: weights resident in device memory, sparse tables such as the embedding read in place if memory is
        short, and logits in memory both processors may read.
      </p>
      <p>
        In the course's order, <a href={href('x1')}>chapter X1</a>, sampling kernels, comes next.{' '}
        <a href={href('x3')}>Chapter X3</a>, reading the GPU backends of llama.cpp, MLX and PyTorch, is where you meet
        this appendix's mechanisms in production code: llama.cpp's pool of reserved and mapped device memory and its
        switch to managed memory, its Metal backend wrapping memory-mapped weights without a copy, and PyTorch's pinned
        host memory and expandable segments.
      </p>
    </>
  );
}
