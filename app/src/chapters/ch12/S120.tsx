import { Ask, type Question } from '../../components/Ask';
import { Aside, Beyond, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';

// Before you start. Prerequisite questions from the notes of chapters 4, 6, 9, 10 and 11;
// vendor facts from notes/today/ch12-facts.md (D3, D4, D6) and ch05-facts.md (B5).
export const questionsS120: Record<string, Question> = {
  'ch12/start/q-lane64': {
    kind: 'numeric',
    id: 'ch12/start/q-lane64',
    prompt: (
      <>
        A one-dimensional block runs on an AMD GPU whose warps (AMD says wavefronts) are 64 lanes wide. Thread 100 of
        the block sits in which lane?
      </>
    ),
    answer: 36,
    unit: 'lane',
    explain: (
      <>
        <p>
          Warps are cut from consecutive thread indices, so 100 = 64 + 36: warp 1, lane 36. On a 32-wide warp the same
          thread is lane 4 of warp 3 (100 = 96 + 4).
        </p>
        <p>
          In a warp vote, lane 36 owns bit 36 of the mask. An <code>unsigned int</code> has no bit 36, and{' '}
          <code>1 &lt;&lt; 36</code> on a 32-bit integer is undefined. That is why this chapter's warp code gives the
          mask its own type, 64 bits on HIP, and counts it with the 64-bit intrinsics. See{' '}
          <a href={href('ch04', '4.4')}>section 4.4</a> for how a block is cut into warps.
        </p>
      </>
    ),
    source: 'ch. 4 §4.4, p. 75; ch. 10 §10.7, p. 239 (laneIdx); AMD widths: notes/today/ch05-facts.md B5; 64-bit masks: ch12-facts B1',
  },
  'ch12/start/q-onecounter': {
    kind: 'choice',
    id: 'ch12/start/q-onecounter',
    prompt: (
      <>
        100,000 threads each atomically add 1. In version A they all add to one counter; in version B each thread adds
        to a counter of its own. Which version takes longer, and why?
      </>
    ),
    options: [
      {
        label: 'Neither: an atomic is a single read-modify-write instruction, so 100,000 of them take the same time whatever addresses they name.',
        why: 'The hardware still allows only one atomic on a given address at a time. The others queue behind it.',
      },
      {
        label: 'A. Atomics on one address are serialised, each taking about a load plus a store latency; on different addresses they overlap.',
        correct: true,
      },
      {
        label: 'B: 100,000 counters means 100,000 separate locations to fetch and write back, where A keeps reusing one.',
        why: 'Memory traffic is not what limits A. The queue on one address is.',
      },
    ],
    explain: (
      <p>
        The rate of atomics on one location is about one per (load latency + store latency). Section 12.2's filter is
        version A: every kept key bumps the same output counter, and the rest of the unstable half of this chapter
        reduces how many times that counter is touched. Back to <a href={href('ch09', '9.3')}>section 9.3</a>.
      </p>
    ),
    source: 'ch. 9 §9.3, pp. 209–210',
  },
  'ch12/start/q-exclusive': {
    kind: 'numeric',
    id: 'ch12/start/q-exclusive',
    prompt: (
      <>
        Take the exclusive scan (with +) of <code>[0, 1, 1, 0, 1, 0, 0, 1]</code>. What is the value at index 7?
      </>
    ),
    answer: 3,
    explain: (
      <p>
        Exclusive means "the sum of everything before me": 0 + 1 + 1 + 0 + 1 + 0 + 0 = 3. When the inputs are 0/1 keep
        flags, that sum is the number of kept items before this one, which is exactly the slot this item takes in a
        compacted list that keeps the original order. The stable filter of section 12.5 is built on it. Back to{' '}
        <a href={href('ch11', '11.1')}>section 11.1</a>.
      </p>
    ),
    source: 'ch. 11 §11.1, p. 253',
  },
  'ch12/start/q-privatise': {
    kind: 'choice',
    id: 'ch12/start/q-privatise',
    prompt: <>What does giving each block a private copy of a shared output buy, and what does it cost?</>,
    options: [
      {
        label: 'It removes the need for atomics: each block owns its copy, so its threads can update it with plain loads and stores.',
        why: 'The threads of one block still share the private copy, so their updates still need atomics: cheaper ones, in shared memory.',
      },
      {
        label: 'A speedup on every input, since shared-memory updates are cheaper than global ones and the final merge is one pass.',
        why: 'Initialising and merging the copies costs something per block. When contention was low to begin with, that cost can outweigh the gain.',
      },
      {
        label: 'Far fewer updates to the public copy, paid for with on-chip memory per block and a merge step at the end.',
        correct: true,
      },
    ],
    explain: (
      <p>
        Section 12.4 gives each block a private output list and a private counter in shared memory. One thread per
        block then reserves space in the public list with a single atomic. Back to{' '}
        <a href={href('ch09', '9.4')}>section 9.4</a>.
      </p>
    ),
    source: 'ch. 9 §9.4, pp. 211–214',
  },
  'ch12/start/q-coalesce': {
    kind: 'choice',
    id: 'ch12/start/q-coalesce',
    prompt: (
      <>
        The 32 lanes of a warp store 4-byte values. In case A lane <i>i</i> writes <code>out[base + i]</code>; in case
        B it writes <code>out[base + 32*i]</code>. Which case is served with fewer memory transactions, and why?
      </>
    ),
    options: [
      {
        label: 'A: the 32 addresses are adjacent and fall into a few aligned segments. B puts every lane in a different segment.',
        correct: true,
      },
      {
        label: 'B: stores 128 bytes apart fall in different memory banks, so all 32 lanes are served at once.',
        why: 'Global memory is served in segments of adjacent bytes. B uses 4 bytes of each segment it touches and pays for all of them.',
      },
      {
        label: 'Neither: each lane stores 4 bytes either way, so both cases move the same 128 bytes in the same number of transactions.',
        why: 'The bytes stored are the same. The number of segments touched is not.',
      },
    ],
    explain: (
      <p>
        NVIDIA serves global memory in 32-byte transactions: case A needs four of them if <code>base</code> is aligned,
        case B needs 32. Where kept keys land is therefore a performance question in this chapter: sections 12.3, 12.4
        and 12.6 all arrange for neighbouring threads to write neighbouring slots. Back to{' '}
        <a href={href('ch06', '6.1')}>section 6.1</a>.
      </p>
    ),
    source: 'ch. 6 §6.1, pp. 124–132; 32-byte transactions: notes/today/ch06-facts.md A2',
  },
  'ch12/start/q-shflmask': {
    kind: 'choice',
    id: 'ch12/start/q-shflmask',
    prompt: (
      <>
        Inside a branch that only some lanes of a warp take, a CUDA kernel calls{' '}
        <code>__shfl_sync(0xffffffff, v, src)</code>. What is wrong?
      </>
    ),
    options: [
      {
        label: 'Nothing: the mask only hints at which lanes will take part, and the hardware skips any lane that is absent.',
        why: 'It is a contract. Every lane named in it must make the call.',
      },
      {
        label: 'Only that the lanes outside the branch receive 0, which is harmless since they never read the result.',
        why: 'No result is defined for this call at all. The programming guide classes it as invalid.',
      },
      {
        label:
          'The mask names lanes that never make the call, and CUDA requires every lane in the mask to make it with the same mask: the call is invalid and may hang.',
        correct: true,
      },
    ],
    explain: (
      <p>
        The mask says which lanes take part in the shuffle. In section 12.3 the book's kernel shuffles inside a branch,
        so it passes the mask of the lanes that are actually there, not <code>0xffffffff</code>. Back to{' '}
        <a href={href('ch10', '10.7')}>section 10.7</a>.
      </p>
    ),
    source: 'ch. 10 §10.7, p. 238; notes/today/ch12-facts.md A4 (CUDA programming guide section 5.4.6.6)',
  },
};

const FAMILIES: { family: string; secs: { id: string; label: string }[]; slot: string; work: string }[] = [
  {
    family: 'Unstable',
    secs: [
      { id: '12.2', label: '12.2' },
      { id: '12.3', label: '12.3' },
      { id: '12.4', label: '12.4' },
    ],
    slot: 'takes the next free slot from one shared counter, with an atomic',
    work: 'touch the counter less often: once per kept key, then once per warp, then once per block',
  },
  {
    family: 'Stable',
    secs: [
      { id: '12.5', label: '12.5' },
      { id: '12.6', label: '12.6' },
      { id: '12.7', label: '12.7' },
    ],
    slot: 'counts the kept keys before it: an exclusive scan of the keep flags',
    work: 'the scan inside the filter kernel, then staging through shared memory for coalesced writes, then the whole thing in place',
  },
  {
    family: 'Relatives',
    secs: [{ id: '12.8', label: '12.8' }],
    slot: 'from its index alone, or by a scan when its value decides',
    work: 'removing duplicates, removing or adding rows and columns',
  },
];

const LABS: { num: string; title: string; where: string }[] = [
  { num: '12.1', title: 'The unstable filter three ways', where: 'emu, race, Metal, CUDA, HIP' },
  { num: '12.2', title: 'Filter hunt', where: 'emu, race, Metal, CUDA, HIP' },
  { num: '12.3', title: 'A stable filter by scan', where: 'emu, race, Metal, CUDA, HIP' },
  { num: '12.4', title: 'Candidate compaction (milestone M5 piece)', where: 'emu, race, Metal, CUDA, HIP' },
  { num: '12.5', title: 'In-place stable filter in a single pass (optional)', where: 'CUDA family only: emu, race, CUDA, HIP' },
];

export function S120() {
  const q = questionsS120;
  return (
    <>
      <p>
        This chapter packs the items of a list that pass a test (the book calls them <b>keys</b>) into a gapless list
        of their own, on the GPU;{' '}
        <a href={href('ch12', '12.1')}>section 12.1</a> defines filtering properly.
      </p>
      <Source pages="289" />
      <p>
        The chapter's subtitle is "And warp voting". The new technique is letting the 32 threads of a warp (or however
        many your hardware has) vote with a bit mask, so that one thread can act for all of them. The other tools are
        ones you already have: atomics and privatisation from chapter 9, and the scan from chapter 11.
      </p>

      <h2>Where you have already met it</h2>
      <p>
        A sampler meets a filter right after its threshold. Top-k, top-p and min-p each decide which tokens of a
        vocabulary row stay in the running, and the survivors, packed together with their token ids, are a filter's
        output. The course builds that step on the GPU in four pieces.
      </p>
      <Beyond>
        <p>
          The book does not mention samplers here: this pipeline is ours. The min-p threshold is llama.cpp's host
          one, which tests logits and needs no softmax (ch12-facts D3).
        </p>
      </Beyond>
      <table className="data" data-part="pipeline">
        <thead>
          <tr>
            <th>Step</th>
            <th>What it computes</th>
            <th>Where you build it</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Row maximum</td>
            <td>the largest logit of each row</td>
            <td>
              <a href={href('ch10', 'labs')}>lab 10.3</a> (milestone M4)
            </td>
          </tr>
          <tr>
            <td>Threshold</td>
            <td>
              max + log p, so that a token survives when its probability is at least p times the top one
            </td>
            <td>a comparison, not a kernel of its own</td>
          </tr>
          <tr>
            <td>Compaction</td>
            <td>the survivors of each row, with their token ids, and how many there are</td>
            <td>
              <a href={href('ch12', 'labs')}>lab 12.4</a> (a piece of milestone M5)
            </td>
          </tr>
          <tr>
            <td>Sampling</td>
            <td>one token drawn from the survivors</td>
            <td>
              module X1, <a href={href('x1', 'X1.8')}>section X1.8</a>
            </td>
          </tr>
        </tbody>
      </table>
      <p>
        You may wonder why a framework does not do the compaction for you. The answer is about shapes: the number of
        survivors depends on the data, and a framework that hands back a tensor must know its shape first. PyTorch
        pays for that with a wait.
      </p>
      <Today checked="2026-11-20 (notes/today/ch12-facts.md, D4, D5)">
        <p>
          <code>torch.nonzero</code> copies the count to the host and waits for it, on CUDA and on MPS.{' '}
          <code>torch.nonzero_static</code> takes the output size up front and does not wait.
        </p>
      </Today>
      <p>
        Others avoid compaction altogether and keep every token, with the losers' logits set to minus infinity: a
        graph whose shapes are fixed has no way to say "as long as the number of survivors". A kernel has no such
        problem. It writes the survivors and a count to device memory, and the next kernel reads the count there,
        with no trip to the host. That is what this chapter teaches you to write.
      </p>
      <Today checked="2026-11-20 (notes/today/ch12-facts.md, D3, D6)">
        <p>
          MLX has no <code>nonzero</code> and "limited support" for output shapes that depend on the data.
          llama.cpp's experimental GPU samplers and mlx-lm mask instead of compacting.
        </p>
      </Today>

      <h2>The chapter in two families</h2>
      <p>
        Every filter kernel has to answer one question for each kept key: which output slot is mine? There are two
        kinds of answer, and they split the chapter in two. An <b>unstable</b> filter lets the kept keys come out in any
        order; a <b>stable</b> one keeps their input order, which costs more and matters when the input was sorted.
      </p>
      <Source section="12.1" pages="289–290, 297" figs="Fig. 12.1" />
      <table className="data" data-part="families">
        <thead>
          <tr>
            <th>Family</th>
            <th>Sections</th>
            <th>How a thread finds its slot</th>
            <th>What the sections improve</th>
          </tr>
        </thead>
        <tbody>
          {FAMILIES.map((f) => (
            <tr key={f.family}>
              <td>{f.family}</td>
              <td className="nowrap">
                {f.secs.map((s, i) => (
                  <span key={s.id}>
                    {i ? ', ' : ''}
                    <a href={href('ch12', s.id)}>{s.label}</a>
                  </span>
                ))}
              </td>
              <td>{f.slot}</td>
              <td>{f.work}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p>
        The unstable half is a study in contention on one counter, and the remedy is the one chapter 9 used for
        histograms, plus the warp vote. The stable half is a study in scan, and it ends with a filter that works in the
        very array it reads, which turns out to be a question about the order of events between blocks.
      </p>
      <Source pages="290–301" />

      <h2>Before you start</h2>
      <p>
        Six questions on what this chapter assumes from chapters 4, 6, 9, 10 and 11. Each explanation links back to
        the section it comes from, and each question returns in <a href={href('review')}>review</a>.
      </p>
      <Ask q={q['ch12/start/q-lane64']} label="From chapter 4" />
      <Ask q={q['ch12/start/q-onecounter']} label="From chapter 9" />
      <Ask q={q['ch12/start/q-exclusive']} label="From chapter 11" />
      <Ask q={q['ch12/start/q-privatise']} label="From chapter 9" />
      <Ask q={q['ch12/start/q-coalesce']} label="From chapter 6" />
      <Ask q={q['ch12/start/q-shflmask']} label="From chapter 10" />

      <h2>What runs where</h2>
      <p>
        Four labs run on every backend the course has; the fifth is optional and needs something only CUDA documents.
        The <a href={href('ch12', 'labs')}>labs page</a> shows what has actually been checked on each.
      </p>
      <Aside>
        <p>
          <b>emu</b> runs one thread at a time and catches wrong indices and missing barriers; <b>race</b> runs threads
          concurrently and reports unsynchronised accesses. Neither models timing.
        </p>
      </Aside>
      <table className="data" data-part="labs">
        <thead>
          <tr>
            <th>Lab</th>
            <th>Title</th>
            <th>Backends</th>
          </tr>
        </thead>
        <tbody>
          {LABS.map((l) => (
            <tr key={l.num}>
              <td className="nowrap">{l.num}</td>
              <td>{l.title}</td>
              <td>{l.where}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p>
        Lab 12.5 has no Metal version because it lets one block wait for another, and Apple documents no guarantee that
        a waiting threadgroup's predecessor will ever run. Section <a href={href('ch12', '12.7')}>12.7</a> explains
        why, and what to do on Metal instead.
      </p>
    </>
  );
}
