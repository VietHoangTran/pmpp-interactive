import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Beyond, Simplification, Source, Today } from '../../components/notes';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { BinnedBlockScene } from './BinnedBlockScene';
import { PairsExplorer } from './PairsExplorer';

// Section 21.5 (pp. 523–527): cutoff summation, bins, per-block neighbourhood lists, fixed
// bin capacity and the host's overflow pass. Every number on the page is the book's, a vendor
// document's, or computed by src/sim/cutoff.ts (pinned in cutoff.test.ts).

export const questionsS215: Record<string, Question> = {
  'ch21/21.5/q-atom-centric': {
    kind: 'choice',
    id: 'ch21/21.5/q-atom-centric',
    prompt: (
      <>
        The sequential cutoff loop visits, for each atom, only the grid points within the cutoff: cheap and simple. Why
        not give each GPU thread one atom and do the same?
      </>
    ),
    options: [
      {
        label: 'Nearby atoms\' spheres overlap, so several threads add to the same points: a scatter that needs atomics, the problem of section 21.2.',
        correct: true,
      },
      { label: 'Atoms cannot be indexed by their coordinates, so a thread cannot find which grid points lie near its atom.', why: 'It is the grid points that are indexed by coordinate, which is exactly why the sequential loop is easy.' },
      { label: 'There are too few atoms in a realistic system to give every thread of a GPU something to do.', why: 'Realistic systems have hundreds of thousands of atoms; parallelism is not the problem.' },
      { label: 'A GPU thread cannot loop over the points of a sphere, because the trip count differs between threads.', why: 'It can. Different trip counts cost some divergence, but they are not the reason.' },
    ],
    explain: (
      <p>
        The sequential loop is atom-centric, a scatter, and a scatter's outputs collide. Bins are what make the
        grid-centric gather possible here: they tell a thread for a grid point which atoms can matter, without scanning
        them all.
      </p>
    ),
    source: 'ch. 21 §21.5, p. 524; the scale of real systems from ch. 22, p. 536',
  },
  'ch21/21.5/q-fully': {
    kind: 'choice',
    id: 'ch21/21.5/q-fully',
    prompt: (
      <>
        Super circle of radius 2b on square bins of side b, centred on the middle bin. Of the 21 bins in the list, how many
        lie entirely inside the circle?
      </>
    ),
    fixedOrder: true,
    options: [
      { label: '5: the centre bin and its four edge neighbours.', correct: true },
      { label: '9: the whole 3 × 3 block around the centre, as Fig. 21.14 says.', why: 'The diagonal neighbours reach 2.12 b from the centre, past a radius of 2 b.' },
      { label: '13: the 3 × 3 block and the four bins two steps out along the axes.', why: 'The bins two steps out reach 2.55 b; they are only partly inside.' },
      { label: '21: every bin in the list is inside the circle.', why: 'In the list means touched, not covered.' },
    ],
    explain: (
      <p>
        A bin is fully inside when its farthest corner is: √(1.5² + 0.5²) ≈ 1.58 for an edge neighbour, √(1.5² + 1.5²) ≈
        2.12 for a diagonal one. So at R = 2b only five bins are fully inside, although 21 are touched. The figure's "9 fully
        covered + 12 partially covered" holds only at R ≈ 2.12 b, which is also the radius at which the four corner bins
        start to touch. The total of 21 is the number that matters for the kernel. Set the explorer to R = 2 bin widths and
        read its counts.
      </p>
    ),
    source: 'ch. 21 Fig. 21.14, p. 526; notes/today/ch21-facts.md E4',
  },
  'ch21/21.5/q-ex5': {
    kind: 'choice',
    id: 'ch21/21.5/q-ex5',
    prompt: <>Exercise 5. How does control divergence arise when the threads of a block process a bin from their neighbourhood list?</>,
    options: [
      {
        label: 'The list is per block and conservative, so a bin is only partly within the cutoff of some points; for an atom near the edge some lanes accept it and some reject it.',
        correct: true,
      },
      { label: 'Different threads of the block load different bins of the list, so they run different numbers of loop iterations.', why: 'All threads of a block walk the same list and load each bin together.' },
      { label: 'The barrier after the cooperative load makes some lanes wait for others, which splits the warp in two.', why: 'Waiting at a barrier is synchronisation, not divergence: everyone takes the same path to it.' },
      { label: 'Dummy atoms take a different code path from real atoms, so a bin that has some of each diverges.', why: 'Dummy atoms take the same path as real ones; their charge of zero adds nothing.' },
    ],
    explain: (
      <p>
        Every thread scans the same staged atoms but tests each against the cutoff around its own point, and the list was
        built to cover the block's farthest point. In the book's 3D case only about a quarter of the tests pass, so most
        bins split most warps. The explorer's mask panel shows one staged atom across the block's lanes.
      </p>
    ),
    source: 'ch. 21 Exercise 5, p. 528; pp. 524–526; notes/today/ch21-facts.md E4',
  },
  'ch21/21.5/q-global': {
    kind: 'choice',
    id: 'ch21/21.5/q-global',
    prompt: <>Why do the atoms move from constant memory to global memory, staged through shared memory, in the binned kernel?</>,
    options: [
      {
        label: 'Each block needs its own neighbourhood of bins, and constant memory cannot hold what all active blocks need at once; staging a bin once per block keeps the reuse.',
        correct: true,
      },
      { label: 'Constant memory can only hold integers and pointers, so the float coordinates have to go somewhere else.', why: 'The DCS kernels keep float atoms in constant memory.' },
      { label: 'Global memory is faster than constant memory once the atoms have been sorted into bins.', why: 'It is not; the change is about capacity and about which atoms each block needs.' },
      { label: 'Shared memory is shared by all the blocks of a grid, so one copy of each bin serves everyone.', why: 'Shared memory is per block: each block stages its own copy.' },
    ],
    explain: (
      <p>
        In direct summation every thread reads every atom of a chunk, so one small chunk in constant memory serves all
        blocks. With bins, different blocks need different atoms, and all the active blocks together need far more than
        constant memory holds. So the atoms go to global memory, and each block copies a bin into shared memory once and lets
        all its threads read it there.
      </p>
    ),
    source: 'ch. 21 §21.5, pp. 526–527',
  },
  'ch21/21.5/q-capacity': {
    kind: 'choice',
    id: 'ch21/21.5/q-capacity',
    prompt: (
      <>
        Your bin capacity leaves 2 % of the atoms on the overflow list. A colleague proposes raising it until nothing
        overflows. What does that cost?
      </>
    ),
    options: [
      {
        label: 'Every bin is padded to the fullest bin\'s count, so dummy atoms take memory, transfers and every block\'s scan time, to save a host pass over 2 % of the atoms.',
        correct: true,
      },
      { label: 'Nothing: dummy atoms have a charge of zero, so a larger capacity adds no work to any block.', why: 'They add nothing to the sum, but each one is still loaded, staged and tested by every thread.' },
      { label: 'The overflow pass becomes inaccurate, because the host then sums the far field the kernel left out.', why: 'The host pass is exact: a sequential cutoff sum over the overflow atoms.' },
      { label: 'The kernel needs atomics once bins are large, because several threads then share one bin.', why: 'Ownership of the grid points does not change; each thread still writes only its own point.' },
    ],
    explain: (
      <p>
        Capacity is a trade the book describes: pick one that fits the vast majority of bins, far below the largest, and
        send the rest down a slower path. A small overflow costs a short host pass that can even run while the device works
        on the next sub-volume. Raising the capacity to the worst bin pays for that one bin in every bin. Drag the capacity
        in the explorer and watch the two counts trade places.
      </p>
    ),
    source: 'ch. 21 §21.5, p. 527',
  },
  'ch21/21.5/q-vllm-pad': {
    kind: 'numeric',
    id: 'ch21/21.5/q-vllm-pad',
    prompt: (
      <>
        vLLM's <code>moe_align_block_size</code> sizes its padded buffer as{' '}
        <code>topk_ids.numel() + num_experts * (block_size - 1)</code>. For 512 tokens routed to the top 2 of 8 experts
        with <code>block_size</code> 64, how many slots is that?
      </>
    ),
    answer: 1528,
    unit: 'slots',
    explain: (
      <p>
        512 × 2 = 1,024 routed entries, plus 8 × 63 = 504: 1,528. Each expert's list is padded up to a multiple of{' '}
        <code>block_size</code>, which adds at most <code>block_size − 1</code> dummy entries per expert. Those dummies are
        the book's dummy atoms with token ids for charges, and they cost memory the same way.
      </p>
    ),
    source: 'ours, not the book: notes/today/ch21-facts.md D6; p. 527',
  },
};

/** Threadgroup memory as the reader's Mac reports it, if it has been recorded. */
function ThreadgroupMemory() {
  const { reports } = useReports();
  const v = Number(reports.metal?.device?.maxThreadgroupMemoryLength);
  return Number.isFinite(v) && v > 0 ? (
    <>your Mac reports {v.toLocaleString('en-US')} bytes per threadgroup (<code>maxThreadgroupMemoryLength</code>)</>
  ) : (
    <>run <code>python3 labs/pmpp.py doctor</code> to see what your Mac allows per threadgroup</>
  );
}

export function S215() {
  return (
    <>
      <p>
        Direct summation is exact and parallel, and it does not scale. For a larger molecular system the number of atoms
        grows in proportion to the volume, and so does the number of grid points. The work, atoms times points, grows with
        the square of the volume.
      </p>
      <Source section="21.5" pages="523, 528" />
      <Predict
        id="ch21/21.5/p-volume"
        prompt="Double the side of the simulated box, keeping the atom density and the grid spacing. By what factor does the work of direct summation grow?"
        unit="times"
      >
        {(v) => (
          <p>
            <b>64.</b> Eight times the atoms times eight times the points. With a cutoff, each point's neighbourhood keeps its
            size, so the work grows eight-fold, in line with the volume.{' '}
            {v === 64 ? 'That is what you had.' : v === 8 ? 'Eight is the cutoff algorithm. Direct summation pays for the new atoms at every new point.' : v === 4 ? 'Four is one side squared; the box is three-dimensional.' : ''}
          </p>
        )}
      </Predict>

      <h2>Choosing a different algorithm</h2>
      <p>
        Algorithms for one problem differ in the number of steps, the parallelism they expose, their numerical stability and
        the memory bandwidth they consume, and usually none wins on all four. Normally an alternative must reach the same
        result. If a slight change in the result is acceptable, more aggressive options open up. <b>Cutoff summation</b> is
        one: a contribution falls off with distance, so only the atoms within a fixed radius of a grid point are summed
        directly, and the many distant, small contributions are handled together by a separate, cheaper method. The work
        then grows linearly with the volume.
      </p>
      <Source section="21.5" pages="523–524" figs="Fig. 21.11" />
      <Beyond>
        <p>
          The book leaves that far-field method undescribed. VMD's own GPU page names multilevel summation, whose work grows
          with atoms plus grid points rather than their product (ch21-facts A1). The explorer below simply drops the far
          field and shows how much that is.
        </p>
      </Beyond>
      <p>
        Sequentially, the cutoff is easy: take one atom at a time and visit only the grid points within the radius of it.
        Grid points live in an array indexed by coordinate, so finding them is a little arithmetic. But that loop is
        atom-centric. It is the scatter of section 21.2 again, and it does not parallelise well. The parallel kernel has to
        be grid-centric: one thread per grid point, finding the atoms that matter to it.
      </p>
      <Source section="21.5" pages="524" />
      <Ask q={questionsS215['ch21/21.5/q-atom-centric']} />

      <h2>Bins and neighbourhoods</h2>
      <p>
        First, sort the atoms into <b>bins</b>: boxes in space, each holding the atoms that fall inside it. A grid point's
        neighbourhood is the set of bins that can hold atoms within its cutoff (Fig. 21.12). Its thread scans every atom of
        those bins and tests each against the cutoff around its own point, because some of them lie in a bin that only
        partly overlaps the cutoff.
      </p>
      <Source section="21.5" pages="524" figs="Fig. 21.12" />
      <p>
        Working out the exact neighbourhood of every point is an expensive geometric problem, so the book defines one
        neighbourhood per block, prepared once on the host before the launch, and simplifies further: the points a block
        computes are exactly one bin. Draw a "super circle" around the block's bin with radius equal to the cutoff plus half
        the bin's diagonal; it contains the cutoff circle of every point in the bin. Every bin the super circle covers fully
        or partly goes on the list, stored as offsets relative to the block's own bin, so one list serves every block
        (Fig. 21.14).
      </p>
      <Source section="21.5" pages="525–526" figs="Figs. 21.13, 21.14" />
      <Predict
        id="ch21/21.5/p-bins"
        prompt="Square bins of side b, super-circle radius 2b, centred on the middle bin. How many bins does the 2D neighbourhood contain?"
        unit="bins"
      >
        {(v) => (
          <p>
            <b>21.</b> The 3 × 3 block, the four bins two steps out along the axes, and the eight beside those; the four
            corner bins of the 5 × 5 square stay out. The count is 21 for any radius between about 1.58 b and 2.12 b.{' '}
            {v === 21 ? 'That is what you had.' : v === 25 ? 'Twenty-five takes the whole 5 × 5 square: the corners only start to touch at 2.12 b.' : v === 9 ? 'Nine is the 3 × 3 block alone: the circle reaches past it along the axes.' : ''}
          </p>
        )}
      </Predict>
      <Simplification>
        <p>
          2D: points and atoms in one plane (the atoms sit a fixed height above it only so that no distance is zero), uniform
          random atoms, the block's points exactly one bin. The far field is dropped, not computed. Neighbour bins outside the
          box are skipped. Tests and pairs are counted, not timed.
        </p>
      </Simplification>
      <PairsExplorer
        title="Cutoff and bins"
        task="Press R = 2 bin widths and read the counts. Then click other bins and points, pick staged atoms, double the box, and drag the capacity."
        views={['cutoff']}
        foot="Model: src/sim/cutoff.ts. Atoms (amber) live in global memory; the shaded bins are the ones this block stages in shared memory."
      />
      <p>
        Two of the figures need care. Fig. 21.14 says 9 bins are fully covered and 12 partly, 21 in all. The 21 is right
        over a whole range of radii, but in that range only 5 bins are fully inside; 9 holds at a single radius, about 2.12
        bin widths. And Fig. 21.13 labels its radius "cutoff distance + length of bin diagonal", the whole diagonal, where
        the text and Fig. 21.14 say half; the explorer offers both. The figure itself is a sketch, not drawn to scale. Look
        at the far-field row as well: with charges of both signs at random positions, what the cutoff drops at a point is
        not small, which is why the book hands the far field to a separate method instead of ignoring it.
      </p>
      <Source section="21.5" pages="525–526" figs="Figs. 21.13, 21.14" />
      <Aside>
        <p>
          The book's worked case (p. 525): 0.5 Å spacing and 8 × 8 × 8 blocks give 4 Å bins, with a 12 Å cutoff. Computed
          here, the 3D list has 461 bins (147 fully inside), 667 with Fig. 21.13's radius, and about 25 % of the tested atoms
          pass (ch21-facts E4). See the 3D readout in the explorer.
        </p>
      </Aside>
      <Ask q={questionsS215['ch21/21.5/q-fully']} label="Check the figure" />

      <h2>The kernel</h2>
      <p>
        The list is passed to the kernel, most likely in constant memory, and every block walks it. For each entry its
        threads add the offset to the block's own bin, load that bin's atoms into shared memory together, and then each
        thread tests each staged atom against the cutoff around its own point and adds the ones that pass. The test is where
        warps diverge: some lanes accept an atom that others reject.
      </p>
      <Source section="21.5" pages="524, 526" />
      <Aside>
        <p>
          The book's outline goes from "load together" straight to "each thread checks". A shared copy needs the two
          barriers of chapter 5's tiled kernel (<a href={href('ch05', '5.4')}>section 5.4</a>): one after the load, one
          before the next bin overwrites it. The animation shows them; they are ours, not the page's.
        </p>
      </Aside>
      <BinnedBlockScene />
      <Ask q={questionsS215['ch21/21.5/q-ex5']} label="Book exercise" />
      <p>
        Notice where the atoms live now. In direct summation every thread read every atom of a chunk, so one small chunk in
        constant memory served the whole grid. Here each thread examines far fewer atoms, and different blocks need different
        bins, so constant memory can no longer hold what all the active blocks need. The atoms go to global memory, and each
        block loads a bin into shared memory once and lets all its threads read it there.
      </p>
      <Source section="21.5" pages="526–527" />
      <Ask q={questionsS215['ch21/21.5/q-global']} />
      <Today checked="2026-11-20 (notes/today/ch05-facts.md A1, A7, B4, C1, C2; ch21-facts C1)">
        <p>
          CUDA and HIP: <code>__shared__</code> and <code>__syncthreads()</code>, same spelling. Metal:{' '}
          <code>threadgroup</code> memory and <code>threadgroup_barrier(mem_flags::mem_threadgroup)</code>; <ThreadgroupMemory />.
          For a variable amount of data Apple advises the <code>device</code> address space over <code>constant</code>, so a
          Metal binned kernel takes its atoms as <code>const device float4*</code>.
        </p>
      </Today>

      <h2>Bins of one size, and an overflow list</h2>
      <p>
        Bins naturally hold different numbers of atoms, some many, some none. To keep each bin's load aligned and coalesced,
        the book gives every bin the same capacity and pads the empty slots with dummy atoms of charge zero. That costs twice:
        dummy atoms take memory and transfer bandwidth, and they lengthen the scan of every block whose bins are sparse. So
        the capacity is chosen to fit the vast majority of bins, far below the largest; an atom whose bin is already full
        goes on an <b>overflow list</b>. After the kernel the grid is copied back and the host runs a sequential cutoff sum
        over the overflow atoms. With under 3 % of atoms on the list, that typically takes less time than the device run, and
        if each launch handles a sub-volume, the host can process one sub-volume's overflow while the device works on the
        next.
      </p>
      <Source section="21.5" pages="527" />
      <Today checked="2026-11-20 (notes/today/ch02-facts.md A5, B3, C4)">
        <p>
          That overlap rests on asynchronous launches: a CUDA or HIP kernel launch returns to the host before the kernel
          finishes, and Metal's <code>commit</code> returns before the command buffer completes.
        </p>
      </Today>
      <Ask q={questionsS215['ch21/21.5/q-capacity']} />

      <h2>The same idea in an LLM</h2>
      <p>
        Routing tokens to the experts of a mixture-of-experts layer is binning with experts as bins, and the libraries you use
        make the book's choices in the open: count, pad to a fixed size, and gather to combine.
      </p>
      <Beyond>
        <p>
          llama.cpp's Metal <code>kernel_mul_mm_id_map0</code> counts each expert's tokens, prefix-sums the counts, emits
          32-row tiles with a valid-row count, and pads each expert's slice with one slot so every lane's store is
          unconditional (the comment that divergent stores are slow is the authors', not Apple's). Its CUDA and HIP{' '}
          <code>mm_ids_helper</code> gives each expert one block. vLLM's <code>moe_align_block_size</code> pads each
          expert's list to a multiple of <code>block_size</code> with dummy token ids. mlx-lm sorts tokens by expert at 64
          or more indices (its own, unexplained threshold) and calls <code>gather_mm(sorted_indices=True)</code>. The
          combine is a gather (llama.cpp's <code>moe_weighted_reduction</code>). ch21-facts D4 to D6.
        </p>
      </Beyond>
      <Ask q={questionsS215['ch21/21.5/q-vllm-pad']} label="Work it out" />
      <p>
        Counting the tokens per expert, the first step of all of these, is the histogram of chapter 9; its summary page has a
        question on exactly that (<a href={href('ch09', 'wrap')}>chapter 9, summary</a>).
      </p>

      <h2>Where this goes next</h2>
      <p>
        Chapter 22 steps back from single kernels. <a href={href('ch22', '22.2')}>Section 22.2</a> sorts every pattern in the
        book by the decision this chapter keeps making, output-centric gather against input-centric scatter, and{' '}
        <a href={href('ch22', '22.1')}>section 22.1</a> returns to cutoff binning as an example of trading a little accuracy
        for a different complexity.
      </p>
    </>
  );
}
