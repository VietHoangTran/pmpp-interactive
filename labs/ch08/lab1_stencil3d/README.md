# Lab 8.1: a 3D stencil, one thread per point, then marching in z with registers

Book: sections 8.2 to 8.6 (pp. 187-198), Figs. 8.6 and 8.12, exercise 1b (p. 198). The
two comparison kernels you are given are Figs. 8.8 and 8.10.

A 7-point stencil sweep over an N x N x N grid of floats, row-major with x fastest.
Every interior point of `out` becomes a weighted sum of the same point of `in` and its
six neighbours along x, y and z. Boundary points keep their input values: the driver
copies `in` into `out` before every launch, and your kernels write interior points only.
The weights and which neighbour each one multiplies are in `stencil.h`.

Write two kernels, in CUDA and in Metal:

1. **Basic** (Fig. 8.6): `stencilBasicKernel` and `launchStencilBasic`. One thread per
   grid point, every operand read from global memory, 8 x 8 x 8 blocks (exercise 1b).
2. **Register-tiled** (Fig. 8.12): `stencilRegTiledKernel` and `launchStencilRegTiled`.
   A block of TILE_WIDTH x TILE_WIDTH threads, one x-y plane of the input tile, marches
   through a slab of `depth` output planes in z, starting at plane `blockIdx.z * depth`.
   Each thread keeps its own previous, current and next values in registers; the
   current plane is also mirrored in one shared array, because it is the only plane other
   threads read. The grid is `ceil(N / (TILE_WIDTH - 2))` blocks in x and y and
   `ceil(N / depth)` in z.

- Edit `work/kernel.cu` (both kernels and both launches), `work/kernel.metal`
  (`stencil_basic`, `stencil_regtile`) and `work/launch.mm` (`encodeStencilBasic`,
  `encodeStencilRegTiled`). Do not edit `stencil.h`, `main.cpp` or `../shared/`.
- `TILE_WIDTH` is defined by the build: 32 by default, and 16 and 8 are also built for the
  benchmark. Metal compiles 8, 16 and 32 at run time; 32 is tested only where the
  pipeline's `maxTotalThreadsPerThreadgroup` is at least 1024, and otherwise the driver
  prints that limit and says it skipped T = 32. Do not hard-code 32.
- Test: `python3 labs/pmpp.py run 8.1`. Start with `--backend emu`. The emulator needs a
  few minutes for this lab (at N = 63 a launch has several hundred thousand threads, run
  one at a time, under three orders). It tests the small grids first, so fix what N = 3,
  4 and 5 show before you wait for N = 63.
- Race check: `python3 labs/pmpp.py run 8.1 --backend race` runs the threads of a block
  concurrently under ThreadSanitizer (N = 3, 4, 5 and 34, T = 32). Your kernel shares
  `inCurr_s` and guards it with two barriers; if one is missing or misplaced, this names
  the two source lines whose accesses are not ordered, even in a run whose numbers happen
  to come out right. A clean run is evidence, not proof.
- Measure: `python3 labs/pmpp.py run 8.1 --bench` (real devices only).

## Where this lab differs from the page

- **c3 and c4.** Figs. 8.6 and 8.8 multiply the y-1 neighbour by c3 and the y+1
  neighbour by c4. Figs. 8.10 and 8.12 print it the other way round. The lab uses the
  first pairing everywhere (`stencil.h`), and the test's seven weights are all different,
  so the printed Fig. 8.12 pairing fails.
- **`inNext_s`.** Fig. 8.12's last line stores `inNext_s` into the shared plane. That
  kernel has no `inNext_s`; the value to store is the register `inNext`.
- **Slab depth.** The book's slab is `OUT_TILE_DIM` planes deep. Here it is the argument
  `depth`, so that the benchmark can compare slabs; with `depth = TILE_WIDTH - 2` your
  kernel is the book's.
- `in` is `const` and the seven weights arrive as one by-value argument. The listings
  leave `c0` to `c6` undeclared.

## Do not return early

Every thread of a block must reach both barriers on every iteration of the z loop. A
thread on the edge of the block, or on the grid's boundary, computes no output but still
loads its element of the halo, which its neighbours read. Put conditions around the
loads and around the store; never `return` before the loop's barriers.

The emulator cannot show you an early return on its own. NVIDIA's guide says a barrier
waits for all *non-exited* threads, Apple says the same for Apple silicon, and the
emulator models exactly that: a thread that has returned stops counting. What the
emulator does catch is the damage, such as a halo value nobody loaded. AMD states no rule
for this case. The portable rule is the book's: every thread of the block executes every
barrier (`notes/today/ch04-facts.md` A6 and C3).

## Reading a failure

- *Those points still hold the input: nothing was written there.* The kernel did not
  write those interior points at all (or did not run).
- *N boundary points changed.* Something wrote outside the interior, usually a missing
  or wrong grid-interior guard.
- *The kernel wrote past the end of out.* The driver puts a guard region after `out` and
  checks it after every launch.
- On the emulator, a missing barrier gives wrong values in almost every case under all
  three thread orders (a lucky order can hide it in a tiny grid, so read all three), and
  AddressSanitizer stops the run at the first out-of-bounds read or write and names the
  line (for example a load of plane `i + 1` when `i + 1` is already past the grid). The
  tests after that stop do not run.

## The experiment

Do this once all tests pass. Before you run `--bench`, write down three predictions:

1. The book's model: the basic kernel asks for 32 bytes per output point (seven 4-byte
   loads and one store), 0.41 FLOP/B (p. 189). The register-tiled kernel at the book's
   slab depth, T = 32 and 30 planes, loads 32 x 32 x 32 values and stores 30 x 30 x 30
   per block: 1.47 FLOP/B. (The book prints 1.52 on p. 195. 1.52 is what you get by
   counting only the x-y halo, the limit as the slab grows without bound; the book does
   not say how it counted. That computation is ours, `notes/today/ch08-facts.md` E2. The
   deeper slabs below test the limit.)
   That is about **3.6 times fewer bytes**. How many times faster do you expect it to
   be on your GPU?
2. The provided three-shared-plane kernel (Fig. 8.10) moves exactly the same global
   data as yours (p. 198); yours swaps two shared planes for registers. How many times
   faster than it do you expect yours to be?
3. Which slab depth will be fastest: the book's T - 2, 8, 16, 64, or one slab for the
   whole column?

Caches already serve many of the basic kernel's neighbour reads, as you saw in chapter 5:
a model's factor says how much less the kernel *asks* of memory, and the speed-up you
measure is smaller wherever the caches were already doing that work. The 3.6 is a
statement about requested bytes, not a promise.

What `--bench` does: it grows N until the basic kernel takes long enough to time, without
letting `in` and `out` together pass 2 GiB or an eighth of the memory your device
reports. It then times every kernel on that grid (the best of at least 5 runs, repeated
until the runs add up to about a quarter of a second, because single sweeps this short
vary a lot) and checks each result against the CPU afterwards; a kernel whose output is wrong is reported, not timed. Each
line shows milliseconds, interior points per second, and "GB/s at 8 B/point": one 4-byte
load and one 4-byte store per interior point, the traffic of the 13/8 FLOP/B ideal, not
what the kernel actually asked of memory. Speed-ups are against the basic kernel and,
for your register-tiled kernel, against the planes kernel of the same tile width.

The slab depths are T - 2 (the book's), 8, 16, 64 and N, marked "1 slab". The grid has
`ceil(N / depth)` slabs and slab 0 starts at plane 0, a boundary plane, so depth N is the
only depth that puts the whole column in one slab; N - 1 or N - 2 would launch a second
slab with at most one plane to compute. "limits" lines record how many threads per
block your device allows for the cube kernel and for your register-tiled kernel at each
tile width (the Metal pipeline's `maxTotalThreadsPerThreadgroup`, which belongs to each
compiled kernel, or `maxThreadsPerBlock` on CUDA and HIP).

On CUDA a 32 x 32 block leaves at most 64 registers per thread; past that the launch
fails rather than running slowly (`notes/today/ch08-facts.md` A4). If a T = 32 launch
ever reports "too many resources requested", that is the reason; `nvcc
--resource-usage` shows the kernel's count.
