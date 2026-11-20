# Lab 5.2: the tiled kernel

Book: section 5.4 (pp. 108-112), Fig. 5.9.

Write the tiled matrix multiplication: two shared-memory tiles, one load of M and one of
N per thread per phase, two barriers. Square matrices whose width is a multiple of
`TILE_WIDTH`; lab 5.3 removes both restrictions.

- Edit `work/kernel.cu` and `work/kernel.metal` + `work/launch.mm`.
- `TILE_WIDTH` is defined by the build (16 by default; 8 and 32 are also built for the
  benchmark; Metal gets 4 to 32 at run time). Do not hard-code 16.
- Test: `python3 labs/pmpp.py run 5.2`   Start with `--backend emu`.
- Measure: `python3 labs/pmpp.py run 5.2 --bench` (compares against a provided naive kernel)

Before you benchmark, write down the speed-up you expect at T = 16 and which tile width
you expect to be fastest. The lesson's model says T times less traffic; your device
decides how much of that turns into time.
