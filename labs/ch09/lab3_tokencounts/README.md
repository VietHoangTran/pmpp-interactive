# Lab 9.3: token counts for repetition penalties

Book: sections 9.2 to 9.5 (pp. 203-217), Figs. 9.6, 9.9 and 9.14. Counting token ids, the
decode-step update and the penalty formula are not in the book. They are the counting
piece of milestone M5; module X1 composes it with top-k and top-p sampling.

A sampler with frequency and presence penalties needs to know, for every request in the
batch, how often each vocabulary id already occurs in that request's context. That is a
histogram whose bins are a whole vocabulary: 50,257 ids for GPT-2, 151,936 for Qwen3-8B,
201,088 for gpt-oss-20b (`notes/today/ch09-facts.md` D7). vLLM keeps exactly this on the
GPU, an `int32` tensor of `[max_num_reqs, vocab_size]` updated with atomic adds (D6).

## Before you write: which privatisation still applies?

Chapter 9 cures contention by giving subsets of threads private copies of the bins. Before
you open `work/`, write down for each item below whether it applies to vocabulary-sized
bins, and one sentence of why. Use your device's figure from
`python3 labs/pmpp.py doctor` (`sharedMemPerBlock` on CUDA and HIP,
`maxThreadgroupMemoryLength` on Metal) where it matters.

1. A private copy per block in shared memory (Fig. 9.10, p. 214).
2. A private copy per block in global memory (Fig. 9.9, p. 212).
3. A private copy per thread block cluster, in distributed shared memory (p. 213).
4. Thread-level privatisation: merge runs of equal values in a register (Fig. 9.15, p. 218).
5. The batch: each request has its own row of counts.
6. Not recounting: during decoding, each step adds one token per request.

Your answers decide what `countTokens` looks like. Ours are at the end of this file.

## Your part

Three kernels, each with a launcher. Row-major, batch first, int32 ids, uint32 counts.

- **`countTokens`**: `counts[r*vocab + t]` = the number of times id `t` occurs among row
  `r`'s `len` tokens. The caller has zeroed `counts`. Any `len` from 1 up; `vocab` from
  1,000 (small tests) to 201,088. Use the book's API from section 9.2,
  `cuda::atomic_ref` and `fetch_add` with `cuda::memory_order_relaxed`, with the scope your
  design needs.
- **`appendTokens`**: one decode step. Row `r` gains one token, `newTokens[r]`; add it to
  that row's counts. Afterwards the counts must equal a count of all `len + 1` tokens.
- **`applyPenalties`**: for every element of the batch's logits,
  `logit -= count * frequencyPenalty + (count > 0) * presencePenalty`. This is llama.cpp's
  frequency and presence penalty (D4). With both penalties 0 the logits must not change.

The sign-aware repetition penalty (divide a positive logit, multiply a negative one) is not
here. Its exact form is not in the facts file yet; module X1 adds it.

## Files

- Edit `work/kernel.cu` (CUDA, HIP and both emulators): `countTokens`, `appendTokens`,
  `applyPenalties` and
  - `launchCountTokens(const int* tokens, unsigned int* counts, int batch, int len, int vocab)`
  - `launchAppendTokens(const int* newTokens, unsigned int* counts, int batch, int vocab)`
  - `launchApplyPenalties(float* logits, const unsigned int* counts, int batch, int vocab, float frequencyPenalty, float presencePenalty)`
- Edit `work/kernel.metal` (`count_tokens`, `append_tokens`, `apply_penalties`) and
  `work/launch.mm` (`encodeCountTokens`, `encodeAppendTokens`, `encodeApplyPenalties`).
  The kernel signatures and buffer bindings are given; the grid sizes are yours.
- Provided, not edited: `provided/count_pool.cu`, `provided/count_pool.metal` and
  `provided/count_pool.mm`. That is Fig. 9.9's design applied to token counts: every block
  gets a private copy of the whole vocabulary in a global-memory pool, counts there with
  block-scope atomics, then commits every non-zero bin to the request's row. One thread
  per token, 256 per block, as in Fig. 9.9. The pool is zeroed by the launcher, and that
  is timed with it. It is the benchmark's comparator; the tests run it too, so a broken
  copy shows.

## Commands

- Test: `python3 labs/pmpp.py run 9.3 --backend emu` first, then `python3 labs/pmpp.py run 9.3`.
- Atomics: `python3 labs/pmpp.py run 9.3 --backend race`.
- Measure: `python3 labs/pmpp.py run 9.3 --bench` (real devices only).
- The reference solution: add `--solution`.

## What the tests do

- Shapes (batch, len, vocab): on `emu` and `race`, (1, 1, 1000), (3, 100, 1000) and
  (2, 64, 50257). Real devices add (1, 1, 50257), (3, 1000, 50257), (2, 4096, 151936),
  (4, 777, 201088) and (8, 2048, 151936).
- Four seeded token distributions per shape: uniform over the vocabulary; Zipf-like, where
  a few ids take most of the row; one id repeated through the whole row (every thread hits
  the same bin); and rows holding ids 0 and vocab-1, which catch a wrong row offset.
- `countTokens`: the counts must match the CPU reference exactly. A failure names the
  first wrong (row, id) and compares the total of all counts with the reference: a total
  that is too small means lost updates; a right total with wrong bins means a wrong index.
- `countTokens then appendTokens`: one more token per row must give the counts of all
  `len + 1` tokens, exactly. If `countTokens` was wrong, `appendTokens` is still checked,
  from the reference counts, and the message says which of the two failed.
- `applyPenalties`, once per shape, on the counts of a Zipf-like batch (many zeros, a few
  large counts) and logits in [-8, 8]: with penalties (0.5, 0.25) the result must be within
  1e-5 times the size of the terms of the CPU reference (Metal does not default to strict
  IEEE arithmetic); with (0, 0) every logit must come back bit for bit unchanged.
- The driver zeroes `counts` before each `countTokens`, as the contract says the caller
  does: device buffers start as garbage (Metal's are filled with `0xA5` bytes).

What each backend can catch:

- `emu` runs one thread at a time under AddressSanitizer, in three fixed orders. It finds
  wrong indices, a row offset that is off by one, and reads or writes past a buffer. It
  cannot see a missing atomic: a plain `++` gives the right counts there every time.
- `race` runs the threads of a block, and small blocks, concurrently under
  ThreadSanitizer. A plain `++` in `countTokens` is reported with its source line once two
  threads that run together hit the same id, which the Zipf-like and repeated rows make
  very likely. `appendTokens` with a plain increment is not reported, and must not be: one
  thread owns each row.
- Metal is a real GPU. On the machine this lab was built on, a plain `+= 1` in
  `count_tokens` lost most of the updates of the repeated rows, and different ones on
  each run.
- Nothing here can catch the wrong scope. If several blocks share a row, a
  `cuda::thread_scope_block` update of that row is a data race and undefined behaviour
  (ch09-facts A2), but both emulators ignore scopes, and Metal has none. Check it by
  reading your code.

## Where the backends differ (not in the book)

- **CUDA**: `cuda::atomic_ref<T, Scope>` defaults to system scope and `fetch_add` to
  `seq_cst` when you leave them out (ch09-facts A1). Pass both, as the book does.
- **HIP**: `<cuda/atomic>` keeps its name through libhipcxx, ROCm 10.1 and later, and the
  HIP build must link it (infra-facts B2). The scope is the same template argument
  (ch09-facts B1). Not run on AMD hardware here: unverified.
- **Metal**: there is no atomic view of a plain array. `count_tokens` declares the counts
  `device atomic_uint*` and updates them with
  `atomic_fetch_add_explicit(..., memory_order_relaxed)`; a read-modify-write takes no
  scope (ch09-facts C1). The atomic type belongs to the kernel's declaration, not to the
  buffer, which holds plain `uint`s: `append_tokens` and `apply_penalties` declare the
  same buffer `device uint*`. PyTorch likewise binds a plain `uint32` tensor as
  `device atomic_uint*` (C1).
- **The provided pool on Metal**: zeroing is a kernel of its own, dispatched before the
  counting kernel in the same compute pass. That pass is serial, so the second dispatch
  sees the first one's writes (ch02-facts C9). The plain read of a private bin in the
  commit becomes `atomic_load_explicit`, and the barrier between the two phases orders
  device memory, `threadgroup_barrier(mem_flags::mem_device)` (ch05-facts C2).

## Before you benchmark, predict

The benchmark runs batch 8, len 4096, vocab 151,936, on uniform and Zipf-like rows. The
provided pool uses 256 threads per block and one token per thread: 16 blocks per row,
128 blocks, each counting 256 tokens into its own copy of 151,936 bins. Write these down
first. The lesson asks for the first under the id in brackets.

1. Is `count_pool` faster or slower than your direct atomics into each request's row, and
   by roughly what factor? Reason from what one block of the pool does: how many updates,
   against how many bins it must zero and read back to commit
   (`ch09/labs/p-lab3-pool`).
2. How many times cheaper is one `appendTokens` step than recounting the whole context
   with `countTokens`? Write down the ratio of the work first, then the ratio of time you
   expect, and say why they could differ.

## Reading the output

The benchmark first prints the size of one private copy of the vocabulary next to your
device's per-block on-chip memory, as the device reports it. Then one table, two blocks of
rows (uniform, Zipf-like). `pmpp.py` also saves the rows in `reports/runs/`, where the app
reads them.

- `count (yours)`: one `countTokens` launch. `counts` is zeroed before each run, outside
  the timed region. `tokens/s` is batch x len divided by the time.
- `count_pool (provided)`: zeroing the pool and Fig. 9.9's kernel, both timed. The row also
  records the blocks per row and the pool's bytes. It runs only if the pool takes at most
  a quarter of the memory your device reports.
- `append (yours)`: one decode step for the whole batch. `tokens/s` is batch divided by the
  time.
- `penalties (yours)`: every logit of the batch with penalties (0.5, 0.25). `elements/s` is
  batch x vocab divided by the time. `GB/s` counts 12 bytes per logit (read the logit, read
  its count, write the logit): the least a kernel must move, not what the memory system
  moved. The same buffers are used on every run, so this is not a measurement of your
  memory bandwidth; lab 5.1 measures that.
- The last line of each block divides the measured times: recount against one append, and
  the pool against your count.
- Every time is the best of at least 10 runs, after a quarter of a second of untimed work
  so the first kernel does not also pay for waking the device. On Metal a time is the
  GPU's start-to-end time of one command buffer. On CUDA and HIP it is host wall-clock
  time around the launch and a synchronise, so a launch's fixed cost is included.

The counts that `count` and `count_pool` leave behind are checked against the CPU
reference after timing; a wrong result is reported as wrong, without a time. The numbers
are measurements of this machine at this size. They are not a property of the kernels.

## Go further

- Section 9.5's cure for the pool is coarsening: fewer blocks, each counting more tokens,
  means fewer copies to zero and commit. Using your measured `count_pool` time, work out
  how much of it 4, 16 and 64 tokens per thread would remove at the benchmark's size, and
  what the fewest blocks, one per row, would cost the counting phase. Does any coarsening
  factor bring the pool level with direct atomics?
- llama.cpp counts penalty tokens on the CPU with a hash map over a window of the last
  `penalty_last_n` tokens, adding the new token and removing the one that leaves the
  window (D4). Write `removeTokens`, the decrement for a sliding window, and say why it
  needs no atomic either.

## Our answers to the first question

Read this after you have written yours.

1. **Shared memory: no.** One `uint32` copy is 196.3 KiB at GPT-2's vocabulary, 593.5 KiB
   at Qwen3's and 785.5 KiB at gpt-oss's. Only GPT-2's fits anywhere, behind the opt-in on a
   few NVIDIA compute capabilities; the two current vocabularies fit in no documented
   per-block limit on any of the three backends. Halving the counters to `uint16` does not
   rescue it: only one compute capability's opt-in limit would then hold Qwen3's, and
   16-bit atomics are emulated on CUDA and HIP and absent on Metal (D7, A1, C2). Compare
   with your `doctor` figure; the benchmark prints both.
2. **Per-block copies in global memory: they work, but cost a whole vocabulary per block.**
   Each block zeroes and commits every bin of its copy, whatever it counted. With 256
   tokens per block and 151,936 bins, that overhead dwarfs the updates it saves. The
   benchmark measures it.
3. **Clusters: not here.** Distributed shared memory needs CUDA compute capability 9.0 or
   later (ch05-facts A6), HIP devices that report cluster launch (infra-facts B4), and
   Metal has none. Neither emulator models clusters (`labs/README.md`).
4. **Run merging: little to gain.** Fig. 9.15 pays off with long runs of one value among a
   thread's elements. With the interleaved index those elements are a block apart in the
   row, and text rarely repeats a token there (our reading, not measured). The repeated
   test rows are the exception: a model stuck in a loop.
5. **The batch: yes, and for free.** Rows of different requests never share a bin. Each
   request's row is already a private histogram that nobody merges, so contention is
   confined to the blocks of one row, and the atomics need device scope only because
   several blocks share that row.
6. **Not recounting: yes, for decoding.** After the prompt is counted once, each step adds
   one token per request. One thread per row owns that row, the owner-computes rule of
   p. 201 holds again, and a plain increment is correct. llama.cpp keeps its counts this
   way on the CPU (D4).
