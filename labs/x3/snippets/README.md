# Snippets

Code that module X3's lessons show and that is not a lab. The files are real sources, so a
lesson shows the checked file and not a copy that can drift. They are not built by the lab
runner; each one's test is in `app/src/sim/`. One row per snippet. Everything here is ours:
module X3 is not in the book.

| File | Shown in | How | Test |
|---|---|---|---|
| `ckpt_check.cpp` | section X3.8, "Check a download by its size": read a llama2.c version 0 header, predict the file size, compare with the size on disk | loaded at run time | `app/src/sim/ckptformat.test.ts` |

## `ckpt_check.cpp`

Host C++ only (no kernel). The size formula is `notes/today/x3-facts.md` E1 over run.c's
`memory_map_weights` order (D1). The test compiles it with `-Wall -Wextra -Werror`, writes
sparse files with the stories15M and stories260K headers (D2) at the length the model predicts,
one 4,096 bytes short, one with a negative `vocab_size` and one shorter than a header, runs the
program on each and compares its output and exit status with `app/src/sim/ckptformat.ts`.
