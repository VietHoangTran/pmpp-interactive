# Snippets

Code that chapter 2's lessons show for the sake of an explanation or a question, and that
is not a lab. The files are kept here as real sources so that the CUDA front-end check in
`./verify.sh` parses every `.cu` file, and so that a lesson shows the checked file and not
a copy that can drift. They are not built or run by the lab runner.

How lessons use them:

- loaded at run time with `useLabFile('ch02/snippets/<name>')`; or
- copied into a section file, with a unit test beside the section that fails if the copy
  and the file differ. Edit the test with the file.

| File | Shown in | How | Test |
|---|---|---|---|
| `vecadd_sequential.cu` | section 2.3, the loop before it moves | loaded | `app/src/chapters/ch02/S23.test.ts` |
| `vecadd_stub.cu` | section 2.4, the stub and its stepper | loaded | `S24.test.ts`, `app/src/sim/offload.test.ts` |
| `vecadd_stub_metal.mm` | section 2.4, the Metal stepper | read by the model's test | `app/src/sim/offload.test.ts` |
| `ex9_foo.cu` | section 2.6, the book's exercise 9 | copied | `S26.snippets.test.ts` |
| `saxpy_stub_defect.cu` | summary page, first question | copied | `Wrap.snippets.test.ts` |
| `count_above_race.cu` | summary page, third question | copied | `Wrap.snippets.test.ts` |

`vecadd_stub_metal.mm` is not part of any build. Its header comment gives the command that
compiles and runs it by hand on a Mac.

Two of these are wrong on purpose. `saxpy_stub_defect.cu` never copies `y` to the device
although its kernel reads it. `count_above_race.cu` updates one counter from many threads
without an atomic. Both parse, and neither may become the basis of a lab.
