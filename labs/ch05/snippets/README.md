# Snippets

Kernels that lessons show for the sake of a question or an exercise, and that are not
labs. They are kept here as real `.cu` files so that `./verify.sh` runs the CUDA
front-end check on them, and a unit test (`app/src/chapters/ch05/snippets.test.ts`)
fails if the copy shown in a lesson drifts from the file. They are not built or run.
