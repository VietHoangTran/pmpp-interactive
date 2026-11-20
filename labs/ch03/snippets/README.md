# Snippets

Code that chapter 3's lessons show for the sake of an explanation or a question, and that
is not a lab. The files are kept here as real `.cu` (and `.metal`) sources so that the CUDA
front-end check in `./verify.sh` parses every `.cu` file, and so that a lesson shows the
checked file and not a copy that can drift. They are not built or run by the lab runner.

How lessons use them:

- loaded at run time with `useLabFile('ch03/snippets/<name>')`
  (`vec_add_1d.cu`, `launch_1d.cu`, `launch_fig31.cu`, `launch_checked.cu`); or
- copied into a section file, with a unit test beside the section
  (`app/src/chapters/ch03/S3x.snippets.test.ts`) that fails if the copy and the file differ.

Lessons refer to some lines by number. `app/src/chapters/ch03/S31.snippets.test.ts` pins
the lines that the "Before you start" page and section 3.1 point at; edit that test with
the file.
