# Chapter notes template

Notes are the working source for lessons, labs and review questions, so later
sessions never have to re-read the PDF. They must be accurate to the PDF (not to
memory of earlier editions), written in our own words, and specific enough that
someone who has not seen the page could re-explain the idea and re-draw the figure.

Every factual item carries a book page reference `(p. 104)`. Book page = the number
printed on the page. `notes/pagemap.json` gives each chapter's PDF range and offset.

Use exactly these headings, in this order.

```markdown
# Chapter N — Title

- **Book pages:** a–b  **PDF pages:** x–y (pdf = book + offset)
- **In one paragraph:** what the chapter is for and what a reader can do afterwards.

## Section map
| § | Title | Book pp. | What it teaches (one or two lines) |

## Concepts
### §N.M Title (pp. a–b)
Bullets, in the order the book develops them. Capture:
- definitions and the exact terms the book uses
- reasoning chains (why X follows from Y), formulas, worked numbers
- every device-specific number, tagged `[device: <model>]` (e.g. GPU model specs)
- pitfalls and misconceptions the book warns about

## Figures
| Fig. | Book p. | Kind | What it shows | Interactive idea |
Kind is one of: diagram, code, plot, table, photo.
"What it shows" must be detailed enough to re-illustrate independently: the
elements, their arrangement, labels, and the point being made. Do not skip any figure.

## Code listings
One entry per code figure:
### Fig. N.M — name (p. X)
- Purpose, signature, launch configuration it assumes
- Walkthrough in our own words (what each stage does, by line range)
- Lines that need care (index arithmetic, sync points, boundary conditions). Quote
  at most a few lines verbatim when the exact expression matters.
- APIs / qualifiers / built-ins used

## Exercises
| # | Book p. | Gist | Type | Answer sketch |
Type: compute, predict, explain, code, debug. Give an answer sketch where you can
derive one with confidence; write "not derived" otherwise.

## Dated or device-specific content
Claims tied to a GPU model, a CUDA version, an API name or a product. These become
"today" notes after checking current docs, so list them precisely with pages.

## Relevance to LLM-inference kernels
How this chapter serves: matmul/GEMM, attention, quantized ops, sampling
(softmax, top-k/top-p, argmax). Be concrete or say "little direct relevance".

## Portability notes
CUDA-specific APIs and features the chapter relies on. For each, what a HIP or Metal
port would need. Mark anything not stated in the book as `[unverified]`: do not
assert HIP/Metal facts from memory as if they were checked.

## Ideas for interactives and labs
Concrete ideas tied to specific figures/listings: steppers, predict-then-reveal,
bug hunts, parameter sliders, labs with measurable outcomes.

## Cross-references
Forward and backward references to other chapters that the book itself makes.

## Unclear in the PDF
Anything illegible, ambiguous, apparently erroneous (possible errata), or that you
could not resolve. Say so rather than guessing.
```
