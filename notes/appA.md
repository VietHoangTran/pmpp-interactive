# Appendix A — Numerical considerations

- **Book pages:** 583–600  **PDF pages:** 607–624 (pdf = book + 24)
- **Contributor line:** "With special contributions from Federico Busato" (p. 583).
- **In one paragraph:** A primer on how floating-point numbers are encoded (sign, excess-coded exponent, normalized mantissa), which values a format can represent exactly, how zero and the region near zero are handled (no-zero, abrupt underflow, denormalization), the IEEE special patterns (infinity, NaN), and the difference between *precision* (a property of the format) and *accuracy* (a property of an operation, measured in ULP). It then shows why the order of additions changes a floating-point sum, so a parallel reduction tree can give a different answer from a sequential loop, and why numerical stability (pivoting in Gaussian elimination) makes some algorithms harder to parallelize. A short section notes that GPUs now have hardware for 16-, 8- and 4-bit floating point. Afterwards a reader can enumerate the representable numbers of a toy format, compute a rounding error in ULP, predict the result of a sum under a given association order, and explain why pivoting is expensive across thread blocks.

Conventions used by the book throughout the appendix (p. 584): a subscript `D` marks decimal place value, a subscript `B` marks binary place value. Example: 0.5_D = 5 × 10^-1 = 0.1_B = 1 × 2^-1. These notes write them as `0.5D`, `0.1B`.

## Section map

| § | Title | Book pp. | What it teaches (one or two lines) |
|---|---|---|---|
| (intro) | untitled opening | 583 | History of floating-point hardware; why FP beats integers for dynamic range; what the appendix covers (accuracy, precision, stability). Mentions LLM training/inference as the current driver of FP throughput. |
| A.1 | Floating-point data representation | 583–586 | IEEE-754 S/E/M fields, value formula (A.1), normalized mantissa with hidden "1.", excess (biased) exponent encoding, worked 6-bit example. |
| A.2 | Representable numbers | 586–590 | Number-line view of a format; five observations on a 5-bit toy format; three ways to treat the region near 0 (no-zero, abrupt underflow, denormalization); precision defined. |
| A.3 | Precision and special bit patterns in IEEE format | 590–591 | Single vs double field widths; infinity, NaN (signaling vs quiet), denormal and zero patterns. |
| A.4 | Arithmetic accuracy and rounding | 591–592 | Accuracy vs precision; why add/multiply need rounding; operand alignment; ULP; 0.5 ULP ideal; approximation-based units can be worse. |
| A.5 | Lower-precision formats | 592–593 | Why sub-32-bit floating point exists; GPU hardware support for 16/8/4-bit FP; H100 FP8 vs single-precision tensor-core throughput. Two paragraphs only. |
| A.6 | Algorithm considerations | 593–594 | Summation order changes the result; sequential vs reduction-tree example; pre-sorting; pointer to Kahan summation. |
| A.7 | Linear solvers and numerical stability | 594–599 | Gaussian elimination + back substitution worked example; a parallel CUDA design; instability when a lead coefficient is zero or tiny; pivoting and its cost in parallel settings. |
| A.8 | Summary | 599 | Recap. |
| A.9 | Exercises | 599–600 | Five exercises. |
| — | References | 600 | Six references (Goldberg; IEEE 754-2019; OCP MX spec; NVIDIA CUDA floating-point reference; Kahan; Ballard et al.). |

## Concepts

### Opening (p. 583)

- Floating-point hardware was once limited to mainframes and supercomputers. 1980s microprocessors had FP coprocessors that were roughly three orders of magnitude slower than those machines. 1990s microprocessors (the book names Intel Pentium III and AMD Athlon) reached supercomputer-class FP. Fast FP is now standard on CPUs and GPUs (p. 583).
- The book states that demand from LLM training and inference has recently pushed GPUs to increase FP throughput dramatically (p. 583).
- Why floating point rather than integers: larger dynamic range of representable values, and more precise representation of tiny values (p. 583).
- Scope stated for the appendix: accuracy of FP operations, precision of FP representation, stability of numerical algorithms, and how these bear on parallel programming (p. 583).
- Footnote 1 (p. 583): there are emerging alternatives to IEEE floating point in the AI context; the example given is the OCP Microscaling Formats (MX), which the book says "have been adopted by recent GPUs" (ref. [3]). No further detail on MX is given anywhere in the appendix.

### §A.1 Floating-point data representation (pp. 583–586)

- IEEE-754 is a common standard for FP representation and arithmetic behaviour; the book cites the IEEE 754-2019 revision (ref. [2]) and says virtually all future microprocessors will fully or almost fully conform (p. 583).
- A value is stored as three bit groups: sign **S**, exponent **E**, mantissa **M** (p. 584). Apart from exceptions covered later, each (S, E, M) pattern identifies one value by Eq. (A.1) (p. 584):

  `value = (-1)^S · 1.M · 2^(E - bias)`

- S: 0 means positive, 1 means negative (because (−1)^0 = 1, (−1)^1 = −1) (p. 584).
- The running toy format for A.1 is **6 bits: 1-bit S, 3-bit E, 2-bit M** (p. 584). (A.2 switches to a 5-bit format; see below.)
- **Normalized representation of M** (p. 584):
  - The mantissa is always read as `1.M`, which makes the mantissa pattern for each number unique. Example: 0.5D must be written `1.0B · 2^-1`; the alternatives `0.1B · 2^0` and `10.0B · 2^-2` do not have the form 1.M.
  - Numbers satisfying this restriction are called **normalized numbers**.
  - Because every normalized mantissa starts with "1.", that bit is not stored. The 2-bit mantissa of 0.5 is `00` (from 1.00 with "1." dropped).
  - Consequence: an m-bit mantissa field effectively carries (m + 1) bits of mantissa.
- **Excess encoding of E** (pp. 584–586):
  - The number of E bits sets the *range*. With exponent value 64 the number lies between 2^64 (> 10^18) and 2^65; with −64 it lies between 2^-64 (< 10^-18) and 2^-63. This is how FP covers a wider range than integer formats (p. 585).
  - IEEE uses **excess (biased) encoding**: with e exponent bits, add `2^(e-1) - 1` to the two's-complement exponent value to get the stored code. For e = 3 the added value is 2^2 − 1 = 3 = `011`, hence "excess-3" (p. 585).
  - Two's complement is defined in passing: negate by complementing every bit and adding one (p. 585).
  - Why excess coding: the codes increase monotonically with the value they represent, so an **unsigned comparator** can compare two exponents directly. Example: codes 001 and 100 compare as 001 < 100, matching −2 < 1 (p. 585).
  - The all-ones code (111 in excess-3) is a **reserved pattern**. The book's argument: zero plus equally many positive and negative values gives an odd count, so one of the 2^e patterns is left over; assigning it to a value would unbalance positives and negatives. IEEE uses it for special values (A.3) (p. 585).
- Worked encoding of 0.5D in the 6-bit format (p. 586): S = 0, E = `010` (exponent −1 in excess-3), M = `00` (from "(1.)00"), giving the bit string `0 010 00` = `001000`.
- General value of (S, E, M) with an n-bit exponent, normalized mantissa and excess coding (p. 586):

  `(-1)^S · 1.M · 2^(E - (2^(n-1) - 1))`

### §A.2 Representable numbers (pp. 586–590)

- **Representable numbers** of a format = the values the format can store exactly (p. 586). Example: a 3-bit unsigned integer represents exactly 0..7 (Fig. A.2); −1 and 9 are not representable (Fig. A.3 marks 0..7 on a number line).
- To keep tables small, A.2 uses a **5-bit format: 1-bit S, 2-bit E (excess-1), 2-bit M (hidden "1.")** (p. 586). E codes: `00` → exponent −1, `01` → 0, `10` → +1, `11` reserved. Fig. A.4 tabulates all values under three conventions (no-zero, abrupt underflow, denorm).
- Positive representable numbers, **no-zero** convention (the plain formula from A.1), from Fig. A.4/A.5 (p. 587):

  | E | exponent | values (M = 00, 01, 10, 11) | spacing |
  |---|---|---|---|
  | 00 | −1 | 0.5, 0.625, 0.75, 0.875 | 0.125 |
  | 01 | 0 | 1, 1.25, 1.5, 1.75 | 0.25 |
  | 10 | +1 | 2, 2.5, 3, 3.5 | 0.5 |
  | 11 | — | reserved | — |

  Negative numbers mirror these (p. 587).
- **Five observations** from the number line of the no-zero format (pp. 587–588):
  1. **Exponent bits define the major intervals.** Intervals run between powers of two. With 2 exponent bits and one reserved code there are three powers of two per side (2^-1, 2^0, 2^1), each starting an interval, so three major intervals on each side of 0 (p. 587).
  2. **Mantissa bits define how many representable numbers each interval holds.** N mantissa bits give 2^N per interval (4 here). A value between two representable numbers is rounded to one of them, so more mantissa bits means higher precision (p. 587).
  3. **0 is not representable** in the no-zero format. The book calls this a serious deficiency (p. 588).
  4. **Representable numbers get denser toward 0.** Each interval is half the width of the one further out (widths 2, 1, 0.5 going inward) but holds the same count. This is desirable: smaller magnitudes need finer absolute resolution. The spacing sets the maximal rounding error in an interval. Analogy: a one-dollar error is invisible on a billion-dollar balance but obvious on a ten-dollar one (p. 588).
  5. **The trend breaks right next to 0.** Because a normalized mantissa cannot be 0, there is a gap between 0 and the smallest interval. Errors for values in (0, 0.5) are 4× larger than for values in (0.5, 1.0); in general, with m mantissa bits the innermost gap has 2^m times the error of the next interval. This hurts methods that test convergence on very small values, and algorithms that divide by small numbers, where the representation error is magnified (p. 588).
- **Abrupt underflow** (p. 588, Fig. A.4 middle columns, Fig. A.6): whenever E is 0, the value is taken to be 0. In the 5-bit format this turns eight patterns (four positive, four negative, between −1.0 and +1.0) into 0. Simple, used by some 1980s mini-computers and, the book says, still used by some arithmetic units that must run very fast. It makes 0 representable but **widens the gap near 0** from 0.5 to 1.0 (2×) (pp. 588–589). Positive values: 0, then 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5.
- **Denormalization** — the method IEEE actually adopts (p. 589, Fig. A.4 right columns, Fig. A.7):
  - When E = 0 the mantissa is read as `0.XX` instead of `1.XX`, and the exponent is taken to be that of the neighbouring (next larger) interval rather than its own. In the 5-bit format: exponent 0 instead of −1.
  - Worked example: pattern `00001` (S=0, E=00, M=01) → 0.01B × 2^0 = 2^-2 = 0.25.
  - Positive values: 0, 0.25, 0.5, 0.75, then 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5. The innermost four values are spread uniformly across the former gap; spacing in the last two intervals ([0,1) and [1,2)) is identical (0.25).
  - General formula when the n-bit exponent field is 0 (p. 589): `0.M · 2^(-2^(n-1) + 2)`. (Check: n = 2 gives 2^0, matching the example.)
  - Cost: hardware must detect the denormal range and switch representation. Implementations with moderate hardware can add *thousands of clock cycles* whenever a denormal is produced or consumed. The book gives this as the reason early CUDA devices did not support denormals; "virtually all recent generations of CUDA devices" do (ref. [4]) (p. 589).
  - The book says denormalization can be switched with an NVCC compiler flag, written as "ftz = true (on) or ftz = false (off)" (p. 589). See "Unclear in the PDF" for the ambiguity in which setting means what.
- **Precision, defined** (p. 590): the maximal error introduced by representing a number as one of the representable numbers. Smaller error = higher precision. Each extra mantissa bit halves the maximal error. This is the single- vs double-precision distinction.

### §A.3 Precision and special bit patterns in IEEE format (pp. 590–591)

- Field widths (p. 590):

  | Format | Total | S | E | M |
  |---|---|---|---|---|
  | single precision (sp) | 32 | 1 | 8 | 23 |
  | double precision (dp) | 64 | 1 | 11 | 52 |

- Double has 29 more mantissa bits, so its largest rounding error is 1/2^29 of single's (p. 590).
- The book says the three extra exponent bits extend "the number of intervals of representable numbers by 2^(2^3) = 256 times" (p. 590). See "Unclear in the PDF".
- Scale illustration: atoms in the universe ≈ 10^80; double precision reaches about 10^308 (p. 590).
- **Special patterns** (Fig. A.8, p. 590):

  | exponent bits | mantissa | meaning |
  |---|---|---|
  | all 1s | ≠ 0 | NaN |
  | all 1s | = 0 | (−1)^S × ∞ |
  | all 0s | ≠ 0 | denormalized |
  | all 0s | = 0 | 0 |

- **Infinity** (p. 590): produced by overflow, e.g. a large number divided by a very small one. FP results **saturate** at ±∞ when they exceed the representable range. The book contrasts this with integer arithmetic, whose out-of-range results wrap around or are undefined. Any representable number divided by ±∞ gives 0.
- **NaN** (pp. 590–591): produced when the result is mathematically undefined. The book's list: 0/0, 0 × ∞, ∞/∞, ∞ − ∞. Also used to mark data that was never initialized.
  - **Signaling NaN (SNaN):** most significant mantissa bit *cleared*. Using one as an arithmetic input raises an exception (e.g. 1.0 + SNaN signals the OS). Used to force interruption, e.g. filling uninitialized data with SNaN to catch its use (p. 591).
  - **Quiet NaN:** most significant mantissa bit *set*. Arithmetic on it yields another quiet NaN with no exception (1.0 + QNaN = QNaN). Used where a user inspects output afterwards; printed as "NaN" (p. 591).
  - **GPU caveat:** "The current generation of GPU hardware does not support signaling NaN", because accurately signaling exceptions during massively parallel execution is hard (p. 591).
  - Many NaN bit patterns exist; the book counts 2^22 − 1 in single precision given one mantissa bit is reserved for signaling/quiet, and notes the spare patterns could encode the cause of the NaN (p. 591).

### §A.4 Arithmetic accuracy and rounding (pp. 591–592)

- **Precision vs accuracy** (p. 591): precision comes from the number of mantissa bits in the *format*; accuracy comes from the *operations*. Accuracy of an operation = the maximal error it introduces. Smaller is better.
- Main error source: the exact result is not representable, so it is rounded. Rounding is needed when the result mantissa needs too many bits (p. 591).
  - Multiplication: the product has twice as many bits as either input.
  - Addition: if exponents differ, the operand with the smaller exponent is **aligned** by shifting its mantissa right one bit and incrementing its exponent, repeatedly, until the exponents match. The sum can then need more bits than the format has (pp. 591–592).
- **Worked alignment example** in the 5-bit format (p. 592): add 1.00B × 2^-2 (stored as S=0, E=00, M=01, i.e. a denormal) to 1.00B × 2^1 (S=0, E=10, M=00).
  - Shift the smaller operand right 3 places: 1.00B × 2^1 + 0.001B × 2^1.
  - Ideal sum: 1.001B × 2^1 (= 2.25). Needs 3 mantissa bits; the format has 2.
  - Best available: 1.01B × 2^1 (2.5) or 1.00B × 2^1 (2.0).
  - Error either way: 0.001B × 2^1, half the place value of the least significant mantissa place.
- **ULP (Units in the Last Place)** (p. 592): the place value of the last mantissa bit; equivalently the gap between adjacent representable numbers. The error above is **0.5 ULP**.
- Perfectly implemented arithmetic with correct rounding introduces at most **0.5 ULP**. The book states: "To our knowledge, this is the accuracy achieved by all CUDA devices today" (p. 592).
- More complex units (division, transcendental functions) are typically built on polynomial approximation. Too few terms gives an error above 0.5 ULP. Example: ideal result of an inversion is 1.00B × 2^1 but hardware returns 1.10B × 2^1; the error 0.10B is twice the last-place unit 0.01B, so **2 ULP**. Some early devices' hardware inversion had 2 ULP error. Recent CUDA devices' hardware arithmetic is, per the book, mathematically accurate and IEEE-conformant (ref. [4]) (p. 592).

### §A.5 Lower-precision formats (pp. 592–593)

This section is two paragraphs. Everything it says:

- Some applications inherently work on small input data types; the book lists **4-bit, 6-bit, 8-bit and 16-bit** (p. 592).
- Application areas named: medical imaging, remote sensing, radio astronomy, seismic analysis, "and other natural data" (p. 592).
- "More recently, deep learning models such as LLMs have been adapted to use smaller floating-point data types" (p. 592).
- The reasoning for *floating-point* rather than integer small types: these applications need only a few significant bits, but they benefit from the larger *range* of representable numbers that floating-point formats give. That motivates floating-point formats smaller than single precision (pp. 592–593).
- Performance argument: a smaller format reduces the number of bytes fetched per arithmetic operation, which improves the application's **compute intensity** (p. 593). (This connects to the FLOP-per-byte reasoning in Chapter 5; the appendix does not restate it.)
- Hardware: "Recent GPU architectures introduced hardware support for computing with 16-bit, 8-bit, and 4-bit floating point numbers" to improve performance and energy efficiency for applications that tolerate lower-precision arithmetic (ref. [3], the OCP MX specification) (p. 593).
- Device numbers `[device: NVIDIA H100]` (p. 593):
  - 8-bit floating point (**FP8**) arithmetic throughput using tensor cores: **3,958 TFLOPS**
  - single-precision throughput using tensor cores: **989 TFLOPS**
  - (ratio ≈ 4×; the ratio is our arithmetic, the book only calls it "a significant improvement")

What §A.5 does **not** contain (so later workers do not attribute it to the book):

- No bit layout (sign/exponent/mantissa widths) for half precision, FP8, or 4-bit floating point.
- No description of the OCP MX formats beyond the name and citation (footnote on p. 583, citation on p. 593).
- No integer quantization (int8/int4 with scale and zero-point), no fixed-point, no mixed-precision accumulation rules.
- No CUDA type names or intrinsics for half precision or FP8, and no code.
- No accuracy analysis (ULP, overflow, range) specific to the small formats. The tools from A.1–A.4 (range from exponent bits, precision from mantissa bits, saturation to ∞, ULP) apply by extension, but the book does not carry out that extension.

### §A.6 Algorithm considerations (pp. 593–594)

- Many numerical algorithms sum many values; the book's example is the dot product inside matrix multiplication (p. 593).
- Mathematically addition is associative, so order should not matter. With finite precision, order affects the accuracy of the result (p. 593).
- **Worked example** in the 5-bit format: sum 1.00B × 2^0 + 1.00B × 2^0 + 1.00B × 2^-2 + 1.00B × 2^-2 (i.e. 1 + 1 + 0.25 + 0.25; exact answer 2.5) (p. 593).
  - **Sequential order:** (1 + 1) = 1.00B × 2^1; then + 0.25 → still 1.00B × 2^1; then + 0.25 → still 1.00B × 2^1. Result **2.0**. In steps two and three the small operand "disappears" because it is too small relative to the large one.
  - **Parallel (pairwise) order:** (1 + 1) + (0.25 + 0.25) = 1.00B × 2^1 + 1.00B × 2^-1 = 1.01B × 2^1. Result **2.5**. The two small values first combine into something large enough to register.
- The pairwise scheme is a **reduction tree** as in Chapter 10 and relies on associativity. Sequential and parallel results differ, which surprises developers unfamiliar with FP (p. 593).
- The parallel version happened to be more accurate here; the book says one can construct a nearby case where the parallel version is *less* accurate. FP addition is not strictly associative (p. 594).
- What experienced developers do (p. 594): either confirm the variation is tolerable, or sort/group the data so the parallel algorithm gives the most accurate result.
- **Pre-sorting** (p. 594): sort ascending before reducing: 0.25 + 0.25 + 1 + 1. Grouping adjacent elements then puts values of similar magnitude together, so each group's additions are accurate. Signs must be taken into account when sorting. Where each thread sequentially reduces its own group, ascending order also helps that sequential pass. The book gives this as a reason sorting is common in massively parallel numerical codes.
- For something more robust the book points to **compensated summation (Kahan's Summation Algorithm)** (ref. [5]) without describing it (p. 594).

### §A.7 Linear solvers and numerical stability (pp. 594–599)

- Operation order matters even more for linear-system solvers: different inputs may need different operation orders to find a solution at all (p. 594).
- **Numerically stable** algorithm: can always find a suitable operation order and thus a solution whenever one exists, for any input. Otherwise **numerically unstable** (p. 594).
- Stability requirements can make efficient parallel algorithms harder to find (p. 594).
- **Gaussian elimination, example 1** (pp. 594–596, Fig. A.9 and A.10):
  - System: 3X + 5Y + 2Z = 19; 2X + 3Y + Z = 11; X + 2Y + 2Z = 11. Geometrically, the intersection point of three planes.
  - Step 1: divide each equation by its X coefficient (3, 2, 1).
  - Step 2: subtract Eq. 1 from Eq. 2 and Eq. 3 → X eliminated from both.
  - Step 3: divide Eq. 2 and Eq. 3 by their Y coefficients (−1/6 and 1/3).
  - Step 4: subtract Eq. 2 from Eq. 3 → Y eliminated from Eq. 3.
  - Step 5: divide Eq. 3 by its Z coefficient → Z = 3.
  - Steps 6–7 (**backward substitution**): Z into Eq. 2 gives Y = 2; Y and Z into Eq. 1 gives X = 1. Called backward because it runs from the last equation to the first.
  - Larger systems just repeat the eliminate step more times.
- **Matrix view** (p. 596, Fig. A.10): store coefficients and right-hand sides as rows of a matrix; equation operations become row operations.
- After elimination the coefficient matrix is **triangular**; the end goal is the **identity matrix** (ones on the diagonal). The book links this to multiplying by the inverse matrix (p. 597).
- **Parallel design sketched by the book** (p. 597), no code given:
  - One thread per matrix row; that thread does all the work for its row.
  - If the coefficients fit in shared memory, a single thread block does the elimination.
  - All threads iterate the steps together. After each division step: barrier synchronization. Then a subtraction step, after which one more thread goes idle (its row is finished until back substitution). Then another barrier so the next step sees updated data.
  - Expect reasonable speedup for systems with many variables.
- **Instability example** (p. 597): replace Eq. 1 with 5Y + 2Z = 16 (X coefficient zero). Step 1 cannot divide Eq. 1 by its X coefficient. The system is still solvable with the same solution (1, 2, 3), so the simple algorithm is numerically unstable.
- **Subtler case** (p. 597): a lead coefficient that is small and already carries rounding error from earlier steps. Dividing by it amplifies error into later steps. Hence instability is often measured as *growth of error over the steps*.
- **Pivoting** (pp. 597–598): pick a remaining equation whose lead-variable coefficient is nonzero and preferably large in absolute value, and swap it with the current top equation. Applied to the example: Eq. 1' = old Eq. 2 (2X + 3Y + Z = 11), Eq. 2' = old Eq. 1 (5Y + 2Z = 16), Eq. 3' unchanged. Elimination then proceeds (Fig. A.11).
  - Extra insight from Fig. A.11: some rows already lack the variable being eliminated (row 2 at step 1), so the thread owning that row skips the division (p. 598).
  - General rule as the book words it: choose the equation with the largest absolute coefficient among all lead variables, swap its row with the current top row "as well as swap the variable (column) with the current variable" (p. 598).
- **Cost of pivoting in parallel** (pp. 598–599):
  - Each thread owns a row, so pivoting needs to inspect, and maybe swap, data spread across threads.
  - Within one block with data in shared memory it is manageable: run a parallel **max reduction** across the block's threads, keeping control-flow divergence within warps under control.
  - Across multiple thread blocks, or multiple cluster nodes, global inspection is extremely expensive. This motivates **communication-avoiding algorithms** (ref. [6]).
  - Two approaches named: (1) **partial pivoting**, which the book describes as restricting swap candidates to a localized set of equations to bound inspection cost, at a slight loss of accuracy; (2) **randomization**, which researchers have shown tends to maintain high accuracy (p. 599).

### §A.8 Summary (p. 599)

- Recap of format and representable numbers as the basis of precision; denormals and why they matter; early CUDA devices lacked denormal support, later generations have it.
- Accuracy of FP operations matters for understanding "the potential lower accuracy of fast arithmetic operations implemented in the special function units".
- Main takeaway for parallel programmers: parallel algorithms can change result accuracy; sorting and other techniques can improve it.

## Figures

| Fig. | Book p. | Kind | What it shows | Interactive idea |
|---|---|---|---|---|
| A.1 | 585 | table | "Excess-3 encoding, sorted by excess-3 ordering." Three columns: 2's complement, Decimal value, Excess-3. Eight rows in increasing excess-3 code: (101, −3, 000), (110, −2, 001), (111, −1, 010), (000, 0, 011), (001, 1, 100), (010, 2, 101), (011, 3, 110), (100, "Reserved pattern", 111). Point: excess codes rise monotonically with the value, so unsigned comparison works; all-ones is reserved. | Slider for exponent width e; show two's-complement vs excess code side by side and let the learner compare two codes as unsigned integers. |
| A.2 | 586 | table | Two-column table mapping 3-bit patterns 000..111 to 0..7. Caption: representable numbers of a 3-bit unsigned integer format. | Trivial warm-up: click a bit pattern, highlight the value. |
| A.3 | 586 | diagram | Horizontal number line with tick labels −1, 0, 1, …, 9. Four-pointed star markers sit on 0 through 7; −1, 8 and 9 have labels but no star. Same caption as A.2. | Combine with A.2: toggling bits moves a marker on the line. |
| A.4 | 587 | table | Representable numbers of the 5-bit format under three conventions. Row keys: E (00, 01, 10, 11) and within each E, M (00, 01, 10, 11). Column groups: No-zero, Abrupt underflow, Denorm, each split into S=0 and S=1. E=00 rows: No-zero gives 2^-1, 2^-1+1·2^-3, 2^-1+2·2^-3, 2^-1+3·2^-3 (negated for S=1); Abrupt underflow gives 0 in all eight cells; Denorm gives 0, 1·2^-2, 2·2^-2, 3·2^-2 (S=1: 0, −1·2^-2, −2·2^-2, −3·2^-2). E=01 rows: 2^0, 2^0+1·2^-2, 2^0+2·2^-2, 2^0+3·2^-2 in all three conventions. E=10 rows: 2^1, 2^1+1·2^-1, 2^1+2·2^-1, 2^1+3·2^-1 in all three. E=11: one merged cell "Reserved pattern". Point: the conventions differ only in the E=00 block. | The central interactive: 5 toggle bits + a three-way convention switch; show the formula, the decimal value, and the position on the number line. |
| A.5 | 587 | diagram | Number line labelled 0, 1, 2, 3, 4 with star markers for the positive no-zero values: 0.5, 0.625, 0.75, 0.875, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5 (12 stars; none at 0 or 4). Shows three intervals of 4 points each, halving in width toward 0, and an empty gap between 0 and 0.5. The printed drawing is not to scale in the 0–1 region (the 0→1 segment is drawn shorter than 1→2). | Draw to scale; let learners add a mantissa or exponent bit (Exercises 1–2) and watch points/intervals change. |
| A.6 | 589 | diagram | Same axis (labels 0–4). Stars at 0, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5. Nothing between 0 and 1. Point: abrupt underflow gains 0 but doubles the gap. | Overlay with A.5 and A.7; highlight the gap width. |
| A.7 | 589 | diagram | Number line with labels 0, 1, 2 only. Stars at 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, then three more evenly spaced (2.5, 3, 3.5), plus a 13th star at the right end of the line (position ≈ 4; see "Unclear in the PDF"). Point: denormals fill the gap with uniform spacing equal to the next interval's. | Same overlay; show "rounding error for a chosen x" under each convention. |
| A.8 | 590 | table | Three columns: exponent, mantissa, meaning. Rows: (11…1, ≠ 0, NaN), (11…1, = 0, (−1)^S × ∞), (00…0, ≠ 0, denormalized), (00…0, = 0, 0). | Classifier quiz: given a 32-bit hex pattern, pick the category. |
| A.9 | 595 | diagram | Gaussian elimination + back substitution as eight equation panels joined by block arrows, each with a caption. **Original:** 3X+5Y+2Z=19; 2X+3Y+Z=11; X+2Y+2Z=11. **Step 1** (divide equation 1 by 3, equation 2 by 2): X+5/3Y+2/3Z=19/3; X+3/2Y+1/2Z=11/2; X+2Y+2Z=11. **Step 2** (subtract equation 1 from equation 2 and equation 3): row 1 unchanged; −1/6Y−1/6Z=−5/6; 1/3Y+4/3Z=14/3. **Step 3** (divide equation 2 by −1/6 and equation 3 by 1/3): Y+Z=5; Y+4Z=14. **Step 4** (subtract equation 2 from equation 3): 3Z=9. **Step 5** (divide equation 3 by 3, "Solution for Z!"): Z=3. **Step 6** (substitute Z into equation 2, "Solution for Y!"): Y=2. **Step 7** (substitute Y and Z into equation 1, "Solution for X!"): X=1, Y=2, Z=3. Layout: three panels on the top row, then two per row, last panel alone. | Stepper with one button per step; show which thread (row) is active and where barriers fall. |
| A.10 | 596 | diagram | Same eight panels as A.9 but as augmented-matrix rows (coefficients + right-hand side), blanks where a coefficient is 0. Original [3 5 2 19; 2 3 1 11; 1 2 2 11] drawn between vertical bars. Step 1: [1 5/3 2/3 19/3; 1 3/2 1/2 11/2; 1 2 2 11]. Step 2: [1 5/3 2/3 19/3; · −1/6 −1/6 −5/6; · 1/3 4/3 14/3]. Step 3: [… ; · 1 1 5; · 1 4 14]. Step 4: [… ; · 1 1 5; · · 3 9]. Step 5: [… ; · 1 1 5; · · 1 3]. Step 6: [1 5/3 2/3 19/3; · 1 · 2; · · 1 3]. Step 7: [1 · · 1; · 1 · 2; · · 1 3]. Captions say "row" where A.9 says "equation" (Step 5's caption still says "divide equation 3 by 3"). | Same stepper in matrix form; toggle equation/matrix views. |
| A.11 | 598 | diagram | Gaussian elimination with pivoting, matrix view, nine panels. **Original:** [· 5 2 16; 2 3 1 11; 1 2 2 11]. **Pivoting** (swap row 1 (Equation 1) with row 2 (Equation 2)): [2 3 1 11; · 5 2 16; 1 2 2 11]. **Step 1** (caption: "divide row 1 by 3, no need to divide row 2 or row 3"): [1 3/2 1/2 11/2; · 5 2 16; 1 2 2 11]. **Step 2** (subtract row 1 from row 3; column 1 of row 2 is already 0): [1 3/2 1/2 11/2; · 5 2 16; · 1/2 3/2 11/2]. **Step 3** (divide row 2 by 5 and row 3 by 1/2): [1 3/2 1/2 11/2; · 1 2/5 16/5; · 1 3 11]. **Step 4** (subtract row 2 from row 3): [… ; · 1 2/5 16/5; · · 13/5 39/5]. **Step 5** (divide row 3 by 13/5, "Solution for Z!"): rows 2–3 = [· 1 2/5 16/5; · · 1 3]; row 1 is printed as "1 5/3 2/3 19/3". **Step 6** (substitute Z into equation 2, "Solution for Y!"): [row 1 printed as "1 5/3 2/3 19/3"; · 1 · 2; · · 1 3]. **Step 7** (substitute Y and Z into equation 1, "Solution for X!"): [1 · · 1; · 1 · 2; · · 1 3]. Two apparent misprints in this figure are listed under "Unclear in the PDF". | Bug hunt: ask learners to find the two misprints. Also a "pivot or fail" stepper where the no-pivot path divides by zero. |

## Code listings

There are no code listings in Appendix A. All eleven figures are tables or diagrams. The parallel Gaussian-elimination kernel is described only in prose (p. 597), and the only tool-level detail is the NVCC `ftz` flag mention (p. 589).

Formulas worth quoting exactly:

- Eq. (A.1), p. 584: `value = (-1)^S · 1.M · 2^(E - bias)`
- Excess bias for an e-bit exponent, p. 585: `2^(e-1) - 1`
- General normalized value with n-bit E, p. 586: `(-1)^S · 1.M · 2^(E - (2^(n-1) - 1))`
- Denormal value when the n-bit E is 0, p. 589: `0.M · 2^(-2^(n-1) + 2)`

## Exercises

| # | Book p. | Gist | Type | Answer sketch |
|---|---|---|---|---|
| 1 | 599 | Redraw Fig. A.5 for a 6-bit format with 1-bit sign, **3-bit mantissa**, 2-bit exponent; explain what each extra mantissa bit does to the representable numbers. | compute / explain | (Our derivation, no-zero convention as in Fig. A.5.) Same three intervals [0.5, 1), [1, 2), [2, 4), now with 8 points each: 0.5 to 0.9375 in steps of 1/16; 1 to 1.875 in steps of 1/8; 2 to 3.75 in steps of 1/4 (24 positive values). Each extra mantissa bit doubles the points per interval and halves the spacing (max rounding error); the intervals and overall range are essentially unchanged, and the gap next to 0 stays, now 2^3 = 8× the neighbouring spacing. |
| 2 | 599 | Redraw Fig. A.5 for a 6-bit format with 1-bit sign, 2-bit mantissa, **3-bit exponent**; explain what each extra exponent bit does. | compute / explain | (Our derivation; excess-3 with 111 reserved, exponents −3..3.) Seven intervals starting at 1/8, 1/4, 1/2, 1, 2, 4, 8, each with 4 points (28 positive values); smallest 0.125, largest 8 + 3·2 = 14. Each extra exponent bit roughly doubles the number of intervals (2^e − 1: 3 → 7), extending the range toward both larger and smaller magnitudes; points per interval (relative precision) are unchanged. |
| 3 | 599 | An adder can only round toward zero (truncate); it keeps enough bits that rounding is the only error. Maximal ULP error for add? | compute | (Our derivation.) Truncation can discard anything up to, but not including, one full unit in the last place, so the maximal error is just under 1 ULP (bounded by 1 ULP), versus 0.5 ULP for round-to-nearest. |
| 4 | 600 | A student reduces a large float array, always sorted smallest to largest, to its sum; to avoid branch divergence he implements "the algorithm of Fig. A.4". Explain why accuracy can drop. | explain | The reference is broken: Fig. A.4 is a table of representable numbers, not an algorithm (see "Unclear in the PDF"). Our inference: the intended figure is the reduced-control-divergence reduction of Chapter 10 (Fig. 10.7/10.8), in which each thread adds the element half an array away to its own element. On an ascending array that pairs every small value with a much larger one, so the small values are lost in alignment/rounding (the "disappearing operand" effect of §A.6), and the benefit of sorting (similar magnitudes combined first) is thrown away. Treat as inferred, not confirmed by the text. |
| 5 | 600 | A sin() unit yields two more accurate mantissa bits per clock cycle and runs 9 cycles; remaining mantissa bits are filled with 0. Max ULP error for IEEE single precision, given the omitted "1." bit must also be generated? | compute | (Our derivation.) 9 × 2 = 18 accurate bits. Single precision has 24 effective mantissa bits (hidden 1 + 23 stored), so 6 low bits are forced to 0. Worst case the true low bits are all ones: error 2^6 − 1 = 63 ULP against the 24-bit result (strictly, just under 64 ULP against the exact value). |

## Dated or device-specific content

- Intel Pentium III and AMD Athlon named as 1990s microprocessors with supercomputer-class FP (p. 583).
- "LLM training and inference" cited as the driver of recent GPU FP throughput increases (p. 583).
- IEEE 754-2019 cited as the current revision (pp. 583, 600).
- OCP Microscaling Formats (MX) "have been adopted by recent GPUs" (footnote, p. 583); spec cited as v1.0, 2023 (p. 600).
- Abrupt underflow used by "some mini-computers in the 1980s" and still by some high-speed arithmetic units (p. 588).
- Early CUDA devices did not support denormalization; "virtually all recent generations of CUDA devices" do (pp. 589, 599).
- NVCC compiler flag `ftz` (true/false) controls denormal behaviour (p. 589). Exact spelling/semantics to be checked against current docs.
- "The current generation of GPU hardware does not support signaling NaN" (p. 591).
- 0.5 ULP is "the accuracy achieved by all CUDA devices today" for correctly rounded arithmetic (p. 592).
- Hardware inversion on "some early devices" had 2 ULP error; recent CUDA devices are IEEE-conformant (p. 592).
- "Recent GPU architectures" support 16-bit, 8-bit and 4-bit floating point in hardware (p. 593).
- `[device: NVIDIA H100]` FP8 with tensor cores: 3,958 TFLOPS; single precision with tensor cores: 989 TFLOPS (p. 593).
- Reference [4]: NVIDIA "Cuda floating-point reference", https://docs.nvidia.com/cuda/floating-point/index.html (p. 600).
- Summary's phrase "fast arithmetic operations implemented in the special function units" (p. 599) refers to hardware units not otherwise described in this appendix.

## Relevance to LLM-inference kernels

What the book itself says that bears on inference kernels:

- **Reduced-precision floating point (quantized/low-precision ops).** §A.5 (pp. 592–593) is the book's statement that LLMs have moved to smaller FP types, that the rationale is "few significant bits but wide range", that fewer bytes per operation raises compute intensity, and that GPUs have 16/8/4-bit FP hardware, with the H100 tensor-core figures above. That is the whole of it; the course will need outside sources for half-precision layout, FP8/MX details, and integer quantization.
- **Matmul/GEMM.** §A.6 uses the dot product in matrix multiplication as its motivating example of a long sum (p. 593). The order in which a kernel accumulates partial products (sequential per thread, tiled, tree-reduced) changes the low bits of the result.
- **Reductions.** Any tree reduction (Chapter 10) regroups the additions; sequential and parallel results legitimately differ (p. 593). A course test harness should compare against a reference with a tolerance, not bit-for-bit.
- **GPU exception model.** No signaling NaN on GPUs (p. 591): a NaN propagates quietly through a kernel and shows up only in the output.
- **Overflow behaviour.** FP saturates to ±∞, integer arithmetic wraps or is undefined (p. 590). The book makes this contrast explicitly.

Our inferences for the course (not stated in the book; flag as such when used):

- *Precision shrinks with mantissa bits, range with exponent bits* (A.2 observations 1–2). Applying the toy-format reasoning to a 16-bit or 8-bit float explains why low-precision accumulation loses small addends sooner: the "disappearing operand" of §A.6 happens whenever the exponent gap exceeds the mantissa width, which is a much smaller gap in a short format. This suggests accumulating in a wider type than the stored operands; the book does not say this.
- *Softmax stability.* The book never mentions softmax. The relevant pieces are: overflow saturates to ∞ (p. 590), ∞/∞ and ∞ − ∞ produce NaN (p. 591), NaNs propagate silently on GPUs (p. 591), and sums are order-sensitive (pp. 593–594). Together these motivate the usual subtract-the-max formulation and an explicit choice of summation order for the normalizer; that conclusion is ours.
- *Sampling (top-k/top-p, cumulative sums).* A cumulative probability sum is a long reduction of values with widely varying magnitudes; §A.6's sorting advice (add small values first) and the Kahan pointer (p. 594) are directly applicable. Top-p already sorts, though usually descending, which is the unfavourable order by the book's reasoning. Our inference.
- *Argmax/max reductions* are exact (no rounding), so order affects only tie-breaking and NaN handling; the book's only max-reduction remark here is its use for pivot search (p. 599).
- *Denormals.* Tiny activations or probabilities can land in the denormal range; the book's claim of "thousands of clock cycles" on moderate hardware (p. 589) and the `ftz` flag are the hooks for a note on flush-to-zero in inference builds. Whether current GPUs pay any such penalty is not stated.
- §A.7 (linear solvers, pivoting): little direct relevance to inference kernels. Useful mainly as an example of barrier-synchronized row-per-thread design and of why global data inspection is costly across blocks.

## Portability notes

- **IEEE-754 formats and special values** (A.1–A.3): presented as a cross-vendor standard (p. 583), so the concepts carry over. The book makes no statement about HIP or Metal conformance. `[unverified]`
- **NVCC `ftz` flag** (p. 589): CUDA-toolchain specific. Equivalent controls in HIP's compiler or in Metal Shading Language are `[unverified]`.
- **Denormal support, 0.5 ULP accuracy, IEEE-conformant division/transcendentals** (pp. 589, 592): the book asserts these for "CUDA devices" only. Behaviour on AMD GPUs (HIP) and Apple GPUs (Metal), including whether fast-math is on by default, is `[unverified]`.
- **No signaling NaN** (p. 591): stated for "the current generation of GPU hardware" generally, not just NVIDIA. HIP/Metal specifics `[unverified]`.
- **Low-precision FP hardware and tensor cores** (p. 593): tensor cores and the TFLOPS numbers are NVIDIA H100 specific. Whether and how HIP or Metal targets expose half-precision, 8-bit or 4-bit floating-point arithmetic, or any matrix-accelerator unit, is `[unverified]`. The book names no CUDA API or type for these formats, so there is nothing API-level to port from this appendix.
- **Reduction order** (A.6): the non-associativity argument is hardware-independent; any port that changes block size, tiling or reduction-tree shape should be expected to change low-order bits. (Follows from the book's argument; not a book statement about ports.)
- **Gaussian elimination sketch** (p. 597): relies on shared memory, thread blocks and barrier synchronization (`__syncthreads()` is not named here). Metal/HIP equivalents are `[unverified]` and are better covered from the chapters that introduce those features.

## Ideas for interactives and labs

- **Toy-float explorer (Fig. A.4/A.5/A.6/A.7).** Five bit toggles (S, E1 E0, M1 M0) and a convention switch (no-zero / abrupt underflow / denorm). Show the evaluated formula, the decimal value and a to-scale number line with all representable values. Add sliders for exponent and mantissa width to answer Exercises 1 and 2 visually.
- **Predict-then-reveal: where is the gap?** Show the no-zero number line and ask for the rounding error of 0.2; then switch conventions.
- **Excess-code comparator (Fig. A.1).** Give two exponents; learner compares the codes as unsigned integers and checks against the signed values.
- **Bit-pattern classifier (Fig. A.8).** Hex input for a 32-bit float; highlight S/E/M fields; classify as normal, denormal, zero, ±∞, quiet NaN, signaling NaN.
- **Alignment and ULP stepper (p. 592).** Animate the right-shift of the smaller operand, show the bits that fall off, the two candidate results and the error in ULP. Let learners change the format width.
- **Summation-order lab (pp. 593–594).** Input list plus a choice of order: left-to-right, pairwise tree, sorted ascending, sorted descending, Kahan (as an extension beyond the book). Run in the 5-bit toy format first (reproduce 2.0 vs 2.5), then in real float32/float16 on a GPU: sum 2^24 ones plus small values, compare against a float64 reference. Measurable outcome: absolute error per ordering.
- **"Make the parallel one worse" challenge.** The book says such a case exists (p. 594) without giving it; ask learners to construct four values where the tree sum is less accurate than the sequential sum in the toy format.
- **Gaussian-elimination stepper (Fig. A.9/A.10).** Toggle equation and matrix views; show one thread per row, barriers after each divide and subtract step, and threads retiring.
- **Pivot-or-fail (Fig. A.11).** Run example 2 without pivoting (division by zero), then with pivoting. Include the two misprints in the book's figure as a bug hunt.
- **Throughput card (p. 593).** H100 FP8 vs single-precision tensor-core TFLOPS as a small bar comparison tagged with the device.
- **Exercise 5 as a calculator.** Bits-per-cycle and cycle-count sliders → accurate bits → max ULP error.

## Cross-references

Made by this appendix:

- Chapter 10: the pairwise sum is a reduction tree "as discussed in Chapter 10" (p. 593).
- Section A.1 formula referenced from A.2 (p. 586); Fig. A.4's 5-bit format reused for the alignment example (p. 592) and the summation example (p. 593).
- Exercise 4 refers to "the algorithm of Fig. A.4" (p. 600); the target is unclear (see below).
- External references: [1] Goldberg 1991; [2] IEEE 754-2019; [3] OCP MX spec 2023; [4] NVIDIA CUDA floating-point reference; [5] Kahan 1965; [6] Ballard, Demmel, Holtz, Schwartz 2011 (p. 600).

Made by other chapters to this appendix (found by searching the extracted text):

- Chapter 8 (p. 185): lists double, single, half precision "and other representations (see Appendix A)".
- Chapter 10 (p. 224): FP addition is not strictly associative; readers are sent to Appendix A.
- Chapter 17 (p. 407): nondeterministic order of atomic updates on FP values "may cause numerical stability issues (see Appendix A)".
- Chapter 24 (p. 582): outline lists Appendix A.
- Related numbers elsewhere: Chapter 1 (p. 3) gives H100 throughput by precision including 16-bit half precision; Chapter 5 (p. 97) quotes the H100 tensor-core peak of 989 TFLOPS. Check those chapters' notes before combining figures.

## Unclear in the PDF

Possible errata and ambiguities. Unless marked "read directly", the judgement that something is an error is ours.

- **`ftz` flag wording (p. 589).** Text: denormalization can be turned "on or off ... by setting the NVCC compiler flag ftz = true (on) or ftz = false (off)". It is ambiguous whether "(on)" means denormalization on or the flag on; if the flag means flush-to-zero, `true` would disable denormals. Also the exact flag spelling is not given in command-line form. Verify against current NVCC docs before teaching.
- **Wrong figure reference for denormalization (p. 589).** "As shown in Fig. A.6, whenever E = 0, the mantissa is no longer assumed to be of the form 1.XX": Fig. A.6 is the abrupt-underflow line; the denorm material is in Fig. A.4 (right columns) and Fig. A.7.
- **Wrong figure reference for missing zero (p. 588).** "missing from the representable numbers in the no-zero column of Fig. A.5": the no-zero *column* is in Fig. A.4; Fig. A.5 is the number line.
- **Fig. A.7 extra star (p. 589).** A 13th star is drawn at the right end of the line at about 4.0, but 4 is not representable in the 5-bit format (E = 11 is reserved in Fig. A.4). Fig. A.5 and A.6 have no star there. Fig. A.7 also labels only 0, 1, 2 on the axis.
- **Fig. A.5 scale (p. 587).** The 0–1 segment is drawn shorter than the 1–2 segment, so the cluster 0.5–0.875 is not positioned to scale. Values are unambiguous from Fig. A.4.
- **"previous interval" wording (p. 589).** For denormals the exponent is "the same as the previous interval: 0 rather than −1", i.e. the next *larger* interval. The summary formula and Fig. A.4 confirm this reading.
- **"2^(2^3) = 256 times" more intervals (p. 590).** Read directly as 2 raised to 2^3. Three extra exponent bits multiply the number of exponent codes (hence intervals) by 2^3 = 8 (8-bit → 11-bit), not 256. Possibly intended differently; do not repeat the 256× claim without checking.
- **NaN count (p. 591).** "2^22 − 1 possible NaN bit patterns" in single precision "given that the most significant mantissa bit is reserved for signaling/quiet". The count appears to be per kind (and ignores sign); recorded as printed.
- **Reserved-pattern argument (p. 585).** "Having the pattern 111 as either even number or odd number would result in unbalanced number of positive and negative numbers": "even or odd" reads like a slip for "positive or negative".
- **Typos read directly:** "throughput of their floating-point throughput" (p. 583); "when view numbers" (p. 585); "0.5D" printed without subscript once (p. 586); "positive infinite" (p. 590); "Quite NaN's" (p. 591); "two floating-point values identical exponents" (missing "have", p. 592); "using dividing the rest" (p. 597).
- **Identical captions (p. 586).** Fig. A.2 and Fig. A.3 carry the same caption.
- **Fig. A.11 Step 1 caption (p. 598).** Says "divide row 1 by 3", but row 1 is [2 3 1 11] and the result shown, [1 3/2 1/2 11/2], is division by 2.
- **Fig. A.11 Steps 5 and 6, row 1 (p. 598).** Printed as "1 5/3 2/3 19/3" (the row 1 of Fig. A.10's example) instead of "1 3/2 1/2 11/2" as in Steps 1–4 of the same figure. The final answer (1, 2, 3) is still correct.
- **Fig. A.10 Step 5 caption (p. 596).** Says "divide equation 3 by 3" in the matrix view where the other captions say "row"; harmless.
- **Pivoting terminology (pp. 598–599).** The "general" rule includes swapping columns as well as rows, and "partial pivoting" is defined as restricting candidates to a localized set of equations. Recorded as the book states it; this may not match usage elsewhere, so check before presenting definitions.
- **Exercise 4 reference (p. 600).** "the algorithm of Fig. A.4" cannot be right: Fig. A.4 is a table. Probably a stale cross-reference to a reduction figure (our guess: Fig. 10.7/10.8).
- **Exercise 1 vs 2 field order (p. 599).** Both list fields as "sign, mantissa, exponent" (not the S, E, M storage order); only the widths matter.
- **A.5 terminology (pp. 592–593).** "smaller input data types (4-bit, 6-bit, 8-bit, and 16-bit)" in the first sentence vs hardware support for "16-bit, 8-bit, and 4-bit floating point" in the second paragraph; whether the first list means integer sample widths or float widths is not stated. The H100 "989 TFLOPS single-precision throughput using tensor cores" is as printed; compare with how Chapters 1 and 5 label H100 figures before reuse.
- **"special function units" (p. 599).** Mentioned only in the summary; not defined in this appendix.
