# Appendix B — Deep learning basics

- **Book pages:** 601–622  **PDF pages:** 625–646 (pdf = book + 24)
- **In one paragraph:** Background appendix (with contributions from Carl Pearson and Boris Ginsburg, p. 601) for Chapters 19 and 20. It builds from a single perceptron to multi-layer perceptrons, defines fully connected and convolutional layers, walks through training (error function, stochastic gradient descent, chain rule, learning rate, batches), and then gives sequential C code for LeNet-5-style layers: convolution forward, subsampling forward, and the two convolution gradients (`dE/dX`, `dE/dW`). There is no CUDA code and there are no exercises. The course's target reader already knows this material, so these notes are short on concepts and precise on three things later workers need: the book's own notation and non-standard definitions, every figure and listing, and the printing errors in the listings (several, see "Unclear in the PDF"). Anything labelled **(derived)** is ours, not the book's.

## Section map
| § | Title | Book pp. | What it teaches (one or two lines) |
|---|---|---|---|
| (intro) | untitled opening | 601 | ML defined; task types (classification, regression, transcription, translation, embedding); the appendix covers the kernels behind neural-network classification. |
| B.1 | Classifiers | 602–605 | Classifier as `y = f(x, θ)`; perceptron `y = sign(W·x + b)`; linear separability; multi-layer perceptrons (unnumbered subsection "Multi-layer Classifiers", p. 603). |
| B.2 | Fully connected and convolutional layers | 605–606 | FC layer = `m × n` weight matrix applied by matrix-vector multiplication; convolutional layers as patch-local, weight-shared classifiers. |
| B.2.1 | Training models | 606–612 | Supervised training, error function, SGD, epoch, back-propagation, chain rule, sigmoid, learning rate, batch, mini-batch, multi-layer training, feed-forward networks. |
| B.3 | Convolutional neural networks | 612–613 | CNN history (near-verbatim duplicate of §19.1's history). |
| B.4 | CNN inference and training: sequential implementation | 613–616 | LeNet-5 layers; sequential C for convolution forward and subsampling forward. |
| B.4.1 | CNN training | 616–621 | Gradient flow through layers; FC gradients in matrix form; convolution `dE/dX` and `dE/dW` formulas and C code; weight update. |
| B.5 | Summary | 621 | Recap. |
| — | References | 621–622 | 9 references. |

## Concepts

### Opening (p. 601)
- "Machine learning" attributed to Arthur Samuel (IBM, 1959). Task examples: classification, regression, transcription, translation, embedding.
- Scope statement: introduce the computation kernels of the neural-network approach to classification; understanding these lets the reader build kernels for other tasks.

### §B.1 Classifiers (pp. 602–605)
- Classifier: `f : Rⁿ → {1, 2, …, k}` (Eq. B.1); parameterised form `y = f(x, θ)` (Eq. B.2). `θ` is called the **model** (p. 602).
- Perceptron: `y = sign(W · x + b)` (Eq. B.3). `sign` returns 1, 0 or −1 for positive, zero, negative input. It is called the **activation function**; it supplies the non-linearity. `W` holds the **weights**, `b` is the **bias**. Perceptrons are **linear classifiers** (p. 602).
- 2D example (p. 603): `(w1, w2) = (2, 3)`, `b = −6` gives the line `2x1 + 3x2 − 6 = 0` through `(3, 0)` and `(0, 2)`. **Inference** = evaluating the classifier: for input `(5, −1)`, `y = sign(2·5 + 3·(−1) − 6) = sign(1) = 1` (Eq. B.4).
- Limitation (pp. 603–604): one line cannot isolate the upper-right quadrant of the unit square (`x1 > 0.5` and `x2 > 0.5`). The line `2x1 + 2x2 − 3 = 0` misclassifies e.g. `(0.55, 0.65)`.
- **Multi-layer perceptron (MLP)** (p. 604): outputs of one layer are inputs of the next. Observation the book leans on: first-layer inputs take infinitely many values, but its outputs take only a few (−1, 0, 1 each).
- Two-layer solution (pp. 604–605): layer 1 has `y1 = sign(x1 − 0.5)` and `y2 = sign(x2 − 0.5)`, giving 9 possible `(y1, y2)` pairs; only `(1, 1)` is the target class; layer 2 separates it with `2y1 + 2y2 − 3 = 0`. `(0.55, 0.65)` now gives `y1 = 1`, `y2 = 1`, `z = 1`.
- Need for more layers (p. 605, Fig. B.3): for an XOR-like pattern the target points land on `(−1, −1)` or `(1, 1)` in `y`-space, which one line cannot separate from the rest. Add a second layer-2 perceptron `z2 = sign(−2y1 − 2y2 − 3)` and a third layer with the line `z1 + z2 + 1 = 0`. (derived check: `(1,1) → (1,−1)`, `(−1,−1) → (−1,1)`, the other seven points `→ (−1,−1)`; the line separates them.)

### §B.2 Fully connected and convolutional layers (pp. 605–606)
- **Fully connected layer (FC)**: every one of `m` outputs depends on all `n` inputs. Weights form an `m × n` matrix `W`, one row per output. Evaluating the layer is a **matrix-vector multiplication** (p. 606).
- The book notes FC layers are core components of many networks "including large language models" and says their GPU implementations are studied further (p. 606; not in this appendix).
- Cost argument (p. 606): `m × n` weights; for images `n` is in the millions and `m` hundreds or more.
- **Convolutional layer** (p. 606): each classifier sees only a patch and all classifiers share weights; the result is an **output feature map**; computationally a 2D convolution. Several weight sets give several output maps.

### §B.2.1 Training models (pp. 606–612)
- **Supervised training** assumed; semi-supervised and reinforcement learning only mentioned (p. 606).
- Labelled points give equations such as `sign(w1 + 5w2 + b) = 1` (Eq. B.5), which are non-linear and do not determine a unique solution (the book says "ill-defined problem"), so training is iterative (p. 607).
- **Error function** / **cost function**: `E = (y − t)² / 2` (Eq. B.6), `t` the label. The 1/2 simplifies the derivative (p. 607). Example: label 1, output −1 gives `E = 2` (p. 608).
- **Stochastic gradient descent** as described (pp. 607–608): repeatedly run permutations of the data, evolve parameters, stop when they change less than a threshold; restart with a new initial guess (or a deeper model) if the total cost is still too high.
- **Epoch**: one pass; data are shuffled first (p. 608). Back-propagation is described as "activated" when the error exceeds a threshold.
- **Back-propagation**: find each parameter's contribution to the error via partial derivatives; decrease a parameter whose increase raises the error (p. 608).
- **Chain rule**: `∂E/∂w1 = (∂E/∂y)(∂y/∂w1)` (Eq. B.7); `∂E/∂y = y − t` (Eq. B.8).
- **Sigmoid, as this book defines it** (p. 609): `s = (1 − e^(−x)) / (1 + e^(−x))`, a smooth stand-in for `sign` with range −1 to 1. This is not the 0-to-1 logistic function (derived: it equals `tanh(x/2)`). Derivative (Eq. B.9): `2e^(−k) / (1 + e^(−k))²`.
- With `k = w1x1 + w2x2 + b` (Eq. B.13): `∂E/∂w1 = (y − t) · 2e^(−k)/(1 + e^(−k))² · x1` (Eq. B.10); same with `x2` for `w2` (Eq. B.11); without the `x` factor for `b` (Eq. B.12). Running example input is `(5, 2)`.
- **Learning rate** `ε` (p. 610): updates `w1 ← w1 − ε ∂E/∂w1`, `w2 ← w2 − ε ∂E/∂w2`, `b ← b − ε ∂E/∂b` (Eqs. B.14–B.16); generic `θ ← θ − ε ∂E/∂θ` (Eq. B.17). Larger `ε` is faster but risks instability.
- **Batch** and **mini-batch, as this book defines them** (pp. 610–611): a batch is a slice of the training set small enough to fit in device memory; a mini-batch is a further slice of a batch assigned to one of several computing devices. Each device runs forward propagation on its mini-batch and back-propagates when its total error is too big; the devices then sum their parameter changes before the next batch. This differs from common usage; keep the book's wording when quoting it.
- **Training multi-layer classifiers** (p. 611): the previous layer's `∂E/∂y` is the current layer's `∂E/∂x`, and `∂E/∂x = (∂E/∂y)(∂y/∂x)`. For the perceptron: `∂E/∂x1 = (y − t) · 2e^(−k)/(1 + e^(−k))² · w1` (Eq. B.18), and with `w2` for `x2` (Eq. B.19).
- **Feed-forward network** (p. 612): no connection from a later layer's output to an earlier layer's input, so back-propagation is a simple backwards sweep.

### §B.3 Convolutional neural networks (pp. 612–613)
- Same history and numbers as §19.1 (late-1980s invention, 2006 revival, 2012 ILSVRC result: 60 million parameters, 650,000 neurons, 1.2 million images, one week on two GPUs, 15.3% vs 26.2%). Adds the claim that CNNs are easier to train and generalise better than other deep learning models (p. 612).

### §B.4 CNN inference and training: sequential implementation (pp. 613–616)
- Same LeNet-5 walk-through as §19.1, with slightly different vocabulary: a set of weights for one (input map, output map) pair is a **filter bank**; the three 2D filter banks feeding one output map can be viewed as one **3D filter bank**; `n` input maps and `m` output maps need `n·m` 2D filter banks, or `m` 3D ones (pp. 613–614).
- Eq. B.20 (p. 614) is the same worked sum as Eq. 19.1 (= 14).
- Shapes (p. 615): `X[C, H, W]`, `Y[M, H−K+1, W−K+1]`, `W[M, C, K, K]`. Footnote 2: `W` names both the image width and the weight array.
- **Subsampling (pooling) layer** (p. 616): reduces map size by combining pixels so later convolutions see larger-scale patterns without larger masks. LeNet S2: 6 maps of 28×28 → 6 maps of 14×14; each output pixel is the average of a 2×2 neighbourhood; a per-map bias `b[m]` is added and the result goes through sigmoid. Other combining rules are possible. Number of maps is unchanged; rows and columns halve.
- The book repeats that it focuses CUDA work on convolution layers because they dominate CNN training and inference time; FC and Gaussian-connection layers are left to the literature (p. 616).

### §B.4.1 CNN training (pp. 616–621)
- Labels give the target output vector: 1.0 for the correct digit, 0.0 elsewhere (p. 617).
- Gradient flow (p. 617): each layer receives `∂E/∂y` for its outputs (which is the next layer's `∂E/∂x`), computes `∂E/∂x` for its inputs, and, if it has weights, `∂E/∂w`.
- FC layer `y = w · x` (Eq. B.21, p. 618): `∂E/∂x = wᵀ ∂E/∂y` (matrix-vector product) and `∂E/∂w = (∂E/∂y) xᵀ` (outer product).
- Convolution, gradient to inputs (Eq. B.22, p. 618), as printed: `∂E/∂x(c, h, w) = Σ_{m=0}^{M−1} Σ_{p=0}^{K−1} Σ_{q=0}^{K−1} ∂E/∂y(m, h−p, w−q) ∗ w(m, c, k−p, k−q)`. The book calls this a "backward convolution" and says it is equivalent to a convolution with a "transposed filter bank `wᵀ`" (p. 619). See "Unclear in the PDF" about the weight indices.
- Indexing intuition (p. 619, Fig. B.9, 3×3 filters): `x[h,w]` contributed to nine outputs, from `y[h−2, w−2]` (via `w[2,2]`) to `y[h,w]` (via `w[0,0]`), so its gradient collects from those nine.
- Convolution, gradient to weights (Eq. B.23, p. 619): `∂E/∂W(m, c, p, q) = Σ_{h=0}^{H_out−1} Σ_{w=0}^{W_out−1} X(c, h+p, w+q) ∗ ∂E/∂Y(m, h, w)`.
- Roles (p. 621): `∂E/∂x` propagates to the previous layer; `∂E/∂w` adjusts the current layer.
- Update (p. 621): `w ← w − ε ∗ ∂E/∂w`; `ε` starts at an empirically chosen value and is reduced over epochs by a user-defined rule.

### §B.5 Summary (p. 621)
- States the appendix is background for Chapters 19 and 20, and recaps the perceptron → MLP → CNN path and the gradient computations.

## Figures
| Fig. | Book p. | Kind | What it shows | Interactive idea |
|---|---|---|---|---|
| B.1 | 603 | diagram | Axes `x1` (right) and `x2` (up). A straight line slopes from upper left to lower right, crossing the `x2` axis at a point labelled `(0, −b/w2)` and the `x1` axis at `(−b/w1, 0)`, and is labelled `w1 x1 + w2 x2 + b = 0`. About a dozen light-blue dots lie below-left of the line in the region labelled `w1 x1 + w2 x2 + b < 0`; about eight dark-blue dots lie above-right in the region labelled `> 0`. To the right: `y = sign((w1, w2) • (x1, x2) + b) = sign(w1 x1 + w2 x2 + b)`. Caption: perceptron linear classifier with 2D input. | Drag `w1`, `w2`, `b`; watch the line and classifications change. Low priority for this audience. |
| B.2 | 604 | diagram | (a) Top: unit square in `x1`–`x2` with corners labelled `(0,1)` and `(1,0)`; dark dots over most of it, light dots near the upper-right corner (the corner `(1,1)` is light); one diagonal line slicing off the upper-right corner. Bottom: dataflow graph of one perceptron: `x1` and `x2` each enter a multiply node (circle marked ×) with weight 2 (one fed from above, one from below); both feed an add node (+) with bias −3; then an activation node (circle marked s); output `y`. (b) Top middle: two copies of the same square, the upper with a vertical line at `x1 = 0.5`, the lower with a horizontal line at `x2 = 0.5`. Top right: `y1`–`y2` plane with a square carrying nine dots (corners, edge midpoints, centre at the origin); the corner labelled `(1,1)` is light, the other eight dark; a diagonal line separates `(1,1)` from the rest. Bottom: two-layer network. Layer 1, upper perceptron: `x1·1 + x2·0 − 0.5 → s → y1`; lower perceptron: `x1·0 + x2·1 − 0.5 → s → y2`. Layer 2: `y1·2 + y2·2 − 3 → s → z`. Labels "Layer 1", "Layer 2". Caption: a multi-layer perceptron example. | Toggle between one-line and two-layer classifiers on the same point set; test point `(0.55, 0.65)`. |
| B.3 | 605 | diagram | (a) Unit square in `x1`–`x2` (labels `(0,1)`, `(1,0)`) with an XOR-like pattern: light dots in the upper-right and lower-left quadrants, dark dots in the upper-left and lower-right. (b) `y1`–`y2` plane with the nine-dot square; `(1,1)` and `(−1,−1)` (both labelled) are light, the rest dark; two parallel diagonal lines, one cutting off each light corner. (c) `z1`–`z2` plane with a square; corner label `(1,1)`; light dots at the upper-left and lower-right corners, a dark dot at the lower-left; one diagonal line separating the dark dot from the two light ones. Blue arrows run from the two light corners in (b) to the two light dots in (c) (from `(−1,−1)` to the upper-left, from `(1,1)` to the lower-right). Caption: need for perceptrons with more than two layers. | Step through layers and watch points collapse onto the nine, then three, positions. |
| B.4 | 613 | diagram | Identical to Fig. 19.1: LeNet-5, INPUT 32×32 (letter "A") → C1 6@28×28 → S2 6@14×14 → C3 16@10×10 → S4 16@5×5 → C5 120 → F6 84 → OUTPUT 10, with operation labels Convolutions, Subsampling, Convolutions, Subsampling, Full connection, Full connection, Gaussian connections. Same caption. | Reuse the Chapter 19 layer strip. |
| B.5 | 614 | diagram | Identical to Fig. 19.2: (a) green "Convolution Layer" circle with a "Filters (weights)" stack above (six downward arrows), "Input Features" stack on the left (three arrows in), "Output Features" stack on the right (two arrows out). (b) Three 3×3 input maps `[1 2 0; 1 1 3; 0 2 2]`, `[0 2 1; 0 3 2; 1 1 0]`, `[1 2 1; 0 1 3; 3 3 2]`; six 2×2 filters (green `[1 1; 2 2]`, `[1 1; 1 1]`, `[0 1; 1 0]`; blue `[1 0; 0 1]`, `[2 1; 2 1]`, `[1 2; 2 0]`); outputs `[14 20; 15 24]` and `[12 24; 17 26]`; top-left patches and the values 14 and 12 circled. | Reuse the Chapter 19 hover figure. |
| B.6 | 615 | code | Sequential `convLayer_forward`; same as Fig. 19.3 except the filter array is named `W` (see listings). | Diff against Fig. 19.3. |
| B.7 | 617 | code | Sequential `subsamplingLayer_forward` with bias and sigmoid (see listings). Caption adds that the activation would sit in the convolutional layer if no subsampling layer followed it. | Bug hunt on the printed typos. |
| B.8 | 618 | diagram | Two copies of the Fig. B.5(a) block picture with a green circle labelled "Layer". (a) `∂E/∂w`: three arrows from "Input Features" into the layer, two arrows from "Output Features" (right) pointing left into the layer, and six arrows pointing up from the layer to the "Filters (weights)" stack. (b) `∂E/∂x`: six arrows from the filters down into the layer, two arrows from "Output Features" into the layer, and three arrows from the layer pointing left to "Input Features". Point: what each gradient computation consumes and produces. | Animated arrows showing forward then the two backward flows. |
| B.9 | 619 | diagram | Left: 7×7 grid, rows labelled `h=0` … `h=H−1`, columns `w=0` … `w=W−1`. The cell at row 3, column 3 (0-based) is dark blue and labelled `x[h,w]`. Two 3×3 patches are outlined thick: one with `x[h,w]` as its bottom-right cell, one with `x[h,w]` as its top-left cell. Right: 5×5 grid, rows `h=0` … `h=H−3`, columns `w=0` … `w=W−3`, with a 3×3 block shaded light blue (rows 1–3, columns 1–3); its top-left cell is labelled `y[h−2,w−2]`, its bottom-right `y[h,w]`. An arrow from the upper-left patch passes the label `x[h,w] * w[2,2]` and ends at `y[h−2,w−2]`; an arrow from the lower-right patch passes `x[h,w] * w[0,0]` and ends at `y[h,w]`. Caption mentions parts (a) `∂E/∂w` and (b) `∂E/∂x`, but the figure has no (a)/(b) labels and only illustrates the `∂E/∂x` indexing. | Click an `x` cell to shade the outputs it influenced and show which weight links each. |
| B.10 | 620 | code | Sequential `convLayer_backward_x_grad` (see listings). | Bug hunt. |
| B.11 | 620 | code | Sequential `convLayer_backward_w_grad` (see listings). | Compare loop structure with the forward pass. |

## Code listings

### Fig. B.6 — `convLayer_forward` (p. 615)
- Purpose: sequential forward path of a convolutional layer, single sample; the Appendix twin of Fig. 19.3.
- Signature as printed: `void convLayer_forward(int M, int C, int H, int W, int K, float* X, float* W, float* Y)`. `W` appears twice (width and weights), acknowledged by footnote 2; it would not compile as printed. Fig. 19.3 uses `F` for the weights.
- Walkthrough: lines 2–3 compute `H_out = H - K + 1`, `W_out = W - K + 1`; lines 4–6 loop over `m`, `h`, `w`; line 7 zeroes `Y[m, h, w]`; lines 8–10 loop over `c`, `p`, `q`.
- Key line: `Y[m, h, w] += X[c, h + p, w + q] * W[m, c, p, q];`
- Multi-dimensional indexing is pseudo-code. No activation.

### Fig. B.7 — `subsamplingLayer_forward` (p. 617)
- Purpose: sequential average-pooling with per-map bias and sigmoid.
- Signature: `void subsamplingLayer_forward(int M, int H, int W, int K, float* Y, float* S)`; `Y` is the input (the convolution output), `S` the subsampled output.
- Walkthrough:
  - Lines 2–4: loops `m < M`, `h < H/K`, `w < W/K`; a comment says the code assumes `H` and `W` are multiples of `K`.
  - Line 5: zero the output element (printed as `S[m, x, y] = 0.;`).
  - Lines 6–8: loops `p, q < K`; `S[m, h, w] += Y[m, K*h + p, K*w+ q] /(K*K);`
  - Line 10: `S[m, h, w] = sigmoid(S[m, h, w] + b[m]);`
- Needs care: line 5 uses undefined `x`, `y` where `h`, `w` are meant; `b` is used but is not a parameter; the division by `K*K` is applied per term.

### Fig. B.10 — `convLayer_backward_x_grad` (p. 620)
- Purpose: sequential `∂E/∂x` for a convolutional layer, given `∂E/∂y` of the same layer.
- Signature: `void convLayer_backward_x_grad(int M, int C, int H_in, int W_in, int K, float* dE_dY, float* W, float* dE_dX)`. The text says `dE_dX` is pre-allocated "in the device memory" (p. 619).
- Walkthrough:
  - Lines 2–3: `H_out = H_in - K + 1`, `W_out = W_in - K + 1`.
  - Lines 4–7: zero `dE_dX[c, h, w]` over `C × H_in × W_in`.
  - Lines 8–13: loops over `m`, `h`, `w`, `c`, `p`, `q`.
  - Line 14: range test that the output index is valid.
  - Line 15: accumulate.
- Lines as printed (several apparent errors, listed under "Unclear in the PDF"):
  - `for(int h = 0; h < H-1; h++)` and `for(int w = 0; w < W-1; w++)`
  - `if(h-p >= 0 && w-p >= 0 && h-p < H_out and w-p < W_OUT)`
  - `dE_dX[c, h, w] += dE_dY[m, h-p, w-p] * W[m, c, k-p, k-q];`
- (derived) Form consistent with the forward code of Fig. B.6: loop `h < H_in`, `w < W_in`; test `h-p` against `[0, H_out)` and `w-q` against `[0, W_out)`; accumulate `dE_dY[m, h-p, w-q] * W[m, c, p, q]`. Reason: forward has `Y[m, h', w'] += X[c, h'+p, w'+q] * W[m, c, p, q]`, so `X[c, h, w]` reached `Y[m, h-p, w-q]` through `W[m, c, p, q]`, which also matches the Fig. B.9 labels (`w[2,2]` ↔ `y[h−2, w−2]`).

### Fig. B.11 — `convLayer_backward_w_grad` (p. 620)
- Purpose: sequential `∂E/∂w` for a convolutional layer.
- Signature: `void convLayer_backward_w_grad(int M, int C, int H, int W, int K, float* dE_dY, float* X, float* dE_dW)`.
- Walkthrough:
  - Lines 2–3: `H_out`, `W_out`.
  - Lines 4–8: zero `dE_dW[m, c, p, q]`.
  - Lines 9–14: loops over `m`, `h < H_out`, `w < W_out`, `c`, `p`, `q`.
  - Line 15 as printed: `dE_dW[m, c, p, q] += X[c, h+p, w+q] * dE_dY[m, c, h, w];`
- Needs care: `dE_dY` is indexed with four subscripts here but with three (`m, h, w`) in Eq. B.23 and in Fig. B.10; the extra `c` looks like an error. Every `(m, c, p, q)` accumulator is a sum over all output pixels, which is a reduction if parallelised.

## Exercises
| # | Book p. | Gist | Type | Answer sketch |
|---|---|---|---|---|
| — | — | Appendix B has no exercises section. | — | — |

The text contains two "reader should verify" prompts: the `z1`–`z2` mapping of Fig. B.3 (p. 605; verified above) and the general statement that the generic update Eq. B.17 applies to each parameter (p. 610).

## Dated or device-specific content
- No GPU model, CUDA version or API name appears. There is no CUDA code in the appendix; one sentence mentions "device memory" for `dE_dX` (p. 619).
- Historical figures (p. 612), same as Chapter 19: 2012 ILSVRC network, about 60 million parameters, 650,000 neurons, 1.2 million ImageNet images, 1000 classes, one week on two GPUs (model not named), 15.3% vs 26.2% test error; "ten times faster than traditional CPUs" for the circa-2006 work.
- cuda-convnet cited at `https://code.google.com/p/cuda-convnet/`, 2014 (reference [8], p. 622).
- "Machine learning" dated to Arthur Samuel, 1959 (p. 601); perceptron reference dated 1957 (p. 621).
- Activation functions covered are only `sign` and the book's ±1 sigmoid; LeNet-era choices (average pooling with bias and sigmoid, Gaussian connections) are presented as the example design.

## Relevance to LLM-inference kernels
- **Matmul / GEMV:** the one directly useful statement is that a fully connected layer is a matrix-vector multiplication with an `m × n` weight matrix, and that such layers are core to LLMs (p. 606). Eq. B.21 gives the matrix forms of the FC gradients, which matter for training, not inference.
- **Attention, quantised ops, sampling (softmax, top-k/top-p, argmax):** little direct relevance. Softmax, ReLU, normalisation and attention do not appear. Chapter 20 footnote 4 (p. 481) sends readers to Appendix B "for the definition of ReLU and activation", but Appendix B defines only `sign` and sigmoid; a text search finds no "ReLU" in the appendix.
- **Convolution gradients (Figs. B.10, B.11):** useful only as extra kernel-writing practice (gather form vs scatter form, reductions), not for inference.
- Recommendation for the course: treat Appendix B as a notation reference (filter bank, feature map, `X`/`Y`/`W` shapes, the book's sigmoid) and as a source of bug-hunt material; do not build lessons on its ML content for this audience.

## Portability notes
- The appendix contains no CUDA-specific API, qualifier or built-in; all listings are sequential C pseudo-code with multi-dimensional index notation that must be linearised by hand in any language.
- The book states nothing about HIP or Metal here. Any port of GPU versions of these routines (asked for by Chapter 19 Exercises 1 and 3) would raise the same points as the Chapter 19 kernels `[unverified]`.

## Ideas for interactives and labs
- **Errata bug hunt (Figs. B.7, B.10, B.11).** Present the listings exactly as printed and ask the learner to find the problems (undefined `x`, `y`, `b`; `H`/`W` vs `H_in`/`W_in`; `w-p` vs `w-q`; `and`; `W_OUT`; `k-p, k-q`; extra `c` subscript). Then check a corrected version against numerical differentiation of Fig. B.6. This suits readers fluent in autograd who are new to hand-written kernels.
- **Gather vs scatter (Fig. B.9, Fig. B.10).** Show the same `∂E/∂x` computed two ways: scatter from each `y` element into nine `x` gradients (needs atomics when parallel) and gather into each `x` element from nine `y` gradients (one writer per element). Ties to Chapter 19's remark about atomics for the inner loops (p. 460).
- **Reduction spotting (Fig. B.11).** Identify that each `dE_dW[m, c, p, q]` is a sum over `H_out × W_out` (and over the batch) and choose a thread mapping.
- **Lab for Chapter 19 Exercise 1.** Port Fig. B.7 to a one-thread-per-output-pixel kernel; verify against the fixed sequential version.
- **Notation card.** One-screen glossary mapping Appendix B names (`W[M, C, K, K]`, filter bank, 3D filter bank) to Chapter 19 names (`F`, filter) and cuDNN names (Table 19.1).
- The perceptron and MLP figures (B.1–B.3) could become a single optional widget (drag the lines, watch points collapse onto the nine `(y1, y2)` positions); low priority for this reader.

## Cross-references
- Made by the appendix: Chapter 7 for the definition of convolution filters (p. 613); Fig. 19.1 for the LeNet-5 subsampling example with `K = 2` (p. 616); Chapters 19 and 20 as the chapters it supports (p. 621); "in preparation for their CUDA implementation" (p. 613) points at Chapter 19.
- Incoming: Chapter 19 refers to Appendix B on pp. 454, 455, 456, 474, 475 (background, perceptrons, sigmoid, subsampling, backward pass). Chapter 20 p. 481 refers to it for the idea that deeper models give more levels of classifiers, and in footnote 4 for "ReLU and activation". Listed in the back-matter outline at p. 582.
- Duplicated material: Fig. B.4 = Fig. 19.1; Fig. B.5 = Fig. 19.2; Fig. B.6 ≈ Fig. 19.3 (weights named `W` instead of `F`); Eq. B.20 = Eq. 19.1; §B.3 history ≈ §19.1 history.

## Unclear in the PDF
- p. 609: "Based on calculus manipulation, `∂y/∂w1` is simply `x1`". From the surrounding chain-rule expression this should be `∂k/∂w1 = x1`. Likely erratum.
- p. 609: the function called "sigmoid" is `(1 − e^(−x))/(1 + e^(−x))`, range −1 to 1. Internally consistent (Eq. B.9 is its derivative) but not the standard logistic sigmoid; flag it wherever the course quotes Eqs. B.9–B.12 or B.18–B.19.
- p. 605: "We show in Fig. B.2(b) that we can add another line by adding another perceptron in the second layer." The figure with two lines in `y`-space is Fig. B.3(b). Likely erratum.
- pp. 603–605: shading convention flips. In Fig. B.1 dark dots are class 1; in Figs. B.2 and B.3 the light dots are the target category 1. p. 604 also says points are "misclassified into category −1 (darkly shaded)".
- p. 602 vs p. 603: Eq. B.1 maps to labels `1 … k`, while the perceptron outputs −1, 0, 1.
- pp. 610–611: "batch" and "mini-batch" are defined in a non-standard way (see Concepts). The Batch paragraph says back-propagation "is triggered for the mini-batch" before mini-batches are introduced; "min-batch" is a typo on p. 611. The description that back-propagation runs only when a batch's total error is "too large" is recorded as printed.
- p. 615, Fig. B.6: two parameters named `W` (acknowledged by footnote 2).
- p. 616: text says "sampling layers" for subsampling layers; cites "Fig. 19.1" for the LeNet-5 example where Fig. B.4 is the local copy; cites "Section B.2.1" for the sigmoid activation; says labels give "the correct letter" (digits are meant).
- p. 617, Fig. B.7: `S[m, x, y] = 0.;` uses undefined `x`, `y`; `b[m]` is not a parameter.
- p. 618, Eq. B.22 and p. 620, Fig. B.10 line 15: weight indexed as `(k−p, k−q)` with a lowercase `k` that is not defined (the size parameter is `K`). Paired with `∂E/∂y(m, h−p, w−q)`, the forward definition implies the weight index should be `(p, q)`; `(K−1−p, K−1−q)` would be right only if the `y` index were written in the flipped (true-convolution) form. As printed the two indexings are mixed. Suspected erratum.
- p. 619: "equivalent to a convolution with a transposed filter bank `wᵀ`". For a convolution this means a spatially flipped filter rather than a matrix transpose; the book's wording is kept.
- p. 620, Fig. B.10: loops `h < H-1`, `w < W-1` use `H` and `W`, which are not declared in this function (`W` is the weight pointer; the sizes are `H_in`, `W_in`), and the `-1` would skip the last row and column. Line 14 uses `w-p` where `w-q` is expected, the word `and` instead of `&&`, and `W_OUT` instead of `W_out`. Line 15 uses `w-p` again. Suspected errata.
- p. 620, Fig. B.11 line 15: `dE_dY[m, c, h, w]` has an extra `c` subscript compared with Eq. B.23 and Fig. B.10. Suspected erratum.
- p. 619, Fig. B.9: caption refers to (a) `∂E/∂w` and (b) `∂E/∂x`, but the figure is a single diagram without (a)/(b) labels, showing only the `∂E/∂x` indexing.
- p. 606: "we will further study their GPU implementations" (of fully connected layers) has no follow-up inside the appendix; presumably it points to Chapter 20.
- Chapter 20 footnote 4 (p. 481) cites Appendix B for the definition of ReLU, which the appendix does not contain.
- Reading method: figure details for B.1, B.2, B.3, B.8, B.9 and listings B.7, B.10, B.11 were read from 220–230-dpi renders of the pages. Dot counts in B.1–B.3 are approximate; everything else was legible.
