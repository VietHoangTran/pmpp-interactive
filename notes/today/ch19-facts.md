# Chapter 19 fact check: CUDA, HIP, Metal

Checked on 2026-11-20 against the live sources listed under each item.

How to read this file:

- Same conventions as `notes/today/ch05-facts.md`. **Verdict** is one of CONFIRMED, WRONG, PARTLY, COULD NOT VERIFY. Where an item has several sub-claims the verdict is for the item as a whole and the text says which sub-claim failed or could not be found. A verdict judges the book's statement, or the assumption a course author would most likely bring (stated in the item title), not the vendor.
- **Quote** is copied from the source (HTML flattened to text, documentation Markdown, GitHub raw files, PDF text, local SDK headers). Line breaks inside quotes are removed.
- Versions the sources reported about themselves on the check date: cuDNN backend 9.28.1 (release notes; support matrix: "the recommended configuration is cuDNN 9.28.1 with CUDA 13.5") and cuDNN frontend 1.28.0; CUDA C++ Best Practices Guide v13.5 (last updated Nov 02, 2026); CUTLASS docs `latest` (convolution page last updated Nov 18, 2026); HIP docs "HIP 7.17.0" (ROCm Core SDK 10.2.0, `ch05-facts.md` B1); MIOpen docs "MIOpen 3.6.2"; HIPIFY docs `latest`; sources at tag `therock-10.2` in `ROCm/rocm-libraries` and `ROCm/rocm-systems`; Metal Shading Language Specification "Version 4.1" (2026-06-04, downloaded today); Metal Feature Set Tables dated October 23, 2026; macOS SDK 27.0 headers on this machine; PyTorch docs 2.15; MLX docs 0.32.4. Repositories (llama.cpp `master`, MLX `main`, PyTorch `main`) read on their default branches today; the last commit date of a file is given where it matters, and line numbers will drift.
- Not checked: anything by running code on a GPU. Nothing here was compiled for CUDA, ROCm or Metal. Group E is our own arithmetic, checked with a short Python script, not a vendor statement.

## Already settled elsewhere (cite, do not re-check)

- `ch05-facts.md` A5 (CUDA block and grid limits: 1024 threads per block, grid x up to 2^31 - 1, y and z up to 65535), A9 (cuBLAS and CUTLASS names, URLs), B3 and B4 (HIP accepts `<<<>>>`, `__global__`, `__shared__`, `__syncthreads()`, `blockIdx`, `threadIdx`, `dim3` with CUDA spelling), B5 (wavefront 64 on gfx9, 32 on gfx10 and later), C1 and C2 (Metal `threadgroup` memory, 32 KB; `threadgroup_barrier`), C4 (`threadgroup_position_in_grid`, `thread_position_in_threadgroup` as `uint3`), C5 (`dispatchThreadgroups` vs `dispatchThreads`; nonuniform threadgroups on Apple4 and later), C7 (MSL is not IEEE-strict by default).
- `ch03-facts.md` A7 and B6 (cuBLAS and rocBLAS are column-major), C7 (`MPSMatrix` is row-major), D2 (llama.cpp hands row-major data to a column-major BLAS).
- `ch06-facts.md` A12 (Nsight Compute: threads reading the same 32-bit word do not conflict); `infra-facts.md` A10, B10, C13 (32 banks of 4 bytes on NVIDIA; 32 or 64 on AMD; nothing documented on Metal). Together these settle the bank assumptions of the notes' Exercise 4 sketch (see "Answers to the notes' portability questions").
- `ch07-facts.md` D1 (cuDNN, MIOpen and PyTorch compute cross-correlation, the book's un-flipped formula; PyTorch padding modes) and D2 (llama.cpp `ggml_conv_2d` is `ggml_im2col` plus `ggml_mul_mat`; MLX's 2D dispatch; PyTorch's `ConvBackend` enum and MPS `convolution2DWithSourceTensor`). This file adds the GEMM-lowering detail on top of D2.
- `ch13-facts.md` C2 (integer `/` and `%` in MSL follow C++ for non-negative operands; `%` with a negative operand is undefined in MSL). Every index in Eq. 19.5 is non-negative.
- `ch15-facts.md` A9 (cuDNN docs moved to `docs.nvidia.com/cudnn/`; arXiv 1410.0759, the book's reference [9], resolves), A10, B6, C6 (`cublasSgemm`, `hipblasSgemm`, `MPSMatrixMultiplication` as library baselines, all FP32 by default), D1 to D4 (where llama.cpp, MLX and PyTorch send a float matmul).
- `ch24-facts.md` D4 (PyTorch's TF32 switches; updated in A6 below).

Totals: 18 CONFIRMED, 0 WRONG, 10 PARTLY, 0 COULD NOT VERIFY (28 items: A1 to A11, B1 to B5, C1 to C6, D1 to D4, E1, E2). Sub-claims that could not be verified are named inside A4, A10, B2, B3, C1, C3 and C5 and collected under "Summary of corrections".

---

## Group A: CUDA, cuDNN, and the book's dated statements

### A1. cuDNN as the book describes it (p. 472): C-language API, common convolution descriptor, opaque descriptors with arbitrary strides, thread-safe, data in device memory

**Verdict: PARTLY.** Every property the book lists is still documented, but on the legacy half of a library that has moved on. Since cuDNN 9.0 the convolution functions the book describes (`cudnnConvolutionForward`, `cudnnSetConvolution2dDescriptor`, ...) are deprecated, though still shipped; NVIDIA now points to a graph API reached through a C++ and Python frontend, with the C backend kept for C-only use.

- What it is (overview, "Latest"): "The NVIDIA CUDA Deep Neural Network library (cuDNN) provides optimized operations for training and inference on NVIDIA GPUs. Use attention, matrix multiplication, convolution, and multi-operation fusion through your framework or the cuDNN Python and C++ APIs."
- Which API: "For most applications, start with the frontend APIs. Use the cuDNN backend API if you need a C-only interface or legacy fixed-function routines that the frontend does not expose."
- The split (Backend API overview): "The cuDNN version 9 library is reorganized into several sub-libraries. This new library structure separates legacy functionality (an imperative API with a fixed set of operations and fusion patterns in cuDNN version 7 and older) from the graph API (a declarative API, introduced in cuDNN version 8), as well as from the engine implementation." "We recommend users to make API calls through the graph API, but the older shim layer ( libcudnn.so ) and legacy libraries ( libcudnn_cnn.so , libcudnn_ops.so , libcudnn_adv.so ) can still be directly used." `cudnn_cnn`: "This entity contains all legacy API functions related to convolutional neural networks."
- Legacy developer guide: "Over time, we are deprecating and removing parts of the legacy API, as the graph API grows to parity with its support surface. However, there is still a gap, and therefore there is still a legacy API."
- `cudnnConvolutionForward()`: "This function has been deprecated in cuDNN 9.0." "This function executes convolutions or cross-correlations over x using filters specified with w , returning results in y ." Signature as printed: `cudnnStatus_t cudnnConvolutionForward(cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y)`.
- Device memory: parameter `x` is a "Data pointer to GPU memory associated with the tensor descriptor xDesc"; the same wording for `w`, `y` and `workSpace`. Core concepts: the handle "is explicitly passed to every subsequent library function that operates on GPU data."
- Common descriptor: `cudnnConvolutionBackwardData` and `cudnnConvolutionBackwardFilter` take the same `const cudnnConvolutionDescriptor_t convDesc` as the forward call; `cudnnConvolutionDescriptor_t` "is a pointer to an opaque structure holding the description of a convolution operation." In the graph API the convolution is an operation node instead (`CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR`); the frontend writes it `conv_fprop(image, filter, Conv_fprop_attributes)` with setters `set_padding`, `set_stride`, `set_dilation`, `set_compute_data_type`, `set_convolution_mode`.
- Arbitrary strides: `cudnnSetTensor4dDescriptorEx` (not marked deprecated): "similarly to cudnnSetTensor4dDescriptor() but with the strides explicitly passed as parameters. This can be used to lay out the 4D tensor in any order or simply to define gaps between dimensions." Two limits the book does not mention: "At present, some cuDNN routines have limited support for strides. Those routines will return CUDNN_STATUS_NOT_SUPPORTED if a 4D tensor object with an unsupported stride is used." and "The total size of a tensor including the potential padding between dimensions is limited to 2 Giga-elements of type datatype."
- Thread safety, with a condition the book omits (Odds and Ends): "The cuDNN library is thread-safe. Its functions can be called from multiple host threads, so long as the threads do not share the same cuDNN handle simultaneously."

For the lesson: present §19.4 as the book's description of the legacy API (name it as such), and add one `<Today>` note: cuDNN 9 deprecates those calls in favour of an operation-graph API; frameworks call cuDNN for you.

- https://docs.nvidia.com/cudnn/overview
- https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/overview.html
- https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-cnn-library.html (`cudnnConvolutionForward`, `cudnnConvolutionBackwardData`, `cudnnConvolutionDescriptor_t`)
- https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-ops-library.html (`cudnnSetTensor4dDescriptorEx`)
- https://docs.nvidia.com/deeplearning/cudnn/backend/latest/developer/legacy-api.html
- https://docs.nvidia.com/deeplearning/cudnn/backend/latest/developer/core-concepts.html
- https://docs.nvidia.com/cudnn/latest/developer/misc.md ("Thread Safety")
- https://docs.nvidia.com/cudnn/latest/operations/Convolutions.md
- https://docs.nvidia.com/deeplearning/cudnn/backend/latest/release-notes.html
- https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html

### A2. Table 19.1 (p. 473): parameter names N, C, H, W, K, R, S, u, v, pad_h, pad_w; the output size P = f(H; R; u; pad_h); the stated purpose of padding

**Verdict: PARTLY.** All eleven names are still cuDNN's, and the docs give the formula the book leaves out (notes, "Unclear in the PDF", p. 473). The book's stated purpose of padding ("improved memory alignment and/or vectorized execution") is not what the parameter documents: `pad_h` and `pad_w` are implicit zero rows and columns in the convolution's arithmetic, and they change the output size.

- "Convolution Terms" table (Odds and Ends, section "Convolution Formulas"), as printed: `x` Input (image) Tensor; `w` Weight Tensor; `y` Output Tensor; `n` Current Batch Size; `c` Current Input Channel; `C` Total Input Channels; `H` Input Image Height; `W` Input Image Width; `k` Current Output Channel; `K` Total Output Channels; `p` Current Output Height Position; `q` Current Output Width Position; `G` Group Count; `pad` Padding Value; `u` Vertical Subsample Stride (along Height); `v` Horizontal Subsample Stride (along Width); `dil h`, `dil w` dilation; `r` Current Filter Height; `R` Total Filter Height; `s` Current Filter Width; `S` Total Filter Width.
- So the book's warning stands: cuDNN's `K` is the chapter's `M`, and `R`, `S` are the chapter's `K`.
- The forward formula, cross-correlation mode: `y[n,k,p,q] = sum_c sum_r sum_s x[n, c, p+r, q+s] * W[k,c,r,s]`; with padding `x[n, c, p+r-pad, q+s-pad]` and `x<0 = 0`, `x>H,>W = 0`; with stride `x[n, c, p*u + r, q*v + s]`; with dilation `x[n, c, p + r*dil_h, q + s*dil_w]`; convolution mode reads `W[k, c, R-r-1, S-s-1]`. With pad 0, stride 1, dilation 1 this is Fig. 19.4 line for line.
- Output size, `cudnnGetConvolution2dForwardOutputDim()` (deprecated in 9.0): `outputDim = 1 + ( inputDim + 2 * pad - ((( filterDim -1 ) * dilation ) + 1 ) ) / convolutionStride ;` (integer division). With pad 0 and stride 1 this is the book's `H - K + 1`. The same formula, written differently, is PyTorch's `H_out = floor((H_in + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1)/stride[0] + 1)` and llama.cpp's `(ins + 2 * p - d * (ks - 1) - 1) / s + 1` (`ggml_calc_conv_output_size`, `ggml/src/ggml.c` line 4464).
- Padding, `cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode, computeType)`: "pad_h Input . Zero-padding height: number of rows of zeros implicitly concatenated onto the top and onto the bottom of input images." (pad_w likewise, left and right). Two parameters the table lacks: dilation (above) and groups (`cudnnSetConvolutionGroupCount`).
- Where alignment does appear, it is internal and automatic, not the user's pad: "For packed NCHW data, when the channel dimension is not a multiple of 8, then the cuDNN library will pad the tensors as needed to enable Tensor Core operations."

For the lesson: give the output-size formula as cuDNN's (stride, padding, dilation), say that the chapter's kernels are the pad 0, stride 1 case, and describe padding as ghost cells of zeros (Chapter 7's word), not as an alignment device.

- https://docs.nvidia.com/deeplearning/cudnn/backend/latest/developer/misc.html (section "Convolution Formulas")
- https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-cnn-library.html (`cudnnGetConvolution2dForwardOutputDim`, `cudnnSetConvolution2dDescriptor`)
- https://docs.nvidia.com/deeplearning/cudnn/backend/latest/developer/legacy-api.html (section "Tensor Transformations")
- https://docs.pytorch.org/docs/2.15/generated/torch.nn.Conv2d.html

### A3. Algorithm families (p. 473): GEMM, Winograd, FFT, "etc."

**Verdict: CONFIRMED,** and the full list is longer. `cudnnConvolutionFwdAlgo_t` (cudnn_ops library) has eight values; descriptions as printed:

| Value | Documented description |
|---|---|
| `CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM` | "expresses the convolution as a matrix product without actually explicitly forming the matrix that holds the input tensor data." |
| `CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM` | "... without actually explicitly forming the matrix that holds the input tensor data, but still needs some memory workspace to precompute some indices in order to facilitate the implicit construction of the matrix that holds the input tensor data." (the enum page misspells it `IMPLICIT_PPRECOMP_GEMM`; the algorithm table on the cudnn_cnn page has the correct identifier) |
| `CUDNN_CONVOLUTION_FWD_ALGO_GEMM` | "expresses convolution as an explicit matrix product. A significant memory workspace is needed to store the matrix that holds the input tensor data." |
| `CUDNN_CONVOLUTION_FWD_ALGO_DIRECT` | "expresses the convolution as a direct convolution (for example, without implicitly or explicitly doing a matrix multiplication)." |
| `CUDNN_CONVOLUTION_FWD_ALGO_FFT` | "uses the Fast-Fourier Transform approach ... A significant memory workspace is needed to store intermediate results." |
| `CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING` | "uses the Fast-Fourier Transform approach but splits the inputs into tiles ... less than CUDNN_CONVOLUTION_FWD_ALGO_FFT for large size images." |
| `CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD` | "uses the Winograd Transform approach ... A reasonably sized workspace is needed". Note: "The CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD algorithm is not supported on GPUs based on the NVIDIA Hopper GPU architecture and later GPU architectures." |
| `CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED` | "uses the Winograd Transform approach ... A significant workspace may be needed to store intermediate results." |

Two more facts worth a sentence: the explicit-GEMM entry is the book's "materialise B" variant, documented with exactly the cost the book names (p. 467); and for the forward call "all algorithms perform deterministic computations. Specifying a separate algorithm can cause changes in performance and support." The enum and the functions that take it are legacy (A1); in the graph API the choice is an engine picked by heuristics.

- https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-ops-library.html (`cudnnConvolutionFwdAlgo_t`)
- https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-cnn-library.html (`cudnnConvolutionForward`, "Supported Algorithms")

### A4. "cuDNN's GEMM-based algorithm ... lazily generating and loading [the expanded matrix] into on-chip memory only", computing the index mapping "dynamically as the computation proceeds", on a matmul "similar to" Tan et al. (pp. 473 to 474)

**Verdict: PARTLY.** cuDNN documents three GEMM-family algorithms, not one: the book's lazy version (`IMPLICIT_GEMM`), a variant that precomputes indices into a workspace instead of computing them all on the fly (`IMPLICIT_PRECOMP_GEMM`), and the explicit one the book argues against (`GEMM`, A3). It even had a public explicit unfold call, `cudnnIm2Col`, deprecated in 9.0. That cuDNN's matmul is "similar to" reference [10] (Tan et al., Fermi DGEMM) COULD NOT be verified: cuDNN documents no kernel internals.

- `cudnnIm2Col()`: "This function has been deprecated in cuDNN 9.0." "This function constructs the A matrix necessary to perform a forward pass of GEMM convolution." "This A matrix has a height of batch_size*y_height*y_width and width of input_channels*filter_height*filter_width". That orientation is the transpose of the book's `B` (rows are output pixels, all samples stacked), the same orientation as CUTLASS (A7) and llama.cpp (D2).
- Precomputed indices are also what NVIDIA's open GEMM library does (A7: "Precomputing kernel-invariant pointer deltas on the host"), so "computes the mapping dynamically" is one of several documented strategies.

- https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-ops-library.html (`cudnnConvolutionFwdAlgo_t`)
- https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-cnn-library.html (`cudnnIm2Col`)

### A5. "cuDNN performs the required tensor transposition to store the result in the user's desired data layout" (p. 474); the layouts Exercise 2 asks about

**Verdict: PARTLY.** cuDNN documents layout conversion, but tied to Tensor Cores, in both directions, and only when they are used; it does not describe a transposition step of the GEMM algorithm in general. What it documents is directly useful for Exercise 2 (NCHW vs NHWC vs CHWN).

- Formats (`cudnnTensorFormat_t`): `CUDNN_TENSOR_NCHW` ("batch size, feature maps, rows, columns ... the columns are the inner dimension"), `CUDNN_TENSOR_NHWC` ("batch size, rows, columns, feature maps ... the feature maps are the inner dimension"), `CUDNN_TENSOR_NCHW_VECT_C` (vectors of feature maps, int8 types only). Other orders through strides (A1).
- Legacy developer guide, "Conversion Between NCHW And NHWC": "Tensor Cores require that the tensors be in the NHWC data layout. Conversion between NCHW and NHWC is performed when the user requests Tensor Op math. However, a request to use Tensor Cores is just that, a request and Tensor Cores may not be used in some cases. The cuDNN library converts between NCHW and NHWC if and only if Tensor Cores are requested and are actually used." "Non-Tensor Op convolutions will not perform conversions between NCHW and NHWC."
- `cudnnSetFilter4dDescriptor`: "Tensor format CUDNN_TENSOR_NHWC has limited support in cudnnConvolutionForward() , cudnnConvolutionBackwardData() , and cudnnConvolutionBackwardFilter() ." (legacy API).
- CUTLASS's recommended configuration is also NHWC (A7), and MLX's and Metal Performance Primitives' convolutions accept only channels-last activations (D3, C5).

For Exercise 2's discussion (ours, from these facts): the libraries' preference for NHWC comes from the matrix units and from making the GEMM K dimension (`C·R·S` with `C` innermost) contiguous, not from fewer bytes moved; the book's direct kernel, with neighbouring threads on neighbouring `w`, prefers NCHW.

- https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-graph-library.html (`cudnnTensorFormat_t`)
- https://docs.nvidia.com/deeplearning/cudnn/backend/latest/developer/legacy-api.html
- https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-ops-library.html (`cudnnSetFilter4dDescriptor`)

### A6. Frameworks that use cuDNN: "PyTorch, Caffe, TensorFlow, MXNet" (p. 472); "used by most deep learning frameworks" (p. 474)

**Verdict: PARTLY.** PyTorch and TensorFlow still depend on cuDNN, and NVIDIA's own page adds JAX/XLA and LLM serving stacks. Two of the four names are no longer active projects: Apache MXNet is retired, and Caffe's repository has had no commit since 2020.

- Apache Attic: "Apache MXNet ... retired in September 2023 and the move to the Attic was completed in February 2024." GitHub `apache/mxnet`: archived.
- GitHub `BVLC/caffe`: not archived; last commit on `master` 2020-02-13 (GitHub API).
- cuDNN overview, "Integrations": PyTorch, JAX / XLA, Megatron Core, Transformer Engine, and "LLM INFERENCE" vLLM and SGLang through FlashInfer ("vLLM → FlashInfer → cuDNN ... Use cuDNN through the FlashInfer integration."). For LLM serving the cuDNN path NVIDIA advertises is attention, not convolution.
- TensorFlow's pip install page still lists "cuDNN SDK"; JAX: "Installs the wheel compatible with NVIDIA CUDA 13 and cuDNN 9."
- PyTorch 2.15, the knobs that map to §19.4's algorithm choice: `torch.backends.cudnn.benchmark`: "A bool that, if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest." `torch.backends.cudnn.deterministic`: "causes cuDNN to only use deterministic convolution algorithms." Reproducibility note: "When a cuDNN convolution is called with a new set of size parameters, an optional feature can run multiple convolution algorithms, benchmarking them to find the fastest one. ... the benchmark may select different algorithms on subsequent runs, even on the same machine."
- TF32 for convolutions (updates `ch24-facts.md` D4): the old flag still reads "The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True." with `torch.backends.cudnn.allow_tf32`, but the backends page now says "allow_tf32 is going to be deprecated", and "After PyTorch 2.9, we provide a new sets of APIs to control the TF32 behavior in a more fine-grained way": `torch.backends.cudnn.conv.fp32_precision = "ieee"` or `"tf32"`. The page shows the new settings by example and does not state their default. Also: "operations implemented using CUDA GEMM can inherit the matmul precision setting, including slow or naive convolution fallbacks."

For the lab README (ours): if the reader compares their kernels with `torch.nn.functional.conv2d` on an NVIDIA GPU, a float32 call may run in TF32 unless they set the flag; compare with a tolerance and print the flag.

- https://attic.apache.org/projects/mxnet.html
- https://github.com/apache/mxnet
- https://github.com/BVLC/caffe
- https://docs.nvidia.com/cudnn/overview
- https://www.tensorflow.org/install/pip
- https://docs.jax.dev/en/latest/installation.html
- https://docs.pytorch.org/docs/2.15/backends.html (`torch.backends.cudnn`)
- https://docs.pytorch.org/docs/2.15/notes/randomness.html ("CUDA convolution benchmarking")
- https://docs.pytorch.org/docs/2.15/notes/cuda.html ("TensorFloat-32 (TF32) on Ampere (and later) devices")

### A7. Beyond the book: NVIDIA's own documentation of §19.3, CUTLASS "Implicit GEMM Convolution"

**Verdict: CONFIRMED.** CUTLASS documents the chapter's whole argument and the book's index map, with the GEMM transposed and the tensors channels-last.

- "2-D convolution may be mapped to matrix multiply by first forming a convolution matrix containing elements of the activations tensor, then multiplying this by a matrix formed from the filters tensor. The earliest form of this algorithm constructs the convolution matrix explicitly via an operation conventionally referred to as im2col . The resulting matrix replicates each activation element by a factor equal to the filter size, consuming additional storage capacity and memory bandwidth." (the book's Eq. 19.4 and p. 467 argument)
- "Instead of constructing the convolution matrix explicitly, it forms tiles of the convolution matrix on the fly as data are loaded from global memory into Shared Memory by carefully updating pointers and predicates. Once the convolution matrix is formed in Shared Memory, the existing warp-level GEMM components accumulate the result of convolution and update the output tensor." (Fig. 19.11's design)
- Shapes: "x(NHWC), w(KRSC), and y(NPQK)"; `int GEMM_M = N * P * Q; int GEMM_N = K; int GEMM_K = C * R * S;` with `gemm_k = s + S * (r + R * c)` and inverse `c = gemm_k / (RS)`, `residual = gemm_k % (RS)`, `r = residual / S`, `s = residual % S`. That decomposition of the reduction index is Eq. 19.5's `u/K²`, `(u%K²)/K`, `(u%K²)%K`; the output index uses the same divide and remainder over `PQ` and `Q`. Output rows are pixels and columns are output channels, the transpose of the book's `Y = F·B`.
- Why the analytic form is not enough: "a tile iterator could be implemented to compute these functions analytically and load the appropriate elements. However, the resulting modulo arithmetic would be computationally intensive, and overhead would limit performance of a GEMM kernel targeting Turing Tensor Cores." The optimised iterators use "a. Precomputing kernel-invariant pointer deltas on the host b. Computing cta-invariant mask predicates on device-side iterator ctors c. Use of fast divmod to map GEMM dimensions to convolution tensors."
- Recommended: "All tensors are 128-bit aligned NHWC tensors", "Channel count (C) is a multiple of 32 elements", "Filter count (K) is a multiple of 32 elements".
- Two things on the page we did not reconcile, quoted as printed: it calls the activation matrix "a row-major matrix of extent NHW -by- RSC" while the output is "NPQ -by- K" (a GEMM needs NPQ rows on both); and its `f(p, r) = p * stride_h + R - r - 1 + pad_h` flips the filter (convolution mode) and adds `pad_h`, unlike cuDNN's cross-correlation `p*u + r - pad` (A2). Use cuDNN's formula in lessons.

- https://docs.nvidia.com/cutlass/latest/media/docs/cpp/implicit_gemm_convolution.html

### A8. Integer division and remainder per loaded element (Eq. 19.5, Fig. 19.11 line 28; "additional indexing arithmetic", p. 474)

**Verdict: CONFIRMED.** NVIDIA documents both operations as expensive, and both NVIDIA's library (A7) and llama.cpp (D1) replace them.

- Best Practices Guide v13.5, 12.1.4 "Division Modulo Operations": "Low Priority: Use shift operations to avoid expensive division and modulo calculations." "Integer division and modulo operations are particularly costly and should be avoided or replaced with bitwise operations whenever possible: If n is a power of 2, ( i/n ) is equivalent to ( i ≫ log2(n) ) and ( i%n ) is equivalent to ( i&(n-1) )." Its summary list: "Integer division and modulo operation are costly as they compile to up to 20 instructions. ... the compiler will perform these conversions if n is literal."
- Consequence for the lab (ours): make `K` a compile-time constant (template parameter or macro) where the lab allows it, so `/K` and `%K` by a literal become cheap; `W_out` stays a runtime divisor unless specialised. llama.cpp does exactly this: compile-time `KW`, `KH` for 3x3 and 1x1, `fastdiv` otherwise (D1).

- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (sections 12.1.4 and the recommendations summary)

### A9. The 2012 network (pp. 454 to 455): "about 60 million parameters, 650,000 neurons, 1.2 million images, 1000 classes, one week on two GPUs, cuda-convnet, 15.3% vs 26.2%"

**Verdict: PARTLY.** The numbers are the paper's, with two qualifications: the paper says "five to six days" (the book rounds to a week) on two GTX 580 GPUs, and the 15.3% was won by "a variant of this model" in ILSVRC-2012, while the 1.2 million images are the LSVRC-2010 training set.

- Paper (NIPS 2012 proceedings PDF), abstract: "We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes. ... The neural network, which has 60 million parameters and 650,000 neurons, consists of five convolutional layers ... We also entered a variant of this model in the ILSVRC-2012 competition and achieved a winning top-5 test error rate of 15.3%, compared to 26.2% achieved by the second-best entry."
- Body: "which took five to six days on two NVIDIA GTX 580 3GB GPUs"; code at "http://code.google.com/p/cuda-convnet/".
- That URL (the book's, p. 475) now redirects to the Google Code archive, https://code.google.com/archive/p/cuda-convnet (HTTP 200).
- Trap: the proceedings landing page (`...-Abstract.html`) carries a different, earlier abstract ("1.3 million", "500,000 neurons", top-5 "18.9%"). Cite the PDF.

These are historical facts about one network, not device figures; present them as the book's story and keep GPU model names out of the lesson body.

- https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
- https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
- https://code.google.com/archive/p/cuda-convnet

### A10. References and links the chapter gives

**Verdict: CONFIRMED** for every link the notes name; one reference's content COULD NOT be read.

- Footnote 3 (p. 464), Pete Warden, "Why GEMM is at the heart of deep learning": https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/ (HTTP 200).
- Reference [8], Chellapilla et al., 2006 (the GEMM formulation): https://inria.hal.science/inria-00112631 (HTTP 200).
- Reference [9], cuDNN, arXiv 1410.0759: resolves (`ch15-facts.md` A9).
- Reference [10], Tan et al., "Fast implementation of DGEMM on Fermi GPU", 2011: DOI https://doi.org/10.1145/2063384.2063431 redirects to the ACM Digital Library, which refused a scripted fetch (HTTP 403). The DOI resolves; the paper was not read.
- cuda-convnet: A9.

### A11. Grid limits for the book's launches (Fig. 19.5 `gridDim(M, T, N)`; the derived Fig. 19.11 grid)

**Verdict: CONFIRMED** from `ch05-facts.md` A5: grid x up to 2^31 - 1 blocks, y and z up to 65535, on every listed compute capability. For Fig. 19.5 that bounds `T` (tiles per output map) and `N` (batch) by 65535; for Fig. 19.11 (x = pixel tiles, y = `M/TILE_WIDTH`, z = `N`) it bounds `M/16` and `N`. llama.cpp's implicit-GEMM launch checks exactly these two: `(OC + 63) / 64 <= 65535 && B <= 65535` (`conv2d.cu` line 404). HIP's limits differ (B2).

---

## Group B: HIP / ROCm

### B1. Do Figs. 19.7 and 19.11 (corrected) compile and run under HIP unchanged?

**Verdict: CONFIRMED** (by `ch05-facts.md` B3 and B4). The kernels use only `__global__`, `__shared__`, `__syncthreads()`, `blockIdx.{x,y,z}`, `threadIdx.{x,y}`, `dim3` and `<<<>>>`, all with CUDA spelling on HIP; none of the portability traps in CLAUDE.md (warp `_sync` masks, cooperative groups, half headers, CUB, matrix intrinsics) appears. A 16 x 16 block is 256 threads: 4 wavefronts of 64 on gfx9, 8 of 32 on gfx10 and later (`ch05-facts.md` B5). The Exercise 4 bank analysis does not carry over unchanged: AMD has 32 or 64 banks (`infra-facts.md` B10).

### B2. Grid limits on AMD (the HIP side of A11)

**Verdict: PARTLY.** y and z match CUDA (65535). x is larger, but AMD documents a cap CUDA does not have: at most 2^32 - 1 threads in the whole grid.

- HIP 7.17.0 "Hardware features" table, the same for every listed architecture (RDNA1, CDNA1, RDNA2, CDNA2, RDNA3, CDNA3): "Maximum dimensionality of grid | 3"; "Maximum x-, y- and z-dimension of a grid | x - 2^32-1 y - 2^16-1 z - 2^16-1"; "Maximum number of threads per grid | 2^32 - 1".
- What `hipDeviceProp_t.maxGridSize` reports: the CLR runtime clamps each dimension to `INT32_MAX`: `info_.maxGridDim_[0] = std::min(grid_max_dim.x, static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));` (`projects/clr/rocclr/device/rocm/rocdevice.cpp` lines 1403 to 1405), copied into `deviceProps.maxGridSize[i]` (`projects/clr/hipamd/src/hip_device.cpp` lines 656 to 658).
- Not verified: whether the table's x figure counts blocks or threads (it does not say), and where the runtime rejects a launch over 2^32 - 1 threads (I read `ihipLaunchKernel_validate` in `hip_module.cpp` and found no such check there).

Consequence (ours): the direct kernel launches about `N · M · H_out · W_out` threads (rounded up to tiles); on HIP that product must stay below 2^32. The lab's sizes are far below it; the explorer should not offer combinations above it for a "HIP" label.

- https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hardware_features.html
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/rocclr/device/rocm/rocdevice.cpp
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/src/hip_device.cpp
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/src/hip_module.cpp

### B3. MIOpen: AMD's counterpart of cuDNN (§19.4)

**Verdict: CONFIRMED.** Same shape as cuDNN's legacy API (handle, opaque descriptors, forward call with an algorithm and a workspace), with three differences worth teaching: five algorithms instead of eight, a Find step that is mandatory for the classic forward call, and an explicit "not supported" for the CHWN layout of Exercise 2.

- Docs ("MIOpen 3.6.2 Documentation"): "MIOpen is the AMD open-source, deep-learning primitives library for GPUs. It implements fusion to optimize for memory bandwidth and GPU launch overheads, providing an auto-tuning infrastructure to overcome the large design space of problem configurations. It also implements different algorithms to optimize convolutions for different filter and input sizes." Source lives in `ROCm/rocm-libraries`.
- Algorithms, `miopen.h` at `therock-10.2` (lines 1204 to 1215): "Convolutional algorithm mode for forward propagation. MIOpen use cross-correlation for its convolution implementation." `miopenConvolutionFwdAlgoGEMM = 0, /*!< GEMM variant */`, `miopenConvolutionFwdAlgoDirect = 1`, `miopenConvolutionFwdAlgoFFT = 2`, `miopenConvolutionFwdAlgoWinograd = 3`, `miopenConvolutionFwdAlgoImplicitGEMM = 5, /*!< Implicit GEMM convolutions */`.
- Descriptor (line 970): `miopenInitConvolutionDescriptor(miopenConvolutionDescriptor_t convDesc, miopenConvolutionMode_t c_mode, int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w)`; "For group/depthwise convolution dilation height and width, only a dilation value of 1 is supported." Output size: `miopenGetConvolutionForwardOutputDim(convDesc, inputTensorDesc, filterDesc, &n, &c, &h, &w)` (line 1123); the header prints no formula.
- Layouts (lines 360 to 375): `miopenTensorNCHW = 0, /*!< NCHW memory layout (Fully supported) */`, `miopenTensorNHWC = 1, /*!< NHWC memory layout (Fully supported) */`, `miopenTensorCHWN = 2, /*!< CHWN memory layout (Not supported) */`, plus partially supported vectorised forms.
- Find (lines 1750 to 1760): `miopenFindConvolutionForwardAlgorithm` "attempts all MIOpen forward convolution algorithms based on the input configuration, and outputs performance metrics ... sorted ... This function is mandatory before using miopenConvolutionForward()." Docs: "miopenFindConvolution*() caches its own results on disk so subsequent calls during the same MIOpen session run faster." and "Internally, MIOpen's find calls compile and benchmark a set of solvers". The alternative: "MIOpen v2.0 introduces immediate mode, which removes the requirement for miopenFindConvolution*() calls" (`miopenConvolutionForwardImmediate`, line 1421), backed by a database with an AI-heuristic fallback by default.
- Thread safety COULD NOT be verified as a general statement. The one statement found is narrower: "Kernel timings are based on the MIOpen handle and is not thread-safe. In order to use multi-threaded profiling, create an MIOpen handle for each concurrent thread." (lines 229 to 231).
- PyTorch's ROCm path calls MIOpen (`ch07-facts.md` D1).

- https://rocm.docs.amd.com/projects/MIOpen/en/latest/index.html
- https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/miopen/include/miopen/miopen.h

### B4. Porting cuDNN calls to AMD: HIPIFY's cuDNN-to-MIOpen table, and hipDNN

**Verdict: PARTLY.** HIPIFY maps the core convolution call and five of the eight algorithm values, but not the descriptor setter the book's API description relies on. AMD also has a cuDNN-graph-style library, hipDNN, which its own README calls early-stage.

- HIPIFY "CUDNN API supported by MIOPEN" maps `cudnnConvolutionForward` → `miopenConvolutionForward`, `cudnnCreate` → `miopenCreate`, `CUDNN_CONVOLUTION_FWD_ALGO_GEMM` → `miopenConvolutionFwdAlgoGEMM`, `..._IMPLICIT_GEMM` → `miopenConvolutionFwdAlgoImplicitGEMM`, `..._FFT` → `...FFT`, `..._DIRECT` → `...Direct`, `..._WINOGRAD` → `...Winograd`. No MIOpen entry is listed for `..._IMPLICIT_PRECOMP_GEMM`, `..._FFT_TILING`, `..._WINOGRAD_NONFUSED`, or for `cudnnSetConvolution2dDescriptor` (MIOpen's initialiser takes the mode first and has no compute type, B3). The page `CUDNN_API_supported_by_HIP.html` returns 404 today; the MIOpen table is the one that exists.
- hipDNN (`projects/hipdnn` at `therock-10.2`): "hipDNN is in the early stages of development. There is currently limited functionality available to execute graphs." "hipDNN is a graph-based deep learning library for AMD GPUs that leverages a flexible plugin architecture". Its operation-support page lists the "MIOpen Provider Plugin" with "Convolution operations (Forward, Dgrad, Wgrad)". Docs at https://rocm.docs.amd.com/projects/hipdnn/en/latest/index.html (HTTP 200; the capitalised `/projects/hipDNN/` path is 404).

- https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUDNN_API_supported_by_MIOPEN.html
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hipdnn/README.md
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hipdnn/docs/OperationSupport.md

### B5. Integer division on AMD (the HIP side of A8)

**Verdict: CONFIRMED,** qualitatively only. HIP performance guidelines, "Instruction throughput optimization": "Division requires many more hardware cycles than multiplication. Similarly, bitwise operations (shifts, AND, OR) are single-cycle instructions on integer units, making them far more efficient than equivalent arithmetic for power-of-two calculations." No instruction count is given. llama.cpp's `fastdiv` runs on HIP too (D1, same source).

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html

---

## Group C: Metal

### C1. The chapter's kernels on Metal, and the largest grid

**Verdict: PARTLY.** Everything the kernels need is documented (`ch05-facts.md` C1, C2, C4, C5): `threadgroup` arrays for `Fds`/`Bds`, `threadgroup_barrier(mem_flags::mem_threadgroup)` for both `__syncthreads()`, `threadgroup_position_in_grid` as a `uint3` for the three grid axes, and `dispatchThreads` with nonuniform threadgroups as an alternative to Fig. 19.5's ceiling division for edge tiles. What COULD NOT be verified is a maximum compute grid size: the Feature Set Tables list grid maxima only for mesh and object shaders ("Maximum threadgroups per mesh shader grid"), and the dispatch pages and "Calculating threadgroup and grid sizes" give none. The position attributes are at most `uint3`, so each axis is a 32-bit quantity (ours, from the attribute types).

- https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf (October 23, 2026)
- https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/dispatchthreads(_:threadsperthreadgroup:)
- https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes

### C2. The largest single buffer: can the unrolled matrix `B` be allocated? (`MTLDevice.maxBufferLength`)

**Verdict: CONFIRMED.** Metal caps one buffer at a device-reported size with a documented floor; the explicit-unroll lab must query it.

- Apple docs: "The largest amount of memory, in bytes, that a GPU device can allocate to a buffer instance." "The property's value is at least 256 MB (268,435,456 bytes)."
- Header (macOS 27.0 SDK, `MTLDevice.h` line 1152): `@property (readonly) NSUInteger maxBufferLength API_AVAILABLE(macos(10.14), ios(12.0));`
- The Feature Set Tables give no number; their only mention is footnote 9: "The maximum texture buffer width, in pixels, is also limited by MTLDevice.maxBufferLength divided by the size of a pixel, in bytes, as well as available memory."
- MLX hits this in production and tiles its explicit unfold because of it (D3).

For the lab (ours, Group E): `B` for one 224 x 224, 64-channel, 3 x 3 layer is 108 MiB in FP32; a batch of 8 is 866 MiB, above the documented floor. `launch.mm` must read `maxBufferLength` and either refuse with a clear message or unroll per sample; lessons must not quote the reader's value, only the floor as Apple's documented minimum.

- https://developer.apple.com/documentation/metal/mtldevice/maxbufferlength
- Local header: `Metal.framework/Headers/MTLDevice.h` (macOS 27.0 SDK)

### C3. Apple's graph-level route: MPSGraph `convolution2D` and `imToCol`

**Verdict: CONFIRMED.** MPSGraph has both halves of the chapter: a convolution op, and an explicit im2col op (the book's materialised `B`). Neither exposes or documents an algorithm choice, and the convolution op does not say whether it flips the filter.

- `MPSGraphConvolutionOps.h` (macOS 27.0 SDK): `-(MPSGraphTensor *) convolution2DWithSourceTensor:(MPSGraphTensor *) source weightsTensor:(MPSGraphTensor *) weights descriptor:(MPSGraphConvolution2DOpDescriptor *) descriptor name:(NSString * _Nullable) name` (macOS 11.0 and later). Descriptor properties: `strideInX`, `strideInY` ("Source x-coordinate, sx is computed from destination x-coordinate, dx as sx = strideInX*dx."), `dilationRateInX/Y`, `paddingLeft/Right/Top/Bottom`, `paddingStyle`, `dataLayout`, `weightsLayout`, `groups`.
- Layouts (`MPSGraphCore.h`): data `MPSGraphTensorNamedDataLayoutNCHW` or `...NHWC`; weights `...OIHW` or `...HWIO`. With NCHW and OIHW the source is "batch x channels x height x width with width as fastest moving dimension" and the weights "outputChannels x inputChannels x kernelHeight x kernelWidth with kernelWidth as fastest moving dimension", the book's layouts.
- `MPSGraphImToColOps.h`: `-(MPSGraphTensor *) imToColWithSourceTensor:(MPSGraphTensor *) source descriptor:(MPSGraphImToColOpDescriptor *) descriptor name:(NSString * _Nullable) name` (macOS 14.0 and later), with kernel size, stride, dilation, padding and `dataLayout`; also `colToImWithSourceTensor:outputShape:descriptor:name:`. The header does not document the shape of the result.
- Flip: neither header says cross-correlation or convolution. PyTorch's MPS backend passes `nn.Conv2d` weights (OIHW) to `convolution2DWithSourceTensor` without reversing them (`aten/src/ATen/native/mps/operations/Convolution.mm` around line 955, last commit 2026-11-07), so in practice it computes cross-correlation (our reading of PyTorch's use; not an Apple statement).
- Two slips in Apple's headers, for anyone who reads them: the im2col descriptor says "e.g. `batch x channels x width x height` for `NCHW` layout" (the convolution header has the correct "batch x channels x height x width"), and the convolution header gives the dilated kernel width as "`(dilationRateInX-1)*kernelWidth+1`" (cuDNN's correct form is `(filterDim-1)*dilation+1`, A2).

- Local headers: `MetalPerformanceShadersGraph.framework/Headers/MPSGraphConvolutionOps.h`, `MPSGraphImToColOps.h`, `MPSGraphCore.h` (macOS 27.0 SDK)
- https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/operations/Convolution.mm

### C4. Apple's kernel-level route: `MPSCNNConvolution`

**Verdict: CONFIRMED,** with a precision caveat that matters for any comparison with a CPU reference.

- Not deprecated (class annotation `MPS_CLASS_AVAILABLE_STARTING( macos(10.13), ios(10.0), macCatalyst(13.0), tvos(10.0))`; only some neuron-fusion members are deprecated). "The MPSCNNConvolution convolves the input image with a set of filters, each producing one feature map in the output image."
- It works on images, not raw buffers: "A MPSCNNKernel consumes one MPSImage and produces one MPSImage." (`MPSCNNKernel.h`).
- Weights: "Currently only OHWI layout is supported." `MPSCNNConvolutionWeightsLayoutOHWI // weights[ outputFeatureChannels ][ kernelHeight ][ kernelWidth ][ inputFeatureChannels/groups ]`. That is channels-last, not the book's `M × C × K × K`.
- Precision: accumulator "Default is MPSNNConvolutionAccumulatorPrecisionOptionFloat." Weights: "dataType precisions of MPSDataTypeFloat32 may return kernelWeightsDataType of MPSDataTypeFloat16 or MPSDataTypeFloat32. When kernelWeightsDataType returns MPSDataTypeFloat32 the accumulatorPrecisionOption on the CNNConvolution object must be set to MPSNNConvolutionAccumulatorPrecisionOptionFloat. When kernelWeightsDataType is unimplemented the kernel will use float16 precision." So FP32 weights are used as FP16 unless the data source opts in.

- Local headers: `MetalPerformanceShaders.framework/Versions/A/Frameworks/MPSNeuralNetwork.framework/Headers/MPSCNNConvolution.h` (lines 390 to 405, 862 to 937), `MPSCNNKernel.h`, `MPSNeuralNetworkTypes.h` (lines 110 to 119), macOS 27.0 SDK

### C5. Apple's shader-side route: Metal Performance Primitives `convolution2d` (Metal 4)

**Verdict: CONFIRMED.** MSL 4.1 section 7.2.2 documents a convolution op inside shaders, channels-last only, used by a whole threadgroup. How it positions the window is not documented; PyTorch measured it.

- MSL 4.1: "The template class convolution2d performs a 2D convolution where 2D stands for two spatial dimensions of width x height." Descriptor: `convolution2d_descriptor(int4 destination_dimensions, int4 source_dimensions, int2 kernel_dimensions, convolution2d_activation_layout activation_layout = nhwc, convolution2d_weights_layout weight_layout = hwio, int2 strides = int2(1, 1), int2 dilations = int2(1, 1), int groups = 1, bool relaxed_precision = false, mode convolution2d_mode = mode::multiply)`. "Currently, the only scope supported is execution_simdgroups<N> where N is simdgroups_per_threadgroup."
- Table 7.8: activation "NHWC layout ... C = input channels (fastest moving dimension)"; weights "HWIO layout ... O = output channels (fastest moving dimension)"; destination "NHWO". `relaxed_precision`: "Relaxed precision allows the operation to truncate the mantissa before the multiplication." (off by default).
- Header (`MPPTensorOpsConvolution2d.h`, macOS 27.0 SDK): `static_assert(Descriptor.activation_layout == convolution2d_activation_layout::nhwc, "only nhwc activation layout supported currently");` likewise "only hwio weights layout supported currently" and "only group size 1 supported currently". The descriptor has no padding field; the op takes a window origin through `set_offsets(int2)`. Guarded by `__HAVE_TENSOR__` (tensors are Apple7 and later, `infra-facts.md` C10).
- In production: PyTorch's `conv3d_mpp` kernel (`aten/src/ATen/native/mps/kernels/Convolution.metal`, last commit 2026-10-13) builds a `convolution2d<desc, execution_simdgroups<NSG>>` with `nhwc`, `hwio`, `multiply_accumulate`, and its comment records the undocumented part: "MPP anchors the window at ceil((K - 1) * dilation / 2) (measured; for even kernels with dilation > 1 this is NOT (K / 2) * dilation)." Its descriptor is a compile-time constant, so PyTorch "bakes popular geometries into bundled entry points; catalog misses use conv3d_simd". llama.cpp's Metal 4 convolution uses `matmul2d` instead (D2).
- COULD NOT VERIFY: the window-anchor rule (Apple documents none).

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 7.2.2, Tables 7.7 and 7.8, pages 365 to 366)
- Local header: `MetalPerformancePrimitives.framework/Versions/A/Headers/MPPTensorOpsConvolution2d.h` (macOS 27.0 SDK)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/kernels/Convolution.metal (lines 1 to 2, 384 to 445)

### C6. Integer division on Apple GPUs (the Metal side of A8)

**Verdict: PARTLY.** Nothing in the MSL 4.1 specification or the current Xcode optimisation pages states a cost. The only Apple statement found is a 2016 talk.

- WWDC16 session 606, "Advanced Metal Shader Optimization" (transcript on developer.apple.com): "On modern GPUs most of them do not have integer division or modulus instructions, integer not float." "So avoid divisional modulus by denominators that are not literal or function consonants" (transcription of "function constants").
- For the Metal lab (ours): pass `K` and the output width as function constants or macros (`ch05-facts.md` C7 `preprocessorMacros`), so the divides in Eq. 19.5 have constant denominators. A function constant cannot size a `threadgroup` array (`ch07-facts.md` C3), so the tile width stays a macro.

- https://developer.apple.com/videos/play/wwdc2016/606/

---

## Group D: How llama.cpp, MLX and PyTorch do this chapter (goal 3)

Read from raw files on the default branches on 2026-11-20. These are the honest LLM bridges: convolution layers do not appear in decoder-only transformer blocks, but they do appear in the encoders that feed multimodal LLMs, and every backend lowers them to GEMM the way §19.3 does.

### D1. llama.cpp, CUDA and HIP: `conv2d_implicit_gemm_f16` is Fig. 19.11 in production

**Verdict: CONFIRMED.** `ggml/src/ggml-cuda/conv2d.cu` (last commit 2026-11-18), compiled for HIP from the same source (`ch05-facts.md` B9).

- Same orientation as the book: rows of the GEMM are output channels, columns are output pixels. `const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;` with `BM = BN = BK = 64` (line 137); the launch is `const dim3 grid(unsigned((OW * OH + 63) / 64), unsigned((OC + 63) / 64), unsigned(B * split_k));` (line 445), the grid the notes derived for Fig. 19.11 (x = pixel tiles, y = channel tiles, z = batch), with ceiling division.
- The weight tile is read straight from the `OC × (IC·KH·KW)` weight array, the book's "F needs no data movement": `((const int4 *) weight)[((m0 + row) * k_total + k0 + col) / 8]`.
- The input tile is gathered with Eq. 19.5: `const int ci = KW && KH ? ki / (KW * KH) : int(fastdiv(uint32_t(ki), kwkh_fd));` (line 207), `const int ky = KW ? r / KW : int(fastdiv(uint32_t(r), kw_fd));` (line 209), `kx = r - ky * kw`, `const int offset = ki < k_total ? (n * ic + ci) * ih * iw + ky * dy * iw + kx * dx : 0;` (line 211). `KW`, `KH` are template parameters for 3 x 3 and 1 x 1 (literal divisors), `fastdiv` with precomputed `init_fastdiv_values` otherwise (A8).
- Bounds: every load checks `m0 + row < oc`, `k0 + col < k_total`, `spatial < ow * oh` and writes zeros otherwise, the checks Fig. 19.11 leaves out. Padding is handled before the kernel by writing a zero-padded copy of the input (`conv2d_pad_f16`), so the gather needs no `if` for ghost cells.
- Three choices the book's p. 472 argument predicts, in the code: (1) "Short reductions on small maps do not amortize conversion and launch costs." `const bool small_conv = IC * KW * KH < 64 && OW * OH < 512;` (lines 392 to 393) sends small layers to the direct kernel `conv2d_kernel`; (2) "Split long reductions so small spatial maps still occupy the GPU." (line 438, split-K over `C·K·K`); (3) the K = 1 case: `const bool pointwise = KW == 1 && KH == 1 && ST_X == 1 && ST_Y == 1 && PD_X == 0 && PD_Y == 0;` (line 390) calls `cublasGemmEx` on the input as it is (line 429), because `B` is `X` itself (notes, "K = 1 special case").
- Precision: "Match im2col's F16 input precision, but expand patches only in shared memory and accumulate in F32." (line 409). Tensor-core and AMD MFMA/WMMA fragments come from `mma.cuh`; the same kernel has a SIMT fallback.

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/conv2d.cu

### D2. llama.cpp: the explicit path, where convolutions occur in LLM serving, and the Metal kernels

**Verdict: CONFIRMED.**

- Explicit path, `ggml/src/ggml.c` (last commit 2026-11-19): `// result: [N, OH, OW, IC*KH*KW]` for `ggml_im2col` (line 4471), so rows of the unfolded matrix are output pixels (the transpose of the book's `B`, like `cudnnIm2Col`, A4). It is stored in F16 unless the weights are BF16: `a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16`. `ggml_conv_2d` then multiplies and fixes the layout with a copy: `result = ggml_reshape_4d(..., a->ne[3]); // [OC, N, OH, OW]` and `result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2)); // [N, OC, OH, OW]` (lines 4679 to 4697). That copy is what the book's ordering avoids for one sample (p. 466); unrolling a whole batch at once makes it necessary (E2).
- Where it is used: multimodal encoders. `tools/mtmd/mtmd-graph.cpp` (last commit 2026-11-19), `mtmd_graph::build_inp()`: "build the input after conv2d (inp_raw --> patches)" with `ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1)`: a vision patch embedding, stride equal to the kernel size, no padding, so the unfolded matrix duplicates nothing (E1). `ggml.c` names the pattern `ggml_conv_2d_sk_p0` ("stride = kernel, padding 0"). Audio: `tools/mtmd/models/whisper-enc.cpp` runs `ggml_conv_1d_ph(ctx0, model.conv1d_1_w, inp, 1, 1)` then a stride-2 one ("stride-2 conv1d with padding 1"). The short causal convolutions of recurrent-hybrid models are `ch07-facts.md` D3.
- Metal, `ggml/src/ggml-metal/kernels/conv.metal` (last commit 2026-11-19): `kernel_im2col` (line 13, explicit unfold), `kernel_conv_2d` (line 164, direct), `kernel_conv_2d_mm` (line 278, implicit GEMM on `simdgroup_half8x8` / `simdgroup_float8x8` with split-K summed by `kernel_conv_2d_mm_reduce`), and under `GGML_METAL_HAS_TENSOR` `kernel_conv_2d_mm_tensor` (line 479), whose comment reads "conv_2d as a matmul on the tensor cores, like kernel_mul_mm" and "The patches of NRA output positions are gathered into threadgroup memory, the kernel rows are read from device memory." It calls `mpp::tensor_ops::matmul2d`, not MPP's `convolution2d` (C5). "Tensor cores" is llama.cpp's word; Apple's documentation names no such unit (`infra-facts.md` C10).

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml.c
- https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/mtmd-graph.cpp
- https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/models/whisper-enc.cpp
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/conv.metal

### D3. MLX on Metal: channels-last convolution, four algorithms, and an explicit unfold tiled by `maxBufferLength`

**Verdict: CONFIRMED.**

- Layout (MLX 0.32.4 docs, `mlx.core.conv2d`): input "Input array of shape `(N, H, W, C_in)`."; weight "Weight array of shape `(C_out, KH, KW, C_in)`." Both channels-last; the weight is still a `C_out × (KH·KW·C_in)` matrix without a copy, with the reduction index ordered channel-fastest instead of the book's channel-slowest.
- Dispatch, `mlx/backend/metal/conv.cpp` (last commit 2026-10-27), `dispatch_conv_2D_gpu` (line 1408): Winograd for 3 x 3, stride 1, `C % 32 == 0 && O % 32 == 0` and large enough inputs ("Only use winograd conv when having enough memory."); implicit GEMM (`implicit_gemm_conv_2D_gpu`, kernels `implicit_gemm_conv_2d_*` in `kernels/steel/conv/`) when channel counts suit the specialised kernel, padding input channels to a multiple of 16 when that pays; a general implicit GEMM otherwise; and "Direct to explicit gemm conv" as the last resort.
- The explicit path's own comment, which is the book's p. 467 cost meeting Metal's buffer cap (C2): "The full unfold buffer can exceed maxBufferLength, so unfold and gemm in row tiles reusing one buffer, keeping peak memory at a single tile." `max_unfold_rows` reads `d.mtl_device()->maxBufferLength()` and throws "A single unfolding row needs ... bytes, which exceeds the maximum Metal buffer size" when even one row does not fit (lines 34 to 50, 113 to 160). The kernel name is `naive_unfold_nd_`.

- https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.conv2d.html
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/conv.cpp
- https://github.com/ml-explore/mlx/tree/main/mlx/backend/metal/kernels/steel/conv

### D4. PyTorch: the CUDA fallback is §19.3 with explicit unfolding; MPS uses MPSGraph and its own kernels

**Verdict: CONFIRMED.**

- `aten/src/ATen/native/cuda/ConvolutionMM2d.cu` (`slow_conv2d_cuda`, last commit 2026-05-27): "For each elt in batch, do:" `at::native::im2col(...)` into `columns`, then `int64_t m = nOutputPlane; int64_t n = outputHeight * outputWidth; int64_t k = nInputPlane*kH*kW;` and `at::cuda::blas::gemm('n', 'n', n, m, k, ...)` ("note: this is a bit confusing because gemm assumes column-major matrices"). That is the book's `F` (`M × C·K·K`) times `B` (`C·K·K × H_out·W_out`), one sample at a time, row-major data through a column-major BLAS (`ch03-facts.md` D2). The unfold is skipped for the K = 1 case: `const bool requires_columns = (kW != 1 || kH != 1 || dW != 1 || dH != 1 || padH != 0 || padW != 0);` (line 156). This path is the `Slow2d` member of `ConvBackend` (`aten/src/ATen/native/ConvUtils.h`); `Cudnn` and `Miopen` are the library paths.
- Weights: `nn.Conv2d.weight` has shape "(out_channels, in_channels/groups, kernel_size[0], kernel_size[1])" (PyTorch 2.15 docs), the book's `F` layout, so the `M × (C·K·K)` view needs no copy in PyTorch either.
- MPS (`aten/src/ATen/native/mps/operations/Convolution.mm`, last commit 2026-11-07): 2D forward builds `[mpsGraph convolution2DWithSourceTensor:...]` (line 955) with `dataLayout` NCHW or NHWC following the tensor's memory format (lines 743 to 747). The same file holds PyTorch's own Metal paths for 3D and 1D convolution, with comments that restate p. 472 in measured form: "A 1x1x1 stride-1 no-pad conv is a per-voxel GEMM over channels; both the direct kernel and im2col (...) lose to a plain matmul, so peel it off first." and "conv1d_mpp reduces over all of C_in * k inside each tile, so a short output with a large reduction leaves most of the GPU idle, im2col + addmm splits the reduction. Measured cutoffs: the im2col copy pays off from a 2048-element reduction". I did not trace every 2D dispatch condition in this file.

- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ConvolutionMM2d.cu
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/ConvUtils.h
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/operations/Convolution.mm
- https://docs.pytorch.org/docs/2.15/generated/torch.nn.Conv2d.html

---

## Group E: The chapter's arithmetic, rechecked (ours, not vendor facts)

### E1. Worked numbers and the notes' listing errata

**Verdict: CONFIRMED.** Every printed number checks; every listing problem the notes flag reproduces. Script run 2026-11-20 (Python, exact integers).

| Book page | As printed | Recomputed | Status |
|---|---|---|---|
| 457, 465 | `Y` matrix rows `14 20 15 24` and `12 24 17 26` | Direct loop nest (Fig. 19.3) and `F·B` agree, same values | correct |
| 465 | `B` columns, e.g. col 0 `1 2 1 1, 0 2 0 3, 1 2 0 1` | All four columns match Fig. 19.8 as described in the notes | correct |
| 466 to 467 | 48 entries in `B` vs 27 inputs, 16/9 | 1.78 | correct |
| 467 | centre pixel 4 times, edges 2, corners 1 | `[[1,2,1],[2,4,2],[1,2,1]]`, total 16 | correct |
| 470 | `B[10,2]` → `X[n,2,2,0]` = 3 | (c, h, w) = (2, 2, 0), value 3 | correct (the text's "u is equal to 2" is the erratum the notes list) |
| 471 | Fig. 19.11 line 28 | Equals Eq. 19.5 linearised for every (u, v) of the example | correct |
| 467 | ratio approaches `K²` | K = 3: 7.91 at 32 x 32, 8.84 at 224, 8.96 at 1024; K = 5: 19.14, 24.12, 24.81 | correct |
| (notes) | LeNet C1 ≈ 19.1, C3 ≈ 12.8 | 19.14, 12.76; C5 gives 1.00 (one output pixel, nothing overlaps) | correct |
| (notes) | LeNet GEMM shapes | C1: `F` 6 x 25, `B` 25 x 784; C3: 16 x 150, 150 x 100; C5: 120 x 400, 400 x 1 | correct |
| 471 (notes) | `C·K·K = 25` with `TILE_WIDTH` 16 drops terms | `25/16` truncates to 1 phase: 16 terms used, 9 dropped; ceiling gives 2 | erratum reproduced |
| 461 (notes) | 28 x 28 output, `W_grid = 28/16` | 1 tile covers 16 of 28; ceiling gives 2 | erratum reproduced |

Two cases for the explorer that go beyond the book (ours): with `K = 1`, the ratio is exactly 1 at every size (`B` is `X` viewed as `C × H·W`); with stride equal to `K` and no padding (a vision patch embedding, D2), the ratio is `K²·(H/K)·(W/K)/(H·W) = 1` whenever `K` divides the image (14 on 224: 1.0; 16 on 224: 1.0). The duplication the chapter warns about comes from overlapping windows, which stride 1 maximises.

### E2. Sizing the explorer and the lab

**Verdict: CONFIRMED** (arithmetic; the limits it is compared with are cited).

| Layer (FP32; `B` for all N samples, stride 1, no padding) | `B` elements | `B` size | `X` size | over int32 element count? | over Metal's 256 MB floor (C2)? |
|---|---|---|---|---|---|
| C = 3, 224 x 224, K = 7, N = 1 | 6,986,028 | 26.6 MiB | 0.6 MiB | no | no |
| C = 64, 224 x 224, K = 3, N = 1 | 28,387,584 | 108.3 MiB | 12.2 MiB | no | no |
| C = 64, 224 x 224, K = 3, N = 8 | 227,100,672 | 866.3 MiB | 98.0 MiB | no | yes |
| C = 256, 56 x 56, K = 3, N = 32 | 214,990,848 | 820.1 MiB | 98.0 MiB | no | yes |

- With `int` indices, `B` for C = 64, K = 3, N = 1 passes 2^31 - 1 elements at a 1933 x 1933 input. cuDNN documents its own tensor cap at "2 Giga-elements" (A1); llama.cpp's launch checks `ggml_nelements(...) <= INT_MAX - 256`. The lab's sizes stay far below; the explorer should show the element count next to the ratio.
- Orientation for the lab (ours): with the milestone convention of row-major `float` matrices and explicit sizes, `F` (`M × C·K·K`, the weight array as stored) times one sample's `B` (`C·K·K × H_out·W_out`) gives that sample's `Y` already in NCHW (p. 466), so the reader's M2/M3 GEMM can be called unchanged once per sample. Unrolling the whole batch into one `C·K·K × N·H_out·W_out` matrix gives `Y` in `M × N × H_out × W_out` order (CNHW) and needs a permute afterwards: the copy llama.cpp makes with `ggml_cont` (D2) and PyTorch avoids by looping over samples (D4). The lab should do per-sample GEMMs or document the permute.
- Test sizes that exercise the book's omitted cases (ours): `C·K·K` not a multiple of the tile width (LeNet C1: 25), an output side not a multiple of 16 (LeNet C1: 28), and `M` not a multiple of 16 (LeNet C1: 6). The corrected kernels must pass all three; the uncorrected Fig. 19.11 fails the first.

---

## Book items with no vendor fact to check

- `TILE_WIDTH = 16` (256 threads) is a design choice, not a device limit (notes, "Dated or device-specific content"); it is within every backend's per-threadgroup limit (`ch05-facts.md` A5, B2, C3).
- LeNet-5's layer sizes (pp. 455 to 459) come from the 1998 LeNet paper, which the notes do not cite and this check did not read. They are used as shapes, not claims about devices.
- The p. 453 copyright line (2027) and the book's terminology drift ("filter bank", "unfolding" vs "unrolling") are editorial.

## Answers to the notes' portability questions

The notes marked these `[unverified]`; each now has a documented answer.

| Notes item | Answer |
|---|---|
| `__global__`, `blockIdx`, `threadIdx`, `dim3`, 3D `<<<>>>` on HIP | Same spelling (`ch05-facts.md` B3, B4); grid limits differ (B2) |
| The same on Metal | `kernel` function with `[[threadgroup_position_in_grid]]` / `[[thread_position_in_threadgroup]]` as `uint3`, dispatched with `dispatchThreadgroups` or `dispatchThreads` (`ch05-facts.md` C4, C5); no documented grid maximum (C1) |
| `__shared__` and `__syncthreads()` | HIP: same; Metal: `threadgroup` arrays and `threadgroup_barrier(mem_flags::mem_threadgroup)` (`ch05-facts.md` B4, C1, C2) |
| Cost of integer `/` and `%` | CUDA "up to 20 instructions" (A8); AMD "many more hardware cycles than multiplication" (B5); Apple, 2016 talk only (C6) |
| cuBLAS and cuDNN counterparts | hipBLAS and MIOpen (hipDNN early) on AMD; `MPSMatrixMultiplication`, MPSGraph `convolution2D` / `imToCol`, `MPSCNNConvolution`, MPP `convolution2d` on Apple (`ch15-facts.md` B6, C6; B3, B4, C3 to C5) |
| Exercise 4: 32 banks of 4 bytes; several threads reading one word are served together | Both confirmed for NVIDIA: 32 banks (`infra-facts.md` A10) and Nsight Compute: "A shared memory request for a warp does not generate a bank conflict between two threads that access any address within the same 32-bit word" (`ch06-facts.md` A12). AMD: 32 or 64 banks (`infra-facts.md` B10). Metal: nothing documented (`infra-facts.md` C13) |

---

## Summary of corrections

Every PARTLY item and every sub-claim that could not be verified, one line each:

- A1: §19.4's cuDNN API (`cudnnConvolutionForward`, convolution and filter descriptors) is deprecated since cuDNN 9.0, kept as a legacy library; NVIDIA now recommends the graph API through the C++/Python frontend, with the C backend for C-only use. Thread safety holds only "so long as the threads do not share the same cuDNN handle simultaneously"; tensors are capped at "2 Giga-elements" and some routines reject unusual strides.
- A2: Table 19.1's names hold, but padding is zero rows and columns in the convolution's arithmetic ("implicitly concatenated onto the top and onto the bottom"), which changes the output size; it is not an alignment device. The missing formula is `outputDim = 1 + (inputDim + 2*pad - ((filterDim-1)*dilation + 1)) / convolutionStride`. Dilation and groups are parameters the table omits.
- A3 (status): cuDNN has eight forward algorithms, not three families; `CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD` "is not supported on GPUs based on the NVIDIA Hopper GPU architecture and later".
- A4: cuDNN's GEMM family has an explicit variant (large workspace), an implicit one, and one that precomputes indices in a workspace; there was a public `cudnnIm2Col` (deprecated 9.0) producing the transpose of the book's `B`. "Similar to Tan et al." could not be verified.
- A5: cuDNN documents NCHW/NHWC conversion only around Tensor Core use ("if and only if Tensor Cores are requested and are actually used"), not as a general final transposition; "Tensor Cores require that the tensors be in the NHWC data layout".
- A6: MXNet retired in 2023 (Attic, 2024); Caffe's last commit is from 2020. PyTorch's `torch.backends.cudnn.allow_tf32` still defaults to True but "is going to be deprecated" in favour of `torch.backends.cudnn.conv.fp32_precision` (from PyTorch 2.9; default not stated on the page).
- A7 (doc slip): CUTLASS's convolution page gives the activation matrix as "NHW -by- RSC" against an "NPQ -by- K" output, and an `f(p, r)` with a flipped filter and `+ pad_h`; use cuDNN's formula.
- A9: the 2012 network trained "five to six days" (not a week) on two GPUs; the 15.3% vs 26.2% belongs to "a variant of this model" in ILSVRC-2012; the proceedings landing page shows an older abstract with different numbers. The cuda-convnet URL redirects to the Google Code archive.
- A10 (could not verify): reference [10]'s DOI resolves to the ACM Digital Library, which refused a scripted fetch; the paper was not read.
- B2: HIP documents at most 2^32 - 1 threads per grid (CUDA documents no such cap), grid y and z up to 2^16 - 1 like CUDA, and the runtime reports `maxGridSize` clamped to `INT32_MAX`. Whether the table's x limit counts blocks or threads, and where an oversized launch is rejected, could not be verified.
- B3 (could not verify): no general thread-safety statement for MIOpen was found; the header says profiling timings per handle are "not thread-safe". MIOpen's Find step "is mandatory before using miopenConvolutionForward()" (or use immediate mode); CHWN is "Not supported".
- B4: HIPIFY maps `cudnnConvolutionForward` to `miopenConvolutionForward` but lists no MIOpen counterpart for `cudnnSetConvolution2dDescriptor`, `IMPLICIT_PRECOMP_GEMM`, `FFT_TILING` or `WINOGRAD_NONFUSED`; hipDNN exists but is "in the early stages of development".
- C1 (could not verify): Apple documents no maximum compute grid size; only mesh and object shader grids have listed maxima.
- C3 (could not verify): whether MPSGraph `convolution2D` flips the filter is not documented; PyTorch passes `nn.Conv2d` weights unflipped. Apple's im2col header misstates NCHW as "batch x channels x width x height", and the convolution header misstates dilated width as "(dilationRateInX-1)*kernelWidth+1".
- C4 (caveat): `MPSCNNConvolution` uses FP16 weights unless the data source reports FP32 ("When kernelWeightsDataType is unimplemented the kernel will use float16 precision"); it takes `MPSImage`s and OHWI weights only.
- C5 (could not verify): MPP `convolution2d` documents no rule for where the window sits relative to `set_offsets`; PyTorch's kernel records a measured rule. The op is NHWC/HWIO, groups 1, whole-threadgroup scope only.
- C6: Apple's only statement on integer division cost is a 2016 talk ("most of them do not have integer division or modulus instructions"); nothing in MSL 4.1.

Status changes worth knowing even where the verdict is CONFIRMED:

- A7, D1, D3, D4: CUTLASS, llama.cpp (CUDA, HIP, Metal), MLX and PyTorch all implement §19.3. Implicit GEMM is the main path in llama.cpp's CUDA backend and MLX; explicit im2col survives as the generic path in llama.cpp's graph (`ggml_conv_2d`), MLX's last resort (tiled to fit `maxBufferLength`) and PyTorch's slow CUDA fallback; all of them special-case 1 x 1 convolutions as a plain GEMM.
- D2: in LLM serving, 2D convolutions appear in vision patch embeddings (stride = kernel, so im2col duplicates nothing) and 1D convolutions in audio encoders; neither is in the decoder's token loop.

## Where the backends really differ

One line per cell, documented facts only unless marked "ours". Nothing here was run on a device.

| Feature | CUDA | HIP (ROCm 10.2) | Metal (MSL 4.1, macOS 27 SDK) |
|---|---|---|---|
| Kernel code of Figs. 19.7 and 19.11 (corrected) | As in the book, indices linearised by hand | Same source through `pmpp/vendors/hip.h` | MSL rewrite: `threadgroup` arrays, `threadgroup_barrier`, `uint3` positions |
| Grid limits | x up to 2^31 - 1 blocks, y and z up to 65535; no per-grid thread cap documented | x 2^32 - 1, y and z 2^16 - 1, at most 2^32 - 1 threads per grid; `maxGridSize` reported up to `INT32_MAX` | No compute-grid maximum documented; position attributes are 32-bit `uint` |
| Edge tiles (Fig. 19.5 fix) | Ceiling division plus bounds checks | Same | `dispatchThreads` with nonuniform threadgroups (Apple4 and later), or ceiling plus checks |
| One buffer for the unrolled `B` | Not checked here | Not checked here | `maxBufferLength`, documented floor 256 MB |
| Integer `/` and `%` cost | "up to 20 instructions"; literal powers of two become shifts | "many more hardware cycles than multiplication" | 2016 talk: no integer divide instruction on most GPUs; use literals or function constants |
| Vendor convolution library | cuDNN 9.28.1: graph API (C++/Python frontend, C backend); legacy `cudnnConvolutionForward` deprecated since 9.0 | MIOpen 3.6.2: `miopenConvolutionForward` after Find, or immediate mode; hipDNN early | MPSGraph `convolution2D`; `MPSCNNConvolution` on `MPSImage`; MPP `convolution2d` in shaders (Metal 4) |
| Forward algorithms | 8: GEMM, IMPLICIT_GEMM, IMPLICIT_PRECOMP_GEMM, DIRECT, FFT, FFT_TILING, WINOGRAD (not Hopper and later), WINOGRAD_NONFUSED | 5: GEMM, Direct, FFT, Winograd, ImplicitGEMM | None exposed or documented |
| Choosing the algorithm | Caller passes `algo` (legacy) or heuristics pick an engine; PyTorch `cudnn.benchmark` | Find benchmarks solvers, caches on disk; immediate mode uses a database and heuristics | Not exposed |
| Explicit im2col as an API | `cudnnIm2Col` (deprecated 9.0), rows = N·P·Q | None in `miopen.h` | MPSGraph `imToCol` (macOS 14 and later) |
| Layouts | NCHW, NHWC, NCHW_VECT_C, any strides (some routines refuse); Tensor Cores require NHWC, converted internally | NCHW and NHWC fully supported; CHWN not supported | MPSGraph NCHW or NHWC (weights OIHW or HWIO); MPSCNN weights OHWI; MPP NHWC and HWIO only |
| Filter flip | Cross-correlation mode in its formulas; frameworks select it | "MIOpen use cross-correlation" | Not documented; PyTorch passes weights unflipped to MPSGraph |
| Precision traps against an FP32 reference | PyTorch lets cuDNN convolutions use TF32 by default (`allow_tf32` True) | Not checked here | `MPSCNNConvolution` weights FP16 unless opted out; MPP `relaxed_precision` off by default; MSL fast math (`ch05-facts.md` C7) |
| GEMM for the explicit path | cuBLAS, column-major | hipBLAS, column-major | `MPSMatrixMultiplication`, row-major |
| Implicit GEMM in real code | CUTLASS (NHWC, fast divmod); llama.cpp `conv2d_implicit_gemm_f16` (NCHW, fastdiv) | Same llama.cpp file with MFMA/WMMA fragments | llama.cpp `kernel_conv_2d_mm` (`simdgroup_half8x8`); MLX `implicit_gemm_conv_2d`; PyTorch `conv3d_mpp` (MPP) |
