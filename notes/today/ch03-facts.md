# Chapter 3 fact check: CUDA, HIP, Metal

Checked on 2026-11-20 against the live sources listed under each item.

How to read this file:

- **Verdict** is one of CONFIRMED, WRONG, PARTLY, COULD NOT VERIFY. Where an item has several sub-claims, the verdict is for the item as a whole and the text says which sub-claim failed.
- **Quote** is copied from the source (HTML docs flattened to text, GitHub raw files, Apple's documentation JSON, PDF text). Line breaks inside quotes are removed.
- "Book" means `notes/ch03.md` with its page numbers. Nothing here is taken from memory of the book.
- Versions the sources reported about themselves on the check date: CUDA Programming Guide "v13.5" (pages last updated Oct 31, 2026), CUDA Runtime API reference (Nov 02, 2026), "cuBLAS 13.5 documentation"; "HIP 7.17.0 Documentation" and "rocBLAS 5.8.0 Documentation" (ROCm 10.2.0), HIP and CLR source at tag `therock-10.2`; Metal Shading Language Specification "Version 4.1", dated 2026-06-04; Metal Feature Set Tables dated October 23, 2026; local macOS SDK 27.0 headers. Project sources read from the default branch: llama.cpp `master` at `b94959a` (2026-11-19), MLX `main` at `8f0f391` (2026-11-20), PyTorch `main` at `538adf4` (2026-11-20), PyTorch docs 2.15.
- Not checked: anything by running code on a GPU. Two things were compiled on this machine and are labelled where they appear: host-only C++ programs for the cast precedence erratum and the effect of truncation (A9, compiled and run with Apple clang 21.0.0) and a Metal kernel snippet (C6, compiled only, `xcrun metal`, "Apple metal version 32023.921").

Already settled in `notes/today/ch05-facts.md` and not repeated here (cited as "ch05 A5" and so on):

| Topic | Item |
|---|---|
| CUDA block and grid limits per compute capability (1024 threads per block; block at most 1024 x 1024 x 64; grid at most 2^31 - 1 in x and 65535 in y and z) | ch05 A5 |
| `cudaDeviceProp` and `hipDeviceProp_t` field names (`maxThreadsPerBlock`, `maxThreadsDim`, `maxGridSize`) | ch05 A1, B2 |
| HIP accepts `kernel<<<grid, block>>>(args)` directly | ch05 B3 |
| HIP spells `__global__`, `threadIdx`, `blockIdx`, `blockDim`, `gridDim` as CUDA does, and needs `hip_runtime.h` | ch05 B4 |
| llama.cpp builds its HIP backend from the CUDA `.cu` files | ch05 B9 |
| Metal threadgroup size limits and pipeline properties | ch05 C3 (one sub-claim corrected in C9 below) |
| MSL kernel attributes `thread_position_in_grid` and friends, their types and the formula relating them | ch05 C4 |
| `dispatchThreads` versus `dispatchThreadgroups`, nonuniform threadgroups | ch05 C5 |
| Shared storage mode and unified memory | ch05 C6 |
| MSL math mode is not IEEE-strict by default | ch05 C7 |
| `GPUStartTime` / `GPUEndTime` for timing | ch05 C9 |
| How llama.cpp and MLX hold their Metal kernels | ch05 C10 |

There is no `notes/today/infra-facts.md` on the check date.

Totals: 24 CONFIRMED, 1 WRONG, 5 PARTLY, 2 COULD NOT VERIFY (32 items). The WRONG item is a correction to ch05-facts C3, not to the book.

---

## Group A: CUDA, and the C++ underneath it

### A1. `dim3` and the execution configuration (book pp. 46 to 47)

**Verdict: CONFIRMED.** The first two `<<< >>>` parameters are the grid size in blocks and the block size in threads, both `dim3`; unspecified components are 1; a plain integer is accepted for a 1D launch.

Quotes:

- "The values used to specify and retrieve the kernel configuration for the grid and blocks along the x, y, and z dimensions are of type dim3. ... Both dim3 and uint3 are trivial structures consisting of three unsigned values named x, y, and z. In C++11 and later, the default value of all components of dim3 is 1."
- "The execution configuration is specified by inserting an expression in the form <<<grid_dim, block_dim, dynamic_smem_bytes, stream>>> between the function name and the parenthesized argument list". "grid_dim is of type dim3 and specifies the dimension and size of the grid, such that grid_dim.x * grid_dim.y * grid_dim.z equals the number of blocks being launched." "block_dim is of type dim3 and specifies the dimension and size of each block, such that block_dim.x * block_dim.y * block_dim.z equals the number of threads per block."
- "The first two parameters to the triple chevron notation are the grid dimensions and the thread block dimensions, respectively. When using 1-dimensional thread blocks or grids, integers can be used to specify dimensions."
- 2D example as printed: `dim3 grid(16,16);` / `dim3 block(8,8);` / `MatAdd<<<grid, block>>>(A, B, C);`
- "The arguments for the execution configuration are evaluated before the arguments for the actual function."

Two notes for the lessons:

- The book sizes the grid with `ceil(n/256.0)` (p. 46). NVIDIA's guide does not print that form. It uses integer arithmetic, `int blocks = (vectorLength + threads-1)/threads;`, and now also a library helper: "The CUDA Core Compute Library (CCCL) provides a convenient utility, cuda::ceil_div, for doing this ceiling divide to calculate the number of blocks needed for a kernel launch. This utility is available by including the header <cuda/cmath>." The book's exercise 3 uses the integer form `(N - 1)/16 + 1`. The `ceil` form relies on an implicit double to unsigned conversion into `dim3`; I did not find a vendor page that shows it, and I did not compile it with nvcc.
- I could not read NVIDIA's `dim3` constructor itself (the toolkit header is not published online). The statements above are from the guide's text.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (sections 5.4.2.2 and 5.4.3)
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html (sections 2.1.2.2 and 2.1.2.3)

### A2. Built-in variables `gridDim`, `blockDim`, `blockIdx`, `threadIdx` (book pp. 45 to 48)

**Verdict: CONFIRMED.** Names, meanings and ranges are as the book states. The two index variables have a different type from the two size variables.

Quotes:

- "Built-in device-only variables: dim3 gridDim: contains the dimensions of the grid, namely the number of thread blocks, along the x, y, and z dimensions. dim3 blockDim: contains the dimensions of the thread block, namely the number of threads, along the x, y, and z dimensions. uint3 blockIdx: contains the block index within the grid, along the x, y, and z dimensions. uint3 threadIdx: contains the thread index within the block, along the x, y, and z dimensions."
- "Each of these intrinsics is a 3-component vector with a .x, .y, and .z member. Dimensions not specified by a launch configuration will default to 1. threadIdx and blockIdx are zero indexed. That is, threadIdx.x will take on values from 0 up to and including blockDim.x-1."
- "This computation of workIndex is very common for 1-dimensional parallelizations. Expanding to two or three dimensions often follows the same pattern in each of those dimensions." (after `int workIndex = threadIdx.x + blockDim.x * blockIdx.x;`)
- On what the extra dimensions buy you: "The use of multi-dimensional thread blocks and grids is for convenience only and does not affect performance. The threads of a block are linearized predictably: the first index x moves the fastest, followed by y and then z. This means that in the linearization of a thread indices, consecutive values of threadIdx.x indicate consecutive threads, threadIdx.y has a stride of blockDim.x, and threadIdx.z has a stride of blockDim.x * blockDim.y. This affects how threads are assigned to warps".

Consequences:

- All four are unsigned. The book stores `row` and `col` in `int` (Figs. 3.4, 3.8, 3.11) and in `unsigned int` in exercise 3. For the blur kernel the signed form matters, because `curRow = row + blurRow` must be able to go negative before the `curRow >= 0` test (this is plain C++ arithmetic, my reasoning, not a vendor statement).
- The book's label order (z, y, x) on p. 48 is a convention of the book. The vendor order everywhere is x first.
- These variables are a property of the per-thread (SIMT) kernels the chapter teaches. See A10 for the second kind of kernel CUDA now has, where they do not exist.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (section 5.4.2.2)
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html (section 2.1.2.3)
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html (section 2.3.2)

### A3. Size limits: 1024 threads per block, grid ranges (book p. 47)

**Verdict: PARTLY.** The numbers are confirmed and settled in ch05 A5. The book's sentence that the 1024 threads can be distributed across the three dimensions in any way is too loose: each dimension has its own cap.

- Confirmed (ch05 A5, Table 33 of the compute capabilities appendix, same value for every listed compute capability 7.5 through 12.x): 1024 threads per block; grid x at most 2^31 - 1; grid y and z at most 65535. The guide's current wording keeps the book's hedge: "On current GPUs, a thread block may contain up to 1024 threads."
- Missing from the book: "Maximum x- or y-dimensionality of a thread block: 1024" and "Maximum z-dimension of a thread block: 64" (ch05 A5). A block of (1, 1, 128) has 128 threads and is still illegal. All four of the book's examples, (512, 1, 1), (8, 16, 4), (32, 16, 2) allowed and (32, 32, 2) rejected, come out the same under both rules, so the book's examples can be used as printed.
- The portable way to get the numbers is to read `maxThreadsPerBlock`, `maxThreadsDim` and `maxGridSize` (ch05 A1).

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html (Table 33)
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html (section 2.1.2.2.1)

### A4. What happens when a launch exceeds the limits

**Verdict: CONFIRMED** (the book does not say; the launch explorer and the labs need it). The launch fails with `cudaErrorInvalidConfiguration`, and because launches are asynchronous you see it only when you ask for the error state.

Quotes:

- "The function call fails if grid_dim or block_dim exceeds the maximum sizes allowed for the device, as specified in Compute Capabilities, or if dynamic_smem_bytes is greater than the available shared memory after accounting for statically allocated memory."
- "enumerator cudaErrorInvalidConfiguration = 9 This indicates that a kernel launch is requesting resources that can never be satisfied by the current device. Requesting more shared memory per block than the device supports will trigger this error, as will requesting too many threads or blocks. See cudaDeviceProp for more device limitations."
- "The CUDA runtime maintains a cudaError_t state for each host thread. The value defaults to cudaSuccess and is overwritten whenever an error occurs. cudaGetLastError returns current error state and then resets it to cudaSuccess. Alternatively, cudaPeekAtLastError returns error state without resetting it."
- "Kernel launches are asynchronous with respect to the host thread."

A `<<< >>>` statement returns nothing, so a bad configuration is silent unless the host calls `cudaGetLastError()` afterwards. PyTorch does this after every launch (D6).

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (section 5.4.3)
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html (sections 2.1.2.2.1 and 2.1.7.1)

### A5. Surplus threads and the range guard (book pp. 50, 52 to 55)

**Verdict: CONFIRMED.** NVIDIA documents the same pattern: round the grid up to whole blocks and make out-of-range threads do nothing.

Quotes:

- "To make the kernel handle any vector length, we can add checks that the memory access is not exceeding the bounds of the arrays as shown below, and then launch one thread block which will have some inactive threads."
- "With the above kernel code, more threads than needed can be launched without causing out-of-bounds accesses to the arrays. When workIndex exceeds vectorLength, threads exit and do not do any work. Launching extra threads in a block that do no work does not incur a large overhead cost, however launching thread blocks in which no threads do work should be avoided."
- The guide's own 2D kernel uses the book's mapping, x for the column and y for the row, with the same guard: `int myCol = blockDim.x * blockIdx.x + threadIdx.x;` / `int myRow = blockDim.y * blockIdx.y + threadIdx.y;` / `if( myRow < m && myCol < m )`.

The second quoted sentence is the one design rule the coverage explorer should respect: a rounded-up grid never contains a wholly idle block, and the book's 62 x 76 example has none (areas 1 to 4 on pp. 54 to 55 all have working threads).

- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html (section 2.1.2.3.1)
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html (section 2.3.4.1.1)

### A6. "Programmers need to explicitly linearize ... in the current CUDA C++" (book pp. 50 to 52)

**Verdict: PARTLY.** Row-major flattening by hand is still what NVIDIA's guide and the inference projects do. It is no longer the only option: nvcc now accepts C++ up to C++23, and NVIDIA's standard library ships `mdspan` for device code.

What still holds:

- The guide flattens by hand, with a macro: "/* macro to index a 1D memory array with 2D indices in row-major order */ /* ld is the leading dimension, i.e. the number of columns in the matrix */ #define INDX( row, col, ld ) ( ( (row) * (ld) ) + (col) )".
- "In both C++ and Python, multi-dimensional array ordering, the last index moves the fastest".
- llama.cpp, MLX and PyTorch all index raw pointers with a computed flat offset (D1 to D6).

What has changed since the sentence was written:

- "nvcc processes CUDA and device code according to the following specifications: C++03 ... C++11 ... C++14 ... C++17 (ISO/IEC 14882:2017), --std=c++17 flag. C++20 (ISO/IEC 14882:2020), --std=c++20 flag. C++23 (ISO/IEC 14882:2024), --std=c++23 flag."
- "CUDA provides an implementation of the C++ Standard Library (STL), called libcu++. ... The functionalities are available on both host and device. ... Provides C++17 backports of C++ Standard Library features available in recent standard versions, including C++20, C++23, and C++26."
- CCCL page for `<cuda/std/mdspan>`: "All features of <mdspan> are made available in C++17 onwards". "The C++23 multidimensional operator[] is replaced with operator() in previous C++ standards". "On device no exceptions are thrown in case of a bad access." "operator[] is disabled even in C++23 mode because it is not yet fully supported by the nvcc compiler."

So a kernel can write `m(row, col)` over a flat buffer through `cuda::std::mdspan`, and the layout underneath is still the row-major arithmetic the chapter teaches. The CCCL page I read is the "unstable" documentation branch; I did not check which toolkit release first shipped it, and I did not compile it. The labs should keep the explicit `row*width + col`, which is also the only form that is the same source on HIP and translates directly to MSL.

- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html (sections 2.3.4.1.1 and 2.3.4.2.2)
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html (sections 5.3 and 5.3.6)
- https://nvidia.github.io/cccl/unstable/libcudacxx/standard_api/container_library/mdspan.html

### A7. Column-major libraries (book pp. 51 to 52)

**Verdict: CONFIRMED.** The book's warning that libraries built for FORTRAN callers use column-major storage applies to NVIDIA's own BLAS.

Quotes (cuBLAS 13.5):

- "For maximum compatibility with existing Fortran environments, the cuBLAS library uses column-major storage, and 1-based indexing. Since C and C++ use row-major storage, applications written in these languages can not use the native array semantics for two-dimensional arrays. Instead, macros or inline functions should be defined to implement matrices on top of one-dimensional arrays."
- "#define IDX2C(i,j,ld) (((j)*(ld))+(i))", where "ld refers to the leading dimension of the matrix, which in the case of column-major storage is the number of rows of the allocated matrix".

`IDX2C` is the book's exercise 4b formula (column times height plus row). How llama.cpp hands its row-major data to this library without copying is in D2.

- https://docs.nvidia.com/cuda/cublas/index.html (section 1.1 "Data Layout")

### A8. Grayscale constants (book p. 52 text versus p. 53 listing)

**Verdict: PARTLY.** The constants in the book's text, 0.299, 0.587, 0.114, are the ITU-R BT.601 luma coefficients. The listing's 0.21, 0.71, 0.07 are not the coefficients of either ITU recommendation I read; they are close to BT.709 rounded, but BT.709's green coefficient rounds to 0.72, and the listing's three sum to 0.99.

Quotes:

- ITU-R BT.601-7, luminance equation: "EY = 0.299 ER + 0.587 EG + 0.114 EB" (primes and subscripts flattened by the PDF text extraction).
- ITU-R BT.709-6, row "Derivation of luminance signal EY": coefficients 0.2126 (ER), 0.7152 (EG), 0.0722 (EB). The text extraction drops the equals and plus signs of this row, so only the three numbers and their symbols are quoted.
- torchvision, `rgb_to_grayscale`: `l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)` (the v2 version: `l_img = r.mul(0.2989).add_(g, alpha=0.587).add_(b, alpha=0.114)`).
- Apple's Metal grayscale sample: "The kernel calculates the grayscale value by applying the Rec. 709 luma coefficients to the color pixel's RGB components" with `half grayValue = dot(colorValue.rgb, kRec709LumaCoefficients);`. The page does not print the values of that constant.

For the lab: use 0.299, 0.587, 0.114. They are the book's own stated formula, an exact published standard, they sum to 1.0 (see A9 for what truncation then does), and they match what the reader gets from `torchvision.transforms.functional.rgb_to_grayscale` to within the fourth decimal of the red coefficient (0.2989 against 0.299). State that the listing on p. 53 prints different numbers.

- https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.601-7-201103-I!!PDF-E.pdf
- https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.709-6-201506-I!!PDF-E.pdf
- https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py (line 160)
- https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/functional/_color.py (line 42)
- https://developer.apple.com/documentation/metal/combining-blit-and-compute-operations-in-a-single-pass

### A9. The cast on line 19 of Fig. 3.4 (book p. 53; notes, "Unclear in the PDF" item 1)

**Verdict: CONFIRMED** that the listing as printed is wrong. `(unsigned char) 0.21f*r+0.71f*g+0.07f*b` casts only `0.21f`, which becomes 0, so the red term is always 0.

From the C++ grammar (working draft, which matches C++17 on these two productions):

- [expr.cast]: "cast-expression: unary-expression | ( type-id ) cast-expression". The operand of a cast is a cast-expression, which cannot contain a top-level `*` or `+`.
- [expr.mul]: "multiplicative-expression: pm-expression | multiplicative-expression * pm-expression". A cast-expression is an operand of `*`, so the cast binds tighter than the multiplication.
- [conv.fpint]: "A prvalue of a floating-point type can be converted to a prvalue of an integer type. The conversion truncates; that is, the fractional part is discarded. The behavior is undefined if the truncated value cannot be represented in the destination type."

So the printed line parses as `((unsigned char)0.21f) * r + 0.71f*g + 0.07f*b`: `(unsigned char)0.21f` is 0, `0 * r` is the `int` 0, the rest is a `float`, and the assignment to `Pout[grayOffset]` truncates that float. The stored value is `trunc(0.71*g + 0.07*b)`.

Compiled and run on this machine (host C++ only, Apple clang 21.0.0, `-std=c++17 -Wall -Wextra`, no warning emitted):

| r, g, b | As printed | Sum parenthesised |
|---|---|---|
| 255, 0, 0 | 0 | 53 |
| 200, 100, 50 | 74 | 116 |

`static_assert`s in the same file confirm the types: `decltype((unsigned char) 0.21f*r)` is `int`, and the whole unparenthesised expression is `float`. The compiler gives no diagnostic, which is why this makes a good bug-hunt question: a pure red image comes out black.

The third sentence of [conv.fpint] matters for the corrected line too: converting a float outside 0 to 255 to `unsigned char` is undefined behaviour in C++. With coefficients that sum to at most 1.0 and inputs in 0 to 255 the value stays in range. The blur kernel's `(unsigned char)((float)pixVal/pixels)` (Fig. 3.8 line 22) is correctly parenthesised and is an average of bytes, so it is in range as well.

Truncation also makes the result sensitive to the order of the float additions. In a second host test here (same compiler, CPU only), computing `(0.299f*r + 0.587f*g) + 0.114f*b` against `0.299f*r + (0.587f*g + 0.114f*b)` over 140,608 sampled colours changed the stored byte, by one grey level, for 40 of them. A GPU compiler may reorder or fuse these operations (MSL does by default, ch05 C7), so the grayscale lab must compare against its CPU reference with a tolerance of one level, not for equality. In the same test a white pixel (255, 255, 255) gave 255 with the BT.601 constants and 252 with the listing's.

- https://eel.is/c++draft/expr.cast
- https://eel.is/c++draft/expr.mul
- https://eel.is/c++draft/conv.fpint

### A10. Is one thread per data element still CUDA's model? "Tile" now has two meanings

**Verdict: CONFIRMED,** with an addition the book does not have. The thread-per-element model of the chapter is what the guide calls SIMT kernels, and it is unchanged. The v13.5 guide also documents a second model, tile programming, in which the programmer does not choose a block size or write per-thread index arithmetic.

Quotes:

- The guide's chapter 2.3 is now titled "Writing SIMT Kernels" and opens "This chapter covers some more details of writing kernels using the SIMT programming model".
- "In addition to the SIMT model described in the preceding sections, CUDA supports a tile programming model. In tile programming, the programmer writes code at the level of an entire thread block, describing operations on multidimensional collections of data called tiles. The compiler maps these operations to the individual threads of the block."
- "Tile kernels are launched on a grid of blocks ... Each block executes the tile kernel and can query its position within the grid to determine which portion of the data it is responsible for. The programmer specifies only the grid dimensions; the number of threads per block is determined by the compiler".
- "For __tile_global__ kernels, the block_dim must be 1 because the compiler will decide how many threads to schedule when launching the tile kernel."
- Of `gridDim`, `blockDim`, `blockIdx`, `threadIdx`: "These variables are not supported in tile code. To retrieve the block ID or grid size in tile code, use the cuda::tiles::bid() and cuda::tiles::num_blocks() APIs."
- The guide itself warns "It is important not to confuse blocks" with tiles: blocks are units of execution, tiles are units of data.

Why it is recorded here: the book says the matmul mapping "divides P into tiles; each block computes one tile" (p. 61), and chapter 5 uses "tiled" for the shared-memory algorithm. Neither is NVIDIA's "tile kernel". A lesson that uses the word should say which meaning it has. I read only the overview (section 1.2.2.3) and the mentions in the language appendices, not the full chapter 2.4 "Writing Tile Kernels"; nothing in the chapter 3 labs uses it, and I found no HIP or Metal counterpart in the pages read for this file (C8 describes a different Metal feature with a similar aim).

- https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html (section 1.2.2.3)
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html (section 2.3)
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (sections 5.4.2.2 and 5.4.3)
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-tile-kernels.html (exists; not read in full)

### A11. Memory Space sidebar: "at least 4G byte-sized locations" (book p. 51)

**Verdict: COULD NOT VERIFY,** and nothing depends on it. This is a general statement about computers, not a vendor fact, and no vendor page I read states it. The related facts that a lesson might want are checkable:

- The amount of device memory is a per-device property (`totalGlobalMem`, ch05 A1), and by the project's ground rules it must come from `pmpp.py doctor`, not from a lesson.
- A flat index held in a 32-bit `int`, as in every listing of the chapter, limits an array to 2^31 - 1 elements. Real code deals with this explicitly. PyTorch's elementwise launcher asserts it: `TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());` (D6). llama.cpp computes flat offsets in 64 bits: `const int64_t dst_idx = i3 * (ne0 * ne1 * ne2) + i2 * (ne0 * ne1) + i1 * ne0 + i0;` (D1). MLX switches to a 64-bit offset when `out.data_size() > UINT32_MAX` (D4).
- Metal's counterpart to "how big can one array be" is `MTLDevice.maxBufferLength`: "The largest amount of memory, in bytes, that a GPU device can allocate to a buffer instance." "The property's value is at least 256 MB (268,435,456 bytes)."

- https://developer.apple.com/documentation/metal/mtldevice/maxbufferlength

---

## Group B: HIP / ROCm

### B1. `dim3` on HIP

**Verdict: CONFIRMED.** Same name, same three fields, same default of 1, and the constructor is not `explicit`, so the book's host lines compile unchanged.

Header at tag `therock-10.2`:

```c
typedef struct dim3 {
  uint32_t x;  ///< x
  uint32_t y;  ///< y
  uint32_t z;  ///< z
#ifdef __cplusplus
  constexpr __host__ __device__ dim3(uint32_t _x = 1, uint32_t _y = 1, uint32_t _z = 1)
      : x(_x), y(_y), z(_z) {};
#endif
} dim3;
```

Docs: "dim3 is a special three-dimensional unsigned integer vector type that is commonly used to specify grid and group dimensions for kernel launch configurations. Its constructor accepts up to three arguments. The unspecified dimensions are initialized to 1."

The docs' own launches wrap the sizes explicitly, `example_kernel<<<dim3(gridSize), dim3(blockSize), 0, 0>>>(a, N);`, but a bare integer converts through the same constructor (my reading of the header; not compiled here).

- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/hip_runtime_api.h (lines 1443 to 1451)
- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html (section "dim3")

### B2. Types of the index built-ins on HIP

**Verdict: PARTLY.** The names and the `.x .y .z` members are the same as CUDA's (ch05 B4). The documented types are not: CUDA says `threadIdx` and `blockIdx` are `uint3`; AMD's docs say all four are `dim3`; AMD's header makes each one its own struct type that converts to `dim3`.

- CUDA: "uint3 blockIdx", "uint3 threadIdx", "dim3 gridDim", "dim3 blockDim" (A2).
- HIP docs: "These built-ins are of type dim3, and are constant for each thread, but differ between the threads or blocks, and are initialized at kernel launch."
- HIP header (`amd_hip_runtime.h`, `therock-10.2`): four types `__hip_builtin_threadIdx_t`, `__hip_builtin_blockIdx_t`, `__hip_builtin_blockDim_t`, `__hip_builtin_gridDim_t`, each with `unsigned int` properties `x`, `y`, `z` backed by compiler builtins (`__builtin_amdgcn_workitem_id_x()`, `__builtin_amdgcn_workgroup_id_x()`, `__builtin_amdgcn_workgroup_size_x()`, `__ockl_get_num_groups(0)`), and each with `__device__ operator dim3() const { return dim3(x, y, z); }`. The variables are declared `extern const __device__ __attribute__((weak)) __hip_builtin_threadIdx_t threadIdx;` and so on.

What this means in practice: every expression in the chapter (`blockIdx.y*blockDim.y + threadIdx.y`) uses the members and is identical on both. Code that copies a whole built-in into a `uint3`, which is legal CUDA, has no matching conversion in the HIP header (my inference from the header; not compiled). Keep to the members.

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html (section "Index built-ins")
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/amd_hip_runtime.h (lines 247 to 339)

### B3. Block and grid limits on AMD

**Verdict: PARTLY.** Limits exist and the runtime enforces them, but the current HIP documentation (7.17.0) no longer states any number. The numbers below come from an older FAQ and from the runtime source; the reliable answer is to query `hipDeviceProp_t`.

What the current docs say: nothing. The FAQ entry "What are the maximum limits of kernel launch parameters?" is absent from the HIP 7.17.0 FAQ, and I found no limit in the language-extensions page or the porting guide.

What the older docs said (HIP 6.2.0 FAQ, still online): "Product of block.x, block.y, and block.z should be less than 1024. Please note, HIP does not support kernel launch with total work items defined in dimension with size gridDim x blockDim >= 2^32, so gridDim.x * blockDim.x, gridDim.y * blockDim.y and gridDim.z * blockDim.z are always less than 2^32."

What the runtime source does at `therock-10.2`:

- Device properties are filled from the driver at run time, not from constants: `deviceProps.maxThreadsPerBlock = info.maxWorkGroupSize_;`, `deviceProps.maxThreadsDim[0] = info.maxWorkItemSizes_[0];`, `deviceProps.maxGridSize[0] = std::min(static_cast<uint64_t>(info.maxGridDim_[0]), kInt32Max);` (`hip_device.cpp`), with `maxWorkGroupSize_` and `maxGridDim_` read through `HSA_AGENT_INFO_WORKGROUP_MAX_SIZE` and `HSA_AGENT_INFO_GRID_MAX_DIM` (`rocdevice.cpp`).
- A launch with too many threads per block is rejected: "// Make sure dispatch doesn't exceed max workgroup size limit" / `if (launch_params.local_.product() > info.maxWorkGroupSize_) { return hipErrorInvalidConfiguration; }` (`hip_module.cpp`, `ihipLaunchKernel_validate`). A zero in any grid or block dimension is rejected the same way.
- The per-dimension product still has to fit in 32 bits, which is the old FAQ's rule in code: the grid is turned into a global size `static_cast<size_t>(gridX) * blockX` and checked with `NDRange32::CanSafelyNarrow` ("Check that the size_t globals fit in uint32_t before the narrowing cast above"); each block dimension must fit in 16 bits (`NDRange16::CanSafelyNarrow`) (`ndrange.hpp`).
- On newer hardware the grid's y and z are capped as on NVIDIA: `constexpr auto gridDimYZmax = static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) + 1;` / `if (device->isa().versionMajor() >= 12 && (gridDim.y > gridDimYZmax || gridDim.z > gridDimYZmax)) { return hipErrorInvalidConfiguration; }` (`hip_platform.cpp`, `ihipLaunchKernel`). That is 65536, one more than CUDA's 65535. I did not check which products report an ISA major version of 12 or more.

Documented differences from CUDA, then: the product limit of gridDim times blockDim per dimension (CUDA documents no such rule; its x limit of 2^31 - 1 blocks times up to 1024 threads exceeds 2^32), and no published per-dimension block caps. I did not find what values AMD devices report in `maxThreadsDim` or `maxThreadsPerBlock`; ch05 B2 has the same gap.

- https://rocm.docs.amd.com/projects/HIP/en/latest/faq.html (no limits entry)
- https://rocm.docs.amd.com/projects/HIP/en/docs-6.2.0/how-to/faq.html (old entry)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/src/hip_device.cpp (lines 652 to 658)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/src/hip_module.cpp (lines 339 to 371)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/src/hip_platform.cpp (lines 685 to 737)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/rocclr/platform/ndrange.hpp (lines 84 to 196)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/rocclr/device/rocm/rocdevice.cpp (lines 1374 to 1405)

### B4. Reporting a failed launch on HIP

**Verdict: CONFIRMED.** The error code and the query functions mirror CUDA's.

- `hipErrorInvalidConfiguration = 9,     ///< Invalide configuration` (spelling as in the header).
- `hipError_t hipGetLastError(void);`: "Returns the last error that has been returned by any of the runtime calls in the same host thread, and then resets the saved error to #hipSuccess." `hipError_t hipPeekAtLastError(void);`: "Unlike hipGetLastError, this function does not reset the saved error code."
- llama.cpp's `vendors/hip.h` renames the CUDA spellings to these (ch05 B9), so one `.cu` source checks errors on both.

- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/hip_runtime_api.h (lines 127 and 3080 to 3116)

### B5. The chapter's kernels on HIP: mapping, guard, flattening

**Verdict: CONFIRMED** that nothing changes: the `__global__` qualifier, the built-ins, the launch and the C++ index arithmetic are the same source (ch05 B3, B4, B9; B1 and B2 above). One thing in AMD's documentation should not be copied.

The HIP language page prints a 3D flattening formula: "A global unique identifier of a three-dimensional grid can be calculated using the following code: (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x + (threadIdx.z + blockIdx.z * blockDim.z) * blockDim.x * blockDim.y".

As printed, the strides are the block's width and area, not the grid's. By my arithmetic it gives distinct values only when the grid is one block wide in x and y: with `gridDim.x = 2` and `blockDim.x = 4`, the thread at global (x 4, y 0) and the thread at global (x 0, y 1) both get 4. The stride that makes the identifier unique is the total thread count per row, `gridDim.x * blockDim.x`, and for data it is the data's width, which is the book's `plane*m*n + row*m + col` (p. 55).

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html (section "threadIdx and blockIdx")

### B6. Column-major libraries on AMD

**Verdict: CONFIRMED.** Same convention as cuBLAS.

- "rocBLAS uses column-major storage for 2D arrays, and 1-based indexing for the functions xMAX and xMIN. This is the same as legacy BLAS and cuBLAS."
- "If you require row-major and 0-based indexing (used in C language arrays), download the file cblas.tgz from the Netlib Repository. Review the CBLAS functions that provide a thin interface to legacy BLAS. They convert from row-major and 0-based to column-major and 1-based. This is done by swapping the order of the function arguments. It is not necessary to transpose the matrices."

I read rocBLAS's page, not hipBLAS's. llama.cpp reaches AMD's BLAS through `#define cublasSgemm hipblasSgemm` (ch05 B9) with the same arguments it gives cuBLAS (D2).

- https://rocm.docs.amd.com/projects/rocBLAS/en/latest/how-to/what-is-rocblas.html

---

## Group C: Metal

Sources as in ch05-facts: the MSL specification (cited "MSL spec" with section and page), Apple developer documentation, the macOS 27.0 SDK headers on this machine, the Metal Feature Set Tables.

### C1. A thread's coordinates: no `blockIdx * blockDim + threadIdx` needed

**Verdict: CONFIRMED.** The attribute names and types are in ch05 C4. Added here: Apple's worked example, the axis convention, and the fact that the book's formula is available in Metal too.

Quotes ("Creating threads and threadgroups"):

- "A compute pass can run a kernel function over a 1D, 2D, or 3D grid. Each point in the grid represents a thread, which is a single instance of your kernel function." The page's example is image processing, with "each thread corresponding to a single pixel of the image being processed."
- "For example, given a grid of 16 x 16 threads partitioned into 2 x 4 threadgroups of 8 x 4 threads, a single thread ... has a position in the grid of (9,10)". "The thread ... is in the threadgroup with a position in the grid of (1,2), and its position in that threadgroup is (1,2)".
- "Depending on the shape of the grid, these position attributes are either a scalar value, or a two- or three-element vector. In the case of a 2D grid, position attributes are two-element vectors, with the origin at the top-left."
- The book's formula, as Apple prints it: `uint2 thread_position_in_grid = (threadgroup_position_in_grid * threads_per_threadgroup) + thread_position_in_threadgroup;`

Checking Apple's example against the formula: x is 1 * 8 + 1 = 9 and y is 2 * 4 + 2 = 10. So "8 x 4 threads" is width by height, `.x` is the horizontal coordinate (column) and `.y` the vertical one (row), the same assignment as CUDA's `col` from x and `row` from y. Apple's samples confirm it: the grid is `MTLSizeMake(texture.width, texture.height, 1)` and the guard is `gridId.x >= outTexture.get_width() || gridId.y >= outTexture.get_height()`.

Mapping for the lessons: `threadgroup_position_in_grid` is `blockIdx`, `thread_position_in_threadgroup` is `threadIdx`, `threads_per_threadgroup` is `blockDim`, `threadgroups_per_grid` is `gridDim`, and `thread_position_in_grid` is the already-combined coordinate. Metal has no (z, y, x) labelling convention; positions are written x first.

- https://developer.apple.com/documentation/metal/creating-threads-and-threadgroups
- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 5.2.3.6, page 153)

### C2. Metal's `dim3`: `MTLSize`

**Verdict: CONFIRMED.** The host-side size type is `MTLSize` with fields `width`, `height`, `depth`. It has no defaulting constructor: all three must be given, and unused ones must be 1.

- SDK header: `MTL_INLINE MTLSize MTLSizeMake(NSUInteger width, NSUInteger height, NSUInteger depth)`.
- "Treat each size instance as a measure of something in 3D, even if it represents something with only one or two dimensions, by assigning 1 to the irrelevant dimensions." "For a 1D array with length 42, set a size's height, depth properties to 1, so that it represents [42, 1, 1]."
- 1D launch in Apple's introductory sample: `MTLSize gridSize = MTLSizeMake(arrayLength, 1, 1);`

The fields are `NSUInteger` (64-bit) where `dim3` has 32-bit fields. That does not mean a grid may be that large; see C4.

- https://developer.apple.com/documentation/metal/mtlsize
- https://developer.apple.com/documentation/metal/performing-calculations-on-a-gpu
- Local header: `Metal.framework/Headers/MTLTypes.h` (macOS 27.0 SDK)

### C3. Launching over 2D data, choosing the threadgroup shape, and surplus threads

**Verdict: CONFIRMED.** The two dispatch calls are in ch05 C5. With `dispatchThreads` the grid is exactly the data size, so there are no surplus threads and no guard. With `dispatchThreadgroups` the situation is the book's: round up, then guard.

Quotes:

- "In iOS 11 and macOS 10.13 and later on devices that support nonuniform dispatch sizes, Metal calculates the number of threadgroups and provides nonuniform threadgroups if the grid size isn't a multiple of the threadgroup size."
- "When Metal performs this calculation, it can generate smaller threadgroups along the edges of your grid. Compared to uniform threadgroups, this technique simplifies kernel code and improves GPU performance."
- "You don't need to check a thread's position in a grid if you use the dispatchThreads(_:threadsPerThreadgroup:) technique."
- For the uniform call: `MTLSizeMake((texture.width + w - 1) / w, (texture.height + h - 1) / h, 1)`, and "This approach can result in the threadgroups generating a grid that's larger than your data. To compensate for the extra threads, you can make your code exit early if the thread position in the grid is outside the bounds of the data."
- Recommended threadgroup shape: `NSUInteger w = pipelineState.threadExecutionWidth;` / `NSUInteger h = pipelineState.maxTotalThreadsPerThreadgroup / w;` / `MTLSize threadsPerThreadgroup = MTLSizeMake(w, h, 1);`
- A fixed shape is also what Apple's Metal 4 sample does: "The configureThreadgroupForComputePasses method sets a 16 × 16 threadgroup size, which runs on any Apple silicon GPU", followed by the same round-up, and "This calculation ensures the grid covers an area that's at least as large as the texture, which is why the kernel checks whether each thread's coordinates are within bounds."

For the centrepiece interactive this gives a clean statement of the difference: the four regions of the book's 62 x 76 example (pp. 54 to 55) exist on Metal only under `dispatchThreadgroups`. Under `dispatchThreads` with 16 x 16 groups the edge threadgroups are smaller instead; the spec says so directly ("the actual threadgroup size of a specific threadgroup may be smaller than the threadgroup size specified in the dispatch", ch05 C4). Real projects use both: llama.cpp and PyTorch's MPS matmul use `dispatchThreadgroups` and guard or stride (D3, D5); MLX uses `dispatchThreads` (D4).

A slip on Apple's page, for anyone checking the arithmetic: with the page's own values (512 threads per group, execution width 16, so w = 16 and h = 32) the code yields 64 by 24 threadgroups for a 1024 by 768 texture. The page says "a width of 32, a height of 48". Both give the 1536 threadgroups and 786,432 threads it then states.

- https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes
- https://developer.apple.com/documentation/metal/combining-blit-and-compute-operations-in-a-single-pass
- https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/dispatchthreads(_:threadsperthreadgroup:)
- https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/dispatchthreadgroups(_:threadsperthreadgroup:)

### C4. Metal's limits on grid size, and what happens beyond a limit

**Verdict: COULD NOT VERIFY** for the two things the book states for CUDA and a port would want: a maximum grid size, and the result of exceeding a limit.

- Threads per threadgroup: settled in ch05 C3 (1024 in the feature set tables for Apple4 and later, per-dimension values from `MTLDevice.maxThreadsPerThreadgroup`, the real limit for a given kernel from `MTLComputePipelineState.maxTotalThreadsPerThreadgroup`). Apple's sample caps its group with that property: `NSUInteger threadGroupSize = _mAddFunctionPSO.maxTotalThreadsPerThreadgroup;` and notes it "varies depending on the complexity of the function used to create the pipeline state object". That is a difference in kind from CUDA, where 1024 does not depend on the kernel.
- Grid size: I found no Apple statement of a maximum number of threads or threadgroups per compute grid. The feature set tables have rows for object and mesh shader grids only. The one concrete figure I found is a code comment in MLX, which is a project's observation and not Apple documentation: "Metal silently truncates a grid dimension past UINT32_MAX threads, so a launch of n_groups threadgroups along x is split into several dispatches that each fit." (D4).
- Exceeding the threadgroup limit: the dispatch methods return `void`, and none of the pages or headers I read says what an oversized `threadsPerThreadgroup` does. There is no documented counterpart of `cudaErrorInvalidConfiguration`.

- https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
- https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/maxtotalthreadsperthreadgroup
- https://developer.apple.com/documentation/metal/performing-calculations-on-a-gpu
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/utils.h (lines 23 to 45)

### C5. Byte images in plain buffers (no textures)

**Verdict: CONFIRMED.** MSL has an 8-bit unsigned scalar and buffers of it are ordinary pointer arguments, so the grayscale and blur kernels can use the same flat `unsigned char` arrays as the CUDA versions. One layout trap: `uchar3` is 4 bytes.

Quotes (MSL spec):

- Table 2.1: "unsigned char, uchar, uint8_t: An unsigned 8-bit integer." Table 2.2: size 1, alignment 1.
- Section 2.8: "MSL implements a buffer as a pointer to a built-in or user defined data type described in the device, constant, or threadgroup address space." "Ordinary Metal buffers may contain: Basic types such as float and int; Vector and matrix types; Arrays of buffer types; Structures of buffer types; Unions of buffer types".
- Section 1.5.4: "You must declare arguments to Metal graphics and kernel functions that are pointers with the Metal device, constant, threadgroup, threadgroup_imageblock, object_data, or ray_data address space attribute."
- Table 2.3: `char3`, `uchar3`: size 4, alignment 4. Table 2.4: `packed_char3`, `packed_uchar3`: size 3, alignment 1.

So an RGB image of three consecutive bytes per pixel must be read as `device const uchar*` with the book's `rgbOffset = grayOffset*CHANNELS` arithmetic (p. 53), or as `packed_uchar3`. A `device const uchar3*` would step 4 bytes per pixel and read the wrong data.

Host side:

- `setBuffer(_:offset:index:)`: "Binds a buffer to the buffer argument table, allowing compute kernels to access its data on the GPU."
- Small scalars such as width and height can be passed without making a buffer: `setBytes(_:length:index:)`, "Copies data directly to the GPU to populate an entry in the buffer argument table.", with the limit "This method only works for data smaller than 4 kilobytes that doesn't persist. Create an MTLBuffer instance if your data exceeds 4 KB, needs to persist on the GPU, or you access results on the CPU." In the kernel such an argument is a reference in the `constant` address space, as in the spec's `constant AAPL::uniforms_t& uniforms [[buffer(2)]]`.

Apple's own image samples use textures (`texture2d<half, access::read>` with `inTexture.read(gridId)`), which hide the flat index and the byte-to-float conversion the chapter is about. Plain buffers are the documented alternative above and are what llama.cpp, MLX and PyTorch use for tensors (D3 to D5).

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (sections 1.5.4, 2.1, 2.2, 2.2.3, 2.8; pages 13, 25 to 28, 33 to 34, 41)
- https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/setbuffer(_:offset:index:)
- https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/setbytes(_:length:index:)

### C6. MSL as a language for these kernels: C++ base, casts, signedness, `double`

**Verdict: CONFIRMED.** The chapter's kernel bodies translate line for line, with three things to watch.

- Language base: "In Metal 4 and later, the Metal programming language is a C++17-based specification with extensions and restrictions. Refer to the C++17 specification (also known as ISO/IEC 14882:2017) for a detailed description of the language grammar. Prior language versions of Metal are a C++14-based specification with extensions and restrictions." The grammar is C++'s, so the cast precedence of A9 applies unchanged: the book's line 19 is the same bug in MSL.
- Conversions: "The static_cast operator converts from a scalar or vector type to another scalar or vector type using the default rounding mode with no saturation (when converting to floating-point, round ties to even; when converting to an integer, round toward zero)." and "Implicit conversions between scalar built-in types (except void) are supported." Truncation toward zero matches C++, so `(uchar)(sum)` gives the same byte as the CUDA version for in-range values. "No saturation" means an out-of-range float is not clamped.
- Signedness: the position attributes accept only unsigned types ("ushort, ushort2, ushort3, uint, uint2, or uint3", ch05 C4), and "Following the usual arithmetic conversions, all arithmetic operators return a result of the same built-in type (integer or floating-point) as the type of the operands." The blur kernel's `curRow >= 0` test therefore needs signed variables: convert the position to `int` before adding the signed offset, as the book's `int row`, `int col` do for CUDA's unsigned built-ins (A2). The reasoning about the blur kernel is mine; the two rules are the spec's.
- No `double`: "Metal does not support the double, long long, unsigned long long, and long double data types." "Metal supports: the f or F suffix to specify a single precision floating-point literal value (such as 0.5f or 0.5F)". Write `0.299f`. (The book's `ceil(m/16.0)` is host code and is unaffected.)
- Not available: "Do not use the C++ standard library in Metal code. Instead, Metal has its own standard library".

Compile-only check on this machine (`xcrun metal -c`, "Apple metal version 32023.921", no diagnostics): a kernel taking `device const uchar* in [[buffer(0)]]`, `device uchar* out [[buffer(1)]]`, `constant uint2& size [[buffer(2)]]`, `uint2 gid [[thread_position_in_grid]]`, with `out[o] = (uchar)(0.299f*r + 0.587f*g + 0.114f*b);`, and also the book's unparenthesised cast, compiled. An unsuffixed literal (`0.299 * r`) compiled without a diagnostic too; the spec passages I read do not say what type it is given, so the labs should always write the suffix. This shows the syntax is accepted. It is not a run and says nothing about results.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (sections 1.5, 1.5.4, 2.1, 2.23, 2.24, 3.1; pages 12 to 13, 25 to 26, 110 to 112)

### C7. Row-major and column-major on Apple's side

**Verdict: CONFIRMED.** A flat buffer has whatever layout your index arithmetic gives it, so `row*width + col` is row-major on Metal exactly as on CUDA. Two neighbouring Apple conventions differ from each other, which is worth one sentence where the book warns about column-major libraries (pp. 51 to 52).

- MSL's built-in matrix types are column-major: "Metal constructs and consumes matrix components in column-major order." and "You can also construct a matrix of type T with n columns and m rows from n vectors of type T with m components." (`float3x2` is 3 columns by 2 rows.) These small matrix types are not what a kernel uses for a tensor.
- Metal Performance Shaders matrices are row-major: "MPSMatrix objects serve as inputs and outputs of MPSMatrixMultiplication objects. Matrix data is assumed to be stored in row-major order."

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 2.3.2, page 37)
- https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrix

### C8. Matrix multiplication that Apple already provides

**Verdict: CONFIRMED.** Two library routes exist beside a hand-written kernel. Neither replaces the chapter's naive kernel as a teaching object, but the reader will meet both in PyTorch's source (D5).

- Metal Performance Shaders: `MPSMatrixMultiplication`, "A matrix multiplication kernel.", "computes the following operation: C = alpha * op(A) * op(B) + beta * C". That is the level-3 BLAS form in the book's sidebar (pp. 59 to 60).
- Inside a kernel, new with Metal 4: MSL spec section 7 "Metal Performance Primitives". "TensorOps are GPU-accelerated functions that operate on tensors and cooperative_tensors". "The template class matmul2d performs a generalized matrix multiplication of two tensors (C = A*B) or matrix multiplication accumulated into a tensor (C = A*B + C). The operation multiplies an M×K tensor A by a K×N tensor B and accumulates the result into an M×N tensor C." The spec's example declares the operands as `tensor<device half, dextents<int, 2>> a [[ buffer(0) ]]`, builds `tensor_ops::matmul2d<matmulDescriptor, execution_simdgroups<4>> matmulOp;` and calls `matmulOp.run(mA, mB, mC);` on slices chosen by `threadgroup_position_in_grid`. Tensor types themselves: "All OS: Metal 4 and later support tensor types." (section 2.22).

I read the overview and the first example only, not the full descriptor and data-type tables. This is the nearest Metal thing to CUDA's tile kernels (A10) in intent, a per-threadgroup operation the compiler spreads over threads, but they are unrelated designs and I found no document linking them.

- https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication
- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (sections 2.22 and 7.2.1; pages 84 and 350 to 351)

### C9. Correction to ch05-facts C3: Apple does use the word "warp"

**Verdict: WRONG** (the sub-claim in ch05-facts C3, not anything in the book). ch05-facts says "No Apple source uses the word 'warp'" and that only the MSL spec links `threadExecutionWidth` to a SIMD-group. An Apple documentation page says both things outright.

Quotes ("Creating threads and threadgroups"):

- "The threads in a threadgroup are further organized into single-instruction, multiple-data (SIMD) groups, also known as warps or wavefronts, that execute concurrently. The threads in a SIMD group execute the same code. Avoid writing code that could cause your kernel function to diverge; that is, to follow different code paths."
- "The number of threads in a SIMD group is returned by the threadExecutionWidth of your compute pipeline state object."
- "Although threadgroups can be multidimensional, SIMD groups are 1D. Therefore, a thread's position within a SIMD group is a scalar value for all threadgroup shapes."
- "Even if a single thread in a SIMD group takes a different path from the others, all threads in that group execute both branches, and the execution time for the group is the sum of the execution time of both branches."

The rest of ch05 C3 stands (property meanings, the feature set limits, no documented fixed width). For chapter 3 this matters where the notes point forward to chapter 4's reuse of the 62 x 76 regions for divergence: the Metal lesson may call a SIMD group Apple's warp and cite this page. ch05-facts itself was not edited.

- https://developer.apple.com/documentation/metal/creating-threads-and-threadgroups

---

## Group D: How llama.cpp, MLX and PyTorch do the same things

Read from each project's default branch on the check date (commits at the top of this file). Line numbers will drift.

### D1. llama.cpp, CUDA: a 3D grid over a 4D tensor, with the 65535 cap handled in the kernel

**Verdict: CONFIRMED.** `ggml/src/ggml-cuda/pad.cu` is a compact real example of everything in sections 3.1 and 3.2: a 3D grid, a 1D block, the surplus-thread guard, the flat row-major index, and the grid limit of A3.

```cpp
    // blockIdx.z: i3*ne2+i2
    // blockIdx.y: i1
    // blockIDx.x: i0 / CUDA_PAD_BLOCK_SIZE
    // gridDim.y and gridDim.z are capped at 65535, blocks stride over larger ne1 and ne2*ne3
    int i0 = threadIdx.x + blockIdx.x * blockDim.x;
    if (i0 >= ne0) {
        return;
    }

    for (int i1 = blockIdx.y; i1 < ne1; i1 += gridDim.y) {
        for (int i23 = blockIdx.z; i23 < ne2 * ne3; i23 += gridDim.z) {
            ...
            const int64_t dst_idx = i3 * (ne0 * ne1 * ne2) + i2 * (ne0 * ne1) + i1 * ne0 + i0;
```

and the launch:

```cpp
    int  num_blocks = (ne0 + CUDA_PAD_BLOCK_SIZE - 1) / CUDA_PAD_BLOCK_SIZE;
    dim3 gridDim(num_blocks, std::min(ne1, 65535), std::min(ne2 * ne3, 65535));
    pad_f32<<<gridDim, CUDA_PAD_BLOCK_SIZE, 0, stream>>>(src, ...);
```

Points for the lessons: the fastest-varying tensor dimension is mapped to x; the flat index is the book's `plane*m*n + row*m + col` extended by one more dimension (p. 55); the integer round-up idiom is the guide's (A1); and the strided loops are the book's second remedy for the grid limit, a thread that computes several elements (p. 64). The same file compiles for AMD (ch05 B9), where a comparable cap exists on newer hardware (B3).

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/pad.cu (lines 10 to 77)

### D2. llama.cpp: tensor layout, and calling a column-major BLAS with row-major data

**Verdict: CONFIRMED.** ggml stores the innermost (fastest-varying) dimension first, and its matmul convention already has the transpose built in.

- `ggml.h`: `int64_t ne[GGML_MAX_DIMS]; // number of elements` and `size_t nb[GGML_MAX_DIMS]; // stride in bytes: // nb[0] = ggml_type_size(type) // nb[1] = nb[0] * (ne[0] / ggml_blck_size(type)) + padding // nb[i] = nb[i-1] * ne[i-1]`. So `ne[0]` is the row length (number of columns) and `ne[1]` the number of rows: the shape is listed in the reverse of PyTorch's order, and the memory is the same row-major layout.
- `ggml_mul_mat`: "// A: k columns, n rows => [ne03, ne02, n, k] // B: k columns, m rows (i.e. we transpose it internally) => [ne03 * x, ne02 * y, m, k] // result is n columns, m rows => [ne03 * x, ne02 * y, m, n]". Both operands are stored with the shared dimension k as their row length, so each output element is a dot product of two contiguous rows. Compare the book's kernel, where N is walked down a column at stride `Width` (p. 62).
- The cuBLAS call for the float case: `cublasSgemm(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N, ne01, ne11, ne10, alpha, src0_ptr, s01, src1_ptr, s11, beta, dst_ptr, ldc)`. A row-major n x k array is, read column-major, a k x n matrix, so telling cuBLAS to transpose the first operand recovers the intended product with no copy. That is the practical form of the book's remark that column-major libraries ask C callers to transpose (pp. 51 to 52; A7).

For milestone M1 this fixes the vocabulary: a linear layer's weight is stored one output feature per row (D6), which is the "B" operand layout above.

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml.h (lines 515 to 524 and 1304 to 1310)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu (lines 2043 to 2053)

### D3. llama.cpp, Metal: one threadgroup per row, uniform dispatch

**Verdict: CONFIRMED.** llama.cpp's Metal backend does not call `dispatchThreads`. Its one dispatch wrapper is `[encoder->obj dispatchThreadgroups:MTLSizeMake(tg0, tg1, tg2) threadsPerThreadgroup:MTLSizeMake(tptg0, tptg1, tptg2)];` in `ggml-metal-device.m`, and a search of that file, `ggml-metal-context.m`, `ggml-metal.cpp` and `ggml-metal-common.cpp` finds no `dispatchThreads:` call.

The usual mapping puts the outer tensor dimensions on the threadgroup position and lets the threads of a group stride along the innermost one. From `kernels/upscale.metal`:

```cpp
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;
    ...
    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
```

launched with `const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);` and `ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);`. Other kernels spread a row over several groups and stride over all of them: `for (int i0 = tgpig.x*ntg.x + tpitg.x; i0 < args.ne0; i0 += ntg.x*ntgs.x)` (`kernels/unary.metal`, `kernel_reglu`).

Reading notes: the loop bound `i0 < args.ne0` is the range guard; the stride loop is again "one thread, several elements"; offsets are computed in bytes from the `nb` strides of D2 (`dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0`); and these kernels are the Metal relative of the book's exercise 1, where one thread owns more than one output element.

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-device.m (lines 890 to 891)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/upscale.metal (lines 5 to 28)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/unary.metal (lines 233 to 252)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-ops.cpp (lines 5527 to 5534)

### D4. MLX: `thread_position_in_grid` with a nonuniform dispatch

**Verdict: CONFIRMED.** MLX takes the opposite route to llama.cpp: it dispatches exactly as many threads as it needs with `dispatchThreads` and flattens the grid position itself.

- Kernel side (`kernels/unary.h`): `uint2 index [[thread_position_in_grid]], uint2 grid_dim [[threads_per_grid]]` and `int64_t offset = N * (index.x + grid_dim.x * int64_t(index.y));`. That is `row*width + col` with the grid's own width as the stride. The 3D form in the same file is `IdxT out_idx = N * index.x + xshape * (index.y + IdxT(grid_dim.y) * index.z);`.
- Host side (`binary.cpp`): `auto group_dims = get_block_dims(dim0, dim1, rest);` / `MTL::Size grid_dims = MTL::Size(dim0, dim1, rest);` / `compute_encoder.dispatch_threads(grid_dims, group_dims);`, where `dim0` is the last tensor axis (`shape[ndim - 1]`), so again the fastest-varying dimension goes on x. The wrapper is `get_command_encoder()->dispatchThreads(grid_dims, group_dims);` (`device.cpp`).
- Threadgroup shape: `get_block_dims_common` picks power-of-two sizes for the three dimensions whose exponents sum to at most 10, that is at most 1024 threads, grown round-robin while each dimension is still smaller than the data (`mlx/backend/common/utils.cpp`).
- The simple one-element-per-thread kernels have no range guard, consistent with Apple's note in C3. The variants that do N elements per thread guard the tail: `if (N > 1 && offset + N > size)`.
- Default layout is row-major: `strides[i] = size; size *= shape[i];` iterating `i` from the last axis down, with the flag `row_contiguous` (`mlx/array.cpp`).
- Large grids: the comment quoted in C4, and `large = out.data_size() > UINT32_MAX;` selecting 64-bit offsets and a 2D grid.

- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/unary.h (lines 22 to 60)
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/binary.cpp (lines 60 to 162)
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/device.cpp (lines 451 to 465)
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/common/utils.cpp (lines 125 to 153)
- https://github.com/ml-explore/mlx/blob/main/mlx/array.cpp (lines 242 to 248)
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/utils.h (lines 23 to 45)

### D5. PyTorch, MPS backend: the book's matmul mapping, in Metal

**Verdict: CONFIRMED.** `aten/src/ATen/native/mps/kernels/Matmul.metal` maps threads to output elements exactly as Fig. 3.11 does: x gives the column, y the row, a third grid dimension the batch, and a guard protects the write.

```cpp
kernel void naive_bmm(
    ...
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  uint batch = group_id.z;
  uint col = group_id.x * TILE_DIM + tid.x;
  uint row = group_id.y * TILE_DIM + tid.y;
  ...
  if (row < sizes.x && col < sizes.z) {
    outputData[batch * strides[8] + col * strides[6] + row * strides[7]] =
        static_cast<T>(sum);
  }
```

The non-batched `matmul` kernel takes `uint2 thread_id [[thread_position_in_grid]]` instead and guards with `if (thread_id.y < sizes.x && thread_id.x < sizes.z)`. The launch (`operations/Matmul.mm`, `do_metal_mm`) is the round-up of section 3.2 with a 16 x 16 group:

```objc
      constexpr uint32_t TILE_DIM = 16; // fastest performance from tests on multiple macs
      uint32_t gridSizeX = (output.size(1) + TILE_DIM - 1) / TILE_DIM;
      uint32_t gridSizeY = (self_.size(0) + TILE_DIM - 1) / TILE_DIM;
      MTLSize threadsPerThreadgroup = MTLSizeMake(TILE_DIM, TILE_DIM, 1);
      MTLSize threadgroupsPerGrid = MTLSizeMake(gridSizeX, gridSizeY, 1);
      [computeEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
```

Three cautions so the lesson does not overstate the match:

- Only the mapping and guard are chapter 3 material. The inner product in these kernels is not the naive loop: it stages 16 x 16 tiles in `threadgroup` arrays with `threadgroup_barrier`, which is the shared-memory algorithm of chapter 5. The kernel name `naive_bmm` is PyTorch's.
- Indexing goes through the tensors' strides (`row * strides[...] + col * strides[...]`), not `row*Width + col`, because a PyTorch tensor need not be contiguous.
- The same file has further paths: `matmul_simd` (SIMD-group matrices) and `matmul_mpp`, which calls Metal Performance Primitives (C8); `Matmul.mm` also still uses MPSGraph's `matrixMultiplicationWithPrimaryTensor` in places. I did not trace which path a given `torch.matmul` call takes.

- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/kernels/Matmul.metal (lines 204 to 222 and 277 to 300)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/operations/Matmul.mm (lines 293 to 322)

### D6. PyTorch: conventions the milestone should match, and the CUDA elementwise launch

**Verdict: CONFIRMED.**

Linear layer (for milestone M1):

- "Applies an affine linear transformation to the incoming data: y = xA^T + b". "weight: the learnable weights of the module of shape (out_features, in_features)". Input "(*, H_in)", output "(*, H_out)". Source: `torch.empty((out_features, in_features), **factory_kwargs)`.
- So with batch first, a batch of B inputs is a B x in_features matrix X, and the layer computes X times the transpose of an out_features x in_features matrix W: element (b, o) is the dot product of row b of X and row o of W. Both operands are walked along a row. This is ggml's convention too (D2), and it is not the book's P = M N, where the second operand is walked down a column (p. 62). M1 has to say which of the two it implements.

Row-major:

- `Tensor.stride`: "Stride is the jump necessary to go from one element to the next one in the specified dimension dim." Example as printed: `x = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])`, `x.stride()` gives `(5, 1)`. A step along dimension 0 skips a whole row of 5: the book's `row*Width + col`.
- "torch.contiguous_format: Tensor is or will be allocated in dense non-overlapping memory. Strides represented by values in decreasing order."

CUDA elementwise kernels (what runs under `a + b` or `torch.relu` on a CUDA tensor):

- A 1D grid whatever the tensor's shape, with the flat index from the book's formula and several elements per thread: `int idx = nv * blockIdx.x + tid;` ... `if (idx < N) { f(idx); idx += nt; }` (`elementwise_kernel`, with `nv = nt * vt`).
- Launch: `dim3 block(nt);` / `dim3 grid((N + block.x * vt - 1) / (block.x * vt));` / `elementwise_kernel<nt, vt, func_t><<<grid, block, 0, stream>>>(N, f);` / `C10_CUDA_KERNEL_LAUNCH_CHECK();`, where `#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())` (A4).
- Block size is fixed in source and differs by vendor: `constexpr uint32_t num_threads() { return C10_WARP_SIZE * 4; }` for CUDA and `constexpr int num_threads() { return 256; }` under `USE_ROCM` (`thread_constants.h`).
- The 32-bit index limit of A11: `TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());`.

This is the bridge for section 3.2: PyTorch does not use a 2D grid for a 2D tensor. A multidimensional grid is a convenience (A2), and a library that must handle any shape and any strides flattens first.

- https://docs.pytorch.org/docs/2.15/generated/torch.nn.Linear.html
- https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py (lines 54 to 109)
- https://docs.pytorch.org/docs/2.15/generated/torch.Tensor.stride.html
- https://docs.pytorch.org/docs/2.15/tensor_attributes.html (section "torch.memory_format")
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/CUDALoops.cuh (lines 596 to 660)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/thread_constants.h
- https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDAException.h (line 85)

---

## Summary of corrections

Items where the book, an earlier facts file, or a vendor page turned out WRONG or PARTLY, or where a lesson would otherwise say something stale:

- A3: the book's "distributed across the three dimensions any way you like" omits the per-dimension block caps (1024, 1024, 64). Its four examples are unaffected.
- A1: NVIDIA's guide sizes grids with integer arithmetic (`(n + threads-1)/threads`) or `cuda::ceil_div` from `<cuda/cmath>`; the book's `ceil(n/256.0)` form appears on no vendor page I read.
- A4: a launch that breaks a limit fails with `cudaErrorInvalidConfiguration` and is silent unless the host calls `cudaGetLastError()`. The book does not mention this.
- A6: "need to explicitly linearize ... in the current CUDA C++" is dated. nvcc accepts C++23 and libcu++ provides `cuda::std::mdspan` for device code (with `operator()`, since `operator[]` is disabled). Hand flattening is still what NVIDIA's guide and all three inference projects do.
- A8: the text's 0.299, 0.587, 0.114 are ITU-R BT.601. The listing's 0.21, 0.71, 0.07 match neither BT.601 nor BT.709 (0.2126, 0.7152, 0.0722) and sum to 0.99. Use the text's constants; torchvision uses 0.2989, 0.587, 0.114.
- A9: line 19 of Fig. 3.4 casts only `0.21f`, so the red term is zero. Confirmed from the grammar and by compiling: pure red gives 0 instead of 53, with no compiler warning. The same holds in MSL (C6).
- A9: because the float to byte conversion truncates, the order of the three additions can change the grayscale result by one level (40 of 140,608 sampled colours in a host test). Check the lab with a tolerance of 1.
- A10: "tile" now has an NVIDIA meaning (tile kernels, `__tile_global__`, compiler-chosen block size, no `threadIdx`) that is different from the book's "tile of P" and from chapter 5's tiled matmul.
- B2: on HIP the index built-ins are not `uint3`. AMD's docs call all four `dim3`; the header gives each its own type convertible to `dim3`. Use the `.x .y .z` members only.
- B3: the current HIP docs state no block or grid limits. The old FAQ's rules (block product "less than 1024", gridDim times blockDim below 2^32 per dimension) survive as checks in the runtime source, and the source adds a 65536 cap on grid y and z for ISA major version 12 and later. Query `hipDeviceProp_t`.
- B5: the 3D "global unique identifier" formula printed in AMD's HIP language page uses the block size as its stride and is unique only for a grid one block wide. Do not copy it.
- C3: Apple's "Calculating threadgroup and grid sizes" prints 32 by 48 threadgroups where its own code and values give 64 by 24 (same total).
- C4: Metal documents no maximum compute grid size and no error for an oversized threadgroup. Do not state either.
- C5: `uchar3` is 4 bytes in MSL. Three-byte RGB pixels must be read through `uchar*` or `packed_uchar3`.
- C9 (correction to ch05-facts C3): Apple's documentation does call SIMD groups "warps or wavefronts" and does say `threadExecutionWidth` returns the SIMD group size. ch05-facts was not edited.
- D2 and D6: both llama.cpp (`ggml_mul_mat`) and PyTorch (`nn.Linear`, weight shape `(out_features, in_features)`, y = xA^T + b) multiply by the transpose of the stored second operand, so both operands are read along rows. The book's P = M N reads N down a column. The milestone must state which it computes.
- D5: PyTorch's Metal kernel named `naive_bmm` uses the book's thread mapping but a tiled, threadgroup-memory inner product. Only its mapping is a chapter 3 example.

Not verified, stated once here:

- NVIDIA's `dim3` constructor text (toolkit header not online), and whether `ceil(n/256.0)` inside `<<< >>>` compiles with nvcc.
- What values AMD devices report for `maxThreadsPerBlock`, `maxThreadsDim` and `maxGridSize`, and which products have ISA major version 12 or more.
- A maximum Metal compute grid size, and Metal's behaviour when `threadsPerThreadgroup` exceeds the pipeline's limit.
- The numeric values of Apple's `kRec709LumaCoefficients`.
- The type MSL gives an unsuffixed floating-point literal.
- The full text of CUDA's tile-kernel chapter and of the MSL tensor and TensorOps tables; hipBLAS's own storage-order statement.
- Which matmul path PyTorch's MPS backend takes for a given call.
- Nothing in this file was run on a GPU.

## Where the backends really differ

| Feature | CUDA | HIP | Metal |
|---|---|---|---|
| Launch size type | `dim3`, unsigned x, y, z, components default to 1 (A1) | `dim3`, `uint32_t` x, y, z, default 1 (B1) | `MTLSize` width, height, depth; no defaults, unused must be 1 (C2) |
| Launch statement | `kernel<<<grid, block>>>(args)`; integers allowed for 1D (A1) | Same statement (ch05 B3) | `dispatchThreads:threadsPerThreadgroup:` (total threads) or `dispatchThreadgroups:threadsPerThreadgroup:` (group count) on a compute encoder (ch05 C5, C3) |
| What you give the launch | Number of blocks and block shape | Same | Either the total thread grid, or the number of threadgroups; plus the group shape |
| Thread coordinates in the kernel | Global variables `blockIdx`, `threadIdx` (`uint3`), `blockDim`, `gridDim` (`dim3`) (A2) | Same names; documented as `dim3`, implemented as distinct types convertible to `dim3`; need `hip_runtime.h` (B2, ch05 B4) | Attributed kernel parameters, unsigned scalar or vector: `threadgroup_position_in_grid`, `thread_position_in_threadgroup`, `threads_per_threadgroup`, `threadgroups_per_grid` (ch05 C4) |
| Global row and column | Computed: `blockIdx.y*blockDim.y + threadIdx.y`, `blockIdx.x*blockDim.x + threadIdx.x` | Same | Given directly: `thread_position_in_grid` (`.y` row, `.x` column); the CUDA formula also works (C1) |
| Surplus threads | Always possible; grid is whole blocks; guard needed (A5) | Same | None with `dispatchThreads` (edge groups shrink, no guard needed); present with `dispatchThreadgroups` (guard needed) (C3) |
| Threads per block or group | 1024 on all listed compute capabilities; per-dimension 1024, 1024, 64; independent of the kernel (A3, ch05 A5) | Not in current docs; read `maxThreadsPerBlock` and `maxThreadsDim`; runtime rejects larger (B3) | 1024 in the feature tables for Apple4 and later, but the real limit is per pipeline: `maxTotalThreadsPerThreadgroup` (ch05 C3, C4) |
| Grid size limit | x at most 2^31 - 1, y and z at most 65535 blocks (ch05 A5) | gridDim times blockDim must fit in 32 bits per dimension; y and z at most 65536 blocks on ISA major 12 and later (source, B3) | None documented (C4) |
| Suggested block or group shape | "256 threads per block ... quite often a good value to start with"; total a multiple of 32 (A1 page, section 2.1.2.3.1 and programming model) | Not stated in pages read | Width `threadExecutionWidth`, height `maxTotalThreadsPerThreadgroup / width`; Apple's sample also uses a fixed 16 x 16 (C3) |
| Launch exceeds a limit | `cudaErrorInvalidConfiguration`, seen via `cudaGetLastError()` (A4) | `hipErrorInvalidConfiguration`, seen via `hipGetLastError()` (B3, B4) | Not documented (C4) |
| Kernel qualifier and language | `__global__`, C++ up to C++23 (A6) | `__global__`, compiled by HIP-Clang (ch05 B4, B6) | `kernel`, MSL based on C++17 (Metal 4) or C++14; no `double`, no standard library (C6) |
| Byte image argument | `unsigned char*` | Same | `device uchar*` with `[[buffer(n)]]`; `uchar3` is 4 bytes, use `uchar` or `packed_uchar3` (C5) |
| Scalar arguments (width, height) | By value in the kernel call | Same | `constant T&` bound with `setBytes:length:atIndex:` (under 4 KB) or a buffer (C5) |
| Float to byte cast | C++ truncation; out of range is undefined (A9) | Same | Truncation toward zero, "no saturation" (C6) |
| Multidimensional view over a flat buffer | Hand flattening, or `cuda::std::mdspan` (A6) | Hand flattening (no equivalent found in pages read) | Hand flattening; MSL 4 `tensor<...>` types exist (C8) |
| Vendor BLAS storage order | cuBLAS: column-major (A7) | rocBLAS: column-major (B6) | `MPSMatrix`: row-major; MSL `floatNxM` types: column-major (C7) |
| Library matmul callable from the host | `cublasSgemm` and relatives (D2) | `hipblasSgemm` through the same call (ch05 B9) | `MPSMatrixMultiplication` (C8) |
| Compiler-distributed block-level operations | Tile kernels (`__tile_global__`), block size chosen by the compiler (A10) | None found | Metal Performance Primitives `matmul2d` inside a kernel (C8) |
