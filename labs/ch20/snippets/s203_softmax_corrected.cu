// Section 20.3's stepper: the book's causal mask-and-softmax kernel (Fig. 20.4, p. 487),
// corrected, with the book's 28 lines kept so that a line number here is the line number
// on the page. The stepper (app/src/sim/masksoftmax.ts) runs exactly these lines.
//
// Where it differs from the page, and why (notes/today/ch20-facts.md, infra-facts.md):
//   - line 6: NEG_INFINITY is not a CUDA name (A5). The identity for max is -FLT_MAX, as in
//     lab 10.3: finite, so it means the same under Metal's fast math (C5).
//   - lines 11 and 20: cub::Max() and cub::Sum() were removed in CCCL 3.0; the current
//     spellings are cuda::maximum<>{} and cuda::std::plus<>{} (infra A11). The result is
//     still valid in thread 0 only (A2), which is why lines 12 to 14 and 21 to 23 exist.
//   - line 27: the book writes the denominator D "for reuse during training". Without the
//     row max, D alone cannot rebuild P; FlashAttention-2 stores the log-sum-exp m + log(D)
//     (A7), and so does this kernel, as L.
//   - exp and log are expf and logf: single precision, said explicitly.
//   - S is const: the kernel only reads it.
//
// Not changed: one block per row (gridDim.x = N; gridDim.y and gridDim.z are 1, not 0 as
// p. 487 says, A1), and the four barriers. Section 20.3 asks which of them the kernel
// actually needs.
//
// Status: front-end-checked (clang's CUDA parser against the course's stub header and the
// stand-in declarations below). Not built with nvcc, not run: unverified on CUDA hardware.
#include <cfloat>

#if defined(PMPP_CUDA_FRONTEND_CHECK)
// The front-end check has no CCCL headers. These declare just the names used below, with
// the shapes infra-facts A11 quotes; they check spelling and arity, not behaviour.
namespace cub {
template <typename T, int BLOCK_DIM_X>
class BlockReduce {
public:
    struct TempStorage { T slot[BLOCK_DIM_X]; };
    __device__ explicit BlockReduce(TempStorage& storage);
    template <typename ReductionOp> __device__ T Reduce(T input, ReductionOp op);
};
}  // namespace cub
namespace cuda {
template <typename T = void> struct maximum { template <typename U> __device__ U operator()(const U& a, const U& b) const; };
namespace std {
template <typename T = void> struct plus { template <typename U> __device__ U operator()(const U& a, const U& b) const; };
}  // namespace std
}  // namespace cuda
__device__ float expf(float x);
__device__ float logf(float x);
#else
#include <cub/cub.cuh>
#endif

#define BLOCK_SIZE 128

__global__ void softmax_kernel(const float* S, float* L, float* P, int N) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_store;
    __shared__ float max_or_sum;
    const float* S_row = &S[blockIdx.x * N];
    float max_val_thread = -FLT_MAX;
    for (int idx = threadIdx.x; idx <= blockIdx.x; idx += blockDim.x) {
        if (S_row[idx] > max_val_thread) max_val_thread = S_row[idx];
    }
    __syncthreads();
    float max_val_row = BlockReduce(temp_store).Reduce(max_val_thread, cuda::maximum<>{});
    if (threadIdx.x == 0) max_or_sum = max_val_row;
    __syncthreads();
    max_val_row = max_or_sum;
    float sum_thread = 0.0f;
    for (int idx = threadIdx.x; idx <= blockIdx.x; idx += blockDim.x) {
        sum_thread += expf(S_row[idx] - max_val_row);
    }
    __syncthreads();
    float sum_row = BlockReduce(temp_store).Reduce(sum_thread, cuda::std::plus<>{});
    if (threadIdx.x == 0) max_or_sum = sum_row;
    __syncthreads();
    sum_row = max_or_sum;
    for (int idx = threadIdx.x; idx < N; idx += blockDim.x) {
        P[blockIdx.x * N + idx] = idx <= blockIdx.x ? expf(S_row[idx] - max_val_row) / sum_row : 0.0f;
    }
    if (threadIdx.x == 0) L[blockIdx.x] = max_val_row + logf(sum_row);
}

// A launch, so the front-end check sees the kernel called the way the book describes:
// one block of BLOCK_SIZE threads per row of S.
void launchSoftmaxCausal(const float* S, float* L, float* P, int N) {
    softmax_kernel<<<N, BLOCK_SIZE>>>(S, L, P, N);
}
