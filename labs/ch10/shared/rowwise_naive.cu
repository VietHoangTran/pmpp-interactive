// Provided for lab 10.3 (do not edit): the benchmark baseline, one thread per row.
//
// Thread r walks all of row r on its own: three passes for the softmax (max, sum of
// exponentials, write), one for the argmax. It is correct and as simple as a row-wise
// kernel can be, and it is what the lab's block-per-row kernels are measured against.
// Two things make it slow, and they are the chapter's subject: a batch of R rows uses R
// threads, so at batch 1 one thread does the whole row; and neighbouring threads read
// addresses `cols` floats apart, so no access is coalesced (chapter 5).
//
// Contract as in the lab: [rows x cols] float32, row-major, finite logits.
#include <cfloat>

// The course's CUDA front-end check (labs/pmpp.py) declares no CUDA Math API functions,
// so it is given expf's device declaration here. nvcc, hipcc and the emulator never see it.
#if defined(PMPP_CUDA_FRONTEND_CHECK)
__device__ float expf(float x);
#endif

#define NAIVE_BLOCK 64

__global__ void softmaxRowsNaiveKernel(const float* x, float* y, int rows, int cols, float scale) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* in = x + (size_t)r * cols;
    float* out = y + (size_t)r * cols;
    float m = -FLT_MAX;
    for (int c = 0; c < cols; ++c) {
        float v = scale * in[c];
        m = v > m ? v : m;
    }
    float s = 0.0f;
    for (int c = 0; c < cols; ++c) {
        s += expf(scale * in[c] - m);
    }
    float inv = 1.0f / s;
    for (int c = 0; c < cols; ++c) {
        out[c] = expf(scale * in[c] - m) * inv;
    }
}

__global__ void argmaxRowsNaiveKernel(const float* x, int* idx, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* in = x + (size_t)r * cols;
    float best = in[0];
    int at = 0;
    for (int c = 1; c < cols; ++c) {
        if (in[c] > best) {   // strictly greater: the first maximum stays
            best = in[c];
            at = c;
        }
    }
    idx[r] = at;
}

void launchSoftmaxRowsNaive(const float* x, float* y, int rows, int cols, float scale) {
    softmaxRowsNaiveKernel<<<(rows + NAIVE_BLOCK - 1) / NAIVE_BLOCK, NAIVE_BLOCK>>>(x, y, rows, cols, scale);
}

void launchArgmaxRowsNaive(const float* x, int* idx, int rows, int cols) {
    argmaxRowsNaiveKernel<<<(rows + NAIVE_BLOCK - 1) / NAIVE_BLOCK, NAIVE_BLOCK>>>(x, idx, rows, cols);
}
