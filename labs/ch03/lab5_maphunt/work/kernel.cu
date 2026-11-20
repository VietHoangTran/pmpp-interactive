// Lab 3.5 starter (CUDA; also built for HIP and for the CPU emulator).
//
// Windowed mean along the sequence axis of a [batch, seq, hidden] float tensor:
//   out[b][t][h] = mean of in[b][s][h] over s = t-r .. t+r, keeping only the
//                  positions s that exist (0 <= s < seq).
// Layout: row-major, batch first, so h is the index that varies fastest.
// Mapping: hidden on x, seq on y, batch on z. Blocks of 8 x 8 x 4 threads.
//
// This kernel is not a listing from the book. Its parts are: the thread-to-data
// mapping of section 3.2 with its 3D form (p. 55), and the clipped loop with its
// counter from the blur kernel (Fig. 3.8, p. 57), kept to one axis.
//
// This file has FOUR independent defects. It compiles, and it passes the test
// named 'cube'. Find and fix all four.
// Run it on the CPU emulator first and fix one thing at a time: README.md says
// what each kind of failure looks like there.

__global__ void seqMeanKernel(const float* in, float* out, int batch, int seq, int hidden, int r) {
    int h = blockIdx.x*blockDim.x + threadIdx.x;   // column: position along hidden
    int t = blockIdx.y*blockDim.y + threadIdx.y;   // row:    position along seq
    int b = blockIdx.z*blockDim.z + threadIdx.z;   // plane:  which sequence of the batch

    if (h < hidden && t < seq) {
        float sum = 0;
        int count = 0;
        for (int d = -r; d < r+1; ++d) {
            int cur = t + d;
            if (cur < seq) {
                sum += in[b*seq*hidden + cur*hidden + h];
                ++count;   // how many positions went into the sum
            }
        }
        out[b*seq*hidden + t*seq + h] = sum/count;
    }
}

void launchSeqMean(const float* in, float* out, int batch, int seq, int hidden, int r) {
    dim3 dimBlock(8, 8, 4);
    dim3 dimGrid((seq    + dimBlock.x - 1) / dimBlock.x,
                 (hidden + dimBlock.y - 1) / dimBlock.y,
                 (batch  + dimBlock.z - 1) / dimBlock.z);
    seqMeanKernel<<<dimGrid, dimBlock>>>(in, out, batch, seq, hidden, r);
}
