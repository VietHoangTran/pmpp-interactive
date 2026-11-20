// Lab 3.5 reference solution (CUDA; also built for HIP and for the CPU emulator):
// the four defects in the starter, fixed.
//
// Windowed mean along the sequence axis of a [batch, seq, hidden] float tensor:
//   out[b][t][h] = mean of in[b][s][h] over s = t-r .. t+r, keeping only the
//                  positions s that exist (0 <= s < seq).
// Layout: row-major, batch first. Mapping: hidden on x, seq on y, batch on z.
//
// This kernel is not a listing from the book. Its parts are: the thread-to-data
// mapping of section 3.2 with the 3D form `P[plane*m*n + row*m + col]` (p. 55),
// where m is the number of columns (here hidden) and n the number of rows (here
// seq); and the clipped loop with its counter from the blur kernel (Fig. 3.8,
// p. 57), kept to one axis. Where it differs from those pages: the radius is an
// argument `r` where the book has the constant BLUR_SIZE, the data is float where
// the blur's is unsigned char, and the grid is sized with integer round-up where
// p. 50 prints ceil(m/16.0).

__global__ void seqMeanKernel(const float* in, float* out, int batch, int seq, int hidden, int r) {
    int h = blockIdx.x*blockDim.x + threadIdx.x;   // column: position along hidden
    int t = blockIdx.y*blockDim.y + threadIdx.y;   // row:    position along seq
    int b = blockIdx.z*blockDim.z + threadIdx.z;   // plane:  which sequence of the batch

    // FIX B: the grid is whole blocks on all three axes, so all three indices need
    // the test. The starter left out `b < batch`: when batch is not a multiple of
    // the block's z extent, the surplus planes of threads read and wrote the
    // memory after the tensors. Every element inside the tensor was still correct,
    // which is why only a bounds check or a guard region shows this one.
    if (h < hidden && t < seq && b < batch) {
        float sum = 0;
        int count = 0;
        for (int d = -r; d < r+1; ++d) {
            int cur = t + d;
            // FIX C: clip the window at both ends. The starter tested only
            // `cur < seq`. A negative `cur` then indexed the row before the
            // sequence's first: the tail of the previous sequence, or for b = 0
            // the memory before the tensor. It was counted too, so the mean was
            // taken over positions that do not exist.
            if (cur >= 0 && cur < seq) {
                sum += in[b*seq*hidden + cur*hidden + h];
                ++count;   // how many positions went into the sum
            }
        }
        // FIX A: a row holds `hidden` elements, so the row stride is hidden. The
        // starter wrote `t*seq`. With seq < hidden, [t][h] and [t+1][h-seq] then
        // named the same element (two threads, one write each, the last one wins)
        // and the end of every plane was never written. With seq > hidden the rows
        // landed too far apart: gaps inside the plane, rows spilling into the next
        // plane, and the last ones past the end of the tensor.
        out[b*seq*hidden + t*hidden + h] = sum/count;
    }
}

void launchSeqMean(const float* in, float* out, int batch, int seq, int hidden, int r) {
    dim3 dimBlock(8, 8, 4);
    // FIX D: dim3 is (x, y, z), lowest dimension first; the tensor is written
    // [batch, seq, hidden], highest first. x must cover hidden and y must cover
    // seq. The starter sized x from seq and y from hidden, so one axis had too
    // few blocks (elements never computed) and the other too many (idle threads).
    dim3 dimGrid((hidden + dimBlock.x - 1) / dimBlock.x,
                 (seq    + dimBlock.y - 1) / dimBlock.y,
                 (batch  + dimBlock.z - 1) / dimBlock.z);
    seqMeanKernel<<<dimGrid, dimBlock>>>(in, out, batch, seq, hidden, r);
}
