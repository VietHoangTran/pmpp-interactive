// Provided for lab 12.4 (do not edit): the benchmark baseline, one thread per row.
//
// Thread r runs llama.cpp's host min-p loop over row r (notes/today/ch12-facts.md D3: keep a
// candidate when its logit is >= the threshold, append it, count it). That loop is a
// sequential stable filter, the operation this lab parallelises, so it is both the
// benchmark baseline and a second implementation of the contract. The tests run it too, so
// a broken copy of this file shows.
//
// What makes it slow is the chapter's subject and lab 10.3's: a batch of R rows uses R
// threads, so at batch 1 one thread walks the whole vocabulary; and neighbouring threads
// read addresses `cols` floats apart, so no access is coalesced (chapter 5).
//
// Contract as in the lab (README.md): [rows x cols] float32 logits, row-major, batch first;
// threshold[rows]; finite values.

#define SERIAL_BLOCK 64

__global__ void compactSerialKernel(const float* logits, const float* threshold, int* outIdx,
                                    float* outVal, unsigned int* counts, int rows, int cols) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const size_t rowStart = (size_t)r * cols;
    const float* in = logits + rowStart;
    const float thr = threshold[r];
    unsigned int n = 0;
    for (int c = 0; c < cols; ++c) {
        const float v = in[c];
        if (v >= thr) {   // llama.cpp: if (logit >= min_logit) push_back
            outIdx[rowStart + n] = c;
            outVal[rowStart + n] = v;
            ++n;
        }
    }
    counts[r] = n;
}

void launchCompactSerial(const float* logits, const float* threshold, int* outIdx, float* outVal,
                         unsigned int* counts, int rows, int cols) {
    compactSerialKernel<<<(rows + SERIAL_BLOCK - 1) / SERIAL_BLOCK, SERIAL_BLOCK>>>(
        logits, threshold, outIdx, outVal, counts, rows, cols);
}
