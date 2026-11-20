// Provided for lab 11.6 (do not edit): a copy with the top-p kernel's traffic.
//
// Element i: read in[i] (4 bytes), write outF[i] = in[i] (4 bytes) and outB[i] = 1 (1 byte),
// the 9 bytes per element a top-p mask must move at least. One thread per element, every
// access coalesced, no arithmetic. Its bandwidth is the yardstick the lab's benchmark reports
// your kernel against ("share of copy"), because it is what this device achieves for this
// traffic in practice, not what a data sheet promises (ch11-facts A2).

#define COPY_BLOCK 256

__global__ void TopPCopyKernel(const float* in, float* outF, unsigned char* outB, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        outF[i] = in[i];
        outB[i] = 1;
    }
}

void launchTopPCopy(const float* in, float* outF, unsigned char* outB, int n) {
    if (n <= 0) return;
    TopPCopyKernel<<<((unsigned int)n + COPY_BLOCK - 1) / COPY_BLOCK, COPY_BLOCK>>>(in, outF, outB, (unsigned int)n);
}
