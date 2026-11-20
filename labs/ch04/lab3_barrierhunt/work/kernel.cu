// Lab 4.3 starter: reverses each block's slice of a[0..n) in place.
// A 1D grid: block b owns the elements b*blockDim.x up to (b+1)*blockDim.x, and
// the last slice stops where the array stops. Every thread reads its element
// into a register, then writes it to the mirrored position of its slice.
//
// This kernel has THREE independent defects. It compiles, and on a GPU it can
// pass for some block sizes and array lengths. Find and fix all three.
// Run it on the CPU emulator first: its thread orders are deterministic, so a
// synchronisation defect fails every time there instead of now and then, and
// AddressSanitizer stops at the first access outside the array.
//
// The launch below is correct. The driver chooses threadsPerBlock, and it does
// not only choose multiples of anything.

__global__ void reverseBlocksKernel(float* a, int n) {
    int t = threadIdx.x;
    int base = blockIdx.x * blockDim.x;     // where this block's slice starts
    int i = base + t;

    float mine = 0.0f;
    if (i < n) mine = a[i];

    if (i < n) a[base + 32 - 1 - t] = mine;
}

void launchReverseBlocks(float* a, int n, int threadsPerBlock) {
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    reverseBlocksKernel<<<blocks, threadsPerBlock>>>(a, n);
}
