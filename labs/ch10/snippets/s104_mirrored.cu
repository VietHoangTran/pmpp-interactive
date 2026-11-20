// Book exercise 3 (p. 249): Fig. 10.8 changed to the mirrored pattern of the unnumbered
// figure on the same page. Thread t owns element t + blockDim.x, in the second half of
// the input; the stride halves from blockDim.x to 1 as in Fig. 10.8; the active threads
// are the last `stride` ones; the total ends in the last element. The answer is ours (the
// book prints none). Section 10.4 shows this file as the reference answer of "Mirror it",
// and app/src/chapters/ch10/S104.snippets.test.ts ties it to the model that runs it.

__global__ void MirroredSumReductionKernel(float* input, float* output) {
    unsigned int i = threadIdx.x + blockDim.x;
    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
        if (threadIdx.x >= blockDim.x - stride) {
            input[i] += input[i - stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == blockDim.x - 1) {
        *output = input[2*blockDim.x - 1];
    }
}
