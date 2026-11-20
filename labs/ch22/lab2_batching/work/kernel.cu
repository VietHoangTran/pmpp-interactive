// Lab 22.2 starter (CUDA; also built for HIP and for the CPU emulators).
// Milestone M8: one greedy decode step for a batch of sequences, built from the project's
// own pieces, so that the benchmark can measure what batching buys.
//
// Book: section 22.5 (p. 539): batching the queries of many users turns each projection's
// vector-matrix products into one matrix product that reuses the weights. Section 22.2
// (pp. 532-533) for the embedding's decomposition. The chapter has no code listings: the
// code here is ours.
//
// decode.h has the shapes, DecodeModel, DecodeState, and the earlier labs' pieces you call.
#include "../decode.h"

// H[r][j] = E[tokens[r]][j] for r < batch, j < d. E is [vocab x d], H is [batch x d],
// both row-major.
// TODO 1: output-centric (section 22.2): one thread per element of H. The thread that owns
//         H[i] works out which sequence and which feature i is, and gathers the one float
//         of E that belongs there. The token ids are device memory: read them in the kernel.
__global__ void embedGatherKernel(const float* E, const int* tokens, float* H, int batch, int d) {
}

void launchEmbed(const float* E, const int* tokens, float* H, int batch, int d) {
    // TODO 2: launch embedGatherKernel with one thread per element of H (M0, lab 2.3, uses
    //         blocks of 256). An empty grid is not a valid launch.
}

void launchDecodeStep(const DecodeModel& m, DecodeState& s) {
    // TODO 3: one greedy step for all s.batch sequences: one launch per operation for the
    //         whole batch, and nothing that waits for the GPU. In order:
    //           embed the current tokens into H;
    //           for each layer l: T = H W_l^T (m.linear), T += b_l, H += T;
    //           logits = H Wout^T, where Wout is the embedding table (tied, decode.h);
    //           probs = softmax(logits), with scale 1 / s.temperature;
    //           the next tokens = argmax(logits).
    //         decode.h gives each piece's argument order. Where must the next tokens go so
    //         that the next step can start without a trip to the host?
}
