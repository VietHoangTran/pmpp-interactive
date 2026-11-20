// Lab 22.2 reference solution (CUDA; also built for HIP and for the CPU emulators).
// Milestone M8: one greedy decode step for a batch of sequences, built from the project's
// own pieces, so that the benchmark can measure what batching buys.
//
// Book: section 22.5 (p. 539): batching the queries of many users turns each projection's
// vector-matrix products into one matrix product that reuses the weights; each query waits
// longer, the group finishes sooner. Section 22.2 (pp. 532-533) for the embedding's
// decomposition. The chapter has no code listings: every line here is ours.
//
// The step, for s.batch sequences at once (decode.h has the shapes):
//   H = E[tokens]                          embed: output-centric gather, one thread per element
//   for each layer l:
//     T = H W_l^T                          m.linear: M2, M3 or lab 22.1's gather
//     T += b_l                             M0 (lab 2.3), in place
//     H += T                               M0
//   logits = H E^T                         the LM head; its weight is E (tied, ours)
//   probs = softmax(logits / temperature)  M4 (lab 10.3); a sampler (M5) reads it,
//                                          greedy decoding does not
//   tokens = argmax(logits)                M4, over the tokens this step read
// One launch per operation for the whole batch, and nothing in between waits for the GPU.
// Launches without a stream go to the default stream, which runs them in the order they
// were issued (ch02-facts A5; HIP: B3), so each one sees what the one before it wrote.
#include "../decode.h"

// Every launch of the embedding uses blocks of this many threads, as M0 does.
const int threadsPerBlock = 256;

// Output-centric (section 22.2): thread i owns output element i = r * d + j of H and
// gathers the one input it needs, E[tokens[r]][j]. Neighbouring threads read neighbouring
// floats of one row of E and write neighbouring floats of one row of H. Two sequences on
// the same token read the same row: reads never conflict.
__global__ void embedGatherKernel(const float* E, const int* tokens, float* H, int batch, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch * d) {
        int r = i / d;   // the sequence
        int j = i % d;   // the feature
        // The token ids stay on the device: the previous step's argmax wrote them.
        H[i] = E[(size_t)tokens[r] * d + j];
    }
}

void launchEmbed(const float* E, const int* tokens, float* H, int batch, int d) {
    int n = batch * d;
    if (n <= 0) return;   // an empty grid is not a valid launch
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    embedGatherKernel<<<numBlocks, threadsPerBlock>>>(E, tokens, H, batch, d);
}

void launchDecodeStep(const DecodeModel& m, DecodeState& s) {
    launchEmbed(m.E, s.tokens, s.H, s.batch, m.d);
    for (int l = 0; l < m.layers; ++l) {
        m.linear(s.H, m.W[l], s.T, s.batch, m.d, m.d);       // T = H W_l^T: one GEMM for the batch
        launchAddBias(s.T, s.T, m.b[l], s.batch, m.d);        // T += b_l
        launchAddResidual(s.H, s.T, s.batch * m.d);           // H += T
    }
    // E is [vocab x d], which is the (out, in) layout the linear layer takes: no transpose.
    m.linear(s.H, m.E, s.logits, s.batch, m.d, m.vocab);
    launchSoftmaxRows(s.logits, s.probs, s.batch, m.vocab, 1.0f / s.temperature);
    // The next tokens go where this step read its tokens, so the next step starts on the
    // device, with no copy to the host and back between steps. (Section 22.3, pp. 537-538:
    // whatever stays on the host bounds the speed-up. The mapping to decoding is ours.)
    launchArgmaxRows(s.logits, s.tokens, s.batch, m.vocab);
}
