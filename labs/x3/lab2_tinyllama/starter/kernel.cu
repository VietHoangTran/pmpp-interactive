// Lab X3.2 starter (CUDA; also built for HIP and for the CPU emulators).
// Milestone M10: one greedy decode step of a Llama-style decoder, composed from the
// project's own pieces, and the loop that generates tokens with it.
//
// Not in the book: the model is llama2.c's `forward` (notes/today/x3-facts.md D1), and the
// code here is ours. llama.h has the model, the state, and every piece you call with its
// argument order. You write no arithmetic kernel: you decide the order of the pieces, their
// arguments, and where each result lives.
#include <cmath>

#include "../llama.h"

// ---------------------------------------------------------------- advance
// One thread per sequence b, after the step's argmax has written tokens[b]:
//   out[pos[b] * batch + b] = tokens[b]           record this step's argmax at its position
//   if pos[b] + 1 < promptLen:                    still inside the prompt:
//       tokens[b] = prompt[(pos[b] + 1) * batch + b]   feed the prompt, not the argmax
//   pos[b] += 1, len[b] += 1, rngStep[b] += 1
// TODO 1: write it. This kernel is what lets the generation loop stay on the device: ask
//         what the host would have to do between steps without it.
__global__ void advancePositionsKernel(int* tokens, int* pos, int* len, const int* prompt, int promptLen, int* out,
                                       unsigned long long* rngStep, int batch) {
}

void launchAdvancePositions(int* tokens, int* pos, int* len, const int* prompt, int promptLen, int* out,
                            unsigned long long* rngStep, int batch) {
    // TODO 2: one thread per sequence. An empty grid is not a valid launch.
}

// ---------------------------------------------------------------- one step
void launchLlamaStep(const LlamaModel& m, LlamaState& s) {
    // TODO 3: one step for all s.batch sequences, one launch per operation for the whole
    //         batch, and nothing that waits for the GPU or copies to the host. In order:
    //           embed s.tokens into s.x;
    //           for each layer l (weights m.layer[l], caches s.Kc[l] and s.Vc[l]):
    //             xb = rmsnorm(x) with rmsAtt;  q, k, v = xb times wq, wk, wv (m.linear);
    //             rotate q (n_heads heads) and k (n_kv_heads heads) at s.pos, pairing m.ropeMode;
    //             append k and v to the caches at s.pos;
    //             att = attention of q over the cache (m.attention), scale 1 / sqrt(128);
    //             xb2 = att times wo;  x += xb2;
    //             xb = rmsnorm(x) with rmsFfn;  hb = xb w1^T, hb2 = xb w3^T;
    //             hb = silu(hb) * hb2;  xb2 = hb w2^T;  x += xb2;
    //           xb = rmsnorm(x) with rmsFinal;  logits = xb E^T (the LM head is E);
    //           the next tokens: s.sample if the driver set it, else the argmax;
    //           then your advance.
    //         llama.h gives each piece's argument order and the shape of every buffer. Which
    //         `len` must the attention see so that it reads the row you just appended, and
    //         no row beyond it? Where must the next tokens go so that the next step can
    //         start without a trip to the host?
}
