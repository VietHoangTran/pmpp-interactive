// Lab X3.2 reference solution (CUDA; also built for HIP and for the CPU emulators).
// Milestone M10: one greedy decode step of a Llama-style decoder, composed from the
// project's own pieces, and the loop that generates tokens with it.
//
// Not in the book: the model is llama2.c's `forward` (notes/today/x3-facts.md D1), and every
// line here is ours. The pieces are earlier labs' reference solutions (llama.h lists them):
// this file only decides their order, their arguments and where each result lives.
//
// The step, for s.batch sequences at once, one launch per operation for the whole batch:
//   x = E[tokens]                                   embed (M8)
//   per layer: rmsnorm, q k v, RoPE on q and k, KV append, attention, wo, residual,
//              rmsnorm, w1 and w3, SwiGLU, w2, residual
//   rmsnorm, logits = x E^T (the tied LM head), argmax into tokens (or the sampler, M5)
//   advance: record the argmax, feed the next prompt token if any, pos and len up by one
// Launches without a stream go to the default stream and run in issue order (ch02-facts A5;
// HIP: B3), so each one sees what the one before it wrote, and nothing here waits.
#include <cmath>

#include "../llama.h"

// ---------------------------------------------------------------- advance
// One thread per sequence. This is the only kernel the step adds: it is what lets the
// whole generation loop stay on the device. The step's argmax is recorded at the position
// it was computed for; inside the prompt the next prompt token replaces it (the prompt is
// fed one token per step, as llama2.c does, D1); then the position moves on.
__global__ void advancePositionsKernel(int* tokens, int* pos, int* len, const int* prompt, int promptLen, int* out,
                                       unsigned long long* rngStep, int batch) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch) {
        const int p = pos[b];
        out[p * batch + b] = tokens[b];              // this step's argmax, for position p + 1
        if (p + 1 < promptLen) {
            tokens[b] = prompt[(p + 1) * batch + b]; // still in the prompt: feed it, not the argmax
        }
        pos[b] = p + 1;
        len[b] = p + 2;                              // the next step attends over p + 2 rows
        rngStep[b] += 1;                             // the sampler's counter follows the position
    }
}

void launchAdvancePositions(int* tokens, int* pos, int* len, const int* prompt, int promptLen, int* out,
                            unsigned long long* rngStep, int batch) {
    if (batch <= 0) {
        return;
    }
    const int threads = 256;
    advancePositionsKernel<<<(batch + threads - 1) / threads, threads>>>(tokens, pos, len, prompt, promptLen, out,
                                                                         rngStep, batch);
}

// ---------------------------------------------------------------- one step
void launchLlamaStep(const LlamaModel& m, LlamaState& s) {
    const LlamaConfig& c = m.cfg;
    const int B = s.batch, d = c.dim, hidden = c.hidden_dim;
    const int dq = c.n_heads * LLAMA_HEAD_DIM;      // = d
    const int dkv = c.n_kv_heads * LLAMA_HEAD_DIM;
    // llama2.c divides the scores by sqrtf(head_size) (D1); the decode kernel multiplies.
    const float scale = 1.0f / sqrtf((float)LLAMA_HEAD_DIM);

    launchEmbed(m.E, s.tokens, s.x, B, d);
    for (int l = 0; l < c.n_layers; ++l) {
        const LlamaLayer& w = m.layer[l];
        // Attention block.
        launchRmsNorm(s.x, w.rmsAtt, s.xb, B, d, m.eps);
        m.linear(s.xb, w.wq, s.q, B, d, dq);
        m.linear(s.xb, w.wk, s.k, B, d, dkv);
        m.linear(s.xb, w.wv, s.v, B, d, dkv);
        launchRope(s.q, s.pos, s.cosT, s.sinT, B, c.n_heads, LLAMA_HEAD_DIM, m.ropeMode);
        launchRope(s.k, s.pos, s.cosT, s.sinT, B, c.n_kv_heads, LLAMA_HEAD_DIM, m.ropeMode);
        // The rotated key goes into the cache: later steps reuse it as it is.
        launchKvAppend(s.k, s.v, s.Kc[l], s.Vc[l], s.pos, B, c.seq_len, c.n_kv_heads);
        // len = pos + 1: the row just appended is the last one the query sees.
        m.attention(s.q, s.Kc[l], s.Vc[l], s.len, s.att, B, c.n_heads, c.n_kv_heads, c.seq_len, scale);
        m.linear(s.att, w.wo, s.xb2, B, dq, d);
        launchAddResidual(s.x, s.xb2, B * d);
        // Feed-forward block.
        launchRmsNorm(s.x, w.rmsFfn, s.xb, B, d, m.eps);
        m.linear(s.xb, w.w1, s.hb, B, d, hidden);
        m.linear(s.xb, w.w3, s.hb2, B, d, hidden);
        launchSwiGLU(s.hb, s.hb2, s.hb, B * hidden);
        m.linear(s.hb, w.w2, s.xb2, B, hidden, d);
        launchAddResidual(s.x, s.xb2, B * d);
    }
    launchRmsNorm(s.x, m.rmsFinal, s.xb, B, d, m.eps);
    // E is [vocab x d], the (out, in) layout the linear layer takes: the LM head needs no copy.
    m.linear(s.xb, m.E, s.logits, B, d, c.vocab_size);
    // The next tokens go where this step read its tokens: the next step starts on the device.
    if (s.sample) {
        s.sample(s.sampleCtx, s.logits, s.tokens, s.rngStep, B, c.vocab_size);
    } else {
        launchArgmaxRows(s.logits, s.tokens, B, c.vocab_size);
    }
    launchAdvancePositions(s.tokens, s.pos, s.len, s.prompt, s.promptLen, s.out, s.rngStep, B);
}
