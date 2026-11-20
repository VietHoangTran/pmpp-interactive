// Lab 9.3 starter (CUDA; also built for HIP and for the CPU emulators).
// Token counts for repetition penalties: the counting piece of milestone M5.
// Book: sections 9.2 to 9.5, pp. 203-217. Counting token ids instead of pixels, and
// the penalty formula, are not in the book.
//
// Contract (row-major, batch first, raw device pointers):
//   tokens     [batch][len]    int32 ids, each in [0, vocab)
//   counts     [batch][vocab]  uint32; the caller zeroes it before countTokens
//   newTokens  [batch]         one id per row, in [0, vocab)
//   logits     [batch][vocab]  float, updated in place
//   batch <= 65535; batch * vocab < 2^32.
// vocab runs from 1,000 in the small tests to 201,088. len runs from 1 to 4,096.
//
// Before you write anything, answer the README's first question: which of chapter 9's
// privatisations still apply when the bins are a whole vocabulary per request?
// Your answer decides what countTokens looks like.
#include <cuda/atomic>

__global__ void countTokens(const int* tokens, unsigned int* counts, int len, int vocab) {
    // TODO 1: which request (row) is this block counting? Point at that row's tokens
    //         and at that row's counts.
    // TODO 2: several blocks may share one row. Visit this thread's share of the row's
    //         len tokens so that, together, the row's blocks cover every token once,
    //         whatever len is and however many blocks there are. Let neighbouring
    //         threads read neighbouring tokens (section 9.5).
    // TODO 3: add 1 to the token's bin. Which threads can update the same bin at the
    //         same time? Write the update in the book's API (section 9.2) with the
    //         narrowest scope that covers all of them.
}

void launchCountTokens(const int* tokens, unsigned int* counts, int batch, int len, int vocab) {
    // TODO 4: choose a block size and a number of blocks per row, and launch one set of
    //         blocks per request.
}

__global__ void appendTokens(const int* newTokens, unsigned int* counts, int batch, int vocab) {
    // TODO 5: one decode step. Each request gains one token, newTokens[row]: add it to
    //         that row's counts. How many threads can touch a given row? What does that
    //         say about the kind of update you need?
}

void launchAppendTokens(const int* newTokens, unsigned int* counts, int batch, int vocab) {
    // TODO 6: launch enough threads for the batch.
}

__global__ void applyPenalties(float* logits, const unsigned int* counts, size_t n,
                               float frequencyPenalty, float presencePenalty) {
    // TODO 7: one thread per element of the batch's logits:
    //           logit -= count * frequencyPenalty + (count > 0 ? 1 : 0) * presencePenalty
    //         (llama.cpp's formula). With both penalties 0 the logits must not change.
}

void launchApplyPenalties(float* logits, const unsigned int* counts, int batch, int vocab,
                          float frequencyPenalty, float presencePenalty) {
    // TODO 8: n = batch * vocab elements; launch enough threads for all of them.
}
