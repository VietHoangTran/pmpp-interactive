// Chapter 21, summary page: a kernel of our own, not the book's, shown for one question.
// It applies a logit bias in the gather form of Fig. 21.6: one thread per vocabulary entry
// scans the whole list of (token id, bias) pairs and adds the ones that name its entry.
// The lesson's copy is the part below the marker, tied to this file by
// app/src/chapters/ch21/Wrap.snippets.test.ts. Only the CUDA front-end check parses it;
// it has never been run.
// ---- shown in the lesson ----
// Ours: a logit bias in the gather form of Fig. 21.6. The list holds n
// (token id, bias) pairs, and a token may appear in it more than once.
__global__ void biasGather(float* logits, int vocab,
                           const int* ids, const float* bias, int n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's logit
    if (v >= vocab) return;
    float add = 0.0f;
    for (int e = 0; e < n; ++e)   // every thread scans every pair
        if (ids[e] == v) add += bias[e];
    logits[v] += add;             // one writer per logit: no atomic
}
