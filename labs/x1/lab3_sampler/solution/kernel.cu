// Lab X1.3 reference solution (CUDA; also built for HIP and for the CPU emulators).
// Milestone M5 complete: a batched GPU sampler. [rows x vocab] logits and token counts in, one
// token id per row out, deterministic per (seed, stream, step).
//
// Not in the book, and ours: sampling (PLAN.md). The stages are earlier labs' kernels, provided
// as copies in labs/x1/shared-g11/; this file adds what no earlier lab has: a counter-based
// random number per row, the repetition penalty, the final draw, and the composition.
//
// The chain (llama.cpp's default order, notes/today/x1-facts.md D1, D2; README.md says why):
//   copy logits to the workspace -> repetition penalty -> frequency and presence (lab 9.3)
//   -> temperature <= 0: argmax, lowest id on ties (lab 10.3), done
//   -> top-k select (lab X1.1; topK <= 0 or >= vocab keeps the whole row)
//   -> sort the k candidates, descending and stable (lab 14.3)
//   -> softmax at scale 1 (lab 10.3) -> top-p keep count (lab 11.6)
//   -> min-p count, n = max(1, min(kept, countMinP)) (provided, minp_g11.cu)
//   -> draw among the first n with weights exp((l_i - l_0) / T) (this file)
// Every tie at every stage goes to the lowest token id: the select keeps the lowest columns, it
// hands them over in column order, and the sort is stable.
//
// The draw (this file, drawRows): one block per row, as FlashInfer's sampling kernel does
// (x1-facts A6). Weights w_i = exp((l_i - l_0) * invT) over the n survivors, l_0 the largest;
// total by block reduction (section 10.8); u in [0, 1) from Philox4x32-10 (x1-facts A4) with
// counter (step low, step high, stream low, stream high) and key (seed low, seed high), word 0,
// converted exactly as (w >> 8) * 2^-24 (x1-facts E2); target = u * total. The row is walked in
// chunks of blockDim.x * DRAW_VEC weights with a running aggregate; a chunk that cannot hold the
// crossing is skipped after its scan; in the chunk that does, the first position with a positive
// weight whose inclusive sum is greater than the target wins (ballot, first set lane, lowest
// warp). If rounding leaves no such position, the last positive weight wins (FlashInfer has the
// same fallback, A6).
#include <cfloat>

// The course's CUDA front-end check (labs/pmpp.py) declares no CUDA Math API functions, so it
// is given expf's device declaration here. nvcc, hipcc and the emulator never see it.
#if defined(PMPP_CUDA_FRONTEND_CHECK)
__device__ float expf(float x);
#endif

#define BLOCK_SIZE 256                          // threads per block for the row kernels
#define DRAW_VEC 4                              // consecutive weights per thread per chunk
#define MIN_WARP_SIZE 32                        // the narrowest warp any backend here has
#define MAX_WARPS (BLOCK_SIZE / MIN_WARP_SIZE)  // per-warp slots in shared memory
#define ELEMENTWISE_BLOCK 256                   // threads per block for copy and repetition

// ---------------------------------------------------------------- provided: the stages of earlier labs
// Lab 9.3 (penalties_g11.cu), lab 10.3 (softmax_g11.cu), lab X1.1 (topk_g11.cu), lab 14.3
// (sortrows_g11.cu), lab 11.6 (topp_g11.cu), and the min-p count (minp_g11.cu).
void launchApplyPenalties(float* logits, const unsigned int* counts, int batch, int vocab,
                          float frequencyPenalty, float presencePenalty);
void launchArgmaxRows(const float* x, int* idx, int rows, int cols);
void launchSoftmaxRows(const float* x, float* y, int rows, int cols, float scale);
void launchTopKRows(const float* logits, int* outIdx, float* outVal, unsigned int* scratch, int rows, int cols,
                    int k);
void launchSortRowsDesc(const float* probs, float* sortedProbs, int* sortedIdx, unsigned int* scratch, int rows,
                        int cols);
void launchTopPMask(const float* probs, float* massBefore, unsigned char* keep, int* kept, int rows, int cols,
                    float p);
void launchMinPCombine(const float* sortedLogits, const int* kept, int* n, int rows, int cols, float minP);

// ---------------------------------------------------------------- provided: warp helpers
// The vote's type and the bit operations that match it (infra-facts B3, ch12-facts B1).
#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
__device__ unsigned int firstLane(lane_mask_t m) { return __ffsll((long long)m) - 1; }   // m != 0
#else
typedef unsigned int lane_mask_t;
__device__ unsigned int firstLane(lane_mask_t m) { return __ffs((int)m) - 1; }           // m != 0
#endif
__device__ unsigned int laneIdx() { return threadIdx.x % warpSize; }
__device__ unsigned int warpIdx() { return threadIdx.x / warpSize; }
__device__ unsigned int warpsInBlock() { return (blockDim.x + warpSize - 1) / warpSize; }

// ---------------------------------------------------------------- 1. Philox4x32-10
// Random123's generator (include/Random123/philox.h; x1-facts A4), written here, not copied.
// Random123 is distributed under this notice:
//   Copyright 2010-2011, D. E. Shaw Research. All rights reserved. Redistribution and use in
//   source and binary forms, with or without modification, are permitted provided that the
//   following conditions are met: redistributions of source code must retain the above
//   copyright notice, this list of conditions, and the following disclaimer; redistributions in
//   binary form must reproduce them in the documentation and/or other materials provided with
//   the distribution; neither the name of D. E. Shaw Research nor the names of its contributors
//   may be used to endorse or promote products derived from this software without specific
//   prior written permission. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
//   CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES ... ARE DISCLAIMED.
//
// The 32 x 32 -> 64-bit product is an unsigned long long multiplication: the same source on
// CUDA, HIP and the emulators (x1-facts A5). __umulhi would do on CUDA and HIP (A5, B3), but
// neither the emulator nor the front-end stub declares it (x1-facts, project notes).
__device__ uint4 philox4x32_10(uint4 ctr, uint2 key) {
    for (int round = 0; round < 10; ++round) {
        if (round > 0) {                         // the key is bumped between rounds, not before the first
            key.x += 0x9E3779B9u;
            key.y += 0xBB67AE85u;
        }
        const unsigned long long p0 = (unsigned long long)0xD2511F53u * ctr.x;
        const unsigned long long p1 = (unsigned long long)0xCD9E8D57u * ctr.z;
        const unsigned int hi0 = (unsigned int)(p0 >> 32), lo0 = (unsigned int)p0;
        const unsigned int hi1 = (unsigned int)(p1 >> 32), lo1 = (unsigned int)p1;
        ctr = make_uint4(hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0);
    }
    return ctr;
}

// The known-answer test: in[i] = (counter x, y, z, w, key x, y), out[i] = the four words.
__global__ void philoxKatKernel(const unsigned int* in, unsigned int* out, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const unsigned int* a = in + (size_t)i * 6;
        const uint4 w = philox4x32_10(make_uint4(a[0], a[1], a[2], a[3]), make_uint2(a[4], a[5]));
        out[(size_t)i * 4 + 0] = w.x;
        out[(size_t)i * 4 + 1] = w.y;
        out[(size_t)i * 4 + 2] = w.z;
        out[(size_t)i * 4 + 3] = w.w;
    }
}

void launchPhiloxKat(const unsigned int* in, unsigned int* out, int count) {
    if (count <= 0) return;
    philoxKatKernel<<<(count + 127) / 128, 128>>>(in, out, count);
}

// M5's mapping (ours): counter = (step low, step high, stream low, stream high), key = (seed
// low, seed high), word 0. One Philox call per (row, step), as cuRAND's curand_init(seed,
// stream, 4 * step) would give by our reading of its header (x1-facts A3).
__device__ unsigned int rowBits(unsigned long long seed, unsigned long long stream, unsigned long long step) {
    const uint4 ctr = make_uint4((unsigned int)step, (unsigned int)(step >> 32), (unsigned int)stream,
                                 (unsigned int)(stream >> 32));
    return philox4x32_10(ctr, make_uint2((unsigned int)seed, (unsigned int)(seed >> 32))).x;
}

// ---------------------------------------------------------------- 2. u from 32 bits
// The top 24 bits times 2^-24: an integer below 2^24 converts to float exactly, and a power of
// two scales exactly, so u is the same float on every backend under any math mode, in
// [0, 1 - 2^-24], never 1 (x1-facts E2).
__device__ float uniformFromBits(unsigned int w) {
    return (float)(w >> 8) * (1.0f / 16777216.0f);
}

// ---------------------------------------------------------------- 3. copy and repetition
// Provided: the workspace copy, one element per thread.
__global__ void copyRows(const float* in, float* out, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i];
    }
}

void launchCopyRows(const float* in, float* out, size_t n) {
    if (n == 0) return;
    copyRows<<<(unsigned int)((n + ELEMENTWISE_BLOCK - 1) / ELEMENTWISE_BLOCK), ELEMENTWISE_BLOCK>>>(in, out, n);
}

// llama.cpp's rule (x1-facts D2): a token seen before has a logit <= 0 multiplied by the
// penalty and a positive one divided by it. The CTRL paper divides regardless of sign, which
// would make a negative logit more likely (D2, D9).
__global__ void applyRepetition(float* logits, const unsigned int* counts, size_t n, float penalty) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && counts[i] > 0u) {
        const float x = logits[i];
        logits[i] = x <= 0.0f ? x * penalty : x / penalty;
    }
}

void launchApplyRepetition(float* logits, const unsigned int* counts, int rows, int vocab, float penalty) {
    const size_t n = (size_t)rows * (size_t)vocab;
    if (n == 0) return;
    applyRepetition<<<(unsigned int)((n + ELEMENTWISE_BLOCK - 1) / ELEMENTWISE_BLOCK), ELEMENTWISE_BLOCK>>>(
        logits, counts, n, penalty);
}

// ---------------------------------------------------------------- 4. the draw
__device__ float warpSum(float v) {
    const lane_mask_t full = ~(lane_mask_t)0;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        v += __shfl_down_sync(full, v, offset);
    }
    return v;
}

__device__ float warpInclusiveScan(float v) {
    const lane_mask_t full = ~(lane_mask_t)0;
    for (int stride = 1; stride < warpSize; stride *= 2) {
        const float left = __shfl_up_sync(full, v, stride);
        if ((int)laneIdx() >= stride) {
            v += left;
        }
    }
    return v;
}

// Row r: survivors sortedLogits[r * stride + i], i < n[r], largest first; sortedPos and candIdx
// map position i to a vocabulary id: candIdx[r * stride + sortedPos[r * stride + i]]. u comes
// from uIn when it is not null (the test entry), else from Philox. uOut, if not null, gets u.
__global__ void drawRows(const float* sortedLogits, const int* sortedPos, const int* candIdx, const int* n,
                         const unsigned long long* seeds, const unsigned long long* streams,
                         const unsigned long long* steps, const float* uIn, int* tokens, float* uOut, int stride,
                         float invT) {
    __shared__ float sums_s[MAX_WARPS];          // warp sums for the total
    __shared__ float chunk_s[2][MAX_WARPS];      // warp totals of a chunk, double-buffered
    __shared__ int hit_s[MAX_WARPS];             // each warp's first winning position, or -1

    const lane_mask_t full = ~(lane_mask_t)0;
    const size_t rowStart = (size_t)blockIdx.x * stride;
    const float* l = sortedLogits + rowStart;
    const int m = n[blockIdx.x];
    const float l0 = l[0];

    // (a) The total: a strided sum per thread, a sum per warp, then warp 0 (section 10.8).
    float part = 0.0f;
    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        part += expf((l[i] - l0) * invT);
    }
    part = warpSum(part);
    if (laneIdx() == 0) {
        sums_s[warpIdx()] = part;
    }
    __syncthreads();
    float total = 0.0f;
    for (unsigned int w = 0; w < warpsInBlock(); ++w) {   // the same order in every thread
        total += sums_s[w];
    }

    // (b) This row's u, and the target.
    const float u = uIn != nullptr ? uIn[blockIdx.x]
                                   : uniformFromBits(rowBits(seeds[blockIdx.x], streams[blockIdx.x], steps[blockIdx.x]));
    const float target = u * total;

    // (c) The chunks. Every thread holds the same aggregate and the same chunk total, so every
    //     decision below is uniform across the block.
    int winner = -1;
    float aggregate = 0.0f;
    int buf = 0;
    for (int base = 0; base < m; base += blockDim.x * DRAW_VEC, buf ^= 1) {
        const int first = base + threadIdx.x * DRAW_VEC;
        float local[DRAW_VEC];
        bool positive[DRAW_VEC];
        float run = 0.0f;
        for (int j = 0; j < DRAW_VEC; ++j) {
            const float w = first + j < m ? expf((l[first + j] - l0) * invT) : 0.0f;
            positive[j] = w > 0.0f;
            run += w;
            local[j] = run;                       // inclusive over this thread's weights
        }
        const float incl = warpInclusiveScan(run);
        if (laneIdx() == (unsigned int)warpSize - 1) {
            chunk_s[buf][warpIdx()] = incl;
        }
        __syncthreads();   // the chunk's warp totals are in; double-buffering means the next
                           // chunk writes the other half, so no second barrier is needed here
        float carry = 0.0f, chunk = 0.0f;
        for (unsigned int w = 0; w < warpsInBlock(); ++w) {
            const float s = chunk_s[buf][w];
            carry += w < warpIdx() ? s : 0.0f;
            chunk += s;
        }
        if (aggregate + chunk > target) {        // the crossing can be in this chunk: find it
            const float before = aggregate + carry + (incl - run);
            int hit = -1;
            for (int j = DRAW_VEC - 1; j >= 0; --j) {
                if (first + j < m && positive[j] && before + local[j] > target) {
                    hit = j;                      // the lowest such j wins
                }
            }
            const lane_mask_t vote = __ballot_sync(full, hit >= 0);
            if (vote == 0) {
                if (laneIdx() == 0) {
                    hit_s[warpIdx()] = -1;
                }
            } else if (laneIdx() == firstLane(vote)) {
                hit_s[warpIdx()] = first + hit;
            }
            __syncthreads();
            for (unsigned int w = 0; w < warpsInBlock() && winner < 0; ++w) {
                winner = hit_s[w];                // the lowest warp with a hit
            }
            // A later chunk rewrites hit_s only after its first barrier, which every thread
            // reaches after this read.
            if (winner >= 0) {
                break;
            }
        }
        aggregate += chunk;
    }

    // (d) Rounding left no sum above the target: the last positive weight (FlashInfer, A6).
    if (threadIdx.x == 0) {
        if (winner < 0) {
            winner = 0;
            for (int i = m - 1; i > 0; --i) {
                if (expf((l[i] - l0) * invT) > 0.0f) {
                    winner = i;
                    break;
                }
            }
        }
        tokens[blockIdx.x] = candIdx[rowStart + sortedPos[rowStart + winner]];
        if (uOut != nullptr) {
            uOut[blockIdx.x] = u;
        }
    }
}

void launchDrawRows(const float* sortedLogits, const int* sortedPos, const int* candIdx, const int* n,
                    const unsigned long long* seeds, const unsigned long long* streams,
                    const unsigned long long* steps, int* tokens, float* uOut, int rows, int stride, float invT) {
    if (rows <= 0) return;
    drawRows<<<rows, BLOCK_SIZE>>>(sortedLogits, sortedPos, candIdx, n, seeds, streams, steps, nullptr, tokens,
                                   uOut, stride, invT);
}

// The test entry: u from a buffer, so the driver can plant boundary cases.
void launchDrawRowsU(const float* sortedLogits, const int* sortedPos, const int* candIdx, const int* n,
                     const float* u, int* tokens, int rows, int stride, float invT) {
    if (rows <= 0) return;
    drawRows<<<rows, BLOCK_SIZE>>>(sortedLogits, sortedPos, candIdx, n, nullptr, nullptr, nullptr, u, tokens,
                                   nullptr, stride, invT);
}

// ---------------------------------------------------------------- 5. the sampler
// The workspace: every intermediate of the chain, each region rounded up to 256 bytes.
static int effectiveTopK(int vocab, int topK) { return topK <= 0 || topK >= vocab ? vocab : topK; }
static size_t region(size_t bytes) { return (bytes + 255) / 256 * 256; }

struct Workspace {
    float* logits;             // [rows x vocab] the penalised logits
    int* candIdx;              // [rows x K] lab X1.1's columns, ascending
    float* candVal;            // [rows x K] and their logits
    unsigned int* selScratch;  // [rows x 4]
    float* sortedVal;          // [rows x K] candidates, largest first
    int* sortedPos;            // [rows x K] their positions in candIdx
    unsigned int* sortScratch; // [4 x rows x K]
    float* probs;              // [rows x K] softmax of sortedVal
    float* mass;               // [rows x K] lab 11.6's mass before each token
    unsigned char* keep;       // [rows x K] lab 11.6's keep flags
    int* kept;                 // [rows] top-p's count
    int* n;                    // [rows] survivors of top-p and min-p
    size_t bytes;
};

static Workspace carve(void* base, int rows, int vocab, int topK) {
    const size_t R = (size_t)rows, V = (size_t)vocab, K = (size_t)effectiveTopK(vocab, topK);
    char* p = (char*)base;
    size_t at = 0;
    auto take = [&](size_t bytes) { char* q = p ? p + at : nullptr; at += region(bytes); return q; };
    Workspace w;
    w.logits = (float*)take(R * V * 4);
    w.candIdx = (int*)take(R * K * 4);
    w.candVal = (float*)take(R * K * 4);
    w.selScratch = (unsigned int*)take(R * 4 * 4);
    w.sortedVal = (float*)take(R * K * 4);
    w.sortedPos = (int*)take(R * K * 4);
    w.sortScratch = (unsigned int*)take(4 * R * K * 4);
    w.probs = (float*)take(R * K * 4);
    w.mass = (float*)take(R * K * 4);
    w.keep = (unsigned char*)take(R * K);
    w.kept = (int*)take(R * 4);
    w.n = (int*)take(R * 4);
    w.bytes = at;
    return w;
}

size_t sampleWorkspaceBytes(int rows, int vocab, int topK) {
    if (rows <= 0 || vocab <= 0) return 0;
    return carve(nullptr, rows, vocab, topK).bytes;
}

void launchSampleRows(const float* logits, const unsigned int* counts, const unsigned long long* seeds,
                      const unsigned long long* streams, const unsigned long long* steps, int* tokens, float* uOut,
                      void* workspace, int rows, int vocab, int topK, float topP, float minP, float temperature,
                      float repeatPenalty, float frequencyPenalty, float presencePenalty) {
    if (rows <= 0 || vocab <= 0) return;
    const int K = effectiveTopK(vocab, topK);
    const Workspace w = carve(workspace, rows, vocab, topK);
    const size_t n = (size_t)rows * (size_t)vocab;

    // Penalties on a copy: the caller's logits are read-only. A neutral setting is skipped, as
    // llama.cpp skips it; applying it would leave the logits bit-identical anyway.
    launchCopyRows(logits, w.logits, n);
    if (repeatPenalty != 1.0f) {
        launchApplyRepetition(w.logits, counts, rows, vocab, repeatPenalty);
    }
    if (frequencyPenalty != 0.0f || presencePenalty != 0.0f) {
        launchApplyPenalties(w.logits, counts, rows, vocab, frequencyPenalty, presencePenalty);
    }

    if (temperature <= 0.0f) {                   // greedy: no draw, and uOut is not written
        launchArgmaxRows(w.logits, tokens, rows, vocab);
        return;
    }

    launchTopKRows(w.logits, w.candIdx, w.candVal, w.selScratch, rows, vocab, K);
    launchSortRowsDesc(w.candVal, w.sortedVal, w.sortedPos, w.sortScratch, rows, K);
    launchSoftmaxRows(w.sortedVal, w.probs, rows, K, 1.0f);
    launchTopPMask(w.probs, w.mass, w.keep, w.kept, rows, K, topP);
    launchMinPCombine(w.sortedVal, w.kept, w.n, rows, K, minP);
    // 1 / T once on the host, multiplied in the kernel: llama.cpp's backend path does the same
    // (ggml_scale by 1 / temp); its host path divides (x1-facts D2).
    launchDrawRows(w.sortedVal, w.sortedPos, w.candIdx, w.n, seeds, streams, steps, tokens, uOut, rows, K,
                   1.0f / temperature);
}
