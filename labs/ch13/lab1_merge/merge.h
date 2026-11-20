// Lab 13.1: the element type, the block size and the entry points. Read-only for you.
//
// Every array in this lab holds key-value pairs. Merging compares .key only; .val rides
// along. That is what makes stability visible: two elements with the same key are told
// apart by their values, so a merge that puts them in the wrong order gives a different
// array. (A merge of plain ints cannot be caught being unstable.)
//
// kernel.metal declares the same struct (a Metal source compiled at run time cannot
// include this file); the two must keep the same two 32-bit ints in the same order.
#pragma once

struct KV {
    int key;
    int val;
};

// Threads per block in every launch of this lab, on every backend.
constexpr int MERGE_BLOCK = 128;

// Entries the tiled merge's `bounds` array needs for `total` outputs at `perThread`
// outputs per thread: one per tile boundary, ceil(total / tile) + 1, where a tile is
// the MERGE_BLOCK * perThread outputs of one block.
inline int mergeBoundsCount(int total, int perThread) {
    const int tile = MERGE_BLOCK * perThread;
    return (total + tile - 1) / tile + 1;
}

// ---- CUDA, HIP and the emulators: implemented in <variant>/kernel.cu ----
// All pointers are device pointers. A and B are sorted by key (m and n elements, either
// may be 0). C receives the first `total` outputs of their stable merge, 0 <= total <= m + n;
// nothing at or past C[total] may be written.

// Provided: out[k] = co_rank(k, A, m, B, n) for every k from 0 to m + n.
void launchCoRank(const KV* A, int m, const KV* B, int n, int* out);
// Part (a): one launch, every thread searches global memory.
void launchMergeBasic(const KV* A, int m, const KV* B, int n, KV* C, int total, int perThread);
// Part (b): a partition launch, then a launch that stages each block's inputs in shared
// memory. `bounds` has mergeBoundsCount(total, perThread) entries.
void launchMergeTiled(const KV* A, int m, const KV* B, int n, KV* C, int total, int perThread, int* bounds);
