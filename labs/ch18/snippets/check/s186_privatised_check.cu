// Runs one kernel of ../s186_privatised_variants.cu for one BFS level and prints the public
// frontier:
//
//   s186_privatised_check <kernel> <order> <capacity> <preset> <blocks> <threads>
//   -> <kernel> <order> <numCurrFrontier> <currFrontier[0]> ... <currFrontier[numVertices-1]>
//
// <order> is forward, reverse or shuffle on the course's emulator (the labs' seed), and
// anything on the race-detection flavour, which runs once. <capacity> is 1 to 8.
// The level is built from a previous frontier of blocks x threads vertices, ids 0 to F - 1,
// all at level 1, one per thread; the new level is 2. The presets (app/src/sim/bfspriv.ts
// builds the same graphs):
//   fanout  vertex i has two new neighbours, F + 2i and F + 2i + 1
//   shared  vertex i has the neighbours F + i and F + (i + 1) mod F, each shared by two
//   hub     vertex 0 has ten new neighbours, F to F + 9; vertex i > 0 has one, F + 9 + i
// The public frontier has exactly numVertices slots and starts as 0xFF bytes, so a slot
// nobody wrote prints as 4294967295 and a write past the end stops the emulator.
// app/src/sim/bfspriv.test.ts appends this file to s186_privatised_variants.cu, translates
// the result with labs/tools/cu2emu.py, builds it as the lab runner builds the emu and race
// backends, and compares every run with the model.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// The tests append this file to ../s186_privatised_variants.cu and define PMPP_SNIPPET_APPENDED first. The
// course's CUDA front-end check (labs/pmpp.py) parses this file on its own, so it includes
// the snippet itself.
#ifndef PMPP_SNIPPET_APPENDED
#include "../s186_privatised_variants.cu"
#endif

typedef void (*Kernel)(CSRGraph, unsigned int*, unsigned int*, unsigned int*, unsigned int,
                       unsigned int*, unsigned int);

template <unsigned int C>
static Kernel pick(const char* name) {
    if (std::strcmp(name, "bfsPrivatised") == 0) return bfsPrivatised<C>;
    if (std::strcmp(name, "bfsPrivatisedClamp") == 0) return bfsPrivatisedClamp<C>;
    if (std::strcmp(name, "bfsPlainStore") == 0) return bfsPlainStore<C>;
    if (std::strcmp(name, "bfsNoBarrier1") == 0) return bfsNoBarrier1<C>;
    if (std::strcmp(name, "bfsNoBarrier2") == 0) return bfsNoBarrier2<C>;
    if (std::strcmp(name, "bfsNoBarrier3") == 0) return bfsNoBarrier3<C>;
    if (std::strcmp(name, "bfsIfCommit") == 0) return bfsIfCommit<C>;
    if (std::strcmp(name, "bfsNoOverflowCheck") == 0) return bfsNoOverflowCheck<C>;
    return nullptr;
}

static Kernel pickCapacity(const char* name, unsigned int capacity) {
    switch (capacity) {
        case 1: return pick<1>(name);
        case 2: return pick<2>(name);
        case 3: return pick<3>(name);
        case 4: return pick<4>(name);
        case 5: return pick<5>(name);
        case 6: return pick<6>(name);
        case 7: return pick<7>(name);
        case 8: return pick<8>(name);
        default: return nullptr;
    }
}

int main(int argc, char** argv) {
    if (argc != 7) {
        std::fprintf(stderr, "usage: %s <kernel> <forward|reverse|shuffle> <capacity> <fanout|shared|hub> <blocks> <threads>\n", argv[0]);
        return 2;
    }
    const char* order = argv[2];
    const unsigned int capacity = (unsigned int)std::atoi(argv[3]);
    const char* preset = argv[4];
    const unsigned int blocks = (unsigned int)std::atoi(argv[5]);
    const unsigned int threads = (unsigned int)std::atoi(argv[6]);
    Kernel kernel = pickCapacity(argv[1], capacity);
    if (!kernel || blocks == 0 || threads == 0) {
        std::fprintf(stderr, "no kernel %s with capacity %u, or no threads\n", argv[1], capacity);
        return 2;
    }
#if defined(PMPP_EMU) && !defined(PMPP_EMU_RACE)
    // The labs' seed (labs/common/src/backend_cudalike.cpp).
    using pmpp_emu::Schedule;
    const Schedule s = std::strcmp(order, "reverse") == 0 ? Schedule::Reverse
                       : std::strcmp(order, "shuffle") == 0 ? Schedule::Shuffle
                                                            : Schedule::Forward;
    pmpp_emu::set_schedule(s, 7);
#endif
    // The graph, as CSR.
    const unsigned int F = blocks * threads;
    std::vector<std::vector<unsigned int>> out(F);
    unsigned int numVertices = 0;
    if (std::strcmp(preset, "fanout") == 0) {
        for (unsigned int i = 0; i < F; ++i) out[i] = {F + 2 * i, F + 2 * i + 1};
        numVertices = 3 * F;
    } else if (std::strcmp(preset, "shared") == 0) {
        for (unsigned int i = 0; i < F; ++i) out[i] = {F + i, F + (i + 1) % F};
        numVertices = 2 * F;
    } else if (std::strcmp(preset, "hub") == 0) {
        for (unsigned int k = 0; k < 10; ++k) out[0].push_back(F + k);
        for (unsigned int i = 1; i < F; ++i) out[i] = {F + 9 + i};
        numVertices = 2 * F + 9;
    } else {
        std::fprintf(stderr, "no preset %s\n", preset);
        return 2;
    }
    std::vector<unsigned int> srcPtrs(numVertices + 1, 0), dst;
    for (unsigned int v = 0; v < numVertices; ++v) {
        if (v < F) dst.insert(dst.end(), out[v].begin(), out[v].end());
        srcPtrs[v + 1] = (unsigned int)dst.size();
    }
    std::vector<unsigned int> level(numVertices, UNVISITED), prev(F);
    for (unsigned int i = 0; i < F; ++i) {
        level[i] = 1;
        prev[i] = i;
    }

    CSRGraph g{numVertices, nullptr, nullptr};
    unsigned int *level_d = nullptr, *prev_d = nullptr, *curr_d = nullptr, *count_d = nullptr;
    cudaMalloc((void**)&g.srcPtrs, srcPtrs.size() * sizeof(unsigned int));
    cudaMalloc((void**)&g.dst, dst.size() * sizeof(unsigned int));
    cudaMalloc((void**)&level_d, numVertices * sizeof(unsigned int));
    cudaMalloc((void**)&prev_d, F * sizeof(unsigned int));
    cudaMalloc((void**)&curr_d, numVertices * sizeof(unsigned int));
    cudaMalloc((void**)&count_d, sizeof(unsigned int));
    cudaMemcpy(g.srcPtrs, srcPtrs.data(), srcPtrs.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(g.dst, dst.data(), dst.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(level_d, level.data(), numVertices * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(prev_d, prev.data(), F * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(curr_d, 0xFF, numVertices * sizeof(unsigned int));
    cudaMemset(count_d, 0, sizeof(unsigned int));

    kernel<<<blocks, threads>>>(g, level_d, prev_d, curr_d, F, count_d, 2u);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::printf("%s %s launch-failed\n", argv[1], order);
        return 3;
    }
    std::vector<unsigned int> curr(numVertices);
    unsigned int count = 0;
    cudaMemcpy(curr.data(), curr_d, numVertices * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&count, count_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    std::printf("%s %s %u", argv[1], order, count);
    for (unsigned int k = 0; k < numVertices; ++k) std::printf(" %u", curr[k]);
    std::printf("\n");
    cudaFree(g.srcPtrs);
    cudaFree(g.dst);
    cudaFree(level_d);
    cudaFree(prev_d);
    cudaFree(curr_d);
    cudaFree(count_d);
    return 0;
}
