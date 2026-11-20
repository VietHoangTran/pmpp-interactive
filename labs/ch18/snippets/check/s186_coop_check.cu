// Runs one kernel of ../s186_coop_swap.cu as a whole BFS in one cooperative launch and
// prints what is left in memory:
//
//   s186_coop_check <kernel> <order> grid <W> <blocks> <threads>
//   s186_coop_check <kernel> <order> tree <b> <depth> <blocks> <threads>
//   -> <kernel> <order> <rc> | levels <level[0..n)> | A <bufA[0..n)> | B <bufB[0..n)> | counters <c...>
//   s186_coop_check toolarge <threads>
//   -> toolarge <residentBlocks> <rc of a launch with one block more>
//
// <order> is forward, reverse or shuffle (the labs' seed). The graphs (app/src/sim/gridbfs.ts
// builds the same ones): a W x W grid, vertex y*W + x, each vertex linked to its up, left,
// right and down neighbours in that order (ascending id), searched from vertex 0; or a
// b-ary tree of the given depth, vertex v linked to b*v + 1 ... b*v + b, searched from 0.
// Both frontier buffers have exactly numVertices slots and start as 0xFF bytes. counters is
// numCurrFrontier for the three-barrier kernels and numFrontier[0..n] for Exercise 3's.
// <rc> is the launch's error code as a number (0 for success).
// app/src/sim/gridbfs.test.ts appends this file to s186_coop_swap.cu, translates the result
// with labs/tools/cu2emu.py, builds it as the lab runner builds the emu backend, and
// compares every run with the model.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

typedef void (*CoopKernel)(CSRGraph, unsigned int*, unsigned int*, unsigned int*, unsigned int,
                           unsigned int*);
struct Entry {
    const char* name;
    CoopKernel kernel;
    bool perLevel;   // Exercise 3: one counter per level
};
static const Entry kKernels[] = {
    {"bfsCoop", bfsCoop, false},
    {"bfsCoopNoSync1", bfsCoopNoSync1, false},
    {"bfsCoopNoSync2", bfsCoopNoSync2, false},
    {"bfsCoopNoSync3", bfsCoopNoSync3, false},
    {"bfsCoopEx3", bfsCoopEx3, true},
    {"bfsCoopEx3NoSync", bfsCoopEx3NoSync, true},
};

static void setOrder(const char* order) {
#if defined(PMPP_EMU) && !defined(PMPP_EMU_RACE)
    using pmpp_emu::Schedule;
    const Schedule s = std::strcmp(order, "reverse") == 0 ? Schedule::Reverse
                       : std::strcmp(order, "shuffle") == 0 ? Schedule::Shuffle
                                                            : Schedule::Forward;
    pmpp_emu::set_schedule(s, 7);   // the labs' seed (labs/common/src/backend_cudalike.cpp)
#else
    (void)order;
#endif
}

int main(int argc, char** argv) {
    if (argc == 3 && std::strcmp(argv[1], "toolarge") == 0) {
        const unsigned int threads = (unsigned int)std::atoi(argv[2]);
        const unsigned int resident = residentBlocks(threads);
        CSRGraph g{1, nullptr, nullptr};
        cudaError_t rc = launchBfsCoop(g, nullptr, nullptr, nullptr, 0, nullptr, resident + 1, threads);
        std::printf("toolarge %u %d\n", resident, (int)rc);
        return 0;
    }
    const bool tree = argc == 8 && std::strcmp(argv[3], "tree") == 0;
    const bool grid = argc == 7 && std::strcmp(argv[3], "grid") == 0;
    if (!tree && !grid) {
        std::fprintf(stderr, "usage: %s <kernel> <order> grid <W> <blocks> <threads> | ... tree <b> <depth> <blocks> <threads>\n", argv[0]);
        return 2;
    }
    const Entry* e = nullptr;
    for (const Entry& k : kKernels)
        if (std::strcmp(k.name, argv[1]) == 0) e = &k;
    if (!e) {
        std::fprintf(stderr, "no kernel named %s\n", argv[1]);
        return 2;
    }
    const char* order = argv[2];
    setOrder(order);
    const unsigned int blocks = (unsigned int)std::atoi(argv[tree ? 6 : 5]);
    const unsigned int threads = (unsigned int)std::atoi(argv[tree ? 7 : 6]);

    // The graph, as CSR.
    std::vector<std::vector<unsigned int>> out;
    if (grid) {
        const unsigned int W = (unsigned int)std::atoi(argv[4]);
        out.resize(W * W);
        for (unsigned int y = 0; y < W; ++y)
            for (unsigned int x = 0; x < W; ++x) {
                std::vector<unsigned int>& o = out[y * W + x];
                if (y > 0) o.push_back((y - 1) * W + x);
                if (x > 0) o.push_back(y * W + x - 1);
                if (x + 1 < W) o.push_back(y * W + x + 1);
                if (y + 1 < W) o.push_back((y + 1) * W + x);
            }
    } else {
        const unsigned int b = (unsigned int)std::atoi(argv[4]);
        const unsigned int depth = (unsigned int)std::atoi(argv[5]);
        unsigned int n = 0, width = 1;
        for (unsigned int d = 0; d <= depth; ++d, width *= b) n += width;
        out.resize(n);
        for (unsigned int v = 0; v < n; ++v)
            for (unsigned int c = 1; c <= b; ++c)
                if (b * v + c < n) out[v].push_back(b * v + c);
    }
    const unsigned int numVertices = (unsigned int)out.size();
    std::vector<unsigned int> srcPtrs(numVertices + 1, 0), dst;
    for (unsigned int v = 0; v < numVertices; ++v) {
        dst.insert(dst.end(), out[v].begin(), out[v].end());
        srcPtrs[v + 1] = (unsigned int)dst.size();
    }
    std::vector<unsigned int> level(numVertices, UNVISITED), bufA(numVertices, 0xFFFFFFFFu);
    level[0] = 0;
    bufA[0] = 0;
    const unsigned int nCounters = e->perLevel ? numVertices + 1 : 1;

    CSRGraph g{numVertices, nullptr, nullptr};
    unsigned int *level_d = nullptr, *a_d = nullptr, *b_d = nullptr, *count_d = nullptr;
    cudaMalloc((void**)&g.srcPtrs, srcPtrs.size() * sizeof(unsigned int));
    cudaMalloc((void**)&g.dst, (dst.size() ? dst.size() : 1) * sizeof(unsigned int));
    cudaMalloc((void**)&level_d, numVertices * sizeof(unsigned int));
    cudaMalloc((void**)&a_d, numVertices * sizeof(unsigned int));
    cudaMalloc((void**)&b_d, numVertices * sizeof(unsigned int));
    cudaMalloc((void**)&count_d, nCounters * sizeof(unsigned int));
    cudaMemcpy(g.srcPtrs, srcPtrs.data(), srcPtrs.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (dst.size()) cudaMemcpy(g.dst, dst.data(), dst.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(level_d, level.data(), numVertices * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(a_d, bufA.data(), numVertices * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(b_d, 0xFF, numVertices * sizeof(unsigned int));
    cudaMemset(count_d, 0, nCounters * sizeof(unsigned int));

    unsigned int numPrevFrontier = 1;
    cudaError_t rc;
    if (e->kernel == bfsCoop) {
        rc = launchBfsCoop(g, level_d, a_d, b_d, numPrevFrontier, count_d, blocks, threads);
    } else {
        void* kernelArgs[] = { (void*)&g, (void*)&level_d, (void*)&a_d, (void*)&b_d,
            (void*)&numPrevFrontier, (void*)&count_d };
        rc = cudaLaunchCooperativeKernel((void*)e->kernel, blocks, threads, kernelArgs, 0, 0);
    }
    if (rc == cudaSuccess) rc = cudaDeviceSynchronize();

    std::vector<unsigned int> counters(nCounters);
    cudaMemcpy(level.data(), level_d, numVertices * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(bufA.data(), a_d, numVertices * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    std::vector<unsigned int> bufB(numVertices);
    cudaMemcpy(bufB.data(), b_d, numVertices * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(counters.data(), count_d, nCounters * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    std::printf("%s %s %d | levels", argv[1], order, (int)rc);
    for (unsigned int v : level) std::printf(" %u", v);
    std::printf(" | A");
    for (unsigned int v : bufA) std::printf(" %u", v);
    std::printf(" | B");
    for (unsigned int v : bufB) std::printf(" %u", v);
    std::printf(" | counters");
    for (unsigned int v : counters) std::printf(" %u", v);
    std::printf("\n");
    cudaFree(g.srcPtrs);
    cudaFree(g.dst);
    cudaFree(level_d);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(count_d);
    return 0;
}
