// Runs the kernels of ../s183_variants.cu on the course's CPU emulator for section 18.3 and
// prints what app/src/chapters/ch18/S183.snippets.test.ts compares with the models behind the
// section (app/src/sim/bfs.ts and app/src/sim/bfsvisit.ts).
//
//   s183_check <forward|reverse|shuffle>
//     bfs <graph> <root> <kernel> calls <c> sizes <s1> ... levels <l0> ...
//         a whole search with each of the eight kernels (the five figures as printed and as
//         corrected), blocks of 4 threads, one launch per level as the book's host loop
//         would make them; "sizes" is the threads needed per launch (frontier kernels only)
//     visit <scenario> <form> level <l0> ... frontier <v> ... counter <c>
//         one level of the stepper's scenarios with each visit form except visitWeakReset
//   s183_check spin
//     visitWeakReset on the trap scenario: it never returns, so the test runs it with a
//     time limit and expects to have to stop it
//
// The emulator runs one thread until it returns (there are no barriers here), so "forward"
// is thread 0 to its end, then thread 1, and so on, and "reverse" the other way round: the
// two orders the model's runOrder() reproduces.
#include <cstdio>
#include <cstring>
#include <vector>
#include "../s183_variants.cu"

typedef std::vector<unsigned int> Vec;
struct Graph {
    const char* name;
    unsigned int n;
    std::vector<std::pair<unsigned int, unsigned int>> edges;  // sorted by (source, destination)
};

static const Graph kBook = {"book", 9, {{0, 1}, {0, 2}, {1, 3}, {1, 4}, {2, 5}, {2, 6}, {2, 7}, {3, 4}, {3, 8},
                                         {4, 5}, {4, 8}, {5, 6}, {6, 8}, {7, 0}, {7, 6}}};
static const Graph kEx1 = {"ex1", 8, {{0, 2}, {0, 5}, {1, 0}, {1, 4}, {1, 7}, {2, 3}, {3, 0}, {3, 6}, {4, 3},
                                       {5, 1}, {5, 7}, {6, 4}, {7, 2}, {7, 4}, {7, 6}}};
static const Graph kTrap = {"trap", 5, {{0, 1}, {0, 2}, {1, 3}, {2, 3}, {2, 4}}};
static const unsigned int kBlock = 4;

static unsigned int* toDevice(const Vec& v) {
    unsigned int* p = nullptr;
    cudaMalloc((void**)&p, (v.empty() ? 1 : v.size()) * sizeof(unsigned int));
    if (!v.empty()) cudaMemcpy(p, v.data(), v.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
    return p;
}
static Vec toHost(const unsigned int* p, size_t n) {
    Vec v(n);
    if (n) cudaMemcpy(v.data(), p, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    return v;
}
static unsigned int read1(const unsigned int* p) { return toHost(p, 1)[0]; }
static void write1(unsigned int* p, unsigned int v) { cudaMemcpy(p, &v, sizeof v, cudaMemcpyHostToDevice); }

struct Arrays {
    Vec srcPtrs, dst, dstPtrs, src, cooSrc, cooDst;
};
static Arrays formats(const Graph& g) {
    Arrays a;
    a.srcPtrs.assign(g.n + 1, 0);
    a.dstPtrs.assign(g.n + 1, 0);
    for (auto& e : g.edges) {
        a.srcPtrs[e.first + 1]++;
        a.dstPtrs[e.second + 1]++;
        a.cooSrc.push_back(e.first);
        a.cooDst.push_back(e.second);
        a.dst.push_back(e.second);
    }
    for (unsigned int i = 0; i < g.n; ++i) {
        a.srcPtrs[i + 1] += a.srcPtrs[i];
        a.dstPtrs[i + 1] += a.dstPtrs[i];
    }
    a.src.assign(g.edges.size(), 0);
    Vec fill(a.dstPtrs.begin(), a.dstPtrs.end() - 1);
    for (auto& e : g.edges) a.src[fill[e.second]++] = e.first;  // sources in increasing order
    return a;
}

static void print(const char* tag, const Vec& v) {
    std::printf(" %s", tag);
    for (unsigned int x : v) std::printf(" %u", x);
}

// One search with a flag kernel (push, pull or edge-centric), as printed or corrected.
template <typename Launch>
static void flagSearch(const Graph& g, unsigned int root, const char* name, unsigned int items, Launch launch) {
    Vec level(g.n, UNVISITED);
    level[root] = 0;
    unsigned int* level_d = toDevice(level);
    unsigned int* flag_d = toDevice(Vec(1, 0));
    unsigned int calls = 0;
    for (unsigned int currLevel = 1;; ++currLevel) {
        write1(flag_d, 0);
        launch((items + kBlock - 1) / kBlock, level_d, flag_d, currLevel);
        cudaDeviceSynchronize();
        ++calls;
        if (!read1(flag_d)) break;
    }
    std::printf("bfs %s %u %s calls %u sizes", g.name, root, name, calls);
    print("levels", toHost(level_d, g.n));
    std::printf("\n");
    cudaFree(level_d);
    cudaFree(flag_d);
}

template <typename Kernel>
static void frontierSearch(const Graph& g, unsigned int root, const char* name, Kernel kernel) {
    Arrays a = formats(g);
    CSRGraph csr = {g.n, toDevice(a.srcPtrs), toDevice(a.dst)};
    Vec level(g.n, UNVISITED);
    level[root] = 0;
    unsigned int* level_d = toDevice(level);
    unsigned int* prev_d = toDevice(Vec(g.n, 0));
    unsigned int* curr_d = toDevice(Vec(g.n, 0));
    unsigned int* count_d = toDevice(Vec(1, 0));
    write1(prev_d, root);
    unsigned int numPrev = 1, calls = 0;
    Vec sizes;
    for (unsigned int currLevel = 1; numPrev > 0; ++currLevel) {
        write1(count_d, 0);
        sizes.push_back(numPrev);
        kernel((numPrev + kBlock - 1) / kBlock, csr, level_d, prev_d, curr_d, numPrev, count_d, currLevel);
        cudaDeviceSynchronize();
        ++calls;
        numPrev = read1(count_d);
        unsigned int* t = prev_d;
        prev_d = curr_d;
        curr_d = t;
    }
    std::printf("bfs %s %u %s calls %u", g.name, root, name, calls);
    print("sizes", sizes);
    print("levels", toHost(level_d, g.n));
    std::printf("\n");
    cudaFree(csr.srcPtrs);
    cudaFree(csr.dst);
    cudaFree(level_d);
    cudaFree(prev_d);
    cudaFree(curr_d);
    cudaFree(count_d);
}

static void searches(const Graph& g, unsigned int root) {
    Arrays a = formats(g);
    CSRGraph csr = {g.n, toDevice(a.srcPtrs), toDevice(a.dst)};
    CSCGraph csc = {g.n, toDevice(a.dstPtrs), toDevice(a.src)};
    COOGraph coo = {(unsigned int)g.edges.size(), toDevice(a.cooSrc), toDevice(a.cooDst)};
    const unsigned int m = (unsigned int)g.edges.size();
    flagSearch(g, root, "push_printed", g.n, [&](unsigned int b, unsigned int* l, unsigned int* f, unsigned int c) {
        fig18_6_printed::bfs_kernel<<<b, kBlock>>>(csr, l, f, c);
    });
    flagSearch(g, root, "push_fixed", g.n, [&](unsigned int b, unsigned int* l, unsigned int* f, unsigned int c) {
        fig18_6_fixed::bfs_kernel<<<b, kBlock>>>(csr, l, f, c);
    });
    flagSearch(g, root, "pull_printed", g.n, [&](unsigned int b, unsigned int* l, unsigned int* f, unsigned int c) {
        fig18_8_printed::bfs_kernel<<<b, kBlock>>>(csc, l, f, c);
    });
    flagSearch(g, root, "pull_fixed", g.n, [&](unsigned int b, unsigned int* l, unsigned int* f, unsigned int c) {
        fig18_8_fixed::bfs_kernel<<<b, kBlock>>>(csc, l, f, c);
    });
    flagSearch(g, root, "edge_printed", m, [&](unsigned int b, unsigned int* l, unsigned int* f, unsigned int c) {
        fig18_10_printed::bfs_kernel<<<b, kBlock>>>(coo, l, f, c);
    });
    flagSearch(g, root, "edge_fixed", m, [&](unsigned int b, unsigned int* l, unsigned int* f, unsigned int c) {
        fig18_10_fixed::bfs_kernel<<<b, kBlock>>>(coo, l, f, c);
    });
    frontierSearch(g, root, "frontier_printed",
                   [](unsigned int b, CSRGraph G, unsigned int* l, unsigned int* p, unsigned int* q, unsigned int np,
                      unsigned int* nq, unsigned int c) { fig18_12_printed::bfs_kernel<<<b, kBlock>>>(G, l, p, q, np, nq, c); });
    frontierSearch(g, root, "frontier_fixed",
                   [](unsigned int b, CSRGraph G, unsigned int* l, unsigned int* p, unsigned int* q, unsigned int np,
                      unsigned int* nq, unsigned int c) { fig18_12_fixed::bfs_kernel<<<b, kBlock>>>(G, l, p, q, np, nq, c); });
    cudaFree(csr.srcPtrs);
    cudaFree(csr.dst);
    cudaFree(csc.dstPtrs);
    cudaFree(csc.src);
    cudaFree(coo.src);
    cudaFree(coo.dst);
}

// ---------------------------------------------------------------- the visit forms
enum Form { Plain, PlainAppend, CAS, Hoisted, Weak, WeakLoop, WeakReset, WeakCorrect };
static const char* kFormName[] = {"plain", "plain-append", "cas", "hoisted", "weak", "weak-loop", "weak-reset", "weak-correct"};

__global__ void visitLevel(int form, CSRGraph g, const unsigned int* prevFrontier, unsigned int numPrevFrontier,
                           unsigned int* level, unsigned int* currFrontier, unsigned int* numCurrFrontier,
                           unsigned int currLevel) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPrevFrontier) return;
    unsigned int v = prevFrontier[i];
    switch (form) {
        case Plain: visitPlain(g, v, level, currFrontier, numCurrFrontier, currLevel); break;
        case PlainAppend: visitPlainAppend(g, v, level, currFrontier, numCurrFrontier, currLevel); break;
        case CAS: visitCAS(g, v, level, currFrontier, numCurrFrontier, currLevel); break;
        case Hoisted: visitHoisted(g, v, level, currFrontier, numCurrFrontier, currLevel); break;
        case Weak: visitWeak(g, v, level, currFrontier, numCurrFrontier, currLevel); break;
        case WeakLoop: visitWeakLoop(g, v, level, currFrontier, numCurrFrontier, currLevel); break;
        case WeakReset: visitWeakReset(g, v, level, currFrontier, numCurrFrontier, currLevel); break;
        case WeakCorrect: visitWeakCorrect(g, v, level, currFrontier, numCurrFrontier, currLevel); break;
    }
}

struct Scenario {
    const char* name;
    const Graph* g;
    Vec level;
    Vec threads;
    unsigned int currLevel;
};

static void visit(const Scenario& s, int form) {
    Arrays a = formats(*s.g);
    CSRGraph csr = {s.g->n, toDevice(a.srcPtrs), toDevice(a.dst)};
    unsigned int* level_d = toDevice(s.level);
    unsigned int* prev_d = toDevice(s.threads);
    unsigned int* curr_d = toDevice(Vec(8, UNVISITED));
    unsigned int* count_d = toDevice(Vec(1, 0));
    const unsigned int n = (unsigned int)s.threads.size();
    visitLevel<<<1, n>>>(form, csr, prev_d, n, level_d, curr_d, count_d, s.currLevel);
    cudaDeviceSynchronize();
    unsigned int count = read1(count_d);
    std::printf("visit %s %s", s.name, kFormName[form]);
    print("level", toHost(level_d, s.g->n));
    print("frontier", toHost(curr_d, count < 8 ? count : 8));
    std::printf(" counter %u\n", count);
    cudaFree(csr.srcPtrs);
    cudaFree(csr.dst);
    cudaFree(level_d);
    cudaFree(prev_d);
    cudaFree(curr_d);
    cudaFree(count_d);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <forward|reverse|shuffle|spin>\n", argv[0]);
        return 2;
    }
    const char* order = argv[1];
#if defined(PMPP_EMU) && !defined(PMPP_EMU_RACE)
    using pmpp_emu::Schedule;
    pmpp_emu::set_schedule(std::strcmp(order, "reverse") == 0   ? Schedule::Reverse
                           : std::strcmp(order, "shuffle") == 0 ? Schedule::Shuffle
                                                                : Schedule::Forward,
                           7);
#endif
    const Scenario book = {"book", &kBook, {0, 1, 1, 2, 2, 2, 2, 2, UNVISITED}, {3, 4, 6}, 3};
    const Scenario trap = {"trap", &kTrap, {0, 1, 1, UNVISITED, UNVISITED}, {1, 2}, 2};
    if (std::strcmp(order, "spin") == 0) {
        visit(trap, WeakReset);  // does not return
        return 0;
    }
    searches(kBook, 0);
    searches(kBook, 2);
    searches(kEx1, 0);
    for (int f = Plain; f <= WeakCorrect; ++f) {
        if (f == WeakReset) continue;
        visit(book, f);
        visit(trap, f);
    }
    std::printf("DONE\n");
    return 0;
}
