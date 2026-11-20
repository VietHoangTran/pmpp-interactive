// Lab 18.1 driver: tests your frontier BFS (part A, every backend) and your cooperative BFS
// (part B, CUDA, HIP and the CPU emulators) against a sequential BFS; on a GPU it then times
// them, two provided kernels, and the fixed cost of a level. Your work goes in work/. You do
// not need to edit this file.
//
// Before every search the driver sets level[v] = UNVISITED for every v except level[root] = 0
// (the kernels never do), fills both public frontier buffers with 0xFFFFFFFF, which is no
// vertex, and writes kGuard guard words after each of them; each buffer has room for
// numVertices entries, which is all a correct search can ever put there. Every case checks:
//   1. the guard words after both frontier buffers are untouched;
//   2. level[] equals the sequential search's levels exactly; unreached vertices stay
//      UNVISITED;
//   3. for the one-launch-per-level searches: the number of launches is the number of levels
//      after the root plus one (the last launch finds nothing), and each level's frontier
//      holds exactly as many vertices as the sequential search found at that level. More
//      means a vertex was added twice; fewer, that one was lost.
// A frontier slot that is counted but never written still holds 0xFFFFFFFF, so the next level
// reads srcPtrs far out of bounds: the emulator stops there with an AddressSanitizer report.
// Levels are whole numbers, so the comparison is exact: the tolerance is zero.
#include <chrono>
#include <climits>
#include <functional>
#include <memory>

#include "pmpp/harness.hpp"
using pmpp::Buf;

#define UNVISITED UINT_MAX

#ifndef BLOCK_DIM
#define BLOCK_DIM 128
#endif
#ifndef PRIVATE_FRONTIER_CAPACITY
#define PRIVATE_FRONTIER_CAPACITY 256
#endif

// What launchBfsCooperative returns: the numbers of BFS_COOP_* in kernel.cu.
enum CoopStatus { kCoopDone = 0, kCoopNotWritten = 1, kCoopUnsupported = 2, kCoopLaunchError = 3 };

static const unsigned kGuard = 32;                 // guard words after each public frontier
static const unsigned kEmptySlot = 0xFFFFFFFFu;    // a frontier slot before the search: no vertex
static unsigned guardWord(unsigned i) { return 0x80000000u | ((0x6A09E667u ^ (i * 0x9E3779B9u)) >> 1); }

// ---------------------------------------------------------------- graphs, in CSR form
// The out-neighbours of v are dst[srcPtrs[v]] .. dst[srcPtrs[v + 1] - 1] (section 18.1, p. 427).
struct Graph {
  std::string name;
  unsigned n = 0;
  std::vector<unsigned> srcPtrs, dst;
  unsigned m() const { return (unsigned)dst.size(); }
};
using Edges = std::vector<std::pair<unsigned, unsigned>>;

// Each vertex's neighbours sorted by id, as in Fig. 18.3; repeated edges are kept.
static Graph fromEdges(const std::string& name, unsigned n, Edges e) {
  std::sort(e.begin(), e.end());
  Graph g;
  g.name = name;
  g.n = n;
  g.srcPtrs.assign((size_t)n + 1, 0);
  g.dst.reserve(e.size());
  for (const auto& x : e) g.srcPtrs[x.first + 1]++;
  for (unsigned v = 0; v < n; ++v) g.srcPtrs[v + 1] += g.srcPtrs[v];
  for (const auto& x : e) g.dst.push_back(x.second);
  return g;
}

// Fig. 18.1 (p. 426): 9 vertices, 15 edges (notes/ch18.md, confirmed by Figs. 18.2 and 18.3).
static Graph fig18_1() {
  return fromEdges("Fig. 18.1 graph", 9, {{0, 1}, {0, 2}, {1, 3}, {1, 4}, {2, 5}, {2, 6}, {2, 7}, {3, 4},
                                          {3, 8}, {4, 5}, {4, 8}, {5, 6}, {6, 8}, {7, 0}, {7, 6}});
}
// Exercise 1 (p. 450): 8 vertices, 15 edges, our reading of the arrowheads (notes/ch18.md).
static Graph exercise1() {
  return fromEdges("Exercise 1 graph", 8, {{0, 5}, {0, 2}, {1, 0}, {1, 4}, {1, 7}, {2, 3}, {3, 0}, {3, 6},
                                           {4, 3}, {5, 1}, {5, 7}, {6, 4}, {7, 2}, {7, 4}, {7, 6}});
}
static Graph singleVertex() { return fromEdges("a single vertex", 1, {}); }
// Vertex 0 has in-edges but no out-edges: the search ends at the root.
static Graph rootWithoutOutEdges() { return fromEdges("root with no out-edges", 4, {{1, 0}, {2, 1}, {3, 2}, {2, 3}}); }
// Vertices 4 to 7 cannot be reached from 0 and must stay UNVISITED.
static Graph twoComponents() {
  return fromEdges("two components", 8, {{0, 1}, {1, 2}, {2, 0}, {1, 3}, {4, 5}, {5, 6}, {6, 4}, {6, 7}});
}
static Graph selfLoopsAndRepeats() {
  return fromEdges("self-loops and repeated edges", 5,
                   {{0, 0}, {0, 1}, {0, 1}, {0, 2}, {1, 1}, {1, 3}, {1, 3}, {1, 3}, {2, 3}, {2, 2}, {3, 4},
                    {3, 4}, {4, 0}, {4, 4}});
}
static Graph chain(unsigned n) {
  Edges e;
  for (unsigned v = 0; v + 1 < n; ++v) e.push_back({v, v + 1});
  return fromEdges("chain of " + std::to_string(n), n, e);
}
// A side x side grid, each vertex joined both ways to its 4 neighbours: a road-network-like
// graph with many levels and small frontiers (p. 445).
static Graph grid2d(unsigned side) {
  Edges e;
  e.reserve((size_t)4 * side * side);
  for (unsigned r = 0; r < side; ++r)
    for (unsigned c = 0; c < side; ++c) {
      const unsigned v = r * side + c;
      if (r > 0) e.push_back({v, v - side});
      if (c > 0) e.push_back({v, v - 1});
      if (c + 1 < side) e.push_back({v, v + 1});
      if (r + 1 < side) e.push_back({v, v + side});
    }
  return fromEdges("grid " + std::to_string(side) + " x " + std::to_string(side), side * side, e);
}
// root -> C children -> 4 new grandchildren each -> 1 great-grandchild each. At level 2 every
// block of BLOCK_DIM children finds 4 x BLOCK_DIM vertices, more than the private capacity, so
// many threads overflow at once and the commit has more entries than the block has threads.
// The great-grandchildren make a grandchild lost from the frontier show as an unreached vertex.
static Graph fanOut(unsigned children) {
  const unsigned C = children;
  Edges e;
  for (unsigned k = 0; k < C; ++k) {
    e.push_back({0, 1 + k});
    for (unsigned j = 0; j < 4; ++j) e.push_back({1 + k, 1 + C + 4 * k + j});
  }
  for (unsigned g = 0; g < 4 * C; ++g) e.push_back({1 + C + g, 1 + 5 * C + g});
  return fromEdges("fan-out", 1 + 9 * C, e);
}
// root -> 128 children; child k -> grandchildren (k + 16j) mod 64, j = 0..3, so each of the 64
// grandchildren has 8 parents competing to visit it; each grandchild -> 1 great-grandchild.
static Graph fanOutShared() {
  Edges e;
  for (unsigned k = 0; k < 128; ++k) {
    e.push_back({0, 1 + k});
    for (unsigned j = 0; j < 4; ++j) e.push_back({1 + k, 129 + (k + 16 * j) % 64});
  }
  for (unsigned g = 0; g < 64; ++g) e.push_back({129 + g, 193 + g});
  return fromEdges("fan-out shared", 257, e);
}
// 0 -> {1, 2}, 1 -> {3}, 2 -> {3, 4}. With the threads in order, vertex 1's thread visits 3
// first; vertex 2's thread loses 3 and must still win 4. A compare-and-swap whose expected
// value is left over from the lost attempt fails on 4.
static Graph staleExpectedTrap() { return fromEdges("stale-expected trap", 5, {{0, 1}, {0, 2}, {1, 3}, {2, 3}, {2, 4}}); }
// root -> `degree` children (one thread finds them all, so it alone overflows the private
// frontier); child i -> grandchild i mod (degree / 5): a large second level with contention.
static Graph hub(unsigned degree) {
  const unsigned G = degree / 5;
  Edges e;
  for (unsigned i = 0; i < degree; ++i) {
    e.push_back({0, 1 + i});
    e.push_back({1 + i, 1 + degree + i % G});
  }
  return fromEdges("hub of degree " + std::to_string(degree), 1 + degree + G, e);
}
// R-MAT (the generator is ours): each edge picks one quadrant of the adjacency matrix per bit
// of the vertex ids, with probabilities a, b, c, d = 0.57, 0.19, 0.19, 0.05 (our choice), which
// gives a skewed, social-network-like degree distribution with few levels and large frontiers.
// The ids are then scattered by multiplying by an odd constant modulo 2^scale (a bijection that
// keeps 0 at 0), and every edge is stored both ways (an undirected graph, p. 426). Vertex 0 is
// the one that picks quadrant a for every bit, the highest-degree vertex: the root.
static Graph rmat(unsigned scale, unsigned pairs, uint32_t seed) {
  const unsigned n = 1u << scale, mask = n - 1;
  uint32_t x = seed * 2654435761u + 1u;
  auto uniform = [&] { x ^= x << 13; x ^= x >> 17; x ^= x << 5; return (x >> 8) / 16777216.0; };
  Edges e;
  e.reserve((size_t)2 * pairs);
  for (unsigned k = 0; k < pairs; ++k) {
    unsigned u = 0, v = 0;
    for (unsigned bit = 0; bit < scale; ++bit) {
      const double r = uniform();
      const unsigned ub = r >= 0.76, vb = (r >= 0.57 && r < 0.76) || r >= 0.95;
      u = (u << 1) | ub;
      v = (v << 1) | vb;
    }
    u = (u * 2654435761u) & mask;
    v = (v * 2654435761u) & mask;
    e.push_back({u, v});
    e.push_back({v, u});
  }
  return fromEdges("R-MAT scale " + std::to_string(scale) + ", seed " + std::to_string(seed), n, e);
}

// ---------------------------------------------------------------- the CPU reference
struct Bfs {
  std::vector<unsigned> level;
  std::vector<unsigned> perLevel;      // perLevel[L] = vertices at level L; perLevel[0] = 1
  unsigned maxLevel = 0;               // levels after the root
  unsigned largestFrontier = 0;
  uint64_t edgesTraversed = 0;         // out-edges of every reached vertex: what a search examines
};
static Bfs cpuBfs(const Graph& g, unsigned root) {
  Bfs b;
  b.level.assign(g.n, UNVISITED);
  b.level[root] = 0;
  b.perLevel.push_back(1);
  b.largestFrontier = 1;
  std::vector<unsigned> cur{root}, next;
  for (unsigned L = 1; !cur.empty(); ++L) {
    next.clear();
    for (unsigned v : cur) {
      b.edgesTraversed += g.srcPtrs[v + 1] - g.srcPtrs[v];
      for (unsigned e = g.srcPtrs[v]; e < g.srcPtrs[v + 1]; ++e) {
        const unsigned u = g.dst[e];
        if (b.level[u] == UNVISITED) { b.level[u] = L; next.push_back(u); }
      }
    }
    if (next.empty()) break;
    b.perLevel.push_back((unsigned)next.size());
    b.maxLevel = L;
    b.largestFrontier = std::max(b.largestFrontier, (unsigned)next.size());
    cur.swap(next);
  }
  return b;
}

// The reference must reproduce the book's own arrays and levels.
static std::string selfCheck() {
  const Graph g = fig18_1();
  if (g.srcPtrs != std::vector<unsigned>{0, 2, 4, 7, 9, 11, 12, 13, 15, 15} ||
      g.dst != std::vector<unsigned>{1, 2, 3, 4, 5, 6, 7, 4, 8, 5, 8, 6, 8, 0, 6})
    return "the CSR arrays of the Fig. 18.1 graph differ from Fig. 18.3(a) (p. 428)";
  if (cpuBfs(g, 0).level != std::vector<unsigned>{0, 1, 1, 2, 2, 2, 2, 2, 3})
    return "the levels from root 0 differ from Fig. 18.4(a) (p. 429)";
  if (cpuBfs(g, 2).level != std::vector<unsigned>{2, 3, 0, 4, 4, 1, 1, 1, 2})
    return "the levels from root 2 differ from Fig. 18.4(b) (p. 429)";
  const Graph x = exercise1();
  if (x.srcPtrs != std::vector<unsigned>{0, 2, 5, 6, 8, 9, 11, 12, 15} ||
      x.dst != std::vector<unsigned>{2, 5, 0, 4, 7, 3, 0, 6, 3, 1, 7, 4, 2, 4, 6})
    return "the CSR arrays of the Exercise 1 graph differ from notes/ch18.md, answer 1b";
  if (cpuBfs(x, 0).level != std::vector<unsigned>{0, 2, 1, 2, 3, 1, 3, 2})
    return "the levels of the Exercise 1 graph differ from notes/ch18.md, answer 1c";
  return "";
}

// ---------------------------------------------------------------- the graph on the device
struct DeviceGraph {
  unsigned n, m;
  Buf *srcPtrs, *dst, *level, *frontierA, *frontierB, *counter, *flag;
  explicit DeviceGraph(const Graph& g) : n(g.n), m(g.m()) {
    srcPtrs = pmpp::alloc(4ull * (n + 1));
    dst = pmpp::alloc(4ull * std::max(m, 1u));
    level = pmpp::alloc(4ull * n);
    frontierA = pmpp::alloc(4ull * (n + kGuard));
    frontierB = pmpp::alloc(4ull * (n + kGuard));
    counter = pmpp::alloc(4);
    flag = pmpp::alloc(4);
    pmpp::upload(srcPtrs, g.srcPtrs.data(), 4ull * (n + 1));
    if (m) pmpp::upload(dst, g.dst.data(), 4ull * m);
  }
  DeviceGraph(const DeviceGraph&) = delete;
  ~DeviceGraph() {
    for (Buf* b : {srcPtrs, dst, level, frontierA, frontierB, counter, flag}) pmpp::release(b);
  }
  void resetLevels(unsigned root) {
    std::vector<unsigned> lv(n, UNVISITED);
    lv[root] = 0;
    pmpp::upload(level, lv.data(), 4ull * n);
  }
  void fillFrontiers() {
    std::vector<unsigned> f((size_t)n + kGuard, kEmptySlot);
    for (unsigned i = 0; i < kGuard; ++i) f[(size_t)n + i] = guardWord(i);
    pmpp::upload(frontierA, f.data(), f.size() * 4);
    pmpp::upload(frontierB, f.data(), f.size() * 4);
  }
  std::string checkGuards() {
    std::vector<unsigned> f((size_t)n + kGuard);
    for (int k = 0; k < 2; ++k) {
      pmpp::download(k ? frontierB : frontierA, f.data(), f.size() * 4);
      for (unsigned i = 0; i < kGuard; ++i)
        if (f[(size_t)n + i] != guardWord(i)) {
          char b[240];
          std::snprintf(b, sizeof b,
                        "guard word %u after public frontier buffer %c was overwritten (it holds %u): something "
                        "wrote past the %u entries a frontier can need",
                        i, k ? 'B' : 'A', f[(size_t)n + i], n);
          return b;
        }
    }
    return "";
  }
  std::vector<unsigned> levels() {
    std::vector<unsigned> lv(n);
    pmpp::download(level, lv.data(), 4ull * n);
    return lv;
  }
};

// What one search did.
struct Outcome {
  unsigned launches = 0;
  int status = kCoopDone;
  double seconds = 0;
  std::string error;
};

// ---------------------------------------------------------------- entry points, by backend
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
// work/launch.mm (yours, written for you) and provided/ref_launch.mm.
unsigned int launchBfsPrivatised(id<MTLCommandQueue>, id<MTLComputePipelineState>, id<MTLBuffer> srcPtrs,
                                 id<MTLBuffer> dst, uint32_t numVertices, uint32_t root, id<MTLBuffer> level,
                                 id<MTLBuffer> frontierA, id<MTLBuffer> frontierB, id<MTLBuffer> numCurrFrontier,
                                 uint32_t* frontierSizes, std::string* error);
unsigned int launchRefBfsPush(id<MTLCommandQueue>, id<MTLComputePipelineState>, id<MTLBuffer> srcPtrs,
                              id<MTLBuffer> dst, uint32_t numVertices, id<MTLBuffer> level,
                              id<MTLBuffer> newVertexVisited, std::string* error);
unsigned int launchRefBfsFrontier(id<MTLCommandQueue>, id<MTLComputePipelineState>, id<MTLBuffer> srcPtrs,
                                  id<MTLBuffer> dst, uint32_t numVertices, uint32_t root, id<MTLBuffer> level,
                                  id<MTLBuffer> frontierA, id<MTLBuffer> frontierB, id<MTLBuffer> numCurrFrontier,
                                  uint32_t* frontierSizes, std::string* error);
void launchBfsFloor(id<MTLCommandQueue>, id<MTLComputePipelineState>, uint32_t launches,
                    id<MTLBuffer> numCurrFrontier, std::string* error);
using pmpp::metal::buffer;

// One queue for the whole run; each level is a command buffer on it.
static id<MTLCommandQueue> queue() {
  static id<MTLCommandQueue> q = [pmpp::metal::device() newCommandQueue];
  return q;
}
// provided/ref_kernels.metal, found from this variant's kernel.metal: <lab>/<variant>/kernel.metal.
static std::string providedMetal() {
  std::string s = PMPP_METAL_SOURCE;
  s = s.substr(0, s.rfind('/'));
  s = s.substr(0, s.rfind('/'));
  return s + "/provided/ref_kernels.metal";
}
static id<MTLComputePipelineState> pipe(const std::string& path, const char* fn, std::string* error) {
  id<MTLComputePipelineState> p = pmpp::metal::pipeline(
      path, fn, {{"BLOCK_DIM", std::to_string(BLOCK_DIM)}, {"PRIVATE_FRONTIER_CAPACITY", std::to_string(PRIVATE_FRONTIER_CAPACITY)}});
  if (!p) {
    std::string e = pmpp::take_error();
    *error = std::string("the Metal kernel ") + fn + " did not build" + (e.empty() ? "" : ": " + e);
  }
  return p;
}
static double secondsSince(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}
static Outcome runPrivatised(DeviceGraph& d, unsigned root, unsigned* sizes) {
  Outcome o;
  id<MTLComputePipelineState> p = pipe(PMPP_METAL_SOURCE, "bfs_privatised", &o.error);
  if (!p) return o;
  const auto t0 = std::chrono::steady_clock::now();
  o.launches = launchBfsPrivatised(queue(), p, buffer(d.srcPtrs), buffer(d.dst), d.n, root, buffer(d.level),
                                   buffer(d.frontierA), buffer(d.frontierB), buffer(d.counter), sizes, &o.error);
  o.seconds = secondsSince(t0);
  return o;
}
static Outcome runFrontier(DeviceGraph& d, unsigned root, unsigned* sizes) {
  Outcome o;
  id<MTLComputePipelineState> p = pipe(providedMetal(), "ref_bfs_frontier", &o.error);
  if (!p) return o;
  const auto t0 = std::chrono::steady_clock::now();
  o.launches = launchRefBfsFrontier(queue(), p, buffer(d.srcPtrs), buffer(d.dst), d.n, root, buffer(d.level),
                                    buffer(d.frontierA), buffer(d.frontierB), buffer(d.counter), sizes, &o.error);
  o.seconds = secondsSince(t0);
  return o;
}
static Outcome runPush(DeviceGraph& d, unsigned /*root: level[] already says*/) {
  Outcome o;
  id<MTLComputePipelineState> p = pipe(providedMetal(), "ref_bfs_push", &o.error);
  if (!p) return o;
  const auto t0 = std::chrono::steady_clock::now();
  o.launches = launchRefBfsPush(queue(), p, buffer(d.srcPtrs), buffer(d.dst), d.n, buffer(d.level), buffer(d.flag),
                                &o.error);
  o.seconds = secondsSince(t0);
  return o;
}
static Outcome runFloor(DeviceGraph& d, unsigned launches) {
  Outcome o;
  id<MTLComputePipelineState> p = pipe(providedMetal(), "bfs_floor", &o.error);
  if (!p) return o;
  const auto t0 = std::chrono::steady_clock::now();
  launchBfsFloor(queue(), p, launches, buffer(d.counter), &o.error);
  o.seconds = secondsSince(t0);
  o.launches = launches;
  return o;
}
static Outcome runCooperative(DeviceGraph&, unsigned) {
  Outcome o;
  o.status = kCoopUnsupported;
  return o;
}
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
#else
// work/kernel.cu (yours) and provided/ref_kernels.cu.
unsigned int launchBfsPrivatised(const unsigned int* srcPtrs, const unsigned int* dst, unsigned int numVertices,
                                 unsigned int root, unsigned int* level, unsigned int* frontierA,
                                 unsigned int* frontierB, unsigned int* numCurrFrontier, unsigned int* frontierSizes);
int launchBfsCooperative(const unsigned int* srcPtrs, const unsigned int* dst, unsigned int numVertices,
                         unsigned int root, unsigned int* level, unsigned int* frontierA, unsigned int* frontierB,
                         unsigned int* numCurrFrontier);
unsigned int launchRefBfsPush(const unsigned int* srcPtrs, const unsigned int* dst, unsigned int numVertices,
                              unsigned int* level, unsigned int* newVertexVisited);
unsigned int launchRefBfsFrontier(const unsigned int* srcPtrs, const unsigned int* dst, unsigned int numVertices,
                                  unsigned int root, unsigned int* level, unsigned int* frontierA,
                                  unsigned int* frontierB, unsigned int* numCurrFrontier, unsigned int* frontierSizes);
void launchBfsFloor(unsigned int launches, unsigned int* numCurrFrontier);

static unsigned* P(Buf* b) { return (unsigned*)pmpp::raw(b); }
// pmpp::run synchronises the device before and after and returns the host's wall-clock time
// in between; a search's launches, copies and read-backs all fall inside it.
static Outcome finish(const pmpp::Timing& t, Outcome o) {
  o.seconds = t.wall_s;
  o.error = pmpp::take_error();
  return o;
}
static Outcome runPrivatised(DeviceGraph& d, unsigned root, unsigned* sizes) {
  Outcome o;
  pmpp::Timing t = pmpp::run([&](void*) {
    o.launches = launchBfsPrivatised(P(d.srcPtrs), P(d.dst), d.n, root, P(d.level), P(d.frontierA), P(d.frontierB),
                                     P(d.counter), sizes);
  });
  return finish(t, o);
}
static Outcome runFrontier(DeviceGraph& d, unsigned root, unsigned* sizes) {
  Outcome o;
  pmpp::Timing t = pmpp::run([&](void*) {
    o.launches = launchRefBfsFrontier(P(d.srcPtrs), P(d.dst), d.n, root, P(d.level), P(d.frontierA), P(d.frontierB),
                                      P(d.counter), sizes);
  });
  return finish(t, o);
}
static Outcome runPush(DeviceGraph& d, unsigned /*root: level[] already says*/) {
  Outcome o;
  pmpp::Timing t = pmpp::run([&](void*) { o.launches = launchRefBfsPush(P(d.srcPtrs), P(d.dst), d.n, P(d.level), P(d.flag)); });
  return finish(t, o);
}
static Outcome runFloor(DeviceGraph& d, unsigned launches) {
  Outcome o;
  o.launches = launches;
  pmpp::Timing t = pmpp::run([&](void*) { launchBfsFloor(launches, P(d.counter)); });
  return finish(t, o);
}
static Outcome runCooperative(DeviceGraph& d, unsigned root) {
  Outcome o;
  pmpp::Timing t = pmpp::run([&](void*) {
    o.status = launchBfsCooperative(P(d.srcPtrs), P(d.dst), d.n, root, P(d.level), P(d.frontierA), P(d.frontierB),
                                    P(d.counter));
  });
  return finish(t, o);
}
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
#endif

// ---------------------------------------------------------------- checks
static std::string levelText(unsigned l) { return l == UNVISITED ? "UNVISITED" : std::to_string(l); }

static std::string checkLevels(const std::vector<unsigned>& got, const Bfs& want) {
  size_t bad = 0, first = 0;
  for (size_t v = 0; v < got.size(); ++v)
    if (got[v] != want.level[v]) {
      if (!bad) first = v;
      ++bad;
    }
  if (!bad) return "";
  char b[300];
  std::snprintf(b, sizeof b, "%zu of %zu vertices have the wrong level; first: vertex %zu has %s, the sequential "
                "search says %s", bad, got.size(), first, levelText(got[first]).c_str(),
                levelText(want.level[first]).c_str());
  return b;
}

// sizes[L] for L = 1 .. launches, as the host loop recorded them.
static std::string checkSizes(const std::vector<unsigned>& sizes, unsigned launches, const Bfs& want) {
  const unsigned wantLaunches = want.maxLevel + 1;
  char b[300];
  for (unsigned L = 1; L <= std::min(launches, wantLaunches); ++L) {
    const unsigned w = L <= want.maxLevel ? want.perLevel[L] : 0;
    if (sizes[L] != w) {
      std::snprintf(b, sizeof b, "the launch for level %u counted %u vertices in its frontier; the sequential search "
                    "found %u at that level (%s)", L, sizes[L], w,
                    sizes[L] > w ? "more: a vertex was added twice" : "fewer: a vertex was lost");
      return b;
    }
  }
  if (launches != wantLaunches) {
    std::snprintf(b, sizeof b, "%u launches; expected %u (one per level from 1 to %u, then one that finds nothing)",
                  launches, wantLaunches, want.maxLevel);
    return b;
  }
  return "";
}

// ---------------------------------------------------------------- test cases
struct Case {
  std::string name;
  std::shared_ptr<const Graph> g;
  unsigned root;
  Bfs want;
};

static std::vector<Case> buildCases() {
  const bool emu = pmpp::is_emu();
  const bool race = std::string(pmpp::backend()) == "race";
  std::vector<Case> v;
  auto add = [&](std::shared_ptr<const Graph> g, unsigned root, const std::string& name) {
    v.push_back(Case{name, g, root, cpuBfs(*g, root)});
  };
  auto fig = std::make_shared<const Graph>(fig18_1());
  add(fig, 0, "Fig. 18.1 graph, root 0 (Fig. 18.4a)");
  add(fig, 2, "Fig. 18.1 graph, root 2 (Fig. 18.4b)");
  add(std::make_shared<const Graph>(exercise1()), 0, "Exercise 1 graph (p. 450), root 0");
  add(std::make_shared<const Graph>(singleVertex()), 0, "a single vertex");
  add(std::make_shared<const Graph>(rootWithoutOutEdges()), 0, "a root with no out-edges");
  add(std::make_shared<const Graph>(twoComponents()), 0, "two components, root in the first");
  add(std::make_shared<const Graph>(selfLoopsAndRepeats()), 0, "self-loops and repeated edges");
  // 300 on a device. On the emulators every level of the cooperative search is 1,024 emulated
  // threads crossing six barriers, one context switch each, so the chain is 100 there.
  const unsigned chainLength = emu ? 100 : 300;
  add(std::make_shared<const Graph>(chain(chainLength)), 0, "a " + std::to_string(chainLength) + "-vertex chain");
  const unsigned side = emu ? 24 : 1024;
  add(std::make_shared<const Graph>(grid2d(side)), 0,
      "grid " + std::to_string(side) + " x " + std::to_string(side) + ", 4 neighbours, root at a corner");
  add(std::make_shared<const Graph>(fanOut(128)), 0,
      "fan-out: 128 children find 512 grandchildren, one block against 256 private slots");
  add(std::make_shared<const Graph>(fanOut(256)), 0, "fan-out: 256 children, two blocks overflowing at once");
  add(std::make_shared<const Graph>(fanOutShared()), 0, "fan-out shared: 128 children share 64 grandchildren");
  add(std::make_shared<const Graph>(staleExpectedTrap()), 0, "stale-expected trap: 0->{1,2}, 1->{3}, 2->{3,4}");
  add(std::make_shared<const Graph>(hub(5000)), 0, "a hub of degree 5,000: one thread overflows alone");
  const unsigned scale = race ? 8 : emu ? 9 : 16;
  for (uint32_t seed : {1u, 2u}) {
    auto g = std::make_shared<const Graph>(rmat(scale, 4u << scale, seed));
    add(g, 0, "R-MAT, " + std::to_string(g->n) + " vertices, " + std::to_string(g->m()) + " edges, seed " +
                  std::to_string(seed));
  }
  return v;
}

static std::string guardsThenLevels(DeviceGraph& d, const Case& c) {
  std::string e = d.checkGuards();
  if (!e.empty()) return e;
  return checkLevels(d.levels(), c.want);
}

// One search through a one-launch-per-level host loop that records the frontier sizes.
static std::string casePerLevel(const Case& c, Outcome (*run)(DeviceGraph&, unsigned, unsigned*)) {
  DeviceGraph d(*c.g);
  d.resetLevels(c.root);
  d.fillFrontiers();
  std::vector<unsigned> sizes((size_t)c.g->n + 1, 0);
  const Outcome o = run(d, c.root, sizes.data());
  if (!o.error.empty()) return "launch error: " + o.error;
  std::string e = guardsThenLevels(d, c);
  if (!e.empty()) return e;
  return checkSizes(sizes, o.launches, c.want);
}

static std::string casePush(const Case& c) {
  DeviceGraph d(*c.g);
  d.resetLevels(c.root);
  const Outcome o = runPush(d, c.root);
  if (!o.error.empty()) return "launch error: " + o.error;
  std::string e = checkLevels(d.levels(), c.want);
  if (!e.empty()) return e;
  if (o.launches != c.want.maxLevel + 1)
    return std::to_string(o.launches) + " launches; expected " + std::to_string(c.want.maxLevel + 1);
  return "";
}

static std::string caseCooperative(const Case& c) {
  DeviceGraph d(*c.g);
  d.resetLevels(c.root);
  d.fillFrontiers();
  const Outcome o = runCooperative(d, c.root);
  if (o.status == kCoopLaunchError)
    return "cudaLaunchCooperativeKernel returned an error" + (o.error.empty() ? std::string() : ": " + o.error);
  if (o.status != kCoopDone) return "launchBfsCooperative returned " + std::to_string(o.status);
  if (!o.error.empty()) return "launch error: " + o.error;
  return guardsThenLevels(d, c);
}

// Every case through one provided kernel, as a single test. "" if all are right.
static std::string allCases(const std::vector<Case>& cases, const std::function<std::string(const Case&)>& one) {
  std::vector<std::string> failures;
  for (const Case& c : cases) {
    const std::string e = one(c);
    if (!e.empty()) failures.push_back(c.name + ": " + e);
  }
  if (failures.empty()) return "";
  std::string out = std::to_string(failures.size()) + " of " + std::to_string(cases.size()) + " graphs wrong. ";
  for (size_t i = 0; i < failures.size() && i < 2; ++i) out += (i ? " | " : "") + failures[i];
  return out;
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  t.check("driver: the CPU reference reproduces Figs. 18.3 and 18.4 and our reading of Exercise 1",
          [] { return selfCheck(); });
  const std::vector<Case> cases = buildCases();

  std::printf("provided kernels (the benchmark's baselines), on every graph below\n");
  t.check("provided push (Fig. 18.6 with relaxed atomics): every graph", [&] { return allCases(cases, casePush); });
  t.check("provided frontier (Figs. 18.12 and 18.14): every graph",
          [&] { return allCases(cases, [](const Case& c) { return casePerLevel(c, runFrontier); }); });

  std::printf("part A: your bfsPrivatisedKernel (Fig. 18.15), one launch per level\n");
  for (const Case& c : cases) t.check("privatised: " + c.name, [&] { return casePerLevel(c, runPrivatised); });

#if defined(PMPP_BACKEND_METAL)
  std::printf("part B: no Metal form (Metal documents no grid-wide barrier, no cooperative launch and no occupancy\n"
              "  query: notes/today/ch18-facts.md C1). Skipped.\n");
#else
  // Ask once, on the smallest graph, whether part B is written and the device can run it.
  const Case* tiny = nullptr;
  for (const Case& c : cases)
    if (c.g->n == 1) tiny = &c;
  int status = kCoopDone;
  {
    DeviceGraph d(*tiny->g);
    d.resetLevels(0);
    d.fillFrontiers();
    status = runCooperative(d, 0).status;
  }
  if (status == kCoopNotWritten) {
    std::printf("part B (optional): launchBfsCooperative still returns BFS_COOP_NOT_WRITTEN, so its tests are\n"
                "  skipped. Write part B to run them.\n");
  } else if (status == kCoopUnsupported) {
    std::printf("part B skipped: this device reports no cooperative launch (cudaDevAttrCooperativeLaunch is 0).\n");
  } else {
    std::printf("part B: your bfsCooperativeKernel (Fig. 18.17), every level in one cooperative launch\n");
    for (const Case& c : cases) t.check("cooperative: " + c.name, [&] { return caseCooperative(c); });
  }
#endif
  return t.finish();
}

// ---------------------------------------------------------------- benchmark
struct BenchGraph {
  std::string key;
  Graph g;
  unsigned root;
  Bfs ref;
};

static int bench(const pmpp::Args& a) {
  // Two graphs with about the same number of edges: the grid (road-network-like: many levels,
  // small frontiers) and R-MAT (social-network-like: few levels, large frontiers). Both are
  // shrunk if the arrays would take more than a quarter of the memory the device reports.
  const double mem = deviceMemory();
  auto bytes = [](double n, double m) { return 4.0 * (n + 1) + 4.0 * m + 4.0 * n + 8.0 * (n + kGuard) + 8.0; };
  unsigned side = 1024, scale = 20;
  while (mem > 0 && side > 64 && bytes((double)side * side, 4.0 * side * side) > mem / 4) side /= 2;
  while (mem > 0 && scale > 12 && bytes(double(1u << scale), 4.0 * (1u << scale)) > mem / 4) --scale;
  std::vector<BenchGraph> graphs;
  graphs.push_back(BenchGraph{"grid", grid2d(side), 0, {}});
  graphs.push_back(BenchGraph{"rmat", rmat(scale, 2u << scale, 20261120u), 0, {}});
  for (BenchGraph& bg : graphs) bg.ref = cpuBfs(bg.g, bg.root);

  std::printf("Whole searches, best of %d after one untimed run. level[] is reset before each run, untimed;\n"
              "the last run's levels are checked against the sequential search before a row is reported.\n"
              "us/level divides by the launches (levels after the root, plus the last that finds nothing);\n"
              "Medges/s counts the out-edges of every reached vertex, which every search here examines.\n",
              a.reps);

  double secs[2][5] = {{0}};   // [graph][push, frontier, privatised, cooperative, floor]
  const char* names[5] = {"push", "frontier", "privatised", "cooperative", "floor"};
  const char* labels[5] = {"push (provided)", "frontier (provided)", "privatised (yours)", "cooperative (yours)",
                           "floor (provided)"};
  bool allRight = true;
  for (size_t gi = 0; gi < graphs.size(); ++gi) {
    BenchGraph& bg = graphs[gi];
    const unsigned launches = bg.ref.maxLevel + 1;
    std::printf("\ngraph '%s': %s, %u vertices, %u edges, root %u: %u levels after the root, largest frontier %u,"
                " %llu edges traversed\n", bg.key.c_str(), bg.g.name.c_str(), bg.g.n, bg.g.m(), bg.root,
                bg.ref.maxLevel, bg.ref.largestFrontier, (unsigned long long)bg.ref.edgesTraversed);
    std::printf("  %-22s %12s %12s %12s\n", "", "ms", "us/level", "Medges/s");
    DeviceGraph d(bg.g);
    d.fillFrontiers();
    for (int k = 0; k < 5; ++k) {
      std::string why;
      double best = 0;
      for (int r = 0; r <= a.reps && why.empty(); ++r) {
        if (k != 4) d.resetLevels(bg.root);
        Outcome o;
        switch (k) {
          case 0: o = runPush(d, bg.root); break;
          case 1: o = runFrontier(d, bg.root, nullptr); break;
          case 2: o = runPrivatised(d, bg.root, nullptr); break;
          case 3: o = runCooperative(d, bg.root); break;
          default: o = runFloor(d, launches); break;
        }
        if (o.status == kCoopUnsupported || o.status == kCoopNotWritten) {
          why = o.status == kCoopNotWritten ? "part B not written" : "no cooperative launch on this backend or device";
          break;
        }
        if (o.status == kCoopLaunchError) why = "cudaLaunchCooperativeKernel returned an error";
        if (!o.error.empty()) why = "error: " + o.error;
        if (why.empty() && r > 0 && (best == 0 || o.seconds < best)) best = o.seconds;
      }
      if (why.empty() && k != 4) {
        why = d.checkGuards();
        if (why.empty()) why = checkLevels(d.levels(), bg.ref);
        if (!why.empty()) {
          why = "WRONG RESULT, not reported: " + why;
          allRight = false;
        }
      }
      if (why.empty() && best <= 0) why = "no time measured";
      if (!why.empty()) {
        std::printf("  %-22s %s\n", labels[k], why.c_str());
        continue;
      }
      secs[gi][k] = best;
      const double usPerLevel = best * 1e6 / launches, medges = bg.ref.edgesTraversed / best / 1e6;
      std::printf("  %-22s %12.3f %12.2f %12.1f\n", labels[k], best * 1e3, usPerLevel, medges);
      pmpp::bench_row(a, names[k]).s("graph", bg.key).n("vertices", bg.g.n).n("edges", bg.g.m())
          .n("levels", bg.ref.maxLevel).n("seconds", best).n("us_per_level", usPerLevel).n("medges_per_s", medges).emit();
    }
  }

  std::printf("\nThe three numbers the README asks you to predict, as measured:\n");
  if (secs[0][4] > 0 && secs[0][2] > 0)
    std::printf("  (1) grid: floor / privatised = %.2f (the share of the per-level search spent on the fixed cost of "
                "a level)\n", secs[0][4] / secs[0][2]);
  if (secs[1][1] > 0 && secs[1][2] > 0)
    std::printf("  (2) rmat: privatised is %.2fx as fast as frontier\n", secs[1][1] / secs[1][2]);
  if (secs[0][3] > 0 && secs[0][2] > 0 && secs[1][3] > 0 && secs[1][2] > 0)
    std::printf("  (3) cooperative is %.2fx as fast as privatised on grid, %.2fx on rmat\n", secs[0][2] / secs[0][3],
                secs[1][2] / secs[1][3]);
  else
    std::printf("  (3) not measured here: part B runs on CUDA and HIP devices that support cooperative launch\n");
  if (!allRight) std::printf("  WARNING: some results were wrong and are not reported; fix the tests first.\n");
  std::printf("These are measurements of this device.\n");
  return allRight ? 0 : 1;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
