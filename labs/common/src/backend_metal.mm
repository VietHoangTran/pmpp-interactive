// pmpp backend for Metal (Apple GPUs).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>

#include "pmpp/backend.hpp"
#include "pmpp/metal.h"

namespace pmpp {

struct Buf { id<MTLBuffer> buffer; size_t bytes; };

static std::string g_error;
static void note(const std::string& e) { if (g_error.empty()) g_error = e; }

namespace metal {
id<MTLDevice> device() {
  static id<MTLDevice> d = MTLCreateSystemDefaultDevice();
  return d;
}
static id<MTLCommandQueue> queue() {
  static id<MTLCommandQueue> q = [device() newCommandQueue];
  return q;
}
id<MTLComputePipelineState> pipeline(const std::string& path, const std::string& function,
                                     const std::vector<std::pair<std::string, std::string>>& macros) {
  static std::map<std::string, id<MTLComputePipelineState>> cache;
  std::string key = path + "|" + function;
  for (auto& m : macros) key += "|" + m.first + "=" + m.second;
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;

  NSError* err = nil;
  NSString* src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]
                                            encoding:NSUTF8StringEncoding error:&err];
  if (!src) { note("cannot read " + path); std::fprintf(stderr, "pmpp: cannot read %s\n", path.c_str()); return nil; }
  MTLCompileOptions* opts = [MTLCompileOptions new];
  NSMutableDictionary* defs = [NSMutableDictionary dictionary];
  for (auto& m : macros)
    defs[[NSString stringWithUTF8String:m.first.c_str()]] = [NSString stringWithUTF8String:m.second.c_str()];
  opts.preprocessorMacros = defs;
  id<MTLLibrary> lib = [device() newLibraryWithSource:src options:opts error:&err];
  if (!lib) {
    std::string msg = err ? err.localizedDescription.UTF8String : "unknown error";
    std::fprintf(stderr, "pmpp: Metal compile error in %s:\n%s\n", path.c_str(), msg.c_str());
    note("Metal compile error: " + msg.substr(0, 300));
    return nil;
  }
  id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:function.c_str()]];
  if (!fn) { note("no kernel function named '" + function + "' in " + path); return nil; }
  id<MTLComputePipelineState> pso = [device() newComputePipelineStateWithFunction:fn error:&err];
  if (!pso) { note(std::string("pipeline error: ") + (err ? err.localizedDescription.UTF8String : "unknown")); return nil; }
  cache[key] = pso;
  return pso;
}
}  // namespace metal

const char* backend() { return "metal"; }

bool device_available(std::string* why) {
  if (metal::device()) return true;
  if (why) *why = "MTLCreateSystemDefaultDevice returned nil";
  return false;
}

std::vector<std::pair<std::string, std::string>> device_props() {
  std::vector<std::pair<std::string, std::string>> out;
  id<MTLDevice> d = metal::device();
  if (!d) return out;
  auto add = [&](const char* k, double v) { char b[64]; std::snprintf(b, sizeof b, "%.0f", v); out.emplace_back(k, b); };
  out.emplace_back("name", d.name.UTF8String);
  add("hasUnifiedMemory", d.hasUnifiedMemory ? 1 : 0);
  add("maxThreadsPerThreadgroup.width", d.maxThreadsPerThreadgroup.width);
  add("maxThreadsPerThreadgroup.height", d.maxThreadsPerThreadgroup.height);
  add("maxThreadsPerThreadgroup.depth", d.maxThreadsPerThreadgroup.depth);
  add("maxThreadgroupMemoryLength", d.maxThreadgroupMemoryLength);
  add("maxBufferLength", d.maxBufferLength);
  add("recommendedMaxWorkingSetSize", d.recommendedMaxWorkingSetSize);
  add("physicalMemory", [NSProcessInfo processInfo].physicalMemory);
  return out;
}

Buf* alloc(size_t bytes) {
  Buf* b = new Buf{nil, bytes};
  // Shared storage: one allocation visible to both CPU and GPU.
  b->buffer = [metal::device() newBufferWithLength:(bytes ? bytes : 4) options:MTLResourceStorageModeShared];
  if (!b->buffer) note("newBufferWithLength failed");
  else std::memset(b->buffer.contents, 0xA5, bytes);   // do not hand out zeroed memory
  return b;
}
void upload(Buf* b, const void* src, size_t bytes) { std::memcpy(b->buffer.contents, src, bytes); }
void download(Buf* b, void* dst, size_t bytes) { std::memcpy(dst, b->buffer.contents, bytes); }
void release(Buf* b) { if (b) { b->buffer = nil; delete b; } }
void* raw(Buf* b) { return (__bridge void*)b->buffer; }

int schedules() { return 1; }
void use_schedule(int) {}
const char* schedule_name(int) { return "device"; }

Timing run(const std::function<void(void*)>& enqueue) {
  @autoreleasepool {
    metal::Pass pass;
    pass.device = metal::device();
    pass.commandBuffer = [metal::queue() commandBuffer];
    pass.encoder = [pass.commandBuffer computeCommandEncoder];
    auto t0 = std::chrono::steady_clock::now();
    enqueue(&pass);
    [pass.encoder endEncoding];
    [pass.commandBuffer commit];
    [pass.commandBuffer waitUntilCompleted];
    auto t1 = std::chrono::steady_clock::now();
    if (pass.commandBuffer.status == MTLCommandBufferStatusError)
      note(std::string("command buffer error: ") +
           (pass.commandBuffer.error ? pass.commandBuffer.error.localizedDescription.UTF8String : "unknown"));
    return Timing{std::chrono::duration<double>(t1 - t0).count(),
                  pass.commandBuffer.GPUEndTime - pass.commandBuffer.GPUStartTime};
  }
}

std::string take_error() { std::string e = g_error; g_error.clear(); return e; }

}  // namespace pmpp
