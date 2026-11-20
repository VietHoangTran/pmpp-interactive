// Metal side of the lab runtime. Objective-C++ only.
#pragma once
#import <Metal/Metal.h>
#include <string>
#include <utility>
#include <vector>

#include "pmpp/backend.hpp"

namespace pmpp { namespace metal {

// What a lab's encode function receives: an open compute pass on the device.
struct Pass {
  id<MTLDevice> device;
  id<MTLCommandBuffer> commandBuffer;
  id<MTLComputeCommandEncoder> encoder;
};

id<MTLDevice> device();

// Compiles `function` from the Metal Shading Language file at `path` (at run
// time, so editing the .metal file needs no rebuild) and returns its pipeline
// state. `macros` become preprocessor definitions, e.g. {{"TILE_WIDTH","16"}}.
// Results are cached. On a compile error the message is printed, recorded for
// pmpp::take_error(), and nil is returned.
id<MTLComputePipelineState> pipeline(const std::string& path, const std::string& function,
                                     const std::vector<std::pair<std::string, std::string>>& macros = {});

// The id<MTLBuffer> behind a pmpp::Buf.
inline id<MTLBuffer> buffer(pmpp::Buf* b) { return (__bridge id<MTLBuffer>)pmpp::raw(b); }

}}  // namespace pmpp::metal
