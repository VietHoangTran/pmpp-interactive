// Lab 4.4: the limits of one SM that the occupancy model works from.
// Given; do not edit. The same file sits in starter/, work/ and solution/.
// Each field has the name of the cudaDeviceProp / hipDeviceProp_t member the
// benchmark fills it from. A value of 0 means "not reported": the model skips
// the rule that needs it.
#pragma once

struct SmLimits {
    int warpSize;                      // threads in a warp
    int maxThreadsPerBlock;            // largest block the device accepts
    int maxBlocksPerMultiProcessor;    // block slots of one SM
    int maxThreadsPerMultiProcessor;   // thread slots of one SM
    int regsPerMultiprocessor;         // registers of one SM
};
