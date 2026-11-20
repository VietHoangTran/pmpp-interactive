// Lab 14.2 starter (CUDA; also built for HIP and for both CPU emulators).
// Book: sections 14.6 to 14.8 (pp. 339-345, Figs. 14.8, 14.9, 14.12, 14.13) and exercises 1, 2
// and 3 (p. 347). The book gives no code for these; the figures give the design.
//
// One LSD radix sort of (key, value) pairs with a radix of r bits (1 to 8) and c keys per
// thread (1 to 8), both chosen at run time. A block has BLOCK_DIM = 256 threads and owns a
// tile of 256c consecutive pairs; the last tile may be partial. One pass over the digit
// (key >> shift) & (2^r - 1) is three launches:
//   radixLocalSort      sort the tile by the digit in shared memory, record bucket sizes and
//                       local bucket starts;
//   launchExclusiveScan (provided) scan the bucket-size table into global bucket starts;
//   radixScatterLocal   write each pair to its global place, bucket by bucket.
// Both kernels take keysPerThread (c) as their last parameter.
//
// Use shared memory and barriers for the block-level scan. A warp-shuffle scan would need a
// lane mask, and on HIP that mask is 64 bits (notes/today/infra-facts.md B3); the shared-memory
// form has no mask to get wrong. Never assume a warp width.
#define BLOCK_DIM 256
#define MAX_KEYS_PER_THREAD 8
#define MAX_RADIX_BITS 8
#define MAX_BUCKETS 256   // 2^MAX_RADIX_BITS. Size shared arrays with plain names like this:
                          // the emulator's translator takes no parentheses in a __shared__
                          // array bound.

// Provided in ../../shared-g11/scan.cu (do not edit it):
//   scanScratchCount(count): unsigned ints of scratch the scan needs for `count` elements.
//   launchExclusiveScan(data, count, scratch): in place, data[0] = 0 and data[i] = the sum of
//   the old data[0..i-1].
int scanScratchCount(int count);
void launchExclusiveScan(unsigned int* data, int count, unsigned int* scratch);

// Reads the block's tile of keysIn/valsIn, sorts it by the digit, and writes:
//   keysLocal/valsLocal  the sorted tile, at the same positions the tile came from;
//   table                table[d * numBlocks + block] = how many of the tile's keys have digit d
//                        (digit-major, as Figs. 14.9 and 14.12 lay it out);
//   localStarts          localStarts[block * 2^r + d] = where digit d's run starts in the
//                        sorted tile.
// radixBits is this pass's digit width (the last pass may be narrower than r).
__global__ void radixLocalSort(const unsigned int* keysIn, const unsigned int* valsIn,
                               unsigned int* keysLocal, unsigned int* valsLocal,
                               unsigned int* table, unsigned int* localStarts,
                               unsigned int n, unsigned int shift, unsigned int radixBits,
                               unsigned int keysPerThread) {
    // TODO 1: shared memory for the tile (256c keys and 256c values: dynamic, the third launch
    //         parameter), for your block-level scan, and for the 2^r + 1 bucket boundaries.
    // TODO 2: how many pairs does this block own (the last tile may be partial)? Load them,
    //         coalesced: consecutive threads read consecutive pairs.
    // TODO 3: radixBits stable one-bit splits, least significant digit bit first (p. 342).
    //         For each: every thread counts the ones among its keys, a block-level exclusive
    //         scan gives the ones before them, and Fig. 14.6's two formulas, applied inside the
    //         tile, give each pair's new place. Mind every barrier: nobody may overwrite the
    //         tile before everybody has read it.
    // TODO 4: from the sorted tile, find where each of the 2^r buckets starts (empty buckets
    //         included), then fill this block's column of the table and its localStarts.
    // TODO 5: write the sorted tile back, coalesced. Padding past n is never counted or written.
}

// Sends each locally sorted pair to tableScanned[d * numBlocks + block] + (its position in
// the tile - localStarts[block * 2^r + d]). Neighbouring threads must take neighbouring tile
// positions, so that inside a bucket they write neighbouring addresses.
__global__ void radixScatterLocal(const unsigned int* keysLocal, const unsigned int* valsLocal,
                                  unsigned int* keysOut, unsigned int* valsOut,
                                  const unsigned int* tableScanned, const unsigned int* localStarts,
                                  unsigned int n, unsigned int shift, unsigned int radixBits,
                                  unsigned int keysPerThread) {
    // TODO 6
}

// ceil(32 / r) passes; the result must end in keys/vals. table and localStarts have room for
// 2^r * numBlocks entries; scanScratch holds scanScratchCount of that. tmpKeys/tmpVals are
// yours to use.
void launchRadixSortMulti(unsigned int* keys, unsigned int* vals, unsigned int* tmpKeys, unsigned int* tmpVals,
                          unsigned int* table, unsigned int* localStarts, unsigned int* scanScratch,
                          int n, int radixBits, int keysPerThread) {
    // TODO 7: how many blocks, how much dynamic shared memory? Then, per pass: local sort,
    //         scan of the 2^bits x numBlocks table, scatter. When 32 is not a multiple of r,
    //         what is the last pass's digit width? (A shift by 32 or more is undefined in C++.)
}
