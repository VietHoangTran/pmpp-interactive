// Lab 14.2 reference solution (CUDA; also built for HIP and for both CPU emulators).
// Book: sections 14.6 to 14.8 (pp. 339-345, Figs. 14.8, 14.9, 14.12, 14.13) and exercises 1, 2
// and 3 (p. 347). The book gives no code for any of them; everything here is ours, built to
// the designs the figures draw.
//
// One LSD radix sort of (key, value) pairs with a radix of r bits (1 to 8) and c keys per
// thread (1 to 8), both chosen at run time. A block has BLOCK_DIM = 256 threads and owns a
// tile of 256c consecutive pairs. One pass over the digit (key >> shift) & (2^r - 1) is three
// launches:
//   radixLocalSort     the block sorts its tile by the digit in shared memory, with r stable
//                      one-bit splits (p. 342), each a block-level exclusive scan (section 14.6
//                      with chapter 11's scan). It writes the sorted tile back to the same
//                      place, its 2^r bucket sizes into a digit-major table, table[d * numBlocks
//                      + block] (Figs. 14.9 and 14.12), and where each bucket starts inside the
//                      tile into localStarts[block * 2^r + d].
//   launchExclusiveScan (provided) turns the table into every local bucket's global start.
//   radixScatterLocal  sends each locally sorted pair to its bucket's global start plus its
//                      offset inside the bucket. Within a bucket, neighbouring threads write
//                      neighbouring addresses: the coalescing that section 14.6 is after.
// The kernel signatures add one parameter to the ones the lab text names: keysPerThread, since
// c is chosen at run time and both kernels need the tile size.
//
// The block-level scan uses shared memory and barriers only (Kogge-Stone, section 11.3), so it
// needs no warp intrinsics and no lane masks on HIP.
#define BLOCK_DIM 256
#define MAX_KEYS_PER_THREAD 8
#define MAX_RADIX_BITS 8
#define MAX_BUCKETS 256   // 2^MAX_RADIX_BITS, spelled out: the emulator's translator takes no
                          // parentheses in a __shared__ array bound
static_assert(MAX_BUCKETS == 1 << MAX_RADIX_BITS, "MAX_BUCKETS must be 2^MAX_RADIX_BITS");

// Provided in ../../shared-g11/scan.cu.
int scanScratchCount(int count);
void launchExclusiveScan(unsigned int* data, int count, unsigned int* scratch);

// Exclusive scan of one value per thread over the block. Every thread must call it. scan_s
// holds 2 * BLOCK_DIM unsigned ints of shared memory. *total receives the block's sum.
// Its first barrier also orders every thread's earlier reads of shared memory before any
// thread's later writes.
__device__ unsigned int blockExclusiveScan(unsigned int value, unsigned int* scan_s, unsigned int* total) {
    const unsigned int t = threadIdx.x;
    unsigned int cur = 0;
    scan_s[t] = value;
    __syncthreads();
    for (unsigned int stride = 1; stride < BLOCK_DIM; stride *= 2) {
        unsigned int v = scan_s[cur * BLOCK_DIM + t];
        if (t >= stride) {
            v += scan_s[cur * BLOCK_DIM + t - stride];
        }
        scan_s[(1 - cur) * BLOCK_DIM + t] = v;
        cur = 1 - cur;
        __syncthreads();
    }
    *total = scan_s[cur * BLOCK_DIM + BLOCK_DIM - 1];
    return scan_s[cur * BLOCK_DIM + t] - value;
}

__global__ void radixLocalSort(const unsigned int* keysIn, const unsigned int* valsIn,
                               unsigned int* keysLocal, unsigned int* valsLocal,
                               unsigned int* table, unsigned int* localStarts,
                               unsigned int n, unsigned int shift, unsigned int radixBits,
                               unsigned int keysPerThread) {
    extern __shared__ unsigned int tile_s[];                 // keys, then values: 2 * 256c
    __shared__ unsigned int scan_s[2 * BLOCK_DIM];
    __shared__ unsigned int start_s[MAX_BUCKETS + 1];

    const unsigned int t = threadIdx.x;
    const unsigned int c = keysPerThread;
    const unsigned int tile = BLOCK_DIM * c;
    const unsigned int base = blockIdx.x * tile;
    const unsigned int count = n - base < tile ? n - base : tile;   // the last tile may be partial
    unsigned int* keys_s = tile_s;
    unsigned int* vals_s = tile_s + tile;

    // 1. Coalesced load: on round k, consecutive threads read consecutive pairs.
    for (unsigned int k = 0; k < c; ++k) {
        const unsigned int p = k * BLOCK_DIM + t;
        if (p < count) {
            keys_s[p] = keysIn[base + p];
            vals_s[p] = valsIn[base + p];
        }
    }
    __syncthreads();

    // 2. r stable one-bit splits, least significant digit bit first. Thread t owns the c
    //    consecutive tile positions t*c .. t*c + c - 1, so its part of the scan is sequential.
    for (unsigned int b = 0; b < radixBits; ++b) {
        const unsigned int bit = shift + b;
        unsigned int myKeys[MAX_KEYS_PER_THREAD];
        unsigned int myVals[MAX_KEYS_PER_THREAD];
        unsigned int myOnes = 0;
        for (unsigned int k = 0; k < c; ++k) {
            const unsigned int p = t * c + k;
            if (p < count) {
                myKeys[k] = keys_s[p];
                myVals[k] = vals_s[p];
                myOnes += (myKeys[k] >> bit) & 1;
            }
        }
        unsigned int onesTotal;
        unsigned int onesBefore = blockExclusiveScan(myOnes, scan_s, &onesTotal);
        const unsigned int zerosTotal = count - onesTotal;
        // Every thread has read its pairs (the scan's barriers), so the tile can be rewritten.
        for (unsigned int k = 0; k < c; ++k) {
            const unsigned int p = t * c + k;
            if (p < count) {
                const unsigned int one = (myKeys[k] >> bit) & 1;
                const unsigned int dst = one ? zerosTotal + onesBefore : p - onesBefore;
                keys_s[dst] = myKeys[k];
                vals_s[dst] = myVals[k];
                onesBefore += one;
            }
        }
        __syncthreads();
    }

    // 3. Where each local bucket starts. The tile is sorted by digit, so bucket d starts at the
    //    first position whose digit is d or more. A position p whose digit differs from the one
    //    before it is the start of every bucket from (previous digit + 1) up to its own digit;
    //    exactly one position writes each bucket that has a key after it. Buckets with no key
    //    at or after them start at count, and start_s[2^r] = count closes the last bucket.
    const unsigned int buckets = 1u << radixBits;
    const unsigned int mask = buckets - 1;
    for (unsigned int d = t; d <= buckets; d += BLOCK_DIM) {
        start_s[d] = count;
    }
    __syncthreads();
    for (unsigned int k = 0; k < c; ++k) {
        const unsigned int p = k * BLOCK_DIM + t;
        if (p < count) {
            const unsigned int digit = (keys_s[p] >> shift) & mask;
            const unsigned int first = p == 0 ? 0 : ((keys_s[p - 1] >> shift) & mask) + 1;
            for (unsigned int d = first; d <= digit; ++d) {
                start_s[d] = p;
            }
        }
    }
    __syncthreads();

    // 4. The table is digit-major: all blocks' counts for digit 0, then for digit 1, ...
    const unsigned int numBlocks = gridDim.x;
    for (unsigned int d = t; d < buckets; d += BLOCK_DIM) {
        table[d * numBlocks + blockIdx.x] = start_s[d + 1] - start_s[d];
        localStarts[blockIdx.x * buckets + d] = start_s[d];
    }

    // 5. Coalesced store of the locally sorted tile, to the same tile position.
    for (unsigned int k = 0; k < c; ++k) {
        const unsigned int p = k * BLOCK_DIM + t;
        if (p < count) {
            keysLocal[base + p] = keys_s[p];
            valsLocal[base + p] = vals_s[p];
        }
    }
}

__global__ void radixScatterLocal(const unsigned int* keysLocal, const unsigned int* valsLocal,
                                  unsigned int* keysOut, unsigned int* valsOut,
                                  const unsigned int* tableScanned, const unsigned int* localStarts,
                                  unsigned int n, unsigned int shift, unsigned int radixBits,
                                  unsigned int keysPerThread) {
    __shared__ unsigned int offset_s[MAX_BUCKETS];

    const unsigned int t = threadIdx.x;
    const unsigned int tile = BLOCK_DIM * keysPerThread;
    const unsigned int base = blockIdx.x * tile;
    const unsigned int count = n - base < tile ? n - base : tile;
    const unsigned int buckets = 1u << radixBits;
    const unsigned int mask = buckets - 1;

    // Global start of this block's bucket d, minus where it starts inside the tile. Unsigned
    // arithmetic wraps, and adding a tile position back gives the right address.
    for (unsigned int d = t; d < buckets; d += BLOCK_DIM) {
        offset_s[d] = tableScanned[d * gridDim.x + blockIdx.x] - localStarts[blockIdx.x * buckets + d];
    }
    __syncthreads();

    // Consecutive threads take consecutive tile positions; inside one bucket they write
    // consecutive global addresses.
    for (unsigned int k = 0; k < keysPerThread; ++k) {
        const unsigned int p = k * BLOCK_DIM + t;
        if (p < count) {
            const unsigned int key = keysLocal[base + p];
            const unsigned int dst = offset_s[(key >> shift) & mask] + p;
            keysOut[dst] = key;
            valsOut[dst] = valsLocal[base + p];
        }
    }
}

// ceil(32 / r) passes; the last one covers only the bits that remain (32 - shift of them), so
// no shift ever reaches 32. Each pass: local sort keys -> tmp, scan the table, scatter tmp ->
// keys. The result is in keys/vals after every pass.
void launchRadixSortMulti(unsigned int* keys, unsigned int* vals, unsigned int* tmpKeys, unsigned int* tmpVals,
                          unsigned int* table, unsigned int* localStarts, unsigned int* scanScratch,
                          int n, int radixBits, int keysPerThread) {
    if (n < 1 || radixBits < 1 || radixBits > MAX_RADIX_BITS || keysPerThread < 1 ||
        keysPerThread > MAX_KEYS_PER_THREAD) {
        return;
    }
    const unsigned int count = (unsigned int)n;
    const unsigned int r = (unsigned int)radixBits;
    const unsigned int c = (unsigned int)keysPerThread;
    const unsigned int tile = BLOCK_DIM * c;
    const unsigned int numBlocks = (count + tile - 1) / tile;
    const size_t sharedBytes = 2 * (size_t)tile * sizeof(unsigned int);
    for (unsigned int shift = 0; shift < 32; shift += r) {
        const unsigned int bits = 32 - shift < r ? 32 - shift : r;
        radixLocalSort<<<numBlocks, BLOCK_DIM, sharedBytes>>>(keys, vals, tmpKeys, tmpVals, table, localStarts,
                                                               count, shift, bits, c);
        launchExclusiveScan(table, (int)((1u << bits) * numBlocks), scanScratch);
        radixScatterLocal<<<numBlocks, BLOCK_DIM>>>(tmpKeys, tmpVals, keys, vals, table, localStarts,
                                                    count, shift, bits, c);
    }
}
