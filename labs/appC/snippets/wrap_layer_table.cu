#include <cstdio>

// Where one layer's keys and values live: two device pointers.
struct LayerKV {
    float* k;
    float* v;
};

// Block b copies the first key of layer b. The table of layers is read from
// memory, and then the pointer found in it is followed.
__global__ void firstKeys(const LayerKV* layers, float* out) {
    if (threadIdx.x == 0) out[blockIdx.x] = layers[blockIdx.x].k[0];
}

void checkLayers(int nLayers, size_t perLayer) {
    LayerKV* layers;
    float* out;
    cudaMallocManaged((void**)&layers, nLayers * sizeof(LayerKV));   // the table: managed
    cudaMallocManaged((void**)&out, nLayers * sizeof(float));
    for (int l = 0; l < nLayers; ++l) {
        cudaMalloc((void**)&layers[l].k, perLayer * sizeof(float));   // the keys: device memory
        cudaMalloc((void**)&layers[l].v, perLayer * sizeof(float));
        cudaMemset(layers[l].k, 0, perLayer * sizeof(float));
    }
    firstKeys<<<nLayers, 32>>>(layers, out);
    cudaDeviceSynchronize();
    printf("%f\n", out[0]);            // line A
    printf("%f\n", layers[0].k[0]);    // line B
}
