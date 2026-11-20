// The code lines of Fig. 21.8 (book p. 520) as printed, with their line numbers kept, for
// the erratum hunt in lesson 21.2. The one comment line is reworded. Line 05 does not
// scale threadIdx.x by COARSEN_FACTOR, so with lines 30 to 33 neighbouring threads write
// overlapping points with a plain +=, and most of each block's points are never written
// (notes/ch21.md, "Unclear in the PDF"; notes/today/ch21-facts.md E2). check.py runs it to
// show exactly that. Do not use it.
#ifndef CHUNK_SIZE
#define CHUNK_SIZE 256
#endif
// The course's CUDA front-end check declares no CUDA Math API functions, so it is given
// sqrtf's device declaration here. nvcc, hipcc and the emulator never see it.
#if defined(PMPP_CUDA_FRONTEND_CHECK)
__device__ float sqrtf(float x);
#endif
// ---- as printed ----
__constant__ float atoms[CHUNK_SIZE*4];
#define COARSEN_FACTOR 4
void __global__ cenergy(float *energygrid, dim3 grid, float gridspacing,
                        float z, int numatoms) {
    int i = blockIdx.x * blockDim.x*COARSEN_FACTOR + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int atomarrdim = numatoms * 4;
    int k = z / gridspacing;
    float y = gridspacing * (float) j;
    float x = gridspacing * (float) i;
    float energy0 = 0.0f;
    float energy1 = 0.0f;
    float energy2 = 0.0f;
    float energy3 = 0.0f;
    // add up the contributions of every atom
    for (int n=0; n<atomarrdim; n+=4) {
        float dx0 = x - atoms[n  ];
        float dx1 = dx0 +   gridspacing;
        float dx2 = dx0 + 2*gridspacing;
        float dx3 = dx0 + 3*gridspacing;
        float dy = y - atoms[n+1];
        float dz = z - atoms[n+2];
        float dysqdzsq = dy*dy + dz*dz;
        float charge = atoms[n+3];
        energy0 += charge / sqrtf(dx0*dx0 + dysqdzsq);
        energy1 += charge / sqrtf(dx1*dx1 + dysqdzsq);
        energy2 += charge / sqrtf(dx2*dx2 + dysqdzsq);
        energy3 += charge / sqrtf(dx3*dx3 + dysqdzsq);
    }
    energygrid[grid.x*grid.y*k + grid.x*j + i  ] += energy0;
    energygrid[grid.x*grid.y*k + grid.x*j + i+1] += energy1;
    energygrid[grid.x*grid.y*k + grid.x*j + i+2] += energy2;
    energygrid[grid.x*grid.y*k + grid.x*j + i+3] += energy3;
}
