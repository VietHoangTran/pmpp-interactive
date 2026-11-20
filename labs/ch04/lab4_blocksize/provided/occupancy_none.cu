// Lab 4.4, provided: the stand-in used when the lab is configured with
// -DPMPP_CH04_OCCUPANCY_API=OFF (see occupancy_api.cu). The benchmark then
// prints your model and the timings without the vendor's column.
int vendorBlocksPerSm(int threadsPerBlock) {
    (void)threadsPerBlock;
    return -1;
}
