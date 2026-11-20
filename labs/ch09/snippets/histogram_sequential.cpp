// Fig. 9.2 (p. 203): the sequential histogram, one bin per pixel value.
// Differs from the page only in `const` on the image. The caller zeroes bins.
void histogram_sequential(const unsigned char* image, unsigned int* bins,
    unsigned int width, unsigned int height) {
    for(unsigned int i = 0; i < width*height; ++i) {
        unsigned char b = image[i];
        ++bins[b];
    }
}
