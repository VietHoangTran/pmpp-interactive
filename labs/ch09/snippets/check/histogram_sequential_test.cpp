// Driver for ../histogram_sequential.cpp. The unit test app/src/sim/histogram.test.ts
// compiles this file, runs it on the lesson's images and compares the 256 counts with
// the model behind section 9.1's explorer (app/src/sim/histogram.ts).
//
//   histogram_sequential_test WIDTH HEIGHT v0 v1 ...   prints the 256 bins, one per line
#include "../histogram_sequential.cpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) return 2;
    unsigned int width = std::strtoul(argv[1], nullptr, 10);
    unsigned int height = std::strtoul(argv[2], nullptr, 10);
    if (argc != 3 + static_cast<int>(width * height)) return 2;
    std::vector<unsigned char> image;
    for (int k = 3; k < argc; ++k) image.push_back(static_cast<unsigned char>(std::strtoul(argv[k], nullptr, 10)));
    std::vector<unsigned int> bins(256, 0u);  // the listing assumes zeroed bins
    histogram_sequential(image.data(), bins.data(), width, height);
    for (unsigned int b : bins) std::printf("%u\n", b);
    return 0;
}
