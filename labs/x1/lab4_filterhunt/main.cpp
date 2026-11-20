// Lab X1.4 driver (the filter hunt): lab X1.2's part B checks, run on the fused kernel in
// work/ (or solution/). Part B only, at smaller sizes, every size with rows longer than one
// chunk of 1,024 tokens included; on a GPU each batch runs ten times on fresh outputs. You do
// not need to edit this file; your work goes in work/kernel.cu and work/kernel.metal.
#define X1_FILTER_HUNT 1
#include "../lab2_filters/main.cpp"
