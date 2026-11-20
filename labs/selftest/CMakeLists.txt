# Self-tests of the CPU emulator (both flavours). Each is one .cu file written
# in plain CUDA plus the shared driver; see pmpp_add_selftest in ../cmake/Lab.cmake.
# Run them with `python3 labs/pmpp.py selftest`; ./verify.sh runs and reports them too.
pmpp_add_selftest(warp)
pmpp_add_selftest(atomics)
pmpp_add_selftest(constant)
pmpp_add_selftest(shared)
pmpp_add_selftest(cgroups)
pmpp_add_selftest(hostapi)
# The race detector's own controls. The plain histogram is correct on the baton
# emulator (which is the point) and wrong on a real GPU, so it is not built for cuda.
pmpp_add_selftest(race_hist_atomic)
pmpp_add_selftest(race_hist_plain EXPECT_RACE BACKENDS emu race)
pmpp_add_selftest(race_negatives EXPECT_RACE BACKENDS race)
