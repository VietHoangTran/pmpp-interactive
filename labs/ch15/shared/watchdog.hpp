// Labs 15.1 to 15.7, provided: a time limit for one test case. You do not need to edit
// this file.
//
// The K = 0 (in = 0) cases exist for one bug: a tile loop bounded by numTiles - 1 with
// an unsigned numTiles of 0. The bound wraps around to 4,294,967,295 and the loop runs
// for hours, so without a limit the program would sit there until the runner gives up.
// A case run under a Watchdog that has not finished in time is printed and recorded as
// failed, with the reason, and the program stops.
#pragma once
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>

#include "pmpp/harness.hpp"

namespace ch15 {

class Watchdog {
 public:
  Watchdog(const pmpp::Args& a, std::string name, int seconds, std::string why) {
    t_ = std::thread([this, a, name, seconds, why] {
      std::unique_lock<std::mutex> lock(m_);
      if (cv_.wait_for(lock, std::chrono::seconds(seconds), [this] { return done_; })) return;
      const std::string detail = "still running after " + std::to_string(seconds) + " s: " + why;
      std::printf("  FAIL  %s  -> %s\n", name.c_str(), detail.c_str());
      pmpp::Json j;
      j.s("kind", "test").s("lab", a.lab).s("variant", a.variant).s("backend", pmpp::backend())
          .s("case", name).b("pass", false).s("detail", detail);
      j.emit();
      std::printf("FAILED: stopped in the case above; the remaining cases did not run (%s, %s)\n",
                  a.variant.c_str(), pmpp::backend());
      std::fflush(stdout);
      std::_Exit(1);
    });
  }
  Watchdog(const Watchdog&) = delete;
  ~Watchdog() {
    {
      std::lock_guard<std::mutex> lock(m_);
      done_ = true;
    }
    cv_.notify_all();
    t_.join();
  }

 private:
  std::mutex m_;
  std::condition_variable cv_;
  bool done_ = false;
  std::thread t_;
};

// The limit for an empty-sum case: it launches one block that loads no tile, so even the
// emulator, one thread at a time under AddressSanitizer, needs a small fraction of this.
constexpr int kEmptySumSeconds = 30;

}  // namespace ch15
