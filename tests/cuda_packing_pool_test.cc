#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#include "src/cuda_packing_pool.h"

namespace {

void Expect(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}

// Bound a deadlock regression without relying on the Bazel test timeout.
class Watchdog {
public:
  Watchdog() : thread_([this] {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!done_.wait_for(lock, std::chrono::seconds(30),
                        [this] { return finished_; })) {
      std::fputs("FAIL: packing pool stalled (a worker missed its task)\n",
                 stderr);
      std::_Exit(1);
    }
  }) {}
  ~Watchdog() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      finished_ = true;
    }
    done_.notify_one();
    thread_.join();
  }

private:
  std::mutex mutex_;
  std::condition_variable done_;
  bool finished_ = false;
  std::thread thread_;
};

class Gate {
public:
  void Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    ready_.wait(lock, [this] { return open_; });
  }
  void Open() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      open_ = true;
    }
    ready_.notify_one();
  }

private:
  std::mutex mutex_;
  std::condition_variable ready_;
  bool open_ = false;
};

struct Counts {
  std::size_t participants;
  std::array<std::size_t, 8> calls{};
  Gate *gate = nullptr;
};

void Count(void *context, std::size_t worker, std::size_t workers) {
  auto &counts = *static_cast<Counts *>(context);
  Expect(workers == counts.participants, "participant count");
  Expect(worker < workers, "worker index");
  ++counts.calls[worker];
  // Run has already published its first task before invoking worker zero.
  if (worker == 0 && counts.gate != nullptr)
    counts.gate->Open();
}

void CheckCounts(const Counts &counts, std::size_t expected) {
  for (std::size_t worker = 0; worker < counts.calls.size(); ++worker)
    Expect(counts.calls[worker] ==
               (worker < counts.participants ? expected : 0),
           "each participant must run each task exactly once");
}

} // namespace

namespace clqr::cuda::detail {

// Delay the real worker entry point until Run has published its task. No
// scheduling assumptions, sleeps, or test hooks in the production hot path.
struct PackingPoolTestAccess {
  static void AddDelayedWorker(PackingPool &pool, Gate &gate) {
    Expect(pool.participants() == 1, "delayed test starts without helpers");
    pool.threads_.emplace_back([&pool, &gate] {
      gate.Wait();
      pool.Worker(1);
    });
  }
};

} // namespace clqr::cuda::detail

namespace {

using clqr::cuda::detail::PackingPool;
using clqr::cuda::detail::PackingPoolTestAccess;

void DelayedStartupAndRestart() {
  PackingPool pool;
  for (int restart = 0; restart < 2; ++restart) {
    Gate gate;
    PackingPoolTestAccess::AddDelayedWorker(pool, gate);
    Counts counts{2, {}, &gate};
    pool.Run(Count, &counts);
    CheckCounts(counts, 1);
    counts.gate = nullptr;
    pool.Run(Count, &counts);
    CheckCounts(counts, 2);
    // Stop resets the generation; the replacement worker must neither replay
    // this context nor miss the first task of the next generation.
    pool.Resize(0);
  }
}

void RepeatedDispatchAndResize() {
  PackingPool pool;
  for (std::size_t helpers : {0, 1, 7, 2, 0, 3, 7, 0}) {
    pool.Resize(helpers);
    pool.Resize(helpers); // Reserving an unchanged size is a no-op.
    Counts counts{helpers + 1};
    for (std::size_t repetition = 1; repetition <= 100; ++repetition) {
      pool.Run(Count, &counts);
      CheckCounts(counts, repetition);
    }
  }
}

void HelperExceptionAndReuse() {
  PackingPool pool;
  pool.Resize(2);
  bool caught = false;
  try {
    pool.Run([](void *, std::size_t worker, std::size_t) {
      if (worker == 1)
        throw std::runtime_error("helper failure");
    }, nullptr);
  } catch (const std::runtime_error &error) {
    caught = std::string(error.what()) == "helper failure";
  }
  Expect(caught, "helper exception reaches caller after workers finish");
  Counts counts{3};
  pool.Run(Count, &counts);
  CheckCounts(counts, 1);
}

} // namespace

int main() {
  Watchdog watchdog;
  DelayedStartupAndRestart();
  RepeatedDispatchAndResize();
  HelperExceptionAndReuse();
  std::puts("CUDA host packing pool tests passed");
}
