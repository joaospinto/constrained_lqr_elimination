#ifndef CLQR_CUDA_PACKING_POOL_H_
#define CLQR_CUDA_PACKING_POOL_H_

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace clqr::cuda::detail {

// Persistent helper threads that pack large host problems into pinned memory.
// They are created during structure preparation, only when the packed problem
// is large enough for the fork/join latency to pay off, and they sleep between
// solves; a prepared solve therefore performs no allocation. Work is a plain
// function pointer plus context, so dispatch never allocates either.
// The workspace owner serializes Resize and Run.
class PackingPool {
public:
  using Task = void (*)(void *context, std::size_t worker, std::size_t workers);

  PackingPool() = default;
  PackingPool(const PackingPool &) = delete;
  PackingPool &operator=(const PackingPool &) = delete;
  ~PackingPool() { Stop(); }

  // Participants in Run, including the calling thread.
  std::size_t participants() const { return threads_.size() + 1; }

  void Resize(std::size_t helper_threads) {
    if (helper_threads == threads_.size())
      return;
    Stop();
    threads_.reserve(helper_threads);
    for (std::size_t index = 0; index < helper_threads; ++index)
      threads_.emplace_back([this, index] { Worker(index + 1); });
  }

  // Runs task(context, worker, participants()) on every participant, the
  // caller acting as worker 0, and returns once all have finished. The first
  // exception thrown by a helper is rethrown on the calling thread.
  void Run(Task task, void *context) {
    if (threads_.empty()) {
      task(context, 0, 1);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      task_ = task;
      context_ = context;
      ++generation_;
      pending_ = threads_.size();
      error_ = nullptr;
    }
    start_.notify_all();
    task(context, 0, participants());
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [this] { return pending_ == 0; });
    if (error_) {
      std::exception_ptr error = error_;
      error_ = nullptr;
      lock.unlock();
      std::rethrow_exception(error);
    }
  }

private:
  friend struct PackingPoolTestAccess;

  void Worker(std::size_t index) {
    // Stop resets the generation to zero before Resize creates any helpers.
    // Start from that value even if Run has already published its first task:
    // sampling generation_ here would mark that unseen task as completed and
    // leave Run waiting forever. A restarted pool cannot replay an old task.
    std::uint64_t seen = 0;
    for (;;) {
      Task task = nullptr;
      void *context = nullptr;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        start_.wait(lock, [&] { return stop_ || generation_ != seen; });
        if (stop_)
          return;
        seen = generation_;
        task = task_;
        context = context_;
      }
      try {
        task(context, index, threads_.size() + 1);
      } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!error_)
          error_ = std::current_exception();
      }
      std::lock_guard<std::mutex> lock(mutex_);
      if (--pending_ == 0)
        done_.notify_one();
    }
  }

  void Stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    start_.notify_all();
    for (std::thread &thread : threads_)
      thread.join();
    threads_.clear();
    stop_ = false;
    // No helper exists now; later helpers start from this generation.
    generation_ = 0;
    pending_ = 0;
  }

  std::vector<std::thread> threads_;
  std::mutex mutex_;
  std::condition_variable start_;
  std::condition_variable done_;
  Task task_ = nullptr;
  void *context_ = nullptr;
  std::uint64_t generation_ = 0;
  std::size_t pending_ = 0;
  bool stop_ = false;
  std::exception_ptr error_;
};

} // namespace clqr::cuda::detail

#endif // CLQR_CUDA_PACKING_POOL_H_
