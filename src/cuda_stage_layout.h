#ifndef CLQR_CUDA_STAGE_LAYOUT_H_
#define CLQR_CUDA_STAGE_LAYOUT_H_

#include <cstddef>
#include <limits>
#include <stdexcept>

#include "cuda_internal.h"

namespace clqr::cuda::detail {

// The same traversal sizes and binds each phase. Keeping the dense layouts
// independent of allocation lets later phases borrow completed scan storage.
class DenseLayoutCursor {
public:
  explicit DenseLayoutCursor(std::byte *base = nullptr) : base_(base) {}

  static std::size_t Product(std::size_t rows, std::size_t columns) {
    if (columns && rows > std::numeric_limits<std::size_t>::max() / columns)
      throw std::invalid_argument("CUDA dense layout product overflows");
    return rows * columns;
  }

  template <typename T> T *Take(std::size_t count) {
    constexpr std::size_t maximum = std::numeric_limits<std::size_t>::max();
    if (bytes_ > maximum - (alignof(T) - 1))
      throw std::invalid_argument("CUDA dense layout alignment overflows");
    bytes_ = (bytes_ + alignof(T) - 1) / alignof(T) * alignof(T);
    const std::size_t size = Product(count, sizeof(T));
    if (size > maximum - bytes_)
      throw std::invalid_argument("CUDA dense layout size overflows");
    T *result =
        base_ == nullptr ? nullptr : reinterpret_cast<T *>(base_ + bytes_);
    bytes_ += size;
    return result;
  }

  Scalar *Matrix(std::size_t rows, std::size_t columns) {
    return Take<Scalar>(Product(rows, columns));
  }
  std::size_t bytes() const { return bytes_; }

private:
  std::byte *base_;
  std::size_t bytes_ = 0;
};

inline StateParam LayoutStateParameter(DenseLayoutCursor &cursor,
                                       std::size_t physical) {
  StateParam out{};
  out.free_columns = cursor.Take<int>(physical);
  out.T = cursor.Matrix(physical, physical);
  out.t = cursor.Take<Scalar>(physical);
  return out;
}

inline ControlParam LayoutControlParameter(DenseLayoutCursor &cursor,
                                           std::size_t physical_control,
                                           std::size_t reduced_state) {
  ControlParam out{};
  out.free_columns = cursor.Take<int>(physical_control);
  out.Y = cursor.Matrix(physical_control, reduced_state);
  // Control rank is determined in the reduction kernel. The state dimension
  // is already known, but the control nullspace still needs its worst case.
  out.Z = cursor.Matrix(physical_control, physical_control);
  out.y = cursor.Take<Scalar>(physical_control);
  return out;
}

inline ReducedStage LayoutReducedStage(DenseLayoutCursor &cursor,
                                       std::size_t reduced_state,
                                       std::size_t reduced_next,
                                       std::size_t physical_control) {
  ReducedStage out{};
  out.A = cursor.Matrix(reduced_next, reduced_state);
  out.B = cursor.Matrix(reduced_next, physical_control);
  out.c = cursor.Take<Scalar>(reduced_next);
  out.Q = cursor.Matrix(reduced_state, reduced_state);
  out.R = cursor.Matrix(physical_control, physical_control);
  out.M = cursor.Matrix(reduced_state, physical_control);
  out.q = cursor.Take<Scalar>(reduced_state);
  out.r = cursor.Take<Scalar>(physical_control);
  return out;
}

inline ReducedTerminal LayoutReducedTerminal(DenseLayoutCursor &cursor,
                                             std::size_t reduced_state) {
  ReducedTerminal out{};
  out.Q = cursor.Matrix(reduced_state, reduced_state);
  out.q = cursor.Take<Scalar>(reduced_state);
  return out;
}

inline Feedback LayoutFeedback(DenseLayoutCursor &cursor,
                               std::size_t reduced_state,
                               std::size_t reduced_next,
                               std::size_t reduced_control) {
  Feedback out{};
  out.K = cursor.Matrix(reduced_control, reduced_state);
  out.k = cursor.Take<Scalar>(reduced_control);
  out.control_factor = cursor.Matrix(reduced_control, reduced_control);
  out.transition = cursor.Matrix(reduced_next, reduced_state);
  out.offset = cursor.Take<Scalar>(reduced_next);
  return out;
}

inline DualParam LayoutDualParameter(DenseLayoutCursor &cursor,
                                     std::size_t physical_dual) {
  DualParam out{};
  out.free_columns = cursor.Take<int>(physical_dual);
  out.basis = cursor.Matrix(physical_dual, physical_dual);
  out.offset = cursor.Take<Scalar>(physical_dual);
  return out;
}

inline StateDualParam LayoutStateDualParameter(DenseLayoutCursor &cursor,
                                               std::size_t constraints,
                                               std::size_t left_free,
                                               std::size_t right_free) {
  StateDualParam out{};
  out.offset = cursor.Take<Scalar>(constraints);
  out.left = cursor.Matrix(constraints, left_free);
  out.right = cursor.Matrix(constraints, right_free);
  return out;
}

} // namespace clqr::cuda::detail

#endif // CLQR_CUDA_STAGE_LAYOUT_H_
