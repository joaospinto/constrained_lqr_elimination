#ifndef CLQR_CUDA_INTERNAL_H_
#define CLQR_CUDA_INTERNAL_H_

#include "clqr/cuda.h"

namespace clqr {
namespace cuda {
namespace detail {

constexpr int kMaxRelationRows = 2 * kMaxStateDimension;
constexpr int kMaxRelationColumns =
    3 * kMaxStateDimension +
    1;  // eliminated boundary, two outer boundaries, rhs
constexpr int kMaxDualColumns =
    kMaxMixedConstraints + kMaxStateConstraints + 2 * kMaxStateDimension + 1;
constexpr int kMaxStageConstraintRows =
    kMaxMixedConstraints + kMaxStateDimension;
constexpr int kMaxStageReductionColumns =
    kMaxControlDimension + kMaxStateDimension + 1;

struct PackedStage {
  int n = 0;
  int next_n = 0;
  int m = 0;
  int mixed = 0;
  int state = 0;
  Scalar A[kMaxStateDimension * kMaxStateDimension]{};
  Scalar B[kMaxStateDimension * kMaxControlDimension]{};
  Scalar c[kMaxStateDimension]{};
  Scalar Q[kMaxStateDimension * kMaxStateDimension]{};
  Scalar R[kMaxControlDimension * kMaxControlDimension]{};
  Scalar M[kMaxStateDimension * kMaxControlDimension]{};
  Scalar q[kMaxStateDimension]{};
  Scalar r[kMaxControlDimension]{};
  Scalar C[kMaxMixedConstraints * kMaxStateDimension]{};
  Scalar D[kMaxMixedConstraints * kMaxControlDimension]{};
  Scalar d[kMaxMixedConstraints]{};
  Scalar E[kMaxStateConstraints * kMaxStateDimension]{};
  Scalar e[kMaxStateConstraints]{};
};

struct PackedTerminal {
  int n = 0;
  int state = 0;
  Scalar Q[kMaxStateDimension * kMaxStateDimension]{};
  Scalar q[kMaxStateDimension]{};
  Scalar E[kMaxStateConstraints * kMaxStateDimension]{};
  Scalar e[kMaxStateConstraints]{};
};

// A relation L*x + R*y = h. Only rows, left_dim, and right_dim are active.
struct Relation {
  int left_dim = 0;
  int right_dim = 0;
  int rows = 0;
  Scalar left[kMaxRelationRows * kMaxStateDimension]{};
  Scalar right[kMaxRelationRows * kMaxStateDimension]{};
  Scalar rhs[kMaxRelationRows]{};
};

// x = T*z + t. The physical and reduced dimensions are carried separately.
struct StateParam {
  int physical_dim = 0;
  int reduced_dim = 0;
  int free_columns[kMaxStateDimension]{};
  Scalar T[kMaxStateDimension * kMaxStateDimension]{};
  Scalar t[kMaxStateDimension]{};
};

// u = Y*z + Z*v + y.
struct ControlParam {
  int physical_dim = 0;
  int state_dim = 0;
  int reduced_dim = 0;
  int free_columns[kMaxControlDimension]{};
  Scalar Y[kMaxControlDimension * kMaxStateDimension]{};
  Scalar Z[kMaxControlDimension * kMaxControlDimension]{};
  Scalar y[kMaxControlDimension]{};
};

struct ReducedStage {
  int n = 0;
  int next_n = 0;
  int m = 0;
  Scalar A[kMaxStateDimension * kMaxStateDimension]{};
  Scalar B[kMaxStateDimension * kMaxControlDimension]{};
  Scalar c[kMaxStateDimension]{};
  Scalar Q[kMaxStateDimension * kMaxStateDimension]{};
  Scalar R[kMaxControlDimension * kMaxControlDimension]{};
  Scalar M[kMaxStateDimension * kMaxControlDimension]{};
  Scalar q[kMaxStateDimension]{};
  Scalar r[kMaxControlDimension]{};
};

struct ReducedTerminal {
  int n = 0;
  Scalar Q[kMaxStateDimension * kMaxStateDimension]{};
  Scalar q[kMaxStateDimension]{};
};

// Associative conditional-value element for an interval. J and eta live at
// the left endpoint; C and b live at the right endpoint; A maps left to right.
struct ValueElement {
  int left_dim = 0;
  int right_dim = 0;
  Scalar A[kMaxStateDimension * kMaxStateDimension]{};
  Scalar b[kMaxStateDimension]{};
  Scalar C[kMaxStateDimension * kMaxStateDimension]{};
  Scalar eta[kMaxStateDimension]{};
  Scalar J[kMaxStateDimension * kMaxStateDimension]{};
};

struct Feedback {
  int state_dim = 0;
  int next_state_dim = 0;
  int control_dim = 0;
  Scalar K[kMaxControlDimension * kMaxStateDimension]{};
  Scalar k[kMaxControlDimension]{};
  Scalar transition[kMaxStateDimension * kMaxStateDimension]{};
  Scalar offset[kMaxStateDimension]{};
};

struct AffineMap {
  int left_dim = 0;
  int right_dim = 0;
  Scalar linear[kMaxStateDimension * kMaxStateDimension]{};
  Scalar offset[kMaxStateDimension]{};
};

struct NodeValue {
  int left_dim = 0;
  int right_dim = 0;
  Scalar left[kMaxStateDimension]{};
  Scalar right[kMaxStateDimension]{};
};

enum DeviceCode : int {
  kDeviceOk = 0,
  kDeviceInfeasible = 1,
  kDeviceNumericalFailure = 2,
};

struct DeviceStatus {
  int code = kDeviceOk;
  int stage = -1;
  int detail = 0;
};

}  // namespace detail
}  // namespace cuda
}  // namespace clqr

#endif  // CLQR_CUDA_INTERNAL_H_
