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
  double A[kMaxStateDimension * kMaxStateDimension]{};
  double B[kMaxStateDimension * kMaxControlDimension]{};
  double c[kMaxStateDimension]{};
  double Q[kMaxStateDimension * kMaxStateDimension]{};
  double R[kMaxControlDimension * kMaxControlDimension]{};
  double M[kMaxStateDimension * kMaxControlDimension]{};
  double q[kMaxStateDimension]{};
  double r[kMaxControlDimension]{};
  double C[kMaxMixedConstraints * kMaxStateDimension]{};
  double D[kMaxMixedConstraints * kMaxControlDimension]{};
  double d[kMaxMixedConstraints]{};
  double E[kMaxStateConstraints * kMaxStateDimension]{};
  double e[kMaxStateConstraints]{};
};

struct PackedTerminal {
  int n = 0;
  int state = 0;
  double Q[kMaxStateDimension * kMaxStateDimension]{};
  double q[kMaxStateDimension]{};
  double E[kMaxStateConstraints * kMaxStateDimension]{};
  double e[kMaxStateConstraints]{};
};

// A relation L*x + R*y = h. Only rows, left_dim, and right_dim are active.
struct Relation {
  int left_dim = 0;
  int right_dim = 0;
  int rows = 0;
  double left[kMaxRelationRows * kMaxStateDimension]{};
  double right[kMaxRelationRows * kMaxStateDimension]{};
  double rhs[kMaxRelationRows]{};
};

// x = T*z + t. The physical and reduced dimensions are carried separately.
struct StateParam {
  int physical_dim = 0;
  int reduced_dim = 0;
  int free_columns[kMaxStateDimension]{};
  double T[kMaxStateDimension * kMaxStateDimension]{};
  double t[kMaxStateDimension]{};
};

// u = Y*z + Z*v + y.
struct ControlParam {
  int physical_dim = 0;
  int state_dim = 0;
  int reduced_dim = 0;
  int free_columns[kMaxControlDimension]{};
  double Y[kMaxControlDimension * kMaxStateDimension]{};
  double Z[kMaxControlDimension * kMaxControlDimension]{};
  double y[kMaxControlDimension]{};
};

struct ReducedStage {
  int n = 0;
  int next_n = 0;
  int m = 0;
  double A[kMaxStateDimension * kMaxStateDimension]{};
  double B[kMaxStateDimension * kMaxControlDimension]{};
  double c[kMaxStateDimension]{};
  double Q[kMaxStateDimension * kMaxStateDimension]{};
  double R[kMaxControlDimension * kMaxControlDimension]{};
  double M[kMaxStateDimension * kMaxControlDimension]{};
  double q[kMaxStateDimension]{};
  double r[kMaxControlDimension]{};
};

struct ReducedTerminal {
  int n = 0;
  double Q[kMaxStateDimension * kMaxStateDimension]{};
  double q[kMaxStateDimension]{};
};

// Associative conditional-value element for an interval. J and eta live at
// the left endpoint; C and b live at the right endpoint; A maps left to right.
struct ValueElement {
  int left_dim = 0;
  int right_dim = 0;
  double A[kMaxStateDimension * kMaxStateDimension]{};
  double b[kMaxStateDimension]{};
  double C[kMaxStateDimension * kMaxStateDimension]{};
  double eta[kMaxStateDimension]{};
  double J[kMaxStateDimension * kMaxStateDimension]{};
};

struct Feedback {
  int state_dim = 0;
  int next_state_dim = 0;
  int control_dim = 0;
  double K[kMaxControlDimension * kMaxStateDimension]{};
  double k[kMaxControlDimension]{};
  double transition[kMaxStateDimension * kMaxStateDimension]{};
  double offset[kMaxStateDimension]{};
};

struct AffineMap {
  int left_dim = 0;
  int right_dim = 0;
  double linear[kMaxStateDimension * kMaxStateDimension]{};
  double offset[kMaxStateDimension]{};
};

struct NodeValue {
  int left_dim = 0;
  int right_dim = 0;
  double left[kMaxStateDimension]{};
  double right[kMaxStateDimension]{};
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
