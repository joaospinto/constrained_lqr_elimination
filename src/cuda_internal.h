#ifndef CLQR_CUDA_INTERNAL_H_
#define CLQR_CUDA_INTERNAL_H_

#include "clqr/cuda.h"

namespace clqr {
namespace cuda {
namespace detail {

constexpr int ConstexprMax(int first, int second) {
  return first > second ? first : second;
}

constexpr int kMaxRelationRows = 2 * kMaxStateDimension;
constexpr int kMaxRelationColumns =
    3 * kMaxStateDimension +
    1; // eliminated boundary, two outer boundaries, rhs
constexpr int kMaxDualParameterDimension =
    kMaxStateDimension + kMaxMixedConstraints;
constexpr int kMaxDualColumns = ConstexprMax(
    kMaxRelationColumns,
    ConstexprMax(
        kMaxControlDimension + 2 * kMaxStateDimension + 1,
        ConstexprMax(kMaxMixedConstraints + kMaxStateConstraints +
                         2 * kMaxStateDimension + 1,
                     ConstexprMax(kMaxStateConstraints +
                                      2 * kMaxDualParameterDimension + 1,
                                  3 * kMaxDualParameterDimension + 1))));
constexpr int kMaxStageConstraintRows =
    kMaxMixedConstraints + kMaxStateDimension;
constexpr int kMaxStageReductionColumns =
    kMaxControlDimension + kMaxStateDimension + 1;

struct PackedStage {
  int n;
  int next_n;
  int m;
  int mixed;
  int state;
  const Scalar *A;
  const Scalar *B;
  const Scalar *c;
  const Scalar *Q;
  const Scalar *R;
  const Scalar *M;
  const Scalar *q;
  const Scalar *r;
  const Scalar *C;
  const Scalar *D;
  const Scalar *d;
  const Scalar *E;
  const Scalar *e;
};

struct PackedTerminal {
  int n;
  int state;
  const Scalar *Q;
  const Scalar *q;
  const Scalar *E;
  const Scalar *e;
};

// A relation L*x + R*y = h. Only rows, left_dim, and right_dim are active.
struct Relation {
  static constexpr int kMaxDimension = kMaxStateDimension;
  int left_dim;
  int right_dim;
  int rows;
  Scalar left[kMaxRelationRows * kMaxStateDimension];
  Scalar right[kMaxRelationRows * kMaxStateDimension];
  Scalar rhs[kMaxRelationRows];
};

// Multiplier recovery contracts only the genuinely free components left after
// enforcing reduced-costate and control-stationarity equations.  Their count
// can include mixed-multiplier directions, so it has its own endpoint capacity
// rather than padding or truncating them to the physical state dimension.
struct DualRelation {
  static constexpr int kMaxDimension = kMaxDualParameterDimension;
  int left_dim;
  int right_dim;
  int rows;
  Scalar left[2 * kMaxDualParameterDimension * kMaxDualParameterDimension];
  Scalar right[2 * kMaxDualParameterDimension * kMaxDualParameterDimension];
  Scalar rhs[2 * kMaxDualParameterDimension];
};

// x = T*z + t. The physical and reduced dimensions are carried separately.
struct StateParam {
  int physical_dim;
  int reduced_dim;
  int free_columns[kMaxStateDimension];
  Scalar T[kMaxStateDimension * kMaxStateDimension];
  Scalar t[kMaxStateDimension];
};

// u = Y*z + Z*v + y.
struct ControlParam {
  int physical_dim;
  int state_dim;
  int reduced_dim;
  int free_columns[kMaxControlDimension];
  Scalar Y[kMaxControlDimension * kMaxStateDimension];
  Scalar Z[kMaxControlDimension * kMaxControlDimension];
  Scalar y[kMaxControlDimension];
};

struct ReducedStage {
  int n;
  int next_n;
  int m;
  Scalar A[kMaxStateDimension * kMaxStateDimension];
  Scalar B[kMaxStateDimension * kMaxControlDimension];
  Scalar c[kMaxStateDimension];
  Scalar Q[kMaxStateDimension * kMaxStateDimension];
  Scalar R[kMaxControlDimension * kMaxControlDimension];
  Scalar M[kMaxStateDimension * kMaxControlDimension];
  Scalar q[kMaxStateDimension];
  Scalar r[kMaxControlDimension];
};

struct ReducedTerminal {
  int n;
  Scalar Q[kMaxStateDimension * kMaxStateDimension];
  Scalar q[kMaxStateDimension];
};

// Associative conditional-value element for an interval. J and eta live at
// the left endpoint; C and b live at the right endpoint; A maps left to right.
struct ValueElement {
  int left_dim;
  int right_dim;
  Scalar *A;
  Scalar *b;
  Scalar *C;
  Scalar *eta;
  Scalar *J;
};

struct Feedback {
  int state_dim;
  int next_state_dim;
  int control_dim;
  Scalar K[kMaxControlDimension * kMaxStateDimension];
  Scalar k[kMaxControlDimension];
  Scalar transition[kMaxStateDimension * kMaxStateDimension];
  Scalar offset[kMaxStateDimension];
};

struct AffineMap {
  int left_dim;
  int right_dim;
  Scalar *linear;
  Scalar *offset;
};

struct DualNodeValue {
  int left_dim;
  int right_dim;
  Scalar left[kMaxDualParameterDimension];
  Scalar right[kMaxDualParameterDimension];
};

// [y_i; lambda_i] = offset + basis * s_i, where y_i is the physical
// dynamics multiplier and lambda_i is the stage mixed-equality multiplier.
// Only physical_dim-by-free_dim entries of basis are active.
struct DualParam {
  int state_dim;
  int mixed_dim;
  int physical_dim;
  int free_dim;
  int free_columns[kMaxDualParameterDimension];
  Scalar basis[kMaxDualParameterDimension * kMaxDualParameterDimension];
  Scalar offset[kMaxDualParameterDimension];
};

// State-only multiplier at node i as an affine function of the free dual
// coordinates on the adjacent stages.  It is obtained from the same RREF that
// emits the residual relation, so recovery does not refactor E_i^T.
struct StateDualParam {
  int constraint_dim;
  int left_dim;
  int right_dim;
  Scalar offset[kMaxStateConstraints];
  Scalar left[kMaxStateConstraints * kMaxDualParameterDimension];
  Scalar right[kMaxStateConstraints * kMaxDualParameterDimension];
};

enum DeviceCode : int {
  kDeviceOk = 0,
  kDeviceInfeasible = 1,
  kDeviceNumericalFailure = 2,
};

struct DeviceStatus {
  int code;
  int stage;
  int detail;
};

} // namespace detail
} // namespace cuda
} // namespace clqr

#endif // CLQR_CUDA_INTERNAL_H_
