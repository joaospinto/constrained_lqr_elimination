#ifndef CLQR_CUDA_INTERNAL_H_
#define CLQR_CUDA_INTERNAL_H_

#include "clqr/cuda.h"

namespace clqr {
namespace cuda {
namespace detail {

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
  int left_dim;
  int right_dim;
  int rows;
  Scalar *left;
  Scalar *right;
  Scalar *rhs;
};

// Multiplier recovery contracts only the genuinely free components left after
// enforcing reduced-costate and control-stationarity equations.  Their count
// can include mixed-multiplier directions, so it has its own endpoint capacity
// rather than padding or truncating them to the physical state dimension.
struct DualRelation {
  int left_dim;
  int right_dim;
  int rows;
  Scalar *left;
  Scalar *right;
  Scalar *rhs;
};

// x = T*z + t. The physical and reduced dimensions are carried separately.
struct StateParam {
  int physical_dim;
  int reduced_dim;
  int *free_columns;
  Scalar *T;
  Scalar *t;
};

// u = Y*z + Z*v + y.
struct ControlParam {
  int physical_dim;
  int state_dim;
  int reduced_dim;
  int *free_columns;
  Scalar *Y;
  Scalar *Z;
  Scalar *y;
};

struct ReducedStage {
  int n;
  int next_n;
  int m;
  Scalar *A;
  Scalar *B;
  Scalar *c;
  Scalar *Q;
  Scalar *R;
  Scalar *M;
  Scalar *q;
  Scalar *r;
};

struct ReducedTerminal {
  int n;
  Scalar *Q;
  Scalar *q;
};

// Matrix-only associative conditional-value element for an interval. J lives
// at the left endpoint, C lives at the right endpoint, and A maps left to
// right. The affine terms are recovered after the matrix scan.
struct ValueElement {
  int left_dim;
  int right_dim;
  Scalar *A;
  Scalar *C;
  Scalar *J;
};

struct Feedback {
  int state_dim;
  int next_state_dim;
  int control_dim;
  Scalar *K;
  Scalar *k;
  // Cholesky factor of R + B^T P B. The same factorization used to compute K
  // is retained and reused to recover k after the affine costate scan.
  Scalar *control_factor;
  Scalar *transition;
  Scalar *offset;
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
  Scalar *left;
  Scalar *right;
};

// [y_i; lambda_i] = offset + basis * s_i, where y_i is the physical
// dynamics multiplier and lambda_i is the stage mixed-equality multiplier.
// Only physical_dim-by-free_dim entries of basis are active.
struct DualParam {
  int state_dim;
  int mixed_dim;
  int physical_dim;
  int free_dim;
  int *free_columns;
  Scalar *basis;
  Scalar *offset;
};

// State-only multiplier at node i as an affine function of the free dual
// coordinates on the adjacent stages.  It is obtained from the same RREF that
// emits the residual relation, so recovery does not refactor E_i^T.
struct StateDualParam {
  int constraint_dim;
  int left_dim;
  int right_dim;
  Scalar *offset;
  Scalar *left;
  Scalar *right;
};

enum DeviceCode : int {
  kDeviceOk = 0,
  kDeviceInfeasible = 1,
  kDeviceNumericalFailure = 2,
  kDeviceInvalidInput = 3,
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
