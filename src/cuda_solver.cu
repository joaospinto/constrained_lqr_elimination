#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "clqr/cuda.h"
#include "cuda_internal.h"

namespace clqr {
namespace cuda {
namespace detail {
namespace {

// Each stage/node carries a small dense problem. A single warp avoids the
// four-warp synchronization and occupancy cost of the former 128-thread
// blocks; lanes stride over every runtime-sized dense workspace.
constexpr int kThreads = 32;
#ifdef CLQR_USE_FLOAT
constexpr Scalar kMinimumFeasibilityConsistencyTolerance = 1e-4f;
constexpr Scalar kMinimumMultiplierRankTolerance = 1e-4f;
constexpr Scalar kMultiplierConsistencyTolerancePerTreeLevel = 2e-2f;
constexpr Scalar kMinimumDualRelationRowScale = 1e-6f;
#else
constexpr Scalar kMinimumFeasibilityConsistencyTolerance = Scalar{0};
constexpr Scalar kMinimumMultiplierRankTolerance = 1e-7;
constexpr Scalar kMultiplierConsistencyTolerancePerTreeLevel = 1e-6;
constexpr Scalar kMinimumDualRelationRowScale = 1e-14;
#endif
constexpr Scalar kScalarMax = std::numeric_limits<Scalar>::max();

__host__ __device__ constexpr std::size_t AlignUp(std::size_t value,
                                                  std::size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

struct ScratchSize {
  std::size_t bytes = 0;

  template <typename T> __host__ __device__ void Add(std::size_t count) {
    constexpr std::size_t maximum = std::numeric_limits<std::size_t>::max();
    if (bytes > maximum - (alignof(T) - 1)) {
      bytes = maximum;
      return;
    }
    bytes = AlignUp(bytes, alignof(T));
    if (count > (maximum - bytes) / sizeof(T)) {
      bytes = maximum;
      return;
    }
    bytes += count * sizeof(T);
  }
};

__host__ __device__ std::size_t
DualRelationLeafScratchBytes(std::size_t matrix_entries, std::size_t rows,
                             std::size_t state_constraints) {
  ScratchSize scratch_size;
  scratch_size.Add<Scalar>(matrix_entries);
  scratch_size.Add<Scalar>(rows);
  scratch_size.Add<Scalar>(state_constraints);
  scratch_size.Add<int>(rows);
  scratch_size.Add<int>(rows);
  return scratch_size.bytes;
}

std::size_t ScratchCheckedProduct(std::size_t first, std::size_t second,
                                  const char *description) {
  if (first != 0 && second > std::numeric_limits<std::size_t>::max() / first) {
    throw std::invalid_argument(std::string(description) + " size overflows");
  }
  return first * second;
}

std::size_t ScratchCheckedSum(std::initializer_list<std::size_t> terms,
                              const char *description) {
  std::size_t result = 0;
  for (const std::size_t term : terms) {
    if (term > std::numeric_limits<std::size_t>::max() - result) {
      throw std::invalid_argument(std::string(description) + " size overflows");
    }
    result += term;
  }
  return result;
}

std::size_t DenseEliminationScratchBytes(std::size_t rows, std::size_t columns,
                                         const char *description) {
  ScratchSize size;
  size.Add<Scalar>(ScratchCheckedProduct(rows, columns, description));
  size.Add<Scalar>(rows);
  size.Add<int>(rows);
  size.Add<int>(rows);
  return size.bytes;
}

struct ScanShape {
  std::size_t left = 0;
  std::size_t right = 0;
  std::size_t rows = 0;
  bool valid = false;
};

ScanShape MakeScanShape(std::size_t left, std::size_t right) {
  return {left, right,
          ScratchCheckedSum({left, right}, "scan relation workspace"), true};
}

ScanShape ComposeScanShapes(const ScanShape &first, const ScanShape &second) {
  if (!first.valid)
    return second;
  if (!second.valid)
    return first;
  if (first.right != second.left) {
    throw std::invalid_argument(
        "internal CUDA scratch scan has incompatible dimensions");
  }
  return MakeScanShape(first.left, second.right);
}

struct ScanPlan {
  std::vector<std::vector<ScanShape>> reductions;
  std::vector<std::vector<ScanShape>> suffix_contexts;
  std::vector<std::vector<ScanShape>> prefix_contexts;
};

ScanPlan BuildScanPlan(const std::vector<ScanShape> &leaves) {
  ScanPlan tree;
  tree.reductions.push_back(leaves);
  while (tree.reductions.back().size() > 1) {
    const auto &children = tree.reductions.back();
    std::vector<ScanShape> parents((children.size() + 1) / 2);
    for (std::size_t parent = 0; parent < parents.size(); ++parent) {
      const std::size_t child = 2 * parent;
      parents[parent] = children[child];
      if (child + 1 < children.size()) {
        parents[parent] =
            ComposeScanShapes(children[child], children[child + 1]);
      }
    }
    tree.reductions.push_back(std::move(parents));
  }

  tree.suffix_contexts.resize(tree.reductions.size());
  tree.prefix_contexts.resize(tree.reductions.size());
  for (std::size_t level = 0; level < tree.reductions.size(); ++level) {
    tree.suffix_contexts[level].resize(tree.reductions[level].size());
    tree.prefix_contexts[level].resize(tree.reductions[level].size());
  }
  if (tree.reductions.size() == 1)
    return tree;

  for (int level = static_cast<int>(tree.reductions.size()) - 2; level >= 1;
       --level) {
    const auto &reductions = tree.reductions[level];
    const auto &suffix_parents = tree.suffix_contexts[level + 1];
    const auto &prefix_parents = tree.prefix_contexts[level + 1];
    auto &suffix_children = tree.suffix_contexts[level];
    auto &prefix_children = tree.prefix_contexts[level];
    for (std::size_t parent = 0; parent < suffix_parents.size(); ++parent) {
      const std::size_t child = 2 * parent;
      const ScanShape &suffix_parent = suffix_parents[parent];
      const ScanShape &prefix_parent = prefix_parents[parent];
      suffix_children[child] = suffix_parent;
      prefix_children[child] = prefix_parent;
      if (child + 1 < reductions.size()) {
        suffix_children[child] =
            ComposeScanShapes(reductions[child + 1], suffix_parent);
        suffix_children[child + 1] = suffix_parent;
        prefix_children[child + 1] =
            ComposeScanShapes(prefix_parent, reductions[child]);
      }
    }
  }
  return tree;
}

std::size_t RelationComposeScratchBytes(const ScanShape &first,
                                        const ScanShape &second,
                                        const char *description) {
  const std::size_t rows =
      ScratchCheckedSum({first.rows, second.rows}, description);
  const std::size_t columns = ScratchCheckedSum(
      {first.right, first.left, second.right, 1}, description);
  return DenseEliminationScratchBytes(rows, columns, description);
}

std::size_t RelationFinalizeScratchBytes(const ScanShape &left,
                                         const ScanShape *right,
                                         const ScanShape &parent) {
  std::size_t scratch_left = left.left;
  std::size_t scratch_right = left.right;
  std::size_t matrix_rows = left.rows;
  std::size_t matrix_columns = ScratchCheckedSum(
      {left.left, left.right, 1}, "primal-relation final workspace");
  if (right != nullptr) {
    scratch_left = std::max(scratch_left, right->left);
    scratch_right = std::max(scratch_right, right->right);
    matrix_rows = std::max(
        matrix_rows, ScratchCheckedSum({left.rows, right->rows},
                                       "primal-relation final workspace"));
    matrix_columns =
        std::max(matrix_columns,
                 ScratchCheckedSum({left.right, left.left, right->right, 1},
                                   "primal-relation final workspace"));
  }
  if (parent.valid) {
    scratch_right = std::max(scratch_right, parent.right);
    const ScanShape &child = right != nullptr ? *right : left;
    matrix_rows = std::max(
        matrix_rows, ScratchCheckedSum({child.rows, parent.rows},
                                       "primal-relation final workspace"));
    matrix_columns =
        std::max(matrix_columns,
                 ScratchCheckedSum({child.right, child.left, parent.right, 1},
                                   "primal-relation final workspace"));
  }
  const std::size_t relation_rows = ScratchCheckedSum(
      {scratch_left, scratch_right}, "primal-relation final workspace");
  const std::size_t relation_entries = ScratchCheckedSum(
      {ScratchCheckedProduct(relation_rows, relation_rows,
                             "primal-relation final workspace"),
       relation_rows},
      "primal-relation final workspace");
  ScratchSize size;
  size.Add<Relation>(1);
  size.Add<Scalar>(relation_entries);
  size.Add<Scalar>(ScratchCheckedProduct(matrix_rows, matrix_columns,
                                         "primal-relation final workspace"));
  size.Add<Scalar>(matrix_rows);
  size.Add<int>(matrix_rows);
  size.Add<int>(matrix_rows);
  return size.bytes;
}

std::size_t ValueComposeScratchBytes(const ScanShape &first,
                                     const char *description) {
  const std::size_t columns = ScratchCheckedSum(
      {ScratchCheckedProduct(2, first.right, description), first.left, 1},
      description);
  return DenseEliminationScratchBytes(first.right, columns, description);
}

std::size_t ValueFinalizeScratchBytes(const ScanShape &left,
                                      const ScanShape *right,
                                      const ScanShape &parent) {
  std::size_t left_capacity = left.left;
  std::size_t right_capacity = left.right;
  std::size_t shared_capacity = left.right;
  std::size_t columns_capacity = ScratchCheckedSum(
      {ScratchCheckedProduct(2, left.right, "value-scan final workspace"),
       left.left, 1},
      "value-scan final workspace");
  if (right != nullptr) {
    left_capacity = std::max(left_capacity, right->left);
    right_capacity = std::max(right_capacity, right->right);
    shared_capacity = std::max(shared_capacity, right->right);
  }
  if (parent.valid) {
    right_capacity = std::max(right_capacity, parent.right);
    const ScanShape &child = right != nullptr ? *right : left;
    shared_capacity = std::max(shared_capacity, child.right);
    columns_capacity = std::max(
        columns_capacity,
        ScratchCheckedSum({ScratchCheckedProduct(2, child.right,
                                                 "value-scan final workspace"),
                           child.left, 1},
                          "value-scan final workspace"));
  }
  const std::size_t composed_entries =
      ScratchCheckedSum({ScratchCheckedProduct(right_capacity, left_capacity,
                                               "value-scan final workspace"),
                         right_capacity,
                         ScratchCheckedProduct(right_capacity, right_capacity,
                                               "value-scan final workspace"),
                         left_capacity,
                         ScratchCheckedProduct(left_capacity, left_capacity,
                                               "value-scan final workspace")},
                        "value-scan final workspace");
  ScratchSize size;
  size.Add<Scalar>(ScratchCheckedProduct(shared_capacity, columns_capacity,
                                         "value-scan final workspace"));
  size.Add<Scalar>(shared_capacity);
  size.Add<ValueElement>(1);
  size.Add<Scalar>(composed_entries);
  size.Add<int>(shared_capacity);
  size.Add<int>(shared_capacity);
  return size.bytes;
}

std::size_t AffineFinalizeScratchBytes(const ScanShape &left,
                                       const ScanShape *right,
                                       const ScanShape &parent) {
  std::size_t left_capacity = left.left;
  std::size_t right_capacity = left.right;
  if (parent.valid)
    left_capacity = std::max(left_capacity, parent.left);
  if (right != nullptr)
    right_capacity = std::max(right_capacity, right->right);
  ScratchSize size;
  size.Add<AffineMap>(1);
  size.Add<Scalar>(
      ScratchCheckedSum({ScratchCheckedProduct(left_capacity, right_capacity,
                                               "affine-scan final workspace"),
                         right_capacity},
                        "affine-scan final workspace"));
  return size.bytes;
}

std::size_t DualRootScratchBytes(const ScanShape &root) {
  const std::size_t variables = root.left;
  ScratchSize size;
  size.Add<Scalar>(ScratchCheckedProduct(
      root.rows, ScratchCheckedSum({variables, 1}, "dual-root workspace"),
      "dual-root workspace"));
  size.Add<Scalar>(root.rows);
  size.Add<Scalar>(
      ScratchCheckedProduct(variables, variables, "dual-root workspace"));
  size.Add<Scalar>(variables);
  size.Add<Scalar>(variables);
  size.Add<int>(variables);
  return size.bytes;
}

std::size_t DualExpandScratchBytes(const ScanShape &left,
                                   const ScanShape &right) {
  const std::size_t shared = left.right;
  const std::size_t rows =
      ScratchCheckedSum({left.rows, right.rows}, "dual-expansion workspace");
  ScratchSize size;
  size.Add<Scalar>(ScratchCheckedProduct(
      rows, ScratchCheckedSum({shared, 1}, "dual-expansion workspace"),
      "dual-expansion workspace"));
  size.Add<Scalar>(rows);
  size.Add<Scalar>(
      ScratchCheckedProduct(shared, shared, "dual-expansion workspace"));
  size.Add<Scalar>(shared);
  size.Add<Scalar>(shared);
  size.Add<int>(shared);
  return size.bytes;
}

struct ScratchRequirements {
  std::size_t primal_leaf = 0;
  std::size_t primal_relation = 0;
  std::size_t primal_relation_final = 0;
  std::size_t state_parameter = 0;
  std::size_t stage_reduction = 0;
  std::size_t value_leaf = 0;
  std::size_t value_compose = 0;
  std::size_t value_finalize = 0;
  std::size_t feedback = 0;
  std::size_t affine_finalize = 0;
  std::size_t dual_parameter = 0;
  std::size_t dual_relation_leaf = 0;
  std::size_t dual_relation = 0;
  std::size_t dual_root = 0;
  std::size_t dual_expand = 0;

  std::size_t Maximum() const {
    return std::max({primal_leaf, primal_relation, primal_relation_final,
                     state_parameter, stage_reduction, value_leaf,
                     value_compose, value_finalize, feedback, affine_finalize,
                     dual_parameter, dual_relation_leaf, dual_relation,
                     dual_root, dual_expand});
  }
};

ScratchRequirements PlanScratch(const Problem &problem) {
  ScratchRequirements result;
  const std::size_t stage_count = problem.stages.size();
  const std::size_t node_count = stage_count + 1;
  std::vector<std::size_t> state_bounds(node_count);
  std::vector<ScanShape> primal_leaves(node_count);
  std::vector<ScanShape> value_leaves(node_count);
  std::vector<ScanShape> affine_leaves(stage_count);
  std::vector<std::size_t> dual_bounds(stage_count);
  for (std::size_t index = 0; index < stage_count; ++index) {
    const Stage &stage = problem.stages[index];
    const std::size_t n = stage.A.cols();
    const std::size_t next = stage.A.rows();
    const std::size_t m = stage.B.cols();
    const std::size_t mixed = stage.C.rows();
    const std::size_t state_constraints = stage.E.rows();
    const std::size_t dual = ScratchCheckedSum({next, mixed}, "dual dimension");
    state_bounds[index] = n;
    primal_leaves[index] = MakeScanShape(n, next);
    value_leaves[index] = MakeScanShape(n, next);
    affine_leaves[index] = MakeScanShape(n, next);
    dual_bounds[index] = dual;

    result.primal_leaf = std::max(
        result.primal_leaf,
        DenseEliminationScratchBytes(
            ScratchCheckedSum({mixed, state_constraints, next},
                              "feasibility row workspace"),
            ScratchCheckedSum({m, n, next, 1}, "feasibility column workspace"),
            "feasibility kernel workspace"));
    result.stage_reduction = std::max(
        result.stage_reduction,
        DenseEliminationScratchBytes(
            ScratchCheckedSum({mixed, next}, "reduction row workspace"),
            ScratchCheckedSum({m, n, 1}, "reduction column workspace"),
            "reduction kernel workspace"));
    ScratchSize value_leaf;
    value_leaf.Add<Scalar>(ScratchCheckedProduct(m, m, "value-leaf workspace"));
    value_leaf.Add<Scalar>(ScratchCheckedProduct(
        m, ScratchCheckedSum({n, next, 1}, "value-leaf workspace"),
        "value-leaf workspace"));
    result.value_leaf = std::max(result.value_leaf, value_leaf.bytes);
    ScratchSize feedback;
    feedback.Add<Scalar>(ScratchCheckedProduct(
        m, ScratchCheckedSum({m, n, 1}, "feedback workspace"),
        "feedback workspace"));
    feedback.Add<Scalar>(
        ScratchCheckedProduct(m, m, "feedback workspace"));
    result.feedback = std::max(result.feedback, feedback.bytes);

    ScratchSize dual_parameter;
    const std::size_t dual_rows =
        ScratchCheckedSum({next, m}, "dual-parameter workspace");
    dual_parameter.Add<Scalar>(ScratchCheckedProduct(
        dual_rows, ScratchCheckedSum({dual, 1}, "dual-parameter workspace"),
        "dual-parameter workspace"));
    dual_parameter.Add<Scalar>(mixed);
    dual_parameter.Add<Scalar>(dual_rows);
    dual_parameter.Add<Scalar>(
        ScratchCheckedProduct(dual, dual, "dual-parameter workspace"));
    dual_parameter.Add<Scalar>(dual);
    dual_parameter.Add<Scalar>(dual);
    dual_parameter.Add<int>(dual);
    result.dual_parameter =
        std::max(result.dual_parameter, dual_parameter.bytes);
  }

  const std::size_t terminal_n = problem.terminal_Q.rows();
  state_bounds.back() = terminal_n;
  primal_leaves.back() = MakeScanShape(terminal_n, 0);
  value_leaves.back() = MakeScanShape(terminal_n, 0);
  result.primal_leaf =
      std::max(result.primal_leaf,
               DenseEliminationScratchBytes(
                   problem.terminal_E.rows(),
                   ScratchCheckedSum({terminal_n, 1}, "terminal workspace"),
                   "terminal feasibility workspace"));

  ScratchSize state_parameter;
  state_parameter.Add<int>(
      *std::max_element(state_bounds.begin(), state_bounds.end()));
  result.state_parameter = state_parameter.bytes;

  if (node_count > 1) {
    const ScanPlan primal_tree = BuildScanPlan(primal_leaves);
    for (std::size_t level = 1; level < primal_tree.reductions.size();
         ++level) {
      const auto &children = primal_tree.reductions[level - 1];
      for (std::size_t parent = 0;
           parent < primal_tree.reductions[level].size(); ++parent) {
        const std::size_t child = 2 * parent;
        if (child + 1 < children.size()) {
          result.primal_relation = std::max(
              result.primal_relation,
              RelationComposeScratchBytes(children[child], children[child + 1],
                                          "primal-relation workspace"));
        }
      }
    }
    for (int level = static_cast<int>(primal_tree.reductions.size()) - 2;
         level >= 1; --level) {
      const auto &children = primal_tree.reductions[level];
      const auto &parents = primal_tree.suffix_contexts[level + 1];
      for (std::size_t parent = 0; parent < parents.size(); ++parent) {
        const std::size_t child = 2 * parent;
        if (child + 1 < children.size()) {
          result.primal_relation = std::max(
              result.primal_relation,
              RelationComposeScratchBytes(children[child + 1], parents[parent],
                                          "primal-relation workspace"));
        }
      }
    }
    const auto &parents = primal_tree.suffix_contexts[1];
    for (std::size_t parent = 0; parent < parents.size(); ++parent) {
      const std::size_t child = 2 * parent;
      const ScanShape *right = child + 1 < primal_leaves.size()
                                   ? &primal_leaves[child + 1]
                                   : nullptr;
      result.primal_relation_final =
          std::max(result.primal_relation_final,
                   RelationFinalizeScratchBytes(primal_leaves[child], right,
                                                parents[parent]));
    }

    const ScanPlan value_tree = BuildScanPlan(value_leaves);
    for (std::size_t level = 1; level < value_tree.reductions.size(); ++level) {
      const auto &children = value_tree.reductions[level - 1];
      for (std::size_t parent = 0; parent < value_tree.reductions[level].size();
           ++parent) {
        const std::size_t child = 2 * parent;
        if (child + 1 < children.size()) {
          result.value_compose =
              std::max(result.value_compose,
                       ValueComposeScratchBytes(children[child],
                                                "value-scan workspace"));
        }
      }
    }
    for (int level = static_cast<int>(value_tree.reductions.size()) - 2;
         level >= 1; --level) {
      const auto &children = value_tree.reductions[level];
      const auto &parents = value_tree.suffix_contexts[level + 1];
      for (std::size_t parent = 0; parent < parents.size(); ++parent) {
        const std::size_t child = 2 * parent;
        if (child + 1 < children.size()) {
          result.value_compose =
              std::max(result.value_compose,
                       ValueComposeScratchBytes(children[child + 1],
                                                "value-scan workspace"));
        }
      }
    }
    const auto &value_parents = value_tree.suffix_contexts[1];
    for (std::size_t parent = 0; parent < value_parents.size(); ++parent) {
      const std::size_t child = 2 * parent;
      const ScanShape *right =
          child + 1 < value_leaves.size() ? &value_leaves[child + 1] : nullptr;
      result.value_finalize =
          std::max(result.value_finalize,
                   ValueFinalizeScratchBytes(value_leaves[child], right,
                                             value_parents[parent]));
    }
  }

  if (stage_count > 1) {
    const ScanPlan affine_tree = BuildScanPlan(affine_leaves);
    const auto &parents = affine_tree.prefix_contexts[1];
    for (std::size_t parent = 0; parent < parents.size(); ++parent) {
      const std::size_t child = 2 * parent;
      const ScanShape *right = child + 1 < affine_leaves.size()
                                   ? &affine_leaves[child + 1]
                                   : nullptr;
      result.affine_finalize =
          std::max(result.affine_finalize,
                   AffineFinalizeScratchBytes(affine_leaves[child], right,
                                              parents[parent]));
    }
  }

  if (stage_count > 0) {
    std::vector<ScanShape> dual_leaves(stage_count);
    for (std::size_t index = 0; index < stage_count; ++index) {
      const std::size_t node = index + 1;
      const std::size_t state_dim = problem.stages[index].A.rows();
      const std::size_t state_constraints = node == stage_count
                                                ? problem.terminal_E.rows()
                                                : problem.stages[node].E.rows();
      const std::size_t right =
          node == stage_count ? 0 : dual_bounds[index + 1];
      const std::size_t columns =
          ScratchCheckedSum({state_constraints, dual_bounds[index], right, 1},
                            "dual-relation leaf workspace");
      result.dual_relation_leaf =
          std::max(result.dual_relation_leaf,
                   DualRelationLeafScratchBytes(
                       ScratchCheckedProduct(state_dim, columns,
                                             "dual-relation leaf workspace"),
                       state_dim, state_constraints));
      dual_leaves[index] = MakeScanShape(dual_bounds[index], right);
    }
    const ScanPlan dual_tree = BuildScanPlan(dual_leaves);
    for (std::size_t level = 1; level < dual_tree.reductions.size(); ++level) {
      const auto &children = dual_tree.reductions[level - 1];
      for (std::size_t parent = 0; parent < dual_tree.reductions[level].size();
           ++parent) {
        const std::size_t child = 2 * parent;
        if (child + 1 < children.size()) {
          result.dual_relation = std::max(
              result.dual_relation,
              RelationComposeScratchBytes(children[child], children[child + 1],
                                          "dual-relation workspace"));
          result.dual_expand = std::max(
              result.dual_expand,
              DualExpandScratchBytes(children[child], children[child + 1]));
        }
      }
    }
    result.dual_root =
        DualRootScratchBytes(dual_tree.reductions.back().front());
  }
  return result;
}

std::vector<int> BuildStructureKey(const Problem &problem) {
  std::vector<int> key;
  key.reserve(5 * problem.stages.size() + 3);
  key.push_back(static_cast<int>(problem.initial_state.size()));
  key.push_back(static_cast<int>(problem.terminal_Q.rows()));
  key.push_back(static_cast<int>(problem.terminal_E.rows()));
  for (const Stage &stage : problem.stages) {
    key.push_back(static_cast<int>(stage.A.cols()));
    key.push_back(static_cast<int>(stage.A.rows()));
    key.push_back(static_cast<int>(stage.B.cols()));
    key.push_back(static_cast<int>(stage.C.rows()));
    key.push_back(static_cast<int>(stage.E.rows()));
  }
  return key;
}

bool RefreshScratchPlan(const Problem &problem, std::vector<int> *structure_key,
                        ScratchRequirements *scratch) {
  std::vector<int> key = BuildStructureKey(problem);
  if (key == *structure_key)
    return false;
  *scratch = PlanScratch(problem);
  *structure_key = std::move(key);
  return true;
}

struct ScratchArena {
  unsigned char *data;
  std::size_t offset = 0;

  template <typename T> __device__ T *Take(std::size_t count) {
    offset = AlignUp(offset, alignof(T));
    T *result = reinterpret_cast<T *>(data + offset);
    offset += count * sizeof(T);
    return result;
  }
};

#ifdef CLQR_CUDA_EMULATION
unsigned char *EmulatedBlockScratch(std::size_t bytes) {
  // vector<unsigned char> does not promise the alignment required by Scalar
  // and the small scan records placed in this arena.  max_align_t does, while
  // retaining the same reusable, exactly runtime-sized emulation buffer.
  static thread_local std::vector<std::max_align_t> storage;
  const std::size_t words =
      bytes / sizeof(std::max_align_t) +
      static_cast<std::size_t>(bytes % sizeof(std::max_align_t) != 0);
  storage.resize(std::max<std::size_t>(words, 1));
  return reinterpret_cast<unsigned char *>(storage.data());
}
#define CLQR_BLOCK_SCRATCH(name, required_bytes)                               \
  ScratchArena name { EmulatedBlockScratch(required_bytes) }
#else
#define CLQR_BLOCK_SCRATCH(name, required_bytes)                               \
  extern __shared__ __align__(16) unsigned char clqr_shared_memory[];          \
  (void)(required_bytes);                                                      \
  ScratchArena name { clqr_shared_memory }
#endif

__device__ inline Scalar DeviceAbs(Scalar x) { return x < Scalar{0} ? -x : x; }

__device__ inline bool DeviceFinite(Scalar x) {
  return x >= -kScalarMax && x <= kScalarMax;
}

__device__ inline int DeviceMax(int first, int second) {
  return first > second ? first : second;
}

#ifdef CLQR_CUDA_EMULATION
constexpr int kActiveWarpWidth = 1;
#else
constexpr int kActiveWarpWidth = 32;
#endif

__device__ Scalar WarpSum(Scalar value) {
#ifdef CLQR_CUDA_EMULATION
  return value;
#else
  constexpr unsigned kFullWarp = 0xffffffffu;
  for (int offset = 16; offset > 0; offset /= 2)
    value += __shfl_down_sync(kFullWarp, value, offset);
  return __shfl_sync(kFullWarp, value, 0);
#endif
}

__device__ Scalar WarpMaximum(Scalar value) {
#ifdef CLQR_CUDA_EMULATION
  return value;
#else
  constexpr unsigned kFullWarp = 0xffffffffu;
  for (int offset = 16; offset > 0; offset /= 2)
    value = fmax(value, __shfl_down_sync(kFullWarp, value, offset));
  return __shfl_sync(kFullWarp, value, 0);
#endif
}

__device__ int WarpBroadcastLaneZero(int value) {
#ifdef CLQR_CUDA_EMULATION
  return value;
#else
  return __shfl_sync(0xffffffffu, value, 0);
#endif
}

__device__ void WarpSynchronize() {
#ifndef CLQR_CUDA_EMULATION
  __syncwarp();
#endif
}

// Factor a small symmetric positive-definite matrix into a dense lower
// Cholesky factor. The coefficient matrix may be the leading block of a wider
// augmented matrix. Native CUDA and kernel emulation use this same source.
__device__ bool FactorPositiveDefiniteBlock(const Scalar *coefficient,
                                            int coefficient_stride,
                                            int dimension, Scalar tolerance,
                                            Scalar *lower,
                                            int *positive_definite) {
  for (int linear = threadIdx.x; linear < dimension * dimension;
       linear += blockDim.x) {
    const int row = linear / dimension;
    const int col = linear % dimension;
    lower[linear] = Scalar{0.5} * (coefficient[row * coefficient_stride + col] +
                                   coefficient[col * coefficient_stride + row]);
  }
  WarpSynchronize();
  if (threadIdx.x == 0) {
    *positive_definite = 1;
    Scalar scale = Scalar{0};
    for (int diagonal = 0; diagonal < dimension; ++diagonal) {
      scale = fmax(scale, DeviceAbs(lower[diagonal * dimension + diagonal]));
    }
    if (!(scale > Scalar{0}) || !DeviceFinite(scale))
      *positive_definite = 0;
    for (int col = 0; col < dimension && *positive_definite; ++col) {
      Scalar diagonal = lower[col * dimension + col];
      for (int k = 0; k < col; ++k) {
        diagonal -= lower[col * dimension + k] * lower[col * dimension + k];
      }
      if (!(diagonal > tolerance * scale) || !DeviceFinite(diagonal)) {
        *positive_definite = 0;
        break;
      }
      lower[col * dimension + col] = sqrt(diagonal);
      for (int row = col + 1; row < dimension; ++row) {
        Scalar value = lower[row * dimension + col];
        for (int k = 0; k < col; ++k) {
          value -= lower[row * dimension + k] * lower[col * dimension + k];
        }
        lower[row * dimension + col] = value / lower[col * dimension + col];
      }
    }
  }
  WarpSynchronize();
  return *positive_definite != 0;
}

// Solve L*L^T*X = B for several right-hand sides. B is overwritten by X and
// may itself be a strided suffix of an augmented matrix.
__device__ void SolvePositiveDefiniteMultipleRhsBlock(
    const Scalar *lower, int dimension, Scalar *right_hand_sides,
    int right_hand_side_stride, int right_hand_side_count) {
  for (int rhs = threadIdx.x; rhs < right_hand_side_count; rhs += blockDim.x) {
    for (int row = 0; row < dimension; ++row) {
      Scalar value = right_hand_sides[row * right_hand_side_stride + rhs];
      for (int col = 0; col < row; ++col) {
        value -= lower[row * dimension + col] *
                 right_hand_sides[col * right_hand_side_stride + rhs];
      }
      right_hand_sides[row * right_hand_side_stride + rhs] =
          value / lower[row * dimension + row];
    }
    for (int reverse = 0; reverse < dimension; ++reverse) {
      const int row = dimension - 1 - reverse;
      Scalar value = right_hand_sides[row * right_hand_side_stride + rhs];
      for (int col = row + 1; col < dimension; ++col) {
        value -= lower[col * dimension + row] *
                 right_hand_sides[col * right_hand_side_stride + rhs];
      }
      right_hand_sides[row * right_hand_side_stride + rhs] =
          value / lower[row * dimension + row];
    }
  }
  WarpSynchronize();
}

__device__ void BindValueElementScratch(ValueElement *element, Scalar *storage,
                                        int left_capacity, int right_capacity) {
  element->A = storage;
  element->b = element->A + right_capacity * left_capacity;
  element->C = element->b + right_capacity;
  element->eta = element->C + right_capacity * right_capacity;
  element->J = element->eta + left_capacity;
}

__device__ void BindAffineMapScratch(AffineMap *map, Scalar *storage,
                                     int left_capacity, int right_capacity) {
  map->linear = storage;
  map->offset = map->linear + right_capacity * left_capacity;
}

template <typename RelationType>
__device__ void BindRelationScratch(RelationType *relation, Scalar *storage,
                                    int left_capacity, int right_capacity) {
  const int rows = left_capacity + right_capacity;
  relation->left = storage;
  relation->right = relation->left + rows * left_capacity;
  relation->rhs = relation->right + rows * right_capacity;
}

#ifdef CLQR_CUDA_EMULATION
__device__ void BindDualRelationScratch(DualRelation *relation, Scalar *storage,
                                        int left_capacity, int right_capacity) {
  BindRelationScratch(relation, storage, left_capacity, right_capacity);
}

__device__ void BindDualValueScratch(DualNodeValue *value, Scalar *storage,
                                     int left_capacity) {
  value->left = storage;
  value->right = value->left + left_capacity;
}
#endif

__device__ void SetFailure(DeviceStatus *status, int code, int stage,
                           int detail) {
  if (atomicCAS(&status->code, kDeviceOk, code) == kDeviceOk) {
    status->stage = stage;
    status->detail = detail;
  }
}

// A global failure can be reported by another block at any time.  Sampling it
// independently in every lane before a later warp synchronization can
// therefore make only part of a warp return.  Have one lane sample the flag
// and broadcast a warp-uniform decision instead.
__device__ bool BlockEnabled(const DeviceStatus *status) {
  int enabled = 0;
  if (threadIdx.x == 0)
    enabled = atomicCAS(const_cast<int *>(&status->code), kDeviceOk,
                        kDeviceOk) == kDeviceOk;
  return WarpBroadcastLaneZero(enabled) != 0;
}

// Scale each nonzero equation before pivoting, then use partial row pivoting.
// This makes rank decisions invariant to independent equation rescaling while
// retaining the deterministic free-column convention of the CPU RREF path.
// Generated relations may supply a positive minimum_row_scale to prevent
// roundoff-level cancellation noise from being normalized to order one.
__device__ void RrefBlock(Scalar *matrix, int rows, int columns,
                          int pivot_limit, Scalar tolerance, int *pivot_columns,
                          int *pivot_rows, int *rank, int *best_row,
                          Scalar *factors,
                          Scalar minimum_row_scale = Scalar{0}) {
  for (int row = threadIdx.x; row < rows; row += blockDim.x) {
    Scalar scale = Scalar{0};
    for (int col = 0; col < pivot_limit; ++col) {
      scale = fmax(scale, DeviceAbs(matrix[row * columns + col]));
    }
    if (scale > minimum_row_scale) {
      for (int col = 0; col < columns; ++col)
        matrix[row * columns + col] /= scale;
    }
  }
  if (threadIdx.x == 0)
    *rank = 0;
  WarpSynchronize();

  for (int col = 0; col < pivot_limit; ++col) {
    if (threadIdx.x == 0) {
      *best_row = -1;
      Scalar best = tolerance;
      for (int row = *rank; row < rows; ++row) {
        const Scalar candidate = DeviceAbs(matrix[row * columns + col]);
        if (candidate > best) {
          best = candidate;
          *best_row = row;
        }
      }
    }
    WarpSynchronize();
    const int selected_row = *best_row;
    // A no-pivot iteration skips the barriers below.  Ensure every thread has
    // consumed best_row before thread 0 reuses it in the next iteration.
    WarpSynchronize();
    if (selected_row < 0)
      continue;

    const int pivot_row = *rank;
    if (selected_row != pivot_row) {
      for (int j = threadIdx.x; j < columns; j += blockDim.x) {
        const Scalar tmp = matrix[pivot_row * columns + j];
        matrix[pivot_row * columns + j] = matrix[selected_row * columns + j];
        matrix[selected_row * columns + j] = tmp;
      }
    }
    WarpSynchronize();

    const Scalar pivot = matrix[pivot_row * columns + col];
    // All threads must load the pivot before any thread normalizes its entry.
    WarpSynchronize();
    for (int j = col + threadIdx.x; j < columns; j += blockDim.x) {
      matrix[pivot_row * columns + j] /= pivot;
    }
    WarpSynchronize();

    for (int row = threadIdx.x; row < rows; row += blockDim.x) {
      factors[row] = row == pivot_row ? Scalar{0} : matrix[row * columns + col];
    }
    WarpSynchronize();
    for (int index = threadIdx.x; index < rows * (columns - col);
         index += blockDim.x) {
      const int row = index / (columns - col);
      const int j = col + index % (columns - col);
      if (row != pivot_row) {
        matrix[row * columns + j] -=
            factors[row] * matrix[pivot_row * columns + j];
      }
    }
    WarpSynchronize();
    if (threadIdx.x == 0) {
      pivot_columns[pivot_row] = col;
      pivot_rows[pivot_row] = pivot_row;
      ++(*rank);
    }
    WarpSynchronize();
    if (*rank == rows)
      break;
  }

  for (int index = threadIdx.x; index < rows * columns; index += blockDim.x) {
    if (DeviceAbs(matrix[index]) <= tolerance)
      matrix[index] = Scalar{0};
  }
  WarpSynchronize();
}

__device__ bool InconsistentRref(const Scalar *matrix, int rows, int columns,
                                 int lhs_columns, Scalar lhs_tolerance,
                                 Scalar rhs_tolerance) {
  for (int row = 0; row < rows; ++row) {
    bool zero = true;
    for (int col = 0; col < lhs_columns; ++col) {
      if (DeviceAbs(matrix[row * columns + col]) > lhs_tolerance) {
        zero = false;
        break;
      }
    }
    if (zero && DeviceAbs(matrix[row * columns + lhs_columns]) > rhs_tolerance)
      return true;
  }
  return false;
}

// Measure the right-hand side relative to each equation's largest coefficient
// before the matrix is overwritten, so long multiplier chains are tested on a
// scale-independent residual.
__device__ Scalar ConditionedRhsScale(const Scalar *matrix, int rows,
                                      int columns, int lhs_columns,
                                      Scalar rank_tolerance) {
  Scalar scale = Scalar{1};
  for (int row = 0; row < rows; ++row) {
    Scalar lhs_scale = Scalar{0};
    for (int col = 0; col < lhs_columns; ++col) {
      const Scalar value = DeviceAbs(matrix[row * columns + col]);
      if (value > lhs_scale)
        lhs_scale = value;
    }
    Scalar rhs_scale = DeviceAbs(matrix[row * columns + lhs_columns]);
    if (lhs_scale > rank_tolerance)
      rhs_scale /= lhs_scale;
    if (rhs_scale > scale)
      scale = rhs_scale;
  }
  return scale;
}

template <typename RelationType>
__device__ void ExtractResidualRelation(const Scalar *matrix, int columns,
                                        int rank, const int *pivot_columns,
                                        int eliminated_columns, int left_dim,
                                        int right_dim, RelationType *output) {
  if (threadIdx.x == 0) {
    int eliminated_rank = 0;
    while (eliminated_rank < rank &&
           pivot_columns[eliminated_rank] < eliminated_columns) {
      ++eliminated_rank;
    }
    output->left_dim = left_dim;
    output->right_dim = right_dim;
    output->rows = rank - eliminated_rank;
  }
  WarpSynchronize();
  const int eliminated_rank = rank - output->rows;
  const int outer_dim = left_dim + right_dim;
  for (int index = threadIdx.x; index < output->rows * outer_dim;
       index += blockDim.x) {
    const int row = index / outer_dim;
    const int col = index % outer_dim;
    const Scalar value =
        matrix[(eliminated_rank + row) * columns + eliminated_columns + col];
    if (col < left_dim) {
      output->left[row * left_dim + col] = value;
    } else {
      output->right[row * right_dim + col - left_dim] = value;
    }
  }
  for (int row = threadIdx.x; row < output->rows; row += blockDim.x) {
    output->rhs[row] = matrix[(eliminated_rank + row) * columns + columns - 1];
  }
  WarpSynchronize();
}

// Eliminate the leading variables by orthogonally projecting the equations
// onto their left nullspace, then retain an orthonormal basis of the resulting
// affine relation.  Pivoted, reorthogonalized modified Gram--Schmidt is used in
// both directions. Stage dimensions are bounded, so the local work is constant
// with respect to the horizon while independent tree nodes remain fully
// parallel.
template <typename RelationType>
__device__ void EliminateRelationOrthogonally(
    Scalar *matrix, int rows, int columns, int eliminated_columns, int left_dim,
    int right_dim, Scalar rank_tolerance, Scalar consistency_tolerance,
    RelationType *relation, int *local_ok) {
  if (threadIdx.x < kActiveWarpWidth) {
    const int lane = threadIdx.x;
    const Scalar rank_threshold_squared = rank_tolerance * rank_tolerance;
    Scalar rhs_scale = Scalar{1};
    for (int row = lane; row < rows; row += kActiveWarpWidth)
      rhs_scale =
          fmax(rhs_scale, DeviceAbs(matrix[row * columns + columns - 1]));
    rhs_scale = WarpMaximum(rhs_scale);

    int eliminated_rank = 0;
    for (int basis = 0; basis < eliminated_columns; ++basis) {
      int best_column = basis;
      Scalar best_norm_squared = Scalar{-1};
      for (int candidate = basis; candidate < eliminated_columns; ++candidate) {
        Scalar norm_squared = Scalar{0};
        for (int row = lane; row < rows; row += kActiveWarpWidth) {
          const Scalar value = matrix[row * columns + candidate];
          norm_squared += value * value;
        }
        norm_squared = WarpSum(norm_squared);
        if (norm_squared > best_norm_squared) {
          best_norm_squared = norm_squared;
          best_column = candidate;
        }
      }
      if (!(best_norm_squared > rank_threshold_squared))
        break;
      if (best_column != basis) {
        for (int row = lane; row < rows; row += kActiveWarpWidth) {
          const Scalar value = matrix[row * columns + basis];
          matrix[row * columns + basis] = matrix[row * columns + best_column];
          matrix[row * columns + best_column] = value;
        }
        WarpSynchronize();
      }
      for (int pass = 0; pass < 2; ++pass) {
        for (int previous = 0; previous < basis; ++previous) {
          Scalar projection = Scalar{0};
          for (int row = lane; row < rows; row += kActiveWarpWidth) {
            projection += matrix[row * columns + basis] *
                          matrix[row * columns + previous];
          }
          projection = WarpSum(projection);
          for (int row = lane; row < rows; row += kActiveWarpWidth) {
            matrix[row * columns + basis] -=
                projection * matrix[row * columns + previous];
          }
          WarpSynchronize();
        }
      }
      Scalar norm_squared = Scalar{0};
      for (int row = lane; row < rows; row += kActiveWarpWidth) {
        const Scalar value = matrix[row * columns + basis];
        norm_squared += value * value;
      }
      norm_squared = WarpSum(norm_squared);
      if (!(norm_squared > rank_threshold_squared))
        break;
      const Scalar inverse_norm = Scalar{1} / sqrt(norm_squared);
      for (int row = lane; row < rows; row += kActiveWarpWidth)
        matrix[row * columns + basis] *= inverse_norm;
      WarpSynchronize();
      for (int candidate = basis + 1; candidate < eliminated_columns;
           ++candidate) {
        for (int pass = 0; pass < 2; ++pass) {
          Scalar projection = Scalar{0};
          for (int row = lane; row < rows; row += kActiveWarpWidth)
            projection += matrix[row * columns + candidate] *
                          matrix[row * columns + basis];
          projection = WarpSum(projection);
          for (int row = lane; row < rows; row += kActiveWarpWidth)
            matrix[row * columns + candidate] -=
                projection * matrix[row * columns + basis];
          WarpSynchronize();
        }
      }
      ++eliminated_rank;
    }

    // Apply the orthogonal projector to the outer coefficients and right-hand
    // side. The remaining rows are precisely the equations that cannot be
    // satisfied by choosing the eliminated variables.
    for (int col = eliminated_columns; col < columns; ++col) {
      for (int basis = 0; basis < eliminated_rank; ++basis) {
        for (int pass = 0; pass < 2; ++pass) {
          Scalar projection = Scalar{0};
          for (int row = lane; row < rows; row += kActiveWarpWidth)
            projection +=
                matrix[row * columns + col] * matrix[row * columns + basis];
          projection = WarpSum(projection);
          for (int row = lane; row < rows; row += kActiveWarpWidth)
            matrix[row * columns + col] -=
                projection * matrix[row * columns + basis];
          WarpSynchronize();
        }
      }
    }

    const int outer_columns = left_dim + right_dim;
    int relation_rank = 0;
    while (relation_rank < rows) {
      int best_row = relation_rank;
      Scalar best_norm_squared = Scalar{-1};
      for (int candidate = relation_rank; candidate < rows; ++candidate) {
        Scalar norm_squared = Scalar{0};
        for (int col = lane; col < outer_columns; col += kActiveWarpWidth) {
          const Scalar value =
              matrix[candidate * columns + eliminated_columns + col];
          norm_squared += value * value;
        }
        norm_squared = WarpSum(norm_squared);
        if (norm_squared > best_norm_squared) {
          best_norm_squared = norm_squared;
          best_row = candidate;
        }
      }
      if (!(best_norm_squared > rank_threshold_squared))
        break;
      if (best_row != relation_rank) {
        for (int outer = lane; outer <= outer_columns;
             outer += kActiveWarpWidth) {
          const int col = eliminated_columns + outer;
          const Scalar value = matrix[relation_rank * columns + col];
          matrix[relation_rank * columns + col] =
              matrix[best_row * columns + col];
          matrix[best_row * columns + col] = value;
        }
        WarpSynchronize();
      }
      for (int pass = 0; pass < 2; ++pass) {
        for (int previous = 0; previous < relation_rank; ++previous) {
          Scalar projection = Scalar{0};
          for (int col = lane; col < outer_columns; col += kActiveWarpWidth) {
            projection +=
                matrix[relation_rank * columns + eliminated_columns + col] *
                matrix[previous * columns + eliminated_columns + col];
          }
          projection = WarpSum(projection);
          for (int col = lane; col <= outer_columns; col += kActiveWarpWidth) {
            matrix[relation_rank * columns + eliminated_columns + col] -=
                projection *
                matrix[previous * columns + eliminated_columns + col];
          }
          WarpSynchronize();
        }
      }
      Scalar norm_squared = Scalar{0};
      for (int col = lane; col < outer_columns; col += kActiveWarpWidth) {
        const Scalar value =
            matrix[relation_rank * columns + eliminated_columns + col];
        norm_squared += value * value;
      }
      norm_squared = WarpSum(norm_squared);
      if (!(norm_squared > rank_threshold_squared))
        break;
      const Scalar inverse_norm = Scalar{1} / sqrt(norm_squared);
      for (int col = lane; col <= outer_columns; col += kActiveWarpWidth)
        matrix[relation_rank * columns + eliminated_columns + col] *=
            inverse_norm;
      WarpSynchronize();
      for (int row = relation_rank + 1; row < rows; ++row) {
        for (int pass = 0; pass < 2; ++pass) {
          Scalar projection = Scalar{0};
          for (int col = lane; col < outer_columns; col += kActiveWarpWidth) {
            projection +=
                matrix[row * columns + eliminated_columns + col] *
                matrix[relation_rank * columns + eliminated_columns + col];
          }
          projection = WarpSum(projection);
          for (int col = lane; col <= outer_columns; col += kActiveWarpWidth) {
            matrix[row * columns + eliminated_columns + col] -=
                projection *
                matrix[relation_rank * columns + eliminated_columns + col];
          }
          WarpSynchronize();
        }
      }
      ++relation_rank;
    }

    Scalar maximum_residual = Scalar{0};
    for (int row = relation_rank + lane; row < rows; row += kActiveWarpWidth)
      maximum_residual = fmax(maximum_residual,
                              DeviceAbs(matrix[row * columns + columns - 1]));
    maximum_residual = WarpMaximum(maximum_residual);
    const bool okay = maximum_residual <= consistency_tolerance * rhs_scale;
    if (lane == 0) {
      *local_ok = okay;
      relation->left_dim = left_dim;
      relation->right_dim = right_dim;
      relation->rows = relation_rank;
    }
    if (okay) {
      const int outer_entries = relation_rank * outer_columns;
      for (int entry = lane; entry < outer_entries; entry += kActiveWarpWidth) {
        const int row = entry / outer_columns;
        const int col = entry % outer_columns;
        const Scalar value = matrix[row * columns + eliminated_columns + col];
        if (col < left_dim) {
          relation->left[row * left_dim + col] = value;
        } else {
          relation->right[row * right_dim + col - left_dim] = value;
        }
      }
      for (int row = lane; row < relation_rank; row += kActiveWarpWidth)
        relation->rhs[row] = matrix[row * columns + columns - 1];
    }
  }
  WarpSynchronize();
}

// Solve an overdetermined system with column-pivoted, reorthogonalized QR.
// Free variables are set to zero in the pivoted coordinates.  The coefficient
// matrix is overwritten by the orthonormal columns.
__device__ void SolveSystemOrthogonally(Scalar *matrix, int rows, int columns,
                                        int variables, Scalar rank_tolerance,
                                        Scalar consistency_tolerance,
                                        Scalar rhs_scale, Scalar *residual_rhs,
                                        Scalar *upper, Scalar *rhs_projection,
                                        Scalar *solution, int *permutation,
                                        int *rank, int *local_ok) {
  if (threadIdx.x < kActiveWarpWidth) {
    const int lane = threadIdx.x;
    for (int index = lane; index < variables * variables;
         index += kActiveWarpWidth)
      upper[index] = Scalar{0};
    for (int variable = lane; variable < variables;
         variable += kActiveWarpWidth) {
      permutation[variable] = variable;
      rhs_projection[variable] = Scalar{0};
      solution[variable] = Scalar{0};
    }
    for (int row = lane; row < rows; row += kActiveWarpWidth)
      residual_rhs[row] = matrix[row * columns + variables];
    WarpSynchronize();

    int computed_rank = 0;
    const Scalar rank_threshold_squared = rank_tolerance * rank_tolerance;
    for (int basis = 0; basis < variables; ++basis) {
      int best_column = basis;
      Scalar best_norm_squared = Scalar{-1};
      for (int candidate = basis; candidate < variables; ++candidate) {
        Scalar norm_squared = Scalar{0};
        for (int row = lane; row < rows; row += kActiveWarpWidth) {
          const Scalar value = matrix[row * columns + candidate];
          norm_squared += value * value;
        }
        norm_squared = WarpSum(norm_squared);
        if (norm_squared > best_norm_squared) {
          best_norm_squared = norm_squared;
          best_column = candidate;
        }
      }
      if (!(best_norm_squared > rank_threshold_squared))
        break;
      if (best_column != basis) {
        for (int row = lane; row < rows; row += kActiveWarpWidth) {
          const Scalar value = matrix[row * columns + basis];
          matrix[row * columns + basis] = matrix[row * columns + best_column];
          matrix[row * columns + best_column] = value;
        }
        for (int previous = lane; previous < basis;
             previous += kActiveWarpWidth) {
          const Scalar value = upper[previous * variables + basis];
          upper[previous * variables + basis] =
              upper[previous * variables + best_column];
          upper[previous * variables + best_column] = value;
        }
        if (lane == 0) {
          const int variable = permutation[basis];
          permutation[basis] = permutation[best_column];
          permutation[best_column] = variable;
        }
        WarpSynchronize();
      }

      Scalar norm_squared = Scalar{0};
      for (int row = lane; row < rows; row += kActiveWarpWidth) {
        const Scalar value = matrix[row * columns + basis];
        norm_squared += value * value;
      }
      norm_squared = WarpSum(norm_squared);
      if (!(norm_squared > rank_threshold_squared))
        break;
      const Scalar norm = sqrt(norm_squared);
      if (lane == 0)
        upper[basis * variables + basis] = norm;
      for (int row = lane; row < rows; row += kActiveWarpWidth)
        matrix[row * columns + basis] /= norm;
      WarpSynchronize();

      for (int pass = 0; pass < 2; ++pass) {
        Scalar projection = Scalar{0};
        for (int row = lane; row < rows; row += kActiveWarpWidth)
          projection += matrix[row * columns + basis] * residual_rhs[row];
        projection = WarpSum(projection);
        if (lane == 0)
          rhs_projection[basis] += projection;
        for (int row = lane; row < rows; row += kActiveWarpWidth)
          residual_rhs[row] -= projection * matrix[row * columns + basis];
        WarpSynchronize();
      }
      for (int candidate = basis + 1; candidate < variables; ++candidate) {
        for (int pass = 0; pass < 2; ++pass) {
          Scalar projection = Scalar{0};
          for (int row = lane; row < rows; row += kActiveWarpWidth) {
            projection += matrix[row * columns + basis] *
                          matrix[row * columns + candidate];
          }
          projection = WarpSum(projection);
          if (lane == 0)
            upper[basis * variables + candidate] += projection;
          for (int row = lane; row < rows; row += kActiveWarpWidth) {
            matrix[row * columns + candidate] -=
                projection * matrix[row * columns + basis];
          }
          WarpSynchronize();
        }
      }
      ++computed_rank;
    }

    Scalar maximum_residual = Scalar{0};
    for (int row = lane; row < rows; row += kActiveWarpWidth)
      maximum_residual = fmax(maximum_residual, DeviceAbs(residual_rhs[row]));
    maximum_residual = WarpMaximum(maximum_residual);
    const bool okay = maximum_residual <= consistency_tolerance * rhs_scale;
    WarpSynchronize();
    if (lane == 0) {
      *rank = computed_rank;
      *local_ok = okay;
    }
    if (okay && lane == 0) {
      for (int reverse = 0; reverse < computed_rank; ++reverse) {
        const int row = computed_rank - 1 - reverse;
        Scalar value = rhs_projection[row];
        for (int col = row + 1; col < computed_rank; ++col)
          value -= upper[row * variables + col] * rhs_projection[col];
        rhs_projection[row] = value / upper[row * variables + row];
      }
      for (int variable = 0; variable < computed_rank; ++variable)
        solution[permutation[variable]] = rhs_projection[variable];
    }
  }
  WarpSynchronize();
}

__global__ void
BuildPrimalLeavesKernel(const PackedStage *stages, int stage_count,
                        const PackedTerminal *terminal_ptr,
                        Scalar rank_tolerance, Scalar consistency_tolerance,
                        Relation *leaves, DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index > stage_count)
    return;
  if (!BlockEnabled(status))
    return;
  const PackedTerminal &terminal = *terminal_ptr;
  const bool is_terminal = index == stage_count;
  const PackedStage *stage = is_terminal ? nullptr : &stages[index];
  const int rows = is_terminal ? terminal.state
                               : stage->mixed + stage->state + stage->next_n;
  const int columns =
      is_terminal ? terminal.n + 1 : stage->m + stage->n + stage->next_n + 1;
  const int eliminated = is_terminal ? 0 : stage->m;
  const int left_dim = is_terminal ? terminal.n : stage->n;
  const int right_dim = is_terminal ? 0 : stage->next_n;
  ScratchSize scratch_size;
  scratch_size.Add<Scalar>(static_cast<std::size_t>(rows) * columns);
  scratch_size.Add<Scalar>(rows);
  scratch_size.Add<int>(rows);
  scratch_size.Add<int>(rows);
  CLQR_BLOCK_SCRATCH(scratch, scratch_size.bytes);
  Scalar *matrix =
      scratch.Take<Scalar>(static_cast<std::size_t>(rows) * columns);
  Scalar *factors = scratch.Take<Scalar>(rows);
  int *pivot_columns = scratch.Take<int>(rows);
  int *pivot_rows = scratch.Take<int>(rows);
  __shared__ int rank;
  __shared__ int best_row;
  __shared__ int local_ok;

  for (int i = threadIdx.x; i < rows * columns; i += blockDim.x)
    matrix[i] = Scalar{0};
  WarpSynchronize();
  if (index == stage_count) {
    for (int linear = threadIdx.x; linear < terminal.state * terminal.n;
         linear += blockDim.x) {
      const int row = linear / terminal.n;
      const int col = linear % terminal.n;
      matrix[row * columns + col] = terminal.E[row * terminal.n + col];
    }
    for (int row = threadIdx.x; row < terminal.state; row += blockDim.x) {
      matrix[row * columns + terminal.n] = -terminal.e[row];
    }
  } else {
    const PackedStage &s = *stage;
    for (int linear = threadIdx.x; linear < s.mixed * s.m;
         linear += blockDim.x) {
      const int row = linear / s.m;
      const int col = linear % s.m;
      matrix[row * columns + col] = s.D[row * s.m + col];
    }
    for (int linear = threadIdx.x; linear < s.mixed * s.n;
         linear += blockDim.x) {
      const int row = linear / s.n;
      const int col = linear % s.n;
      matrix[row * columns + s.m + col] = s.C[row * s.n + col];
    }
    for (int row = threadIdx.x; row < s.mixed; row += blockDim.x) {
      matrix[row * columns + columns - 1] = -s.d[row];
    }
    for (int linear = threadIdx.x; linear < s.state * s.n;
         linear += blockDim.x) {
      const int row = linear / s.n;
      const int col = linear % s.n;
      matrix[(s.mixed + row) * columns + s.m + col] = s.E[row * s.n + col];
    }
    for (int row = threadIdx.x; row < s.state; row += blockDim.x) {
      matrix[(s.mixed + row) * columns + columns - 1] = -s.e[row];
    }
    const int dynamics_row = s.mixed + s.state;
    for (int linear = threadIdx.x; linear < s.next_n * s.m;
         linear += blockDim.x) {
      const int row = linear / s.m;
      const int col = linear % s.m;
      matrix[(dynamics_row + row) * columns + col] = -s.B[row * s.m + col];
    }
    for (int linear = threadIdx.x; linear < s.next_n * s.n;
         linear += blockDim.x) {
      const int row = linear / s.n;
      const int col = linear % s.n;
      matrix[(dynamics_row + row) * columns + s.m + col] =
          -s.A[row * s.n + col];
    }
    for (int row = threadIdx.x; row < s.next_n; row += blockDim.x) {
      matrix[(dynamics_row + row) * columns + s.m + s.n + row] = Scalar{1};
      matrix[(dynamics_row + row) * columns + columns - 1] = s.c[row];
    }
  }
  WarpSynchronize();
  RrefBlock(matrix, rows, columns, columns - 1, rank_tolerance, pivot_columns,
            pivot_rows, &rank, &best_row, factors);
  if (threadIdx.x == 0) {
    local_ok = !InconsistentRref(matrix, rows, columns, columns - 1,
                                 rank_tolerance, consistency_tolerance);
    if (!local_ok)
      SetFailure(status, kDeviceInfeasible, index, 1);
  }
  WarpSynchronize();
  if (!local_ok)
    return;
  ExtractResidualRelation(matrix, columns, rank, pivot_columns, eliminated,
                          left_dim, right_dim, &leaves[index]);
}

template <typename RelationType>
__device__ void
ComposeRelationsBlock(const RelationType &first, const RelationType &second,
                      Scalar rank_tolerance, Scalar consistency_tolerance,
                      RelationType *output, DeviceStatus *status, int stage,
                      int inconsistency_code, int inconsistency_detail,
                      Scalar *matrix, Scalar *factors, int *pivot_columns,
                      int *pivot_rows, int *rank, int *best_row, int *local_ok,
                      bool orthonormalize_output = false) {
  if (first.right_dim != second.left_dim) {
    if (threadIdx.x == 0)
      SetFailure(status, kDeviceNumericalFailure, stage, 2);
    return;
  }
  const int shared = first.right_dim;
  const int rows = first.rows + second.rows;
  const int columns = shared + first.left_dim + second.right_dim + 1;
  for (int i = threadIdx.x; i < rows * columns; i += blockDim.x)
    matrix[i] = Scalar{0};
  WarpSynchronize();
  for (int linear = threadIdx.x; linear < first.rows * shared;
       linear += blockDim.x) {
    const int row = linear / shared;
    const int col = linear % shared;
    matrix[row * columns + col] = first.right[row * first.right_dim + col];
  }
  for (int linear = threadIdx.x; linear < first.rows * first.left_dim;
       linear += blockDim.x) {
    const int row = linear / first.left_dim;
    const int col = linear % first.left_dim;
    matrix[row * columns + shared + col] =
        first.left[row * first.left_dim + col];
  }
  for (int row = threadIdx.x; row < first.rows; row += blockDim.x) {
    matrix[row * columns + columns - 1] = first.rhs[row];
  }
  for (int linear = threadIdx.x; linear < second.rows * shared;
       linear += blockDim.x) {
    const int row = linear / shared;
    const int col = linear % shared;
    matrix[(first.rows + row) * columns + col] =
        second.left[row * second.left_dim + col];
  }
  for (int linear = threadIdx.x; linear < second.rows * second.right_dim;
       linear += blockDim.x) {
    const int row = linear / second.right_dim;
    const int col = linear % second.right_dim;
    matrix[(first.rows + row) * columns + shared + first.left_dim + col] =
        second.right[row * second.right_dim + col];
  }
  for (int row = threadIdx.x; row < second.rows; row += blockDim.x) {
    matrix[(first.rows + row) * columns + columns - 1] = second.rhs[row];
  }
  WarpSynchronize();
  if (orthonormalize_output) {
    EliminateRelationOrthogonally(matrix, rows, columns, shared, first.left_dim,
                                  second.right_dim, rank_tolerance,
                                  consistency_tolerance, output, local_ok);
    if (threadIdx.x == 0 && !*local_ok)
      SetFailure(status, inconsistency_code, stage, inconsistency_detail);
    WarpSynchronize();
    return;
  }
  RrefBlock(matrix, rows, columns, columns - 1, rank_tolerance, pivot_columns,
            pivot_rows, rank, best_row, factors);
  if (threadIdx.x == 0) {
    *local_ok = !InconsistentRref(matrix, rows, columns, columns - 1,
                                  rank_tolerance, consistency_tolerance);
    if (!*local_ok)
      SetFailure(status, inconsistency_code, stage, inconsistency_detail);
  }
  WarpSynchronize();
  if (!*local_ok)
    return;
  ExtractResidualRelation(matrix, columns, *rank, pivot_columns, shared,
                          first.left_dim, second.right_dim, output);
}

template <typename RelationType>
__device__ void CopyRelationBlock(const RelationType &input,
                                  RelationType *output) {
  if (threadIdx.x == 0) {
    output->left_dim = input.left_dim;
    output->right_dim = input.right_dim;
    output->rows = input.rows;
  }
  for (int linear = threadIdx.x; linear < input.rows * input.left_dim;
       linear += blockDim.x) {
    const int row = linear / input.left_dim;
    const int col = linear % input.left_dim;
    output->left[row * input.left_dim + col] =
        input.left[row * input.left_dim + col];
  }
  for (int linear = threadIdx.x; linear < input.rows * input.right_dim;
       linear += blockDim.x) {
    const int row = linear / input.right_dim;
    const int col = linear % input.right_dim;
    output->right[row * input.right_dim + col] =
        input.right[row * input.right_dim + col];
  }
  for (int row = threadIdx.x; row < input.rows; row += blockDim.x)
    output->rhs[row] = input.rhs[row];
}

__device__ bool InvalidScanRelation(const Relation &relation) {
  return relation.left_dim < 0;
}

__device__ void SetInvalidScanRelation(Relation *relation) {
  if (threadIdx.x == 0) {
    relation->left_dim = -1;
    relation->right_dim = 0;
    relation->rows = 0;
  }
}

__device__ void
ComposeScanRelationBlock(const Relation &first, const Relation &second,
                         Scalar rank_tolerance, Scalar consistency_tolerance,
                         Relation *output, DeviceStatus *status, int stage,
                         int inconsistency_detail, Scalar *matrix,
                         Scalar *factors, int *pivot_columns, int *pivot_rows,
                         int *rank, int *best_row, int *local_ok) {
  if (InvalidScanRelation(first)) {
    if (InvalidScanRelation(second)) {
      SetInvalidScanRelation(output);
    } else {
      CopyRelationBlock(second, output);
    }
    return;
  }
  if (InvalidScanRelation(second)) {
    CopyRelationBlock(first, output);
    return;
  }
  ComposeRelationsBlock(first, second, rank_tolerance, consistency_tolerance,
                        output, status, stage, kDeviceInfeasible,
                        inconsistency_detail, matrix, factors, pivot_columns,
                        pivot_rows, rank, best_row, local_ok);
}

__global__ void
ReduceRelationLeavesKernel(const Relation *leaves, int count, int parent_count,
                           Scalar rank_tolerance, Scalar consistency_tolerance,
                           Relation *parents, DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index >= parent_count)
    return;
  if (!BlockEnabled(status))
    return;
  const int left = 2 * index;
  if (left + 1 >= count) {
    CopyRelationBlock(leaves[left], &parents[index]);
    return;
  }
  const Relation &first = leaves[left];
  const Relation &second = leaves[left + 1];
  const int rows = first.rows + second.rows;
  const int columns = first.right_dim + first.left_dim + second.right_dim + 1;
  ScratchSize scratch_size;
  scratch_size.Add<Scalar>(static_cast<std::size_t>(rows) * columns);
  scratch_size.Add<Scalar>(rows);
  scratch_size.Add<int>(rows);
  scratch_size.Add<int>(rows);
  CLQR_BLOCK_SCRATCH(scratch, scratch_size.bytes);
  Scalar *matrix =
      scratch.Take<Scalar>(static_cast<std::size_t>(rows) * columns);
  Scalar *factors = scratch.Take<Scalar>(rows);
  int *pivot_columns = scratch.Take<int>(rows);
  int *pivot_rows = scratch.Take<int>(rows);
  __shared__ int rank;
  __shared__ int best_row;
  __shared__ int local_ok;
  ComposeRelationsBlock(first, second, rank_tolerance, consistency_tolerance,
                        &parents[index], status, index, kDeviceInfeasible, 19,
                        matrix, factors, pivot_columns, pivot_rows, &rank,
                        &best_row, &local_ok);
}

__global__ void ReduceRelationTreeLevelKernel(Relation *tree, int child_offset,
                                              int parent_offset,
                                              int child_count, int parent_count,
                                              Scalar rank_tolerance,
                                              Scalar consistency_tolerance,
                                              DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index >= parent_count)
    return;
  if (!BlockEnabled(status))
    return;
  const int left = child_offset + 2 * index;
  if (2 * index + 1 >= child_count) {
    CopyRelationBlock(tree[left], &tree[parent_offset + index]);
    return;
  }
  const int right = left + 1;
  const Relation &first = tree[left];
  const Relation &second = tree[right];
  const int rows = first.rows + second.rows;
  const int columns = first.right_dim + first.left_dim + second.right_dim + 1;
  ScratchSize scratch_size;
  scratch_size.Add<Scalar>(static_cast<std::size_t>(rows) * columns);
  scratch_size.Add<Scalar>(rows);
  scratch_size.Add<int>(rows);
  scratch_size.Add<int>(rows);
  CLQR_BLOCK_SCRATCH(scratch, scratch_size.bytes);
  Scalar *matrix =
      scratch.Take<Scalar>(static_cast<std::size_t>(rows) * columns);
  Scalar *factors = scratch.Take<Scalar>(rows);
  int *pivot_columns = scratch.Take<int>(rows);
  int *pivot_rows = scratch.Take<int>(rows);
  __shared__ int rank;
  __shared__ int best_row;
  __shared__ int local_ok;
  ComposeRelationsBlock(first, second, rank_tolerance, consistency_tolerance,
                        &tree[parent_offset + index], status, index,
                        kDeviceInfeasible, 19, matrix, factors, pivot_columns,
                        pivot_rows, &rank, &best_row, &local_ok);
}

__global__ void InitializeRelationContextRootKernel(Relation *tree,
                                                    int root_offset) {
  if (blockIdx.x == 0)
    SetInvalidScanRelation(&tree[root_offset]);
}

__global__ void ExpandRelationContextLevelKernel(
    Relation *tree, int child_offset, int parent_offset, int child_count,
    int parent_count, Scalar rank_tolerance, Scalar consistency_tolerance,
    DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index >= parent_count)
    return;
  if (!BlockEnabled(status))
    return;
  const int left = child_offset + 2 * index;
  const Relation &parent_context = tree[parent_offset + index];
  if (2 * index + 1 >= child_count) {
    CopyRelationBlock(parent_context, &tree[left]);
    return;
  }
  const int right = left + 1;
  const Relation &first = tree[right];
  const Relation &second = parent_context;
  const int rows = first.rows + second.rows;
  const int columns = first.right_dim + first.left_dim + second.right_dim + 1;
  ScratchSize scratch_size;
  scratch_size.Add<Scalar>(static_cast<std::size_t>(rows) * columns);
  scratch_size.Add<Scalar>(rows);
  scratch_size.Add<int>(rows);
  scratch_size.Add<int>(rows);
  CLQR_BLOCK_SCRATCH(scratch, scratch_size.bytes);
  Scalar *matrix =
      scratch.Take<Scalar>(static_cast<std::size_t>(rows) * columns);
  Scalar *factors = scratch.Take<Scalar>(rows);
  int *pivot_columns = scratch.Take<int>(rows);
  int *pivot_rows = scratch.Take<int>(rows);
  __shared__ int rank;
  __shared__ int best_row;
  __shared__ int local_ok;
  ComposeScanRelationBlock(tree[right], parent_context, rank_tolerance,
                           consistency_tolerance, &tree[left], status, index,
                           20, matrix, factors, pivot_columns, pivot_rows,
                           &rank, &best_row, &local_ok);
  WarpSynchronize();
  CopyRelationBlock(parent_context, &tree[right]);
}

__global__ void FinalizeRelationSuffixFromParentsKernel(
    Relation *leaves, int count, const Relation *parent_contexts,
    int parent_count, Scalar rank_tolerance, Scalar consistency_tolerance,
    DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index >= parent_count)
    return;
  if (!BlockEnabled(status))
    return;
  const int left = 2 * index;
  const Relation &parent = parent_contexts[index];
  const int right = left + 1;
  int scratch_left = leaves[left].left_dim;
  int scratch_right = leaves[left].right_dim;
  int matrix_rows = leaves[left].rows;
  int matrix_columns = leaves[left].left_dim + leaves[left].right_dim + 1;
  if (left + 1 < count) {
    scratch_left = DeviceMax(scratch_left, leaves[right].left_dim);
    scratch_right = DeviceMax(scratch_right, leaves[right].right_dim);
    matrix_rows =
        DeviceMax(matrix_rows, leaves[left].rows + leaves[right].rows);
    matrix_columns = DeviceMax(matrix_columns, leaves[left].right_dim +
                                                   leaves[left].left_dim +
                                                   leaves[right].right_dim + 1);
  }
  if (!InvalidScanRelation(parent)) {
    scratch_right = DeviceMax(scratch_right, parent.right_dim);
    const Relation &child = left + 1 < count ? leaves[right] : leaves[left];
    matrix_rows = DeviceMax(matrix_rows, child.rows + parent.rows);
    matrix_columns =
        DeviceMax(matrix_columns,
                  child.right_dim + child.left_dim + parent.right_dim + 1);
  }
  const int relation_rows = scratch_left + scratch_right;
  const std::size_t relation_entries =
      static_cast<std::size_t>(relation_rows) * (scratch_left + scratch_right) +
      relation_rows;
  ScratchSize scratch_size;
  scratch_size.Add<Relation>(1);
  scratch_size.Add<Scalar>(relation_entries);
  scratch_size.Add<Scalar>(static_cast<std::size_t>(matrix_rows) *
                           matrix_columns);
  scratch_size.Add<Scalar>(matrix_rows);
  scratch_size.Add<int>(matrix_rows);
  scratch_size.Add<int>(matrix_rows);
  CLQR_BLOCK_SCRATCH(scratch, scratch_size.bytes);
  Relation *composed_ptr = scratch.Take<Relation>(1);
  Scalar *composed_storage = scratch.Take<Scalar>(relation_entries);
  Scalar *matrix = scratch.Take<Scalar>(static_cast<std::size_t>(matrix_rows) *
                                        matrix_columns);
  Scalar *factors = scratch.Take<Scalar>(matrix_rows);
  int *pivot_columns = scratch.Take<int>(matrix_rows);
  int *pivot_rows = scratch.Take<int>(matrix_rows);
  Relation &composed = *composed_ptr;
  __shared__ int rank;
  __shared__ int best_row;
  __shared__ int local_ok;
  if (threadIdx.x == 0)
    BindRelationScratch(&composed, composed_storage, scratch_left,
                        scratch_right);
  WarpSynchronize();
  if (left + 1 >= count) {
    if (!InvalidScanRelation(parent)) {
      ComposeScanRelationBlock(leaves[left], parent, rank_tolerance,
                               consistency_tolerance, &composed, status, left,
                               21, matrix, factors, pivot_columns, pivot_rows,
                               &rank, &best_row, &local_ok);
      WarpSynchronize();
      CopyRelationBlock(composed, &leaves[left]);
    }
    return;
  }
  if (!InvalidScanRelation(parent)) {
    ComposeScanRelationBlock(leaves[right], parent, rank_tolerance,
                             consistency_tolerance, &composed, status, right,
                             21, matrix, factors, pivot_columns, pivot_rows,
                             &rank, &best_row, &local_ok);
    WarpSynchronize();
    CopyRelationBlock(composed, &leaves[right]);
    WarpSynchronize();
  }
  ComposeRelationsBlock(leaves[left], leaves[right], rank_tolerance,
                        consistency_tolerance, &composed, status, left,
                        kDeviceInfeasible, 21, matrix, factors, pivot_columns,
                        pivot_rows, &rank, &best_row, &local_ok);
  WarpSynchronize();
  CopyRelationBlock(composed, &leaves[left]);
}

__global__ void StateParamKernel(const Relation *suffix, int count,
                                 StateParam *params, int *state_dimensions,
                                 DeviceStatus *status, Scalar tolerance) {
  const int index = blockIdx.x;
  if (index >= count || threadIdx.x != 0 || status->code != kDeviceOk)
    return;
  const Relation &relation = suffix[index];
  if (relation.right_dim != 0 || relation.rows > relation.left_dim) {
    SetFailure(status, kDeviceNumericalFailure, index, 4);
    return;
  }
  StateParam &out = params[index];
  out.physical_dim = relation.left_dim;
  for (int row = 0; row < relation.left_dim; ++row) {
    out.t[row] = Scalar{0};
  }
  ScratchSize scratch_size;
  scratch_size.Add<int>(relation.left_dim);
  CLQR_BLOCK_SCRATCH(scratch, scratch_size.bytes);
  int *pivot_row = scratch.Take<int>(relation.left_dim);
  for (int i = 0; i < relation.left_dim; ++i)
    pivot_row[i] = -1;
  for (int row = 0; row < relation.rows; ++row) {
    int column = -1;
    for (int col = 0; col < relation.left_dim; ++col) {
      if (DeviceAbs(relation.left[row * relation.left_dim + col]) > tolerance) {
        column = col;
        break;
      }
    }
    if (column < 0 || pivot_row[column] >= 0) {
      SetFailure(status, kDeviceNumericalFailure, index, 5);
      return;
    }
    pivot_row[column] = row;
  }
  int reduced = 0;
  for (int col = 0; col < relation.left_dim; ++col) {
    if (pivot_row[col] < 0)
      out.free_columns[reduced++] = col;
  }
  out.reduced_dim = reduced;
  for (int row = 0; row < relation.left_dim; ++row)
    for (int col = 0; col < reduced; ++col)
      out.T[row * reduced + col] = Scalar{0};
  for (int col = 0; col < relation.left_dim; ++col) {
    if (pivot_row[col] >= 0) {
      const int row = pivot_row[col];
      const Scalar diagonal = relation.left[row * relation.left_dim + col];
      out.t[col] = relation.rhs[row] / diagonal;
      for (int free = 0; free < reduced; ++free) {
        out.T[col * reduced + free] =
            -relation.left[row * relation.left_dim + out.free_columns[free]] /
            diagonal;
      }
    }
  }
  for (int free = 0; free < reduced; ++free) {
    out.T[out.free_columns[free] * reduced + free] = Scalar{1};
  }
  if (state_dimensions != nullptr) {
    state_dimensions[2 * index] = out.physical_dim;
    state_dimensions[2 * index + 1] = out.reduced_dim;
  }
}

} // namespace
} // namespace detail
} // namespace cuda
} // namespace clqr

#ifndef CLQR_CUDA_EMULATION
namespace clqr {
namespace cuda {
namespace detail {
namespace {

__global__ void ReduceStagesKernel(const PackedStage *, const Relation *,
                                   const StateParam *, int, Scalar, Scalar,
                                   ControlParam *, ReducedStage *, int *,
                                   DeviceStatus *);
__global__ void ReduceTerminalKernel(const PackedTerminal *, const StateParam *,
                                     int, ReducedTerminal *);
__global__ void InitialReducedStateKernel(const StateParam *, const Scalar *,
                                          Scalar *, Scalar, DeviceStatus *);
__global__ void BuildValueElementsKernel(const ReducedStage *,
                                         const ReducedTerminal *, int, Scalar,
                                         ValueElement *, DeviceStatus *);
__global__ void ReduceValueLeavesKernel(const ValueElement *, int, int, Scalar,
                                        DeviceStatus *, ValueElement *);
__global__ void ReduceValueTreeLevelKernel(ValueElement *, int, int, int, int,
                                           Scalar, DeviceStatus *);
__global__ void InitializeValueContextRootKernel(ValueElement *, int);
__global__ void ExpandValueContextLevelKernel(ValueElement *, int, int, int,
                                              int, Scalar, DeviceStatus *);
__global__ void FinalizeValueSuffixFromParentsKernel(ValueElement *, int,
                                                     const ValueElement *, int,
                                                     Scalar, DeviceStatus *);
__global__ void FeedbackKernel(const ReducedStage *, const ValueElement *, int,
                               Scalar, Feedback *, DeviceStatus *);
__global__ void InitializeAffineMapsKernel(const Feedback *, int, AffineMap *);
__global__ void ReduceAffineLeavesKernel(const AffineMap *, int, int,
                                         AffineMap *, DeviceStatus *);
__global__ void ReduceAffineTreeLevelKernel(AffineMap *, int, int, int, int,
                                            DeviceStatus *);
__global__ void InitializeAffineContextRootKernel(AffineMap *, int);
__global__ void ExpandAffineContextLevelKernel(AffineMap *, int, int, int, int,
                                               DeviceStatus *);
__global__ void FinalizeAffinePrefixFromParentsKernel(AffineMap *, int,
                                                      const AffineMap *, int,
                                                      DeviceStatus *);
__global__ void ReconstructPrimalKernel(const AffineMap *, const StateParam *,
                                        const ControlParam *, const Feedback *,
                                        const Scalar *, const int *,
                                        const int *, const int *, int, Scalar *,
                                        Scalar *, Scalar *, Scalar *);
__global__ void BuildDualParametersKernel(const PackedStage *,
                                          const StateParam *,
                                          const ValueElement *, const Scalar *,
                                          const Scalar *, const Scalar *,
                                          const int *, const int *, const int *,
                                          int, Scalar, Scalar, DualParam *,
                                          int *, int *, DeviceStatus *);
__global__ void BuildDualParameterRelationsKernel(
    const PackedStage *, const PackedTerminal *, const DualParam *, int,
    const Scalar *, const Scalar *, const int *, const int *, Scalar, Scalar,
    DualRelation *, const int *, StateDualParam *, DeviceStatus *);
__global__ void
RecoverParameterizedMultipliersKernel(const DualParam *, const StateDualParam *,
                                      const DualNodeValue *, const int *,
                                      const int *, const int *, int, Scalar *,
                                      Scalar *, Scalar *, Scalar *);
__global__ void RecoverInitialMultiplierKernel(
    const PackedStage *, const PackedTerminal *, int, const Scalar *,
    const Scalar *, const Scalar *, const Scalar *, const int *, const int *,
    const int *, const int *, const int *, Scalar *, Scalar *, Scalar *);
__global__ void ReduceDualTreeLevelKernel(const DualRelation *, int, int, int,
                                          int, Scalar, Scalar, DualRelation *,
                                          const int *, DeviceStatus *);
__global__ void SolveDualRootKernel(const DualRelation *, DualNodeValue *,
                                    const int *, DeviceStatus *, Scalar);
__global__ void ExpandDualTreeLevelKernel(const DualRelation *, int, int, int,
                                          int, Scalar, Scalar,
                                          const DualNodeValue *,
                                          DualNodeValue *, const int *,
                                          DeviceStatus *);
void CudaCheck(cudaError_t error, const char *operation) {
  if (error == cudaSuccess)
    return;
  throw std::runtime_error(std::string(operation) + ": " +
                           cudaGetErrorString(error));
}

template <typename T> class DeviceBuffer {
public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) { Allocate(count); }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  DeviceBuffer(DeviceBuffer &&other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        count_(std::exchange(other.count_, 0)) {}
  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
    if (this != &other) {
      Release();
      data_ = std::exchange(other.data_, nullptr);
      count_ = std::exchange(other.count_, 0);
    }
    return *this;
  }
  ~DeviceBuffer() { Release(); }

  void Allocate(std::size_t count) {
    Release();
    count_ = std::max<std::size_t>(count, 1);
    if (count_ > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::invalid_argument("CUDA device allocation size overflows");
    CudaCheck(cudaMalloc(reinterpret_cast<void **>(&data_), count_ * sizeof(T)),
              "cudaMalloc");
  }
  void Reserve(std::size_t count) {
    const std::size_t required = std::max<std::size_t>(count, 1);
    if (count_ < required)
      Allocate(required);
  }
  void Release() {
    if (data_ != nullptr)
      cudaFree(data_);
    data_ = nullptr;
    count_ = 0;
  }
  T *get() { return data_; }
  const T *get() const { return data_; }
  std::size_t count() const { return count_; }

private:
  T *data_ = nullptr;
  std::size_t count_ = 0;
};

template <typename T> class PinnedBuffer {
public:
  PinnedBuffer() = default;
  PinnedBuffer(const PinnedBuffer &) = delete;
  PinnedBuffer &operator=(const PinnedBuffer &) = delete;
  ~PinnedBuffer() { Release(); }

  void Reserve(std::size_t count) {
    const std::size_t required = std::max<std::size_t>(count, 1);
    if (capacity_ >= required)
      return;
    if (required > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::invalid_argument("CUDA pinned allocation size overflows");
    Release();
    CudaCheck(
        cudaMallocHost(reinterpret_cast<void **>(&data_), required * sizeof(T)),
        "cudaMallocHost");
    capacity_ = required;
  }
  void Resize(std::size_t count) {
    Reserve(count);
    size_ = count;
  }
  void Release() {
    if (data_ != nullptr)
      cudaFreeHost(data_);
    data_ = nullptr;
    size_ = 0;
    capacity_ = 0;
  }
  T *data() { return data_; }
  const T *data() const { return data_; }
  T *begin() { return data_; }
  T *end() { return data_ + size_; }
  T &operator[](std::size_t index) { return data_[index]; }
  const T &operator[](std::size_t index) const { return data_[index]; }
  std::size_t size() const { return size_; }

private:
  T *data_ = nullptr;
  std::size_t size_ = 0;
  std::size_t capacity_ = 0;
};

struct WorkspaceStorage {
  int device = -1;
  cudaEvent_t event_start = nullptr;
  cudaEvent_t event_stop = nullptr;
  DeviceBuffer<PackedStage> device_stages;
  DeviceBuffer<PackedTerminal> device_terminal;
  DeviceBuffer<Scalar> device_problem_data;
  DeviceBuffer<Scalar> device_initial;
  DeviceBuffer<DeviceStatus> device_status;
  DeviceBuffer<Relation> relation_leaves;
  DeviceBuffer<Relation> relation_scan;
  DeviceBuffer<Scalar> relation_data;
  DeviceBuffer<StateParam> state_params;
  DeviceBuffer<ControlParam> control_params;
  DeviceBuffer<ReducedStage> reduced_stages;
  DeviceBuffer<ReducedTerminal> reduced_terminal;
  DeviceBuffer<Scalar> stage_data;
  DeviceBuffer<int> stage_indices;
  DeviceBuffer<Scalar> reduced_initial;
  DeviceBuffer<ValueElement> value_leaves;
  DeviceBuffer<ValueElement> value_scan;
  DeviceBuffer<Scalar> value_data;
  DeviceBuffer<Feedback> feedback;
  DeviceBuffer<AffineMap> map_leaves;
  DeviceBuffer<AffineMap> map_scan;
  DeviceBuffer<Scalar> map_data;
  DeviceBuffer<Scalar> reduced_states;
  DeviceBuffer<Scalar> reduced_controls;
  DeviceBuffer<Scalar> states;
  DeviceBuffer<Scalar> controls;
  DeviceBuffer<int> state_offsets;
  DeviceBuffer<int> reduced_state_offsets;
  DeviceBuffer<int> control_offsets;
  DeviceBuffer<int> dynamics_offsets;
  DeviceBuffer<int> mixed_offsets;
  DeviceBuffer<int> state_constraint_offsets;
  DeviceBuffer<int> state_dimensions;
  DeviceBuffer<int> control_dimensions;
  DeviceBuffer<DualParam> dual_params;
  DeviceBuffer<int> dual_dimensions;
  DeviceBuffer<int> dual_scan_needed;
  DeviceBuffer<StateDualParam> state_dual_params;
  DeviceBuffer<DualRelation> dual_tree;
  DeviceBuffer<DualNodeValue> dual_values;
  DeviceBuffer<Scalar> dual_relation_data;
  DeviceBuffer<Scalar> dual_value_data;
  DeviceBuffer<Scalar> initial_multiplier;
  DeviceBuffer<Scalar> dynamics_multipliers;
  DeviceBuffer<Scalar> mixed_multipliers;
  DeviceBuffer<Scalar> state_multipliers;
  DeviceBuffer<Scalar> terminal_multiplier;

  PinnedBuffer<PackedStage> host_stages;
  PinnedBuffer<PackedTerminal> host_terminal;
  PinnedBuffer<Scalar> host_problem_data;
  PinnedBuffer<Scalar> host_initial;
  PinnedBuffer<Scalar> host_states;
  PinnedBuffer<Scalar> host_controls;
  PinnedBuffer<Scalar> host_initial_multiplier;
  PinnedBuffer<Scalar> host_dynamics;
  PinnedBuffer<Scalar> host_mixed;
  PinnedBuffer<Scalar> host_state_multipliers;
  PinnedBuffer<Scalar> host_terminal_multiplier;
  PinnedBuffer<int> host_state_offsets;
  PinnedBuffer<int> host_reduced_state_offsets;
  PinnedBuffer<int> host_control_offsets;
  PinnedBuffer<int> host_dynamics_offsets;
  PinnedBuffer<int> host_mixed_offsets;
  PinnedBuffer<int> host_state_constraint_offsets;
  PinnedBuffer<int> host_state_dimensions;
  PinnedBuffer<int> host_control_dimensions;
  PinnedBuffer<Relation> host_relation_leaves;
  PinnedBuffer<Relation> host_relation_scan;
  PinnedBuffer<StateParam> host_state_params;
  PinnedBuffer<ControlParam> host_control_params;
  PinnedBuffer<ReducedStage> host_reduced_stages;
  PinnedBuffer<ReducedTerminal> host_reduced_terminal;
  PinnedBuffer<Feedback> host_feedback;
  PinnedBuffer<DualParam> host_dual_params;
  PinnedBuffer<StateDualParam> host_state_dual_params;
  PinnedBuffer<ValueElement> host_value_leaves;
  PinnedBuffer<ValueElement> host_value_scan;
  PinnedBuffer<AffineMap> host_map_leaves;
  PinnedBuffer<AffineMap> host_map_scan;
  PinnedBuffer<int> host_dual_scan_needed;
  PinnedBuffer<int> host_dual_dimensions;
  PinnedBuffer<DualRelation> host_dual_tree;
  PinnedBuffer<DualNodeValue> host_dual_values;
  PinnedBuffer<DeviceStatus> host_status;
  std::vector<int> node_level_offsets;
  std::vector<int> node_level_counts;
  std::vector<int> stage_level_offsets;
  std::vector<int> stage_level_counts;
  std::vector<int> relation_layout_key;
  std::vector<int> stage_layout_key;
  bool stage_layout_uploaded = false;
  std::vector<int> value_layout_key;
  std::vector<int> map_layout_key;
  std::vector<int> dual_layout_key;
  std::vector<int> structure_key;
  ScratchRequirements scratch;
  bool structure_ready = false;
  bool relation_layout_uploaded = false;

  ~WorkspaceStorage() {
    if (device >= 0)
      cudaSetDevice(device);
    if (event_stop != nullptr)
      cudaEventDestroy(event_stop);
    if (event_start != nullptr)
      cudaEventDestroy(event_start);
  }

  void Reserve(int requested_device, int stage_count, int node_count,
               std::size_t state_entries, std::size_t control_entries,
               std::size_t mixed_entries, std::size_t state_constraint_entries,
               std::size_t problem_data_entries, int initial_state_dimension,
               int terminal_constraint_dimension) {
    if (device >= 0 && device != requested_device) {
      throw std::invalid_argument(
          "a CUDA workspace cannot be reused across devices");
    }
    device = requested_device;
    if (event_start == nullptr)
      CudaCheck(cudaEventCreate(&event_start), "cudaEventCreate(start)");
    if (event_stop == nullptr)
      CudaCheck(cudaEventCreate(&event_stop), "cudaEventCreate(stop)");
    int total_tree_nodes = 0;
    for (int level_count = node_count;; level_count = (level_count + 1) / 2) {
      total_tree_nodes += level_count;
      if (level_count == 1)
        break;
    }
    int total_stage_tree_nodes = 0;
    for (int level_count = std::max(stage_count, 1);;
         level_count = (level_count + 1) / 2) {
      total_stage_tree_nodes += level_count;
      if (level_count == 1)
        break;
    }
    device_stages.Reserve(stage_count);
    device_terminal.Reserve(1);
    device_problem_data.Reserve(problem_data_entries);
    device_initial.Reserve(initial_state_dimension);
    device_status.Reserve(1);
    relation_leaves.Reserve(node_count);
    relation_scan.Reserve(total_tree_nodes - node_count);
    state_params.Reserve(node_count);
    control_params.Reserve(stage_count);
    reduced_stages.Reserve(stage_count);
    reduced_terminal.Reserve(1);
    reduced_initial.Reserve(initial_state_dimension);
    value_leaves.Reserve(node_count);
    value_scan.Reserve(total_tree_nodes - node_count);
    feedback.Reserve(stage_count);
    map_leaves.Reserve(stage_count);
    map_scan.Reserve(total_stage_tree_nodes - std::max(stage_count, 1));
    states.Reserve(state_entries);
    reduced_controls.Reserve(control_entries);
    controls.Reserve(control_entries);
    state_offsets.Reserve(node_count + 1);
    reduced_state_offsets.Reserve(node_count + 1);
    control_offsets.Reserve(stage_count + 1);
    dynamics_offsets.Reserve(stage_count + 1);
    mixed_offsets.Reserve(stage_count + 1);
    state_constraint_offsets.Reserve(stage_count + 1);
    state_dimensions.Reserve(static_cast<std::size_t>(2) * node_count);
    control_dimensions.Reserve(static_cast<std::size_t>(2) * stage_count);
    dual_params.Reserve(stage_count);
    dual_dimensions.Reserve(stage_count);
    dual_scan_needed.Reserve(1);
    state_dual_params.Reserve(stage_count);
    initial_multiplier.Reserve(initial_state_dimension);
    dynamics_multipliers.Reserve(state_entries - initial_state_dimension);
    mixed_multipliers.Reserve(mixed_entries);
    state_multipliers.Reserve(state_constraint_entries);
    terminal_multiplier.Reserve(terminal_constraint_dimension);

    host_stages.Resize(stage_count);
    host_terminal.Resize(1);
    host_problem_data.Resize(problem_data_entries);
    host_initial.Resize(initial_state_dimension);
    host_states.Resize(state_entries);
    host_controls.Resize(control_entries);
    host_initial_multiplier.Resize(initial_state_dimension);
    host_dynamics.Resize(state_entries - initial_state_dimension);
    host_mixed.Resize(mixed_entries);
    host_state_multipliers.Resize(state_constraint_entries);
    host_terminal_multiplier.Resize(terminal_constraint_dimension);
    host_state_offsets.Resize(node_count + 1);
    host_reduced_state_offsets.Resize(node_count + 1);
    host_control_offsets.Resize(stage_count + 1);
    host_dynamics_offsets.Resize(stage_count + 1);
    host_mixed_offsets.Resize(stage_count + 1);
    host_state_constraint_offsets.Resize(stage_count + 1);
    host_state_dimensions.Resize(static_cast<std::size_t>(2) * node_count);
    host_control_dimensions.Resize(static_cast<std::size_t>(2) * stage_count);
    host_relation_leaves.Resize(node_count);
    host_relation_scan.Resize(total_tree_nodes - node_count);
    host_state_params.Resize(node_count);
    host_control_params.Resize(stage_count);
    host_reduced_stages.Resize(stage_count);
    host_reduced_terminal.Resize(1);
    host_feedback.Resize(stage_count);
    host_dual_params.Resize(stage_count);
    host_state_dual_params.Resize(stage_count);
    host_value_leaves.Resize(node_count);
    host_value_scan.Resize(total_tree_nodes - node_count);
    host_map_leaves.Resize(stage_count);
    host_map_scan.Resize(total_stage_tree_nodes - std::max(stage_count, 1));
    host_dual_scan_needed.Resize(1);
    host_dual_dimensions.Resize(stage_count);
    host_status.Resize(1);
  }
};

template <typename Function>
double TimeGpu(WorkspaceStorage &workspace, Function &&function) {
  CudaCheck(cudaEventRecord(workspace.event_start), "cudaEventRecord(start)");
  function();
  CudaCheck(cudaGetLastError(), "CUDA kernel launch");
  CudaCheck(cudaEventRecord(workspace.event_stop), "cudaEventRecord(stop)");
  CudaCheck(cudaEventSynchronize(workspace.event_stop), "cudaEventSynchronize");
  float milliseconds = 0.0f;
  CudaCheck(cudaEventElapsedTime(&milliseconds, workspace.event_start,
                                 workspace.event_stop),
            "cudaEventElapsedTime");
  return milliseconds;
}

// Queue setup transfers before the start event and result/control transfers
// after the stop event, then synchronize the whole default stream once.  The
// returned duration therefore measures only the kernels in `function`, while
// preserving the same single synchronization needed before the host consumes
// `after`'s results.
template <typename Before, typename Function, typename After>
double TimeGpuKernels(WorkspaceStorage &workspace, Before &&before,
                      Function &&function, After &&after) {
  before();
  CudaCheck(cudaGetLastError(), "CUDA phase setup");
  CudaCheck(cudaEventRecord(workspace.event_start), "cudaEventRecord(start)");
  function();
  CudaCheck(cudaGetLastError(), "CUDA kernel launch");
  CudaCheck(cudaEventRecord(workspace.event_stop), "cudaEventRecord(stop)");
  after();
  CudaCheck(cudaGetLastError(), "CUDA phase result transfer");
  CudaCheck(cudaStreamSynchronize(nullptr), "cudaStreamSynchronize");
  float milliseconds = 0.0f;
  CudaCheck(cudaEventElapsedTime(&milliseconds, workspace.event_start,
                                 workspace.event_stop),
            "cudaEventElapsedTime");
  return milliseconds;
}

bool Finite(const Matrix &matrix) {
  for (Scalar value : matrix.data()) {
    if (!std::isfinite(value))
      return false;
  }
  return true;
}

bool Finite(const Vector &vector) {
  for (Scalar value : vector.data()) {
    if (!std::isfinite(value))
      return false;
  }
  return true;
}

void Require(bool condition, const std::string &message) {
  if (!condition)
    throw std::invalid_argument(message);
}

void RequireAtStage(bool condition, const char *message, std::size_t stage) {
  if (!condition) {
    throw std::invalid_argument(std::string(message) + " at stage " +
                                std::to_string(stage));
  }
}

std::size_t CheckedProduct(std::size_t first, std::size_t second,
                           const char *description) {
  if (first != 0 && second > std::numeric_limits<std::size_t>::max() / first) {
    throw std::invalid_argument(std::string("CUDA ") + description +
                                " size overflows size_t");
  }
  return first * second;
}

void CheckedAccumulate(std::size_t count, std::size_t *total,
                       const char *description) {
  if (count > std::numeric_limits<std::size_t>::max() - *total) {
    throw std::invalid_argument(std::string("CUDA ") + description +
                                " size overflows size_t");
  }
  *total += count;
}

std::size_t CheckedSum(std::initializer_list<std::size_t> terms,
                       const char *description) {
  std::size_t result = 0;
  for (const std::size_t term : terms)
    CheckedAccumulate(term, &result, description);
  return result;
}

std::size_t BalancedTreeNodeCount(std::size_t leaf_count) {
  Require(leaf_count > 0, "CUDA tree must contain at least one leaf");
  std::size_t total = 0;
  for (std::size_t level_count = leaf_count;;
       level_count = level_count / 2 + level_count % 2) {
    Require(level_count <= std::numeric_limits<std::size_t>::max() - total,
            "CUDA tree size overflows size_t");
    total += level_count;
    if (level_count == 1)
      return total;
  }
}

bool ValidateCudaProblem(const Problem &problem, const Options &options,
                         const std::vector<int> &structure_key,
                         bool validate_values) {
  Require(std::isfinite(options.tolerance) && options.tolerance > Scalar{0},
          "CUDA tolerance must be finite and positive");
  Require(options.device >= 0, "CUDA device index must be nonnegative");
  const std::size_t count = problem.stages.size();
  bool structure_matches =
      structure_key.size() ==
      ScratchCheckedSum(
          {ScratchCheckedProduct(5, count, "CUDA structure key"), 3},
          "CUDA structure key");
  std::size_t key_index = 0;
  const auto match_dimension = [&](std::size_t dimension) {
    if (structure_matches) {
      structure_matches &=
          structure_key[key_index] == static_cast<int>(dimension);
    }
    ++key_index;
  };
  match_dimension(problem.initial_state.size());
  match_dimension(problem.terminal_Q.rows());
  match_dimension(problem.terminal_E.rows());
  Require(count < static_cast<std::size_t>(std::numeric_limits<int>::max()),
          "too many stages for CUDA indices");
  constexpr std::size_t kMaxTreeIndex =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  Require(BalancedTreeNodeCount(count + 1) <= kMaxTreeIndex &&
              BalancedTreeNodeCount(std::max<std::size_t>(count, 1)) <=
                  kMaxTreeIndex,
          "too many stages for CUDA tree indices");
  Require(problem.terminal_Q.rows() == problem.terminal_Q.cols(),
          "terminal_Q must be square");
  Require(problem.terminal_Q.rows() <=
              static_cast<std::size_t>(std::numeric_limits<int>::max()),
          "terminal state dimension exceeds CUDA index range");
  Require(problem.terminal_q.size() == problem.terminal_Q.rows(),
          "terminal_q shape mismatch");
  Require(problem.terminal_E.cols() == problem.terminal_Q.rows(),
          "terminal_E shape mismatch");
  Require(problem.terminal_E.rows() <=
              static_cast<std::size_t>(std::numeric_limits<int>::max()),
          "terminal constraint count exceeds CUDA index range");
  Require(problem.terminal_e.size() == problem.terminal_E.rows(),
          "terminal_e shape mismatch");
  if (validate_values) {
    Require(Finite(problem.terminal_Q) && Finite(problem.terminal_q) &&
                Finite(problem.terminal_E) && Finite(problem.terminal_e) &&
                Finite(problem.initial_state),
            "problem contains a non-finite terminal or initial value");
  }
  if (count == 0) {
    Require(problem.initial_state.size() == problem.terminal_Q.rows(),
            "initial_state and terminal state dimensions differ");
    return structure_matches;
  }
  Require(problem.initial_state.size() == problem.stages.front().A.cols(),
          "initial_state and first stage dimensions differ");
  for (std::size_t i = 0; i < count; ++i) {
    const Stage &s = problem.stages[i];
    const std::size_t n = s.A.cols();
    const std::size_t next = s.A.rows();
    const std::size_t m = s.B.cols();
    constexpr std::size_t kMaxIndex =
        static_cast<std::size_t>(std::numeric_limits<int>::max());
    RequireAtStage(n <= kMaxIndex && next <= kMaxIndex,
                   "state dimension exceeds CUDA index range", i);
    RequireAtStage(m <= kMaxIndex,
                   "control dimension exceeds CUDA index range", i);
    RequireAtStage(s.C.rows() <= kMaxIndex,
                   "mixed constraint count exceeds CUDA index range", i);
    RequireAtStage(s.E.rows() <= kMaxIndex,
                   "state constraint count exceeds CUDA index range", i);
    const std::size_t mixed = s.C.rows();
    const std::size_t state_constraints = s.E.rows();
    match_dimension(n);
    match_dimension(next);
    match_dimension(m);
    match_dimension(mixed);
    match_dimension(state_constraints);
    // Scratch/index bounds depend only on the five dimensions recorded in the
    // structure key.  Rechecking all overflow-safe sums on every solve is
    // measurable for long horizons, while a matching cached key proves that
    // the same bounds were checked when the workspace layout was prepared.
    if (!structure_matches) {
      const auto require_index_sum =
          [&](std::initializer_list<std::size_t> terms,
              const char *description) {
            if (CheckedSum(terms, description) > kMaxIndex) {
              throw std::invalid_argument(std::string(description) +
                                          " exceeds CUDA index range at stage " +
                                          std::to_string(i));
            }
          };
      require_index_sum({next, mixed}, "dual dimension");
      require_index_sum({mixed, state_constraints, next},
                        "feasibility row dimension");
      require_index_sum({m, n, next, 1}, "feasibility column dimension");
      require_index_sum({mixed, next}, "reduction row dimension");
      require_index_sum({m, n, 1}, "reduction column dimension");
      require_index_sum({next, m}, "dual-system row dimension");
    }
    RequireAtStage(s.B.rows() == next && s.c.size() == next,
                   "dynamics shape mismatch", i);
    RequireAtStage(s.Q.rows() == n && s.Q.cols() == n && s.q.size() == n,
                   "state-cost shape mismatch", i);
    RequireAtStage(s.R.rows() == m && s.R.cols() == m && s.r.size() == m,
                   "control-cost shape mismatch", i);
    RequireAtStage(s.M.rows() == n && s.M.cols() == m,
                   "cross-cost shape mismatch", i);
    RequireAtStage(s.C.cols() == n && s.D.rows() == s.C.rows() &&
                       s.D.cols() == m && s.d.size() == s.C.rows(),
                   "mixed-constraint shape mismatch", i);
    RequireAtStage(s.E.cols() == n && s.e.size() == s.E.rows(),
                   "state-constraint shape mismatch", i);
    const std::size_t expected_next = i + 1 == count
                                          ? problem.terminal_Q.rows()
                                          : problem.stages[i + 1].A.cols();
    RequireAtStage(next == expected_next,
                   "neighboring state dimensions differ", i);
    if (validate_values) {
      RequireAtStage(Finite(s.A) && Finite(s.B) && Finite(s.c) &&
                         Finite(s.Q) && Finite(s.R) && Finite(s.M) &&
                         Finite(s.q) && Finite(s.r) && Finite(s.C) &&
                         Finite(s.D) && Finite(s.d) && Finite(s.E) &&
                         Finite(s.e),
                     "problem contains a non-finite value", i);
    }
  }
  return structure_matches;
}

bool PackScalars(const Scalar *source, std::size_t entries,
                 Scalar **host_cursor, Scalar **device_cursor,
                 const Scalar **target) {
  *target = *device_cursor;
  bool finite = true;
  for (std::size_t index = 0; index < entries; ++index) {
    const Scalar value = source[index];
    (*host_cursor)[index] = value;
    finite &= std::isfinite(value);
  }
  *host_cursor += entries;
  *device_cursor += entries;
  return finite;
}

bool PackMatrix(const Matrix &source, Scalar **host_cursor,
                Scalar **device_cursor, const Scalar **target) {
  return PackScalars(source.data().data(), source.rows() * source.cols(),
                     host_cursor, device_cursor, target);
}

bool PackVector(const Vector &source, Scalar **host_cursor,
                Scalar **device_cursor, const Scalar **target) {
  return PackScalars(source.data().data(), source.size(), host_cursor,
                     device_cursor, target);
}

bool PackStage(const Stage &source, Scalar **host_cursor,
               Scalar **device_cursor, PackedStage *out) {
  out->n = static_cast<int>(source.A.cols());
  out->next_n = static_cast<int>(source.A.rows());
  out->m = static_cast<int>(source.B.cols());
  out->mixed = static_cast<int>(source.C.rows());
  out->state = static_cast<int>(source.E.rows());
  bool finite = true;
  finite &= PackMatrix(source.A, host_cursor, device_cursor, &out->A);
  finite &= PackMatrix(source.B, host_cursor, device_cursor, &out->B);
  finite &= PackVector(source.c, host_cursor, device_cursor, &out->c);
  finite &= PackMatrix(source.Q, host_cursor, device_cursor, &out->Q);
  finite &= PackMatrix(source.R, host_cursor, device_cursor, &out->R);
  finite &= PackMatrix(source.M, host_cursor, device_cursor, &out->M);
  finite &= PackVector(source.q, host_cursor, device_cursor, &out->q);
  finite &= PackVector(source.r, host_cursor, device_cursor, &out->r);
  finite &= PackMatrix(source.C, host_cursor, device_cursor, &out->C);
  finite &= PackMatrix(source.D, host_cursor, device_cursor, &out->D);
  finite &= PackVector(source.d, host_cursor, device_cursor, &out->d);
  finite &= PackMatrix(source.E, host_cursor, device_cursor, &out->E);
  finite &= PackVector(source.e, host_cursor, device_cursor, &out->e);
  return finite;
}

bool PackTerminal(const Problem &problem, Scalar **host_cursor,
                  Scalar **device_cursor, PackedTerminal *out) {
  out->n = static_cast<int>(problem.terminal_Q.rows());
  out->state = static_cast<int>(problem.terminal_E.rows());
  bool finite = true;
  finite &=
      PackMatrix(problem.terminal_Q, host_cursor, device_cursor, &out->Q);
  finite &=
      PackVector(problem.terminal_q, host_cursor, device_cursor, &out->q);
  finite &=
      PackMatrix(problem.terminal_E, host_cursor, device_cursor, &out->E);
  finite &=
      PackVector(problem.terminal_e, host_cursor, device_cursor, &out->e);
  return finite;
}

std::string DeviceFailureMessage(const DeviceStatus &status) {
  std::ostringstream out;
  if (status.code == kDeviceInfeasible) {
    out << "CUDA feasibility elimination found an inconsistent relation";
  } else if (status.detail == 19) {
    out << "CUDA conditional-value scan requires a positive-definite reduced "
           "control Hessian";
  } else if (status.detail == 20) {
    out << "CUDA conditional-value scan encountered a singular or "
           "incompatible interval composition";
  } else {
    out << "CUDA backend encountered a rank or consistency failure";
  }
  if (status.stage >= 0)
    out << " at stage/node " << status.stage;
  out << " (diagnostic " << status.detail << ")";
  return out.str();
}

bool ApplyDeviceFailure(const DeviceStatus &status, Solution *solution) {
  if (status.code == kDeviceOk)
    return false;
  solution->status = status.code == kDeviceInfeasible
                         ? SolveStatus::kInfeasible
                         : SolveStatus::kNumericalFailure;
  solution->message = DeviceFailureMessage(status);
  return true;
}

struct CompactEntryCounts {
  std::size_t states = 0;
  std::size_t controls = 0;
  std::size_t mixed = 0;
  std::size_t state_constraints = 0;
  std::size_t problem_data = 0;
};

CompactEntryCounts CountCompactEntries(const Problem &problem) {
  CompactEntryCounts counts;
  counts.states = problem.initial_state.size();
  for (const Stage &stage : problem.stages) {
    const std::size_t n = stage.A.cols();
    const std::size_t next_n = stage.A.rows();
    const std::size_t m = stage.B.cols();
    const std::size_t mixed = stage.C.rows();
    const std::size_t state_constraints = stage.E.rows();
    CheckedAccumulate(next_n, &counts.states, "compact state buffer");
    CheckedAccumulate(m, &counts.controls, "compact control buffer");
    CheckedAccumulate(mixed, &counts.mixed, "compact mixed-constraint buffer");
    CheckedAccumulate(state_constraints, &counts.state_constraints,
                      "compact state-constraint buffer");
    const std::size_t stage_problem_data =
        CheckedSum({CheckedProduct(next_n, n, "packed problem data"),
                    CheckedProduct(next_n, m, "packed problem data"), next_n,
                    CheckedProduct(n, n, "packed problem data"),
                    CheckedProduct(m, m, "packed problem data"),
                    CheckedProduct(n, m, "packed problem data"), n, m,
                    CheckedProduct(mixed, n, "packed problem data"),
                    CheckedProduct(mixed, m, "packed problem data"), mixed,
                    CheckedProduct(state_constraints, n, "packed problem data"),
                    state_constraints},
                   "packed problem data");
    CheckedAccumulate(stage_problem_data, &counts.problem_data,
                      "packed problem data");
  }
  const std::size_t terminal_n = problem.terminal_Q.rows();
  const std::size_t terminal_constraints = problem.terminal_E.rows();
  CheckedAccumulate(CheckedSum({CheckedProduct(terminal_n, terminal_n,
                                               "packed terminal data"),
                                terminal_n,
                                CheckedProduct(terminal_constraints, terminal_n,
                                               "packed terminal data"),
                                terminal_constraints},
                               "packed terminal data"),
                    &counts.problem_data, "packed problem data");
  constexpr std::size_t kMaxOffset =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  Require(counts.states <= kMaxOffset && counts.controls <= kMaxOffset &&
              counts.mixed <= kMaxOffset &&
              counts.state_constraints <= kMaxOffset,
          "CUDA compact buffer offsets exceed 32-bit indexing");
  return counts;
}

template <typename T> T *TakeStorage(T **cursor, std::size_t count) {
  T *result = *cursor;
  *cursor += count;
  return result;
}

// Bind every small dense object to a compact runtime-sized slice.  The slices
// use the physical dimensions as safe capacities; kernels continue to perform
// arithmetic only over their active reduced dimensions.
void PrepareStageStorage(const Problem &problem, WorkspaceStorage *workspace) {
  const std::size_t stage_count = problem.stages.size();
  const std::size_t node_count = stage_count + 1;
  std::vector<int> layout_key;
  layout_key.reserve(5 * stage_count + 2);
  for (const Stage &stage : problem.stages) {
    layout_key.push_back(static_cast<int>(stage.A.cols()));
    layout_key.push_back(static_cast<int>(stage.A.rows()));
    layout_key.push_back(static_cast<int>(stage.B.cols()));
    layout_key.push_back(static_cast<int>(stage.C.rows()));
    layout_key.push_back(static_cast<int>(stage.E.rows()));
  }
  layout_key.push_back(static_cast<int>(problem.terminal_Q.rows()));
  layout_key.push_back(static_cast<int>(problem.terminal_E.rows()));
  if (layout_key == workspace->stage_layout_key)
    return;
  std::size_t scalar_entries = 0;
  std::size_t index_entries = 0;
  auto scalars = [&](std::size_t count) {
    CheckedAccumulate(count, &scalar_entries, "stage workspace");
  };
  auto indices = [&](std::size_t count) {
    CheckedAccumulate(count, &index_entries, "stage index workspace");
  };
  auto square = [&](std::size_t dimension) {
    return CheckedProduct(dimension, dimension, "dense workspace");
  };
  auto rectangle = [&](std::size_t rows, std::size_t columns) {
    return CheckedProduct(rows, columns, "dense workspace");
  };

  for (std::size_t node = 0; node < node_count; ++node) {
    const std::size_t n = node == stage_count ? problem.terminal_Q.rows()
                                              : problem.stages[node].A.cols();
    indices(n);
    scalars(square(n)); // StateParam T.
    scalars(n);         // StateParam t.
  }
  for (std::size_t stage = 0; stage < stage_count; ++stage) {
    const Stage &source = problem.stages[stage];
    const std::size_t n = source.A.cols();
    const std::size_t next = source.A.rows();
    const std::size_t m = source.B.cols();
    const std::size_t dual =
        CheckedSum({next, source.C.rows()}, "dual dimension");
    indices(m); // ControlParam free columns.
    scalars(rectangle(m, n));
    scalars(square(m));
    scalars(m);
    scalars(rectangle(next, n));
    scalars(rectangle(next, m));
    scalars(next);
    scalars(square(n));
    scalars(square(m));
    scalars(rectangle(n, m));
    scalars(n);
    scalars(m); // ReducedStage.
    scalars(rectangle(m, n));
    scalars(m);
    scalars(rectangle(next, n));
    scalars(next); // Feedback.
    indices(dual); // DualParam free columns.
    scalars(square(dual));
    scalars(dual);

    const std::size_t node = stage + 1;
    const std::size_t constraints = node == stage_count
                                        ? problem.terminal_E.rows()
                                        : problem.stages[node].E.rows();
    const std::size_t left_dual = dual;
    const std::size_t right_dual =
        node == stage_count ? 0
                            : CheckedSum({problem.stages[node].A.rows(),
                                          problem.stages[node].C.rows()},
                                         "right dual dimension");
    scalars(constraints);
    scalars(rectangle(constraints, left_dual));
    scalars(rectangle(constraints, right_dual)); // StateDualParam.
  }
  const std::size_t terminal_n = problem.terminal_Q.rows();
  scalars(square(terminal_n));
  scalars(terminal_n);

  workspace->stage_data.Reserve(scalar_entries);
  workspace->stage_indices.Reserve(index_entries);
  Scalar *scalar_cursor = workspace->stage_data.get();
  int *index_cursor = workspace->stage_indices.get();

  for (std::size_t node = 0; node < node_count; ++node) {
    const std::size_t n = node == stage_count ? problem.terminal_Q.rows()
                                              : problem.stages[node].A.cols();
    StateParam &out = workspace->host_state_params[node];
    out = {};
    out.free_columns = TakeStorage(&index_cursor, n);
    out.T = TakeStorage(&scalar_cursor, square(n));
    out.t = TakeStorage(&scalar_cursor, n);
  }
  for (std::size_t stage = 0; stage < stage_count; ++stage) {
    const Stage &source = problem.stages[stage];
    const std::size_t n = source.A.cols();
    const std::size_t next = source.A.rows();
    const std::size_t m = source.B.cols();
    const std::size_t dual =
        CheckedSum({next, source.C.rows()}, "dual dimension");

    ControlParam &control = workspace->host_control_params[stage];
    control = {};
    control.free_columns = TakeStorage(&index_cursor, m);
    control.Y = TakeStorage(&scalar_cursor, rectangle(m, n));
    control.Z = TakeStorage(&scalar_cursor, square(m));
    control.y = TakeStorage(&scalar_cursor, m);

    ReducedStage &reduced = workspace->host_reduced_stages[stage];
    reduced = {};
    reduced.A = TakeStorage(&scalar_cursor, rectangle(next, n));
    reduced.B = TakeStorage(&scalar_cursor, rectangle(next, m));
    reduced.c = TakeStorage(&scalar_cursor, next);
    reduced.Q = TakeStorage(&scalar_cursor, square(n));
    reduced.R = TakeStorage(&scalar_cursor, square(m));
    reduced.M = TakeStorage(&scalar_cursor, rectangle(n, m));
    reduced.q = TakeStorage(&scalar_cursor, n);
    reduced.r = TakeStorage(&scalar_cursor, m);

    Feedback &feedback = workspace->host_feedback[stage];
    feedback = {};
    feedback.K = TakeStorage(&scalar_cursor, rectangle(m, n));
    feedback.k = TakeStorage(&scalar_cursor, m);
    feedback.transition = TakeStorage(&scalar_cursor, rectangle(next, n));
    feedback.offset = TakeStorage(&scalar_cursor, next);

    DualParam &dual_param = workspace->host_dual_params[stage];
    dual_param = {};
    dual_param.free_columns = TakeStorage(&index_cursor, dual);
    dual_param.basis = TakeStorage(&scalar_cursor, square(dual));
    dual_param.offset = TakeStorage(&scalar_cursor, dual);

    const std::size_t node = stage + 1;
    const std::size_t constraints = node == stage_count
                                        ? problem.terminal_E.rows()
                                        : problem.stages[node].E.rows();
    const std::size_t right_dual =
        node == stage_count ? 0
                            : CheckedSum({problem.stages[node].A.rows(),
                                          problem.stages[node].C.rows()},
                                         "right dual dimension");
    StateDualParam &state_dual = workspace->host_state_dual_params[stage];
    state_dual = {};
    state_dual.offset = TakeStorage(&scalar_cursor, constraints);
    state_dual.left = TakeStorage(&scalar_cursor, rectangle(constraints, dual));
    state_dual.right =
        TakeStorage(&scalar_cursor, rectangle(constraints, right_dual));
  }
  ReducedTerminal &terminal = workspace->host_reduced_terminal[0];
  terminal = {};
  terminal.Q = TakeStorage(&scalar_cursor, square(terminal_n));
  terminal.q = TakeStorage(&scalar_cursor, terminal_n);

  Require(scalar_cursor == workspace->stage_data.get() + scalar_entries,
          "internal CUDA stage-workspace layout size mismatch");
  Require(index_cursor == workspace->stage_indices.get() + index_entries,
          "internal CUDA stage-index layout size mismatch");
  workspace->stage_layout_key = std::move(layout_key);
  workspace->stage_layout_uploaded = false;
}

void RequireScratchFits(const ScratchRequirements &scratch, int device) {
  int shared_memory_limit = 0;
  CudaCheck(cudaDeviceGetAttribute(&shared_memory_limit,
                                   cudaDevAttrMaxSharedMemoryPerBlock, device),
            "query CUDA shared-memory limit");
  constexpr std::size_t kStaticSharedMemoryAllowance = 256;
  const std::size_t usable_shared_memory =
      static_cast<std::size_t>(shared_memory_limit) >
              kStaticSharedMemoryAllowance
          ? static_cast<std::size_t>(shared_memory_limit) -
                kStaticSharedMemoryAllowance
          : 0;
  Require(scratch.Maximum() <= usable_shared_memory,
          "active dimensions require " + std::to_string(scratch.Maximum()) +
              " bytes of dynamic per-block workspace, exceeding device " +
              "shared-memory resources (" +
              std::to_string(shared_memory_limit) + " bytes per block)");
}

void BuildCompactOffsets(const Problem &problem, WorkspaceStorage *workspace) {
  auto &state = workspace->host_state_offsets;
  auto &control = workspace->host_control_offsets;
  auto &dynamics = workspace->host_dynamics_offsets;
  auto &mixed = workspace->host_mixed_offsets;
  auto &state_constraint = workspace->host_state_constraint_offsets;
  state[0] = 0;
  control[0] = 0;
  dynamics[0] = 0;
  mixed[0] = 0;
  state_constraint[0] = 0;
  const auto append_offset = [](int previous, std::size_t count,
                                const char *description) {
    constexpr std::size_t kMaximum =
        static_cast<std::size_t>(std::numeric_limits<int>::max());
    Require(
        previous >= 0 && count <= kMaximum - static_cast<std::size_t>(previous),
        std::string("CUDA ") + description + " offsets exceed 32-bit indexing");
    return previous + static_cast<int>(count);
  };
  for (std::size_t index = 0; index < problem.stages.size(); ++index) {
    const Stage &stage = problem.stages[index];
    state[index + 1] = append_offset(state[index], stage.A.cols(), "state");
    control[index + 1] =
        append_offset(control[index], stage.B.cols(), "control");
    dynamics[index + 1] =
        append_offset(dynamics[index], stage.A.rows(), "dynamics");
    mixed[index + 1] =
        append_offset(mixed[index], stage.C.rows(), "mixed-constraint");
    state_constraint[index + 1] = append_offset(
        state_constraint[index], stage.E.rows(), "state-constraint");
  }
  state[problem.stages.size() + 1] = append_offset(
      state[problem.stages.size()], problem.terminal_Q.rows(), "state");
}

struct ValueCapacity {
  std::size_t a = 0;
  std::size_t b = 0;
  std::size_t c = 0;
  std::size_t eta = 0;
  std::size_t j = 0;

  void Include(const ScanShape &shape) {
    if (!shape.valid)
      return;
    a = std::max(a, CheckedProduct(shape.right, shape.left, "value layout"));
    b = std::max(b, shape.right);
    c = std::max(c, CheckedProduct(shape.right, shape.right, "value layout"));
    eta = std::max(eta, shape.left);
    j = std::max(j, CheckedProduct(shape.left, shape.left, "value layout"));
  }

  std::size_t Entries() const {
    return CheckedSum({a, b, c, eta, j}, "value layout");
  }
};

struct RelationCapacity {
  std::size_t left = 0;
  std::size_t right = 0;
  std::size_t rhs = 0;

  void Include(const ScanShape &shape) {
    if (!shape.valid)
      return;
    left = std::max(left,
                    CheckedProduct(shape.rows, shape.left, "relation layout"));
    right = std::max(
        right, CheckedProduct(shape.rows, shape.right, "relation layout"));
    rhs = std::max(rhs, shape.rows);
  }

  std::size_t Entries() const {
    return CheckedSum({left, right, rhs}, "relation layout");
  }
};

struct MapCapacity {
  std::size_t linear = 0;
  std::size_t offset = 0;

  void Include(const ScanShape &shape) {
    if (!shape.valid)
      return;
    linear =
        std::max(linear, CheckedProduct(shape.left, shape.right, "map layout"));
    offset = std::max(offset, shape.right);
  }

  std::size_t Entries() const {
    return CheckedSum({linear, offset}, "map layout");
  }
};

template <typename Capacity>
void PlanSuffixScanStorage(const std::vector<ScanShape> &leaves,
                           const std::vector<int> &level_offsets,
                           std::vector<Capacity> *leaf_capacity,
                           std::vector<Capacity> *internal_capacity) {
  const int leaf_count = static_cast<int>(leaves.size());
  const ScanPlan tree = BuildScanPlan(leaves);
  const int internal_count = level_offsets.back() +
                             static_cast<int>(tree.reductions.back().size()) -
                             leaf_count;
  leaf_capacity->assign(leaf_count, Capacity{});
  internal_capacity->assign(internal_count, Capacity{});
  for (int leaf = 0; leaf < leaf_count; ++leaf)
    (*leaf_capacity)[leaf].Include(leaves[leaf]);

  for (std::size_t level = 1; level < tree.reductions.size(); ++level) {
    const int offset = level_offsets[level] - leaf_count;
    for (std::size_t node = 0; node < tree.reductions[level].size(); ++node)
      (*internal_capacity)[offset + node].Include(tree.reductions[level][node]);
  }

  if (leaf_count == 1)
    return;
  for (std::size_t level = 1; level < tree.suffix_contexts.size(); ++level) {
    const int offset = level_offsets[level] - leaf_count;
    for (std::size_t node = 0; node < tree.suffix_contexts[level].size();
         ++node) {
      (*internal_capacity)[offset + node].Include(
          tree.suffix_contexts[level][node]);
    }
  }

  for (std::size_t parent = 0; parent < tree.suffix_contexts[1].size();
       ++parent) {
    const int child = 2 * parent;
    const ScanShape &parent_context = tree.suffix_contexts[1][parent];
    if (child + 1 >= leaf_count) {
      (*leaf_capacity)[child].Include(
          ComposeScanShapes(leaves[child], parent_context));
      continue;
    }
    const ScanShape right_suffix =
        ComposeScanShapes(leaves[child + 1], parent_context);
    const ScanShape left_suffix =
        ComposeScanShapes(leaves[child], right_suffix);
    (*leaf_capacity)[child + 1].Include(right_suffix);
    (*leaf_capacity)[child].Include(left_suffix);
  }
}

void BindValueStorage(ValueElement *element, Scalar **cursor,
                      const ValueCapacity &capacity) {
  element->left_dim = -1;
  element->right_dim = 0;
  element->A = *cursor;
  *cursor += capacity.a;
  element->b = *cursor;
  *cursor += capacity.b;
  element->C = *cursor;
  *cursor += capacity.c;
  element->eta = *cursor;
  *cursor += capacity.eta;
  element->J = *cursor;
  *cursor += capacity.j;
}

void BindRelationStorage(Relation *relation, Scalar **cursor,
                         const RelationCapacity &capacity) {
  relation->left_dim = -1;
  relation->right_dim = 0;
  relation->rows = 0;
  relation->left = *cursor;
  *cursor += capacity.left;
  relation->right = *cursor;
  *cursor += capacity.right;
  relation->rhs = *cursor;
  *cursor += capacity.rhs;
}

void BindMapStorage(AffineMap *map, Scalar **cursor,
                    const MapCapacity &capacity) {
  map->left_dim = -1;
  map->right_dim = 0;
  map->linear = *cursor;
  *cursor += capacity.linear;
  map->offset = *cursor;
  *cursor += capacity.offset;
}

bool PrepareRelationStorage(const Problem &problem,
                            WorkspaceStorage *workspace) {
  const int stage_count = static_cast<int>(problem.stages.size());
  const int node_count = stage_count + 1;
  bool layout_matches =
      workspace->relation_layout_key.size() ==
      static_cast<std::size_t>(2) * static_cast<std::size_t>(node_count);
  for (int stage = 0; stage < stage_count && layout_matches; ++stage) {
    layout_matches &= workspace->relation_layout_key[2 * stage] ==
                          static_cast<int>(problem.stages[stage].A.cols()) &&
                      workspace->relation_layout_key[2 * stage + 1] ==
                          static_cast<int>(problem.stages[stage].A.rows());
  }
  if (layout_matches) {
    layout_matches &= workspace->relation_layout_key[2 * stage_count] ==
                          static_cast<int>(problem.terminal_Q.rows()) &&
                      workspace->relation_layout_key[2 * stage_count + 1] == 0;
  }
  if (layout_matches)
    return false;
  std::vector<ScanShape> leaves(node_count);
  std::vector<int> key;
  key.reserve(static_cast<std::size_t>(2) * node_count);
  for (int stage = 0; stage < stage_count; ++stage) {
    leaves[stage] = MakeScanShape(problem.stages[stage].A.cols(),
                                  problem.stages[stage].A.rows());
    key.push_back(static_cast<int>(leaves[stage].left));
    key.push_back(static_cast<int>(leaves[stage].right));
  }
  leaves[stage_count] = MakeScanShape(problem.terminal_Q.rows(), 0);
  key.push_back(static_cast<int>(leaves[stage_count].left));
  key.push_back(0);
  std::vector<RelationCapacity> leaf_capacity;
  std::vector<RelationCapacity> internal_capacity;
  PlanSuffixScanStorage(leaves, workspace->node_level_offsets, &leaf_capacity,
                        &internal_capacity);
  std::size_t entries = 0;
  for (const RelationCapacity &capacity : leaf_capacity)
    CheckedAccumulate(capacity.Entries(), &entries, "relation layout");
  for (const RelationCapacity &capacity : internal_capacity)
    CheckedAccumulate(capacity.Entries(), &entries, "relation layout");
  workspace->relation_data.Reserve(entries);
  Scalar *cursor = workspace->relation_data.get();
  for (int node = 0; node < node_count; ++node)
    BindRelationStorage(&workspace->host_relation_leaves[node], &cursor,
                        leaf_capacity[node]);
  for (std::size_t node = 0; node < internal_capacity.size(); ++node)
    BindRelationStorage(&workspace->host_relation_scan[node], &cursor,
                        internal_capacity[node]);
  Require(cursor == workspace->relation_data.get() + entries,
          "internal CUDA relation layout size mismatch");
  workspace->relation_layout_key = std::move(key);
  return true;
}

bool PrepareValueStorage(WorkspaceStorage *workspace, int stage_count) {
  const int node_count = stage_count + 1;
  bool layout_matches = workspace->value_layout_key.size() ==
                        static_cast<std::size_t>(node_count);
  for (int node = 0; node < node_count && layout_matches; ++node) {
    layout_matches &= workspace->value_layout_key[node] ==
                      workspace->host_state_dimensions[2 * node + 1];
  }
  if (layout_matches)
    return false;
  const auto &level_offsets = workspace->node_level_offsets;
  std::vector<ScanShape> leaves(node_count);
  std::vector<int> key;
  key.reserve(node_count);
  for (int node = 0; node < node_count; ++node) {
    const int left = workspace->host_state_dimensions[2 * node + 1];
    const int right = node == stage_count
                          ? 0
                          : workspace->host_state_dimensions[2 * node + 3];
    leaves[node] = MakeScanShape(left, right);
    key.push_back(left);
  }

  const int internal_count =
      static_cast<int>(workspace->host_value_scan.size());
  std::vector<ValueCapacity> leaf_capacity;
  std::vector<ValueCapacity> internal_capacity;
  PlanSuffixScanStorage(leaves, level_offsets, &leaf_capacity,
                        &internal_capacity);

  std::size_t entries = 0;
  for (const ValueCapacity &capacity : leaf_capacity)
    CheckedAccumulate(capacity.Entries(), &entries, "value layout");
  for (const ValueCapacity &capacity : internal_capacity)
    CheckedAccumulate(capacity.Entries(), &entries, "value layout");
  workspace->value_data.Reserve(entries);
  Scalar *cursor = workspace->value_data.get();
  for (int node = 0; node < node_count; ++node)
    BindValueStorage(&workspace->host_value_leaves[node], &cursor,
                     leaf_capacity[node]);
  for (int node = 0; node < internal_count; ++node)
    BindValueStorage(&workspace->host_value_scan[node], &cursor,
                     internal_capacity[node]);
  Require(cursor == workspace->value_data.get() + entries,
          "internal CUDA value layout size mismatch");
  workspace->value_layout_key = std::move(key);
  return true;
}

bool PrepareMapStorage(WorkspaceStorage *workspace, int stage_count) {
  if (stage_count == 0)
    return false;
  const int node_count = stage_count + 1;
  bool layout_matches =
      workspace->map_layout_key.size() == static_cast<std::size_t>(node_count);
  for (int node = 0; node < node_count && layout_matches; ++node) {
    layout_matches &= workspace->map_layout_key[node] ==
                      workspace->host_state_dimensions[2 * node + 1];
  }
  if (layout_matches)
    return false;
  const auto &level_offsets = workspace->stage_level_offsets;
  std::vector<ScanShape> leaves(stage_count);
  std::vector<int> key;
  key.reserve(stage_count + 1);
  key.push_back(workspace->host_state_dimensions[1]);
  std::vector<MapCapacity> leaf_capacity(stage_count);
  for (int stage = 0; stage < stage_count; ++stage) {
    leaves[stage] =
        MakeScanShape(workspace->host_state_dimensions[2 * stage + 1],
                      workspace->host_state_dimensions[2 * stage + 3]);
    leaf_capacity[stage].Include(leaves[stage]);
    key.push_back(static_cast<int>(leaves[stage].right));
  }

  const int internal_count = static_cast<int>(workspace->host_map_scan.size());
  std::vector<MapCapacity> internal_capacity(internal_count);
  const ScanPlan tree = BuildScanPlan(leaves);
  for (std::size_t level = 1; level < tree.reductions.size(); ++level) {
    const int offset = level_offsets[level] - stage_count;
    for (std::size_t node = 0; node < tree.reductions[level].size(); ++node) {
      internal_capacity[offset + node].Include(tree.reductions[level][node]);
      internal_capacity[offset + node].Include(
          tree.prefix_contexts[level][node]);
    }
  }

  if (stage_count > 1) {
    for (std::size_t parent = 0; parent < tree.prefix_contexts[1].size();
         ++parent) {
      const std::size_t child = 2 * parent;
      const ScanShape left_prefix =
          ComposeScanShapes(tree.prefix_contexts[1][parent], leaves[child]);
      leaf_capacity[child].Include(left_prefix);
      if (child + 1 < leaves.size()) {
        leaf_capacity[child + 1].Include(
            ComposeScanShapes(left_prefix, leaves[child + 1]));
      }
    }
  }

  std::size_t entries = 0;
  for (const MapCapacity &capacity : leaf_capacity)
    CheckedAccumulate(capacity.Entries(), &entries, "map layout");
  for (const MapCapacity &capacity : internal_capacity)
    CheckedAccumulate(capacity.Entries(), &entries, "map layout");
  workspace->map_data.Reserve(entries);
  Scalar *cursor = workspace->map_data.get();
  for (int stage = 0; stage < stage_count; ++stage)
    BindMapStorage(&workspace->host_map_leaves[stage], &cursor,
                   leaf_capacity[stage]);
  for (int node = 0; node < internal_count; ++node)
    BindMapStorage(&workspace->host_map_scan[node], &cursor,
                   internal_capacity[node]);
  Require(cursor == workspace->map_data.get() + entries,
          "internal CUDA map layout size mismatch");
  workspace->map_layout_key = std::move(key);
  return true;
}

struct DualValueCapacity {
  std::size_t left = 0;
  std::size_t right = 0;

  void Include(const ScanShape &shape) {
    left = std::max(left, shape.left);
    right = std::max(right, shape.right);
  }

  std::size_t Entries() const {
    return CheckedSum({left, right}, "dual-value layout");
  }
};

void BindDualRelationStorage(DualRelation *relation, Scalar **cursor,
                             const RelationCapacity &capacity) {
  relation->left_dim = -1;
  relation->right_dim = 0;
  relation->rows = 0;
  relation->left = *cursor;
  *cursor += capacity.left;
  relation->right = *cursor;
  *cursor += capacity.right;
  relation->rhs = *cursor;
  *cursor += capacity.rhs;
}

void BindDualValueStorage(DualNodeValue *value, Scalar **cursor,
                          const DualValueCapacity &capacity) {
  value->left_dim = -1;
  value->right_dim = 0;
  value->left = *cursor;
  *cursor += capacity.left;
  value->right = *cursor;
  *cursor += capacity.right;
}

bool PrepareDualStorage(WorkspaceStorage *workspace, int stage_count) {
  bool layout_matches = workspace->dual_layout_key.size() ==
                        static_cast<std::size_t>(stage_count);
  for (int stage = 0; stage < stage_count && layout_matches; ++stage) {
    layout_matches &= workspace->dual_layout_key[stage] ==
                      workspace->host_dual_dimensions[stage];
  }
  if (layout_matches)
    return false;
  std::vector<int> key(workspace->host_dual_dimensions.begin(),
                       workspace->host_dual_dimensions.end());
  const auto &level_offsets = workspace->stage_level_offsets;
  std::vector<ScanShape> leaves(stage_count);
  for (int stage = 0; stage < stage_count; ++stage) {
    const int right = stage + 1 == stage_count
                          ? 0
                          : workspace->host_dual_dimensions[stage + 1];
    leaves[stage] =
        MakeScanShape(workspace->host_dual_dimensions[stage], right);
  }
  const ScanPlan tree = BuildScanPlan(leaves);
  const int tree_size =
      level_offsets.back() + static_cast<int>(tree.reductions.back().size());
  std::vector<RelationCapacity> relation_capacity(tree_size);
  std::vector<DualValueCapacity> value_capacity(tree_size);
  for (std::size_t level = 0; level < tree.reductions.size(); ++level) {
    const int offset = level_offsets[level];
    for (std::size_t node = 0; node < tree.reductions[level].size(); ++node) {
      relation_capacity[offset + node].Include(tree.reductions[level][node]);
      value_capacity[offset + node].Include(tree.reductions[level][node]);
    }
  }

  std::size_t relation_entries = 0;
  std::size_t value_entries = 0;
  for (int node = 0; node < tree_size; ++node) {
    CheckedAccumulate(relation_capacity[node].Entries(), &relation_entries,
                      "dual-relation layout");
    CheckedAccumulate(value_capacity[node].Entries(), &value_entries,
                      "dual-value layout");
  }
  workspace->dual_tree.Reserve(tree_size);
  workspace->dual_values.Reserve(tree_size);
  workspace->dual_relation_data.Reserve(relation_entries);
  workspace->dual_value_data.Reserve(value_entries);
  workspace->host_dual_tree.Resize(tree_size);
  workspace->host_dual_values.Resize(tree_size);
  Scalar *relation_cursor = workspace->dual_relation_data.get();
  Scalar *value_cursor = workspace->dual_value_data.get();
  for (int node = 0; node < tree_size; ++node) {
    BindDualRelationStorage(&workspace->host_dual_tree[node], &relation_cursor,
                            relation_capacity[node]);
    BindDualValueStorage(&workspace->host_dual_values[node], &value_cursor,
                         value_capacity[node]);
  }
  Require(relation_cursor ==
              workspace->dual_relation_data.get() + relation_entries,
          "internal CUDA dual-relation layout size mismatch");
  Require(value_cursor == workspace->dual_value_data.get() + value_entries,
          "internal CUDA dual-value layout size mismatch");
  workspace->dual_layout_key = std::move(key);
  return true;
}

void BuildTreeLevels(int leaf_count, std::vector<int> *offsets,
                     std::vector<int> *counts) {
  offsets->assign(1, 0);
  counts->assign(1, leaf_count);
  int tree_size = leaf_count;
  while (counts->back() > 1) {
    offsets->push_back(tree_size);
    counts->push_back((counts->back() + 1) / 2);
    tree_size += counts->back();
  }
}

void PrepareProblemStructure(const Problem &problem, int device,
                             WorkspaceStorage *workspace) {
  workspace->structure_ready = false;
  RefreshScratchPlan(problem, &workspace->structure_key, &workspace->scratch);
  RequireScratchFits(workspace->scratch, device);
  const int stage_count = static_cast<int>(problem.stages.size());
  const int node_count = stage_count + 1;
  const CompactEntryCounts entries = CountCompactEntries(problem);
  workspace->Reserve(device, stage_count, node_count, entries.states,
                     entries.controls, entries.mixed, entries.state_constraints,
                     entries.problem_data,
                     static_cast<int>(problem.initial_state.size()),
                     static_cast<int>(problem.terminal_E.rows()));
  BuildCompactOffsets(problem, workspace);
  PrepareStageStorage(problem, workspace);
  BuildTreeLevels(node_count, &workspace->node_level_offsets,
                  &workspace->node_level_counts);
  BuildTreeLevels(std::max(stage_count, 1), &workspace->stage_level_offsets,
                  &workspace->stage_level_counts);
  if (PrepareRelationStorage(problem, workspace))
    workspace->relation_layout_uploaded = false;
  workspace->structure_ready = true;
}

Scalar ObjectiveFromCompact(const Problem &problem, const int *state_offsets,
                            const int *control_offsets, const Scalar *states,
                            const Scalar *controls) {
  Scalar objective = Scalar{0};
  for (std::size_t i = 0; i < problem.stages.size(); ++i) {
    const Stage &s = problem.stages[i];
    const Scalar *x = states + state_offsets[i];
    const Scalar *u = controls + control_offsets[i];
    for (std::size_t row = 0; row < s.Q.rows(); ++row) {
      objective += s.q[row] * x[row];
      for (std::size_t col = 0; col < s.Q.cols(); ++col)
        objective += Scalar{0.5} * x[row] * s.Q(row, col) * x[col];
      for (std::size_t col = 0; col < s.M.cols(); ++col)
        objective += x[row] * s.M(row, col) * u[col];
    }
    for (std::size_t row = 0; row < s.R.rows(); ++row) {
      objective += s.r[row] * u[row];
      for (std::size_t col = 0; col < s.R.cols(); ++col)
        objective += Scalar{0.5} * u[row] * s.R(row, col) * u[col];
    }
  }
  const Scalar *x = states + state_offsets[problem.stages.size()];
  for (std::size_t row = 0; row < problem.terminal_Q.rows(); ++row) {
    objective += problem.terminal_q[row] * x[row];
    for (std::size_t col = 0; col < problem.terminal_Q.cols(); ++col)
      objective += Scalar{0.5} * x[row] * problem.terminal_Q(row, col) * x[col];
  }
  return objective;
}

Solution &SolveImpl(const Problem &problem, WorkspaceStorage &workspace,
                    Solution &solution, const Options &options) {
  const bool structure_matches =
      ValidateCudaProblem(problem, options, workspace.structure_key, false);
  int device_count = 0;
  CudaCheck(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  Require(options.device < device_count, "CUDA device index is out of range");
  if (workspace.structure_ready) {
    Require(workspace.device == options.device,
            "a CUDA workspace cannot be reused across devices");
  }
  CudaCheck(cudaSetDevice(options.device), "cudaSetDevice");
  if (!workspace.structure_ready || !structure_matches)
    PrepareProblemStructure(problem, options.device, &workspace);
  const ScratchRequirements &scratch = workspace.scratch;
  const auto total_start = std::chrono::steady_clock::now();
  solution.status = SolveStatus::kInvalidInput;
  solution.message.clear();
  solution.objective = Scalar{0};
  solution.timings = Timings{};
  const int stage_count = static_cast<int>(problem.stages.size());
  const int node_count = stage_count + 1;
  auto &level_offsets = workspace.node_level_offsets;
  auto &level_counts = workspace.node_level_counts;
  const int feasibility_scan_levels = static_cast<int>(level_counts.size()) - 1;
  auto &stage_level_offsets = workspace.stage_level_offsets;
  auto &stage_level_counts = workspace.stage_level_counts;
  auto &host_stages = workspace.host_stages;
  Scalar *host_problem_cursor = workspace.host_problem_data.data();
  Scalar *device_problem_cursor = workspace.device_problem_data.get();
  bool finite = true;
  for (std::size_t index = 0; index < problem.stages.size(); ++index) {
    finite &= PackStage(problem.stages[index], &host_problem_cursor,
                        &device_problem_cursor, &host_stages[index]);
  }
  PackedTerminal &terminal = workspace.host_terminal[0];
  finite &= PackTerminal(problem, &host_problem_cursor, &device_problem_cursor,
                         &terminal);
  Require(host_problem_cursor == workspace.host_problem_data.data() +
                                     workspace.host_problem_data.size(),
          "internal CUDA problem-packing size mismatch");
  auto &host_initial = workspace.host_initial;
  for (std::size_t i = 0; i < problem.initial_state.size(); ++i) {
    const Scalar value = problem.initial_state[i];
    host_initial[i] = value;
    finite &= std::isfinite(value);
  }
  Require(finite, "problem contains a non-finite value");

  auto &device_stages = workspace.device_stages;
  auto &device_terminal = workspace.device_terminal;
  auto &device_initial = workspace.device_initial;
  auto &device_status = workspace.device_status;
  auto &state_offsets = workspace.state_offsets;
  auto &reduced_state_offsets = workspace.reduced_state_offsets;
  auto &control_offsets = workspace.control_offsets;
  auto &dynamics_offsets = workspace.dynamics_offsets;
  auto &mixed_offsets = workspace.mixed_offsets;
  auto &state_constraint_offsets = workspace.state_constraint_offsets;
  auto &relation_a = workspace.relation_leaves;
  auto &relation_b = workspace.relation_scan;
  auto &state_params = workspace.state_params;
  auto &control_params = workspace.control_params;
  auto &state_dimensions = workspace.state_dimensions;
  auto &control_dimensions = workspace.control_dimensions;
  auto &reduced_stages = workspace.reduced_stages;
  auto &reduced_terminal = workspace.reduced_terminal;
  auto &reduced_initial = workspace.reduced_initial;
  const bool stage_layout_upload = !workspace.stage_layout_uploaded;
  const bool relation_layout_upload = !workspace.relation_layout_uploaded;

  solution.timings.upload_ms = TimeGpu(workspace, [&] {
    if (relation_layout_upload) {
      CudaCheck(cudaMemcpyAsync(
                    relation_a.get(), workspace.host_relation_leaves.data(),
                    workspace.host_relation_leaves.size() * sizeof(Relation),
                    cudaMemcpyHostToDevice),
                "upload compact relation leaves");
      if (workspace.host_relation_scan.size() > 0) {
        CudaCheck(cudaMemcpyAsync(
                      relation_b.get(), workspace.host_relation_scan.data(),
                      workspace.host_relation_scan.size() * sizeof(Relation),
                      cudaMemcpyHostToDevice),
                  "upload compact relation tree");
      }
    }
    if (stage_count > 0) {
      CudaCheck(cudaMemcpyAsync(device_stages.get(), host_stages.data(),
                                host_stages.size() * sizeof(PackedStage),
                                cudaMemcpyHostToDevice),
                "upload stages");
    }
    if (stage_layout_upload) {
      if (stage_count > 0) {
        CudaCheck(cudaMemcpyAsync(control_params.get(),
                                  workspace.host_control_params.data(),
                                  workspace.host_control_params.size() *
                                      sizeof(ControlParam),
                                  cudaMemcpyHostToDevice),
                  "upload control-parameter layouts");
        CudaCheck(cudaMemcpyAsync(reduced_stages.get(),
                                  workspace.host_reduced_stages.data(),
                                  workspace.host_reduced_stages.size() *
                                      sizeof(ReducedStage),
                                  cudaMemcpyHostToDevice),
                  "upload reduced-stage layouts");
        CudaCheck(cudaMemcpyAsync(
                      workspace.feedback.get(), workspace.host_feedback.data(),
                      workspace.host_feedback.size() * sizeof(Feedback),
                      cudaMemcpyHostToDevice),
                  "upload feedback layouts");
        CudaCheck(cudaMemcpyAsync(workspace.dual_params.get(),
                                  workspace.host_dual_params.data(),
                                  workspace.host_dual_params.size() *
                                      sizeof(DualParam),
                                  cudaMemcpyHostToDevice),
                  "upload dual-parameter layouts");
        CudaCheck(cudaMemcpyAsync(workspace.state_dual_params.get(),
                                  workspace.host_state_dual_params.data(),
                                  workspace.host_state_dual_params.size() *
                                      sizeof(StateDualParam),
                                  cudaMemcpyHostToDevice),
                  "upload state-dual layouts");
      }
      CudaCheck(cudaMemcpyAsync(
                    state_params.get(), workspace.host_state_params.data(),
                    workspace.host_state_params.size() * sizeof(StateParam),
                    cudaMemcpyHostToDevice),
                "upload state-parameter layouts");
      CudaCheck(cudaMemcpyAsync(reduced_terminal.get(),
                                workspace.host_reduced_terminal.data(),
                                sizeof(ReducedTerminal),
                                cudaMemcpyHostToDevice),
                "upload reduced-terminal layout");
    }
    CudaCheck(cudaMemcpyAsync(device_terminal.get(), &terminal,
                              sizeof(PackedTerminal), cudaMemcpyHostToDevice),
              "upload terminal data");
    CudaCheck(
        cudaMemcpyAsync(workspace.device_problem_data.get(),
                        workspace.host_problem_data.data(),
                        workspace.host_problem_data.size() * sizeof(Scalar),
                        cudaMemcpyHostToDevice),
        "upload compact problem data");
    CudaCheck(cudaMemcpyAsync(device_initial.get(), host_initial.data(),
                              host_initial.size() * sizeof(Scalar),
                              cudaMemcpyHostToDevice),
              "upload initial state");
    CudaCheck(cudaMemcpyAsync(state_offsets.get(),
                              workspace.host_state_offsets.data(),
                              workspace.host_state_offsets.size() * sizeof(int),
                              cudaMemcpyHostToDevice),
              "upload state offsets");
    CudaCheck(cudaMemcpyAsync(
                  control_offsets.get(), workspace.host_control_offsets.data(),
                  workspace.host_control_offsets.size() * sizeof(int),
                  cudaMemcpyHostToDevice),
              "upload control offsets");
    CudaCheck(
        cudaMemcpyAsync(dynamics_offsets.get(),
                        workspace.host_dynamics_offsets.data(),
                        workspace.host_dynamics_offsets.size() * sizeof(int),
                        cudaMemcpyHostToDevice),
        "upload dynamics-multiplier offsets");
    CudaCheck(cudaMemcpyAsync(mixed_offsets.get(),
                              workspace.host_mixed_offsets.data(),
                              workspace.host_mixed_offsets.size() * sizeof(int),
                              cudaMemcpyHostToDevice),
              "upload mixed offsets");
    CudaCheck(cudaMemcpyAsync(state_constraint_offsets.get(),
                              workspace.host_state_constraint_offsets.data(),
                              workspace.host_state_constraint_offsets.size() *
                                  sizeof(int),
                              cudaMemcpyHostToDevice),
              "upload state-constraint offsets");
    CudaCheck(cudaMemsetAsync(device_status.get(), 0, sizeof(DeviceStatus)),
              "clear CUDA status");
  });
  workspace.stage_layout_uploaded = true;
  workspace.relation_layout_uploaded = true;

  Scalar feasibility_consistency_tolerance = std::max(
      options.tolerance, kMinimumFeasibilityConsistencyTolerance *
                             static_cast<Scalar>(feasibility_scan_levels + 2));
  solution.timings.feasibility_ms = TimeGpuKernels(
      workspace,
      [&] {
        // A failing block may cause other blocks to stop before publishing
        // their dimensions. The host downloads status and dimensions in one
        // synchronization, so keep the ignored-on-failure entries initialized.
        CudaCheck(cudaMemsetAsync(state_dimensions.get(), 0,
                                  workspace.host_state_dimensions.size() *
                                      sizeof(int)),
                  "initialize reduced state dimensions");
      },
      [&] {
        BuildPrimalLeavesKernel<<<node_count, kThreads, scratch.primal_leaf>>>(
            device_stages.get(), stage_count, device_terminal.get(),
            options.tolerance, feasibility_consistency_tolerance,
            relation_a.get(), device_status.get());
        if (node_count > 1) {
          const int first_parent_count = level_counts[1];
          ReduceRelationLeavesKernel<<<first_parent_count, kThreads,
                                       scratch.primal_relation>>>(
              relation_a.get(), node_count, first_parent_count,
              options.tolerance, feasibility_consistency_tolerance,
              relation_b.get(), device_status.get());
          for (std::size_t level = 1; level + 1 < level_counts.size();
               ++level) {
            ReduceRelationTreeLevelKernel<<<level_counts[level + 1], kThreads,
                                            scratch.primal_relation>>>(
                relation_b.get(), level_offsets[level] - node_count,
                level_offsets[level + 1] - node_count, level_counts[level],
                level_counts[level + 1], options.tolerance,
                feasibility_consistency_tolerance, device_status.get());
          }
          InitializeRelationContextRootKernel<<<1, kThreads>>>(
              relation_b.get(), level_offsets.back() - node_count);
          for (int level = static_cast<int>(level_counts.size()) - 2;
               level >= 1; --level) {
            ExpandRelationContextLevelKernel<<<
                level_counts[level + 1], kThreads, scratch.primal_relation>>>(
                relation_b.get(), level_offsets[level] - node_count,
                level_offsets[level + 1] - node_count, level_counts[level],
                level_counts[level + 1], options.tolerance,
                feasibility_consistency_tolerance, device_status.get());
          }
          FinalizeRelationSuffixFromParentsKernel<<<
              first_parent_count, kThreads, scratch.primal_relation_final>>>(
              relation_a.get(), node_count, relation_b.get(),
              first_parent_count, options.tolerance,
              feasibility_consistency_tolerance, device_status.get());
        }
        StateParamKernel<<<node_count, kThreads, scratch.state_parameter>>>(
            relation_a.get(), node_count, state_params.get(),
            state_dimensions.get(), device_status.get(), options.tolerance);
      },
      [&] {
        CudaCheck(cudaMemcpyAsync(workspace.host_status.data(),
                                  device_status.get(), sizeof(DeviceStatus),
                                  cudaMemcpyDeviceToHost),
                  "read feasibility status");
        CudaCheck(cudaMemcpyAsync(workspace.host_state_dimensions.data(),
                                  state_dimensions.get(),
                                  workspace.host_state_dimensions.size() *
                                      sizeof(int),
                                  cudaMemcpyDeviceToHost),
                  "download reduced state dimensions");
      });
  Relation *suffix = relation_a.get();
  DeviceStatus status = workspace.host_status[0];
  if (ApplyDeviceFailure(status, &solution)) {
    solution.timings.total_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - total_start)
            .count();
    return solution;
  }
  workspace.host_reduced_state_offsets[0] = 0;
  for (int index = 0; index < node_count; ++index) {
    workspace.host_reduced_state_offsets[index + 1] =
        workspace.host_reduced_state_offsets[index] +
        workspace.host_state_dimensions[2 * index + 1];
  }
  workspace.reduced_states.Reserve(
      workspace.host_reduced_state_offsets[node_count]);

  solution.timings.reduction_ms = TimeGpuKernels(
      workspace,
      [&] {
        CudaCheck(cudaMemcpyAsync(reduced_state_offsets.get(),
                                  workspace.host_reduced_state_offsets.data(),
                                  workspace.host_reduced_state_offsets.size() *
                                      sizeof(int),
                                  cudaMemcpyHostToDevice),
                  "upload reduced-state offsets");
        if (stage_count > 0) {
          CudaCheck(cudaMemsetAsync(control_dimensions.get(), 0,
                                    workspace.host_control_dimensions.size() *
                                        sizeof(int)),
                    "initialize reduced control dimensions");
        }
      },
      [&] {
        if (stage_count > 0) {
          ReduceStagesKernel<<<stage_count, kThreads,
                               scratch.stage_reduction>>>(
              device_stages.get(), suffix, state_params.get(), stage_count,
              options.tolerance, feasibility_consistency_tolerance,
              control_params.get(), reduced_stages.get(),
              control_dimensions.get(), device_status.get());
        }
        ReduceTerminalKernel<<<1, kThreads>>>(device_terminal.get(),
                                              state_params.get(), stage_count,
                                              reduced_terminal.get());
        InitialReducedStateKernel<<<1, kThreads>>>(
            state_params.get(), device_initial.get(), reduced_initial.get(),
            options.tolerance, device_status.get());
      },
      [&] {
        if (stage_count > 0) {
          CudaCheck(cudaMemcpyAsync(workspace.host_control_dimensions.data(),
                                    control_dimensions.get(),
                                    workspace.host_control_dimensions.size() *
                                        sizeof(int),
                                    cudaMemcpyDeviceToHost),
                    "download reduced control dimensions");
        }
        CudaCheck(cudaMemcpyAsync(workspace.host_status.data(),
                                  device_status.get(), sizeof(DeviceStatus),
                                  cudaMemcpyDeviceToHost),
                  "read reduction status");
      });
  status = workspace.host_status[0];
  if (ApplyDeviceFailure(status, &solution)) {
    solution.timings.total_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - total_start)
            .count();
    return solution;
  }

  const bool value_layout_changed =
      PrepareValueStorage(&workspace, stage_count);
  const bool map_layout_changed = PrepareMapStorage(&workspace, stage_count);

  auto &value_a = workspace.value_leaves;
  auto &value_b = workspace.value_scan;
  auto &feedback = workspace.feedback;
  const ValueElement *value_suffix = value_a.get();
  solution.timings.riccati_ms = TimeGpuKernels(
      workspace,
      [&] {
        if (value_layout_changed) {
          CudaCheck(cudaMemcpyAsync(value_a.get(),
                                    workspace.host_value_leaves.data(),
                                    workspace.host_value_leaves.size() *
                                        sizeof(ValueElement),
                                    cudaMemcpyHostToDevice),
                    "upload compact value leaves");
          if (workspace.host_value_scan.size() > 0) {
            CudaCheck(cudaMemcpyAsync(value_b.get(),
                                      workspace.host_value_scan.data(),
                                      workspace.host_value_scan.size() *
                                          sizeof(ValueElement),
                                      cudaMemcpyHostToDevice),
                      "upload compact value tree");
          }
        }
      },
      [&] {
        BuildValueElementsKernel<<<node_count, kThreads, scratch.value_leaf>>>(
            reduced_stages.get(), reduced_terminal.get(), stage_count,
            options.tolerance, value_a.get(), device_status.get());
        if (node_count > 1) {
          const int first_parent_count = level_counts[1];
          ReduceValueLeavesKernel<<<first_parent_count, kThreads,
                                    scratch.value_compose>>>(
              value_a.get(), node_count, first_parent_count, options.tolerance,
              device_status.get(), value_b.get());
          for (std::size_t level = 1; level + 1 < level_counts.size();
               ++level) {
            ReduceValueTreeLevelKernel<<<level_counts[level + 1], kThreads,
                                         scratch.value_compose>>>(
                value_b.get(), level_offsets[level] - node_count,
                level_offsets[level + 1] - node_count, level_counts[level],
                level_counts[level + 1], options.tolerance,
                device_status.get());
          }
          InitializeValueContextRootKernel<<<1, kThreads>>>(
              value_b.get(), level_offsets.back() - node_count);
          for (int level = static_cast<int>(level_counts.size()) - 2;
               level >= 1; --level) {
            ExpandValueContextLevelKernel<<<level_counts[level + 1], kThreads,
                                            scratch.value_compose>>>(
                value_b.get(), level_offsets[level] - node_count,
                level_offsets[level + 1] - node_count, level_counts[level],
                level_counts[level + 1], options.tolerance,
                device_status.get());
          }
          FinalizeValueSuffixFromParentsKernel<<<first_parent_count, kThreads,
                                                 scratch.value_finalize>>>(
              value_a.get(), node_count, value_b.get(), first_parent_count,
              options.tolerance, device_status.get());
        }
        if (stage_count > 0) {
          FeedbackKernel<<<stage_count, kThreads, scratch.feedback>>>(
              reduced_stages.get(), value_suffix, stage_count,
              options.tolerance, feedback.get(), device_status.get());
        }
      },
      [&] {
        CudaCheck(cudaMemcpyAsync(workspace.host_status.data(),
                                  device_status.get(), sizeof(DeviceStatus),
                                  cudaMemcpyDeviceToHost),
                  "read Riccati status");
      });

  status = workspace.host_status[0];
  if (ApplyDeviceFailure(status, &solution)) {
    solution.timings.total_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - total_start)
            .count();
    return solution;
  }

  auto &map_a = workspace.map_leaves;
  auto &map_b = workspace.map_scan;
  auto &reduced_states = workspace.reduced_states;
  auto &states = workspace.states;
  auto &controls = workspace.controls;
  AffineMap *prefix = map_a.get();
  solution.timings.reconstruction_ms = TimeGpuKernels(
      workspace,
      [&] {
        if (stage_count > 0 && map_layout_changed) {
          CudaCheck(cudaMemcpyAsync(
                        map_a.get(), workspace.host_map_leaves.data(),
                        workspace.host_map_leaves.size() * sizeof(AffineMap),
                        cudaMemcpyHostToDevice),
                    "upload compact affine leaves");
          if (workspace.host_map_scan.size() > 0) {
            CudaCheck(cudaMemcpyAsync(
                          map_b.get(), workspace.host_map_scan.data(),
                          workspace.host_map_scan.size() * sizeof(AffineMap),
                          cudaMemcpyHostToDevice),
                      "upload compact affine tree");
          }
        }
      },
      [&] {
        if (stage_count > 0) {
          InitializeAffineMapsKernel<<<stage_count, kThreads>>>(
              feedback.get(), stage_count, map_a.get());
          if (stage_count > 1) {
            const int first_parent_count = stage_level_counts[1];
            ReduceAffineLeavesKernel<<<first_parent_count, kThreads>>>(
                map_a.get(), stage_count, first_parent_count, map_b.get(),
                device_status.get());
            for (std::size_t level = 1; level + 1 < stage_level_counts.size();
                 ++level) {
              ReduceAffineTreeLevelKernel<<<stage_level_counts[level + 1],
                                            kThreads>>>(
                  map_b.get(), stage_level_offsets[level] - stage_count,
                  stage_level_offsets[level + 1] - stage_count,
                  stage_level_counts[level], stage_level_counts[level + 1],
                  device_status.get());
            }
            InitializeAffineContextRootKernel<<<1, kThreads>>>(
                map_b.get(), stage_level_offsets.back() - stage_count);
            for (int level = static_cast<int>(stage_level_counts.size()) - 2;
                 level >= 1; --level) {
              ExpandAffineContextLevelKernel<<<stage_level_counts[level + 1],
                                               kThreads>>>(
                  map_b.get(), stage_level_offsets[level] - stage_count,
                  stage_level_offsets[level + 1] - stage_count,
                  stage_level_counts[level], stage_level_counts[level + 1],
                  device_status.get());
            }
            FinalizeAffinePrefixFromParentsKernel<<<
                first_parent_count, kThreads, scratch.affine_finalize>>>(
                map_a.get(), stage_count, map_b.get(), first_parent_count,
                device_status.get());
          }
          prefix = map_a.get();
        }
        const int state_blocks = (node_count + kThreads - 1) / kThreads;
        ReconstructPrimalKernel<<<state_blocks, kThreads>>>(
            prefix, state_params.get(), control_params.get(), feedback.get(),
            reduced_initial.get(), reduced_state_offsets.get(),
            state_offsets.get(), control_offsets.get(), stage_count,
            reduced_states.get(), workspace.reduced_controls.get(),
            states.get(), controls.get());
      },
      [&] {
        CudaCheck(cudaMemcpyAsync(workspace.host_status.data(),
                                  device_status.get(), sizeof(DeviceStatus),
                                  cudaMemcpyDeviceToHost),
                  "read reconstruction status");
      });
  status = workspace.host_status[0];
  if (ApplyDeviceFailure(status, &solution)) {
    solution.timings.total_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - total_start)
            .count();
    return solution;
  }

  // First recover the part of each dynamics/mixed multiplier fixed by the
  // reduced costate and control stationarity.  Only genuinely free components
  // remain in the balanced relation tree; in the common full-column-rank case
  // that tree has zero-dimensional endpoints.  State-only and endpoint
  // multipliers then follow independently from state stationarity.
  const Scalar multiplier_rank_tolerance =
      std::max(options.tolerance, kMinimumMultiplierRankTolerance);
  const Scalar multiplier_consistency_tolerance =
      options.enforce_multiplier_consistency
          ? std::max(multiplier_rank_tolerance,
                     kMultiplierConsistencyTolerancePerTreeLevel *
                         stage_level_counts.size())
          : kScalarMax;
  auto &dual_params = workspace.dual_params;
  auto &dual_dimensions = workspace.dual_dimensions;
  auto &dual_scan_needed = workspace.dual_scan_needed;
  auto &state_dual_params = workspace.state_dual_params;
  auto &dual_tree = workspace.dual_tree;
  auto &dual_values = workspace.dual_values;
  auto &initial_multiplier = workspace.initial_multiplier;
  auto &dynamics_multipliers = workspace.dynamics_multipliers;
  auto &mixed_multipliers = workspace.mixed_multipliers;
  auto &state_multipliers = workspace.state_multipliers;
  auto &terminal_multiplier = workspace.terminal_multiplier;
  int &host_dual_scan_needed = workspace.host_dual_scan_needed[0];
  host_dual_scan_needed = 0;
  solution.timings.multiplier_ms = 0.0;
  if (stage_count > 0) {
    solution.timings.multiplier_ms += TimeGpuKernels(
        workspace,
        [&] {
          CudaCheck(cudaMemsetAsync(dual_scan_needed.get(), 0, sizeof(int)),
                    "initialize dual scan flag");
          CudaCheck(cudaMemsetAsync(dual_dimensions.get(), 0,
                                    workspace.host_dual_dimensions.size() *
                                        sizeof(int)),
                    "initialize dual dimensions");
        },
        [&] {
          BuildDualParametersKernel<<<stage_count, kThreads,
                                      scratch.dual_parameter>>>(
              device_stages.get(), state_params.get(), value_suffix,
              reduced_states.get(), states.get(), controls.get(),
              reduced_state_offsets.get(), state_offsets.get(),
              control_offsets.get(), stage_count, multiplier_rank_tolerance,
              multiplier_consistency_tolerance, dual_params.get(),
              dual_scan_needed.get(), dual_dimensions.get(),
              device_status.get());
        },
        [&] {
          CudaCheck(cudaMemcpyAsync(&host_dual_scan_needed,
                                    dual_scan_needed.get(), sizeof(int),
                                    cudaMemcpyDeviceToHost),
                    "read dual scan flag");
          CudaCheck(cudaMemcpyAsync(workspace.host_dual_dimensions.data(),
                                    dual_dimensions.get(),
                                    workspace.host_dual_dimensions.size() *
                                        sizeof(int),
                                    cudaMemcpyDeviceToHost),
                    "read dual dimensions");
          CudaCheck(cudaMemcpyAsync(workspace.host_status.data(),
                                    device_status.get(), sizeof(DeviceStatus),
                                    cudaMemcpyDeviceToHost),
                    "read dual parameter status");
        });
    status = workspace.host_status[0];
    if (ApplyDeviceFailure(status, &solution)) {
      solution.timings.total_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - total_start)
              .count();
      return solution;
    }
    const bool dual_layout_changed =
        host_dual_scan_needed != 0 ? PrepareDualStorage(&workspace, stage_count)
                                   : false;
    DualRelation *dual_relations =
        host_dual_scan_needed != 0 ? dual_tree.get() : nullptr;
    DualNodeValue *dual_leaf_values =
        host_dual_scan_needed != 0 ? dual_values.get() : nullptr;
    solution.timings.multiplier_ms += TimeGpuKernels(
        workspace,
        [&] {
          if (dual_layout_changed) {
            CudaCheck(cudaMemcpyAsync(dual_tree.get(),
                                      workspace.host_dual_tree.data(),
                                      workspace.host_dual_tree.size() *
                                          sizeof(DualRelation),
                                      cudaMemcpyHostToDevice),
                      "upload compact dual relation tree");
            CudaCheck(cudaMemcpyAsync(dual_values.get(),
                                      workspace.host_dual_values.data(),
                                      workspace.host_dual_values.size() *
                                          sizeof(DualNodeValue),
                                      cudaMemcpyHostToDevice),
                      "upload compact dual value tree");
          }
        },
        [&] {
          BuildDualParameterRelationsKernel<<<stage_count, kThreads,
                                              scratch.dual_relation_leaf>>>(
              device_stages.get(), device_terminal.get(), dual_params.get(),
              stage_count, states.get(), controls.get(), state_offsets.get(),
              control_offsets.get(), multiplier_rank_tolerance,
              multiplier_consistency_tolerance, dual_relations,
              dual_scan_needed.get(), state_dual_params.get(),
              device_status.get());
          if (host_dual_scan_needed != 0) {
            for (std::size_t level = 0; level + 1 < stage_level_counts.size();
                 ++level) {
              ReduceDualTreeLevelKernel<<<stage_level_counts[level + 1],
                                          kThreads, scratch.dual_relation>>>(
                  dual_tree.get(), stage_level_offsets[level],
                  stage_level_offsets[level + 1], stage_level_counts[level],
                  stage_level_counts[level + 1], multiplier_rank_tolerance,
                  multiplier_consistency_tolerance, dual_tree.get(),
                  dual_scan_needed.get(), device_status.get());
            }
            const int root_offset = stage_level_offsets.back();
            SolveDualRootKernel<<<1, kThreads, scratch.dual_root>>>(
                dual_tree.get() + root_offset, dual_values.get() + root_offset,
                dual_scan_needed.get(), device_status.get(),
                multiplier_rank_tolerance);
            for (int level = static_cast<int>(stage_level_counts.size()) - 2;
                 level >= 0; --level) {
              ExpandDualTreeLevelKernel<<<stage_level_counts[level + 1],
                                          kThreads, scratch.dual_expand>>>(
                  dual_tree.get(), stage_level_offsets[level],
                  stage_level_offsets[level + 1], stage_level_counts[level],
                  stage_level_counts[level + 1], multiplier_rank_tolerance,
                  multiplier_consistency_tolerance, dual_values.get(),
                  dual_values.get(), dual_scan_needed.get(),
                  device_status.get());
            }
          }
          const int recovery_blocks = (stage_count + kThreads - 1) / kThreads;
          RecoverParameterizedMultipliersKernel<<<recovery_blocks, kThreads>>>(
              dual_params.get(), state_dual_params.get(), dual_leaf_values,
              dynamics_offsets.get(), mixed_offsets.get(),
              state_constraint_offsets.get(), stage_count,
              dynamics_multipliers.get(), mixed_multipliers.get(),
              state_multipliers.get(), terminal_multiplier.get());
          RecoverInitialMultiplierKernel<<<1, kThreads>>>(
              device_stages.get(), device_terminal.get(), stage_count,
              states.get(), controls.get(), dynamics_multipliers.get(),
              mixed_multipliers.get(), state_offsets.get(),
              control_offsets.get(), dynamics_offsets.get(),
              mixed_offsets.get(), state_constraint_offsets.get(),
              initial_multiplier.get(), state_multipliers.get(),
              terminal_multiplier.get());
        },
        [&] {
          CudaCheck(cudaMemcpyAsync(workspace.host_status.data(),
                                    device_status.get(), sizeof(DeviceStatus),
                                    cudaMemcpyDeviceToHost),
                    "read multiplier status");
        });
  }
  if (stage_count == 0) {
    solution.timings.multiplier_ms += TimeGpuKernels(
        workspace, [] {},
        [&] {
          RecoverInitialMultiplierKernel<<<1, kThreads>>>(
              device_stages.get(), device_terminal.get(), stage_count,
              states.get(), controls.get(), dynamics_multipliers.get(),
              mixed_multipliers.get(), state_offsets.get(),
              control_offsets.get(), dynamics_offsets.get(),
              mixed_offsets.get(), state_constraint_offsets.get(),
              initial_multiplier.get(), state_multipliers.get(),
              terminal_multiplier.get());
        },
        [&] {
          CudaCheck(cudaMemcpyAsync(workspace.host_status.data(),
                                    device_status.get(), sizeof(DeviceStatus),
                                    cudaMemcpyDeviceToHost),
                    "read multiplier status");
        });
  }
  status = workspace.host_status[0];
  if (ApplyDeviceFailure(status, &solution)) {
    solution.timings.total_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - total_start)
            .count();
    return solution;
  }

  auto &host_states = workspace.host_states;
  auto &host_controls = workspace.host_controls;
  auto &host_initial_multiplier = workspace.host_initial_multiplier;
  auto &host_dynamics = workspace.host_dynamics;
  auto &host_mixed = workspace.host_mixed;
  auto &host_state_multipliers = workspace.host_state_multipliers;
  auto &host_terminal_multiplier = workspace.host_terminal_multiplier;
  auto &host_state_dimensions = workspace.host_state_dimensions;
  auto &host_control_dimensions = workspace.host_control_dimensions;
  solution.timings.download_ms = TimeGpu(workspace, [&] {
    CudaCheck(cudaMemcpyAsync(host_states.data(), states.get(),
                              host_states.size() * sizeof(Scalar),
                              cudaMemcpyDeviceToHost),
              "download states");
    CudaCheck(cudaMemcpyAsync(host_controls.data(), controls.get(),
                              host_controls.size() * sizeof(Scalar),
                              cudaMemcpyDeviceToHost),
              "download controls");
    CudaCheck(cudaMemcpyAsync(host_initial_multiplier.data(),
                              initial_multiplier.get(),
                              host_initial_multiplier.size() * sizeof(Scalar),
                              cudaMemcpyDeviceToHost),
              "download initial multiplier");
    CudaCheck(cudaMemcpyAsync(host_dynamics.data(), dynamics_multipliers.get(),
                              host_dynamics.size() * sizeof(Scalar),
                              cudaMemcpyDeviceToHost),
              "download dynamics multipliers");
    CudaCheck(cudaMemcpyAsync(host_mixed.data(), mixed_multipliers.get(),
                              host_mixed.size() * sizeof(Scalar),
                              cudaMemcpyDeviceToHost),
              "download mixed multipliers");
    CudaCheck(cudaMemcpyAsync(host_state_multipliers.data(),
                              state_multipliers.get(),
                              host_state_multipliers.size() * sizeof(Scalar),
                              cudaMemcpyDeviceToHost),
              "download state multipliers");
    CudaCheck(cudaMemcpyAsync(host_terminal_multiplier.data(),
                              terminal_multiplier.get(),
                              host_terminal_multiplier.size() * sizeof(Scalar),
                              cudaMemcpyDeviceToHost),
              "download terminal multiplier");
  });

  solution.states.resize(node_count);
  solution.reduced_state_dimensions.resize(node_count);
  for (int i = 0; i < node_count; ++i) {
    const int n = host_state_dimensions[2 * i];
    solution.states[i].resize(n);
    for (int row = 0; row < n; ++row)
      solution.states[i][row] =
          host_states[workspace.host_state_offsets[i] + row];
    solution.reduced_state_dimensions[i] = host_state_dimensions[2 * i + 1];
  }
  solution.controls.resize(stage_count);
  solution.reduced_control_dimensions.resize(stage_count);
  solution.dynamics_multipliers.resize(stage_count);
  solution.mixed_multipliers.resize(stage_count);
  solution.state_multipliers.resize(stage_count);
  for (int i = 0; i < stage_count; ++i) {
    const PackedStage &s = host_stages[i];
    solution.controls[i].resize(s.m);
    for (int row = 0; row < s.m; ++row)
      solution.controls[i][row] =
          host_controls[workspace.host_control_offsets[i] + row];
    solution.reduced_control_dimensions[i] = host_control_dimensions[2 * i + 1];
    solution.dynamics_multipliers[i].resize(s.next_n);
    for (int row = 0; row < s.next_n; ++row)
      solution.dynamics_multipliers[i][row] =
          host_dynamics[workspace.host_dynamics_offsets[i] + row];
    solution.mixed_multipliers[i].resize(s.mixed);
    for (int row = 0; row < s.mixed; ++row)
      solution.mixed_multipliers[i][row] =
          host_mixed[workspace.host_mixed_offsets[i] + row];
    solution.state_multipliers[i].resize(s.state);
    for (int row = 0; row < s.state; ++row)
      solution.state_multipliers[i][row] =
          host_state_multipliers[workspace.host_state_constraint_offsets[i] +
                                 row];
  }
  solution.initial_multiplier.resize(problem.initial_state.size());
  for (std::size_t row = 0; row < problem.initial_state.size(); ++row)
    solution.initial_multiplier[row] = host_initial_multiplier[row];
  solution.terminal_state_multiplier.resize(terminal.state);
  for (int row = 0; row < terminal.state; ++row)
    solution.terminal_state_multiplier[row] = host_terminal_multiplier[row];
  solution.objective =
      ObjectiveFromCompact(problem, workspace.host_state_offsets.data(),
                           workspace.host_control_offsets.data(),
                           host_states.data(), host_controls.data());
  solution.status = SolveStatus::kOptimal;
  solution.message = "optimal (parallel CUDA conditional-value scan)";
  if (!options.enforce_multiplier_consistency)
    solution.message += "; multiplier consistency unchecked";
  solution.timings.total_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - total_start)
          .count();
  return solution;
}

} // namespace
} // namespace detail

struct Workspace::Impl {
  detail::WorkspaceStorage storage;
};

Workspace::Workspace() : impl_(std::make_unique<Impl>()) {}
Workspace::~Workspace() {
  if (impl_ && impl_->storage.device >= 0)
    cudaSetDevice(impl_->storage.device);
}
Workspace::Workspace(Workspace &&) noexcept = default;
Workspace &Workspace::operator=(Workspace &&other) noexcept {
  if (this != &other) {
    if (impl_ && impl_->storage.device >= 0)
      cudaSetDevice(impl_->storage.device);
    impl_ = std::move(other.impl_);
  }
  return *this;
}

void Workspace::Reserve(const Problem &problem, const Options &options) {
  if (!impl_)
    impl_ = std::make_unique<Impl>();
  const bool structure_matches = detail::ValidateCudaProblem(
      problem, options, impl_->storage.structure_key, true);
  int device_count = 0;
  detail::CudaCheck(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  detail::Require(options.device < device_count,
                  "CUDA device index is out of range");
  if (impl_->storage.structure_ready) {
    detail::Require(impl_->storage.device == options.device,
                    "a CUDA workspace cannot be reused across devices");
  }
  detail::CudaCheck(cudaSetDevice(options.device), "cudaSetDevice");
  if (!impl_->storage.structure_ready || !structure_matches) {
    detail::PrepareProblemStructure(problem, options.device, &impl_->storage);
  }
}

bool Available() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::string DeviceDescription(int device) {
  cudaDeviceProp properties{};
  const cudaError_t error = cudaGetDeviceProperties(&properties, device);
  if (error != cudaSuccess)
    return std::string("CUDA unavailable: ") + cudaGetErrorString(error);
  std::ostringstream out;
  out << properties.name << " (compute " << properties.major << "."
      << properties.minor << ", "
      << static_cast<double>(properties.totalGlobalMem) / (1024.0 * 1024.0)
      << " MiB)";
  return out.str();
}

Solution &Solve(const Problem &problem, Workspace &workspace, Solution &result,
                const Options &options) {
  try {
    if (!Available()) {
      result.status = SolveStatus::kInvalidInput;
      result.message = "no CUDA device is available";
      return result;
    }
    if (!workspace.impl_)
      workspace.impl_ = std::make_unique<Workspace::Impl>();
    return detail::SolveImpl(problem, workspace.impl_->storage, result,
                             options);
  } catch (const std::invalid_argument &error) {
    result.status = SolveStatus::kInvalidInput;
    result.message = error.what();
    return result;
  } catch (const std::exception &error) {
    result.status = SolveStatus::kNumericalFailure;
    result.message = error.what();
    return result;
  }
}

Solution Solve(const Problem &problem, const Options &options) {
  Workspace workspace;
  Solution result;
  Solve(problem, workspace, result, options);
  return result;
}

} // namespace cuda
} // namespace clqr
#endif // CLQR_CUDA_EMULATION

namespace clqr {
namespace cuda {
namespace detail {
namespace {

__global__ void InitializeAffineMapsKernel(const Feedback *feedback, int count,
                                           AffineMap *maps) {
  const int index = blockIdx.x;
  if (index >= count)
    return;
  const Feedback &fb = feedback[index];
  if (threadIdx.x == 0) {
    maps[index].left_dim = fb.state_dim;
    maps[index].right_dim = fb.next_state_dim;
  }
  for (int linear = threadIdx.x; linear < fb.next_state_dim * fb.state_dim;
       linear += blockDim.x) {
    const int row = linear / fb.state_dim;
    const int col = linear % fb.state_dim;
    maps[index].linear[row * fb.state_dim + col] =
        fb.transition[row * fb.state_dim + col];
  }
  for (int row = threadIdx.x; row < fb.next_state_dim; row += blockDim.x)
    maps[index].offset[row] = fb.offset[row];
}

__device__ void CopyAffineMapBlock(const AffineMap &input, AffineMap *output) {
  if (threadIdx.x == 0) {
    output->left_dim = input.left_dim;
    output->right_dim = input.right_dim;
  }
  for (int linear = threadIdx.x; linear < input.right_dim * input.left_dim;
       linear += blockDim.x) {
    const int row = linear / input.left_dim;
    const int col = linear % input.left_dim;
    output->linear[row * input.left_dim + col] =
        input.linear[row * input.left_dim + col];
  }
  for (int row = threadIdx.x; row < input.right_dim; row += blockDim.x)
    output->offset[row] = input.offset[row];
}

__device__ bool InvalidScanAffineMap(const AffineMap &map) {
  return map.left_dim < 0;
}

__device__ void SetInvalidScanAffineMap(AffineMap *map) {
  if (threadIdx.x == 0) {
    map->left_dim = -1;
    map->right_dim = 0;
  }
}

__device__ void ComposeAffineMapsBlock(const AffineMap &first,
                                       const AffineMap &second,
                                       AffineMap *output, DeviceStatus *status,
                                       int index) {
  if (InvalidScanAffineMap(first)) {
    if (InvalidScanAffineMap(second)) {
      SetInvalidScanAffineMap(output);
    } else {
      CopyAffineMapBlock(second, output);
    }
    return;
  }
  if (InvalidScanAffineMap(second)) {
    CopyAffineMapBlock(first, output);
    return;
  }
  if (first.right_dim != second.left_dim) {
    SetFailure(status, kDeviceNumericalFailure, index, 11);
    return;
  }
  if (threadIdx.x == 0) {
    output->left_dim = first.left_dim;
    output->right_dim = second.right_dim;
  }
  for (int linear = threadIdx.x; linear < second.right_dim * first.left_dim;
       linear += blockDim.x) {
    const int row = linear / first.left_dim;
    const int col = linear % first.left_dim;
    Scalar value = Scalar{0};
    for (int k = 0; k < first.right_dim; ++k) {
      value += second.linear[row * second.left_dim + k] *
               first.linear[k * first.left_dim + col];
    }
    output->linear[row * first.left_dim + col] = value;
  }
  for (int row = threadIdx.x; row < second.right_dim; row += blockDim.x) {
    Scalar value = second.offset[row];
    for (int k = 0; k < first.right_dim; ++k) {
      value += second.linear[row * second.left_dim + k] * first.offset[k];
    }
    output->offset[row] = value;
  }
}

__global__ void ReduceAffineLeavesKernel(const AffineMap *leaves, int count,
                                         int parent_count, AffineMap *parents,
                                         DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index >= parent_count)
    return;
  if (!BlockEnabled(status))
    return;
  const int left = 2 * index;
  if (left + 1 >= count) {
    CopyAffineMapBlock(leaves[left], &parents[index]);
    return;
  }
  ComposeAffineMapsBlock(leaves[left], leaves[left + 1], &parents[index],
                         status, index);
}

__global__ void ReduceAffineTreeLevelKernel(AffineMap *tree, int child_offset,
                                            int parent_offset, int child_count,
                                            int parent_count,
                                            DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index >= parent_count)
    return;
  if (!BlockEnabled(status))
    return;
  const int left = child_offset + 2 * index;
  if (2 * index + 1 >= child_count) {
    CopyAffineMapBlock(tree[left], &tree[parent_offset + index]);
    return;
  }
  const int right = left + 1;
  ComposeAffineMapsBlock(tree[left], tree[right], &tree[parent_offset + index],
                         status, index);
}

__global__ void InitializeAffineContextRootKernel(AffineMap *tree,
                                                  int root_offset) {
  if (blockIdx.x == 0)
    SetInvalidScanAffineMap(&tree[root_offset]);
}

__global__ void
ExpandAffineContextLevelKernel(AffineMap *tree, int child_offset,
                               int parent_offset, int child_count,
                               int parent_count, DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index >= parent_count)
    return;
  if (!BlockEnabled(status))
    return;
  const int left = child_offset + 2 * index;
  const AffineMap &parent_context = tree[parent_offset + index];
  if (2 * index + 1 >= child_count) {
    CopyAffineMapBlock(parent_context, &tree[left]);
    return;
  }
  const int right = left + 1;
  ComposeAffineMapsBlock(parent_context, tree[left], &tree[right], status,
                         index);
  WarpSynchronize();
  CopyAffineMapBlock(parent_context, &tree[left]);
}

__global__ void
FinalizeAffinePrefixFromParentsKernel(AffineMap *leaves, int count,
                                      const AffineMap *parent_contexts,
                                      int parent_count, DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index >= parent_count)
    return;
  if (!BlockEnabled(status))
    return;
  const int left = 2 * index;
  const AffineMap &parent = parent_contexts[index];
  const int right = left + 1;
  int left_capacity = leaves[left].left_dim;
  int right_capacity = leaves[left].right_dim;
  if (!InvalidScanAffineMap(parent))
    left_capacity = DeviceMax(left_capacity, parent.left_dim);
  if (left + 1 < count)
    right_capacity = DeviceMax(right_capacity, leaves[right].right_dim);
  ScratchSize scratch_size;
  scratch_size.Add<AffineMap>(1);
  scratch_size.Add<Scalar>(static_cast<std::size_t>(left_capacity) *
                               right_capacity +
                           right_capacity);
  CLQR_BLOCK_SCRATCH(scratch, scratch_size.bytes);
  AffineMap *composed_ptr = scratch.Take<AffineMap>(1);
  Scalar *composed_storage = scratch.Take<Scalar>(
      static_cast<std::size_t>(left_capacity) * right_capacity +
      right_capacity);
  AffineMap &composed = *composed_ptr;
  if (threadIdx.x == 0)
    BindAffineMapScratch(&composed, composed_storage, left_capacity,
                         right_capacity);
  WarpSynchronize();
  if (left + 1 >= count) {
    if (!InvalidScanAffineMap(parent)) {
      ComposeAffineMapsBlock(parent, leaves[left], &composed, status, left);
      WarpSynchronize();
      CopyAffineMapBlock(composed, &leaves[left]);
    }
    return;
  }
  if (!InvalidScanAffineMap(parent)) {
    ComposeAffineMapsBlock(parent, leaves[left], &composed, status, left);
    WarpSynchronize();
    CopyAffineMapBlock(composed, &leaves[left]);
    WarpSynchronize();
  }
  ComposeAffineMapsBlock(leaves[left], leaves[right], &composed, status, right);
  WarpSynchronize();
  CopyAffineMapBlock(composed, &leaves[right]);
}

__global__ void ReconstructPrimalKernel(
    const AffineMap *prefix, const StateParam *state_params,
    const ControlParam *control_params, const Feedback *feedback,
    const Scalar *initial, const int *reduced_state_offsets,
    const int *state_offsets, const int *control_offsets, int stage_count,
    Scalar *reduced_states, Scalar *reduced_controls, Scalar *states,
    Scalar *controls) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index > stage_count)
    return;
  const StateParam &state = state_params[index];
  Scalar *z = reduced_states + reduced_state_offsets[index];
  if (index == 0) {
    for (int col = 0; col < state.reduced_dim; ++col)
      z[col] = initial[col];
  } else {
    const AffineMap &map = prefix[index - 1];
    for (int row = 0; row < map.right_dim; ++row) {
      Scalar value = map.offset[row];
      for (int col = 0; col < map.left_dim; ++col)
        value += map.linear[row * map.left_dim + col] * initial[col];
      z[row] = value;
    }
  }
  for (int x = 0; x < state.physical_dim; ++x) {
    Scalar value = state.t[x];
    for (int col = 0; col < state.reduced_dim; ++col) {
      value += state.T[x * state.reduced_dim + col] * z[col];
    }
    states[state_offsets[index] + x] = value;
  }
  if (index == stage_count)
    return;
  const ControlParam &control = control_params[index];
  const Feedback &fb = feedback[index];
  Scalar *v = reduced_controls + control_offsets[index];
  for (int row = 0; row < control.reduced_dim; ++row) {
    Scalar value = fb.k[row];
    for (int col = 0; col < fb.state_dim; ++col) {
      value += fb.K[row * fb.state_dim + col] * z[col];
    }
    v[row] = value;
  }
  for (int u = 0; u < control.physical_dim; ++u) {
    Scalar value = control.y[u];
    for (int col = 0; col < control.state_dim; ++col) {
      value += control.Y[u * control.state_dim + col] * z[col];
    }
    for (int col = 0; col < control.reduced_dim; ++col) {
      value += control.Z[u * control.reduced_dim + col] * v[col];
    }
    controls[control_offsets[index] + u] = value;
  }
}

__global__ void BuildDualParametersKernel(
    const PackedStage *stages, const StateParam *state_params,
    const ValueElement *value_suffix, const Scalar *reduced_states,
    const Scalar *states, const Scalar *controls,
    const int *reduced_state_offsets, const int *state_offsets,
    const int *control_offsets, int stage_count, Scalar rank_tolerance,
    Scalar consistency_tolerance, DualParam *params, int *scan_needed,
    int *dual_dimensions, DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index >= stage_count)
    return;
  if (!BlockEnabled(status))
    return;
  const PackedStage &stage = stages[index];
  const StateParam &next = state_params[index + 1];
  const ValueElement &next_value = value_suffix[index + 1];
  const int variables = stage.next_n + stage.mixed;
  const int rows = next.reduced_dim + stage.m;
  const int columns = variables + 1;
  ScratchSize scratch_size;
  scratch_size.Add<Scalar>(static_cast<std::size_t>(rows) * columns);
  scratch_size.Add<Scalar>(stage.mixed);
  scratch_size.Add<Scalar>(rows);
  scratch_size.Add<Scalar>(static_cast<std::size_t>(variables) * variables);
  scratch_size.Add<Scalar>(variables);
  scratch_size.Add<Scalar>(variables);
  scratch_size.Add<int>(variables);
  CLQR_BLOCK_SCRATCH(scratch, scratch_size.bytes);
  Scalar *matrix =
      scratch.Take<Scalar>(static_cast<std::size_t>(rows) * columns);
  Scalar *constraint_scales = scratch.Take<Scalar>(stage.mixed);
  Scalar *residual_rhs = scratch.Take<Scalar>(rows);
  Scalar *upper =
      scratch.Take<Scalar>(static_cast<std::size_t>(variables) * variables);
  Scalar *rhs_projection = scratch.Take<Scalar>(variables);
  Scalar *solution = scratch.Take<Scalar>(variables);
  int *permutation = scratch.Take<int>(variables);
  __shared__ int rank;
  __shared__ int local_ok;
  __shared__ Scalar conditioned_rhs_scale;

  for (int entry = threadIdx.x; entry < rows * columns; entry += blockDim.x)
    matrix[entry] = Scalar{0};
  for (int constraint = threadIdx.x; constraint < stage.mixed;
       constraint += blockDim.x) {
    Scalar scale = Scalar{0};
    for (int state = 0; state < stage.n; ++state)
      scale = fmax(scale, DeviceAbs(stage.C[constraint * stage.n + state]));
    for (int control = 0; control < stage.m; ++control)
      scale = fmax(scale, DeviceAbs(stage.D[constraint * stage.m + control]));
    constraint_scales[constraint] = scale > Scalar{0} ? scale : Scalar{1};
  }
  WarpSynchronize();

  for (int linear = threadIdx.x; linear < next.reduced_dim * stage.next_n;
       linear += blockDim.x) {
    const int row = linear / stage.next_n;
    const int state = linear % stage.next_n;
    matrix[row * columns + state] = next.T[state * next.reduced_dim + row];
  }
  const Scalar *next_reduced_state =
      reduced_states + reduced_state_offsets[index + 1];
  for (int row = threadIdx.x; row < next.reduced_dim; row += blockDim.x) {
    Scalar costate = -next_value.eta[row];
    for (int col = 0; col < next.reduced_dim; ++col)
      costate -= next_value.J[row * next_value.left_dim + col] *
                 next_reduced_state[col];
    matrix[row * columns + variables] = costate;
  }
  for (int linear = threadIdx.x; linear < stage.m * stage.next_n;
       linear += blockDim.x) {
    const int control = linear / stage.next_n;
    const int state = linear % stage.next_n;
    matrix[(next.reduced_dim + control) * columns + state] =
        stage.B[state * stage.m + control];
  }
  for (int linear = threadIdx.x; linear < stage.m * stage.mixed;
       linear += blockDim.x) {
    const int control = linear / stage.mixed;
    const int constraint = linear % stage.mixed;
    matrix[(next.reduced_dim + control) * columns + stage.next_n + constraint] =
        -stage.D[constraint * stage.m + control] /
        constraint_scales[constraint];
  }
  const Scalar *state = states + state_offsets[index];
  const Scalar *control = controls + control_offsets[index];
  for (int row = threadIdx.x; row < stage.m; row += blockDim.x) {
    Scalar gradient = stage.r[row];
    for (int col = 0; col < stage.n; ++col)
      gradient += stage.M[col * stage.m + row] * state[col];
    for (int col = 0; col < stage.m; ++col)
      gradient += stage.R[row * stage.m + col] * control[col];
    matrix[(next.reduced_dim + row) * columns + variables] = gradient;
  }
  WarpSynchronize();
  if (threadIdx.x == 0) {
    conditioned_rhs_scale =
        ConditionedRhsScale(matrix, rows, columns, variables, rank_tolerance);
  }
  WarpSynchronize();
  SolveSystemOrthogonally(matrix, rows, columns, variables, rank_tolerance,
                          consistency_tolerance, conditioned_rhs_scale,
                          residual_rhs, upper, rhs_projection, solution,
                          permutation, &rank, &local_ok);
  if (threadIdx.x == 0) {
    if (!local_ok) {
      SetFailure(status, kDeviceNumericalFailure, index, 24);
    } else {
      DualParam &out = params[index];
      out.state_dim = stage.next_n;
      out.mixed_dim = stage.mixed;
      out.physical_dim = variables;
      out.free_dim = variables - rank;
      if (dual_dimensions != nullptr)
        dual_dimensions[index] = out.free_dim;
      for (int free = 0; free < out.free_dim; ++free)
        out.free_columns[free] = permutation[rank + free];
      if (out.free_dim > 0)
        atomicExch(scan_needed, 1);
      for (int variable = 0; variable < variables; ++variable) {
        const Scalar scale = variable < stage.next_n
                                 ? Scalar{1}
                                 : constraint_scales[variable - stage.next_n];
        out.offset[variable] = solution[variable] / scale;
        for (int free = 0; free < out.free_dim; ++free)
          out.basis[variable * out.free_dim + free] = Scalar{0};
      }
      for (int free = 0; free < out.free_dim; ++free) {
        for (int position = 0; position < variables; ++position)
          solution[position] = Scalar{0};
        solution[rank + free] = Scalar{1};
        for (int reverse = 0; reverse < rank; ++reverse) {
          const int row = rank - 1 - reverse;
          Scalar value = -upper[row * variables + rank + free];
          for (int col = row + 1; col < rank; ++col) {
            value -= upper[row * variables + col] * solution[col];
          }
          solution[row] = value / upper[row * variables + row];
        }
        for (int position = 0; position < variables; ++position) {
          const int variable = permutation[position];
          const Scalar scale = variable < stage.next_n
                                   ? Scalar{1}
                                   : constraint_scales[variable - stage.next_n];
          out.basis[variable * out.free_dim + free] =
              solution[position] / scale;
        }
      }
    }
  }
  WarpSynchronize();
}

__global__ void BuildDualParameterRelationsKernel(
    const PackedStage *stages, const PackedTerminal *terminal_ptr,
    const DualParam *params, int stage_count, const Scalar *states,
    const Scalar *controls, const int *state_offsets,
    const int *control_offsets, Scalar rank_tolerance,
    Scalar consistency_tolerance, DualRelation *relations,
    const int *scan_needed, StateDualParam *state_params,
    DeviceStatus *status) {
  const int relation_index = blockIdx.x;
  if (relation_index >= stage_count)
    return;
  if (!BlockEnabled(status))
    return;
  const int node = relation_index + 1;
  const bool is_terminal = node == stage_count;
  const PackedTerminal &terminal = *terminal_ptr;
  const DualParam &left = params[node - 1];
  const DualParam *right = is_terminal ? nullptr : &params[node];
  const int state_dim = is_terminal ? terminal.n : stages[node].n;
  const int state_constraints =
      is_terminal ? terminal.state : stages[node].state;
  const int right_dim = is_terminal ? 0 : right->free_dim;
  const int rows = state_dim;
  const int columns = state_constraints + left.free_dim + right_dim + 1;
  const std::size_t scratch_bytes = DualRelationLeafScratchBytes(
      static_cast<std::size_t>(rows) * columns, rows, state_constraints);
  CLQR_BLOCK_SCRATCH(scratch, scratch_bytes);
  Scalar *matrix =
      scratch.Take<Scalar>(static_cast<std::size_t>(rows) * columns);
  Scalar *factors = scratch.Take<Scalar>(rows);
  Scalar *constraint_scales = scratch.Take<Scalar>(state_constraints);
  int *pivot_columns = scratch.Take<int>(rows);
  int *pivot_rows = scratch.Take<int>(rows);
  __shared__ int rank;
  __shared__ int best_row;
  __shared__ int local_ok;

  for (int entry = threadIdx.x; entry < rows * columns; entry += blockDim.x)
    matrix[entry] = Scalar{0};
  for (int constraint = threadIdx.x; constraint < state_constraints;
       constraint += blockDim.x) {
    Scalar scale = Scalar{0};
    for (int state = 0; state < state_dim; ++state) {
      const Scalar value =
          is_terminal ? terminal.E[constraint * terminal.n + state]
                      : stages[node].E[constraint * stages[node].n + state];
      scale = fmax(scale, DeviceAbs(value));
    }
    constraint_scales[constraint] = scale > Scalar{0} ? scale : Scalar{1};
  }
  WarpSynchronize();
  for (int linear = threadIdx.x; linear < rows * state_constraints;
       linear += blockDim.x) {
    const int state = linear / state_constraints;
    const int constraint = linear % state_constraints;
    const Scalar value =
        is_terminal ? terminal.E[constraint * terminal.n + state]
                    : stages[node].E[constraint * stages[node].n + state];
    matrix[state * columns + constraint] =
        value / constraint_scales[constraint];
  }
  for (int linear = threadIdx.x; linear < state_dim * left.free_dim;
       linear += blockDim.x) {
    const int state = linear / left.free_dim;
    const int free = linear % left.free_dim;
    matrix[state * columns + state_constraints + free] =
        left.basis[state * left.free_dim + free];
  }
  if (!is_terminal) {
    const PackedStage &stage = stages[node];
    for (int linear = threadIdx.x; linear < state_dim * right_dim;
         linear += blockDim.x) {
      const int state = linear / right_dim;
      const int free = linear % right_dim;
      Scalar value = Scalar{0};
      for (int next_state = 0; next_state < stage.next_n; ++next_state) {
        value -= stage.A[next_state * stage.n + state] *
                 right->basis[next_state * right_dim + free];
      }
      for (int constraint = 0; constraint < stage.mixed; ++constraint) {
        value += stage.C[constraint * stage.n + state] *
                 right->basis[(stage.next_n + constraint) * right_dim + free];
      }
      matrix[state * columns + state_constraints + left.free_dim + free] =
          value;
    }
  }
  const Scalar *state = states + state_offsets[node];
  for (int row = threadIdx.x; row < state_dim; row += blockDim.x) {
    Scalar rhs = -left.offset[row];
    if (is_terminal) {
      rhs -= terminal.q[row];
      for (int col = 0; col < terminal.n; ++col)
        rhs -= terminal.Q[row * terminal.n + col] * state[col];
    } else {
      const PackedStage &stage = stages[node];
      const Scalar *control = controls + control_offsets[node];
      rhs -= stage.q[row];
      for (int col = 0; col < stage.n; ++col)
        rhs -= stage.Q[row * stage.n + col] * state[col];
      for (int col = 0; col < stage.m; ++col)
        rhs -= stage.M[row * stage.m + col] * control[col];
      for (int next_state = 0; next_state < stage.next_n; ++next_state)
        rhs += stage.A[next_state * stage.n + row] * right->offset[next_state];
      for (int constraint = 0; constraint < stage.mixed; ++constraint) {
        rhs -= stage.C[constraint * stage.n + row] *
               right->offset[stage.next_n + constraint];
      }
    }
    matrix[row * columns + columns - 1] = rhs;
  }
  WarpSynchronize();
  // A redundant mixed-constraint multiplier is a true null direction. Its
  // contribution to state stationarity can cancel exactly or leave only
  // architecture-dependent roundoff, which must not become a new equation.
  RrefBlock(matrix, rows, columns, columns - 1, rank_tolerance, pivot_columns,
            pivot_rows, &rank, &best_row, factors,
            kMinimumDualRelationRowScale);
  if (threadIdx.x == 0) {
    local_ok = !InconsistentRref(matrix, rows, columns, columns - 1,
                                 rank_tolerance, consistency_tolerance);
    if (!local_ok) {
      SetFailure(status, kDeviceNumericalFailure, relation_index, 17);
    }
  }
  WarpSynchronize();
  if (!local_ok)
    return;
  if (threadIdx.x == 0) {
    StateDualParam &out = state_params[relation_index];
    out.constraint_dim = state_constraints;
    out.left_dim = left.free_dim;
    out.right_dim = right_dim;
    for (int constraint = 0; constraint < state_constraints; ++constraint) {
      out.offset[constraint] = Scalar{0};
      for (int free = 0; free < left.free_dim; ++free)
        out.left[constraint * left.free_dim + free] = Scalar{0};
      for (int free = 0; free < right_dim; ++free)
        out.right[constraint * right_dim + free] = Scalar{0};
    }
    for (int pivot = 0; pivot < rank; ++pivot) {
      const int constraint = pivot_columns[pivot];
      if (constraint >= state_constraints)
        break;
      const Scalar inverse_scale = Scalar{1} / constraint_scales[constraint];
      out.offset[constraint] =
          matrix[pivot * columns + columns - 1] * inverse_scale;
      for (int free = 0; free < left.free_dim; ++free) {
        out.left[constraint * left.free_dim + free] =
            -matrix[pivot * columns + state_constraints + free] * inverse_scale;
      }
      for (int free = 0; free < right_dim; ++free) {
        out.right[constraint * right_dim + free] =
            -matrix[pivot * columns + state_constraints + left.free_dim +
                    free] *
            inverse_scale;
      }
    }
  }
  WarpSynchronize();
  if (*scan_needed != 0) {
    ExtractResidualRelation(matrix, columns, rank, pivot_columns,
                            state_constraints, left.free_dim, right_dim,
                            &relations[relation_index]);
  }
}

__global__ void RecoverParameterizedMultipliersKernel(
    const DualParam *params, const StateDualParam *state_params,
    const DualNodeValue *leaf_values, const int *dynamics_offsets,
    const int *mixed_offsets, const int *state_constraint_offsets,
    int stage_count, Scalar *dynamics_multipliers, Scalar *mixed_multipliers,
    Scalar *state_multipliers, Scalar *terminal_multiplier) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= stage_count)
    return;
  const DualParam &param = params[index];
  for (int row = 0; row < param.physical_dim; ++row) {
    Scalar result = param.offset[row];
    for (int free = 0; free < param.free_dim; ++free) {
      result += param.basis[row * param.free_dim + free] *
                leaf_values[index].left[free];
    }
    if (row < param.state_dim) {
      dynamics_multipliers[dynamics_offsets[index] + row] = result;
    } else {
      mixed_multipliers[mixed_offsets[index] + row - param.state_dim] = result;
    }
  }
  const int node = index + 1;
  const StateDualParam &state_param = state_params[index];
  for (int constraint = 0; constraint < state_param.constraint_dim;
       ++constraint) {
    Scalar multiplier = state_param.offset[constraint];
    for (int free = 0; free < state_param.left_dim; ++free) {
      multiplier += state_param.left[constraint * state_param.left_dim + free] *
                    leaf_values[index].left[free];
    }
    for (int free = 0; free < state_param.right_dim; ++free) {
      multiplier +=
          state_param.right[constraint * state_param.right_dim + free] *
          leaf_values[index].right[free];
    }
    if (node == stage_count) {
      terminal_multiplier[constraint] = multiplier;
    } else {
      state_multipliers[state_constraint_offsets[node] + constraint] =
          multiplier;
    }
  }
}

__global__ void RecoverInitialMultiplierKernel(
    const PackedStage *stages, const PackedTerminal *terminal_ptr,
    int stage_count, const Scalar *states, const Scalar *controls,
    const Scalar *dynamics_multipliers, const Scalar *mixed_multipliers,
    const int *state_offsets, const int *control_offsets,
    const int *dynamics_offsets, const int *mixed_offsets,
    const int *state_constraint_offsets, Scalar *initial_multiplier,
    Scalar *state_multipliers, Scalar *terminal_multiplier) {
  if (blockIdx.x != 0)
    return;
  const PackedTerminal &terminal = *terminal_ptr;
  if (stage_count == 0) {
    for (int row = threadIdx.x; row < terminal.state; row += blockDim.x)
      terminal_multiplier[row] = Scalar{0};
    for (int row = threadIdx.x; row < terminal.n; row += blockDim.x) {
      Scalar value = -terminal.q[row];
      for (int col = 0; col < terminal.n; ++col)
        value -=
            terminal.Q[row * terminal.n + col] * states[state_offsets[0] + col];
      initial_multiplier[row] = value;
    }
    return;
  }
  const PackedStage &stage = stages[0];
  for (int row = threadIdx.x; row < stage.state; row += blockDim.x)
    state_multipliers[state_constraint_offsets[0] + row] = Scalar{0};
  for (int row = threadIdx.x; row < stage.n; row += blockDim.x) {
    Scalar value = -stage.q[row];
    for (int col = 0; col < stage.n; ++col)
      value -= stage.Q[row * stage.n + col] * states[state_offsets[0] + col];
    for (int col = 0; col < stage.m; ++col)
      value -=
          stage.M[row * stage.m + col] * controls[control_offsets[0] + col];
    for (int col = 0; col < stage.next_n; ++col)
      value += stage.A[col * stage.n + row] *
               dynamics_multipliers[dynamics_offsets[0] + col];
    for (int constraint = 0; constraint < stage.mixed; ++constraint) {
      value -= stage.C[constraint * stage.n + row] *
               mixed_multipliers[mixed_offsets[0] + constraint];
    }
    initial_multiplier[row] = value;
  }
}

__global__ void
ReduceDualTreeLevelKernel(const DualRelation *tree, int child_offset,
                          int parent_offset, int child_count, int parent_count,
                          Scalar rank_tolerance, Scalar consistency_tolerance,
                          DualRelation *mutable_tree, const int *scan_needed,
                          DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index >= parent_count || *scan_needed == 0)
    return;
  if (!BlockEnabled(status))
    return;
  if (2 * index + 1 >= child_count) {
    CopyRelationBlock(tree[child_offset + 2 * index],
                      &mutable_tree[parent_offset + index]);
    return;
  }
  const DualRelation &first = tree[child_offset + 2 * index];
  const DualRelation &second = tree[child_offset + 2 * index + 1];
  const int rows = first.rows + second.rows;
  const int columns = first.right_dim + first.left_dim + second.right_dim + 1;
  ScratchSize scratch_size;
  scratch_size.Add<Scalar>(static_cast<std::size_t>(rows) * columns);
  scratch_size.Add<Scalar>(rows);
  scratch_size.Add<int>(rows);
  scratch_size.Add<int>(rows);
  CLQR_BLOCK_SCRATCH(scratch, scratch_size.bytes);
  Scalar *matrix =
      scratch.Take<Scalar>(static_cast<std::size_t>(rows) * columns);
  Scalar *factors = scratch.Take<Scalar>(rows);
  int *pivot_columns = scratch.Take<int>(rows);
  int *pivot_rows = scratch.Take<int>(rows);
  __shared__ int rank;
  __shared__ int best_row;
  __shared__ int local_ok;
  ComposeRelationsBlock(first, second, rank_tolerance, consistency_tolerance,
                        &mutable_tree[parent_offset + index], status, index,
                        kDeviceNumericalFailure, 18, matrix, factors,
                        pivot_columns, pivot_rows, &rank, &best_row, &local_ok,
                        true);
}

__global__ void SolveDualRootKernel(const DualRelation *relation,
                                    DualNodeValue *value,
                                    const int *scan_needed,
                                    DeviceStatus *status, Scalar tolerance) {
  if (blockIdx.x != 0 || *scan_needed == 0)
    return;
  if (!BlockEnabled(status))
    return;
  if (relation->right_dim != 0) {
    if (threadIdx.x == 0)
      SetFailure(status, kDeviceNumericalFailure, 0, 13);
    return;
  }
  const int variables = relation->left_dim;
  const int columns = variables + 1;
  ScratchSize scratch_size;
  scratch_size.Add<Scalar>(static_cast<std::size_t>(relation->rows) * columns);
  scratch_size.Add<Scalar>(relation->rows);
  scratch_size.Add<Scalar>(static_cast<std::size_t>(variables) * variables);
  scratch_size.Add<Scalar>(variables);
  scratch_size.Add<Scalar>(variables);
  scratch_size.Add<int>(variables);
  CLQR_BLOCK_SCRATCH(scratch, scratch_size.bytes);
  Scalar *matrix =
      scratch.Take<Scalar>(static_cast<std::size_t>(relation->rows) * columns);
  Scalar *residual_rhs = scratch.Take<Scalar>(relation->rows);
  Scalar *upper =
      scratch.Take<Scalar>(static_cast<std::size_t>(variables) * variables);
  Scalar *rhs_projection = scratch.Take<Scalar>(variables);
  Scalar *solution = scratch.Take<Scalar>(variables);
  int *permutation = scratch.Take<int>(variables);
  __shared__ int rank;
  __shared__ int local_ok;
  __shared__ Scalar rhs_scale;
  for (int linear = threadIdx.x; linear < relation->rows * variables;
       linear += blockDim.x) {
    const int row = linear / variables;
    const int col = linear % variables;
    matrix[row * columns + col] =
        relation->left[row * relation->left_dim + col];
  }
  for (int row = threadIdx.x; row < relation->rows; row += blockDim.x)
    matrix[row * columns + variables] = relation->rhs[row];
  WarpSynchronize();
  if (threadIdx.x == 0)
    rhs_scale = ConditionedRhsScale(matrix, relation->rows, columns, variables,
                                    tolerance);
  WarpSynchronize();
  SolveSystemOrthogonally(matrix, relation->rows, columns, variables, tolerance,
                          tolerance, rhs_scale, residual_rhs, upper,
                          rhs_projection, solution, permutation, &rank,
                          &local_ok);
  if (!local_ok) {
    if (threadIdx.x == 0)
      SetFailure(status, kDeviceNumericalFailure, 0, 14);
    return;
  }
  if (threadIdx.x == 0) {
    value->left_dim = variables;
    value->right_dim = 0;
  }
  for (int col = threadIdx.x; col < variables; col += blockDim.x)
    value->left[col] = solution[col];
}

__global__ void ExpandDualTreeLevelKernel(
    const DualRelation *tree, int child_offset, int parent_offset,
    int child_count, int parent_count, Scalar rank_tolerance,
    Scalar consistency_tolerance, const DualNodeValue *parent_values,
    DualNodeValue *values, const int *scan_needed, DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index >= parent_count || *scan_needed == 0)
    return;
  if (!BlockEnabled(status))
    return;
  if (2 * index + 1 >= child_count) {
    const DualNodeValue &parent = parent_values[parent_offset + index];
    DualNodeValue &child = values[child_offset + 2 * index];
    if (threadIdx.x == 0) {
      child.left_dim = parent.left_dim;
      child.right_dim = parent.right_dim;
    }
    for (int entry = threadIdx.x; entry < parent.left_dim; entry += blockDim.x)
      child.left[entry] = parent.left[entry];
    for (int entry = threadIdx.x; entry < parent.right_dim; entry += blockDim.x)
      child.right[entry] = parent.right[entry];
    return;
  }
  const DualRelation &left = tree[child_offset + 2 * index];
  const DualRelation &right = tree[child_offset + 2 * index + 1];
  const DualNodeValue &parent = parent_values[parent_offset + index];
  if (left.left_dim != parent.left_dim || right.right_dim != parent.right_dim ||
      left.right_dim != right.left_dim) {
    if (threadIdx.x == 0)
      SetFailure(status, kDeviceNumericalFailure, index, 15);
    return;
  }
  const int shared = left.right_dim;
  const int rows = left.rows + right.rows;
  const int columns = shared + 1;
  ScratchSize scratch_size;
  scratch_size.Add<Scalar>(static_cast<std::size_t>(rows) * columns);
  scratch_size.Add<Scalar>(rows);
  scratch_size.Add<Scalar>(static_cast<std::size_t>(shared) * shared);
  scratch_size.Add<Scalar>(shared);
  scratch_size.Add<Scalar>(shared);
  scratch_size.Add<int>(shared);
  CLQR_BLOCK_SCRATCH(scratch, scratch_size.bytes);
  Scalar *matrix =
      scratch.Take<Scalar>(static_cast<std::size_t>(rows) * columns);
  Scalar *residual_rhs = scratch.Take<Scalar>(rows);
  Scalar *upper =
      scratch.Take<Scalar>(static_cast<std::size_t>(shared) * shared);
  Scalar *rhs_projection = scratch.Take<Scalar>(shared);
  Scalar *shared_solution = scratch.Take<Scalar>(shared);
  int *permutation = scratch.Take<int>(shared);
  __shared__ int rank;
  __shared__ int local_ok;
  __shared__ Scalar conditioned_rhs_scale;
  for (int i = threadIdx.x; i < rows * columns; i += blockDim.x)
    matrix[i] = Scalar{0};
  WarpSynchronize();
  for (int linear = threadIdx.x; linear < left.rows * shared;
       linear += blockDim.x) {
    const int row = linear / shared;
    const int col = linear % shared;
    matrix[row * columns + col] = left.right[row * left.right_dim + col];
  }
  for (int row = threadIdx.x; row < left.rows; row += blockDim.x) {
    Scalar rhs = left.rhs[row];
    for (int col = 0; col < left.left_dim; ++col) {
      rhs -= left.left[row * left.left_dim + col] * parent.left[col];
    }
    matrix[row * columns + shared] = rhs;
  }
  for (int linear = threadIdx.x; linear < right.rows * shared;
       linear += blockDim.x) {
    const int row = linear / shared;
    const int col = linear % shared;
    matrix[(left.rows + row) * columns + col] =
        right.left[row * right.left_dim + col];
  }
  for (int row = threadIdx.x; row < right.rows; row += blockDim.x) {
    Scalar rhs = right.rhs[row];
    for (int col = 0; col < right.right_dim; ++col) {
      rhs -= right.right[row * right.right_dim + col] * parent.right[col];
    }
    matrix[(left.rows + row) * columns + shared] = rhs;
  }
  WarpSynchronize();
  if (threadIdx.x == 0) {
    conditioned_rhs_scale =
        ConditionedRhsScale(matrix, rows, columns, shared, rank_tolerance);
  }
  WarpSynchronize();
  SolveSystemOrthogonally(matrix, rows, columns, shared, rank_tolerance,
                          consistency_tolerance, conditioned_rhs_scale,
                          residual_rhs, upper, rhs_projection, shared_solution,
                          permutation, &rank, &local_ok);
  if (threadIdx.x == 0 && !local_ok)
    SetFailure(status, kDeviceNumericalFailure, index, 16);
  WarpSynchronize();
  if (!local_ok)
    return;
  if (threadIdx.x == 0) {
    DualNodeValue &left_value = values[child_offset + 2 * index];
    DualNodeValue &right_value = values[child_offset + 2 * index + 1];
    left_value.left_dim = left.left_dim;
    left_value.right_dim = shared;
    right_value.left_dim = shared;
    right_value.right_dim = right.right_dim;
    for (int col = 0; col < left.left_dim; ++col)
      left_value.left[col] = parent.left[col];
    for (int col = 0; col < right.right_dim; ++col)
      right_value.right[col] = parent.right[col];
    for (int col = 0; col < shared; ++col) {
      left_value.right[col] = shared_solution[col];
      right_value.left[col] = shared_solution[col];
    }
  }
}

} // namespace
} // namespace detail
} // namespace cuda
} // namespace clqr

namespace clqr {
namespace cuda {
namespace detail {
namespace {

__global__ void BuildValueElementsKernel(const ReducedStage *stages,
                                         const ReducedTerminal *terminal_ptr,
                                         int stage_count, Scalar tolerance,
                                         ValueElement *elements,
                                         DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index > stage_count)
    return;
  if (!BlockEnabled(status))
    return;
  if (index == stage_count) {
    const ReducedTerminal &terminal = *terminal_ptr;
    if (threadIdx.x == 0) {
      elements[index].left_dim = terminal.n;
      elements[index].right_dim = 0;
    }
    for (int linear = threadIdx.x; linear < terminal.n * terminal.n;
         linear += blockDim.x) {
      const int row = linear / terminal.n;
      const int col = linear % terminal.n;
      elements[index].J[row * terminal.n + col] =
          terminal.Q[row * terminal.n + col];
    }
    for (int row = threadIdx.x; row < terminal.n; row += blockDim.x) {
      elements[index].eta[row] = terminal.q[row];
    }
    return;
  }

  const ReducedStage &s = stages[index];
  ValueElement &out = elements[index];
  if (threadIdx.x == 0) {
    out.left_dim = s.n;
    out.right_dim = s.next_n;
  }
  if (s.m == 0) {
    for (int linear = threadIdx.x; linear < s.next_n * s.n;
         linear += blockDim.x) {
      const int row = linear / s.n;
      const int col = linear % s.n;
      out.A[row * s.n + col] = s.A[row * s.n + col];
    }
    for (int row = threadIdx.x; row < s.next_n; row += blockDim.x)
      out.b[row] = s.c[row];
    for (int linear = threadIdx.x; linear < s.n * s.n; linear += blockDim.x) {
      const int row = linear / s.n;
      const int col = linear % s.n;
      out.J[row * s.n + col] = s.Q[row * s.n + col];
    }
    for (int row = threadIdx.x; row < s.n; row += blockDim.x)
      out.eta[row] = s.q[row];
    for (int linear = threadIdx.x; linear < s.next_n * s.next_n;
         linear += blockDim.x)
      out.C[linear] = Scalar{0};
    return;
  }

  const int rhs_count = s.n + 1 + s.next_n;
  ScratchSize scratch_size;
  scratch_size.Add<Scalar>(static_cast<std::size_t>(s.m) * s.m);
  scratch_size.Add<Scalar>(static_cast<std::size_t>(s.m) * rhs_count);
  CLQR_BLOCK_SCRATCH(scratch, scratch_size.bytes);
  Scalar *cholesky =
      scratch.Take<Scalar>(static_cast<std::size_t>(s.m) * s.m);
  Scalar *right_hand_sides =
      scratch.Take<Scalar>(static_cast<std::size_t>(s.m) * rhs_count);
  __shared__ int positive_definite;
  for (int linear = threadIdx.x; linear < s.m * rhs_count;
       linear += blockDim.x) {
    const int row = linear / rhs_count;
    const int col = linear % rhs_count;
    if (col < s.n) {
      right_hand_sides[linear] = s.M[col * s.m + row];
    } else if (col == s.n) {
      right_hand_sides[linear] = s.r[row];
    } else {
      right_hand_sides[linear] = s.B[(col - s.n - 1) * s.m + row];
    }
  }
  WarpSynchronize();
  if (!FactorPositiveDefiniteBlock(s.R, s.m, s.m, tolerance, cholesky,
                                   &positive_definite)) {
    if (threadIdx.x == 0)
      SetFailure(status, kDeviceNumericalFailure, index, 19);
    return;
  }
  SolvePositiveDefiniteMultipleRhsBlock(cholesky, s.m, right_hand_sides,
                                        rhs_count, rhs_count);

  // The solved columns are R^{-1}*M^T, R^{-1}*r, and R^{-1}*B^T.
  for (int linear = threadIdx.x; linear < s.n * s.n; linear += blockDim.x) {
    const int a = linear / s.n;
    const int b = linear % s.n;
    Scalar value = s.Q[a * s.n + b];
    for (int u = 0; u < s.m; ++u) {
      value -= s.M[a * s.m + u] * right_hand_sides[u * rhs_count + b];
    }
    out.J[a * s.n + b] = value;
  }
  for (int a = threadIdx.x; a < s.n; a += blockDim.x) {
    Scalar value = s.q[a];
    for (int u = 0; u < s.m; ++u) {
      value -= s.M[a * s.m + u] * right_hand_sides[u * rhs_count + s.n];
    }
    out.eta[a] = value;
  }
  for (int linear = threadIdx.x; linear < s.next_n * s.n;
       linear += blockDim.x) {
    const int row = linear / s.n;
    const int col = linear % s.n;
    Scalar value = s.A[row * s.n + col];
    for (int u = 0; u < s.m; ++u) {
      value -= s.B[row * s.m + u] * right_hand_sides[u * rhs_count + col];
    }
    out.A[row * s.n + col] = value;
  }
  for (int row = threadIdx.x; row < s.next_n; row += blockDim.x) {
    Scalar value = s.c[row];
    for (int u = 0; u < s.m; ++u) {
      value -= s.B[row * s.m + u] * right_hand_sides[u * rhs_count + s.n];
    }
    out.b[row] = value;
  }
  for (int linear = threadIdx.x; linear < s.next_n * s.next_n;
       linear += blockDim.x) {
    const int a = linear / s.next_n;
    const int b = linear % s.next_n;
    Scalar value = Scalar{0};
    for (int u = 0; u < s.m; ++u) {
      value += s.B[a * s.m + u] * right_hand_sides[u * rhs_count + s.n + 1 + b];
    }
    out.C[a * s.next_n + b] = value;
  }
}

__device__ void
ComposeValueElementsBlock(const ValueElement &first, const ValueElement &second,
                          Scalar tolerance, ValueElement *output,
                          DeviceStatus *status, int node, Scalar *augmented,
                          Scalar *factors, int *pivot_columns, int *pivot_rows,
                          int *rank, int *best_row) {
  const int shared = first.right_dim;
  if (shared != second.left_dim) {
    if (threadIdx.x == 0)
      SetFailure(status, kDeviceNumericalFailure, node, 20);
    return;
  }
  const int left = first.left_dim;
  const int right = second.right_dim;
  if (threadIdx.x == 0) {
    output->left_dim = left;
    output->right_dim = right;
  }
  if (shared == 0) {
    // A zero-dimensional interface disconnects the two value elements.  The
    // composed cross map is therefore the right-by-left zero matrix.  Compact
    // scan storage is intentionally uninitialized, so publish those zeros
    // explicitly rather than relying on allocator contents.
    for (int linear = threadIdx.x; linear < right * left;
         linear += blockDim.x)
      output->A[linear] = Scalar{0};
    for (int linear = threadIdx.x; linear < left * left; linear += blockDim.x) {
      const int row = linear / left;
      const int col = linear % left;
      output->J[row * left + col] = first.J[row * first.left_dim + col];
    }
    for (int row = threadIdx.x; row < left; row += blockDim.x)
      output->eta[row] = first.eta[row];
    for (int linear = threadIdx.x; linear < right * right;
         linear += blockDim.x) {
      const int row = linear / right;
      const int col = linear % right;
      output->C[row * right + col] = second.C[row * second.right_dim + col];
    }
    for (int row = threadIdx.x; row < right; row += blockDim.x)
      output->b[row] = second.b[row];
    return;
  }

  const int rhs_columns = left + 1 + shared;
  const int columns = shared + rhs_columns;
  for (int linear = threadIdx.x; linear < shared * columns;
       linear += blockDim.x) {
    const int row = linear / columns;
    const int col = linear % columns;
    Scalar value = Scalar{0};
    if (col < shared) {
      value = row == col ? Scalar{1} : Scalar{0};
      for (int k = 0; k < shared; ++k) {
        value += first.C[row * first.right_dim + k] *
                 second.J[k * second.left_dim + col];
      }
    } else if (col < shared + left) {
      value = first.A[row * first.left_dim + col - shared];
    } else if (col == shared + left) {
      value = first.b[row];
      for (int k = 0; k < shared; ++k) {
        value -= first.C[row * first.right_dim + k] * second.eta[k];
      }
    } else {
      value = first.C[row * first.right_dim + col - shared - left - 1];
    }
    augmented[linear] = value;
  }
  WarpSynchronize();
  RrefBlock(augmented, shared, columns, shared, tolerance, pivot_columns,
            pivot_rows, rank, best_row, factors);
  if (*rank != shared) {
    if (threadIdx.x == 0)
      SetFailure(status, kDeviceNumericalFailure, node, 20);
    return;
  }

  // A = A2*S^{-1}*A1 and b = A2*S^{-1}(b1+C1*eta2)+b2.
  for (int linear = threadIdx.x; linear < right * left; linear += blockDim.x) {
    const int row = linear / left;
    const int col = linear % left;
    Scalar value = Scalar{0};
    for (int k = 0; k < shared; ++k) {
      value += second.A[row * second.left_dim + k] *
               augmented[k * columns + shared + col];
    }
    output->A[row * left + col] = value;
  }
  for (int row = threadIdx.x; row < right; row += blockDim.x) {
    Scalar value = second.b[row];
    for (int k = 0; k < shared; ++k) {
      value += second.A[row * second.left_dim + k] *
               augmented[k * columns + shared + left];
    }
    output->b[row] = value;
  }
  for (int linear = threadIdx.x; linear < right * right; linear += blockDim.x) {
    const int row = linear / right;
    const int col = linear % right;
    Scalar value = second.C[row * second.right_dim + col];
    for (int p = 0; p < shared; ++p) {
      for (int q = 0; q < shared; ++q) {
        value += second.A[row * second.left_dim + p] *
                 augmented[p * columns + shared + left + 1 + q] *
                 second.A[col * second.left_dim + q];
      }
    }
    output->C[row * right + col] = value;
  }
  for (int linear = threadIdx.x; linear < left * left; linear += blockDim.x) {
    const int row = linear / left;
    const int col = linear % left;
    Scalar value = first.J[row * first.left_dim + col];
    for (int p = 0; p < shared; ++p) {
      for (int q = 0; q < shared; ++q) {
        value += first.A[p * first.left_dim + row] *
                 second.J[p * second.left_dim + q] *
                 augmented[q * columns + shared + col];
      }
    }
    output->J[row * left + col] = value;
  }
  for (int row = threadIdx.x; row < left; row += blockDim.x) {
    Scalar value = first.eta[row];
    for (int p = 0; p < shared; ++p) {
      Scalar dual = second.eta[p];
      for (int q = 0; q < shared; ++q) {
        dual += second.J[p * second.left_dim + q] *
                augmented[q * columns + shared + left];
      }
      value += first.A[p * first.left_dim + row] * dual;
    }
    output->eta[row] = value;
  }
  WarpSynchronize();
  for (int linear = threadIdx.x; linear < left * left; linear += blockDim.x) {
    const int row = linear / left;
    const int col = linear % left;
    if (row < col) {
      const Scalar value = Scalar{0.5} * (output->J[row * left + col] +
                                          output->J[col * left + row]);
      output->J[row * left + col] = value;
      output->J[col * left + row] = value;
    }
  }
  for (int linear = threadIdx.x; linear < right * right; linear += blockDim.x) {
    const int row = linear / right;
    const int col = linear % right;
    if (row < col) {
      const Scalar value = Scalar{0.5} * (output->C[row * right + col] +
                                          output->C[col * right + row]);
      output->C[row * right + col] = value;
      output->C[col * right + row] = value;
    }
  }
}

__device__ bool InvalidScanValueElement(const ValueElement &element) {
  return element.left_dim < 0;
}

__device__ void SetInvalidScanValueElement(ValueElement *element) {
  if (threadIdx.x == 0) {
    element->left_dim = -1;
    element->right_dim = 0;
  }
}

__device__ void CopyValueElementBlock(const ValueElement &input,
                                      ValueElement *output) {
  if (InvalidScanValueElement(input)) {
    SetInvalidScanValueElement(output);
    return;
  }
  if (threadIdx.x == 0) {
    output->left_dim = input.left_dim;
    output->right_dim = input.right_dim;
  }
  for (int linear = threadIdx.x; linear < input.right_dim * input.left_dim;
       linear += blockDim.x) {
    const int row = linear / input.left_dim;
    const int col = linear % input.left_dim;
    output->A[row * input.left_dim + col] = input.A[row * input.left_dim + col];
  }
  for (int row = threadIdx.x; row < input.right_dim; row += blockDim.x)
    output->b[row] = input.b[row];
  for (int linear = threadIdx.x; linear < input.right_dim * input.right_dim;
       linear += blockDim.x) {
    const int row = linear / input.right_dim;
    const int col = linear % input.right_dim;
    output->C[row * input.right_dim + col] =
        input.C[row * input.right_dim + col];
  }
  for (int row = threadIdx.x; row < input.left_dim; row += blockDim.x)
    output->eta[row] = input.eta[row];
  for (int linear = threadIdx.x; linear < input.left_dim * input.left_dim;
       linear += blockDim.x) {
    const int row = linear / input.left_dim;
    const int col = linear % input.left_dim;
    output->J[row * input.left_dim + col] = input.J[row * input.left_dim + col];
  }
}

__device__ void ComposeScanValueBlock(const ValueElement &first,
                                      const ValueElement &second,
                                      Scalar tolerance, ValueElement *output,
                                      DeviceStatus *status, int node,
                                      Scalar *augmented, Scalar *factors,
                                      int *pivot_columns, int *pivot_rows,
                                      int *rank, int *best_row) {
  if (InvalidScanValueElement(first)) {
    if (InvalidScanValueElement(second)) {
      SetInvalidScanValueElement(output);
    } else {
      CopyValueElementBlock(second, output);
    }
    return;
  }
  if (InvalidScanValueElement(second)) {
    CopyValueElementBlock(first, output);
    return;
  }
  ComposeValueElementsBlock(first, second, tolerance, output, status, node,
                            augmented, factors, pivot_columns, pivot_rows, rank,
                            best_row);
}

__global__ void ReduceValueLeavesKernel(const ValueElement *leaves, int count,
                                        int parent_count, Scalar tolerance,
                                        DeviceStatus *status,
                                        ValueElement *parents) {
  const int index = blockIdx.x;
  if (index >= parent_count)
    return;
  if (!BlockEnabled(status))
    return;
  const int left = 2 * index;
  if (left + 1 >= count) {
    CopyValueElementBlock(leaves[left], &parents[index]);
    return;
  }
  const ValueElement &first = leaves[left];
  const ValueElement &second = leaves[left + 1];
  const int shared = first.right_dim;
  const int columns = 2 * shared + first.left_dim + 1;
  ScratchSize scratch_size;
  scratch_size.Add<Scalar>(static_cast<std::size_t>(shared) * columns);
  scratch_size.Add<Scalar>(shared);
  scratch_size.Add<int>(shared);
  scratch_size.Add<int>(shared);
  CLQR_BLOCK_SCRATCH(scratch, scratch_size.bytes);
  Scalar *augmented =
      scratch.Take<Scalar>(static_cast<std::size_t>(shared) * columns);
  Scalar *factors = scratch.Take<Scalar>(shared);
  int *pivot_columns = scratch.Take<int>(shared);
  int *pivot_rows = scratch.Take<int>(shared);
  __shared__ int rank;
  __shared__ int best_row;
  ComposeValueElementsBlock(first, second, tolerance, &parents[index], status,
                            index, augmented, factors, pivot_columns,
                            pivot_rows, &rank, &best_row);
}

__global__ void ReduceValueTreeLevelKernel(ValueElement *tree, int child_offset,
                                           int parent_offset, int child_count,
                                           int parent_count, Scalar tolerance,
                                           DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index >= parent_count)
    return;
  if (!BlockEnabled(status))
    return;
  const int left = child_offset + 2 * index;
  if (2 * index + 1 >= child_count) {
    CopyValueElementBlock(tree[left], &tree[parent_offset + index]);
    return;
  }
  const int right = left + 1;
  const ValueElement &first = tree[left];
  const ValueElement &second = tree[right];
  const int shared = first.right_dim;
  const int columns = 2 * shared + first.left_dim + 1;
  ScratchSize scratch_size;
  scratch_size.Add<Scalar>(static_cast<std::size_t>(shared) * columns);
  scratch_size.Add<Scalar>(shared);
  scratch_size.Add<int>(shared);
  scratch_size.Add<int>(shared);
  CLQR_BLOCK_SCRATCH(scratch, scratch_size.bytes);
  Scalar *augmented =
      scratch.Take<Scalar>(static_cast<std::size_t>(shared) * columns);
  Scalar *factors = scratch.Take<Scalar>(shared);
  int *pivot_columns = scratch.Take<int>(shared);
  int *pivot_rows = scratch.Take<int>(shared);
  __shared__ int rank;
  __shared__ int best_row;
  ComposeValueElementsBlock(first, second, tolerance,
                            &tree[parent_offset + index], status,
                            parent_offset + index, augmented, factors,
                            pivot_columns, pivot_rows, &rank, &best_row);
}

__global__ void InitializeValueContextRootKernel(ValueElement *tree,
                                                 int root_offset) {
  if (blockIdx.x == 0)
    SetInvalidScanValueElement(&tree[root_offset]);
}

__global__ void ExpandValueContextLevelKernel(
    ValueElement *tree, int child_offset, int parent_offset, int child_count,
    int parent_count, Scalar tolerance, DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index >= parent_count)
    return;
  if (!BlockEnabled(status))
    return;
  const int left = child_offset + 2 * index;
  const ValueElement &parent_context = tree[parent_offset + index];
  if (2 * index + 1 >= child_count) {
    CopyValueElementBlock(parent_context, &tree[left]);
    return;
  }
  const int right = left + 1;
  const ValueElement &first = tree[right];
  const int shared = first.right_dim;
  const int columns = 2 * shared + first.left_dim + 1;
  ScratchSize scratch_size;
  scratch_size.Add<Scalar>(static_cast<std::size_t>(shared) * columns);
  scratch_size.Add<Scalar>(shared);
  scratch_size.Add<int>(shared);
  scratch_size.Add<int>(shared);
  CLQR_BLOCK_SCRATCH(scratch, scratch_size.bytes);
  Scalar *augmented =
      scratch.Take<Scalar>(static_cast<std::size_t>(shared) * columns);
  Scalar *factors = scratch.Take<Scalar>(shared);
  int *pivot_columns = scratch.Take<int>(shared);
  int *pivot_rows = scratch.Take<int>(shared);
  __shared__ int rank;
  __shared__ int best_row;
  ComposeScanValueBlock(tree[right], parent_context, tolerance, &tree[left],
                        status, left, augmented, factors, pivot_columns,
                        pivot_rows, &rank, &best_row);
  WarpSynchronize();
  CopyValueElementBlock(parent_context, &tree[right]);
}

__global__ void FinalizeValueSuffixFromParentsKernel(
    ValueElement *leaves, int count, const ValueElement *parent_contexts,
    int parent_count, Scalar tolerance, DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index >= parent_count)
    return;
  if (!BlockEnabled(status))
    return;
  const int left = 2 * index;
  const ValueElement &parent = parent_contexts[index];
  const int right = left + 1;
  int left_capacity = leaves[left].left_dim;
  int right_capacity = leaves[left].right_dim;
  int shared_capacity = leaves[left].right_dim;
  int columns_capacity = 2 * leaves[left].right_dim + leaves[left].left_dim + 1;
  if (left + 1 < count) {
    left_capacity = DeviceMax(left_capacity, leaves[right].left_dim);
    right_capacity = DeviceMax(right_capacity, leaves[right].right_dim);
    shared_capacity = DeviceMax(shared_capacity, leaves[right].right_dim);
    columns_capacity =
        DeviceMax(columns_capacity,
                  2 * leaves[left].right_dim + leaves[left].left_dim + 1);
  }
  if (!InvalidScanValueElement(parent)) {
    right_capacity = DeviceMax(right_capacity, parent.right_dim);
    const ValueElement &child = left + 1 < count ? leaves[right] : leaves[left];
    shared_capacity = DeviceMax(shared_capacity, child.right_dim);
    columns_capacity =
        DeviceMax(columns_capacity, 2 * child.right_dim + child.left_dim + 1);
  }
  const std::size_t composed_entries =
      static_cast<std::size_t>(right_capacity) * left_capacity +
      right_capacity +
      static_cast<std::size_t>(right_capacity) * right_capacity +
      left_capacity + static_cast<std::size_t>(left_capacity) * left_capacity;
  ScratchSize scratch_size;
  scratch_size.Add<Scalar>(static_cast<std::size_t>(shared_capacity) *
                           columns_capacity);
  scratch_size.Add<Scalar>(shared_capacity);
  scratch_size.Add<ValueElement>(1);
  scratch_size.Add<Scalar>(composed_entries);
  scratch_size.Add<int>(shared_capacity);
  scratch_size.Add<int>(shared_capacity);
  CLQR_BLOCK_SCRATCH(scratch, scratch_size.bytes);
  Scalar *augmented = scratch.Take<Scalar>(
      static_cast<std::size_t>(shared_capacity) * columns_capacity);
  Scalar *factors = scratch.Take<Scalar>(shared_capacity);
  ValueElement *composed_ptr = scratch.Take<ValueElement>(1);
  Scalar *composed_storage = scratch.Take<Scalar>(composed_entries);
  int *pivot_columns = scratch.Take<int>(shared_capacity);
  int *pivot_rows = scratch.Take<int>(shared_capacity);
  ValueElement &composed = *composed_ptr;
  __shared__ int rank;
  __shared__ int best_row;
  if (threadIdx.x == 0)
    BindValueElementScratch(&composed, composed_storage, left_capacity,
                            right_capacity);
  WarpSynchronize();
  if (left + 1 >= count) {
    if (!InvalidScanValueElement(parent)) {
      ComposeScanValueBlock(leaves[left], parent, tolerance, &composed, status,
                            left, augmented, factors, pivot_columns, pivot_rows,
                            &rank, &best_row);
      WarpSynchronize();
      CopyValueElementBlock(composed, &leaves[left]);
    }
    return;
  }
  if (!InvalidScanValueElement(parent)) {
    ComposeScanValueBlock(leaves[right], parent, tolerance, &composed, status,
                          right, augmented, factors, pivot_columns, pivot_rows,
                          &rank, &best_row);
    WarpSynchronize();
    CopyValueElementBlock(composed, &leaves[right]);
    WarpSynchronize();
  }
  ComposeValueElementsBlock(leaves[left], leaves[right], tolerance, &composed,
                            status, left, augmented, factors, pivot_columns,
                            pivot_rows, &rank, &best_row);
  WarpSynchronize();
  CopyValueElementBlock(composed, &leaves[left]);
}

__device__ void BuildFeedbackSystem(const ReducedStage &s,
                                    const ValueElement &next, Scalar *augmented,
                                    int columns) {
  for (int linear = threadIdx.x; linear < s.m * columns; linear += blockDim.x) {
    const int row = linear / columns;
    const int col = linear % columns;
    Scalar value = Scalar{0};
    if (col < s.m) {
      value = s.R[row * s.m + col];
      for (int a = 0; a < s.next_n; ++a) {
        for (int b = 0; b < s.next_n; ++b) {
          value += s.B[a * s.m + row] * next.J[a * next.left_dim + b] *
                   s.B[b * s.m + col];
        }
      }
    } else if (col < s.m + s.n) {
      const int x = col - s.m;
      value = -s.M[x * s.m + row];
      for (int a = 0; a < s.next_n; ++a) {
        for (int b = 0; b < s.next_n; ++b) {
          value -= s.B[a * s.m + row] * next.J[a * next.left_dim + b] *
                   s.A[b * s.n + x];
        }
      }
    } else {
      value = -s.r[row];
      for (int a = 0; a < s.next_n; ++a) {
        Scalar future = next.eta[a];
        for (int b = 0; b < s.next_n; ++b) {
          future += next.J[a * next.left_dim + b] * s.c[b];
        }
        value -= s.B[a * s.m + row] * future;
      }
    }
    augmented[linear] = value;
  }
}

__device__ void ExtractFeedback(const ReducedStage &s, const Scalar *augmented,
                                int columns, Feedback *feedback) {
  if (threadIdx.x == 0) {
    feedback->state_dim = s.n;
    feedback->next_state_dim = s.next_n;
    feedback->control_dim = s.m;
  }
  for (int linear = threadIdx.x; linear < s.m * s.n; linear += blockDim.x) {
    const int row = linear / s.n;
    const int col = linear % s.n;
    feedback->K[row * s.n + col] = augmented[row * columns + s.m + col];
  }
  for (int row = threadIdx.x; row < s.m; row += blockDim.x) {
    feedback->k[row] = augmented[row * columns + s.m + s.n];
  }
  WarpSynchronize();
  for (int linear = threadIdx.x; linear < s.next_n * s.n;
       linear += blockDim.x) {
    const int row = linear / s.n;
    const int col = linear % s.n;
    Scalar value = s.A[row * s.n + col];
    for (int u = 0; u < s.m; ++u) {
      value += s.B[row * s.m + u] * feedback->K[u * s.n + col];
    }
    feedback->transition[row * s.n + col] = value;
  }
  for (int row = threadIdx.x; row < s.next_n; row += blockDim.x) {
    Scalar value = s.c[row];
    for (int u = 0; u < s.m; ++u) {
      value += s.B[row * s.m + u] * feedback->k[u];
    }
    feedback->offset[row] = value;
  }
}

__device__ bool SolveFeedbackBlock(const ReducedStage &stage,
                                   const ValueElement &next, Scalar tolerance,
                                   Feedback *feedback, Scalar *augmented,
                                   Scalar *cholesky, int *positive_definite,
                                   DeviceStatus *status, int stage_index,
                                   int diagnostic) {
  if (stage.m == 0) {
    if (threadIdx.x == 0) {
      feedback->state_dim = stage.n;
      feedback->next_state_dim = stage.next_n;
      feedback->control_dim = 0;
    }
    for (int linear = threadIdx.x; linear < stage.next_n * stage.n;
         linear += blockDim.x) {
      const int row = linear / stage.n;
      const int col = linear % stage.n;
      feedback->transition[row * stage.n + col] = stage.A[row * stage.n + col];
    }
    for (int row = threadIdx.x; row < stage.next_n; row += blockDim.x)
      feedback->offset[row] = stage.c[row];
    return true;
  }

  const int columns = stage.m + stage.n + 1;
  BuildFeedbackSystem(stage, next, augmented, columns);
  WarpSynchronize();
  if (!FactorPositiveDefiniteBlock(augmented, columns, stage.m, tolerance,
                                   cholesky, positive_definite)) {
    if (threadIdx.x == 0)
      SetFailure(status, kDeviceNumericalFailure, stage_index, diagnostic);
    return false;
  }
  SolvePositiveDefiniteMultipleRhsBlock(cholesky, stage.m, augmented + stage.m,
                                        columns, stage.n + 1);
  ExtractFeedback(stage, augmented, columns, feedback);
  return true;
}

__global__ void FeedbackKernel(const ReducedStage *stages,
                               const ValueElement *suffix, int stage_count,
                               Scalar tolerance, Feedback *feedback,
                               DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index >= stage_count)
    return;
  if (!BlockEnabled(status))
    return;
  const ReducedStage &s = stages[index];
  const ValueElement &next = suffix[index + 1];
  Feedback &out = feedback[index];
  const int columns = s.m + s.n + 1;
  ScratchSize scratch_size;
  scratch_size.Add<Scalar>(static_cast<std::size_t>(s.m) * columns);
  scratch_size.Add<Scalar>(static_cast<std::size_t>(s.m) * s.m);
  CLQR_BLOCK_SCRATCH(scratch, scratch_size.bytes);
  Scalar *augmented =
      scratch.Take<Scalar>(static_cast<std::size_t>(s.m) * columns);
  Scalar *cholesky =
      scratch.Take<Scalar>(static_cast<std::size_t>(s.m) * s.m);
  __shared__ int positive_definite;
  SolveFeedbackBlock(s, next, tolerance, &out, augmented, cholesky,
                     &positive_definite, status, index, 9);
}

} // namespace
} // namespace detail
} // namespace cuda
} // namespace clqr

namespace clqr {
namespace cuda {
namespace detail {
namespace {

__global__ void
ReduceStagesKernel(const PackedStage *stages, const Relation *suffix,
                   const StateParam *state_params, int stage_count,
                   Scalar rank_tolerance, Scalar consistency_tolerance,
                   ControlParam *control_params, ReducedStage *reduced,
                   int *control_dimensions, DeviceStatus *status) {
  const int index = blockIdx.x;
  if (index >= stage_count)
    return;
  if (!BlockEnabled(status))
    return;
  const PackedStage &s = stages[index];
  const StateParam &current = state_params[index];
  const StateParam &next = state_params[index + 1];
  const Relation &next_relation = suffix[index + 1];
  const int rows = s.mixed + next_relation.rows;
  const int columns = s.m + current.reduced_dim + 1;
  ScratchSize scratch_size;
  scratch_size.Add<Scalar>(static_cast<std::size_t>(rows) * columns);
  scratch_size.Add<Scalar>(rows);
  scratch_size.Add<int>(rows);
  scratch_size.Add<int>(rows);
  CLQR_BLOCK_SCRATCH(scratch, scratch_size.bytes);
  Scalar *matrix =
      scratch.Take<Scalar>(static_cast<std::size_t>(rows) * columns);
  Scalar *factors = scratch.Take<Scalar>(rows);
  int *pivot_columns = scratch.Take<int>(rows);
  int *pivot_rows = scratch.Take<int>(rows);
  __shared__ int rank;
  __shared__ int best_row;
  __shared__ int control_rank;
  __shared__ int local_ok;
  for (int i = threadIdx.x; i < rows * columns; i += blockDim.x)
    matrix[i] = Scalar{0};
  WarpSynchronize();

  // Original mixed equalities after x = T*z + t.
  for (int linear = threadIdx.x; linear < s.mixed * s.m; linear += blockDim.x) {
    const int row = linear / s.m;
    const int col = linear % s.m;
    matrix[row * columns + col] = s.D[row * s.m + col];
  }
  for (int linear = threadIdx.x; linear < s.mixed * current.reduced_dim;
       linear += blockDim.x) {
    const int row = linear / current.reduced_dim;
    const int z = linear % current.reduced_dim;
    Scalar value = Scalar{0};
    for (int x = 0; x < s.n; ++x) {
      value += s.C[row * s.n + x] * current.T[x * current.reduced_dim + z];
    }
    matrix[row * columns + s.m + z] = value;
  }
  for (int row = threadIdx.x; row < s.mixed; row += blockDim.x) {
    Scalar value = -s.d[row];
    for (int x = 0; x < s.n; ++x) {
      value -= s.C[row * s.n + x] * current.t[x];
    }
    matrix[row * columns + columns - 1] = value;
  }

  // The successor must belong to its propagated feasible-state set.
  for (int linear = threadIdx.x; linear < next_relation.rows * s.m;
       linear += blockDim.x) {
    const int row = linear / s.m;
    const int u = linear % s.m;
    Scalar value = Scalar{0};
    for (int xp = 0; xp < s.next_n; ++xp) {
      value += next_relation.left[row * next_relation.left_dim + xp] *
               s.B[xp * s.m + u];
    }
    matrix[(s.mixed + row) * columns + u] = value;
  }
  for (int linear = threadIdx.x;
       linear < next_relation.rows * current.reduced_dim;
       linear += blockDim.x) {
    const int row = linear / current.reduced_dim;
    const int z = linear % current.reduced_dim;
    Scalar value = Scalar{0};
    for (int xp = 0; xp < s.next_n; ++xp) {
      Scalar at = Scalar{0};
      for (int x = 0; x < s.n; ++x) {
        at += s.A[xp * s.n + x] * current.T[x * current.reduced_dim + z];
      }
      value += next_relation.left[row * next_relation.left_dim + xp] * at;
    }
    matrix[(s.mixed + row) * columns + s.m + z] = value;
  }
  for (int row = threadIdx.x; row < next_relation.rows; row += blockDim.x) {
    Scalar value = next_relation.rhs[row];
    for (int xp = 0; xp < s.next_n; ++xp) {
      Scalar affine = s.c[xp];
      for (int x = 0; x < s.n; ++x) {
        affine += s.A[xp * s.n + x] * current.t[x];
      }
      value -= next_relation.left[row * next_relation.left_dim + xp] * affine;
    }
    matrix[(s.mixed + row) * columns + columns - 1] = value;
  }
  WarpSynchronize();

  // Normalize raw mixed equations by their original scale, while treating a
  // successor equation produced entirely at the roundoff level as zero.  A
  // second relative row test removes constraints that vanish after applying
  // the current feasible-state parameterization.  This must happen before the
  // generic RREF equilibration, which would otherwise magnify cancellation
  // noise into a unit pivot.
  for (int row = threadIdx.x; row < rows; row += blockDim.x) {
    Scalar scale = Scalar{0};
    if (row < s.mixed) {
      for (int x = 0; x < s.n; ++x)
        scale = fmax(scale, DeviceAbs(s.C[row * s.n + x]));
      for (int u = 0; u < s.m; ++u)
        scale = fmax(scale, DeviceAbs(s.D[row * s.m + u]));
      scale = fmax(scale, DeviceAbs(s.d[row]));
    } else {
      for (int col = 0; col < columns; ++col)
        scale = fmax(scale, DeviceAbs(matrix[row * columns + col]));
    }
    factors[row] = scale;
  }
  WarpSynchronize();
  for (int linear = threadIdx.x; linear < rows * columns;
       linear += blockDim.x) {
    const int row = linear / columns;
    const Scalar scale = factors[row];
    if (row >= s.mixed && scale <= rank_tolerance) {
      matrix[linear] = Scalar{0};
    } else if (scale > Scalar{0}) {
      matrix[linear] /= scale;
    }
  }
  WarpSynchronize();
  for (int row = threadIdx.x; row < rows; row += blockDim.x) {
    Scalar scale = Scalar{0};
    for (int col = 0; col < columns; ++col)
      scale = fmax(scale, DeviceAbs(matrix[row * columns + col]));
    factors[row] = scale;
  }
  WarpSynchronize();
  for (int linear = threadIdx.x; linear < rows * columns;
       linear += blockDim.x) {
    if (factors[linear / columns] <= rank_tolerance)
      matrix[linear] = Scalar{0};
  }
  WarpSynchronize();

  RrefBlock(matrix, rows, columns, columns - 1, rank_tolerance, pivot_columns,
            pivot_rows, &rank, &best_row, factors);
  if (threadIdx.x == 0) {
    local_ok = 1;
    if (InconsistentRref(matrix, rows, columns, columns - 1, rank_tolerance,
                         consistency_tolerance)) {
      SetFailure(status, kDeviceInfeasible, index, 6);
      local_ok = 0;
    }
    control_rank = 0;
    while (control_rank < rank && pivot_columns[control_rank] < s.m)
      ++control_rank;
    if (control_rank < rank) {
      // The suffix relation says every current reduced state is feasible, so
      // elimination must not discover an additional condition on z.
      SetFailure(status, kDeviceNumericalFailure, index, 7);
      local_ok = 0;
    }
    if (local_ok) {
      ControlParam &cp = control_params[index];
      cp.physical_dim = s.m;
      cp.state_dim = current.reduced_dim;
      cp.reduced_dim = s.m - control_rank;
      int free = 0;
      for (int u = 0; u < s.m; ++u) {
        bool is_pivot = false;
        for (int p = 0; p < control_rank; ++p)
          is_pivot |= pivot_columns[p] == u;
        if (!is_pivot)
          cp.free_columns[free++] = u;
      }
      ReducedStage &rs = reduced[index];
      rs.n = current.reduced_dim;
      rs.next_n = next.reduced_dim;
      rs.m = cp.reduced_dim;
      if (control_dimensions != nullptr) {
        control_dimensions[2 * index] = cp.physical_dim;
        control_dimensions[2 * index + 1] = cp.reduced_dim;
      }
    }
  }
  WarpSynchronize();
  if (!local_ok)
    return;

  ControlParam &initialized_control = control_params[index];
  for (int u = threadIdx.x; u < initialized_control.physical_dim;
       u += blockDim.x)
    initialized_control.y[u] = Scalar{0};
  for (int linear = threadIdx.x; linear < initialized_control.physical_dim *
                                              initialized_control.state_dim;
       linear += blockDim.x)
    initialized_control.Y[linear] = Scalar{0};
  for (int linear = threadIdx.x; linear < initialized_control.physical_dim *
                                              initialized_control.reduced_dim;
       linear += blockDim.x) {
    const int u = linear / initialized_control.reduced_dim;
    const int v = linear % initialized_control.reduced_dim;
    initialized_control.Z[u * initialized_control.reduced_dim + v] = Scalar{0};
  }
  WarpSynchronize();
  if (threadIdx.x == 0) {
    for (int p = 0; p < control_rank; ++p) {
      const int u = pivot_columns[p];
      initialized_control.y[u] = matrix[p * columns + columns - 1];
      for (int z = 0; z < current.reduced_dim; ++z) {
        initialized_control.Y[u * initialized_control.state_dim + z] =
            -matrix[p * columns + s.m + z];
      }
      for (int v = 0; v < initialized_control.reduced_dim; ++v) {
        initialized_control.Z[u * initialized_control.reduced_dim + v] =
            -matrix[p * columns + initialized_control.free_columns[v]];
      }
    }
    for (int v = 0; v < initialized_control.reduced_dim; ++v) {
      initialized_control.Z[initialized_control.free_columns[v] *
                                initialized_control.reduced_dim +
                            v] = Scalar{1};
    }
  }
  WarpSynchronize();

  const ControlParam &cp = control_params[index];
  ReducedStage &rs = reduced[index];

  // Reduced dynamics, selecting the free physical coordinates at node i+1.
  for (int linear = threadIdx.x; linear < rs.next_n * rs.n;
       linear += blockDim.x) {
    const int row = linear / rs.n;
    const int z = linear % rs.n;
    const int xp = next.free_columns[row];
    Scalar value = Scalar{0};
    for (int x = 0; x < s.n; ++x) {
      value += s.A[xp * s.n + x] * current.T[x * current.reduced_dim + z];
    }
    for (int u = 0; u < s.m; ++u) {
      value += s.B[xp * s.m + u] * cp.Y[u * cp.state_dim + z];
    }
    rs.A[row * rs.n + z] = value;
  }
  for (int linear = threadIdx.x; linear < rs.next_n * rs.m;
       linear += blockDim.x) {
    const int row = linear / rs.m;
    const int v = linear % rs.m;
    const int xp = next.free_columns[row];
    Scalar value = Scalar{0};
    for (int u = 0; u < s.m; ++u) {
      value += s.B[xp * s.m + u] * cp.Z[u * cp.reduced_dim + v];
    }
    rs.B[row * rs.m + v] = value;
  }
  for (int row = threadIdx.x; row < rs.next_n; row += blockDim.x) {
    const int xp = next.free_columns[row];
    Scalar value = s.c[xp] - next.t[xp];
    for (int x = 0; x < s.n; ++x) {
      value += s.A[xp * s.n + x] * current.t[x];
    }
    for (int u = 0; u < s.m; ++u) {
      value += s.B[xp * s.m + u] * cp.y[u];
    }
    rs.c[row] = value;
  }

  // Reduced quadratic and bilinear terms.
  for (int linear = threadIdx.x; linear < rs.n * rs.n; linear += blockDim.x) {
    const int a = linear / rs.n;
    const int b = linear % rs.n;
    Scalar value = Scalar{0};
    for (int x = 0; x < s.n; ++x) {
      for (int y = 0; y < s.n; ++y) {
        value += current.T[x * current.reduced_dim + a] * s.Q[x * s.n + y] *
                 current.T[y * current.reduced_dim + b];
      }
      for (int u = 0; u < s.m; ++u) {
        value += current.T[x * current.reduced_dim + a] * s.M[x * s.m + u] *
                 cp.Y[u * cp.state_dim + b];
        value += cp.Y[u * cp.state_dim + a] * s.M[x * s.m + u] *
                 current.T[x * current.reduced_dim + b];
      }
    }
    for (int u = 0; u < s.m; ++u) {
      for (int v = 0; v < s.m; ++v) {
        value += cp.Y[u * cp.state_dim + a] * s.R[u * s.m + v] *
                 cp.Y[v * cp.state_dim + b];
      }
    }
    rs.Q[a * rs.n + b] = value;
  }
  for (int linear = threadIdx.x; linear < rs.m * rs.m; linear += blockDim.x) {
    const int a = linear / rs.m;
    const int b = linear % rs.m;
    Scalar value = Scalar{0};
    for (int u = 0; u < s.m; ++u) {
      for (int v = 0; v < s.m; ++v) {
        value += cp.Z[u * cp.reduced_dim + a] * s.R[u * s.m + v] *
                 cp.Z[v * cp.reduced_dim + b];
      }
    }
    rs.R[a * rs.m + b] = value;
  }
  for (int linear = threadIdx.x; linear < rs.n * rs.m; linear += blockDim.x) {
    const int z = linear / rs.m;
    const int v = linear % rs.m;
    Scalar value = Scalar{0};
    for (int x = 0; x < s.n; ++x) {
      for (int u = 0; u < s.m; ++u) {
        value += current.T[x * current.reduced_dim + z] * s.M[x * s.m + u] *
                 cp.Z[u * cp.reduced_dim + v];
      }
    }
    for (int u = 0; u < s.m; ++u) {
      for (int w = 0; w < s.m; ++w) {
        value += cp.Y[u * cp.state_dim + z] * s.R[u * s.m + w] *
                 cp.Z[w * cp.reduced_dim + v];
      }
    }
    rs.M[z * rs.m + v] = value;
  }

  for (int z = threadIdx.x; z < rs.n; z += blockDim.x) {
    Scalar value = Scalar{0};
    for (int x = 0; x < s.n; ++x) {
      Scalar gx = s.q[x];
      for (int y = 0; y < s.n; ++y)
        gx += s.Q[x * s.n + y] * current.t[y];
      for (int u = 0; u < s.m; ++u)
        gx += s.M[x * s.m + u] * cp.y[u];
      value += current.T[x * current.reduced_dim + z] * gx;
    }
    for (int u = 0; u < s.m; ++u) {
      Scalar gu = s.r[u];
      for (int x = 0; x < s.n; ++x)
        gu += s.M[x * s.m + u] * current.t[x];
      for (int v = 0; v < s.m; ++v)
        gu += s.R[u * s.m + v] * cp.y[v];
      value += cp.Y[u * cp.state_dim + z] * gu;
    }
    rs.q[z] = value;
  }
  for (int v = threadIdx.x; v < rs.m; v += blockDim.x) {
    Scalar value = Scalar{0};
    for (int u = 0; u < s.m; ++u) {
      Scalar gu = s.r[u];
      for (int x = 0; x < s.n; ++x)
        gu += s.M[x * s.m + u] * current.t[x];
      for (int w = 0; w < s.m; ++w)
        gu += s.R[u * s.m + w] * cp.y[w];
      value += cp.Z[u * cp.reduced_dim + v] * gu;
    }
    rs.r[v] = value;
  }
}

__global__ void ReduceTerminalKernel(const PackedTerminal *terminal_ptr,
                                     const StateParam *state_params,
                                     int terminal_index,
                                     ReducedTerminal *reduced) {
  const PackedTerminal &terminal = *terminal_ptr;
  const StateParam &param = state_params[terminal_index];
  if (threadIdx.x == 0)
    reduced->n = param.reduced_dim;
  for (int linear = threadIdx.x; linear < param.reduced_dim * param.reduced_dim;
       linear += blockDim.x) {
    const int a = linear / param.reduced_dim;
    const int b = linear % param.reduced_dim;
    Scalar value = Scalar{0};
    for (int x = 0; x < terminal.n; ++x) {
      for (int y = 0; y < terminal.n; ++y) {
        value += param.T[x * param.reduced_dim + a] *
                 terminal.Q[x * terminal.n + y] *
                 param.T[y * param.reduced_dim + b];
      }
    }
    reduced->Q[a * reduced->n + b] = value;
  }
  for (int a = threadIdx.x; a < param.reduced_dim; a += blockDim.x) {
    Scalar value = Scalar{0};
    for (int x = 0; x < terminal.n; ++x) {
      Scalar gx = terminal.q[x];
      for (int y = 0; y < terminal.n; ++y) {
        gx += terminal.Q[x * terminal.n + y] * param.t[y];
      }
      value += param.T[x * param.reduced_dim + a] * gx;
    }
    reduced->q[a] = value;
  }
}

__global__ void InitialReducedStateKernel(const StateParam *state_params,
                                          const Scalar *initial_state,
                                          Scalar *reduced_initial,
                                          Scalar tolerance,
                                          DeviceStatus *status) {
  const StateParam &param = state_params[0];
  for (int z = threadIdx.x; z < param.reduced_dim; z += blockDim.x) {
    const int physical = param.free_columns[z];
    reduced_initial[z] = initial_state[physical] - param.t[physical];
  }
  WarpSynchronize();
  if (threadIdx.x == 0) {
    Scalar scale = Scalar{1};
    Scalar residual = Scalar{0};
    for (int x = 0; x < param.physical_dim; ++x) {
      Scalar value = param.t[x];
      for (int z = 0; z < param.reduced_dim; ++z) {
        value += param.T[x * param.reduced_dim + z] * reduced_initial[z];
      }
      scale = fmax(scale, DeviceAbs(initial_state[x]));
      residual = fmax(residual, DeviceAbs(value - initial_state[x]));
    }
    if (residual > Scalar{20} * tolerance * scale) {
      SetFailure(status, kDeviceInfeasible, 0, 8);
    }
  }
}

} // namespace
} // namespace detail
} // namespace cuda
} // namespace clqr
