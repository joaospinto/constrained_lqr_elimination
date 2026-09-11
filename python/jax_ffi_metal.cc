#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "python/jax_ffi_metal.h"
#include "src/metal_dual_objective_source.h"
#include "src/metal_layout.h"
#include "src/metal_planner.h"
#include "src/metal_primal_scan_source.h"
#include "src/metal_reduction_recovery_source.h"
#include "src/metal_value_affine_source.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

#ifdef CLQR_USE_FLOAT
constexpr ffi::DataType kScalarType = ffi::DataType::F32;
#else
constexpr ffi::DataType kScalarType = ffi::DataType::F64;
#endif

using ScalarBufferR1 = ffi::BufferR1<kScalarType>;
using ScalarBufferR2 = ffi::BufferR2<kScalarType>;
using ScalarBufferR3 = ffi::BufferR3<kScalarType>;
using ScalarResultR1 = ffi::ResultBufferR1<kScalarType>;
using ScalarResultR2 = ffi::ResultBufferR2<kScalarType>;

using clqr::metal::detail::CheckedSizeProduct;
using clqr::metal::detail::CheckedSizeSum;
using clqr::metal::detail::HasCooperativeThreadgroupOccupancy;
using clqr::metal::detail::InvocationLayout;
using clqr::metal::detail::KernelParams;
using clqr::metal::detail::PlanInvocation;
using clqr::metal::detail::PlanLaneSlicedThreadgroupLanes;
using clqr::metal::detail::TreePlan;
using clqr::metal::detail::WithScratch;

std::string ErrorText(NSError *error) {
  if (error == nil)
    return "unknown Metal error";
  return std::string(error.localizedDescription.UTF8String);
}

class MetalRuntime {
public:
  MetalRuntime() {
    @autoreleasepool {
      device_ = MTLCreateSystemDefaultDevice();
      if (device_ == nil)
        throw std::runtime_error("no Metal device is available");
      if (![device_ supportsFamily:MTLGPUFamilyApple1] ||
          !device_.hasUnifiedMemory) {
        throw std::runtime_error(
            "the CLQR Metal backend requires an Apple-silicon "
            "unified-memory GPU");
      }
      queue_ = [device_ newCommandQueue];
      if (queue_ == nil)
        throw std::runtime_error(
            "failed to create the CLQR Metal command queue");

      primal_library_ = CompileLibrary(
          clqr::metal::detail::kMetalPrimalScanSource,
          sizeof(clqr::metal::detail::kMetalPrimalScanSource) - 1,
          "primal scan");
      reduction_recovery_library_ = CompileLibrary(
          clqr::metal::detail::kMetalReductionRecoverySource,
          sizeof(clqr::metal::detail::kMetalReductionRecoverySource) - 1,
          "reduction/recovery");
      std::string dual_source(
          clqr::metal::detail::kMetalDualObjectiveSourcePart0,
          sizeof(clqr::metal::detail::kMetalDualObjectiveSourcePart0) - 1);
      dual_source.append(
          clqr::metal::detail::kMetalDualObjectiveSourcePart1,
          sizeof(clqr::metal::detail::kMetalDualObjectiveSourcePart1) - 1);
      dual_objective_library_ = CompileLibrary(
          dual_source.data(), dual_source.size(), "dual/objective");
      value_affine_library_ = CompileLibrary(
          clqr::metal::detail::kMetalValueAffineSource,
          sizeof(clqr::metal::detail::kMetalValueAffineSource) - 1,
          "value/affine scan");
      check_finite_inputs_ =
          Pipeline(primal_library_, "clqr_check_finite_inputs");
      build_primal_leaves_ =
          Pipeline(primal_library_, "clqr_build_primal_leaves");
      build_primal_leaves_threadgroup_sliced_ = Pipeline(
          primal_library_, "clqr_build_primal_leaves_threadgroup_sliced");
      reduce_primal_relations_ =
          Pipeline(primal_library_, "clqr_reduce_primal_relations");
      expand_primal_suffix_context_ =
          Pipeline(primal_library_, "clqr_expand_primal_suffix_context");
      finalize_primal_suffix_ =
          Pipeline(primal_library_, "clqr_finalize_primal_suffix");
      extract_state_parameters_ =
          Pipeline(primal_library_, "clqr_extract_state_parameters");
      reduce_stages_ =
          Pipeline(reduction_recovery_library_, "clqr_reduce_stages");
      reduce_terminal_ =
          Pipeline(reduction_recovery_library_, "clqr_reduce_terminal");
      initial_reduced_state_ =
          Pipeline(reduction_recovery_library_, "clqr_initial_reduced_state");
      matrix_feedback_ =
          Pipeline(reduction_recovery_library_, "clqr_matrix_feedback");
      initialize_costate_maps_ =
          Pipeline(reduction_recovery_library_, "clqr_initialize_costate_maps");
      recover_costates_ =
          Pipeline(reduction_recovery_library_, "clqr_recover_costates");
      finalize_feedback_ =
          Pipeline(reduction_recovery_library_, "clqr_finalize_feedback");
      initialize_state_maps_ =
          Pipeline(reduction_recovery_library_, "clqr_initialize_state_maps");
      reconstruct_primal_ =
          Pipeline(reduction_recovery_library_, "clqr_reconstruct_primal");
      build_dual_parameters_ =
          Pipeline(dual_objective_library_, "clqr_build_dual_parameters");
      build_dual_parameters_threadgroup_sliced_ =
          Pipeline(dual_objective_library_,
                   "clqr_build_dual_parameters_threadgroup_sliced");
      build_dual_relation_leaves_ =
          Pipeline(dual_objective_library_, "clqr_build_dual_relation_leaves");
      build_dual_relation_leaves_threadgroup_sliced_ =
          Pipeline(dual_objective_library_,
                   "clqr_build_dual_relation_leaves_threadgroup_sliced");
      reduce_dual_relations_ =
          Pipeline(dual_objective_library_, "clqr_reduce_dual_relations");
      solve_dual_root_ =
          Pipeline(dual_objective_library_, "clqr_solve_dual_root");
      expand_dual_relations_ =
          Pipeline(dual_objective_library_, "clqr_expand_dual_relations");
      recover_parameterized_multipliers_ = Pipeline(
          dual_objective_library_, "clqr_recover_parameterized_multipliers");
      recover_initial_multiplier_ =
          Pipeline(dual_objective_library_, "clqr_recover_initial_multiplier");
      build_objective_terms_ =
          Pipeline(dual_objective_library_, "clqr_build_objective_terms");
      reduce_objective_ =
          Pipeline(dual_objective_library_, "clqr_reduce_objective");
      finalize_objective_ =
          Pipeline(dual_objective_library_, "clqr_finalize_objective");
      build_value_leaves_ =
          Pipeline(value_affine_library_, "clqr_build_value_leaves");
      reduce_value_ = Pipeline(value_affine_library_, "clqr_reduce_value");
      reduce_value_threadgroup_ =
          Pipeline(value_affine_library_, "clqr_reduce_value_threadgroup");
      expand_value_context_ =
          Pipeline(value_affine_library_, "clqr_expand_value_context");
      expand_value_context_threadgroup_ = Pipeline(
          value_affine_library_, "clqr_expand_value_context_threadgroup");
      finalize_value_suffix_ =
          Pipeline(value_affine_library_, "clqr_finalize_value_suffix");
      finalize_value_suffix_threadgroup_ = Pipeline(
          value_affine_library_, "clqr_finalize_value_suffix_threadgroup");
      reduce_affine_ = Pipeline(value_affine_library_, "clqr_reduce_affine");
      expand_affine_context_ =
          Pipeline(value_affine_library_, "clqr_expand_affine_context");
      finalize_affine_prefix_ =
          Pipeline(value_affine_library_, "clqr_finalize_affine_prefix");
    }
  }

  MetalRuntime(const MetalRuntime &) = delete;
  MetalRuntime &operator=(const MetalRuntime &) = delete;

  id<MTLDevice> device() const { return device_; }
  id<MTLCommandQueue> queue() const { return queue_; }
  id<MTLComputePipelineState> check_finite_inputs() const {
    return check_finite_inputs_;
  }
  id<MTLComputePipelineState> build_primal_leaves() const {
    return build_primal_leaves_;
  }
  id<MTLComputePipelineState> build_primal_leaves_threadgroup_sliced() const {
    return build_primal_leaves_threadgroup_sliced_;
  }
  id<MTLComputePipelineState> reduce_primal_relations() const {
    return reduce_primal_relations_;
  }
  id<MTLComputePipelineState> expand_primal_suffix_context() const {
    return expand_primal_suffix_context_;
  }
  id<MTLComputePipelineState> finalize_primal_suffix() const {
    return finalize_primal_suffix_;
  }
  id<MTLComputePipelineState> extract_state_parameters() const {
    return extract_state_parameters_;
  }
  id<MTLComputePipelineState> reduce_stages() const { return reduce_stages_; }
  id<MTLComputePipelineState> reduce_terminal() const {
    return reduce_terminal_;
  }
  id<MTLComputePipelineState> initial_reduced_state() const {
    return initial_reduced_state_;
  }
  id<MTLComputePipelineState> matrix_feedback() const {
    return matrix_feedback_;
  }
  id<MTLComputePipelineState> initialize_costate_maps() const {
    return initialize_costate_maps_;
  }
  id<MTLComputePipelineState> recover_costates() const {
    return recover_costates_;
  }
  id<MTLComputePipelineState> finalize_feedback() const {
    return finalize_feedback_;
  }
  id<MTLComputePipelineState> initialize_state_maps() const {
    return initialize_state_maps_;
  }
  id<MTLComputePipelineState> reconstruct_primal() const {
    return reconstruct_primal_;
  }
  id<MTLComputePipelineState> build_dual_parameters() const {
    return build_dual_parameters_;
  }
  id<MTLComputePipelineState> build_dual_parameters_threadgroup_sliced() const {
    return build_dual_parameters_threadgroup_sliced_;
  }
  id<MTLComputePipelineState> build_dual_relation_leaves() const {
    return build_dual_relation_leaves_;
  }
  id<MTLComputePipelineState>
  build_dual_relation_leaves_threadgroup_sliced() const {
    return build_dual_relation_leaves_threadgroup_sliced_;
  }
  id<MTLComputePipelineState> reduce_dual_relations() const {
    return reduce_dual_relations_;
  }
  id<MTLComputePipelineState> solve_dual_root() const {
    return solve_dual_root_;
  }
  id<MTLComputePipelineState> expand_dual_relations() const {
    return expand_dual_relations_;
  }
  id<MTLComputePipelineState> recover_parameterized_multipliers() const {
    return recover_parameterized_multipliers_;
  }
  id<MTLComputePipelineState> recover_initial_multiplier() const {
    return recover_initial_multiplier_;
  }
  id<MTLComputePipelineState> build_objective_terms() const {
    return build_objective_terms_;
  }
  id<MTLComputePipelineState> reduce_objective() const {
    return reduce_objective_;
  }
  id<MTLComputePipelineState> finalize_objective() const {
    return finalize_objective_;
  }
  id<MTLComputePipelineState> build_value_leaves() const {
    return build_value_leaves_;
  }
  id<MTLComputePipelineState> reduce_value() const { return reduce_value_; }
  id<MTLComputePipelineState> reduce_value_threadgroup() const {
    return reduce_value_threadgroup_;
  }
  id<MTLComputePipelineState> expand_value_context() const {
    return expand_value_context_;
  }
  id<MTLComputePipelineState> expand_value_context_threadgroup() const {
    return expand_value_context_threadgroup_;
  }
  id<MTLComputePipelineState> finalize_value_suffix() const {
    return finalize_value_suffix_;
  }
  id<MTLComputePipelineState> finalize_value_suffix_threadgroup() const {
    return finalize_value_suffix_threadgroup_;
  }
  id<MTLComputePipelineState> reduce_affine() const { return reduce_affine_; }
  id<MTLComputePipelineState> expand_affine_context() const {
    return expand_affine_context_;
  }
  id<MTLComputePipelineState> finalize_affine_prefix() const {
    return finalize_affine_prefix_;
  }

private:
  id<MTLLibrary> CompileLibrary(const char *bytes, std::size_t size,
                                const char *description) {
    NSString *source = [[NSString alloc] initWithBytes:bytes
                                                length:size
                                              encoding:NSUTF8StringEncoding];
    MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
    options.fastMathEnabled = NO;
    NSError *error = nil;
    id<MTLLibrary> library =
        [device_ newLibraryWithSource:source options:options error:&error];
    if (library == nil) {
      throw std::runtime_error(std::string("failed to compile CLQR Metal ") +
                               description + " kernels: " + ErrorText(error));
    }
    return library;
  }

  id<MTLComputePipelineState> Pipeline(id<MTLLibrary> library,
                                       const char *name) {
    NSString *function_name = [NSString stringWithUTF8String:name];
    id<MTLFunction> function = [library newFunctionWithName:function_name];
    if (function == nil)
      throw std::runtime_error(std::string("missing Metal kernel ") + name);
    NSError *error = nil;
    id<MTLComputePipelineState> pipeline =
        [device_ newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
      throw std::runtime_error(std::string("failed to create Metal pipeline ") +
                               name + ": " + ErrorText(error));
    }
    constexpr NSUInteger kRequiredThreadExecutionWidth = 32;
    if (pipeline.threadExecutionWidth != kRequiredThreadExecutionWidth) {
      throw std::runtime_error(
          std::string("CLQR Metal requires 32-lane SIMD groups; pipeline ") +
          name + " reports " + std::to_string(pipeline.threadExecutionWidth));
    }
    return pipeline;
  }

  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> queue_ = nil;
  id<MTLLibrary> primal_library_ = nil;
  id<MTLLibrary> reduction_recovery_library_ = nil;
  id<MTLLibrary> dual_objective_library_ = nil;
  id<MTLLibrary> value_affine_library_ = nil;
  id<MTLComputePipelineState> check_finite_inputs_ = nil;
  id<MTLComputePipelineState> build_primal_leaves_ = nil;
  id<MTLComputePipelineState> build_primal_leaves_threadgroup_sliced_ = nil;
  id<MTLComputePipelineState> reduce_primal_relations_ = nil;
  id<MTLComputePipelineState> expand_primal_suffix_context_ = nil;
  id<MTLComputePipelineState> finalize_primal_suffix_ = nil;
  id<MTLComputePipelineState> extract_state_parameters_ = nil;
  id<MTLComputePipelineState> reduce_stages_ = nil;
  id<MTLComputePipelineState> reduce_terminal_ = nil;
  id<MTLComputePipelineState> initial_reduced_state_ = nil;
  id<MTLComputePipelineState> matrix_feedback_ = nil;
  id<MTLComputePipelineState> initialize_costate_maps_ = nil;
  id<MTLComputePipelineState> recover_costates_ = nil;
  id<MTLComputePipelineState> finalize_feedback_ = nil;
  id<MTLComputePipelineState> initialize_state_maps_ = nil;
  id<MTLComputePipelineState> reconstruct_primal_ = nil;
  id<MTLComputePipelineState> build_dual_parameters_ = nil;
  id<MTLComputePipelineState> build_dual_parameters_threadgroup_sliced_ = nil;
  id<MTLComputePipelineState> build_dual_relation_leaves_ = nil;
  id<MTLComputePipelineState> build_dual_relation_leaves_threadgroup_sliced_ =
      nil;
  id<MTLComputePipelineState> reduce_dual_relations_ = nil;
  id<MTLComputePipelineState> solve_dual_root_ = nil;
  id<MTLComputePipelineState> expand_dual_relations_ = nil;
  id<MTLComputePipelineState> recover_parameterized_multipliers_ = nil;
  id<MTLComputePipelineState> recover_initial_multiplier_ = nil;
  id<MTLComputePipelineState> build_objective_terms_ = nil;
  id<MTLComputePipelineState> reduce_objective_ = nil;
  id<MTLComputePipelineState> finalize_objective_ = nil;
  id<MTLComputePipelineState> build_value_leaves_ = nil;
  id<MTLComputePipelineState> reduce_value_ = nil;
  id<MTLComputePipelineState> reduce_value_threadgroup_ = nil;
  id<MTLComputePipelineState> expand_value_context_ = nil;
  id<MTLComputePipelineState> expand_value_context_threadgroup_ = nil;
  id<MTLComputePipelineState> finalize_value_suffix_ = nil;
  id<MTLComputePipelineState> finalize_value_suffix_threadgroup_ = nil;
  id<MTLComputePipelineState> reduce_affine_ = nil;
  id<MTLComputePipelineState> expand_affine_context_ = nil;
  id<MTLComputePipelineState> finalize_affine_prefix_ = nil;
};

[[maybe_unused]] MetalRuntime &Runtime() {
  static MetalRuntime runtime;
  return runtime;
}

class SharedBuffer {
public:
  void Reserve(id<MTLDevice> device, std::size_t bytes) {
    const std::size_t requested = std::max<std::size_t>(bytes, 4);
    if (buffer_ != nil && buffer_.length >= requested)
      return;
    buffer_ = [device newBufferWithLength:requested
                                  options:MTLResourceStorageModeShared];
    if (buffer_ == nil)
      throw std::bad_alloc();
  }

  id<MTLBuffer> get() const { return buffer_; }
  void *contents() const { return buffer_.contents; }

private:
  id<MTLBuffer> buffer_ = nil;
};

struct ThreadWorkspace {
  SharedBuffer dimensions;
  SharedBuffer inputs;
  SharedBuffer outputs;
  SharedBuffer float_workspace;
  SharedBuffer int_workspace;
};

thread_local ThreadWorkspace thread_workspace;

[[maybe_unused]] void BindWorkspace(id<MTLComputeCommandEncoder> encoder,
                                    const ThreadWorkspace &workspace) {
  [encoder setBuffer:workspace.dimensions.get() offset:0 atIndex:0];
  [encoder setBuffer:workspace.inputs.get() offset:0 atIndex:1];
  [encoder setBuffer:workspace.outputs.get() offset:0 atIndex:2];
  [encoder setBuffer:workspace.float_workspace.get() offset:0 atIndex:3];
  [encoder setBuffer:workspace.int_workspace.get() offset:0 atIndex:4];
}

void EncodeKernel(id<MTLComputeCommandEncoder> encoder,
                  id<MTLComputePipelineState> pipeline,
                  const ThreadWorkspace &workspace, const KernelParams &params,
                  std::uint32_t thread_count) {
  if (thread_count == 0)
    return;
  (void)workspace;
  [encoder setComputePipelineState:pipeline];
  [encoder setBytes:&params length:sizeof(params) atIndex:5];
  const NSUInteger group_size =
      std::min<NSUInteger>(thread_count, pipeline.threadExecutionWidth);
  [encoder dispatchThreads:MTLSizeMake(thread_count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(group_size, 1, 1)];
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

void EncodeCooperativeKernel(id<MTLComputeCommandEncoder> encoder,
                             id<MTLComputePipelineState> pipeline,
                             const ThreadWorkspace &workspace,
                             const KernelParams &params,
                             std::uint32_t threadgroup_count) {
  if (threadgroup_count == 0)
    return;
  (void)workspace;
  [encoder setComputePipelineState:pipeline];
  [encoder setBytes:&params length:sizeof(params) atIndex:5];
  const NSUInteger group_size = pipeline.threadExecutionWidth;
  [encoder dispatchThreadgroups:MTLSizeMake(threadgroup_count, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(group_size, 1, 1)];
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

[[maybe_unused]] void EncodeCooperativeKernelWithThreadgroupMemory(
    id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline,
    id<MTLDevice> device, const ThreadWorkspace &workspace,
    const KernelParams &params, std::uint32_t threadgroup_count,
    std::size_t float_bytes, std::size_t integer_bytes) {
  if (threadgroup_count == 0)
    return;
  (void)workspace;
  const NSUInteger group_size = pipeline.threadExecutionWidth;
  if (group_size > pipeline.maxTotalThreadsPerThreadgroup) {
    throw std::runtime_error(
        "CLQR Metal cooperative kernel exceeds the pipeline thread limit");
  }
  const std::size_t total_threadgroup_bytes =
      CheckedSizeSum({std::size_t(pipeline.staticThreadgroupMemoryLength),
                      float_bytes, integer_bytes},
                     "Metal threadgroup memory");
  if (total_threadgroup_bytes >
      std::size_t(device.maxThreadgroupMemoryLength)) {
    throw std::runtime_error(
        "CLQR Metal reduced-stage scratch exceeds device threadgroup memory");
  }
  [encoder setComputePipelineState:pipeline];
  [encoder setBytes:&params length:sizeof(params) atIndex:5];
  [encoder setThreadgroupMemoryLength:float_bytes atIndex:0];
  [encoder setThreadgroupMemoryLength:integer_bytes atIndex:1];
  [encoder dispatchThreadgroups:MTLSizeMake(threadgroup_count, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(group_size, 1, 1)];
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

[[maybe_unused]] bool EncodeCooperativeKernelWithThreadgroupMemoryAtOccupancy(
    id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline,
    id<MTLDevice> device, const ThreadWorkspace &workspace,
    const KernelParams &params, std::uint32_t threadgroup_count,
    std::size_t float_bytes, std::size_t minimum_resident_groups) {
  if (threadgroup_count == 0)
    return true;
  (void)workspace;
  const NSUInteger group_size = pipeline.threadExecutionWidth;
  if (group_size > pipeline.maxTotalThreadsPerThreadgroup ||
      !HasCooperativeThreadgroupOccupancy(
          std::size_t(pipeline.staticThreadgroupMemoryLength), float_bytes,
          std::size_t(device.maxThreadgroupMemoryLength),
          minimum_resident_groups)) {
    return false;
  }
  [encoder setComputePipelineState:pipeline];
  [encoder setBytes:&params length:sizeof(params) atIndex:5];
  [encoder setThreadgroupMemoryLength:float_bytes atIndex:0];
  [encoder dispatchThreadgroups:MTLSizeMake(threadgroup_count, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(group_size, 1, 1)];
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  return true;
}

[[maybe_unused]] bool EncodeLaneSlicedKernelWithThreadgroupMemory(
    id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline,
    id<MTLDevice> device, const ThreadWorkspace &workspace,
    const KernelParams &params, std::uint32_t thread_count,
    std::size_t per_lane_float_bytes, std::size_t per_lane_integer_bytes) {
  if (thread_count == 0)
    return true;
  (void)workspace;
  const std::size_t per_lane_bytes =
      CheckedSizeSum({per_lane_float_bytes, per_lane_integer_bytes},
                     "Metal per-lane threadgroup memory");
  const std::size_t static_bytes =
      std::size_t(pipeline.staticThreadgroupMemoryLength);
  const std::size_t maximum_bytes =
      std::size_t(device.maxThreadgroupMemoryLength);
  if (per_lane_bytes == 0 || static_bytes > maximum_bytes)
    return false;
  const NSUInteger lanes =
      static_cast<NSUInteger>(PlanLaneSlicedThreadgroupLanes(
          static_bytes, maximum_bytes, per_lane_bytes,
          pipeline.maxTotalThreadsPerThreadgroup,
          pipeline.threadExecutionWidth));
  if (lanes == 0)
    return false;
  const std::size_t float_bytes =
      CheckedSizeProduct({per_lane_float_bytes, std::size_t(lanes)},
                         "Metal lane-sliced float memory");
  const std::size_t integer_bytes =
      CheckedSizeProduct({per_lane_integer_bytes, std::size_t(lanes)},
                         "Metal lane-sliced integer memory");
  [encoder setComputePipelineState:pipeline];
  [encoder setBytes:&params length:sizeof(params) atIndex:5];
  [encoder setThreadgroupMemoryLength:float_bytes atIndex:0];
  [encoder setThreadgroupMemoryLength:integer_bytes atIndex:1];
  const NSUInteger groups = (NSUInteger(thread_count) + lanes - 1u) / lanes;
  [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(lanes, 1, 1)];
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  return true;
}

class PackedInputWriter {
public:
  PackedInputWriter(float *destination, std::size_t capacity)
      : destination_(destination), capacity_(capacity) {}

  void Copy(std::uint32_t offset, const float *source, std::size_t count) {
    const std::size_t begin = offset;
    if (begin < cursor_ || begin > capacity_ || count > capacity_ - begin) {
      throw std::runtime_error("invalid CLQR Metal packed-input layout");
    }
    ZeroRange(cursor_, begin);
    if (count != 0)
      std::memcpy(destination_ + begin, source, count * sizeof(float));
    cursor_ = begin + count;
  }

  void Finish() {
    ZeroRange(cursor_, capacity_);
    cursor_ = capacity_;
  }

private:
  void ZeroRange(std::size_t begin, std::size_t end) {
    if (end > begin)
      std::memset(destination_ + begin, 0, (end - begin) * sizeof(float));
  }

  float *destination_;
  std::size_t capacity_;
  std::size_t cursor_ = 0;
};

[[maybe_unused]] void CopyOutputFloats(float *destination, const float *source,
                                       std::size_t count) {
  if (count != 0)
    std::memcpy(destination, source, count * sizeof(float));
}

[[maybe_unused]] void EncodeReductionTree(id<MTLComputeCommandEncoder> encoder,
                                          id<MTLComputePipelineState> pipeline,
                                          const ThreadWorkspace &workspace,
                                          const KernelParams &base,
                                          const TreePlan &tree) {
  for (std::size_t level = 0; level + 1 < tree.counts.size(); ++level) {
    KernelParams invocation = base;
    invocation.child_offset = tree.offsets[level];
    invocation.parent_offset = tree.offsets[level + 1];
    invocation.child_count = tree.counts[level];
    invocation.parent_count = tree.counts[level + 1];
    invocation.phase_detail = static_cast<std::uint32_t>(level);
    EncodeKernel(encoder, pipeline, workspace, invocation,
                 invocation.parent_count);
  }
}

[[maybe_unused]] void
EncodeCooperativeReductionTree(id<MTLComputeCommandEncoder> encoder,
                               id<MTLComputePipelineState> pipeline,
                               const ThreadWorkspace &workspace,
                               const KernelParams &base, const TreePlan &tree) {
  for (std::size_t level = 0; level + 1 < tree.counts.size(); ++level) {
    KernelParams invocation = base;
    invocation.child_offset = tree.offsets[level];
    invocation.parent_offset = tree.offsets[level + 1];
    invocation.child_count = tree.counts[level];
    invocation.parent_count = tree.counts[level + 1];
    invocation.phase_detail = static_cast<std::uint32_t>(level);
    EncodeCooperativeKernel(encoder, pipeline, workspace, invocation,
                            invocation.parent_count);
  }
}

[[maybe_unused]] void EncodeCooperativeReductionTreeWithThreadgroupMemory(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> threadgroup_pipeline,
    id<MTLComputePipelineState> global_pipeline, id<MTLDevice> device,
    const ThreadWorkspace &workspace, const KernelParams &base,
    const TreePlan &tree, std::size_t float_bytes) {
  constexpr std::size_t kMinimumResidentThreadgroups = 4;
  for (std::size_t level = 0; level + 1 < tree.counts.size(); ++level) {
    KernelParams invocation = base;
    invocation.child_offset = tree.offsets[level];
    invocation.parent_offset = tree.offsets[level + 1];
    invocation.child_count = tree.counts[level];
    invocation.parent_count = tree.counts[level + 1];
    invocation.phase_detail = static_cast<std::uint32_t>(level);
    if (!EncodeCooperativeKernelWithThreadgroupMemoryAtOccupancy(
            encoder, threadgroup_pipeline, device, workspace, invocation,
            invocation.parent_count, float_bytes,
            kMinimumResidentThreadgroups)) {
      EncodeCooperativeKernel(encoder, global_pipeline, workspace, invocation,
                              invocation.parent_count);
    }
  }
}

[[maybe_unused]] void EncodeCooperativeTreeContexts(
    id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> expand,
    id<MTLComputePipelineState> finalize, const ThreadWorkspace &workspace,
    const KernelParams &base, const TreePlan &tree,
    std::uint32_t temporary_offset) {
  if (tree.counts.size() <= 1)
    return;
  KernelParams invocation = base;
  for (std::size_t level = tree.counts.size() - 2; level > 0; --level) {
    invocation = base;
    invocation.child_offset = tree.offsets[level];
    invocation.parent_offset = tree.offsets[level + 1];
    invocation.child_count = tree.counts[level];
    invocation.parent_count = tree.counts[level + 1];
    invocation.phase_detail = static_cast<std::uint32_t>(level);
    EncodeCooperativeKernel(encoder, expand, workspace, invocation,
                            invocation.parent_count);
  }
  invocation = base;
  invocation.child_offset = tree.offsets[0];
  invocation.parent_offset = tree.offsets[1];
  invocation.child_count = tree.counts[0];
  invocation.parent_count = tree.counts[1];
  invocation.temporary_offset = temporary_offset;
  EncodeCooperativeKernel(encoder, finalize, workspace, invocation,
                          invocation.parent_count);
}

[[maybe_unused]] void EncodeCooperativeTreeContextsWithThreadgroupMemory(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> expand_threadgroup,
    id<MTLComputePipelineState> expand_global,
    id<MTLComputePipelineState> finalize_threadgroup,
    id<MTLComputePipelineState> finalize_global, id<MTLDevice> device,
    const ThreadWorkspace &workspace, const KernelParams &base,
    const TreePlan &tree, std::uint32_t temporary_offset,
    std::size_t float_bytes) {
  if (tree.counts.size() <= 1)
    return;
  constexpr std::size_t kMinimumResidentThreadgroups = 4;
  KernelParams invocation = base;
  for (std::size_t level = tree.counts.size() - 2; level > 0; --level) {
    invocation = base;
    invocation.child_offset = tree.offsets[level];
    invocation.parent_offset = tree.offsets[level + 1];
    invocation.child_count = tree.counts[level];
    invocation.parent_count = tree.counts[level + 1];
    invocation.phase_detail = static_cast<std::uint32_t>(level);
    if (!EncodeCooperativeKernelWithThreadgroupMemoryAtOccupancy(
            encoder, expand_threadgroup, device, workspace, invocation,
            invocation.parent_count, float_bytes,
            kMinimumResidentThreadgroups)) {
      EncodeCooperativeKernel(encoder, expand_global, workspace, invocation,
                              invocation.parent_count);
    }
  }
  invocation = base;
  invocation.child_offset = tree.offsets[0];
  invocation.parent_offset = tree.offsets[1];
  invocation.child_count = tree.counts[0];
  invocation.parent_count = tree.counts[1];
  invocation.temporary_offset = temporary_offset;
  if (!EncodeCooperativeKernelWithThreadgroupMemoryAtOccupancy(
          encoder, finalize_threadgroup, device, workspace, invocation,
          invocation.parent_count, float_bytes, kMinimumResidentThreadgroups)) {
    EncodeCooperativeKernel(encoder, finalize_global, workspace, invocation,
                            invocation.parent_count);
  }
}

template <typename Buffer>
bool HasShape(const Buffer &buffer,
              std::initializer_list<std::int64_t> expected) {
  const auto actual = buffer.dimensions();
  if (actual.size() != expected.size())
    return false;
  std::size_t index = 0;
  for (const std::int64_t dimension : expected) {
    if (actual[index++] != dimension)
      return false;
  }
  return true;
}

[[maybe_unused]] std::string ValidateActiveDimensions(
    const std::int32_t *dimensions, std::uint32_t stage_count,
    std::uint32_t state_capacity, std::uint32_t control_capacity,
    std::uint32_t mixed_capacity, std::uint32_t state_constraint_capacity,
    std::uint32_t terminal_constraint_capacity) {
  const std::size_t N = stage_count;
  for (std::size_t node = 0; node <= N; ++node) {
    const std::int32_t value = dimensions[node];
    if (value < 0 || std::uint32_t(value) > state_capacity)
      return "state dimension exceeds the padded state capacity";
  }
  for (std::size_t stage = 0; stage < N; ++stage) {
    const std::int32_t control = dimensions[N + 1 + stage];
    if (control < 0 || std::uint32_t(control) > control_capacity)
      return "control dimension exceeds the padded control capacity";
    const std::int32_t mixed = dimensions[2 * N + 1 + stage];
    if (mixed < 0 || std::uint32_t(mixed) > mixed_capacity)
      return "mixed-constraint dimension exceeds its padded capacity";
    const std::int32_t state_constraint = dimensions[3 * N + 1 + stage];
    if (state_constraint < 0 ||
        std::uint32_t(state_constraint) > state_constraint_capacity)
      return "state-constraint dimension exceeds its padded capacity";
  }
  const std::int32_t terminal = dimensions[4 * N + 1];
  if (terminal < 0 || std::uint32_t(terminal) > terminal_constraint_capacity)
    return "terminal-constraint dimension exceeds its padded capacity";
  return {};
}

ffi::Error SolveMetalImpl(
    float tolerance, ffi::BufferR1<ffi::DataType::S32> dimensions,
    ScalarBufferR3 A, ScalarBufferR3 B, ScalarBufferR2 c, ScalarBufferR3 Q,
    ScalarBufferR3 R, ScalarBufferR3 M, ScalarBufferR2 q, ScalarBufferR2 r,
    ScalarBufferR3 C, ScalarBufferR3 D, ScalarBufferR2 d, ScalarBufferR3 E,
    ScalarBufferR2 e, ScalarBufferR2 terminal_E, ScalarBufferR1 terminal_e,
    ScalarBufferR1 initial_state,
    ffi::ResultBufferR1<ffi::DataType::S32> diagnostics,
    ScalarResultR1 objective, ScalarResultR2 states, ScalarResultR2 controls,
    ScalarResultR1 initial_multiplier, ScalarResultR2 dynamics_multipliers,
    ScalarResultR2 mixed_multipliers, ScalarResultR2 state_multipliers,
    ScalarResultR1 terminal_state_multiplier) {
#ifndef CLQR_USE_FLOAT
  return ffi::Error::InvalidArgument(
      "Apple Metal does not support float64; rebuild CLQR with "
      "--config=fp32 to use backend='metal'");
#else
  try {
    const auto a_shape = A.dimensions();
    if (a_shape.size() != 3 || a_shape[0] < 0 || a_shape[1] < 0 ||
        a_shape[1] != a_shape[2]) {
      return ffi::Error::InvalidArgument(
          "A must have shape (stages, max_state, max_state)");
    }
    const std::int64_t stage_count = a_shape[0];
    const std::int64_t state_capacity = a_shape[1];
    const auto b_shape = B.dimensions();
    if (b_shape.size() != 3 || b_shape[0] != stage_count ||
        b_shape[1] != state_capacity || b_shape[2] < 0) {
      return ffi::Error::InvalidArgument(
          "B must have shape (stages, max_state, max_control)");
    }
    const std::int64_t control_capacity = b_shape[2];
    const auto c_constraint_shape = C.dimensions();
    const auto e_constraint_shape = E.dimensions();
    const auto terminal_constraint_shape = terminal_E.dimensions();
    if (c_constraint_shape.size() != 3 ||
        c_constraint_shape[0] != stage_count || c_constraint_shape[1] < 0 ||
        c_constraint_shape[2] != state_capacity ||
        e_constraint_shape.size() != 3 ||
        e_constraint_shape[0] != stage_count || e_constraint_shape[1] < 0 ||
        e_constraint_shape[2] != state_capacity ||
        terminal_constraint_shape.size() != 2 ||
        terminal_constraint_shape[0] < 0 ||
        terminal_constraint_shape[1] != state_capacity) {
      return ffi::Error::InvalidArgument(
          "constraint matrices have inconsistent padded shapes");
    }
    const std::int64_t mixed_capacity = c_constraint_shape[1];
    const std::int64_t state_constraint_capacity = e_constraint_shape[1];
    const std::int64_t terminal_constraint_capacity =
        terminal_constraint_shape[0];
    const std::int64_t maximum =
        std::max({stage_count, state_capacity, control_capacity, mixed_capacity,
                  state_constraint_capacity, terminal_constraint_capacity});
    if (maximum > std::numeric_limits<std::uint32_t>::max()) {
      return ffi::Error::InvalidArgument(
          "padded Metal problem dimension exceeds uint32");
    }
    if (!(tolerance > 0.0f) || !std::isfinite(tolerance)) {
      return ffi::Error::InvalidArgument(
          "Metal solve tolerance must be finite and positive");
    }

    if (!HasShape(dimensions, {4 * stage_count + 2}) ||
        !HasShape(c, {stage_count, state_capacity}) ||
        !HasShape(Q, {stage_count + 1, state_capacity, state_capacity}) ||
        !HasShape(R, {stage_count, control_capacity, control_capacity}) ||
        !HasShape(M, {stage_count, state_capacity, control_capacity}) ||
        !HasShape(q, {stage_count + 1, state_capacity}) ||
        !HasShape(r, {stage_count, control_capacity}) ||
        !HasShape(D, {stage_count, mixed_capacity, control_capacity}) ||
        !HasShape(d, {stage_count, mixed_capacity}) ||
        !HasShape(e, {stage_count, state_constraint_capacity}) ||
        !HasShape(terminal_e, {terminal_constraint_capacity}) ||
        !HasShape(initial_state, {state_capacity})) {
      return ffi::Error::InvalidArgument(
          "one or more packed input arrays have inconsistent shapes");
    }
    if (!HasShape(*diagnostics, {3}) || !HasShape(*objective, {1}) ||
        !HasShape(*states, {stage_count + 1, state_capacity}) ||
        !HasShape(*controls, {stage_count, control_capacity}) ||
        !HasShape(*initial_multiplier, {state_capacity}) ||
        !HasShape(*dynamics_multipliers, {stage_count, state_capacity}) ||
        !HasShape(*mixed_multipliers, {stage_count, mixed_capacity}) ||
        !HasShape(*state_multipliers,
                  {stage_count, state_constraint_capacity}) ||
        !HasShape(*terminal_state_multiplier, {terminal_constraint_capacity})) {
      return ffi::Error::InvalidArgument(
          "one or more packed output arrays have inconsistent shapes");
    }

    const auto N = static_cast<std::uint32_t>(stage_count);
    const auto nx = static_cast<std::uint32_t>(state_capacity);
    const auto nu = static_cast<std::uint32_t>(control_capacity);
    const auto nc = static_cast<std::uint32_t>(mixed_capacity);
    const auto ne = static_cast<std::uint32_t>(state_constraint_capacity);
    const auto nt = static_cast<std::uint32_t>(terminal_constraint_capacity);
    const std::string dimension_error = ValidateActiveDimensions(
        dimensions.typed_data(), N, nx, nu, nc, ne, nt);
    if (!dimension_error.empty())
      return ffi::Error::InvalidArgument(dimension_error);
    const InvocationLayout layout =
        PlanInvocation(N, nx, nu, nc, ne, nt, tolerance);
    MetalRuntime &runtime = Runtime();
    ThreadWorkspace &workspace = thread_workspace;
    workspace.dimensions.Reserve(runtime.device(), dimensions.element_count() *
                                                       sizeof(std::int32_t));
    workspace.inputs.Reserve(runtime.device(),
                             layout.input_floats * sizeof(float));
    workspace.outputs.Reserve(runtime.device(),
                              layout.output_floats * sizeof(float));
    workspace.float_workspace.Reserve(runtime.device(),
                                      layout.workspace_floats * sizeof(float));
    workspace.int_workspace.Reserve(runtime.device(), layout.workspace_ints *
                                                          sizeof(std::int32_t));
    std::memcpy(workspace.dimensions.contents(), dimensions.typed_data(),
                dimensions.element_count() * sizeof(std::int32_t));
    std::memset(workspace.outputs.contents(), 0,
                layout.output_floats * sizeof(float));
    std::memset(workspace.int_workspace.contents(), 0,
                layout.workspace_ints * sizeof(std::int32_t));
    PackedInputWriter packed(static_cast<float *>(workspace.inputs.contents()),
                             layout.input_floats);
    const KernelParams &base = layout.params;
    KernelParams feasibility = base;
    feasibility.consistency_tolerance = std::max(
        tolerance,
        1.0e-4f * static_cast<float>(layout.node_tree.counts.size() + 1));
    KernelParams multiplier_tree = base;
    multiplier_tree.rank_tolerance = std::max(tolerance, 1.0e-4f);
    multiplier_tree.consistency_tolerance =
        std::max(multiplier_tree.rank_tolerance,
                 2.0e-2f * static_cast<float>(layout.stage_tree.counts.size()));
    KernelParams multiplier_leaf = multiplier_tree;
    multiplier_leaf.consistency_tolerance =
        std::max(multiplier_leaf.rank_tolerance, 2.0e-2f);
    packed.Copy(base.input_A, A.typed_data(), std::size_t(N) * nx * nx);
    packed.Copy(base.input_B, B.typed_data(), std::size_t(N) * nx * nu);
    packed.Copy(base.input_c, c.typed_data(), std::size_t(N) * nx);
    packed.Copy(base.input_Q, Q.typed_data(), std::size_t(N + 1) * nx * nx);
    packed.Copy(base.input_R, R.typed_data(), std::size_t(N) * nu * nu);
    packed.Copy(base.input_M, M.typed_data(), std::size_t(N) * nx * nu);
    packed.Copy(base.input_q, q.typed_data(), std::size_t(N + 1) * nx);
    packed.Copy(base.input_r, r.typed_data(), std::size_t(N) * nu);
    packed.Copy(base.input_C, C.typed_data(), std::size_t(N) * nc * nx);
    packed.Copy(base.input_D, D.typed_data(), std::size_t(N) * nc * nu);
    packed.Copy(base.input_d, d.typed_data(), std::size_t(N) * nc);
    packed.Copy(base.input_E, E.typed_data(), std::size_t(N) * ne * nx);
    packed.Copy(base.input_e, e.typed_data(), std::size_t(N) * ne);
    packed.Copy(base.input_terminal_E, terminal_E.typed_data(),
                std::size_t(nt) * nx);
    packed.Copy(base.input_terminal_e, terminal_e.typed_data(), nt);
    packed.Copy(base.input_initial_state, initial_state.typed_data(), nx);
    packed.Finish();

    id<MTLCommandBuffer> metal_command_buffer = [runtime.queue() commandBuffer];
    if (metal_command_buffer == nil)
      throw std::runtime_error("failed to create CLQR Metal command buffer");
    id<MTLComputeCommandEncoder> command_buffer =
        [metal_command_buffer computeCommandEncoder];
    if (command_buffer == nil)
      throw std::runtime_error("failed to create CLQR Metal command encoder");
    BindWorkspace(command_buffer, workspace);
    EncodeKernel(command_buffer, runtime.check_finite_inputs(), workspace, base,
                 N + 1);
    if (N < 64u ||
        !EncodeLaneSlicedKernelWithThreadgroupMemory(
            command_buffer, runtime.build_primal_leaves_threadgroup_sliced(),
            runtime.device(), workspace,
            WithScratch(feasibility, layout.primal_leaves), N + 1,
            layout.primal_leaf_float_bytes, layout.primal_leaf_integer_bytes)) {
      EncodeKernel(command_buffer, runtime.build_primal_leaves(), workspace,
                   WithScratch(feasibility, layout.primal_leaves), N + 1);
    }
    EncodeCooperativeReductionTree(
        command_buffer, runtime.reduce_primal_relations(), workspace,
        WithScratch(feasibility, layout.primal_relations), layout.node_tree);
    EncodeCooperativeTreeContexts(
        command_buffer, runtime.expand_primal_suffix_context(),
        runtime.finalize_primal_suffix(), workspace,
        WithScratch(feasibility, layout.primal_relations), layout.node_tree,
        layout.node_tree.slots);
    KernelParams invocation = WithScratch(feasibility, layout.state_parameters);
    invocation.child_offset = layout.node_tree.offsets[0];
    EncodeKernel(command_buffer, runtime.extract_state_parameters(), workspace,
                 invocation, N + 1);
    EncodeCooperativeKernelWithThreadgroupMemory(
        command_buffer, runtime.reduce_stages(), runtime.device(), workspace,
        feasibility, N, layout.reduced_stage_float_bytes,
        layout.reduced_stage_integer_bytes);
    EncodeCooperativeKernelWithThreadgroupMemory(
        command_buffer, runtime.reduce_terminal(), runtime.device(), workspace,
        base, 1, layout.reduced_terminal_float_bytes,
        layout.reduced_terminal_integer_bytes);
    EncodeKernel(command_buffer, runtime.initial_reduced_state(), workspace,
                 base, 1);
    EncodeCooperativeKernel(command_buffer, runtime.build_value_leaves(),
                            workspace, WithScratch(base, layout.value_leaves),
                            N + 1);
    EncodeCooperativeReductionTreeWithThreadgroupMemory(
        command_buffer, runtime.reduce_value_threadgroup(),
        runtime.reduce_value(), runtime.device(), workspace,
        WithScratch(base, layout.value_compositions), layout.node_tree,
        layout.value_composition_float_bytes);
    EncodeCooperativeTreeContextsWithThreadgroupMemory(
        command_buffer, runtime.expand_value_context_threadgroup(),
        runtime.expand_value_context(),
        runtime.finalize_value_suffix_threadgroup(),
        runtime.finalize_value_suffix(), runtime.device(), workspace,
        WithScratch(base, layout.value_compositions), layout.node_tree,
        layout.node_tree.slots, layout.value_composition_float_bytes);
    EncodeCooperativeKernelWithThreadgroupMemory(
        command_buffer, runtime.matrix_feedback(), runtime.device(), workspace,
        base, N, layout.feedback_float_bytes, layout.feedback_integer_bytes);
    EncodeKernel(command_buffer, runtime.initialize_costate_maps(), workspace,
                 WithScratch(base, layout.affine_rhs), N);
    EncodeCooperativeReductionTree(command_buffer, runtime.reduce_affine(),
                                   workspace, base, layout.stage_tree);
    EncodeCooperativeTreeContexts(
        command_buffer, runtime.expand_affine_context(),
        runtime.finalize_affine_prefix(), workspace, base, layout.stage_tree,
        layout.stage_tree.slots);
    EncodeKernel(command_buffer, runtime.recover_costates(), workspace, base,
                 N + 1);
    EncodeKernel(command_buffer, runtime.finalize_feedback(), workspace,
                 WithScratch(base, layout.affine_rhs), N);
    EncodeKernel(command_buffer, runtime.initialize_state_maps(), workspace,
                 base, N);
    EncodeCooperativeReductionTree(command_buffer, runtime.reduce_affine(),
                                   workspace, base, layout.stage_tree);
    EncodeCooperativeTreeContexts(
        command_buffer, runtime.expand_affine_context(),
        runtime.finalize_affine_prefix(), workspace, base, layout.stage_tree,
        layout.stage_tree.slots);
    EncodeKernel(command_buffer, runtime.reconstruct_primal(), workspace, base,
                 N + 1);
    if (N < 64u ||
        !EncodeLaneSlicedKernelWithThreadgroupMemory(
            command_buffer, runtime.build_dual_parameters_threadgroup_sliced(),
            runtime.device(), workspace,
            WithScratch(multiplier_tree, layout.dual_parameters), N,
            layout.dual_parameter_float_bytes,
            layout.dual_parameter_integer_bytes)) {
      EncodeKernel(command_buffer, runtime.build_dual_parameters(), workspace,
                   WithScratch(multiplier_tree, layout.dual_parameters), N);
    }
    if (N < 64u ||
        !EncodeLaneSlicedKernelWithThreadgroupMemory(
            command_buffer,
            runtime.build_dual_relation_leaves_threadgroup_sliced(),
            runtime.device(), workspace,
            WithScratch(multiplier_leaf, layout.dual_leaves), N,
            layout.dual_leaf_float_bytes, layout.dual_leaf_integer_bytes)) {
      EncodeKernel(command_buffer, runtime.build_dual_relation_leaves(),
                   workspace, WithScratch(multiplier_leaf, layout.dual_leaves),
                   N);
    }
    EncodeReductionTree(
        command_buffer, runtime.reduce_dual_relations(), workspace,
        WithScratch(multiplier_tree, layout.dual_relations), layout.stage_tree);
    if (!layout.stage_tree.counts.empty()) {
      invocation = WithScratch(multiplier_tree, layout.dual_solves);
      invocation.child_offset = layout.stage_tree.offsets.back();
      invocation.parent_offset = layout.stage_tree.offsets.back();
      EncodeKernel(command_buffer, runtime.solve_dual_root(), workspace,
                   invocation, 1);
    }
    if (!layout.stage_tree.counts.empty()) {
      if (layout.stage_tree.counts.size() > 1) {
        for (std::size_t level = layout.stage_tree.counts.size() - 2;;) {
          invocation = WithScratch(multiplier_tree, layout.dual_solves);
          invocation.child_offset = layout.stage_tree.offsets[level];
          invocation.parent_offset = layout.stage_tree.offsets[level + 1];
          invocation.child_count = layout.stage_tree.counts[level];
          invocation.parent_count = layout.stage_tree.counts[level + 1];
          invocation.phase_detail = static_cast<std::uint32_t>(level);
          EncodeKernel(command_buffer, runtime.expand_dual_relations(),
                       workspace, invocation, invocation.parent_count);
          if (level == 0)
            break;
          --level;
        }
      }
    }
    EncodeKernel(command_buffer, runtime.recover_parameterized_multipliers(),
                 workspace, base, N);
    EncodeKernel(command_buffer, runtime.recover_initial_multiplier(),
                 workspace, base, 1);
    EncodeKernel(command_buffer, runtime.build_objective_terms(), workspace,
                 base, N + 1);
    EncodeReductionTree(command_buffer, runtime.reduce_objective(), workspace,
                        base, layout.node_tree);
    invocation = base;
    invocation.child_offset = layout.node_tree.offsets.back();
    EncodeKernel(command_buffer, runtime.finalize_objective(), workspace,
                 invocation, 1);

    [command_buffer endEncoding];
    [metal_command_buffer commit];
    [metal_command_buffer waitUntilCompleted];
    if (metal_command_buffer.status == MTLCommandBufferStatusError) {
      throw std::runtime_error("Metal command buffer failed: " +
                               ErrorText(metal_command_buffer.error));
    }
    const std::int32_t device_code = static_cast<const std::int32_t *>(
        workspace.int_workspace.contents())[base.status];
    std::int32_t *packed_diagnostics = diagnostics->typed_data();
    packed_diagnostics[0] =
        device_code == 2 ? 3 : (device_code == 3 ? 2 : device_code);
    packed_diagnostics[1] = 0;
    packed_diagnostics[2] = 0;
    const float *packed_output =
        static_cast<const float *>(workspace.outputs.contents());
    CopyOutputFloats(objective->typed_data(),
                     packed_output + base.output_objective, 1);
    CopyOutputFloats(states->typed_data(), packed_output + base.output_states,
                     std::size_t(N + 1) * nx);
    CopyOutputFloats(controls->typed_data(),
                     packed_output + base.output_controls, std::size_t(N) * nu);
    CopyOutputFloats(initial_multiplier->typed_data(),
                     packed_output + base.output_initial_multiplier, nx);
    CopyOutputFloats(dynamics_multipliers->typed_data(),
                     packed_output + base.output_dynamics_multipliers,
                     std::size_t(N) * nx);
    CopyOutputFloats(mixed_multipliers->typed_data(),
                     packed_output + base.output_mixed_multipliers,
                     std::size_t(N) * nc);
    CopyOutputFloats(state_multipliers->typed_data(),
                     packed_output + base.output_state_multipliers,
                     std::size_t(N) * ne);
    CopyOutputFloats(terminal_state_multiplier->typed_data(),
                     packed_output + base.output_terminal_state_multiplier, nt);
    return ffi::Error::Success();
  } catch (const std::exception &exception) {
    return ffi::Error::Internal(std::string("CLQR Metal FFI failed: ") +
                                exception.what());
  }
#endif
}

} // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(ClqrMetalFfi, SolveMetalImpl,
                              ffi::Ffi::Bind()
                                  .Attr<float>("tolerance")
                                  .Arg<ffi::BufferR1<ffi::DataType::S32>>()
                                  .Arg<ScalarBufferR3>()
                                  .Arg<ScalarBufferR3>()
                                  .Arg<ScalarBufferR2>()
                                  .Arg<ScalarBufferR3>()
                                  .Arg<ScalarBufferR3>()
                                  .Arg<ScalarBufferR3>()
                                  .Arg<ScalarBufferR2>()
                                  .Arg<ScalarBufferR2>()
                                  .Arg<ScalarBufferR3>()
                                  .Arg<ScalarBufferR3>()
                                  .Arg<ScalarBufferR2>()
                                  .Arg<ScalarBufferR3>()
                                  .Arg<ScalarBufferR2>()
                                  .Arg<ScalarBufferR2>()
                                  .Arg<ScalarBufferR1>()
                                  .Arg<ScalarBufferR1>()
                                  .Ret<ffi::BufferR1<ffi::DataType::S32>>()
                                  .Ret<ScalarBufferR1>()
                                  .Ret<ScalarBufferR2>()
                                  .Ret<ScalarBufferR2>()
                                  .Ret<ScalarBufferR1>()
                                  .Ret<ScalarBufferR2>()
                                  .Ret<ScalarBufferR2>()
                                  .Ret<ScalarBufferR2>()
                                  .Ret<ScalarBufferR1>());
