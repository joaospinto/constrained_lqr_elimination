cmake_minimum_required(VERSION 3.24)
project(constrained_lqr_elimination LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CLQR_ENABLE_CUDA "Build the optional CUDA backend" OFF)
option(CLQR_BUILD_TESTS "Build tests" ON)
option(CLQR_BUILD_BENCHMARKS "Build benchmarks" ON)
set(CLQR_PRECISION "FP64" CACHE STRING "Arithmetic precision: FP64 or FP32")
set_property(CACHE CLQR_PRECISION PROPERTY STRINGS FP64 FP32)
if(NOT CLQR_PRECISION STREQUAL "FP64" AND
   NOT CLQR_PRECISION STREQUAL "FP32")
  message(FATAL_ERROR "CLQR_PRECISION must be FP64 or FP32")
endif()

add_library(clqr STATIC src/clqr.cc src/linalg.cc)
target_include_directories(clqr PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(clqr PRIVATE $<$<CONFIG:Release>:-O3>)
if(CLQR_PRECISION STREQUAL "FP32")
  target_compile_definitions(clqr PUBLIC CLQR_USE_FLOAT=1)
endif()

if(CLQR_BUILD_TESTS)
  enable_testing()
  add_executable(clqr_test tests/clqr_test.cc)
  target_link_libraries(clqr_test PRIVATE clqr)
  add_test(NAME clqr_test COMMAND clqr_test)

  add_executable(clqr_cuda_kernel_emulation_test
                 tests/cuda_kernel_emulation_test.cc)
  target_include_directories(clqr_cuda_kernel_emulation_test
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/emulated_cuda)
  target_link_libraries(clqr_cuda_kernel_emulation_test PRIVATE clqr)
  add_test(NAME clqr_cuda_kernel_emulation_test
           COMMAND clqr_cuda_kernel_emulation_test)
endif()

if(CLQR_BUILD_BENCHMARKS)
  add_executable(clqr_benchmark benchmarks/clqr_benchmark.cc)
  target_link_libraries(clqr_benchmark PRIVATE clqr)
endif()

if(CLQR_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_library(clqr_cuda STATIC src/cuda_solver.cu)
  target_include_directories(clqr_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(clqr_cuda PUBLIC clqr CUDA::cudart)
  target_compile_features(clqr_cuda PUBLIC cxx_std_17 cuda_std_17)
  set_target_properties(
    clqr_cuda
    PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      POSITION_INDEPENDENT_CODE ON
  )
  target_compile_options(
    clqr_cuda
    PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-lineinfo>
  )

  if(CLQR_BUILD_TESTS)
    add_executable(clqr_cuda_test tests/cuda_solver_test.cc)
    target_link_libraries(clqr_cuda_test PRIVATE clqr_cuda)
    add_test(NAME clqr_cuda_test COMMAND clqr_cuda_test)

    add_executable(clqr_cuda_jax_fixture tests/cuda_jax_fixture.cc)
    target_link_libraries(clqr_cuda_jax_fixture PRIVATE clqr_cuda)
  endif()

  if(CLQR_BUILD_BENCHMARKS)
    add_executable(clqr_cuda_benchmark benchmarks/cuda_benchmark.cc)
    target_link_libraries(clqr_cuda_benchmark PRIVATE clqr_cuda)
  endif()
else()
  add_library(clqr_cuda_stub STATIC src/cuda_stub.cc)
  target_include_directories(clqr_cuda_stub PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(clqr_cuda_stub PUBLIC clqr)
  if(CLQR_BUILD_TESTS)
    add_executable(clqr_cuda_stub_test tests/cuda_stub_test.cc)
    target_link_libraries(clqr_cuda_stub_test PRIVATE clqr_cuda_stub)
    add_test(NAME clqr_cuda_stub_test COMMAND clqr_cuda_stub_test)

    add_executable(clqr_cuda_jax_fixture tests/cuda_jax_fixture.cc)
    target_link_libraries(clqr_cuda_jax_fixture PRIVATE clqr_cuda_stub)
  endif()
endif()
