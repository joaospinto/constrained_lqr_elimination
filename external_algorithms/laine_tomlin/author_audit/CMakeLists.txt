cmake_minimum_required(VERSION 3.20)
project(laine_author_audit LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(LAINE_SOURCE_DIR "" CACHE PATH "Unmodified parallel_lqr checkout")
set(EIGEN_SOURCE_DIR "" CACHE PATH "Eigen 3 headers")
option(AUDIT_EIGEN_ASSERTIONS "Enable Eigen assertions in the original code" OFF)
if(NOT EXISTS "${LAINE_SOURCE_DIR}/src/trajectory.cpp" OR
   NOT EXISTS "${EIGEN_SOURCE_DIR}/Eigen/Core")
  message(FATAL_ERROR "Set LAINE_SOURCE_DIR and EIGEN_SOURCE_DIR")
endif()
set(CLQR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../..")
# Adapt the old include spelling without editing the downloaded source.
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include")
file(CREATE_LINK "${EIGEN_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/include/eigen3" SYMBOLIC)
add_library(laine_author STATIC
  "${LAINE_SOURCE_DIR}/src/trajectory.cpp"
  "${LAINE_SOURCE_DIR}/src/dynamics.cpp"
  "${LAINE_SOURCE_DIR}/src/running_constraint.cpp"
  "${LAINE_SOURCE_DIR}/src/endpoint_constraint.cpp"
  "${LAINE_SOURCE_DIR}/src/running_cost.cpp"
  "${LAINE_SOURCE_DIR}/src/terminal_cost.cpp"
  "${LAINE_SOURCE_DIR}/src/numerical_gradient.cpp")
target_include_directories(laine_author PUBLIC "${LAINE_SOURCE_DIR}/src")
target_include_directories(laine_author SYSTEM PUBLIC
  "${EIGEN_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/include")
# Old headers use std::function without including <functional> themselves.
target_compile_options(laine_author PRIVATE -include functional)
if(AUDIT_EIGEN_ASSERTIONS)
  target_compile_options(laine_author PUBLIC -UNDEBUG)
endif()
add_executable(author_audit audit.cc "${CLQR_ROOT}/src/clqr.cc" "${CLQR_ROOT}/src/linalg.cc")
target_include_directories(author_audit PRIVATE "${CLQR_ROOT}" "${CLQR_ROOT}/include")
target_link_libraries(author_audit PRIVATE laine_author)
enable_testing()
foreach(case analytic-coupling analytic-feedback analytic-affine)
  add_test(NAME ${case} COMMAND author_audit ${case})
endforeach()
