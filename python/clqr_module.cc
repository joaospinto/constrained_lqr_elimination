#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstddef>
#include <stdexcept>
#include <string>

#include "clqr/clqr.h"

namespace nb = nanobind;

namespace {

using Array1 = nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::c_contig>;
using Array2 = nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig>;

nb::object Get(nb::dict dict, const char* key, bool required = true) {
  PyObject* value = PyDict_GetItemString(dict.ptr(), key);
  if (value != nullptr) return nb::borrow<nb::object>(value);
  if (PyErr_Occurred()) nb::raise_python_error();
  if (required) throw nb::key_error(("missing required key '" + std::string(key) + "'").c_str());
  return nb::none();
}

void Set(nb::dict dict, const char* key, nb::handle value) {
  if (PyDict_SetItemString(dict.ptr(), key, value.ptr()) != 0) nb::raise_python_error();
}

clqr::Vector ReadVector(nb::handle object, const char* name) {
  Array1 array;
  try {
    array = nb::cast<Array1>(object);
  } catch (const std::exception& e) {
    throw std::runtime_error("failed to read vector '" + std::string(name) + "': " + e.what());
  }
  clqr::Vector out(static_cast<std::size_t>(array.shape(0)));
  const double* data = array.data();
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = data[i];
  return out;
}

clqr::Matrix ReadMatrix(nb::handle object, const char* name) {
  Array2 array;
  try {
    array = nb::cast<Array2>(object);
  } catch (const std::exception& e) {
    throw std::runtime_error("failed to read matrix '" + std::string(name) + "': " + e.what());
  }
  clqr::Matrix out(static_cast<std::size_t>(array.shape(0)),
                   static_cast<std::size_t>(array.shape(1)));
  const double* data = array.data();
  for (std::size_t i = 0; i < out.rows(); ++i) {
    for (std::size_t j = 0; j < out.cols(); ++j) {
      out(i, j) = data[i * out.cols() + j];
    }
  }
  return out;
}

clqr::Matrix ReadOptionalMatrix(nb::dict dict, const char* key, std::size_t rows,
                                std::size_t cols) {
  nb::object value = Get(dict, key, false);
  if (value.is_none()) return clqr::Matrix(rows, cols);
  return ReadMatrix(value, key);
}

clqr::Vector ReadOptionalVector(nb::dict dict, const char* key, std::size_t size) {
  nb::object value = Get(dict, key, false);
  if (value.is_none()) return clqr::Vector(size);
  return ReadVector(value, key);
}

clqr::Stage ReadStage(nb::handle object) {
  nb::dict dict;
  try {
    dict = nb::cast<nb::dict>(object);
  } catch (const std::exception& e) {
    throw std::runtime_error("failed to read stage dict: " + std::string(e.what()));
  }
  clqr::Stage stage;
  stage.A = ReadMatrix(Get(dict, "A"), "A");
  stage.B = ReadMatrix(Get(dict, "B"), "B");
  stage.c = ReadVector(Get(dict, "c"), "c");
  stage.Q = ReadMatrix(Get(dict, "Q"), "Q");
  stage.R = ReadMatrix(Get(dict, "R"), "R");
  stage.M = ReadMatrix(Get(dict, "M"), "M");
  stage.q = ReadVector(Get(dict, "q"), "q");
  stage.r = ReadVector(Get(dict, "r"), "r");
  stage.C = ReadOptionalMatrix(dict, "C", 0, stage.A.cols());
  stage.D = ReadOptionalMatrix(dict, "D", stage.C.rows(), stage.B.cols());
  stage.d = ReadOptionalVector(dict, "d", stage.C.rows());
  stage.E = ReadOptionalMatrix(dict, "E", 0, stage.A.cols());
  stage.e = ReadOptionalVector(dict, "e", stage.E.rows());
  return stage;
}

auto VectorViewToNumpy(const clqr::VectorView& x) {
  double* data = new double[x.size];
  for (std::size_t i = 0; i < x.size; ++i) data[i] = x[i];
  nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
  return nb::ndarray<nb::numpy, double, nb::ndim<1>>(
      data, {static_cast<unsigned long>(x.size)}, owner);
}

nb::list VectorViewListToPython(const clqr::VectorView* vectors, std::size_t count) {
  nb::list out;
  for (std::size_t i = 0; i < count; ++i) out.append(VectorViewToNumpy(vectors[i]));
  return out;
}

nb::dict Solve(nb::object problem_object,
               clqr::Scalar tolerance = clqr::SolveOptions{}.tolerance) {
  std::string step = "start";
  try {
    step = "cast problem";
    nb::dict problem = nb::cast<nb::dict>(problem_object);
    clqr::Problem cproblem;
    step = "read stages";
    nb::object stages_object = Get(problem, "stages");
    nb::sequence stages;
    try {
      stages = nb::cast<nb::sequence>(stages_object);
    } catch (const std::exception& e) {
      throw std::runtime_error("failed to read stages sequence: " + std::string(e.what()));
    }
    cproblem.stages.reserve(static_cast<std::size_t>(nb::len(stages)));
    for (nb::handle item : stages) cproblem.stages.push_back(ReadStage(item));
    step = "read terminal";
    cproblem.terminal_Q = ReadMatrix(Get(problem, "terminal_Q"), "terminal_Q");
    cproblem.terminal_q = ReadVector(Get(problem, "terminal_q"), "terminal_q");
    cproblem.terminal_E =
        ReadOptionalMatrix(problem, "terminal_E", 0, cproblem.terminal_Q.rows());
    cproblem.terminal_e =
        ReadOptionalVector(problem, "terminal_e", cproblem.terminal_E.rows());
    cproblem.initial_state = ReadVector(Get(problem, "initial_state"), "initial_state");

    step = "solve";
    clqr::SolveOptions options;
    options.tolerance = tolerance;
    clqr::Workspace workspace;
    workspace.Reserve(cproblem, options);
    clqr::SolutionView solution = clqr::Solve(cproblem, workspace, options);

    step = "build result";
    nb::dict result;
    nb::str status(clqr::StatusName(solution.status));
    nb::str message(solution.message);
    nb::str newton_kkt_diagnostic(solution.newton_kkt_diagnostic);
    nb::float_ objective(solution.objective);
    nb::list states = VectorViewListToPython(solution.states, solution.state_count);
    nb::list controls = VectorViewListToPython(solution.controls, solution.control_count);
    nb::list dynamics_multipliers =
        VectorViewListToPython(solution.dynamics_multipliers,
                               solution.dynamics_multiplier_count);
    nb::list mixed_multipliers =
        VectorViewListToPython(solution.mixed_multipliers, solution.mixed_multiplier_count);
    nb::list state_multipliers =
        VectorViewListToPython(solution.state_multipliers, solution.state_multiplier_count);
    Set(result, "status", status);
    Set(result, "message", message);
    result[nb::str("newton_kkt_singular")] = nb::bool_(solution.newton_kkt_singular);
    result[nb::str("newton_kkt_wrong_inertia")] =
        nb::bool_(solution.newton_kkt_wrong_inertia);
    Set(result, "newton_kkt_diagnostic", newton_kkt_diagnostic);
    Set(result, "objective", objective);
    Set(result, "states", states);
    Set(result, "controls", controls);
    result[nb::str("initial_multiplier")] = VectorViewToNumpy(solution.initial_multiplier);
    Set(result, "dynamics_multipliers", dynamics_multipliers);
    Set(result, "mixed_multipliers", mixed_multipliers);
    Set(result, "state_multipliers", state_multipliers);
    result[nb::str("terminal_state_multiplier")] =
        VectorViewToNumpy(solution.terminal_state_multiplier);
    return result;
  } catch (const std::exception& e) {
    throw std::runtime_error("clqr.solve failed during " + step + ": " + e.what());
  }
}

}  // namespace

NB_MODULE(_clqr, module) {
  module.doc() = "Sequential constrained LQR solver.";
  module.def("solve", &Solve, nb::arg("problem"),
             nb::arg("tolerance") = clqr::SolveOptions{}.tolerance);
}
