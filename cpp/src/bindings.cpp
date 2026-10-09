#include "fairino_ik/solver.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_native, module) {
  module.doc() = "C++ Pinocchio + constrained eiquadprog dual-arm IK for the Fairino wheel-legged robot";
  auto options = py::class_<fairino_ik::Options>(module, "Options", py::module_local()).def(py::init<>());
#define OPTION(name) options.def_readwrite(#name, &fairino_ik::Options::name)
  OPTION(max_iterations); OPTION(qp_max_iterations); OPTION(dt); OPTION(damping);
  OPTION(step_size); OPTION(max_step_norm); OPTION(tolerance);
  OPTION(position_tolerance); OPTION(orientation_tolerance);
  OPTION(position_cost); OPTION(orientation_cost); OPTION(frame_gain);
  OPTION(posture_weight); OPTION(posture_gain); OPTION(posture_target);
  OPTION(default_velocity_limit); OPTION(torso_velocity_limit); OPTION(torso_step_weight); OPTION(arms_first_iterations);
  OPTION(body_avoidance); OPTION(body_activation_distance); OPTION(body_min_distance);
  OPTION(body_link_radius); OPTION(body_avoidance_weight); OPTION(body_avoidance_gain);
  OPTION(torso_position); OPTION(head_position);
#undef OPTION
  auto result = py::class_<fairino_ik::Result>(module, "Result", py::module_local());
#define RESULT(name) result.def_readonly(#name, &fairino_ik::Result::name)
  RESULT(q); RESULT(full_q); RESULT(success); RESULT(iterations); RESULT(qp_iterations);
  RESULT(qp_active_constraints); RESULT(solve_time_ms); RESULT(solver); RESULT(error);
  RESULT(left_position_error); RESULT(right_position_error);
  RESULT(left_orientation_error); RESULT(right_orientation_error);
  RESULT(body_collision_free); RESULT(body_min_distance); RESULT(body_nearest_segment);
  RESULT(status); RESULT(message);
#undef RESULT
  py::class_<fairino_ik::Model>(module, "Model", py::module_local())
      .def(py::init<const std::string&, const std::string&, const std::string&, bool>(),
           py::arg("urdf_path"), py::arg("left_frame"), py::arg("right_frame"),
           py::arg("active_torso") = false)
      .def(py::init<const std::string&, const std::string&, const std::string&,
                    const std::array<double, 16>&, bool>(),
           py::arg("urdf_path"), py::arg("left_frame"), py::arg("right_frame"),
           py::arg("base_pose"), py::arg("active_torso") = false)
      .def_property_readonly("nq", &fairino_ik::Model::nq)
      .def_property_readonly("nv", &fairino_ik::Model::nv)
      .def_property_readonly("active_torso", &fairino_ik::Model::activeTorso)
      .def_property_readonly("joint_names", &fairino_ik::Model::jointNames)
      .def_property_readonly("full_nq", &fairino_ik::Model::fullNq)
      .def_property_readonly("full_nv", &fairino_ik::Model::fullNv)
      .def_property_readonly("joint_limits", &fairino_ik::Model::jointLimits)
      .def_property_readonly("torso_limits", &fairino_ik::Model::torsoLimits)
      .def_property_readonly("head_limits", &fairino_ik::Model::headLimits)
      .def_property_readonly("frame_names", &fairino_ik::Model::frameNames)
      .def_property_readonly("joint_coordinates", &fairino_ik::Model::jointCoordinates)
      .def("velocity_limits", &fairino_ik::Model::velocityLimits)
      .def("full_configuration", &fairino_ik::Model::fullConfiguration, py::call_guard<py::gil_scoped_release>())
      .def("frame_placements", &fairino_ik::Model::framePlacements, py::call_guard<py::gil_scoped_release>())
      .def("forward_kinematics", &fairino_ik::Model::forwardKinematics, py::call_guard<py::gil_scoped_release>())
      .def("solve_ik", &fairino_ik::Model::solveIK, py::call_guard<py::gil_scoped_release>())
      .def("solve_ik_batch", &fairino_ik::Model::solveIKBatch,
           py::call_guard<py::gil_scoped_release>());
}
