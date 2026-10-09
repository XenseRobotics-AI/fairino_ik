#include "fairino_ik/solver.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <tinyxml2.h>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/spatial/explog.hpp>

namespace fairino_ik {
namespace {

constexpr double kFiniteLimit = 1e6;

// Operational flange frames added below each wrist link.  The wrist mesh
// (left/right_waist2_Link.STL) ends 0.114 m along the link's -x axis.
constexpr std::array<const char*, 2> kWristFrameNames{{
    "left_waist2_Link", "right_waist2_Link"}};
constexpr std::array<const char*, 2> kFlangeFrameNames{{
    "left_flange", "right_flange"}};
const Eigen::Vector3d kFlangeOffset(-0.114, 0.0, 0.0);

// Torso collision boxes, expressed in their own link frames.  They are the
// bounding boxes of the URDF meshes; body_Link is cut at z = 0.45 m so the
// neck and head mount do not block the space above the shoulders.
struct BodyBox {
  const char* frame;
  Eigen::Vector3d center;
  Eigen::Vector3d half_extents;
};
const std::array<BodyBox, kBodyBoxCount> kBodyBoxes{{
    {"body_Link", Eigen::Vector3d(0.0, 0.0075, 0.225),
     Eigen::Vector3d(0.16, 0.1687, 0.225)},
    {"waist_Link", Eigen::Vector3d(0.0, 0.03005, -0.1291),
     Eigen::Vector3d(0.118, 0.10995, 0.1859)},
}};

// Shoulder, elbow, wrist center and flange.  Coincident URDF frames
// (upperarm1/forearm1/waist1) are skipped so every segment has length.
constexpr std::array<std::array<const char*, 4>, 2> kBodyArmFrameNames{{
    {{"left_shoulder_Link", "left_upperarm2_Link", "left_forearm2_Link",
      "left_flange"}},
    {{"right_shoulder_Link", "right_upperarm2_Link", "right_forearm2_Link",
      "right_flange"}},
}};

constexpr std::array<double, 4> kSegmentSamples{{0.0, 0.25, 0.5, 0.75}};

// Labels depend only on the fixed sample layout, not on the configuration.
const auto kBodySegmentLabels = [] {
  std::array<std::array<std::array<std::string, kBodyBoxCount>, 4>, 2> labels;
  for (std::size_t side = 0; side < labels.size(); ++side) {
    for (std::size_t segment = 0; segment < labels[side].size(); ++segment) {
      std::string name = kBodyArmFrameNames[side][segment];
      if (segment + 1 < labels[side].size())
        name += std::string("->") + kBodyArmFrameNames[side][segment + 1];
      for (std::size_t box = 0; box < kBodyBoxCount; ++box)
        labels[side][segment][box] = name + " vs " + kBodyBoxes[box].frame;
    }
  }
  return labels;
}();

struct BoxDistance {
  double distance = std::numeric_limits<double>::infinity();
  Eigen::Vector3d outward_normal = Eigen::Vector3d::Zero();
};

// Signed distance to an oriented box; the normal is returned in world axes.
BoxDistance signedDistanceToBox(
    const Eigen::Vector3d& point,
    const Eigen::Isometry3d& box_pose,
    const Eigen::Vector3d& half_extents) {
  const Eigen::Vector3d local = box_pose.inverse() * point;
  const Eigen::Vector3d closest =
      local.cwiseMax(-half_extents).cwiseMin(half_extents);
  const Eigen::Vector3d delta = local - closest;
  const double outside_norm = delta.norm();
  if (outside_norm > 1e-12) {
    return {outside_norm, box_pose.linear() * (delta / outside_norm)};
  }

  // The point is inside (or exactly on) the box.  A negative signed
  // distance reports penetration, and the nearest face supplies a stable
  // escape direction for the IK repulsion task.
  const Eigen::Vector3d face_distance = half_extents - local.cwiseAbs();
  Eigen::Index axis = 0;
  const double penetration = face_distance.minCoeff(&axis);
  Eigen::Vector3d normal = Eigen::Vector3d::Zero();
  normal[axis] = local[axis] >= 0.0 ? 1.0 : -1.0;
  return {-penetration, box_pose.linear() * normal};
}

bool finite(double value) {
  return std::isfinite(value);
}

Eigen::Vector3d rotationError(
    const Eigen::Matrix3d& current, const Eigen::Matrix3d& target) {
  // Pinocchio's log3 returns the shortest local rotation vector. Convert it
  // into the world frame used by LOCAL_WORLD_ALIGNED Jacobians.
  const Eigen::Vector3d local =
      pinocchio::log3(current.transpose() * target);
  return current * local;
}

}  // namespace

Model::Model(
    const std::string& urdf_path,
    const std::string& left_frame,
    const std::string& right_frame) {
  if (urdf_path.empty()) {
    throw std::invalid_argument("Fairino IK URDF path cannot be empty");
  }
  try {
    pinocchio::urdf::buildModel(urdf_path, full_model_);
  } catch (const std::exception& error) {
    throw std::runtime_error(
        "Could not parse Fairino URDF with C++ Pinocchio: " +
        std::string(error.what()));
  }
  addFlangeFrames();
  full_data_ = pinocchio::Data(full_model_);
  left_frame_name_ = left_frame;
  right_frame_name_ = right_frame;
  for (int i = 0; i < kArmDof; ++i) {
    const auto id = full_model_.getJointId(arm_joint_names_[i]);
    if (id == 0 || id >= static_cast<pinocchio::JointIndex>(full_model_.njoints) ||
        full_model_.joints[id].nq() != 1 || full_model_.joints[id].nv() != 1) {
      throw std::invalid_argument("Missing 1-DoF arm joint " + arm_joint_names_[i]);
    }
    full_q_indices_[i] = full_model_.joints[id].idx_q();
  }
  for (const auto& name : torso_joint_names_) {
    if (!full_model_.existJointName(name) ||
        full_model_.joints[full_model_.getJointId(name)].nq() != 1) {
      throw std::invalid_argument("Missing 1-DoF torso joint " + name);
    }
  }
  if (!full_model_.existJointName(head_joint_name_) ||
      full_model_.joints[full_model_.getJointId(head_joint_name_)].nq() != 1) {
    throw std::invalid_argument("Missing 1-DoF head joint " + head_joint_name_);
  }
  // The current description has no mimic joints, but a gripper added later
  // may.  Resolve URDF mimic coordinates here, not in the Python wrapper.
  tinyxml2::XMLDocument document;
  if (document.LoadFile(urdf_path.c_str()) != tinyxml2::XML_SUCCESS ||
      document.FirstChildElement("robot") == nullptr) {
    throw std::invalid_argument("Could not read URDF mimic joints");
  }
  std::map<int, Mimic> relations;
  const auto coordinate = [&](const char* name) {
    if (!name || !full_model_.existJointName(name))
      throw std::invalid_argument("Missing URDF mimic joint");
    const auto id = full_model_.getJointId(name);
    if (full_model_.joints[id].nq() != 1)
      throw std::invalid_argument("Mimic joints must have one coordinate");
    return full_model_.joints[id].idx_q();
  };
  for (auto* joint = document.FirstChildElement("robot")->FirstChildElement("joint");
       joint; joint = joint->NextSiblingElement("joint")) {
    const auto* mimic = joint->FirstChildElement("mimic");
    if (!mimic) continue;
    Mimic relation{coordinate(joint->Attribute("name")),
                   coordinate(mimic->Attribute("joint")), 1.0, 0.0};
    for (const auto* key : {"multiplier", "offset"}) {
      if (!mimic->Attribute(key)) continue;
      double& value = std::string(key) == "multiplier" ? relation.multiplier : relation.offset;
      if (mimic->QueryDoubleAttribute(key, &value) != tinyxml2::XML_SUCCESS || !finite(value))
        throw std::invalid_argument("Invalid URDF mimic coefficient");
    }
    relations.emplace(relation.child, relation);
  }
  std::map<int, int> visit_state;
  std::function<void(int)> visit = [&](int child) {
    if (relations.count(child) == 0 || visit_state[child] == 2) return;
    if (visit_state[child] == 1) throw std::invalid_argument("Cyclic URDF mimic relation");
    visit_state[child] = 1;
    visit(relations.at(child).parent);
    mimic_.push_back(relations.at(child));
    visit_state[child] = 2;
  };
  for (const auto& relation : relations) visit(relation.first);
  configureReducedModel(Options{});
}

Model::Model(const std::string& urdf_path, const std::string& left_frame,
             const std::string& right_frame, const std::array<double, 16>& base_pose)
    : Model(urdf_path, left_frame, right_frame) {
  base_pose_ = poseFromArray(base_pose);
}

void Model::addFlangeFrames() {
  for (std::size_t side = 0; side < kWristFrameNames.size(); ++side) {
    if (!full_model_.existFrame(kWristFrameNames[side])) {
      throw std::invalid_argument(
          std::string("Missing wrist link ") + kWristFrameNames[side]);
    }
    if (full_model_.existFrame(kFlangeFrameNames[side])) continue;
    const auto wrist_id = full_model_.getFrameId(kWristFrameNames[side]);
    const pinocchio::Frame& wrist = full_model_.frames[wrist_id];
    full_model_.addFrame(pinocchio::Frame(
        kFlangeFrameNames[side], wrist.parentJoint, wrist_id,
        wrist.placement * pinocchio::SE3(Eigen::Matrix3d::Identity(), kFlangeOffset),
        pinocchio::OP_FRAME));
  }
}

void Model::configureReducedModel(const Options& options) {
  std::array<double, kTorsoDof + 1> coordinates{};
  std::copy(options.torso_position.begin(), options.torso_position.end(),
            coordinates.begin());
  coordinates.back() = options.head_position;
  if (reduced_initialized_ && coordinates == fixed_coordinates_) return;

  std::vector<pinocchio::JointIndex> locked;
  locked.reserve(full_model_.njoints);
  for (pinocchio::JointIndex joint_id = 1;
       joint_id < static_cast<pinocchio::JointIndex>(full_model_.njoints);
       ++joint_id) {
    const std::string& name = full_model_.names[joint_id];
    const bool active = std::find(
                            arm_joint_names_.begin(), arm_joint_names_.end(),
                            name) != arm_joint_names_.end();
    if (!active && full_model_.joints[joint_id].nv() > 0) {
      locked.push_back(joint_id);
    }
  }

  const Eigen::VectorXd reference = fullConfigurationImpl(std::vector<double>(kArmDof, 0.0), options);
  reduced_model_ =
      pinocchio::buildReducedModel(full_model_, locked, reference);
  reduced_data_ = pinocchio::Data(reduced_model_);
  if (reduced_model_.nq != kArmDof || reduced_model_.nv != kArmDof) {
    throw std::runtime_error(
        "Fairino C++ IK reduced model must have 14 DoF, got nq=" +
        std::to_string(reduced_model_.nq) + ", nv=" +
        std::to_string(reduced_model_.nv));
  }

  for (int i = 0; i < kArmDof; ++i) {
    const pinocchio::JointIndex joint_id =
        reduced_model_.getJointId(arm_joint_names_[i]);
    if (joint_id == 0 ||
        reduced_model_.names[joint_id] != arm_joint_names_[i]) {
      throw std::runtime_error(
          "Reduced Fairino IK model is missing joint " + arm_joint_names_[i]);
    }
    const int index = reduced_model_.joints[joint_id].idx_q();
    arm_q_indices_[i] = index;
    double lower = reduced_model_.lowerPositionLimit[index];
    double upper = reduced_model_.upperPositionLimit[index];
    if (!finite(lower)) lower = -kFiniteLimit;
    if (!finite(upper)) upper = kFiniteLimit;
    if (lower > upper) {
      throw std::runtime_error(
          "Invalid limits for Fairino IK joint " + arm_joint_names_[i]);
    }
    lower_limits_[i] = lower;
    upper_limits_[i] = upper;
    const int velocity_index = reduced_model_.joints[joint_id].idx_v();
    if (velocity_index != index) throw std::runtime_error("Arm q/v ordering mismatch");
    const double speed = reduced_model_.velocityLimit[velocity_index];
    urdf_velocity_limits_[i] = finite(speed) && speed > 0.0 ? speed : 0.0;
  }

  const auto& left_frame = left_frame_name_;
  const auto& right_frame = right_frame_name_;
  if (!reduced_model_.existFrame(left_frame) ||
      !reduced_model_.existFrame(right_frame)) {
    throw std::runtime_error(
        "Fairino IK end-effector frame not found: " + left_frame + " / " +
        right_frame);
  }
  left_frame_id_ = reduced_model_.getFrameId(left_frame);
  right_frame_id_ = reduced_model_.getFrameId(right_frame);

  for (std::size_t side = 0; side < kBodyArmFrameNames.size(); ++side) {
    for (std::size_t index = 0;
         index < kBodyArmFrameNames[side].size(); ++index) {
      const std::string frame_name = kBodyArmFrameNames[side][index];
      if (!reduced_model_.existFrame(frame_name)) {
        throw std::runtime_error(
            "Reduced Fairino IK model is missing body check frame " +
            frame_name);
      }
      body_arm_frame_ids_[side][index] =
          reduced_model_.getFrameId(frame_name);
    }
  }
  for (std::size_t box = 0; box < kBodyBoxes.size(); ++box) {
    if (!reduced_model_.existFrame(kBodyBoxes[box].frame)) {
      throw std::runtime_error(
          std::string("Reduced Fairino IK model is missing body box frame ") +
          kBodyBoxes[box].frame);
    }
    body_box_frame_ids_[box] = reduced_model_.getFrameId(kBodyBoxes[box].frame);
  }
  fixed_coordinates_ = coordinates;
  reduced_initialized_ = true;
}

Eigen::VectorXd Model::fullConfigurationImpl(const std::vector<double>& q,
                                             const Options& options) const {
  validateQ(q, "q");
  Eigen::VectorXd result = pinocchio::neutral(full_model_);
  for (int i = 0; i < kArmDof; ++i) result[full_q_indices_[i]] = q[i];
  const auto set_locked = [&](const std::string& name, double value) {
    if (!finite(value)) throw std::invalid_argument("Fixed joint values must be finite");
    const int index = full_model_.joints[full_model_.getJointId(name)].idx_q();
    result[index] = std::clamp(value, full_model_.lowerPositionLimit[index],
                               full_model_.upperPositionLimit[index]);
  };
  for (int i = 0; i < kTorsoDof; ++i)
    set_locked(torso_joint_names_[i], options.torso_position[i]);
  set_locked(head_joint_name_, options.head_position);
  for (const auto& relation : mimic_)
    result[relation.child] = relation.multiplier * result[relation.parent] + relation.offset;
  return result;
}

std::vector<double> Model::fullConfiguration(const std::vector<double>& q, const Options& options) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto full = fullConfigurationImpl(q, options);
  return {full.data(), full.data() + full.size()};
}

std::vector<std::array<double, 2>> Model::jointLimits() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::array<double, 2>> result;
  for (int i = 0; i < kArmDof; ++i) result.push_back({lower_limits_[i], upper_limits_[i]});
  return result;
}

std::vector<std::array<double, 2>> Model::torsoLimits() const {
  std::vector<std::array<double, 2>> result;
  for (const auto& name : torso_joint_names_) {
    const int index = full_model_.joints[full_model_.getJointId(name)].idx_q();
    result.push_back({full_model_.lowerPositionLimit[index],
                      full_model_.upperPositionLimit[index]});
  }
  return result;
}

std::array<double, 2> Model::headLimits() const {
  const int index =
      full_model_.joints[full_model_.getJointId(head_joint_name_)].idx_q();
  return {full_model_.lowerPositionLimit[index],
          full_model_.upperPositionLimit[index]};
}

double Model::velocityLimit(int index, const Options& options) const {
  const double urdf = urdf_velocity_limits_[index];
  return urdf > 0.0 ? urdf : options.default_velocity_limit;
}

std::vector<double> Model::velocityLimits(const Options& options) const {
  std::lock_guard<std::mutex> lock(mutex_);
  validateOptions(options);
  std::vector<double> result(kArmDof);
  for (int i = 0; i < kArmDof; ++i) result[i] = velocityLimit(i, options);
  return result;
}

std::vector<std::string> Model::frameNames() const {
  std::vector<std::string> result;
  for (const auto& frame : full_model_.frames) result.push_back(frame.name);
  return result;
}

std::map<std::string, int> Model::jointCoordinates() const {
  std::map<std::string, int> result;
  for (int id = 1; id < full_model_.njoints; ++id)
    if (full_model_.joints[id].nq() == 1)
      result[full_model_.names[id]] = full_model_.joints[id].idx_q();
  return result;
}

std::map<std::string, std::array<double, 16>> Model::framePlacements(
    const std::vector<double>& q, const Options& options) {
  std::lock_guard<std::mutex> lock(mutex_);
  pinocchio::forwardKinematics(full_model_, full_data_, fullConfigurationImpl(q, options));
  pinocchio::updateFramePlacements(full_model_, full_data_);
  std::map<std::string, std::array<double, 16>> result;
  for (std::size_t i = 0; i < full_model_.frames.size(); ++i)
    result[full_model_.frames[i].name] = poseToArray(base_pose_ *
        Eigen::Isometry3d(full_data_.oMf[i].toHomogeneousMatrix()));
  return result;
}

void Model::validateQ(
    const std::vector<double>& q, const char* name) const {
  if (q.size() != kArmDof) {
    throw std::invalid_argument(
        std::string(name) + " must contain 14 values, got " +
        std::to_string(q.size()));
  }
  for (double value : q) {
    if (!finite(value)) {
      throw std::invalid_argument(std::string(name) + " contains NaN or Inf");
    }
  }
}

void Model::validateOptions(const Options& options) const {
  if (options.max_iterations < 1) {
    throw std::invalid_argument("max_iterations must be >= 1");
  }
  if (options.qp_max_iterations < 1) throw std::invalid_argument("qp_max_iterations must be >= 1");
  const std::array<double, 17> values = {
      options.dt,
      options.damping,
      options.step_size,
      options.max_step_norm,
      options.tolerance,
      options.position_tolerance,
      options.orientation_tolerance,
      options.position_cost,
      options.orientation_cost,
      options.frame_gain,
      options.posture_weight,
      options.default_velocity_limit,
      options.body_activation_distance,
      options.body_min_distance,
      options.body_link_radius,
      options.body_avoidance_weight,
      options.body_avoidance_gain,
  };
  for (double value : values) {
    if (!finite(value) || value < 0.0) {
      throw std::invalid_argument(
          "Fairino C++ IK options must be finite and non-negative");
    }
  }
  if (options.dt <= 0.0 ||
      options.step_size <= 0.0 || options.max_step_norm <= 0.0 ||
      options.tolerance <= 0.0 || options.default_velocity_limit <= 0.0 ||
      (options.position_cost == 0.0 && options.orientation_cost == 0.0) ||
      options.posture_gain < 0.0 || !finite(options.posture_gain)) {
    throw std::invalid_argument(
        "Fairino C++ IK core options contain a non-positive value");
  }
  if (options.frame_gain <= 0.0 || options.frame_gain > 1.0) {
    throw std::invalid_argument("frame_gain must be in (0, 1]");
  }
  if (options.body_avoidance &&
      options.body_activation_distance <= options.body_min_distance) {
    throw std::invalid_argument(
        "body_activation_distance must be greater than body_min_distance");
  }
  if (options.body_avoidance &&
      (options.body_avoidance_weight <= 0.0 ||
       options.body_avoidance_gain <= 0.0)) {
    throw std::invalid_argument(
        "body avoidance weight and gain must be positive");
  }
  for (double value : options.torso_position) {
    if (!finite(value)) throw std::invalid_argument("torso_position must be finite");
  }
  if (!finite(options.head_position)) {
    throw std::invalid_argument("head_position must be finite");
  }
  if (!options.posture_target.empty()) {
    validateQ(options.posture_target, "posture_target");
  }
}

Eigen::VectorXd Model::armToReduced(
    const std::vector<double>& q_arm) const {
  validateQ(q_arm, "q");
  Eigen::VectorXd result = pinocchio::neutral(reduced_model_);
  for (int i = 0; i < kArmDof; ++i) {
    result[arm_q_indices_[i]] =
        std::clamp(q_arm[i], lower_limits_[i], upper_limits_[i]);
  }
  return result;
}

std::vector<double> Model::reducedToArm(
    const Eigen::VectorXd& q_reduced) const {
  if (q_reduced.size() != kArmDof) {
    throw std::invalid_argument("reduced q must contain 14 values");
  }
  std::vector<double> result(kArmDof);
  for (int i = 0; i < kArmDof; ++i) {
    result[i] = std::clamp(
        q_reduced[arm_q_indices_[i]], lower_limits_[i], upper_limits_[i]);
  }
  return result;
}

Eigen::Isometry3d Model::poseFromArray(
    const std::array<double, 16>& values) const {
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  for (int row = 0; row < 4; ++row) {
    for (int column = 0; column < 4; ++column) {
      const double value = values[row * 4 + column];
      if (!finite(value)) {
        throw std::invalid_argument("target pose contains NaN or Inf");
      }
      result.matrix()(row, column) = value;
    }
  }
  const double determinant = result.linear().determinant();
  if (!finite(determinant) || std::abs(determinant - 1.0) > 1e-6 ||
      (result.linear().transpose() * result.linear() - Eigen::Matrix3d::Identity()).norm() > 1e-6 ||
      (result.matrix().row(3) - Eigen::RowVector4d(0, 0, 0, 1)).norm() > 1e-8) {
    throw std::invalid_argument("target pose must have a proper rotation and homogeneous last row");
  }
  return result;
}

std::array<double, 16> Model::poseToArray(
    const Eigen::Isometry3d& pose) const {
  std::array<double, 16> result{};
  for (int row = 0; row < 4; ++row) {
    for (int column = 0; column < 4; ++column) {
      result[row * 4 + column] = pose.matrix()(row, column);
    }
  }
  return result;
}

void Model::updateKinematics(const Eigen::VectorXd& q) {
  // The q-taking Jacobian pass also updates joint placements (oMi/liMi).
  pinocchio::computeJointJacobians(reduced_model_, reduced_data_, q);
  pinocchio::updateFramePlacements(reduced_model_, reduced_data_);
}

Eigen::Isometry3d Model::framePose(
    const Eigen::VectorXd& q, pinocchio::FrameIndex frame) {
  updateKinematics(q);
  const pinocchio::SE3& local = reduced_data_.oMf[frame];
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.matrix() = local.toHomogeneousMatrix();
  return base_pose_ * pose;
}

std::vector<double> Model::forwardKinematics(
    const std::vector<double>& q_arm, const std::string& side, const Options& options) {
  std::lock_guard<std::mutex> lock(mutex_);
  validateOptions(options);
  configureReducedModel(options);
  const Eigen::VectorXd q = armToReduced(q_arm);
  const pinocchio::FrameIndex frame =
      (side == "left" || side == "l")
          ? left_frame_id_
          : (side == "right" || side == "r")
                ? right_frame_id_
                : throw std::invalid_argument(
                      "side must be 'left' or 'right'");
  const auto values = poseToArray(framePose(q, frame));
  return std::vector<double>(values.begin(), values.end());
}

Model::PoseErrors Model::evaluate(
    const Eigen::VectorXd& q,
    const Eigen::Isometry3d& left_target,
    const Eigen::Isometry3d& right_target) {
  updateKinematics(q);
  const Eigen::Isometry3d left_current =
      base_pose_ * Eigen::Isometry3d(reduced_data_.oMf[left_frame_id_]
                                         .toHomogeneousMatrix());
  const Eigen::Isometry3d right_current =
      base_pose_ * Eigen::Isometry3d(reduced_data_.oMf[right_frame_id_]
                                         .toHomogeneousMatrix());

  PoseErrors result;
  result.task_error.setZero();
  const Eigen::Vector3d left_position =
      left_target.translation() - left_current.translation();
  const Eigen::Vector3d right_position =
      right_target.translation() - right_current.translation();
  const Eigen::Vector3d left_orientation =
      rotationError(left_current.rotation(), left_target.rotation());
  const Eigen::Vector3d right_orientation =
      rotationError(right_current.rotation(), right_target.rotation());
  result.task_error.segment<3>(0) = left_position;
  result.task_error.segment<3>(3) = left_orientation;
  result.task_error.segment<3>(6) = right_position;
  result.task_error.segment<3>(9) = right_orientation;
  result.left_position = left_position.norm();
  result.left_orientation = left_orientation.norm();
  result.right_position = right_position.norm();
  result.right_orientation = right_orientation.norm();
  result.max_error = std::max(
      std::max(result.left_position, result.left_orientation),
      std::max(result.right_position, result.right_orientation));
  return result;
}

Model::BodyMetrics Model::evaluateBodyAvoidance(
    const Options& options,
    Eigen::Matrix<double, kArmDof, kArmDof>* hessian,
    Eigen::Matrix<double, kArmDof, 1>* rhs,
    Eigen::Matrix<double, kBodySamples, kArmDof>* constraints,
    Eigen::Matrix<double, kBodySamples, 1>* offsets) {
  BodyMetrics metrics;
  if (constraints) constraints->setZero();
  if (offsets) offsets->setOnes();
  if (!options.body_avoidance) return metrics;

  const bool add_task = hessian != nullptr && rhs != nullptr;
  // Torso joints are locked, so the boxes are constant during a solve; the
  // reduced model still carries their frames.  Everything here is in the
  // model root frame, as are the Jacobians.
  std::array<Eigen::Isometry3d, kBodyBoxCount> box_poses;
  std::array<Eigen::Vector3d, kBodyBoxCount> inflated_half_extents;
  for (std::size_t box = 0; box < kBodyBoxes.size(); ++box) {
    const Eigen::Isometry3d frame(
        reduced_data_.oMf[body_box_frame_ids_[box]].toHomogeneousMatrix());
    box_poses[box] = frame * Eigen::Translation3d(kBodyBoxes[box].center);
    inflated_half_extents[box] = kBodyBoxes[box].half_extents +
        Eigen::Vector3d::Constant(options.body_link_radius);
  }
  int sample = 0;

  for (std::size_t side = 0; side < body_arm_frame_ids_.size(); ++side) {
    std::array<Eigen::Vector3d, 4> positions;
    std::array<Eigen::Matrix<double, 3, kArmDof>, 4> linear_jacobians;
    for (std::size_t index = 0;
         index < body_arm_frame_ids_[side].size(); ++index) {
      const auto frame_id = body_arm_frame_ids_[side][index];
      positions[index] = reduced_data_.oMf[frame_id].translation();
      if (add_task) {
        Eigen::Matrix<double, 6, kArmDof> frame_jacobian;
        frame_jacobian.setZero();
        pinocchio::getFrameJacobian(
            reduced_model_, reduced_data_, frame_id,
            pinocchio::LOCAL_WORLD_ALIGNED, frame_jacobian);
        linear_jacobians[index] = frame_jacobian.topRows<3>();
      }
    }

    const auto process_sample = [&](
        const Eigen::Vector3d& point,
        const Eigen::Matrix<double, 3, kArmDof>* point_jacobian,
        const std::array<std::string, kBodyBoxCount>& labels) {
      for (std::size_t box_index = 0; box_index < kBodyBoxes.size(); ++box_index) {
        const int row = sample++;
        const BoxDistance box = signedDistanceToBox(
            point, box_poses[box_index], inflated_half_extents[box_index]);
        if (box.distance < metrics.min_distance) {
          metrics.min_distance = box.distance;
          metrics.nearest_segment = labels[box_index];
        }
        if (!add_task || point_jacobian == nullptr ||
            box.distance >= options.body_activation_distance) {
          continue;
        }

        const Eigen::Matrix<double, 1, kArmDof> distance_jacobian =
            box.outward_normal.transpose() * (*point_jacobian);
        // The shoulder sample is fixed to the torso.  A zero row cannot
        // change the clearance and would only make the QP infeasible.
        if (distance_jacobian.squaredNorm() < 1e-12) continue;
        if (constraints && offsets) {
          constraints->row(row) = distance_jacobian;
          // Linearized minimum-clearance barrier: Jd*step + gain*(d-dmin) >= 0.
          // Capping the gain keeps it conservative for arbitrary user settings.
          (*offsets)[row] = std::min(1.0, options.body_avoidance_gain) *
                            (box.distance - options.body_min_distance);
        }
        const double desired_clearance_rate =
            options.body_avoidance_gain *
            std::max(0.0, options.body_min_distance - box.distance);
        hessian->noalias() +=
            options.body_avoidance_weight *
            distance_jacobian.transpose() * distance_jacobian;
        rhs->noalias() +=
            options.body_avoidance_weight * desired_clearance_rate *
            distance_jacobian.transpose();
      }
    };

    for (std::size_t segment = 0; segment + 1 < positions.size(); ++segment) {
      const auto& label = kBodySegmentLabels[side][segment];
      for (double alpha : kSegmentSamples) {
        const Eigen::Vector3d point =
            (1.0 - alpha) * positions[segment] +
            alpha * positions[segment + 1];
        if (add_task) {
          const Eigen::Matrix<double, 3, kArmDof> point_jacobian =
              (1.0 - alpha) * linear_jacobians[segment] +
              alpha * linear_jacobians[segment + 1];
          process_sample(point, &point_jacobian, label);
        } else {
          process_sample(point, nullptr, label);
        }
      }
    }

    // The segment sampling excludes alpha=1 to avoid duplicate joint-frame
    // points, so add the flange endpoint once.
    const std::size_t last = positions.size() - 1;
    if (add_task) {
      process_sample(
          positions[last], &linear_jacobians[last],
          kBodySegmentLabels[side][last]);
    } else {
      process_sample(
          positions[last], nullptr, kBodySegmentLabels[side][last]);
    }
  }
  return metrics;
}

Result Model::solveIK(
    const std::array<double, 16>& left_values,
    const std::array<double, 16>& right_values,
    const std::vector<double>& q_seed,
    const Options& options) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto started = std::chrono::steady_clock::now();
  validateOptions(options);
  validateQ(q_seed, "q_seed");
  configureReducedModel(options);
  Result result = solveIKUnlocked(left_values, right_values, q_seed, options);
  result.solve_time_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
  return result;
}

std::vector<Result> Model::solveIKBatch(
    const std::vector<std::array<double, 16>>& left_targets,
    const std::vector<std::array<double, 16>>& right_targets,
    const std::vector<std::vector<double>>& q_seeds,
    const Options& options) {
  if (left_targets.size() != right_targets.size() ||
      left_targets.size() != q_seeds.size() || left_targets.empty()) {
    throw std::invalid_argument(
        "left_targets, right_targets and q_seeds must have the same non-zero size");
  }
  std::lock_guard<std::mutex> lock(mutex_);
  validateOptions(options);
  configureReducedModel(options);
  for (const auto& seed : q_seeds) validateQ(seed, "q_seed");
  std::vector<Result> results;
  results.reserve(left_targets.size());
  for (std::size_t i = 0; i < left_targets.size(); ++i) {
    results.push_back(
        solveIKUnlocked(left_targets[i], right_targets[i], q_seeds[i], options));
  }
  return results;
}

Result Model::solveIKUnlocked(
    const std::array<double, 16>& left_values,
    const std::array<double, 16>& right_values,
    const std::vector<double>& q_seed,
    const Options& options) {
  const auto started = std::chrono::steady_clock::now();
  const Eigen::Isometry3d left_target =
      poseFromArray(left_values);
  const Eigen::Isometry3d right_target =
      poseFromArray(right_values);
  Eigen::VectorXd q = armToReduced(q_seed);
  Eigen::VectorXd posture = q;
  if (!options.posture_target.empty()) {
    posture = armToReduced(options.posture_target);
  }
  const double position_tolerance =
      options.position_tolerance > 0.0 ? options.position_tolerance
                                       : options.tolerance;
  const double orientation_tolerance =
      options.orientation_tolerance > 0.0 ? options.orientation_tolerance
                                          : options.tolerance;

  Result result;
  PoseErrors errors = evaluate(q, left_target, right_target);
  BodyMetrics body = evaluateBodyAvoidance(options, nullptr, nullptr);
  auto converged = [&]() {
    return errors.left_position <= position_tolerance &&
           errors.right_position <= position_tolerance &&
           errors.left_orientation <= orientation_tolerance &&
           errors.right_orientation <= orientation_tolerance &&
           (!options.body_avoidance ||
            body.min_distance >= options.body_min_distance);
  };
  if (converged()) {
    result.success = true;
    result.status = "converged";
  } else {
    result.status = "max_iterations";
    Eigen::Matrix<double, 12, kArmDof> jacobian;
    Eigen::Matrix<double, 12, 12> weight =
        Eigen::Matrix<double, 12, 12>::Zero();
    for (int block = 0; block < 2; ++block) {
      weight.block<3, 3>(block * 6, block * 6).diagonal().setConstant(
          options.position_cost);
      weight.block<3, 3>(block * 6 + 3, block * 6 + 3).diagonal().setConstant(
          options.orientation_cost);
    }

    for (int iteration = 1; iteration <= options.max_iterations; ++iteration) {
      // evaluate() has already updated placements and Jacobians at this q,
      // initially and after every applied QP step. Reuse that state here.
      Eigen::Matrix<double, 6, kArmDof> left_jacobian;
      Eigen::Matrix<double, 6, kArmDof> right_jacobian;
      // getFrameJacobian only writes the columns that belong to the frame's
      // kinematic subtree.  Initialize the fixed-size buffers explicitly so
      // the other arm's columns are deterministic zeros.
      left_jacobian.setZero();
      right_jacobian.setZero();
      pinocchio::getFrameJacobian(
          reduced_model_,
          reduced_data_,
          left_frame_id_,
          pinocchio::LOCAL_WORLD_ALIGNED,
          left_jacobian);
      pinocchio::getFrameJacobian(
          reduced_model_,
          reduced_data_,
          right_frame_id_,
          pinocchio::LOCAL_WORLD_ALIGNED,
          right_jacobian);
      // Pinocchio's getFrameJacobian API exposes [linear; angular] rows for
      // this reference frame, which matches the public [position;
      // orientation] task layout.  Rotate both blocks into the public/world
      // frame if a non-identity base pose is supplied.
      const Eigen::Matrix3d base_rotation = base_pose_.rotation();
      left_jacobian.topRows<3>() = base_rotation * left_jacobian.topRows<3>();
      left_jacobian.bottomRows<3>() =
          base_rotation * left_jacobian.bottomRows<3>();
      right_jacobian.topRows<3>() =
          base_rotation * right_jacobian.topRows<3>();
      right_jacobian.bottomRows<3>() =
          base_rotation * right_jacobian.bottomRows<3>();
      jacobian.block<6, kArmDof>(0, 0) = left_jacobian;
      jacobian.block<6, kArmDof>(6, 0) = right_jacobian;

      Eigen::Matrix<double, kArmDof, kArmDof> hessian =
          jacobian.transpose() * weight * jacobian;
      // Strict convexity is required by the active-set solver, even when the
      // caller requests zero damping or the arm is stretched (singular).
      hessian.diagonal().array() += std::max(1e-10, options.damping * options.damping);
      Eigen::Matrix<double, kArmDof, 1> rhs =
          jacobian.transpose() * weight *
          (options.frame_gain * errors.task_error);
      if (options.posture_weight > 0.0) {
        hessian.diagonal().array() += options.posture_weight;
        rhs += options.posture_weight * options.posture_gain * (posture - q);
      }
      Eigen::Matrix<double, kBodySamples, kArmDof> body_constraints;
      Eigen::Matrix<double, kBodySamples, 1> body_offsets;
      body = evaluateBodyAvoidance(options, &hessian, &rhs, &body_constraints, &body_offsets);
      Eigen::Matrix<double, kArmDof, 1> lower, upper, step;
      // Optimize configuration increments, not a post-clipped velocity.
      // The conservative trust box guarantees ||step|| <= max_step_norm.
      const double trust = options.max_step_norm / std::sqrt(double(kArmDof));
      for (int i = 0; i < kArmDof; ++i) {
        const int index = arm_q_indices_[i];
        const double limit = std::min(
            trust, velocityLimit(i, options) * options.dt * options.step_size);
        lower[index] = std::max(lower_limits_[i] - q[index], -limit);
        upper[index] = std::min(upper_limits_[i] - q[index], limit);
      }
      const auto qp_status = qp_.solve(hessian, -options.step_size * rhs, lower, upper,
                                       options.qp_max_iterations, step, body_constraints, body_offsets);
      result.qp_iterations += qp_.iterations();
      result.qp_active_constraints = std::max(result.qp_active_constraints, qp_.activeConstraints());
      if (qp_status != eiquadprog::solvers::RT_EIQUADPROG_OPTIMAL || !step.allFinite() ||
          (step.array() < lower.array() - 1e-8).any() || (step.array() > upper.array() + 1e-8).any()) {
        result.status = qp_status == eiquadprog::solvers::RT_EIQUADPROG_MAX_ITER_REACHED
            ? "qp_max_iterations" : "qp_solver_error";
        result.message = "C++ eiquadprog failed; no unverified QP step was applied (status=" +
                         std::to_string(int(qp_status)) + ")";
        result.iterations = iteration;
        break;
      }
      q += step;
      for (int i = 0; i < kArmDof; ++i) {
        q[arm_q_indices_[i]] =
            std::clamp(q[arm_q_indices_[i]], lower_limits_[i], upper_limits_[i]);
      }
      errors = evaluate(q, left_target, right_target);
      body = evaluateBodyAvoidance(options, nullptr, nullptr);
      result.iterations = iteration;
      if (converged()) {
        result.success = true;
        result.status = "converged";
        break;
      }
    }
  }

  result.q = reducedToArm(q);
  // Keep the public full configuration in a plain vector without another
  // Python/native round trip.  fullConfigurationImpl is already under the
  // model mutex.
  const Eigen::VectorXd full = fullConfigurationImpl(result.q, options);
  result.full_q.assign(full.data(), full.data() + full.size());
  result.error = errors.max_error;
  result.left_position_error = errors.left_position;
  result.left_orientation_error = errors.left_orientation;
  result.right_position_error = errors.right_position;
  result.right_orientation_error = errors.right_orientation;
  result.body_min_distance = body.min_distance;
  result.body_nearest_segment = body.nearest_segment;
  result.body_collision_free =
      !options.body_avoidance ||
      body.min_distance >= options.body_min_distance;
  if (!result.body_collision_free) {
    result.success = false;
    result.status = "body_collision";
    std::ostringstream message;
    message << "C++ dual-arm IK rejected by torso boxes: clearance="
            << body.min_distance << " m, required>="
            << options.body_min_distance << " m";
    if (!body.nearest_segment.empty()) {
      message << ", nearest=" << body.nearest_segment;
    }
    result.message = message.str();
  }
  if (!result.success && result.message.empty()) {
    result.message =
        "C++ dual-arm IK did not reach the requested tolerances after " +
        std::to_string(result.iterations) + " iterations";
  }
  result.solve_time_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
  return result;
}

}  // namespace fairino_ik
