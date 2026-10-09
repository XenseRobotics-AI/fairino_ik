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
#include <Eigen/SVD>

namespace fairino_ik {
namespace {

constexpr double kFiniteLimit = 1e6;
// The linearized barrier drives a blocked arm onto the clearance margin;
// its curvature error (sub-micron, larger when the torso moves) is accepted.
constexpr double kClearanceTolerance = 1e-5;
// eiquadprog can overshoot an active bound by ~1e-5 rad when a joint sits on
// its limit in the redundant 19-DoF problem.  Larger violations still reject
// the step; smaller ones are projected back into the box.
constexpr double kStepBoundTolerance = 1e-4;

// Operational flange frames added below each wrist link.  The wrist mesh
// (left/right_waist2_Link.STL) ends 0.114 m along the link's -x axis.
constexpr std::array<const char*, 2> kWristFrameNames{{
    "left_waist2_Link", "right_waist2_Link"}};
constexpr std::array<const char*, 2> kFlangeFrameNames{{
    "left_flange", "right_flange"}};
const Eigen::Vector3d kFlangeOffset(-0.114, 0.0, 0.0);

// Collision boxes, expressed in their own link frames.  They are the
// bounding boxes of the URDF meshes; body_Link is cut at z = 0.45 m so the
// neck and head mount do not block the space above the shoulders.  The
// base_link box is the chassis, which the hands reach when the torso squats.
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
    {"base_link", Eigen::Vector3d(0.0, -0.14933, 0.02025),
     Eigen::Vector3d(0.30165, 0.38233, 0.19125)},
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

Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d result;
  result << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return result;
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
    const std::string& right_frame,
    bool active_torso)
    : active_torso_(active_torso), nq_(active_torso ? kMaxDof : kArmDof) {
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
  for (int i = 0; i < kMaxDof; ++i) {
    const auto id = full_model_.getJointId(joint_names_[i]);
    if (id == 0 || id >= static_cast<pinocchio::JointIndex>(full_model_.njoints) ||
        full_model_.joints[id].nq() != 1 || full_model_.joints[id].nv() != 1) {
      throw std::invalid_argument("Missing 1-DoF joint " + joint_names_[i]);
    }
    full_q_indices_[i] = full_model_.joints[id].idx_q();
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
             const std::string& right_frame, const std::array<double, 16>& base_pose,
             bool active_torso)
    : Model(urdf_path, left_frame, right_frame, active_torso) {
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
  // Arms and torso always stay in the reduced model, so only the locked
  // head yaw (and the irrelevant wheels) require a rebuild.
  if (reduced_initialized_ && options.head_position == fixed_head_position_) return;

  std::vector<pinocchio::JointIndex> locked;
  locked.reserve(full_model_.njoints);
  for (pinocchio::JointIndex joint_id = 1;
       joint_id < static_cast<pinocchio::JointIndex>(full_model_.njoints);
       ++joint_id) {
    const std::string& name = full_model_.names[joint_id];
    const bool active = std::find(
                            joint_names_.begin(), joint_names_.end(),
                            name) != joint_names_.end();
    if (!active && full_model_.joints[joint_id].nv() > 0) {
      locked.push_back(joint_id);
    }
  }

  const Eigen::VectorXd reference = fullConfigurationImpl(std::vector<double>(nq_, 0.0), options);
  reduced_model_ =
      pinocchio::buildReducedModel(full_model_, locked, reference);
  reduced_data_ = pinocchio::Data(reduced_model_);
  if (reduced_model_.nq != kMaxDof || reduced_model_.nv != kMaxDof) {
    throw std::runtime_error(
        "Fairino C++ IK reduced model must have 19 DoF, got nq=" +
        std::to_string(reduced_model_.nq) + ", nv=" +
        std::to_string(reduced_model_.nv));
  }

  for (int i = 0; i < kMaxDof; ++i) {
    const pinocchio::JointIndex joint_id =
        reduced_model_.getJointId(joint_names_[i]);
    if (joint_id == 0 ||
        reduced_model_.names[joint_id] != joint_names_[i]) {
      throw std::runtime_error(
          "Reduced Fairino IK model is missing joint " + joint_names_[i]);
    }
    const int index = reduced_model_.joints[joint_id].idx_q();
    reduced_q_indices_[i] = index;
    double lower = reduced_model_.lowerPositionLimit[index];
    double upper = reduced_model_.upperPositionLimit[index];
    if (!finite(lower)) lower = -kFiniteLimit;
    if (!finite(upper)) upper = kFiniteLimit;
    if (lower > upper) {
      throw std::runtime_error(
          "Invalid limits for Fairino IK joint " + joint_names_[i]);
    }
    lower_limits_[i] = lower;
    upper_limits_[i] = upper;
    const int velocity_index = reduced_model_.joints[joint_id].idx_v();
    if (velocity_index != index) throw std::runtime_error("Reduced q/v ordering mismatch");
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
  fixed_head_position_ = options.head_position;
  reduced_initialized_ = true;
}

double Model::torsoValue(const std::vector<double>& q, const Options& options,
                         int index) const {
  return active_torso_ ? q[kArmDof + index] : options.torso_position[index];
}

Eigen::VectorXd Model::fullConfigurationImpl(const std::vector<double>& q,
                                             const Options& options) const {
  validateQ(q, "q");
  Eigen::VectorXd result = pinocchio::neutral(full_model_);
  const auto set_clamped = [&](int index, double value) {
    if (!finite(value)) throw std::invalid_argument("Joint values must be finite");
    result[index] = std::clamp(value, full_model_.lowerPositionLimit[index],
                               full_model_.upperPositionLimit[index]);
  };
  for (int i = 0; i < kArmDof; ++i) result[full_q_indices_[i]] = q[i];
  for (int i = 0; i < kTorsoDof; ++i)
    set_clamped(full_q_indices_[kArmDof + i], torsoValue(q, options, i));
  set_clamped(full_model_.joints[full_model_.getJointId(head_joint_name_)].idx_q(),
              options.head_position);
  for (const auto& relation : mimic_)
    result[relation.child] = relation.multiplier * result[relation.parent] + relation.offset;
  return result;
}

std::vector<double> Model::fullConfiguration(const std::vector<double>& q, const Options& options) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto full = fullConfigurationImpl(q, options);
  return {full.data(), full.data() + full.size()};
}

std::vector<std::string> Model::jointNames() const {
  return {joint_names_.begin(), joint_names_.begin() + nq_};
}

std::vector<std::array<double, 2>> Model::jointLimits() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::array<double, 2>> result;
  for (int i = 0; i < nq_; ++i) result.push_back({lower_limits_[i], upper_limits_[i]});
  return result;
}

std::vector<std::array<double, 2>> Model::torsoLimits() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::array<double, 2>> result;
  for (int i = kArmDof; i < kMaxDof; ++i) result.push_back({lower_limits_[i], upper_limits_[i]});
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
  if (urdf > 0.0) return urdf;
  return index < kArmDof ? options.default_velocity_limit
                         : options.torso_velocity_limit;
}

std::vector<double> Model::velocityLimits(const Options& options) const {
  std::lock_guard<std::mutex> lock(mutex_);
  validateOptions(options);
  std::vector<double> result(nq_);
  for (int i = 0; i < nq_; ++i) result[i] = velocityLimit(i, options);
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
  if (static_cast<int>(q.size()) != nq_) {
    throw std::invalid_argument(
        std::string(name) + " must contain " + std::to_string(nq_) +
        " values, got " + std::to_string(q.size()));
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
  if (options.arms_first_iterations < 0) throw std::invalid_argument("arms_first_iterations must be >= 0");
  const std::array<double, 20> values = {
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
      options.posture_tolerance,
      options.default_velocity_limit,
      options.torso_velocity_limit,
      options.torso_step_weight,
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
      options.torso_velocity_limit <= 0.0 || options.posture_tolerance <= 0.0 ||
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

Eigen::VectorXd Model::toReduced(
    const std::vector<double>& q, const Options& options) const {
  validateQ(q, "q");
  Eigen::VectorXd result = pinocchio::neutral(reduced_model_);
  for (int i = 0; i < kMaxDof; ++i) {
    const double value = i < kArmDof ? q[i] : torsoValue(q, options, i - kArmDof);
    result[reduced_q_indices_[i]] =
        std::clamp(value, lower_limits_[i], upper_limits_[i]);
  }
  return result;
}

std::vector<double> Model::reducedToPublic(
    const Eigen::VectorXd& q_reduced) const {
  if (q_reduced.size() != kMaxDof) {
    throw std::invalid_argument("reduced q must contain 19 values");
  }
  std::vector<double> result(nq_);
  for (int i = 0; i < nq_; ++i) {
    result[i] = std::clamp(
        q_reduced[reduced_q_indices_[i]], lower_limits_[i], upper_limits_[i]);
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
    const std::vector<double>& q_public, const std::string& side, const Options& options) {
  std::lock_guard<std::mutex> lock(mutex_);
  validateOptions(options);
  configureReducedModel(options);
  const Eigen::VectorXd q = toReduced(q_public, options);
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
    bool torso_free,
    FullMatrix* hessian,
    FullVector* rhs,
    ConstraintMatrix* constraints,
    ConstraintVector* offsets) {
  BodyMetrics metrics;
  if (constraints) constraints->setZero();
  if (offsets) offsets->setOnes();
  if (!options.body_avoidance) return metrics;

  const bool add_task = hessian != nullptr && rhs != nullptr;
  // Everything here is in the model root frame, as are the Jacobians.  The
  // body and waist boxes ride on the torso joints, so a clearance changes
  // with the relative velocity between an arm point and the box.
  std::array<Eigen::Isometry3d, kBodyBoxCount> box_poses;
  std::array<Eigen::Vector3d, kBodyBoxCount> box_origins;
  std::array<Eigen::Matrix<double, 6, kMaxDof>, kBodyBoxCount> box_jacobians;
  std::array<Eigen::Vector3d, kBodyBoxCount> inflated_half_extents;
  for (std::size_t box = 0; box < kBodyBoxes.size(); ++box) {
    const auto frame_id = body_box_frame_ids_[box];
    const Eigen::Isometry3d frame(
        reduced_data_.oMf[frame_id].toHomogeneousMatrix());
    box_poses[box] = frame * Eigen::Translation3d(kBodyBoxes[box].center);
    box_origins[box] = frame.translation();
    inflated_half_extents[box] = kBodyBoxes[box].half_extents +
        Eigen::Vector3d::Constant(options.body_link_radius);
    if (add_task) {
      box_jacobians[box].setZero();
      pinocchio::getFrameJacobian(
          reduced_model_, reduced_data_, frame_id,
          pinocchio::LOCAL_WORLD_ALIGNED, box_jacobians[box]);
    }
  }
  int sample = 0;

  for (std::size_t side = 0; side < body_arm_frame_ids_.size(); ++side) {
    std::array<Eigen::Vector3d, 4> positions;
    std::array<Eigen::Matrix<double, 3, kMaxDof>, 4> linear_jacobians;
    for (std::size_t index = 0;
         index < body_arm_frame_ids_[side].size(); ++index) {
      const auto frame_id = body_arm_frame_ids_[side][index];
      positions[index] = reduced_data_.oMf[frame_id].translation();
      if (add_task) {
        Eigen::Matrix<double, 6, kMaxDof> frame_jacobian;
        frame_jacobian.setZero();
        pinocchio::getFrameJacobian(
            reduced_model_, reduced_data_, frame_id,
            pinocchio::LOCAL_WORLD_ALIGNED, frame_jacobian);
        linear_jacobians[index] = frame_jacobian.topRows<3>();
      }
    }

    const auto process_sample = [&](
        const Eigen::Vector3d& point,
        const Eigen::Matrix<double, 3, kMaxDof>* point_jacobian,
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

        // Velocity of the box-fixed point that coincides with the sample:
        // v_o + w x (p - o).
        const auto& box_jacobian = box_jacobians[box_index];
        const Eigen::Matrix<double, 3, kMaxDof> box_point_jacobian =
            box_jacobian.topRows<3>() -
            skew(point - box_origins[box_index]) * box_jacobian.bottomRows<3>();
        Eigen::Matrix<double, 1, kMaxDof> distance_jacobian =
            box.outward_normal.transpose() *
            (*point_jacobian - box_point_jacobian);
        // A fixed torso cannot help to keep the clearance.
        if (!torso_free) {
          for (int i = kArmDof; i < kMaxDof; ++i)
            distance_jacobian[reduced_q_indices_[i]] = 0.0;
        }
        // A sample rigidly attached to the box (the shoulder vs body_Link,
        // or anything vs base_link with a locked torso) cannot change its
        // clearance; a zero row would only make the QP infeasible.
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
          const Eigen::Matrix<double, 3, kMaxDof> point_jacobian =
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

template <int N>
eiquadprog::solvers::RtEiquadprog_status Model::solveActiveQP(
    BoxQP<N, kBodySamples>& qp, const FullMatrix& hessian,
    const FullVector& gradient, const FullVector& lower,
    const FullVector& upper, const ConstraintMatrix& constraints,
    const ConstraintVector& offsets, int max_iterations, FullVector& step,
    int* iterations, int* active_constraints) {
  typename BoxQP<N, kBodySamples>::Matrix h;
  typename BoxQP<N, kBodySamples>::Vector g, lo, hi, x;
  Eigen::Matrix<double, kBodySamples, N> c;
  for (int i = 0; i < N; ++i) {
    const int ri = reduced_q_indices_[i];
    g[i] = gradient[ri];
    lo[i] = lower[ri];
    hi[i] = upper[ri];
    c.col(i) = constraints.col(ri);
    for (int j = 0; j < N; ++j) h(i, j) = hessian(ri, reduced_q_indices_[j]);
  }
  const auto status = qp.solve(h, g, lo, hi, max_iterations, x, c, offsets);
  *iterations = qp.iterations();
  *active_constraints = qp.activeConstraints();
  step.setZero();
  for (int i = 0; i < N; ++i) step[reduced_q_indices_[i]] = x[i];
  return status;
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
  Eigen::VectorXd q = toReduced(q_seed, options);
  Eigen::VectorXd posture = q;
  if (!options.posture_target.empty()) {
    posture = toReduced(options.posture_target, options);
  }
  const double position_tolerance =
      options.position_tolerance > 0.0 ? options.position_tolerance
                                       : options.tolerance;
  const double orientation_tolerance =
      options.orientation_tolerance > 0.0 ? options.orientation_tolerance
                                          : options.tolerance;

  Result result;
  PoseErrors errors = evaluate(q, left_target, right_target);
  BodyMetrics body = evaluateBodyAvoidance(options, false, nullptr, nullptr);
  const auto targets_reached = [&]() {
    return errors.left_position <= position_tolerance &&
           errors.right_position <= position_tolerance &&
           errors.left_orientation <= orientation_tolerance &&
           errors.right_orientation <= orientation_tolerance &&
           (!options.body_avoidance ||
            body.min_distance >= options.body_min_distance - kClearanceTolerance);
  };
  // The posture task lives in the null space of the hand tasks, so it keeps
  // moving after the targets are reached; it has settled once the steps it
  // produces are negligible (or it is blocked by limits/collisions).
  const bool posture_enabled = options.posture_weight > 0.0;
  double last_step = std::numeric_limits<double>::infinity();
  const auto converged = [&]() {
    return targets_reached() &&
           (!posture_enabled || last_step <= options.posture_tolerance);
  };
  Eigen::Matrix<double, 12, 12> weight =
      Eigen::Matrix<double, 12, 12>::Zero();
  for (int block = 0; block < 2; ++block) {
    weight.block<3, 3>(block * 6, block * 6).diagonal().setConstant(
        options.position_cost);
    weight.block<3, 3>(block * 6 + 3, block * 6 + 3).diagonal().setConstant(
        options.orientation_cost);
  }

  // Runs up to `budget` QP iterations over the arms, or over arms and torso
  // when `torso_free`.  Returns false after a rejected QP step.
  const auto run_phase = [&](bool torso_free, int budget) {
    const int active = torso_free ? kMaxDof : kArmDof;
    // The trust box is per active coordinate, so ||step|| <= max_step_norm.
    const double trust = options.max_step_norm / std::sqrt(double(active));
    Eigen::Matrix<double, 12, kMaxDof> jacobian;
    for (int count = 0; count < budget; ++count) {
      const int iteration = ++result.iterations;
      // evaluate() has already updated placements and Jacobians at this q,
      // initially and after every applied QP step. Reuse that state here.
      Eigen::Matrix<double, 6, kMaxDof> left_jacobian;
      Eigen::Matrix<double, 6, kMaxDof> right_jacobian;
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
      jacobian.block<6, kMaxDof>(0, 0) = left_jacobian;
      jacobian.block<6, kMaxDof>(6, 0) = right_jacobian;

      // Everything below is assembled over all 19 reduced coordinates;
      // solveActiveQP drops the torso columns when the torso is fixed.
      FullMatrix hessian = jacobian.transpose() * weight * jacobian;
      // Strict convexity is required by the active-set solver, even when the
      // caller requests zero damping or the arm is stretched (singular).
      hessian.diagonal().array() += std::max(1e-10, options.damping * options.damping);
      FullVector rhs =
          jacobian.transpose() * weight *
          (options.frame_gain * errors.task_error);
      if (posture_enabled) {
        // Null-space projector of the weighted hand tasks over the active
        // joints.  Pulling towards the posture only along it leaves the
        // converged hand poses unbiased.
        Eigen::Matrix<double, 12, kMaxDof> task = weight.diagonal().cwiseSqrt().asDiagonal() * jacobian;
        if (!torso_free) {
          for (int i = kArmDof; i < kMaxDof; ++i) task.col(reduced_q_indices_[i]).setZero();
        }
        const Eigen::JacobiSVD<Eigen::Matrix<double, 12, kMaxDof>> svd(task, Eigen::ComputeFullV);
        const auto& singular = svd.singularValues();
        const double threshold = 1e-6 * std::max(1.0, singular[0]);
        FullMatrix null_space = FullMatrix::Identity();
        for (int k = 0; k < singular.size() && singular[k] > threshold; ++k)
          null_space -= svd.matrixV().col(k) * svd.matrixV().col(k).transpose();
        const FullVector pull = null_space * (posture - q);
        double residual = 0.0;
        for (int i = 0; i < active; ++i)
          residual = std::max(residual, std::abs(pull[reduced_q_indices_[i]]));
        result.posture_residual = residual;
        hessian.noalias() += options.posture_weight * null_space;
        rhs.noalias() += options.posture_weight * options.posture_gain * pull;
      }
      for (int i = kArmDof; i < kMaxDof; ++i)
        hessian(reduced_q_indices_[i], reduced_q_indices_[i]) += options.torso_step_weight;
      ConstraintMatrix body_constraints;
      ConstraintVector body_offsets;
      body = evaluateBodyAvoidance(options, torso_free, &hessian, &rhs,
                                   &body_constraints, &body_offsets);
      FullVector lower = FullVector::Zero(), upper = FullVector::Zero(), step;
      // Optimize configuration increments, not a post-clipped velocity.
      for (int i = 0; i < active; ++i) {
        const int index = reduced_q_indices_[i];
        const double limit = std::min(
            trust, velocityLimit(i, options) * options.dt * options.step_size);
        lower[index] = std::max(lower_limits_[i] - q[index], -limit);
        upper[index] = std::min(upper_limits_[i] - q[index], limit);
      }
      int qp_iterations = 0;
      int qp_active = 0;
      const FullVector gradient = -options.step_size * rhs;
      const auto qp_status = torso_free
          ? solveActiveQP<kMaxDof>(torso_qp_, hessian, gradient, lower, upper,
                                   body_constraints, body_offsets,
                                   options.qp_max_iterations, step,
                                   &qp_iterations, &qp_active)
          : solveActiveQP<kArmDof>(arm_qp_, hessian, gradient, lower, upper,
                                   body_constraints, body_offsets,
                                   options.qp_max_iterations, step,
                                   &qp_iterations, &qp_active);
      result.qp_iterations += qp_iterations;
      result.qp_active_constraints = std::max(result.qp_active_constraints, qp_active);
      if (qp_status != eiquadprog::solvers::RT_EIQUADPROG_OPTIMAL || !step.allFinite() ||
          (step.array() < lower.array() - kStepBoundTolerance).any() ||
          (step.array() > upper.array() + kStepBoundTolerance).any()) {
        result.status = qp_status == eiquadprog::solvers::RT_EIQUADPROG_MAX_ITER_REACHED
            ? "qp_max_iterations" : "qp_solver_error";
        const double violation = std::max(
            (lower - step).maxCoeff(), (step - upper).maxCoeff());
        result.message = "C++ eiquadprog failed; no unverified QP step was applied (status=" +
                         std::to_string(int(qp_status)) + ", bound violation=" +
                         std::to_string(violation) + ", iteration=" +
                         std::to_string(iteration) + ")";
        return false;
      }
      // Remove the solver's rounding at an active joint limit.
      step = step.cwiseMax(lower).cwiseMin(upper);
      last_step = step.cwiseAbs().maxCoeff();
      q += step;
      for (int i = 0; i < active; ++i) {
        q[reduced_q_indices_[i]] =
            std::clamp(q[reduced_q_indices_[i]], lower_limits_[i], upper_limits_[i]);
      }
      errors = evaluate(q, left_target, right_target);
      body = evaluateBodyAvoidance(options, torso_free, nullptr, nullptr);
      if (converged()) {
        result.success = true;
        result.status = "converged";
        return true;
      }
    }
    return true;
  };

  if (converged()) {
    result.success = true;
    result.status = "converged";
  } else {
    result.status = "max_iterations";
    if (!active_torso_) {
      run_phase(false, options.max_iterations);
    } else {
      // Arms first: the heavy torso only moves when the arms cannot reach
      // the targets from the current torso pose within a short budget.
      // A posture target also covers the torso, so its null-space pull
      // continues with the torso free once the arms have reached.
      const int arm_budget = std::min(options.arms_first_iterations, options.max_iterations);
      if (arm_budget > 0) run_phase(false, arm_budget);
      if ((!result.success || posture_enabled) &&
          result.iterations < options.max_iterations) {
        result.success = false;
        result.status = "max_iterations";
        result.message.clear();
        last_step = std::numeric_limits<double>::infinity();
        run_phase(true, options.max_iterations - result.iterations);
      }
    }
  }
  if (!result.success && result.status == "max_iterations" && targets_reached()) {
    // The hands are on target; only the null-space posture ran out of budget.
    result.success = true;
    result.status = "converged";
    result.message = "targets reached; posture still settling after " +
                     std::to_string(result.iterations) + " iterations";
  }

  result.q = reducedToPublic(q);
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
      body.min_distance >= options.body_min_distance - kClearanceTolerance;
  if (!result.body_collision_free) {
    result.success = false;
    result.status = "body_collision";
    std::ostringstream message;
    message << "C++ dual-arm IK rejected by body boxes: clearance="
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
