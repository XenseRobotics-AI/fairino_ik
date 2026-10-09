#pragma once

#include <array>
#include <limits>
#include <mutex>
#include <map>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include "fairino_ik/box_qp.hpp"

namespace fairino_ik {

// The URDF exports every joint as jN without a torso/arm prefix.  The groups
// below are the convention used by this package:
//   left arm   j1..j7   (left_arm_base_Link  -> left_waist2_Link)
//   right arm  j8..j14  (right_arm_base_Link -> right_waist2_Link)
//   torso      j20, j19, j18, j16, j15 (base_link -> body_Link, root first)
//   head       j17
//   wheels     j21..j28 (steering + drive, irrelevant to the arms)
constexpr int kArmDof = 14;
constexpr int kTorsoDof = 5;
constexpr int kBodyBoxCount = 2;  // body_Link and waist_Link
constexpr int kArmSamplesPerSide = 13;  // 3 segments * 4 samples + endpoint
constexpr int kBodySamples = kBodyBoxCount * 2 * kArmSamplesPerSide;

struct Options {
  int max_iterations = 500;
  int qp_max_iterations = 100;
  double dt = 0.02;
  double damping = 1e-5;
  double step_size = 1.0;
  double max_step_norm = 0.15;
  double tolerance = 1e-5;
  double position_tolerance = 0.0;
  double orientation_tolerance = 0.0;
  double position_cost = 1.0;
  double orientation_cost = 1.0;
  double frame_gain = 1.0;
  double posture_weight = 0.0;
  double posture_gain = 1.0;
  // The exported URDF has velocity="0" on every joint.  This speed (rad/s)
  // is used for any arm joint without a positive URDF velocity limit.
  double default_velocity_limit = 3.14159265358979323846;
  // Lightweight self-collision avoidance against the torso.  Boxes fitted
  // to the body_Link and waist_Link meshes are checked against sampled arm
  // center lines; each box is inflated by body_link_radius so the samples
  // approximate capsules.
  bool body_avoidance = true;
  double body_activation_distance = 0.04;
  double body_min_distance = 0.01;
  double body_link_radius = 0.06;
  double body_avoidance_weight = 4.0;
  double body_avoidance_gain = 1.0;
  // Optional public-order (left seven, right seven) posture target.  An
  // empty vector means that the seed is used when posture regularization is
  // enabled without an explicit target.
  std::vector<double> posture_target;
  // Locked during IK: j20, j19, j18, j16, j15 and the head yaw j17.
  std::array<double, kTorsoDof> torso_position{};
  double head_position = 0.0;
};

struct Result {
  std::vector<double> q;
  // Expanded full URDF configuration.  Keeping this in the native result
  // avoids a second Python -> C++ call after every IK solve.
  std::vector<double> full_q;
  bool success = false;
  int iterations = 0;
  int qp_iterations = 0;
  int qp_active_constraints = 0;
  double solve_time_ms = 0.0;
  std::string solver = "eiquadprog";
  double error = 0.0;
  double left_position_error = 0.0;
  double left_orientation_error = 0.0;
  double right_position_error = 0.0;
  double right_orientation_error = 0.0;
  bool body_collision_free = true;
  double body_min_distance = std::numeric_limits<double>::infinity();
  std::string body_nearest_segment;
  std::string status;
  std::string message;
};

class Model {
 public:
  explicit Model(
      const std::string& urdf_path,
      const std::string& left_frame = "left_waist2_Link",
      const std::string& right_frame = "right_waist2_Link");
  Model(const std::string& urdf_path, const std::string& left_frame,
        const std::string& right_frame, const std::array<double, 16>& base_pose);

  int nq() const noexcept { return kArmDof; }
  int nv() const noexcept { return kArmDof; }
  int fullNq() const noexcept { return full_model_.nq; }
  int fullNv() const noexcept { return full_model_.nv; }
  std::vector<std::array<double, 2>> jointLimits() const;
  std::vector<std::array<double, 2>> torsoLimits() const;
  std::array<double, 2> headLimits() const;
  std::vector<double> velocityLimits(const Options& options) const;
  std::vector<std::string> frameNames() const;
  std::map<std::string, int> jointCoordinates() const;
  std::vector<double> fullConfiguration(const std::vector<double>& q,
                                        const Options& options);
  std::map<std::string, std::array<double, 16>> framePlacements(
      const std::vector<double>& q, const Options& options);

  std::vector<double> forwardKinematics(
      const std::vector<double>& q_arm, const std::string& side,
      const Options& options = Options{});

  Result solveIK(
      const std::array<double, 16>& left_target,
      const std::array<double, 16>& right_target,
      const std::vector<double>& q_seed,
      const Options& options);

  std::vector<Result> solveIKBatch(
      const std::vector<std::array<double, 16>>& left_targets,
      const std::vector<std::array<double, 16>>& right_targets,
      const std::vector<std::vector<double>>& q_seeds,
      const Options& options);

 private:
  Result solveIKUnlocked(
      const std::array<double, 16>& left_target,
      const std::array<double, 16>& right_target,
      const std::vector<double>& q_seed,
      const Options& options);
  void addFlangeFrames();
  void configureReducedModel(const Options& options);
  Eigen::VectorXd fullConfigurationImpl(const std::vector<double>& q,
                                        const Options& options) const;
  struct Mimic { int child; int parent; double multiplier; double offset; };
  std::vector<Mimic> mimic_;  // Topologically ordered at construction.
  std::string left_frame_name_, right_frame_name_;
  std::array<double, kTorsoDof + 1> fixed_coordinates_{};
  bool reduced_initialized_ = false;
  Eigen::VectorXd armToReduced(const std::vector<double>& q_arm) const;
  std::vector<double> reducedToArm(const Eigen::VectorXd& q_reduced) const;
  Eigen::Isometry3d poseFromArray(const std::array<double, 16>& values) const;
  std::array<double, 16> poseToArray(const Eigen::Isometry3d& pose) const;
  Eigen::Isometry3d framePose(const Eigen::VectorXd& q, pinocchio::FrameIndex frame);
  void updateKinematics(const Eigen::VectorXd& q);
  double velocityLimit(int index, const Options& options) const;
  void validateOptions(const Options& options) const;
  void validateQ(const std::vector<double>& q, const char* name) const;

  struct PoseErrors {
    Eigen::Matrix<double, 12, 1> task_error;
    double left_position = 0.0;
    double left_orientation = 0.0;
    double right_position = 0.0;
    double right_orientation = 0.0;
    double max_error = 0.0;
  };

  struct BodyMetrics {
    double min_distance = std::numeric_limits<double>::infinity();
    std::string nearest_segment;
  };

  PoseErrors evaluate(
      const Eigen::VectorXd& q,
      const Eigen::Isometry3d& left_target,
      const Eigen::Isometry3d& right_target);
  BodyMetrics evaluateBodyAvoidance(
      const Options& options,
      Eigen::Matrix<double, kArmDof, kArmDof>* hessian,
      Eigen::Matrix<double, kArmDof, 1>* rhs,
      Eigen::Matrix<double, kBodySamples, kArmDof>* constraints = nullptr,
      Eigen::Matrix<double, kBodySamples, 1>* offsets = nullptr);

  pinocchio::Model full_model_;
  pinocchio::Data full_data_;
  pinocchio::Model reduced_model_;
  pinocchio::Data reduced_data_;
  pinocchio::FrameIndex left_frame_id_{0};
  pinocchio::FrameIndex right_frame_id_{0};
  std::array<std::array<pinocchio::FrameIndex, 4>, 2> body_arm_frame_ids_{};
  std::array<pinocchio::FrameIndex, kBodyBoxCount> body_box_frame_ids_{};
  Eigen::Isometry3d base_pose_{Eigen::Isometry3d::Identity()};
  std::array<std::string, kArmDof> arm_joint_names_{{
      "j1", "j2", "j3", "j4", "j5", "j6", "j7",
      "j8", "j9", "j10", "j11", "j12", "j13", "j14",
  }};
  std::array<std::string, kTorsoDof> torso_joint_names_{{
      "j20", "j19", "j18", "j16", "j15",
  }};
  std::string head_joint_name_ = "j17";
  std::array<int, kArmDof> arm_q_indices_{};
  std::array<double, kArmDof> lower_limits_{};
  std::array<double, kArmDof> upper_limits_{};
  // Raw URDF values; zero means "use Options::default_velocity_limit".
  std::array<double, kArmDof> urdf_velocity_limits_{};
  std::array<int, kArmDof> full_q_indices_{};
  BoxQP<kArmDof, kBodySamples> qp_;
  mutable std::mutex mutex_;
};

}  // namespace fairino_ik
