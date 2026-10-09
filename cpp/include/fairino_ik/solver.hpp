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
//
// The public configuration is [left 7 | right 7] in arm-only mode and
// [left 7 | right 7 | torso 5] when the torso is active, so the arm slices
// mean the same thing in both modes.
constexpr int kArmDof = 14;
constexpr int kTorsoDof = 5;
constexpr int kMaxDof = kArmDof + kTorsoDof;
constexpr int kBodyBoxCount = 3;  // body_Link, waist_Link and base_link
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
  // Posture task in the null space of the hand tasks: with posture_weight > 0
  // the redundant joints move towards posture_target without biasing the
  // hand poses.  posture_gain is the fraction of the remaining null-space
  // offset removed per iteration; the solve keeps iterating until the
  // posture steps fall below posture_tolerance (rad).
  double posture_weight = 0.0;
  double posture_gain = 1.0;
  double posture_tolerance = 1e-4;
  // The exported URDF has velocity="0" on every joint.  These speeds (rad/s)
  // are used for joints without a positive URDF velocity limit.
  double default_velocity_limit = 3.14159265358979323846;
  double torso_velocity_limit = 0.5;
  // Active-torso mode only.  The solve first tries the arms alone for up to
  // arms_first_iterations, so the torso stays put whenever the arms can
  // reach; then it continues with the torso free.  0 frees the torso from
  // the start.  torso_step_weight is an extra quadratic cost on each torso
  // step in that phase; it slows convergence near arm singularities.
  int arms_first_iterations = 50;
  double torso_step_weight = 0.0;
  // Lightweight self-collision avoidance.  Boxes fitted to the body_Link,
  // waist_Link and base_link meshes are checked against sampled arm center
  // lines; each box is inflated by body_link_radius so the samples
  // approximate capsules.
  bool body_avoidance = true;
  double body_activation_distance = 0.04;
  double body_min_distance = 0.01;
  double body_link_radius = 0.06;
  double body_avoidance_weight = 4.0;
  double body_avoidance_gain = 1.0;
  // Optional public-order posture target with nq values.  An empty vector
  // means that the seed is used when posture regularization is enabled
  // without an explicit target.
  std::vector<double> posture_target;
  // Arm-only mode locks j20, j19, j18, j16, j15 here; active-torso mode
  // takes them from q instead.  The head yaw j17 is always locked.
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
  // Largest remaining null-space posture offset (rad); 0 without a posture task.
  double posture_residual = 0.0;
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
      const std::string& right_frame = "right_waist2_Link",
      bool active_torso = false);
  Model(const std::string& urdf_path, const std::string& left_frame,
        const std::string& right_frame, const std::array<double, 16>& base_pose,
        bool active_torso = false);

  int nq() const noexcept { return nq_; }
  int nv() const noexcept { return nq_; }
  bool activeTorso() const noexcept { return active_torso_; }
  int fullNq() const noexcept { return full_model_.nq; }
  int fullNv() const noexcept { return full_model_.nv; }
  std::vector<std::string> jointNames() const;
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
      const std::vector<double>& q, const std::string& side,
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
  using FullVector = Eigen::Matrix<double, kMaxDof, 1>;
  using FullMatrix = Eigen::Matrix<double, kMaxDof, kMaxDof>;
  using ConstraintMatrix = Eigen::Matrix<double, kBodySamples, kMaxDof>;
  using ConstraintVector = Eigen::Matrix<double, kBodySamples, 1>;

  Result solveIKUnlocked(
      const std::array<double, 16>& left_target,
      const std::array<double, 16>& right_target,
      const std::vector<double>& q_seed,
      const Options& options);
  // Solves the step QP over the first N public joints.  All inputs use the
  // reduced-model coordinate order; inactive columns are dropped.
  template <int N>
  eiquadprog::solvers::RtEiquadprog_status solveActiveQP(
      BoxQP<N, kBodySamples>& qp, const FullMatrix& hessian,
      const FullVector& gradient, const FullVector& lower,
      const FullVector& upper, const ConstraintMatrix& constraints,
      const ConstraintVector& offsets, int max_iterations, FullVector& step,
      int* iterations, int* active_constraints);
  void addFlangeFrames();
  void configureReducedModel(const Options& options);
  Eigen::VectorXd fullConfigurationImpl(const std::vector<double>& q,
                                        const Options& options) const;
  struct Mimic { int child; int parent; double multiplier; double offset; };
  std::vector<Mimic> mimic_;  // Topologically ordered at construction.
  std::string left_frame_name_, right_frame_name_;
  double fixed_head_position_ = 0.0;
  bool reduced_initialized_ = false;
  // Public torso values: from q when active, else from the options.
  double torsoValue(const std::vector<double>& q, const Options& options,
                    int index) const;
  Eigen::VectorXd toReduced(const std::vector<double>& q,
                            const Options& options) const;
  std::vector<double> reducedToPublic(const Eigen::VectorXd& q_reduced) const;
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
      bool torso_free,
      FullMatrix* hessian,
      FullVector* rhs,
      ConstraintMatrix* constraints = nullptr,
      ConstraintVector* offsets = nullptr);

  bool active_torso_ = false;
  int nq_ = kArmDof;
  pinocchio::Model full_model_;
  pinocchio::Data full_data_;
  pinocchio::Model reduced_model_;
  pinocchio::Data reduced_data_;
  pinocchio::FrameIndex left_frame_id_{0};
  pinocchio::FrameIndex right_frame_id_{0};
  std::array<std::array<pinocchio::FrameIndex, 4>, 2> body_arm_frame_ids_{};
  std::array<pinocchio::FrameIndex, kBodyBoxCount> body_box_frame_ids_{};
  Eigen::Isometry3d base_pose_{Eigen::Isometry3d::Identity()};
  // Public order: left arm, right arm, torso (root first).
  std::array<std::string, kMaxDof> joint_names_{{
      "j1", "j2", "j3", "j4", "j5", "j6", "j7",
      "j8", "j9", "j10", "j11", "j12", "j13", "j14",
      "j20", "j19", "j18", "j16", "j15",
  }};
  std::string head_joint_name_ = "j17";
  // Public index -> reduced-model coordinate.  The reduced model always
  // contains all 19 joints; arm-only mode fixes the torso coordinates.
  std::array<int, kMaxDof> reduced_q_indices_{};
  std::array<double, kMaxDof> lower_limits_{};
  std::array<double, kMaxDof> upper_limits_{};
  // Raw URDF values; zero means "use the Options fallback speed".
  std::array<double, kMaxDof> urdf_velocity_limits_{};
  std::array<int, kMaxDof> full_q_indices_{};
  BoxQP<kArmDof, kBodySamples> arm_qp_;
  BoxQP<kMaxDof, kBodySamples> torso_qp_;
  mutable std::mutex mutex_;
};

}  // namespace fairino_ik
