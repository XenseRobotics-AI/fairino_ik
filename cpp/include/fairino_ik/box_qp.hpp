#pragma once

#include <Eigen/Core>
#include <eiquadprog/eiquadprog-rt.hpp>

namespace fairino_ik {

// Fixed-size, allocation-free active-set QP workspace. Unlike a clipped
// least-squares solution, active constraints participate in the optimization.
// min 0.5 x' H x + g' x, subject to lower <= x <= upper.
template <int N, int M = 0> class BoxQP {
 public:
  using Vector = Eigen::Matrix<double, N, 1>;
  using Matrix = Eigen::Matrix<double, N, N>;
  using Status = eiquadprog::solvers::RtEiquadprog_status;

  Status solve(const Matrix& hessian, const Vector& gradient,
               const Vector& lower, const Vector& upper, int max_iterations,
               Vector& solution,
               const Eigen::Matrix<double, M, N>& extra = Eigen::Matrix<double, M, N>::Zero(),
               const Eigen::Matrix<double, M, 1>& extra_offsets = Eigen::Matrix<double, M, 1>::Ones()) {
    // The RT backend labels some infeasible systems "unbounded". A crossed
    // box can be detected unambiguously before invoking it.
    if ((lower.array() > upper.array()).any())
      return eiquadprog::solvers::RT_EIQUADPROG_INFEASIBLE;
    solver_.setMaxIter(max_iterations);
    Eigen::Matrix<double, 2 * N + M, N> constraints;
    constraints.template topRows<N>().setIdentity();
    constraints.template middleRows<N>(N) = -Matrix::Identity();
    constraints.template bottomRows<M>() = extra;
    Eigen::Matrix<double, 2 * N + M, 1> offsets;
    offsets.template head<N>() = -lower;
    offsets.template segment<N>(N) = upper;
    offsets.template tail<M>() = extra_offsets;
    // eiquadprog uses CI*x+ci0 >= 0 (and zero equality rows here).
    return solver_.solve_quadprog(hessian, gradient,
        Eigen::Matrix<double, 0, N>(), Eigen::Matrix<double, 0, 1>(),
        constraints, offsets, solution);
  }

  int iterations() const { return solver_.getIteratios(); }
  int activeConstraints() const { return solver_.getActiveSetSize(); }

 private:
  eiquadprog::solvers::RtEiquadprog<N, 0, 2 * N + M> solver_;
};

}  // namespace fairino_ik
