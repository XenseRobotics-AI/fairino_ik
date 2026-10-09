#include "fairino_ik/box_qp.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
  fairino_ik::BoxQP<2> qp;
  fairino_ik::BoxQP<2>::Matrix h;
  h << 2., 1., 1., 2.;
  fairino_ik::BoxQP<2>::Vector g, lower, upper, x;
  g << -2., -6.; lower << -10., -10.; upper << 10., 1.;
  int check = 0;
  auto require = [&](bool condition) { ++check; if (!condition) throw std::runtime_error("QP regression failed at check " + std::to_string(check)); };
  require(qp.solve(h, g, lower, upper, 100, x) == eiquadprog::solvers::RT_EIQUADPROG_OPTIMAL);
  // Unconstrained solution is (-2/3, 10/3). Simply clipping it is wrong:
  // the true constrained optimum reoptimizes the other coordinate to 0.5.
  require(std::abs(x[0] - .5) < 1e-10 && std::abs(x[1] - 1.) < 1e-10);
  require(qp.activeConstraints() == 1);
  require(qp.solve(h, g, lower, upper, 1, x) == eiquadprog::solvers::RT_EIQUADPROG_MAX_ITER_REACHED);
  lower[1] = 2.;
  require(qp.solve(h, g, lower, upper, 100, x) == eiquadprog::solvers::RT_EIQUADPROG_INFEASIBLE);

  fairino_ik::BoxQP<2, 1> barrier_qp;
  Eigen::Matrix<double, 1, 2> c; c << -1., 0.;
  Eigen::Matrix<double, 1, 1> offset; offset << 0.; // x[0] <= 0
  lower << -10., -10.;
  require(barrier_qp.solve(h, g, lower, upper, 100, x, c, offset) == eiquadprog::solvers::RT_EIQUADPROG_OPTIMAL);
  require(std::abs(x[0]) < 1e-10 && std::abs(x[1] - 1.) < 1e-10);
  std::cout << "C++ active-bound, barrier, infeasibility and iteration-limit QP checks passed\n";
}
