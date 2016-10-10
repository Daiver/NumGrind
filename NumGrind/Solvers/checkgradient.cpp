#include "checkgradient.h"

#include "numericalgradient.h"

bool ::NumGrind::Solvers::isGradientOk(std::function<float(const Eigen::VectorXf &)> func,
                                       std::function<void(const Eigen::VectorXf &, Eigen::VectorXf &)> grad,
                                       const Eigen::VectorXf &vars, const float tolerance, const float dx) {
    Eigen::VectorXf grad1(vars.size());
    grad1.fill(0);
    grad(vars, grad1);
    Eigen::VectorXf grad2(vars.size());
    numericalGradient(func, vars, dx, grad2);
    const float diff = (grad1 - grad2).norm() / vars.size();
    return diff < tolerance;
}
