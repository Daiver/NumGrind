#include "StochasticGradientDescentSolver.h"

#include <iostream>

NumGrind::Solvers::StochasticGradientDescentSolver::StochasticGradientDescentSolver(
            const NumGrind::Solvers::SolverSettings &settings, const float stepSize, const Eigen::VectorXf &vars)
        : settings(settings), stepSize(stepSize), mVars(vars)
{

}

void NumGrind::Solvers::StochasticGradientDescentSolver::makeStep(std::function<float(const Eigen::VectorXf &)> func,
                                                                  std::function<void(const Eigen::VectorXf &,
                                                                                     Eigen::VectorXf &)> gradFunc)
{
    this->grad.resize(mVars.size());
    grad.fill(0);
    gradFunc(mVars, grad);
    mVars += -stepSize * grad;
    if(this->settings.verbose)
        std::cout << "err " << func(mVars) << std::endl;
}
