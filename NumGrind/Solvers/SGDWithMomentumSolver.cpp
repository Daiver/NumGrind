#include <iostream>
#include "SGDWithMomentumSolver.h"

NumGrind::Solvers::SGDWithMomentumSolver::SGDWithMomentumSolver(const NumGrind::Solvers::SolverSettings &settings,
                                                                const float stepSize, const float momentumCoeff,
                                                                const Eigen::VectorXf &vars)
        : NumGrind::Solvers::SGDSolver(settings, stepSize, vars), momentumCoeff(momentumCoeff) {
    this->oldGrad.resize(vars.size());
    this->oldGrad.fill(0);
    assert(false);//Not implemented
}

void NumGrind::Solvers::SGDWithMomentumSolver::makeStep(std::function<float(const Eigen::VectorXf &)> func,
                                                        std::function<void(const Eigen::VectorXf &,
                                                                           Eigen::VectorXf &)> gradFunc) {
    this->grad.resize(mVars.size());
    grad.fill(0);
    gradFunc(mVars, grad);
    grad *= stepSize;
    grad += this->momentumCoeff * this->oldGrad;
    oldGrad = grad;
    mVars -= grad;
    const float err = func(mVars);
    this->updateBestVars(err);
    if (this->settings.verbose)
        std::cout << "err " << err << std::endl;
}
