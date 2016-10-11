#include "SGDSolver.h"

#include <iostream>

NumGrind::Solvers::SGDSolver::SGDSolver(
        const NumGrind::Solvers::SolverSettings &settings, const float stepSize, const Eigen::VectorXf &vars)
        : settings(settings), stepSize(stepSize), mVars(vars), mBestVars(vars) {

}

void NumGrind::Solvers::SGDSolver::makeStep(std::function<float(const Eigen::VectorXf &)> func,
                                            std::function<void(const Eigen::VectorXf &,
                                                               Eigen::VectorXf &)> gradFunc) {
    this->grad.resize(mVars.size());
    grad.fill(0);
    gradFunc(mVars, grad);
    mVars += -stepSize * grad;
    const float err = func(mVars);
    this->updateBestVars(err);
    if (this->settings.verbose)
        std::cout << "err " << err << std::endl;
}

void NumGrind::Solvers::SGDSolver::updateBestVars(const float err) {
    if(err < bestErr){
        bestErr = err;
        mBestVars = mVars;
    }
}
