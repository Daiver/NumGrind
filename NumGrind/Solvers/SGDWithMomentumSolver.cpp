//
// Created by daiver on 11.10.16.
//

#include "SGDWithMomentumSolver.h"

NumGrind::Solvers::SGDWithMomentumSolver::SGDWithMomentumSolver(const NumGrind::Solvers::SolverSettings &settings,
                                                                const float stepSize, const Eigen::VectorXf &vars)
        : NumGrind::Solvers::SGDSolver(settings, stepSize, vars) {
    assert(false);//Not implemented
}
