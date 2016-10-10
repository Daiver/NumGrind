#ifndef NUMGRIND_SGDWITHMOMENTUMSOLVER_H
#define NUMGRIND_SGDWITHMOMENTUMSOLVER_H

#include "SGDSolver.h"

namespace NumGrind {
    namespace Solvers {

        class SGDWithMomentumSolver : public SGDSolver{
        public:
            SGDWithMomentumSolver(
                    const NumGrind::Solvers::SolverSettings &settings,
                    const float stepSize,
                    const Eigen::VectorXf &vars);
        };

    }
}

#endif //NUMGRIND_SGDWITHMOMENTUMSOLVER_H
