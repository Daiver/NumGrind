#ifndef NUMGRIND_SGDWITHMOMENTUMSOLVER_H
#define NUMGRIND_SGDWITHMOMENTUMSOLVER_H

#include "SGDSolver.h"

namespace NumGrind {
    namespace Solvers {

        class SGDWithMomentumSolver : public SGDSolver {
        public:
            SGDWithMomentumSolver(
                    const NumGrind::Solvers::SolverSettings &settings,
                    const float stepSize,
                    const float momentumCoeff,
                    const Eigen::VectorXf &vars);

            void makeStep(std::function<float(const Eigen::VectorXf &)> func,
                          std::function<void(const Eigen::VectorXf &,
                                             Eigen::VectorXf &)> gradFunc);

        protected:
            Eigen::VectorXf oldGrad;
            float momentumCoeff;
        };

    }
}

#endif //NUMGRIND_SGDWITHMOMENTUMSOLVER_H
