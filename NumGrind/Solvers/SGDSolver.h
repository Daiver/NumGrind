#ifndef NUMGRIND_STOCHASTICGRADIENTDESCENTSOLVER_H
#define NUMGRIND_STOCHASTICGRADIENTDESCENTSOLVER_H

#include <functional>
#include "Eigen/Core"
#include "SolverSettings.h"

namespace NumGrind {
    namespace Solvers {

        class SGDSolver {
        public:
            SGDSolver(
                    const NumGrind::Solvers::SolverSettings &settings,
                    const float stepSize,
                    const Eigen::VectorXf &vars);

            void makeStep(std::function<float(const Eigen::VectorXf &)> func,
                          std::function<void(const Eigen::VectorXf &,
                                             Eigen::VectorXf &)> gradFunc);

            const Eigen::VectorXf &vars() { return this->mVars; }

        protected:
            const SolverSettings settings;
            const float stepSize;
            Eigen::VectorXf grad;
            Eigen::VectorXf mVars;
        };

    }
}

#endif //NUMGRIND_STOCHASTICGRADIENTDESCENTSOLVER_H
