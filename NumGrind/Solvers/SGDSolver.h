#ifndef NUMGRIND_STOCHASTICGRADIENTDESCENTSOLVER_H
#define NUMGRIND_STOCHASTICGRADIENTDESCENTSOLVER_H

#include <limits>
#include <functional>
#include <cfloat>
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

            const Eigen::VectorXf &vars()     { return this->mVars; }
            const Eigen::VectorXf &bestVars() { return this->mBestVars; }

        protected:
            void updateBestVars(const float err);
            const SolverSettings settings;
            const float stepSize;
            float bestErr = FLT_MAX;
            Eigen::VectorXf grad;
            Eigen::VectorXf mVars;
            Eigen::VectorXf mBestVars;
        };

    }
}

#endif //NUMGRIND_STOCHASTICGRADIENTDESCENTSOLVER_H
