#ifndef NUMGRIND_STOCHASTICGRADIENTDESCENTSOLVER_H
#define NUMGRIND_STOCHASTICGRADIENTDESCENTSOLVER_H

#include <functional>
#include "Eigen/Core"
#include "SolverSettings.h"

namespace NumGrind {
    namespace solvers {

        class StochasticGradientDescentSolver {
        public:
            StochasticGradientDescentSolver(const SolverSettings &settings, const float stepSize);

            void makeStep(std::function<float (const Eigen::VectorXf &)> func, std::function<void (const Eigen::VectorXf &, Eigen::VectorXf &)> gradFunc, Eigen::VectorXf &vars);

        protected:
            const SolverSettings settings;
            const float stepSize;
            Eigen::VectorXf grad;
        };

    }
}

#endif //NUMGRIND_STOCHASTICGRADIENTDESCENTSOLVER_H
