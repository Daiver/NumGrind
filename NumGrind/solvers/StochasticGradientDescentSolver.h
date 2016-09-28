#ifndef NUMGRIND_STOCHASTICGRADIENTDESCENTSOLVER_H
#define NUMGRIND_STOCHASTICGRADIENTDESCENTSOLVER_H

#include "SolverSettings.h"

namespace NumGrind {
    namespace solvers {

        class StochasticGradientDescentSolver {
        public:
            StochasticGradientDescentSolver(const SolverSettings &settings);

        protected:
            const SolverSettings settings;
        };

    }
}

#endif //NUMGRIND_STOCHASTICGRADIENTDESCENTSOLVER_H
