#include "GradientDescentSolver.h"

#include <iostream>

using namespace NumGrind;

void Solvers::gradientDescent(
        const SolverSettings &settings,
        const float stepSize,
        std::function<float(const Eigen::VectorXf &)> func,
        std::function<void(const Eigen::VectorXf &, Eigen::VectorXf &)> gradF, Eigen::VectorXf &vars) {
    Eigen::VectorXf grad = Eigen::VectorXf::Zero(vars.size());
    float errOld = func(vars);
    if(settings.verbose)
        std::cout << "Initial err " << errOld << std::endl;
    for (int iter = 0; iter < settings.nMaxIterations; ++iter) {
        grad.fill(0);
        gradF(vars, grad);
        vars -= grad * stepSize;
        const float err = func(vars);
        const float dErr = fabs(err - errOld);
        const float gradL2 = grad.norm();
        if(settings.verbose)
            std::cout << iter << "> err:" << err << " |g|_2:" << gradL2 << " dErr:" << dErr << std::endl;
//        std::cout << grad(0, 0) << std::endl;
//        std::cout << mVars(0, 0) << std::endl;

        //if(gradL2 < settings.minGradL2)
            //break;
        //if(dErr < settings.minDErr)
            /*break;*/

        errOld = err;
    }
}
