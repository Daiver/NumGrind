#include "GradientDescentSolver.h"

#include <iostream>

void solvers::gradientDescent(const int nIters, const float stepSize, GNScalarOutput &function, Eigen::VectorXf &vars) {
    Eigen::VectorXf grad = Eigen::VectorXf::Zero(vars.size());
    function.forwardPass(vars);
    const float err = function.value();
    std::cout << "Initial err " << err << std::endl;
    for(int iter = 0; iter < nIters; ++iter){
        function.backwardPass(1, grad);
        vars -= grad * stepSize;
        function.forwardPass(vars);
        const float err = function.value();
        std::cout << iter << " : " << err << std::endl;
    }
}
