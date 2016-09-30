#include "numericalgradient.h"
#include <iostream>

void NumGrind::solvers::numericalGradient(
        std::function<float(const Eigen::VectorXf &)> func,
        const Eigen::VectorXf &varsInit,
        const float dx,
        Eigen::VectorXf &grad)
{
    assert(varsInit.size() == grad.size());
    Eigen::VectorXf vars = varsInit;
    const float valInit = func(vars);
    const int nVars = vars.size();

    grad.fill(0);

    for(int varInd = 0; varInd < nVars; ++varInd){
        vars[varInd] += dx;
        const float valCurrent = func(vars);
        vars[varInd] -= dx;
        grad[varInd] = (valCurrent - valInit) / dx;
    }
}
