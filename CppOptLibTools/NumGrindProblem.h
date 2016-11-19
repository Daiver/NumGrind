#ifndef NUMGRIND_NUMGRINDPROBLEM_H
#define NUMGRIND_NUMGRINDPROBLEM_H

#include "numgrind.h"
#include "cppoptlib/problem.h"


namespace NumGrind {
    class NumGrindProblem : public cppoptlib::Problem<float> {
    public:
        using typename cppoptlib::Problem<float>::TVector;
        using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;

        NumGrindProblem(NumGrind::SymbolicGraph::SymbolicScalarNode funcToMinimize) : funcToMinimize(funcToMinimize) {}

        float value(const TVector &params) {
            funcToMinimize.node()->forwardPass(params);
            return funcToMinimize.value();
        }

        void gradient(const TVector &params, TVector &grad) {
            grad.fill(0);//EXTREMELY IMPORTANT!
            funcToMinimize.node()->forwardPass(params);
            funcToMinimize.node()->backwardPass(1.0, grad);
        }

    protected:
        NumGrind::SymbolicGraph::SymbolicScalarNode funcToMinimize;

    };
}

#endif //NUMGRIND_NUMGRINDPROBLEM_H
