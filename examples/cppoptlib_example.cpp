#include "numgrind.h"
#include "cppoptlib/problem.h"
#include "cppoptlib/meta.h"
#include "cppoptlib/solver/bfgssolver.h"
#include "cppoptlib/solver/gradientdescentsolver.h"

class NumGrindProblem : public cppoptlib::Problem<float> {
public:
    using typename cppoptlib::Problem<float>::TVector;
    using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;

    NumGrindProblem(NumGrind::SymbolicGraph::SymbolicScalarNode funcToMinimize): funcToMinimize(funcToMinimize) {}

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

int main() {
    NumGrind::GraphManager gm;

    auto a = gm.variable(10) - 2;
    auto b = a * a + 5;

    NumGrindProblem problem(b);

    auto vars = gm.initializeVariables();
    cppoptlib::BfgsSolver<NumGrindProblem> solver;
//    cppoptlib::GradientDescentSolver<NumGrindProblem> solver;

    solver.minimize(problem, vars);
    std::cout << vars << std::endl;

    return 0;
}
