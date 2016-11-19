#include "numgrind.h"
#include "CppOptLibTools/NumGrindProblem.h"
#include "cppoptlib/solver/bfgssolver.h"
#include "cppoptlib/solver/gradientdescentsolver.h"


int main() {
    NumGrind::GraphManager gm;

    auto a = gm.variable(10) - 2;
    auto b = a * a + 5;

    NumGrind::NumGrindProblem problem(b);

    auto vars = gm.initializeVariables();
    cppoptlib::BfgsSolver<NumGrind::NumGrindProblem> solver;
//    cppoptlib::GradientDescentSolver<NumGrindProblem> solver;

    solver.minimize(problem, vars);
    std::cout << vars << std::endl;

    return 0;
}
