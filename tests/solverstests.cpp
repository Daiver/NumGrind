#include <cmath>
#include <iostream>
#include "gtest/gtest.h"
#include "numgrind.h"
#include "Utils/utils.h"
#include "Solvers/GradientDescentSolver.h"
#include "testhelpers.h"

using namespace NumGrind;

TEST(NumGrindSolvers, testLinear01) {
    //std::cout << "WARNING: testLinear01 not fully implemented";
    using namespace NumGrind::SymbolicGraph;
    NumGrind::GraphManager gm;

    Eigen::MatrixXf data(4, 2);
    Eigen::VectorXf targets(4);
    data << 0, 0,
            1, 0,
            2, 0,
            3, 0;
    targets << 1, 3, 5, 7;

    auto X = gm.constant(data);
    auto y = gm.constant(targets);

    auto W1 = gm.variable(NumGrind::Utils::evecf({0, 0}));
    auto b1 = gm.variable(0);
    auto f = matmult(X, W1) + b1;
    auto err = sumOfSquares(f - y);

    auto vars = gm.initializeVariables();
    auto func = gm.funcFromNode(&err);
    //auto grad1 = gm.initializeGradient(vars);
    auto grad = gm.initializeGradient(vars);

    NumGrind::Solvers::SolverSettings settings;
    settings.nMaxIterations = 1000;
    settings.verbose = false;
    NumGrind::Solvers::gradientDescent(
        settings, 0.01,
        gm.funcFromNode(&err), gm.gradFromNode(&err), vars);
    ASSERT_TRUE(fabs(vars[0] - 1) < 0.0001);
    ASSERT_TRUE(fabs(vars[1] - 2) < 0.0001);
    ASSERT_TRUE(fabs(vars[2] - 0) < 0.0001);
}


