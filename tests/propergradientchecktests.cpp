#include <cmath>
#include <iostream>
#include "gtest/gtest.h"
#include "numgrind.h"
#include "Utils/utils.h"

#include "testhelpers.h"

using namespace NumGrind;

TEST(NumGrindProperCheckGrad, testLinear01) {
    std::cout << "WARNING: testLinear01 not fully implemented";
    using namespace NumGrind::SymbolicGraph;
    NumGrind::GraphManager gm;

    Eigen::MatrixXf data(4, 2);
    Eigen::VectorXf targets(4);
    data << 0, 0,
            0, 1,
            1, 0,
            1, 1;
    targets << 0, 1, 1, 0;

    std::default_random_engine generator;
    generator.seed(42);

    auto X = gm.constant(data);
    auto y = gm.constant(targets);

    auto W1 = gm.variable(NumGrind::Utils::gaussf(2, 1, 0.0, 0.5, generator));
    auto b1 = gm.variable(NumGrind::Utils::gaussf(0.0, 0.5, generator));
    auto f = matmult(X, W1) + b1;
    auto residual = f - y;
    auto err = dot(residual, residual);

    auto vars = gm.initializeVariables();
    auto func = gm.funcFromNode(&err);
    auto grad1 = gm.initializeGradient(vars);
    auto grad2 = gm.initializeGradient(vars);

    vars.fill(0);
    err.node()->forwardPass(vars);
    err.node()->backwardPass(1, grad2);
    //ASSERT_TRUE((grad1 - grad2).norm() < 0.01);
}

