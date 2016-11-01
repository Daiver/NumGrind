#include <cmath>
#include <iostream>
#include "gtest/gtest.h"
#include "numgrind.h"
#include "Utils/utils.h"

#include "testhelpers.h"

using namespace NumGrind;

TEST(NumGrindProperCheckGrad, testLinear01) {
    //std::cout << "WARNING: testLinear01 not fully implemented";
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
    auto err = sumOfSquares(f - y);

    auto vars = gm.initializeVariables();
    auto func = gm.funcFromNode(&err);
    //auto grad1 = gm.initializeGradient(vars);
    auto grad = gm.initializeGradient(vars);

    vars.fill(0);
    err.node()->forwardPass(vars);
    err.node()->backwardPass(1, grad);
    //std::cout << grad;
    ASSERT_FLOAT_EQ(grad[0], -4);
    ASSERT_FLOAT_EQ(grad[1], -2);
    ASSERT_FLOAT_EQ(grad[2], -2);
    //ASSERT_TRUE((grad1 - grad2).norm() < 0.01);
}

TEST(NumGrindProperCheckGrad, testLinear02) {
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
    auto err = sumOfSquares(f - y);

    auto vars = gm.initializeVariables();
    auto func = gm.funcFromNode(&err);
    auto grad = gm.initializeGradient(vars);
    auto grad2 = gm.initializeGradient(vars);

    vars << 5, 0, 0;
    err.node()->forwardPass(vars);
    err.node()->backwardPass(1, grad);
    //std::cout << grad;
    ASSERT_FLOAT_EQ(grad[0], 36);
    ASSERT_FLOAT_EQ(grad[1], 8 + 10);
    ASSERT_FLOAT_EQ(grad[2], 18);
    //ASSERT_TRUE((grad1 - grad2).norm() < 0.01);
}



