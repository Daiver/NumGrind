
#include <cmath>
#include <iostream>
#include "gtest/gtest.h"
#include "numgrind.h"
#include "utils.h"
#include "Solvers/numericalgradient.h"

#include "testhelpers.h"

using namespace NumGrind;

TEST(NumGrindCheckGrad, testNumericalGradient01) {
    auto f = [](const Eigen::VectorXf &vars) { return vars.dot(vars); };
    const Eigen::VectorXf vars = Utils::vec2EVecf({1, 2, 3});
    Eigen::VectorXf grad(vars.size());
    Solvers::numericalGradient(f, vars, 0.00001, grad);

    ASSERT_TRUE(fabs(grad[0] - 2.0) < 0.01);
    ASSERT_TRUE(fabs(grad[1] - 4.0) < 0.01);
    ASSERT_TRUE(fabs(grad[2] - 6.0) < 0.01);
}

TEST(NumGrindCheckGrad, testSquare01) {
    using namespace NumGrind;
    using namespace NumGrind::SymbolicGraph;

    GraphManager gm;

    auto w = gm.variable(2, 1, 0);
    auto f = apply<testhelpers::square, testhelpers::squareDer>(w);
    auto err = dot(f, f);
    auto vars = gm.initializeVariables();
    vars << 0.1, 0.02;
    auto func = gm.funcFromNode(&err);

    auto grad1 = gm.initializeGradient(vars);
    auto grad2 = gm.initializeGradient(vars);

    err.node()->forwardPass(vars);
    err.node()->backwardPass(1, grad2);

    Solvers::numericalGradient(func, vars, 0.001, grad1);

    ASSERT_TRUE((grad1 - grad2).norm() < 0.01);
}


TEST(NumGrindCheckGrad, testSigmoid01) {

    auto f = [](const Eigen::VectorXf &vars) { return testhelpers::sigmoid(vars[0]); };

    const float val = 0.02;

    const Eigen::VectorXf vars = Utils::vec2EVecf({val});
    Eigen::VectorXf grad(vars.size());
    Solvers::numericalGradient(f, vars, 0.00001, grad);
    const float realDer = testhelpers::sigmoidDer(val);
    ASSERT_TRUE(fabs(grad[0] - realDer) < 0.001);
}

TEST(NumGrindCheckGrad, testLogistic01) {

    using namespace NumGrind;
    using namespace NumGrind::SymbolicGraph;

    GraphManager gm;

    Eigen::MatrixXf data(4, 2);
    Eigen::VectorXf targets(4);
    data << 0, 0,
            0, 1,
            1, 0,
            1, 1;
    targets << 0, 0, 0, 1;

    auto X = gm.constant(data);
    auto y = gm.constant(targets);
    auto w = gm.variable(2, 1, 0);
    auto b = gm.variable(0);
    auto f = apply<testhelpers::sigmoid, testhelpers::sigmoidDer>(matmult(X, w) + b);
    auto residual = f - y;
    auto err = dot(residual, residual);

    auto vars = gm.initializeVariables();
    auto func = gm.funcFromNode(&err);

    auto grad1 = gm.initializeGradient(vars);
    auto grad2 = gm.initializeGradient(vars);

    vars << -0.1, 0.02, 0.03;
    Solvers::numericalGradient(func, vars, 0.001, grad1);
    err.node()->forwardPass(vars);
    err.node()->backwardPass(1, grad2);
    ASSERT_TRUE((grad1 - grad2).norm() < 0.01);
}

TEST(NumGrindCheckGrad, testLogistic02) {

    using namespace NumGrind;
    using namespace NumGrind::SymbolicGraph;

    GraphManager gm;

    Eigen::MatrixXf data(4, 2);
    Eigen::VectorXf targets(4);
    data << 0, 0,
            0, 1,
            1, 0,
            1, 1;
    targets << 0, 0, 0, 1;

    auto X = gm.constant(data);
    auto y = gm.constant(targets);
    auto w = gm.variable(2, 1, 0);
    auto b = gm.variable(0);
    auto f = apply<testhelpers::sigmoid, testhelpers::sigmoidDer>(matmult(X, w) + b);
    auto residual = f - y;
    auto err = dot(residual, residual);

    auto vars = gm.initializeVariables();
    auto func = gm.funcFromNode(&err);

    auto grad1 = gm.initializeGradient(vars);
    auto grad2 = gm.initializeGradient(vars);

    vars << 0.7, -0.52, 0.13;
    err.node()->forwardPass(vars);
    err.node()->backwardPass(1, grad2);
    Solvers::numericalGradient(func, vars, 0.001, grad1);
    ASSERT_TRUE((grad1 - grad2).norm() < 0.01);
}

TEST(NumGrindCheckGrad, testMLP01) {
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

    auto W1 = gm.variable(NumGrind::Utils::gaussf(2, 2, 0.0, 0.5, generator));
    auto b1 = gm.variable(NumGrind::Utils::gaussf(1, 2, 0.0, 0.5, generator));
    auto W2 = gm.variable(NumGrind::Utils::gaussf(2, 1, 0.0, 0.01, generator));
    auto b2 = gm.variable(NumGrind::Utils::gaussf(0.0f, 0.01f, generator));
    auto f1 = apply<testhelpers::sigmoid, testhelpers::sigmoidDer>(matmult(X, W1) + b1);
    auto f2 = apply<testhelpers::sigmoid, testhelpers::sigmoidDer>(matmult(f1, W2) + b2);
    auto residual = f2 - y;
    auto err = dot(residual, residual);

    auto vars = gm.initializeVariables();

    auto func = gm.funcFromNode(&err);

    auto grad1 = gm.initializeGradient(vars);
    auto grad2 = gm.initializeGradient(vars);

    vars <<
            0.7, -0.52,
            0.13, -0.03,
            1.0, 1.2,
            -0.04, 0.01,
            0.05;
    err.node()->forwardPass(vars);
    err.node()->backwardPass(1, grad2);
    Solvers::numericalGradient(func, vars, 0.001, grad1);
    ASSERT_TRUE((grad1 - grad2).norm() < 0.01);
}