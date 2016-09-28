#include "gradientchecktests.h"

#include <cmath>
#include <iostream>
#include "gtest/gtest.h"
#include "numgrind.h"
#include "utils.h"
#include "solvers/numericalgradient.h"

#include "helpers.h"

using namespace NumGrind;

TEST(NumGrindCheckGrad, testNumericalGradient01) {
    auto f = [](const Eigen::VectorXf &vars) { return vars.dot(vars); };
    const Eigen::VectorXf vars = utils::vec2EVecf({1, 2, 3});
    Eigen::VectorXf grad(vars.size());
    solvers::numericalGradient(f, vars, 0.00001, grad);

    ASSERT_TRUE(fabs(grad[0] - 2.0) < 0.01);
    ASSERT_TRUE(fabs(grad[1] - 4.0) < 0.01);
    ASSERT_TRUE(fabs(grad[2] - 6.0) < 0.01);
}

TEST(NumGrindCheckGrad, testSigmoidmain01) {
    using namespace NumGrind;
    using namespace NumGrind::SymbolicNodeOps;

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
    auto f = apply<helpers::sigmoid, helpers::sigmoidDer>(matmult(X, w));
    auto residual = f - y;
    auto err = dot(residual, residual);
    auto vars = gm.initializeVariables();
    vars << 0.1, 0.02;
    auto func = gm.funcFromNode(&err);

    auto grad1 = gm.initializeGradient(vars);
    auto grad2 = gm.initializeGradient(vars);

    solvers::numericalGradient(func, vars, 0.00001, grad1);

    err.node()->forwardPass(vars);
    err.node()->backwardPass(1, grad2);
    std::cout << grad1 << std::endl;
    std::cout << grad2 << std::endl;
    ASSERT_TRUE((grad1 - grad2).norm() < 0.01);
}
