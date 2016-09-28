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
