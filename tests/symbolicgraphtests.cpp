#include "symbolicgraphtests.h"

#include <cmath>
#include <iostream>
#include "gtest/gtest.h"
#include "numgrind.h"
#include "utils.h"

#include "helpers.h"

using namespace NumGrind;

TEST(NumGrindGraphManagerSuit, testInitializeVarsAndGrad01) {

    using namespace SymbolicNodeOps;
    GraphManager manager;
    auto a = manager.variable(1);
    auto b = manager.variable(2);
    auto c = a + b;

    auto vars = manager.initializeVariables();
    auto grad = manager.initializeGradient(vars);

    ASSERT_FLOAT_EQ(vars[0], 1.0);
    ASSERT_FLOAT_EQ(vars[1], 2.0);

    c.node()->forwardPass(vars);
    c.node()->backwardPass(1.0, grad);

    ASSERT_FLOAT_EQ(grad[0], 1.0);
    ASSERT_FLOAT_EQ(grad[1], 1.0);
}

TEST(NumGrindGraphManagerSuit, testInitializeVarsAndGrad02) {

    using namespace SymbolicNodeOps;
    GraphManager manager;
    auto a = manager.variable(7);
    auto b = manager.variable(12);
    auto c = a - b;
    auto d = a*c;

    auto vars = manager.initializeVariables();
    auto grad = manager.initializeGradient(vars);

    ASSERT_FLOAT_EQ(vars[0], 7.0);
    ASSERT_FLOAT_EQ(vars[1], 12.0);

    d.node()->forwardPass(vars);
    d.node()->backwardPass(1.0, grad);

    ASSERT_FLOAT_EQ(grad[0], 2.0);
    ASSERT_FLOAT_EQ(grad[1], -7.0);
}


TEST(NumGrindGraphManagerSuit, testInitializeVarsAndGrad03) {

    using namespace SymbolicNodeOps;
    GraphManager manager;
    auto a = manager.variable(3);
    auto b = manager.variable(2);
    auto c = manager.variable(12);
    auto d = a + b;
    auto e = a * b;

    auto vars = manager.initializeVariables();
    auto grad = manager.initializeGradient(vars);

    ASSERT_FLOAT_EQ(vars[0], 3.0);
    ASSERT_FLOAT_EQ(vars[1], 2.0);
    ASSERT_FLOAT_EQ(vars[2], 12.0);

    e.node()->forwardPass(vars);
    e.node()->backwardPass(1.0, grad);

    ASSERT_FLOAT_EQ(grad[0], 2.0);
    ASSERT_FLOAT_EQ(grad[1], 3.0);
    ASSERT_FLOAT_EQ(grad[2], 0.0);
}

TEST(NumGrindGraphManagerSuit, testInitializeVarsAndGrad04) {

    using namespace SymbolicNodeOps;
    GraphManager manager;
    auto a = 2.0f*manager.variable(3);
    auto b = a + manager.variable(10);

    auto vars = manager.initializeVariables();
    auto grad = manager.initializeGradient(vars);

    ASSERT_FLOAT_EQ(vars[0], 3.0);
    ASSERT_FLOAT_EQ(vars[1], 10.0);

    b.node()->forwardPass(vars);
    b.node()->backwardPass(1.0, grad);

    ASSERT_FLOAT_EQ(grad[0], 2.0);
    ASSERT_FLOAT_EQ(grad[1], 1.0);
}

TEST(NumGrindGraphManagerSuit, testInitializeVarsAndGrad05) {

    using namespace SymbolicNodeOps;
    GraphManager manager;
    auto f = 2*manager.variable(13) * manager.variable(16) - 10 + manager.variable(2)/3.0;

    auto vars = manager.initializeVariables();
    auto grad = manager.initializeGradient(vars);

    ASSERT_FLOAT_EQ(vars[2], 13.0);
    ASSERT_FLOAT_EQ(vars[1], 16.0);
    ASSERT_FLOAT_EQ(vars[0], 2.0);

    f.node()->forwardPass(vars);
    f.node()->backwardPass(1.0, grad);

    ASSERT_FLOAT_EQ(grad[2], 2*16.0);
    ASSERT_FLOAT_EQ(grad[1], 2*13.0);
    ASSERT_FLOAT_EQ(grad[0], 1.0/3.0);
}

TEST(NumGrindGraphManagerSuit, testInitializeVarsAndGradMat01) {

    using namespace SymbolicNodeOps;
    GraphManager manager;
    Eigen::VectorXf val1(3);
    val1 << 4, 6, 1;
    Eigen::MatrixXf val2(2, 3);
    val2 << 2, 4, 6,
            8, 10, 12;
    auto a = manager.variable(val1);
    auto b = manager.variable(val2);
    auto c = manager.variable(1, 4, 13);

    auto vars = manager.initializeVariables();

    ASSERT_EQ(vars.size(), 3 + 6 + 4);

    ASSERT_FLOAT_EQ(vars[0], 4.0);
    ASSERT_FLOAT_EQ(vars[1], 6.0);
    ASSERT_FLOAT_EQ(vars[2], 1.0);

    ASSERT_FLOAT_EQ(vars[3], 2.0);
    ASSERT_FLOAT_EQ(vars[4], 4.0);
    ASSERT_FLOAT_EQ(vars[5], 6.0);
    ASSERT_FLOAT_EQ(vars[6], 8.0);
    ASSERT_FLOAT_EQ(vars[7], 10.0);
    ASSERT_FLOAT_EQ(vars[8], 12.0);

    ASSERT_FLOAT_EQ(vars[9],  13.0);
    ASSERT_FLOAT_EQ(vars[10], 13.0);
    ASSERT_FLOAT_EQ(vars[11], 13.0);
}

TEST(NumGrindGraphMatrixSuit, testMatSum01) {

    using namespace SymbolicNodeOps;
    GraphManager gm;

    Eigen::MatrixXf aMat(3, 2);
    aMat << 1, 2,
            3, 4,
            5, 6;
    Eigen::MatrixXf bMat(1, 2);
    bMat << 10, 20;
    auto a = gm.variable(aMat);
    auto b = gm.variable(bMat);
    auto c = a + b;
    auto d = apply<helpers::square, helpers::squareDer>(c);
    auto e = reduceSum(d);
    auto vars = gm.initializeVariables();
    auto grad = gm.initializeGradient(vars);
    c.node()->forwardPass(vars);
    auto res1 = c.value();

    ASSERT_FLOAT_EQ(res1(0, 0), 11);
    ASSERT_FLOAT_EQ(res1(0, 1), 22);
    ASSERT_FLOAT_EQ(res1(1, 0), 13);
    ASSERT_FLOAT_EQ(res1(1, 1), 24);
    ASSERT_FLOAT_EQ(res1(2, 0), 15);
    ASSERT_FLOAT_EQ(res1(2, 1), 26);

    e.node()->forwardPass(vars);
    e.node()->backwardPass(1.0, grad);

    ASSERT_FLOAT_EQ(grad[0], 2*11);
    ASSERT_FLOAT_EQ(grad[1], 2*22);
    ASSERT_FLOAT_EQ(grad[2], 2*13);
    ASSERT_FLOAT_EQ(grad[3], 2*24);
    ASSERT_FLOAT_EQ(grad[4], 2*15);
    ASSERT_FLOAT_EQ(grad[5], 2*26);

    ASSERT_FLOAT_EQ(grad[6], 2*11 + 2*13 + 2*15);
    ASSERT_FLOAT_EQ(grad[7], 2*22 + 2*24 + 2*26);

}
