#include <cmath>
#include <iostream>
#include "gtest/gtest.h"
#include "numgrind.h"
#include "utils.h"

using namespace NumGrind;
using namespace NumGrind::CompGraph;

TEST(NumGrindScalarSuit, test01) {

    auto n1 = GNScalarVariable(0);
    auto n2 = GNScalarVariable(1);
    auto n3 = GNScalarConst(2.0);

    auto n4 = GNScalarMult(&n1, &n3);
    auto n5 = GNScalarSum(&n2, &n4);
    auto n6 = GNScalarMult(&n1, &n5);

    auto graph = n6;

    Eigen::VectorXf vars = utils::vec2EVecf({3, 4});
    Eigen::VectorXf grad = Eigen::VectorXf::Zero(vars.size());
    graph.forwardPass(vars);
    graph.backwardPass(1, grad);

    ASSERT_FLOAT_EQ(grad[0], 16.0f);
    ASSERT_FLOAT_EQ(grad[1], 3.0f);
}


TEST(NumGrindVectorSuit, testDotProduct01) {
    auto n1 = GNVectorVariable({0, 1});
    auto n2 = GNVectorVariable({2, 3});

    auto n3 = GNMatrixElementWiseProduct(&n1, &n2);
    auto n4 = GNDotProduct(&n3, &n1);

    auto graph = n4;

    Eigen::VectorXf vars = utils::vec2EVecf({1, 2, 3, 4});
    Eigen::VectorXf grad = Eigen::VectorXf::Zero(vars.size());
    graph.forwardPass(vars);
    graph.backwardPass(1, grad);

    ASSERT_FLOAT_EQ(grad[0], 6.0f);
    ASSERT_FLOAT_EQ(grad[1], 16.0f);
    ASSERT_FLOAT_EQ(grad[2], 1.0f);
    ASSERT_FLOAT_EQ(grad[3], 4.0f);
}

TEST(NumGrindVectorSuit, testDotProduct02) {
    Eigen::MatrixXf mat(3, 1);
    mat << 5, 12, 13;
    auto a = GNMatrixConstant(mat);
    auto b = GNVectorVariable({0, 1, 2});
    auto expr = GNDotProduct(&a, &b);

    Eigen::VectorXf vars = utils::vec2EVecf({1, 2, 3});
    Eigen::VectorXf grad = Eigen::VectorXf::Zero(vars.size());
    expr.forwardPass(vars);
    expr.backwardPass(1, grad);
    ASSERT_FLOAT_EQ(grad[0], 5.0f);
    ASSERT_FLOAT_EQ(grad[1], 12.0f);
    ASSERT_FLOAT_EQ(grad[2], 13.0f);
}

TEST(NumGrindMatrixSuit, test01) {
    auto n1 = GNMatrixVariable(2, 2, {0, 1, 2, 3});
    Eigen::VectorXf vars(4);
    vars << 1, 2, 3, 4;
    n1.forwardPass(vars);
    auto res = n1.value();
//    std::cout << std::endl << res << std::endl;
    ASSERT_FLOAT_EQ(res(0, 0), 1.0f);
    ASSERT_FLOAT_EQ(res(0, 1), 2.0f);
    ASSERT_FLOAT_EQ(res(1, 0), 3.0f);
    ASSERT_FLOAT_EQ(res(1, 1), 4.0f);
}

TEST(NumGrindMatrixSuit, test02) {
    auto n1 = GNMatrixVariable(2, 2, {0, 1, 2, 3});
    auto n2 = GNMatrixReduceSum(&n1);
    Eigen::VectorXf vars(4);
    vars << 1, 2, 3, 4;
    Eigen::VectorXf grad = Eigen::VectorXf::Zero(vars.size());

    n2.forwardPass(vars);
    n2.backwardPass(1.0, grad);

    ASSERT_FLOAT_EQ(grad[0], 1.0f);
    ASSERT_FLOAT_EQ(grad[1], 1.0f);
    ASSERT_FLOAT_EQ(grad[2], 1.0f);
    ASSERT_FLOAT_EQ(grad[3], 1.0f);
}

TEST(NumGrindMatrixSuit, test03) {
    auto n1 = GNMatrixVariable(2, 2, {0, 1, 2, 3});
    auto n2 = GNMatrixVariable(2, 2, {4, 5, 6, 7});
    auto n3 = GNMatrixProduct(&n1, &n2);
    Eigen::VectorXf vars(8);
    vars << 1, 2, 3, 4, 5, 6, 7, 8;
    n3.forwardPass(vars);

    auto res = n3.value();
    ASSERT_FLOAT_EQ(res(0, 0), 19.0f);
    ASSERT_FLOAT_EQ(res(0, 1), 22.0f);
    ASSERT_FLOAT_EQ(res(1, 0), 43.0f);
    ASSERT_FLOAT_EQ(res(1, 1), 50.0f);
}

TEST(NumGrindMatrixSuit, test04) {
    auto n1 = GNMatrixVariable(2, 2, {0, 1, 2, 3});
    auto n2 = GNMatrixVariable(2, 2, {4, 5, 6, 7});
    auto n3 = GNMatrixProduct(&n1, &n2);
    auto n4 = GNMatrixReduceSum(&n3);
    Eigen::VectorXf vars(8);
    vars << 1, 2, 3, 4,
            5, 6, 7, 8;
    Eigen::VectorXf grad = Eigen::VectorXf::Zero(vars.size());
    n4.forwardPass(vars);
    n4.backwardPass(1.0, grad);

    ASSERT_FLOAT_EQ(grad[0], 5 + 6);
    ASSERT_FLOAT_EQ(grad[1], 7 + 8);
    ASSERT_FLOAT_EQ(grad[2], 5 + 6);
    ASSERT_FLOAT_EQ(grad[3], 7 + 8);

    ASSERT_FLOAT_EQ(grad[4], 1 + 3);
    ASSERT_FLOAT_EQ(grad[5], 1 + 3);
    ASSERT_FLOAT_EQ(grad[6], 2 + 4);
    ASSERT_FLOAT_EQ(grad[7], 2 + 4);
}

TEST(NumGrindMatrixSuit, test05) {
    auto A = GNMatrixVariable(2, 3, {0, 1, 2, 3, 4, 5});
    auto B = GNMatrixVariable(3, 1, {6, 7, 8});
    auto C = GNMatrixProduct(&A, &B);
    auto expr = GNMatrixReduceSum(&C);
    Eigen::VectorXf vars(9);
    vars << 1, 2, 3, 4, 5, 6,
            7, 8, 9;
    Eigen::VectorXf grad = Eigen::VectorXf::Zero(vars.size());
    expr.forwardPass(vars);
    expr.backwardPass(1.0, grad);

    ASSERT_FLOAT_EQ(grad[0], 7);
    ASSERT_FLOAT_EQ(grad[1], 8);
    ASSERT_FLOAT_EQ(grad[2], 9);
    ASSERT_FLOAT_EQ(grad[3], 7);
    ASSERT_FLOAT_EQ(grad[4], 8);
    ASSERT_FLOAT_EQ(grad[5], 9);

    ASSERT_FLOAT_EQ(grad[6], 1 + 4);
    ASSERT_FLOAT_EQ(grad[7], 2 + 5);
    ASSERT_FLOAT_EQ(grad[8], 3 + 6);
}

TEST(NumGrindMatrixSuit, test06) {
    auto A = GNMatrixVariable(2, 3, {0, 1, 2, 3, 4, 5});
    auto B = GNMatrixVariable(3, 1, {6, 7, 8});
    auto c = GNMatrixProduct(&A, &B);
    auto w = GNVectorVariable({9, 10});
//    auto ct = GNMatrixTranspose(&c);
    auto expr = GNDotProduct(&c, &w);
    Eigen::VectorXf vars(11);
    vars << 1, 2, 3, 4, 5, 6,
            7, 8, 9,
            0.13, 0.22;
    Eigen::VectorXf grad = Eigen::VectorXf::Zero(vars.size());
    expr.forwardPass(vars);
    expr.backwardPass(1.0, grad);

    ASSERT_FLOAT_EQ(grad[0], 7 * 0.13);
    ASSERT_FLOAT_EQ(grad[1], 8 * 0.13);
    ASSERT_FLOAT_EQ(grad[2], 9 * 0.13);
    ASSERT_FLOAT_EQ(grad[3], 7 * 0.22);
    ASSERT_FLOAT_EQ(grad[4], 8 * 0.22);
    ASSERT_FLOAT_EQ(grad[5], 9 * 0.22);

    ASSERT_FLOAT_EQ(grad[6], 1 * 0.13 + 4 * 0.22);
    ASSERT_FLOAT_EQ(grad[7], 2 * 0.13 + 5 * 0.22);
    ASSERT_FLOAT_EQ(grad[8], 3 * 0.13 + 6 * 0.22);

    ASSERT_FLOAT_EQ(grad[9], 1*7 + 2*8 + 3*9);
    ASSERT_FLOAT_EQ(grad[10], 4*7 + 5*8 + 6*9);
}

TEST(NumGrindMatrixSuit, test07) {
    auto A = GNMatrixVariable(2, 3, {0, 1, 2, 3, 4, 5});
    auto B = GNMatrixVariable(3, 1, {6, 7, 8});
    auto C = GNMatrixProduct(&A, &B);
    auto expr = GNDotProduct(&C, &C);
    Eigen::VectorXf vars(9);
    vars << 1, 2, 3, 4, 5, 6,
            7, 8, 9;
    Eigen::VectorXf grad = Eigen::VectorXf::Zero(vars.size());
    expr.forwardPass(vars);
    expr.backwardPass(1.0, grad);

    ASSERT_FLOAT_EQ(grad[0], 2*(1*7 + 2*8 + 3*9)*7);
    ASSERT_FLOAT_EQ(grad[1], 2*(1*7 + 2*8 + 3*9)*8);
    ASSERT_FLOAT_EQ(grad[2], 2*(1*7 + 2*8 + 3*9)*9);
    ASSERT_FLOAT_EQ(grad[3], 2*(4*7 + 5*8 + 6*9)*7);
    ASSERT_FLOAT_EQ(grad[4], 2*(4*7 + 5*8 + 6*9)*8);
    ASSERT_FLOAT_EQ(grad[5], 2*(4*7 + 5*8 + 6*9)*9);

    ASSERT_FLOAT_EQ(grad[6], 2*(1*7 + 2*8 + 3*9)*1 + 2*(4*7 + 5*8 + 6*9)*4);
    ASSERT_FLOAT_EQ(grad[7], 2*(1*7 + 2*8 + 3*9)*2 + 2*(4*7 + 5*8 + 6*9)*5);
    ASSERT_FLOAT_EQ(grad[8], 2*(1*7 + 2*8 + 3*9)*3 + 2*(4*7 + 5*8 + 6*9)*6);
}

float sigmoid(float z)
{
    return 1.0/(1.0 + exp(-z));
}

float sigmoidDer(float z)
{
    return sigmoid(z) * sigmoid(1.0 - z);
}

TEST(NumGrindMatrixSuit, test08) {
    auto A = GNMatrixVariable(2, 3, {0, 1, 2, 3, 4, 5});

    auto sA = GNMatrixMapUnaryFunction<float, sigmoid, sigmoidDer>(&A);
    auto expr = GNMatrixReduceSum(&sA);

    Eigen::VectorXf vars(6);
    vars << -0.2, -0.1, 0.0, 0.1, 0.2, 0.3;
    Eigen::VectorXf grad = Eigen::VectorXf::Zero(vars.size());
    expr.forwardPass(vars);
    expr.backwardPass(1.0, grad);

    const float eps = 1e-4;

    auto res = expr.value();
    ASSERT_TRUE(fabs(res - 3.0744425168116591) < eps);

    ASSERT_TRUE(fabs(grad[0] - 0.3459637297540461) < eps);
    ASSERT_TRUE(fabs(grad[1] - 0.35638916496192907) < eps);
    ASSERT_TRUE(fabs(grad[2] - 0.36552928931500245) < eps);
    ASSERT_TRUE(fabs(grad[3] - 0.37323369222663105) < eps);
    ASSERT_TRUE(fabs(grad[4] - 0.37937142700199805) < eps);
    ASSERT_TRUE(fabs(grad[5] - 0.38383546554705678) < eps);

}


TEST(NumGrindMatrixSuit, test09) {
    auto A = GNMatrixVariable(3, 1, {0, 1, 2});
    auto b = GNScalarVariable(3);
    auto c = GNMatrixScalarSum(&A, &b);
    auto expr = GNDotProduct(&c, &c);

    Eigen::VectorXf vars(4);
    vars << 1, 2, 3, 4;
    Eigen::VectorXf grad = Eigen::VectorXf::Zero(vars.size());
    expr.forwardPass(vars);
    expr.backwardPass(1.0, grad);

    ASSERT_FLOAT_EQ(grad[0], 2*(1 + 4));
    ASSERT_FLOAT_EQ(grad[1], 2*(2 + 4));
    ASSERT_FLOAT_EQ(grad[2], 2*(3 + 4));
    ASSERT_FLOAT_EQ(grad[3], 2*(1 + 4) + 2*(2 + 4) + 2*(3 + 4));
}


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

float square(float x) { return x*x; }
float squareDer(float x) { return 2*x; }

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
    auto d = apply<square, squareDer>(c);
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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
