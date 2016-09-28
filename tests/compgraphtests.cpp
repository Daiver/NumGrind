
#include "compgraphtests.h"

#include <cmath>
#include <iostream>
#include "gtest/gtest.h"
#include "numgrind.h"
#include "utils.h"

#include "helpers.h"

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

TEST(NumGrindMatrixSuit, test08) {
    auto A = GNMatrixVariable(2, 3, {0, 1, 2, 3, 4, 5});

    auto sA = GNMatrixMapUnaryFunction<float, helpers::sigmoid, helpers::sigmoidDer>(&A);
    auto expr = GNMatrixReduceSum(&sA);

    Eigen::VectorXf vars(6);
    vars << -0.2, -0.1, 0.0, 0.1, 0.2, 0.3;
    Eigen::VectorXf grad = Eigen::VectorXf::Zero(vars.size());
    expr.forwardPass(vars);
    expr.backwardPass(1.0, grad);

    const float eps = 1e-4;

    auto res = expr.value();
    ASSERT_TRUE(fabs(res - 3.0744425168116591) < eps);

    ASSERT_TRUE(fabs(grad[0] - 0.24751657271) < eps);
    ASSERT_TRUE(fabs(grad[1] - 0.24937604019) < eps);
    ASSERT_TRUE(fabs(grad[2] - 0.25) < eps);
    ASSERT_TRUE(fabs(grad[3] - 0.24937604019) < eps);
    ASSERT_TRUE(fabs(grad[4] - 0.24751657271) < eps);
    ASSERT_TRUE(fabs(grad[5] - 0.24445831169) < eps);

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