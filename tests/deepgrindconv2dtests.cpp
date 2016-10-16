#include <cmath>
#include <iostream>
#include "gtest/gtest.h"
#include "numgrind.h"
#include "DeepGrind/deepgrind.h"

TEST(DeepGrindConv2DTests, testConv2DNParams01) {
    DeepGrind::Conv2DFilterShape params(2, 3, 10, 6);
    const int nParamsFor1Filter = params.nParams1FilterBiased();
    const int nParams = params.nParamsBiased();
    ASSERT_EQ(nParamsFor1Filter, 61);
    ASSERT_FLOAT_EQ(nParams, 61*6);
}

TEST(DeepGrindConv2DTests, testConv2DResShape01) {
    auto params = DeepGrind::Conv2DFilterShape(3, 3, 1, 1);
    auto nodeParams = NumGrind::CompGraph::CGMatrixConstant(Eigen::MatrixXf::Zero(params.nParamsBiased(), 1));
    auto nodeVal = NumGrind::CompGraph::CGMatrixConstant(Eigen::MatrixXf::Zero(7, 7));
    auto nodeConv = NumGrind::CompGraph::CGConv2DNode(params, 1, 1, 0, &nodeParams, &nodeVal);
    auto vars = Eigen::VectorXf::Zero(params.nParamsBiased());
    nodeConv.forwardPass(vars);
    auto res = nodeConv.value();
    ASSERT_EQ(res.rows(), 5);
    ASSERT_EQ(res.cols(), 5);
}

TEST(DeepGrindConv2DTests, testConv2DResShape02) {
    auto params = DeepGrind::Conv2DFilterShape(3, 3, 1, 1);
    auto nodeParams = NumGrind::CompGraph::CGMatrixConstant(Eigen::MatrixXf::Zero(params.nParamsBiased(), 1));
    auto nodeVal = NumGrind::CompGraph::CGMatrixConstant(Eigen::MatrixXf::Zero(7, 9));
    auto nodeConv = NumGrind::CompGraph::CGConv2DNode(params, 1, 1, 0, &nodeParams, &nodeVal);
    auto vars = Eigen::VectorXf::Zero(params.nParamsBiased());
    nodeConv.forwardPass(vars);
    auto res = nodeConv.value();
    ASSERT_EQ(res.rows(), 5);
    ASSERT_EQ(res.cols(), 7);
}

TEST(DeepGrindConv2DTests, testForwardPass01) {

}