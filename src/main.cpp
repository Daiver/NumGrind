#include <iostream>

#include "utils.h"

#include "numgrind.h"
#include "GradientDescentSolver.h"
#include "Eigen/Core"

float sigmoid(float z)
{
    return (float)(1.0f/(1.0f + exp(-z)));
}

float sigmoidDer(float z)
{
    return sigmoid(z) * sigmoid(1.0f - z);
}

void logisticRegressionExample()
{
    Eigen::MatrixXf data(4, 2);
    Eigen::VectorXf targets(4);
    data << 0, 0,
            0, 1,
            1, 0,
            1, 1;
    targets << 0, 0, 0, 1;

    auto X = GNMatrixConstant(data);
    auto y = GNMatrixConstant(targets);

    auto w = GNMatrixVariable(2, 1, {0, 1});
    auto b = GNScalarVariable(2);
    auto f1 = GNMatrixProduct(&X, &w);
    auto f2 = GNMatrixScalarSum(&f1, &b);
    auto f3 = GNMatrixMapUnaryFunction<float, sigmoid, sigmoidDer>(&f2);
    auto f4 = GNMatrixSub(&f3, &y);
    auto f  = GNDotProduct(&f4, &f4);

    Eigen::VectorXf vars = Eigen::VectorXf::Zero(3);

    f.forwardPass(vars);
    std::cout << "Err " << f.value() << std::endl;
    solvers::gradientDescent(10, 0.25, f, vars);
    std::cout << vars << std::endl;

}

int main() {
    logisticRegressionExample();
//    std::cout << "Hello, World!" << std::endl;
//
//    auto n1 = GNVectorVariable({0, 1});
//    auto n2 = GNVectorVariable({2, 3});
//
//    auto n3 = GNVectorElementWiseProduct(&n1, &n2);
//    auto n4 = GNDotProduct(&n3, &n1);
//
//    auto graph = n4;
//
//    Eigen::VectorXf vars = utils::vec2EVecf({1, 2, 3, 4});
//    Eigen::VectorXf grad = Eigen::VectorXf::Zero(vars.size());
//    graph.forwardPass(vars);
//    graph.backwardPass(1, grad);
//
//    std::cout << graph.toString() << std::endl;
//    for(int i = 0; i < grad.size(); ++i)
//        std::cout << i << ":" << grad[i] << " ";
//    std::cout << std::endl;

    return 0;
}