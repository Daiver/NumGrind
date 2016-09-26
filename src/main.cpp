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

void logisticRegressionOperatorAndExample01()
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
    solvers::gradientDescent(10, 0.2, f, vars);
    std::cout << vars << std::endl;

}

void logisticRegressionOperatorAndExample02()
{
    using namespace SymbolicNodeOps;

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
    auto f = apply<sigmoid, sigmoidDer>(matmult(X, w) + b);
    auto residual = f - y;
    auto err = dot(residual, residual);

    auto vars = gm.initializeVariables();
    auto grad = gm.initializeGradient(vars);

    solvers::gradientDescent(20, 0.1, *err.node(), vars);
    f.node()->forwardPass(vars);
    std::cout << "Function result" << std::endl;
    std::cout << f.value() << std::endl;
    std::cout << "W:" << std::endl;
    std::cout << w.value() << std::endl;
    std::cout << "b:" << std::endl;
    std::cout << b.value() << std::endl;
}

void mlpOperatorOrExample01()
{
    using namespace SymbolicNodeOps;

    GraphManager gm;

    Eigen::MatrixXf data(4, 3);
    Eigen::VectorXf targets(4);
    data << 0, 0, 1,
            0, 1, 1,
            1, 0, 1,
            1, 1, 1;
    targets << 0, 1, 1, 0;

    //NumGrind currently have no random initializators
    Eigen::MatrixXf w1Init(3, 3);
    w1Init << 0.01, 0.02, -0.01,
              0.03, 0.04, -0.02,
              -0.03, 0.005, 0.06;
    w1Init *= 10;
//    Eigen::MatrixXf b1Init(3, 1);
//    b1Init << -0.03, -0.01, 0.05;

    Eigen::MatrixXf w2Init(3, 1);
    w2Init << -0.01, 0.03, 0.04;

    auto X = gm.constant(data);
    auto y = gm.constant(targets);
    auto W1 = gm.variable(w1Init);
//    auto b1 = gm.variable(b1Init);
    auto W2 = gm.variable(w2Init);
    auto b2 = gm.variable(0.002);
    auto f1 = apply<sigmoid, sigmoidDer>(matmult(X, W1));
    auto f2 = apply<sigmoid, sigmoidDer>(matmult(f1, W2) + b2);
    auto residual = f2 - y;
    auto err = dot(residual, residual);

    auto vars = gm.initializeVariables();
    auto grad = gm.initializeGradient(vars);

    solvers::gradientDescent(100, 0.5, *err.node(), vars);
    f2.node()->forwardPass(vars);
    std::cout << "Function result" << std::endl;
    std::cout << f2.value() << std::endl;
    std::cout << "W1:" << std::endl;
    std::cout << W1.value() << std::endl;
//    std::cout << "b1:" << std::endl;
//    std::cout << b1.value() << std::endl;

    std::cout << "W2:" << std::endl;
    std::cout << W2.value() << std::endl;
    std::cout << "b2:" << std::endl;
    std::cout << b2.value() << std::endl;
}

int main() {
//    logisticRegressionOperatorAndExample01();
//    logisticRegressionOperatorAndExample02();
    mlpOperatorOrExample01();

    return 0;
}
