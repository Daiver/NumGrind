#include <iostream>

#include "utils.h"
#include "numgrind.h"
#include "solvers/GradientDescentSolver.h"
#include "Eigen/Core"


float sigmoid(float z)
{
    return (float)(1.0f/(1.0f + exp(-z)));
}

float sigmoidDer(float z)
{
    return sigmoid(z) * sigmoid(1.0f - z);
}

void logisticRegressionOperatorAndExample02()
{
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
    auto b = gm.variable(0);
    auto f = apply<sigmoid, sigmoidDer>(matmult(X, w) + b);
    auto residual = f - y;
    auto err = dot(residual, residual);

    auto vars = gm.initializeVariables();

    SolverSettings settings;
    settings.nMaxIterations = 20;
    solvers::gradientDescent(settings, 0.1, [&](const Eigen::VectorXf &vars) {
                                 err.node()->forwardPass(vars);
                                 return err.node()->value();
                             },
                             [&](const Eigen::VectorXf &vars, Eigen::VectorXf &grad) {
                                 err.node()->forwardPass(vars);
                                 err.node()->backwardPass(1.0, grad);
                             }, vars);
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
    using namespace NumGrind;
    using namespace NumGrind::SymbolicNodeOps;
    GraphManager gm;

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

    auto W1 = gm.variable(utils::gaussf(2, 2, 0.0, 0.5, generator));
    auto b1 = gm.variable(utils::gaussf(1, 2, 0.0, 0.5, generator));
    auto W2 = gm.variable(utils::gaussf(2, 1, 0.0, 0.01, generator));
    auto b2 = gm.variable(utils::gaussf(0.0f, 0.01f, generator));
    auto f1 = apply<sigmoid, sigmoidDer>(matmult(X, W1) + b1);
    auto f2 = apply<sigmoid, sigmoidDer>(matmult(f1, W2) + b2);
    auto residual = f2 - y;
    auto err = dot(residual, residual);

    auto vars = gm.initializeVariables();
    auto grad = gm.initializeGradient(vars);

    SolverSettings settings;
    settings.nMaxIterations = 40;
    settings.minDErr = 1e-5;
    solvers::gradientDescent(settings, 2.0, [&](const Eigen::VectorXf &vars) {
                                 err.node()->forwardPass(vars);
                                 return err.node()->value();
                             },
                             [&](const Eigen::VectorXf &vars, Eigen::VectorXf &grad) {
                                 err.node()->forwardPass(vars);
                                 err.node()->backwardPass(1.0, grad);
                             }, vars);
    f2.node()->forwardPass(vars);
    std::cout << "Function result" << std::endl;
    std::cout << f2.value() << std::endl;
    std::cout << "W1:" << std::endl;
    std::cout << W1.value() << std::endl;

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

