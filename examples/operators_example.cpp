
#include <iostream>
#include "utils.h"
#include "numgrind.h"
#include "Solvers/GradientDescentSolver.h"
#include "Solvers/SGDSolver.h"
#include "Solvers/checkgradient.h"
#include "mnist.h"

float sigmoid(float z)
{
    return (float) (1.0f / (1.0f + exp(-z)));
}

float sigmoidDer(float z)
{
    const float sigZ = sigmoid(z);
    return sigZ * (1.0f - sigZ);
}

void logisticRegressionOperatorAndExample02() {
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
    auto f = apply<sigmoid, sigmoidDer>(matmult(X, w) + b);
    auto residual = f - y;
    auto err = dot(residual, residual);

    auto vars = gm.initializeVariables();

    Solvers::SolverSettings settings;
    settings.nMaxIterations = 20;
    Solvers::gradientDescent(settings, 0.1, gm.funcFromNode(&err), gm.gradFromNode(&err), vars);
    f.node()->forwardPass(vars);
    std::cout << "Function result" << std::endl;
    std::cout << f.value() << std::endl;
    std::cout << "W:" << std::endl;
    std::cout << w.value() << std::endl;
    std::cout << "b:" << std::endl;
    std::cout << b.value() << std::endl;
}

void mlpOperatorOrExample01() {
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
    auto f1 = apply<sigmoid, sigmoidDer>(matmult(X, W1) + b1);
    auto f2 = apply<sigmoid, sigmoidDer>(matmult(f1, W2) + b2);
    auto residual = f2 - y;
    auto err = dot(residual, residual);

    auto vars = gm.initializeVariables();

    NumGrind::Solvers::SolverSettings settings;
    settings.nMaxIterations = 40;

    std::cout << "is gradient ok? "
              << NumGrind::Solvers::isGradientOk(gm.funcFromNode(&err), gm.gradFromNode(&err), vars) << std::endl;

    NumGrind::Solvers::gradientDescent(settings, 2.0, gm.funcFromNode(&err), gm.gradFromNode(&err), vars);
    f2.node()->forwardPass(vars);
    std::cout << "Function result" << std::endl << f2.value() << std::endl;
    std::cout << "W1:" << std::endl << W1.value() << std::endl;
    std::cout << "W2:" << std::endl << W2.value() << std::endl;
    std::cout << "b2:" << std::endl << b2.value() << std::endl;
}


void mlpOperatorOrExample02() {
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
    auto f1 = apply<sigmoid, sigmoidDer>(matmult(X, W1) + b1);
    auto f2 = apply<sigmoid, sigmoidDer>(matmult(f1, W2) + b2);
    auto err = sumOfSquares(f2 - y);

    auto vars = gm.initializeVariables();

    NumGrind::Solvers::SolverSettings settings;
    settings.nMaxIterations = 500;
    NumGrind::Solvers::SGDSolver solver(settings, 1.0, vars);

    std::cout << "is gradient ok? "
              << NumGrind::Solvers::isGradientOk(gm.funcFromNode(&err), gm.gradFromNode(&err), vars)
              << std::endl;

    const int nIters = 100;
    std::uniform_int_distribution<int> dist(0, data.rows() - 1);
    for (int iter = 0; iter < nIters; ++iter) {
        const int index = dist(generator);
        const Eigen::MatrixXf sample = data.row(index);
        const Eigen::MatrixXf label = targets.row(index);
        X.setValue(sample);
        y.setValue(label);
        solver.makeStep(gm.funcFromNode(&err), gm.gradFromNode(&err));
    }

    X.setValue(data);
    y.setValue(targets);
    f2.node()->forwardPass(solver.vars());
    std::cout << "Function result" << std::endl << f2.value() << std::endl;
    std::cout << "W1:" << std::endl << W1.value() << std::endl;
    std::cout << "W2:" << std::endl << W2.value() << std::endl;
    std::cout << "b2:" << std::endl << b2.value() << std::endl;
}

void mlpOperatorOrAndExample03() {
    using namespace NumGrind::SymbolicGraph;
    NumGrind::GraphManager gm;

    Eigen::MatrixXf data(4, 2);
    Eigen::MatrixXf targets(4, 3);
    data << 0, 0,
            0, 1,
            1, 0,
            1, 1;
    targets << 0, 0, 0,
            1, 0, 1,
            1, 0, 1,
            0, 1, 1;

    std::default_random_engine generator;
    generator.seed(42);

    auto X = gm.constant(data);
    auto y = gm.constant(targets);

    auto W1 = gm.variable(NumGrind::Utils::gaussf(2, 2, 0.0, 0.5, generator));
    auto b1 = gm.variable(NumGrind::Utils::gaussf(1, 2, 0.0, 0.5, generator));
    auto W2 = gm.variable(NumGrind::Utils::gaussf(2, 3, 0.0, 0.01, generator));
    auto b2 = gm.variable(NumGrind::Utils::gaussf(1, 3, 0.0f, 0.01f, generator));
    //auto b2 = gm.variable(NumGrind::Utils::gaussf(0.0f, 0.01f, generator));
    auto f1 = apply<sigmoid, sigmoidDer>(matmult(X, W1) + b1);
    auto f2 = apply<sigmoid, sigmoidDer>(matmult(f1, W2) + b2);
    auto err = sumOfSquares(f2 - y);

    auto vars = gm.initializeVariables();

    NumGrind::Solvers::SolverSettings settings;
    settings.nMaxIterations = 500;
    NumGrind::Solvers::SGDSolver solver(settings, 4.0, vars);

    std::cout << "is gradient ok? "
              << NumGrind::Solvers::isGradientOk(gm.funcFromNode(&err), gm.gradFromNode(&err), vars)
              << std::endl;

    const int nIters = 100;
    std::uniform_int_distribution<int> dist(0, data.rows() - 1);
    for (int iter = 0; iter < nIters; ++iter) {
        const int index = dist(generator);
        const Eigen::MatrixXf sample = data.row(index);
        const Eigen::MatrixXf label = targets.row(index);
        X.setValue(data);
        y.setValue(targets);
        solver.makeStep(gm.funcFromNode(&err), gm.gradFromNode(&err));
    }

    f2.node()->forwardPass(solver.vars());
    std::cout << "Function result" << std::endl << f2.value() << std::endl;
    std::cout << "W1:" << std::endl << W1.value() << std::endl;
    std::cout << "W2:" << std::endl << W2.value() << std::endl;
    std::cout << "b1:" << std::endl << b1.value() << std::endl;
    std::cout << "b2:" << std::endl << b2.value() << std::endl;
}

int main()
{

    return 0;
}
