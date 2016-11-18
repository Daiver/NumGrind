#include <iostream>
#include <Utils/utils.h>
#include "numgrind.h"
#include "DeepGrind/deepgrind.h"
#include "Solvers/SGDWithMomentumSolver.h"
#include "Solvers/GradientDescentSolver.h"
#include "Utils/eigenimport.h"
#include "Utils/Normalizer.h"

int main() {
    using namespace NumGrind::SymbolicGraph;
    const std::string dataDir = "/home/daiver/coding/NumGrind/examples/data/house_prices/";
    auto trainData = NumGrind::Utils::readMatFromTxt(dataDir + "X_train.txt");
    const auto trainTargets = NumGrind::Utils::readMatFromTxt(dataDir + "y_train.txt");
    auto testData = NumGrind::Utils::readMatFromTxt(dataDir + "X_test.txt");
    const auto testTargets = NumGrind::Utils::readMatFromTxt(dataDir + "y_test.txt");
    std::cout << "n Features " << trainData.cols() << std::endl;
    std::cout << "n Train samples " << trainData.rows() << std::endl;
    std::cout << "n Test samples " << testData.rows() << std::endl;

    NumGrind::Utils::Normalizer normalizer(trainData);
    trainData = normalizer.transform(trainData);
    testData = normalizer.transform(testData);

    NumGrind::GraphManager gm;

    std::default_random_engine generator;

    auto X = gm.constant(trainData);
    auto y = gm.constant(trainTargets);

    const int hiddenLayerSize = 20;

    auto w1 = gm.variable(NumGrind::Utils::gaussf(trainData.cols(), hiddenLayerSize, 0.0, 0.001, generator));
    auto b1 = gm.variable(NumGrind::Utils::gaussf(1, hiddenLayerSize, 0.0, 0.001, generator));
    auto w2 = gm.variable(NumGrind::Utils::gaussf(hiddenLayerSize, 1, 0.0, 0.001, generator));
    auto b2 = gm.variable(NumGrind::Utils::gaussf(0.0, 0.001, generator));
//    auto f1 = apply<DeepGrind::sigmoid, DeepGrind::sigmoidDer>(matmult(X, w1) + b1);
    auto f1 = apply<DeepGrind::relu, DeepGrind::reluDer>(matmult(X, w1) + b1);
    auto f2 = matmult(f1, w2) + b2;

    auto output = f2;
    auto err = sumOfSquares(output - y);

    auto vars = gm.initializeVariables();
    NumGrind::Solvers::SolverSettings settings;
    settings.verbose = false;
    settings.nMaxIterations = 10;
    //NumGrind::Solvers::gradientDescent(settings, 0.000000001, gm.funcFromNode(&err), gm.gradFromNode(&err), vars);
    //NumGrind::Solvers::SGDWithMomentumSolver solver(settings, 0.000000010, 0.9, vars);
    NumGrind::Solvers::SGDWithMomentumSolver solver(settings, 0.00000003, 0.9, vars);
    //NumGrind::Solvers::SGDWithMomentumSolver solver(settings, 0.000000003, 0.9, vars);

    const int batchSize = 32;
    //const int batchSize = 8;

    Eigen::MatrixXf trainDataSamples(batchSize, trainData.cols());
    Eigen::MatrixXf trainLabelsSamples(batchSize, 1);

    float bestErr = 1e10;
    for (int iterInd = 0; iterInd < 200001; ++iterInd) {
        for (int innerIter = 0; innerIter < trainData.rows() / batchSize; ++innerIter) {
            NumGrind::Utils::rowDataTargetRandomSampling<float, float>(
                    trainData, trainTargets, generator, trainDataSamples, trainLabelsSamples);
            X.setValue(trainDataSamples);
            y.setValue(trainLabelsSamples);
            solver.makeStep(gm.funcFromNode(&err), gm.gradFromNode(&err));
        }
        if (iterInd % 100 == 0) {
            X.setValue(testData);
            y.setValue(testTargets);
            err.node()->forwardPass(solver.vars());
            auto res = err.value();
            const float fErr = res / testTargets.rows();
            if (fErr < bestErr) bestErr = fErr;
            X.setValue(trainData);
            y.setValue(trainTargets);
            err.node()->forwardPass(solver.vars());
            auto trainErr = err.value() / trainData.rows();
            std::cout
                    << "Epoch: " << iterInd << ", "
                    << "Train error " << trainErr << ", "
                    << "Test error " << fErr << ", "
                    << "best " << bestErr << " " << std::endl;
        }
    }

    return 0;
}
