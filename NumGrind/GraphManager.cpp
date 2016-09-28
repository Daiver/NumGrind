#include "GraphManager.h"

#include "CompGraph/CGMatrixConstant.h"

using namespace NumGrind;
using namespace NumGrind::CompGraph;
using namespace NumGrind::SymbolicGraph;

GraphManager::GraphManager() {

}

GraphManager::~GraphManager() {
    for(const CompGraphNode *node : mGraphNodes)
        if(node != nullptr)
            delete node;
}

void GraphManager::addGraphNode(CompGraphNode *node) {
    assert(node != nullptr);
    this->mGraphNodes.push_back(node);
}

SymbolicScalarPlaceholder GraphManager::variable(const float val) {
    CGScalarVariable *node = new CGScalarVariable(-1);
    node->setValue(val);
    this->mScalarVariables.push_back(node);
    this->addGraphNode(node);
    return SymbolicScalarPlaceholder(this, node, true);
}

SymbolicTensorPlaceholder GraphManager::variable(const int nRows, const int nCols, const float val) {
    return variable(Eigen::MatrixXf::Constant(nRows, nCols, val));
}

SymbolicTensorPlaceholder GraphManager::variable(const Eigen::MatrixXf &value) {
    CGMatrixVariable *node = new CGMatrixVariable(value.rows(), value.cols());
    node->setValue(value);
    this->mMatrixVariables.push_back(node);
    this->addGraphNode(node);
    return SymbolicTensorPlaceholder(this, node, true);
}

Eigen::VectorXf GraphManager::initializeGradient(const Eigen::VectorXf &vars) {
    return Eigen::VectorXf::Zero(vars.size());
}


int GraphManager::nVarsForMatrices() const {
    int res = 0;
    for(CGMatrixVariable *var : mMatrixVariables)
        res += var->nRows() * var->nCols();

    return res;
}


Eigen::VectorXf GraphManager::initializeVariables() {
    const int nScalarVars = mScalarVariables.size();
    const int nTensorVars = nVarsForMatrices();
    const int nVariables = nScalarVars + nTensorVars;
    Eigen::VectorXf res = Eigen::VectorXf::Zero(nVariables);

    for(int i = 0; i < nScalarVars; ++i) {
        res[i] = mScalarVariables[i]->value();
        mScalarVariables[i]->setIndex(i);
    }

    int tensorVarsOffset = nScalarVars;
    for(int tvarInd = 0; tvarInd < mMatrixVariables.size(); ++tvarInd){
        const int nRows = mMatrixVariables[tvarInd]->nRows();
        const int nCols = mMatrixVariables[tvarInd]->nCols();
        const int nVarsPerTensor = nRows * nCols;
        std::vector<int> varsIndices;
        varsIndices.reserve(nVarsPerTensor);
        for(int i = 0; i < nRows; ++i){
            for(int j = 0; j < nCols; ++j){
                varsIndices.push_back(tensorVarsOffset + i*nCols + j);
                res[tensorVarsOffset + i*nCols + j] = mMatrixVariables[tvarInd]->value()(i, j);
            }
        }
        mMatrixVariables[tvarInd]->setIndices(varsIndices);
        tensorVarsOffset += nVarsPerTensor;
    }

    return res;
}

SymbolicTensorNode GraphManager::constant(const Eigen::MatrixXf &value) {
    auto node = new CGMatrixConstant(value);
    this->addGraphNode(node);
    return SymbolicTensorNode(this, node);
}

