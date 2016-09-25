
#include "GraphManager.h"


GraphManager::GraphManager() {

}

GraphManager::~GraphManager() {
    for(const GraphNode *node : mGraphNodes)
        if(node != nullptr)
            delete node;
}

void GraphManager::addGraphNode(GraphNode *node) {
    assert(node != nullptr);
    this->mGraphNodes.push_back(node);
}

SymbolicScalarPlaceholder GraphManager::variable(const float val) {
    GNScalarVariable *node = new GNScalarVariable(-1);
    node->setValue(val);
    this->mScalarVariables.push_back(node);
    this->addGraphNode(node);
    return SymbolicScalarPlaceholder(this, node, true);
}

Eigen::VectorXf GraphManager::initializeVariables() {
    const int nScalarVars = mScalarVariables.size();
    const int nVariables = nScalarVars;
    Eigen::VectorXf res = Eigen::VectorXf::Zero(nVariables);
    for(int i = 0; i < nScalarVars; ++i) {
        res[i] = mScalarVariables[i]->value();
        mScalarVariables[i]->setIndex(i);
    }
    return res;
}

Eigen::VectorXf GraphManager::initializeGradient(const Eigen::VectorXf &vars) {
    return Eigen::VectorXf::Zero(vars.size());
}

