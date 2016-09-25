
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

