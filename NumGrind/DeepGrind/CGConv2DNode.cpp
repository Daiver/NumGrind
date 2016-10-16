#include "CGConv2DNode.h"

NumGrind::CompGraph::CGConv2DNode::CGConv2DNode(const DeepGrind::Conv2DFilterShape &filterParams,
                                                const int xStride,
                                                const int yStride,
                                                const int zeroPadding,
                                                NumGrind::CompGraph::CGTensorOutput *argFilterWeights,
                                                NumGrind::CompGraph::CGTensorOutput *argValue) :
        filterParams(filterParams),
        xStride(xStride),
        yStride(yStride),
        zeroPadding(zeroPadding),
        argFilterWeights(argFilterWeights),
        argValue(argValue) {

}

void NumGrind::CompGraph::CGConv2DNode::forwardPass(const Eigen::VectorXf &vars) {
    this->argFilterWeights->forwardPass(vars);
    this->argValue->forwardPass(vars);

}

