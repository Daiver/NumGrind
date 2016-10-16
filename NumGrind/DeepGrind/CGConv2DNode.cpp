#include "CGConv2DNode.h"

NumGrind::CompGraph::CGConv2DNode::CGConv2DNode(const DeepGrind::Conv2DFilterShape &filterParams,
                                                const int xStride,
                                                const int yStride,
                                                const int zeroPadding,
                                                NumGrind::CompGraph::CGTensorOutput *argFilterWeights,
                                                NumGrind::CompGraph::CGTensorOutput *argValue) :
        filterShape(filterParams),
        xStride(xStride),
        yStride(yStride),
        zeroPadding(zeroPadding),
        argFilterWeights(argFilterWeights),
        argValue(argValue) {

}

std::pair<int, int>
NumGrind::CompGraph::CGConv2DNode::resSizeFromShapes(const int xValShape, const int yValShape) {
    // (W−F+2P)/S+1 <- https://cs231n.github.io/convolutional-networks/
    const int xDiff = xValShape - filterShape.xShape() + 2 * this->zeroPadding;
    const int yDiff = yValShape - filterShape.yShape() + 2 * this->zeroPadding;
    assert(xDiff % this->xStride == 0);
    assert(yDiff % this->yStride == 0);
    const int xResShape = xDiff / this->xStride + 1;
    const int yResShape = yDiff / this->yStride + 1;
    return std::make_pair(xResShape, yResShape);
}

void NumGrind::CompGraph::CGConv2DNode::forwardPass(const Eigen::VectorXf &vars) {
    this->argFilterWeights->forwardPass(vars);
    this->argValue->forwardPass(vars);
    auto filter = argFilterWeights->value();
    auto value = argValue->value();
    assert(filter.rows() == this->filterShape.nParamsBiased());
    assert(filter.cols() == 1);

    const int xValShape = value.cols();
    const int yValShape = value.rows();

    const auto resShape = resSizeFromShapes(xValShape, yValShape);
    this->mValue = Eigen::MatrixXf::Zero(resShape.second, resShape.first);
}

void NumGrind::CompGraph::CGConv2DNode::backwardPass(const Eigen::MatrixXf &sensitivity, Eigen::VectorXf &grad) {
    assert(false); //Not implemented yet
}

