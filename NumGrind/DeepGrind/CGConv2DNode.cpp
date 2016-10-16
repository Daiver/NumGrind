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
    assert(filterShape.zShape() == 1);
    assert(filterShape.nFilters() == 1);

    const int xValShape = value.cols();
    const int yValShape = value.rows();

    const auto resShape = resSizeFromShapes(xValShape, yValShape);
    const int xResShape = resShape.first;
    const int yResShape = resShape.second;
    this->mValue = Eigen::MatrixXf::Zero(yResShape, xResShape);

    //TODO: just rewrite it
//    const int xOffset = (zeroPadding == 0) ? (xValShape - xResShape) : 0;
//    const int yOffset = (zeroPadding == 0) ? (yValShape - yResShape) : 0;
//    for(int iRes = 0; iRes < yResShape; ++iRes){
//        for(int jRes = 0; jRes < xResShape; ++jRes){
//            for(int iFilter = 0; iFilter < filterShape.yShape(); iFilter++){
//                for(int jFilter = 0; jFilter < filterShape.xShape(); jFilter++){
//                    const int realFilterWidth  = filterShape.xShape() * xStride;
//                    const int realFilterHeight = filterShape.yShape() * yStride;
//                    int iToTake = iRes - (realFilterHeight / 2) + iFilter + xOffset;
//                    int jToTake = jRes - (realFilterWidth  / 2) + jFilter + yOffset;
//                    iToTake = (iToTake < 0) ? 0 : iToTake;
//                    iToTake = (iToTake >= yResShape) ? (yResShape - 1) : iToTake;
//                    jToTake = (jToTake < 0) ? 0 : jToTake;
//                    jToTake = (jToTake >= xResShape) ? (xResShape - 1) : jToTake;
//                    const float pixelVal  = value(iToTake, jToTake);
//                    const float filterVal = filter(iFilter, jFilter);
//                    mValue(iRes, jRes) += pixelVal * filterVal;
//                }
//            }
//        }
//    }
}

void NumGrind::CompGraph::CGConv2DNode::backwardPass(const Eigen::MatrixXf &sensitivity, Eigen::VectorXf &grad) {
    assert(false); //Not implemented yet
}

