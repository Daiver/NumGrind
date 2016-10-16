#ifndef DEEPGRIND_CGCONVOLUTIONNODE_H
#define DEEPGRIND_CGCONVOLUTIONNODE_H

#include "Utils.h"
#include "Conv2DFilterShape.h"
#include "CompGraph/CGTensorOutput.h"

//Just because i don't understand Tensors
namespace NumGrind {
    namespace CompGraph{
        class CGConv2DNode : public CGTensorOutput{
        public:
            CGConv2DNode(
                    const DeepGrind::Conv2DFilterShape &filterParams,
                    const int xStride,
                    const int yStride,
                    const int zeroPadding,
                    CGTensorOutput *argFilterWeights,
                    CGTensorOutput *argValue);

            virtual void forwardPass(const Eigen::VectorXf &vars) override;

            virtual void backwardPass(const Eigen::MatrixXf &sensitivity, Eigen::VectorXf &grad) override;

            virtual const Eigen::MatrixXf &value() const override { return mValue; }

            std::pair<int, int> resSizeFromShapes(const int xValShape, const int yValShape);

        private:
            DeepGrind::Conv2DFilterShape filterShape;
            int xStride;
            int yStride;
            int zeroPadding;
            CGTensorOutput *argFilterWeights;
            CGTensorOutput *argValue;
            Eigen::MatrixXf mValue;

        };
    }
}

#endif //NUMGRIND_CGCONVOLUTIONNODE_H
