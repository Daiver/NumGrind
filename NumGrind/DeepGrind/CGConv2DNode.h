#ifndef DEEPGRIND_CGCONVOLUTIONNODE_H
#define DEEPGRIND_CGCONVOLUTIONNODE_H

#include "Utils.h"
#include "Conv2DFilterShape.h"
#include "CompGraph/CGTensorOutput.h"

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

            virtual const Eigen::MatrixXf &value() const override { return mValue; }

        private:
            DeepGrind::Conv2DFilterShape filterParams;
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
