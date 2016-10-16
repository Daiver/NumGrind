#ifndef DEEPGRIND_CGCONVOLUTIONNODE_H
#define DEEPGRIND_CGCONVOLUTIONNODE_H

#include "Utils.h"
#include "Conv2DParams.h"
#include "CompGraph/CGTensorOutput.h"

namespace NumGrind {
    namespace CompGraph{
        class CGConv2DNode : public CGTensorOutput{
        public:
            virtual void forwardPass(const Eigen::VectorXf &vars) override;

            virtual const Eigen::MatrixXf &value() const override { return mValue; }

        private:
            DeepGrind::Conv2DParams filterParams;
            CGTensorOutput *argFilterWeights;
            CGTensorOutput *argValue;
            Eigen::MatrixXf mValue;

        };
    }
}

#endif //NUMGRIND_CGCONVOLUTIONNODE_H
