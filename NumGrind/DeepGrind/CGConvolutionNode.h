#ifndef NUMGRIND_CGCONVOLUTIONNODE_H
#define NUMGRIND_CGCONVOLUTIONNODE_H

#include "Utils.h"
#include "CompGraph/CGTensorOutput.h"

namespace NumGrind {
    namespace CompGraph{
        class CGConvolutionNode : public CGTensorOutput{
        public:
            virtual void forwardPass(const Eigen::VectorXf &vars) override;

            virtual const Eigen::MatrixXf &value() const override { return mValue; }

        private:
            CGTensorOutput *argFilterWeights;
            CGTensorOutput *argValue;
            Eigen::MatrixXf mValue;

        };
    }
}

#endif //NUMGRIND_CGCONVOLUTIONNODE_H
