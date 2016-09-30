#ifndef NUMGRIND_CGSUMOFSQUARES_H
#define NUMGRIND_CGSUMOFSQUARES_H

#include "CGTensorOutput.h"
#include "CGScalarOutput.h"

namespace NumGrind {
    namespace CompGraph {
        class CGSumOfSquares : public CGScalarOutput{
        public:
            CGSumOfSquares(CGTensorOutput *arg);

            virtual void backwardPass(const float sensitivity, Eigen::VectorXf &grad) override;

            virtual void forwardPass(const Eigen::VectorXf &vars) override;

            virtual float value() const override;

        protected:
            CGTensorOutput *arg;
            float mValue;
        };

    }
}


#endif //NUMGRIND_CGSUMOFSQUARES_H
