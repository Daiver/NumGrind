#ifndef NUMGRIND_GNSCALARSUB_H
#define NUMGRIND_GNSCALARSUB_H

#include "CGScalarOutput.h"

namespace NumGrind {
    namespace CompGraph {
        class CGScalarSub : public CGScalarOutput {
        public:
            CGScalarSub(CGScalarOutput *arg1, CGScalarOutput *arg2);

            virtual void forwardPass(const Eigen::VectorXf &vars) override;

            virtual void backwardPass(const float sensitivity, Eigen::VectorXf &grad) override;

            virtual float value() const override {
                return mValue;
            }

        private:
            float mValue;
            CGScalarOutput *arg1;
            CGScalarOutput *arg2;
        };

    }
}

#endif //NUMGRIND_GNSCALARSUB_H
