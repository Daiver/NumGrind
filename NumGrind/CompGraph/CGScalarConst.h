#ifndef NUMGRINDTEST01_GRAPHNODESCALARCONST_H
#define NUMGRINDTEST01_GRAPHNODESCALARCONST_H

#include "CGScalarOutput.h"

namespace NumGrind {
    namespace CompGraph {

        class CGScalarConst : public CGScalarOutput {
        public:
            CGScalarConst(const float value) : mValue(value) {
            }

            void forwardPass(const Eigen::VectorXf &vars) {}

            float value() const { return this->mValue; }

            virtual void backwardPass(const float sensitivity, Eigen::VectorXf &grad) override {}

        private:
            float mValue;
        };

    }
}
#endif //NUMGRINDTEST01_GRAPHNODESCALARCONST_H
