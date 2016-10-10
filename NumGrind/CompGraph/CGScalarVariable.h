#ifndef NUMGRINDTEST01_GRAPHNODESCALARVARIABLE_H
#define NUMGRINDTEST01_GRAPHNODESCALARVARIABLE_H

#include "CGScalarOutput.h"

namespace NumGrind {
    namespace CompGraph {
        class CGScalarVariable : public CGScalarOutput {
        public:
            CGScalarVariable(const int index);

            void forwardPass(const Eigen::VectorXf &vars) override;

            virtual void backwardPass(const float sensitivity, Eigen::VectorXf &grad) override;

            float value() const override { return this->mValue; }

            void setValue(const float value) { this->mValue = value; }

            void setIndex(const int index) { this->index = index; }

        private:
            int index;
            float mValue;
        };
    }
}

#endif //NUMGRINDTEST01_GRAPHNODESCALARVARIABLE_H
