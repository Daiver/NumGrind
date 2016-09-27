#ifndef NUMGRINDTEST01_GRAPHNODESCALARVARIABLE_H
#define NUMGRINDTEST01_GRAPHNODESCALARVARIABLE_H

#include "GNScalarOutput.h"

namespace NumGrind {
    namespace CompGraph {
        class GNScalarVariable : public GNScalarOutput {
        public:
            GNScalarVariable(const int index) : index(index), mValue(0.0) {
            }

            void forwardPass(const Eigen::VectorXf &vars) {
                assert(index < vars.size());
                this->mValue = vars[index];
            }

            virtual void backwardPass(const float sensitivity, Eigen::VectorXf &grad) override {
                assert(index < grad.size());
                grad[this->index] += sensitivity;
            }

            float value() const { return this->mValue; }

            void setValue(const float value) { this->mValue = value; }

            void setIndex(const int index) { this->index = index; }

        private:
            int index;
            float mValue;
        };
    }
}

#endif //NUMGRINDTEST01_GRAPHNODESCALARVARIABLE_H
