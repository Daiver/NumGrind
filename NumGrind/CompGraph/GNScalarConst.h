#ifndef NUMGRINDTEST01_GRAPHNODESCALARCONST_H
#define NUMGRINDTEST01_GRAPHNODESCALARCONST_H

#include "GNScalarOutput.h"

namespace NumGrind {
    namespace CompGraph {

        class GNScalarConst : public GNScalarOutput {
        public:
            GNScalarConst(const float value) : mValue(value) {
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
