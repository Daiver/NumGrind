#ifndef NUMGRINDTEST01_GRAPHNODEVECTORVARIABLE_H
#define NUMGRINDTEST01_GRAPHNODEVECTORVARIABLE_H

#include <vector>
#include "CGTensorOutput.h"

namespace NumGrind {
    namespace CompGraph {
        class CGVectorVariable : public CGTensorOutput {
        public:
            CGVectorVariable(const std::vector<int> &indices);

            void forwardPass(const Eigen::VectorXf &vars) override;

            virtual void backwardPass(const Eigen::MatrixXf &sensitivity, Eigen::VectorXf &grad) override;

            const Eigen::MatrixXf &value() const override { return this->mValue; }

        private:
            const std::vector<int> indices;
            Eigen::MatrixXf mValue;
        };
    }
}

#endif //NUMGRINDTEST01_GRAPHNODEVECTORVARIABLE_H
