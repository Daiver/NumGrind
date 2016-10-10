#ifndef NUMGRINDTEST01_GRAPHNODEVECTORVARIABLE_H
#define NUMGRINDTEST01_GRAPHNODEVECTORVARIABLE_H

#include <vector>
#include "CGTensorOutput.h"

namespace NumGrind {
    namespace CompGraph {
        class CGVectorVariable : public CGTensorOutput {
        public:
            CGVectorVariable(const std::vector<int> &indices) : indices(indices),
                                                                mValue(Eigen::VectorXf::Zero(indices.size())) {
            }

            void forwardPass(const Eigen::VectorXf &vars) override {
                assert(this->mValue.cols() == 1);
                for (int i = 0; i < indices.size(); ++i)
                    this->mValue(i, 0) = vars[indices[i]];
            }

            virtual void backwardPass(const Eigen::MatrixXf &sensitivity, Eigen::VectorXf &grad) override {
                assert(sensitivity.cols() == 1);
                for (int i = 0; i < indices.size(); ++i)
                    grad[indices[i]] += sensitivity(i, 0);
            }

            const Eigen::MatrixXf &value() const { return this->mValue; }

        private:
            const std::vector<int> indices;
            Eigen::MatrixXf mValue;
        };
    }
}

#endif //NUMGRINDTEST01_GRAPHNODEVECTORVARIABLE_H
