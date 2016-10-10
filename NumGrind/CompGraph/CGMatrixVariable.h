#ifndef NUMGRINDTEST01_GNMATRIXVARIABLE_H
#define NUMGRINDTEST01_GNMATRIXVARIABLE_H

#include <algorithm>
#include "CGTensorOutput.h"

namespace NumGrind {
    namespace CompGraph {
        class CGMatrixVariable : public CGTensorOutput {
        public:

            CGMatrixVariable(const int nRows, const int nCols);

            CGMatrixVariable(const int nRows, const int nCols, const std::vector<int> &indices);

            void forwardPass(const Eigen::VectorXf &vars) override;

            virtual void backwardPass(const Eigen::MatrixXf &sensitivity, Eigen::VectorXf &grad) override;

            void setIndices(const std::vector<int> &indices) { this->indices = indices; }

            void setValue(const Eigen::MatrixXf &value);

            int nRows() const { return mNRows; }

            int nCols() const { return mNCols; }

            const Eigen::MatrixXf &value() const override { return this->mValue; }

        protected:
            std::pair<int, int> flatIndTo2DInd(const int flatInd);

        private:
            int mNRows;
            int mNCols;
            std::vector<int> indices;
            Eigen::MatrixXf mValue;
        };

    }
}

#endif //NUMGRINDTEST01_GNMATRIXVARIABLE_H

