#ifndef NUMGRINDTEST01_GNMATRIXVARIABLE_H
#define NUMGRINDTEST01_GNMATRIXVARIABLE_H

#include <algorithm>
#include "CGTensorOutput.h"

namespace NumGrind {
    namespace CompGraph {
        class CGMatrixVariable : public CGTensorOutput {
        public:

            CGMatrixVariable(const int nRows, const int nCols) :
                    mNRows(nRows), mNCols(nCols), mValue(Eigen::MatrixXf::Zero(nRows, nCols)) {}

            CGMatrixVariable(const int nRows, const int nCols, const std::vector<int> &indices) :
                    mNRows(nRows), mNCols(nCols), indices(indices), mValue(Eigen::MatrixXf::Zero(nRows, nCols)) {
                assert(nRows * nCols == indices.size());
            }

            void forwardPass(const Eigen::VectorXf &vars) {
                for (int i = 0; i < indices.size(); ++i) {
                    auto inds2D = flatIndTo2DInd(i);
                    this->mValue(inds2D.first, inds2D.second) = vars[indices[i]];
                }
            }

            virtual void backwardPass(const Eigen::MatrixXf &sensitivity, Eigen::VectorXf &grad) override {
                assert(sensitivity.rows() == mNRows);
                assert(sensitivity.cols() == mNCols);
                for (int i = 0; i < indices.size(); ++i) {
                    auto inds2D = flatIndTo2DInd(i);
                    grad[indices[i]] += sensitivity(inds2D.first, inds2D.second);
                }
            }

            void setIndices(const std::vector<int> &indices) { this->indices = indices; }

            void setValue(const Eigen::MatrixXf &value);

            int nRows() const { return mNRows; }

            int nCols() const { return mNCols; }

            const Eigen::MatrixXf &value() const { return this->mValue; }

        protected:
            std::pair<int, int> flatIndTo2DInd(const int flatInd) {
                const int row = flatInd / mNCols;
                const int col = flatInd % mNCols;
                return std::make_pair(row, col);
            }

        private:
            int mNRows;
            int mNCols;
            std::vector<int> indices;
            Eigen::MatrixXf mValue;
        };

    }
}

#endif //NUMGRINDTEST01_GNMATRIXVARIABLE_H

