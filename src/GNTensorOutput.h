#ifndef NUMGRINDTEST01_GRAPHNODEVECTOROUTPUT_H
#define NUMGRINDTEST01_GRAPHNODEVECTOROUTPUT_H

#include "GraphNode.h"

class GNTensorOutput : public GraphNode
{
public:
    virtual void backwardPass(const Eigen::MatrixXf &sensitivity, Eigen::VectorXf &grad) = 0;
    virtual const Eigen::MatrixXf &value() const = 0;
};


#endif //NUMGRINDTEST01_GRAPHNODEVECTOROUTPUT_H
