#ifndef NUMGRINDTEST01_GRAPHNODEFUNCTION_H
#define NUMGRINDTEST01_GRAPHNODEFUNCTION_H

#include <assert.h>

#include "GNScalarOutput.h"

class GNScalarFunction : public GNScalarOutput
{
public:
    float value() const { return this->mValue; }
protected:
    std::vector<GNScalarOutput *> arguments;
    float mValue;
};



#endif //NUMGRINDTEST01_GRAPHNODEFUNCTION_H
