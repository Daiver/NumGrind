#ifndef NUMGRIND_CONV2DPARAMS_H
#define NUMGRIND_CONV2DPARAMS_H

namespace DeepGrind {
    class Conv2DFilterShape {
    public:
        Conv2DFilterShape();
        Conv2DFilterShape(
            const int xShape,
            const int yShape,
            const int zShape,
            const int nFilters);

        const int xShape()   const { return mXShape; }
        const int yShape()   const { return mYShape; }
        const int zShape()   const { return mZShape; }
        const int nFilters() const { return mNFilters; }

        const int nParams1FilterBiased() const { return xShape() * yShape() * zShape() + 1; }
        const int nParamsBiased()        const { return nParams1FilterBiased() * nFilters(); }

    private:
        const int mXShape   = 0;
        const int mYShape   = 0;
        const int mZShape   = 0;
        const int mNFilters = 0;
    };
}

#endif //NUMGRIND_CONV2DPARAMS_H
