#include "C2Decomp.hpp"

int C2Decomp::allocX(double*& var)
{

    int xsize = decompMain.xsz[0];
    int ysize = decompMain.xsz[1];
    int zsize = decompMain.xsz[2];

    var = new double[xsize * ysize * zsize];
    return xsize * ysize * zsize;
}

int C2Decomp::allocY(double*& var)
{

    int xsize = decompMain.ysz[0];
    int ysize = decompMain.ysz[1];
    int zsize = decompMain.ysz[2];

    var = new double[xsize * ysize * zsize];
    return xsize * ysize * zsize;
}

std::vector<double> C2Decomp::allocY_()
{
    std::size_t n = static_cast<std::size_t>(decompMain.ysz[0]) *
                    static_cast<std::size_t>(decompMain.ysz[1]) *
                    static_cast<std::size_t>(decompMain.ysz[2]);

    return std::vector<double>(n); // value-initialized to 0.0
}

int C2Decomp::allocZ(double*& var)
{

    int xsize = decompMain.zsz[0];
    int ysize = decompMain.zsz[1];
    int zsize = decompMain.zsz[2];

    var = new double[xsize * ysize * zsize];
    return xsize * ysize * zsize;
}

void C2Decomp::deallocXYZ(double*& var)
{
    delete[] var;
    var = NULL;
}
