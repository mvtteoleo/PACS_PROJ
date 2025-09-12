#pragma once
#include "C2Decomp.hpp"

template <typename myType>
void C2Decomp<myType>::allocX(myType*& var)
{

    int xsize = decompMain.xsz[0];
    int ysize = decompMain.xsz[1];
    int zsize = decompMain.xsz[2];

    var = new myType[xsize * ysize * zsize];
}

template <typename myType>
void C2Decomp<myType>::allocY(myType*& var)
{

    int xsize = decompMain.ysz[0];
    int ysize = decompMain.ysz[1];
    int zsize = decompMain.ysz[2];

    var = new myType[xsize * ysize * zsize];
}

template <typename myType>
void C2Decomp<myType>::allocZ(myType*& var)
{

    int xsize = decompMain.zsz[0];
    int ysize = decompMain.zsz[1];
    int zsize = decompMain.zsz[2];

    var = new myType[xsize * ysize * zsize];
}

template <typename myType>
void C2Decomp<myType>::deallocXYZ(myType*& var)
{
    delete[] var;
    var = NULL;
}
