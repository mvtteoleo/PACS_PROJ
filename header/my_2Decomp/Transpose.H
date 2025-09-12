#pragma once
#include "C2Decomp.hpp"

template <typename myType>
void C2Decomp<myType>::transposeX2Y_MajorIndex(myType* src, myType* dst)
{

    auto& s1 = decompMain.xsz[0];
    auto& s2 = decompMain.xsz[1];
    auto& s3 = decompMain.xsz[2];

    auto& d1 = decompMain.ysz[0];
    auto& d2 = decompMain.ysz[1];
    auto& d3 = decompMain.ysz[2];

    // Always comes in major order...
    memSplitXY(src, s1, s2, s3, work1_r, dims[0], decompMain.x1dist);

    MPI_Alltoallv(work1_r, decompMain.x1cnts, decompMain.x1disp, myType_MPI, work2_r,
                  decompMain.y1cnts, decompMain.y1disp, myType_MPI, DECOMP_2D_COMM_COL);

    memMergeXY_YMajor(work2_r, d1, d2, d3, dst, dims[0], decompMain.y1dist);
}

template <typename myType>
void C2Decomp<myType>::transposeY2X_MajorIndex(myType* src, myType* dst)
{

    auto& s1 = decompMain.ysz[0];
    auto& s2 = decompMain.ysz[1];
    auto& s3 = decompMain.ysz[2];

    auto& d1 = decompMain.xsz[0];
    auto& d2 = decompMain.xsz[1];
    auto& d3 = decompMain.xsz[2];

    memSplitYX_YMajor(src, s1, s2, s3, work1_r, dims[0], decompMain.y1dist);

    MPI_Alltoallv(work1_r, decompMain.y1cnts, decompMain.y1disp, myType_MPI, work2_r,
                  decompMain.x1cnts, decompMain.x1disp, myType_MPI, DECOMP_2D_COMM_COL);

    // X pencil is already in index major format
    memMergeYX(work2_r, d1, d2, d3, dst, dims[0], decompMain.x1dist);
}

template <typename myType>
void C2Decomp<myType>::transposeY2Z_MajorIndex(myType* src, myType* dst)
{

    auto& s1 = decompMain.ysz[0];
    auto& s2 = decompMain.ysz[1];
    auto& s3 = decompMain.ysz[2];

    auto& d1 = decompMain.zsz[0];
    auto& d2 = decompMain.zsz[1];
    auto& d3 = decompMain.zsz[2];

    memSplitYZ_YMajor(src, s1, s2, s3, work1_r, dims[1], decompMain.y2dist);

    MPI_Alltoallv(work1_r, decompMain.y2cnts, decompMain.y2disp, myType_MPI, work2_r,
                  decompMain.z2cnts, decompMain.z2disp, myType_MPI, DECOMP_2D_COMM_ROW);

    // Just do the transpose here...
    for (int kp = 0; kp < d3; ++kp)
        for (int jp = 0; jp < d2; ++jp)
            for (int ip = 0; ip < d1; ++ip)
            {
                int ii   = kp * d2 * d1 + jp * d1 + ip;
                int iip  = jp * d3 * d1 + ip * d3 + kp;
                dst[iip] = work2_r[ii];
            }
}

template <typename myType>
void C2Decomp<myType>::transposeZ2Y_MajorIndex(myType* src, myType* dst)
{

    auto& s1 = decompMain.zsz[0];
    auto& s2 = decompMain.zsz[1];
    auto& s3 = decompMain.zsz[2];

    auto& d1 = decompMain.ysz[0];
    auto& d2 = decompMain.ysz[1];
    auto& d3 = decompMain.ysz[2];

    // Just do the transpose here...
    for (int kp = 0; kp < s3; ++kp)
        for (int jp = 0; jp < s2; ++jp)
            for (int ip = 0; ip < s1; ++ip)
            {
                int ii      = kp * s2 * s1 + jp * s1 + ip;
                int iip     = jp * s3 * s1 + ip * s3 + kp;
                work1_r[ii] = src[iip];
            }

    MPI_Alltoallv(work1_r, decompMain.z2cnts, decompMain.z2disp, myType_MPI, work2_r,
                  decompMain.y2cnts, decompMain.y2disp, myType_MPI, DECOMP_2D_COMM_ROW);

    memMergeZY_YMajor(work2_r, d1, d2, d3, dst, dims[1], decompMain.y2dist);
}
