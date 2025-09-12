#ifndef _2DECOMPCH_
#define _2DECOMPCH_

#include "MPI_types.hpp"
#include "math.h"
#include < mpi.h >
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory.h>
#include <string>

template <typename myType>
class C2Decomp
{

  public:
    // Just assume that we're using double precision all the time
    // MPI_Datatype myType_MPI = MPI_DOUBLE; // MpiTypeMap<myType>::type;
    auto myType_MPI = mpi_get_type<myType>()  ;

    int myTypeBytes{0};

    // Global Size
    int nxGlobal, nyGlobal, nzGlobal;

    // MPI rank info
    int nRank, nProc;

  public:
    // parameters for 2D Cartesian Topology
    int dims[2], coord[2];
    int periodic[2];

  public:
    MPI_Comm DECOMP_2D_COMM_CART_X = MPI_COMM_NULL, DECOMP_2D_COMM_CART_Y = MPI_COMM_NULL,
             DECOMP_2D_COMM_CART_Z = MPI_COMM_NULL;
    MPI_Comm DECOMP_2D_COMM_ROW = MPI_COMM_NULL, DECOMP_2D_COMM_COL = MPI_COMM_NULL;

  private:
    // Defining neighboring blocks
    int neighbor[3][6];
    // Flags for periodic condition in 3D
    bool periodicX, periodicY, periodicZ;

  public:
    // Struct used to store decomposition info for a given global data size
    typedef struct decompinfo
    {
        int xst[3], xen[3], xsz[3];
        int yst[3], yen[3], ysz[3];
        int zst[3], zen[3], zsz[3];

        int *x1dist, *y1dist, *y2dist, *z2dist;
        int *x1cnts, *y1cnts, *y2cnts, *z2cnts;
        int *x1disp, *y1disp, *y2disp, *z2disp;

        int x1count, y1count, y2count, z2count;

        bool even;
    } DecompInfo;

  public:
    // main default decomposition information for global size nx*ny*nz
    DecompInfo decompMain;
    int        decompBufSize;

  public:
    // Starting/ending index and size of data held by the current processor
    // duplicate 'decompMain', needed by apps to define data structure
    int xStart[3], xEnd[3], xSize[3]; // x-pencil
    int yStart[3], yEnd[3], ySize[3]; // y-pencil
    int zStart[3], zEnd[3], zSize[3]; // z-pencil

  private:
    // These are the buffers used by MPI_ALLTOALL(V) calls
    myType* work1_r;
    myType* work2_r; // Only implementing real for now...

  public:
    C2Decomp(int nx, int ny, int nz, int pRow, int pCol, bool periodicBC[3])
    {

        nxGlobal = nx;
        nyGlobal = ny;
        nzGlobal = nz;

        periodicX = periodicBC[0];
        periodicY = periodicBC[1];
        periodicZ = periodicBC[2];

        decompBufSize = 0;
        work1_r       = NULL;
        work2_r       = NULL;

        decomp2DInit(pRow, pCol);
    }

    void decomp2DInit(int pRow, int pCol);

    void best2DGrid(int nProc, int& pRow, int& pCol);
    void FindFactor(int num, int* factors, int& nfact);

    void decomp2DFinalize();
    // Get Transposes but with array indexing with the major index of the pencil...
    void transposeX2Y_MajorIndex(myType* src, myType* dst);
    void transposeY2Z_MajorIndex(myType* src, myType* dst);
    void transposeZ2Y_MajorIndex(myType* src, myType* dst);
    void transposeY2X_MajorIndex(myType* src, myType* dst);

    void decompInfoInit();
    void decompInfoFinalize();

    // only doing real
    void allocX(myType*& var);
    void allocY(myType*& var);
    void allocZ(myType*& var);
    void deallocXYZ(myType*& var);

    void decomp2DAbort(int errorCode, std::string msg);
    void initNeighbor();
    void getDist();
    void distribute(int data1, int proc, int* st, int* en, int* sz);
    void partition(int nx, int ny, int nz, int* pdim, int* lstart, int* lend, int* lsize);
    void prepareBuffer(DecompInfo* dii);

    void getDecompInfo(DecompInfo dcompinfo_in);

    void memSplitXY(myType* in, int n1, int n2, int n3, myType* out, int iproc, int* dist);

    void memMergeXY_YMajor(myType* in, int n1, int n2, int n3, myType* out, int iproc, int* dist);

    void memSplitYZ_YMajor(myType* in, int n1, int n2, int n3, myType* out, int iproc, int* dist);

    void memMergeZY_YMajor(myType* in, int n1, int n2, int n3, myType* out, int iproc, int* dist);

    void memSplitYX_YMajor(myType* in, int n1, int n2, int n3, myType* out, int iproc, int* dist);

    void memMergeYX(myType* in, int n1, int n2, int n3, myType* out, int iproc, int* dist);
};

#include "Alloc.H"
#include "Best2DGrid.H"
#include "C2Decomp.H"
#include "MemSplitMerge.H"
#include "Transpose.H"

#endif
