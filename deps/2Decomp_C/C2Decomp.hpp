#ifndef _2DECOMPCH_
#define _2DECOMPCH_

#include "math.h"
#include "mpi.h"
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory.h>
#include <string>
using std::string, std::cout, std::min, std::max, std::endl;

class C2Decomp
{

  public:
    // Just assume that we're using double precision all the time
    typedef double myType;
    MPI_Datatype   realType;
    int            myTypeBytes;

    // Global Size
    int nxGlobal, nyGlobal, nzGlobal;

    // MPI rank info
    int nRank, nProc;

  public:
    // parameters for 2D Cartesian Topology
    int dims[2], coord[2];
    int periodic[2];
    // Defining neighboring blocks
    int neighbor[3][6];

  public:
    MPI_Comm DECOMP_2D_COMM_CART_X, DECOMP_2D_COMM_CART_Y, DECOMP_2D_COMM_CART_Z;
    MPI_Comm DECOMP_2D_COMM_ROW, DECOMP_2D_COMM_COL;

  private:
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
    double *work1_r, *work2_r; // Only implementing real for now...

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

        realType = MPI_DOUBLE_PRECISION;

        decomp2DInit(pRow, pCol);
    }

    void decomp2DInit(int pRow, int pCol);

    void best2DGrid(int nProc, int& pRow, int& pCol);
    void FindFactor(int num, int* factors, int& nfact);

    void decomp2DFinalize()
    {
        // std::cout << "Freeing C2Decop structures and buffers\n";
        // --- Free main decomp arrays (if allocated) ---
        // x1dist, y1dist, y2dist, z2dist
        if (decompMain.x1dist)
        {
            delete[] decompMain.x1dist;
            decompMain.x1dist = nullptr;
        }
        if (decompMain.y1dist)
        {
            delete[] decompMain.y1dist;
            decompMain.y1dist = nullptr;
        }
        if (decompMain.y2dist)
        {
            delete[] decompMain.y2dist;
            decompMain.y2dist = nullptr;
        }
        if (decompMain.z2dist)
        {
            delete[] decompMain.z2dist;
            decompMain.z2dist = nullptr;
        }

        // x1cnts, y1cnts, y2cnts, z2cnts
        if (decompMain.x1cnts)
        {
            delete[] decompMain.x1cnts;
            decompMain.x1cnts = nullptr;
        }
        if (decompMain.y1cnts)
        {
            delete[] decompMain.y1cnts;
            decompMain.y1cnts = nullptr;
        }
        if (decompMain.y2cnts)
        {
            delete[] decompMain.y2cnts;
            decompMain.y2cnts = nullptr;
        }
        if (decompMain.z2cnts)
        {
            delete[] decompMain.z2cnts;
            decompMain.z2cnts = nullptr;
        }

        // x1disp, y1disp, y2disp, z2disp
        if (decompMain.x1disp)
        {
            delete[] decompMain.x1disp;
            decompMain.x1disp = nullptr;
        }
        if (decompMain.y1disp)
        {
            delete[] decompMain.y1disp;
            decompMain.y1disp = nullptr;
        }
        if (decompMain.y2disp)
        {
            delete[] decompMain.y2disp;
            decompMain.y2disp = nullptr;
        }
        if (decompMain.z2disp)
        {
            delete[] decompMain.z2disp;
            decompMain.z2disp = nullptr;
        }

        // Clear counts
        decompMain.x1count = decompMain.y1count = decompMain.y2count = decompMain.z2count = 0;
        decompMain.even                                                                   = false;

        // --- Free work buffers used for alltoall/alltoallv ---
        if (work1_r)
        {
            delete[] work1_r;
            work1_r = nullptr;
        }
        if (work2_r)
        {
            delete[] work2_r;
            work2_r = nullptr;
        }
        decompBufSize = 0;

        // --- Free / reset x/y/z pencil start/end/size arrays (they are fixed-size ints in class:
        // just zero them) ---
        for (int i = 0; i < 3; ++i)
        {
            xStart[i] = xEnd[i] = xSize[i] = 0;
            yStart[i] = yEnd[i] = ySize[i] = 0;
            zStart[i] = zEnd[i] = zSize[i] = 0;
        }

        // --- Free MPI communicators safely ---
        // Helper lambda to free a communicator if it's valid
        auto freeCommIfValid = [](MPI_Comm& comm)
        {
            if (comm != MPI_COMM_NULL)
            {
                // MPI_Comm_free requires a pointer to the communicator handle
                MPI_Comm tmp = comm;
                MPI_Comm_free(&tmp);
                // After free, tmp is set to MPI_COMM_NULL by MPI (but ensure our reference is null)
                comm = MPI_COMM_NULL;
            }
        };

        freeCommIfValid(DECOMP_2D_COMM_CART_X);
        freeCommIfValid(DECOMP_2D_COMM_CART_Y);
        freeCommIfValid(DECOMP_2D_COMM_CART_Z);
        freeCommIfValid(DECOMP_2D_COMM_ROW);
        freeCommIfValid(DECOMP_2D_COMM_COL);

        // --- Reset topology-related integers ---
        dims[0] = dims[1] = 0;
        coord[0] = coord[1] = 0;
        periodic[0] = periodic[1] = 0;
        periodicX = periodicY = periodicZ = false;

        // --- Reset neighbor info ---
        for (int d = 0; d < 3; ++d)
            for (int k = 0; k < 6; ++k)
                neighbor[d][k] = -1;

        // --- Reset global sizes and mpi info  ---
        nxGlobal = nyGlobal = nzGlobal = 0;
        nRank = nProc = 0;
        myTypeBytes   = 0;
    };

    // Just get it running without the optional decomp for now...
    void transposeX2Y(double* src, double* dst);
    void transposeY2Z(double* src, double* dst);
    void transposeZ2Y(double* src, double* dst);
    void transposeY2X(double* src, double* dst);

    // Get Transposes but with array indexing with the major index of the pencil...
    void transposeX2Y_MajorIndex(double* src, double* dst);
    void transposeY2Z_MajorIndex(double* src, double* dst);
    void transposeZ2Y_MajorIndex(double* src, double* dst);
    void transposeY2X_MajorIndex(double* src, double* dst);

    // calls for overlapping communication and computation...
    void transposeX2Y_Start(MPI_Request& handle, double* src, double* dst, double* sbuf,
                            double* rbuf);
    void transposeX2Y_Wait(MPI_Request& handle, double* src, double* dst, double* sbuf,
                           double* rbuf);

    void transposeY2Z_Start(MPI_Request& handle, double* src, double* dst, double* sbuf,
                            double* rbuf);
    void transposeY2Z_Wait(MPI_Request& handle, double* src, double* dst, double* sbuf,
                           double* rbuf);

    void transposeZ2Y_Start(MPI_Request& handle, double* src, double* dst, double* sbuf,
                            double* rbuf);
    void transposeZ2Y_Wait(MPI_Request& handle, double* src, double* dst, double* sbuf,
                           double* rbuf);

    void transposeY2X_Start(MPI_Request& handle, double* src, double* dst, double* sbuf,
                            double* rbuf);
    void transposeY2X_Wait(MPI_Request& handle, double* src, double* dst, double* sbuf,
                           double* rbuf);

    // calls for overlapping communication and computation...
    void transposeX2Y_MajorIndex_Start(MPI_Request& handle, double* src, double* dst, double* sbuf,
                                       double* rbuf);
    void transposeX2Y_MajorIndex_Wait(MPI_Request& handle, double* src, double* dst, double* sbuf,
                                      double* rbuf);

    void transposeY2Z_MajorIndex_Start(MPI_Request& handle, double* src, double* dst, double* sbuf,
                                       double* rbuf);
    void transposeY2Z_MajorIndex_Wait(MPI_Request& handle, double* src, double* dst, double* sbuf,
                                      double* rbuf);

    void transposeZ2Y_MajorIndex_Start(MPI_Request& handle, double* src, double* dst, double* sbuf,
                                       double* rbuf);
    void transposeZ2Y_MajorIndex_Wait(MPI_Request& handle, double* src, double* dst, double* sbuf,
                                      double* rbuf);

    void transposeY2X_MajorIndex_Start(MPI_Request& handle, double* src, double* dst, double* sbuf,
                                       double* rbuf);
    void transposeY2X_MajorIndex_Wait(MPI_Request& handle, double* src, double* dst, double* sbuf,
                                      double* rbuf);

    void decompInfoInit();
    void decompInfoFinalize();

    // only doing real
    void allocX(double*& var);
    void allocY(double*& var);
    void allocZ(double*& var);
    void deallocXYZ(double*& var);

    void updateHalo(double* in, double*& out, int level, int ipencil);

    void decomp2DAbort(int errorCode, string msg);
    void initNeighbor();
    void getDist();
    void distribute(int data1, int proc, int* st, int* en, int* sz);
    void partition(int nx, int ny, int nz, int* pdim, int* lstart, int* lend, int* lsize);
    void prepareBuffer(DecompInfo* dii);

    void getDecompInfo(DecompInfo dcompinfo_in);

    void memSplitXY(double* in, int n1, int n2, int n3, double* out, int iproc, int* dist);

    void memMergeXY(double* in, int n1, int n2, int n3, double* out, int iproc, int* dist);
    void memMergeXY_YMajor(double* in, int n1, int n2, int n3, double* out, int iproc, int* dist);

    void memSplitYZ(double* in, int n1, int n2, int n3, double* out, int iproc, int* dist);
    void memSplitYZ_YMajor(double* in, int n1, int n2, int n3, double* out, int iproc, int* dist);

    void memMergeYZ(double* in, int n1, int n2, int n3, double* out, int iproc, int* dist);
    void memMergeYZ_ZMajor(double* in, int n1, int n2, int n3, double* out, int iproc, int* dist);

    void memSplitZY(double* in, int n1, int n2, int n3, double* out, int iproc, int* dist);
    void memSplitZY_ZMajor(double* in, int n1, int n2, int n3, double* out, int iproc, int* dist);

    void memMergeZY(double* in, int n1, int n2, int n3, double* out, int iproc, int* dist);
    void memMergeZY_YMajor(double* in, int n1, int n2, int n3, double* out, int iproc, int* dist);

    void memSplitYX(double* in, int n1, int n2, int n3, double* out, int iproc, int* dist);
    void memSplitYX_YMajor(double* in, int n1, int n2, int n3, double* out, int iproc, int* dist);

    void memMergeYX(double* in, int n1, int n2, int n3, double* out, int iproc, int* dist);

    // IO
    void writeOne(int ipencil, double* var, string filename);
    void writeVar(MPI_File& fh, MPI_Offset& disp, int ipencil, double* var);
    void writeScalar(MPI_File& fh, MPI_Offset& disp, int n, double* var);
    void writePlane(int ipencil, double* var, int iplane, int n, string filename);
    void writeEvery(int ipencil, double* var, int iskip, int jskip, int kskip, string filename,
                    bool from1);

    void readOne(int ipencil, double* var, string filename);
    void readVar(MPI_File& fh, MPI_Offset& disp, int ipencil, double* var);
    void readScalar(MPI_File& fh, MPI_Offset& disp, int n, double* var);
};

#endif
