#pragma once

#include "MPI_types.hpp"
#include "math.h"
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory.h>
#include <mpi.h>
#include <string>

template <typename T>
class C2Decomp
{

  public:
    // Just assume that we're using double precision all the time
    using myType            = T;
    MPI_Datatype myType_MPI = mpi_get_type<T>();

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

    void decomp2DInit(int pRow, int pCol)
    {

        int errorcode, ierr, row, col;

        row = 0;
        col = 0;

        // Get the mpi rank and size
        ierr = MPI_Comm_size(MPI_COMM_WORLD, &nProc);
        ierr = MPI_Comm_rank(MPI_COMM_WORLD, &nRank);

        if (pRow == 0 && pCol == 0)
        {
            best2DGrid(nProc, row, col);
        }
        else
        {
            if (nProc != pRow * pCol)
            {
                errorcode               = 1;
                std::string errorstring = "Invalid 2D processor grid - nproc /= p_row*p_col\n";
                decomp2DAbort(errorcode, errorstring);
            }
            else
            {
                row = pRow;
                col = pCol;
            }
        }

        dims[0] = row;
        dims[1] = col;

        // Set up the cartesian coordinates...
        periodic[0] = (int) periodicY;
        periodic[1] = (int) periodicZ;
        ierr        = MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 0, &DECOMP_2D_COMM_CART_X);

        periodic[0] = (int) periodicX;
        periodic[1] = (int) periodicZ;
        ierr        = MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 0, &DECOMP_2D_COMM_CART_Y);

        periodic[0] = (int) periodicX;
        periodic[1] = (int) periodicY;
        ierr        = MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 0, &DECOMP_2D_COMM_CART_Z);

        // Get the current rank's coordinate in the Cartesian communicator...
        ierr = MPI_Cart_coords(DECOMP_2D_COMM_CART_X, nRank, 2, coord);

        // Derive communicators defining sub-groups for Alltoall(v)
        int remain[2] = {1, 0};
        ierr          = MPI_Cart_sub(DECOMP_2D_COMM_CART_X, remain, &DECOMP_2D_COMM_COL);

        remain[0] = 0;
        remain[1] = 1;
        ierr      = MPI_Cart_sub(DECOMP_2D_COMM_CART_X, remain, &DECOMP_2D_COMM_ROW);

        //////////////////////////
        // This could be a pitfall right here because of the column major to row major conversion...
        //////////////////////////

        // gather information for halo-cell support
        initNeighbor();

        decompInfoInit();

        for (int i = 0; i < 3; i++)
        {
            // minus 1 to get C style zero start indices
            xStart[i] = decompMain.xst[i] - 1;
            yStart[i] = decompMain.yst[i] - 1;
            zStart[i] = decompMain.zst[i] - 1;

            xEnd[i] = decompMain.xen[i] - 1;
            yEnd[i] = decompMain.yen[i] - 1;
            zEnd[i] = decompMain.zen[i] - 1;

            xSize[i] = decompMain.xsz[i];
            ySize[i] = decompMain.ysz[i];
            zSize[i] = decompMain.zsz[i];
        }

        ierr = MPI_Type_size(myType_MPI, &myTypeBytes);
    }

    void best2DGrid(int iproc, int& best_pRow, int& best_pCol)
    {

        if (!nRank)
        {
            std::cout << "C2Decomp: In auto-tuning mode..." << std::endl;
        }

        double best_time = HUGE_VAL;
        double t2, t1;

        best_pRow = -1;
        best_pCol = -1;

        myType *u1, *u2, *u3;

        int  factSize = (int) sqrt((double) iproc) + 10;
        int* factors  = new int[factSize];
        int  nfact    = 0;

        // Get the factors of the number of processes
        FindFactor(iproc, factors, nfact);

        if (!nRank)
        {
            std::cout << "    factors: ";
            for (int ip = 0; ip < nfact; ip++)
            {
                std::cout << factors[ip] << " ";
            }
            std::cout << std::endl;
        }

        for (int ip = 0; ip < nfact; ip++)
        {

            int row = factors[ip];
            int col = iproc / row;

            if (std::min(nxGlobal, nyGlobal) >= row && std::min(nyGlobal, nzGlobal) >= col)
            {
                dims[0] = row;
                dims[1] = col;

                periodic[0] = 0;
                periodic[1] = 0;

                MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 0, &DECOMP_2D_COMM_CART_X);
                MPI_Cart_coords(DECOMP_2D_COMM_CART_X, nRank, 2, coord);

                int remain[2] = {1, 0};
                MPI_Cart_sub(DECOMP_2D_COMM_CART_X, remain, &DECOMP_2D_COMM_COL);

                remain[0] = 0;
                remain[1] = 1;
                MPI_Cart_sub(DECOMP_2D_COMM_CART_X, remain, &DECOMP_2D_COMM_ROW);

                decompInfoInit();

                allocX(u1);
                allocY(u2);
                allocZ(u3);

                t1 = MPI_Wtime();
                for (int numTransTest = 0; numTransTest < 50; numTransTest++)
                {
                    transposeX2Y_MajorIndex(u1, u2);
                    transposeY2Z_MajorIndex(u2, u3);
                    transposeZ2Y_MajorIndex(u3, u2);
                    transposeY2X_MajorIndex(u2, u1);
                }
                t2 = MPI_Wtime() - t1;

                deallocXYZ(u1);
                deallocXYZ(u2);
                deallocXYZ(u3);

                decompInfoFinalize();

                MPI_Allreduce(&t2, &t1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

                t1 /= (double) nProc;

                if (!nRank)
                {
                    std::cout << "    Processor Grid " << row << " by " << col << ", time = " << t1
                              << std::endl;
                }

                if (best_time > t1)
                {
                    best_time = t1;
                    best_pRow = row;
                    best_pCol = col;
                }

                // set pointers back to NULL
                DECOMP_2D_COMM_CART_X = MPI_COMM_NULL;
                DECOMP_2D_COMM_ROW    = MPI_COMM_NULL;
                DECOMP_2D_COMM_COL    = MPI_COMM_NULL;
            }
        }

        delete[] factors;

        if (best_pRow != -1)
        {
            if (!nRank)
            {
                std::cout << "    ==============================================================="
                          << std::endl;
                std::cout << "    The best processor grid is probably " << best_pRow << " by "
                          << best_pCol << std::endl;
            }
        }
        else
        {
            int         errorcode = 9;
            std::string errorstring =
                "The processor=grid auto-tuning code fail. The number of processes "
                "requested is probably too large ";
            decomp2DAbort(errorcode, errorstring);
        }
    }

    void FindFactor(int num, int* factors, int& nfact)
    {

        int m;

        // Finding factors <= sqrt(num)
        m     = (int) sqrt((double) num);
        nfact = 1;
        for (int ip = 1; ip < m + 1; ip++)
        {
            if (num / ip * ip == num)
            {
                factors[nfact - 1] = ip;
                nfact++;
            }
        }
        nfact--;

        // Finding factors > sqrt(num)
        if (factors[nfact - 1] * factors[nfact - 1] != num)
        {
            for (int ip = nfact + 1; ip < 2 * nfact + 1; ip++)
            {
                factors[ip - 1] = num / factors[2 * nfact - ip];
            }
            nfact *= 2;
        }
        else
        {
            for (int ip = nfact + 1; ip < 2 * nfact; ip++)
            {
                factors[ip - 1] = num / factors[2 * nfact - ip - 1];
            }
            nfact = nfact * 2 - 1;
        }
    }

    void transposeX2Y_MajorIndex(myType* src, myType* dst)
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

    void transposeY2X_MajorIndex(myType* src, myType* dst)
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

    void transposeY2Z_MajorIndex(myType* src, myType* dst)
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

    void transposeZ2Y_MajorIndex(myType* src, myType* dst)
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

    void decompInfoInit()
    {

        int bufSize, nx, ny, nz, errorcode;
        nx = nxGlobal;
        ny = nyGlobal;
        nz = nzGlobal;

        // verify the global size can actually be distributed as pencils
        if (nx < dims[0] || ny < dims[0] || ny < dims[1] || nz < dims[1])
        {
            errorcode = 6;
            std::string msg =
                "Invalid 2D processor grid. \n Make sure that min(nx, ny) > p_row and min(ny, "
                "nz) >= p_col.";
            decomp2DAbort(errorcode, msg);
        }

        if (nx % dims[0] == 0 && ny % dims[0] == 0 && ny % dims[1] == 0 && nz % dims[1] == 0)
        {
            decompMain.even = true;
        }
        else
        {
            decompMain.even = false;
        }

        decompMain.x1dist = new int[dims[0]];
        decompMain.y1dist = new int[dims[0]];
        decompMain.y2dist = new int[dims[1]];
        decompMain.z2dist = new int[dims[1]];

        getDist();

        int pdim[3] = {0, 1, 2};
        partition(nx, ny, nz, pdim, decompMain.xst, decompMain.xen, decompMain.xsz);

        pdim[0] = 1;
        pdim[1] = 0;
        pdim[2] = 2;
        partition(nx, ny, nz, pdim, decompMain.yst, decompMain.yen, decompMain.ysz);

        pdim[0] = 1;
        pdim[1] = 2;
        pdim[2] = 0;
        partition(nx, ny, nz, pdim, decompMain.zst, decompMain.zen, decompMain.zsz);

        decompMain.x1cnts = new int[dims[0]];
        decompMain.y1cnts = new int[dims[0]];
        decompMain.y2cnts = new int[dims[1]];
        decompMain.z2cnts = new int[dims[1]];

        decompMain.x1disp = new int[dims[0]];
        decompMain.y1disp = new int[dims[0]];
        decompMain.y2disp = new int[dims[1]];
        decompMain.z2disp = new int[dims[1]];

        prepareBuffer(&decompMain);

        int size1 = decompMain.xsz[0] * decompMain.xsz[1] * decompMain.xsz[2];
        int size2 = decompMain.ysz[0] * decompMain.ysz[1] * decompMain.xsz[2];
        int size3 = decompMain.zsz[0] * decompMain.zsz[1] * decompMain.zsz[2];

        bufSize = std::max(size2, size3);
        bufSize = std::max(size1, bufSize);

        if (bufSize > decompBufSize)
        {
            if (work1_r != NULL)
            {
                delete[] work1_r;
                work1_r = NULL;
            }

            if (work2_r != NULL)
            {
                delete[] work2_r;
                work2_r = NULL;
            }

            work1_r       = new myType[bufSize];
            work2_r       = new myType[bufSize];
            decompBufSize = bufSize;
        }
    }
    void decompInfoFinalize()
    {

        decompBufSize = 0;

        delete[] decompMain.x1dist;
        delete[] decompMain.y1dist;
        delete[] decompMain.y2dist;
        delete[] decompMain.z2dist;

        delete[] decompMain.x1cnts;
        delete[] decompMain.y1cnts;
        delete[] decompMain.y2cnts;
        delete[] decompMain.z2cnts;

        delete[] decompMain.x1disp;
        delete[] decompMain.y1disp;
        delete[] decompMain.y2disp;
        delete[] decompMain.z2disp;

        delete[] work1_r;
        delete[] work2_r;

        decompMain.x1dist = NULL;
        decompMain.y1dist = NULL;
        decompMain.y2dist = NULL;
        decompMain.z2dist = NULL;

        decompMain.x1cnts = NULL;
        decompMain.y1cnts = NULL;
        decompMain.y2cnts = NULL;
        decompMain.z2cnts = NULL;

        decompMain.x1disp = NULL;
        decompMain.y1disp = NULL;
        decompMain.y2disp = NULL;
        decompMain.z2disp = NULL;

        work1_r = NULL;
        work2_r = NULL;
    }

    void decomp2DAbort(int errorCode, std::string msg)
    {
        int ierr;
        if (!nRank)
        {
            std::cout << "2Decopm_c Error - errorCode: " << errorCode << std::endl;
            std::cout << "Error Message: " << msg << std::endl;
        }
        ierr = MPI_Abort(MPI_COMM_WORLD, errorCode);
    }
    void initNeighbor()
    {

        // X-pencil
        neighbor[0][0] = MPI_PROC_NULL;
        neighbor[0][1] = MPI_PROC_NULL;
        MPI_Cart_shift(DECOMP_2D_COMM_CART_X, 0, 1, &neighbor[0][3], &neighbor[0][2]);
        MPI_Cart_shift(DECOMP_2D_COMM_CART_X, 1, 1, &neighbor[0][5], &neighbor[0][4]);

        // Y-pencil
        MPI_Cart_shift(DECOMP_2D_COMM_CART_Y, 0, 1, &neighbor[1][1], &neighbor[1][0]);
        neighbor[1][2] = MPI_PROC_NULL;
        neighbor[1][3] = MPI_PROC_NULL;
        MPI_Cart_shift(DECOMP_2D_COMM_CART_Y, 1, 1, &neighbor[1][5], &neighbor[1][4]);

        // Z-pencil
        MPI_Cart_shift(DECOMP_2D_COMM_CART_Z, 0, 1, &neighbor[2][1], &neighbor[2][0]);
        MPI_Cart_shift(DECOMP_2D_COMM_CART_Z, 1, 1, &neighbor[2][3], &neighbor[2][2]);
        neighbor[2][4] = MPI_PROC_NULL;
        neighbor[2][5] = MPI_PROC_NULL;
    }
    void getDist()
    {

        int nx, ny, nz;
        nx = nxGlobal;
        ny = nyGlobal;
        nz = nzGlobal;

        int* st1 = new int[dims[0]];
        int* en1 = new int[dims[0]];

        distribute(nx, dims[0], st1, en1, decompMain.x1dist);
        distribute(ny, dims[0], st1, en1, decompMain.y1dist);

        delete[] st1;
        delete[] en1;

        int* st2 = new int[dims[1]];
        int* en2 = new int[dims[1]];

        distribute(ny, dims[1], st2, en2, decompMain.y2dist);
        distribute(nz, dims[1], st2, en2, decompMain.z2dist);

        delete[] st2;
        delete[] en2;
    }
    void distribute(int data1, int proc, int* st, int* en, int* sz)
    {

        int size1, nl, nu;

        size1 = data1 / proc;
        nu    = data1 - size1 * proc;
        nl    = proc - nu;

        st[0] = 1;
        sz[0] = size1;
        en[0] = size1;

        for (int i = 1; i < nl; i++)
        {
            st[i] = st[i - 1] + size1;
            sz[i] = size1;
            en[i] = en[i - 1] + size1;
        }

        size1 = size1 + 1;

        for (int i = nl; i < proc; i++)
        {
            st[i] = en[i - 1] + 1;
            sz[i] = size1;
            en[i] = en[i - 1] + size1;
        }

        en[proc - 1] = data1;
        sz[proc - 1] = data1 - st[proc - 1] + 1;
    }
    void partition(int nx, int ny, int nz, int* pdim, int* lstart, int* lend, int* lsize)
    {

        int gsize;

        for (int i = 0; i < 3; i++)
        {

            if (i == 0)
            {
                gsize = nx;
            }
            else if (i == 1)
            {
                gsize = ny;
            }
            else if (i == 2)
            {
                gsize = nz;
            }

            if (pdim[i] == 0)
            {
                lstart[i] = 1;
                lend[i]   = gsize;
                lsize[i]  = gsize;
            }
            else if (pdim[i] == 1)
            {
                int *st, *en, *sz;
                st = new int[dims[0]];
                en = new int[dims[0]];
                sz = new int[dims[0]];

                distribute(gsize, dims[0], st, en, sz);

                lstart[i] = st[coord[0]];
                lend[i]   = en[coord[0]];
                lsize[i]  = sz[coord[0]];

                delete[] st;
                delete[] en;
                delete[] sz;
            }
            else if (pdim[i] == 2)
            {

                int *st, *en, *sz;
                st = new int[dims[1]];
                en = new int[dims[1]];
                sz = new int[dims[1]];

                distribute(gsize, dims[1], st, en, sz);

                lstart[i] = st[coord[1]];
                lend[i]   = en[coord[1]];
                lsize[i]  = sz[coord[1]];

                delete[] st;
                delete[] en;
                delete[] sz;
            }
        }
    }

    void prepareBuffer(DecompInfo* dii)
    {

        // MPI_Alltoallv buffer info

        for (int i = 0; i < dims[0]; i++)
        {
            dii->x1cnts[i] = dii->x1dist[i] * dii->xsz[1] * dii->xsz[2];
            dii->y1cnts[i] = dii->y1dist[i] * dii->ysz[0] * dii->ysz[2];
            if (i == 0)
            {
                dii->x1disp[i] = 0;
                dii->y1disp[i] = 0;
            }
            else
            {
                dii->x1disp[i] = dii->x1disp[i - 1] + dii->x1cnts[i - 1];
                dii->y1disp[i] = dii->y1disp[i - 1] + dii->y1cnts[i - 1];
            }
        }

        for (int i = 0; i < dims[1]; i++)
        {
            dii->y2cnts[i] = dii->ysz[0] * dii->y2dist[i] * dii->ysz[2];
            dii->z2cnts[i] = dii->zsz[0] * dii->zsz[1] * dii->z2dist[i];
            if (i == 0)
            {
                dii->y2disp[i] = 0;
                dii->z2disp[i] = 0;
            }
            else
            {
                dii->y2disp[i] = dii->y2disp[i - 1] + dii->y2cnts[i - 1];
                dii->z2disp[i] = dii->z2disp[i - 1] + dii->z2cnts[i - 1];
            }
        }

        dii->x1count = dii->x1dist[dims[0] - 1] * dii->y1dist[dims[0] - 1] * dii->xsz[2];
        dii->y1count = dii->x1count;

        dii->y2count = dii->y2dist[dims[1] - 1] * dii->z2dist[dims[1] - 1] * dii->zsz[0];
        dii->z2count = dii->y2count;
    }
    void getDecompInfo(DecompInfo dcompinfo_in);

    template <bool YMajor = false>
    void memTransfer(myType*& in, myType*& out, int n1, int n2, int n3, int iproc, int* dist,
                     int* disp)
    {
        for (int m = 0; m < iproc; ++m)
        {
            int i1 = (m == 0) ? 1 : 0;
            int i2 = (m == 0) ? dist[0] : 0;

            if (m != 0)
            {
                i1 = i2 + 1;
                i2 = i1 + dist[m] - 1;
            }

            int pos = disp[m];

            for (int k = 0; k < n3; ++k)
            {
                for (int j = i1 - 1; j < i2; ++j)
                {
                    if constexpr (!YMajor)
                    {
                        // contiguous in i
                        std::memcpy(out + pos, in + k * n2 * n1 + j * n1, sizeof(myType) * n1);
                        pos += n1;
                    }
                    else
                    {
                        for (int i = 0; i < n1; ++i)
                        {
                            int ii  = i * n3 * n2 + k * n2 + j;
                            out[ii] = in[pos++];
                        }
                    }
                }
            }
        }
    }
    /* MINIMAL SETUP, JUST WHAT IS USED */
    // ----------------------- XY -----------------------
    void memSplitXY(myType* in, int n1, int n2, int n3, myType* out, int iproc, int* dist)
    {
        memTransfer<false>(in, out, n1, n2, n3, iproc, dist, decompMain.x1disp);
    }

    void memMergeXY_YMajor(myType* in, int n1, int n2, int n3, myType* out, int iproc, int* dist)
    {
        memTransfer<true>(in, out, n1, n2, n3, iproc, dist, decompMain.y1disp);
    }

    // ----------------------- YX -----------------------
    void memSplitYX_YMajor(myType* in, int n1, int n2, int n3, myType* out, int iproc, int* dist)
    {
        memTransfer<true>(in, out, n1, n2, n3, iproc, dist, decompMain.y1disp);
    }

    void memMergeYX(myType* in, int n1, int n2, int n3, myType* out, int iproc, int* dist)
    {
        memTransfer<false>(in, out, n1, n2, n3, iproc, dist, decompMain.x1disp);
    }

    // ----------------------- YZ -----------------------

    void memSplitYZ_YMajor(myType* in, int n1, int n2, int n3, myType* out, int iproc, int* dist)
    {
        memTransfer<true>(in, out, n1, n2, n3, iproc, dist, decompMain.y2disp);
    }

    // ----------------------- ZY -----------------------
    void memMergeZY_YMajor(myType* in, int n1, int n2, int n3, myType* out, int iproc, int* dist)
    {
        memTransfer<true>(in, out, n1, n2, n3, iproc, dist, decompMain.y2disp);
    }

    void allocX(myType*& var)
    {

        int xsize = decompMain.xsz[0];
        int ysize = decompMain.xsz[1];
        int zsize = decompMain.xsz[2];

        var = new myType[xsize * ysize * zsize];
    }

    void allocY(myType*& var)
    {

        int xsize = decompMain.ysz[0];
        int ysize = decompMain.ysz[1];
        int zsize = decompMain.ysz[2];

        var = new myType[xsize * ysize * zsize];
    }

    void allocZ(myType*& var)
    {

        int xsize = decompMain.zsz[0];
        int ysize = decompMain.zsz[1];
        int zsize = decompMain.zsz[2];

        var = new myType[xsize * ysize * zsize];
    }

    void deallocXYZ(myType*& var)
    {
        delete[] var;
        var = NULL;
    }

    void decomp2DFinalize()
    {
        std::cout << "Freeing C2Decop structures and buffers\n";
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

        // --- Reset global sizes and mpi info (optional but cleans state) ---
        nxGlobal = nyGlobal = nzGlobal = 0;
        nRank = nProc = 0;

        // myTypeBytes can be reset if desired
        myTypeBytes = 0;
    }
};
