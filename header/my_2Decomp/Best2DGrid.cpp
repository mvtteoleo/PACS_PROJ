#include "C2Decomp.hpp"
#include <iterator>
#include <memory>
#include <ostream>

void C2Decomp::FindFactor(int num, int* factors, int& nfact)
{

    int m;

    // Finding factors <= sqrt(num)
    m     = (int) sqrt((double) num);
    nfact = 1;
    for (int ip = 1; ip < m + 1; ++ip)
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
        for (int ip = nfact + 1; ip < 2 * nfact + 1; ++ip)
        {
            factors[ip - 1] = num / factors[2 * nfact - ip];
        }
        nfact *= 2;
    }
    else
    {
        for (int ip = nfact + 1; ip < 2 * nfact; ++ip)
        {
            factors[ip - 1] = num / factors[2 * nfact - ip - 1];
        }
        nfact = nfact * 2 - 1;
    }
}

void C2Decomp::best2DGrid(int iproc, int& best_pRow, int& best_pCol)
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
        for (int ip = 0; ip < nfact; ++ip)
        {
            std::cout << factors[ip] << " ";
        }
        std::cout << std::endl;
    }

    for (int ip = 0; ip < nfact; ++ip)
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
