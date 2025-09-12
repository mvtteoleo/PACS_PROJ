#include "mpi.h"
#include "C2Decomp.hpp"

using Real = int;

int main (int argc, char *argv[]) {
    
    int N = 5;
    int nx = N;
    int ny = N;
    int nz = N;
    int pRow = 0;
    int pCol = 0;
    bool periodic[3] = {true, true,true};
    C2Decomp<Real> c2d ( nx,  ny,  nz,  pRow,  pCol, periodic);

    

    return 0;
}

