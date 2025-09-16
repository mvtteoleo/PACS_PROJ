#include "decompose.hpp"
#include "tensors.hpp"
#include <cstddef>
#include <fftw3.h>
#include <memory>
#include <random>

/*
 * At init I need:
 *  - BC (Just homo for now)
 *  - decomp infos (To handle the sizes of the buffers)
 *  - Build the fftw datastructures
 *
 */
namespace numPDE
{
    enum BC
    {
        NeuHomo,
        DirHomo,
        // No support for periodic for the moment
        // Periodic
    };

    enum SIDES
    {
        NORTH, // ++i
        SOUTH, // --i
        WEST,  // ++j
        EST,   // --j
        TOP,   // ++k
        BOTTOM // --k
    };

    struct BoudaryConditions
    {
        BC BC_x = NeuHomo;
        BC BC_y = NeuHomo;
        BC BC_z = NeuHomo;
    };

    template <typename T = double>
    class FastPoissonSolver
    {
      public:
        FastPoissonSolver(NewDecomp<T>& decomp, BoudaryConditions& Bcs)
            : p_Decomp(std::make_shared<NewDecomp<T>>(decomp)), m_BCs{Bcs}
        {

            auto& Lx = p_Decomp->xSize()[0];
            auto& Ly = p_Decomp->ySize()[1];
            auto& Lz = p_Decomp->zSize()[2];

            auto buf_size = size_t{std::max({Lx, Ly, Lz})};
            auto y_size =
                std::accumulate(decomp.ySize()[0], decomp.ySize()[2], size_t{1}, std::multiplies{});
            auto z_size =
                std::accumulate(decomp.zSize()[0], decomp.zSize()[2], size_t{1}, std::multiplies{});

            // Allocate memory for the buffers
            m_fftbuf   = static_cast<T*>(fftw_malloc(sizeof(T) * buf_size));
            m_Y_Pencil = (T*) fftw_malloc(sizeof(T) * y_size);
            m_Z_Pencil = (T*) fftw_malloc(sizeof(T) * z_size);

            m_Nx = (m_BCs.BC_x == DirHomo) ? Lx - 2 : Lx;
            m_Ny = (m_BCs.BC_y == DirHomo) ? Ly - 2 : Ly;
            m_Nz = (m_BCs.BC_z == DirHomo) ? Lz - 2 : Lz;

            fft_x =
                fftw_plan(m_Nx, m_fftbuf, m_fftbuf,
                          (m_BCs.BC_x == DirHomo) ? FFTW_REDFFT00 : FFTW_RODFFT00, FFTW_ESTIMATE);
            fft_y =
                fftw_plan(m_Ny, m_fftbuf, m_fftbuf,
                          (m_BCs.BC_y == DirHomo) ? FFTW_REDFFT00 : FFTW_RODFFT00, FFTW_ESTIMATE);
            fft_z =
                fftw_plan(m_Nz, m_fftbuf, m_fftbuf,
                          (m_BCs.BC_z == DirHomo) ? FFTW_REDFFT00 : FFTW_RODFFT00, FFTW_ESTIMATE);
        };

        ~FastPoissonSolver();

      private:
        std::shared_ptr<NewDecomp<T>> p_Decomp;
        BoudaryConditions             m_BCs;
        T*                            m_fftbuf   = nullptr;
        T*                            m_Y_Pencil = nullptr;
        T*                            m_Z_Pencil = nullptr;
        // Just one because we can leverage the symmetry DCT and DST are equal in this case
        fftw_plan fft_x = nullptr;
        fftw_plan fft_y = nullptr;
        fftw_plan fft_z = nullptr;
        size_t    m_Nx;
        size_t    m_Ny;
        size_t    m_Nz;
    };
} // namespace numPDE
