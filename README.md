This project is a direct numerical solver for computational fluid dynamic using
Chorin-Themam decomposition, the NSSolver class is templated so that is able to
solve the Poisson equation for pressure both using Spectral solver and a
Multigrid solver, a better explanation can be seen [here](./doc/pacs_rep.pdf) 
The C++ standard of choice is C++23, to manage the dependencies I here use nix,
the only real dependency.

If one is missing nix can avoid installing it globally using [nix-portable](https://github.com/DavHau/nix-portable).

The [flake.nix](./flake.nix) file is the one that handles dependencies and packages needed.

To run one needs to either call `nix develop` or `./nix-portable nix develop
--extra-experimental-features "nix-command flakes"` to enter in the
nix-shell. 

The directory is divided as can be seeen below.
```
.
├── build/
├── debug.pbs
├── flake.lock
├── flake.nix
├── include/
│   ├── datastructs/
│   ├── impl/
│   └── third_party/
│       ├── 2Decomp_C/
│       ├── gnuplot-iostream.h
│       └── MPI_types.hpp
├── Makefile
├── new_flake.nix
├── output
├── perf_test.pbs
├── README.md
├── scaling.pbs
├── tests/
│   ├── parallel/
│   └── serial/
└── tools

17 directories, 114 files
```

All the dependency not present in the flake.nix file are inside the include/third_party/ directory. 

  -  [2DecompC](https://github.com/emathew1/2Decomp_C) To handle domain
  decomposition and data transpositions necessary for the FastPoissonSolver.
  - [gnuplot-iostrem](https://github.com/dstahlke/gnuplot-iostream) In order to
  make plots in a more fast  way and avoid the complex sintax and verbosity of
  gnuplot.
  -  [MPI_Types](https://gist.github.com/2b-t/50d85115db8b12ed263f8231abf07fa2)
  To handle the MPI types in a consistent way give the template usage.

The tests are divided in parallel and serial for debug purposes.
All the executables will be in the build directory and then each is divided
between serial and parallel subdirectory.

Once inside the nix-shell is enough to call `make parallel` or `make serial` or
`make all` to build all the executables `make main` will build main.cpp.


The test are quite a few and each one has its own description on the top of
the source code.

One note on the tools/ directory, it contains some .py to handle manufactured
solutions and in general helpers.

Some more interesting aspect in the project:

  - VTK written in parallel using the .pvts format, only support for the scalar fields. (Like pressure)
  - Curiously recurring templates instead of inheritance from virtual
  - Policy based designed
  - Expression templates

