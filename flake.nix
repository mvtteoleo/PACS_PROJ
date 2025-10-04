 {
  description = "Pacs environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    # Helper for making flakes portable across systems (linux, macos, etc.)
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    # This function applies the configuration to common systems.
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # Python environment with specified packages
        pyEnv = pkgs.python311.withPackages (ps: with ps; [
          pandas
          sympy
          pyvista
          matplotlib
          numpy
          vtk
          black
        ]);

      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            # Version control
            pkgs.git

            # Compilers and build tools
            pkgs.gnumake
            pkgs.gcc
            pkgs.clang-tools
            # NOTE: You have two MPI implementations. You typically only need one.
            # Choose either mpich or openmpi depending on your project's needs.
                         #pkgs.mpich
             pkgs.openmpi
         # pkgs.opensycl

            # C++ Libraries
            pkgs.ginac
            pkgs.fftw
            pkgs.eigen
            pkgs.petsc

            # Debugging
            pkgs.heaptrack
            pkgs.gdb

            # Python Environment
            pyEnv
          ];

          # This hook now runs correctly inside the shell definition.
          shellHook = ''
            # Correct paths for libraries
            export EIGEN_INCLUDE_DIR="${pkgs.eigen}/include/eigen3"
            export FFTW_INCLUDE_DIR="${pkgs.fftw}/include"

            # PETSc build systems typically use PETSC_DIR and PETSC_ARCH
            export PETSC_DIR="${pkgs.petsc}"
            export PETSC_ARCH=""

            echo "✅ Pacs environment activated"
          '';
        };
      }
    );
}
