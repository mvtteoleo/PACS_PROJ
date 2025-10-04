{
    description = "Pacs environment";

    inputs = {
        nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05"; # or any branch/tag/commit
    };

    outputs = { self, nixpkgs }:
        let
            system = "x86_64-linux";
            pkgs = import nixpkgs { inherit system; };

            pyEnv = pkgs.python311.withPackages (ps: with ps; [
                pandas
                sympy
                pyvista
                matplotlib
                numpy
                vtk
                black
            ]);

        in {
            devShells.${system}.default = pkgs.mkShell {
                buildInputs = [
                    pkgs.git
                    # debug stuff
                    pkgs.heaptrack
                    pkgs.gdb
                    # libraries
                    pkgs.ginac
                    pkgs.fftw
                    pkgs.eigen
                    pkgs.petsc
                    # compile and run
                    pkgs.gnumake
                    pkgs.gcc
                    pkgs.mpich
                    pkgs.openmpi
                    pkgs.opensycl
                    pkgs.clang-tools
                    # Python stuff
                    pyEnv
                ];
            };
             shellHook = ''
        export PKG_CONFIG_PATH="${nixpkgs.fftw}/lib/pkgconfig:${nixpkgs.fftw}/lib64/pkgconfig:$PKG_CONFIG_PATH"
        echo "FFTW and OpenMPI are available"
      '';
        };
}

