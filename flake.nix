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
                    if [ -n "$LD_LIBRARY_PATH" ]; then
                    echo "⚠️  Warning: LD_LIBRARY_PATH is set ($LD_LIBRARY_PATH)"
                    echo "🔒 This shell expects a pure environment."
                    exit 1
                    fi
                    echo "✅ Pacs environment activated"
            '';
        };
}

