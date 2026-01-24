{
  description = "Pacs environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            # Version control
            pkgs.git

            # Compilers
            pkgs.gnumake
            pkgs.gcc15
            pkgs.clang-tools
            pkgs.openmpi

            # Libraries
            pkgs.ginac
            pkgs.fftw
            pkgs.eigen
	    pkgs.petsc
            pkgs.boost
            pkgs.gnuplot
            pkgs.tbb

            # Tools
            pkgs.heaptrack
            pkgs.gdb
	    pkgs.neovim
          ];

          shellHook = ''
            # Set up environment variables
            export EIGEN_INCLUDE_DIR="${pkgs.eigen}/include/eigen3"
            export FFTW_INCLUDE_DIR="${pkgs.fftw}/include"
            export PETSC_DIR="${pkgs.petsc}"
            export PETSC_ARCH=""

            # FIX: Do not exit if LD_LIBRARY_PATH exists.
            # Instead, just unset it inside the shell to ensure purity.
            unset LD_LIBRARY_PATH

            echo "✅ Thesis environment activated"
          '';
        };
      }
    );
}

