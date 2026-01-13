{
  description = "Pacs environment with GCC 15";

  inputs = {
    # CRITICAL: GCC 15 is too new for 24.05. You MUST use unstable.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };

      # 1. Define GCC 15 Environment
      gcc15Env = pkgs.gcc15Stdenv;

      # 2. Override MPI to use GCC 15
      # This fixes the "mpicc" binary incompatibility
      mpiGCC15 = pkgs.openmpi.override { stdenv = gcc15Env; };

        pyEnv = pkgs.python311.withPackages (ps: with ps; [
                    pandas
                    sympy
                    pyvista
                    matplotlib
                    numpy
                    vtk
                    black
        ]);

        # Shared Dependencies
        myDevTools = [
          pkgs.git
          pkgs.gnumake
          pkgs.gcc15
          pkgs.clang-tools
          pkgs.bashInteractive
          pkgs.coreutils
          

          # HPC Libraries
          pkgs.openmpi
          pkgs.openmpi.dev 

          pkgs.fftw
          pkgs.fftw.dev
          
          pkgs.eigen
          pkgs.petsc
          pkgs.boost
          pkgs.tbb
          pkgs.ginac

          # Vis & Debug
          pkgs.gnuplot
          pkgs.heaptrack
          pkgs.gdb

          pyEnv
        ];

    in 
        {
        # 1. Local Shell
        devShells.${system}.default = (pkgs.mkShell.override { stdenv = gcc15Env; }) {
          buildInputs = myDevTools;
          shellHook = ''
            export EIGEN_INCLUDE_DIR="${pkgs.eigen}/include/eigen3"
            export FFTW_INCLUDE_DIR="${pkgs.fftw.dev}/include"
            export PETSC_DIR="${pkgs.petsc}"
            export PETSC_ARCH=""
            echo "✅ Thesis environment activated"
          '';
        };

        # 2. Apptainer/Docker Image
        packages.container = pkgs.dockerTools.buildImage {
          name = "pacs-env";
          tag = "latest";
          
          copyToRoot = pkgs.buildEnv {
            name = "image-root";
            paths = myDevTools;
            pathsToLink = [ "/bin" "/lib" "/include" "/share" ];
          };

          # NEW: Fix for 'passwd file missing' warnings
          runAsRoot = ''
            mkdir -p /tmp
            chmod 1777 /tmp
            mkdir -p /etc
            echo "root:x:0:0::/root:/bin/bash" > /etc/passwd
            echo "root:x:0:" > /etc/group
          '';

          config = {
            Env = [
              "EIGEN_INCLUDE_DIR=${pkgs.eigen}/include/eigen3"
              "FFTW_INCLUDE_DIR=${pkgs.fftw.dev}/include"
              "PETSC_DIR=${pkgs.petsc}"
              "PETSC_ARCH="
              # Ensure /bin is first so system tools are found
              "PATH=/bin:/usr/bin:${pyEnv}/bin"
            ];
            Cmd = [ "/bin/bash" ];
          };
        };
      };
}

