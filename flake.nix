{
  description = "Pacs environment with Apptainer support";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # Python Environment
        pyEnv = pkgs.python311.withPackages (ps: with ps; [
          pandas sympy pyvista matplotlib numpy vtk black
        ]);

        # Shared Dependencies
        myDevTools = [
          pkgs.git
          pkgs.gnumake
          pkgs.gcc
          pkgs.clang-tools
          pkgs.bashInteractive
          pkgs.coreutils
          
          # NEW: Explicitly add 'which' so you can debug
          pkgs.which 

          # HPC Libraries
          # NEW: We include BOTH the runtime and the dev output (compilers)
          pkgs.openmpi
          pkgs.openmpi.dev # <--- CRITICAL: Contains mpic++

          pkgs.fftw
          pkgs.fftw.dev    # Good practice to include headers explicitly if split
          
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
        devShells.default = pkgs.mkShell {
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
      }
    );
}

