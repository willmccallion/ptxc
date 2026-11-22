{
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      lib = nixpkgs.lib;
      # ptxas ships in NVIDIA's unfree CUDA redistributables (nvcc and the packages it
      # pulls in); allow exactly the packages under the CUDA licence and nothing else.
      isCudaLicensed = pkg: builtins.all (l: (l.shortName or "") == "CUDA EULA") (lib.toList pkg.meta.license);
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfreePredicate = isCudaLicensed;
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          cargo
          rustc
          rust-analyzer
          clippy
          rustfmt
          # Python baselines in baselines/.
          uv
          # ptxas, as a test-only checker of emitted PTX; never on the compiler path.
          cudaPackages.cuda_nvcc
        ];
        # cargo llvm-cov needs llvm-profdata and llvm-cov from the LLVM rustc was built
        # with; nixpkgs' rustc has no llvm-tools component.
        LLVM_PROFDATA = "${lib.getBin pkgs.rustc.llvmPackages.llvm}/bin/llvm-profdata";
        LLVM_COV = "${lib.getBin pkgs.rustc.llvmPackages.llvm}/bin/llvm-cov";
        # ptxc-cuda dlopens libcuda.so.1, which NixOS keeps in the driver's directory.
        shellHook = ''
          export LD_LIBRARY_PATH="/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        '';
      };
    };
}
