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
          # cuBLAS, the matrix-multiply baseline only ptxc-bench loads; never on the
          # compiler path.
          cudaPackages.libcublas
        ];
        # cargo llvm-cov needs llvm-profdata and llvm-cov from the LLVM rustc was built
        # with; nixpkgs' rustc has no llvm-tools component.
        LLVM_PROFDATA = "${lib.getBin pkgs.rustc.llvmPackages.llvm}/bin/llvm-profdata";
        LLVM_COV = "${lib.getBin pkgs.rustc.llvmPackages.llvm}/bin/llvm-cov";
        # ptxc-cuda dlopens libcuda.so.1, which NixOS keeps in the driver's directory,
        # and ptxc-bench dlopens libcublas.so from its package.
        # uv's standalone Python looks for CA certificates where NixOS keeps none, so
        # baselines' downloads (torchvision weights) need SSL_CERT_FILE; stdenv's setup
        # replaces the variable, so the hook sets it after.
        shellHook = ''
          export LD_LIBRARY_PATH="/run/opengl-driver/lib:${lib.getLib pkgs.cudaPackages.libcublas}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          export SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
        '';
      };
    };
}
