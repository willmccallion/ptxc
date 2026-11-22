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
        # ptxc-cuda dlopens libcuda.so.1, which NixOS keeps in the driver's directory.
        shellHook = ''
          export LD_LIBRARY_PATH="/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        '';
      };
    };
}
