
{
  description = "Application packaged using poetry2nix";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.poetry2nix = {
    url = "github:nix-community/poetry2nix";
    inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # see https://github.com/nix-community/poetry2nix/tree/master#api for more functions and examples.
        inherit (poetry2nix.legacyPackages.${system}) mkPoetryApplication;
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        packages = {
          myapp = mkPoetryApplication { projectDir = self; };
          default = self.packages.${system}.myapp;
        };

        devShells.default = pkgs.mkShell rec {
          packages = [ poetry2nix.packages.${system}.poetry ];
          buildInputs = with pkgs; [
            linuxPackages.nvidia_x11
            cudaPackages.cudnn
            cudaPackages.nccl
            cudaPackages.cudatoolkit
            python311
            poetry
            zlib
            rustup
            curl
            leveldb
            stdenv.cc.cc.lib
          ];

          CUDA_TOOLKIT = "${pkgs.cudaPackages.cudatoolkit}/lib";

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;

          shellHook = ''
            export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_TOOLKIT"
          '';
        };
      });
}
