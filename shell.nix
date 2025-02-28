{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python3
    pkgs.python3Packages.pandas
    pkgs.python3Packages.jupyterlab
    pkgs.python3Packages.deep-translator
    pkgs.python3Packages.tqdm
    pkgs.python3Packages.nltk
    # Include any other dependencies you might need
  ];

}

