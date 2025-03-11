{ pkgs, ... }:

let
  # let steam-run to take care of FHS environment!
  # https://nixos.wiki/wiki/Nix_Cookbook#Wrapping_packages
  with-steam-run = name: pkgs.writeShellScriptBin name ''
    exec ${pkgs.steam-run}/bin/steam-run ''${VIRTUAL_ENV}/bin/${name} "$@"
  '';
  # if `${VIRTUAL_ENV}/bin/...` or `${DEVENV_ROOT}/.venv/bin/...` does not work,
  # the alternative could be below
  # `exec ${pkgs.steam-run}/bin/steam-run ${pkgs.poetry}/bin/poetry run ${bin} "$@"`

  wrapped-bins = builtins.map (name: name) [
    "python"
    "python3"
    "pip"
    "pip3"
    # if you want to cover more binaries, you could list all via `ls .venv/bin`
  ];
  wrapped-bins-paths = builtins.concatStringsSep "/bin:" wrapped-bins;

  export-path =
    # too bad there is no stdenv.isNixOS but it will do
     ''
      export PATH="${wrapped-bins-paths}/bin:''$PATH"
      export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
    '';
in
{
  # now you don't need this as `wrapped-bins` will handle bdist packages! (probably not all though)
  # env.POETRY_INSTALLER_NO_BINARY = ":all:";

  enterShell = ''
    # this runs after .venv/bin/activate
    ${export-path}

    jupyter notebook --NotebookApp.token="" --NotebookApp.password="" -no-browser

  '';

  # # https://devenv.sh/languages/
  languages.python = {
    enable = true;
    poetry = {
      enable = true;
    };
  };
}