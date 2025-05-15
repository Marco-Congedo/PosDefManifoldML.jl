#   This script is not part of the PosDefManifoldML package.
#   It allows to build the PosDefManifoldML package
#   and its documentation locally from the source code,
#   without actually installing the package.
#   It is useful for developing purposes using the Julia
#   `Revise` package (that you need to have installed on your PC,
#   together with the `Documenter` package for building the documentation).
#   You won't need this script for using the package.
#
#   MIT License
#   Copyright (c) 2019_2025, Marco Congedo, CNRS, Grenobe, France:
#   https://sites.google.com/site/marcocongedo/home
#
#   DIRECTIONS:
#   1) If you have installed PosDefManifoldML from github or Julia registry,
#      uninstall it.
#   2) Change the `juliaCodeDir` path here below to the path
#           where the PosDefManifoldML folder is located on your computer.
#   3) Under Linux, replace all '\\' with `/`
#   4) Put the cursor in this unit and hit SHIFT+CTRL+ENTER
#
#   Nota Bene: all you need for building the package is actually
#   the 'push!' line and the 'using' line.
#   You can safely delete the rest once
#   you have identified the 'srcDir' to be used in the push command.

begin
  juliaCodeDir = homedir()*"\\Documents\\Documenti\\Code\\julia\\"
  scrDir       = juliaCodeDir*"PosDefManifoldML2\\src\\" ### whatch 2
  docsDir      = juliaCodeDir*"PosDefManifoldML2\\docs\\" ### whatch 2

  push!(LOAD_PATH, scrDir)
  using LinearAlgebra,
        Documenter, Statistics, PosDefManifold, Revise, PosDefManifoldML, 
        Diagonalizations, Folds, StatsAPI, Serialization,
        PermutationTests 

#  cd(docsDir)
#  clipboard("""makedocs(sitename="PosDefManifoldML", modules=[PosDefManifoldML])""")
#  @info("\nhit CTRL+V+ENTER on the REPL for building the documentation.");
end
