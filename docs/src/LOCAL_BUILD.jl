#   This script is not part of the PosDefManifoldML package.
#   It allows to build the PosDefManifoldML documentation
#   locally from the source code,
#   without actually installing the package.
#   You won't need this script for using the package.
#
#   MIT License
#   Copyright (c) 2019_2023, Marco Congedo, CNRS, Grenobe, France:
#   https://sites.google.com/site/marcocongedo/home
#
#   DIRECTIONS:
#   1) Set the `docs` folder as julia envoronment
#   2) Change the `juliaCodeDir` path here below to the path
#           where the PosDefManifoldML folder is located on your computer.
#   3) Under Linux, replace all '\\' with `/`
#   4) Put the cursor in this unit and hit SHIFT+CTRL+ENTER
#
#   Nota Bene: all you need for building the documentation is actually
#   the 'push!' line and the 'using' line.
#   You can safely delete the rest once
#   you have identified the 'srcDir' to be used in the push command.

begin
  juliaCodeDir = homedir()*"\\Documents\\@ Documenti\\Code\\julia\\"
  scrDir       = juliaCodeDir*"PosDefManifoldML\\src\\"
  docsDir      = juliaCodeDir*"PosDefManifoldML\\docs\\"

  push!(LOAD_PATH, scrDir)
  using PosDefManifoldML
  push!(LOAD_PATH, docsDir)
  using Documenter
  cd(docsDir)
  clipboard("""makedocs(sitename="PosDefManifoldML", modules=[PosDefManifoldML])""")
  @info("\nhit CTRL+V+ENTER on the REPL for building the documentation.");
end
