begin
  juliaCodeDir= homedir()*"\\Documents\\Code\\julia\\"
  scrDir      = juliaCodeDir*"PosDefManifoldML\\src\\"
  docsDir     = juliaCodeDir*"PosDefManifoldML\\docs\\"

  push!(LOAD_PATH, scrDir)

  using LinearAlgebra, Statistics, Base.Threads, Random,
        PosDefManifold, Revise, PosDefManifoldML


  cd(docsDir)
  clipboard("""include("make.jl")""")
  @info("\nhit CTRL+V+ENTER on the REPL for building the documentation.");

end
