push!(LOAD_PATH,"../src/")
using Documenter, PosDefManifoldML

makedocs(
   sitename="PosDefManifoldML",
   authors="Saloni Jain, Marco Congedo",
   modules=[PosDefManifoldML],
   pages =
   [
      "index.md",
      "Tutorial" => "tutorial.md",
      "Main Module" => "MainModule.md",
      "Training-Testing" => "train_test.md",
      "Tools" => "tools.md",

      "Machine Learning Models" => Any[
                           "Minimum Distance to Mean" => "mdm.md",
      ],
   ]
)

deploydocs(
   # root
   # target = "build", # add this folder to .gitignore!
   repo = "github.com/Marco-Congedo/PosDefManifoldML.jl.git",
   # branch = "gh-pages",
   # osname = "linux",
   # deps = Deps.pip("pygments", "mkdocs"),
   # devbranch = "dev",
   # devurl = "dev",
   # versions = ["stable" => "v^", "v#.#", devurl => devurl],
)
