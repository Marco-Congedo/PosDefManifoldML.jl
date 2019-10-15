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
      "Tools" => "tools.md",

      "ML Models: PD Manifold" => Any[
                  "Minimum Distance to Mean" => "mdm.md",
      ],
      "ML Models: PD Tangent Space" => Any[
                  "Elastic-Net Logistic Regression" => "enlr.md",
      ],
      "fit, predict, cv" => "cv.md",

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
