push!(LOAD_PATH,"../src/")
push!(LOAD_PATH,"docs/src/")
using Documenter, PosDefManifoldML

makedocs(
   sitename="PosDefManifoldML",
   authors="Marco Congedo, Saloni Jain, Fahim Doumi, Anton Andreev",
   format = Documenter.HTML(repolink = "..."),
   modules=[PosDefManifoldML],
   remotes = nothing,
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
                  "Support-Vector Machine" => "svm.md",
      ],
      "Fit, Predict, CV" => "cv.md",
      "Statistics" => Any[
         "Descriptive" => "stats_descriptive.md",
         "Inferential" => "stats_inferential.md",
      ],    
      "Conditioners & Pipelines" => "conditioners.md",
      "How to contribute" => "contribute.md", 
   ]
)

deploydocs(
   # root
   # target = "build", # add this folder to .gitignore!
   repo = "github.com/Marco-Congedo/PosDefManifoldML.jl.git",
   branch = "gh-pages",
   # osname = "linux",
   # deps = Deps.pip("pygments", "mkdocs"),
   devbranch = "dev",
   devurl = "dev",
   push_preview = true,
   # versions = ["stable" => "v^", "v#.#", devurl => devurl],
)
