push!(LOAD_PATH,"../src/")
using Documenter, PosDefManifoldML

makedocs(
   sitename="PosDefManifoldML",
   authors="Saloni Jain, Marco Congedo",
   modules=[PosDefManifoldML],
   pages =
   [
      "index.md",
      "Tutorials" => "tutorial.md",
      "Main Module" => "MainModule.md",
      "Training-Testing" => "train_test.md",
      "Tools" => "tools.md",

      "Machine Learning Models" => Any[
                           "Minimum Distance to Mean" => "mdm.md",
      ],
   ]
)

#deploydocs(
#    repo = "https://github.com/tosalonijain/RiemannianML.git"
#)
