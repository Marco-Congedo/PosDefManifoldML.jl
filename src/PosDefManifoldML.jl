#   Unit "simulations.jl" of the PosDefManifoldML Package for Julia language
#   v 0.0.1 - last update 28th of September 2019
#
#   MIT License
#   Copyright (c) 2019,
#   Saloni Jain, Indian Institute of Technology, Kharagpur, India
#   Marco Congedo, CNRS, Grenobe, France:
#   https://sites.google.com/site/marcocongedo/home

# ? CONTENTS :
#   Main module of the PosDefManifoldML package.
#
#   This pachwge works in conjunction with the PosDefManifold package
#   https://github.com/Marco-Congedo/PosDefManifold.jl

module  PosDefManifoldML

using   LinearAlgebra, Base.Threads, Random, Statistics, PosDefManifold

# Special instructions and variables
BLAS.set_num_threads(Sys.CPU_THREADS-Threads.nthreads())

# constants #
const 📌            = "PosDefManifoldML"
const titleFont     = "\x1b[32m"
const separatorFont = "\x1b[92m"
const defaultFont   = "\x1b[0m"
const greyFont      = "\x1b[90m"
const dice = ("⚀", "⚁", "⚂", "⚃", "⚄", "⚅")

# types #
abstract type MLmodel end

IntVector=Vector{Int}

import Base: show

export

    # from this module
    MLmodel,
    IntVector,

    # from mdm.jl
    MDM,
    getMeans,
    getDistances,
    CV_mdm,

    # from train_test.jl
    fit!,
    predict,
    CVscore,

    # from tools.jl
    projectOnTS,
    CVsetup,
    gen2ClassData

include("tools.jl")
include("mdm.jl")
include("train_test.jl")

println("\n⭐ "," Welcome to the",titleFont," ",📌," ",defaultFont,"package", " ⭐\n")
@info " "
println(" Your Machine `",gethostname(),"` (",Sys.MACHINE, ")")
println(" runs on kernel ",Sys.KERNEL," with word size ",Sys.WORD_SIZE,".")
println(" CPU  Threads: ",Sys.CPU_THREADS)
# Sys.BINDIR # julia bin directory
println(" Base.Threads: ", "$(Threads.nthreads())")
println(" BLAS Threads: ", "$(Sys.CPU_THREADS-Threads.nthreads())", "\n")

end # module
