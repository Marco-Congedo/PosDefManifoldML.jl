#   Unit "simulations.jl" of the PosDefManifoldML Package for Julia language
#   v 0.2.1 - last update 18th of October 2019
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

 __precompile__()

module  PosDefManifoldML

using LinearAlgebra, Base.Threads, Random, Statistics, PosDefManifold
using Dates:now
using GLMNet:GLMNet, glmnet, glmnetcv, GLMNetPath, GLMNetCrossValidation
using Distributions:Distributions, Binomial

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
abstract type MLmodel end # all machine learning models
abstract type PDmodel<:MLmodel end # PD manifold models
abstract type TSmodel<:MLmodel end # tangent space models

IntVector=Vector{Int}

import Base:show
import GLMNet.predict
import Distributions.fit

export

    # from this module
    MLmodel,
    PDmodel,
    TSmodel,
    IntVector,

    # from mdm.jl
    MDMmodel,
    MDM,
    fit,
    predict,
    getMean,
    getDistances,

    # from enlr.jl
    ENLRmodel,
    ENLR,

    # from cv.jl
    CVacc,
    cvAcc,
    cvSetup,

    # from tools.jl
    tsMap,
    gen2ClassData,
    predictErr

include("mdm.jl")
include("enlr.jl")
include("cv.jl")
include("tools.jl")


println("\n⭐ "," Welcome to the",titleFont," ",📌," ",defaultFont,"package", " ⭐\n")
@info " "
println(" Your Machine `",gethostname(),"` (",Sys.MACHINE, ")")
println(" runs on kernel ",Sys.KERNEL," with word size ",Sys.WORD_SIZE,".")
println(" CPU  Threads: ",Sys.CPU_THREADS)
# Sys.BINDIR # julia bin directory
println(" Base.Threads: ", "$(Threads.nthreads())")
println(" BLAS Threads: ", "$(Sys.CPU_THREADS-Threads.nthreads())", "\n")

end # module
