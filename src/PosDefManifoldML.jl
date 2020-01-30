#   Unit "simulations.jl" of the PosDefManifoldML Package for Julia language
#   v 0.3.4 - last update 31st of January 2020
#
#   MIT License
#   Copyright (c) 2019,
#   Marco Congedo, CNRS, Grenobe, France:
#   https://sites.google.com/site/marcocongedo/home

# ? CONTENTS :
#   Main module of the PosDefManifoldML package.
#
#   This pachwge works in conjunction with the PosDefManifold package
#   https://github.com/Marco-Congedo/PosDefManifold.jl

 __precompile__()

module PosDefManifoldML

using LinearAlgebra, Base.Threads, Random, StatsBase, Statistics, PosDefManifold
using Dates:now
using GLMNet:GLMNet, glmnet, glmnetcv, GLMNetPath, GLMNetCrossValidation
using Distributions:Distributions, Binomial
using LIBSVM: svmpredict, svmtrain, SVC, NuSVC, OneClassSVM, NuSVR, EpsilonSVR,
      LinearSVC, Linearsolver, Kernel


# Special instructions and variables
BLAS.set_num_threads(Sys.CPU_THREADS)

# constants #
const 📌            = "PosDefManifoldML"
const titleFont     = "\x1b[32m"
const separatorFont = "\x1b[92m"
const defaultFont   = "\x1b[0m"
const greyFont      = "\x1b[90m"
const dice = ("⚀", "⚁", "⚂", "⚃", "⚄", "⚅")

# shortcut to LIBSVM kernels enum type
const Linear 		= Kernel.KERNEL(0)
const Polynomial 	= Kernel.KERNEL(1)
const RadialBasis 	= Kernel.KERNEL(2)
const Sigmoid 		= Kernel.KERNEL(3)
const Precomputed 	= Kernel.KERNEL(4)


# types #
abstract type MLmodel end # all machine learning models
abstract type PDmodel<:MLmodel end # PD manifold models
abstract type TSmodel<:MLmodel end # tangent space models

IntVector=Vector{Int}

import Base:show
#import GLMNet.predict
#import Distributions.fit
# import LIBSVM.predict
import StatsBase:fit, predict

export

    # from this module
    MLmodel,
    PDmodel,
    TSmodel,
    IntVector,
	Linear,
	Polynomial,
	RadialBasis,
	Sigmoid,
	Precomputed,

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

	# from libSVM.jl
	SVMmodel,
	SVM,
	SVC,
	NuSVC,
	OneClassSVM,
	NuSVR,
	EpsilonSVR,
	LinearSVC,
	Linearsolver,
	Kernel,

    # from cv.jl
    CVacc,
    cvAcc,
    cvSetup,

    # from tools.jl
    tsMap,
    tsWeights,
    gen2ClassData,
    predictErr,
	rescale!

include("mdm.jl")
include("enlr.jl")
include("cv.jl")
include("tools.jl")
include("svm.jl")


println("\n⭐ "," Welcome to the", titleFont," ",📌,".jl ",defaultFont,"package", " ⭐\n")
@info " "
println(" Your Machine `",gethostname(),"` (",Sys.MACHINE, ")")
println(" runs on kernel ",Sys.KERNEL," with word size ",Sys.WORD_SIZE,".")
println(" CPU  Threads: ", Sys.CPU_THREADS)
println(" Base.Threads: ", "$(Threads.nthreads())")
println(" BLAS Threads: ", "$(Sys.CPU_THREADS)", "\n")

end # module
