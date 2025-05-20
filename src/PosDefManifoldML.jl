#   Unit "simulations.jl" of the PosDefManifoldML Package for Julia language
#   v 0.5.7 - last update May 2025
#
#   MIT License
#   Copyright (c) 2019-2025,
#   Marco Congedo, CNRS, Grenobe, France:
#   https://sites.google.com/site/marcocongedo/home

# ? CONTENTS :
#   Main module of the PosDefManifoldML package.
#
#   This pachwge works in conjunction with the PosDefManifold package
#   https://github.com/Marco-Congedo/PosDefManifold.jl

module PosDefManifoldML

using LinearAlgebra, Base.Threads, Random, StatsBase, Statistics, PosDefManifold

using Dates: Dates, now
using GLMNet: GLMNet, glmnet, glmnetcv, GLMNetPath, GLMNetCrossValidation
using Distributions: Distributions, Binomial, cdf, ccdf, Normal
using PermutationTests: PermutationTests, Left, Right, Both
using LIBSVM: LIBSVM, svmpredict, svmtrain, SVC, NuSVC, OneClassSVM, NuSVR, EpsilonSVR,
      LinearSVC, Linearsolver, Kernel
# 2025
using Diagonalizations, Folds, Serialization

# Special instructions and variables
BLAS.set_num_threads(Sys.CPU_THREADS)

# constants #

const titleFont     = "\x1b[32m"
const separatorFont = "\x1b[92m"
const defaultFont   = "\x1b[0m"
const greyFont      = "\x1b[90m"
const 📌            = titleFont*"PosDefManifoldML"*defaultFont
const dice = ("⚀", "⚁", "⚂", "⚃", "⚄", "⚅")

# shortcut to LIBSVM kernels enum type
const Linear 		= Kernel.KERNEL(0)
const Polynomial 	= Kernel.KERNEL(1)
const RadialBasis 	= Kernel.KERNEL(2)
const Sigmoid 		= Kernel.KERNEL(3)
const Precomputed 	= Kernel.KERNEL(4)

# Machine Learning Model types
abstract type MLmodel end # all machine learning models

abstract type PDmodel<:MLmodel end # PD manifold models
abstract type TSmodel<:MLmodel end # tangent space models


# Conditioner types
abstract type Conditioner end # all SPD matrix transformations

# Cross-Validation result type
abstract type CVresult end 

IntVector=Vector{Int}

import Base: show, getindex, length, iterate, isempty
import StatsBase: fit, predict
import StatsAPI.fit!
import LinearAlgebra.normalize!
import PosDefManifold.dim
import DataFrames.transform!

export

    # from this module
    MLmodel,
    PDmodel,
    TSmodel,
    Conditioner,
    CVresult,

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
    barycenter,
    distances,

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
    CVres,
    crval,
    cvSetup,

    # from stats_descriptive.jl
  	confusionMat,
	predictAcc,
    predictErr,
    binaryloss,


    # from stats_inferential.jl
    testCV,

    # from tools.jl
    tsMap,
    tsWeights,
    gen2ClassData,
	rescale!,
    demean!,
    normalize!,
    standardize!,
    saveas,
    load,

    # from conditioners.jl
    Pipeline,
    @pipeline,
    @→,
    Tikhonov,
    Recenter,
    Compress,
    Equalize,
    Shrink,
    fit!,
    transform!,
    includes,
    pickfirst,
    dim

include("conditioners.jl")
include("mdm.jl")
include("enlr.jl")
include("svm.jl")
include("private.jl")
include("stats_descriptive.jl")
include("stats_inferential.jl")
include("tools.jl")
include("cv.jl")


println("\n⭐ "," Welcome to the ", 📌, " package", " ⭐\n")
@info " "
println(" Your Machine `", separatorFont, gethostname(), defaultFont, "` (",Sys.MACHINE, ")")
println(" runs on kernel ",Sys.KERNEL," with word size ",Sys.WORD_SIZE,".")
println(" CPU  Threads: ", separatorFont, Sys.CPU_THREADS, defaultFont)
println(" Base.Threads: ", separatorFont, "$(Threads.nthreads())", defaultFont)
println(" BLAS Threads: ", separatorFont, "$(BLAS.get_num_threads())", "\n", defaultFont)

end # module
