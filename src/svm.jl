#   Unit "libSVM.jl" of the PosDefManifoldML Package for Julia language

#   MIT License
#   Copyright (c) 2019-2025
#   Anton Andreev, Marco Congedo, CNRS, Grenoble, France:

# ? CONTENTS :
#   This unit implements a wrapper to libSVM. It projects data to tangent space
#   and it applies SVM classification using Julia's SVM wrapper.

"""
```julia
abstract type SVMmodel<:TSmodel end
```
Abstract type for **Support-Vector Machine (SVM)**
learning models. See [MLmodel](@ref).
"""
abstract type SVMmodel<:TSmodel end

"""
```julia
mutable struct SVM <: SVMmodel
	metric		:: Metric
	svmType		:: Type
	kernel		:: Kernel.KERNEL
	pipeline 	:: Pipeline
	normalize	:: Union{Function, Tuple, Nothing}
	meanISR		:: Union{HermitianVector, Nothing, UniformScaling}
	vecRange	:: UnitRange
	featDim		:: Int
	svmModel #store the training model from the SVM library
```
SVM machine learning models are incapsulated in this
mutable structure. Fields:

`.metric`, of type
[Metric](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#Metric::Enumerated-type-1),
is the metric that will be adopted to compute the mean used
as base-point for tangent space projection. By default the
Fisher metric is adopted. See [mdm.jl](@ref)
for the available metrics. If the data used to train the model
are not positive definite matrices, but Euclidean feature vectors,
the `.metric` field has no use.  In order to use metrics you need to install the
*PosDefManifold* package.

`svmType`, a generic Type of SVM models used in LIBSVM.
Available types are:
- `SVC`: *C-Support Vector Classification*. The fit time complexity is more
   than quadratic with the number of observations. The multiclass support is handled
   according to a one-vs-one scheme,
- `NuSVC`: *Nu-Support Vector Classification*. Similar to SVC but uses a
   parameter to control the number of support vectors,
- `OneClassSVM`: Unsupervised outlier detection. Estimate the support of a high-dimensional distribution,
- `EpsilonSVR`: *Epsilon-Support Vector Regression*,
- `NuSVR`: *Nu-Support Vector Regression*.
The default is `SVC`, unless labels are not provided while fitting
the model, in which case it defaults to `OneClassSVM`.

`kernel`, a kernel type.
Available kernels are declared as constants in the main module. They are:
- `Linear` 		(default)
- `RadialBasis` 
- `Polynomial`
- `Sigmoid`
- `Precomputed` (not supported).

All other fields do not correspond to arguments passed
upon creation of the model by the default creator.
Instead, they are filled later when a model is created by the
[`fit`](@ref) function:

For the content of fields `vecRange` and `normalize`, please see the documentation
of the [`fit`](@ref) function for the ENLR model.

For the content of the `.meanISR`, `.featDim` and `pipeline` fields please
see the documentation of the [`ENLR`](@ref) structure.

`svmModel` holds the model structure created by LIBSVM when the model is fitted
(declared [here](https://github.com/mpastell/LIBSVM.jl/blob/master/src/LibSVMtypes.jl)).

**Examples**:
```julia
# Note: creating models with the default creator is possible,
# but not useful in general.

using PosDefManifoldML, PosDefManifold

# Create an empty SVM model
m = SVM(Fisher)

# Since the Fisher metric is the default metric,
# this is equivalent to
m = SVM()

# Create an empty SVM model using the logEuclidean metric
m = SVM(logEuclidean)

# Generate some data
PTr, PTe, yTr, yTe = gen2ClassData(10, 30, 40, 60, 80, 0.1);

# Empty models can be passed as first argument of the `fit` function
# to fit a model. For instance, this will fit an SVM model of the same
# kind of `m` and put the fitted model in `m1`:
m1 = fit(m, PTr, yTr)

# In general you don't need this machinery for fitting a model,
# since you can specify a model by creating one on the fly:
m2 = fit(SVM(logEuclidean), PTr, yTr; kernel=Linear)

# which is equivalent to
m2 = fit(m, PTr, yTr; kernel=Linear)

# note that, albeit model `m` has been created as an SVM model
# with the default kernel (RadialBasis),
# you have passed `m` and overwritten the `kernel` type.
# You can also overwrite the `svmType`.
# The metric, instead, cannot be overwritten.
```
"""
mutable struct SVM <: SVMmodel
    	metric        :: Metric
		svmType       :: Type
		kernel        :: Kernel.KERNEL
		pipeline
		normalize
		meanISR
		vecRange
		featDim
		# LIBSVM args
		svmModel #used to store the training model from the SVM library
    function SVM( metric :: Metric=Fisher;
				  svmType 	  	= SVC,
                  kernel  	  	= Linear,
				  pipeline		= nothing,
				  normalize  	= nothing,
				  meanISR 	  	= nothing,
				  vecRange    	= nothing,
				  featDim 	  	= nothing,
				  svmModel 	  	= nothing)
	   	 new(metric, svmType, kernel, pipeline, normalize, meanISR,
		     vecRange, featDim, svmModel)
    end
end



"""
```julia
function fit(model     :: SVMmodel,
               𝐏Tr     :: Union{HermitianVector, Matrix{Float64}},
               yTr     :: IntVector=[];

	# pipeline (data transformations)
	pipeline    :: Union{Pipeline, Nothing} = nothing,

	# parameters for projection onto the tangent space
	w		:: Union{Symbol, Tuple, Vector} = Float64[],
	meanISR 	:: Union{Hermitian, Nothing} = nothing,
	meanInit 	:: Union{Hermitian, Nothing} = nothing,
	vecRange	:: UnitRange = 𝐏Tr isa HermitianVector ? (1:size(𝐏Tr[1], 2)) : (1:size(𝐏Tr, 2)),
	normalize	:: Union{Function, Tuple, Nothing} = normalize!,

	# SVM parameters
	svmType 	:: Type = SVC,
	kernel 		:: Kernel.KERNEL = Linear,
	epsilon 	:: Float64 = 0.1,
	cost		:: Float64 = 1.0,
	gamma 		:: Float64	= 1/_getDim(𝐏Tr, vecRange),
	degree 		:: Int64	= 3,
	coef0 		:: Float64	= 0.,
	nu 			:: Float64 = 0.5,
	shrinking 	:: Bool = true,
	probability	:: Bool = false,
	weights 	:: Union{Dict{Int, Float64}, Nothing} = nothing,
	cachesize 	:: Float64	= 200.0,
	checkArgs	:: Bool = true,

	# Generic and common parameters
	tol		:: Real = 1e-5,
	verbose		:: Bool = true,
	⏩		:: Bool = true)
```

Create and fit a **1-class** or **2-class** support vector machine ([`SVM`](@ref)) machine learning model,
with training data `𝐏Tr`, of type
[ℍVector](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#%E2%84%8DVector-type-1),
and corresponding labels `yTr`, of type [IntVector](@ref).
The label vector can be omitted if the `svmType` is `OneClassSVM`
(see [`SVM`](@ref)).
Return the fitted model as an instance of the [`SVM`](@ref) structure.

!!! warning "Class Labels"
    Labels must be provided using the natural numbers, *i.e.*,
    `1` for the first class, `2` for the second class, etc.

As for all ML models acting in the tangent space,
fitting an SVM model involves computing a mean (barycenter) of all the
matrices in `𝐏Tr`, projecting all matrices onto the tangent space
after parallel transporting them at the identity matrix
and vectorizing them using the
[vecP](https://marco-congedo.github.io/PosDefManifold.jl/dev/riemannianGeometry/#PosDefManifold.vecP)
operation. Once this is done, the support-vector machine is fitted.

**Optional keyword arguments**

For the following keyword arguments see the documentation of the [`fit`](@ref) 
funtion for the ENLR (Elastic Net Logistic Regression) machine learning model:

- `pipeline` (pre-conditioning),
- `w`, `meanISR`, `meanInit`, `vecRange` (tangent space projection),

!!! tip "Euclidean SVM models"
    ML models acting on the tangent space allows to fit a model passing as
    training data `𝐏Tr` directly a matrix of feature vectors,
    where each feature vector is a row of the matrix.
    In this case none of the above keyword arguments are used.

- `normalize` (tangent or feature vectors normalization).

**Optional keyword arguments for fitting the model(s) using LIBSVM.jl**

`svmType` and `kernel` allow to chose among several
available SVM models. See the documentation of the [`SVM`](@ref) structure.

`epsilon`, with default 0.1, is the epsilon in loss function
of the `epsilonSVR` SVM model.

`cost`, with default 1.0, is the cost parameter *C* of `SVC`,
`epsilonSVR`, and `nuSVR` SVM models.

`gamma`, defaulting to 1 divided by the length of the tangent (or feature) vectors,
is the *γ* parameter for `RadialBasis`, `Polynomial` and `Sigmoid` kernels.
The provided argument `gamma` will be ignored if a pre-conditioning `pipeline` 
is passed as argument and if the pipeline changes the dimension of the input matrices, 
thus of the tangent vectors. In this case it will be set to its default value using 
the new dimension. To force the use of the provided `gamma` value instead, 
set	`checkArgs` to false (true by default).

`degree`, with default 3, is the degree for `Polynomial` kernels

`coef0`, zero by default, is a parameter for the `Sigmoid` and `Polynomial` kernel.

`nu`, with default 0.5, is the parameter *ν* of `nuSVC`,
`OneClassSVM`, and `nuSVR` SVM models. It should be in the interval (0, 1].

`shrinking`, true by default, sets whether to use the shrinking heuristics.

`probability`, false by default sets whether to train a `SVC` or `SVR` model
allowing probability estimates.

if a `Dict{Int, Float64}` is passed as `weights` argument, it will be used
to give weights to the classes. By default it is equal to `nothing`, implying
equal weights to all classes.

`cachesize` for the kernel, 200.0 by defaut (in MB), can be increased for
very large problems.

`tol` is the convergence criterion for both the computation
of a mean for projecting onto the tangent space
(if the metric requires an iterative algorithm)
and for the LIBSVM fitting algorithm. Defaults to 1e-5.

If `verbose` is true (default), information is printed in the REPL.
This option is included to allow repeated calls to this function
without crowding the REPL. 

The `⏩` argument (true by default) is passed to the [`tsMap`](@ref)
function for projecting the matrices in `𝐏Tr` onto the tangent space
and to the LIBSVM function that perform the fit in order to run them
in multi-threaded mode.

For further information on tho LIBSVM arguments, refer to the
resources on the LIBSVM package [🎓](@ref).

**See**: [notation & nomenclature](@ref), [the ℍVector type](@ref)

**See also**: [`predict`](@ref), [`crval`](@ref)

**Tutorial**: [Examples using SVM models](@ref)

**Examples**
```julia
using PosDefManifoldML, PosDefManifold

# Generate some data
PTr, PTe, yTr, yTe = gen2ClassData(10, 30, 40, 60, 80, 0.1);

# Fit an SVC SVM model and find the best model by cross-validation:
m = fit(SVM(), PTr, yTr)

# The same but using a pre-conditioning pipeline:
p = @→ Recenter(; eVar=0.999) Compress Shrink(Fisher; radius=0.02)
m = fit(SVM(), PTr, yTr; pipeline=p)

# ... balancing the weights for tangent space mapping
m = fit(SVM(), PTr, yTr; w=:b)

# ... using the log-Eucidean metric for tangent space projection
m = fit(SVM(logEuclidean), PTr, yTr)

# ... using the linear kernel
m = fit(SVM(logEuclidean), PTr, yTr, kernel=Linear)

# or

m = fit(SVM(logEuclidean; kernel=Linear), PTr, yTr)

# ... using the Nu-Support Vector Classification
m = fit(SVM(logEuclidean), PTr, yTr, kernel=Linear, svmtype=NuSVC)

# or

m = fit(SVM(logEuclidean; kernel=Linear, svmtype=NuSVC), PTr, yTr)

# N.B. all other keyword arguments must be passed to the fit function
# and not to the SVM constructor.
```
"""
function fit(model     :: SVMmodel,
               𝐏Tr     :: Union{ℍVector, Matrix{Float64}},
               yTr     :: IntVector=[];

            # pipeline
            pipeline    :: Union{Pipeline, Nothing} = nothing,

			# parameters for projection onto the tangent space
			w			:: Union{Symbol, Tuple, Vector} = Float64[],
			meanISR     :: Union{ℍ, Nothing} = nothing,
			meanInit    :: Union{ℍ, Nothing} = nothing,
			vecRange    :: UnitRange = 𝐏Tr isa ℍVector ? (1:size(𝐏Tr[1], 2)) : (1:size(𝐏Tr, 2)),
			normalize	:: Union{Function, Tuple, Nothing} = normalize!,

			# paramters for LIBSVM svmtrain function
			svmType     :: Type 			= SVC,
			kernel      :: Kernel.KERNEL	= Linear,
			epsilon     :: Float64 	  		= 0.1,
			cost        :: Float64 	  		= 1.0,
			gamma       :: Float64 	  		= 1/_getDim(𝐏Tr, vecRange),
			degree      :: Int64   	  		= 3,
			coef0		:: Float64	  		= 0.,
			nu			:: Float64	  		= 0.5,
			shrinking   :: Bool		  		= true,
			probability :: Bool		  		= false,
			weights     :: Union{Dict{Int, Float64}, Nothing} = nothing,
			cachesize   :: Float64     		= 200.0,
			checkArgs	:: Bool 			= true,

			# Generic and common parameters
			tol         :: Real 		  	= 1e-5,
			verbose     :: Bool 		  	= true,
			⏩  	      :: Bool 			   = true)

    ⌚=now() # get the time in ms
    ℳ=deepcopy(model) # output model

	# checks
	isempty(yTr) && svmType≠OneClassSVM && throw(ArgumentError, "only for the `OneClassSVM` svmtpe the `y` vector may be empty")

	# overwrite fields in `ℳ` if the user has passed them here as arguments,
	# otherwise use as arguments the values in the fields of `ℳ`, e.g., the default
	if svmType ≠ SVC ℳ.svmType = svmType else svmType = ℳ.svmType end
	if kernel ≠ Linear ℳ.kernel = kernel else kernel = ℳ.kernel end

	# check w argument and get weights for input matrices
    (w=_getWeights(w, yTr, "fit ("*_model2Str(ℳ)*" model)")) == nothing && return

	# other checks. Note: for this model the check on feature weights is forced to succeed since those weights are not supported
    𝐏Tr isa ℍVector ? nObs=length(𝐏Tr) : nObs=size(𝐏Tr, 1)
    !_check_fit(ℳ, nObs, length(yTr), length(w), length(yTr), "SVM") && return

    # apply pre-conditioning pipeline and reset the gamma keyword arg to fit the model 
    # if the pipeline change the input matrix dimension
    if 𝐏Tr isa ℍVector # only for tangent vectors (not if 𝐏Tr is a matrix of tangent vectors)     

        originalDim = size(𝐏Tr[1], 2)
        # pipeline (pre-conditioners)
        if !(pipeline===nothing)
            verbose && println(greyFont, "Fitting pipeline...")
            ℳ.pipeline = fit!(𝐏Tr, pipeline)
        end

        newDim = size(𝐏Tr[1], 2) # some pre-conditioners can change the dimension
        if newDim ≠ originalDim && checkArgs # reset the vecrange and gamma arg to the default using the new dimension
			vecRange = 1:newDim
			gamma = 1/_getDim(𝐏Tr, vecRange)
        end
    end


	# project data onto the tangent space or just copy the features if 𝐏Tr is a matrix
	verbose && println(greyFont, "Lifting SPD matrices onto the tangent space...")
	X = _getTSvec_fit!(ℳ, 𝐏Tr, meanISR, meanInit, tol, w, vecRange, false, verbose, ⏩)

	# Normalize data if the user asks for. 
	# For SVM models the tangent vectors are the columns of X, thus dims=1
	_normalizeTSvec!(X, normalize; dims=1)
		
    # set multi-threading in line with LIBSVM functioning
	⏩==true ? nthreads=Sys.CPU_THREADS : nthreads=1

    verbose && println(defaultFont, "Fitting SVM model...", greyFont)
    model = svmtrain(X, yTr;
				 svmtype     = svmType,
				 kernel      = kernel,
				 epsilon     = epsilon,
				 cost        = cost,
				 gamma       = gamma,
				 tolerance   = tol,
				 verbose     = verbose,
				 degree      = degree,
				 coef0	     = coef0,
				 nu		     = nu,
				 shrinking   = shrinking,
				 probability = probability,
				 weights     = weights,
				 cachesize   = cachesize,
				 nt          = nthreads);

	# write some fields in output model struct				 
    ℳ.svmModel 	= model
	ℳ.svmType  	= svmType
	ℳ.kernel   	= kernel
	ℳ.vecRange 	= vecRange
	ℳ.featDim  	= size(X, 1)
	ℳ.normalize = normalize

    verbose && println(defaultFont, "Done in ", now()-⌚,".")
    return ℳ
end


"""
```julia
function predict(model :: SVMmodel,
			𝐏Te	:: Union{ℍVector, Matrix{Float64}},
			what	:: Symbol = :labels;
		meanISR	:: Union{ℍ, Nothing, UniformScaling} = nothing,
		pipeline:: Union{Pipeline, Nothing} = nothing,
		verbose	:: Bool = true,
		⏩	:: Bool = true)
```

Compute predictions given an [`SVM`](@ref) `model` trained (fitted) on 2 classes
and a testing set of *k* positive definite matrices `𝐏Te` of type
[ℍVector](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#%E2%84%8DVector-type-1).

For the meaning of arguments `what`, `meanISR`, `pipeline` and `verbose`,
see the documentation of the [`predict`](@ref) function
for the ENLR model.

If ⏩ = true (default) and `𝐏Te` is an ℍVector type, the projection onto the
tangent space will be multi-threaded. Also, the prediction of the LIBSVM.jl prediction function
will be multi-threaded.

**See**: [notation & nomenclature](@ref), [the ℍVector type](@ref)

**See also**: [`fit`](@ref), [`crval`](@ref), [`predictErr`](@ref)

**Examples** 

see the examples for the [`predict`](@ref) function
for the ENLR model; the syntax is identical, only the model
used there has to be changed with a `SVMmodel`.
"""
function predict(model   :: SVMmodel,
                 𝐏Te     :: Union{ℍVector, Matrix{Float64}},
                 what    :: Symbol = :labels;
			meanISR	:: Union{ℍ, Nothing, UniformScaling} = nothing,
			pipeline:: Union{Pipeline, Nothing} = nothing,
            verbose :: Bool = true,
            ⏩      :: Bool = true)

    ⌚=now() # time in milliseconds

    # checks
    _whatIsValid(what, "predict ("*_model2Str(model)*")") || return 

	verbose && println(greyFont, "Applying pipeline...")
	_applyPipeline!(𝐏Te, pipeline, model)

	# projection onto the tangent space. NB `false` put feature vecs in cols of X
	X = _getTSvec_Predict!(model, 𝐏Te, meanISR, model.vecRange, false, verbose, ⏩)

   	# Normalize data if the user have asked for while fitting the model. 
	# For SVM models the tangent vectors are the cols of X, thus dims=1
    _normalizeTSvec!(X, model.normalize; dims=1)

	# set multi-threading in line with LIBSVM functioning
	⏩==true ? nthreads=Sys.CPU_THREADS : nthreads=1

    # prediction
	verbose && println("Predicting using "*_model2Str(model)*" model...")
	(labels, π) = svmpredict(model.svmModel, X; nt=nthreads)

	if     what == :functions     || what == :f 🃏=π[1, :]
    elseif what == :labels 		  || what == :l 🃏=labels
    elseif what == :probabilities || what == :p 🃏=[softmax([π[1, i], 0]) for i=1:size(π, 2)]
    end

    verbose && println(defaultFont, "Done in ", now()-⌚,".")
    verbose && println(titleFont, "\nPredicted ",_what2Str(what),":", defaultFont)
    return 🃏
end

# ++++++++++++++++++++  Show override  +++++++++++++++++++ # (REPL output)
function Base.show(io::IO, ::MIME{Symbol("text/plain")}, M::SVM)
	if M.svmModel == nothing
        println(io, greyFont, "\n↯ SVM LIBSVM machine learning model")
        println(io, "⭒  ⭒    ⭒       ⭒          ⭒")
        println(io, ".metric  : ", string(M.metric))
        println(io, ".svmType : ", "$(_model2Str(M))")
		println(io, ".kernel  : ", "$(string(M.kernel))", defaultFont)
        println(io, "Unfitted model")
        return
    end

	println(io, titleFont, "\n↯ SVM "*_model2Str(M)*" machine learning model")
    println(io, separatorFont, "⭒  ⭒    ⭒       ⭒          ⭒", defaultFont)
    println(io, "type    : PD Tangent Space model")
    println(io, "features: tangent vectors of length $(M.featDim)")
    println(io, "classes : 2, with labels (1, 2)")
    println(io, separatorFont, "Fields  : ")
	# # #
	println(io, separatorFont," .featDim     ", defaultFont, "$(M.featDim)")
	println(io, greyFont, " Tangent Space Parametrization", defaultFont)
	println(io, separatorFont," .metric      ", defaultFont, string(M.metric))
	if M.meanISR == nothing
        println(io, greyFont, " .meanISR      not created")
    elseif M.meanISR isa Hermitian
        n=size(M.meanISR, 1)
        println(io, separatorFont," .meanISR     ", defaultFont, "$(n)x$(n) Hermitian matrix")
    end
	println(io, separatorFont," .vecRange    ", defaultFont, "$(M.vecRange)")
	println(io, separatorFont," .normalize   ", defaultFont, "$(string(M.normalize))")
    # # #
	println(io, greyFont, " SVM Parametrization", defaultFont)
    println(io, separatorFont," .svmType     ", defaultFont, "$(_model2Str(M))")
	println(io, separatorFont," .kernel      ", defaultFont, "$(string(M.kernel))")
    println(io, separatorFont," .svmModel ", defaultFont, "   LIBSVM model struct")
end
