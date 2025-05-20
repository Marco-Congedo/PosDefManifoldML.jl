#   Unit "enlr.jl" of the PosDefManifoldML Package for Julia language

#   MIT License
#   Copyright (c) 2019-2025
#   Marco Congedo, CNRS, Grenoble, France:
#   https://sites.google.com/site/marcocongedo/home

# ? CONTENTS :
#   This unit implements the elastic net logistic regression (ENLR)
#   machine learning classifier, including the lasso LR (default) and Ridge LR
#   as specific instances.

"""
```julia
abstract type ENLRmodel<:TSmodel end
```

Abstract type for **Elastic Net Logistic Rgression (ENLR)**
machine learning models. See [MLmodel](@ref).
"""
abstract type ENLRmodel<:TSmodel end


"""
```julia
mutable struct ENLR <: ENLRmodel
    metric      :: Metric = Fisher;
    alpha       :: Real = 1.0
    pipeline    :: Pipeline
    normalize	:: Union{Function, Tuple, Nothing}
    intercept   :: Bool
    meanISR     :: Union{Hermitian, Nothing, UniformScaling}
    vecRange    :: UnitRange
    featDim     :: Int
    # GLMNet Models
    path        :: GLMNet.GLMNetPath
    cvλ         :: GLMNet.GLMNetCrossValidation
    best        :: GLMNet.GLMNetPath
end
```

ENLR machine learning models are incapsulated in this
mutable structure. Fields:

`.metric`, of type
[Metric](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#Metric::Enumerated-type-1),
is the metric that will be adopted to compute the mean used
as base-point for tangent space projection. By default the
Fisher metric is adopted. See [mdm.jl](@ref)
for the available metrics. If the data used to train the model
are not positive definite matrices, but Euclidean feature vectors,
the `.metric` field has no use. In order to use metrics you need to install the
*PosDefManifold* package.

`.alpha` is the hyperparameter in ``[0, 1]`` trading-off
the **elestic-net model**. *α=0* requests a pure **ridge** model and
*α=1* a pure **lasso** model. By default, *α=1* is specified (lasso model).
This argument is usually passed as parameter to
the [`fit`](@ref) function, defaulting therein to *α=1* too.
See the examples here below.

All other fields do not correspond to arguments passed
upon creation of the model by the default creator.
Instead, they are filled later when a model is created by the
[`fit`](@ref) function:

The field `pipeline`, of type [`Pipeline`](@ref), holds an optional
sequence of data pre-conditioners to be applied to the data. 
The pipeline is learnt when a ML model is fitted - see [`fit`](@ref) - 
and stored in the model. If the pipeline is fitted, it is 
automatically applied to the data at each call of the [`predict`](@ref) function.    

For the content of fields `normalize`, `intercept`, `meanISR` and `vecRange`,
please see the documentation of the [`fit`](@ref) function for the ENLR model.

if the data used to train the model are positive definite matrices,
`.featDim` is the length of the vectorized tangent vectors.
This is given by *n(n+1)÷2* (integer division), where *n*
is the dimension of the original PD matrices on which the model is applied
once they are mapped onto the tangent space.
If feature vectors are used to train the model, `.featDim` is the length
of these vectors. If for fitting the model you have provided an optional
keyword argument `vecRange`, `.featDim` will be reduced accordingly.

`.path` is an instance of the following `GLMNetPath`
structure of the
[GLMNet.jl](https://github.com/JuliaStats/GLMNet.jl) package.
It holds the regularization path
that is created when the [`fit`](@ref) function is invoked
with optional keyword parameter `fitType` = `:path` or = `:all`:

```julia
struct GLMNetPath{F<:Distribution}
    family::F                        # Binomial()
    a0::Vector{Float64}              # intercept values for each solution
    betas::CompressedPredictorMatrix # coefficient values for each solution
    null_dev::Float64                # Null deviance of the model
    dev_ratio::Vector{Float64}       # R^2 values for each solution
    lambda::Vector{Float64}          # lambda values for each solution
    npasses::Int                     # actual number of passes over the
                                     # data for all lamda values
end
```

`.cvλ` is an instance of the following `GLMNetCrossValidation`
structure of the [GLMNet.jl](https://github.com/JuliaStats/GLMNet.jl) package.
It holds information about the
cross-validation used for estimating the optimal lambda
hyperparameter by the [`fit`](@ref) function when this is invoked
with optional keyword parameter `fitType` = `:best` (default) or = `:all`:

```julia
struct GLMNetCrossValidation
    path::GLMNetPath            # the cv path
    nfolds::Int                 # the number of folds for the cv
    lambda::Vector{Float64}     # lambda values for each solution
    meanloss::Vector{Float64}   # mean loss for each solution
    stdloss::Vector{Float64}    # standard deviation of the mean losses
end
```

`.best` is an instance of the `GLMNetPath`
structure (see above). It holds the model with the optimal
lambda parameter found by cross-validation
that is created by default
when the [`fit`](@ref) function is invoked.


**Examples**:
```julia
# Note: creating models with the default creator is possible,
# but not useful in general.

using PosDefManifoldML, PosDefManifold

# Create an empty lasso model
m = ENLR(Fisher)

# Since the Fisher metric is the default metric,
# this is equivalent to
m = ENLR()

# Create an empty ridge model using the logEuclidean metric
m = ENLR(logEuclidean; alpha=0)

# Empty models can be passed as first argument of the `fit` function
# to fit a model. For instance, this will fit a ridge model of the same
# kind of `m` and put the fitted model in `m1`:
m1 = fit(m, PTr, yTr)

# In general you don't need this machinery for fitting a model,
# since you can specify a model by creating one on the fly:
m2 = fit(ENLR(logEuclidean; alpha=0), PTr, yTr)

# which is equivalent to
m2 = fit(ENLR(logEuclidean), PTr, yTr; alpha=0)

# Note that, albeit model `m` has been created as a ridge model,
# you have passed `m` and overwritten the `alpha` hyperparameter.
# The metric, instead, cannot be overwritten.
```
"""
mutable struct ENLR <: ENLRmodel
    metric   :: Metric
    alpha    :: Real
    pipeline
    normalize
    intercept
    meanISR
	vecRange
    featDim
    path
    cvλ
    best
    function ENLR(metric    :: Metric=Fisher;
               alpha        :: Real = 1.0,
               pipeline     = nothing,
               normalize    = nothing,
               intercept    = nothing,
               meanISR      = nothing,
			   vecRange     = nothing,
               featDim      = nothing,
               path         = nothing,
               cvλ          = nothing,
               best         = nothing)
        new(metric, alpha, pipeline, normalize, intercept,
            meanISR, vecRange, featDim, path, cvλ, best)
    end
end


"""
```julia
function fit(model	:: ENLRmodel,
             𝐏Tr	 :: Union{HermitianVector, Matrix{Float64}},
             yTr	:: IntVector;

    # pipeline (data transformations)
    pipeline    :: Union{Pipeline, Nothing} = nothing,

    # parameters for projection onto the tangent space
    w           :: Union{Symbol, Tuple, Vector} = Float64[],
    meanISR     :: Union{Hermitian, Nothing, UniformScaling} = nothing,
    meanInit    :: Union{Hermitian, Nothing} = nothing,
    vecRange    :: UnitRange = 𝐏Tr isa ℍVector ? (1:size(𝐏Tr[1], 2)) : (1:size(𝐏Tr, 2)),
    normalize	:: Union{Function, Tuple, Nothing} = normalize!,

    # arguments for `GLMNet.glmnet` function
    alpha           :: Real = model.alpha,
    weights         :: Vector{Float64} = ones(Float64, length(yTr)),
    intercept       :: Bool = true,
    fitType         :: Symbol = :best,
    penalty_factor  :: Vector{Float64} = ones(Float64, _getDim(𝐏Tr, vecRange)),
    constraints     :: Matrix{Float64} = [x for x in (-Inf, Inf), y in 1:_getDim(𝐏Tr, vecRange)],
    offsets         :: Union{Vector{Float64}, Nothing} = nothing,
    dfmax           :: Int = _getDim(𝐏Tr, vecRange),
    pmax            :: Int = min(dfmax*2+20, _getDim(𝐏Tr, vecRange)),
    nlambda         :: Int = 100,
    lambda_min_ratio:: Real = (length(yTr)*2 < _getDim(𝐏Tr, vecRange) ? 1e-2 : 1e-4),
    lambda          :: Vector{Float64} = Float64[],
    maxit           :: Int = 1000000,
    algorithm       :: Symbol = :newtonraphson,
    checkArgs       :: Bool = true,

    # selection method
    λSelMeth        :: Symbol = :sd1,

    # arguments for `GLMNet.glmnetcv` function
    nfolds          :: Int = min(10, div(size(yTr, 1), 3)),
    folds           :: Vector{Int} =
    begin
        n, r = divrem(size(yTr, 1), nfolds)
        shuffle!([repeat(1:nfolds, outer=n); 1:r])
    end,
    parallel        :: Bool=true,

    # Generic and common parameters
    tol             :: Real = 1e-5,
    verbose         :: Bool = true,
    ⏩              :: Bool = true,
)
```

Create and fit an **2-class** elastic net logistic regression ([`ENLR`](@ref)) machine learning model,
with training data `𝐏Tr`, of type
[ℍVector](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#%E2%84%8DVector-type-1),
and corresponding labels `yTr`, of type [IntVector](@ref).
Return the fitted model(s) as an instance of the [`ENLR`](@ref) structure.

!!! warning "Class Labels"
    Labels must be provided using the natural numbers, *i.e.*,
    `1` for the first class, `2` for the second class, etc.

As for all ML models acting in the tangent space,
fitting an ENLR model involves computing a mean (barycenter) of all the
matrices in `𝐏Tr`, projecting all matrices onto the tangent space
after parallel transporting them at the identity matrix
and vectorizing them using the
[vecP](https://marco-congedo.github.io/PosDefManifold.jl/dev/riemannianGeometry/#PosDefManifold.vecP)
operation. Once this is done, the ENLR is fitted.

The mean is computed according to the `.metric` field
of the `model`, with optional weights `w`.
The `.metric` field of the `model` is passed internally to the [`tsMap`](@ref) function.
By default the metric is the Fisher metric. See the examples
here below to see how to change metric.
See [mdm.jl](@ref) or check out directly the documentation
of [PosDefManifold.jl](https://marco-congedo.github.io/PosDefManifold.jl/dev/)
for the available metrics.

**Optional keyword arguments**

If a `pipeline`, of type [`Pipeline`](@ref) is provided, 
all necessary parameters of the sequence of conditioners are fitted and all matrices 
are transformed according to the specified pipeline before fitting the
ML model. The parameters are stored in the output ML model.
Note that the fitted pipeline is automatically applied by any successive call 
to function [`predict`](@ref) to which the output ML model is passed as argument.

By default, uniform weights will be given to all observations
for computing the mean to project the data in the tangent space.
This is equivalent to passing as argument `w=:uniform` (or `w=:u`).
You can also pass as argument:

- `w=:balanced` (or simply `w=:b`). If the two classes are unbalanced,
  the weights should better be inversely proportional to the number of examples
  for each class, in such a way that each class contributes equally
  to the computation of the mean.
  This is equivalent of passing `w=tsWeights(yTr)`. See the
  [`tsWeights`](@ref) function for details.
- `w=v`, where `v` is a user defined vector of non-negative weights for the
  observations, thus, `v` must contain the same number of elements as `yTr`.
  For example, `w=[1.0, 1.0, 2.0, 2.0, ...., 1.0]`
- `w=t`, where `t` is a 2-tuple of real weights, one weight for each class,
  for example `w=(0.5, 1.5)`.
  This is equivalent to passing `w=tsWeights(yTr; classWeights=collect(0.5, 1.5))`,
  see the [`tsWeights`](@ref) function for details.

By default `meanISR=nothing` and the inverse square root (ISR) of the mean
used for projecting the matrices onto the tangent space (see [`tsMap`](@ref))
is computed. An Hermitian matrix or `I` (the identity matrix) can also be passed 
as argument `meanISR` and in this case this matrix will be used as the ISR of the mean.
Passed or computed, it will be written in the `.meanISR` field of the 
model structure created by this function. Notice that passing `I`, the matrices
will be projected onto the tangent space at the identity without recentering them.
This is possible if the matrices have been recentered bt a pre-conditioning pipeline
(see [`Pipeline`](@ref)).

If `meanISR` is not provided and the `.metric` field of the `model`
is Fisher, logdet0 or Wasserstein, the tolerance of the iterative algorithm
used to compute the mean is set to argument `tol` (default 1e-5).
Also, in this case a particular initialization for those iterative algorithms
can be provided as an `Hermitian` matrix with argument `meanInit`.

!!! tip "Euclidean ENLR models"
    ML models acting on the tangent space allows to fit a model passing as
    training data `𝐏Tr` directly a matrix of feature vectors,
    where each feature vector is a row of the matrix.
    In this case none of the above keyword arguments are used.  

**The following optional keyword arguments act on any kind of input,
that is, tangent vectors and generic feature vectors**

If a `UnitRange` is passed with optional keyword argument `vecRange`,
then if `𝐏Tr` is a vector of `Hermitian` matrices, the vectorization
of those matrices once they are projected onto the tangent space
concerns only the rows (or columns) given in the specified range,
else if `𝐏Tr` is a matrix with feature vectors arranged in its rows, then
only the columns of `𝐏Tr` given in the specified range will be used.
Argument `vecRange` will be ignored if a pre-conditioning pipeline is used 
and if the pipeline changes the dimension of the input matrices.
In this case it will be set to its default value using the new dimension.
You are not allowed to change this behavior.

With `normalize` the tangent (or feature) vectors can be normalized individually.
Three functions can be passed, namely 
 - [`demean!`](@ref) to remove the mean,
 - [`normalize!`](@ref) to fix the norm (default),
 - [`standardize!`](@ref) to fix the mean to zero and the standard deviation to 1.

As argument `normalize` you can also pass a 2-tuple of real numbers.
In this case the numbers will be the lower and upper limit
to bound the vectors within these limits - see [`rescale!`](@ref).

!!! tip "Rescaling"
    If you wish to rescale, use `(-1, 1)`, since tangent vectors
    of SPD matrices have positive and negative elements. If `𝐏Tr`
    is a feature matrix and the features are only positive, use `(0, 1)`
    instead. 

If you pass `nothing` as argument `normalize`, no normalization will be carried out.

The remaining optional keyword arguments, are

- the arguments passed to the `GLMNet.glmnet` function for fitting the models.
  Those are always used.

- the `λSelMeth` argument and the arguments passed to the `GLMNet.glmnetcv`
  function for finding the best lambda hyperparamater by cross-validation.
  Those are used only if `fitType` = `:path` or = `:all`.

**Optional keyword arguments for fitting the model(s) using GLMNet.jl**

`alpha`: the hyperparameter in ``[0, 1]`` to trade-off
an elestic-net model. *α=0* requests a pure *ridge* model and
*α=1* a pure *lasso* model. This defaults to 1.0,
which specifies a lasso model, unless the input [`ENLR`](@ref) `model`
has another value in the `alpha` field, in which case this value
is used. If argument `alpha` is passed here, it will overwrite
the `alpha` field of the input `model`.

`weights`: a vector of weights for each matrix (or feature vectors)
of the same size as `yTr`.
It defaults to 1 for all matrices.

`intercept`: whether to fit an intercept term.
The intercept is always unpenalized. Defaults to true.

If `fitType` = `:best` (default), a cross-validation procedure is run
to find the best lambda hyperparameter for the given training data.
This finds a single model that is written into the `.best` field
of the [`ENLR`](@ref) structure that will be created.

If `fitType` = `:path`, the regularization path for several values of
the lambda hyperparameter is found for the given training data.
This creates several models, which are written into the
`.path` field of the [`ENLR`](@ref) structure that will be created,
none of which is optimal, in the cross-validation sense, for the
given training data.

If `fitType` = `:all`, both the above fits are performed and all fields
of the [`ENLR`](@ref) structure that will be created will be filled in.

`penalty_factor`: a vector of length *n(n+1)/2*, where *n*
is the dimension of the original PD matrices on which the model is applied,
of penalties for each predictor in the tangent vectors.
This defaults to all ones, which weights each predictor equally.
To specify that a predictor should be unpenalized,
set the corresponding entry to zero.

`constraints`: an *[n(n+1)/2]* x *2* matrix specifying lower bounds
(first column) and upper bounds (second column) on each predictor.
By default, this is [-Inf Inf] for each predictor (each element
of tangent vectors).

`offset`: see documentation of original GLMNet package [🎓](@ref).

`dfmax`: The maximum number of predictors in the largest model.

`pmax`: The maximum number of predictors in any model.

`nlambda`: The number of values of *λ* along the path to consider.

`lambda_min_ratio`: The smallest *λ* value to consider,
as a ratio of the value of *λ* that gives the null model
(*i.e.*, the model with only an intercept).
If the number of observations exceeds the number of variables,
this defaults to 0.0001, otherwise 0.01.

`lambda`: The *λ* values to consider for fitting.
By default, this is determined
from `nlambda` and `lambda_min_ratio`.

`maxit`: The maximum number of iterations of the cyclic coordinate
descent algorithm. If convergence is not achieved, a warning is returned.

`algorithm`: the algorithm used to find the regularization path.
Possible values are `:newtonraphson` (default) and
`:modifiednewtonraphson`.

For further informations on those arguments, refer to the
resources on the GLMNet package [🎓](@ref).

!!! warning "Possible change of dimension"
    The provided arguments `penalty_factor`, `constraints`, `dfmax`, `pmax` and 
    `lambda_min_ratio` will be ignored if a pre-conditioning `pipeline` is passed 
    as argument and if the pipeline	changes the dimension of the input matrices, 
    thus of the tangent vectors. In this case they will be set to their 
    default values using the new dimension. To force the use of the provided values 
    instead, set `checkArgs` to false (true by default). Note however that in this 
    case you must provide suitable values for all the abova arguments.

**Optional Keyword arguments for finding the best model by cv**

`λSelMeth` = `:sd1` (default), the best model is defined as the one
allowing the highest `cvλ.meanloss` within one standard deviation of the
minimum, otherwise it is defined as the one allowing the minimum `cvλ.meanloss`.
Note that in selecting a model, the model with only the intercept term,
if it exists, is ignored. See [`ENLRmodel`](@ref) for a description
of the `.cvλ` field of the model structure.

Arguments `nfolds`, `folds` and `parallel` are passed to the
`GLMNet.glmnetcv` function along with the `⏩` argument.
Please refer to the resources on GLMNet
for details [🎓](@ref).

`tol`: Is the convergence criterion for both the computation
of a mean for projecting onto the tangent space
(if the metric requires an iterative algorithm)
and for the GLMNet fitting algorithm. Defaults to 1e-5.
In order to speed up computations, you may try to set a lower `tol`;
The convergence will be faster but more coarse,
with a possible drop of classification accuracy,
depending on the signal-to-noise ratio of the input features.

If `verbose` is true (default), information is printed in the REPL.

The `⏩` argument (true by default) is passed to the [`tsMap`](@ref)
function for projecting the matrices in `𝐏Tr` onto the tangent space
and to the `GLMNet.glmnetcv` function to run inner cross-validation
to find the `best` model using multi-threading.

**See**: [notation & nomenclature](@ref), [the ℍVector type](@ref)

**See also**: [`predict`](@ref), [`crval`](@ref)

**Tutorial**: [Examples using the ENLR model](@ref)

**Examples**
```julia
using PosDefManifoldML, PosDefManifold

# Generate some data
PTr, PTe, yTr, yTe = gen2ClassData(10, 30, 40, 60, 80, 0.1)

# Fit an ENLR lasso model and find the best model by cross-validation
m = fit(ENLR(), PTr, yTr)

# ... standardizing the tangent vectors
m = fit(ENLR(), PTr, yTr; pipeline=p, normalize=standardize!)

# ... balancing the weights for tangent space mapping
m = fit(ENLR(), PTr, yTr; w=tsWeights(yTr))

# ... using the log-Eucidean metric for tangent space projection
m = fit(ENLR(logEuclidean), PTr, yTr)

# Fit an ENLR ridge model and find the best model by cv:
m = fit(ENLR(Fisher), PTr, yTr; alpha=0)

# Fit an ENLR elastic-net model (α=0.9) and find the best model by cv:
m = fit(ENLR(Fisher), PTr, yTr; alpha=0.9)

# Fit an ENLR lasso model and its regularization path:
m = fit(ENLR(), PTr, yTr; fitType=:path)

# Fit an ENLR lasso model, its regularization path
# and the best model found by cv:
m = fit(ENLR(), PTr, yTr; fitType=:all)

# Fit using a pre-conditioning pipeline:
p = @→ Recenter(; eVar=0.999) Compress Shrink(Fisher; radius=0.02)
m = fit(ENLR(PosDefManifold.Euclidean), PTr, yTr; pipeline=p)

# Use a recentering pipeline and project the data
# onto the tangent space at the identity matrix.
# In this case the metric is irrilevant as the barycenter
# for determining the base point is not computed.
# Note that the previous call to 'fit' has modified `PTr`,
# so we generate new data.
PTr, PTe, yTr, yTe = gen2ClassData(10, 30, 40, 60, 80, 0.1)
p = @→ Recenter(; eVar=0.999) Compress Shrink(Fisher; radius=0.02)
m = fit(ENLR(), PTr, yTr; pipeline=p, meanISR=I)
```
"""
function fit(model  :: ENLRmodel,
                𝐏Tr  :: Union{ℍVector, Matrix{Float64}},
                yTr  :: IntVector;
            # pipeline
            pipeline    :: Union{Pipeline, Nothing} = nothing,
            # parameters for projection onto the tangent space
            w        	:: Union{Symbol, Tuple, Vector} = Float64[],
            meanISR  	:: Union{ℍ, Nothing, UniformScaling} = nothing,
            meanInit 	:: Union{ℍ, Nothing} = nothing,
            vecRange 	:: UnitRange = 𝐏Tr isa ℍVector ? (1:size(𝐏Tr[1], 2)) : (1:size(𝐏Tr, 2)),
            normalize	:: Union{Function, Tuple, Nothing} = normalize!,

            # arguments for `GLMNet.glmnet` function
            alpha           :: Real = model.alpha,
            weights         :: Vector{Float64} = ones(Float64, length(yTr)),
            intercept       :: Bool = true,
            fitType  	    :: Symbol = :best,
            penalty_factor  :: Vector{Float64} = ones(Float64, _getDim(𝐏Tr, vecRange)),
            constraints     :: Matrix{Float64} = [x for x in (-Inf, Inf), y in 1:_getDim(𝐏Tr, vecRange)],
            offsets         :: Union{Vector{Float64}, Nothing} = nothing,
            dfmax           :: Int = _getDim(𝐏Tr, vecRange),
            pmax            :: Int = min(dfmax*2+20, _getDim(𝐏Tr, vecRange)),
            nlambda         :: Int = 100,

            #lambda_min_ratio :: Real = (length(yTr)*2 < _getDim(𝐏Tr, vecRange) ? 1e-2 : 1e-4),
            lambda_min_ratio:: Real = (length(yTr) < _getDim(𝐏Tr, vecRange) ? 1e-2 : 1e-4), # May 2025
            lambda          :: Vector{Float64} = Float64[],
            maxit           :: Int = 1000000,
            algorithm       :: Symbol = :newtonraphson,
            checkArgs       :: Bool = true,

            # selection method
            λSelMeth    :: Symbol = :sd1,
            
            # arguments for `GLMNet.glmnetcv` function
            nfolds      :: Int = min(10, div(size(yTr, 1), 3)),
            folds       :: Vector{Int} =
            begin
                n, r = divrem(size(yTr, 1), nfolds)
                shuffle!([repeat(1:nfolds, outer=n); 1:r])
            end,
            parallel    :: Bool=true,
            tol         :: Real = 1e-7, # Mai 2025
            verbose     :: Bool = true,
            ⏩      	   :: Bool = true,
)

    ⌚=now() # get the time in ms
    ℳ=deepcopy(model) # output model

	# overwrite fields in `ℳ` if the user has passed them here as arguments,
	# otherwise use as arguments the values in the fields of `ℳ`, e.g., the default
	if alpha ≠ 1.0 ℳ.alpha = alpha else alpha = ℳ.alpha end

    # check w argument and get weights for input matrices
    (w=_getWeights(w, yTr, "fit ("*_model2Str(ℳ)*" model)")) == nothing && return

    # other checks
    𝐏Tr isa ℍVector ? nObs=length(𝐏Tr) : nObs=size(𝐏Tr, 1)
    !_check_fit(ℳ, nObs, length(yTr), length(w), length(weights), "ENLR") && return

    # apply pre-conditioning pipeline and reset some keyword args to fit the model 
    # if the pipeline change the input matrix dimension
    if 𝐏Tr isa ℍVector # only for tangent vectors (not if 𝐏Tr is a matrix of tangent vectors)     

        originalDim = size(𝐏Tr[1], 2)
        # pipeline (pre-conditioners)
        if !(pipeline===nothing)
            verbose && println(greyFont, "Fitting pipeline...")
            ℳ.pipeline = fit!(𝐏Tr, pipeline)
        end

        newDim = size(𝐏Tr[1], 2) # some pre-conditioners can change the dimension
        if newDim ≠ originalDim && checkArgs # reset these arguments to the default using the new dimension
            vecRange = 1:newDim
            manifoldDim = (newDim * (newDim+1)) ÷ 2
            penalty_factor  = ones(Float64, manifoldDim)
            constraints = [x for x in (-Inf, Inf), y in 1:manifoldDim]
            dfmax = manifoldDim
            pmax = min(dfmax*2+20, manifoldDim)
            lambda_min_ratio = (length(yTr) < manifoldDim ? 1e-2 : 1e-4)
        end
    end

	# project data onto the tangent space or just copy the features if 𝐏Tr is a matrix
    verbose && println(greyFont, "Lifting SPD matrices onto the tangent space...")
	X = _getTSvec_fit!(ℳ, 𝐏Tr, meanISR, meanInit, tol, w, vecRange, true, verbose, ⏩)

   	# Normalize data if the user asks for. 
	# For ENLR models the tangent vectors are the rows of X, thus dims=2
    _normalizeTSvec!(X, normalize; dims=2)

    # convert labels in GLMNet format # 
    y = convert(Matrix{Float64}, [(yTr.==1) (yTr.==2)])

    # write some fields in output model struct
    ℳ.intercept   = intercept
    ℳ.featDim     = size(X, 2)
	ℳ.vecRange    = vecRange
    ℳ.normalize   = normalize

    # collect the argumenst for `glmnet` function excluding the `lambda` argument
    fitArgs_λ = (alpha            = alpha,
                 weights          = weights,
                 standardize      = false, # we take care of the normalization
                 intercept        = intercept,
                 penalty_factor   = penalty_factor,
                 constraints      = constraints,
                 offsets          = offsets,
                 dfmax            = dfmax,
                 pmax             = pmax,
                 nlambda          = nlambda,
                 lambda_min_ratio = lambda_min_ratio,
                 tol              = tol,
                 maxit            = maxit,
                 algorithm        = algorithm)

    # Fit a path 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    if fitType ∈(:path, :all)
        # fit the regularization path
        verbose && println("Fitting "*_model2Str(ℳ)*" reg. path...")
        ℳ.path = glmnet(X, y, Binomial();
                         lambda = lambda,
                         fitArgs_λ...) # glmnet Args but lambda
    end
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Fit the best model
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    if fitType ∈(:best, :all)
        verbose && println("Fitting best "*_model2Str(ℳ)*" model...")
        ℳ.cvλ=glmnetcv(X, y, Binomial();
                        nfolds   = nfolds,
                        folds    = folds,
                        parallel = parallel,
                        lambda   = lambda,
                        fitArgs_λ...) # glmnet Args but lambda

        # Never consider the model with only the intercept (0 degrees of freedom)
        l, i=length(ℳ.cvλ.lambda), max(argmin(ℳ.cvλ.meanloss), 1+intercept)

        # if bestλsel==:sd1 select the highest model with mean loss withinh 1sd of the minimum
        # otherwise the model with the smallest mean loss.
        thr=ℳ.cvλ.meanloss[i]+ℳ.cvλ.stdloss[i]
        λSelMeth==:sd1 ? (while i<l ℳ.cvλ.meanloss[i+1]<=thr ? i+=1 : break end) : nothing

        # fit the best model (only for the optimal lambda)
        ℳ.best = glmnet(X, y, Binomial();
                         lambda  = [ℳ.cvλ.path.lambda[i]],
                         fitArgs_λ...) # glmnet Args but lambda
    end
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    verbose && println(defaultFont, "Done in ", now()-⌚,".")
    return ℳ
end


"""
```julia
function predict(model   :: ENLRmodel,
		𝐏Te         :: Union{ℍVector, Matrix{Float64}},
		what        :: Symbol = :labels,
		fitType     :: Symbol = :best,
		onWhich     :: Int    = Int(fitType==:best);
    pipeline    :: Union{Pipeline, Nothing} = nothing,
    meanISR     :: Union{ℍ, Nothing, UniformScaling} = nothing,
    verbose     :: Bool = true,
    ⏩          :: Bool = true)
```

Given an [`ENLR`](@ref) `model` trained (fitted) on 2 classes
and a testing set of *k* positive definite matrices `𝐏Te` of type
[ℍVector](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#%E2%84%8DVector-type-1),

if `what` is `:labels` or `:l` (default), return
the predicted **class labels** for each matrix in `𝐏Te`,
as an [IntVector](@ref).
Those labels are '1' for class 1 and '2' for class 2;

if `what` is `:probabilities` or `:p`, return the predicted **probabilities**
for each matrix in `𝐏Te` to belong to each classe, as a *k*-vector
of *z* vectors holding reals in *[0, 1]* (probabilities).
The 'probabilities' are obtained passing to a
[softmax function](https://en.wikipedia.org/wiki/Softmax_function)
the output of the ENLR model and zero;

if `what` is `:f` or `:functions`, return the **output function** of the model,
which is the raw output of the ENLR model.

If `fitType` = `:best` (default), the best model that has been found by
cross-validation is used for prediction.

If `fitType` = `:path`,

- if `onWhich` is a valid serial number for a model in the `model.path`,
then this model is used for prediction,

- if `onWhich` is zero, all models in the `model.path` will be used for predictions, thus the output will be multiplied by the number of models in `model.path`.

Argument `onWhich` has no effect if `fitType` = `:best`.

!!! note "Nota Bene"
    By default, the [`fit`](@ref) function fits only the `best` model.
    If you want to use the `fitType` = `:path` option you need to invoke
    the fit function with optional keyword argument `fitType`=`:path` or
    `fitType`=`:all`. See the [`fit`](@ref) function for details.

Optional keyword argument `meanISR` can be used to specify the principal
inverse square root (ISR) of a new mean to be used as base point for
projecting the matrices in testing set `𝐏Te` onto the tangent space.
By default `meanISR` is equal to nothing,
implying that the base point will be the mean used to fit the model.
This corresponds to the classical 'training-test' mode of operation.

Passing with argument `meanISR` a new mean ISR
allows the *adaptation* first described in
Barachant et *al.* (2013)[🎓](@ref). Typically `meanISR` is the ISR
of the mean of the matrices in `𝐏Te` or of a subset of them.
Notice that this actually performs *transfer learning* by parallel
transporting both the training and test data to the identity matrix
as defined in Zanini et *al.* (2018) and later taken up in
Rodrigues et *al.* (2019)[🎓](@ref).  
You can aslo pass `meanISR=I`, in which case the base point
is taken as the identity matrix. This is possible if the set
`𝐏Te` is centered to the identity, for instance, if a recentering
pre-conditioner is included in a pipeline and the pipeline 
is adapted as well (see the example below).

If `verbose` is true (default), information is printed in the REPL.
This option is included to allow repeated calls to this function
without crowding the REPL.

If ⏩ = true (default) and `𝐏Te` is an ℍVector type, the projection onto the
tangent space is multi-threaded.

Note that if the field `pipeline` of the provided `model` is not `nothing`,
implying that a pre-conditioning pipeline has been fitted during the
fitting of the model,
the pipeline is applied to the data before to carry out the prediction.
If you wish to **adapt** the pipeline to the testing data, 
just pass the same pipeline as argument `pipeline` in this function.

!!! warning "Adapting the Pipeline"
    Be careful when adapting a pipeline; if a [`Recenter`](@ref) conditioner is included in the
    pipeline and dimensionality reduction was sought (parameter `eVar` different 
    from `nothing`), then `eVar` must be set to an integer so that the
    dimension of the training ad testing data is the same after adaptation.
    See the example here below.

**See**: [notation & nomenclature](@ref), [the ℍVector type](@ref)

**See also**: [`fit`](@ref), [`crval`](@ref), [`predictErr`](@ref)

**Examples**
```julia
using PosDefManifoldML, PosDefManifold

# Generate some data
PTr, PTe, yTr, yTe = gen2ClassData(10, 30, 40, 60, 80)

# Fit an ENLR lasso model and find the best model by cv
m = fit(ENLR(Fisher), PTr, yTr)

# Predict labels from the best model
yPred = predict(m, PTe, :l)

# Prediction error
predErr = predictErr(yTe, yPred)

# Predict probabilities from the best model
predict(m, PTe, :p)

# Output functions from the best model
predict(m, PTe, :f)

# Fit a regularization path for an ENLR lasso model
m = fit(ENLR(Fisher), PTr, yTr; fitType=:path)

# Predict labels using a specific model
yPred = predict(m, PTe, :l, :path, 10)

# Predict labels for all models
yPred = predict(m, PTe, :l, :path, 0)

# Prediction error for all models
predErr = [predictErr(yTe, yPred[:, i]) for i=1:size(yPred, 2)]

# Predict probabilities from a specific model
predict(m, PTe, :p, :path, 12)

# Predict probabilities from all models
predict(m, PTe, :p, :path, 0)

# Output functions from specific model
predict(m, PTe, :f, :path, 3)

# Output functions for all models
predict(m, PTe, :f, :path, 0)

## Adapting the base point
PTr, PTe, yTr, yTe = gen2ClassData(10, 30, 40, 60, 80)
m = fit(ENLR(Fisher), PTr, yTr)
predict(m, PTe, :l; meanISR=invsqrt(mean(Fisher, PTe)))

# Also using and adapting a pre-conditioning pipeline
# For adaptation, we need to set `eVar` to an integer or to `nothing`.
# We will use the dimension determined on training data.
# Note that the adaptation does not work well if the class proportions
# of the training data is different from the class proportions of the test data.
p = @→ Recenter(; eVar=0.999) Compress Shrink(Fisher; radius=0.02)

# Fit the model using the pre-conditioning pipeline
m = fit(ENLR(), PTr, yTr; pipeline = p)

# Define the same pipeline with fixed dimensionality reduction parameter
p = @→ Recenter(; eVar=dim(m.pipeline)) Compress Shrink(Fisher; radius=0.02)

# Fit the pipeline to testing data (adapt) and use the identity matrix as base point:
predict(m, PTe, :l; pipeline=p, meanISR=I) 

# Suppose we want to adapt recentering, but not shrinking, which also has a 
# learnable parameter. We would then use this pipeline instead:
p = deepcopy(m.pipeline)
p[1].eVar = dim(m.pipeline)
```
"""
function predict(model   :: ENLRmodel,
                 𝐏Te     :: Union{ℍVector, Matrix{Float64}},
                 what    :: Symbol = :labels,
                 fitType :: Symbol = :best,
                 onWhich :: Int    = Int(fitType==:best);
			meanISR     :: Union{ℍ, Nothing, UniformScaling} = nothing,
            pipeline    :: Union{Pipeline, Nothing} = nothing,
            verbose     :: Bool = true,
            ⏩          :: Bool = true)

    ⌚=now()

    # checks
    _whatIsValid(what, "predict ("*_model2Str(model)*")") || return
    _fitTypeIsValid(fitType, "predict ("*_model2Str(model)*")") || return
    fitType==:best && model.best==nothing && @error(📌*", predict function: the best model has not been fitted; run the `fit`function with keyword argument `fitType=:best` or `fitType=:all`") === nothing && return 
    fitType==:path && model.path==nothing && @error(📌*", predict function: the regularization path has not been fitted; run the `fit`function with keyword argument `fitType=:path` or `fitType=:all`") === nothing && return 
    _ENLRonWhichIsValid(model, fitType, onWhich, "predict ("*_model2Str(model)*")") || return

    verbose && println(greyFont, "Applying pipeline...")
    _applyPipeline!(𝐏Te, pipeline, model)

    # projection onto the tangent space
	X = _getTSvec_Predict!(model, 𝐏Te, meanISR, model.vecRange, true, verbose, ⏩)

   	# Normalize data if the user have asked for while fitting the model. 
	# For ENLR models the tangent vectors are the rows of X, thus dims=2
    _normalizeTSvec!(X, model.normalize; dims=2)

    # prediction
    verbose && println("Predicting "*_ENLRonWhichStr(model, fitType, onWhich)*"...")
    if 		fitType==:best
        	path=model.best
        	onWhich=1
    elseif  fitType==:path
        	path=model.path
    end

    onWhich==0 ? π=GLMNet.predict(path, X) : π=GLMNet.predict(path, X, onWhich)

    k, l=size(π, 1), length(path.lambda)
    if     	what == :functions || what == :f
        	🃏=π
    elseif 	what == :labels || what == :l
        	onWhich==0 ? 🃏=[π[i, j]<0 ? 1 : 2 for i=1:k, j=1:l] : 🃏=[y<0 ? 1 : 2 for y ∈ π]
    elseif 	what == :probabilities || what == :p
        	onWhich==0 ? 🃏=[softmax([-π[i, j], 0]) for i=1:k, j=1:l] : 🃏=[softmax([-y, 0]) for y ∈ π]
    end

    verbose && println(defaultFont, "Done in ", now()-⌚,".")
    verbose && println(titleFont, "\nPredicted ",_what2Str(what),":", defaultFont)
    return 🃏
end



# ++++++++++++++++++++  Show override  +++++++++++++++++++ # (REPL output)
function Base.show(io::IO, ::MIME{Symbol("text/plain")}, M::ENLR)
    if M.path==nothing
        if M.best==nothing
            println(io, greyFont, "\n↯ ENLR GLMNet machine learning model")
            println(io, "⭒  ⭒    ⭒       ⭒          ⭒")
            println(io, ".metric : ", string(M.metric))
            println(io, ".alpha  : ", "$(round(M.alpha, digits=3))", defaultFont)
            println(io, "Unfitted model")
            return
        end
    end

    println(io, titleFont, "\n↯ GLMNet ENLR machine learning model")
    println(io, separatorFont, "  ", _model2Str(M))
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
	println(io, greyFont, " ENLR Parametrization", defaultFont)
    println(io, separatorFont," .alpha       ", defaultFont, "$(round(M.alpha, digits=3))")
    println(io, separatorFont," .intercept   ", defaultFont, string(M.intercept))

    if M.path==nothing
        println(io, greyFont," .path struct `GLMNetPath`:")
        println(io, "       not created ")
    else
        println(io, separatorFont," .path", defaultFont," struct `GLMNetPath`:")
        println(io, titleFont,"       .family, .a0, .betas, .null_dev, ")
        println(io, titleFont,"       .dev_ratio, .lambda, .npasses")
    end
    if M.cvλ==nothing
        println(io, greyFont," .cvλ  struct `GLMNetCrossValidation`:")
        println(io, "       not created ")
    else
        println(io, separatorFont," .cvλ", defaultFont,"  struct `GLMNetCrossValidation`:")
        println(io, titleFont,"       .path, .nfolds, .lambda, ")
        println(io, titleFont,"       .meanloss, stdloss")
    end
    if M.best==nothing
        println(io, greyFont," .best struct `GLMNetPath`:")
        println(io, "       not created ")
    else
        println(io, separatorFont," .best", defaultFont," struct `GLMNetPath`:")
        println(io, titleFont,"       .family, .a0, .betas, .null_dev, ")
        println(io, titleFont,"       .dev_ratio, .lambda, .npasses")
    end
end
