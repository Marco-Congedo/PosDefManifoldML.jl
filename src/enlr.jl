#   Unit "enlr.jl" of the PosDefManifoldML Package for Julia language

#   MIT License
#   Copyright (c) 2019,
#   Marco Congedo, CNRS, Grenoble, France:
#   https://sites.google.com/site/marcocongedo/home

# ? CONTENTS :
#   This unit implements the elastic net logistic regression (ENLR)
#   machine learning classifier, including the lasso LR (default) and Ridge LR
#   as specific instances.

"""
```
abstract type ENLRmodel<:TSmodel end
```

Abstract type for **Elastic Net Logistic Rgression (ENLR)**
machine learning models. See [MLmodel](@ref).
"""
abstract type ENLRmodel<:TSmodel end


"""
```
mutable struct ENLR <: ENLRmodel
    metric      :: Metric = Fisher;
    alpha       :: Real = 1.0
    standardize :: Bool
    intercept   :: Bool
	meanISR     :: Union{ℍVector, Nothing}
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
the **elestic-net model**. ``α=0`` requests a pure **ridge** model and
``α=1`` a pure **lasso** model. By default, ``α=1`` is specified (lasso model).
This argument is usually passed as parameter to
the [`fit`](@ref) function, defaulting therein to ``α=1`` too.
See the examples here below.

All other fields do not correspond to arguments passed
upon creation of the model by the default creator.
Instead, they are filled later when a model is created by the
[`fit`](@ref) function:

For the content of fields `standardize`, `intercept`, `meanISR` and `vecRange`,
please see the documentation of the [`fit`](@ref) function.

if the data used to train the model are positive definite matrices,
`.featDim` is the length of the vectorized tangent vectors.
This is given by ``n(n+1)÷2`` (integer division), where ``n``
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

```
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

```
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
```
# Note: creating models with the default creator is possible,
# but not useful in general.

using PosDefManifoldML, PosDefManifold

# create an empty lasso model
m = ENLR(Fisher)

# since the Fisher metric is the default metric,
# this is equivalent to
m = ENLR()

# create an empty ridge model using the logEuclidean metric
m = ENLR(logEuclidean; alpha=0)

# Empty models can be passed as first argument of the `fit` function
# to fit a model. For instance, this will fit a ridge model of the same
# kind of `m` and put the fitted model in `m1`:
m1=fit(m, PTr, yTr)

# in general you don't need this machinery for fitting a model,
# since you can specify a model by creating one on the fly:
m2=fit(ENLR(logEuclidean; alpha=0), PTr, yTr)

# which is equivalent to
m2=fit(ENLR(logEuclidean), PTr, yTr; alpha=0)

# note that, albeit model `m` has been created as a ridge model,
# you have passed `m` and overwritten the `alpha` hyperparameter.
# The metric, instead, cannot be overwritten.

```
"""
mutable struct ENLR <: ENLRmodel
    metric   :: Metric
    alpha    :: Real
    standardize
    intercept
    meanISR
	vecRange
    featDim
    path
    cvλ
    best
    function ENLR(metric :: Metric=Fisher;
               alpha     :: Real = 1.0,
               standardize = nothing,
               intercept   = nothing,
               meanISR     = nothing,
			   vecRange    = nothing,
               featDim     = nothing,
               path        = nothing,
               cvλ         = nothing,
               best        = nothing)
        new(metric, alpha, standardize, intercept,
            meanISR, vecRange, featDim, path, cvλ, best)
    end
end


"""
```
function fit(model	:: ENLRmodel,
             𝐏Tr	 :: Union{ℍVector, Matrix{Float64}},
             yTr	:: IntVector;
	# parameters for projection onto the tangent space
	w		:: Union{Symbol, Tuple, Vector} = [],
	meanISR	:: Union{ℍ, Nothing} = nothing,
	meanInit:: Union{ℍ, Nothing} = nothing,
	vecRange:: UnitRange = 𝐏Tr isa ℍVector ? (1:size(𝐏Tr[1], 2)) : (1:size(𝐏Tr, 2)),
	fitType	:: Symbol = :best,
	verbose	:: Bool = true,
	⏩	   :: Bool = true,
	# arguments for `GLMNet.glmnet` function
	alpha			:: Real = model.alpha,
	weights			:: Vector{Float64} = ones(Float64, length(yTr)),
	intercept		:: Bool = true,
	standardize		:: Bool = true,
	penalty_factor	:: Vector{Float64} = ones(Float64, _getDim(𝐏Tr, vecRange)),
	constraints		:: Matrix{Float64} = [x for x in (-Inf, Inf), y in 1:_getDim(𝐏Tr, vecRange)],
	offsets			:: Union{Vector{Float64}, Nothing} = nothing,
	dfmax			:: Int = _getDim(𝐏Tr, vecRange),
	pmax			:: Int = min(dfmax*2+20, _getDim(𝐏Tr, vecRange)),
	nlambda			:: Int = 100,
	lambda_min_ratio:: Real = (length(yTr) < _getDim(𝐏Tr, vecRange) ? 1e-2 : 1e-4),
	lambda			:: Vector{Float64} = Float64[],
	tol				:: Real = 1e-5,
	maxit			:: Int = 1000000,
	algorithm		:: Symbol = :newtonraphson,
	# selection method
	λSelMeth	:: Symbol = :sd1,
	# arguments for `GLMNet.glmnetcv` function
	nfolds		:: Int = min(10, div(size(yTr, 1), 3)),
	folds		:: Vector{Int} =
	begin
		n, r = divrem(size(yTr, 1), nfolds)
		shuffle!([repeat(1:nfolds, outer=n); 1:r])
	end,
	parallel 	:: Bool=true)
```

Create and fit an [`ENLR`](@ref) machine learning model,
with training data `𝐏Tr`, of type
[ℍVector](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#%E2%84%8DVector-type-1),
and corresponding labels `yTr`, of type [IntVector](@ref).
Return the fitted model(s) as an instance of the [`ENLR`](@ref) structure.

As for all ML models acting in the tangent space,
fitting an ENLR model involves computing a mean of all the
matrices in `𝐏Tr`, mapping all matrices onto the tangent space
after parallel transporting them at the identity matrix
and vectorizing them using the
[vecP](https://marco-congedo.github.io/PosDefManifold.jl/dev/riemannianGeometry/#PosDefManifold.vecP)
operation. Once this is done, the elastic net logistic regression is fitted.

The mean is computed according to the `.metric` field
of the `model`, with optional weights `w`.
The `.metric` field of the `model` is passed to the [`tsMap`](@ref) function.
By default the metric is the Fisher metric. See the examples
here below to see how to change metric.
See [mdm.jl](@ref) or check out directly the documentation
of [PosDefManifold.jl](https://marco-congedo.github.io/PosDefManifold.jl/dev/)
for the available metrics.

**Optional keyword arguments**

By default, uniform weights will be given to all observations
for computing the mean to pass in the tangent space.
This is equivalent to passing as argument `w=:uniform` (or `w=:u`).
You can also pass as argument:

- `w=:balanced` (or simply `w=:b`). If the two classes are unbalanced,
  the weights should be inversely proportional to the number of examples
  for each class, in such a way that each class contributes equally
  to the computation of the mean.
  This is equivalent of passing `w=tsWeights(yTr)`. See the
  [`tsWeights`](@ref) function for details.
- `w=v`, where `v` is a user defined vector of non-negative weights for the
  observations, thus, `v` must contain the same number of elements as `yTr`.
  For example, `w=[1.0, 1.0, 2.0, 2.0, ...., 1.0]`
- `w=t`, where `t` is a 2-tuple of real weights, one weight for each class,
  for example `w=(0.5, 1.5)`.
  This is equivalent to passing `w=tsWeights(yTr; classWeights=collect(t))`,
  see the [`tsWeights`](@ref) function for details.

If `meanISR` is passed as argument, the mean is not computed,
instead this matrix is the inverse square root (ISR) of the mean
used for projecting the matrices in the tangent space (see [`tsMap`](@ref)).
Passed or computed, the inverse square root (ISR) of the mean
will be written in the `.meanISR` field of the created [`ENLR`](@ref) structure.
If `meanISRis` is not provided and the `.metric` field of the `model`
is Fisher, logdet0 or Wasserstein, the tolerance of the iterative algorithm
used to compute the mean is set to the argument passed as `tol` (default 1e-5).
Also, in this case a particular initialization for those iterative algorithms
can be provided as an `Hermitian` matrix with argument `meanInit`.

This function also allows to fit a model passing as
training data `𝐏Tr` directly a matrix of feature vectors,
where each feature vector is a row of the matrix.
In this case the `metric` of the ENLR model and argument `meanISR` are not used.
Therefore, the `.meanISR` field of the created [`ENLR`](@ref) structure
will be set to `nothing`.

If a `UnitRange` is passed with optional keyword argument `vecRange`,
then if `𝐏Tr` is a vector of `Hermitian` matrices, the vectorization
of those matrices once they are projected onto the tangent space
concerns only the rows (or columns) given in the specified range,
else if `𝐏Tr` is a matrix with feature vectors arranged in its rows, then
only the columns of `𝐏Tr` given in the specified range will be used.

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

If `verbose` is true (default), information is printed in the REPL.
This option is included to allow repeated calls to this function
without crowding the REPL.

The `⏩` argument (true by default) is passed to the [`tsMap`](@ref)
function for projecting the matrices in `𝐏Tr` onto the tangent space
and to the `GLMNet.glmnetcv` function to run inner cross-validation
to find the `best` model using multi-threading.

The remaining optional keyword arguments, are

- the arguments passed to the `GLMNet.glmnet` function for fitting the models.
  Those are always used.

- the `λSelMeth` argument and the arguments passed to the `GLMNet.glmnetcv`
  function for finding the best lambda hyperparamater by cross-validation.
  Those are used only if `fitType` = `:path` or = `:all`.

**Optional keyword arguments for fitting the model(s) using GLMNet**

`alpha`: the hyperparameter in ``[0, 1]`` to trade-off
an elestic-net model. ``α=0`` requests a pure *ridge* model and
``α=1`` a pure *lasso* model. This defaults to 1.0,
which specifies a lasso model, unless the input [`ENLR`](@ref) `model`
has another value in the `alpha` field, in which case this value
is used. If argument `alpha` is passed here, it will overwrite
the `alpha` field of the input `model`.

`weights`: a vector of weights for each matrix (or feature vectors)
of the same size as `yTr`.
It defaults to 1 for all matrices.

`intercept`: whether to fit an intercept term.
The intercept is always unpenalized. Defaults to true.

`standardize`: if true (default), GLMNet standardize the predictors
(presumably this amounts to transform to unit variance) so that they are
in the same units. This is a common choice for regularized
regression models.

`penalty_factor`: a vector of length ``n(n+1)/2``, where ``n``
is the dimension of the original PD matrices on which the model is applied,
of penalties for each predictor in the tangent vectors.
This defaults to all ones, which weights each predictor equally.
To specify that a predictor should be unpenalized,
set the corresponding entry to zero.

`constraints`: an ``[n(n+1)/2]`` x ``2`` matrix specifying lower bounds
(first column) and upper bounds (second column) on each predictor.
By default, this is [-Inf Inf] for each predictor (each element
of tangent vectors).

`offset`: see documentation of original GLMNet package [🎓](@ref).

`dfmax`: The maximum number of predictors in the largest model.

`pmax`: The maximum number of predictors in any model.

`nlambda`: The number of values of ``λ`` along the path to consider.

`lambda_min_ratio`: The smallest ``λ`` value to consider,
as a ratio of the value of ``λ`` that gives the null model
(*i.e.*, the model with only an intercept).
If the number of observations exceeds the number of variables,
this defaults to 0.0001, otherwise 0.01.

`lambda`: The ``λ`` values to consider for fitting.
By default, this is determined
from `nlambda` and `lambda_min_ratio`.

`tol`: Is the convergence criterion for both the computation
of a mean for projecting onto the tangent space
(if the metric requires an iterative algorithm)
and for the GLMNet fitting algorithm. Defaults to 1e-5.
In order to speed up computations, you may try to set a lower `tol`;
The convergence will be faster but more coarse,
with a possible drop of classification accuracy,
depending on the signal-to-noise ratio of the input features.

`maxit`: The maximum number of iterations of the cyclic coordinate
descent algorithm. If convergence is not achieved, a warning is returned.

`algorithm`: the algorithm used to find the regularization path.
Possible values are `:newtonraphson` (default) and
`:modifiednewtonraphson`.

For further informations on those arguments, refer to the
resources on the GLMNet package [🎓](@ref).

**Optional Keyword arguments for finding the best model by cv**

`λSelMeth` = `:sd1` (default), the best model is defined as the one
allowing the highest `cvλ.meanloss` within one standard deviation of the
minimum, otherwise it is defined as the one allowing the minimum `cvλ.meanloss`.
Note that in selecting a model, the model with only the intercept term,
if it exists, is ignored. See [`ENLRmodel`](@ref) for a description
of the `.cvλ` field of the model structure.

Arguments `nfolds` and `folds` are passed to the
`GLMNet.glmnetcv` function along with the `⏩` argument.
Please refer to the resources on GLMNet
for details [🎓](@ref).

**See**: [notation & nomenclature](@ref), [the ℍVector type](@ref).

**See also**: [`predict`](@ref), [`cvAcc`](@ref).

**Tutorial**: [Example using the ENLR model](@ref).

**Examples**
```
using PosDefManifoldML, PosDefManifold

# generate some data
PTr, PTe, yTr, yTe=gen2ClassData(10, 30, 40, 60, 80, 0.1)

# Fit an ENLR lasso model and find the best model by cross-validation:
m=fit(ENLR(), PTr, yTr)

# ... balancing the weights for tangent space mapping
m=fit(ENLR(), PTr, yTr; w=tsWeights(yTr))

# ... using the log-Eucidean metric for tangent space projection
m=fit(ENLR(logEuclidean), PTr, yTr)

# Fit an ENLR ridge model and find the best model by cv:
m=fit(ENLR(Fisher), PTr, yTr; alpha=0)

# Fit an ENLR elastic-net model (α=0.9) and find the best model by cv:
m=fit(ENLR(Fisher), PTr, yTr; alpha=0.9)

# Fit an ENLR lasso model and its regularization path:
m=fit(ENLR(), PTr, yTr; fitType=:path)

# Fit an ENLR lasso model, its regularization path
# and the best model found by cv:
m=fit(ENLR(), PTr, yTr; fitType=:all)

```

"""
function fit(model  :: ENLRmodel,
               𝐏Tr  :: Union{ℍVector, Matrix{Float64}},
               yTr  :: IntVector;
		   # parameters for projection onto the tangent space
           w        	:: Union{Symbol, Tuple, Vector} = [],
           meanISR  	:: Union{ℍ, Nothing} = nothing,
		   meanInit 	:: Union{ℍ, Nothing} = nothing,
           vecRange 	:: UnitRange = 𝐏Tr isa ℍVector ? (1:size(𝐏Tr[1], 2)) : (1:size(𝐏Tr, 2)),
           fitType  	:: Symbol = :best,
		   verbose  	:: Bool = true,
           ⏩      	   :: Bool = true,
           # arguments for `GLMNet.glmnet` function
           alpha            :: Real = model.alpha,
           weights          :: Vector{Float64} = ones(Float64, length(yTr)),
           intercept        :: Bool = true,
		   standardize  	:: Bool = true,
           penalty_factor   :: Vector{Float64} = ones(Float64, _getDim(𝐏Tr, vecRange)),
           constraints      :: Matrix{Float64} = [x for x in (-Inf, Inf), y in 1:_getDim(𝐏Tr, vecRange)],
           offsets          :: Union{Vector{Float64}, Nothing} = nothing,
           dfmax            :: Int = _getDim(𝐏Tr, vecRange),
           pmax             :: Int = min(dfmax*2+20, _getDim(𝐏Tr, vecRange)),
           nlambda          :: Int = 100,
           lambda_min_ratio :: Real = (length(yTr) < _getDim(𝐏Tr, vecRange) ? 1e-2 : 1e-4),
           lambda           :: Vector{Float64} = Float64[],
           tol              :: Real = 1e-5,
           maxit            :: Int = 1000000,
           algorithm        :: Symbol = :newtonraphson,
           # selection method
           λSelMeth :: Symbol = :sd1,
           # arguments for `GLMNet.glmnetcv` function
           nfolds   :: Int = min(10, div(size(yTr, 1), 3)),
           folds    :: Vector{Int} =
           begin
               n, r = divrem(size(yTr, 1), nfolds)
               shuffle!([repeat(1:nfolds, outer=n); 1:r])
           end,
           parallel ::Bool=true)

    ⌚=now() # get the time in ms
    ℳ=deepcopy(model) # output model

	# overwrite fields in `ℳ` if the user has passed them here as arguments,
	# otherwise use as arguments the values in the fields of `ℳ`, e.g., the default
	if alpha ≠ 1.0 ℳ.alpha = alpha else alpha = ℳ.alpha end

    # check w argument and get weights for input matrices
    (w=_getWeights(w, yTr, "fit ("*_modelStr(ℳ)*" model)")) == nothing && return

    # other checks
    𝐏Tr isa ℍVector ? nObs=length(𝐏Tr) : nObs=size(𝐏Tr, 1)
    !_check_fit(ℳ, nObs, length(yTr), length(w), length(weights), "ENLR") && return

	# project data onto the tangent space or just copy the features if 𝐏Tr is a matrix
	X=_getFeat_fit!(ℳ, 𝐏Tr, meanISR, meanInit, tol, w, vecRange, true, verbose, ⏩)

    # convert labels in GLMNet format
    y = convert(Matrix{Float64}, [(yTr.==1) (yTr.==2)])

    # write some fields in output model struct
    ℳ.standardize = standardize
    ℳ.intercept   = intercept
    ℳ.featDim     = size(X, 2)
	ℳ.vecRange    = vecRange

    # collect the argumenst for `glmnet` function excluding the `lambda` argument
    fitArgs_λ = (alpha            = alpha,
                 weights          = weights,
                 standardize      = standardize,
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

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    if fitType ∈(:path, :all)
        # fit the regularization path
        verbose && println("Fitting "*_modelStr(ℳ)*" reg. path...")
        ℳ.path = glmnet(X, y, Binomial();
                         lambda = lambda,
                         fitArgs_λ...) # glmnet Args but lambda
    end
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    if fitType ∈(:best, :all)
        verbose && println("Fitting best "*_modelStr(ℳ)*" model...")
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
```
function predict(model   :: ENLRmodel,
		𝐏Te     :: Union{ℍVector, Matrix{Float64}},
		what    :: Symbol = :labels,
		fitType :: Symbol = :best,
		onWhich :: Int    = Int(fitType==:best);
		transfer   :: Union{ℍ, Nothing} = nothing,
		verbose    :: Bool = true,
		⏩        :: Bool = true)
```

Given an [`ENLR`](@ref) `model` trained (fitted) on 2 classes
and a testing set of ``k`` positive definite matrices `𝐏Te` of type
[ℍVector](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#%E2%84%8DVector-type-1),

if `what` is `:labels` or `:l` (default), return
the predicted **class labels** for each matrix in `𝐏Te`,
as an [IntVector](@ref).
Those labels are '1' for class 1 and '2' for class 2;

if `what` is `:probabilities` or `:p`, return the predicted **probabilities**
for each matrix in `𝐏Te` to belong to each classe, as a ``k``-vector
of ``z`` vectors holding reals in ``[0, 1]`` (probabilities).
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

- if `onWhich` is zero, all model in the `model.path` will be used for
predictions, thus the output will be multiplied by the number of
models in `model.path`.

Argumet `onWhich` has no effect if `fitType` = `:best`.

!!! note "Nota Bene"
    By default, the [`fit`](@ref) function fits only the `best` model.
    If you want to use the `fitType` = `:path` option you need to invoke
    the fit function with optional keyword argument `fitType`=`:path` or
    `fitType`=`:all`. See the [`fit`](@ref) function for details.

Optional keyword argument `transfer` can be used to specify the principal
inverse square root (ISR) of a new mean to be used as base point for
projecting the matrices in `𝐏Te` onto the tangent space.
By default `transfer` is equal to nothing,
implying that the base point will be the mean used to fit the model.
Passing a new mean ISR allows the *adaptation* first described in
Barachant et *al.*(2013). Typically `transfer` is the ISR
of the mean of the matrices in `𝐏Te` or of a subset of them.
Notice that this actually performs *transfer learning* by parallel
transporting both the training and test data to the identity matrix
as defined in Zanini et *al.*(2018) and later taken up in
Rodrigues et *al.*(2019)[🎓](@ref).

If `verbose` is true (default), information is printed in the REPL.
This option is included to allow repeated calls to this function
without crowding the REPL.

If ⏩ = true (default) and `𝐏Te` is an ℍVector type, the projection onto the
tangent space is multi-threaded.

**See**: [notation & nomenclature](@ref), [the ℍVector type](@ref).

**See also**: [`fit`](@ref), [`cvAcc`](@ref), [`predictErr`](@ref).

**Examples**
```
using PosDefManifoldML, PosDefManifold

# generate some data
PTr, PTe, yTr, yTe=gen2ClassData(10, 30, 40, 60, 80)

# fit an ENLR lasso model and find the best model by cv
m=fit(ENLR(Fisher), PTr, yTr)

# predict labels from the best model
yPred=predict(m, PTe, :l)
# prediction error
predErr=predictErr(yTe, yPred)

# predict probabilities from the best model
predict(m, PTe, :p)

# output functions from the best model
predict(m, PTe, :f)

# fit a regularization path for an ENLR lasso model
m=fit(ENLR(Fisher), PTr, yTr; fitType=:path)

# predict labels using a specific model
yPred=predict(m, PTe, :l, :path, 10)

# predict labels for all models
yPred=predict(m, PTe, :l, :path, 0)
# prediction error for all models
predErr=[predictErr(yTe, yPred[:, i]) for i=1:size(yPred, 2)]

# predict probabilities from a specific model
predict(m, PTe, :p, :path, 12)

# predict probabilities from all models
predict(m, PTe, :p, :path, 0)

# output functions from specific model
predict(m, PTe, :f, :path, 3)

# output functions for all models
predict(m, PTe, :f, :path, 0)

```

"""
function predict(model   :: ENLRmodel,
                 𝐏Te     :: Union{ℍVector, Matrix{Float64}},
                 what    :: Symbol = :labels,
                 fitType :: Symbol = :best,
                 onWhich :: Int    = Int(fitType==:best);
			transfer   :: Union{ℍ, Nothing} = nothing,
            verbose    :: Bool = true,
            ⏩        :: Bool = true)

    ⌚=now()

    # checks
    if !_whatIsValid(what, "predict ("*_modelStr(model)*")") return end
    if !_fitTypeIsValid(fitType, "predict ("*_modelStr(model)*")") return end
    if fitType==:best && model.best==nothing @error 📌*", predict function: the best model has not been fitted; run the `fit`function with keyword argument `fitType=:best` or `fitType=:all`"; return end
    if fitType==:path && model.path==nothing @error 📌*", predict function: the regularization path has not been fitted; run the `fit`function with keyword argument `fitType=:path` or `fitType=:all`"; return end
    if !_ENLRonWhichIsValid(model, fitType, onWhich, "predict ("*_modelStr(model)*")") return end

    # projection onto the tangent space
	X=_getFeat_Predict!(model, 𝐏Te, transfer, model.vecRange, true, verbose, ⏩)

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
    println(io, separatorFont, "  ", _modelStr(M))
    println(io, separatorFont, "⭒  ⭒    ⭒       ⭒          ⭒", defaultFont)
    println(io, "type    : PD Tangent Space model")
    println(io, "features: tangent vectors of length $(M.featDim)")
    println(io, "classes : 2")
    println(io, separatorFont, "Fields  : ")
	# # #
	println(io, greyFont, " Tangent Space Parametrization", defaultFont)
    println(io, separatorFont," .metric      ", defaultFont, string(M.metric))
	if M.meanISR == nothing
        println(io, greyFont, " .meanISR      not created")
    else
        n=size(M.meanISR, 1)
        println(io, separatorFont," .meanISR     ", defaultFont, "$(n)x$(n) Hermitian matrix")
    end
    println(io, separatorFont," .vecRange    ", defaultFont, "$(M.vecRange)")
    println(io, separatorFont," .featDim     ", defaultFont, "$(M.featDim)")
	# # #
	println(io, greyFont, " ENLR Parametrization", defaultFont)
    println(io, separatorFont," .alpha       ", defaultFont, "$(round(M.alpha, digits=3))")
    println(io, separatorFont," .intercept   ", defaultFont, string(M.intercept))
	println(io, separatorFont," .standardize ", defaultFont, string(M.standardize))


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
