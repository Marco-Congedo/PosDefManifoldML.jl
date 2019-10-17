#   Unit "enlr.jl" of the PosDefManifoldML Package for Julia language
#   v 0.2.1 - last update 18th of October 2019
#
#   MIT License
#   Copyright (c) 2019,
#   Saloni Jain, Indian Institute of Technology, Kharagpur, India
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
    alpha       :: Real
    standardize :: Bool
    intercept   :: Bool
    meanISR     :: Union{ℍVector, Nothing}
    featDim     :: Int
    path        :: GLMNet.GLMNetPath
    cvλ         :: GLMNet.GLMNetCrossValidation
    best        :: GLMNet.GLMNetPath
end
```
ENLR machine learning models are incapsulated in this
mutable structure. Fields:

`.metric`, of type
[Metric](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#Metric::Enumerated-type-1),
is to be specified by the user.
It is the metric that will be adopted to compute the mean used
as base-point for tangent space projection.

All other fields do not correspond to arguments passed
upon creation of the model. Instead, they are filled later by the
[`fit`](@ref) function:

`.alpha` is the hyperparameter in ``[0, 1]`` trading-off
the **elestic-net model**. ``α=0`` requests a pure **ridge** model and
``α=1`` a pure **lasso** model. This is passed as parameter to
the [`fit`](@ref) function, defaulting therein to ``α=1``.

`.standardize`. If true, predictors are standardized so that
they are in the same units. By default is true for lasso models
(``α=1``), false otherwise (``0≤α<1``).

`.intercept` is true (default) if the logistic regression model
has an intercept term.

`.meanISR` is optionally passed to the [`fit`](@ref)
function. By default it is computed thereby.

`.featDim` is the length of the vectorized tangent vectors.
This is given by ``n(n+1)/2``, where ``n``
is the dimension of the original PD matrices on which the model is applied
once they are mapped onto the tangent space.

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
using PosDefManifoldML

# create an empty model
m = ENLR(Fisher)

# since the Fisher metric is the default metric,
# this is equivalent to
m = ENLR()
```

Note that in general you need to invoke these constructors
only when an ENLR model is needed as an argument to a function,
otherwise you will create and fit an ENLR model using
the [`fit`](@ref) function.

"""
mutable struct ENLR <: ENLRmodel
    metric   :: Metric
    alpha    :: Real
    standardize
    intercept
    meanISR
    featDim
    path
    cvλ
    best
    function ENLR(metric :: Metric=Fisher;
               alpha     :: Real = 1.0,
               standardize = nothing,
               intercept   = nothing,
               meanISR     = nothing,
               featDim     = nothing,
               path        = nothing,
               cvλ         = nothing,
               best        = nothing)
        new(metric, alpha, standardize, intercept,
            meanISR, featDim, path, cvλ, best)
    end
end


"""
```
function fit(model :: ENLRmodel,
               𝐏Tr :: Union{ℍVector, Matrix{Float64}},
               yTr :: Vector;
           w       :: Vector            = [],
           meanISR :: Union{ℍ, Nothing} = nothing,
           fitType :: Symbol            = :best,
           verbose :: Bool              = true,
           ⏩     :: Bool               = true,
                # arguments for `GLMNet.glmnet` function
           alpha            :: Real = model.alpha < 1.0 ? model.alpha : 1.0,
           intercept        :: Bool = true,
           standardize      :: Bool = alpha≈1.0 ? true : false,
           penalty_factor   :: Vector{Float64} = ones(_triNum(𝐏Tr[1])),
           constraints      :: Matrix{Float64} = [x for x in (-Inf, Inf), y in 1:_triNum(𝐏Tr[1])],
           offsets          :: Union{Vector{Float64}, Nothing} = nothing,
           dfmax            :: Int = _triNum(𝐏Tr[1]),
           pmax             :: Int = min(dfmax*2+20, _triNum(𝐏Tr[1])),
           nlambda          :: Int = 100,
           lambda_min_ratio :: Real = (length(yTr) < _triNum(𝐏Tr[1]) ? 1e-2 : 1e-4),
           lambda           :: Vector{Float64} = Float64[],
           tol              :: Real = 1e-7,
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
           parallel ::Bool=false)

```
Create and fit an [`ENLR`](@ref) machine learning model,
with training data `𝐏Tr`, of type
[ℍVector](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#%E2%84%8DVector-type-1),
and corresponding labels `yTr`, of type [IntVector](@ref).
Return the fitted model.

As for all ML models acting in the tangent space,
fitting an ENLR model involves computing a mean of all the
matrices in `𝐏Tr`, mapping all matrices onto the tangent space
after parallel transporting them at the identity matrix
and vectorizing them using the
[vecP](https://marco-congedo.github.io/PosDefManifold.jl/dev/riemannianGeometry/#PosDefManifold.vecP)
operation. Once this is done, the elastic net logistic regression is fitted.

The mean is computed according to the `.metric` field
of the `model`, with optional weights `w`.
If weights are used, they should be inversely proportional to
the number of examples for each class, in such a way that each class
contributes equally to the computation of the mean. The `.metric` field
of the `model` is passed to the [`tsMap`](@ref) function.
By default the metric is the Fisher metric. See the examples
here below to see how to change metric. See [mdm.jl](@ref)
for the available metrics.

If `meanISR` is passed as argument, the mean is not computed,
instead this matrix is the inverse square root (ISR) of the mean
used for projecting the matrices in the tangent space (see [`tsMap`](@ref)).

If `fitType` = `:best` (default), a cross-validation procedure is run
to find the best lambda hyperparameter for the training data. This finds
a single model that is written into the `.best` field of the model
that will be created.

If `fitType` = `:path`, the regularization path for several values of
the lambda hyperparameter if found for the training data.
This creates several models, which are written into the
`.path` field of the model
that will be created, none of which
is optimal, in the cross-validation sense, for the training data.

If `fitType` = `:all`, both the above fits are performed and all fields
of the model that will be created will be filled in.

If `verbose` is true (default), information is printed in the REPL.
This option is included to allow repeated calls to this function
without crowding the REPL.

The `⏩` argument (true by default) is passed to the [`tsMap`](@ref)
function for projecting the matrices in `𝐏Tr` onto the tangent space
using multi-threading.

The remaining optional keyword arguments, are

- the arguments passed to the `GLMNet.glmnet` function for fitting the models.
  Those are always used.

- the `λSelMeth` argument and the arguments passed to the `GLMNet.glmnetcv`
  function for finding the best lambda hyperparamater by cross-validation.
  Those are used only if `fitType` = `:path` or = `:all`.

**Optional Keyword arguments for fitting the model(s)**

`weights`: a vector of weights for each matrix of the same size as `yTr`.
Argument `w` is passed, defaulting to 1 for all matrices.

`alpha`: the hyperparameter in ``[0, 1]`` to trade-off
an elestic-net model. ``α=0`` requests a pure *ridge* model and
``α=1`` a pure *lasso* model. This defaults to 1.0,
which specifies a lasso model.

`intercept`: whether to fit an intercept term.
The intercept is always unpenalized. Defaults to true.

`standardize`: whether to standardize predictors so that they are in the
same units. Differently from GLMNet.jl, by default this is true for lasso models
(``α=1``), false otherwise (``0≤α<1``).

`penalty_factor`: a vector of length ``n(n+1)/2``, where ``n``
is the dimension of the original PD matrices on which the model is applied,
of penalties for each predictor in the tangent vectors.
This defaults to all ones, which weights each predictor equally.
To specify that a predictor should be unpenalized,
set the corresponding entry to zero.

`constraints`: an ``n(n+1)/2`` x ``2`` matrix specifying lower bounds
(first column) and upper bounds (second column) on each predictor.
By default, this is [-Inf Inf] for each predictor (each element
of tangent vectors).

`dfmax`: The maximum number of predictors in the largest model.

`pmax`: The maximum number of predictors in any model.

`nlambda`: The number of values of ``λ`` along the path to consider.

`lambda_min_ratio`: The smallest ``λ`` value to consider,
as a ratio of the value of ``λ`` that gives the null model
(*i.e.*, the model with only an intercept).
If the number of observations exceeds the number of variables,
this defaults to 0.0001, otherwise 0.01.

`lambda`: The ``λ`` values to consider. By default, this is determined
from `nlambda` and `lambda_min_ratio`.

`tol`: Convergence criterion. Defaults to 1e-7.

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

Arguments `nfolds`, `folds` and `parallel` are passed to the
`GLMNet.glmnetcv` function. Please refer to the resources on GLMNet
for details [🎓](@ref).

**See**: [notation & nomenclature](@ref), [the ℍVector type](@ref).

**See also**: [`predict`](@ref), [`cvAcc`](@ref).

**Examples**
```
using PosDefManifoldML

# generate some data
PTr, PTe, yTr, yTe=gen2ClassData(10, 30, 40, 60, 80, 0.1)

# Fit an ENLR lasso model and find the best model by cross-validation:
m=fit(ENLR(), PTr, yTr)

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
function fit(model :: ENLRmodel,
               𝐏Tr :: Union{ℍVector, Matrix{Float64}},
               yTr :: Vector;
           w       :: Vector            = [],
           meanISR :: Union{ℍ, Nothing} = nothing,
           fitType :: Symbol            = :best,
           verbose :: Bool              = true,
           ⏩     :: Bool               = true,
           # arguments for `GLMNet.glmnet` function
           alpha            :: Real = model.alpha < 1.0 ? model.alpha : 1.0,
           intercept        :: Bool = true,
           standardize      :: Bool = alpha≈1.0 ? true : false,
           penalty_factor   :: Vector{Float64} = ones(_triNum(𝐏Tr[1])),
           constraints      :: Matrix{Float64} = [x for x in (-Inf, Inf), y in 1:_triNum(𝐏Tr[1])],
           offsets          :: Union{Vector{Float64}, Nothing} = nothing,
           dfmax            :: Int = _triNum(𝐏Tr[1]),
           pmax             :: Int = min(dfmax*2+20, _triNum(𝐏Tr[1])),
           nlambda          :: Int = 100,
           lambda_min_ratio :: Real = (length(yTr) < _triNum(𝐏Tr[1]) ? 1e-2 : 1e-4),
           lambda           :: Vector{Float64} = Float64[],
           tol              :: Real = 1e-7,
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
           parallel ::Bool=false)

    ⌚=now()

    # checks
    𝐏Tr isa ℍVector ? dim=length(𝐏Tr) : dim=size(𝐏Tr, 1)
    !_check_fit(model, dim, length(yTr), length(w), "ENLR") && return

    # projection onto the tangent space
    if 𝐏Tr isa ℍVector
        verbose && println(greyFont, "Projecting data onto the tangent space...")
        if meanISR==nothing
            (X, G⁻½)=tsMap(model.metric, 𝐏Tr; w=w, ⏩=⏩)
            model.meanISR = G⁻½
        else
            X=tsMap(model.metric, 𝐏Tr; w=w, ⏩=⏩, meanISR=meanISR)
            model.meanISR = meanISR
        end
    else X=𝐏Tr
    end

    # convert labels
    y = convert(Matrix{Float64}, [(yTr.==1) (yTr.==2)])

    # get weights for GLMNet
    glmnetWeights = isempty(w) ? ones(size(y, 1)) : w

    # write some fields in model struct
    model.alpha       = alpha
    model.standardize = standardize
    model.intercept   = intercept
    model.featDim     = size(X, 2)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    if fitType ∈(:path, :all)
        # fit the regularization path
        verbose && println("Fitting "*_modelStr(model)*" reg. path...")
        model.path = glmnet(X, y, Binomial();
                         weights           = glmnetWeights,
                         alpha             = alpha,
                         standardize       = standardize,
                         intercept         = intercept,
                         penalty_factor    = penalty_factor,
                         constraints       = constraints,
                         offsets           = offsets,
                         dfmax             = dfmax,
                         pmax              = pmax,
                         nlambda           = nlambda,
                         lambda_min_ratio  = lambda_min_ratio,
                         lambda            = lambda,
                         tol               = tol,
                         maxit             = maxit,
                         algorithm         = algorithm)
    end
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    if fitType ∈(:best, :all)
        verbose && println("Fitting best "*_modelStr(model)*" model...")
        model.cvλ=glmnetcv(X, y, Binomial();
                        weights  = glmnetWeights,
                        nfolds   = nfolds,
                        folds    = folds,
                        parallel = parallel,
                        # glmnetArgs
                        alpha             = alpha,
                        standardize       = standardize,
                        intercept         = intercept,
                        penalty_factor    = penalty_factor,
                        constraints       = constraints,
                        offsets           = offsets,
                        dfmax             = dfmax,
                        pmax              = pmax,
                        nlambda           = nlambda,
                        lambda_min_ratio  = lambda_min_ratio,
                        lambda            = lambda,
                        tol               = tol,
                        maxit             = maxit,
                        algorithm         = algorithm)

        # Never consider the model with only the intercept (0 degrees of freedom)
        l, i=length(model.cvλ.lambda), max(argmin(model.cvλ.meanloss), 1+intercept)

        # if bestλsel==:sd1 select the highest model with mean loss withinh 1sd of the minimum
        # otherwise the model with the smallest mean loss.
        thr=model.cvλ.meanloss[i]+model.cvλ.stdloss[i]
        λSelMeth==:sd1 ? (while i<l model.cvλ.meanloss[i+1]<=thr ? i+=1 : break end) : nothing

        # fit the best model (only for the optimal lambda)
        model.best = glmnet(X, y, Binomial();
                          weights           = isempty(w) ? ones(size(y, 1)) : w,
                          alpha             = alpha,
                          standardize       = standardize,
                          intercept         = intercept,
                          penalty_factor    = penalty_factor,
                          constraints       = constraints,
                          offsets           = offsets,
                          dfmax             = dfmax,
                          pmax              = pmax,
                          nlambda           = nlambda,
                          lambda_min_ratio  = lambda_min_ratio,
                          lambda            = [model.cvλ.path.lambda[i]],
                          tol               = tol,
                          maxit             = maxit,
                          algorithm         = algorithm)
    end
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    verbose && println(defaultFont, "Done in ", now()-⌚,".")
    return model
end



"""
```
function predict(model   :: ENLRmodel,
                 𝐏Te     :: Union{ℍVector, Matrix{Float64}},
                 what    :: Symbol = :labels,
                 fitType :: Symbol = :best,
                 onWhich :: Int    = Int(fitType==:best);
            checks  :: Bool = true,
            verbose :: Bool = true,
            ⏩     :: Bool = true)
```

Given an [`ENLR`](@ref) `model` trained (fitted) on 2 classes
and a testing set of ``k`` positive definite matrices `𝐏Te` of type
[ℍVector](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#%E2%84%8DVector-type-1),

if `what` is `:labels` or `:l` (default), return
the predicted **class labels** for each matrix in `𝐏Te`,
as an [IntVector](@ref).
Those labels are '1' for class 1 and '2' for class 2;

if `what` is `:probabilities` or `:p`, return the predicted **probabilities**
for each matrix in `𝐏Te` to belong to a all classes, as a ``k``-vector
of ``z`` vectors holding reals in ``[0, 1]`` (probabilities).
The 'probabilities' are obtained passing to a
[softmax function](https://en.wikipedia.org/wiki/Softmax_function)
the output of the ENLR model and zero;

if `what` is `:f` or `:functions`, return the **output function** of the model,
which is the raw output of the ENLR model.

If `fitType` = `:best` (default), the best model that has been found by
cross-validation is used for prediction.

If `fitType` = `:path` (default),

- if `onWhich` is a valid serial number for a model in the `model.path`, this model is used for prediction,

- if `onWhich` is zero, all model in the `model.path` will be used for predictions, thus the output will be multiplied by the number of models in `model.path`.

Argumet `onWhich` has no effect if `fitType` = `:best`.

!!! note "Nota Bene"
    By default, the [`fit`](@ref) function fits only the `best` model.
    If you want to use the `fitType` = `:path` option you need to invoke
    the fit function with optional keyword argument `fitType`=`:path` or
    `fitType`=`:all`. See the [`fit`](@ref) function for details.

If `checks` is true (default), checks on the validity of the arguments
are performed. This can be set to false to spped up computations.

If `verbose` is true (default), information is printed in the REPL.
This option is included to allow repeated calls to this function
without crowding the REPL.

If ⏩ = true (default) and `𝐏Te` is an ℍVector type, the projection onto the
tangent space is multi-threaded.

**See**: [notation & nomenclature](@ref), [the ℍVector type](@ref).

**See also**: [`fit`](@ref), [`cvAcc`](@ref), [`predictErr`](@ref).

**Examples**
```
using PosDefManifoldML

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
            checks  :: Bool = true,
            verbose :: Bool = true,
            ⏩     :: Bool = true)

    ⌚=now()

    # checks
    if checks
        if !_whatIsValid(what, "predict ("*_modelStr(model)*")") return end
        if !_fitTypeIsValid(fitType, "predict ("*_modelStr(model)*")") return end
        if fitType==:best && model.best==nothing @error 📌*", predict function: the best model has not been fitted"; return end
        if fitType==:path && model.path==nothing @error 📌*", predict function: the regularization path has not been fitted"; return end
        if !_ENLRonWhichIsValid(model, fitType, onWhich, "predict ("*_modelStr(model)*")") return end
    end

    # projection onto the tangent space
    if 𝐏Te isa ℍVector
        verbose && println(greyFont, "Projecting data onto the tangent space...")
        X=tsMap(model.metric, 𝐏Te; meanISR=model.meanISR, ⏩=⏩)
    else X=𝐏Te end

    # prediction
    verbose && println("Predicting "*_ENLRonWhichStr(model, fitType, onWhich)*"...")
    if 		fitType==:best
        path=model.best
        onWhich=1
    elseif  fitType==:path
        path=model.path
    end
    onWhich==0 ? π=GLMNet.predict(path, X) :
                 π=GLMNet.predict(path, X, onWhich)
    k, l=size(π, 1), length(path.lambda)
    if     what == :functions || what == :f
        🃏=π
    elseif what == :labels || what == :l
        onWhich==0 ? 🃏=[π[i, j]<0 ? 1 : 2 for i=1:k, j=1:l] : 🃏=[y<0 ? 1 : 2 for y ∈ π]
    elseif what == :probabilities || what == :p
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
            println(io, ".metric :", string(M.metric))
            println(io, ".alpha  :", "$(round(M.alpha, digits=3))", defaultFont)
            println(io, "Non-fitted model")
            return
        end
    end

    n=size(M.meanISR, 1)
    println(io, titleFont, "\n↯ ENLR GLMNet machine learning model")
    println(io, defaultFont, "  ", _modelStr(M))
    println(io, separatorFont, "⭒  ⭒    ⭒       ⭒          ⭒", defaultFont)
    println(io, "type    : PD Tangent Space model")
    println(io, "features: tangent vectors of length $(M.featDim)")
    println(io, "classes : 2")
    println(io, "fields  : ")
    println(io, separatorFont," .metric      ", defaultFont, string(M.metric))
    println(io, separatorFont," .alpha       ", defaultFont, "$(round(M.alpha, digits=3))")
    println(io, separatorFont," .intercept   ", defaultFont, string(M.intercept))
    println(io, separatorFont," .standardize ", defaultFont, string(M.standardize))
    println(io, separatorFont," .meanISR     ", defaultFont, "$(n)x$(n) Hermitian matrix")
    println(io, separatorFont," .featDim     ", defaultFont, "$(M.featDim)")
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
