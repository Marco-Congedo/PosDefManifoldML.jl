#   Unit "enlr.jl" of the PosDefManifoldML Package for Julia language
#   v 0.2.0 - last update 11th of October 2019
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
    intercept   :: Bool
    standardize :: Bool
    meanISR     :: Union{ℍVector, Nothing}
    featDim     :: Int
    path        :: GLMNet.GLMNetPath
    cvλ         :: GLMNet.GLMNetCrossValidation
    bestModel   :: Int
    bestλ       :: Real
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
[`fit`](@ref) and [`cvLambda!`](@ref) functions:

`.alpha` is the hyperparameter in ``[0, 1]`` trading-off
the **elestic-net model**. ``α=0`` requests a pure **ridge** model and
``α=1`` a pure **lasso** model. This is passed as parameter to
the [`fit`](@ref) function, defaulting therein to ``α=1``.

`.intercept` is true (default) if the logistic regression model
has an intercept term.

`.standardize`. If true, predictors are standardized so that
they are in the same units. By default is true for lasso models
(``α=1``), false otherwise (``0≤α<1``).

`.meanISR` is optionally passed to the [`fit`](@ref)
function. By default it is computed thereby.

`.featDim` is the length of the vectorized tangent vectors.
This is given by ``n(n+1)/2``, where ``n``
is the dimension of the original PD matrices on which the model is applied
once they are mapped onto the tangent space.

`.path` is the following structure created using the
[GLMNet.jl](https://github.com/JuliaStats/GLMNet.jl) package
when the [`fit`](@ref) function is invoked:

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

`.cvλ` is the following structure created using the
[GLMNet.jl](https://github.com/JuliaStats/GLMNet.jl) package
when the [`cvLambda!`](@ref) function is invoked.

```
struct GLMNetCrossValidation
    path::GLMNetPath            # the cv path
    nfolds::Int                 # the number of folds for the cv
    lambda::Vector{Float64}     # lambda values for each solution
    meanloss::Vector{Float64}   # mean loss for each solution
    stdloss::Vector{Float64}    # standard deviation of the mean losses
end
```

`.bestModel` is the serial number of the model in `cvλ.path`
allowing the minimum `cvλ.meanloss`, or the highest `cvλ.meanloss`
within one standard deviation of the minimum, as specified by the user
when invoking the [`cvLambda!`](@ref) function.

`.bestλ` is the value of the lambda hyperparameter corresponding
to the `bestModel`.

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
only when an MDM model is needed as an argument to a function,
otherwise you will create and fit an ENLR model using
the [`fit`](@ref) function.

"""
mutable struct ENLR <: ENLRmodel
    metric   :: Metric
    alpha    :: Real
    intercept
    standardize
    meanISR
    featDim
    path
    cvλ
    bestModel
    bestλ
    function ENLR(metric :: Metric=Fisher;
               alpha     :: Real = 1.0,
               intercept   = nothing,
               standardize = nothing,
               meanISR     = nothing,
               featDim     = nothing,
               path        = nothing,
               cvλ         = nothing,
               bestModel   = nothing,
               bestλ       = nothing)
        new(metric, alpha, intercept, standardize,
            meanISR, featDim, path, cvλ, bestModel, bestλ)
    end
end


"""
```
function fit(model :: ENLRmodel,
              𝐏Tr   :: Union{ℍVector, Matrix{Float64}},
              yTr   :: Vector;
           w       :: Vector            = [],
           meanISR :: Union{ℍ, Nothing} = nothing,
           verbose :: Bool              = true,
           ⏩     :: Bool               = true,
           # (GLMNet) fitArgs
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
           algorithm        :: Symbol = :newtonraphson)

```
Fit an [`ENLR`](@ref) machine learning model,
with training data `𝐏Tr`, of type
[ℍVector](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#%E2%84%8DVector-type-1),
and corresponding labels `yTr`, of type [IntVector](@ref).
Return the fitted model.

Fitting an ENLR model involves computing a mean of all the
matrices in `𝐏Tr`, mapping all matrices onto the tangent space
after parallel transporting them at the identity matrix
and vectorizing them using the
[vecP](https://marco-congedo.github.io/PosDefManifold.jl/dev/riemannianGeometry/#PosDefManifold.vecP)
operation. Once this is done the elastic net logistic regression is fitted.

The mean is computed according to the `.metric` field
of the `model`, with optional weights `w`.
If weights are used, they should be inversely proportional to
the number of examples for each class, in such a way that each class
contributes equally to the computation of the mean. This argument is
passed to the [`tsMap`](@ref) function.

If `meanISR` is passed as argument, the mean is not computed,
instead this matrix is the inverse square root (ISR) of the mean
used for projecting the matrices in the tangent space (see [`tsMap`](@ref)).

If `verbose` is true (default), information is printed in the REPL.
This option is included to allow repeated calls to this function
without crowding the REPL.

The `⏩` argument (true by default) is passed to the [`tsMap`](@ref)
function for projecting the matrices in `𝐏Tr` onto the tangent space.

The remaining optional arguments, referred to as **fitArgs**,
are passed to the *GLMNet.glmnet* function to fit the elastic-net model.

`weights`: a vector of weights for each matrix of the same size as `yTr`.
Argument `w` is passed, defaulting to 1 for all matrices.

`alpha`: the hyperparameter in ``[0, 1]`` to trade-off
an elestic-net model. ``α=0`` requests a pure *ridge* model and
``α=1`` a pure *lasso* model. This defaults to 1.0,
which specifies a lasso model.

`intercept`: whether to fit an intercept term.
The intercept is always unpenalized. Defaults to true.

`standardize`: whether to standardize predictors so that they are in the
same units. Differently from *GLMNet.jl*, by default this is true for lasso models
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

`algorithm`: the algorithm used to find the `path`.

For further informations on the **fitArgs** arguments, refer to the
resources on the GLMNet package [🎓](@ref).

**See**: [notation & nomenclature](@ref), [the ℍVector type](@ref).

**See also**: [`predict`](@ref), [`cvAcc`](@ref).

**Examples**
```
using PosDefManifoldML

# generate some data
PTr, PTe, yTr, yTe=gen2ClassData(10, 30, 40, 60, 80, 0.25)

# create and fit an ENLR lasso model:
m=fit(ENLR(Fisher), PTr, yTr)

# create and fit an ENLR ridge model:
m=fit(ENLR(Fisher), PTr, yTr; alpha=0)

# create and fit an ENLR elastic net model with alpha=0.9:
m=fit(ENLR(Fisher), PTr, yTr; alpha=0.9)

```

"""
function fit(model :: ENLRmodel,
              𝐏Tr   :: Union{ℍVector, Matrix{Float64}},
              yTr   :: Vector;
           w       :: Vector            = [],
           meanISR :: Union{ℍ, Nothing} = nothing,
           verbose :: Bool              = true,
           ⏩     :: Bool               = true,
           # arguments of GLMNet
           alpha            :: Real = 1.0,
           standardize      :: Bool = alpha≈1.0 ? true : false,
           intercept        :: Bool = true,
           penalty_factor   :: Vector{Float64} = ones(_triNum(𝐏Tr[1])),
           constraints      :: Matrix{Float64} = [x for x in (-Inf, Inf), y in 1:_triNum(𝐏Tr[1])],
           offsets          :: Union{Vector{Float64},Nothing} = nothing,
           dfmax            :: Int = _triNum(𝐏Tr[1]),
           pmax             :: Int = min(dfmax*2+20, _triNum(𝐏Tr[1])),
           nlambda          :: Int = 100,
           lambda_min_ratio :: Real = (length(yTr) < _triNum(𝐏Tr[1]) ? 1e-2 : 1e-4),
           lambda           :: Vector{Float64} = Float64[],
           tol              :: Real = 1e-7,
           maxit            :: Int = 1000000,
           algorithm        :: Symbol = :newtonraphson)

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

    # write some fields in model struct
    model.alpha       = alpha
    model.intercept   = intercept
    model.standardize = standardize
    model.featDim     = size(X, 2)

    # fit the model
    verbose && println("Fitting "*_modelStr(model)*" models...")
    model.path = glmnet(X, y, Binomial();
                     offsets           = offsets,
                     weights           = isempty(w) ? ones(size(y, 1)) : w,
                     alpha             = alpha,
                     penalty_factor    = penalty_factor,
                     constraints       = constraints,
                     dfmax             = dfmax,
                     pmax              = pmax,
                     nlambda           = nlambda,
                     lambda_min_ratio  = lambda_min_ratio,
                     lambda            = lambda,
                     tol               = tol,
                     standardize       = standardize,
                     intercept         = intercept,
                     maxit             = maxit,
                     algorithm         = algorithm)

    verbose && println("Done in ", defaultFont, now()-⌚,".")
    return model
end


"""
```
function cvLambda!(model :: ENLRmodel,
                   𝐏Tr   :: Union{ℍVector, Matrix{Float64}},
                   yTr   :: Vector;
             w        :: Vector = [],
             λSelMeth :: Symbol = :sd1,
             verbose  :: Bool   = true,
             ⏩      :: Bool   = true,
             # arguments of GLMNet
             nfolds  :: Int = min(10, div(size(yTr, 1), 3)),
             folds   :: Vector{Int} =
             begin
                 n, r = divrem(size(yTr, 1), nfolds)
                 shuffle!([repeat(1:nfolds, outer=n); 1:r])
             end,
             parallel::Bool=false,
             fitArgs...)
```

Run a cross-validation to find the `bestModel`, using training data `𝐏Tr`,
of type
[ℍVector](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#%E2%84%8DVector-type-1),
and corresponding labels `yTr`, of type [IntVector](@ref).
Return the fitted model filling the `model.cvλ` field.

!!! note "Nota bene"
    The `model` must have been fitted calling the [`fit`](@ref)
    function before calling this function.

The optionl `w` weights are passed to the [`tsMap`](@ref) function,
as in the [`fit`](@ref) function.
By defaults all weights are equal to 1, but
here too they should be inversely proportional to
the number of examples for each class, in such a way that each class
contributes equally to the computation of the mean for projecting the
matrices onto the tangent space.

If `λSelMeth` = `:sd1` (default), the best model is the one
allowing the highest `cvλ.meanloss` within one standard deviation of the
minimum, otherwise it is the one allowing the minimum `cvλ.meanloss`.
Note that in selecting a model, the model with only the intercept term,
which is the first model if the [`fit`](@ref) function
has been run with argument `intercept`=true (default), is ignored.

If `verbose` is true (default), information is printed in the REPL.
This option is included to allow repeated calls to this function
without crowding the REPL.

The `⏩` argument (true by default) is passed to the [`tsMap`](@ref)
function for projecting the matrices in `𝐏Tr` onto the tangent space.

Arguments `nfolds`, `folds` and `parallel` are passed to the
`GLMNet.glmnetcv` function. Please refer to the resources on GLMNet
for details [🎓](@ref).

Arguments `fitArgs...` may be any set of **fitArgs** arguments
to be passed to the *GLMNet.glmnet* function.
See the documentation of the [`fit`](@ref) function.

**Examples**
```
using PosDefManifoldML

# generate some data
PTr, PTe, yTr, yTe=gen2ClassData(10, 30, 40, 60, 80, 0.25)

# create and fit an ENLR lasso model:
m=fit(ENLR(Fisher), PTr, yTr)

# estimate the best model by cross-validation
cvLambda!(m, PTr, yTr)

```

"""
function cvLambda!(model :: ENLRmodel,
                   𝐏Tr   :: Union{ℍVector, Matrix{Float64}},
                   yTr   :: Vector;
             w        :: Vector = [],
             λSelMeth :: Symbol = :sd1,
             verbose  :: Bool   = true,
             ⏩      :: Bool   = true,
             # arguments of GLMNet
             nfolds  :: Int = min(10, div(size(yTr, 1), 3)),
             folds   :: Vector{Int} =
             begin
                 n, r = divrem(size(yTr, 1), nfolds)
                 shuffle!([repeat(1:nfolds, outer=n); 1:r])
             end,
             parallel::Bool=false,
             fitArgs...)

    ⌚=now()

    # checks
    if model.path==nothing @error "the model has not been fitted"; return end
    𝐏Tr isa ℍVector ? dim=length(𝐏Tr) : dim=size(𝐏Tr, 1)
    !_check_fit(model, dim, length(yTr), length(w), "ENLR") && return

    # projection on the tangent space
    if 𝐏Tr isa ℍVector
        verbose && println(greyFont, "Projecting data onto the tangent space...")
        X=tsMap(model.metric, 𝐏Tr; w=w, ⏩=⏩, meanISR=model.meanISR)
    else X=𝐏Tr end

    # convert labels
    y = convert(Matrix{Float64}, [(yTr.==1) (yTr.==2)])

    # cross-validation
    verbose && println("Cross-validating "*_modelStr(model)*" models...")
    model.cvλ=glmnetcv(X, y, Binomial();
                   weights  = isempty(w) ? ones(size(y, 1)) : w,
                   nfolds   = nfolds,
                   folds    = folds,
                   parallel = parallel,
                   fitArgs...)

    # select best model
    # never consider the model with only the intercept (0 degrees of freedom)
    l, i=length(model.cvλ.lambda), max(argmin(model.cvλ.meanloss), 1+model.intercept)
    thr=model.cvλ.meanloss[i]+model.cvλ.stdloss[i]
    # if bestλsel==:sd1 select the highest model with means loss withinh 1sd of the minimum
    # otherwise the model with the smallest mean loss.
    λSelMeth==:sd1 ? (while i<l model.cvλ.meanloss[i+1]<=thr ? i+=1 : break end) : nothing
    # model in model.cvλ.path with lowest meanloss (+1sd)
    model.bestModel=i
    model.bestλ=(model.cvλ.path.lambda[i])

    # not sure the model shouldn't be found on the original path instead?
    # find the model in model.path with lambda value closest to the best model.cv.lambda value
    # bestModel=i
    # model.bestModel=argmin(abs.(broadcast(-, model.cvλ.lambda[i], model.path.lambda)))
    # model.bestλ=(model.path.lambda[bestModel])

    verbose && println("Done in ", defaultFont, now()-⌚,".\n")
    return model
end


"""
```
function predict(model  :: ENLRmodel,
                 𝐏Te    :: Union{ℍVector, Matrix{Float64}},
                 what   :: Symbol = :labels,
                 which  :: Union{Int, Nothing} = model.bestModel;
            verbose :: Bool = true,
            ⏩     :: Bool = true)
```

Given an [`ENLR`](@ref) `model` trained (fitted) on 2 classes
and a testing set of ``k`` positive definite matrices `𝐏Te` of type
[ℍVector](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#%E2%84%8DVector-type-1),

if `what` is `:labels` or `:l` (default), return
the predicted **class labels** for each matrix in `𝐏Te` as an [IntVector](@ref);

if `what` is `:probabilities` or `:p`, return the predicted **probabilities**
for each matrix in `𝐏Te` to belong to a all classes, as a ``k``-vector
of ``z`` vectors holding reals in ``[0, 1]`` (probabilities).

if `what` is `:f` or `:functions`, return the **output function** of the model
(see below).

If `verbose` is true (default), information is printed in the REPL.
This option is included to allow repeated calls to this function
without crowding the REPL.

The predicted class 'label' are '1' for class 1 and '2' for class 2.

The 'probabilities' are obtained passing to a
[softmax function](https://en.wikipedia.org/wiki/Softmax_function)
the output of the ENLR model and zero.

The 'functions' are the raw output of the ENLR model.

If the [`cvLambda!`](@ref) function has been previously called on the
`model`, the best model for the model has been selected and by default
this model will be used for prediction, otherwise

- if `which` is a valid serial number for a model in the `model.path`, this model is used for prediction,

- if `which` is zero, all model in the `model.path` will be used for predictions, thus the output will be multiplied by the number of models in `model.path`.


**See**: [notation & nomenclature](@ref), [the ℍVector type](@ref).

**See also**: [`fit`](@ref), [`cvAcc`](@ref), [`predictErr`](@ref).

**Examples**
```
using PosDefManifoldML

# generate some data
PTr, PTe, yTr, yTe=gen2ClassData(10, 30, 40, 60, 80)

# craete and fit an ENLR lasso model
m=fit(ENLR(Fisher), PTr, yTr)

# estimate the best model by cross-validation
cvLambda!(m, PTr, yTr)

# predict labels using the best model
yPred=predict(m, PTe, :l)
# prediction error
predErr=predictErr(yTe, yPred)

# predict labels using a specific model
yPred=predict(m, PTe, :l, 10)

# predict labels for all models
yPred=predict(m, PTe, :l, 0)
# prediction error for all models
predErr=[predictErr(yTe, yPred[:, i]) for i=1:size(yPred, 2)]

# predict probabilities using best model
predict(m, PTe, :p)

# predict probabilities using a specific model
predict(m, PTe, :p, 12)

# predict probabilities for all models
predict(m, PTe, :p, 0)

# output functions using best model
predict(m, PTe, :f)

# output functions using a specific model
predict(m, PTe, :f, 3)

# output functions for all models
predict(m, PTe, :f, 0)

```

"""
function predict(model  :: ENLRmodel,
                 𝐏Te    :: Union{ℍVector, Matrix{Float64}},
                 what   :: Symbol = :labels,
                 which  :: Union{Int, Nothing} = model.bestModel;
            verbose :: Bool = true,
            ⏩     :: Bool = true)

    ⌚=now()

    # checks
    if model.path==nothing @error "the model has not been fitted"; return end
    if !_whatIsValid(what, "predict (ENLR model)") return end
    if !_whichIsValid(model.path, which, "predict (ENLR model)") return end
    if which==nothing which=0 end # if cvLambda has not been run

    # projection onto the tangent space
    if 𝐏Te isa ℍVector
        verbose && println(greyFont, "Projecting data onto the tangent space...")
        X=tsMap(model.metric, 𝐏Te; meanISR=model.meanISR, ⏩=⏩)
    else X=𝐏Te end

    # prediction
    verbose && println("Predicting "*_whichENLRStr(model, which)*"...")
    path=model.cvλ.path # ??? path=model.path
    which==0 ? π=GLMNet.predict(path, X) :
               π=GLMNet.predict(path, X, which)
    k, l=size(π, 1), length(path.lambda)
    if     what == :functions || what == :f
        🃏=π
    elseif what == :labels || what == :l
        which==0 ? 🃏=[π[i, j]<0 ? 1 : 2 for i=1:k, j=1:l] : 🃏=[y<0 ? 1 : 2 for y ∈ π]
    elseif what == :probabilities || what == :p
        which==0 ? 🃏=[softmax([-π[i, j], 0]) for i=1:k, j=1:l] : 🃏=[softmax([-y, 0]) for y ∈ π]
    end

    verbose && println(greyFont, "Done in ", defaultFont, now()-⌚,".")
    verbose && println(titleFont, "\nPredicted ",_what2Str(what),":", defaultFont)
    return 🃏
end



# ++++++++++++++++++++  Show override  +++++++++++++++++++ # (REPL output)
function Base.show(io::IO, ::MIME{Symbol("text/plain")}, M::ENLR)
    if M.path==nothing
        println(io, greyFont, "\n↯ ENLR GLMNet machine learning model")
        println(io, "⭒  ⭒    ⭒       ⭒          ⭒")
        println(io, ".metric :", string(M.metric))
        println(io, ".alpha  :", "$(round(M.alpha, digits=3))", defaultFont)
        println(io, "Non-fitted model")
    else
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
        println(io, separatorFont," .path", defaultFont," struct `GLMNetPath`:")
        println(io, titleFont,"       .family, .a0, .betas, .null_dev, ")
        println(io, titleFont,"       .dev_ratio, .lambda, .npasses")
        if M.cvλ==nothing
            println(io, greyFont," .cvλ  struct `GLMNetCrossValidation`:")
            println(io, "       not created ")
            println(io, " .bestModel   unknown")
            println(io, " .bestλ       unknown")

        else
            println(io, separatorFont," .cvλ", defaultFont,"  struct `GLMNetCrossValidation`:")
            println(io, titleFont,"       .path, .nfolds, .lambda, ")
            println(io, titleFont,"       .meanloss, stdloss")
            println(io, separatorFont," .bestModel   ", defaultFont, "$(M.bestModel)")
            println(io, separatorFont," .bestλ       ", defaultFont, "$(round(M.bestλ; digits=5))")
        end
    end
end
