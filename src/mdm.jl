#   Unit "mdm.jl" of the PosDefManifoldML Package for Julia language

#   MIT License
#   Copyright (c) 2019-2020
#   Marco Congedo, CNRS, Grenoble, France:
#   https://sites.google.com/site/marcocongedo/home
#   Saloni Jain, Indian Institute of Technology, Kharagpur, India

# ? CONTENTS :
#   This unit implements the Riemannian minimum distance to mean
#   machine learning classifier using package PosDefManifold.


"""
Abstract type for MDM (Minimum Distance to Mean)
machine learning models
"""
abstract type MDMmodel<:PDmodel end


"""
```
mutable struct MDM <: MDMmodel
    metric  :: Metric = Fisher;
    featDim :: Int
    means   :: ℍVector
    imeans  :: ℍVector
end
```

MDM machine learning models are incapsulated in this
mutable structure. MDM models have three fields:
`.metric`, `.featDim` and `.means`.

The field `metric`, of type
[Metric](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#Metric::Enumerated-type-1),
is to be specified by the user.
It is the metric that will be adopted to compute the class means
and the distances to the mean.

The field `featDim` is the dimension of the manifold in which
the model acts. This is given by ``n(n+1)/2``, where ``n``
is the dimension of the PD matrices.
This field is not to be specified by the user, instead,
it is computed when the MDM model is fit using the [`fit`](@ref)
function and is accessible only thereafter.

The field `means` is an
[ℍVector](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#%E2%84%8DVector-type-1)
holding the class means, i.e., one mean for each class.
This field is not to be specified by the user, instead,
the means are computed when the MDM model is fitted using the
[`fit`](@ref) function and are accessible only thereafter.

The field `imeans` is an ℍVector holding the inverse of the
matrices in `means`. This also is not to be specified by the user,
is computed when the model is fitted and is accessible only thereafter.
It is used to optimize the computation of distances if the
model is fitted useing the Fisher metric (default).

**Examples**:
```
using PosDefManifoldML, PosDefManifold

# create an empty model
m = MDM(Fisher)

# since the Fisher metric is the default metric,
# this is equivalent to
m = MDM()
```

Note that in general you need to invoke these constructors
only when an MDM model is needed as an argument to a function,
otherwise you can more simply create and fit an MDM model using
the [`fit`](@ref) function.

"""
mutable struct MDM <: MDMmodel
    metric :: Metric
    featDim
    means
    imeans
    function MDM(metric :: Metric = Fisher;
              featDim = nothing,
              means   = nothing,
              imeans  = nothing)
        new(metric, featDim, means)
    end
end



"""
```
function fit(model :: MDMmodel,
              𝐏Tr   :: ℍVector,
              yTr   :: IntVector;
       w        :: Vector = [],
       ✓w       :: Bool  = true,
       meanInit :: Union{ℍVector, Nothing} = nothing,
       tol      :: Real  = 1e-5,
       verbose  :: Bool  = true,
       ⏩       :: Bool  = true)
```

Fit an [`MDM`](@ref) machine learning model,
with training data `𝐏Tr`, of type
[ℍVector](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#%E2%84%8DVector-type-1),
and corresponding labels `yTr`, of type [IntVector](@ref).
Return the fitted model.

Labels must be provided using the natural numbers, i.e.,
`1` for the first class, `2` for the second class, etc.

Fitting an MDM model involves only computing a mean of all the
matrices in each class. Those class means are computed according
to the metric specified by the [`MDM`](@ref) constructor.

Optional keyword argument `w` is a vector of non-negative weights
associated with the matrices in `𝐏Tr`.
This weights are used to compute the mean for each class.
See method (3) of the [mean](https://marco-congedo.github.io/PosDefManifold.jl/dev/riemannianGeometry/#Statistics.mean)
function for the meaning of the arguments
`w`, `✓w` and `⏩`, to which they are passed.
Keep in mind that here the weights should sum up to 1
separatedly for each class, which is what is ensured by this
function if `✓w` is true.

Optional keyword argument `tol` is the tolerance required for those algorithms
that compute the mean iteratively (they are those adopting the Fisher, logde0
or Wasserstein metric). It defaults to 1e-5. For details on this argument see
the functions that are called for computing the means:
- Fisher metric: [gmean](https://marco-congedo.github.io/PosDefManifold.jl/dev/riemannianGeometry/#PosDefManifold.geometricMean)
- logdet0 metric: [ld0mean](https://marco-congedo.github.io/PosDefManifold.jl/dev/riemannianGeometry/#PosDefManifold.logdet0Mean)
- Wasserstein metric: [Wasmean](https://marco-congedo.github.io/PosDefManifold.jl/dev/riemannianGeometry/#PosDefManifold.wasMean).
For those algorithm an initialization can be provided with optional keyword
argument `meanInit`. If provided, this must be a vector of `Hermitian`
matrices of the [ℍVector](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#%F0%9D%95%84Vector-type-1)
type and must contain as many initializations as classes, in the
natural order corresponding to the class labels (see above).

If `verbose` is true (default), information is printed in the REPL.
This option is included to allow repeated calls to this function
without crowding the REPL.

**See**: [notation & nomenclature](@ref), [the ℍVector type](@ref).

**See also**: [`predict`](@ref), [`cvAcc`](@ref).

**Examples**
```
using PosDefManifoldML, PosDefManifold

# generate some data
PTr, PTe, yTr, yTe=gen2ClassData(10, 30, 40, 60, 80, 0.25)

# create and fit a model:
m=fit(MDM(Fisher), PTr, yTr)
```

"""
function fit(model :: MDMmodel,
              𝐏Tr   :: ℍVector,
              yTr   :: IntVector;
       w        :: Vector = [],
       ✓w       :: Bool   = true,
       meanInit :: Union{ℍVector, Nothing} = nothing,
       tol      :: Real   = 1e-5,
       verbose  :: Bool   = true,
       ⏩       :: Bool   = true)

    ⌚=now()

    ℳ=deepcopy(model) # output model

    k=length(𝐏Tr) # number of matrices
    !_check_fit(ℳ, k, length(yTr), length(w), 0, "MDM") && return

    verbose && println(greyFont, "Computing class means...")
    z = length(unique(yTr)) # number of classes
    𝐏 = [ℍ[] for i = 1:z]
    W = [Float64[] for i = 1:z]
    for j = 1:k push!(𝐏[yTr[j]], 𝐏Tr[j]) end
    if !isempty(w) for j = 1:k push!(W[yTr[j]], w[j]) end end

    meanInit==nothing ? ℳ.means = ℍVector([getMean(ℳ.metric, 𝐏[i];
                                              w=W[i], ✓w=✓w, tol=tol, ⏩=⏩) for i=1:z]) :
                        ℳ.means = ℍVector([getMean(ℳ.metric, 𝐏[i];
                                             w=W[i], ✓w=✓w, meanInit=meanInit[i], tol=tol, ⏩=⏩) for i=1:z])

    # store the inverse of the means for optimizing distance computations
    # if the metric is Fisher and the matrices are small
    if ℳ.metric==Fisher
        if size(𝐏Tr[1], 1)<=100
            if ⏩
                ℳ.imeans=ℍVector(undef, length(ℳ.means))
                @threads for i=1:length(ℳ.means) @inbounds ℳ.imeans[i]=inv(ℳ.means[i]) end
            else
                ℳ.imeans=ℍVector([inv(G) for G ∈ ℳ.means])
            end
        end
    else ℳ.imeans=nothing
    end

    ℳ.featDim =_triNum(𝐏Tr[1])

    verbose && println(defaultFont, "Done in ", now()-⌚,".")
    return ℳ
end




"""
```
function predict(model  :: MDMmodel,
                 𝐏Te    :: ℍVector,
                 what   :: Symbol = :labels;
        verbose :: Bool = true,
        ⏩     :: Bool = true)
```
Given an [`MDM`](@ref) `model` trained (fitted) on ``z`` classes
and a testing set of ``k`` positive definite matrices `𝐏Te` of type
[ℍVector](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#%E2%84%8DVector-type-1),

if `what` is `:labels` or `:l` (default), return
the predicted **class labels** for each matrix in `𝐏Te`,
as an [IntVector](@ref).
For MDM models, the predicted class 'label' of an unlabeled matrix is the
serial number of the class whose mean is the closest to the matrix
(minimum distance to mean).
The labels are '1' for class 1, '2' for class 2, etc;

if `what` is `:probabilities` or `:p`, return the predicted **probabilities**
for each matrix in `𝐏Te` to belong to a all classes, as a ``k``-vector
of ``z`` vectors holding reals in ``[0, 1]`` (probabilities).
The 'probabilities' are obtained passing to a
[softmax function](https://en.wikipedia.org/wiki/Softmax_function)
minus the squared distances of each unlabeled matrix to all class means;

if `what` is `:f` or `:functions`, return the **output function** of the model.
The ratio of the squared distance to all classes to
their geometric mean gives the 'functions'.

If `verbose` is true (default), information is printed in the REPL.
This option is included to allow repeated calls to this function
without crowding the REPL.

It f `⏩` is true (default), the computation of distances is multi-threaded.


**See**: [notation & nomenclature](@ref), [the ℍVector type](@ref).

**See also**: [`fit`](@ref), [`cvAcc`](@ref), [`predictErr`](@ref).

**Examples**
```
using PosDefManifoldML, PosDefManifold

# generate some data
PTr, PTe, yTr, yTe=gen2ClassData(10, 30, 40, 60, 80)

# craete and fit an MDM model
m=fit(MDM(Fisher), PTr, yTr)

# predict labels
yPred=predict(m, PTe, :l)

# prediction error
predErr=predictErr(yTe, yPred)

# predict probabilities
predict(m, PTe, :p)

# output functions
predict(m, PTe, :f)

```
"""
function predict(model  :: MDMmodel,
                 𝐏Te    :: ℍVector,
                 what   :: Symbol = :labels;
            verbose :: Bool = true,
            ⏩     :: Bool = true)

    if !_whatIsValid(what, "predict (MDM model)") return end
    ⌚=now()

    verbose && println(greyFont, "Computing distances...")
    D = getDistances(model.metric, model.means, 𝐏Te; imeans=model.imeans, ⏩=⏩)
    (z, k)=size(D)

    verbose && println("Predicting...")
    if     what == :functions || what == :f
           gmeans=[PosDefManifold.mean(Fisher, D[:, j]) for j = 1:k]
           func(j::Int)=[D[i, j]/gmeans[j] for i=1:z]
           🃏 = [func(j) for j = 1:k]
    elseif what == :labels || what == :l
           🃏 = [findmin(D[:, j])[2] for j = 1:k]
    elseif what == :probabilities || what == :p
           🃏 = [softmax(-D[:, j]) for j = 1:k]
    end

    verbose && println(defaultFont, "Done in ", now()-⌚,".")
    verbose && println(titleFont, "\nPredicted ",_what2Str(what),":", defaultFont)
    return 🃏
end




"""
```
function getMean(metric :: Metric,
                 𝐏      :: ℍVector;
              w        :: Vector = [],
              ✓w       :: Bool    = true,
              meanInit :: Union{ℍ, Nothing} = nothing,
              tol      :: Real   = 0.,
              ⏩      :: Bool    = true)
```

Typically, you will not need this function as it is called by the
[`fit`](@ref) function.

Given a `metric` of type
[Metric](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#Metric::Enumerated-type-1),
an [ℍVector](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#%E2%84%8DVector-type-1)
of Hermitian matrices `𝐏` and an optional
non-negative real weights vector `w`,
return the (weighted) mean of the matrices in `𝐏`.
This is used to fit MDM models.

This function calls the appropriate mean functions of package
[PostDefManifold](https://marco-congedo.github.io/PosDefManifold.jl/dev/),
depending on the chosen `metric`,
and check that, if the mean is found by an iterative algorithm,
then the iterative algorithm converges.

See method (3) of the [mean](https://marco-congedo.github.io/PosDefManifold.jl/dev/riemannianGeometry/#Statistics.mean)
function for the meaning of the optional keyword arguments
`w`, `✓w`, `meanInit`, `tol` and `⏩`, to which they are passed.

The returned mean is flagged by Julia as an Hermitian matrix
(see [LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/)).

"""
function getMean(metric :: Metric,
                 𝐏      :: ℍVector;
              w        :: Vector = [],
              ✓w       :: Bool    = true,
              meanInit :: Union{ℍ, Nothing} = nothing,
              tol      :: Real   = 0.,
              ⏩      :: Bool    = true)

  tol==0. ? tolerance = √eps(real(eltype(𝐏[1]))) : tolerance = tol

  if      metric == Fisher
              G, iter, convergence = gMean(𝐏; w=w, ✓w=✓w, init=meanInit, tol=tolerance, ⏩=⏩)
  elseif  metric == logdet0
              G, iter, convergence = ld0Mean(𝐏; w=w, ✓w=✓w, init=meanInit, tol=tolerance, ⏩=⏩)
  elseif  metric == Wasserstein
              G, iter, convergence = wasMean(𝐏; w=w, ✓w=✓w, init=meanInit, tol=tolerance, ⏩=⏩)
  else        G = mean(metric, 𝐏, w=w, ✓w=✓w, ⏩=⏩)
  end

  if metric ∈ (Fisher, logdet0, Wasserstein) && convergence > tolerance
      tolerance == 0. ? toltype="default" : toltype="chosen"
      @error 📌*", getMean function: the iterative algorithm for computing
      the means did not converge using the "*toltype*" tolerance.
      Check your data and try an higher tolerance (with the `tol`=... argument)."
  else
      return G
  end
end



"""
```
function getDistances(metric :: Metric,
                      means  :: ℍVector,
                      𝐏      :: ℍVector;
                imeans  :: Union{ℍVector, Nothing} = nothing,
                scale   :: Bool = false,
                ⏩      :: Bool = true)
```
Typically, you will not need this function as it is called by the
[`predict`](@ref) function.

Given an [ℍVector](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#%E2%84%8DVector-type-1)
`𝐏` holding ``k`` Hermitian matrices and
an ℍVector `means` holding ``z`` matrix means,
return the *square of the distance* of each matrix in `𝐏` to the means
in `means`.

The squared distance is computed according to the chosen `metric`, of type
[Metric](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#Metric::Enumerated-type-1).
See [metrics](https://marco-congedo.github.io/PosDefManifold.jl/dev/introToRiemannianGeometry/#metrics-1)
for details on the supported distance functions.

The computation of distances is optimized for the Fisher metric
if an ℍVector holding the inverse of the means in `means` is passed as
optional keyword argument `imeans`. For other metrics this argument
is ignored.

If `scale` is true,
the distances are divided by the size of the matrices in `𝐏`.
This is used to compare disctances computed on manifolds with
different dimensions.

If `⏩` is true, the distances are computed using multi-threading,
unless the number of threads Julia is instructed to use is <2 or <3k.

The result is a ``z``x``k`` matrix of squared distances.

"""
function getDistances(metric :: Metric,
             means  :: ℍVector,
             𝐏      :: ℍVector;
          imeans :: Union{ℍVector, Nothing} = nothing,
          scale  :: Bool = false,
          ⏩    :: Bool = true)

    z, k = length(means), length(𝐏)
    if ⏩
        D = Matrix{eltype(𝐏[1])}(undef, z, k)
        threads, ranges = _GetThreadsAndLinRanges(length(𝐏), "getDistances")

        dist(i::Int, r::Int) =
           if metric==Fisher && imeans≠nothing
               for j in ranges[r] D[i, j]=sum(log.(eigvals(imeans[i]*𝐏[j]; permute=false, scale=false)).^2) end
           else
               for j in ranges[r] D[i, j]=distance²(metric, 𝐏[j], means[i]) end
           end

        for i=1:z @threads for r=1:length(ranges) dist(i, r) end end
    else
        D=[distance²(metric, 𝐏[j], means[i]) for i=1:z, j=1:k]
    end
    # optimize in PosDefManifold, don't need to compute all distances for some metrics
    return scale ? D./size(𝐏[1], 1) : D
end



# ++++++++++++++++++++  Show override  +++++++++++++++++++ # (REPL output)
function Base.show(io::IO, ::MIME{Symbol("text/plain")}, M::MDM)
    if M.means==nothing
        println(io, greyFont, "\n↯ MDM machine learning model")
        println(io, "⭒  ⭒    ⭒       ⭒          ⭒")
        println(io, ".metric : ", string(M.metric), defaultFont)
        println(io, "Unfitted model")
    else
        println(io, titleFont, "\n↯ MDM machine learning model")
        println(io, separatorFont, "⭒  ⭒    ⭒       ⭒          ⭒", defaultFont)
        nc=length(M.means)
        n=size(M.means[1], 1)
        println(io, "type    : PD Manifold model")
        println(io, "features: $(n)x$(n) Hermitian matrices")
        println(io, "classes : $(nc)")
        println(io, separatorFont, "Fields  : ")
        # # #
        println(io, greyFont, " MDM Parametrization", defaultFont)
        println(io, separatorFont," .metric  ", defaultFont, string(M.metric))
        println(io, separatorFont," .means   ", defaultFont, "vector of $(nc) Hermitian matrices")
        println(io, separatorFont," .imeans  ", defaultFont, "vector of $(nc) Hermitian matrices")
        println(io, separatorFont," .featDim ", defaultFont, "$(M.featDim) ($(n)*($(n)+1)/2)")
    end
end
