#   Unit "mdm.jl" of the PosDefManifoldML Package for Julia language
#   v 0.0.1 - last update 28th of September 2019
#
#   MIT License
#   Copyright (c) 2019,
#   Saloni Jain, Indian Institute of Technology, Kharagpur, India
#   Marco Congedo, CNRS, Grenobe, France:
#   https://sites.google.com/site/marcocongedo/home

# ? CONTENTS :
#   This unit implements the Riemannian minimum distance to mean
#   machine learning classifier using package PosDefManifold.


"""
```
(1)
mutable struct MDM <: MLmodel
    metric :: Metric
    means
    function MDM(metric :: Metric; means = nothing)
        new(metric, means)
    end
end

(2)
function MDM(metric :: Metric,
             𝐏Tr    :: ℍVector,
             yTr    :: IntVector;
           w  :: Vector = [],
           ✓w :: Bool  = true)
```
(1)

MDM machine learning models are incapsulated in this
mutable structure. MDM models have two fields: `.metric` and `.means`.

The field `metric`, of type
[Metric](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#Metric::Enumerated-type-1),
is to be specified by the user.
It is the metric that will be adopted to compute the class means.

The field `means` is an
[ℍVector](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#%E2%84%8DVector-type-1)
holding the class means, i.e., one mean for each class.
This field is not to be specified by the user, instead,
the means are computed when the MDM model is fit using the [`fit!`](@ref)
function and are accessible only thereafter.

(2)

Constructor creating and fitting an MDM model with training data `𝐏Tr`, an
[ℍVector](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#%E2%84%8DVector-type-1)
type, and labels `yTr`, an [IntVector](@ref) type.
The class means are computed according to the chosen `metric`,
of type
[Metric](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#Metric::Enumerated-type-1).
See [here](https://marco-congedo.github.io/PosDefManifold.jl/dev/introToRiemannianGeometry/#metrics-1)
for details on the metrics. Supported metrics are listed in the section
about creating an [MDM model](@ref).

Optional keyword arguments `w` and `✓w` are passed to the [`fit!`](@ref)
function and have the same meaning therein.

**Examples**:
```
using PosDefManifoldML

# (1)
# generate some data
𝐏Tr, 𝐏Te, yTr, yTe = gen2ClassData(10, 30, 40, 60, 80)

# create a model
model = MDM(Fisher)

# fit the model with training data
fit!(model, 𝐏Tr, yTr)

# (2) equivalently and faster:
𝐏Tr, 𝐏Te, yTr, yTe=gen2ClassData(10, 30, 40, 60, 80)
model = MDM(Fisher, 𝐏Tr, yTr)
```

"""
mutable struct MDM <: MLmodel
    metric :: Metric
    means
    function MDM(metric :: Metric; means = nothing)
        new(metric, means)
    end
end


MDM(metric :: Metric,
    𝐏Tr      :: ℍVector,
    yTr      :: IntVector;
  w  :: Vector = [],
  ✓w :: Bool  = true) = fit!(MDM(metric), 𝐏Tr, yTr; w=w, ✓w=✓w)



"""
```
function getMeans(metric :: Metric,
                  𝐏      :: ℍVector;
              tol :: Real = 0.,
              w   :: Vector = [],
              ✓w :: Bool   = true,
              ⏩ :: Bool   = true)
```

Typically, you will not need this function as it is called by the
[`fit!`](@ref) function.

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
`w`, `✓w` and `⏩`, to which they are passed.

The returned mean is flagged by Julia as an Hermitian matrix
(see [LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/)).

"""
function getMeans(metric :: Metric,
                  𝐏      :: ℍVector;
              tol :: Real = 0.,
              w   :: Vector = [],
              ✓w :: Bool   = true,
              ⏩ :: Bool   = true)

    tol==0. ? tolerance = √eps(real(eltype(𝐏[1]))) : tolerance = tol

    if      metric == Fisher
                G, iter, convergence = gMean(𝐏; w=w, ✓w=✓w, ⏩=⏩)
    elseif  metric == logdet0
                G, iter, convergence = ld0Mean(𝐏; w=w, ✓w=✓w, ⏩=⏩)
    elseif  metric == Wasserstein
                G, iter, convergence = wasMean(𝐏; w=w, ✓w=✓w, ⏩=⏩)
    else    G = mean(metric, 𝐏, w=w, ✓w=✓w, ⏩=⏩)
    end

    if metric ∈ (Fisher, logdet0, Wasserstein) && convergence > tolerance
        tolerance == 0. ? toltype="defualt" : toltype="chosen"
        @error 📌*", getMeans function: the iterative algorithm for computing
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
                      𝐏      :: ℍVector)
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

The result is a ``z``x``k`` matrix of squared distances.

"""
getDistances(metric :: Metric,
             means  :: ℍVector,
             𝐏      :: ℍVector) =
  [PosDefManifold.distance²(metric, 𝐏[j], means[i]) for i=1:length(means), j=1:length(𝐏)]
# optimize in PosDefManifold, don't need to compute all distances for some metrics



"""
```
function CV_mdm(metric :: Metric,
                𝐏Tr    :: ℍVector,
                yTr    :: IntVector,
                nCV    :: Int;
            scoring   :: Symbol = :b,
            confusion :: Bool   = false,
            shuffle   :: Bool   = false)
```

Typically, you will not need this function as it is called by the
[`CVscore`](@ref) function.

This function return the same thing and has the same arguments
as the [`CVscore`](@ref) function, with the exception of the first argument,
that here is a `metric` of type
[Metric](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#Metric::Enumerated-type-1).

"""
function CV_mdm(metric :: Metric,
                𝐏Tr    :: ℍVector,
                yTr    :: IntVector,
                nCV    :: Int;
            scoring   :: Symbol = :b,
            confusion :: Bool   = false,
            shuffle   :: Bool   = false)

    nc=length(unique!(copy(yTr)))
    𝐏 = [ℍ[] for i = 1: nc] # All data by classes
    𝐏Tr = [ℍ[] for i = 1: nc] # training data by classes
    𝐏Te = [ℍ[] for i = 1: nc] # test data by classes
    score = Array{Float64, 1}(undef, nCV)
    cnf_mat = [zeros(Float64 , (nc,  nc)) for i = 1:nCV] # confusion matrix final
    for j = 1:dim(𝐏Tr, 1)  push!(𝐏[yTr[j]], 𝐏Tr[j]) end
    println(titleFont, "\nPerforming random cross-validations...", defaultFont)
    for k = 1:nCV
        print(rand(dice), " ")
        model=MDM(metric)
        for i = 1:nc
            nTrain, nTest, indTrain, indTest = CVsetup(length(𝐏[i]), nCV; shuffle=shuffle)
            𝐏Tr[i]  = [𝐏[i][j] for j in(indTrain[k])]
            𝐏Te[i] =  [𝐏[i][j] for j in(indTest[k])]
        end
        if nc<=2 # use nthreads here
            model.means = ℍVector([getMeans(metric, 𝐏Tr[Int(l)]) for l= 1:nc])
        else
            model.means = ℍVector(undef, nc)
            @threads for l in classes model.means[l]=getMeans(metric, 𝐏Tr[Int(l)], ⏩=false) end
        end
        result = [Int[] for i = 1:nc]
        for i = 1: nc
            result[i] = predict(model, 𝐏Te[i], :l, verbose=false)
            for s = 1: length(result[i])
                cnf_mat[k][i, result[i][s]] = cnf_mat[k][i, result[i][s]] + 1.
            end
        end
        scoring == :b ? score[k] = (𝚺( [ cnf_mat[k][i,i] / 𝚺( cnf_mat[k][i,:] ) for i = 1:nc ])) / nc :
                        score[k] = 𝚺( [ cnf_mat[k][i,i] for i = 1:nc ] )/ 𝚺( cnf_mat[k] )

    end
    println(" Done!\n", defaultFont)
    avg=mean(score);    avgs=round(avg; digits=3)
    sd=stdm(score, avg); sds=round(sd; digits=3)
    scoringString = scoring == :b ? "balanced accuracy" : "accuracy"
    println("mean(sd) ", titleFont, scoringString,": ", defaultFont, avgs,"(", sds,")", defaultFont, "\n")
    return confusion ? (score, cnf_mat) : score
end


# ++++++++++++++++++++  Show override  +++++++++++++++++++ # (REPL output)
function Base.show(io::IO, ::MIME{Symbol("text/plain")}, M::MDM)
    if M.means==nothing
        println(io, greyFont, "\n↯ MDM machine learning model")
        println(io, "⭒  ⭒    ⭒       ⭒          ⭒", defaultFont)
        println(io, "The model has been created. \nNext, fit it with data.")
    else
        println(io, titleFont, "\n↯ MDM machine learning model")
        println(io, separatorFont, "⭒  ⭒    ⭒       ⭒          ⭒", defaultFont)
        nc=length(M.means)
        n=size(M.means[1], 1)
        println(io, "features: PD matrices of size $(n)x$(n)")
        println(io, "classes : $(nc)")
        println(io, "fields  : (accessed by . notation)")
        println(io, "  .metric, .means.")
    end
end
