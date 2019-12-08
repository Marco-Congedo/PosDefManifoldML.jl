#   Unit "tools.jl" of the PosDefManifoldML Package for Julia language
#   v 0.3.0 - last update 8th of December 2019
#
#   MIT License
#   Copyright (c) 2019,
#   Marco Congedo, CNRS, Grenoble, France:
#   https://sites.google.com/site/marcocongedo/home

# ? CONTENTS :
#   This unit implements tools that are useful for building
#   Riemannian and Euclidean machine learning classifiers.

"""
```
function tsMap(	metric :: Metric,
				𝐏      :: ℍVector;
		w    	  :: Vector	= [],
		✓w   	  :: Bool = true,
		⏩   	 :: Bool = true,
		meanISR   :: Union{ℍ, Nothing}  = nothing,
		meanInit  :: Union{ℍ, Nothing}  = nothing,
		tol       :: Real               = 0.,
		transpose :: Bool   			 = true,
		vecRange  :: UnitRange          = 1:size(𝐏[1], 1))
```

The [tangent space mapping](https://marco-congedo.github.io/PosDefManifold.jl/dev/riemannianGeometry/#PosDefManifold.logMap)
of positive definite matrices ``P_i``, ``i=1...k`` with mean ``G``, once
those points have been parallel transported to the identity matrix,
is given by:

``S_i=\\textrm{log}(G^{-1/2} P_i G^{-1/2})``.

Given a vector of ``k`` matrices `𝐏` flagged by julia as `Hermitian`,
return a matrix ``X`` with such tangent vectors of the matrices in `𝐏`
vectorized as per the [vecP](https://marco-congedo.github.io/PosDefManifold.jl/dev/riemannianGeometry/#PosDefManifold.vecP)
operation.

The mean ``G`` of the matrices in `𝐏` is found according to the
specified `metric`, of type
[Metric](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#Metric::Enumerated-type-1).
A natural choice is the
[Fisher metric](https://marco-congedo.github.io/PosDefManifold.jl/dev/introToRiemannianGeometry/#Fisher-1).
If the metric is Fisher, logdet0 or Wasserstein the mean is found with an iterative
algorithm with tolerance given by optional keyword argument `tol`.
By default `tol` is set by the function
[mean](https://marco-congedo.github.io/PosDefManifold.jl/dev/riemannianGeometry/#Statistics.mean).
For those iterative algorithms a particular initialization can be provided
as an Hermitian matrix by optional keyword argument `meanInit`.

A set of ``k`` optional non-negative weights `w` can be provided
for computing a weighted mean ``G``, for any metrics.
If `w` is non-empty and optional keyword argument `✓w` is true (default),
the weights are normalized so as to sum up to 1,
otherwise they are used as they are passed and should be already normalized.
This option is provided to allow calling this function
repeatedly without normalizing the same weights vector each time.

If an Hermitian matrix is provided as optional keyword argument `meanISR`,
then the mean ``G`` is not computed, intead this matrix is used
directly in the formula as the inverse square root (ISR) ``G^{-1/2}``.
If `meanISR` is provided, arguments `tol` and `meanInit` have no effect
whatsoever.

If `meanISR` is not provided, return the 2-tuple ``(X, G^{-1/2})``,
otherwise return only matrix ``X``.

If an `UnitRange` is provided with the optional keyword argument `vecRange`,
the vectorization concerns only the columns (or rows) of the matrices `𝐏`
specified by the range.

If optional keyword argument `transpose` is true (default),
``X`` holds the ``k`` vectorized tangent vectors in its rows,
otherwise they are arranged in its columns.
The dimension of the rows in the former case and of the columns is the latter
case is ``n(n+1)÷2`` (integer division), where ``n`` is the size of the
matrices in `𝐏`, unless a `vecRange` spanning a subset of the columns or rows
of the matrices in `𝐏` has been provided, in which case the dimension will
be smaller. (see [vecP](https://marco-congedo.github.io/PosDefManifold.jl/dev/riemannianGeometry/#PosDefManifold.vecP)
).

if optional keyword argument `⏩` if true (default),
the computation of the mean and the projection on the tangent space
are multi-threaded. Multi-threading is automatically disabled if the
number of threads Julia is instructed to use is ``<2`` or ``<2k``.

**Examples**:
```
using PosDefManifoldML

# generate four random symmetric positive definite 3x3 matrices
Pset = randP(3, 4)

# project and vectorize in the tangent space
X, G⁻½ = tsMap(Fisher, Pset)

# X is a 4x6 matrix, where 6 is the size of the
# vectorized tangent vectors (n=3, n*(n+1)/2=6)

# If repeated calls have to be done, faster computations are obtained
# providing the inverse square root of the matrices in Pset, e.g.,
X1 = tsMap(Fisher, ℍVector(Pset[1:2]); meanISR = G⁻½)
X2 = tsMap(Fisher, ℍVector(Pset[3:4]); meanISR = G⁻½)
```

**See**: [the ℍVector type](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#%E2%84%8DVector-type-1).

"""
function tsMap(metric :: Metric,
               𝐏      :: ℍVector;
         w    	   :: Vector 			 = [],
         ✓w   	   :: Bool   			 = true,
         ⏩   	  :: Bool   		    = true,
		 meanISR   :: Union{ℍ, Nothing}  = nothing,
		 meanInit  :: Union{ℍ, Nothing}  = nothing,
	  	 tol       :: Real               = 0.,
		 transpose :: Bool   			 = true,
		 vecRange  :: UnitRange          = 1:size(𝐏[1], 1))

	k, n, getMeanISR = dim(𝐏, 1), dim(𝐏, 2), meanISR==nothing
    getMeanISR ? G⁻½ = pow(mean(metric, 𝐏;
	                            w=w,
								✓w=✓w,
								init=meanInit,
								tol=tol,
								⏩=⏩), -0.5) : G⁻½ = meanISR

	# length of the tangent vectors for the given vecRange
	m=_triNum(𝐏[1], vecRange)

	if transpose
		V = Array{eltype(𝐏[1]), 2}(undef, k, m)
	    ⏩==true ? (@threads  for i = 1:k V[i, :] = vecP(log(cong(G⁻½, 𝐏[i], ℍ)); range=vecRange) end) :
	                (@inbounds for i = 1:k V[i, :] = vecP(log(cong(G⁻½, 𝐏[i], ℍ)); range=vecRange) end)
	else
		V = Array{eltype(𝐏[1]), 2}(undef, m, k)
		⏩==true ? (@threads  for i = 1:k V[:, i] = vecP(log(cong(G⁻½, 𝐏[i], ℍ)); range=vecRange) end) :
	                (@inbounds for i = 1:k V[:, i] = vecP(log(cong(G⁻½, 𝐏[i], ℍ)); range=vecRange) end)
	end
    return getMeanISR ? (V, G⁻½) : V
end



"""
```
function tsWeights(y::Vector{Int}; classWeights=[])
```

Given an [IntVector](@ref) of labels `y`, return a vector
of weights summing up to 1 such that the overall
weight is the same for all classes (balancing).
This is useful for machine learning models in the tangent
space with unbalanced classes for computing the mean,
that is, the base point to map PD matrices onto the tangent space.
For this mapping, giving equal weights to all observations
actually overweights the larger classes
and downweight the smaller classes.

Class labels for ``n`` classes must be the first ``n`` natural numbers,
that is, `1` for class 1, `2` for class 2, etc.
The labels in `y` can be provided in any order.

if a vector of ``n`` weights is specified as optional
keyword argument `classWeights`, the overall weights
for each class will be first balanced (see here above),
then weighted by the `classWeights`.
This allow user-defined control of weighting independently
from the number of observations in each class.
The weights in `classWeights` can be any integer or real
non-negative numbers. The returned weight vector will
nonetheless sum up to 1.

When you invoke the [`fit`](@ref) function for tangent space models
you don't actually need this function, as you can invoke it
implicitly passing symbol `:balanced` (or just `:b`) or a tuple
with the class weights as optional keyword argument `w`.

**Examples**
```
# generate some data; the classes are unbalanced
PTr, PTe, yTr, yTe=gen2ClassData(10, 30, 40, 60, 80, 0.1)

# Fit an ENLR lasso model and find the best model by cross-validation
# balancing the weights for tangent space mapping
m=fit(ENLR(), PTr, yTr; w=tsWeights(yTr))

# A simpler syntax is
m=fit(ENLR(), PTr, yTr; w=:balanced)

# to balance the weights and then give overall weight 0.5 to class 1
# and 1.5 to class 2:
m=fit(ENLR(), PTr, yTr; w=(0.5, 1.5))

# which is equivalent to
m=fit(ENLR(), PTr, yTr; w=tsWeights(yTr; classWeights=(0.5, 1.5)))


# This is how it works:

julia> y=[1, 1, 1, 1, 2, 2]
6-element Array{Int64,1}:
 1
 1
 1
 1
 2
 2

We want the four observations of class 1 to count as much
as the two observations of class 2.

julia> tsWeights(y)
6-element Array{Float64,1}:
 0.125
 0.125
 0.125
 0.125
 0.25
 0.25

i.e., 0.125*4 = 1.25*2
and all weights sum up to 1

Now, suppose we want to give to class 2 a weight
four times bigger as compared to class 1:

julia> tsWeights(y, classWeights=[1, 4])
6-element Array{Float64,1}:
 0.05
 0.05
 0.05
 0.05
 0.4
 0.4

and, again, all weights sum up to 1

```
"""
function tsWeights(y::Vector{Int}; classWeights=[])

    Nobs = length(y)
    Nclass=length(unique(y))
    Nobsxclass=[count(i->(i==j), y) for j=1:Nclass]
    NobsxclassProp=[n/Nobs for n in Nobsxclass]
    minProportion=minimum(Nobsxclass)/Nobs
    minProportion<(1/(Nclass*10)) && @warn "the smallest class contains les then 10% of the observation as compared to a balances design" minProportion

    w=[1/(Nclass*Nobsxclass[l]) for l ∈ y]

    if !isempty(classWeights)
       if length(classWeights)≠Nclass
          @warn "the number of elements in argument ClassWeights is different from the number of unique classes in label vector y. Class weights have not been applied" length(classWeights)
       else
         for i=1:length(w) w[i]*= classWeights[y[i]] end
         w./=sum(w)
       end
    end

    return w
end


"""
```
function gen2ClassData(n        ::  Int,
                       k1train  ::  Int,
                       k2train  ::  Int,
                       k1test   ::  Int,
                       k2test   ::  Int,
                       separation :: Real = 0.1)
```

Generate a *training set* of `k1train`+`k2train`
and a *test set* of `k1test`+`k2test`
symmetric positive definite matrices.
All matrices have size ``n``x``n``.

The training and test sets can be used to train and test any [MLmodel](@ref).

`separation` is a coefficient determining how well the two classs are
separable; the higher it is, the more separable the two classes are.
It must be in [0, 1] and typically a value of 0.5 already
determines complete separation.

Return a 4-tuple with

- an [ℍVector](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#%E2%84%8DVector-type-1) holding the `k1train`+`k2train` matrices in the training set,
- an ℍVector holding the `k1test`+`k2test` matrices in the test set,
- a vector holding the `k1train`+`k2train` labels (integers) corresponding to the matrices of the training set,
- a vector holding the `k1test`+`k2test` labels corresponding to the matrices of the test set (``1`` for class 1 and ``2`` for class 2).

**Examples**

```
using PosDefManifoldML

PTr, PTe, yTr, yTe=gen2ClassData(10, 30, 40, 60, 80, 0.25)

# PTr=training set: 30 matrices for class 1 and 40 matrices for class 2
# PTe=testing set: 60 matrices for class 1 and 80 matrices for class 2
# all matrices are 10x10
# yTr=a vector of 70 labels for the training set
# yTe=a vector of 140 labels for the testing set

```
"""
function gen2ClassData(n        ::  Int,
                       k1train  ::  Int,
                       k2train  ::  Int,
                       k1test   ::  Int,
                       k2test   ::  Int,
                       separation :: Real = 0.1)
	if separation<0 || separation >1
		@error 📌*", function "*gen2ClassData*": argument `separation` must be in range [0, 1]."
		return
	end

	G1=randP(n)
    G2=randP(n)

    # Create a set of k1+k2 random matrices and move the along
    # the Fisher Geodesic with arclength (1-a) the first k1 toward G1
    # and the last k2 toward G2. Geodesics are computed with the Schur method
    function getMatrices(k1::Int, k2::Int, n::Int, a::Real, G1::ℍ, G2::ℍ)

        function getChol(G::ℍ)
            L = cholesky(G, check=false)
            U⁻¹ = inv(L.U)
            return L, U⁻¹
        end

        k=k1+k2
        𝐗=ℍVector(undef, k)
            L, U⁻¹=getChol(G1)
            for i=1:k1
                F = schur(U⁻¹' * randP(n) * U⁻¹)
                𝐗[i]=ℍ(L.U' * (F.Z * F.T^a* F.Z') * L.U)
            end
            L, U⁻¹=getChol(G2)
            for i=k1+1:k
                F = schur(U⁻¹' * randP(n) * U⁻¹)
                𝐗[i]=ℍ(L.U' * (F.Z * F.T^a* F.Z') * L.U)
            end
        return 𝐗
    end

    𝐗train=getMatrices(k1train, k2train, n, 1-separation, G1, G2)
    𝐗test=getMatrices(k1test, k2test, n, 1-separation, G1, G2)
    y𝐗train=IntVector([repeat([1], k1train); repeat([2], k2train)])
    y𝐗test=IntVector([repeat([1], k1test); repeat([2], k2test)])

    return 𝐗train, 𝐗test, y𝐗train, y𝐗test
end


"""
```
function predictErr(yTrue::IntVector, yPred::IntVector;
	          		digits::Int=3))
```

Return the percent prediction error given a vector of true labels and a vector
of predicted labels.

The order of arguments does not matter.

The error is rounded to the number of optional keyword argument
`digits`, 3 by default.

**See** [`predict`](@ref)

**Examples**

```
using PosDefManifoldML
predictErr([1, 1, 2, 2], [1, 1, 1, 2])
# return: 25.0
```
"""
function predictErr(yTrue::IntVector, yPred::IntVector;
	          digits::Int=3)
	n1=length(yTrue)
	n2=length(yPred)
	if n1≠n2
		@error 📌*", function predictErr: the length of the two argument vectors must be equal." n1 n2
		return
	else
		round(sum(y1≠y2 for (y1, y2) ∈ zip(yTrue, yPred))/n1*100; digits=digits)
	end
end



"""
```
function rescale!(	X::Matrix{T},
					bounds::Tuple=(-1, 1);
					dims::Int=1) where T<:Real
```
Rescale the columns or the rows of real matrix `X` to be in range [a, b],
where a and b are the first and seconf elements of tuple `bounds`.

By default rescaling apply to the columns. Use `dims=2` for rescaling
the rows.

This function is used, for instance, by the SVM fit and predict functions.
"""
function rescale!(X::Matrix{T},
	              bounds::Tuple=(-1, 1);
				  dims::Int=1) where T<:Real
 dims∉(1, 2) && throw(ArgumentError, "rescale! function: the `dims` keyword argument must be either 1 or 2; dims=$dims")
 length(bounds) ≠ 2 && throw(ArgumentError, "rescale! function: tuple `bounds` must contain two elements; bounds=$bounds")
 a=first(bounds)
 b=last(bounds)
 a isa Number && b isa Number || throw(ArgumentError, "rescale! function: the two elements of tuple `bounds` must be numbers; bounds=$bounds")
 c=b-a
 c<=0 && throw(ArgumentError, "rescale! function: the two elements (a, b) of tuple `bounds` must verify b-a>0; bounds=$bounds")
 @inbounds dims==1 ? (for j=1:size(X, 2)
			 	        minX, maxX=extrema(X[:, j])
				        range=maxX-minX
			 	        for i=1:size(X, 1) X[i, j]=a+(c*(X[i, j]-minX)/range) end
					end) :
			        (for i=1:size(X, 1)
			 			minX, maxX=extrema(X[i, :])
						range=maxX-minX
			 			for j=1:size(X, 2) X[i, j]=a+(c*(X[i, j]-minX)/range) end
					end)
end

# -------------------------------------------------------- #
# INTERNAL FUNCTIONS #

# return a vector of ranges partitioning lineraly and
# as much as possible evenly `n` elements in `threads` ranges.
# `threads` is the number of threads to which the ranges are to be
# dispatched. If `threads` is not provided, it is set to the number
# of threads Julia is currently instructed to use.
# For example, for `k`=99
# and `threads`=4, return Array{UnitRange{Int64},1}:[1:25, 26:50, 51:75, 76:99].
function _partitionLinRange4threads(n::Int, threads::Int=0)
    threads==0 ? thr=nthreads() : thr=threads
    n<thr ? thr = n : nothing
    d = max(round(Int64, n / thr), 1)
    return [(r<thr ? (d*r-d+1:d*r) : (d*thr-d+1:n)) for r=1:thr]
end


function _GetThreads(n::Int, callingFunction::String)
	threads=Threads.nthreads()
	threads==1 && @warn 📌*", function "*callingFunction*": Julia is instructed to use only one thread."
	if n<threads && n<3
		@warn 📌*", function "*callingFunction*": the number of operations (n) is too low for taking advantage of multi-threading" threads n
		threads=1
	end
	return threads
end

function _GetThreadsAndLinRanges(n::Int, callingFunction::String)
	threads = _GetThreads(n, callingFunction)
	ranges=_partitionLinRange4threads(n, threads)
	return threads, ranges
end

# checks for `fit function`
function _check_fit(model       :: MLmodel,
              		 dim𝐏Tr     :: Int,
              		 dimyTr     :: Int,
           			 dimw  	    :: Int,
					 dimWeights :: Int,
					 modelName  :: String)
    errMsg1="the number of data do not match the number of labels."
	errMsg2="the number of data do not match the number of elements in `w`."
	errMsg3="the number of data do not match the number of elements in `weights`."
    if dim𝐏Tr ≠ dimyTr
		@error 📌*", fit function, model "*modelName*": "*errMsg1
		return false
	end
    if dimw ≠ 0 && dimw ≠ dimyTr
		@error 📌*", fit function, model "*modelName*": "*errMsg2
		return false
	end
	if dimWeights ≠ 0 && dimWeights ≠ dimyTr
		@error 📌*", fit function, model "*modelName*": "*errMsg3
		return false
	end

	return true
end

# check for argument `what` in `predict` function
_whatIsValid(what::Symbol, funcName::String) =
	if what ∉ (:l, :labels, :p, :probabilities, :f, :functions)
		@error 📌*", "*funcName*" function: the `what` symbol is not supported."
		return false
	else
		return true
	end

# translate the `what` argument to a string to give feedback
_what2Str(what::Symbol) =
	if      what ∈(:f, :fucntions)      return "functions"
	elseif  what ∈(:l, :labels)         return "labels"
	elseif  what ∈(:p, :probabilities)  return "prob. of belonging to each class"
	end

# return the dimension of the manifold of PD matrices: (n*n+1)/2
_triNum(P::ℍ, vecRange::UnitRange) =
	if length(vecRange)==size(P, 1)
		result=_triNum(P)
	else
		m=0; for j=vecRange, i=j:size(P, 1) m+=1 end
		result=m
	end

_triNum(P::ℍ) = ( size(P, 1) * (size(P, 1)+1) ) ÷ 2

# dimension of the manifold if 𝐏Tr is an ℍVector,
# dimension of the tagent(feature) vectors if 𝐏Tr is a Matrix
_getDim(𝐏Tr :: Union{ℍVector, Matrix{Float64}}, vecRange::UnitRange) =
	𝐏Tr isa ℍVector ? _triNum(𝐏Tr[1], vecRange) : length(vecRange)

_getDim(𝐏Tr :: Union{ℍVector, Matrix{Float64}}) =
	𝐏Tr isa ℍVector ? _triNum(𝐏Tr[1]) : size(𝐏Tr, 2)


# convert a ML model in a atring to release information
_modelStr(model::MLmodel) =
  if 		model isa MDMmodel
	  		return "MDM"

  elseif    model isa ENLRmodel
    		if     model.alpha≈1. return "Lasso logistic regression"
    		elseif model.alpha≈0. return "Ridge logistic regression"
    		else                  return "El. Net (α=$(round(model.alpha; digits=2))) log. reg."
			end

  elseif    model isa SVMmodel
			if 		model.svmType==SVC 			return "SVC"
			elseif  model.svmType==C-SVM 		return "C-SVM"
			elseif  model.svmType==EpsilonSVR 	return "EpsilonSVR"
			elseif  model.svmType==OneClassSVM 	return "OneClassSVM"
			elseif  model.svmType==NuSVR 		return "NuSVR"
			else    return  "Warning: the SVM type is unknown"
			end

  else      return "unknown"
  end

# check on `what` argument of `fit` function
_fitTypeIsValid(what::Symbol, funcName::String) =
	if what ∉ (:best, :path)
		@error 📌*", "*funcName*" function: the `fitType` symbol must be `:best` or `:path`."
		return false
	else
		return true
	end

# check on `onWhich` argument of `predict` function
function _ENLRonWhichIsValid(model::ENLRmodel, fitType::Symbol,
                    onWhich::Int, funcName::String)
    if fitType==:best
		return true
	else #fitType==:path
		i=length(model.path.lambda)
		if !(0<=onWhich<=i)
			@error 📌*", "*funcName*" function: the `onWhich` integer argument must be comprised between 0 (all models) and $i."
			return false
		else
			return true
		end
	end
end

# return a string to give information on what model is used to predict
# based on arguments `fitType` and `onWhich`
_ENLRonWhichStr(model::ENLRmodel, fitType::Symbol, onWhich::Int) =
	if 		fitType==:best
		return "from the best "*_modelStr(model)*" model (λ=$(round(model.best.lambda[1]; digits=5)))"
	else # :path
		if onWhich == 0
			return "from all "*_modelStr(model)*" models"
		else
			return "from "*_modelStr(model)*" model $(onWhich) (λ=$(round(model.path.lambda[onWhich]; digits=5)))"
		end
	end


# create a copy of optional keyword arguments `args`
# removing all arguments with names listed in tuple `remove`.
# Examples:
# fitArgs✔=_rmArgs((:meanISR, :fitType); fitArgs...)
# fitArgs✔=_rmArgs((:meanISR,); fitArgs...) # notice the comma after `meanISR`
# note: named tuples are immutable, that's why a copy must be created
function _rmArgs(remove::Tuple; args...)
	D = Dict(args)
	for key ∈ remove delete!(D, key) end
    return Tuple(D)
end


# given optional keyword arguments `args`,
# return the value of the argument with key `key`.
# If the argument does not exist, return `nothing`
function _getArgValue(key::Symbol; args...)
   D = Dict(args)
   return haskey(D, key) ? D[key] : nothing
end

# get a valid weigts `w` object and perform check given
# the user-defined `w` argument. Used in `fit` and `cvAcc` functions.
function _getWeights(w :: Union{Symbol, Tuple, Vector}, y::IntVector, funcName::String)
	if 		(w isa Vector && isempty(w)) || w==:uniform || w==:u return []
	elseif	w isa Vector && !isempty(w)
		    nObs, length_w = length(y), length(w)
		    if length_w==nObs return w
			else @error 📌*", "*funcName*"invalid vector `w`. `w` must contain as many elements as there are observations" length_w nObs
			end
	elseif  w==:balanced || w==:b return tsWeights(y)
	elseif  w isa Tuple
		    nClasses, length_w = length(unique(y)), length(w)
			if length_w==nClasses return tsWeights(y; classWeights=collect(w))
			else @error 📌*", "*funcName*"invalid tuple `w`. `w` must contain as many elements as there are classes" length_w nClasses
			end
	else
			@error 📌*", "*funcName*"invalid argument `w`. `w` must be a vector, an empty vector, a tuple of as many real numbers as classes, or symbol `:balanced`, or symbol `:uniform`"
			return nothing
	end
end


# Get the feature matrix for fit functions of ML model in the tangent space:
# if `𝐏Tr` is a matrix just return the columns in `vecRange` (by default all).
# if `𝐏Tr` is vector of Hermitian matrices, they are projected onto the
# tangent space. If the the inversesquare root of a base point `meanISR`
# is provided, the projection is obtained at this base point, otherwise the
# mean of all points is computed and used as base point.
# If the mean is to be computed by an iterative algorithm (e.g., if the metric
# of the model is the Fisher metric), an initialization `meanInit`, weights
# `w` and a tolerance `tol` are used.
# Once projected onto the tangent spave, the matrces in `𝐏Tr` are vectorized
# using only the rows (or columns) specified by `vecRange`.
# if `verbose` is true, print "Projecting data onto the tangent space..."
# if `transpose` the feature vectors are in the rows of `X`, otherwise in the
# columns of `X`.
# if ⏩ is true, the projection onto the tangent space
# and the algorithm to compute the mean are multi-threaded
function _getFeat_fit!(ℳ	    :: TSmodel,
					   𝐏Tr	  	  :: Union{ℍVector, Matrix{Float64}},
			 		   meanISR	 :: Union{ℍ, Nothing},
					   meanInit	 :: Union{ℍ, Nothing},
					   tol		 :: Real,
			 	 	   w		 :: Union{Symbol, Tuple, Vector},
			 	 	   vecRange	 :: UnitRange,
					   transpose :: Bool,
					   verbose	 :: Bool,
					   ⏩	   :: Bool)
	if 𝐏Tr isa ℍVector
		verbose && println(greyFont, "Projecting data onto the tangent space...")
		if meanISR==nothing
			(X, G⁻½)=tsMap(ℳ.metric, 𝐏Tr;
			               w=w,
						   ⏩=⏩,
						   vecRange=vecRange,
						   meanInit=meanInit,
						   tol=tol,
						   transpose=transpose)
			ℳ.meanISR = G⁻½
		else
			X=tsMap(ℳ.metric, 𝐏Tr;
			        w=w,
					⏩=⏩,
					vecRange=vecRange,
					meanISR=meanISR,
					transpose=transpose)
			ℳ.meanISR = meanISR
		end
	else X=𝐏Tr[:, vecRange]
	end
	return X
end

# Get the feature matrix for predict functions of ML model in the tangent space:
# if `𝐏Te` is a matrix just return the columns in `vecRange` (by default all).
# if `𝐏Te` is vector of Hermitian matrices, they are projected onto the
# tangent space. If an inverse square root 5ISR) `tranfer` is provided
# (typically the ISR of the mean of the matrices in `𝐏Te`), this is used
# as ISR of the base point, otherwise the ISR of the base point stored in the
# model ℳ is used. The latter is the classical approach, the former realizes the
# adaptation (transfer learning) explained in Barachant et al. (2013).
# Once projected onto the tangent spave, the matrces in `𝐏Te` are vectorized
# using only the rows (or columns) specified by `vecRange`.
# if `transpose` the feature vectors are in the rows of `X`, otherwise in the
# columns of `X`.
# if `verbose` is true, print "Projecting data onto the tangent space..."
# if ⏩ is true, the projection onto the tangent space is multi-threaded
_getFeat_Predict!(ℳ	   		:: TSmodel,
			      𝐏Te		  :: Union{ℍVector, Matrix{Float64}},
				  transfer   :: Union{ℍ, Nothing},
				  vecRange	 :: UnitRange,
				  transpose  :: Bool,
				  verbose	 :: Bool,
				  ⏩	       :: Bool) =
	if 𝐏Te isa ℍVector
		verbose && println(greyFont, "Projecting data onto the tangent space...")
		return tsMap(ℳ.metric, 𝐏Te;
				     meanISR = transfer==nothing ? ℳ.meanISR : transfer,
				     ⏩=⏩,
				     vecRange=vecRange,
					 transpose=transpose)
	else
		return 𝐏Te[:, vecRange]
	end
