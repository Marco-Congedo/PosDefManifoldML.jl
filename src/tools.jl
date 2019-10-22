#   Unit "tools.jl" of the PosDefManifoldML Package for Julia language
#   v 0.2.1 - last update 18th of October 2019
#
#   MIT License
#   Copyright (c) 2019,
#   Saloni Jain, Indian Institute of Technology, Kharagpur, India
#   Marco Congedo, CNRS, Grenoble, France:
#   https://sites.google.com/site/marcocongedo/home

# ? CONTENTS :
#   This unit implements tools that are useful for building
#   Riemannian and Euclidean machine learning classifiers.

"""
```
function tsMap(	metric :: Metric,
		𝐏 :: ℍVector;
		w :: Vector = [],
		✓w :: Bool = true,
		⏩ :: Bool = true,
		meanISR :: Union{ℍ, Nothing} = nothing,
		transpose :: Bool = true)
```

The [tangent space mapping](https://marco-congedo.github.io/PosDefManifold.jl/dev/riemannianGeometry/#PosDefManifold.logMap)
of matrices ``P_i``, ``i=1...k`` with geometric mean ``G``, once
those points have been parallel transported to the identity matrix,
is given by:

``S_i=\\textrm{log}(G^{-1/2} P_i G^{-1/2})``.

Given a vector of ``k`` Hermitian matrices `𝐏`,
return a matrix ``X`` with such tangent vectors of the matrices in `𝐏`
vectorized as per the [vecP](https://marco-congedo.github.io/PosDefManifold.jl/dev/riemannianGeometry/#PosDefManifold.vecP)
operation.

The mean ``G`` of the matrices in `𝐏` is found according to the
specified `metric`, of type
[Metric](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#Metric::Enumerated-type-1).
A natural choice is the
[Fisher metric](https://marco-congedo.github.io/PosDefManifold.jl/dev/introToRiemannianGeometry/#Fisher-1).

A set of ``k`` optional non-negative weights `w` can be provided
for computing instead the weighted mean ``G``.
If `w` is non-empty and optional keyword argument `✓w` is true (default),
the weights are normalized so as to sum up to 1,
otherwise they are used as they are passed and should be already normalized.
This option is provided to allow calling this function
repeatedly without normalizing the same weights vector each time.

If an Hermitian matrix is provided as optional keyword argument `meanISR`,
then the mean ``G`` is not computed, intead this matrix is used
directly in the formula as the inverse square root (ISR) ``G^{-1/2}``.

If `meanISR` is not provided, return the 2-tuple ``(X, G^{-1/2})``,
otherwise return only matrix ``X``.

If optional keyword argument `transpose` is true (default),
``X`` holds the ``k`` vectorized tangent vectors in its rows,
otherwise they are arranged in its columns.
The dimension of the rows in the former case and of the columns is the latter
case is ``n(n+1)/2``, where ``n`` is the size of the matrices in `𝐏`
(see [vecP](https://marco-congedo.github.io/PosDefManifold.jl/dev/riemannianGeometry/#PosDefManifold.vecP)
).

if optional keyword argument `⏩` if true (default),
the computation of the mean (if this is obtained
with an iterative algorithm, e.g., using the Fisher metric)
and the projection on the tangent space are multi-threaded.
Multi-threading is automatically disabled if the number of threads
Julia is instructed to use is ``<2`` or ``<3k``.

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
X1 = tsMap(Fisher, ℍVector(Pset[1:2]); meanISR =G⁻½)
X2 = tsMap(Fisher, ℍVector(Pset[3:4]); meanISR =G⁻½)

```

**See**: [the ℍVector type](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#%E2%84%8DVector-type-1).

"""
function tsMap(metric :: Metric,
               𝐏      :: ℍVector;
         w    	   :: Vector 			 = [],
         ✓w   	   :: Bool   			 = true,
         ⏩   	  :: Bool   		    = true,
		 meanISR    :: Union{ℍ, Nothing} = nothing,
		 transpose :: Bool   			 = true)

	k, n, getMeanISR = dim(𝐏, 1), dim(𝐏, 2), meanISR==nothing
    getMeanISR ? G⁻½ = pow(mean(metric, 𝐏; w=w, ✓w=✓w, ⏩=⏩), -0.5) : G⁻½ = meanISR
	if transpose
		V = Array{eltype(𝐏[1]), 2}(undef, k, Int(n*(n+1)/2))
	    ⏩==true ? (@threads for i = 1:k V[i, :] = vecP(ℍ(log(ℍ(G⁻½ * 𝐏[i] * G⁻½)))) end) :
	                         (for i = 1:k V[i, :] = vecP(ℍ(log(ℍ(G⁻½ * 𝐏[i] * G⁻½)))) end)
	else
		V = Array{eltype(𝐏[1]), 2}(undef, Int(n*(n+1)/2), k)
		⏩==true ? (@threads for i = 1:k V[:, i] = vecP(ℍ(log(ℍ(G⁻½ * 𝐏[i] * G⁻½)))) end) :
	                         (for i = 1:k V[:, i] = vecP(ℍ(log(ℍ(G⁻½ * 𝐏[i] * G⁻½)))) end)
	end
    return getMeanISR ? (V, G⁻½) : V
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
	if n<threads*3
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

function _check_fit(model     :: MLmodel,
              		 dim𝐏Tr    :: Int,
              		 dimyTr    :: Int,
           			 dimw  	   :: Int,
					 modelName :: String)
    errMsg1="the number of data do not match the number of labels."
	errMsg2="the numberof data do not match the number of weights."
    if dim𝐏Tr ≠ dimyTr
		@error 📌*", fit function, model "*modelName*": "*errMsg1
		return false
	end
    if dimw ≠ 0 && dimw ≠ k
		@error 📌*", fit function, model "*modelName*": "*errMsg2
		return false
	end
	return true
end


_whatIsValid(what::Symbol, funcName::String) =
	if what ∉ (:l, :labels, :p, :probabilities, :f, :functions)
		@error 📌*", "*funcName*" function: the `what` symbol is not supported."
		return false
	else
		return true
	end

_what2Str(what::Symbol) =
	if      what ∈(:f, :fucntions)      return "functions"
	elseif  what ∈(:l, :labels)         return "labels"
	elseif  what ∈(:p, :probabilities)  return "prob. of belonging to each class"
	end

_triNum(P::ℍ) = ( size(P, 1) * (size(P, 1)+1) ) ÷ 2

_modelStr(model::MLmodel) =
  if 		model isa MDMmodel
	  		return "MDM"
  elseif    model isa ENLRmodel
    		if     model.alpha≈1. return "Lasso logit regression"
    		elseif model.alpha≈0. return "Ridge logit regression"
    		else                  return "Elastic Net (α=$(round(model.alpha; digits=2))) logit regression"
			end
  else      return "unknown"
  end


_fitTypeIsValid(what::Symbol, funcName::String) =
	if what ∉ (:best, :path)
		@error 📌*", "*funcName*" function: the `fitType` symbol is not supported."
		return false
	else
		return true
	end

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
