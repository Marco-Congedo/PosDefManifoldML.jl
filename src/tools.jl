#   Unit "tools.jl" of the PosDefManifoldML Package for Julia language

#   MIT License
#   Copyright (c) 2019-2025
#   Marco Congedo, CNRS, Grenoble, France:
#   https://sites.google.com/site/marcocongedo/home

# ? CONTENTS :
#   This unit implements tools that are useful for building
#   Riemannian and Euclidean machine learning classifiers.

"""
```julia
function tsMap(	metric :: Metric, 𝐏 :: ℍVector;
    w           :: Vector = Float64[],
    ✓w         :: Bool = true,
    ⏩          :: Bool = true,
    meanISR     :: Union{ℍ, Nothing, UniformScaling}  = nothing,
    meanInit    :: Union{ℍ, Nothing}  = nothing,
    tol         :: Real = 0.,
    transpose   :: Bool = true,
    vecRange    :: UnitRange = 1:size(𝐏[1], 1))
```

The [tangent space mapping](https://marco-congedo.github.io/PosDefManifold.jl/dev/riemannianGeometry/#PosDefManifold.logMap)
of positive definite matrices ``P_i``, *i=1...k* with mean *G*, once
those points have been parallel transported to the identity matrix,
is given by:

``S_i=\\textrm{log}(G^{-1/2} P_i G^{-1/2})``.

Given a vector of *k* matrices `𝐏` flagged by julia as `Hermitian`,
return a matrix *X* with such tangent vectors of the matrices in `𝐏`
vectorized as per the [vecP](https://marco-congedo.github.io/PosDefManifold.jl/dev/riemannianGeometry/#PosDefManifold.vecP)
operation.

The mean *G* of the matrices in `𝐏` is found according to the
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

A set of *k* optional non-negative weights `w` can be provided
for computing a weighted mean *G*, for any metrics.
If `w` is non-empty and optional keyword argument `✓w` is true (default),
the weights are normalized so as to sum up to 1,
otherwise they are used as they are passed and should be already normalized.
This option is provided to allow calling this function
repeatedly without normalizing the same weights vector each time.

If an Hermitian matrix is provided as optional keyword argument `meanISR`,
then the mean *G* is not computed, intead this matrix is used
directly in the formula as the inverse square root (ISR) ``G^{-1/2}``.
If `meanISR` is provided, arguments `tol` and `meanInit` have no effect
whatsoever.

If `meanISR=I` is used, then the tangent space mapping is obtained 
at the identity matrix as

``S_i=\\textrm{log}(P_i)``.

This corresponds to the tangent space mapping adopting the
log-Euclidean metric. It is also useful when the data has been
already recentered, for example by means of a [`Recenter`](@ref)
pre-conditioner. If `meanISR=I` is used, arguments 
`w`, `✓w`, `meanInit`, and `tol` are ignored.

If `meanISR` is not provided, return the 2-tuple ``(X, G^{-1/2})``,
otherwise return only matrix *X*.

If an `UnitRange` is provided with the optional keyword argument `vecRange`,
the vectorization concerns only the columns (or rows) of the matrices `𝐏`
specified by the range.

If optional keyword argument `transpose` is true (default),
*X* holds the *k* vectorized tangent vectors in its rows,
otherwise they are arranged in its columns.
The dimension of the rows in the former case and of the columns is the latter
case is *n(n+1)÷2* (integer division), where *n* is the size of the
matrices in `𝐏`, unless a `vecRange` spanning a subset of the columns or rows
of the matrices in `𝐏` has been provided, in which case the dimension will
be smaller. (see [vecP](https://marco-congedo.github.io/PosDefManifold.jl/dev/riemannianGeometry/#PosDefManifold.vecP)
).

if optional keyword argument `⏩` if true (default),
the computation of the mean and the projection on the tangent space
are multi-threaded. Multi-threading is automatically disabled if the
number of threads Julia is instructed to use is *<2* or *<2k*.

**Examples**:
```julia
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
function tsMap(	metric :: Metric, 𝐏 :: ℍVector;
				w			:: Vector = Float64[],
				✓w   	  	:: Bool = true,
				⏩			:: Bool = true,
				meanISR   	:: Union{ℍ, Nothing, UniformScaling}  = nothing,
				meanInit  	:: Union{ℍ, Nothing}  = nothing,
				tol       	:: Real = 0.,
				transpose 	:: Bool = true,
				vecRange  	:: UnitRange = 1:size(𝐏[1], 1))

	k, n, getMeanISR = dim(𝐏, 1), dim(𝐏, 2), meanISR===nothing
    G⁻½ = getMeanISR ? PosDefManifold.pow(PosDefManifold.mean(metric, 𝐏; 
		w, ✓w, init=meanInit, tol, ⏩), -0.5) : meanISR

	# length of the tangent vectors for the given vecRange
	m = _manifoldDim(𝐏[1], vecRange)

	##################################################
	tangentvector(P, vecRange, G⁻½) = 
		if G⁻½ isa UniformScaling # identity matrix
			vecP(ℍ(log(P)); range=vecRange)
		else
			vecP(log(cong(G⁻½, P, ℍ)); range=vecRange)
		end
	##################################################
	
	V = Array{eltype(𝐏[1]), 2}(undef, m, k)
	if ⏩
		@threads for i = 1:k 
			V[:, i] = tangentvector(𝐏[i], vecRange, G⁻½)
		end
	else
		@inbounds for i = 1:k 
			V[:, i] = tangentvector(𝐏[i], vecRange, G⁻½)
		end
	end

	# Apply here normalization and rescaling !

	if transpose
		return getMeanISR ? (Matrix(V'), G⁻½) : Matrix(V')
	else
		return getMeanISR ? (V, G⁻½) : V
	end
end



"""
```julia
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

Class labels for *n* classes must be the first *n* natural numbers,
that is, `1` for class 1, `2` for class 2, etc.
The labels in `y` can be provided in any order.

if a vector of *n* weights is specified as optional
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
```julia
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

```

This is how it works:

```julia
using PosDefManifoldML

# Suppose these are the labels:

y=[1, 1, 1, 1, 2, 2]

# We want the four observations of class 1 to count as much
# as the two observations of class 2.

tsWeights(y)

# 6-element Array{Float64,1}:
# 0.125
# 0.125
# 0.125
# 0.125
# 0.25
# 0.25

# i.e., 0.125*4 = 1.25*2
# and all weights sum up to 1

# Now, suppose we want to give to class 2 a weight
# four times bigger as compared to class 1:

tsWeights(y, classWeights=[1, 4])

# 6-element Array{Float64,1}:
# 0.05
# 0.05
# 0.05
# 0.05
# 0.4
# 0.4

# and, again, all weights sum up to 1.
```
"""
function tsWeights(y::Vector{Int}; classWeights=[])

    Nobs = length(y)
    Nclass=length(unique(y))
    Nobsxclass=[count(i->(i==j), y) for j=1:Nclass] #zzz
    NobsxclassProp=[n/Nobs for n in Nobsxclass]
    minProportion=minimum(Nobsxclass)/Nobs
    minProportion<(1/(Nclass*10)) && @warn 📌*" package, tsWights function: the smallest class contains les then 10% of the observation as compared to a balanced design" minProportion

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
```julia
function gen2ClassData(n        ::  Int,
                       k1train  ::  Int,
                       k2train  ::  Int,
                       k1test   ::  Int,
                       k2test   ::  Int,
                       separation :: Real = 0.1)
```

Generate for two classes (1 and 2) a random *training set*
holding `k1train`+`k2train` and a random *test set* holding `k1test`+`k2test`
symmetric positive definite matrices.
All matrices have size *n*x*n*.

The training and test sets can be used to train and test any [MLmodel](@ref).

`separation` is a coefficient determining how well the two classs are
separable; the higher it is, the more separable the two classes are.
It must be in [0, 1] and typically a value of 0.5 already
determines complete separation.

Return a 4-tuple with

- an [ℍVector](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#%E2%84%8DVector-type-1) holding the `k1train`+`k2train` matrices in the training set,
- an ℍVector holding the `k1test`+`k2test` matrices in the test set,
- a vector holding the `k1train`+`k2train` labels (integers) corresponding to the matrices of the training set,
- a vector holding the `k1test`+`k2test` labels corresponding to the matrices of the test set (*1* for class 1 and *2* for class 2).

**Examples**

```julia
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

    # Create a set of k1+k2 random matrices and move them along
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
```julia
function rescale!(X::Matrix{T},	bounds::Tuple=(-1, 1);
		dims::Int=1) where T<:Real
```
Rescale the columns or the rows of real matrix `X` to be in range [a, b],
where a and b are the first and seconf elements of tuple `bounds`.

By default it applies to the columns. Use `dims=2` for rescaling
the rows.

This function is used for normalizing tangent space (feature) vectors.
Typically, you won't need it. When fitting a model with [`fit`](@ref) 
or performing a cross-validation
with [`crval`](@ref), you can simply pass `bounds` to the argument `rescale`
of these functions.

"""
function rescale!(X::Matrix{T},
	              bounds::Tuple=(-1, 1);
				  dims::Int=1) where T<:Real
	dims∉(1, 2) && throw(ArgumentError(📌*" package, rescale! function: the `dims` keyword argument must be either 1 or 2; dims=$dims"))
	length(bounds) ≠ 2 && throw(ArgumentError(📌*" package, rescale! function: tuple `bounds` must contain two elements; bounds=$bounds"))
	a=first(bounds)
	b=last(bounds)
	a isa Number && b isa Number || throw(ArgumentError( 📌*" package, rescale! function: the two elements of tuple `bounds` must be numbers; bounds=$bounds"))
	c=b-a
	c<=0 && throw(ArgumentError(📌*" package, rescale! function: the two elements (a, b) of tuple `bounds` must verify b-a>0; bounds=$bounds"))
	
	@inbounds dims==1 ? (for j=1:size(X, 2)
							minX, maxX=extrema(X[:, j])
							range=maxX-minX
							for i=1:size(X, 1) 
								X[i, j]=a+(c*(X[i, j]-minX)/range) 
							end
						end) :
						(for i=1:size(X, 1)
							minX, maxX=extrema(X[i, :])
							range=maxX-minX
							for j=1:size(X, 2) 
								X[i, j]=a+(c*(X[i, j]-minX)/range) 
							end
						end)
end	


"""
```julia
function demean!(X::Matrix{T}; dims::Int=1) where T<:Real 
```
Remove the mean from the columns or from the rows of real matrix `X`.

By default it applies to the columns. Use `dims=2` for demeaning
the rows.

This function is used for normalizing tangent space (feature) vectors.
Typically, you won't need it. When fitting a model with [`fit`](@ref) 
or performing a cross-validation
with [`crval`](@ref), you can simply pass this function as the `normalize`
argument.

"""
function demean!(X::Matrix{T}; dims::Int=1) where T<:Real 
	dims∉(1, 2) && throw(ArgumentError(📌*" package, demean! function: the `dims` keyword argument must be either 1 or 2; dims=$dims"))
    m = mean(X; dims)
    X[:] = X.-m
    return X
end


"""
```julia
function normalize!(X::Matrix{T}; dims::Int=1) where T<:Real 
```
Normalize the columns or the rows of real matrix `X` so that
their 2-norm divided by their number of elements is 1.0.
This way the value of each element of matrix `X` gravitates 
around 1.0, regardless its size.

By default it applies to the columns. Use `dims=2` for normlizing
the rows.

This function is used for normalizing tangent space (feature) vectors.
Typically, you won't need it. When fitting a model with [`fit`](@ref) 
or performing a cross-validation
with [`crval`](@ref), you can simply pass this function as the `normalize`
argument.
"""
function normalize!(X::Matrix{T}; dims::Int=1) where T<:Real 
	dims∉(1, 2) && throw(ArgumentError(📌*" package, normalize! function: the `dims` keyword argument must be either 1 or 2; dims=$dims"))
    verso = dims==1 ? eachcol : eachrow
    𝐧 = collect(norm(x) for x ∈ verso(X))./size(X, dims)
    X[:] = X./(dims==1 ? 𝐧' : 𝐧)
    return X
end


"""
```julia
function standardize!(X::Matrix{T}; dims::Int=1) where T<:Real 
```
Standardize the columns or the rows of real matrix `X`
so that they have zero mean and unit (uncorrected) standard deviation.

By default it applies to the columns. Use `dims=2` for standardizing
the rows.

This function is used for normalizing tangent space (feature) vectors.
Typically, you won't need it. When fitting a model with [`fit`](@ref) 
or performing a cross-validation
with [`crval`](@ref), you can simply pass this function as the `normalize`
argument.

"""
function standardize!(X::Matrix{T}; dims::Int=1) where T<:Real     
    verso = dims==1 ? eachcol : eachrow
    demean!(X; dims) # check for argument `dims` is done in function `demean!`
    𝐧 = sqrt.(collect(sum(i->i^2, x) for x ∈ verso(X))./size(X, dims))
    X[:] = X./(dims==1 ? 𝐧' : 𝐧)
    return X
end


"""
```julia
    function saveas(object, filename::String)
```
Save the `object` to a file, which full path is given as `filemane`.
It can be used, for instance, to save [`CVres`](@ref) structures and
[`Pipeline`](@ref) tuples.

**See**: [`load`](@ref)

**Examples**

```julia
using PosDefManifoldML, PosDefManifold, Serialization

## Save and then load a cross-validation structure

P, _dummyP, y, _dummyy = gen2ClassData(10, 40, 40, 30, 40, 0.15)

cv = crval(MDM(logEuclidean), P, y; scoring=:a)

filename=joinpath(@__DIR__, "mycv.jls")
saveas(cv, filename)

mycv = load(filename)

# retrive the p-value of the cross-validation
mycv.p

## Save and then load a pipeline

# Generate some 'training' and `test` data
PTr=randP(3, 20) # 20 random 3x3 Hermitian matrices
PTe=randP(3, 5) # 5 random 3x3 Hermitian matrices

# fit a pipeline and transform the training data
p = fit!(PTr, @pipeline Recenter(; eVar=0.99) Compress)

# save the fitted pipeline
filename=joinpath(@__DIR__, "pipeline.jls")
saveas(p, filename) 

mypipeline = load(filename)

# transform the testing data using the loaded pipeline
transform!(PTe, mypipeline)
```
"""
function saveas(object, filename::String)
    open(filename, "w") do io
        serialize(io, object)
    end
end

"""
```julia
    function load(filename::String)
```
Load and retur an `object` stored to a file, 
which full path is given as `filemane`.

It can be used, for instance, to load [`CVres`](@ref) structures and
[`Pipeline`](@ref) tuples.

For pipelines, there is no check that it matches the dimension of the data 
to which it will be applied. This is the user's responsibility.

**See** [`saveas`](@ref)
"""
function load(filename::String)
    return open(filename, "r") do io
        deserialize(io)
    end
end
