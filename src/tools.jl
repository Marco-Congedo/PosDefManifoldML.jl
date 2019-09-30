#   Unit "tools.jl" of the PosDefManifoldML Package for Julia language
#   v 0.0.1 - last update 28th of September 2019
#
#   MIT License
#   Copyright (c) 2019,
#   Saloni Jain, Indian Institute of Technology, Kharagpur, India
#   Marco Congedo, CNRS, Grenobe, France:
#   https://sites.google.com/site/marcocongedo/home

# ? CONTENTS :
#   This unit implements tools that are useful for building
#   Riemannian and Euclidean machine learning classifiers.

"""
```
function projectOnTS(metric :: Metric,
                     𝐏      :: ℍVector;
                  w  :: Vector = [],
                  ✓w :: Bool   = true,
                  ⏩ :: Bool   = true)
```

Given a vector of ``k`` Hermitian matrices `𝐏`
and corresponding optional non-negative weights `w`,
return a matrix with the matrices `𝐏` mapped onto the tangent space
at base-point given by their mean and vectorized as per the
[vecP](https://marco-congedo.github.io/PosDefManifold.jl/dev/riemannianGeometry/#PosDefManifold.vecP)
operation.

[Tangent space mapping](https://marco-congedo.github.io/PosDefManifold.jl/dev/riemannianGeometry/#PosDefManifold.logMap)
of matrices ``P_i, i=1...k`` at base point ``G``
according to the Fisher metric is given by:

``S_i=G^{½} \\textrm{log}(G^{-½} P_i G^{-½}) G^{½}``.

!!! note "Nota Bene"
    the tangent space projection is currently supported only for the
    Fisher metric, therefore this metric is used for the projection.

The mean of the meatrices in `𝐏` is computed according to the
specified `metric`, of type
[Metric](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#Metric::Enumerated-type-1).
A natural choice is the
[Fisher metric](https://marco-congedo.github.io/PosDefManifold.jl/dev/introToRiemannianGeometry/#Fisher-1).
The weighted mean is computed if weights vector `w` is non-empty.
By default the unweighted mean is computed.

If `w` is non-empty and optional keyword argument `✓w` is true (default),
the weights are normalized so as to sum up to 1,
otherwise they are used as they are passed and should be already normalized.
This option is provided to allow calling this function
repeatedly without normalizing the same weights vector each time.

if optional keyword argument `⏩` if true (default),
the computation of the mean is multi-threaded if this is obtained
with an iterative algorithm (e.g., using the Fisher metric).
Multi-threading is automatically disabled if the number of threads
Julia is instructed to use is ``<2`` or ``<4k``.

Return a matrix holding the ``k`` mapped matrices in its columns.
The dimension of the columns is ``n(n+1)/2``, where ``n`` is the size
of the matrices in `𝐏`
(see [vecP](https://marco-congedo.github.io/PosDefManifold.jl/dev/riemannianGeometry/#PosDefManifold.vecP)
).
The arrangement of tangent vectors in the columns of a matrix is natural
in Julia, however if you export the tagent vectors to be
used as feature vectors keep in mind that several ML packages, for example
Python *scikitlearn*, expect them to be arranged in rows.

**Examples**:
```
using PosDefManifoldML

# generate four random symmetric positive definite 3x3 matrices
𝐏=randP(3, 4)

# project and vectorize in the tangent space
T=projectOnTS(Fisher, 𝐏)

# The result is a 6x4 matrix, where 6 is the size of the
# vectorized tangent vectors (n=3, n*(n+1)/2=6)
```

**See**: [the ℍVector type](@ref).

"""
function projectOnTS(metric :: Metric,
                     𝐏      :: ℍVector;
                  w  :: Vector = [],
                  ✓w :: Bool   = true,
                  ⏩ :: Bool   = true)

    G = mean(metric, 𝐏; w=w, ✓w=✓w, ⏩=⏩)
    k, n = dim(𝐏, 1), dim(𝐏, 2)
    G½, G⁻½=pow(G, 0.5, -0.5)
    Vec = Array{eltype(𝐏[1]), 2}(undef, Int(n*(n+1)/2), k)
    ⏩==true ? (@threads for i = 1:k Vec[:, i] = vecP(ℍ(G½ * log(ℍ(G⁻½ * 𝐏[i] * G⁻½)) * G½)) end) :
                         (for i = 1:k Vec[:, i] = vecP(ℍ(G½ * log(ℍ(G⁻½ * 𝐏[i] * G⁻½)) * G½)) end)
    return Vec
end




"""
```
function CVsetup(k       :: Int,
                 nCV     :: Int;
                 shuffle :: Bool = false)
```
Given `k` elements and a parameter `nCV`, a nCV-fold cross-validation
is obtained defining ``nCV`` permutations of ``k`` elements
in ``nTest=k÷nCV`` (intger division) elements for the test and
``k-nTest`` elements for the training,
in such a way that each element is represented in only one permutation.

Said differently, given a length `k` and the number of desired cross-validations
`nCV`, this function generates indices from the sequence of natural numbers
``1,..,k`` to obtain all nCV-fold cross-validation sets.
Specifically, it generates ``nCV`` vectors of indices for generating test sets
and ``nCV`` vectors of indices for geerating training sets.

If optional keyword argument `shuffle` is true,
the sequence of natural numbers ``1,..,k`` is shuffled before
running the function, thus in this case two successive runs of this function
will give different cross-validation sets, hence different accuracy scores.
By default `shuffle` is false, so as to allow exactly the same result
in successive runs.
Notae that no random initialization for the shuffling is provided, so as to
allow the replication of the same random sequences starting again
the random generation from scratch.

This function is used in [`CV_mdm`](@ref). It constitutes the fundamental
basis to implement customized cross-validation iprocedures.

Return the 4-tuple with:

- The size of each training set (integer),
- The size of each testing set (integer),
- A vector of `nCV` vectors holding the indices for the training sets,
- A vector of `nCV` vectors holding the indices for the corresponding test sets.

**Examples**
```
using PosDefManifoldML

CVsetup(10, 2)
# return:
# (5, 5,
# Array{Int64,1}[[6, 7, 8, 9, 10], [1, 2, 3, 4, 5]])
# Array{Int64,1}[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],

CVsetup(10, 2, shuffle=true)
# return:
# (5, 5,
# Array{Int64,1}[[5, 4, 6, 1, 9], [3, 7, 8, 2, 10]])
# Array{Int64,1}[[3, 7, 8, 2, 10], [5, 4, 6, 1, 9]],

CVsetup(10, 3)
# return:
# (7, 3,
# Array{Int64,1}[[4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6]])
# Array{Int64,1}[[1, 2, 3], [4, 5, 6], [7, 8, 9, 10]],

```

"""
function CVsetup(k       :: Int,
                 nCV     :: Int;
                 shuffle :: Bool = false)
    if nCV == 1 @error 📌*", CVsetup function: The number of cross-validation must be bigger than one" end
    nTest = k÷nCV
    nTrain = k-nTest
    #rng = MersenneTwister(1900)
    shuffle ? a=shuffle!( Vector(1:k)) : a=Vector(1:k)
    indTrain = [Vector{Int64}(undef, 0) for i=1:nCV]
    indTest  = [Vector{Int64}(undef, 0) for i=1:nCV]
    # vectors of indices for test and training sets
    j=1
    for i=1:nCV-1
        indTest[i]=a[j:j+nTest-1]
        for g=j+nTest:length(a) push!(indTrain[i], a[g]) end
        for l=i+1:nCV, g=j:j+nTest-1 push!(indTrain[l], a[g]) end
        j+=nTest
    end
    indTest[nCV]=a[j:end]
    return nTrain, nTest, indTrain, indTest
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

The training and test sets can be used to train and test an [ML model](@ref).

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

𝐏Tr, 𝐏Te, yTr, yTe=gen2ClassData(10, 30, 40, 60, 80, 0.25)

# 𝐏Tr=training set: 30 matrices for class 1 and 40 matrices for class 2
# 𝐏Te=testing set: 60 matrices for class 1 and 80 matrices for class 2
# all matrices are 10x10
# yTr=a vector of 70 labels for 𝐓r
# yTe=a vector of 140 labels for 𝐓e

```
"""
function gen2ClassData(n        ::  Int,
                       k1train  ::  Int,
                       k2train  ::  Int,
                       k1test   ::  Int,
                       k2test   ::  Int,
                       separation :: Real = 0.1)
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
