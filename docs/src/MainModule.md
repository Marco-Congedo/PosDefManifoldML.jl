# MainModule

This is the main unit containing the **PosDefManifoldML** *module*.

## dependencies

| standard Julia packages |     external packages    |
|:-----------------------:|:-----------------------:|
| [LinearAlgebra](https://bit.ly/2W5Wq8W) |  [PosDefManifold](https://github.com/Marco-Congedo/PosDefManifold.jl)|
| [Statistics](https://bit.ly/2Oem3li) |  [GLMNet](https://github.com/JuliaStats/GLMNet.jl)|
| [Random](https://github.com/JuliaStdlibs/Random.jl) | [Distributions](https://github.com/JuliaStats/Distributions.jl)|
| [Dates](https://github.com/JuliaStdlibs/Dates.jl)| [LIBSVM](https://github.com/mpastell/LIBSVM.jl)|
| [StatsBase](https://github.com/JuliaStats/StatsBase.jl) | [PermutationTests](https://github.com/Marco-Congedo/PermutationTests.jl) |
| [Diagonalizations](https://github.com/Marco-Congedo/Diagonalizations.jl) | [Folds](https://github.com/JuliaFolds/Folds.jl) |
| [Serialization](https://github.com/JuliaLang/julia/blob/master/stdlib/Serialization/src/Serialization.jl) |  |


## types

The following types are used.

### MLmodel

This type is created (a `struct` in Julia) to hold a ML model.
The abstract type for all machine learning models is

```julia
abstract type MLmodel end
```

The abstract type for all machine learning
models acting on the positive definite (PD) **manifold** (for example, see [`MDM`](@ref)) is

```julia
abstract type PDmodel<:MLmodel end
```

The abstract type for all machine learning
models acting on the **tangent space** (for example, see [`ENLR`](@ref)) is

```julia
abstract type TSmodel<:MLmodel end
```

### Conditioner

Conditioners (also called pre-conditioners) are data transformations for the manifold
of positive-definite matrices. A sequence of conditioners forms a **pipeline** (see [`Pipeline`](@ref)).
Their role of a transforming pipeline is to improve the classification performance
and/or to reduce the computational complexity.

A type is created (a `struct` in Julia) to specify a conditioner. 
The abstract type for all conditioners. is

```julia
abstract type Conditioner end # all SPD matrix transformations
```

These here below are the types for the conditioners currently implemented:

```julia
    abstract type Tikhonov      <: Conditioner end # Tikhonov regularization
    abstract type Recentering   <: Conditioner end # Recentering with or w/o dim reduction
    abstract type Compressing   <: Conditioner end # Compressing (global scaling)
    abstract type Equalizing    <: Conditioner end # Equalizing (individual scaling)
    abstract type Shrinking     <: Conditioner end # Geodesic Shrinking
```

### IntVector

In all concerned functions, *class labels* are given as a vector of integers,
of type

```julia
IntVector=Vector{Int}
```

Class labels must be natural numbers in ``[1,...,z]``, where ``z`` is the number
of classes.

## Tips & Tricks

### working with metrics

In order to work with metrics for the manifold of positive definite matrices, make sure you install the *PosDefManifold* package.

### the ℍVector type

Check this documentation on [typecasting matrices](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#typecasting-matrices-1).

### notation & nomenclature

Throughout the code of this package the following
notation is followed:

- **scalars** and **vectors** are denoted using lower-case letters, e.g., `y`,
- **matrices** using upper case letters, e.g., `X`
- **sets (vectors) of matrices** using bold upper-case letters, e.g., `𝐗`.

The following nomenclature is used consistently:

- `𝐏Tr`: a **training set** of positive definite matrices
- `𝐏Te`: a **testing set** of positive definite matrices
- `𝐏`, `𝐐`: a **generic set** of positive definite matrices.
- `w`: a **weights vector** of non-negative real numbers
- `yTr`: a **training set class labels vector** of positive integer numbers (1, 2,...)
- `yTe`: a **testing set class labels vector** of positive integer numbers
- `y`: a **generic class labels vector** of positive integer numbers.
- `z`: **number of classes** of a ML model
- `k`: **number of matrices** in a set

In the examples, bold upper-case letters are replaced by
upper case letters in order to allow reading in the REPL.

### acronyms

- MDM: minimum distance to mean
- ENLR: Elastic-Net Logistic Regression
- SVM: Support-Vector Machine
- cv: cross-validation
