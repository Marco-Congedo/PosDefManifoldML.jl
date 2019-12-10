# MainModule

This is the main unit containing the **PosDefManifoldML** *module*.

## dependencies

| standard Julia packages |     external packages    |
|:-----------------------:|:-----------------------:|
| [LinearAlgebra](https://bit.ly/2W5Wq8W) |  [PosDefManifold](https://github.com/Marco-Congedo/PosDefManifold.jl)|
| [Statistics](https://bit.ly/2Oem3li) |  [GLMNet](https://github.com/JuliaStats/GLMNet.jl)|
| [Random](https://github.com/JuliaStdlibs/Random.jl) | [Distributions](https://github.com/JuliaStats/Distributions.jl)|
| [Dates](https://github.com/JuliaStdlibs/Dates.jl)| [LIBSVM](https://github.com/mpastell/LIBSVM.jl)|
| [StatsBase](https://github.com/JuliaStats/StatsBase.jl) |  |

The main module does not contains functions.

## types

### MLmodel

As typical in machine learning packages, a type is created (a `struct` in Julia) to specify a ML model. *Supertype*

```abstract type MLmodel end```

is the abstract type for all machine learning
models. *Supertype*

```abstract type PDmodel<:MLmodel end```

is the abstract type for all machine learning
models acting on the positive definite (PD) **manifold** (for example, see [`MDM`](@ref)). *Supertype*

```abstract type TSmodel<:MLmodel end```

is the abstract type for all machine learning
models acting on the **tangent space** (for example, see [`ENLR`](@ref)).

### IntVector

In all concerned functions *class labels* are given as a vector of integers,
of type

```IntVector=Vector{Int}```.

Class labels are natural numbers in ``[1,...,z]``, where ``z`` is the number
of classes.

## Tips & Tricks

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
- `𝐏`: a **generic set** of positive definite matrices.
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
