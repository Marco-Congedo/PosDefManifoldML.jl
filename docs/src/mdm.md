# mdm.jl

This unit implemets the Riemannian **MDM (Minimum Distance to Mean)**
classifier for the manifold of positive definite (PD) matrices.
The MDM is a *simple*, yet *efficient*, *deterministic* and *paramater-free* classifier acting
directly on the manifold of positive definite matrices (Barachat el *al.*, 2012; Congedo et *al.*, 2017a [🎓](@ref)): given a number of PD matrices representing *class means*, the MDM classify an unknown datum (also a PD matrix) as belonging to the class whose mean is the closest to the datum. The process is
illustrated in the upper part of this
[figure](https://raw.githubusercontent.com/Marco-Congedo/PosDefManifoldML.jl/master/docs/src/assets/Fig1.jpg).

The MDM classifier involves only the concepts of a *distance function* for two PD matrices and a *mean* (center of mass) for a number of them. Those are defined for any given metric, a [Metric](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#Metric::Enumerated-type-1)
enumerated type declared in [PosDefManifold](https://marco-congedo.github.io/PosDefManifold.jl/dev/).

Currently supported metrics are:

| metric (distance) |   mean estimation |    known also as            |
|:----------------- |:------------------|:----------------------------|
| Euclidean         | arithmetic        |                             |
| invEuclidean      | harmonic          |                             |
| ChoEuclidean      | Cholesky Euclidean|                             |
| logEuclidean      | log-Euclidean     |                             |
| logCholesky       | log-Cholesky      |                             |
| Fisher            | Fisher | Cartan, Karcher, Pusz-Woronowicz, Affine-Invariant, ...  |
| logdet0           | logDet | S, α, Bhattacharyya, Jensen, ...       |
| Jeffrey           | Jeffrey | symmetrized Kullback-Leibler          |
| Wasserstein       | Wasserstein | Bures, Hellinger, optimal transport, ...|

Do not use the Von Neumann metric, which is also supported in *PosDefManifold*,
since it does not allow a definition of mean. See
[here](https://marco-congedo.github.io/PosDefManifold.jl/dev/introToRiemannianGeometry/) for details on the metrics.

Besides the [`MDM`](@ref) type declaration and the declaration of some
constructors for it, this unit also include the following functions,
which typically you will not need to access directly as they are called
by functions working in the same way for all models declared in
[train_test.jl](@ref). They are provided nonetheless to facilitate
low-level jobs with MDM classifiers:


|         function       |           description             |
|:-----------------------|:----------------------------------|
| [`getMeans`](@ref)     | compute means of training data for fitting the MDM model |
| [`getDistances`](@ref) | compute the distances of a matrix set to a set of means |
| [`CV_mdm`](@ref)       | perform cross-validations for the MDM classifiers |

```@docs
MDM
getMeans
getDistances
CV_mdm
```
