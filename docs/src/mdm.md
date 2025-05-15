# mdm.jl

This unit implements the Riemannian **MDM (Minimum Distance to Mean)**
classifier for the manifold of positive definite (PD) matrices,
both real (symmetric PD) or complex (Hermitian PD) matrices.
The MDM is a *simple*, yet *efficient*, *deterministic* and *paramater-free* classifier acting
directly on the manifold of positive definite matrices (Barachat el *al.*, 2012; Congedo et *al.*, 2017a [🎓](@ref)): given a number of PD matrices representing *class means*, the MDM classify an unknown datum (also a PD matrix) as belonging to the class whose mean is the closest to the datum. The process is
illustrated in the upper part of this
[figure](https://raw.githubusercontent.com/Marco-Congedo/PosDefManifoldML.jl/master/docs/src/assets/Fig1.jpg).

The MDM classifier involves only the concepts of a *distance function* for two PD matrices and a *mean* (center of mass or barycenter) for a number of them. Those are defined for any given metric, a [Metric](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#Metric::Enumerated-type-1)
enumerated type declared in [PosDefManifold](https://marco-congedo.github.io/PosDefManifold.jl/dev/).

Currently supported metrics are:

| metric (distance) |   mean estimation |    known also as            |
|:----------------- |:------------------|:----------------------------|
| Euclidean         | arithmetic        |  Arithmetic                 |
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
In order to use these metrics you need to install the
*PosDefManifold* package.

The **fit**, **predict** and **crval** functions for the MDM model are
reported in the [cv.jl](@ref) unit, since those are homogeneous across all
machine learning models.
Here it is reported the [`MDMmodel`](@ref)
abstract type, the [`MDM`](@ref) structure and the following functions,
which typically you will not need to access directly, but are
nonetheless provided to facilitate low-level operations with MDM classifiers:

|         function       |           description             |
|:-----------------------|:----------------------------------|
| [`barycenter`](@ref)   | compute the barycenter of positive definite matrices for fitting the MDM model |
| [`distances`](@ref) | compute the distances of a matrix set to a set of means |


```@docs
MDMmodel
MDM
barycenter
distances
```
