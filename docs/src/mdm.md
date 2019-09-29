# mdm.jl

This unit implemets the Riemannian **MDM (Minimum Distance to Mean)**
classifier for the manifold of positive definite matrices.

Besides the [`MDM`](@ref) type declaration and the declaration of some
constructors for it, this unit also include the following functions,
which typically you will not need to access directly:

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
