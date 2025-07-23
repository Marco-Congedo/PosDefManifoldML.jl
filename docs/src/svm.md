# svm.jl

This unit implements several **Suport-Vector Machine (SVM)**
machine learning models on the tangent space for symmetric positive definite
(SDP) matrices, *i.e.*, real PD matrices.
Several models can be obtained with different combinations of the `svmType` and the `kernel` arguments when the model is fitted.
Optimal hyperparameters for the given training data
are found using cross-validation.

All SVM models are implemented using the Julia
[LIBSVM.jl](https://github.com/mpastell/LIBSVM.jl) package.
See [🎓](@ref) for resources on the original LIBSVM C library and learn
how to use purposefully these models.

The **fit**, **predict** and **crval** functions for the SVM models are
reported in the [cv.jl](@ref) unit, since those are homogeneous across all
machine learning models. Here it is reported the [`SVMmodel`](@ref)
abstract type and the [`SVM`](@ref) structure.

```@docs
SVMmodel
SVM
```
