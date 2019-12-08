# enlr.jl

This unit implements the **elastic net logistic regression (ENLR)**
machine learning model on the tangent space for symmetric positive definite (SDP) matrices, *i.e.*, real PD matrices. This model
features two hyperparameters: a user-defined **alpha** hyperparameter, in range ``[0, 1]``, where ``α=0`` allows a pure **Ridge** LR model and ``α=1`` a pure **lasso** LR model and the **lambda** (regularization) hyperparameter. When the model is fitted, we can request to find the optimal lambda hyperparameter for the given training data using cross-validation and/or to find the regularization path.

The lasso model (default) has enjoyed popularity in the field of *brain-computer interaces* due to the [winning score](http://alexandre.barachant.org/challenges/)
obtained in six international data classification competitions.

The ENLR model is implemented using the Julia
[GLMNet.jl](https://github.com/JuliaStats/GLMNet.jl) package.
See [🎓](@ref) for resources on GLMNet.jl and learn how to use purposefully
this model.

The **fit**, **predict** and **cvAcc** functions for the ENRL models are
reported in the [cv.jl](@ref) unit, since those are homogeneous across all
machine learning models. Here it is reported the [`ENLRmodel`](@ref)
abstract type and the [`ENLR`](@ref) structure.

```@docs
ENLRmodel
ENLR
```
