# conditioners.jl

This unit implements *conditioners*, also called *pre-conditioners* and **pipelines**,
which are specified sequences of conditioners.

Pipelines are applied to the data (symmetric positive-definite matrices) in order
to increase the classification performance and/or to reduce the computational complexity of classifiers.

## Conditioners

The available conditioners are

|  Conditioner type      |           description             |
|:-----------------------|:----------------------------------|
| [`Tikhonov`](@ref) | Tikhonov regularization |
| [`Recenter`](@ref) | Recentering aroung the identity matrix w/o dimensionality reduction |
| [`Compress`](@ref)| Global scaling reducing the norm |
| [`Equalize`](@ref)| Individual scaling (normalization) to reduce the norm |
| [`Shrink`](@ref)| Move along the geodesic to reduce the norm |

## Pipelines

Pipelines are stored as a dedicated tuple type - see [`Pipeline`](@ref).

## Content

|         function       |           description             |
|:-----------------------|:----------------------------------|
| [`@pipeline`](@ref)     | Macro to create a pipeline |
| [`fit!`](@ref)| Fit a pipeline and transform the data |
| [`transform!`](@ref)| Transform the data using a fitted pipeline |
| [`pickfirst`](@ref)| Return a copy of a specific conditioner in a pipeline|
| [`includes`](@ref)| Check whether a conditioner type is in a pipeline |
| [`dim`](@ref)| Dimension determined by a recentering pre-conditioner in a pipeline|



```@docs
Tikhonov
Recenter
Compress
Equalize
Shrink
Pipeline
@pipeline
fit!
transform!
pickfirst
includes
dim
```
