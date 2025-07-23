# tools.jl

This unit implements tools that are useful for building Riemannian
and Euclidean machine learning classifiers. For basic usage of the package 
you won't need these functions.

## Content

|         function       |           description             |
|:-----------------------|:----------------------------------|
| [`tsMap`](@ref)        | project data on a tangent space to apply Euclidean ML models therein |
| [`tsWeights`](@ref)| generate weights for tagent space mapping |
| [`gen2ClassData`](@ref)| generate 2-class positive definite matrix data for testing Riemannian ML models |
| [`rescale!`](@ref)| rescale the rows or columns of a real matrix to be in range [a, b] |
| [`demean!`](@ref)| remove the mean of the rows or columns of a real matrix |
| [`normalize!`](@ref)| normalize the rows or columns of a real matrix |
| [`standardize!`](@ref)| standardize the rows or columns of a real matrix |
| [`saveas`](@ref)| save a whatever object to a file |
| [`load`](@ref)| load a whatever object from a file |

```@docs
tsMap
tsWeights
gen2ClassData
rescale!
demean!
normalize!
standardize!
saveas
load
```
