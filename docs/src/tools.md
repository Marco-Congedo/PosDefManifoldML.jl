# tools.jl

This unit implements tools that are useful for building Riemannian
and Euclidean machine learning classifiers.

## Content

|         function       |           description             |
|:-----------------------|:----------------------------------|
| [`tsMap`](@ref)        | project data on a tangent space to apply Euclidean ML models therein |
| [`tsWeights`](@ref)| generator of weights for tagent space mapping |
| [`gen2ClassData`](@ref)| generate 2-class positive definite matrix data for testing Riemannian ML models |
| [`rescale!`](@ref)| Rescale the rows or columns of a real matrix to be in range [a, b] |
| [`demean!`](@ref)| Remove the mean of the rows or columns of a real matrix |
| [`normalize!`](@ref)| Normalize the rows or columns of a real matrix |
| [`standardize!`](@ref)| Standardize the rows or columns of a real matrix |
| [`saveas`](@ref)| Save a whatever object to a file |
| [`load`](@ref)| Load a whatever object from a file |

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
