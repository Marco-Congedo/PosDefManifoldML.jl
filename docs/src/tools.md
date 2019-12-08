# tools.jl

This unit implements tools that are useful for building Riemannian
and Euclidean machine learning classifiers.


## Content

|         function       |           description             |
|:-----------------------|:----------------------------------|
| [`tsMap`](@ref)        | project data on a tangent space to apply Euclidean ML models therein |
| [`tsWeights`](@ref)| generator of weights for tagent space mapping |
| [`gen2ClassData`](@ref)| generate 2-class positive definite matrix data for testing Riemannian ML models |
| [`predictErr`](@ref)| prediction error given a vector of true labels and a vector of predicted labels |
| [`rescale!`](@ref)| Rescale the rows of a real matrix to be in range [a, b] |




```@docs
tsMap
tsWeights
gen2ClassData
predictErr
rescale!
```
