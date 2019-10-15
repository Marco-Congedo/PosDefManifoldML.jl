# tools.jl

This unit implements tools that are useful for building Riemannian
and Euclidean machine learning classifiers.


## Content

|         function       |           description             |
|:-----------------------|:----------------------------------|
| [`tsMap`](@ref)        | project data on a tangent space to apply Euclidean ML models therein |
| [`gen2ClassData`](@ref)| generate 2-class positive definite matrix data for testing Riemannian ML models |
| [`predictErr`](@ref)| prediction error given a vector of true labels and a vector of predicted labels |


```@docs
tsMap
gen2ClassData
predictErr
```
