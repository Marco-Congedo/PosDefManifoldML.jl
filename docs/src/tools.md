# tools.jl

This unit implements tools that are useful for building Riemannian
and Euclidean machine learning classifiers.


## Content

|         function       |           description             |
|:-----------------------|:----------------------------------|
| [`projectOnTS`](@ref)  | project data on a tangent space to apply Euclidean ML models |
| [`CVsetup`](@ref)      | generate indexes for performing cross-validtions |
| [`gen2ClassData`](@ref)| generate 2-class positive definite matrix data for testing Riemannian ML models |


```@docs
projectOnTS
CVsetup
gen2ClassData
```
