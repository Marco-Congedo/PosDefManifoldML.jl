# stats_inferential.jl

This unit implements **inferential statistics** (hypothesis tests) for cross-validation analysis.

## Content

|         function       |           description             |
|:-----------------------|:----------------------------------|
| [`testCV`](@ref)    | Test the average error loss observed in a cross-validation against an expected (chance) level|
| [`testCV`](@ref)    | Test the average error loss observed in two cross-validation obtained by two different models and/or processing pipelines on the same data |


**See also** [stats_descriptive.jl](@ref)

```@docs
testCV
```
