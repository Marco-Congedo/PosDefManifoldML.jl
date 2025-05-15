# stats_descriptive.jl

This unit implements **descriptive statistics** for cross-validation analysis.

## Content

|         function       |           description             |
|:-----------------------|:----------------------------------|
| [`confusionMat`](@ref)| Confusion matrix given true and predicted labels |
| [`predictAcc`](@ref)| Prediction accuracy given true and predicted labels or a confusion matrix|
| [`predictErr`](@ref)| Prediction error given true and predicted labels or a confusion matrix|
| [`binaryloss`](@ref)| Binary error loss |

**See also** [stats_inferential.jl](@ref)

```@docs
confusionMat
predictAcc
predictErr
binaryloss
```
