# cv.jl

This unit implements **cross-validation** procedures for estimating
the **accuracy** and **balanced accuracy** of machine learning models.
It also reports the documentation of the **fit** and **predict** functions, as they are common to all models.

**Content**

|     struct       |           description             |
|:-----------------|:----------------------------------|
| [`CVres`](@ref)  | it encapsulates the results of cross-validation procedures for estimating accuracy|

|         function       |           description             |
|:-----------------------|:----------------------------------|
| [`fit`](@ref)     | fit a model with training data, or create and fit it |
| [`predict`](@ref) | preidct labels, probabilities or scoring functions on test data |
| [`crval`](@ref)   | estimate accuracy of a model by cross-validation|
| [`cvSetup`](@ref) | generate indexes for performing cross-validtions |

```@docs
CVres
fit
predict
crval
cvSetup
```
