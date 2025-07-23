# cv.jl

This unit implements **cross-validation** procedures for estimating
the **accuracy** and **balanced accuracy** of machine learning models.
It also reports the documentation of the **fit** and **predict** functions, as they are common to all models.

**Content**

|     struct       |           description             |
|:-----------------|:----------------------------------|
| [`CVres`](@ref)  | encapsulate the results of cross-validation procedures for estimating accuracy|

|         function       |           description             |
|:-----------------------|:----------------------------------|
| [`fit`](@ref)     | fit a machine learning model with training data |
| [`predict`](@ref) | given a fitted model, predict labels, probabilities or scoring functions on test data |
| [`crval`](@ref)   | perform a cross-validation and store accuracies, error losses, confusion matrices, the results of a statistical test and other informations|
| [`cvSetup`](@ref) | generate indexes for performing cross-validtions |

```@docs
CVres
fit
predict
crval
cvSetup
```
