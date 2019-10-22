# Tutorial

*PosDefManifoldML* features two bacic **pipelines**:

**1)** a machine learning (ML) model is first **fit** (trained), then it can be used to **predict** the *labels* of testing data or the *probability* of the data to belong to each class. The raw prediction function of the models is available as well.

**2)** a **k-fold cross-validation** procedure allows to estimate directly the **accuracy** of ML models and compare them.

What *PosDefManifoldML* does for you is to allow an homogeneous syntax to run these two pipelines for all implemented ML models,
it does not matter if they act directly on the manifold of positive definite matrices or on the tangent space.

**get data**

A real data example will be added soon. For now, let us create simulated data for a **2-class example**.
First, let us create symmetric positive definite matrices (real positive definite matrices):

```
using PosDefManifoldML

PTr, PTe, yTr, yTe=gen2ClassData(10, 30, 40, 60, 80)
```

- `PTr` is the simulated training set, holding 30 matrices for class 1 and 40 matrices for class 2
- `PTe` is the testing set, holding 60 matrices for class 1 and 80 matrices for class 2.
- `yTr` is a vector of 70 labels for the training set
- `yTe` is a vector of 140 labels for the testing set

All matrices are of size 10x10.

## Example using the MDM model

The **minimum distance to mean (MDM)** classifier is an example of classifier acting directly on the manifold. It is deterministic and no hyperparameter
tuning is requested.

### MDM Pipeline 1. (fit and predict)

**Craete and fit an MDM model**

An MDM model is created and fit with training data such as

```
m = fit(MDM(Fisher), PTr, yTr)
```

where `Fisher` is the usual choice of a [Metric](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#Metric::Enumerated-type-1)
as declared in the parent package [PosDefManifold](https://marco-congedo.github.io/PosDefManifold.jl/dev/).

The model can also be just created by

```
empty_model = MDM(Fisher)
```

and fitted later using the [`fit`](@ref) function.


**Predict (classify data)**

In order to predict the labels of unlabeled data, we invoke

```
yPred=predict(m, PTe, :l)
```

The prediction error in percent can be retrived with

```
predictErr(yTe, yPred)
```

If instead we wish to estimate the probabilities for the matrices in `PTe` of belonging to each class,

```
predict(m, PTe, :p)
```

Finally, the output functions of the MDM are obtaine by (see [`predict`](@ref))

```
predict(m, PTe, :f)
```

### MDM Pipeline 2. (cross-validation)

The balanced accuracy estimated by a *k-fold cross-validation* is obtained
such as

```
cv = cvAcc(MDM(Fisher), PTr, yTr, 10)
```

where `10` is the number of folds. This implies that
at each cross-validation, 1/5th of the matrices is used for training and the remaining for testing.

Struct `cv` has been created and therein you have access to average accuracy and confusion matrix as well as accuracies
and confusion matrices for all folds. For example,
print the average confusion matrix:

```
cv.avgCnf
```

See [`CVacc`](@ref) for details on the fields of cross-validation objects.

## Example using the ENLR model

The **elastic net logistic regression (ENLR)** classifier is an example of classifier acting on the tangent space. It has two hyperparameters. The **alpha** hyperparameter is supplied by the user and allows to trade off
between a pure **ridge** LR model (``α=0``) and a pure **lasso** LR model
(``α=1``). Given an alpha value, the model is fitted with a number of values for the **lambda** (regularization)
hyperparameter. Thus, differently from the previous example, tuning this hyperparameter is necessary. Also, keep in mind
that the [`fit`](@ref) and [`predict`](@ref) methods for ENLR models accept optional keyword arguments that are specific to this model.

**get data**

Let us get some simulated data as in the previous example.

```
PTr, PTe, yTr, yTe=gen2ClassData(10, 30, 40, 60, 80)
```

### ENLR Pipeline 1. (fit and predict)

**Craete and fit ENLR models**

By default, a lasso model is fitted and the best value
for the lambda hyperparameter is found:

```
m1 = fit(ENLR(Fisher), PTr, yTr)
```

The optimal value of lambda for this training data is

```
m1.best.lambda
```

The intercept and beta terms are retrived by
```
m1.best.a0
m1.best.betas
```

The number of non-zero beta coefficients can be found for example by

```
length(unique(m1.best.betas))-1
```

In order to fit a ridge LR model:

```
m2 = fit(ENLR(Fisher), PTr, yTr; alpha=0)
```

Values of `alpha` in range ``(0, 1)`` fit instead an elastic net LR model. In the following we also request to standardize predictors:

```
m3 = fit(ENLR(Fisher), PTr, yTr; alpha=0.9, standardize=true)
```

In order to find the regularization path we use the
`fitType` keyword argument:

m1 = fit(ENLR(Fisher), PTr, yTr; fitType=:path)

The values of lambda along the path are given by

```
m1.path.lambda
```

In order to find the best value of the lambda hyperparameter and the regularization path at once:

m1 = fit(ENLR(Fisher), PTr, yTr; fitType=:all)

See the documentation of the [`fit`](@ref) ENLR method for
details on all available optional arguments.


**Classify data (predict)**

For prediction, we can request to use the best model (optimal lambda) to use a specific model of the regularization path or all of them.
Note that with the last call both the `.best` and `.path` field of the `m1` structure have been created.

By default, prediction is obtained from the best model
and we request to predict the labels:

```
yPred=predict(m1, PTe)

# prediction error in percent
predictErr(yPred, yTe)

# predict probabilities of matrices in `PTe` to belong to each class
predict(m1, PTe, :p)

# output function of the model for each class
predict(m1, PTe, :f)
```

In order to request the predition of labels for all models
in the regularization path:

```
yPred=predict(m1, PTe, :l, :path, 0)
```

while for a specific model in the path:

```
yPred=predict(m1, PTe, :l, :path, 10)
```

### ENLR Pipeline 2. (cross-validation)

The balanced accuracy estimated by a *k-fold cross-validation* is obtained with the exact same syntax for all models, thus, for example:

```
cv = cvAcc(ENLR(Fisher), PTr, yTr, 10)
```

In order to perform another 10-fold cross-validation
arranging the training data differently in the folds:

```
cv = cvAcc(ENLR(Fisher), PTr, yTr, 10; shuffle=true)
```

This last command can be invoked repeatedly.
