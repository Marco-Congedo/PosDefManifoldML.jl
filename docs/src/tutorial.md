# Tutorial

*PosDefManifoldML* features two bacic **pipelines**:

**1)** a machine learning (ML) model is first **fit** (trained), then it can be used to **predict** the *labels* of testing data or the *probability* of the data to belong to each class. The raw prediction function of the models is available as well.

**2)** a **k-fold cross-validation** procedure allows to estimate the **accuracy** of ML models and compare them.

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
cv = cvAcc(MDM(Fisher), PTe, yTe, 5)
```

where `5` is the number of folds. This implies that
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
(``α=1``). Given an alpha value, the model is fitted with a number of values for the **lambda**
hyperparameter. Thus, differently from the previous example, an additional
step for tuning this hyperparameter is necessary. Also, keep in mind
that the [`fit`](@ref) and [`predict`](@ref) methods for ENLR models accept optional keyword arguments that are specific to this model.

**get data**

Let us get some simulated data as in the previous example.

```
PTr, PTe, yTr, yTe=gen2ClassData(10, 30, 40, 60, 80)
```

### ENLR Pipeline 1. (fit and predict)

**Craete and fit ENLR models**

By default, a lasso model is fitted:

```
m1 = fit(ENLR(Fisher), PTr, yTr)
```

In order to fit a ridge LR model:

```
m2 = fit(ENLR(Fisher), PTr, yTr; alpha=0)
```

Values of `alpha` in range ``(0, 1)`` fit an elastic net LR model. In the following we also request to standardize predictors:

```
m3 = fit(ENLR(Fisher), PTr, yTr; alpha=0.9, standardize=true)
```

See the documentation of the [`fit`](@ref) ENLR method for
details on all available optional arguments.


#### Select the best model for ENLR

```
cvLambda!(m1, PTr, yTr)
```

Note that the [`fit`](@ref) and [`cvLambda!`](@ref) function
have populated the `m1` struct. For example, try the following:

```
m1.path
m1.path.lambda
m1.bestModel
m1.bestλ
m1.cvλ.meanloss
m1.cvλ.stdloss
m1.cvλ.lambda
```

**Classify data (predict)**

Since we have invoked the `cvLambda!` function to find
the best model by cross-validation, by default this model is used for prediction:

```
yPred=predict(m1, PTe, :l)

# prediction error in percent
predictErr(yPred, yTe)

# predict probabilities of matrices in `PTe` to belong to each class
predict(m1, PTe, :p)

# output function of the model for each class
predict(m1, PTe, :f)
```

The following two lines are equivalent

```
yPred=predict(m1, PTe, :l)
yPred=predict(m1, PTe, :l, m1.bestModel)
```

We can also request the predition for all models, as

```
yPred=predict(m1, PTe, :l, 0)
```

or of a specific model in the path, as

```
yPred=predict(m1, PTe, :l, 10)
```

### ENLR Pipeline 2. (cross-validation)

The balanced accuracy estimated by a *k-fold cross-validation* is obtained with the exact same syntax for all models, thus, for example:

```
cv = cvAcc(ENLR(Fisher), PTe, yTe, 5)
```
