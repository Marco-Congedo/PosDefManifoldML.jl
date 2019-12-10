# Tutorial

If you didn't, please read first the [Overview](@ref).

*PosDefManifoldML* features two bacic **pipelines**:

**1)** a machine learning (ML) model is first **fit** (trained), then it can be used to **predict** the *labels* of testing data or the *probability* of the data to belong to each class. The raw prediction function of the models is available as well.

**2)** a **k-fold cross-validation** procedure allows to estimate directly the **accuracy** of ML models and compare them.

What *PosDefManifoldML* does for you is to allow an homogeneous syntax to run these two pipelines for all implemented ML models,
it does not matter if they act directly on the manifold of positive definite matrices or on the tangent space.
Furthermore, models acting on the tangent space can take as input Euclidean feature vectors instead of positive definite matrices, thus they can be used in many more situations.

**get data**

A real data example will be added soon. For now, let us create simulated data for a **2-class example**.
First, let us create symmetric positive definite matrices (real positive definite matrices):

```
using PosDefManifoldML

PTr, PTe, yTr, yTe=gen2ClassData(10, 30, 40, 60, 80);
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

Since the Fisher metric is the default (for all ML models),
the above is equivalent to:

```
m = fit(MDM(), PTr, yTr)
```

In order to adopt another metric:

```
m1 = fit(MDM(logEuclidean), PTr, yTr)
```

**Predict (classify data)**

In order to predict the labels of unlabeled data (which we have stored in `PTe`), we invoke

```
yPred=predict(m, PTe, :l)
```

The prediction error in percent can be retrived with

```
predictErr(yTe, yPred)
```

or by

```
predictErr(yPred, yTe)
```

where in `yTe` we have stored the *true* labels for the
matrices in `PTe`.

If instead we wish to estimate the probabilities for the matrices in `PTe` of belonging to each class:

```
predict(m, PTe, :p)
```

Finally, the output functions of the MDM are obtaine by (see [`predict`](@ref))

```
predict(m, PTe, :f)
```

### MDM Pipeline 2. (cross-validation)

The balanced accuracy estimated by a *k-fold cross-validation* is obtained such as (10-fold by default)

```
cv = cvAcc(MDM(), PTr, yTr)
```

Struct `cv` has been created and therein you have access to average accuracy and confusion matrix as well as accuracies
and confusion matrices for all folds. For example,
print the average confusion matrix:

```
cv.avgCnf
```

See [`CVacc`](@ref) for details on the fields of cross-validation objects.


## Example using the ENLR model

The **elastic net logistic regression (ENLR)** classifier is an example of classifier acting on the tangent space. Besides the **metric** (see above) used to compute a base-point for projecting the data onto the tangent space, it has a parameter **alpha** and an hyperparameter **lambda**. The **alpha** parameter allows to trade off between a pure **ridge** LR model (``α=0``) and a pure **lasso** LR model (``α=1``), which is the default. Given an alpha value, the model is fitted with a number of values for the ``λ`` (regularization) hyperparameter. Thus, differently from the previous example, tuning the ``λ`` hyperparameter is necessary.

Also, keep in mind
that the [`fit`](@ref) and [`predict`](@ref) methods for ENLR models accept optional keyword arguments that are specific to this model.

**get data**

Let us get some simulated data (see the previous example for explanations).

```
PTr, PTe, yTr, yTe=gen2ClassData(10, 30, 40, 60, 80);
```

### ENLR Pipeline 1. (fit and predict)

**Craete and fit ENLR models**

By default, the Fisher metric ic adopted and a lasso model is fitted. The best value
for the lambda hyperparameter is found by cross-validation:

```
m1 = fit(ENLR(), PTr, yTr; w=:balanced)
```

Notice that since the class are unbalanced, with the `w=:balanced`
argument (we may as well just use `w=:b`) we have requested to compute a balanced mean for projecting the matrices in `PTr` onto the tangent space.

The optimal value of lambda for this training data is

```
m1.best.lambda
```

As in *GLMNet.jl*, the intercept and beta terms are retrived by
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
m2 = fit(ENLR(), PTr, yTr; w=:b, alpha=0)
```

Values of `alpha` in range ``(0, 1)`` fit instead an elastic net LR model. In the following we also request not to standardize predictors:

```
m3 = fit(ENLR(Fisher), PTr, yTr; w=:b, alpha=0.9, standardize=false)
```

In order to find the regularization path we use the
`fitType` keyword argument:

```
m1 = fit(ENLR(Fisher), PTr, yTr; w=:b, fitType=:path)
```

The values of lambda along the path are given by

```
m1.path.lambda
```

We can also find the best value of the lambda hyperparameter and the regularization path at once, calling:

```
m1 = fit(ENLR(Fisher), PTr, yTr; w=:b, fitType=:all)
```

For changing the metric see [MDM Pipeline 1. (fit and predict)](@ref).

See the documentation of the [`fit`](@ref) ENLR method for
details on all available optional arguments.


**Classify data (predict)**

For prediction, we can request to use the best model (optimal lambda), to use a specific model of the regularization path or to use all the model
in the regalurization path.
Note that with the last call we have done here above both the `.best` and `.path` field of the `m1` structure have been created.

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

while for a specific model in the path (e.g., the 1Oth model):

```
yPred=predict(m1, PTe, :l, :path, 10)
```

### ENLR Pipeline 2. (cross-validation)

The balanced accuracy estimated by a *k-fold cross-validation* is obtained with the exact same basic syntax for all models, with
some specific optional keyword arguments for
models acting in the tangent space, for example:

```
cv = cvAcc(ENLR(), PTr, yTr; w=:b)
```

In order to perform another cross-validation
arranging the training data differently in the folds:

```
cv = cvAcc(ENLR(), PTr, yTr; w=:b, shuffle=true)
```

This last command can be invoked repeatedly.



## Example using SVM models

The SVM ML model actually encapsulates several **support-vector classification** and **support-vector regression** models.
Here we are here concerned with the former, which include
the **C-Support Vector Classification (SVC)**, the **Nu-Support Vector Classification (NuSVC)**, similar to SVC but using a parameter to control the number of support vectors, and the **One-Class SVM (OneClassSVM)**, which is used in general for unsupervised outlier detection. They all act in the tangent space like ENLR models.
Besides the **metric** (see [MDM Pipeline 1. (fit and predict)](@ref)) used to compute a base-point for projecting the data onto the tangent space and the type of SVM model
(the **svmType**, = `SVC` (default), `NuSVC` or `OneClassSVM`), the main parameter is the **kernel**.
Avaiable kernels are:
- `RadialBasis` (default)
- `Linear`
- `Polynomial`
- `Sigmoid`

Several parameters are available for building all these kernels
besides the linear one, which has no parameter. Like for ENLR, for SVM models also an hyperparameter is to be found by cross-validation.

**get data**

Let us get some simulated data as in the previous examples.

```
PTr, PTe, yTr, yTe=gen2ClassData(10, 30, 40, 60, 80);
```

### SVM Pipeline 1. (fit and predict)

**Craete and fit SVM models**

By default, a C-Support Vector Classification model is fitted:

```
m1 = fit(SVM(), PTr, yTr; w=:b)
```

Notice that as for the example above with for ENLR model, we have requested to compute a balanced mean for projecting the matrices in `PTr` onto the tangent space.

In order to fit a Nu-Support Vector Classification model:

```
m2 = fit(SVM(), PTr, yTr; w=:b, svmType=NuSVC)
```

For using other kernels, e.g.:

```
m3 = fit(SVM(), PTr, yTr; w=:b, svmType=NuSVC, kernel=Linear)
```

In the following we also request not to rescale predictors:

```
m3 = fit(SVM(), PTr, yTr;
        w=:b, svmType=NuSVC, kernel=Linear, rescale=())
```

By default the Fisher metric is used. For changing it see [MDM Pipeline 1. (fit and predict)](@ref).

See the documentation of the [`fit`](@ref) ENLR method for
details on all available optional arguments.


**Classify data (predict)**

Just the same as for the other models:

```
yPred=predict(m1, PTe)

# prediction error in percent
predictErr(yPred, yTe)

# predict probabilities of matrices in `PTe` to belong to each class
predict(m1, PTe, :p)

# output function of the model for each class
predict(m1, PTe, :f)
```

### SVM Pipeline 2. (cross-validation)

Again, the balanced accuracy estimated by a *k-fold cross-validation* is obtained with the exact same basic syntax for all models, with
some specific optional keyword arguments for
models acting in the tangent space, for example:

```
cv = cvAcc(SVM(), PTr, yTr; w=:b)
```
