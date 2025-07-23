# Tutorial

If you didn't, please read first the [Overview](@ref).

*PosDefManifoldML* features two bacic machine learning modes of operation:

 - **train-test**: a machine learning (ML) model is first **fitted** (trained), then it can be used to **predict** (test) the *labels* of testing data or the *probability* of the data to belong to each class. The raw prediction function of the models is available as well.

 - a **k-fold cross-validation** procedure allows to estimate the **accuracy** of ML models and compare them.

The train-test mode is useful in **cross-subject** and **cross-session** settings, 
while cross-validation is the standard for **within-session** settings.

What *PosDefManifoldML* does for you is to allow an homogeneous syntax to operate in these two modes for all implemented ML models,
it does not matter if they act directly on the manifold of positive definite matrices or on the tangent space.
It also features 

- **Pre-conditining pipelines**, which can drastically reduce the execution time
- **Adaptation** techniques, which, besides being very useful in cross-session and cross-subject settings, are instrumental for implementing on-line modes of operation.

Note that models acting on the tangent space can take as input Euclidean feature vectors instead of positive definite matrices, thus they can be used in many more situations.

**Get data**

Let us create simulated data for a **2-class example**.
First, let us create symmetric positive definite matrices (real positive definite matrices):

```julia
using PosDefManifoldML, PosDefManifold

PTr, PTe, yTr, yTe=gen2ClassData(10, 30, 40, 60, 80, 0.1);
```

- `PTr` is the simulated training set, holding 30 matrices for class 1 and 40 matrices for class 2
- `PTe` is the testing set, holding 60 matrices for class 1 and 80 matrices for class 2.
- `yTr` is a vector of 70 labels for the training set
- `yTe` is a vector of 140 labels for the testing set

All matrices are of size 10x10.

## Examples using the MDM model

The **minimum distance to mean (MDM)** classifier is an example of classifier acting directly on the manifold. It is deterministic and no hyperparameter tuning is needed.

### MDM train-test

**Craete and fit an MDM model**

An MDM model is created and fitted with training data such as

```julia
m = fit(MDM(Fisher), PTr, yTr)
```

where `Fisher` (affine-invariant) is the usual choice of a [Metric](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#Metric::Enumerated-type-1)
as declared in the parent package [PosDefManifold](https://marco-congedo.github.io/PosDefManifold.jl/dev/).

Since the Fisher metric is the default (for all ML models),
the above is equivalent to:

```julia
m = fit(MDM(), PTr, yTr)
```

In order to adopt another metric:

```julia
m1 = fit(MDM(logEuclidean), PTr, yTr)
```

**Predict (classify data)**

In order to predict the labels of unlabeled data (which we have stored in `PTe`), we invoke

```julia
yPred=predict(m, PTe, :l)
```

The prediction error in percent can be retrived with

```julia
predictErr(yTe, yPred)
```

the predicton accuracy as

```julia
predictAcc(yTe, yPred)
```

and the confusion matrix as

```julia
confusionMat(yTe, yPred)
```

where in `yTe` we have stored the *true* labels for the
matrices in `PTe`.

If instead we wish to estimate the probabilities for the matrices in `PTe` of belonging to each class:

```julia
predict(m, PTe, :p)
```

Finally, the output functions of the MDM are obtaine by (see [`predict`](@ref))

```julia
predict(m, PTe, :f)
```

### MDM cross-validation

The balanced accuracy estimated by a *k-fold cross-validation* is obtained such as (10-fold by default)

```julia
cv = crval(MDM(), PTr, yTr)
```

As for all functions in the Julia language, the first time you run a function it is compiled,
so it is slow. To appreciate the speed, run it again 

```julia
cv = crval(MDM(), PTr, yTr)
```

Struct `cv` has been created and therein you have access to average accuracy and confusion matrix as well as accuracies and confusion matrices for all folds. For example,
print the average confusion matrix (expressed in *proportions*):

```julia
cv.avgCnf
```

See [`CVres`](@ref) for details on the fields of cross-validation objects.

### MDM adaptation

Let's see how to adapt a pre-conditioning pipeline. Suppose you have data from two 
sessions or two subjects, `s1` and `s2`.
We want to use `s1` to train a machine learning model on the tangent space and `s2` to test it.
A pipeline is fitted on `s1` and we want this pipeline to adapt to `s2` for testing.
If the pipeline includes a recentering pre-conditioner, we need to make sure that
the dimensionality reduction determined on `s2` is the same as in `s1`.

**Get data**

Let us get some simulated data.
We generate random data and labels for session (or subject) 1 and 2.

```julia
Ps1, Ps2, ys1, ys2 = gen2ClassData(10, 30, 40, 60, 80);
```

**Define the pre-conditioning pipeline for s1**

```julia
p = @→ Recenter(; eVar=0.999) → Compress → Shrink(Fisher; radius=0.02)
```

**Fit an MDM model on s1 using the pipeline**

```julia
m = fit(MDM(), Ps1, ys1; pipeline = p)
```

The fitted pipeline with all learnt parameters is stored in model `m`.
Instead of transforming the data in `s2` using this pipeline,
which is the default behavior of the `predict` function,
let us define the same pipeline with a dimensionality 
reduction parameter fixed as it has been learnt on `s1`. 
This way this paramater cannot change and the transformed matrices
in `s1` and `s2` will have equal size.
That is, we allow adaptation of all parameters, but force the same dimension.

```julia
p = @→ Recenter(; eVar=dim(m.pipeline)) → Compress → Shrink(Fisher; radius=0.02)
```

**Fit the pipeline to s2 and predict**

```julia
predict(m, Ps2, :l; pipeline=p)
```

## Examples using the ENLR model

The **elastic net logistic regression (ENLR)** classifier is an example of classifier acting on the tangent space. Besides the **metric** used to compute a base-point for projecting the data onto the tangent space, it has a parameter **alpha** and an hyperparameter **lambda**. The **alpha** parameter allows to trade off between a pure **ridge** LR model (``α=0``) and a pure **lasso** LR model (``α=1``), which is the default. Given an alpha value, the model is fitted with a number of values for the ``λ`` (regularization) hyperparameter. Thus, differently from the previous example, tuning the ``λ`` hyperparameter is necessary.

Also, keep in mind
that the [`fit`](@ref) and [`predict`](@ref) methods for ENLR models accept optional keyword arguments that are specific to this model.

**Get data**

Let us get some simulated data (see the previous example for explanations).

```julia
using PosDefManifoldML, PosDefManifold

PTr, PTe, yTr, yTe=gen2ClassData(10, 30, 40, 60, 80, 0.1);
```

### ENLR train-test

**Craete and fit ENLR models**

By default, the Fisher metric ic adopted and a lasso model is fitted. The best value for the lambda hyperparameter is found by cross-validation:

```julia
m1 = fit(ENLR(), PTr, yTr; w=:balanced)
```

Notice that since the class are unbalanced, with the `w=:balanced`
argument (we may as well just use `w=:b`) we have requested to compute a balanced mean for projecting the matrices in `PTr` onto the tangent space.

The optimal value of lambda for this training data is

```julia
m1.best.lambda
```

As in *GLMNet.jl*, the intercept and beta terms are retrived by

```julia
m1.best.a0
m1.best.betas
```

The number of non-zero beta coefficients can be found, for example, by

```julia
length(unique(m1.best.betas))-1
```

In order to fit a ridge LR model:

```julia
m2 = fit(ENLR(), PTr, yTr; w=:b, alpha=0)
```

Values of `alpha` in range ``(0, 1)`` fit instead an elastic net LR model. In the following we also request not to normalize predictors (by default they norm is fixed):

```julia
m3 = fit(ENLR(Fisher), PTr, yTr; w=:b, alpha=0.9, normalize=nothing)
```

Instead we could standardize predictors:

```julia
m4 = fit(ENLR(Fisher), PTr, yTr; w=:b, alpha=0.9, normalize=standardize!)
```

or rescale them within custom limits:

```julia
m5 = fit(ENLR(Fisher), PTr, yTr; w=:b, alpha=0.9, normalize=(-1.0, 1.0))
```

In order to find the regularization path we use the
`fitType` keyword argument:

```julia
m1 = fit(ENLR(Fisher), PTr, yTr; w=:b, fitType=:path)
```

The values of lambda along the path are given by

```julia
m1.path.lambda
```

We can also find the best value of the lambda hyperparameter and the regularization path at once, calling:

```julia
m1 = fit(ENLR(Fisher), PTr, yTr; w=:b, fitType=:all)
```

For changing the metric see [MDM train-test](@ref).

See the documentation of the [`fit`](@ref) ENLR method for
details on all available optional arguments.


**Classify data (predict)**

For prediction, we can request to use the best model (optimal lambda), to use a specific model of the regularization path or to use all the models in the regalurization path.
Note that with the last call we have done here above both the `.best` and `.path` field of the `m1` structure have been created.

By default, prediction is obtained from the best model
and we request to predict the labels:

```julia
yPred=predict(m1, PTe)

# prediction accuracy (in proportion)
predictAcc(yPred, yTe)

# confusion matrix
confusionMat(yPred, yTe)

# predict probabilities of matrices in `PTe` to belong to each class
predict(m1, PTe, :p)

# output function of the model for each class
predict(m1, PTe, :f)

```

In order to request the predition of labels for all models
in the regularization path:

```julia
yPred=predict(m1, PTe, :l, :path, 0)
```

while for a specific model in the path (e.g., model #10):

```julia
yPred=predict(m1, PTe, :l, :path, 10)
```

### ENLR cross-validation

The balanced accuracy estimated by a *k-fold cross-validation* is obtained with the exact same basic syntax for all models, with some specific optional keyword arguments for models acting in the tangent space, for example:

```julia
cv = crval(ENLR(), PTr, yTr; w=:b)
```

In order to perform another cross-validation
arranging the training data differently in the folds:

```julia
cv = crval(ENLR(), PTr, yTr; w=:b, shuffle=true)
```

This last command can be invoked repeatedly.

### ENLR adaptation

First, let's see how to adapt the base point for projecting the data onto
the tangent space. Suppose you have data from two sessions or two subjects, `s1` and `s2`.
We want to use `s1` to train a machine learning model on the tangent space and `s2` to test it,
however, the barycenter `s1` cannot be assumed equal to the barycenter of `s2`.
The barycenter determines the base point, therefore, we adapt it.

**Get data**

Let us get some simulated data.
We generate random data and labels for session (or subject) 1 and 2.

```julia
Ps1, Ps2, ys1, ys2 = gen2ClassData(10, 30, 40, 60, 80);
```

**Craete and fit an ENLR model on s1**

```julia
m = fit(ENLR(Fisher), Ps1, ys1)
```

**Classify (predict) data of s2 adapting the base point**

```julia
predict(m, PTe, :l; meanISR=invsqrt(mean(Fisher, PTe)))
```

Second, let's see how to adapt a pre-conditioning pipeline like we have done 
here above for the base point. Since the pipeline we will employ recenter the data 
around the identity, we can skip altogether the computation of the barycenter 
for `s2`, using the identity matrix as the base point.

The pipeline we will define comprises a recentering pre-conditioner 
with dimensionality reduction. While adapting the pipeline to `s2`, 
we need to make sure that the matrices in `s2` are reduced to the same 
dimension as the matrices in `s1`, otherwise the 
machine learning model we fit on `s1` cannot operate 
on `s2`. For this, we need to set the `eVar` argmument of the [`Recenter`](@ref)
pre-conditioner to a integer matching the reduced dimension of `s1`.
Note that the adaptation may not work well if the class proportions
is different in `s1` and `s2`.

**Define the pre-conditioning pipeline for s1**

```julia
p = @→ Recenter(; eVar=0.999) → Compress → Shrink(Fisher; radius=0.02)
```

**Fit the model on s1 using the pipeline**

```julia
m = fit(ENLR(), Ps1, ys1; pipeline = p)
```

**Define the same pipeline with fixed dimensionality reduction parameter**

```julia
p = @→ Recenter(; eVar=dim(m.pipeline)) → Compress → Shrink(Fisher; radius=0.02)
```

**Fit the pipeline to s2 (adapt) and use the identity matrix as base point**

```julia
predict(m, Ps2, :l; pipeline=p, meanISR=I) 
```

## Examples using SVM models

The SVM ML model actually encapsulates several **support-vector classification** and **support-vector regression** models.
Here we are concerned with the former, which include the **C-Support Vector Classification (SVC)**, the **Nu-Support Vector Classification (NuSVC)**, similar to SVC but using a parameter to control the number of support vectors, and the **One-Class SVM (OneClassSVM)**, which is used in general for unsupervised outlier detection. They all act in the tangent space like ENLR models.
Besides the **metric** (see [MDM train-test](@ref)) used to compute a base-point for projecting the data onto the tangent space and the type of SVM model (the **svmType**, = `SVC` (default), `NuSVC` or `OneClassSVM`), the main parameter is the **kernel**.
Avaiable kernels are:
- `Linear`      (default)
- `RadialBasis` 
- `Polynomial`
- `Sigmoid`

Several parameters are available for building all these kernels besides the linear one, which has no parameter. Like for ENLR, for SVM models also an hyperparameter is to be found by cross-validation.

**Get data**

Let us get some simulated data as in the previous examples.

```julia
using PosDefManifoldML, PosDefManifold

PTr, PTe, yTr, yTe=gen2ClassData(10, 30, 40, 60, 80, 0.1);
```

### SVM train-test

**Craete and fit SVM models**

By default, a C-Support Vector Classification model is fitted:

```julia
m1 = fit(SVM(), PTr, yTr; w=:b)
```

Notice that, as for the example above with for ENLR model, we have requested to compute a balanced mean for projecting the matrices in `PTr` onto the tangent space.

In order to fit a Nu-Support Vector Classification model:

```julia
m2 = fit(SVM(), PTr, yTr; w=:b, svmType=NuSVC)
```

For using other kernels, e.g.:

```julia
m3 = fit(SVM(), PTr, yTr; w=:b, svmType=NuSVC, kernel=Polynomial)
```

In the following we request not to normalize predictors (by default they norm is fixed):

```julia
m4 = fit(SVM(), PTr, yTr; w=:b, normalize=nothing)
```

Instead we could standardize predictors:

```julia
m5 = fit(SVM(), PTr, yTr; w=:b, normalize=standardize!)
```

or rescale them within custom limits:

```julia
m6 = fit(SVM(), PTr, yTr; w=:b, normalize=(-1.0, 1.0))
```

By default the Fisher metric is used. For changing it, see [MDM train-test](@ref).

See the documentation of the [`fit`](@ref) SVM method for
details on all available optional arguments.


**Classify data (predict)**

Just the same as for the other models:

```julia
yPred=predict(m1, PTe)

# prediction accuracy (in proportion)
predictAcc(yPred, yTe)

# confusion matrix
confusionMat(yPred, yTe)

# predict probabilities of matrices in `PTe` to belong to each class
predict(m1, PTe, :p)

# output function of the model for each class
predict(m1, PTe, :f)
```

### SVM cross-validation

Again, the balanced accuracy estimated by a *k-fold cross-validation* is obtained with the exact same basic syntax for all models, with some specific optional keyword arguments for
models acting in the tangent space, for example:

```julia
cv = crval(SVM(), PTr, yTr; w=:b)
```

### SVM adaptation

See the tutirial on [ENLR adaptation](@ref); the code needed
is exactly the same changing the machine learning model from `ENLR`
to `SVM`.