# Tutorial

*PosDefManifoldML* mimicks the functioning of [ScikitLearn](https://scikit-learn.org/stable/):
first a **machine learning (ML) model** is created, then data is used to
**fit** (train) the model. The above two steps can actually be carried out at one. Once this is done the model
allows to **predict** the labels of test data or the probability of the data to belong to each class.

In order to compare ML models, a **k-fold cross-validation** procedure is implemented.

## ML models

For the moment being, only the **Riemannian minimum distance to mean** (MDM) ML model is implemented. See Barachat el *al.* (2012) and Congedo et *al.* (2017a) [🎓](@ref).

### MDM model

An MDM model is created and fit with trainng data such as

```
model=MDM(Fisher, 𝐏Tr, yTr)
```

where `metric` is of the [Metric](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#Metric::Enumerated-type-1)
enumerated type declared in [PosDefManifold](https://marco-congedo.github.io/PosDefManifold.jl/dev/). This is a metric
in the manifold of positive definite matrices allowing the definition of both a *distance function* and a *mean* (center of mass).

Currently supported metrics are:

| metric (distance) | resulting mean estimation                     |
|:----------------- |:--------------------------------------------- |
| Euclidean         | arithmetic                                    |
| invEuclidean      | harmonic                                      |
| ChoEuclidean      | Cholesky Euclidean                            |
| logEuclidean      | log-Euclidean                                 |
| logCholesky       | log-Cholesky                                  |
| Fisher            | Fisher (Cartan, Karcher, Pusz-Woronowicz,...) |
| logdet0           | logDet (S, α, Bhattacharyya, Jensen,...)      |
| Jeffrey           | Jeffrey (symmetrized Kullback-Leibler)        |
| Wasserstein       | Wasserstein (Bures, Hellinger, ...)           |

Do not use the Von Neumann metric, which is also supported in *PosDefManifold*,
since it does not allow a definition of mean.

## use data

A real data example will be added soon.

Now let us create some simulated data for a **2-class example**.
First, let us create symmetric positive definite matrices (real positive definite matrices):

```
using PosDefManifoldML

𝐏Tr, 𝐏Te, yTr, yTe=gen2ClassData(10, 30, 40, 60, 80)
```

-`𝐏Tr` is the simulated training set, holding 30 matrices for class 1 and 40 matrices for class 2
- `𝐏Te` is the testing set, holding 60 matrices for class 1 and 80 matrices for class 2.
- `yTr` is a vector of 70 labels for 𝐓r
- `yTe` is a vector of 140 labels for 𝐓e

All matrices are of size 10x10.

## craete and fit an MDM model
```
model=MDM(Fisher, 𝐏Tr, yTr)
```

## classify data (predict)

```
predict(model, 𝐏Te, :l)
```

If instead we wish to estimate the probabilities for the matrices in `𝐏Te` of belonging to each class, we invoke

```
predict(model, 𝐏Te, :p)
```

## cross-validation

A *k-fold cross-validation* is obtained as

```
CVscore(model, 𝐏Te, y, 5)
```

where `5` is the number of folds. This implies that
at each CV, 1/5th of the matrices is used for training and the
remaining for testing.
