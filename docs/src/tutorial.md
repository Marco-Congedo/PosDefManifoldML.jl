# Tutorial

*PosDefManifoldML* mimicks the functioning of [ScikitLearn](https://scikit-learn.org/stable/) (good to know if you are familir with it):
first a **machine learning (ML) model** is created, then data is used to
**fit** (train) the model. Once this is done the model
allows to **predict** the labels of test data or the probability of the data to belong to each class.

In order to compare ML models, a **cross-validation** procedure is
implemented.

## ML models

For the moment being, only the **Riemannian minimum distance to mean** (MDM)
ML model is implemented.
The creation of other models will follow the same rationale and syntax,
along the lines of *ScikitLearn*.

### MDM model

An MDM model is created as

```
model = MDM(metric)
```

where `metric` must be a metric in the manifold of positive definite matrices
allowing the definition of both a distance function and of a mean (center of mass).

Currently supported metrics are:

| Metric (distance) | Resulting mean estimation                     |
|:----------------- |:--------------------------------------------- |
| Euclidean         | Arithmetic                                    |
| invEuclidean      | Harmonic                                      |
| ChoEuclidean      | Cholesky Euclidean                            |
| logEuclidean      | Log-Euclidean                                 |
| logCholesky       | Log-Cholesky                                  |
| Fisher            | Fisher (Cartan, Karcher, Pusz-Woronowicz,...) |
| logdet0           | LogDet (S, α, Bhattacharyya, Jensen,...)      |
| Jeffrey           | Jeffrey (symmetrized Kullback-Leibler)        |
| Wasserstein       | Wasserstein (Bures, Hellinger, ...)           |

Do not use the Von Neumann metric, which is also supported in *PosDefManifold*,
since it does not allow a definition of mean.

You can also create and fit an MDM model in one pass.
See [`MDM`](@ref).

## load data

After a classifier instance is created we need to fit the classifier model with  data. For convenience, let us use the **npz** format(.npz),
which allows sharing data between *Python* and *Julia*.
For example:

```
	using NPZ

	path = "/home/saloni/PosDefManifoldML/src/" # where files are stored
	filename = "subject_1.npz" # for subject number i
	data = npzread(path*filename)
	X = data["data"] # retrive the epochs
	y = data["labels"] # retrive the corresponding labels
```

This corresponds to the training part. The training data should be in the form of :

- `𝐗` :-  Vector of Hermitian Matrices. These Hermitian Matrices are the covariance matrices formed   		    out of the raw data. The raw data of signals first need to be converted into their 		   corresponding covariance matrices. This can be very easily done using the [gram](https://marco-congedo.github.io/PosDefManifold.jl/latest/signalProcessing/#PosDefManifold.gram) function of 		    **PosDefManifold**. This is what is done in the below code.
- `y` :-  Labels corresponding to each training sample. Labels should be integers from 1 to n, where  		   n is the number of classes.


	train_size = size(X,1)
	𝐗 = ℍVector(undef, sam_size)
	@threads for i = 1:train_size
    	 	𝐗[i] = gram(X[i,:,:])
	end

## train a model (fit)

Once you are ready with your data, you can fit your model
calling the `fit!` function.

	fit!(model, 𝐗 ,y)


Note that you can also use the construtor

```model=MDM(Fisher, 𝐗, y)```,

which creates and fits the model at once.

## classify data (predict)

Let 𝐓 be a vector of Hermitian matrices forming the **testing set**.
In order to classify them, i.e., to associate a class label to them,
we invoke

```predict(model, 𝐓, :l)```.

If instead we wish to estimate the probabilities for the matrices in 𝐓
of belonging to each class, we invoke

```predict(model, 𝐓, :p)```.

### cross-validation

In order to assess the performance of a model usually a **cross-validation (CV)**
procedure is adopted. In *PosDefManifoldML* a random CV procedure
is implemented. We invoke it by

```CVscore(model, 𝐓, y, 5)```,

where `5` is the number of CVs. This implies that
at each CV, 1/5th of the matrices is used for training and the
remaining for testing.
