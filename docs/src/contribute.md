# How to Contribute

You can easily contribute a new ML model to *PosDefManifoldML.jl*
package following these steps:

Let's say you want to contribute an ML model named `ABC`.

If the model act on the manifold of positive definite matrices (PSD), use as
template the *mdm.jl* unit. If it acts on the tangent space, use as template
the *svm.jl* unit (you can also check the *enlr.jl* unit).

Save the template unit as unit `abc.jl` in the same directory where the template file is.

Implementing your `ABC` model entails the following five steps:

**1) Declare an abstract type for the model**

if your model acts on the manifold of PSD matrices, this will be

```
abstract type ABCmodel<:PDmodel end
```

if your model act on the tangent space, this will be

```
abstract type ABCmodel<:TSmodel end
```

**2) Declare the struct to hold the model and its default creator**

This will look like:

```
mutable struct ABC <: ABCmodel
		metric        :: Metric
		defaultkwarg1 :: Type_of_defaultkwarg1
		defaultkwarg2 :: Type_of_defaultkwarg2
		...
		voidkwarg1
		voidkwarg2
		...
		function ABC(metric :: Metric=Fisher;
				defaultkwarg1 :: Type_of_defaultkwarg1 = default_value,
				defaultkwarg2 :: Type_of_defaultkwarg2 = default_value,
				...
				voidkwarg1 = nothing,
				voidkwarg2 = nothing,
				...)
			new(metric, defaultkwarg1, defaultkwarg1,...,
				voidkwarg1, voidkwarg2,...)
		end
end
```

In the above example `defaultkwarg` are essential parameters that should be
set by default upon creation. Use as few as those as needed to obtain
a working ML model when the user does not pass any argument.

The `voidkwarg` arguments are arguments that you wish be accessible to the user in the structure once the model has been fitted. Include here as few of them as possible.

**3) write the `fit` function**

This function will fit the model. Its default behavior should fit the model
and tune hyperparameters in order to find the best model by cross-validation
if the `ABC` model has hyperparameters.

Your `fit` function declaration will look like:

```
function fit(   model :: ABCmodel,
			    𝐏Tr :: ℍVector,
				# if the model acts on the tangent space use:
				# 𝐏Tr :: Union{ℍVector, Matrix{Float64}}
				yTr :: Vector;
				verbose :: Bool = true
				kwarg1  :: type-of-kwarg1 = default-value,
				kwarg2  :: type-of-kwarg2 = default-value,
				...)
end
```

Here you can use as many `kwarg` arguments as you wish.
Currently, all ML models have a `verbose` argument.
Your `fit` function should starts with:

```
⌚=now() # time in milliseconds
ℳ=deepcopy(model) # output model
```
and ends with

```
verbose && println(defaultFont, "Done in ", now()-⌚,".")
return ℳ
```

In between these two blocks you will fit the model
and write into `ℳ` the `voidkwarg` arguments you have
declared in the `ABC` struct (see above).

Keep in mind that if the model acts in the tangent space
you will need to project the data therein. For doing so
you can use the `_getFeat_fit!` internal function
(declared in unit *tools.jl*),
as it is done for the *ENLR* and *SVM* models.
This entails using some standard arguments for tangent space projection,
which should be given as options to the user, as done for these models. See how this is done in the `fit` function of the
*ENLR* and *SVM* models (unit `enlr.jl` and `svm.jl`, respectively).

Once you have finished, a call such as

```
m1 = fit(ABC(), PTr, yTr)
```

for some data `PTr`(a matrix or [ℍVector](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#%E2%84%8DVector-type-1) type) and labels `yTr` (see [`fit`](@ref))
should fit the model in such a way that it is ready to allow a call
to the `predict` function and return the model.

**4) write the `predict` function**


Your `predict` function declaration will look like:

```
function predict(model   :: ABCmodel,
				𝐏Te :: Union{ℍVector, Matrix{Float64}},
				# if the model acts on the tangent space use:
				# 𝐏Tr :: Union{ℍVector, Matrix{Float64}}
				what :: Symbol = :labels;
				kwarg1 :: type_of_kwarg1 = default_value
				kwarg2 :: type_of_kwarg2 = default_value
				...
				verbose  :: Bool = true)
```

where in general here you will not need `kwarg` arguments.

Your `predict` function should starts with:

```
⌚=now() # time in milliseconds
```

and ends with

```
verbose && println(defaultFont, "Done in ", now()-⌚,".")
verbose && println(titleFont, "\nPredicted ",_what2Str(what),":", defaultFont)
return 🃏
```

where in between these two blocks variable `🃏` has been filled with the prediction.

To use this template code,
in the declaration of the `_what2Str` function (declared in unit *tools.jl*),
add a line to allow returning the full name of your model as a string.

As for the `fit` function, if the model acts in the tangent space
you will need to project the data therein. For doing so
you can use here the `_getFeat_Predict!` internal function
(declared in unit *tools.jl*),
as it is done for the *ENLR* and *SVM* models.
This entails using some standard arguments for tangent space projection,
which should be given as options to the user, as done for these models.
Note that `_getFeat_Predict!` is similar, but not the same as
`_getFeat_fit!` function.

All ML models implemented so far allow three types of prediction,
depending on the symbol passed by the user with argument `what`.
See the documentation of the [`predict`](@ref) function for the
*ENLR* model to see what these three types of predictions are
and the code of the function in unit *enlr.jl* for an example on
how to compute them. Note that the returned type of the `predict` function,
the variable `🃏`, depends on `what` is predicted.


**5) Allow the `cvAcc` function to support your model properly**

If you have been following these guidelines so far, the [`cvAcc`](@ref) function
(declared in unit *cv.jl*) will be able to perform k-fold cross-validation
on data using your `ABC` model.

This function allows the user to pass an arbitrary number of optional
keyword arguments, so the user will be able to pass here the
`kwarg` arguments you have declared in your `fit` function for the `ABC` model
and those will be passed to the `fit` function when fitting the model
at each fold.

This will be done automatically, however it is necessary to
prevent the user from passing here optional keyword arguments
that should not be used in a k-fold cross-validation setting
(if in your `fit` function you have declared such arguments).

Also, it is necessary here to overwrite into the `ABC` model
that is passed by the user as argument to the [`cvAcc`](@ref) function
the `defaultkwarg1` fields of the `ABC` struct you have declared,
if the user can request different values for those fields
using optional keyword arguments passed to the
[`cvAcc`](@ref) function. For example, consider the *SVM*
model; suppose the user pass to the [`cvAcc`](@ref) function
a default SVM model. For such a model the kernel is a radial basis
kernel. Suppose the user has passed to the [`cvAcc`](@ref) function
argument `kernel=linear`. If you don't overwrite the kernel into the
model, the radianl basis kernel will be used instead of the desired
linear kernel.

You can do so easily using the
`_rmArgs` and `_getArgValue` internal functions
(declared in unit *tools.jl*), as done for the *SVM* and *ENLR* model
(see the code of the [`cvAcc`](@ref) function).
