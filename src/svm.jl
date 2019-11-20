#   Unit "libSVM.jl" of the PosDefManifoldML Package for Julia language
#   v 0.2.1 - last update 18th of October 2019
#
#   MIT License
#   Copyright (c) 2019,
#   Anton Andreev, CNRS, Grenoble, France:

# ? CONTENTS :
#   This unit implements a wrapper to libSVM. It projects data to tangent space
#   and it applies SVM classification using Julia's SVM wrapper.

"""
**Examples**:
```
# Note: creating models with the default creator is possible,
# but not useful in general.

using PosDefManifoldML

# generate data
PTr, PTe, yTr, yTe=gen2ClassData(10, 30, 40, 60, 80)

# create and train an SVM model with default parameters for tangent space calculation and SVM
model=fit(svm(), PTr, yTr)

# predict using this model
yPred=predict(model, PTe, :l)

# calculate prediction error
predictErr(yTe, yPred)

You can supply parameters for both tangent space calculaton and SVM:

s = svm(Fisher, nothing, nothing, LIBSVM.SVC, Kernel.RadialBasis, 0.1, 1.0, -1)

model=fit(s, PTr, yTr)

```
"""

mutable struct svm <: TSmodel
    	metric        :: Metric
		internalModel #used to store the training model from the SVM library
		meanISR       :: Union{ℍVector, Nothing}
		svmtype       :: Type
		kernel        :: Kernel.KERNEL
		epsilon       :: Float64
		cost          :: Float64
		gamma         :: Float64
    function svm(	              metric = Fisher,
		         		   internalModel = nothing,
				                 meanISR = nothing,
								 svmtype = LIBSVM.SVC,
                           		  kernel = Kernel.RadialBasis,
								 epsilon = 0.1,
								    cost = 1.0,
								   gamma = -1 #can not set a default value herem because it requires the number of observations
								 )
	   	 			 new(metric, internalModel, meanISR,
                         svmtype, kernel, epsilon, cost, gamma)
    end
end

"""
```
mutable struct svm <: TSmodel
    	metric        :: Metric
		internalModel
		meanISR

		svmtype       ::Type
		kernel        ::Kernel.KERNEL
		epsilon       ::Float64
		cost          ::Float64
		gamma         ::Float64
end
```

`.metric`, of type
[Metric](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#Metric::Enumerated-type-1),
is the metric that will be adopted to compute the mean used
as base-point for tangent space projection. By default the
Fisher metric is adopted. See [mdm.jl](@ref)
for the available metrics. If the data used to train the model
are not positive definite matrices, but Euclidean feature vectors,
the `.metric` field has no use.

`.meanISR` is optionally passed to the [`fit`](@ref)
function. By default it is computed thereby.
If the data used to train the model
are not positive definite matrices, but Euclidean feature vectors,
the `.meanISR` field has no use and is set to `nothing`.

The following are parameters that are passed to the LIBSVM package:

`svmtype::Type=LIBSVM.SVC`: Type of SVM to train `SVC` (for C-SVM), `NuSVC`
    `OneClassSVM`, `EpsilonSVR` or `NuSVR`. Defaults to `OneClassSVM` if
    `y` is not used

`kernel::Kernels.KERNEL=Kernel.RadialBasis`: Model kernel `Linear`, `polynomial`,
    `RadialBasis`, `Sigmoid` or `Precomputed`

`gamma::Float64=1.0/size(X, 1)` : γ for kernels

`cost::Float64=1.0`: cost parameter C of C-SVC, epsilon-SVR, and nu-SVR

`epsilon::Float64=0.1`: epsilon in loss function of epsilon-SVR
"""


function fit(model :: svm,
               𝐏Tr :: Union{ℍVector, Matrix{Float64}},
               yTr :: IntVector,
           meanISR :: Union{ℍ, Nothing} = nothing,
           verbose :: Bool = true,
		         ⏩ :: Bool = true,
          parallel :: Bool=false)

    #println(defaultFont, "Start")
    ⌚=now() # get the time in ms

    # output model
    ℳ=deepcopy(model)

    # checks
    𝐏Tr isa ℍVector ? nObs=length(𝐏Tr) : nObs=size(𝐏Tr, 1)

    # projection onto the tangent space
    if 𝐏Tr isa ℍVector
        verbose && println(greyFont, "Projecting data onto the tangent space...")
        if meanISR==nothing
            (X, G⁻½)=tsMap(ℳ.metric, 𝐏Tr; ⏩=⏩)
            ℳ.meanISR = G⁻½
        else
            X=tsMap(ℳ.metric, 𝐏Tr; ⏩=⏩, meanISR=meanISR)
            ℳ.meanISR = meanISR
        end
    else
        X=𝐏Tr
    end

    nFeatures = size(X,2)

    if ℳ.gamma == -1
		ℳ.gamma = 1 / nFeatures
	end

	verbose && println(defaultFont, "nFeatures: " * string(nFeatures))
	verbose && println(defaultFont, "nObservations: " * string(nObs))
	verbose && println(defaultFont, "gamma: " * string(ℳ.gamma))
	verbose && println(defaultFont, "epsilon: " * string(ℳ.epsilon))

	#convert data to LIBSVM format
	#first dimension is features
	#second dimension is observations
	instances = X'

    verbose && println(defaultFont, "Calculating")
    model = LIBSVM.svmtrain(instances, yTr; svmtype = ℳ.svmtype, kernel = ℳ.kernel, epsilon = ℳ.epsilon, cost=ℳ.cost, gamma = ℳ.gamma);

    ℳ.internalModel = model

    verbose && println(defaultFont, "Done in ", now()-⌚,".")
    return ℳ
end

function predict(model   :: svm,
                 𝐏Te     :: Union{ℍVector, Matrix{Float64}},
                 what    :: Symbol = :labels,
                vecRange :: UnitRange = 𝐏Te isa ℍVector ? (1:size(𝐏Te[1], 2)) : (1:size(𝐏Te, 2)),
                 checks  :: Bool = true,
                 verbose :: Bool = true,
                  ⏩     :: Bool = true)

    ⌚=now()

    # checks
    if checks
        if !_whatIsValid(what, "predict ("*_modelStr(model)*")") return end
    end

    # projection onto the tangent space
    if 𝐏Te isa ℍVector
        verbose && println(greyFont, "Projecting data onto the tangent space...")
        X=tsMap(model.metric, 𝐏Te; meanISR=model.meanISR, ⏩=⏩, vecRange=vecRange)
    else X=𝐏Te[:, vecRange] end

    #convert data to LIBSVM format
    #first dimension is features
    #second dimension is observations
    instances = X'

    #(predicted_labels, decision_values) = svmpredict(model.internalModel, instances);
	(predicted_labels, decision_values) = svmpredict(model.internalModel, instances;)
    🃏 = predicted_labels

    verbose && println(defaultFont, "Done in ", now()-⌚,".")
    verbose && println(titleFont, "\nPredicted ",_what2Str(what),":", defaultFont)
    return 🃏
end
