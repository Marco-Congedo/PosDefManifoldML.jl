#   Unit "libSVM.jl" of the PosDefManifoldML Package for Julia language
#   v 0.2.1 - last update 18th of October 2019
#
#   MIT License
#   Copyright (c) 2019,
#   Anton Andreev, CNRS, Grenoble, France:

# ? CONTENTS :
#   This unit implements a wrapper to libSVM. It projects data to tangent space
#   and it applies SVM classification using Julia's SVM wrapper.

abstract type SVMmodel<:TSmodel end

"""
**Examples**:
```
# Note: creating models with the default creator is possible,
# but not useful in general.

using PosDefManifoldML

# generate data
PTr, PTe, yTr, yTe=gen2ClassData(10, 30, 40, 60, 80)

# create and train an SVM model with default parameters for tangent space calculation and SVM
model=fit(SVM(), PTr, yTr)

# predict using this model
yPred=predict(model, PTe, :l)

# calculate prediction error
predictErr(yTe, yPred)

You can supply parameters for both tangent space calculaton and SVM:

model=fit(SVM(), PTr, yTr, cost = 1.0)

```
"""

mutable struct SVM <: SVMmodel
    	metric        :: Metric
		svmtype       :: Type
		kernel        :: Kernel.KERNEL
		meanISR
		featDim
		# LIBSVM args
		epsilon
		cost
		gamma
		svmModel #used to store the training model from the SVM library
    function SVM( metric :: Metric=Fisher;
				  svmtype = SVC,
                  kernel  = Kernel.RadialBasis,
				  meanISR = nothing,
				  featDim = nothing,
				  epsilon = nothing,
				  cost    = nothing,
				  gamma   = nothing,
				  svmModel = nothing)
	   	 			 new(metric, svmtype, kernel, meanISR, featDim,
					     epsilon, cost, gamma, svmModel)
    end
end

"""
```
mutable struct SVM <: SVMmodel
    	metric        :: Metric
		svmtype       :: Type
		kernel        :: Kernel.KERNEL
		meanISR
		epsilon
		cost
		gamma
		svmModel
end
```

ATTENTION: This class is not used to set the parameters, just to store them. You need to set the parameterts in the FIT function.
Example:
```
model=fit(SVM(), PTr, yTr, cost = 1.0)
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

`svmtype::Type=SVC`: Type of SVM to train `SVC` (for C-SVM), `NuSVC`
    `OneClassSVM`, `EpsilonSVR` or `NuSVR`. Defaults to `OneClassSVM` if
    `y` is not used

`kernel::Kernels.KERNEL=Kernel.RadialBasis`: Model kernel `Linear`, `polynomial`,
    `RadialBasis`, `Sigmoid` or `Precomputed`

`gamma::Float64=1.0/size(X, 1)` : γ for kernels

`cost::Float64=1.0`: cost parameter C of C-SVC, epsilon-SVR, and nu-SVR

`epsilon::Float64=0.1`: epsilon in loss function of epsilon-SVR
"""


function fit(model  :: SVMmodel,
               𝐏Tr  :: Union{ℍVector, Matrix{Float64}},
               yTr  :: IntVector;
		   # Tnagent space parameters
		   w        :: Union{Symbol, Tuple, Vector} = [],
           meanISR  :: Union{ℍ, Nothing} = nothing,
		   vecRange :: UnitRange = 𝐏Tr isa ℍVector ? (1:size(𝐏Tr[1], 2)) : (1:size(𝐏Tr, 2)),
		   # SVM paramters
		   svmtype  :: Type = SVC,
		   kernel   :: Kernel.KERNEL = Kernel.RadialBasis,
		   epsilon  :: Float64 = 0.001,
		   cost     :: Float64 = 1.0,
		   gamma    :: Float64 = 1/_getDim(𝐏Tr, vecRange),
		   # Generic parametes
           verbose  :: Bool = true,
		         ⏩  :: Bool = true,
          parallel  :: Bool=false)

    #println(defaultFont, "Start")
    ⌚=now() # get the time in ms

    # output model
    ℳ=deepcopy(model)

    # checks
    # 𝐏Tr isa ℍVector ? nObs=length(𝐏Tr) : nObs=size(𝐏Tr, 1)

	# check w argument and get weights
    (w=_getWeights(w, yTr, "fit ("*_modelStr(ℳ)*" model)")) == nothing && return

    # projection onto the tangent space
    if 𝐏Tr isa ℍVector
        verbose && println(greyFont, "Projecting data onto the tangent space...")
        if meanISR==nothing
            (X, G⁻½)=tsMap(ℳ.metric, 𝐏Tr; w=w, vecRange=vecRange, ⏩=⏩)
            ℳ.meanISR = G⁻½
        else
            X=tsMap(ℳ.metric, 𝐏Tr; w=w, vecRange=vecRange, meanISR=meanISR, ⏩=⏩)
            ℳ.meanISR = meanISR
        end
    else
        X=𝐏Tr
    end

	ℳ.svmtype = svmtype
	ℳ.kernel = kernel
    ℳ.gamma = gamma
	ℳ.epsilon = epsilon
	ℳ.cost = cost
	ℳ.featDim = size(X, 2)

	#convert data to LIBSVM format; first dim is features, second dim is observations
	instances = X'

    verbose && println(defaultFont, "Fitting SVM model...")
    model = svmtrain(instances, yTr; svmtype = ℳ.svmtype, kernel = ℳ.kernel, epsilon = ℳ.epsilon, cost=ℳ.cost, gamma = ℳ.gamma);

    ℳ.svmModel = model

    verbose && println(defaultFont, "Done in ", now()-⌚,".")
    return ℳ
end



function predict(model   :: SVMmodel,
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

    #convert data to LIBSVM format first dim is features, second dim is observations
    instances = X'

	(predicted_labels, decision_values) = svmpredict(model.svmModel, instances;)
    🃏 = predicted_labels

    verbose && println(defaultFont, "Done in ", now()-⌚,".")
    verbose && println(titleFont, "\nPredicted ",_what2Str(what),":", defaultFont)
    return 🃏
end



# ++++++++++++++++++++  Show override  +++++++++++++++++++ # (REPL output)
function Base.show(io::IO, ::MIME{Symbol("text/plain")}, M::SVM)
    println(io, titleFont, "\n↯ SVM LIBSVM machine learning model")
    println(io, defaultFont, "  ", _modelStr(M))
    println(io, separatorFont, "⭒  ⭒    ⭒       ⭒          ⭒", defaultFont)
    println(io, "type    : PD Tangent Space model")
    println(io, "features: tangent vectors of length $(M.featDim)")
    println(io, "classes : 2")
    println(io, "fields  : ")
	println(io, separatorFont," .metric      ", defaultFont, string(M.metric))

	if 		M.svmtype==SVC s="SVC"
	elseif  M.svmtype==C-SVM s="C-SVM"
	elseif  M.svmtype==EpsilonSVR s="EpsilonSVR"
	elseif  M.svmtype==OneClassSVM s="OneClassSVM"
	elseif  M.svmtype==NuSVR s="NuSVR"
	else    s = "Warning: the SVM type is unknown"
	end
	println(io, separatorFont," .svmtype     ", defaultFont, "$s")

	println(io, separatorFont," .kernel      ", defaultFont, "$(string(M.kernel))")

	#`kernel::Kernels.KERNEL=Kernel.RadialBasis`: Model kernel `Linear`, `polynomial`,
	#    `RadialBasis`, `Sigmoid` or `Precomputed`

	if M.meanISR == nothing
        println(io, greyFont, " .meanISR      not created")
    else
        n=size(M.meanISR, 1)
        println(io, separatorFont," .meanISR     ", defaultFont, "$(n)x$(n) Hermitian matrix")
    end

	M.epsilon==nothing ? println(io, "       not created ") :
						 println(io, separatorFont," .epsilon     ", defaultFont, "$(round(M.epsilon, digits=9))")

    M.cost==nothing ?    println(io, "       not created ") :
						 println(io, separatorFont," .cost        ", defaultFont, "$(round(M.cost, digits=3))")

    M.gamma==nothing ?   println(io, "       not created ") :
 						 println(io, separatorFont," .gamma       ", defaultFont, "$(round(M.gamma, digits=5))")

	println(io, separatorFont," .svmModel ", defaultFont, "   (LIBSVM model)")
end
