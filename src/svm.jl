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

# create and train an SVM model
model=fit(svm(), PTr, yTr)

# predict using this model
yPred=predict(model, PTe, :l)

# calculate prediction error
predictErr(yTe, yPred)
```
"""

mutable struct svm <: TSmodel
    	metric        :: Metric
		internalModel
		meanISR
		svmtype       ::Type
		kernel        ::Kernel.KERNEL
		epsilon       ::Float64
		cost          ::Float64
		gamma         ::Float64
    function svm(	    metric :: Metric = Fisher,
		         		   internalModel = nothing,
				                 meanISR = nothing,
								 svmtype = LIBSVM.SVC,
                           		  kernel = Kernel.RadialBasis,
								 epsilon = 0.1,
								    cost = 1.0,
								   gamma = -1
								 )
	   	 			 new(metric, internalModel, meanISR,
                         svmtype, kernel, epsilon, cost, gamma) #can not set a default value here
    end
end

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
