using LIBSVM

mutable struct wrapperSVM <: TSmodel
    internalModel :: LIBSVM.SVM
	metric        :: Metric
    function wrapperSVM(metric :: Metric=Fisher;)
	   println(defaultFont, "constructor wrapperSVM")
	   new(nothing,metric)
    end
end

function fit(model :: wrapperSVM,
               𝐏Tr :: Union{ℍVector, Matrix{Float64}},
               yTr :: IntVector,
           meanISR :: Union{ℍ, Nothing} = nothing,
           verbose :: Bool = true,
		         ⏩ :: Bool = true,
          parallel :: Bool=false)

    println(defaultFont, "Start")
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
			typeof(X)
            ℳ.meanISR = G⁻½
        else
            X=tsMap(ℳ.metric, 𝐏Tr; ⏩=⏩, meanISR=meanISR)
            ℳ.meanISR = meanISR
        end
    else
        X=𝐏Tr
    end

    #convert data to LIBSVM format
	instances = X

    # convert labels to LIBSVM format
    labels = yTr

    model = LIBSVM.svmtrain(instances, labels);

    ℳ.internalModel = model

    verbose && println(defaultFont, "Done in ", now()-⌚,".")
    return ℳ
end

#end #end of module
