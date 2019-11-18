mutable struct wrapperSVM <: TSmodel
    	metric        :: Metric
		internalModel
		meanISR
    function wrapperSVM(metric :: Metric = Fisher;
		         		   internalModel = nothing,
				                 meanISR = nothing)
	   				 println(defaultFont, "constructor wrapperSVM")
	   	 			 new(metric,internalModel,meanISR)
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
			println(defaultFont, "meanISR is nothing")
            (X, G⁻½)=tsMap(ℳ.metric, 𝐏Tr; ⏩=⏩)
            ℳ.meanISR = G⁻½
        else
			println(defaultFont, "meanISR is NOT nothing")
            X=tsMap(ℳ.metric, 𝐏Tr; ⏩=⏩, meanISR=meanISR)
            ℳ.meanISR = meanISR
        end
    else
        X=𝐏Tr
    end

    println(defaultFont, "Converting")
    #convert data to LIBSVM format
	#first dimension is features
	#second dimension is observations
	instances = X'

    println(defaultFont, "Calculating")
    model = LIBSVM.svmtrain(instances, yTr);

    ℳ.internalModel = model

    verbose && println(defaultFont, "Done in ", now()-⌚,".")
    return ℳ
end

function predict(model   :: wrapperSVM,
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

    (predicted_labels, decision_values) = svmpredict(model.internalModel, instances);
    🃏 = predicted_labels

    verbose && println(defaultFont, "Done in ", now()-⌚,".")
    verbose && println(titleFont, "\nPredicted ",_what2Str(what),":", defaultFont)
    return 🃏
end
