using LIBSVM
using PosDefManifoldML

mutable struct libSVM <: TSmodel
    meanISR
    # TODO: SVM parameters

    internalModel :: LIBSVM.SVM
    function ENLR(
               meanISR     = nothing
        new(meanISR)
    end
end

function fit(model :: libSVM,
               𝐏Tr :: Union{ℍVector, Matrix{Float64}},
               yTr :: IntVector,
           meanISR :: Union{ℍ, Nothing} = nothing,
           verbose :: Bool = true,
          parallel :: Bool=false)

    ⌚=now() # get the time in ms

    # output model
    ℳ=deepcopy(model)

    # checks
    𝐏Tr isa ℍVector ? nObs=length(𝐏Tr) : nObs=size(𝐏Tr, 1)

    # projection onto the tangent space
    if 𝐏Tr isa ℍVector
        verbose && println(greyFont, "Projecting data onto the tangent space...")
        if meanISR==nothing
            (X, G⁻½)=tsMap(ℳ.metric, 𝐏Tr; w=w, ⏩=⏩)
            ℳ.meanISR = G⁻½
        else
            X=tsMap(ℳ.metric, 𝐏Tr; w=w, ⏩=⏩, meanISR=meanISR)
            ℳ.meanISR = meanISR
        end
    else X=𝐏Tr
    end

    #convert data to LIBSVM format

    # convert labels to LIBSVM format
    #y = convert(Matrix{Float64}, [(yTr.==1) (yTr.==2)])

    #model = svmtrain(instances[:, 1:2:end], labels[1:2:end]);

    #ℳ.internalModel = model

    verbose && println(defaultFont, "Done in ", now()-⌚,".")
    return ℳ
end
