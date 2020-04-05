#   Unit "cv.jl" of the PosDefManifoldML Package for Julia language

#   MIT License
#   Copyright (c) 2019-2020
#   Marco Congedo, CNRS, Grenoble, France:
#   https://sites.google.com/site/marcocongedo/home
#   Saloni Jain, Indian Institute of Technology, Kharagpur, India

# ? CONTENTS :
#   This unit implements cross-validation procedures for estimating
#   accuracy of all machine learning models.

"""
```
struct CVacc
    cvType    :: String
    scoring   :: Union{String, Nothing}
    modelType :: Union{String, Nothing}
    cnfs      :: Union{Vector{Matrix{T}}, Nothing} where T<:Real
    avgCnf    :: Union{Matrix{T}, Nothing} where T<:Real
    accs      :: Union{Vector{T}, Nothing} where T<:Real
    avgAcc    :: Union{Real, Nothing}
    stdAcc    :: Union{Real, Nothing}
end
```

A call to [`cvAcc`](@ref) results in an instance of this structure.
Fields:

`.cvTpe` is the type of cross-validation technique, given as a string
(e.g., "10-kfold")

`.scoring` is the type of accuracy that is computed, given as a string.
This has been passed as argument to [`cvAcc`](@ref).
Currently *accuracy* and *balanced accuracy* are supported.

`.modelType` is type of the machine learning used for performing the
cross-validation, given as a string.

`.cnfs` is a vector of matrices holding the *confusion matrices*
obtained at each fold of the cross-validation.

`.avgCnf` is the *average confusion matrix* across the folds of the
cross-validation.

`.accs` is a vector of real numbers holding the *accuracies* obtained
at each fold of the cross-validation.

`.avgAcc` is the *average accuracy* across the folds of the
cross-validation.

`.stdAcc` is the *standard deviation of the accuracy* across the folds of the
cross-validation.

"""
struct CVacc
    cvType    :: String
    scoring   :: Union{String, Nothing}
    modelType :: Union{String, Nothing}
    cnfs      :: Union{Vector{Matrix{T}}, Nothing} where T<:Real
    avgCnf    :: Union{Matrix{T}, Nothing} where T<:Real
    accs      :: Union{Vector{T}, Nothing} where T<:Real
    avgAcc    :: Union{Real, Nothing}
    stdAcc    :: Union{Real, Nothing}
end

"""
```
CVacc(s::String) =
     CVacc(s, nothing, nothing, nothing,
           nothing, nothing, nothing, nothing)
```

Construct an instance of the CVacc structure giving only the `.cvtype`
field. All other fields are filled with `nothing`. This is useful to construct
manually cvAcc objects.
"""
CVacc(s::String)=CVacc(s, nothing, nothing, nothing, nothing, nothing, nothing, nothing)


"""
```
function cvAcc(model    :: MLmodel,
               𝐏Tr     :: ℍVector,
               yTr      :: IntVector;
           nFolds       :: Int      = min(10, length(yTr)÷3),
           scoring      :: Symbol   = :b,
           shuffle      :: Bool     = false,
           verbose      :: Bool     = true,
           outModels    :: Bool     = false,
           ⏩           :: Bool     = true,
           fitArgs...)
```
Cross-validation accuracy for a machine learning `model`:
given an ℍVector `𝐏Tr` holding ``k`` Hermitian matrices,
an [IntVector](@ref) `yTr` holding the ``k`` labels for these matrices and
the number of folds `nFolds`,
return a [`CVacc`](@ref) structure.

**optional keyword arguments**

`nFolds` by default is set to the minimum between 10 and the number
of observation ÷ 3 (integer division).

If `scoring`=:b (default) the **balanced accuracy** is computed.
Any other value will make the function returning the regular **accuracy**.
Balanced accuracy is to be preferred for unbalanced classes.
For balanced classes the balanced accuracy reduces to the
regular accuracy, therefore there is no point in using regular accuracy
if not to avoid a few unnecessary computations when the class are balanced.

For the meaning of the `shuffle` argument (false by default),
see function [`cvSetup`](@ref), to which this argument is passed.

If `verbose` is true (default), information is printed in the REPL.
This option is included to allow repeated calls to this function
without crowding the REPL.

If `outModels` is true return a 2-tuple holding a [`CVacc`](@ref) structure
and a `nFolds`-vector of the model fitted for each fold,
otherwise (default), return only a [`CVacc`](@ref) structure.

If `⏩` the folds and some other computations are multi-threaded.
It is true by default. Set it to false if there are problems in running
this function.

`fitArgs` are optional keyword arguments that are passed to the
[`fit`](@ref) function called for each fold of the cross-validation.
For each machine learning model, all optional keyword arguments of
their fit method are elegible to be passed here, however,
the arguments listed in the following table for each model should not be passed.
Note that if they are passed, they will be disabled:

| MDM/MDMF |   ENLR    |    SVM    |
|:--------:|:---------:|:---------:|
| `verbose`| `verbose` | `verbose` |
|  `⏩`   | `⏩`      | `⏩`      |
|          | `meanISR` | `meanISR` |
|          | `meanInit`| `meanInit`|
|          | `fitType` |
|          | `offsets` |
|          | `lambda`  |
|          | `folds`   |

Also, if you pass a `w` (weights for tangent space projection) argument,
do not pass a vector of weights, just pass a symbol, e.g., `w=:b`
for balancing weights.

**See**: [notation & nomenclature](@ref), [the ℍVector type](@ref).

**See also**: [`fit`](@ref), [`predict`](@ref).

**Examples**
```
using PosDefManifoldML, PosDefManifold

# generate some data
PTr, PTe, yTr, yTe=gen2ClassData(10, 30, 40, 60, 80)

# perform 10-fold cross-validation using the minimum distance to mean classifier
cv=cvAcc(MDM(Fisher), PTr, yTr)

# ...using the lasso logistic regression classifier
cv=cvAcc(ENLR(Fisher), PTr, yTr)

# ...using the support-vector machine classifier
cv=cvAcc(SVM(Fisher), PTr, yTr)

# ...With a Polynomial kernel of order 3 (default)
cv=cvAcc(SVM(Fisher), PTr, yTr; kernel=kernel.Polynomial)

# perform 8-fold cross-validation instead
# (and see that you can go pretty fast if your PC has 8 threads)
cv=cvAcc(SVM(Fisher), PTr, yTr; nFolds=8)

# ...balance the weights for tangent space projection
cv=cvAcc(ENLR(Fisher), PTr, yTr; nFolds=8, w=:b)

# perform another cross-validation shuffling the folds
cv=cvAcc(ENLR(Fisher), PTr, yTr; shuffle=true, nFolds=8, w=:b)

```
"""
function cvAcc(model    :: MLmodel,
               𝐏Tr     :: ℍVector,
               yTr      :: IntVector;
           nFolds       :: Int      = min(10, length(yTr)÷3),
           scoring      :: Symbol   = :b,
           shuffle      :: Bool     = false,
           verbose      :: Bool     = true,
           outModels    :: Bool     = false,
           ⏩           :: Bool     = true,
           fitArgs...)

    ⌚ = now()
    verbose && println(greyFont, "\nPerforming $(nFolds)-fold cross-validation...")

    z  = length(unique(yTr))            # number of classes
    𝐐  = [ℍ[] for i=1:z]               # data arranged by class
    for j=1:length(𝐏Tr) push!(𝐐[yTr[j]], 𝐏Tr[j]) end

    # pre-allocated memory
    𝐐Tr = [ℍ[] for f=1:nFolds]                 # training data in 1 vector per folds
    zTr = [Int64[] for f=1:nFolds]              # training labels in 1 vector per fold
    𝐐Te = [[ℍ[] for i=1:z] for f=1:nFolds]     # testing data arranged by classes per fold
    CM  = [zeros(Float64, z, z) for f=1:nFolds] # confusion matrices per fold
    s   = Vector{Float64}(undef, nFolds)        # accuracy scores per fold
    pl  = [[Int[] for i=1:z] for f=1:nFolds]    # predicted labels per fold
    indTr = [[[]] for i=1:z]                    # indeces for training sets per fold
    indTe = [[[]] for i=1:z]                    # indeces for test sets per fold
    ℳ=Vector{MLmodel}(undef, nFolds)            # ML models

    # get indeces for all CVs (separated for each class)
    if ⏩
       @threads for i=1:z indTr[i], indTe[i] = cvSetup(length(𝐐[i]), nFolds; shuffle=shuffle) end
    else
        for i=1:z indTr[i], indTe[i] = cvSetup(length(𝐐[i]), nFolds; shuffle=shuffle) end
    end

    fitArgs✔=()
    # make sure the user doesn't pass arguments that skrew up the cv
    if model isa ENLRmodel
        fitArgs✔=_rmArgs((:meanISR, :meanInit, :fitType, :verbose, :⏩,
                         :offsets, :lambda, :folds); fitArgs...)
        # overwrite the `alpha` value in `model` if the user has passed keyword `alpha`
        if (a=_getArgValue(:alpha; fitArgs...)) ≠ nothing model.alpha=a end
    end

    if model isa SVMmodel
        # make sure the user doesn't pass arguments that skrew up the cv
        fitArgs✔=_rmArgs((:meanISR, :meanInit, :verbose, :⏩); fitArgs...)
        # overwrite the `svmType` and `kernel` values in `model` if the user has passed then as kwargs
        if (a=_getArgValue(:svmType; fitArgs...)) ≠ nothing model.svmType=a end
        if (a=_getArgValue(:kernel;  fitArgs...)) ≠ nothing model.kernel =a end
    end

    # get means initializations. Be careful here with the behavior for other models
    # NB for the MDM model the initialization is a vector of means for each class,
    # for the TSmodels it is the mean of such means.
    # This is a quick approximation since the initialization is not critical,
    # but it hastens the computation time since itera. alg. require less iters.
    #=
    if      model.metric in (Fisher, logdet0)
                M0=means(logEuclidean, 𝐐; ⏩=⏩)
                if model isa TSmodel M0=mean(logEuclidean, M0; ⏩=⏩) end
    elseif  model.metric == Wasserstein
                M0=ℍVector([generalizedMean(𝐐[i], 0.5; ⏩=⏩) for i=1:length(𝐐)])
                if model isa Tsmodel M0=generalizedMean(M0, 0.5; ⏩=⏩) end
    else    M0=nothing;
    end
    =#

    # perform cv
    function fold(f::Int)
        @static if VERSION >= v"1.3" print(defaultFont, rand(dice), " ") end # print a random dice in the REPL

        # get testing data for current fold
        for i=1:z 𝐐Te[f][i] = [𝐐[i][j] for j ∈ indTe[i][f]] end

        # get training labels for current fold
        for i=1:z, j ∈ indTr[i][f] push!(zTr[f], Int64(i)) end

        # get training data for current fold
        for i=1:z, j ∈ indTr[i][f] push!(𝐐Tr[f], 𝐐[i][j]) end

        # fit machine learning model
        ℳ[f]=fit(model, 𝐐Tr[f], zTr[f];
                  #meanInit=M0,
                  verbose=false,
                  fitArgs✔...)


        # predict labels for current fold
        for i=1:z pl[f][i]=predict(ℳ[f], 𝐐Te[f][i], :l; verbose=false) end

        # compute confusion matrix for current fold
        for i=1:z, s=1:length(pl[f][i]) CM[f][i, pl[f][i][s]]+=1. end

        # compute balanced accuracy or accuracy for current CV
        sumCM=sum(CM[f])
        scoring == :b ? s[f] = 𝚺(CM[f][i, i]/𝚺(CM[f][i, :]) for i=1:z) / z :
                        s[f] = 𝚺(CM[f][i, i] for i=1:z)/ sumCM

        CM[f]/=sumCM # confusion matrices in proportions
    end

    ⏩ ? (@threads for f=1:nFolds fold(f) end) : (for f=1:nFolds fold(f) end)
    verbose && println(greyFont, "\nDone in ", defaultFont, now()-⌚)

    # compute mean and sd (balanced) accuracy
    avg=mean(s);
    std=stdm(s, avg);
    scoStr = scoring == :b ? "balanced accuracy" : "accuracy"

    # create cv struct
    cv=CVacc("$nFolds-fold", scoStr, _modelStr(model), CM, mean(CM), s, avg, std)
    return outModels ? (cv, ℳ) : cv
end



"""
```
function cvSetup(k       :: Int,
                 nCV     :: Int;
                 shuffle :: Bool = false)
```
Given `k` elements and a parameter `nCV`, a nCV-fold cross-validation
is obtained defining ``nCV`` permutations of ``k`` elements
in ``nTest=k÷nCV`` (integer division) elements for the test and
``k-nTest`` elements for the training,
in such a way that each element is represented in only one permutation.

Said differently, given a length `k` and the number of desired cross-validations
`nCV`, this function generates indices from the sequence of natural numbers
``1,..,k`` to obtain all nCV-fold cross-validation sets.
Specifically, it generates ``nCV`` vectors of indices for generating test sets
and ``nCV`` vectors of indices for geerating training sets.

If optional keyword argument `shuffle` is true,
the sequence of natural numbers ``1,..,k`` is shuffled before
running the function, thus in this case two successive runs of this function
will give different cross-validation sets, hence different accuracy scores.
By default `shuffle` is false, so as to allow exactly the same result
in successive runs.
Note that no random initialization for the shuffling is provided, so as to
allow the replication of the same random sequences starting again
the random generation from scratch.

This function is used in [`cvAcc`](@ref). It constitutes the fundamental
basis to implement customized cross-validation procedures.

Return the 2-tuple with:

- A vector of `nCV` vectors holding the indices for the training sets,
- A vector of `nCV` vectors holding the indices for the corresponding test sets.

**Examples**
```
using PosDefManifoldML, PosDefManifold

cvSetup(10, 2)
# return:
# (Array{Int64,1}[[6, 7, 8, 9, 10], [1, 2, 3, 4, 5]],
#  Array{Int64,1}[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

cvSetup(10, 2, shuffle=true)
# return:
# (Array{Int64,1}[[5, 4, 6, 1, 9], [3, 7, 8, 2, 10]],
#  Array{Int64,1}[[3, 7, 8, 2, 10], [5, 4, 6, 1, 9]])

cvSetup(10, 3)
# return:
# (Array{Int64,1}[[4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6]],
#  Array{Int64,1}[[1, 2, 3], [4, 5, 6], [7, 8, 9, 10]])

```

"""
function cvSetup(k       :: Int,
                 nCV     :: Int;
                 shuffle :: Bool = false)

    if nCV == 1 @error 📌*", cvSetup function: The number of cross-validation must be bigger than one" end
    nTest = k÷nCV # nTrain = k-nTest
    #rng = MersenneTwister(1900)
    shuffle ? a=shuffle!( Vector(1:k)) : a=Vector(1:k)
    indTrain = [IntVector(undef, 0) for i=1:nCV]
    indTest  = [IntVector(undef, 0) for i=1:nCV]

    # vectors of indices for test and training sets
    j=1
    for i=1:nCV-1
        indTest[i]=a[j:j+nTest-1]
        for g=j+nTest:length(a) push!(indTrain[i], a[g]) end
        for l=i+1:nCV, g=j:j+nTest-1 push!(indTrain[l], a[g]) end
        j+=nTest
    end
    indTest[nCV]=a[j:end]
    return indTrain, indTest
end


# ++++++++++++++++++++  Show override  +++++++++++++++++++ # (REPL output)
function Base.show(io::IO, ::MIME{Symbol("text/plain")}, cv::CVacc)
                            println(io, titleFont, "\n◕ Cross-Validation Accuracy")
                            println(io, separatorFont, "⭒  ⭒    ⭒       ⭒         ⭒", defaultFont)
                            println(io, separatorFont, ".cvType   :", defaultFont," $(cv.cvType)")
    cv.scoring   ≠ nothing && println(io, separatorFont, ".scoring  :", defaultFont," $(cv.scoring)")
    cv.modelType ≠ nothing && println(io, separatorFont, ".modelType:", defaultFont," $(cv.modelType)")
    cv.cnfs      ≠ nothing && println(io, separatorFont, ".cnfs      ", defaultFont,"(confusion mat. per fold)")
    cv.avgCnf    ≠ nothing && println(io, separatorFont, ".avgCnf    ", defaultFont,"(average confusion mat. )")
    cv.accs      ≠ nothing && println(io, separatorFont, ".accs      ", defaultFont,"(accuracies per fold    )")
    cv.avgAcc    ≠ nothing && println(io, separatorFont, ".avgAcc   :", defaultFont," $(round(cv.avgAcc; digits=3)) (average accuracy)")
    cv.stdAcc    ≠ nothing && println(io, separatorFont, ".stdAcc   :", defaultFont," $(round(cv.stdAcc; digits=3)) (st. dev accuracy)")
end
