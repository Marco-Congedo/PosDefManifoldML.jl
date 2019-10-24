#   Unit "cv.jl" of the PosDefManifoldML Package for Julia language
#   v 0.2.1 - last update 18th of October 2019
#
#   MIT License
#   Copyright (c) 2019,
#   Saloni Jain, Indian Institute of Technology, Kharagpur, India
#   Marco Congedo, CNRS, Grenoble, France:
#   https://sites.google.com/site/marcocongedo/home

# ? CONTENTS :
#   This unit implements cross-validation procedures for estimating
#   accuracy of all machine learning models.

"""
```
struct CVacc
    cvType  :: String
    scoring :: Union{String, Nothing}
    model   :: Union{MLmodel, Nothing}
    cnfs    :: Union{Vector{Matrix{T}}, Nothing} where T<:Real
    avgCnf  :: Union{Matrix{T}, Nothing} where T<:Real
    accs    :: Union{Vector{T}, Nothing} where T<:Real
    avgAcc  :: Union{Real, Nothing}
    stdAcc  :: Union{Real, Nothing}
end
```

A call to [`cvAcc`](@ref) results in an instance of this structure.
Fields:

`.cvTpe` is the type of cross-validation technique, given as a string
(e.g., "10-kfold")

`.scoring` is the type of accuracy that is computed, given as a string.
This has been passed as argument to [`cvAcc`](@ref).
Currently *accuracy* and *balanced accuracy* are supported.

`.model` is the machine learning model that has been passed as argument
to [`cvAcc`](@ref), e.g., an [`MDMmodel`](@ref) or an [`ENLRmodel`](@ref).

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
    cvType  :: String
    scoring :: Union{String, Nothing}
    model   :: Union{MLmodel, Nothing}
    cnfs    :: Union{Vector{Matrix{T}}, Nothing} where T<:Real
    avgCnf  :: Union{Matrix{T}, Nothing} where T<:Real
    accs    :: Union{Vector{T}, Nothing} where T<:Real
    avgAcc  :: Union{Real, Nothing}
    stdAcc  :: Union{Real, Nothing}
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
function cvAcc(model   :: MLmodel,
               𝐏Tr     :: ℍVector,
               yTr     :: IntVector,
               nCV     :: Int;
           scoring   :: Symbol = :b,
           shuffle   :: Bool   = false,
           verbose   :: Bool   = true,
           fitArgs...)
```
Cross-validation accuracy for a machine learning `model`:
given an ℍVector `𝐏Tr` holding ``k`` Hermitian matrices,
an [IntVector](@ref) `yTr` holding the ``k`` labels for these matrices and
the number of cross-validations `nCV`,
return a [`CVacc`](@ref) structure.

If `scoring`=:b (default) the **balanced accuracy** is computed.
Any other value will make the function returning the regular **accuracy**.
Balanced accuracy is to be preferred for unbalanced classes.
In any case, for balanced classes the balanced accuracy reduces to the
regular accuracy, therefore there is no point in using regular accuracy.

For the meaning of the `shuffle` argument (false by default),
see function [`cvSetup`](@ref), to which this argument is passed.

If `verbose` is true (default), information is printed in the REPL.
This option is included to allow repeated calls to this function
without crowding the REPL.

`fitArgs` are optional keyword arguents that are passed to the
[`fit`](@ref) function within this function.

**See**: [notation & nomenclature](@ref), [the ℍVector type](@ref).

**See also**: [`fit`](@ref), [`predict`](@ref).

**Examples**
```
using PosDefManifoldML

# generate some data
PTr, PTe, yTr, yTe=gen2ClassData(10, 30, 40, 60, 80)

# perform cross-validation using the minimum distance to mean classifier
cv=cvAcc(MDM(Fisher), PTr, yTr, 10)

# ...using the lasso logistic regression classifier
cv=cvAcc(ENLR(Fisher), PTr, yTr, 10)

# ...using the elastic-net logistic regression (α=0.9) classifier
cv=cvAcc(ENLR(Fisher), PTr, yTr, 10; alpha=0.9)

# perform another cross-validation
cv=cvAcc(MDM(Fisher), PTr, yTr, 10; shuffle=true)


```
"""
function cvAcc(model   :: MLmodel,
               𝐏Tr     :: ℍVector,
               yTr     :: IntVector,
               nCV     :: Int;
           scoring   :: Symbol = :b,
           shuffle   :: Bool   = false,
           verbose   :: Bool   = true,
           fitArgs...)

    ⌚=now()
    verbose && println(greyFont, "\nPerforming $(nCV) cross-validations...")

    z  = length(unique(yTr))            # number of classes
    𝐐  = [ℍ[] for i=1:z]               # data arranged by class
    for j=1:length(𝐏Tr) push!(𝐐[yTr[j]], 𝐏Tr[j]) end

    # pre-allocated memory
    𝐐Tr = [ℍ[] for k=1:nCV]                 # training data in 1 vector per CV
    zTr = [Int64[] for k=1:nCV]              # training labels in 1 vector per CV
    𝐐Te = [[ℍ[] for i=1:z] for k=1:nCV]     # testing data arranged by classes per CV
    CM  = [zeros(Float64, z, z) for k=1:nCV] # CV confusion matrices
    s   = Vector{Float64}(undef, nCV)        # CV accuracy scores
    pl  = [[Int[] for i=1:z] for k=1:nCV]    # CV predicted labels
    indTr = [[[]] for i=1:z]                 # CV indeces for training sets
    indTe = [[[]] for i=1:z]                 # CV indeces for test sets
    m=Vector{MLmodel}(undef, nCV)            # ML models

    # get indeces for all CVs (separated for each class)
    @threads for i=1:z indTr[i], indTe[i] = cvSetup(length(𝐐[i]), nCV; shuffle=shuffle) end

    @threads for k=1:nCV
        # get testing data for current cross-validation (CV)
        for i=1:z @inbounds 𝐐Te[k][i] = [𝐐[i][j] for j ∈ indTe[i][k]] end

        # get training labels for current cross-validation (CV)
        for i=1:z, j ∈ indTr[i][k] push!(zTr[k], Int64(i)) end

        # get training data for current cross-validation (CV)
        for i=1:z, j ∈ indTr[i][k] push!(𝐐Tr[k], 𝐐[i][j]) end

        if      model isa MDMmodel
                # create and fit MDM model
                m[k]=fit(MDM(model.metric), 𝐐Tr[k], zTr[k]; verbose=false, ⏩=false)

        elseif  model isa ENLRmodel
                # create and fit ENLR model
                m[k]=fit(ENLR(model.metric), 𝐐Tr[k], zTr[k]; verbose=false, ⏩=false, fitArgs...)
        end

        # predict labels and compute confusion matrix for current CV
        # NB: make sure the default predict method is adequate for all models
        for i=1:z
            @inbounds pl[k][i]=predict(m[k], 𝐐Te[k][i], :l; verbose=false, ⏩=false)
            for s=1:length(pl[k][i]) @inbounds CM[k][i, pl[k][i][s]] += 1. end
        end

        # compute balanced accuracy or accuracy for current CV
        sumCM=sum(CM[k])
        scoring == :b ? s[k] = 𝚺(CM[k][i, i]/𝚺(CM[k][i, :]) for i=1:z) / z :
                        s[k] = 𝚺(CM[k][i, i] for i=1:z)/ sumCM

        CM[k]/=sumCM # confusion matrices in percent

        # activate this when @spawn is used (Julia v0.3)
        # print(rand(dice), " ") # print a random dice in the REPL
    end
    verbose && println("Done in ", defaultFont, now()-⌚)

    # compute meand and sd (balanced) accuracy
    avg=mean(s);
    std=stdm(s, avg);
    scoStr = scoring == :b ? "balanced accuracy" : "accuracy"

    return CVacc("$nCV-fold", scoStr, model, CM, mean(CM), s, avg, std)

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
using PosDefManifoldML

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
    println(io, separatorFont, ".cvType  :", defaultFont," $(cv.cvType)")
    println(io, separatorFont, ".scoring :", defaultFont," $(cv.scoring)")
    println(io, separatorFont, ".model    ", defaultFont,"(",_modelStr(cv.model),")")
    println(io, separatorFont, ".cnfs     ", defaultFont,"(confusion mat. per fold)")
    println(io, separatorFont, ".avgCnf   ", defaultFont,"(average confusion mat. )")
    println(io, separatorFont, ".accs     ", defaultFont,"(accuracies per fold    )")
    println(io, separatorFont, ".avgAcc  :", defaultFont," $(round(cv.avgAcc; digits=3)) (average accuracy)")
    println(io, separatorFont, ".stdAcc  :", defaultFont," $(round(cv.stdAcc; digits=3)) (st. dev accuracy)")
end
