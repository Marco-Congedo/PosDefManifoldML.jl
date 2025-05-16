#   Unit "cv.jl" of the PosDefManifoldML Package for Julia language

#   MIT License
#   Copyright (c) 2019-2025
#   Marco Congedo, CNRS, Grenoble, France:
#   https://sites.google.com/site/marcocongedo/home
#   Saloni Jain, Indian Institute of Technology, Kharagpur, India

# ? CONTENTS :
#   This unit implements cross-validation procedures for estimating
#   accuracy of all machine learning models.


"""
```julia
struct CVres <: CVresult
    cvType      :: String
    scoring     :: Union{String, Nothing}
    modelType   :: Union{String, Nothing}
    predLabels  :: Union{Vector{Vector{Vector{I}}}, Nothing} where I<:Int
    losses      :: Union{Vector{BitVector}, Nothing}
    cnfs        :: Union{Vector{Matrix{I}}, Nothing} where I<:Int
    avgCnf      :: Union{Matrix{T}, Nothing} where T<:Real
    accs        :: Union{Vector{T}, Nothing} where T<:Real
    avgAcc      :: Union{Real, Nothing}
    stdAcc      :: Union{Real, Nothing}
    z           :: Union{Real, Nothing}
    p           :: Union{Real, Nothing}
end
```

A call to [`crval`](@ref) results in an instance of this structure.

**Fields:**

`.cvTpe` is the type of cross-validation technique, given as a string
(e.g., "10-kfold") or xxx

`.scoring` is the type of accuracy that is computed, given as a string.
This has been passed as argument to [`crval`](@ref).
Currently *accuracy* and *balanced accuracy* are supported.

`.modelType` is type of the machine learning used for performing the
cross-validation, given as a string.

`.predLabels` is an f-vector of z integer vectors holding the vectors of 
predicted labels. There is one vector for each fold (f) and each containes
as many vector as classes (z), in turn each one containing the predicted labels
for the trials.

`.losses` is an f-vector holding BitVector types (vectors of booleans), 
each holding the binary loss for a fold. 

`.cnfs` is an f-vector of matrices holding the *confusion matrices*
obtained at each fold of the cross-validation. These matrices holds 
*frequencies* (counts), that is, the sum of all elements equals
the number of trials used for each fold.

`.avgCnf` is the *average confusion matrix* of proportions 
across the folds of the cross-validation. This matrix holds *proportions*,
that is, the sum of all elements equal 1.0.

`.accs` is an f-vector of real numbers holding the *accuracies* obtained
at each fold of the cross-validation.

`.avgAcc` is the *average accuracy* across the folds of the
cross-validation.

`.stdAcc` is the *standard deviation of the accuracy* across the folds of the
cross-validation.

`.z` is the test-statistic fot the hypothesis that the observed average error loss is 
inferior to the specified expected value.

`.p` is the p-value of the above hypothesis test.

See [`crval`](@ref) for more informations

"""
struct CVres <: CVresult 
    cvType      :: String
    scoring     :: Union{String, Nothing}
    modelType   :: Union{String, Nothing}
    predLabels  :: Union{Vector{Vector{Vector{I}}}, Nothing} where I<:Int
    losses      :: Union{Vector{BitVector}, Nothing}
    cnfs        :: Union{Vector{Matrix{I}}, Nothing} where I<:Int
    avgCnf      :: Union{Matrix{T}, Nothing} where T<:Real
    accs        :: Union{Vector{T}, Nothing} where T<:Real
    avgAcc      :: Union{Real, Nothing}
    stdAcc      :: Union{Real, Nothing}
    z           :: Union{Real, Nothing}
    p           :: Union{Real, Nothing}
end

"""
```julia
CVres(s::String) =
     CVres(s, nothing, nothing, nothing, nothing, nothing,
           nothing, nothing, nothing, nothing, nothing, nothing)
```

Construct an instance of the CVres structure giving only the `.cvtype`
field. All other fields are filled with `nothing`. This is useful to construct
manually crval objects.
"""
CVres(s::String)=CVres(s, nothing, nothing, nothing, nothing, nothing, 
                        nothing, nothing, nothing, nothing, nothing, nothing)

                       

"""
```julia
function crval(model    :: MLmodel,
               𝐏        :: ℍVector,
               y        :: IntVector;
        pipeline    :: Union{Pipeline, Nothing} = nothing,
        nFolds      :: Int     = min(10, length(y)÷3),
        shuffle     :: Bool    = false,
        scoring     :: Symbol  = :b,
        hyptest     :: Symbol  = :Bayle,
        verbose     :: Bool    = true,
        outModels   :: Bool    = false,
        ⏩           :: Bool    = true,
        fitArgs...)
```
Stratified cross-validation accuracy for a machine learning `model`
given an ℍVector ``𝐏`` holding *k* Hermitian matrices,
an [IntVector](@ref) `y` holding the *k* labels for these matrices and
the number of folds `nFolds`,
return a [`CVres`](@ref) structure.

For each fold, a machine learning model is fitted on training data and labels
are predicted on testing data. Summary classification performance statistics
are stored in the output structure.

**Optional keyword arguments**

If a `pipeline`, of type [`Pipeline`](@ref) is provided, 
the pipeline is fitted on training data and applied for predicting the testing data.

`nFolds` by default is set to the minimum between 10 and the number
of observation ÷ 3 (integer division).

If `scoring`=:b (default) the **balanced accuracy** is computed.
Any other value will make the function returning the regular **accuracy**.
Balanced accuracy is to be preferred for unbalanced classes.
For balanced classes the balanced accuracy reduces to the
regular accuracy, therefore there is no point in using regular accuracy
if not to avoid a few unnecessary computations when the class are balanced.

!!! info "Don't be surprised"
    Note that this function computes the error loss for each fold (see [`CVres`](@ref)).
    The average error loss across folds is the complement of accuracy, 
    not of balanced accuracy. If the classes are balanced and you use `scoring`=:a (accuracy), 
    the average error loss across folds is equal to 1 minus the average accuracy, 
    also computed by this function, but this is not true is the classes are unbalanced
    and you use `scoring`=:b (default).

`hyptest` is a symbol specifying the kind of statistical test to be carried out.
This function tests that the average observed binary error loss is inferior to what is to be 
expected by the hypothesis of random chance, which is set to ``1-\\frac{1}{z}``, where
``z`` is the number of classes. At the moment, only Bayle's test (see [`testCV`](@ref))
is implemented.

For the meaning of the `shuffle` argument (false by default),
see function [`cvSetup`](@ref), to which this argument is passed.

If `verbose` is true (default), information is printed in the REPL.

If `outModels` is true, return a 2-tuple holding a [`CVres`](@ref) structure
and a `nFolds`-vector of the model fitted for each fold,
otherwise (default), return only a [`CVres`](@ref) structure.

If `⏩` the computations are multi-threaded across folds.
It is true by default. Set it to false if there are problems in running
this function and for debugging.

!!! note "Multi-threading"
    If you run the cross-validation with independent threads per fold setting `⏩=true`(default),
    the [`fit!`](@ref) and [`predict`](@ref) function that will be called within each fold will
    we run in single-threaded mode. Vice versa, if you pass `⏩=false`, these two functions
    will be run in multi-threaded mode. This is done to avoid
    overshooting the number of threads to be activated.

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
|`meanInit`| `meanInit`| `meanInit`|
|          | `fitType` |
|          | `offsets` |
|          | `lambda`  |
|          | `folds`   |

If you pass the `meanISR` argument, this must be nothing (default) 
or I (the identity matrix). If you pass `meanISR=I` for a tangent space model,
parallel transport of the points to the identity before lifting
the points to the tangent space will not be carried out.
This can be used if a recentering conditioner is passed in the `pipeline`.

Also, if you pass a `w` argument (weights for barycenter estimations),
do not pass a vector of weights, just pass a symbol, e.g., `w=:b`
for balancing weights.

**See**: [notation & nomenclature](@ref), [the ℍVector type](@ref)

**See also**: [`fit`](@ref), [`predict`](@ref)

**Examples**
```julia
using PosDefManifoldML, PosDefManifold

# Generate some data
P, _dummyP, y, _dummyy = gen2ClassData(10, 60, 80, 30, 40, 0.2)

# Perform 10-fold cross-validation using the minimum distance to mean classifier
cv = crval(MDM(Fisher), P, y)

# Do the same applying a pre-conditioning pipeline
p = @→ Recenter(; eVar=0.999) Compress Shrink(Fisher; radius=0.02)
cv = crval(MDM(Fisher), P, y; pipeline = p)

# Perform 10-fold cross-validation using the lasso logistic regression classifier
cv = crval(ENLR(Fisher), P, y)

# ...using the support-vector machine classifier
cv = crval(SVM(Fisher), P, y)

# ...with a Polynomial kernel of order 3 (default)
cv = crval(SVM(Fisher), P, y; kernel=kernel.Polynomial)

# Perform 8-fold cross-validation instead
# (and see that you can go pretty fast if your PC has 8 threads)
cv = crval(SVM(Fisher), P, y; nFolds=8)

# ...balance the weights for tangent space projection
cv = crval(ENLR(Fisher), P, y; nFolds=8, w=:b)

# perform another cross-validation shuffling the folds
cv = crval(ENLR(Fisher), P, y; shuffle=true, nFolds=8, w=:b)

```
"""
function crval(model    :: MLmodel,
               𝐏        :: ℍVector,
               y        :: IntVector;
            # Conditioners (Data transformation)
            pipeline        :: Union{Pipeline, Nothing} = nothing,
            # Cross-validation parameters
            nFolds          :: Int      = min(10, length(y) ÷ 3),           
            shuffle         :: Bool     = false,
            # Default performance metric and statistical test
            scoring         :: Symbol   = :b,
            hyptest         :: Symbol   = :Bayle,
            # arguments for this function
            verbose         :: Bool     = true,
            outModels       :: Bool     = false,
            ⏩              :: Bool     = true,
            BLAS_threads    :: Int = max(1, Threads.nthreads()-nFolds),
            # Optional arguments for fit functions
            fitArgs...)

    ⌚ = now()

    if ⏩
        blasThreads=BLAS.get_num_threads()
        BLAS.set_num_threads(BLAS_threads) # let all threads free for CV multithreading
    end

    verbose && println(greyFont, "\nPerforming $(nFolds)-fold cross-validation...")

    clabels=unique(y)           # class labels
    z  = length(clabels)        # number of classes
    𝐐  = [ℍ[] for i=1:z]       # data arranged by class
    for j=1:length(𝐏) 
        push!(𝐐[y[j]], 𝐏[j]) 
    end

    # pre-allocated memory
    yTr     = [Int64[] for f=1:nFolds]              # training labels in 1 vector per fold
    𝐐Tr     = [ℍ[] for f=1:nFolds]                  # training data in 1 vector per folds
    yTe     = [Int64[] for f=1:nFolds]              # testing labels in 1 vector per fold
    𝐐Te     = [[ℍ[] for i=1:z] for f=1:nFolds]      # testing data arranged by classes per fold
    CM      = [zeros(Int, z, z) for f=1:nFolds]     # confusion matrices per fold
    as      = Vector{Float64}(undef, nFolds)        # accuracy scores per fold
    errls   = Vector{BitVector}(undef, nFolds)      # binary error loss per fold
    predLab = [[Int64[] for i=1:z] for f=1:nFolds]  # predicted labels by classes per fold
    indTr   = [[Int64[]] for i=1:z]                 # indeces for training sets by fold per class
    indTe   = [[Int64[]] for i=1:z]                 # indeces for test sets by fold per class
    ℳ = Vector{MLmodel}(undef, nFolds)             # ML models

    # get indeces for all CVs (separated for each class)
    for i=1:z 
        indTr[i], indTe[i] = cvSetup(length(𝐐[i]), nFolds; shuffle=shuffle) 
    end

    fitArgs✔ = ()
    # make sure the user doesn't pass arguments that skrew up the cv

    if model isa MDMmodel
        fitArgs✔ = _rmArgs((:meanInit, :normalize, :verbose, :⏩); fitArgs...)
    end

    if model isa ENLRmodel
        fitArgs✔ = _rmArgs((:meanInit, :fitType, :verbose, :⏩,
                                :offsets, :lambda, :folds); fitArgs...)
        # overwrite the `alpha` value in `model` if the user has passed keyword `alpha`
        if (a = _getArgValue(:alpha; fitArgs...)) ≠ nothing model.alpha = a end
    end

    if model isa SVMmodel
        fitArgs✔ = _rmArgs((:meanInit, :verbose, :⏩); fitArgs...)
        # overwrite the `svmType` and `kernel` values in `model` if the user has passed then as kwargs
        if (a = _getArgValue(:svmType; fitArgs...)) ≠ nothing model.svmType = a end
        if (a = _getArgValue(:kernel;  fitArgs...)) ≠ nothing model.kernel = a end
    end
 
    # perform cv
    function fold(f::Int)
        verbose && @static if VERSION >= v"1.3" print(defaultFont, rand(dice), " ") end # print a random dice in the REPL

        # get training labels for current fold
        for i=1:z, j ∈ indTr[i][f] 
            @inbounds push!(yTr[f], Int64(i)) 
        end

        # get training data for current fold
        for i=1:z, j ∈ indTr[i][f] 
            @inbounds push!(𝐐Tr[f], 𝐐[i][j]) 
        end

        # get testing labels for current fold (only used to compute the loss)
        for i=1:z, j ∈ indTe[i][f] 
            @inbounds push!(yTe[f], Int64(i)) 
        end

        # get testing data for current fold
        for i=1:z 
            @inbounds 𝐐Te[f][i] = [𝐐[i][j] for j ∈ indTe[i][f]] 
        end      
        
        # fit machine learning model
        ℳ[f] = fit(model, 𝐐Tr[f], yTr[f];
                    pipeline,
                    meanInit=nothing,
                    verbose=false,
                    ⏩=!⏩, # flase if CV is multi-threaded, true if not
                    fitArgs✔...)

        # predict labels in current fold for each class
        for i=1:z 
            @inbounds predLab[f][i] = predict(ℳ[f], 𝐐Te[f][i], :l; verbose=false, ⏩=!⏩) 
        end

        # compute confusion matrix (as frequencies) for current fold
        for i=1:z, s=1:length(predLab[f][i]) 
            @inbounds CM[f][i, predLab[f][i][s]] += 1
        end

        # compute balanced accuracy or accuracy for current fold
        scoring == :b ? as[f] = sum(CM[f][i, i]/sum(CM[f][i, :]) for i=1:z) / z :
                        as[f] = sum(CM[f][i, i] for i=1:z) / sum(CM[f])

        # compute binary error loss for current fold
        errls[f] = binaryloss(vcat(yTe[f]...), vcat(predLab[f]...)) 

    end # function fold(f)

    # This actually runs the cross-validation
    if ⏩ 
        @threads for f=1:nFolds 
            fold(f) 
        end
    else 
        @simd for f=1:nFolds 
            @inbounds fold(f) 
        end
    end
    verbose && println(greyFont, "\nDone in ", defaultFont, now()-⌚)

    # compute mean and sd (balanced) accuracy
    avg = mean(as);
    std = stdm(as, avg);
    sStr = scoring == :b ? "balanced accuracy" : "accuracy"

    # compute the mean of the confusion matrices as proportions
    mCM = mean(CM./(sum.(CM)))

    if hyptest === :Bayle 
        zstat, pvalue, ase = testCV(CM)
    end

    # create cv struct
    cv = CVres("$nFolds-fold", sStr, _model2Str(model), predLab, errls, CM, mCM, as, avg, std, zstat, pvalue)
    
    # restore the number of threads for BLAS as the user had before invoking this function
    ⏩ && (BLAS.set_num_threads(blasThreads))

    return outModels ? (cv, ℳ) : cv
end



"""
```julia
function cvSetup(k       :: Int,
                 nCV     :: Int;
                 shuffle :: Bool = false)
```

Given `k` elements and a parameter `nCV`, a nCV-fold cross-validation
is obtained defining `nCV` permutations of *k* elements
in *nTest=k÷nCV* (integer division) elements for the test and
*k-nTest* elements for the training,
in such a way that each element is represented in only one permutation.

Said differently, given a length `k` and the number of desired cross-validations
`nCV`, this function generates indices from the sequence of natural numbers
*1,..,k* to obtain all nCV-fold cross-validation sets.
Specifically, it generates `nCV` vectors of indices for generating test sets
and `nCV` vectors of indices for geerating training sets.

If optional keyword argument `shuffle` is true,
the sequence of natural numbers *1,..,k* is shuffled before
running the function, thus in this case two successive runs of this function
will give different cross-validation sets, hence different accuracy scores.
By default `shuffle` is false, so as to allow exactly the same result
in successive runs.
Note that no random initialization for the shuffling is provided, so as to
allow the replication of the same random sequences starting again
the random generation from scratch.

This function is used in [`crval`](@ref). It constitutes the fundamental
basis to implement customized cross-validation procedures.

Return the 2-tuple with:

- A vector of `nCV` vectors holding the indices for the training sets,
- A vector of `nCV` vectors holding the indices for the corresponding test sets.

**Examples**
```julia
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
    shuffle ? (a=shuffle!( Vector(1:k))) : (a=Vector(1:k))
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
function Base.show(io::IO, ::MIME{Symbol("text/plain")}, cv::CVres)
                            println(io, titleFont, "\n◕ Cross-Validation Accuracy")
                            println(io, separatorFont, "⭒  ⭒    ⭒       ⭒         ⭒", defaultFont)
                            println(io, separatorFont, ".cvType   :", defaultFont," $(cv.cvType)")
    cv.scoring      ≠ nothing && println(io, separatorFont, ".scoring  :", defaultFont," $(cv.scoring)")
    cv.modelType    ≠ nothing && println(io, separatorFont, ".modelType:", defaultFont," $(cv.modelType)")
    cv.predLabels   ≠ nothing && println(io, separatorFont, ".predLabels ", defaultFont,"a vector of #classes vectors of predicted labels per fold")
    cv.losses       ≠ nothing && println(io, separatorFont, ".losses     ", defaultFont,"a vector of binary loss per fold")
    cv.cnfs         ≠ nothing && println(io, separatorFont, ".cnfs       ", defaultFont,"a confusion matrix per fold (frequencies)")
    cv.avgCnf       ≠ nothing && println(io, separatorFont, ".avgCnf     ", defaultFont,"average confusion matrix (proportions)")
    cv.accs         ≠ nothing && println(io, separatorFont, ".accs       ", defaultFont,"a vector of accuracies, one per fold")
    cv.avgAcc       ≠ nothing && println(io, separatorFont, ".avgAcc   :", defaultFont," $(round(cv.avgAcc; digits=3))")
    cv.z            ≠ nothing && println(io, separatorFont, ".z        :", defaultFont," $(round(cv.z; digits=4))")
    if cv.p ≠ nothing 
        if cv.p<0.001
            println(io, separatorFont, ".p        :", defaultFont," < 0.0001")
        else
            println(io, separatorFont, ".p        :", defaultFont," $(round(cv.p; digits=4))")
        end
    end
end
