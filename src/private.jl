#   Unit "private.jl" of the PosDefManifoldML Package for Julia language

#   MIT License
#   Copyright (c) 2019-2025
#   Marco Congedo, CNRS, Grenoble, France:
#   https://sites.google.com/site/marcocongedo/home

# ? CONTENTS :
#   This unit implements internal functions

# -------------------------------------------------------- #

# return a vector of ranges partitioning lineraly and
# as much as possible evenly `n` elements in `threads` ranges.
# `threads` is the number of threads to which the ranges are to be
# dispatched. If `threads` is not provided, it is set to the number
# of threads Julia is currently instructed to use.
# For example, for `k`=99
# and `threads`=4, return Array{UnitRange{Int64},1}:[1:25, 26:50, 51:75, 76:99].
function _partitionLinRange4threads(n::Int, threads::Int=0)
    threads == 0 ? thr=nthreads() : thr=threads
    n<thr ? thr = n : nothing
    d = max(round(Int64, n / thr), 1)
    return [(r<thr ? (d*r-d+1:d*r) : (d*thr-d+1:n)) for r=1:thr]
end


function _getThreads(n::Int, callingFunction::String)
	threads = Threads.nthreads()
	threads == 1 && @warn 📌*", function "*callingFunction*": Julia is instructed to use only one thread."
	if n<threads && n<3
		@warn 📌*", function "*callingFunction*": the number of operations (n) is too low for taking advantage of multi-threading" threads n
		threads=1
	end
	return threads
end

function _getThreadsAndLinRanges(n::Int, callingFunction::String)
	threads = _getThreads(n, callingFunction)
	ranges = _partitionLinRange4threads(n, threads)
	return threads, ranges
end


# checks for `fit function`
function _check_fit(model       :: MLmodel,
              		 dim𝐏Tr     :: Int,
              		 dimyTr     :: Int,
           			 dimw  	    :: Int,
					 dimWeights :: Int,
					 modelName  :: String)
    errMsg1="the number of data do not match the number of labels."
	errMsg2="the number of data do not match the number of elements in `w`."
	errMsg3="the number of data do not match the number of elements in `weights`."
    if dim𝐏Tr ≠ dimyTr
		@error 📌*", fit function, model "*modelName*": "*errMsg1
		return false
	end
    if dimw ≠ 0 && dimw ≠ dimyTr
		@error 📌*", fit function, model "*modelName*": "*errMsg2
		return false
	end
	if dimWeights ≠ 0 && dimWeights ≠ dimyTr
		@error 📌*", fit function, model "*modelName*": "*errMsg3
		return false
	end

	return true
end

# check for argument `what` in `predict` function
_whatIsValid(what::Symbol, funcName::String) =
	if what ∉ (:l, :labels, :p, :probabilities, :f, :functions)
		@error 📌*", "*funcName*" function: the `what` symbol is not supported."
		return false
	else
		return true
	end

# translate the `what` argument to a string to give feedback
_what2Str(what::Symbol) =
	if      what ∈(:f, :fucntions)      return "functions"
	elseif  what ∈(:l, :labels)         return "labels"
	elseif  what ∈(:p, :probabilities)  return "prob. of belonging to each class"
	end

# return the dimension of the manifold of PD matrices: (n*n+1)/2
_manifoldDim(P::ℍ, vecRange::UnitRange) =
	if length(vecRange)==size(P, 1)
		result=_manifoldDim(P)
	else
		m=0; for j=vecRange, i=j:size(P, 1) m+=1 end
		result=m
	end

_manifoldDim(P::ℍ) = ( size(P, 1) * (size(P, 1)+1) ) ÷ 2

# dimension of the manifold if 𝐏Tr is an ℍVector,
# dimension of the tagent(feature) vectors if 𝐏Tr is a Matrix
_getDim(𝐏Tr :: Union{ℍVector, Matrix{Float64}}, vecRange::UnitRange) =
	𝐏Tr isa ℍVector ? _manifoldDim(𝐏Tr[1], vecRange) : length(vecRange)

_getDim(𝐏Tr :: Union{ℍVector, Matrix{Float64}}) =
	𝐏Tr isa ℍVector ? _manifoldDim(𝐏Tr[1]) : size(𝐏Tr, 2)


# convert a ML model in a atring to release information
_model2Str(model::MLmodel) =
  if 		model isa MDMmodel
	  		return "MDM"

  elseif    model isa ENLRmodel
    		if     model.alpha≈1. return "Lasso logistic regression"
    		elseif model.alpha≈0. return "Ridge logistic regression"
    		else                  return "El. Net (α=$(round(model.alpha; digits=2))) log. reg."
			end

  elseif    model isa SVMmodel
			if 		model.svmType==SVC 			return "SVC"
			elseif  model.svmType==NuSVC 		return "NuSVC"
			elseif  model.svmType==EpsilonSVR 	return "EpsilonSVR"
			elseif  model.svmType==OneClassSVM 	return "OneClassSVM"
			elseif  model.svmType==NuSVR 		return "NuSVR"
			else    return  "Warning: the SVM type is unknown"
			end

  else      return "unknown"
  end

# check on `what` argument of `fit` function
_fitTypeIsValid(what::Symbol, funcName::String) =
	if what ∉ (:best, :path)
		@error 📌*", "*funcName*" function: the `fitType` symbol must be `:best` or `:path`."
		return false
	else
		return true
	end

# check on `onWhich` argument of `predict` function
function _ENLRonWhichIsValid(model::ENLRmodel, fitType::Symbol,
                    onWhich::Int, funcName::String)
    if fitType==:best
		return true
	else #fitType==:path
		i=length(model.path.lambda)
		if !(0<=onWhich<=i)
			@error 📌*", "*funcName*" function: the `onWhich` integer argument must be comprised between 0 (all models) and $i."
			return false
		else
			return true
		end
	end
end

# return a string to give information on what model is used to predict
# based on arguments `fitType` and `onWhich`
_ENLRonWhichStr(model::ENLRmodel, fitType::Symbol, onWhich::Int) =
	if 		fitType==:best
		return "from the best "*_model2Str(model)*" model (λ=$(round(model.best.lambda[1]; digits=5)))"
	else # :path
		if onWhich == 0
			return "from all "*_model2Str(model)*" models"
		else
			return "from "*_model2Str(model)*" model $(onWhich) (λ=$(round(model.path.lambda[onWhich]; digits=5)))"
		end
	end


# create a copy of optional keyword arguments `args`
# removing all arguments with names listed in tuple `remove`.
# Examples:
# fitArgs✔=_rmArgs((:meanISR, :fitType); fitArgs...)
# fitArgs✔=_rmArgs((:meanISR,); fitArgs...) # notice the comma after `meanISR`
# note: named tuples are immutable, that's why a copy must be created
function _rmArgs(remove::Tuple; args...)
	D = Dict(args)
	for key ∈ remove delete!(D, key) end
    return Tuple(D)
end


# given optional keyword arguments `args`,
# return the value of the argument with key `key`.
# If the argument does not exist, return `nothing`
function _getArgValue(key::Symbol; args...)
   D = Dict(args)
   return haskey(D, key) ? D[key] : nothing
end

# get a valid weigts `w` object and perform check given
# the user-defined `w` argument. Used in `fit` and `crval` functions.
function _getWeights(w :: Union{Symbol, Tuple, Vector}, y::IntVector, funcName::String)
	if 		(w isa Vector && isempty(w)) || w==:uniform || w==:u return Float64[]
	elseif	w isa Vector && !isempty(w)
		    nObs, length_w = length(y), length(w)
		    if length_w==nObs return w
			else @error 📌*", "*funcName*"invalid vector `w`. `w` must contain as many elements as there are observations" length_w nObs
			end
	elseif  w==:balanced || w==:b return tsWeights(y)
	elseif  w isa Tuple
		    nClasses, length_w = length(unique(y)), length(w)
			if length_w==nClasses return tsWeights(y; classWeights=collect(w))
			else @error 📌*", "*funcName*"invalid tuple `w`. `w` must contain as many elements as there are classes" length_w nClasses
			end
	else
			@error 📌*", "*funcName*"invalid argument `w`. `w` must be a vector, an empty vector, a tuple of as many real numbers as classes, or symbol `:balanced`, or symbol `:uniform`"
			return nothing
	end
end


# Get the feature matrix for fit functions of ML model in the tangent space:
# if `𝐏Tr` is a matrix just return the columns in `vecRange` (by default all).
# if `𝐏Tr` is vector of Hermitian matrices, they are projected onto the
# tangent space. If the inverse square root of a base point `meanISR`
# is provided, the projection is obtained at this base point, otherwise the
# mean of all points is computed and used as base point.
# if `meanISR=I` is used, the tangent space is defined at the identity matrix.
# If the mean is to be computed by an iterative algorithm (e.g., if the metric
# of the model is the Fisher metric), an initialization `meanInit`, weights
# `w` and a tolerance `tol` are used.
# Once projected onto the tangent space, the matrices in `𝐏Tr` are vectorized
# using only the rows (or columns) specified by `vecRange`.
# if `verbose` is true, print "Projecting data onto the tangent space..."
# if `transpose` the feature vectors are in the rows of `X`, otherwise in the
# columns of `X`.
# if ⏩ is true, the projection onto the tangent space
# and the algorithm to compute the mean are multi-threaded
function _getTSvec_fit!(ℳ :: TSmodel,
            𝐏Tr         :: Union{ℍVector, Matrix{Float64}},
            meanISR	    :: Union{ℍ, Nothing, UniformScaling},
            meanInit	:: Union{ℍ, Nothing},
            tol		    :: Real,
            w		    :: Union{Symbol, Tuple, Vector},
            vecRange	:: UnitRange,
            transpose   :: Bool,
            verbose	    :: Bool,
            ⏩	       :: Bool)
	if 𝐏Tr isa ℍVector
		verbose && println(greyFont, "Projecting data onto the tangent space...")
		if meanISR==nothing
			(X, G⁻½)=tsMap(ℳ.metric, 𝐏Tr;
			               w,⏩, vecRange, meanInit, tol, transpose)
			ℳ.meanISR = G⁻½
		else
			X=tsMap(ℳ.metric, 𝐏Tr;
			        w, ⏩, vecRange, meanISR, transpose)
			ℳ.meanISR = meanISR
		end
	else X=𝐏Tr[:, vecRange]
	end
	return X
end

# Get the feature matrix for predict functions of ML model in the tangent space:
# if `𝐏Te` is a matrix just return the columns in `vecRange` (by default all).
# if `𝐏Te` is vector of Hermitian matrices, they are projected onto the
# tangent space. If an inverse square root (ISR) `tranfer` is provided
# (typically the ISR of the mean of the matrices in `𝐏Te`), this is used
# as ISR of the base point, otherwise the ISR of the base point stored in the
# model ℳ is used. The latter is the classical approach, the former realizes the
# adaptation (transfer learning) explained in Barachant et al. (2013).
# Once projected onto the tangent space, the matrces in `𝐏Te` are vectorized
# using only the rows (or columns) specified by `vecRange`.
# if `transpose` the feature vectors are in the rows of `X`, otherwise in the
# columns of `X`.
# if `verbose` is true, print "Projecting data onto the tangent space..."
# if ⏩ is true, the projection onto the tangent space is multi-threaded
_getTSvec_Predict!(ℳ	   		:: TSmodel,
        𝐏Te		  :: Union{ℍVector, Matrix{Float64}},
        transfer   :: Union{ℍ, Nothing, UniformScaling},
        vecRange	 :: UnitRange,
        transpose  :: Bool,
        verbose	 :: Bool,
        ⏩	       :: Bool) =
	if 𝐏Te isa ℍVector
		verbose && println(greyFont, "Projecting data onto the tangent space...")
		return tsMap(ℳ.metric, 𝐏Te;
				     meanISR = transfer==nothing ? ℳ.meanISR : transfer,
				     ⏩=⏩,
				     vecRange=vecRange,
					 transpose=transpose)
	else
		return 𝐏Te[:, vecRange]
	end



# internal function to perform normalization of matrix `X`, which holds the 
# tangent vectors in the columns (dims=1) or rows (dims=2).
# `normalization` is either nothing, a tuple (calls `rescale!`) or a function
# either `normalize!` or `standardize`.
function _normalizeTSvec!(X, normalize; dims)
	if !(normalize===nothing)
		if normalize isa Tuple 
			rescale!(X, normalize; dims)
		else
			normalize(X; dims=2)
		end
	end
end

# internal function to adapt the pipeline.
# This is called in the `predict` method of all models.
# If `pipeline` is a valid Pipeline, it is used to fit it to the data
# `𝐏Te`, which will also transform it,
# otherwise, if a pipeline has been stored in `model`
# `𝐏Te` will be transformed accordingly. 
function _applyPipeline!(𝐏Te, pipeline, model)
	if pipeline isa Pipeline
		adapted_pipeline = fit!(𝐏Te, pipeline) # This fits and transforms
	elseif model.pipeline isa Pipeline
		transform!(𝐏Te, model.pipeline) # This transforms only using the model pipeline
	end
end