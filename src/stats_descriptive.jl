#   Unit "stats_descriptive.jl" of the PosDefManifoldML Package for Julia language

#   MIT License
#   Copyright (c) 2019-2025
#   Marco Congedo, CNRS, Grenoble, France:
#   https://sites.google.com/site/marcocongedo/home

# ? CONTENTS :
#   This unit implements descrptive statistics for cross-validation.


"""
```julia
function confusionMat(yTrue::IntVector, yPred::IntVector)
```

Return the *confusion matrix* expressing *frequencies* (counts),
given integer vectors of true label `yTrue`
and predicted labels `yPred`.

The length of `yTrue` and `yPred` must be equal. Furthermore,
the `yTrue` vector must comprise all natural numbers
in between 1 and *z*, where *z* is the number of classes.

The confusion matrix will have size *z*. It is computed
starting from a matrix filled everywhere with zeros and
adding, for each label, 1 at entry [`i`, `j`] of the matrix, where
`i` is the true label and `j` the predicted label.
Thus, the first row will report the true labels for class 1, 
the second row the true labels for class 2, etc.

The returned matrix is a matrix of integers.

**See** [`predict`](@ref), [`predictAcc`](@ref), [`predictErr`](@ref)

**Examples**

```julia
using PosDefManifoldML
confusionMat([1, 1, 1, 2, 2], [1, 1, 1, 1, 2])
# return: [3 0; 1 1]
```
"""
function confusionMat(yTrue::IntVector, yPred::IntVector)

	n1=length(yTrue)
	n2=length(yPred)
	if n1≠n2
		@error 📌*", function ConfusionMat: the length of the two argument vectors must be equal." n1 n2
		return
	end

	cTrue=sort(unique(yTrue))
	z = length(cTrue)
	if cTrue≠[i for i∈1:z]
		@error 📌*", function ConfusionMat: the `yTrue` vector must contains all natural numbers from 1 to the number of classes. It contains instead: " cTrue
		return
	end

	CM = zeros(Int, z, z)
	for i=1:n1 CM[yTrue[i], yPred[i]]+=1 end
	return CM #/=sum(CM)
end

"""
```julia
(1)
function predictAcc(yTrue::IntVector, yPred::IntVector;
		scoring	:: Symbol = :b,
		digits	:: Int=3)

(2)
function predictAcc(CM::Union{Matrix{R}, Matrix{S}};
		scoring	:: Symbol = :b,
		digits	:: Int=3) where {R<:Real, S<:Int}
```

Return the prediction accuracy as a proportion, that is, ∈``[0, 1]``,
given

- (1) the integer vectors of true labels `yTrue` and of predicted labels `yPred`, or
- (2) a confusion matrix.

The confusion matrix may hold integers, in which case it is interpreted as expressing *frequencies*
(counts) or may hold real numbers, in which case it is interpreted as expressing *proportions*.

If `scoring`=:b (default) the **balanced accuracy** is computed.
Any other value will make the function returning the regular **accuracy**.
Balanced accuracy is to be preferred for unbalanced classes.
For balanced classes the balanced accuracy reduces to the
regular accuracy, therefore there is no point in using regular accuracy
if not to avoid a few unnecessary computations when the class are balanced.

The error is rounded to the number of optional keyword argument
`digits`, 3 by default.

**Maths**

The regular *accuracy* is given by sum of the diagonal elements
of the confusion matrix expressing *proportions*.

For the *balanced accuracy*, the diagonal elements
of the confusion matrix are divided by the respective row sums
and their mean is taken.

**See** [`predict`](@ref), [`predictErr`](@ref), [`confusionMat`](@ref)

**Examples**

```julia
using PosDefManifoldML
predictAcc([1, 1, 1, 2, 2], [1, 1, 1, 1, 2]; scoring=:a)
# regular accuracy, return: 0.8
predictAcc([1, 1, 1, 2, 2], [1, 1, 1, 1, 2])
# balanced accuracy, return: 0.75
```
"""
function predictAcc(yTrue::IntVector, yPred::IntVector;
					scoring:: Symbol = :b,
	          		digits::Int=3)
	n1=length(yTrue)
	n2=length(yPred)
	if n1≠n2
		@error 📌*", function `predictAcc` or `predictErr`: the length of the two argument vectors must be equal." n1 n2
		return
	end

	if scoring≠:b # regular accuracy
		return round(sum(y1==y2 for (y1, y2) ∈ zip(yTrue, yPred))/n1; digits=digits)
	else # balanced accuracy
		CM=confusionMat(yTrue, yPred)
		z=size(CM, 1)
		return round(sum(CM[i, i]/sum(CM[i, :]) for i=1:z) / z; digits=digits)
	end
end

function predictAcc(CM:: Matrix{R};
					scoring:: Symbol = :b,
					digits::Int=3) where R<:Real

	num_of_rows, num_of_cols = size(CM)
	if num_of_rows≠num_of_cols
		@error 📌*", function predictAcc or predictErr: the `CM` argument must be square as this must be a confusion matrix." num_of_rows num_of_cols
		return
	end

	sum_of_elements=sum(CM)
	if sum_of_elements ≉  1.0
		@error 📌*", function predictAcc or predictErr: the elements of `CM` matrix argument must sum up to 1.0 as this must be a confusion matrix." sum_of_elements
		return
	end

	return scoring==:b ? round(sum(CM[i, i]/sum(CM[i, :]) for i=1:size(CM, 1)) / size(CM, 1);
								digits=digits) :
						 round(tr(CM);
						 		digits=digits)
end


predictAcc(CM:: Matrix{S}; 
			scoring:: Symbol = :b,
			digits::Int=3) where S<:Int = predictAcc(CM./tr(CM); scoring, digits)

"""
```julia
(1)
function predictErr(yTrue::IntVector, yPred::IntVector;
		scoring	:: Symbol = :b,
		digits	:: Int=3)
(2)
function predictErr(CM::Union{Matrix{R}, Matrix{S}};
		scoring	:: Symbol = :b,
		digits	:: Int=3) where {R<:Real, S<:Int}
```

Return the complement of the predicted accuracy, that is, 1.0 minus
the result of [`predictAcc`](@ref), given

- (1) the integer vectors of true labels `yTrue` and of predicted labels `yPred`, or
- (2) a confusion matrix.

**See** [`predictAcc`](@ref), [`confusionMat`](@ref)
"""
predictErr(yTrue::IntVector, yPred::IntVector;
			scoring::Symbol = :b,
	        digits::Int=3) =
	return (acc=predictAcc(yTrue, yPred;
				scoring=scoring, digits=8))≠nothing ? round(1.0-acc;
													  digits=digits) : nothing
predictErr(CM:: Union{Matrix{R}, Matrix{S}};
			scoring:: Symbol = :b,
			digits::Int=3) where {R<:Real, S<:Int} =
	return (acc=predictAcc(CM;
				scoring=scoring, digits=8))≠nothing ? round(1.0-acc;
													  digits=digits) : nothing


"""
```julia
function binaryloss(yTrue::IntVector, yPred::IntVector)
```
Binary error loss given a vector of true labels `yTrue` and a vector of predicted labels `yPred`.
These two vectors must have the same size.
The error loss is 1 if the corresponding labels are different, zero otherwise.
Return a BitVector, that is, a vector of booleans.

**See** [`predict`](@ref).

**Examples**

```julia
using PosDefManifoldML, Random
dummy1, dummy2, yTr, yPr=gen2ClassData(2, 10, 10, 10, 10, 0.1);
shuffle!(yPr)
[yTr yPr binaryloss(yTr, yPr)]
```
"""
binaryloss(yTrue::IntVector, yPred::IntVector) = yTrue .!= yPred

