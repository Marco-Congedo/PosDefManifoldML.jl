#   Unit "stats_inferential.jl" of the PosDefManifoldML Package for Julia language

#   MIT License
#   Copyright (c) 2019-2025
#   Marco Congedo, CNRS, Grenoble, France:
#   https://sites.google.com/site/marcocongedo/home

# ? CONTENTS :
#   This unit implements hypothesis tests for predictions.

"""
```julia
(1) function testCV(cv:CVresult, 
			expected::Union{Float64, Symbol} = :randomchance)

(2) function testCV(𝐂::Vector{Vector{Int}}; 
			expected::Union{Float64, Symbol} = :randomchance)

```
(1) Given a [`CVres`](@ref) stucture (which is of type `CVresult`)
obtained calling the [`crval`](@ref) method, 
performs Bayles's test (Bayles et *al.*, 2020[🎓](@ref)) 
on the difference in distribution of the average binary error losses obtained in the 
cross-validation as compared to a given `expected` average error loss. 
This can be used to test whether the performance obtained in the cross-validation
is superior to a specified chance level.

(2) As method (1) but giving as first argument a vector of `f` confusion matrices, 
The format of the confusion matrices (`Vector{Vector{Int}}`) is the same 
used to store the confusion matrices in a [`CVres`](@ref) stucture.

The test is always directional. Denoting ``μ`` the average loss and ``E`` the
expected average, the test if performed according to null hypethesis ``H_0: μ = E``
against alternative ``μ < E``; if the test is significant, the error loss
is statistically inferior to the specified expected level, which means that the 
observed accuracy is statistically **superior** to the expected accuracy.

By default, `expected` is set to ``1-\\frac{1}{z}``, where ``z`` is the number of classes.
You can pass another value, which must be a real number.

Return the 3-tuple (*z, p, ase*) holding, the ``z`` test-statistic (a standard deviate), 
the ``p``-value and the asymptotic standard error.

!!! warning "Class Inbalance"
	The error lost reflects the accuracy, not the balance accuracy. If the class are inbalanced,
	the test will be driven by the majority classes. See also the documentation on [`crval`](@ref).

**Examples**:
```julia
using PosDefManifoldML

# Generate a set of random data
P, _dummyP, y, _dummyy = gen2ClassData(10, 60, 80, 30, 40, 0.1)

# Perform a cross-validation
cv = crval(MDM(Fisher), P, y; scoring=:a)

# The `z` and `p` values are printed out as they are computed automatically
# and stored in the created structure. To compute it yourself:

z, p, ase = testCV(cv)

```
"""
function testCV(𝐂::Vector{Matrix{Int}}; 
					expected::Union{Float64, Symbol} = :randomchance)

    z = size(𝐂[1], 1) # number of classes
    k = length(𝐂) # number of folds
    
    # number of instances ∀f
    𝐧 = sum.(𝐂)

    # total number of instances
    N = sum(𝐧)

    # expected error loss
    ex = expected == :randomchance ? 1.0-(1/z) : expected

    𝐞 = ((𝐧.-tr.(𝐂))./𝐧) # mean error loss difference

    σ = sqrt((1/k) * sum((𝐧./(𝐧.-1)).*(𝐞.*(1.0.-𝐞))))
    #σ = sqrt((1/k) * sum((n/(n-1)) * (e*(1.0-e)) for (n, e) in (𝐧, 𝐞)))
    e = mean(𝐞)
    # println("e ", e);  println("σ ", σ )

    z = (e-ex) / (σ/sqrt(N))
    return z, _pvalue(PermutationTests.Left(), z), σ # z ~ N(0, 1) 
end


testCV(cv::CVresult; expected = :randomchance) = testCV(cv.cnfs; expected)


"""
```julia
(1) function testCV(cv1::CVresult, cv2::CVresult; 
               	direction = PermutationTests.Both())

(2) function testCV(𝐞₁::Vector{BitVector}, 𝐞₂::Vector{BitVector}; 
               	direction = PermutationTests.Both())

```

(1) Given two [`CVres`](@ref) stuctures `cv1` and `cv2`, which can be obtaoined 
calling the [`crval`](@ref) method,
performs Bayles's test (Bayles et *al.*, 2020[🎓](@ref)) on the difference in distribution 
of the average binary error losses obtained in cross-validations `cv1` and `cv2`. 
This can be used to compare the performance
obtained by different machine learning models and/or different pre-processing,
processing or pre-conditoning pipelines on the same data. 

!!! warning "Be careful"
    When you compare two models and/or pipelines with cross-validation, 
	do not use keyword argument `shuffle=true` in [`crval`](@ref), as in this case 
	the actual data composing the folds would not be identical. Also,
	if you use the `seed` keyword argument, make sure the same seed is employed
	for the two cross-validations.

(2) The function also accepts as argument the error losses directly, 
denoted `𝐞₁` and `𝐞₂` in method (2) here above. 
In this case 𝐞₁ and 𝐞₂ are vectors holding the 
vectors of error losses obtained at each fold. 
The format is the same used to store the error losses in 
a [`CVres`](@ref) stucture.

Denoting ``μ_1`` and ``μ_2`` the average loss for `cv1` and `cv2` (or `𝐞₁` and `𝐞₂`), 
respectively, the `direction` keyword argument determines the test directionality:

- ``H_1: μ_1 > μ_2`` (use `direction=PermutationTests.Right()`)
- ``H_1: μ_1 < μ_2`` (use `direction=PermutationTests.Left()`)
- ``H_1: μ_1 <> μ_2`` (use `direction=PermutationTests.Both()`, the default)

In al cases ``H_0: μ_1 = μ_2``.

!!! tip "Test direction in terms of accuracy"
	Note that if the test is significant with `direction=PermutationTests.Right()`,
	the  error loss of `cv1`(or `𝐞₁`) is statistically superior to the error loss of `cv2`(or `𝐞₂`),
	which means that the accuracy in `cv1`(or `𝐞₁`) is statistically **inferior** to the accuracy 
	in `cv2`(or `𝐞₂`).

Return the 3-tuple (*z, p, ase*) holding, the ``z`` test-statistic (a standard deviate), 
the ``p``-value and the asymptotic standard error.

!!! warning "Class Inbalance"
	The error lost reflects the accuracy, not the balance accuracy. If the class are inbalanced,
	the test will be driven by the majority classes. See also the documentation on [`crval`](@ref).

**Examples**:
```julia
using PosDefManifoldML

## Test the performance of the same model and pipeline on different data:

# Generate two sets of random data with the same distribution
P1, _dummyP, y1, _dummyy = gen2ClassData(10, 60, 80, 30, 40, 0.01)
P2, _dummyP, y2, _dummyy = gen2ClassData(10, 60, 80, 30, 40, 0.01)

# Perform a cross-validation on the two sets of data
cv1 = crval(MDM(Fisher), P1, y1; scoring=:a)
cv2 = crval(MDM(Fisher), P2, y2; scoring=:a)

z, p, ase = testCV(cv1, cv2) # two-sided test

# This is equivalent to
z, p, ase = testCV(cv1.losses, cv2.losses)

## Test the performance of, for example, different metrics on the same data:

# Generate two sets of random data with the same distribution
P, _dummyP, y, _dummyy = gen2ClassData(10, 40, 40, 30, 40, 0.15)

cv1 = crval(MDM(logEuclidean), P, y; scoring=:a)
cv2 = crval(MDM(Wasserstein), P, y; scoring=:a)

z, p, ase = testCV(cv1, cv2) # two-sided test

```
"""
function testCV(𝐞₁::Vector{BitVector}, 𝐞₂::Vector{BitVector}; corrected=true,
                    direction=PermutationTests.Both())

    𝐝 = [e₁-e₂ for (e₁, e₂) in zip(𝐞₁, 𝐞₂)] # error loss differences in each fold
    N = sum(length(d) for d in 𝐝) # total number of samples
    δ = (1/N) * sum(sum(d) for d in 𝐝)
    σ = sqrt(mean(var(d; corrected) for d in 𝐝))
    # println("δ ", δ);  println("σ  ", σ )

    z = δ / (σ/sqrt(N))
    return z, _pvalue(direction, z), σ # z ~ N(0, 1) 
end


testCV(cv1::CVresult, cv2::CVresult; corrected=true, direction = PermutationTests.Both()) = 
		testCV(cv1.losses, cv2.losses; corrected, direction)

# private

_pvalue(::PermutationTests.Both, z) = cdf(Normal(), -abs(z)) + ccdf(Normal(), abs(z))
_pvalue(::PermutationTests.Left, z) = cdf(Normal(), z)
_pvalue(::PermutationTests.Right, z) = ccdf(Normal(), z)
