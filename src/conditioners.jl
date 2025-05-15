#   Unit "conditioners.jl" of the PosDefManifoldML Package for Julia language

#   MIT License
#   Copyright (c) 2025
#   Marco Congedo, CNRS, Grenoble, France:
#   https://sites.google.com/site/marcocongedo/home

# ? CONTENTS :
#   This unit implements (pre) conditioners for fast Riemannian 
#   machine learning classifier using package PosDefManifold.

include("conditioners_lowlevel.jl")

#= To be included on the top of the doc file on conditioners:
Note that if you only need to fit a conditioner, you can simply create it on the fly
as argument of the [`fit!`](@ref) function - see examples therein.
If you want to apply the conditioner to training and testing (validation) data,
you need instead to first create the conditioner and then fit it
passing the created conditioner as argument.
This is because the learnt parameters will be needed for the transformation of the testing data.
=#

#=

"""
```julia
    abstract type Tikhonov    <: Conditioner end # Tikhonov Regularization
    abstract type Recentering <: Conditioner end # Recentering with or w/o dim reduction
    abstract type Compressing <: Conditioner end # Compressing (global scaling)
    abstract type Equalizing  <: Conditioner end # Equalizing  (individual scaling)
    abstract type Shrinking   <: Conditioner end # Geodesic Shrinking
```
Abstract types for **Conditioners**. Those are the elemntary pipes
to build a [Pipeline](@ref). The available conditioners are

- Tikhonov regularization (diagonal loading)
- Recentering by whitening with or w/o dimensionality reduction
- Compressing by global scaling
- Equalizing by individual scaling
- Shrinking by moving along geodesics towards the identity matrix
"""
=#

###########################################################################
# Structures
###########################################################################

"""
```julia
mutable struct Tikhonov <: Conditioner
    α
    threaded 
```

Mutable structure for the **Tikhonov regularization** conditioner. 

Given a set of points ``𝐏`` in the manifold of positive-definite matrices,
transforms the set such as 

 ``P_j+αI, \\ j=1,...,k``,

where ``I`` is the identity matrix and ``α`` is a non-negative number.

This conditoner structure has only two fields: 

- `α`, which is written in the structure when it is 'fitted' to some data.

- `threaded`, to detrmine if the transformation is done in multi-threading mode (true by default)

This conditioner is not data-driven: the `α` parameter must be given
explicitly upon construction (it is zero by default).

**Examples**:
```julia
using PosDefManifoldML, PosDefManifold

# create a conditioner
T = Tikhonov(0.001)
```
**See also**: [`fit!`](@ref), [`transform!`](@ref), [`crval`](@ref).
"""
mutable struct Tikhonov <: Conditioner
    α 
    threaded
    function Tikhonov(α::Union{Float64, Int} = 0.0;
                        threaded:: Bool = true)
        # constructor 
        errhead = "Tikhonov conditioner constructor: "
        α<0 && throw(ArgumentError(errhead*"the argument (α) must be non-negative"))
        new(α, threaded)
    end
end


"""
```julia
mutable struct Recenter <: Conditioner
    metric 
    eVar 
    w 
    ✓w 
    init 
    tol 
    verbose 
    forcediag 
    threaded 
    ## Fitted parameters
    Z 
    iZ 
```

Mutable structure for the **recentering** conditioner. 

Given a set of points ``𝐏`` in the manifold of positive-definite matrices,
transforms the set such as 

 ``ZP_jZ^T, \\ j=1,...,k``,

where ``Z`` is the whitening matrix of the barycenter of ``𝐏`` as specified by the conditioner,
*i.e.*, if ``G`` is the barycenter of `𝐏`, then ``ZGZ^T=I``.

After recentering the barycenter becomes the identity matrix and the mean of the eigenvalues of the
whitened matrices is 1. In the manifold of positive-definite matrices, recentering is equivalent to
parallel transport of all points to the identity barycenter.

Depending on the `eVar` value used to define the [`Recenter`](@ref) conditioner,
matrices ``Z`` may determine a dimensionality reduction of the input points as well.
In this case ``Z`` is not square, but a wide matrix.

This conditoner may behave in a supervised way; providing the class labels 
when it is fitted (see [`fit!`](@ref)), the classes are equally weighted to compute
the barycenter ``G``, like [`tsWeights`](@ref)
does for computing the barycenter used for tangent space mapping.
If the classes are balanced, the weighting has no effect.

This conditoner structure has the following fields:

- `.metric`, of type  [Metric](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#Metric::Enumerated-type-1), is to be specified by the user. It is the metric that will be adopted to compute the class means and the distances to the mean. default: `PosDefManifold.Euclidean`.

- `.eVar`, the desired explained variance for the whitening. It can be a Real, Int or `nothing`. See the documentation of method [whitening](https://marco-congedo.github.io/Diagonalizations.jl/dev/whitening/#Diagonalizations.whitening) in Diagonalizations.jl. It is 0.9999 by default.

- Fields `.w`, `.✓w`, `.init` and `.tol` are passed to the [mean](https://marco-congedo.github.io/PosDefManifold.jl/latest/riemannianGeometry/#Statistics.mean) method of PosDefManifold.jl for computing the barycenter ``G``. Refer to the documentation therein for details.

- `.verbose` is a boolean determining if information is to be printed to the REPL when using `fit!` and `transform!` with this conditioner. It is false by default.

- `.forcediag` is a boolean for forcing diagonalization. It is true by default. If false, whitening is carried out only if a dimensionality reduction is needed, as determined by `eVar`.

- If `.threaded` is true (default), all operations are multi-threaded.

**Fitted parameters**

When the conditioner is fitted, the following fields are written:

- `.Z`, the whitening matrix of the fitted set ``P_j, \\ j=1:k``, such that ``ZP_jZ^T`` is whitened;

- `.iZ`, the left inverse ``Z^*`` of Z, such that ``Z^*Z=I`` (identity matrix) if no dimensionality reduction is operated.
 If no dimensionality reduction is operated, ``Z^*Z`` will be smaller then ``I`` in the Löwner sense.

**Examples**:
```julia
using PosDefManifoldML, PosDefManifold

# create a default conditioner
R = Recenter(PosDefManifold.Euclidean)

# since the Euclidean metric is the default metric,
# this is equivalent to
R = Recenter()

# do not perform dimensionality reduction
R = Recenter(PosDefManifold.Fisher; eVar=nothing)

# use class labels to balance the weights across classes
# let `y` be a vector of int holding the class labels
R = Recenter(PosDefManifold.Fisher; labels=y)

```
**See also**: [`fit!`](@ref), [`transform!`](@ref), [`crval`](@ref).
"""
mutable struct Recenter <: Conditioner
    metric :: PosDefManifold.Metric
    eVar # Explained variance: Real, Int or nothing
    w # weights for mean computation
    ✓w # boolean for checking weights normalization
    init # initial mean for iterative algorithms
    tol # tolerance for mean iterative algorithms
    verbose # boolean for printing info in the REPL
    forcediag # boolean for forcing diagonalization
    threaded # boolean for using multi-threading
    ## Fitted parameters
    Z # Whitening matrix such that Q[i]=Z*P[i]*Z' is whitened
    iZ # Inverse of Z such that P[i]=iZ*Q[i]*iZ' if no dimensionality reduction is operated
    function Recenter(metric    :: PosDefManifold.Metric = PosDefManifold.Euclidean;
                eVar            :: Union{T, Int, Nothing} = 0.9999,
                w               :: Vector{T} = Float64[],
                ✓w              :: Bool = true,
                init            :: Union{Hermitian, Nothing} = nothing,
                tol             :: Float64 = 0.0,
                verbose         :: Bool = true,
                forcediag       :: Bool = true,
                threaded        :: Bool = true,
                Z = nothing,
                iZ = nothing) where T<:Real
        # constructor
        errhead = "Recenter conditioner constructor: "
        eVar<0 && throw(ArgumentError(errhead*"eVar must be non-negative"))
        tol<0 && throw(ArgumentError(errhead*"tol must be non-negative"))
        new(metric, eVar, w, ✓w, init, tol, verbose, forcediag, threaded, Z, iZ)
    end
end


"""
```julia
mutable struct Compress <: Conditioner
    threaded
    β 
end
```

Mutable structure for the **compressing** conditioner.

Given a set of points ``𝐏`` in the manifold of positive-definite matrices,
transforms the set such as 

 ``βP_j, \\ j=1,...,k``,

where ``β`` is chosen to minimize the average Euclidean norm of the transformed set,
*i.e.*, the average distance to the identity matrix according to the specifies metric.

Since the Euclidean norm is the Euclidean distance to the identity, compressing 
a recentered set of points minimizes the average dispersion of the set around 
the identity, thus it should be performed after conditioner [`Recenter`](@ref). 

The structure has one field only:

- If `threaded` is true (default), all operations are multi-threaded.

**Fitted parameters**

When the conditioner is fitted, the following field is written:

- `.β`, a positive scalar minimizing the average Euclidean norm of the fitted set.

**Examples**:
```julia
using PosDefManifoldML, PosDefManifold

# create the conditioner
C = Compress()
```
**See also**: [`fit!`](@ref), [`transform!`](@ref), [`crval`](@ref).
"""
mutable struct Compress <: Conditioner
    threaded
    ## Fitted parameters
    β # global scaling minimizing the Euclidean norm
    function Compress(; threaded::Bool = true, 
                        β = nothing)
        # constructor
        new(threaded, β)
    end
end


"""
```julia
mutable struct Equalize <: Conditioner
    threaded
    β 
end
```
Mutable structure for the **equalizing** conditioner.

Given a set of points ``𝐏`` in the manifold of positive-definite matrices,
transforms the set such as 

 ``β_jP_j, \\ j=1,...,k``,

where the elements ``β_j`` are chosen so as to minimize the Euclidean norm 
of the transformed matrices individually.

Since the Euclidean norm is the Euclidean distance to the identity, equalizing 
a recentered set of points minimizes the average dispersion of the set around 
the identity, thus it should be performed after conditioner [`Recenter`](@ref). 

As compared to compression, equalization is more effective for reducing the distance 
to the identity, however it is not an isometry.

Also, in contrast to compression, the transformation of the 
matrices in set ``𝐏`` is individual, so fitting equalization does not imply 
a learning process - see [`fit!`](@ref).

If `threaded` is true (default), all operations are multi-threaded.

**Fitted parameters**

When the conditioner is fitted, the following field is written:

- `.β`, a vector of positive scalars minimizing the Euclidean norm individually for each matrix in the fitted set. 

**Examples**:
```julia
using PosDefManifoldML, PosDefManifold

# create the conditioner
E = Equalize()
```
**See also**: [`fit!`](@ref), [`transform!`](@ref), [`crval`](@ref).
"""
mutable struct Equalize <: Conditioner
    threaded
    ## Fitted parameters
    β # individual scaling minimizing the Euclidean norm
    function Equalize(; threaded::Bool = true, 
                        β = nothing)
        # constructor
        new(threaded, β)
    end
end


"""
```julia
mutable struct Shrink <: Conditioner
        metric 
        radius 
        refpoint 
        reshape 
        epsilon 
        verbose 
        threaded 
        ## Fitted parameters
        γ 
        m 
        sd 
```

Mutable structure of the **geodesic shrinking** conditioner. 

Given a set of points ``𝐏`` in the manifold of positive-definite matrices,
this conditioner moves all points towards the identity matrix ``I`` along geodesics
on the manifold defined in accordance to the specified `metric`. This effectively defines a ball
centered at ``I``.

The step-size ``γ`` of the geodesics from ``I`` to each point ``P`` in ``𝐏`` is given by

``\\gamma=\\frac{r\\sqrt{n}}{δ(P, I) + ϵ}``

where ``r`` is the `radius` argument, ``n`` is the dimension of ``P``, 
``δ(P, I)`` is the norm of ``P`` according to the specified `metric` and ``ϵ`` is an optional 
small positive number given as argument `epsilon`.

The conditioner has the following fields:

`.metric`, of type
[Metric](https://marco-congedo.github.io/PosDefManifold.jl/dev/MainModule/#Metric::Enumerated-type-1),
Default: `PosDefManifold.Euclidean`.

After shrinking, the set of points ``𝐏`` acquires a sought `.radius` (default: 0.02).
This is a measure of the distances from the identity (norms), specifically, the maximum distance 
if `.refpoint`=:max, the mean eccentricity if `.refpoint`=:mean (default). 
In the first case it defines a ball confining all points, with radius equal
to the maximum distance from the identity of the transformed points + ``ϵ``. 
In the second case, the actual radius of the ball is equal to 

``\\sqrt{\\frac{1}{n}\\sum_{j=1}^{k}δ(P_j, I) + ϵ}``.

`.reshape` is a boolean for reshaping the eigenvalues of the set ``𝐏`` after shrinking. 
It applies only to the Fisher (affine-invariant) metric. Default: false. See below.

`.epsilon`, a non-negative real number, the ``ϵ`` above. Default: 0.0.

`.verbose`, a boolean. If true, information is printed in the REPL. Default: false

`.threaded`, a boolean for using multi-threading. Default: true

**Fitted parameters**

When the conditioner is fitted, the following fields are written:

`.γ` step-size for geodesic (according to `metric`) from ``I`` to the each matrix in ``𝐏``.

`.m` and `.sd`, mean and standard deviation of the eigenvalues of the set after shrinking.
This is used for reshaping, which applies only if the Fisher metric is adopted.
Reshaping is meaningful only if the input set has been recentered (see [`Recenter`](@ref)).
It recenters again the eigenvalues of the set after shrinking (mean = 1), 
and normalize them so as to have standard deviation equal to `.radius`.

**Examples**:
```julia
using PosDefManifoldML, PosDefManifold

# create a conditioner adopting the Fisher Metric
S = Shrink(PosDefManifold.Fisher; reshape = true)
```
**See also**: [`fit!`](@ref), [`transform!`](@ref), [`crval`](@ref).
"""
mutable struct Shrink <: Conditioner
    metric :: PosDefManifold.Metric
    radius # Radius of the ball, Real or Int
    refpoint # point defining the radius, :mean or :max
    reshape # boolean for reshaping the eigvals. Applies only to the Fisher metric
    epsilon # added to radius, real
    verbose # boolean for printing info in the REPL
    threaded # boolean for using multi-threading
    ## Fitted parameteres
    γ # step-size for geodesic (according to metric) from I to the matrix
    m # mean of eigvals after shrinking (for reshaping)
    sd # standard deviation of eigvals after shrinking (for reshaping) 
    function Shrink(metric  :: PosDefManifold.Metric = PosDefManifold.Euclidean;
                radius      :: Union{Float64, Int} = 0.02,
                refpoint    :: Symbol = :mean,
                reshape     :: Bool = false,
                epsilon     :: Float64 = 0.0,
                verbose     :: Bool = true,
                threaded    :: Bool = true,
                γ           = nothing,
                m           = nothing,
                sd          = nothing)
        # constructor
        errhead = "Shring conditioner constructor: "
        metric == PosDefManifold.VonNeumann && throw(ArgumentError(errhead*"shrinking is not defined adopting metric $(metric)"))
        metric ∈ (PosDefManifold.logdet0, PosDefManifold.Jeffrey) && @warn errhead*"shrinking can be achieved only approximately adopting metric $(metric)"
        refpoint ∈ (:max, :mean) || throw(ArgumentError(errhead*"`refpoint` must be :max or :mean"))
        epsilon<0 && throw(ArgumentError(errhead*"`epsilon` must be non-negative"))
        new(metric, radius, refpoint, reshape, epsilon, verbose, threaded, γ, m, sd)
    end
end

###########################################################################
# Pipeline (Tuple of Conditioner instances)
###########################################################################

"""
`Pipeline` is a type for tuples holding conditioners.

A pipeline holds a sequence of conditioners learned
and optionally applied using [`fit!`](@ref). It can be 
subsequently applied on other data using the [`transform!`](@ref) function.
All `fit!` methods return a pipeline.

Pipelines comprising a single conditioner are allowed.

Pipelines can be saved to a file using the [`saveas`](@ref) function
and loaded from a file using the [`load`](@ref) function.

Note that in Julia tuples are immutable, thus it is not possible to modify
a pipeline.

In order to create a pipeline use the [`@pipeline`](@ref) macro.

**See also**: [`fit!`](@ref), [`transform!`](@ref).
"""
struct Pipeline{T<:Tuple}
    pipe::T
    function Pipeline(pipe::T) where {T<:Tuple}
        all(isa(e, Conditioner) for e in pipe) || error("All elements of a Pipeline tuple must be of type Conditioner")
        new{T}(pipe)
    end
end


"""
```julia
macro pipeline(args...)
```

Create a [`Pipeline`](@ref) chaining the provided expressions.

As an example, the sintax is:

```julia
p = @pipeline Recenter() → Compress → Shrink(Fisher; threaded=false)
```

Note that:

- The symbol → (escape "\\to") separating the conditioners is optional.

- This macro has alias `@→`. 

- As in the example above, expressions may be instantiated conditioners, like `Recenter()`, or their type, like `Compress`, in which case the default conditioner of that type is created.

The example above is thus equivalent to

```julia
    p = @→ Recenter() Compress() Shrink(Fisher; threaded=false)
```
Conditioners are not callable by the macro. Thus if you want to pass a variable, do not write

```julia
    R = Recenter()
    p = @→ R
```
but 

```julia
    R = Recenter()
    p = @→ eval(R)
```

**Available conditioners to form pipelines**

[`Tikhonov`](@ref), [`Recenter`](@ref), [`Compress`](@ref), [`Equalize`](@ref), [`Shrink`](@ref)

**See also**: [`fit!`](@ref), [`transform!`](@ref)

**Examples**
```julia
using PosDefManifoldML, PosDefManifold

P=randP(3, 5)
pipeline = fit!(P, @→ Recenter → Compress)
```
"""
macro pipeline(args...)
    # Recursively extract valid args by skipping any `→` expressions
    function extract_args(expr)
        if expr == :→
            return []
        elseif isa(expr, Expr) && expr.head == :call && expr.args[1] == :→
            # Recurse into both sides of the → expression
            return vcat(extract_args(expr.args[2]), extract_args(expr.args[3]))
        else
            return [expr]
        end
    end

    # Flatten and filter arguments
    flat_args = reduce(vcat, extract_args.(args))

    # Auto-instantiate types or keep full expressions
    processed_args = [
        isa(arg, Symbol) || (isa(arg, Expr) && arg.head == :curly) ? :( $(esc(arg))() ) : esc(arg)
        for arg in flat_args
    ]

    # Runtime check for Conditioner constraint
    check_expr = :(all(isa(e, Conditioner) for e in ($(processed_args...),)) || error("All elements must be of type Conditioner"))
    build_expr = :(Pipeline(($(processed_args...),)))

    return quote
        $check_expr
        $build_expr
    end
end

macro →(args...)
    return esc(:(@pipeline $(args...)))
end

Base.getindex(p::Pipeline, i::Int) = p.pipe[i]
Base.length(p::Pipeline) = length(p.pipe)
Base.iterate(p::Pipeline, s=1) = s > length(p) ? nothing : (p[s], s+1)
isempty(p::Pipeline) = isempty(p.pipe)


###########################################################################
# fit! methods
###########################################################################

"""
```julia
    function fit!(𝐏 :: ℍVector, pipeline :: Union{Pipeline, Conditioner}; 
        transform = true,
        labels = nothing)
```

Fit the given `pipeline` or `Conditioner` to ``𝐏`` and return a fitted [`Pipeline`](@ref) object.
``𝐏`` must be of [the ℍVector type](@ref).

A single `Conditioner` can be given as argument instead of a pipeline;
a fitted pipeline with a single element will be returned.
The type of the conditioner can be gives as well, in which case
the default conditioner will be used - see examples below.

If `pipeline` in an empty tuple, return an empty pipeline without doing anything.

if `transform` is true (default), ``𝐏`` is transformed (in-place),
otherwise the pipeline is fitted but ``𝐏`` is not transformed.

If `labels` is a vector of integers holding the class labels of the points 
in ``𝐏``, the conditioners are supervised (*i.e.*, labels-aware), 
otherwise, if it is `nothing` (default), they are unsupervised. 
Currently the only conditioners that can behave in a supervised manner
is [`Recenter`](@ref). When supervised, the barycenter for recentering
is computed given balanced weights to each class, like [`tsWeights`](@ref)
does for computing the barycenter used for tangent space mapping.
If the classes are balanced, the weighting has no effect.

The returned pipeline can be used as argument for the 
[`transform!`](@ref) function, ensuring that the fitted parameters are properly applied.
It can also be saved to a file using the [`saveas`](@ref) function
and loaded from a file using the [`load`](@ref) function.

Note that the pipeline given as argument is not modified.

**Learning parameters during fit**

For some of the conditioners there is no parameter to be learnt during training. 
For those, a call to the `fit!` function is equivalent to a call to the [`transform!`](@ref) function,
with the exception that when the `fit!` function is called the parameters used for the tranformation
are stored in the returned pipeline.

**See also**: [`transform!`](@ref)

**Examples**
```julia
using PosDefManifoldML, PosDefManifold

## Example 1 (single conditioner): 

# generate some data
P=randP(3, 5) # 5 random 3x3 Hermitian matrices
Q=copy(P)

# fit the default recentering conditioner (whitening)
pipeline = fit!(P, Recenter) 

# This is equivalent to
pipeline = fit!(Q, Recenter())

pipeline[1].Z # a learnt parameter (whitening matrix)

## Example 2 (pipeline): 

# Fit a pipeline comprising Tikhonov regularization, 
# recentering, compressing and shrinking according to the Fisher metric.
# The matrices in P will be first regularized, then recentered, 
# then compressed and finally shrunk.

P=randP(3, 5)  
Q=copy(P)

pipeline = fit!(P, 
        @→ Tikhonov(0.0001) → Recenter → Compress → Shrink(Fisher; radius=0.01))

# or 
pipeline = fit!(Q, @→ Recenter Compress Shrink(Fisher; radius=0.01))

# the whitening matrices of the the recentering conditioner,
pipeline[1].Z

# the scaling factors of the compressing conditioner,
pipeline[2].β

# and the step-size of the shrinking conditioner,
pipeline[3]

## Example 3 (pipeline with a single conditioner):
P=randP(3, 5)  
pipeline = fit!(P, @→ Recenter)

```
"""
function fit!(𝐏 :: ℍVector, conditioner :: Tikhonov; 
                transform :: Bool = true,
                labels :: Union{IntVector, Nothing} = nothing) 
    c = deepcopy(conditioner)
    𝐏 = tikhonov!(𝐏, c.α; transform, c.threaded)
    return Pipeline((c,))
end


function fit!(𝐏 :: ℍVector, conditioner :: Recenter; 
                transform :: Bool = true,
                labels :: Union{IntVector, Nothing} = nothing) 
    c = deepcopy(conditioner)
    𝐏, c.Z, c.iZ = recenter!(c.metric, 𝐏; transform, labels,
                            c.eVar, c.w, c.✓w, c.init, 
                            c.tol, c.verbose, c.forcediag, c.threaded)                                                    
    return Pipeline((c,))
end

function fit!(𝐏 :: ℍVector, conditioner :: Compress; 
                transform :: Bool = true,
                labels :: Union{IntVector, Nothing} = nothing)
    c = deepcopy(conditioner)
    𝐏, c.β  = compress!(𝐏; transform, labels, c.threaded)
    return Pipeline((c,))
end

function fit!(𝐏 :: ℍVector, conditioner :: Equalize; 
                transform :: Bool = true,
                labels :: Union{IntVector, Nothing} = nothing)
    c = deepcopy(conditioner)
    𝐏, c.β = equalize!(𝐏; transform, labels, c.threaded)
    return Pipeline((c,))
end

function fit!(𝐏 :: ℍVector, conditioner :: Shrink; 
            transform :: Bool = true,
            labels :: Union{IntVector, Nothing} = nothing)
    c = deepcopy(conditioner)
    𝐏, c.γ, c.m, c.sd = shrink!(c.metric, 𝐏, c.radius; 
                                transform, labels, c.refpoint, c.reshape, 
                                c.epsilon, c.verbose, c.threaded)
    return Pipeline((c,))
end


fit!(𝐏 :: ℍVector, T :: Type{<:Conditioner}; 
        transform :: Bool = true,
        labels :: Union{IntVector, Nothing} = nothing) = fit!(𝐏, T(); transform, labels) 

# Fit a pipeline, tranform the data if `transform` is true and return
# the fitted pipeline.
function fit!(𝐏::ℍVector, pipeline::Pipeline; 
                transform :: Bool = true,
                labels :: Union{IntVector, Nothing} = nothing)
    isempty(pipeline) && (return 𝐏, pipeline)
    c = Vector{Conditioner}(undef, length(pipeline))
    for (i, p) in enumerate(pipeline)
        c[i] = fit!(𝐏, p; transform, labels)[1] # a pipeline with one element is created at each pass
    end
    return Pipeline(tuple(c...))
end

###########################################################################
# transform! methods
###########################################################################


"""
```julia
function transform!(𝐐 :: Union{ℍVector, ℍ}, Union{Pipeline, Conditioner})

```

Given a fitted `conditioner` or [`Pipeline`](@ref), transform all matrices 
in ``𝐐`` using the parameters learnt during the fitting process. Return ``𝐐``.

In a training-test setting, a fitted conditioner or pipeline is given as argument 
to these methods to make sure that the parameters learnt during the fitting process 
(on trained data) are used for transforming the (testing) data.
More in general, this function can be used to transform the data in ``𝐐``.

If `pipeline` in an empty tuple, return ``𝐐`` without doing anything.

``𝐐`` can be a single Hermitian matrix or a vector of [the ℍVector type](@ref).
It is transformed in-place.

The dimension of matrix(ces) in ``𝐐`` must be the same of the dimension of the matrices 
used to fit the conditioner or pipeline.

In contrast to the `fit!` function, only instantiated conditioner can be used.
For general use, this is transparent to the user as the [`fit!`](@ref) function
always returns pipelines with instantiated conditioners.

**See**: [`fit!`](@ref)

**Examples**
```julia
using PosDefManifoldML, PosDefManifold

## Example 1 (single conditioner)

# generate some 'training' and 'testing' data
PTr=randP(3, 20) # 20 random 3x3 Hermitian matrices
PTe=randP(3, 5) # 5 random 3x3 Hermitian matrices

# Fit the default recentering conditioner (whitening)
# Matrices in PTr will be transformed (recentered)
R = fit!(PTr, Recenter()) 

# Transform PTe using recentering as above
# using the parameters for recentering learnt
# on PTr during the fitting process. 
transform!(PTe, R)

mean(PTr)-I # Should be close to the zero matrix.
mean(PTe)-I # Should not be close to the zero matrix
# as the recentering parameter is learnt on PTr, not on PTe.

## Example 2 (pipeline)

# Generate some 'training' and 'testing' data
PTr=randP(3, 20) # 20 random 3x3 Hermitian matrices
PTe=randP(3, 5) # 5 random 3x3 Hermitian matrices
QTr=copy(PTr)
QTe=copy(PTe)

p = @→ Tikhonov(0.0002) Recenter(; eVar=0.99) Compress Shrink(Fisher; radius=0.01)
pipeline = fit!(QTr, p)
transform!(QTe, pipeline)

## Example 3 (pipeline with a single conditioner):
P=randP(3, 5)  
# For the Equalize conditioner there is no need to fit some data
transform!(P, @→ Equalize)
# this gives an error as Recenter needs to learn parameters:
transform!(P, @→ Recenter)

```
"""
function transform!(𝐏 :: ℍVector, c :: Tikhonov)
    c.α≈0.0 && warn("The Tikhonov conditioner passed as argument has the α parameter equal to zero or very close to it")
    tikhonov!(𝐏, c.α; transform=true, threaded=c.threaded)
    return 𝐏
end

function transform!(𝐏 :: ℍVector, c :: Recenter)
    c.Z===nothing && throw(ArgumentError("The Recenter conditioner passed as argument has not stored the whitening matrix. Create the conditioner, fit it and only then call this function"))
    recenter!(𝐏, c.Z; threaded=c.threaded) 
    return 𝐏
end

function transform!(P :: ℍ, c :: Recenter)
    c.Z===nothing && throw(ArgumentError("The Recenter conditioner passed as argument has not stored the whitening matrix. Create the conditioner, fit it and only then call this function"))
    return P.data.=c.Z*P*c.Z'
end

function transform!(𝐏 :: ℍVector, c :: Compress)
    c.β===nothing && throw(ArgumentError("The Compress conditioner passed as argument has not stored the scaling factor. Create the conditioner, fit it and only then call this function"))
    compress!(𝐏, c.β; threaded=c.threaded) 
    return 𝐏
end

function transform!(P :: ℍ, c :: Compress)
    c.β===nothing && throw(ArgumentError("The Compress conditioner passed as argument has not stored the scaling factor. Create the conditioner, fit it and only then call this function"))
    return P.data.*=c.β
end

function transform!(𝐏 :: ℍVector, c :: Equalize)
    equalize!(𝐏; c.threaded)
    return 𝐏
end

function transform!(P :: ℍ, c :: Equalize)
    return P.data.*=(tr(P) / PosDefManifold.sumOfSqr(P))
end

function _checkFisherConditioner(c::Shrink)
    c.γ===nothing && throw(ArgumentError("The Shrink conditioner passed as argument has not stored the step-size for geodesic shrinking. Create the conditioner, fit it and only then call this function"))
    if c.reshape && c.metric==PosDefManifold.Fisher 
        c.m===nothing && throw(ArgumentError("The Shrink conditioner passed as argument has not stored the m (mean) parameter for reshaping. Create the conditioner, fit it and only then call this function"))
        c.sd===nothing && throw(ArgumentError("The Shrink conditioner passed as argument has not stored the sd (standard deviation) parameter for reshaping. Create the conditioner, fit it and only then call this function"))
    end
    
end

function transform!(𝐏::ℍVector, c::Shrink)
    _checkFisherConditioner(c)
    shrink!(c.metric, 𝐏, c.γ, c.radius, c.m, c.sd, c.reshape; threaded=c.threaded)
    return 𝐏
end

function transform!(P::ℍ, c::Shrink)
    _checkFisherConditioner(c)
    return P.data.=getShrinkedP(P, c.γ, c.radius, c.m, c.sd, c.reshape).data
end

transform!(𝐏::Union{ℍ, ℍVector}, c::Type{<:Conditioner}) = 
    transform!(𝐏, c()) 

function transform!(𝐏::Union{ℍ, ℍVector}, pipeline::Pipeline)
    isempty(pipeline) && (return 𝐏)
    for p in pipeline
        p isa Conditioner || throw(ArgumentError("transform! function: element $p given as `pipeline` argument is not an instance of the Conditioner type"))
        transform!(𝐏, p)
    end
    #foreach(p -> transform!(𝐏, p), pipeline)
    return 𝐏
end    

############################################################################
# Tools
############################################################################


"""
```julia
function pickfirst(pipeline, conditioner)
```
Return a copy of the first conditioner of the `pipeline` which is of 
the same type as `conditioner`.
If no such conditioner is found, return `nothing`.
Note that a copy is returned, not the conditioner in the pipeline itself.

The provided `conditioner` can be a type or an instance of a conditioner.
The returned element will always be an instance, as pipelines holds instances only.

**See**: [`includes`](@ref)

**Examples**
```julia
using PosDefManifoldML

pipeline = @→ Recenter() Shrink()
S = pickfirst(pipeline, Shrink) 
S isa Conditioner # true
S isa Shrink # true

# retrive a parameter of the conditioner
pickfirst(pipeline, Shrink).radius

```
"""
function pickfirst(pipeline::Pipeline, 
                conditioner::Union{Conditioner, Type{T}}) where T<:Conditioner

    isempty(pipeline) && (return nothing)
    c = conditioner isa Conditioner ? typeof(conditioner) : conditioner
    for p in pipeline
        typeof(p)===c && (return deepcopy(p))
    end
    return nothing
end


"""
```julia
function includes(pipeline, conditioner)
```
Return true if the given [`Pipeline`](@ref) includes a conditioner 
of the same type as `conditioner`.

The provided `conditioner` can be a type or an instance of a conditioner.

**See**: [`pickfirst`](@ref), [`@pipeline`](@ref)

**Examples**
```julia
using PosDefManifoldML

pipeline= @→ Recenter() → Shrink()

includes(pipeline, Shrink) # true

includes(pipeline, Shrink()) # true

# same type, althoug a different instance
includes(pipeline, Shrink(Fisher; radius=0.1)) # true

includes(pipeline, Compress) # false
```

**Learn the package**: check out [`saveas`](@ref)
"""
includes(pipeline::Pipeline, 
            conditioner::Union{Conditioner, Type{T}}) where T<:Conditioner =
        !(pickfirst(pipeline, conditioner)===nothing)


"""
```julia
function dim(pipeline::Pipeline)
```
Return the dimension determined by a fitted [`Recenter`](@ref) pre-conditioner 
if the `pipeline` comprises such a pre-conditioner, `nothing` otherwise. 
This is used to adapt pipelines - see the documentation of the [`fit!`](@ref)
function for ENLR machine learning models for an example.

**Examples**
```julia
using PosDefManifoldML, PosDefManifold

pipeline = @→ Recenter(; eVar=0.9) → Shrink()
dim(pipeline) # return false, as it is not fitted

P = randP(10, 5)
p = fit!(P, pipeline)
dim(p) # return an integer ≤ 10
```

**Learn the package**: check out [`@pipeline`](@ref)
"""
function dim(pipeline::Pipeline)
    p = pickfirst(pipeline, Recenter)
    return  p===nothing     ? nothing       : 
            p.Z isa Matrix  ? size(p.Z, 1)  : 
            nothing
end


# ++++++++++++++++++++  Show override  +++++++++++++++++++ # (REPL output)
function Base.show(io::IO, ::MIME{Symbol("text/plain")}, p::Pipeline)        
    println(io, titleFont, "\n→ Pipeline with $(length(p.pipe)) element(s):")
    println(io, separatorFont, "⭒  ⭒    ⭒       ⭒          ⭒", defaultFont)
    for (i, val) in enumerate(p.pipe)
        println(io, " ($i) → ", val)
    end
end

