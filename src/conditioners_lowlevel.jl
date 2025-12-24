#   Unit "conditioners_lowlevel.jl" of the PosDefManifoldML Package for Julia language

#   MIT License
#   Copyright (c) 2025
#   Marco Congedo, CNRS, Grenoble, France:
#   https://sites.google.com/site/marcocongedo/home

# ? CONTENTS :
#   This unit holds low-level code for the (pre) conditioners implemented 
#   in unit "conditioners.jl". None of the functions in thus unit are public



# TRAINING and TESTINg Tikhovov regularization: add a small epsilon to diagonal elements 
# of each matric in input set 𝐂. By default return the input set unchanged.
function tikhonov!(𝐂::PosDefManifold.HermitianVector, tikh=0; 
                    transform=true, labels :: Union{IntVector, Nothing} = nothing, threaded=true)
    if tikh > 0 && transform
        ϵI = Hermitian(Matrix{eltype(𝐂[1])}(tikh*I, size(𝐂[1])))
        if threaded 
            @threads for i ∈ eachindex(𝐂) 
                𝐂[i] += ϵI 
            end
        else
            @simd for i ∈ eachindex(𝐂)  
                @inbounds 𝐂[i] += ϵI 
            end
        end
    end
    return 𝐂
end

tikhonov(𝐂::PosDefManifold.HermitianVector, tikh=0; 
            transform=true, labels :: Union{IntVector, Nothing} = nothing, threaded=false) = 
                    tikhonov!(copy(𝐂), tikh; transform, threaded)

# TESTING: act on a single matrix with provided parameter `tikh` obtained in a training
tikhonov!(C::Hermitian, tikh=0) = 
    if tikh>0
        @simd for i=1:size(C, 1) 
            @inbounds C[i, i] += tikh
        end
    end

tikhonov(C::Hermitian, tikh=0) = tikhonov!(copy(C), tikh)


# TRAINING: Reduce the dimension of the set 𝐏 based on the whitening or pca (pased as `meth`) 
# of the mean of the set. The mean is found using PosDefManifold.mean with arguments:
# metric, w, ✓w, init, tol, verbose, threaded.
# The dimensionality reduction is obtained as a truncated PCA or whitening, on such mean, using 
# Diagonalizations.pca or Diagonalizations.whitening as argument for `meth`, with argument eVar.
# The multiplications dimensionality reduction is multi-threaded if threaded is true (default).
# If infoIfReduced is true (default), an info message is printed if the dimension is reduced.
# If forceDiagonalization is true (default), the diagonalization is performed in any case,
#    even if the dimension is not reduced. Thus this function can be used also for recentering.
# If the dimension is not reduced, the whitener is the inverse of the principal square root 
# of the mean (U * w^-1/2 * U'), otherwise it is (w^-1/2 * U') with w and U reduced.
# Return 3-tuple (𝐏, W, V), where 𝐏 is overwritten, W*mean*W' is a diagonal (pca) or the 
# identity matrix (whitening) and V is the left-inverse of W.
# Example:
# P = randP(100, 10)
# M = mean(Euclidean, P)
# recenter!(Euclidean, P);
# 𝐏, Z, iZ = recenter!(Euclidean, P);
# Z * M * Z' # must be the identity
# force no dimensionality reduction
# P = randP(100, 10)
# M = mean(Euclidean, P)
# # 𝐏, Z, iZ = recenter!(Euclidean, P, eVar=size(P[1], 1)); 
# Z * M * Z' # must be the identity
function recenter!(metric::PosDefManifold.Metric, 𝐏::ℍVector;
                    transform :: Bool = true,
                    labels :: Union{IntVector, Nothing} = nothing,
                    eVar::Union{Real, Int, Nothing}=0.99,
                    w::Vector=Flaot64[],
                    ✓w::Bool=true,
                    init::Union{Hermitian,Nothing}=nothing,
                    tol::Real=0.0,
                    verbose::Bool=true,
                    forcediag::Bool=true,
                    threaded::Bool=true)

    w_  = isnothing(labels) ? w : tsWeights(labels)
    ✓w_ = isnothing(labels) 
    meth, barycen = Diagonalizations.whitening, PosDefManifold.mean
    W = meth(barycen(metric, 𝐏; 
                    w=w_, ✓w=✓w_, init, tol, verbose, ⏩=threaded); 
            eVar)

    # if no dimensionality reduction is needed, use the inverse of the principal
    # sqrt of the mean, otherwise use the reduced inverse sqrt of the eigenvalues
    # times the transpose of the reduced eigenvectors to obtain the reduction
    p = size(W.D, 1) # W.D are eigenvalues of the mean, see Diagonalizations.jl
    sqrtD = sqrt(W.D)
    Z = p==size(𝐏[1], 1) ? W.F*(sqrtD*W.F') : W.F # sqrtD*W.F' are the eigenvectors of the mean
    iZ = p==size(𝐏[1], 1) ? (W.iF'*inv(sqrtD))*W.iF : W.iF # inverse of Z

    if transform
        if p ≠ size(𝐏[1], 1) || forcediag
            p ≠ size(𝐏[1], 1) && verbose && @info "Recenter conditioner: the dimension has been reduced " p eVar
            if threaded
                @threads for i ∈ eachindex(𝐏)
                    𝐏[i] = typeofMatrix(𝐏[i])(Z' * 𝐏[i] * Z)
                end
            else
                @simd for i ∈ eachindex(𝐏)
                    @inbounds 𝐏[i] = typeofMatrix(𝐏[i])(Z' * 𝐏[i] * Z)
                end
            end
        end
    end # if transform
    return 𝐏, Matrix(Z'), Matrix(iZ')
end


# TESTING: act on ℍVector `𝐏` with provided parameter `Zt` obtained in a training
# Zt is given as output of the function above.
function recenter!(𝐏::ℍVector, Zt::Union{Matrix, Hermitian}; threaded::Bool=true) 
    if threaded
        @threads for i ∈ eachindex(𝐏)
            𝐏[i] = typeofMatrix(𝐏[i])(Zt * 𝐏[i] * Zt')
        end
    else
        @simd for i ∈ eachindex(𝐏)
            @inbounds 𝐏[i] = typeofMatrix(𝐏[i])(Zt * 𝐏[i] * Zt')
        end
    end
end

# TESTING: act on a single matrix `P` with provided parameter `Zt` obtained in a training
# Zt is Z' which is given as output of the function `reducedim!` above.
function recenter!(P::Hermitian, Zt::Union{Matrix, Hermitian}; threaded::Bool=true) 
    P.data .= (Zt * P * Zt')
end

# TRAINING :
# Given a vector of matrices, compute the global scaling β=sum(tr(P))/sum(tr(P²))
# to be multiplied to every matrix in 𝐏 so as to solve functional
# min_β sum_j(||β*P_j - I||²).
# This is a global scaling.
function compress!(𝐏::PosDefManifold.ℍVector; 
                    transform :: Bool = true,
                    labels :: Union{IntVector, Nothing} = nothing,
                    threaded :: Bool=true)
    num, den = 0.0, 0.0
    if threaded
        num = Folds.sum(tr, P for P in 𝐏)
        den = Folds.sum(PosDefManifold.sumOfSqr, P for P in 𝐏)
    else
        @simd for i ∈ eachindex(𝐏)
            @inbounds num += tr(𝐏[i])
            @inbounds den += PosDefManifold.sumOfSqr(𝐏[i])
        end
    end

    β = num/den

    if transform
        if threaded
            @threads for i ∈ eachindex(𝐏)
                𝐏[i] = 𝐏[i] * β
            end
        else
            @simd for i ∈ eachindex(𝐏)
                @inbounds 𝐏[i] = 𝐏[i] * β
            end
        end
    end # if transform

    return 𝐏, β
end


# Compute the scaling β=tr(P)/tr(P²) to be multiplied to P so as to solve functional
# min_β ||β*P - I||². This is a single-matrix version of the above. Not used for training.
# s is the x coordinate ("-b/2a") of the minimum of a parabola opening to the top with equation
# ||β*P - I||² = tr(β²*P² -2*β*P + I) = β²*Tr(P²) -β*2*tr(P) + n.
# The matrix β*P is the closest to the identity that can be achieved by scaling
# according to the Euclidean distance.
# Return 2-tuple (P*β, β)
function compress!(P::Hermitian; threaded::Bool=true)
    β = real((tr(P) / PosDefManifold.sumOfSqr(P))) # see PosDefManifold for the method sumOfSqr(P)=tr(P, P)=tr(P^2)=norm(P)^2
    P.data .= P * β
    return P, β
end

# TESTING: act on ℍVector `P` with provided parameter `β` obtained in a training
# as output of the Training version of the function `compress!` above.
function compress!(𝐏::ℍVector, β::Real; threaded::Bool=true) 
    if threaded
        @threads for i ∈ eachindex(𝐏)
            𝐏[i] = β * 𝐏[i]
        end
    else
        @simd for i ∈ eachindex(𝐏)
            @inbounds 𝐏[i] = β * 𝐏[i]
        end
    end
end

# For a single matrix
function compress!(P::Hermitian, β::Real; threaded::Bool=true)
    P.data .= P * β
end

# TRAINING:
# Given  a vector of matrices, it applies the above compress! scaling individually to all 
# matrices in 𝐏. This means that each matrix is scaled individually. 
# This approach is actually a normalization.
function equalize!(𝐏::PosDefManifold.ℍVector; 
                    transform :: Bool = true,
                    labels :: Union{IntVector, Nothing} = nothing,
                    threaded::Bool=true)
    if threaded
        𝛃 = Vector{eltype(𝐏[1])}(undef, length(𝐏))
        @threads for i ∈ eachindex(𝐏)
            𝛃[i] = tr(𝐏[i]) / PosDefManifold.sumOfSqr(𝐏[i])
        end
    else
        𝛃 = [tr(P) / PosDefManifold.sumOfSqr(P) for P ∈ 𝐏] # see PosDefManifold for the method tr(P, P)=tr(P^2)=sumOfSqr(P)=norm(P)^2 
    end

    if transform
        if threaded
            @threads for i ∈ eachindex(𝐏)
                𝐏[i] = 𝐏[i] * 𝛃[i]
            end
        else
            @simd for i ∈ eachindex(𝐏)
                𝐏[i] = 𝐏[i] * 𝛃[i]
            end
        end
    end # if transform

    return 𝐏, 𝛃
end


# For a single matrix equalize and compress coincide. Not to be used for training
equalize!(P::Hermitian; threaded::Bool=true) = compress!(P; threaded)


# TESTING: act on ℍVector `P` with provided parameter `s` obtained in a training
# as output of the Training version of the function `equalize!` above.
function equalize!(𝐏::PosDefManifold.ℍVector, 𝛃::Vector{T}; threaded::Bool=true) where T<: Real
    if threaded
        @threads for i ∈ eachindex(𝐏)
            𝐏[i] = 𝐏[i] * 𝛃[i]
        end
    else
        @simd for i ∈ eachindex(𝐏)
            𝐏[i] = 𝐏[i] * 𝛃[i]
        end
    end
end

# Testing for a single matrix
equalize!(P::Hermitian, β::Real; threaded::Bool=true) = compress!(P, β; threaded)


# Return the vector of eigenvectors and the vector of eigenvalues of matrices in 𝐏
# Example: P=randP(3, 4); λ, U = evds(P); U[1]*(λ[1].*U[1]')≈P[1] # must be true
function evds(𝐏::PosDefManifold.ℍVector; threaded::Bool=true)

    𝛌 = Vector{Vector{eltype(𝐏[1])}}(undef, length(𝐏))
    𝐔 = Vector{Matrix{eltype(𝐏[1])}}(undef, length(𝐏))

    if threaded
        @threads for i ∈ eachindex(𝐏)
            𝛌[i], 𝐔[i] = eigen(𝐏[i])
        end
    else
        @simd for i ∈ eachindex(𝐏)
            @inbounds 𝛌[i], 𝐔[i] = eigen(𝐏[i])
        end
    end

    return 𝛌, 𝐔
end


# TRAINING:
# Return 2-tuple (𝐏, γ), where 𝐏 is the set of input matrices 𝐏 shrinked in an open ball with given `radius` r 
# according to the given `metric`, and γ ∈ (0, 1] is the shrinking parameter. 
# The radius r englobes the most distant point of set 𝐏 if `refpoint`=:max, the mean eccentricity
# of the points in 𝐏 if `refpoint`=:mean. This letter option means that the points are shrinked so that r 
# is equal to sqrt(1/n * mean(norm(P) for P in 𝐏).
# The shrinking is done moving, for each matrix P_k in the set, along the gedesic relying I to P_k,
# according to the given metric. The distance according to the given metric is evaluated for all matrices
# in 𝐏 and γ is computed so as to set the maximum or mean eccentricity equal to r + ϵ,
# where ϵ is the `epsilon` kwarg.
# The ball containing 𝐏 may also be increased (the opposite of shrinking).
# For all metrics γ = (r*√n) / (δ(P, I) + ϵ). The defaults are r = 0.02 and ϵ = 0

# Proof (for three metrics, see PosDefManifold.jl for available metrics supporting a geodesic equation):

# Euclidean: set n⁻¹*δ_E(P, I) = r. 
#   This means n⁻¹*||(1-γ)I+γP-I||=r.
#   We have ||I-γI+γP-I|| = r*√n, ||γP-γI|| = r*√n, γ||P-I|| = r*√n, γ = (r*√n)/δ_E(P, I)

# log-Cholesky: set n⁻¹*δ_lC(P, I) = r. 
#   This means δ_lC(γS+DD^(γ+1), I) = r*√n, i.e., √(γ²||S||²+||Log(DD^(γ+1)))||² = r*√n,
#   where S and D are the matrices obtrained nullifying the upper triangular part and diagonal part of P, respectively.
#   Now, since ||Log(DD^(γ+1)))||² = ||Log(D) + (γ+1)Log(D)||² = ||Log(D)(I+(γ-1)I)||² = ||Log(D)(γI)||² =
#   = ||γ Log(D)||² = γ²||Log(D)||², this is
#    √(γ²||S||²+γ²||Log(D)||²) = r*√n, or √(γ²(||S||²+||Log(D)||²)) = r*√n, γ√(||S||²+||Log(D)||²) = r*√n, hence
#   γ = (r*√n)/δ_lC(P, I)

# Fisher: set n⁻¹*δ_F(P, I) = r. 
#   This means ||Log(P^γ)|| = r*√n, ||γ Log(P)|| = r*√n, γ ||Log(P)|| = r*√n, hence
#   γ = (r*√n)/δ_F(P, I)

# If all points are shrinked within a ball of radius 1, for each matrix Q in the returned set, 
# the infinite series -sum_1:n ((I-Q)^n)/n 
# converges to log(Q), with n=1...Inf (Theorem 2.7, page 34 of B.C. Hall (2003), "Lie Groups, 
# Lie Algebras, and Representations An Elementary Introduction", Springer).

### logCholesky geodesic from I to the matrix with step-size γ. L is the Cholesky factor of the matrix
function geoLCfromI(L, γ)
    D = Diagonal(L)
    Z = (γ * tril(L, -1) + D * D^(γ - 1))  
    return ℍ(Z * Z')
end

function shrink!(metric::PosDefManifold.Metric, 𝐏::PosDefManifold.ℍVector, radius::Union{Float64, Int}=0.02; 
                transform :: Bool = true,
                labels :: Union{IntVector, Nothing} = nothing,
                refpoint::Symbol=:mean,
                reshape::Bool=false, # only for Fisher metric
                epsilon::Float64=0.0, 
                verbose::Bool=true, 
                threaded::Bool=true) 

    n = size(𝐏[1], 1)
    m, sd = 0., 1.

    T = typeofMatrix(𝐏[1])
    γ = 1.0
    # The code could be very simple and generic: compute distances, get the radius, compute alpha and move along 
    # the geodesics, all using PosDefManifold.jl with the given metric, as it is done for generic metrics.
    # However, efficient code is written for the Fisher and log-Cholesky metric. 

    if metric == PosDefManifold.logCholesky

        𝐋 = LowerTriangularVector(undef, length(𝐏))
        funcLC(L) = real(sst(L, -1)) + ssd(𝑓𝔻(log, L)) # distance

        if threaded
            @threads for i = 1:length(𝐏)
                𝐋[i] = choL(𝐏[i])
            end
        else
            @simd for i = 1:length(𝐏)
                𝐋[i] = choL(𝐏[i])
            end
        end

        if refpoint==:max
            d = threaded ? Folds.maximum(funcLC, 𝐋) : maximum(funcLC, 𝐋)
        else
            d = threaded ? Folds.sum(funcLC, 𝐋)/length(𝐋) : sum(funcLC, 𝐋)/length(𝐋)
        end

        # if d ≥ radius # d is the maximal or mean logCholesky distance squared to I
            γ = (radius * sqrt(n)) / (sqrt(d) + epsilon)
            0 ≤ γ || throw(ArgumentError("Shrink conditioner with $(metric) metric; shrinkage parameter γ≤0"))
            # γ < 1 || @warn "Shrink conditioner with $(metric) metric; shrinkage parameter γ≥1" γ radius

            # move on the logCholesky geodesic relying I to 𝐏[i] with step-size γ
            if transform
                if threaded
                    @threads for i ∈ eachindex(𝐏)
                        𝐏[i] = T(geoLCfromI(𝐋[i], γ))
                    end
                else
                    @simd for i ∈ eachindex(𝐏)
                        @inbounds 𝐏[i] = T(geoLCfromI(𝐋[i], γ))
                    end
                end
            end # if transform
        # else
            # verbose && @warn "Shrink conditioner with method $(metric): no shrinking is necessary" d radius
        # end

    elseif metric == PosDefManifold.Fisher
        #eltype(𝐏[1]) <: Complex && throw(ArgumentError("The metric Fisher for function shrink! and shrink is defined only for real matrices"))
        𝛌, 𝐔 = evds(𝐏; threaded)

        # here the eigenvalues could be normalized to constant variance

        funcF(λ) = sqrt(sum(x -> real(log(x))^2, λ)) # norm
        if refpoint==:max
            d = threaded ? Folds.maximum(funcF, 𝛌) : maximum(funcF, 𝛌)
        else
            d = threaded ? Folds.sum(funcF, 𝛌)/length(𝛌) : sum(funcF, 𝛌)/length(𝛌) # average norm
        end

        # if d ≥ radius 
            γ = (radius * sqrt(n)) / (d + epsilon) # γ is the Fisher distance to I
            #            println("γ, d ", γ, " ", d)
            0 ≤ γ || throw(ArgumentError("Shrink conditioner with $(metric) metric; computed shrinkage parameter γ≤0"))
            # γ < 1 || @warn "Shrink conditioner with $(metric) metric; shrinkage parameter γ≥1" γ radius

            # move on the Fisher geodesic relying I to 𝐏[i] with step-size γ
            # and re-recenter the eigenvalues if recenter=true         
            if transform || reshape
                for i ∈ eachindex(𝛌)
                    for j ∈ eachindex(𝛌[i])
                        𝛌[i][j] ^= γ
                    end 
                end
            end

            if reshape
                m=(mean(mean.(𝛌))) # mean
                sd=0. 
                for i ∈ eachindex(𝛌)
                    for j ∈ eachindex(𝛌[i])
                        sd += (𝛌[i][j]-m)^2
                    end
                end
                sd = sqrt(sd/(sum(length(λ) for λ ∈ 𝛌))) # standard deviation
                if transform
                    for i ∈ eachindex(𝛌)
                        for j ∈ eachindex(𝛌[i])
                            𝛌[i][j]=1.0+((radius/sd)*(𝛌[i][j]-m))#/(sd/radius)
                        end
                    end
                end
            end # if reshape

            if transform
                if threaded
                    @threads for i ∈ eachindex(𝐏)
                        𝐏[i] = T(𝐔[i] * (𝛌[i] .* 𝐔[i]'))
                    end
                else
                    @simd for i ∈ eachindex(𝐏)
                        @inbounds 𝐏[i] = T(𝐔[i] * (𝛌[i] .* 𝐔[i]'))
                    end
                end
            end # if transform
        # else
            # verbose && @warn "Shrink conditioner with method $(metric): no shrinking was necessary" d radius
        # end

    else # all other supported metrics
        func(P) = distance(metric, P) # distance
        if refpoint==:max
            d = threaded ? Folds.maximum(func, 𝐏) : maximum(func, 𝐏)
        else
            d = threaded ? Folds.sum(func, 𝐏)/length(𝐏) : sum(func, 𝐏)/length(𝐏)
        end
        
        # if d ≥ radius # γ is the metric distance to I
            γ = (radius * sqrt(n)) / (d + epsilon)
            0 ≤ γ || throw(ArgumentError("Shrink conditioner with $(metric) metric; computed shrinkage parameter γ≤0"))
            #γ < 1 || @warn "Shrink conditioner with $(metric) metric; shrinkage parameter γ≥1" γ radius

            if transform
                #0 < γ ≤ 1 || throw(ArgumentError("shrink! or shrink function with $(metric) metric; shrinkage parameter γ ∉(0, 1]"))
                # move on the geodesic relying I to 𝐏[i] with step-size γ
                Id = eltype(𝐏[1]) <: Real ? Hermitian(Matrix(1.0 * I, size(𝐏[1])...)) : Hermitian(Matrix((1.0 + 0im) * I, size(𝐏[1])...))
                if threaded
                    @threads for i ∈ eachindex(𝐏)
                        𝐏[i] = T(PosDefManifold.geodesic(metric, Id, 𝐏[i], γ))
                    end
                else
                    @simd for i ∈ eachindex(𝐏)
                        @inbounds 𝐏[i] = T(PosDefManifold.geodesic(metric, Id, 𝐏[i], γ))
                    end
                end
            end # if transform
        # else
            # verbose && @warn "Shrink conditioner with metric $(metric): no shrinking was necessary" d radius
        # end
    end
    
    return (𝐏, γ, m, sd)
end



# as above but for a single matrix. Not used for training.
# Threaded is not used, it is here for compatibility with previous methods
function shrink!(metric::PosDefManifold.Metric, P::Hermitian, radius::Union{Float64, Int}=0.02;
                transform :: Bool = true,
                refpoint::Symbol=:mean,
                labels :: Union{IntVector, Nothing} = nothing,
                reshape::Bool=false, # only for Fisher metric
                epsilon::Float64=0.0,
                verbose::Bool=true, 
                threaded::Bool=true)

    n = size(P, 1)
    m, sd = 0., 1.

    T = typeofMatrix(P)
    γ = 1.0

    if metric == PosDefManifold.logCholesky
        #eltype(P) <: Complex && throw(ArgumentError("The metric logCholesky for function shrink! and shrink is defined only for real matrices"))        λ, U = eigen(P)
        L = choL(P)
        ssdD = PosDefManifold.ssd(log(real(Diagonal(L))))
        sstL = real(sst(L, -1))
        d = sstL + ssdD
        if d ≥ radius # d is the logCholesky distance squared to I
            γ = (radius *sqrt(n)) / (sqrt(d) + epsilon)
            0 ≤ γ || throw(ArgumentError("Shrink conditioner with $(metric) metric; computed shrinkage parameter γ≤0"))
            γ < 1 || @warn "Shrink conditioner with $(metric) metric; shrinkage parameter γ≥1" γ radius

            # move on the logCholesky geodesic relying I to 𝐏[i] with step-size ϕ
            if transform
                P = T(geoLCfromI(L, γ))
            end
        else
            verbose && @info "Shrink conditioner with metric $(metric): no shrinking was necessary" d radius
        end

    elseif metric == PosDefManifold.Fisher
        #eltype(P) <: Complex && throw(ArgumentError("The metric Fisher for function shrink! and shrink is defined only for real matrices"))        λ, U = eigen(P)
        λ, U = eigen(P)
        d = sqrt(sum(x -> abs2(log(x)), λ)) + epsilon

        if d ≥ radius # d is the Fisher distance to I
            γ = (radius *sqrt(n)) / (d + epsilon)
            0 ≤ γ || throw(ArgumentError("Shrink conditioner with $(metric) metric; computed shrinkage parameter γ≤0"))
            γ < 1 || @warn "Shrink conditioner with $(metric) metric; shrinkage parameter γ≥1" γ radius

            if transform || reshape
                λ.^= γ
            end
            if reshape 
                m=mean(λ)
                sd=std(λ, mean=m)
                if transform
                    λ.-=m-1.0
                    λ./=(sd/radius)
                end
            end

            if transform
                P = T(U * (λ .* U'))
            end
        else
            verbose && @info "Shrink conditioner with metric $(metric): no shrinking was necessary" d radius
        end
    else
        d = PosDefManifold.distance(metric, P)
        if d ≥ radius # d is the metric distance to I
            γ = (radius * sqrt(n)) / (d + epsilon)
            0 ≤ γ || throw(ArgumentError("Shrink conditioner with $(metric) metric; computed shrinkage parameter γ≤0"))
            γ < 1 || @warn "Shrink conditioner with $(metric) metric; shrinkage parameter γ≥1" γ radius

            # move on the geodesic relying I to 𝐏[i] with step-size γ
            if transform
                Id = eltype(P) <: Real ? Hermitian(Matrix(1.0 * I, size(P)...)) : Hermitian(Matrix((1.0 + 0im) * I, size(P)...))
                P = T(PosDefManifold.geodesic(metric, Id, P, γ))
            end
        else
            verbose && @info "Shrink conditioner with metric $(metric): no shrinking was necessary" d radius
        end       
    end

    return (P, γ, m, sd)
end



function getShrinkedP(P, γ, radius, m, sd, reshape)
    λ, U = eigen(P)
    λ.^= γ
    if reshape 
        for j ∈ eachindex(λ)
            λ[j] = 1.0 + ((radius/sd)*(λ[j]-m)) 
        end 
    end
    return Hermitian(U * (λ .* U'))
end # func getP

# TESTING: act on ℍVector `𝐏` with provided parameter `γ` obtained in a training
# For the Fisher metric, if reshape is true, `radius`, `m` and `sd` must be provided and different from nothing.
function shrink!(metric::PosDefManifold.Metric, 𝐏::ℍVector, γ::Union{Float64, Int}, radius::Union{Real, Nothing},
                m::Union{Real, Nothing}, sd::Union{Real, Nothing}, reshape::Bool=false; threaded::Bool=true)
    metric == PosDefManifold.Fisher && reshape==true && (radius===nothing || m===nothing || sd===nothing) && throw(ArgumentError("Shrink conditioner with the Fisher metric, `radius`, `m` and `sd` must be provided and different from nothing. Check that the fit! function employed a Shrink conditioner adopting the Fisher metric and that the `reshape` argument was true"))

    if metric == PosDefManifold.logCholesky
        if threaded
            @threads for i ∈ eachindex(𝐏)
                𝐏[i] = geoLCfromI(choL(𝐏[i]), γ)
            end
        else
            @simd for i ∈ eachindex(𝐏)
                @inbounds 𝐏[i] = geoLCfromI(choL(𝐏[i]), γ)
            end
        end
    elseif metric == PosDefManifold.Fisher
        if threaded
            @threads for i ∈ eachindex(𝐏)
                𝐏[i] = getShrinkedP(𝐏[i], γ, radius, m, sd, reshape)
            end
        else
            @simd for i ∈ eachindex(𝐏)
                @inbounds 𝐏[i] = getShrinkedP(𝐏[i], γ, radius, m, sd, reshape)
            end
        end
    else
        Id = eltype(𝐏[1]) <: Real ? Hermitian(Matrix(1.0 * I, size(𝐏[1])...)) : Hermitian(Matrix((1.0 + 0im) * I, size(𝐏[1])...))
        if threaded
            @threads for i ∈ eachindex(𝐏)
                𝐏[i] =  PosDefManifold.geodesic(metric, Id, 𝐏[i], γ)
            end
        else
            @simd for i ∈ eachindex(𝐏)
                @inbounds 𝐏[i] = PosDefManifold.geodesic(metric, Id, 𝐏[i], γ)
            end
        end
    end
end

# TESTING: act on a single matrix `P` with provided parameter `γ` obtained in a training
# as output of the Training version of the function `shrink!` above.
# For the Fisher metric, if reshape is true, `radius`, `m` and `sd` must be provided and different from nothing.
function shrink!(metric::PosDefManifold.Metric, P::Hermitian, γ::Union{Float64, Int}, radius::Union{Real, Nothing},
                m::Union{Real, Nothing}, sd::Union{Real, Nothing}, reshape::Bool=false; threaded::Bool=true)
    
    if metric == PosDefManifold.logCholesky
        P.data .= geoLCfromI(choL(P), γ)
    elseif metric == PosDefManifold.Fisher
        P.data .= (getShrinkedP(P, γ, radius, m, sd, reshape)).data
    else
        Id = eltype(𝐏[1]) <: Real ? Hermitian(Matrix(1.0 * I, size(𝐏[1])...)) : Hermitian(Matrix((1.0 + 0im) * I, size(𝐏[1])...))
        P.data .= (PosDefManifold.geodesic(metric, Id, P, γ)).data
    end
end
