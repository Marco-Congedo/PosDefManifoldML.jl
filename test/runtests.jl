using PosDefManifold, PosDefManifoldML, Statistics, Test

# Give two symmetric positive definite matrices A and B,
# 80 matrices will be created randomly as points on the the geodesic
# relying A to B and 80 matrices as points on the the geodesic relying B to A,
# for 80 arc-lengths given by function rand(Float64)*exp(-i/(160))),
# for 1=1:80. This makes 80 two class problems with 160 observations each
# that must be progressively better separable as i increases.
# For each of three default machine learning models (MDM, ENLR and SVM)
# the average accuracy is found by a 10-fold cross-validation for each i
# dataset. The Pearson correlation of the accuracy with the vector [1 2...80]
# is computed and a statistical test on the probability that the correlation is
# equal to zero is performed, requesting the p-value be <1e-8
# (i.e., the correlation with 80-2 degrees of freedom must be > 0.6)

# This is not a formal test, but a general test on the functioning of all
# main functions of the packages and on the expected behavior of ML models.

A= ℍ([1.98 0.97 0.17; 0.96 1.45 1.37; 0.17 1.37 6.33])
B= ℍ([1.21 2.82 -1.48; 2.82  7.09 -3.69; -1.48 -3.69  2.00])
n=80

g(α::Float64)=geodesic(Fisher, A, B, α)

gVec(n, i, P, Q)=
   ℍVector([geodesic(Fisher, P, Q, rand(Float64)*exp(-i/(n*2))) for j=1:n])

PTr=ℍVector₂([vcat(gVec(n, i, A, B), gVec(n, i, B, A)) for i=1:n])
PTe=ℍVector₂([vcat(gVec(n, i, A, B), gVec(n, i, B, A)) for i=1:n])
y=vcat(fill(1, n), fill(2, n))

@testset "All Models" begin
    println("\nTesting MDM model...")
    cv=[crval(MDM(), PTr[i], y; verbose=false, scoring=:b).avgAcc for i=1:n]
    @test cor(cv, [i for i=1:n]) > 0.6

    println("\nTesting ENLR model...")
    cv=[crval(ENLR(), PTr[i], y; verbose=false).avgAcc for i=1:n]
    @test cor(cv, [i for i=1:n]) > 0.6

    println("\nTesting SVM model...")
    cv=[crval(SVM(), PTr[i], y; verbose=false).avgAcc for i=1:n]
    @test cor(cv, [i for i=1:n]) > 0.6
end;