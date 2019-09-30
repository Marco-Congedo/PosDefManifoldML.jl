# PosDefManifoldML.jl

| **Documentation**  | 
|:---------------------------------------:|
| [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://Marco-Congedo.github.io/PosDefManifoldML.jl/dev) |

**PosDefManifoldML** is a [**Julia**](https://julialang.org/) package for classifying data in the [**Riemannian manifolds**](https://en.wikipedia.org/wiki/Riemannian_manifold) **P** of real or complex [**positive definite matrices**](https://en.wikipedia.org/wiki/Definiteness_of_a_matrix). It is based on [PosDefManifold.jl](https://github.com/Marco-Congedo/PosDefManifold.jl). 

Machine learning in **P** can either operate directly on the manifold, which requires dedicated Riemannian methods, or the data can be projected onto the **tangent space**, where standard (Euclidean) machine learning methods apply (e.g., linear discriminant analysis, support-vector machine, logistic regression, random forest, deep neuronal networks, etc). 

![](/docs/src/assets/Fig1.jpg)

For the moment being, the Riemannian Minimum Distance to Mean (MDM) classifier has been implemented. Furthermore, the package allows projecting the data on the tangent space for applying traditional machine learning elsewhere.

## Installation

The package is still not registered. To install it,
execute the following command in Julia's REPL:

    ]add https://github.com/Marco-Congedo/PosDefManifoldML

## Disclaimer

This package is still in a pre-release stage.
Independent reviewers are more then welcome.

## Examples

```
using PosDefManifoldML

# simulate symmetric positive definite matrices data
𝐏Tr, 𝐏Te, yTr, yTe=gen2ClassData(10, 30, 40, 60, 80)

# craete and fit a Riemannian Minimum Distance to Mean (MDM) model
model=MDM(Fisher, 𝐏Tr, yTr)

# predict labels
predict(model, 𝐏Te, :l)

# predict probabilities
predict(model, 𝐏Te, :p)

# average accuracy obtained by 5-fold cross-validation
CVscore(model, 𝐏Tr, yTr, 5)

```

## About the Authors

Saloni Jain is a student at the
[Indian Institute of Technology, Kharagpur](http://www.iitkgp.ac.in/), India.

[Marco Congedo](https://sites.google.com/site/marcocongedo), corresponding
author, is a research scientist of [CNRS](http://www.cnrs.fr/en) (Centre National de la Recherche Scientifique), working in [UGA](https://www.univ-grenoble-alpes.fr/english/) (University of Grenoble Alpes). **contact**: marco *dot* congedo *at* gmail *dot* com

| **Documentation**  | 
|:---------------------------------------:|
| [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://Marco-Congedo.github.io/PosDefManifoldML.jl/dev) |



