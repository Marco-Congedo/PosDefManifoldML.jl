var documenterSearchIndex = {"docs":
[{"location":"tutorial/#Tutorial-1","page":"Tutorials","title":"Tutorial","text":"","category":"section"},{"location":"tutorial/#","page":"Tutorials","title":"Tutorials","text":"PosDefManifoldML mimicks the functioning of ScikitLearn (good to know if you are familir with it): first a machine learning (ML) model is created, then data is used to fit (train) the model. The above two steps can actually be carried out at one. Once this is done the model allows to predict the labels of test data or the probability of the data to belong to each class.","category":"page"},{"location":"tutorial/#","page":"Tutorials","title":"Tutorials","text":"In order to compare ML models, a k-fold cross-validation procedure is implemented.","category":"page"},{"location":"tutorial/#ML-models-1","page":"Tutorials","title":"ML models","text":"","category":"section"},{"location":"tutorial/#","page":"Tutorials","title":"Tutorials","text":"For the moment being, only the Riemannian minimum distance to mean (MDM) ML model is implemented. See Barachat el al. (2012) and Congedo et al. (2017a) 🎓.","category":"page"},{"location":"tutorial/#MDM-model-1","page":"Tutorials","title":"MDM model","text":"","category":"section"},{"location":"tutorial/#","page":"Tutorials","title":"Tutorials","text":"An MDM model is created and fit with trainng data such as","category":"page"},{"location":"tutorial/#","page":"Tutorials","title":"Tutorials","text":"model=MDM(Fisher, 𝐏Tr, yTr)","category":"page"},{"location":"tutorial/#","page":"Tutorials","title":"Tutorials","text":"where metric is be a Metric enumerated type declared in PosDefManifold, a metric in the manifold of positive definite matrices allowing the definition of both a distance function and of a mean (center of mass).","category":"page"},{"location":"tutorial/#","page":"Tutorials","title":"Tutorials","text":"Currently supported metrics are:","category":"page"},{"location":"tutorial/#","page":"Tutorials","title":"Tutorials","text":"metric (distance) resulting mean estimation\nEuclidean Arithmetic\ninvEuclidean Harmonic\nChoEuclidean Cholesky Euclidean\nlogEuclidean Log-Euclidean\nlogCholesky Log-Cholesky\nFisher Fisher (Cartan, Karcher, Pusz-Woronowicz,...)\nlogdet0 LogDet (S, α, Bhattacharyya, Jensen,...)\nJeffrey Jeffrey (symmetrized Kullback-Leibler)\nWasserstein Wasserstein (Bures, Hellinger, ...)","category":"page"},{"location":"tutorial/#","page":"Tutorials","title":"Tutorials","text":"Do not use the Von Neumann metric, which is also supported in PosDefManifold, since it does not allow a definition of mean.","category":"page"},{"location":"tutorial/#use-data-1","page":"Tutorials","title":"use data","text":"","category":"section"},{"location":"tutorial/#","page":"Tutorials","title":"Tutorials","text":"A real data example will be added soon.","category":"page"},{"location":"tutorial/#","page":"Tutorials","title":"Tutorials","text":"Now let us create some simulated data for a 2-class example. First, let us create symmetric positive definite matrices (real positive definite matrices):","category":"page"},{"location":"tutorial/#","page":"Tutorials","title":"Tutorials","text":"using PosDefManifoldML\n\n𝐏Tr, 𝐏Te, yTr, yTe=gen2ClassData(10, 30, 40, 60, 80)","category":"page"},{"location":"tutorial/#","page":"Tutorials","title":"Tutorials","text":"-𝐏Tr is the simulated training set, holding 30 matrices for class 1 and 40 matrices for class 2","category":"page"},{"location":"tutorial/#","page":"Tutorials","title":"Tutorials","text":"𝐏Te is the testing set, holding 60 matrices for class 1 and 80 matrices for class 2.\nyTr is a vector of 70 labels for 𝐓r\nyTe is a vector of 140 labels for 𝐓e","category":"page"},{"location":"tutorial/#","page":"Tutorials","title":"Tutorials","text":"All matrices are of size 10x10.","category":"page"},{"location":"tutorial/#craete-and-fit-an-MDM-model-1","page":"Tutorials","title":"craete and fit an MDM model","text":"","category":"section"},{"location":"tutorial/#","page":"Tutorials","title":"Tutorials","text":"model=MDM(Fisher, 𝐏Tr, yTr)","category":"page"},{"location":"tutorial/#classify-data-(predict)-1","page":"Tutorials","title":"classify data (predict)","text":"","category":"section"},{"location":"tutorial/#","page":"Tutorials","title":"Tutorials","text":"predict(model, 𝐏Te, :l)","category":"page"},{"location":"tutorial/#","page":"Tutorials","title":"Tutorials","text":"If instead we wish to estimate the probabilities for the matrices in 𝐏Te of belonging to each class, we invoke","category":"page"},{"location":"tutorial/#","page":"Tutorials","title":"Tutorials","text":"predict(model, 𝐏Te, :p)","category":"page"},{"location":"tutorial/#cross-validation-1","page":"Tutorials","title":"cross-validation","text":"","category":"section"},{"location":"tutorial/#","page":"Tutorials","title":"Tutorials","text":"A k-fold cross-validation is obtained as","category":"page"},{"location":"tutorial/#","page":"Tutorials","title":"Tutorials","text":"CVscore(model, 𝐏Te, y, 5)","category":"page"},{"location":"tutorial/#","page":"Tutorials","title":"Tutorials","text":"where 5 is the number of folds. This implies that at each CV, 1/5th of the matrices is used for training and the remaining for testing.","category":"page"},{"location":"mdm/#mdm.jl-1","page":"Minimum Distance to Mean","title":"mdm.jl","text":"","category":"section"},{"location":"mdm/#","page":"Minimum Distance to Mean","title":"Minimum Distance to Mean","text":"This unit implemets the Riemannian MDM (Minimum Distance to Mean) classifier for the manifold of positive definite matrices.","category":"page"},{"location":"mdm/#","page":"Minimum Distance to Mean","title":"Minimum Distance to Mean","text":"Besides the MDM type declaration and the declaration of some constructors for it, this unit also include the following functions, which typically you will not need to access directly:","category":"page"},{"location":"mdm/#","page":"Minimum Distance to Mean","title":"Minimum Distance to Mean","text":"function description\ngetMeans compute means of training data for fitting the MDM model\ngetDistances compute the distances of a matrix set to a set of means\nCV_mdm perform cross-validations for the MDM classifiers","category":"page"},{"location":"mdm/#","page":"Minimum Distance to Mean","title":"Minimum Distance to Mean","text":"MDM\ngetMeans\ngetDistances\nCV_mdm","category":"page"},{"location":"mdm/#PosDefManifoldML.MDM","page":"Minimum Distance to Mean","title":"PosDefManifoldML.MDM","text":"(1)\nmutable struct MDM <: MLmodel\n    metric :: Metric\n    means\n    function MDM(metric :: Metric; means = nothing)\n        new(metric, means)\n    end\nend\n\n(2)\nfunction MDM(metric :: Metric,\n             𝐏Tr    :: ℍVector,\n             yTr    :: IntVector;\n           w  :: Vector = [],\n           ✓w :: Bool  = true)\n\n(1)\n\nMDM machine learning models are incapsulated in this mutable structure. MDM models have two fields: .metric and .means.\n\nThe field metric, of type Metric, is to be specified by the user. It is the metric that will be adopted to compute the class means.\n\nThe field means is an ℍVector holding the class means, i.e., one mean for each class. This field is not to be specified by the user, instead, the means are computed when the MDM model is fit using the fit! function and are accessible only thereafter.\n\n(2)\n\nConstructor creating and fitting an MDM model with training data 𝐏Tr, an ℍVector type, and labels yTr, an IntVector type. The class means are computed according to the chosen metric, of type Metric. See here for details on the metrics. Supported metrics are listed in the section about creating an MDM model.\n\nOptional keyword arguments w and ✓w are passed to the fit! function and have the same meaning therein.\n\nExamples:\n\nusing PosDefManifoldML\n\n# (1)\n# generate some data\n𝐏Tr, 𝐏Te, yTr, yTe = gen2ClassData(10, 30, 40, 60, 80)\n\n# create a model\nmodel = MDM(Fisher)\n\n# fit the model with training data\nfit!(model, 𝐏Tr, yTr)\n\n# (2) equivalently and faster:\n𝐏Tr, 𝐏Te, yTr, yTe=gen2ClassData(10, 30, 40, 60, 80)\nmodel = MDM(Fisher, 𝐏Tr, yTr)\n\n\n\n\n\n","category":"type"},{"location":"mdm/#PosDefManifoldML.getMeans","page":"Minimum Distance to Mean","title":"PosDefManifoldML.getMeans","text":"function getMeans(metric :: Metric,\n                  𝐏      :: ℍVector;\n              tol :: Real = 0.,\n              w   :: Vector = [],\n              ✓w :: Bool   = true,\n              ⏩ :: Bool   = true)\n\nTypically, you will not need this function as it is called by the fit! function.\n\nGiven a metric of type Metric, an ℍVector of Hermitian matrices 𝐏 and an optional non-negative real weights vector w, return the (weighted) mean of the matrices in 𝐏. This is used to fit MDM models.\n\nThis function calls the appropriate mean functions of package PostDefManifold, depending on the chosen metric, and check that, if the mean is found by an iterative algorithm, then the iterative algorithm converges.\n\nSee method (3) of the mean function for the meaning of the optional keyword arguments w, ✓w and ⏩, to which they are passed.\n\nThe returned mean is flagged by Julia as an Hermitian matrix (see LinearAlgebra).\n\n\n\n\n\n","category":"function"},{"location":"mdm/#PosDefManifoldML.getDistances","page":"Minimum Distance to Mean","title":"PosDefManifoldML.getDistances","text":"function getDistances(metric :: Metric,\n                      means  :: ℍVector,\n                      𝐏      :: ℍVector)\n\nTypically, you will not need this function as it is called by the predict function.\n\nGiven an ℍVector 𝐏 holding k Hermitian matrices and an ℍVector means holding z matrix means, return the distance of each matrix in 𝐏 to the means in means.\n\nThe distance is computed according to the chosen metric, of type Metric. See metrics for details on the supported distance functions.\n\nThe result is a zxk matrix of distances.\n\n\n\n\n\n","category":"function"},{"location":"mdm/#PosDefManifoldML.CV_mdm","page":"Minimum Distance to Mean","title":"PosDefManifoldML.CV_mdm","text":"function CV_mdm(metric :: Metric,\n                𝐏Tr    :: ℍVector,\n                yTr    :: IntVector,\n                nCV    :: Int;\n            scoring   :: Symbol = :b,\n            confusion :: Bool   = false,\n            shuffle   :: Bool   = false)\n\nTypically, you will not need this function as it is called by the CVscore function.\n\nThis function return the same thing and has the same arguments as the CVscore function, with the exception of the first argument, that here is a metric of type Metric.\n\n\n\n\n\n","category":"function"},{"location":"MainModule/#MainModule-1","page":"Main Module","title":"MainModule","text":"","category":"section"},{"location":"MainModule/#","page":"Main Module","title":"Main Module","text":"This is the main unit containing the PosDefManifoldML module.","category":"page"},{"location":"MainModule/#dependencies-1","page":"Main Module","title":"dependencies","text":"","category":"section"},{"location":"MainModule/#","page":"Main Module","title":"Main Module","text":"standard Julia packages external packages\nLinearAlgebra PosDefManifold\nStatistics \nRandom ","category":"page"},{"location":"MainModule/#","page":"Main Module","title":"Main Module","text":"The main module does not contains functions.","category":"page"},{"location":"MainModule/#types-1","page":"Main Module","title":"types","text":"","category":"section"},{"location":"MainModule/#ML-model-1","page":"Main Module","title":"ML model","text":"","category":"section"},{"location":"MainModule/#","page":"Main Module","title":"Main Module","text":"Similarly to what is done in ScikitLearn, a type is created (a struct in Julia) to specify a ML model. Supertype","category":"page"},{"location":"MainModule/#","page":"Main Module","title":"Main Module","text":"abstract type MLmodel end","category":"page"},{"location":"MainModule/#","page":"Main Module","title":"Main Module","text":"is the abstract type that should be used to derive all machine learning models to be implemented. See the MDM model as an example.","category":"page"},{"location":"MainModule/#IntVector-1","page":"Main Module","title":"IntVector","text":"","category":"section"},{"location":"MainModule/#","page":"Main Module","title":"Main Module","text":"In all concerned functions class labels are given as a vector of integers, of type","category":"page"},{"location":"MainModule/#","page":"Main Module","title":"Main Module","text":"IntVector=Vector{Int}.","category":"page"},{"location":"MainModule/#","page":"Main Module","title":"Main Module","text":"Class labels are natural numbers in 1z, where z is the number of classes.","category":"page"},{"location":"MainModule/#Tips-and-Tricks-1","page":"Main Module","title":"Tips & Tricks","text":"","category":"section"},{"location":"MainModule/#the-ℍVector-type-1","page":"Main Module","title":"the ℍVector type","text":"","category":"section"},{"location":"MainModule/#","page":"Main Module","title":"Main Module","text":"Check this documentation on typecasting matrices.","category":"page"},{"location":"MainModule/#notation-and-nomenclature-1","page":"Main Module","title":"notation & nomenclature","text":"","category":"section"},{"location":"MainModule/#","page":"Main Module","title":"Main Module","text":"Throughout the code and the examples of this package the following notation is followed:","category":"page"},{"location":"MainModule/#","page":"Main Module","title":"Main Module","text":"scalars and vectors are denoted using lower-case letters, e.g., y,\nmatrices using upper case letters, e.g., X\nsets (vectors) of matrices using bold upper-case letters, e.g., 𝐗.","category":"page"},{"location":"MainModule/#","page":"Main Module","title":"Main Module","text":"The following nomenclature is used consistently:","category":"page"},{"location":"MainModule/#","page":"Main Module","title":"Main Module","text":"𝐏Tr: a training set of positive definite matrices\n𝐏Te: a testing set of positive definite matrices\n𝐏: a generic set of positive definite matrices.\nw: a weights vector of non-negative real numbers\nyTr: a training set class labels vector of positive integer numbers (1, 2,...)\nyTe: a testing set class labels vector of positive integer numbers\ny: a generic class labels vector of positive integer numbers.\nz: number of classes of a ML model\nk: number of matrices in a set","category":"page"},{"location":"MainModule/#acronyms-1","page":"Main Module","title":"acronyms","text":"","category":"section"},{"location":"MainModule/#","page":"Main Module","title":"Main Module","text":"MDM: minimum distance to mean\nCV: cross-validation","category":"page"},{"location":"#PosDefManifoldML-Documentation-1","page":"PosDefManifoldML Documentation","title":"PosDefManifoldML Documentation","text":"","category":"section"},{"location":"#Requirements-and-Installation-1","page":"PosDefManifoldML Documentation","title":"Requirements & Installation","text":"","category":"section"},{"location":"#","page":"PosDefManifoldML Documentation","title":"PosDefManifoldML Documentation","text":"Julia version ≥ 1.1.1","category":"page"},{"location":"#","page":"PosDefManifoldML Documentation","title":"PosDefManifoldML Documentation","text":"Packages: see the dependencies of the main module.","category":"page"},{"location":"#","page":"PosDefManifoldML Documentation","title":"PosDefManifoldML Documentation","text":"The package is still not registered. To install it, execute the following command in Julia's REPL:","category":"page"},{"location":"#","page":"PosDefManifoldML Documentation","title":"PosDefManifoldML Documentation","text":"]add https://github.com/Marco-Congedo/PosDefManifoldML","category":"page"},{"location":"#Disclaimer-1","page":"PosDefManifoldML Documentation","title":"Disclaimer","text":"","category":"section"},{"location":"#","page":"PosDefManifoldML Documentation","title":"PosDefManifoldML Documentation","text":"This package is still in a pre-release stage. Any independent reviewer for both the code and the documentation is welcome.","category":"page"},{"location":"#About-the-Authors-1","page":"PosDefManifoldML Documentation","title":"About the Authors","text":"","category":"section"},{"location":"#","page":"PosDefManifoldML Documentation","title":"PosDefManifoldML Documentation","text":"Saloni Jain is a student at the Indian Institute of Technology, Kharagpur, India.","category":"page"},{"location":"#","page":"PosDefManifoldML Documentation","title":"PosDefManifoldML Documentation","text":"Marco Congedo, corresponding author, is a research scientist of CNRS (Centre National de la Recherche Scientifique), working in UGA (University of Grenoble Alpes).","category":"page"},{"location":"#","page":"PosDefManifoldML Documentation","title":"PosDefManifoldML Documentation","text":"Contact: first name dot last name at gmail dot com","category":"page"},{"location":"#Overview-1","page":"PosDefManifoldML Documentation","title":"Overview","text":"","category":"section"},{"location":"#","page":"PosDefManifoldML Documentation","title":"PosDefManifoldML Documentation","text":"Riemannian geometry studies smooth manifolds, multi-dimensional curved spaces with peculiar geometries endowed with non-Euclidean metrics. In these spaces Riemannian geometry allows the definition of angles, geodesics (shortest path between two points), distances between points, centers of mass of several points, etc.","category":"page"},{"location":"#","page":"PosDefManifoldML Documentation","title":"PosDefManifoldML Documentation","text":"In several fields of research such as computer vision and brain-computer interface, treating data in the manifold of positive definite matrices has allowed the introduction of machine learning approaches with remarkable characteristics, such as simplicity of use, excellent classification accuracy, as demonstrated by the winning score obtained in six international data classification competitions, and the ability to operate transfer learning (Congedo et al., 2017a, Brachant et al., 2012)🎓.","category":"page"},{"location":"#","page":"PosDefManifoldML Documentation","title":"PosDefManifoldML Documentation","text":"In this package we are concerned with making use of Riemannian Geometry for classifying data in the form of positive definite matrices (e.g., covariance matrices, Fourier cross-spectral matrices, etc.). This can be done in two ways: either directly in the manifold of positive definite matrices using Riemannian machine learning methods or in the tangent space, where traditional (Euclidean) machine learning methods apply (i.e., linear discriminant analysis, support-vector machine, logistic regression, random forest, etc.).","category":"page"},{"location":"#","page":"PosDefManifoldML Documentation","title":"PosDefManifoldML Documentation","text":"(Image: Figure 1) Figure 1","category":"page"},{"location":"#","page":"PosDefManifoldML Documentation","title":"PosDefManifoldML Documentation","text":"Schematic representation of Riemannian classification. Data points are either natively positive definite matrices or are converted into this form. The classification can be performed by Riemannian methods in the manifold of positive definite matrices or by Euclidean methods after projection onto the tangent space.","category":"page"},{"location":"#","page":"PosDefManifoldML Documentation","title":"PosDefManifoldML Documentation","text":"For a formal introduction to the manifold of positive definite matrices the reader is referred to the monography written by Bhatia(2007)🎓.","category":"page"},{"location":"#","page":"PosDefManifoldML Documentation","title":"PosDefManifoldML Documentation","text":"For an introduction to Riemannian geometry and an overview of mathematical tools implemented in the PostDefManifold package, which is heavily used here, see Intro to Riemannian Geometry.","category":"page"},{"location":"#Code-units-1","page":"PosDefManifoldML Documentation","title":"Code units","text":"","category":"section"},{"location":"#","page":"PosDefManifoldML Documentation","title":"PosDefManifoldML Documentation","text":"PosDefManifoldML is light-weight. It includes four code units (.jl files):","category":"page"},{"location":"#","page":"PosDefManifoldML Documentation","title":"PosDefManifoldML Documentation","text":"Unit Description\nMainModule Main module, declaring internal constants and types\nmdm.jl Unit implementing the MDM( Minimum Distance to Mean) model\ntrain_test.jl Unit allowing fitting models, getting a prediction from there and performing cross-validations\ntools.jl Unit containing tools useful for classification","category":"page"},{"location":"#-1","page":"PosDefManifoldML Documentation","title":"🎓","text":"","category":"section"},{"location":"#","page":"PosDefManifoldML Documentation","title":"PosDefManifoldML Documentation","text":"References","category":"page"},{"location":"#","page":"PosDefManifoldML Documentation","title":"PosDefManifoldML Documentation","text":"A. Barachant, S. Bonnet, M. Congedo, C. Jutten (2012) Multi-class Brain Computer Interface Classification by Riemannian Geometry, IEEE Transactions on Biomedical Engineering, 59(4), 920-928.","category":"page"},{"location":"#","page":"PosDefManifoldML Documentation","title":"PosDefManifoldML Documentation","text":"A. Barachant, S. Bonnet, M. Congedo, C. Jutten (2013) Classification of covariance matrices using a Riemannian-based kernel for BCI applications, Neurocomputing, 112, 172-178.","category":"page"},{"location":"#","page":"PosDefManifoldML Documentation","title":"PosDefManifoldML Documentation","text":"R. Bhatia (2007) Positive Definite Matrices, Princeton University press.","category":"page"},{"location":"#","page":"PosDefManifoldML Documentation","title":"PosDefManifoldML Documentation","text":"M. Congedo, A. Barachant, R. Bhatia R (2017a) Riemannian Geometry for EEG-based Brain-Computer Interfaces; a Primer and a Review, Brain-Computer Interfaces, 4(3), 155-174.","category":"page"},{"location":"#","page":"PosDefManifoldML Documentation","title":"PosDefManifoldML Documentation","text":"M. Congedo, A. Barachant, E. Kharati Koopaei (2017b) Fixed Point Algorithms for Estimating Power Means of Positive Definite Matrices, IEEE Transactions on Signal Processing, 65(9), 2211-2220.","category":"page"},{"location":"#Contents-1","page":"PosDefManifoldML Documentation","title":"Contents","text":"","category":"section"},{"location":"#","page":"PosDefManifoldML Documentation","title":"PosDefManifoldML Documentation","text":"Pages = [       \"index.md\",\n\t\t\t\t\t\t\t\t\"tutorials.md\",\n                \"MainModule.md\",\n                \"mdm.md\",\n                \"train_test.md\",\n                \"tools.md\",\n\t\t]\nDepth = 1","category":"page"},{"location":"#Index-1","page":"PosDefManifoldML Documentation","title":"Index","text":"","category":"section"},{"location":"#","page":"PosDefManifoldML Documentation","title":"PosDefManifoldML Documentation","text":"","category":"page"},{"location":"tools/#tools.jl-1","page":"Tools","title":"tools.jl","text":"","category":"section"},{"location":"tools/#","page":"Tools","title":"Tools","text":"This unit implements tools that are useful for building Riemannian and Euclidean machine learning classifiers.","category":"page"},{"location":"tools/#Content-1","page":"Tools","title":"Content","text":"","category":"section"},{"location":"tools/#","page":"Tools","title":"Tools","text":"function description\nprojectOnTS project data on a tangent space to apply Euclidean ML models\nCVsetup generate indexes for performing cross-validtions\ngen2ClassData generate 2-class positive definite matrix data for testing Riemannian ML models","category":"page"},{"location":"tools/#","page":"Tools","title":"Tools","text":"projectOnTS\nCVsetup\ngen2ClassData","category":"page"},{"location":"tools/#PosDefManifoldML.projectOnTS","page":"Tools","title":"PosDefManifoldML.projectOnTS","text":"function projectOnTS(metric :: Metric,\n                     𝐏      :: ℍVector;\n                  w  :: Vector = [],\n                  ✓w :: Bool   = true,\n                  ⏩ :: Bool   = true)\n\nGiven a vector of k Hermitian matrices 𝐏 and corresponding optional non-negative weights w, return a matrix with the matrices 𝐏 mapped onto the tangent space at base-point given by their mean and vectorized as per the vecP operation.\n\nTangent space mapping of matrices P_i i=1k at base point G according to the Fisher metric is given by:\n\nS_i=G^½ textrmlog(G^-½ P_i G^-½) G^½.\n\nnote: Nota Bene\nthe tangent space projection is currently supported only for the Fisher metric, therefore this metric is used for the projection.\n\nThe mean of the meatrices in 𝐏 is computed according to the specified metric, of type Metric. A natural choice is the Fisher metric. The weighted mean is computed if weights vector w is non-empty. By default the unweighted mean is computed.\n\nIf w is non-empty and optional keyword argument ✓w is true (default), the weights are normalized so as to sum up to 1, otherwise they are used as they are passed and should be already normalized. This option is provided to allow calling this function repeatedly without normalizing the same weights vector each time.\n\nif optional keyword argument ⏩ if true (default), the computation of the mean is multi-threaded if this is obtained with an iterative algorithm (e.g., using the Fisher metric). Multi-threading is automatically disabled if the number of threads Julia is instructed to use is 2 or 4k.\n\nReturn a matrix holding the k mapped matrices in its columns. The dimension of the columns is n(n+1)2, where n is the size of the matrices in 𝐏 (see vecP ). The arrangement of tangent vectors in the columns of a matrix is natural in Julia, however if you export the tagent vectors to be used as feature vectors keep in mind that several ML packages, for example Python scikitlearn, expect them to be arranged in rows.\n\nExamples:\n\nusing PosDefManifoldML\n\n# generate four random symmetric positive definite 3x3 matrices\n𝐏=randP(3, 4)\n\n# project and vectorize in the tangent space\nT=projectOnTS(Fisher, 𝐏)\n\n# The result is a 6x4 matrix, where 6 is the size of the\n# vectorized tangent vectors (n=3, n*(n+1)/2=6)\n\nSee: the ℍVector type.\n\n\n\n\n\n","category":"function"},{"location":"tools/#PosDefManifoldML.CVsetup","page":"Tools","title":"PosDefManifoldML.CVsetup","text":"function CVsetup(k       :: Int,\n                 nCV     :: Int;\n                 shuffle :: Bool = false)\n\nGiven k elements and a parameter nCV, a nCV-fold cross-validation is obtained defining nCV permutations of k elements in nTest=knCV (intger division) elements for the test and k-nTest elements for the training, in such a way that each element is represented in only one permutation.\n\nSaid differently, given a length k and the number of desired cross-validations nCV, this function generates indices from the sequence of natural numbers 1k to obtain all nCV-fold cross-validation sets. Specifically, it generates nCV vectors of indices for generating test sets and nCV vectors of indices for geerating training sets.\n\nIf optional keyword argument shuffle is true, the sequence of natural numbers 1k is shuffled before running the function, thus in this case two successive runs of this function will give different cross-validation sets, hence different accuracy scores. By default shuffle is false, so as to allow exactly the same result in successive runs. Notae that no random initialization for the shuffling is provided, so as to allow the replication of the same random sequences starting again the random generation from scratch.\n\nThis function is used in CV_mdm. It constitutes the fundamental basis to implement customized cross-validation iprocedures.\n\nReturn the 4-tuple with:\n\nThe size of each training set (integer),\nThe size of each testing set (integer),\nA vector of nCV vectors holding the indices for the training sets,\nA vector of nCV vectors holding the indices for the corresponding test sets.\n\nExamples\n\nusing PosDefManifoldML\n\nCVsetup(10, 2)\n# return:\n# (5, 5,\n# Array{Int64,1}[[6, 7, 8, 9, 10], [1, 2, 3, 4, 5]])\n# Array{Int64,1}[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],\n\nCVsetup(10, 2, shuffle=true)\n# return:\n# (5, 5,\n# Array{Int64,1}[[5, 4, 6, 1, 9], [3, 7, 8, 2, 10]])\n# Array{Int64,1}[[3, 7, 8, 2, 10], [5, 4, 6, 1, 9]],\n\nCVsetup(10, 3)\n# return:\n# (7, 3,\n# Array{Int64,1}[[4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6]])\n# Array{Int64,1}[[1, 2, 3], [4, 5, 6], [7, 8, 9, 10]],\n\n\n\n\n\n\n","category":"function"},{"location":"tools/#PosDefManifoldML.gen2ClassData","page":"Tools","title":"PosDefManifoldML.gen2ClassData","text":"function gen2ClassData(n        ::  Int,\n                       k1train  ::  Int,\n                       k2train  ::  Int,\n                       k1test   ::  Int,\n                       k2test   ::  Int,\n                       separation :: Real = 0.1)\n\nGenerate a training set of k1train+k2train and a test set of k1test+k2test symmetric positive definite matrices. All matrices have size nxn.\n\nThe training and test sets can be used to train and test an ML model.\n\nseparation is a coefficient determining how well the two classs are separable; the higher it is, the more separable the two classes are. It must be in [0, 1] and typically a value of 0.5 already determines complete separation.\n\nReturn a 4-tuple with\n\nan ℍVector holding the k1train+k2train matrices in the training set,\nan ℍVector holding the k1test+k2test matrices in the test set,\na vector holding the k1train+k2train labels (integers) corresponding to the matrices of the training set,\na vector holding the k1test+k2test labels corresponding to the matrices of the test set (1 for class 1 and 2 for class 2).\n\nExamples\n\nusing PosDefManifoldML\n\n𝐏Tr, 𝐏Te, yTr, yTe=gen2ClassData(10, 30, 40, 60, 80, 0.25)\n\n# 𝐏Tr=training set: 30 matrices for class 1 and 40 matrices for class 2\n# 𝐏Te=testing set: 60 matrices for class 1 and 80 matrices for class 2\n# all matrices are 10x10\n# yTr=a vector of 70 labels for 𝐓r\n# yTe=a vector of 140 labels for 𝐓e\n\n\n\n\n\n\n","category":"function"},{"location":"train_test/#train_test.jl-1","page":"Training-Testing","title":"train_test.jl","text":"","category":"section"},{"location":"train_test/#","page":"Training-Testing","title":"Training-Testing","text":"This unit implements train-testing and cross-validation procedures.","category":"page"},{"location":"train_test/#Content-1","page":"Training-Testing","title":"Content","text":"","category":"section"},{"location":"train_test/#","page":"Training-Testing","title":"Training-Testing","text":"Function Description\nfit! fit (train) a machine larning model\npredict predict labels or probabilities\nCVscore cross-validation score","category":"page"},{"location":"train_test/#","page":"Training-Testing","title":"Training-Testing","text":"fit!\npredict\nCVscore","category":"page"},{"location":"train_test/#PosDefManifoldML.fit!","page":"Training-Testing","title":"PosDefManifoldML.fit!","text":"function fit!(model :: MLmodel,\n              𝐏Tr     :: ℍVector,\n              yTr     :: Vector;\n           w  :: Vector= [],\n           ✓w :: Bool  = true,\n           ⏩ :: Bool  = true)\n\nFit a machine learning model ML model, with training data 𝐏Tr, of type ℍVector, and corresponding labels yTr, of type IntVector. Return the fitted model.\n\nOnly the MDM model is supported for the moment being.\n\nMDM model\n\nFor this model, fitting involves computing a mean of all the matrices in each class. Those class means are computed according to the metric specified by the MDM constructor.\n\nSee method (3) of the mean function for the meaning of the optional keyword arguments w, ✓w and ⏩, to which they are passed.\n\nNote that the MDM model can be created and fitted in one pass using a specia MDM constructor.\n\nSee: notation & nomenclature, the ℍVector type.\n\nSee also: predict, CVscore.\n\nExamples\n\nusing PosDefManifoldML\n\n# generate some data\n𝐏Tr, 𝐏Te, yTr, yTe=gen2ClassData(10, 30, 40, 60, 80, 0.25)\n\n# create an MDM model\nmodel = MDM(Fisher)\n\n# fit (train) the model\nfit!(model, 𝐏Tr, yTr)\n\n# using a special constructor you don't need the fit! function:\n𝐏Tr, 𝐏Te, yTr, yTe=gen2ClassData(10, 30, 40, 60, 80, 0.25)\nmodel=MDM(Fisher, 𝐏Tr, yTr)\n\n\n\n\n\n","category":"function"},{"location":"train_test/#PosDefManifoldML.predict","page":"Training-Testing","title":"PosDefManifoldML.predict","text":"function predict(model  :: MLmodel,\n                 𝐏Te    :: ℍVector,\n                 what   :: Symbol=:labels)\n\nGiven a ML model model trained (fitted) on z classes and a testing set of k positive definite matrices 𝐏Te of type ℍVector,\n\nif what is :labels or :l (default), return the predicted class labels for each matrix in 𝐏Te as an IntVector;\n\nif what is :probabilities or :p, return the predicted probabilities for each matrix in 𝐏Te to belong to a all classes, as a k-vector of z vectors holding reals in (0 1) (probabilities).\n\nOnly the MDM model is supported for the moment being.\n\nMDM model\n\nFor this model, the predicted class of an unlabeled matrix is the class whose mean is the closest to the matrix (minimum distance to mean).\n\nThe probabilities instead are obtained passing to a softmax function the distances of each unlabeled matrix to all class means.\n\nSee: notation & nomenclature, the ℍVector type.\n\nSee also: fit!, CVscore.\n\nExamples\n\nusing PosDefManifoldML\n\n# generate some data\n𝐏Tr, 𝐏Te, yTr, yTe=gen2ClassData(10, 30, 40, 60, 80)\n\n# craete and fit an MDM model\nmodel=MDM(Fisher, 𝐏Tr, yTr)\n\n# predict labels\npredict(model, 𝐏Te, :l)\n\n# predict probabilities\npredict(model, 𝐏Te, :p)\n\n\n\n\n\n","category":"function"},{"location":"train_test/#PosDefManifoldML.CVscore","page":"Training-Testing","title":"PosDefManifoldML.CVscore","text":"function CVscore(model :: MLmodel,\n                 𝐏Tr   :: ℍVector,\n                 yTr   :: Vector,\n                 nCV   :: Int = 5;\n            scoring   :: Symbol = :b,\n            confusion :: Bool   = false,\n            shuffle   :: Bool   = false)\n\nCross-validation: Given an ℍVector 𝐏Tr holding k Hermitian matrices, a Vector yTr holding the k labels for these matrices, the number of cross-validations nCV and an ML model model, retrun a vector scores of nCV accuracies, one for each cross-validation.\n\nIf scoring= :b (default) the balanced accuracy is computed. Any other value will make the function returning the regular accuracy.\n\nIf confusion=true (dafault=false), return the 2-tuple C, scores, where C is a nCV-vector of the confusion matrices for each cross-validation set, otherwise return only scores.\n\nSee: notation & nomenclature, the ℍVector type.\n\nSee also: fit!, predict.\n\nExamples\n\nusing PosDefManifoldML\n\n# generate some data\n𝐏Tr, 𝐏Te, yTr, yTe=gen2ClassData(10, 30, 40, 60, 80)\n\n# craete and fit an MDM model\nmodel=MDM(Fisher, 𝐏Tr, yTr)\n\n# perform cross-validation\nCVscore(model, 𝐏Te, yTe, 5)\n\n\n\n\n\n","category":"function"}]
}
