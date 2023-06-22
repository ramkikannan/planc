# Parallel Low-rank Approximation with Nonnegativity Constraints (PLANC)

Given an input matrix A, Nonnegative Matrix Factorization (NMF) is about determining two nonnegative matrices called factors, corresponding to the rows and columns of A respectively,  such that the product of them closely approximates A. In recent times, there is a huge interest for nonnegative factorizations for data with more than two modes, which can be represented via a generalization of matrices known as tensors. There are wide applications of nonnegative matrix and tensor factorization (NTF) in the scientific community such as spectral unmixing, compressing scientific data, scientific visualization, inverse problems, feature engineering for deep learning etc., as the factors are scientifically interpretable and an important cornerstone for explainable AI and representation learning. Similarly for internet applications such as topic modeling, background separation from video data, hyper-spectral imaging, web-scale clustering, and community detection.  There are many popular algorithms of NMF and NTF such as multiplicative update (MU), hierarchical alternating least squares (HALS), ANLS/BPP etc., and it was a technically highly complex task for scaling them to large number of processor to handle the internet scale scientific data.

PLANC delivers the high performance, flexibility, and scalability necessary to tackle the ever-growing size of today's internet and scientific data sets. Rather than developing separate software for each problem domain, algorithm and mathematical technique, flexibility has been achieved by characterizing nearly all of the current NMF and NTF algorithms in the context of a Block Coordinate Descent (BCD) framework.

In this repository, we offer highly tuned MPI and OpenMP implementation for MU, HALS, Nesterov, ADMM and ANLS/BPP based NMF and NTF algorithms that delivers the high performance, flexibility, and scalability necessary to tackle the ever-growing size of today's internet and scientific data sets. This can run off the shelf as well easy to integrate in other source code. We have tested
this software in [NERSC](http://www.nersc.gov/users/computational-systems/edison/), [OLCF](https://www.olcf.ornl.gov/), and [PACE](https://pace.gatech.edu/) clusters. The OpenMP implementation is tested on many different linux variants with Intel processors. The library works well for both sparse and dense matrix. 

The following variants of matrix and tensor approximations are supported.
* [Shared Memory NMF](nmf/README.md) 
* [Distributed Memory NMF](distnmf/README.md)
* [Shared Memory NTF](ntf/README.md)
* [Distributed Memory NTF](distntf/README.md)
* [Hierarchical NMF](hiernmf/README.md)
* [Shared Memory JointNMF](jointnmf/README.md)
* [Distributed Memory JointNMF](distjointnmf/README.md)

For other NMF implementations from Prof. Park's lab, please visit [smallk](https://github.com/smallk/smallk)

## Documentation

[PLANC API](https://ramkikannan.github.io/planc-api/)

## Contributors

* [Ramakrishnan Kannan](https://ramkikannan.github.io) - Oak Ridge National Laboratory
* [Grey Ballard](https://users.wfu.edu/ballard/) - Wake Forest University
* [Haesun Park](https://faculty.cc.gatech.edu/~hpark/) - Georgia Institute of Technology, GA
* [Rich Vuduc](https://vuduc.org/v2/) - Georgia Institute of Technology, GA
* [Oguz Kaya](https://kayaogz.github.io/) - Université Paris-Sud/Paris-Saclay
* [Michael A. Matheson](https://www.ornl.gov/staff-profile/michael-matheson) - Oak Ridge National Laboratory
* [Srinivas Eswar](https://seswar3.gitlab.io) - Argonne National Laboratory
* [Koby Hayashi](https://hayakb95.github.io/) - Georgia Institute of Technology, GA
* Lawton Manning - Wake Forest University
* [Ben Cobb](http://www.ben-cobb.com/) - Georgia Institute of Technology, GA

## Build Procedure

* For building individual factorizations, please look at the individual directories
* For building all the components, please refer to the [README](build/README.md) under the build directory 

## Acknowledgements

This research was supported by UT-Battelle, LLC under Contract No. DE-AC05-00OR22725 with the U.S. Department of Energy. This project was partially funded by the Laboratory Director`s Research and Development fund. This research used resources of the Oak Ridge Leadership Computing Facility at the Oak Ridge National Laboratory, which is supported by the Office of Science of the U.S. Department of Energy.

Also, partial funding for this work was provided by AFOSR Grant FA9550-13-1-0100, National Science Foundation (NSF) grants IIS-1348152 and ACI-1338745, Defense Advanced Research Projects Agency (DARPA) XDATA program grant FA8750-12-2-0309.

The United States Government retains and the publisher, by accepting the article for publication, acknowledges that the United States Government retains a non-exclusive, paid-up, irrevocable, world-wide license to publish or reproduce the published form of this manuscript, or allow others to do so, for United States Government purposes. The Department of Energy will provide public access to these results of federally sponsored research in accordance with the [DOE Public Access Plan](http://energy.gov/downloads/doepublic-access-plan). 

Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the USDOE, NERSC, AFOSR, NSF or DARPA.

## How to cite
If you use this code, kindly cite the following [papers](papers.md) appropriately. [Datasets](datasets.md) used in the different papers are also available. The software papers are shown below.

> 1.  Srinivas Eswar, Koby Hayashi, Grey Ballard, Ramakrishnan Kannan, Michael A. Matheson, and Haesun Park. 2021. PLANC: Parallel Low-rank Approximation with Nonnegativity Constraints. ACM Trans. Math. Softw. 47, 3, Article 20 (September 2021), 37 pages. [[doi]](https://doi.org/10.1145/3432185)
> 2. Ramakrishnan Kannan, Grey Ballard and Haesun Park, "MPI-FAUN: An MPI-Based Framework for Alternating-Updating Nonnegative Matrix Factorization," in IEEE Transactions on Knowledge and Data Engineering, vol. 30, no. 3, pp. 544-558, 1 March 2018. [[doi]](https://doi.org/10.1109/TKDE.2017.2767592)

## Release Details
### Version 0.82
* Added JointNMF for handling multimodal inputs in the form of attributed graphs (shared and distributed-memory)
* New input methods for reading in sparse matrices from a single file

### Version 0.81

* Added symmetric regularization for NMF (SymNMF) for the ANLS variant
* Added Gauss-Newton algorithm for SymNMF
* Handle uneven split of rows and columns of the input matrix in the processor grid
* Added MPI-IO for NMF
* Added Rank-2 NMF and HierNMF

### Version 0.8

* Newly added algorithms - CP-ALS, Nesterov, ADMM for both NMF and NTF
* New build procedure
* Offloding dense operations to GPU through NVBLAS
* Loading and processing [NumPy](https://numpy.org/) arrays

### Version 0.51

* Moved from single precision to double precision

### Version 0.5

* Support for L1 and L2 regularization
* Removed boost and mkl requirements. Works on Cray systems such as EOS and Titan as well
* Bug fixes on error computation
