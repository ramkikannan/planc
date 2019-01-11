# Parallel Low-rank Approximations with Non-negativity Constraints (PLANC)

Given an input matrix A, Non-negative Matrix Factorization(NMF) is about determining two non-negative matrices called factors; each for rows and columns of A;  such that the product of them closely approximates the A. In the recent times, there is a huge interest for non-negative tensor factorization for determining factors of every mode of higher order matrices called tensors. There are wider applications of this matrix and tensor factorization in scientific community such as spectral unmixing, compressing scientific data, scientific visualization, inverse problems etc., as the factors are scientifically interpretable and an important cornerstone for explainable AI and representation learning. Similarly for internet applications such as topic modeling, background separation from video data, hyper-spectral imaging, web-scale clustering, and community detection.  There are many popular algorithms of NMF and NTF such as multiplicative update(MU), hierarchical alternating least squares(HALS), ANLS/BPP etc., and it was a technically highly complex task for scaling them to large number of processor to handle the internet scale scientific data.

PLANC delivers the high performance, flexibility, and scalability necessary to tackle the ever-growing size of today's internet and scientific data sets. Rather than developing separate software for each problem domain, algorithm and mathematical technique, flexibility has been achieved by characterizing nearly all of the current NMF and NTF algorithms in the context of a Block Coordinate Descent(BCD) framework.

In this repository, we offer highly tuned MPI and OPENMP implementation for MU, HALS, Nesterov, ADMM and ANLS/BPP based Non-negative Matrix Factorization(NMF) and Non-negative Tensor Factorization(NTF) algorithms that delivers the high performance, flexibility, and scalability necessary to tackle the ever-growing size of today's internet and scientific data sets. This can run off the shelf as well easy to integrate in other source code. We have tested
this software in [NERSC](http://www.nersc.gov/users/computational-systems/edison/) as well [OLCF](https://www.olcf.ornl.gov/) cluster. The openmp implementation is tested on
many different linux variants with intel processors. The library works well for both sparse and dense matrix. If you use this code, kindly cite the following papers appropriately.

* Ramakrishnan Kannan, Grey Ballard, and Haesun Park. 2016. A high-performance parallel algorithm for nonnegative matrix factorization. In Proceedings of the 21st ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP '16). ACM, New York, NY, USA, , Article 9 , 11 pages. DOI: http://dx.doi.org/10.1145/2851141.2851152
* James P. Fairbanks, Ramakrishnan Kannan, Haesun Park, David A. Bader, Behavioral clusters in dynamic graphs, Parallel Computing, Volume 47, August 2015, Pages 38-50, ISSN 0167-8191, http://dx.doi.org/10.1016/j.parco.2015.03.002.
* Kannan, Ramakrishnan. "SCALABLE AND DISTRIBUTED CONSTRAINED LOW RANK APPROXIMATIONS." (Doctoral Disseration) (2016). https://smartech.gatech.edu/handle/1853/54962
* 	Ramakrishnan Kannan, Grey Ballard, Haesun Park:
MPI-FAUN: An MPI-Based Framework for Alternating-Updating Nonnegative Matrix Factorization. IEEE Trans. Knowl. Data Eng. 30(3): 544-558 (2018)
* Oguz Kaya, Ramakrishnan Kannan, Grey Ballard:
Partitioning and Communication Strategies for Sparse Non-negative Matrix Factorization. ICPP 2018: 90:1-90:10
* 	Grey Ballard, Koby Hayashi, Ramakrishnan Kannan:
Parallel Nonnegative CP Decomposition of Dense Tensors. 25th {IEEE} International Conference on High Performance Computing(HiPC) 2018, Accepted

In this library we support the following

* [Shared Memory NMF](nmf/README.md)  based implementation of NMF and NTF
* [Distributed Memory NMF](distnmf/README.md) based implementation
* [Shared Memory NTF](ntf/README.md)  based implementation of NMF and NTF
* [Distributed Memory NTF](distntf/README.md) based implementation

For other NMF implementations from Prof. Park's lab, please visit [smallk](https://github.com/smallk/smallk)

## Contributors

* [Ramakrishnan Kannan](https://ramkikannan.github.io) - Oak Ridge National Laboratory
* [Grey Ballard](http://users.wfu.edu/ballard/) - Wake Forest University
* [Haesun Park](http://www.cc.gatech.edu/~hpark/) - Georgia Institute of Technology, GA
* Srinivas Eswar - Georgia Institute of Technology, GA
* Koby Hayashi - Georgia Institute of Technology, GA
* Michael A. Matheson - Oak Ridge National Laboratory
* [Oguz Kaya](http://kayaogz.github.io/) - Assistant professorship position (maître de conférences) at Université Paris-Sud/Paris-Saclay

## Build Procedure

* For building individual Non-negative Matrix and Tensor Factorization look at the individual directories
* For building all the components, refer [README](build/README.md) under build directory. 

## Acknowledgements

This research was supported by UT-Battelle, LLC under Contract No. DE-AC05-00OR22725 with the U.S. Department of Energy. This project was partially funded by the Laboratory Director`s Research and Development fund. This research used resources of the Oak Ridge Leadership Computing Facility at the Oak Ridge National Laboratory, which is supported by the Office of Science of the U.S. Department of Energy.

Also, partial funding for this work was provided by AFOSR Grant FA9550-13-1-0100, National Science Foundation (NSF) grants IIS-1348152 and ACI-1338745, Defense Advanced Research Projects Agency (DARPA) XDATA program grant FA8750-12-2-0309.

The United States Government retains and the publisher, by accepting the article for publication, acknowledges that the United States Government retains a non-exclusive, paid-up, irrevocable, world-wide license to publish or reproduce the published form of this manuscript, or allow others to do so, for United States Government purposes. The Department of Energy will provide public access to these results of federally sponsored research in accordance with the DOE Public Access Plan http://energy.gov/downloads/doepublic-access-plan. 

Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the USDOE, NERSC, AFOSR, NSF or DARPA.

## Release Details

### Version 0.51

* Moved from Single precision to double precision

### Version 0.5

* Support for L1 and L2 regularization
* Removed boost and mkl requirements. Works on Cray systems such as EOS and Titan as well
* Bug fixes on error computation

### Version 0.8

* Newly added algorithms - CP-ALS, Nesterov, ADMM for both NMF and NTF
* New build procedure
* Offloding dense operations to GPU through NVBLAS
* Loading and processing NUMPY arrays