Distributed NMF Library
=======================

In this repository, we offer both MPI and OPENMP implementation for MU, HALS and ANLS/BPP based NMF algorithms. This can run off the shelf as well easy to integrate in other source code. 
These are very highly tuned NMF algorithms to work on super computers. We have tested
this software in [NERSC](http://www.nersc.gov/users/computational-systems/edison/) as well [OLCF](https://www.olcf.ornl.gov/) cluster. The openmp implementation is tested on
many different linux variants with intel processors. The library works well for both sparse and dense matrix.
If you use this code, kindly cite the following papers appropriately.

* Ramakrishnan Kannan, Grey Ballard, and Haesun Park. 2016. A high-performance parallel algorithm for nonnegative matrix factorization. In Proceedings of the 21st ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP '16). ACM, New York, NY, USA, , Article 9 , 11 pages. DOI: http://dx.doi.org/10.1145/2851141.2851152
* James P. Fairbanks, Ramakrishnan Kannan, Haesun Park, David A. Bader, Behavioral clusters in dynamic graphs, Parallel Computing, Volume 47, August 2015, Pages 38-50, ISSN 0167-8191, http://dx.doi.org/10.1016/j.parco.2015.03.002.
* Kannan, Ramakrishnan. "SCALABLE AND DISTRIBUTED CONSTRAINED LOW RANK APPROXIMATIONS." (Doctoral Disseration) (2016). https://smartech.gatech.edu/handle/1853/54962

In this library we support the following

* Shared Memory [OPENMP](openmp/README.md)  based implementation
* Distributed Memory [MPI](mpi/README.md) based implementation

For other NMF implementations from Prof. Park's lab, please visit [smallk](https://github.com/smallk/smallk)

Contributors
=============

* [Ramakrishnan Kannan](https://sites.google.com/site/ramakrishnankannan/) - Oak Ridge National Laboratory
* [Grey Ballard](http://users.wfu.edu/ballard/) - Wake Forest University
* [Haesun Park](http://www.cc.gatech.edu/~hpark/) - Georgia Institute of Technology, GA

Acknowledgements
================

This research was supported by UT-Battelle, LLC under Contract No. DE-AC05-00OR22725 with the U.S. Department of Energy. This project was partially funded by the Laboratory Director`s Research and Development fund. This research used resources of the Oak Ridge Leadership Computing Facility at the Oak Ridge National Laboratory, which is supported by the Office of Science of the U.S. Department of Energy.

Also, partial funding for this work was provided by AFOSR Grant FA9550-13-1-0100, National Science Foundation (NSF) grants IIS-1348152 and ACI-1338745, Defense Advanced Research Projects Agency (DARPA) XDATA program grant FA8750-12-2-0309.

The United States Government retains and the publisher, by accepting the article for publication, acknowledges that the United States Government retains a non-exclusive, paid-up, irrevocable, world-wide license to publish or reproduce the published form of this manuscript, or allow others to do so, for United States Government purposes. The Department of Energy will provide public access to these results of federally sponsored research in accordance with the DOE Public Access Plan http://energy.gov/downloads/doepublic-access-plan. 

Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the USDOE, NERSC, AFOSR, NSF or DARPA.

Version 0.51
===========
* Moved from Single precision to double precision

Version 0.5
===========
* Support for L1 and L2 regularization
* Removed boost and mkl requirements. Works on Cray systems such as EOS and Titan as well
* Bug fixes on error computation
