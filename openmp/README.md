Install Instructions
====================

This program depends on:

- Armadillo library which can be found at https://arma.sourceforge.net
- OpenBLAS https://github.com/xianyi/OpenBLAS
- Boost C++ libraries 1.55 http://www.boost.org/ Set Operations

Once you have installed these libraries set the following environment variables.

export ARMADILLO_INCLUDE_DIR=/home/rnu/libraries/armadillo-6.600.5/include/
export LIB=$LIB:/home/rnu/libraries/openblas/lib:
export INCLUDE=$INCLUDE:/home/rnu/libraries/openblas/include:$ARMADILLO_INCLUDE_DIR:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rnu/libraries/openblas/lib/:
export CPATH=CPATH:$INCLUDE:

cd build and run cmake ../

Sparse NMF
---------
cd build and run cmake ../ -DCMAKE_BUILD_SPARSE

Sparse Debug build
------------------
cd build and run cmake ../ -DCMAKE_BUILD_SPARSE -DCMAKE_BUILD_TYPE=Debug

Once cmake is ready without any errors, run make. 


Runtime usage
=============
Tell OpenBlas how many threads you want to use. For example on a quad core system use the following.
````
export OPENBLAS_NUM_THREADS=4
 export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:MKL_LIB
````

Intel MKL vs Openblas
=====================
export LD_LIBRARY_PATH=MKL_LIB path

1. Change in BPPNNLS.hpp header inclusion and the call to dpotrs in the function solveSymmetricLinearEquations
2. In the NMF.cpp comment cblas.h
3. make sure there is no compilation problem/warning specifically about openmp


Citation
========

If you are using this openmp implementation, kindly cite.

James P. Fairbanks, Ramakrishnan Kannan, Haesun Park, David A. Bader, Behavioral clusters in dynamic graphs, Parallel Computing, Volume 47, August 2015, Pages 38-50, ISSN 0167-8191, http://dx.doi.org/10.1016/j.parco.2015.03.002.