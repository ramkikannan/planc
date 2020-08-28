# OpenMP Non-negative Matrix Factorization 

## Install Instructions


This program depends on:

- Armadillo library which can be found at https://arma.sourceforge.net
- OpenBLAS https://github.com/xianyi/OpenBLAS. If building with OpenBLAS
and mkl is discoverable by cmake, use -DCMAKE_IGNORE_MKL=1. 

Once you have installed these libraries set the following environment variables.

````
export ARMADILLO_INCLUDE_DIR=/home/rnu/libraries/armadillo-6.600.5/include/
export LIB=$LIB:/home/rnu/libraries/openblas/lib:
export INCLUDE=$INCLUDE:/home/rnu/libraries/openblas/include:$ARMADILLO_INCLUDE_DIR:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rnu/libraries/openblas/lib/:
export CPATH=CPATH:$INCLUDE:
````

* Create a build directory. 
* Change to the build directory 
* In case of MKL, source the ````$MKLROOT/bin/mkl_vars.sh intel64````
* run cmake [PATH TO THE CMakeList.txt]
* make

### Sparse NMF

Run cmake with -DCMAKE_BUILD_SPARSE=1

### Sparse Debug build

Run cmake with -DCMAKE_BUILD_SPARSE -DCMAKE_BUILD_TYPE=Debug

### Building on Cray-EOS/Titan

CC=CC CXX=CC cmake ~/nmflibrary/mpi/ -DCMAKE_IGNORE_MKL=1

## Intel MKL vs Openblas

- ````export LD_LIBRARY_PATH=MKL_LIB path````
- source the ````$MKLROOT/bin/mkl_vars.sh intel64````

## Runtime usage

Tell OpenBlas how many threads you want to use. For example on a quad core system use the following.

````
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:MKL_LIB
````

## Command Line options

The single alphabet is called short option and the string equivalent is called
long option. For eg., "input" is the long equivalent of short option 'i'. 
Typically long option is passed with "--algo=3" and short option with "-a 0".
The following is the brief description of  the various command line options. 

* {"input",'i'} - Either it can be a path to a sparse/dense 
matrix file or synthetically generate matrix. If this 
option is not passed, we generate synthetic matrix.
* {"algo",'a'} - We support four algorithms. 
  0 - Multiplicative update (MU)
  1 - Hierarchical Alternating Least Squares (HALS)
  2 - ANLS/BPP implementation  
* {"lowrank",'k'} - Low rank 'k'. 
* {"iter",'t'} - Number of iterations
* {"dimensions",'d'} - This is applicable only for synthetic matrices. It takes
 space separated dimensions as string. For eg., "21000 16000" means 21000 rows 
 and 16000 columns.
* {'o'} - File name to dump W and H. _w and _h will be appended to distinguish
 W and H matrix.  
* {"sparsity",'s'} - Density for the synthetic sparse matrix. 

Few usage examples are
Usage 1 : Sparse/Dense NMF for an input file with lowrank k=20 for 20 iterations.  
````NMFLibrary --algo=[0/1/2] --lowrank=20 --input=filename --iter=20 ````

Usage 2 : Sparse/Dense synthetic NMF for a 20000x10000 matrix
````NMFLibrary --algo=[0/1/2] --lowrank=20 -p "20000 10000" --iter=20 ````

Usage3 : Sparse/Dense NMF for an input file with lowrank k=20 for 20 iterations starting
from the initialization matrix defined in winit and hinit. Finally, it dumps the output
W and H in the specified file
````NMFLibrary --algo=[0/1/2] --lowrank=20 --input=filename --winit=filename --hinit=filename --w=woutputfilename --h=outputfilename --iter=20````


Citation
========

If you are using this openmp implementation, kindly cite.

James P. Fairbanks, Ramakrishnan Kannan, Haesun Park, David A. Bader, Behavioral clusters in dynamic graphs, Parallel Computing, Volume 47, August 2015, Pages 38-50, ISSN 0167-8191, http://dx.doi.org/10.1016/j.parco.2015.03.002.
