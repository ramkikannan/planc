Install Instructions
====================

This program depends on:

- Armadillo library which can be found at https://arma.sourceforge.net
- OpenBLAS https://github.com/xianyi/OpenBLAS
- Boost C++ libraries 1.55 http://www.boost.org/ Set Operations

Once you have installed these libraries set the following environment variables.

````
export ARMADILLO_INCLUDE_DIR=/home/rnu/libraries/armadillo-6.600.5/include/
export LIB=$LIB:/home/rnu/libraries/openblas/lib:
export INCLUDE=$INCLUDE:/home/rnu/libraries/openblas/include:$ARMADILLO_INCLUDE_DIR:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rnu/libraries/openblas/lib/:
export NMFLIB_DIR=/ccs/home/ramki/rhea/research/nmflib/
export INCLUDE=$INCLUDE:$ARMADILLO_INCLUDE_DIR:$NMFLIB_DIR
export CPATH=$CPATH:$INCLUDE:
export MKLROOT=/ccs/compilers/intel/rh6-x86_64/16.0.0/mkl/
````

If you have got MKL, please source MKLVARS.sh before running make/cmake

Sparse NMF
---------
Run cmake with -DCMAKE_BUILD_SPARSE

Sparse Debug build
------------------
Run cmake with -DCMAKE_BUILD_SPARSE -DCMAKE_BUILD_TYPE=Debug

Other Macros
-------------

* CMAKE macros

  For sparse NMF - cmake -DBUILD_SPARSE=1 - Default dense build
  For timing with barrier after mpi calls - cmake -DCMAKE_WITH_BARRIER_TIMING - Default with barrier timing
  For performance, disable the WITH__BARRIER__TIMING. Run as "cmake -DCMAKE_WITH_BARRIER_TIMING:BOOL=OFF"

* Code level macros - Defined in distutils.h

  MPI_VERBOSE - Be doubly sure about what you do. It prints all intermediary matrices.
			   Try this only for very very small matrix that is of size less than 10.
  WRITE_RAND_INPUT - for dumping the generated random matrix

Command Line options
====================

The single alphabet is called short option and the string equivalent is called long option. 
For eg., "input" is the long equivalent of short option 'i'. Typically long option is passed
with "--algo=3" and short option with "-a 0". The following is the brief description of 
the various command line options. 

* {"input",'i'} - Either it can be a path to a sparse/dense 
matrix file or synthetically generate matrix. Synthetic matrices can be 
generated as rand_normal, rand_uniform or rand_lowrank
* {"algo",'a'} - We support four algorithms. 
  0 - Multiplicative update (MU)
  1 - Hierarchical Alternating Least Squares (HALS)
  2 - ANLS/BPP 2D implementation
  3 - Naive ANLS/BPP 
* {"error",'e'} - Must be called as -e 1 to compute error
after every iteration.
* {"lowrank",'k'} - Low rank 'k'. 
* {"iter",'t'} - Number of iterations
* {"rows",'m'} - This is applicable only for synthetic matrices. 
* {"columns",'n'} - This is applicable only for synthetic matrices. 
* {"output",'o'} - File name to dump W and H matrix. 
It will saved as outputfilename_W_MPIRANK and outputfilename_H_MPIRANK. 
* {"sparsity",'s'} - Density for the synthetic sparse matrix. 
* {"pr"} - Number of 2D rows
* {"pc",} - Number of 2D cols

Few usage examples are
Sparse synthetic distributed NMF on 96 processors 
````mpirun -np 96 ./distnmf -a 2 -m 207360 -n 138240 -k 50 --pr=12 --pc=8 -i rand_lowrank -t 30  -s 0.001````

Dense synthetic distributed NMF. Remember you have to compile dense nmf and sparse nmf differently. 
Look for build instructions above. 
````mpirun -np 96 ./distnmf -a 0 -m 207360 -n 138240 -k 200 --pr=8 --pc=12 -i rand_lowrank -t 10````

Sparse real world distributed NMF on 864 processors
````mpirun -np 864 ./distnmf -a 3  -k 50 --pr=36 --pc=24 -i /lustre/atlas/proj-shared/csc209/ramki/sparserw/nbpp/864/A -t 30````

Dense real world distributed NMF. In this case, we are running a distributed NMF built for dense matrices.
Also the input matrix is double 1D distributed which is differented from 2D input matrix. Refer the paper 
for details. The preprocessing scripts can be found under utilities/. 
````mpirun -np 1536 ./distnmf -a 3 -m 1299456 -n 3456 -k 50 --pr=1536 --pc=1 -i /lustre/atlas/proj-shared/csc209/ramki/videorw/nbpp/1536cores/A -t 30````

Output interpretation
======================
For W matrix row major ordering. That is., W_0, W_1, .., W_p
For H matrix column major ordering. That is., for 6 processes
with pr=3, pc=2, interpret as H_0, H_2, H_4, H_1, H_3, H_5

Citation:
=========

If you are using this MPI implementation, kindly cite.

Ramakrishnan Kannan, Grey Ballard, and Haesun Park. 2016. A high-performance parallel algorithm for nonnegative matrix factorization. In Proceedings of the 21st ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP '16). ACM, New York, NY, USA, , Article 9 , 11 pages. DOI: http://dx.doi.org/10.1145/2851141.2851152
