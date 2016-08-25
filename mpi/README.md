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
export NMFLIB_DIR=/ccs/home/ramki/rhea/research/nmflib/
export INCLUDE=$INCLUDE:$ARMADILLO_INCLUDE_DIR:$NMFLIB_DIR
export CPATH=$CPATH:$INCLUDE:
export MKLROOT=/ccs/compilers/intel/rh6-x86_64/16.0.0/mkl/

If you have got MKL, please source MKLVARS.sh before running make/cmake

Sparse NMF
---------
cd build and run cmake ../ -DCMAKE_BUILD_SPARSE

Sparse Debug build
------------------
cd build and run cmake ../ -DCMAKE_BUILD_SPARSE -DCMAKE_BUILD_TYPE=Debug

Once cmake is ready without any errors, run make. 

CMAKE Macros

For sparse NMF - cmake -DBUILD_SPARSE=1 - Default dense build
For timing with barrier after mpi calls - cmake -DCMAKE_WITH_BARRIER_TIMING - Default with barrier timing
For performance, disable the WITH__BARRIER__TIMING. Run as "cmake -DCMAKE_WITH_BARRIER_TIMING:BOOL=OFF"

Code level macros - Define in distutils.h

MPI_VERBOSE - Be doubly sure about what you do. It prints all intermediary matrices.
			  Try this only for very very small matrix that is of size less than 10.
WRITE_RAND_INPUT - for dumping the generated random matrix

Output interpretation
======================
For W matrix row major ordering. That is., W_0, W_1, .., W_p
For H matrix column major ordering. That is., for 6 processes
with pr=3, pc=2, interpret as H_0, H_2, H_4, H_1, H_3, H_5
