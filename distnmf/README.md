# MPI Based DistNMF and DistNTF 

## Install Instructions

This program depends on:

- Download Armadillo library which can be found at https://arma.sourceforge.net
- Download and build OpenBLAS https://github.com/xianyi/OpenBLAS

Once the above steps are completed, set the following environment variables.

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

### Sparse NMF

Run cmake with -DCMAKE_BUILD_SPARSE

### Sparse Debug build

Run cmake with -DCMAKE_BUILD_SPARSE -DCMAKE_BUILD_TYPE=Debug

### Building on Cray-EOS/Titan

CC=CC CXX=CC cmake ~/nmflibrary/distnmf/ -DCMAKE_IGNORE_MKL=1

### Building on Titan with NVBLAS

We are using NVBLAS to offload computations to GPU.
By default we enable building with cuda in Titan.
The sample configurations files for nvblas can be found at conf/nvblas_cuda75.conf
and conf/nvblas_cuda91.conf for CUDA Toolkit 7.5 and 9.1 respectively.

CC=CC CXX=CC cmake ~/nmflibrary/distnmf/ -DCMAKE_IGNORE_MKL=1 -DCMAKE_BUILD_CUDA=1

- Module swap PrgEnv-pgi PrgEnv-gnu
- Module load cmake3
- Module load cudatoolkit
- mkdir /lustre/atlas/world-shared/csc209/ramki/titan/planc-build
- CC=CC CXX=CC cmake ~/planc-github/ -DCMAKE_INSTALL_PREFIX=/lustre/atlas/world-shared/csc209/ramki/titan/planc/
- Make
- Make install
- Cd /lustre/atlas/world-shared/csc209/ramki/titan/planc/bin
- Copied planc-github/conf/nvblas91.conf nvblas.conf
- ldd dense_disntf to get the libsci_gnu_mp_61.so.5 path and set the NVBLAS_CPU_BLAS_LIB
- Run the experiment

### Other Macros

* CMAKE macros

  For sparse NMF - cmake -DBUILD_SPARSE=1 - Default dense build
  For timing with barrier after mpi calls - cmake -DCMAKE_WITH_BARRIER_TIMING - Default with barrier timing
  For performance, disable the WITH__BARRIER__TIMING. Run as "cmake -DCMAKE_WITH_BARRIER_TIMING:BOOL=OFF"
  For building cuda - -DCMAKE_BUILD_CUDA=1 - Default is off.

* Code level macros - Defined in distutils.h

  MPI_VERBOSE - Be doubly sure about what you do. It prints all intermediary matrices.
			   Try this only for very very small matrix that is of size less than 10.
  WRITE_RAND_INPUT - for dumping the generated random matrix

## Output interpretation

For W matrix row major ordering. That is., W_0, W_1, .., W_p
For H matrix column major ordering. That is., for 6 processes
with pr=3, pc=2, interpret as H_0, H_2, H_4, H_1, H_3, H_5

## Running

mpirun -np 16 ./distnmf -a [0/1/2/3] -i rand_[lowrank/uniform] -d "rows cols" -p "pr pc" -r "W_l2 W_l1 H_l2 H_l0" -k 20 -t 20 -e 1

## Citation:


If you are using this MPI implementation, kindly cite.

Ramakrishnan Kannan, Grey Ballard, and Haesun Park. 2016. A high-performance parallel algorithm for nonnegative matrix factorization. In Proceedings of the 21st ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP '16). ACM, New York, NY, USA, , Article 9 , 11 pages. DOI: http://dx.doi.org/10.1145/2851141.2851152
