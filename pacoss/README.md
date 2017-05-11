# Pacoss: Partitioning and Communication Framework for Sparse Structures #

## What is Pacoss?
Pacoss is a C++ library that provides data partitioning, distribution, and communication tools to easily develop scalable high performance parallel sparse irregular linear (multi)linear algebra applications using MPI in the back-end.
It greatly simplifies the code for developing such parallel without any compromise in terms of performance.

By using Pacoss, one can

* easily solve a sparse problem in parallel using a point-to-point communication scheme,
* try alternative communication schemes, such as embedding point-to-point communications in an all-to-all framework to reduce communication latency,
* minimize the risk of introducing bugs in the communication step,
* and effectively partition the problem data to establish load balance and minimize communication volume for better scalability.

## What type of applications can be developed using Pacoss?
Including, but not limited to, parallel

* sparse matrix and tensor factorization
* sparse matrix-vector multiplication (hence all sorts of iterative solvers)
* optimization problems
* graph and hypergraph algorithms

etc.

## How do I get Pacoss to work? ###

#### Requirements/Dependencies
Pacoss is implemented in C++11; therefore, an MPI compiler with full C++11 support is required.

Pacoss uses a lightweight MPI wrapper library called TMPI for 32/64 bit integer compatibility with MPI libraries, templated MPI routines, and gracefully disabling all MPI dependencies of the code with a single macro definition.

Pacoss Partititioner also has an optional dependency to PaToH hypergraph partitioning library for better partitioning.

#### Installation
1. Clone the Pacoss repository.
1. Clone the TMPI repository.
1. Provide a Makefile.inc file (see Makefile.inc.tmp for an example) which specifies the following variables:
  1. **BUILD**: Set to either RELEASE or DEBUG.
  1. **CPP**: MPI C++11 compiler command
  1. **TMPI_SRC_FOLDER**: Folder including TMPI source files
  1. **PATOH_FOLDER** *(Optional)*: PaToH root folder
1. Type **make**. This builds the **pacoss** executable under the **exe/** directory, and **libpacoss** library under the **lib/** directory.

#### Pacoss Executable
Type `./exe/pacoss` to display the complete list of usage options. Pacoss executable involve three programs:

1. `pacoss partition` partitions a given sparse struct.
    * Example: `pacoss partition -f sparse-struct.txt -p 128 -m fine-random -n balance-comm` partitions the sparse struct `sparse-struct.txt` using fine-grain random partitioning for sparse struct nonzeros, and communication balancing scheme for partitioning rows at each dimension. 
1. `pacoss distribute` creates individual partitioned sparse struct files as well as mode owner mapping for each process, using a sparse struct and associated partition files.
    * Example: `pacoss distribute -f sparse-struct.txt` distributes the sparse struct `sparse-struct.txt` using the previously created partition partition files `sparse-struct.txt.nzpart` and `sparse-struct.txt.dpart`.
    You can alternatively specify the nonzero and dimension partition file names using `-z` and `-d` options.
1. `pacoss check-stats` displays the partition statistics (such as communication costs, load balance, number of messages) for a given sparse struct and associated partition files.
    * Example: `pacoss check-stats -f sparse-struct.txt -z sparse-struct.txt.nzpart -d sparse-struct.txt.dpart` computes and prints the partition statistics for the given sparse struct and partition files.
1. `pacoss convert-sparse-struct` converts any N-dimensional sparse tensor in coordinate format to Pacoss sparse struct.
    * Example: `pacoss convert-sparse-struct -f tensor-coor.txt --one-based-idx` creates the sparse struct file `tensor-coor.txt.ss` that involves dimensionality of the sparse struct and its dimension sizes.
    Use the option `--one-based-idx` if nonzero coordinates in `tensor-coor.txt` are one-based.

#### Pacoss Partitioner API

#### Pacoss Communicator API

## Contribution

Please contact the main developer (oguz.kaya@outlook.com) if you are planning to contribute to Pacoss.
