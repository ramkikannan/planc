/* Copyright 2016 Ramakrishnan Kannan */
// utility functions
#ifndef COMMON_UTILS_H_
#define COMMON_UTILS_H_

// #ifndef _VERBOSE
// #define _VERBOSE 1
// #endif

enum algotype { MU, HALS, ANLSBPP, NAIVEANLSBPP, AOADMM, NESTEROV, CPALS };

enum normtype { NONE, L2NORM, MAXNORM };

// #if !defined(ARMA_64BIT_WORD)
// #define ARMA_64BIT_WORD
#define ARMA_DONT_USE_WRAPPER
#define ARMA_USE_BLAS
#define ARMA_USE_LAPACK
// #endif
#include <armadillo>
#include <cmath>
#include <iostream>
#include <vector>

// using namespace std;

#ifndef ERR
#define ERR std::cerr
#endif

#ifndef WARN
#define WARN std::cerr
#endif

#ifndef INFO
#define INFO std::cout
#endif

#ifndef OUTPUT
#define OUTPUT std::cout
#endif

#define EPSILON_1EMINUS16 0.00000000000000001
#define EPSILON 0.000001
#define EPSILON_1EMINUS12 1e-12
#define NUMBEROF_DECIMAL_PLACES 12
#define RAND_SEED 100
#define RAND_SEED_SPARSE 100

// defines for namespace confusion
#define FMAT arma::fmat
#define MAT arma::mat
#define FROWVEC arma::frowvec
#define ROWVEC arma::rowvec
#define FVEC arma::fvec
#define SP_FMAT arma::sp_fmat
#define SP_MAT arma::sp_mat
#define UVEC arma::uvec
#define IVEC arma::ivec
#define UWORD arma::uword
#define VEC arma::vec

#define PRINTMATINFO(A) "::" #A "::" << (A).n_rows << "x" << (A).n_cols

#define PRINTMAT(A) PRINTMATINFO((A)) << endl << (A)

typedef std::vector<int> STDVEC;
typedef unsigned int UINT;
typedef unsigned long ULONG;

void absmat(const FMAT *X);

inline void tic();
inline double toc();

int random_sieve(const int);

template <typename FVT>
inline void fillVector(const FVT value, std::vector<FVT> *a) {
  for (unsigned int ii = 0; ii < a->size(); ii++) {
    (*a)[ii] = value;
  }
}

#endif  // COMMON_UTILS_H_
