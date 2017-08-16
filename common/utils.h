/* Copyright 2016 Ramakrishnan Kannan */
// utility functions
#ifndef COMMON_UTILS_H_
#define COMMON_UTILS_H_

// #ifndef _VERBOSE
// #define _VERBOSE 1
// #endif


enum algotype {MU_NMF, HALS_NMF, BPP_NMF};

// #if !defined(ARMA_64BIT_WORD)
#define ARMA_64BIT_WORD
#define ARMA_DONT_USE_WRAPPER
#define ARMA_USE_BLAS
#define ARMA_USE_LAPACK
// #endif
#include <getopt.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <armadillo>

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
#define NUMBEROF_DECIMAL_PLACES 12
#define RAND_SEED 100
#define RAND_SEED_SPARSE 100

// defines for namespace confusion
#define FMAT    arma::fmat
#define MAT     arma::mat
#define FROWVEC arma::frowvec
#define ROWVEC arma::rowvec
#define FVEC    arma::fvec
#define SP_FMAT arma::sp_fmat
#define UVEC    arma::uvec
#define UWORD   arma::uword
#define VEC     arma::vec

//#define PRINTMATINFO(A) \
"::"#A"::" << (A).n_rows << "x" << (A).n_cols << "::norm::" << norm((A),"fro")

#define PRINTMATINFO(A) "::"#A"::" << (A).n_rows << "x" << (A).n_cols

#define PRINTMAT(A) PRINTMATINFO((A)) << endl << (A)

typedef std::vector<int> STDVEC;
typedef unsigned int UINT;
typedef unsigned long ULONG;

void absmat(const FMAT *X);

inline void tic();
inline double toc();

int random_sieve(const int);

// usage scenarios
// NMFLibrary algotype lowrank AfileName numIteration
// NMFLibrary algotype lowrank m n numIteration
// NMFLibrary algotype lowrank Afile WInitfile HInitfile numIteration
// NMFLibrary algotype lowrank Afile WInitfile HInitfile WoutputFile HoutputFile numIteration
#define WINITFLAG 1000
#define HINITFLAG 1001
#define REGWFLAG  1002
#define REGHFLAG  1003

struct option nmfopts[] = {
  {"input",       required_argument, 0, 'i'},
  {"algo",        required_argument, 0, 'a'},
  {"lowrank",     optional_argument, 0, 'k'},
  {"iter",        optional_argument, 0, 't'},
  {"rows",        optional_argument, 0, 'm'},
  {"columns",     optional_argument, 0, 'n'},
  {"winit",       optional_argument, 0, WINITFLAG},
  {"hinit",       optional_argument, 0, HINITFLAG},
  {"wout",        optional_argument, 0, 'w'},
  {"hout",        optional_argument, 0, 'h'},
  {"regw",        optional_argument, 0, REGWFLAG},
  {"regh",        optional_argument, 0, REGHFLAG},
  {0,         0,                 0,  0 }
};

template<typename FVT>
inline void fillVector(const FVT value, std::vector<FVT> *a) {
  for (int ii = 0; ii < a->size(); ii++) {
    (*a)[ii] = value;
  }
}

#endif  // COMMON_UTILS_H_
