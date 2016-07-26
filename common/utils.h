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

using namespace std;
using namespace arma;

#ifndef ERR
#define ERR cerr
#endif

#ifndef WARN
#define WARN cerr
#endif

#ifndef INFO
#define INFO cout
#endif

#ifndef OUTPUT
#define OUTPUT cout
#endif

#define EPSILON_1EMINUS16 0.00000000000000001
#define EPSILON 0.000001
#define NUMBEROF_DECIMAL_PLACES 12
#define RAND_SEED 100
#define RAND_SEED_SPARSE 100

//#define PRINTMATINFO(A) \
"::"#A"::" << (A).n_rows << "x" << (A).n_cols << "::norm::" << norm((A),"fro")

#define PRINTMATINFO(A) "::"#A"::" << (A).n_rows << "x" << (A).n_cols

#define PRINTMAT(A) PRINTMATINFO((A)) << endl << (A)

typedef std::vector<int> STDVEC;
typedef unsigned int UINT;
typedef unsigned long ULONG;

void absmat(const fmat *X);

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

struct option nmfopts[] = {
  {"input",   optional_argument, 0, 'i'},
  {"algo",  optional_argument, 0, 'a'},
  {"lowrank",     optional_argument, 0, 'k'},
  {"iter",   optional_argument,       0, 't'},
  {"rows",   optional_argument,       0, 'm'},
  {"columns",   optional_argument,       0, 'n'},
  {"winit",   optional_argument,       0, WINITFLAG},
  {"hinit",   optional_argument,       0, HINITFLAG},
  {"wout",   optional_argument,       0, 'w'},
  {"hout",   optional_argument,       0, 'h'},
  {0,         0,                 0,  0 }
};

template<typename FVT>
inline void fillVector(const FVT value, vector<FVT> *a) {
  for (int ii = 0; ii < a->size(); ii++) {
    (*a)[ii] = value;
  }
}

#endif /* COMMON_UTILS_H_ */
