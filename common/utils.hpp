/* Copyright 2016 Ramakrishnan Kannan */
// utility functions
#ifndef COMMON_UTILS_HPP_
#define COMMON_UTILS_HPP_
#include <stdio.h>
#include <assert.h>
#include <omp.h>
#include <typeinfo>
#include <stack>
#include <ctime>
#include "utils.h"
#ifdef MKL_FOUND
#include <mkl.h>
#else
#include <lapacke.h>
#include <cblas.h>
#endif

static ULONG powersof10[16] = {
    1, 10, 100, 1000,
    10000, 100000, 1000000, 10000000,
    100000000, 1000000000, 10000000000, 100000000000,
    1000000000000, 10000000000000, 100000000000000, 1000000000000000
};

static std::stack<clock_t> tictoc_stack;

inline void tic() {
    tictoc_stack.push(clock());
}

inline double toc() {
    double rc = (static_cast<double>
                 (clock() - tictoc_stack.top())) / CLOCKS_PER_SEC;
    tictoc_stack.pop();
    return rc;
}

template <class T> void fixNumericalError(T *X) {
    for (UINT i = 0; i < X->n_rows; i++) {
        for (UINT j = 0; j < X->n_cols; j++) {
            if ((*X)(i, j) < EPSILON) {
                (*X)(i, j) = EPSILON;
            }
        }
    }
}

template <class T> void fixDecimalPlaces(T *X) {
    for (UINT i = 0; i < X.n_rows; i++) {
        for (UINT j = 0; j < X.n_cols; j++) {
            (*X)(i, j) = floorf(X(i, j) * powersof10[NUMBEROF_DECIMAL_PLACES])
                         / powersof10[NUMBEROF_DECIMAL_PLACES];
        }
    }
}


/*
* Returns the nth prime number.
* There are totally 10000 prime numbers within 104000;
*/
int random_sieve(const int nthprime) {
    int i, m, k;
    int klimit, nlimit;
    int *mark;

    nlimit = 104000;

    mark = reinterpret_cast<int *>(calloc(nlimit, sizeof(int)));

    /* Calculate limit for k */
    klimit = static_cast<int>(sqrt((double)nlimit) + 1);

    /* Mark the composites */
    /* Special case */
    mark[1] = -1;

    /* Set k=1. Loop until k >= sqrt(n) */
    for (k = 3; k <= klimit; k = m) {
        /* Find first non-composite in list > k */
        for (m = k + 1; m < nlimit; m++)
            if (!mark[m])
                break;

        /* Mark the numbers 2m, 3m, 4m, ... */
        for (i = m * 2; i < nlimit; i += m)
            mark[i] = -1;
    }

    /* Now display results - all unmarked numbers are prime */
    int rcprime = -1;
    for (k = 0, i = 1; i < nlimit; i++) {
        if (!mark[i]) {
            k++;
            if (k == nthprime + 1) {
                rcprime = i;
                break;
            }
        }
    }
    free(mark);
    return rcprime;
}

template <class T> void absmat(T *X) {
    UVEC negativeIdx = find((*X) < 0);
    (*X)(negativeIdx) = (*X)(negativeIdx) * -1;
}

template <class T> void makeSparse(const double sparsity, T (*X)) {
    // make a matrix sparse
    srand(RAND_SEED_SPARSE);
    #pragma omp parallel for
    for (int j = 0; j < X->n_cols; j++) {
        for (int i = 0; i < X->n_rows; i++) {
            if (arma::randu() > sparsity) (*X)(i, j) = 0;
        }
    }
}

template <class T> void randNMF(const UWORD m, const UWORD n,
                                const UWORD k, const double sparsity, T *A) {
#ifdef BUILD_SPARSE
    T temp = arma::sprandu<T >(m, n, sparsity);
    A = &temp;
#else
    srand(RAND_SEED);
    FMAT W = 10 * arma::randu<FMAT >(m, k);
    FMAT H = 10 * arma::randu<FMAT >(n, k);
    if (sparsity < 1) {
        makeSparse<FMAT>(sparsity, &W);
        makeSparse<FMAT>(sparsity, &H);
    }
    T temp = ceil(W * trans(H));
    A = &temp;
#endif
}

template <class T>
void printVector(const std::vector<T> &x) {
    for (int i = 0; i < x.size(); i++) {
        INFO << x[i] << ' ';
    }
    INFO << std::endl;
}

/*
* can be called by external people for sparse input matrix.
*/
template <class INPUTTYPE, class LRTYPE>
double computeObjectiveError(const INPUTTYPE &A,
                             const LRTYPE &W, const LRTYPE &H) {
    // 1. over all nnz (a_ij - w_i h_j)^2
    // 2. over all nnz (w_i h_j)^2
    // 3. Compute R of W ahd L of H through QR
    // 4. use sgemm to compute RL
    // 5. use slange to compute ||RL||_F^2
    // 6. return nnzsse+nnzwh-||RL||_F^2
    UWORD k = W.n_cols;
    UWORD m = A.n_rows;
    UWORD n = A.n_cols;
    tic();
    float nnzsse = 0;
    float nnzwh = 0;
    LRTYPE Rw(k, k);
    LRTYPE Rh(k, k);
    LRTYPE Qw(m, k);
    LRTYPE Qh(n, k);
    LRTYPE RwRh(k, k);
    #pragma omp parallel for reduction (+ : nnzsse, nnzwh)
    for (UWORD jj = 1; jj <= A.n_cols; jj++) {
        UWORD startIdx = A.col_ptrs[jj - 1];
        UWORD endIdx = A.col_ptrs[jj];
        UWORD col = jj - 1;
        float nnzssecol = 0;
        float nnzwhcol = 0;
        for (UWORD ii = startIdx; ii < endIdx; ii++) {
            UWORD row = A.row_indices[ii];
            float tempsum = 0;
            for (UWORD kk = 0; kk < k; kk++) {
                tempsum += (W(row, kk) * H(col, kk));
            }
            nnzwhcol += tempsum * tempsum;
            nnzssecol += (A.values[ii] - tempsum) * (A.values[ii] - tempsum);
        }
        nnzsse += nnzssecol;
        nnzwh += nnzwhcol;
    }
    qr_econ(Qw, Rw, W);
    qr_econ(Qh, Rh, H);
    RwRh = Rw * Rh.t();
    double normWH = arma::norm(RwRh, "fro");
    Rw.clear();
    Rh.clear();
    Qw.clear();
    Qh.clear();
    RwRh.clear();
    INFO << "error compute time " << toc() << std::endl;
    double fastErr = sqrt(nnzsse + (normWH * normWH - nnzwh));
    return (fastErr);
}

#if defined(MKL_FOUND) && defined(BUILD_SPARSE)
/*
* mklMat is csc representation
* Bt is the row major order of the arma B matrix
* Ct is the row major order of the arma C matrix
* Once you receive Ct, transpose again to print
* C using arma
*/
void ARMAMKLSCSCMM(const SP_FMAT &mklMat, char transa, const FMAT &Bt,
                   float *Ct) {
    MKL_INT m, k, n, nnz;
    m = static_cast<MKL_INT>(mklMat.n_rows);
    k = static_cast<MKL_INT>(mklMat.n_cols);
    n = static_cast<MKL_INT>(Bt.n_rows);
    // FMAT B = B.t();
    // C = alpha * A * B + beta * C;
    // mkl_?cscmm - https://software.MKL_INTel.com/en-us/node/468598
    // char transa = 'N';
    float alpha = 1.0;
    float beta = 0.0;
    char* matdescra = "GUNC";
    MKL_INT ldb = n;
    MKL_INT ldc = n;
    MKL_INT* pntrb = (MKL_INT *)(mklMat.col_ptrs);
    MKL_INT* pntre = pntrb + 1;
    mkl_scscmm(&transa, &m, &n, &k, &alpha, matdescra,
               mklMat.values, (MKL_INT *)mklMat.row_indices,
               pntrb, pntre,
               (float *)(Bt.memptr()), &ldb,
               &beta, Ct, &ldc);
}
#endif

/*
* This is an sgemm wrapper for armadillo matrices
* Something is going crazy with armadillo
*/

void cblas_sgemm(const FMAT &A, const FMAT &B, float *C) {
    UWORD m = A.n_rows;
    UWORD n = B.n_cols;
    UWORD k = A.n_cols;
    double alpha = 1.0;
    double beta = 0.0;
    cblas_sgemm(CblasColMajor,
                CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, A.memptr(),
                m, B.memptr(), k, beta, C, m);
}

#endif  // COMMON_UTILS_HPP_
