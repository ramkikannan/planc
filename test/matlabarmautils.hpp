#ifndef MATLAB_ARMA_UTILS_HPP_
#define MATLAB_ARMA_UTILS_HPP_

#include <armadillo>

// defines for namespace confusion
#define FMAT    arma::fmat
#define MAT     arma::mat
#define FROWVEC arma::frowvec
#define FVEC    arma::fvec
#define SP_FMAT arma::sp_fmat
#define UVEC    arma::uvec
#define UWORD   arma::uword
#define VEC     arma::vec



template <class ARMAMATTYPE> void matlab2arma(size_t m_rows, size_t n_cols,
        double *in, ARMAMATTYPE *out) {
    for (int j = 0; j < n_cols; j++) {
        for (int i = 0; i < m_rows; i++) {
            (*out)(i, j) = in[j * m_rows + i];
        }
    }

}

template <class ARMAMATTYPE>  void arma2matlab(size_t m_rows, size_t n_cols,
        const ARMAMATTYPE& in, double *out) {
    for (int j = 0; j < n_cols; j++) {
        for (int i = 0; i < m_rows; i++) {
            out[j * m_rows + i] = in(i, j);
        }
    }

}


#endif  // MATLAB_ARMA_UTILS_HPP_