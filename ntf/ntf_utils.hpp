/* Copyright 2017 Ramakrishnan Kannan */

#ifndef NTF_NTF_UTILS_HPP
#define NTF_NTF_UTILS_HPP

#include "ncpfactors.hpp"
#include "tensor.hpp"

// A is of size m x k
// B is of size n x k
// Returns C of size mn x k
void khatrirao(const MAT &i_A, const MAT &i_B, MAT *o_C) {
    assert(i_A.n_cols == i_B.n_cols);
    FVEC acol = arma::zeros<FVEC>(i_A.n_rows);
    FVEC bcol = arma::zeros<FVEC>(i_B.n_rows);
    for (int i = 0; i < i_A.n_cols; i++) {
        acol = i_A.col(i);
        bcol = i_B.col(i);
        for (int j = 0; j < acol.n_rows; j++) {
            (*o_C)(arma::span(j * bcol.n_rows, (j + 1)*bcol.n_rows - 1), i) = acol(j) * bcol;
        }

    }
}
inline void kronecker(const FVEC &i_acol, const FVEC &i_bcol, FVEC *o_c) {
    for (int j = 0; j < i_acol.n_rows; j++) {
        (*o_c).rows(arma::span(j * i_bcol.n_rows, (j + 1)*i_bcol.n_rows - 1)) = i_acol(j) * i_bcol;
    }
}

void mttkrp(const int i_n, PLANC::Tensor& X,
            PLANC::NCPFactors& i_F, MAT *o_mttkrp) {
    MAT krp = i_F.krp_leave_out_one(i_n);    
    X.mttkrp(i_n, krp, o_mttkrp);
}

#endif  // NTF_NTF_UTILS_HPP