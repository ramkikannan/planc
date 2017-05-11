/* Copyright 2017 Ramakrishnan Kannan */

#ifndef NTF_UPDATEALGOS_HPP_
#define NTF_UPDATEALGOS_HPP_

#include "ntf_utils.h"
#include "nnls.hpp"

FMAT update(const ntfalgo i_algo, const FMAT &gram, const FMAT &rhs) {
    FMAT rc;
    switch (i_algo) {
    case NTF_BPP:
        rc = updateBPP(gram, rhs);
        break;
    default:
        cout << "wrong algo type" << endl;
    }
}

FMAT updateBPP(const FMAT &gram, const FMAT &rhs) {
    MAT nnlsgram;
    MAT nnlsrhs;
    nnlsgram = arma::conv_to<MAT>::from(gram);
    nnlsrhs = arma::conv_to<MAT>::from(nnlsrhs);
    BPPNNLS<MAT, VEC > subProblem(nnlsgram, nnlsrhs, true);
    subProblem.solveNNLS();
    FMAT rc = arma::conv_to<FMAT >::from(subProblem.getSolutionMatrix());
    return rc;
}

#endif  // NTF_UPDATEALGOS_HPP_ 