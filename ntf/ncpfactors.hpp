/* Copyright 2017 Ramakrishnan Kannan */
#ifndef NTF_CPFACTORS_HPP_
#define NTF_CPFACTORS_HPP_

#include <cassert>
#include "utils.h"
#include "tensor.hpp"

// ncp_factors contains the factors of the ncp
// every ith factor is of size n_i * k
// number of factors is called as order of the tensor
// all idxs are zero idx.

namespace PLANC {

class NCPFactors {
    FMAT *ncp_factors;
    int m_order;
    int m_k;
    UVEC m_dimensions;
  public:

    //constructors
    NCPFactors(const UVEC i_dimensions, const int i_k)
        : m_dimensions(i_dimensions) {
        ncp_factors = new FMAT[4];
        this->m_order = i_dimensions.n_rows;
        this->m_k = i_k;
        for (int i = 0; i < this->m_order; i++) {
            ncp_factors[i] = arma::randu<FMAT>(i_dimensions[i], this->m_k);
        }
    }
    // getters
    int rank() const {return m_k;}
    UVEC dimensions() const {return m_dimensions;}
    FMAT factor(const int i_n) const {return ncp_factors[i_n];}

    // setters
    void set(const int i_n, const FMAT &i_factor) {
        assert(i_factor.size() == this->ncp_factors[i_n].size());
        this->ncp_factors[i_n] = i_factor;
    }

    //computations
    void gram(const int i_n, FMAT *o_UtU) {
        (*o_UtU) = ncp_factors[i_n] * trans(ncp_factors[i_n]);
    }
    // find the hadamard product of all the factor grams
    // except the n. This is equation 50 of the JGO paper.
    void gram_leave_out_one(const int i_n, FMAT *o_UtU) {
        FMAT currentGram(this->m_k, this->m_k);
        (*o_UtU) = arma::ones<FMAT>(this->m_k, this->m_k);
        for (int i = 0; i < this->m_order && i != i_n; i++) {
            currentGram = ncp_factors[i] * trans(ncp_factors[i]);
            (*o_UtU) = (*o_UtU) % currentGram;
        }
    }

    FMAT krp_leave_out_one(const int i_n){
        UWORD krpsize = arma::prod(this->m_dimensions);
        krpsize /= this->m_dimensions[i_n];
        FMAT krp(krpsize,this->m_k);
        krp_leave_out_one(i_n, &krp);
        return krp;
    }
    // construct low rank tensor using the factors

    // khatrirao leaving out one. we are using the implementation
    // from tensor toolbox. Always krp for mttkrp is computed in
    // reverse. Hence assuming the same. The order of the computation
    // is same a tensor tool box.
    // size of krp must be product of all dimensions leaving out nxk
    void krp_leave_out_one(const int i_n, FMAT *o_krp) {
        // matorder = length(A):-1:1;
        // Always krp for mttkrp is computed in
        // reverse. Hence assuming the same.
        UVEC matorder = arma::zeros<UVEC>(this->m_order - 1);
        int current_ncols = this->m_k;
        int j = 0;
        for (int i = this->m_order - 1; i >= 0; i--) {
            if (i != i_n ) {
                matorder(j++) = i;
            }
        }
#ifdef NTF_VERBOSE
        INFO << "::" << __PRETTY_FUNCTION__
             << "::" << __LINE__
             << "::matorder::" << matorder << endl;
#endif
        (*o_krp).zeros();
        // N = ncols(1);
        // This is our k. So keep N = k in our case.
        // P = A{matorder(1)};
        // take the first factor of matorder
        /*UWORD current_nrows = ncp_factors[matorder(0)].n_rows - 1;
        (*o_krp).rows(0, current_nrows) = ncp_factors[matorder(0)];
        // this is factor by factor
        for (int i = 1; i < this->m_order - 1; i++) {
            // remember always krp in reverse order.
            // That is if A krp B krp C, we compute as
            // C krp B krp A.
            // prev_nrows = current_nrows;
            // rightkrp.n_rows;
            // we are populating column by column
            FMAT& rightkrp = ncp_factors[matorder[i]];
            for (int j = 0; j < this->m_k; j++) {
                FVEC krpcol = (*o_krp)(arma::span(0, current_nrows), j);
                // krpcol.each_rows*rightkrp.col(i);
                for (int k = 0; k < rightkrp.n_rows; k++) {
                    (*o_krp)(arma::span(k * krpcol.n_rows, (k + 1)*krpcol.n_rows - 1), j) = krpcol * rightkrp(k, j);
                }
            }
            current_nrows *= rightkrp.n_rows;
        }*/
// Loop through all the columns
// for n = 1:N
//     % Loop through all the matrices
//     ab = A{matorder(1)}(:,n);
//     for i = matorder(2:end)
//        % Compute outer product of nth columns
//        ab = A{i}(:,n) * ab(:).';
//     end
//     % Fill nth column of P with reshaped result
//     P(:,n) = ab(:);
// end
        for (int n = 0; n < this->m_k; n++) {
            FMAT ab = ncp_factors[matorder[0]].col(n);
            for (int i = 1; i < this->m_order - 1; i++) {
                FVEC abvec = arma::vectorise(ab);
                ab = abvec * trans(ncp_factors[matorder[i]].col(n));
            }
            (*o_krp).col(n) = arma::vectorise(ab);
        }
    }
// caller must free
    Tensor rankk_tensor() {
        UWORD krpsize = arma::prod(this->m_dimensions);
        krpsize /= this->m_dimensions[0];
        FMAT krpleavingzero = arma::zeros<FMAT>(krpsize, this->m_k);
        krp_leave_out_one(0, &krpleavingzero);
        FMAT lowranktensor(this->m_dimensions[0], krpsize);
        lowranktensor = this->ncp_factors[0] * trans(krpleavingzero);
        Tensor rc(this->m_dimensions, lowranktensor.memptr());
        return rc;
    }

    void print() {
        cout << "order::" << this->m_order << "::k::" << this->m_k;
        cout << "::dims::"  << endl << this->m_dimensions << endl;
        for (int i = 0; i < this->m_order; i++) {
            cout << i << "th factor" << endl << "=============" << endl;
            cout << this->ncp_factors[i];
        }
    }
    void print(const int i_n) {
        cout << i_n << "th factor" << endl << "=============" << endl;
        cout << this->ncp_factors[i_n];
    }
};
}  // PLANC

#endif  //  NTF_CPFACTORS_HPP_