#ifndef NTF_AUNTF_HPP
#define NTF_AUNTF_HPP

#include <armadillo>
#include "luc.hpp"
#include "ntf_utils.h"
#include "tensor.hpp"

namespace PLANC {

#define TENSOR_DIM (m_input_tensor.dimensions())
#define TENSOR_NUMEL (m_input_tensor.numel())


class AUNTF {
  private:
    const Tensor &m_input_tensor;
    NCPFactors m_ncp_factors;
    int m_num_it;
    MAT gram_without_one;
    const int m_low_rank_k;
    MAT *ncp_krp;
    MAT *ncp_mttkrp;
    const ntfalgo m_updalgo;
    LUC *m_luc;
    Tensor *lowranktensor;

  public:
    AUNTF(const Tensor &i_tensor, const int i_k, ntfalgo i_algo) :
        m_ncp_factors(i_tensor.dimensions(), i_k),
        m_input_tensor(i_tensor),
        m_low_rank_k(i_k),
        m_updalgo(i_algo) {
        gram_without_one = arma::zeros<MAT>(i_k, i_k);
        ncp_mttkrp = new MAT[i_tensor.order()];
        ncp_krp = new MAT[i_tensor.order()];
        for (int i = 0; i < i_tensor.order(); i++) {
            UWORD current_size = TENSOR_NUMEL / TENSOR_DIM[i];
            ncp_krp[i] = arma::zeros <MAT>(current_size, i_k);
            ncp_mttkrp[i] = arma::zeros<MAT>(TENSOR_DIM[i], i_k);
        }
        lowranktensor = new Tensor(i_tensor.dimensions());
        m_luc = new LUC(m_updalgo);
        m_num_it = 20;
    }
    NCPFactors ncp_factors() const {return m_ncp_factors;}
    void num_it(const int i_n) { this->m_num_it = i_n;}
    void computeNTF() {
        for (int i = 0; i < m_num_it; i++) {
            for (int j = 0; j < this->m_input_tensor.order(); j++) {
                m_ncp_factors.gram_leave_out_one(j, &gram_without_one);
                m_ncp_factors.krp_leave_out_one(j, &ncp_krp[j]);
                m_input_tensor.mttkrp(j, ncp_krp[j], &ncp_mttkrp[j]);
                MAT factor = m_luc->update(gram_without_one,
                                            ncp_mttkrp[j].t());
                m_ncp_factors.set(j, factor.t());
            }
            std::cout << "error at it::" << i << "::"
                      << computeObjectiveError() << std::endl;
        }
    }
    double computeObjectiveError() {
        // current low rank tensor
        // UWORD krpsize = arma::prod(this->m_dimensions);
        // krpsize /= this->m_dimensions[0];
        // MAT krpleavingzero = arma::zeros<MAT>(krpsize, this->m_k);
        // krp_leave_out_one(0, &krpleavingzero);
        // MAT lowranktensor(this->m_dimensions[0], krpsize);
        // lowranktensor = this->ncp_factors[0] * trans(krpleavingzero);

        // compute current low rank tensor as above.
        m_ncp_factors.krp_leave_out_one(0, &ncp_krp[0]);
        // gemm of krpleavingzero and zeroth factor
        // cblas_sgemm (const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa,
        // const CBLAS_TRANSPOSE transb, const MKL_INT m, const MKL_INT n,
        // const MKL_INT k, const float alpha, const float * a,
        // const MKL_INT lda, const float * b, const MKL_INT ldb,
        // const float beta, float * c, const MKL_INT ldc);
        // sgemm (TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
        char transa = 'T';
        char transb = 'N';
        int m = m_ncp_factors.factor(0).n_rows;
        int n = ncp_krp[0].n_cols;
        int k = m_ncp_factors.factor(0).n_cols;
        int lda = m_ncp_factors.factor(0).n_rows;
        int ldb = m_ncp_factors.factor(0).n_cols;
        int ldc = m_ncp_factors.factor(0).n_rows;
        float alpha = 1;
        float beta = 1;
        sgemm_(&transa, &transb, &m, &n, &k, &alpha, m_ncp_factors.factor(0).memptr(),
               &lda, ncp_krp[0].memptr() , &ldb, &beta, lowranktensor->data() , &ldc);
        double err = m_input_tensor.err(*lowranktensor);
        return err;
    }
};  // class AUNTF
}  // namespace PLANC
#endif  // NTF_AUNTF_HPP