/* Copyright Ramakrishnan Kannan 2017 */

#ifndef NTF_AUNTF_HPP_
#define NTF_AUNTF_HPP_

#include <armadillo>
#include <cblas.h>
#include "tensor.hpp"
#include "ncpfactors.hpp"
#include "luc.hpp"
#include "ntf_utils.h"
#include "dimtree/kobydt.hpp"

namespace planc {

#define TENSOR_DIM (m_input_tensor.dimensions())
#define TENSOR_NUMEL (m_input_tensor.numel())


#ifndef NTF_VERBOSE
#define NTF_VERBOSE 1
#endif

// extern "C" void cblas_dgemm_(const CBLAS_LAYOUT Layout,
//                              const CBLAS_TRANSPOSE transa,
//                              const CBLAS_TRANSPOSE transb,
//                              const int m, const int n,
//                              const int k, const double alpha,
//                              const double *a, const int lda,
//                              const double *b, const int ldb,
//                              const double beta, double *c,
//                              const int ldc);
class AUNTF {
  private:
    const planc::Tensor &m_input_tensor;
    planc::NCPFactors m_ncp_factors;
    int m_num_it;
    MAT gram_without_one;
    const int m_low_rank_k;
    MAT *ncp_krp;
    MAT *ncp_mttkrp_t;
    const algotype m_updalgo;
    LUC *m_luc;
    planc::Tensor *lowranktensor;
    KobyDimensionTree *kdt;
    bool m_enable_dim_tree;

  public:
    AUNTF(const planc::Tensor &i_tensor, const int i_k, algotype i_algo) :
        m_input_tensor(i_tensor),
        m_ncp_factors(i_tensor.dimensions(), i_k, false),
        m_low_rank_k(i_k),
        m_updalgo(i_algo) {
        m_ncp_factors.normalize();
        gram_without_one.zeros(i_k, i_k);
        ncp_mttkrp_t = new MAT[i_tensor.modes()];
        ncp_krp = new MAT[i_tensor.modes()];
        for (int i = 0; i < i_tensor.modes(); i++) {
            UWORD current_size = TENSOR_NUMEL / TENSOR_DIM[i];
            ncp_krp[i].zeros(current_size, i_k);
            ncp_mttkrp_t[i].zeros(i_k, TENSOR_DIM[i]);
        }
        lowranktensor = new planc::Tensor(i_tensor.dimensions());
        m_luc = new LUC();
        m_num_it = 20;
        INFO << "Init factors for NCP" << std::endl << "======================";
        m_ncp_factors.print();
        this->m_enable_dim_tree = false;
    }
    ~AUNTF() {
        for (int i = 0; i < m_input_tensor.modes(); i++) {
            ncp_krp[i].clear();
            ncp_mttkrp_t[i].clear();
        }
        delete[] ncp_krp;
        delete[] ncp_mttkrp_t;
        delete m_luc;
        if (this->m_enable_dim_tree) {
            delete kdt;
        }
        delete lowranktensor;
    }
    NCPFactors& ncp_factors() {return m_ncp_factors;}
    void dim_tree(bool i_dim_tree) {
        this->m_enable_dim_tree = i_dim_tree;
        if (i_dim_tree) {
            this->kdt = new KobyDimensionTree(m_input_tensor, m_ncp_factors,
                                              m_input_tensor.modes() / 2);
        }
    }
    void num_it(const int i_n) { this->m_num_it = i_n;}
    void computeNTF() {
        for (int i = 0; i < m_num_it; i++) {
            INFO << "iter::" << i << std::endl;
            for (int j = 0; j < this->m_input_tensor.modes(); j++) {
                m_ncp_factors.gram_leave_out_one(j, &gram_without_one);
#ifdef NTF_VERBOSE
                INFO << "gram_without_" << j << "::"
                     << arma::cond(gram_without_one) << std::endl
                     << gram_without_one << std::endl;
#endif
                m_ncp_factors.krp_leave_out_one(j, &ncp_krp[j]);
#ifdef NTF_VERBOSE
                INFO << "krp_leave_out_" << j << std::endl
                     << ncp_krp[j] << std::endl;
#endif
                if (this->m_enable_dim_tree) {
                    kdt->in_order_reuse_MTTKRP(j, ncp_mttkrp_t[j].memptr(), false);
                } else {
                    m_input_tensor.mttkrp(j, ncp_krp[j], &ncp_mttkrp_t[j]);
                }
#ifdef NTF_VERBOSE
                INFO << "mttkrp for factor" << j << std::endl
                     << ncp_mttkrp_t[j] << std::endl;
#endif
                MAT factor = m_luc->update(m_updalgo, gram_without_one,
                                           ncp_mttkrp_t[j]);
#ifdef NTF_VERBOSE
                INFO << "iter::" << i << "::factor:: " << j << std::endl
                     << factor << std::endl;
#endif
                m_ncp_factors.set(j, factor.t());
                if (j == 0) {
                    INFO << "error at it::" << i << "::"
                         << computeObjectiveError() << std::endl;
                }
                m_ncp_factors.normalize(j);
                factor = m_ncp_factors.factor(j).t();
                if (m_enable_dim_tree) {
                    kdt->set_factor(factor.memptr(), j);
                }
            }
#ifdef NTF_VERBOSE
            INFO << "ncp factors" << std::endl;
            m_ncp_factors.print();
#endif
        }
    }
    double computeObjectiveError() {
        // current low rank tensor
        // UWORD krpsize = arma::prod(this->m_dimensions);
        // krpsize /= this->m_dimensions[0];
        // MAT krpleavingzero.zeros(krpsize, this->m_k);
        // krp_leave_out_one(0, &krpleavingzero);
        // MAT lowranktensor(this->m_dimensions[0], krpsize);
        // lowranktensor = this->ncp_factors[0] * trans(krpleavingzero);

        // compute current low rank tensor as above.
        m_ncp_factors.krp_leave_out_one(0, &ncp_krp[0]);
        // cblas_dgemm_(const CBLAS_LAYOUT Layout,
        //              const CBLAS_TRANSPOSE transa,
        //              const CBLAS_TRANSPOSE transb,
        //              const MKL_INT m, const MKL_INT n,
        //              const MKL_INT k, const double alpha,
        //              const double * a, const MKL_INT lda,
        //              const double * b, const MKL_INT ldb,
        //              const double beta, double * c,
        //              const MKL_INT ldc);
        // char transa = 'T';
        // char transb = 'N';
        int m = m_ncp_factors.factor(0).n_rows;
        int n = ncp_krp[0].n_rows;
        int k = m_ncp_factors.factor(0).n_cols;
        int lda = m;
        int ldb = n;
        int ldc = m;
        double alpha = 1;
        double beta = 0;
        // double *output_tensor = new double[ldc * n];
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    m, n, k, alpha,
                    m_ncp_factors.factor(0).memptr(),
                    lda, ncp_krp[0].memptr() , ldb, beta,
                    lowranktensor->m_data , ldc);
        // INFO << "lowrank tensor::" << std::endl;
        // lowranktensor->print();
        // for (int i=0; i < ldc*n; i++){
        //     INFO << i << ":" << output_tensor[i] << std::endl;
        // }
        double err = m_input_tensor.err(*lowranktensor);
        return err;
    }
};  // class AUNTF
}  // namespace planc
#endif  // NTF_AUNTF_HPP_