#ifndef NTF_AUNTF_HPP
#define NTF_AUNTF_HPP

#include <armadillo>
#include "updatealgos.hpp"
#include "ntf_utils.h"

namespace PLANC {

#define TENSOR_DIM (m_input_tensor.dimension())
#define TENSOR_NUMEL (m_input_tensor.numel())


class AUNTF {
  private:
    const Tensor m_input_tensor;
    NCPFactors m_ncp_factors;
    int m_num_it;
    FMAT gram_without_one;
    const int m_low_rank_k;
    FMAT *ncp_krp;
    FMAT *ncp_mttkrp;
    const ntfalgo m_updalgo;
  public:
    AUNTF(const Tensor &i_tensor, const int i_k, ntfalgo i_algo) :
        m_ncp_factors(i_tensor.dimensions(), i_k),
        m_input_tensor(i_tensor.dimensions(), i_tensor.data()),
        m_low_rank_k(i_k),
        m_updalgo(i_algo) {
        gram_without_one = arma::zeros<FMAT>(i_k, i_k);
        ncp_mttkrp = arma::zeros<FMAT>(TENSOR_NUMEL, i_k);
        for (int i = 0; i < i_tensor.order(); i++) {
            UWORD current_size = TENSOR_NUMEL - TENSOR_DIM[i];
            ncp_krp[i] = arma::zeros <FMAT>(current_size, i_k);
            ncp_mttkrp[i] = arma::zeros<FMAT>(TENSOR_DIM[i], i_k);
        }
    }
    void num_it(const int i_n) { this->m_num_it = i_n;}
    void computeNTF() {
        for (int i = 0; i < m_num_it; i++) {
            for (int j = 0; j < this->m_input_tensor.order(); j++) {
                m_ncp_factors.gram_leave_out_one(j, &gram_without_one);
                m_ncp_factors.krp_leave_out_one(j, &ncp_krp[j]);
                m_input_tensor.mttkrp(j, &ncp_krp[j], &ncp_mttkrp[j]);
                FMAT factor = update(m_updalgo, gram_without_one,
                                     ncp_mttkrp[j]);
                m_ncp_factors.set(j, factor);
            }
        }
    }
}
}  // namespace PLANC
#endif  // NTF_AUNTF_HPP