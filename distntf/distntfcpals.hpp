/* Copyright Ramakrishnan Kannan 2018 */

#ifndef DISTNTF_DISTNTFCPALS_HPP_
#define DISTNTF_DISTNTFCPALS_HPP_

#include "distntf/distauntf.hpp"

namespace planc {

class DistNTFCPALS : public DistAUNTF {
 protected:
  /**
   * This is unconstrained CP ALS update function.
   * Given the MTTKRP and the hadamard of all the grams, we
   * determine the factor matrix to be updated.
   * @param[in] Mode of the factor to be updated
   * @returns The new updated factor
   */
  MAT update(const int mode) {
    MAT Ht = this->m_local_ncp_factors.factor(mode).t();
    if (m_nls_sizes[mode] > 0) {
      Ht = arma::solve(this->global_gram, this->ncp_local_mttkrp_t[mode]);
    } else {  // Return unmodified factor
      Ht.zeros();
    }
    return Ht;
  }

 public:
  DistNTFCPALS(const Tensor &i_tensor, const int i_k, algotype i_algo,
               const UVEC &i_global_dims, const UVEC &i_local_dims,
               const UVEC &i_nls_sizes, const UVEC &i_nls_idxs,
               const NTFMPICommunicator &i_mpicomm)
      : DistAUNTF(i_tensor, i_k, i_algo, i_global_dims, i_local_dims,
                  i_nls_sizes, i_nls_idxs, i_mpicomm) {}
};  // class DistNTFMU

}  // namespace planc

#endif  // DISTNTF_DISTNTFCPALS_HPP_
