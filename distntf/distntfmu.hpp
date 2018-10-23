/* Copyright Ramakrishnan Kannan 2018 */

#ifndef DISTNTF_DISTNTFMU_HPP_
#define DISTNTF_DISTNTFMU_HPP_

#include "distntf/distauntf.hpp"

namespace planc {

class DistNTFMU : public DistAUNTF {
 protected:
  MAT update(const int mode) {
    MAT H(this->m_local_ncp_factors.factor(mode));
    if (m_nls_sizes[mode] > 0) {
      MAT temp = H * this->global_gram + EPSILON;
      MAT rhs = this->ncp_local_mttkrp_t[mode].t();
      H = (H % rhs) / temp;
    } else {  // Return unmodified factor
      H.zeros();
    }
    return H.t();
  }

 public:
  DistNTFMU(const Tensor &i_tensor, const int i_k, algotype i_algo,
            const UVEC &i_global_dims, const UVEC &i_local_dims,
            const UVEC &i_nls_sizes, const UVEC &i_nls_idxs,
            const NTFMPICommunicator &i_mpicomm)
      : DistAUNTF(i_tensor, i_k, i_algo, i_global_dims, i_local_dims,
                  i_nls_sizes, i_nls_idxs, i_mpicomm) {}
};  // class DistNTFMU

}  // namespace planc

#endif  // DISTNTF_DISTNTFMU_HPP_
