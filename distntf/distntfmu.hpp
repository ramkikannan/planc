/* Copyright Ramakrishnan Kannan 2018 */

#ifndef DISTNTF_DISTNTFMU_HPP_
#define DISTNTF_DISTNTFMU_HPP_

#include "distntf/distauntf.hpp"

namespace planc {

class DistNTFMU : public DistAUNTF {
 protected:
  MAT update(const int mode) {
    MAT H(this->m_local_ncp_factors.factor(mode));
    MAT temp = H * this->global_gram + EPSILON;
    H = (H % this->ncp_local_mttkrp_t[mode].t()) / temp;
    return H.t();
  }

 public:
  DistNTFMU(const Tensor &i_tensor, const int i_k, algotype i_algo,
            const UVEC &i_global_dims, const UVEC &i_local_dims,
            const NTFMPICommunicator &i_mpicomm)
      : DistAUNTF(i_tensor, i_k, i_algo, i_global_dims, i_local_dims,
                  i_mpicomm) {}
};  // class DistNTFMU

}  // namespace planc

#endif  // DISTNTF_DISTNTFMU_HPP_
