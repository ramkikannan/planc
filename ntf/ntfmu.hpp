/* Copyright Ramakrishnan Kannan 2018 */

#ifndef NTF_NTFMU_HPP_
#define NTF_NTFMU_HPP_

#include "ntf/auntf.hpp"

namespace planc {
class NTFMU : public AUNTF {
 protected:
  MAT update(const int mode) {
    MAT H(this->m_ncp_factors.factor(mode));
    MAT temp = H * this->gram_without_one + EPSILON;
    H = (H % this->ncp_mttkrp_t[mode].t()) / temp;
    return H.t();
  }

 public:
  NTFMU(const Tensor &i_tensor, const int i_k, algotype i_algo)
      : AUNTF(i_tensor, i_k, i_algo) {}
};  // class NTFMU
}  // namespace planc

#endif  // NTF_NTFMU_HPP_
