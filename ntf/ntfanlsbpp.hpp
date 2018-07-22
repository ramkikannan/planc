/* Copyright Ramakrishnan Kannan 2018 */

#ifndef NTF_NTFANLSBPP_HPP_
#define NTF_NTFANLSBPP_HPP_

#include "ntf/auntf.hpp"
#include "nnls/bppnnls.hpp"

namespace planc {

class NTFANLSBPP : public AUNTF {
 protected:
  MAT update(const int mode) {
    BPPNNLS<MAT, VEC> subProblem(this->gram_without_one,
                                 this->ncp_mttkrp_t[mode], true);
    subProblem.solveNNLS();
    return subProblem.getSolutionMatrix();
  }

 public:
  NTFANLSBPP(const Tensor &i_tensor, const int i_k, algotype i_algo)
      : AUNTF(i_tensor, i_k, i_algo) {}
};  // class NTFANLSBPP

}  // namespace planc

#endif  // NTF_NTFANLSBPP_HPP_
