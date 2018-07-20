/* Copyright Ramakrishnan Kannan 2018 */

#ifndef DISTNTF_DISTNTFANLSBPP_HPP_
#define DISTNTF_DISTNTFANLSBPP_HPP_

#include "distntf/distauntf.hpp"
#include "nnls/bppnnls.hpp"

namespace planc {

class DistNTFANLSBPP : public DistAUNTF {
 protected:
  MAT update(const int mode) {
    BPPNNLS<MAT, VEC> subProblem(this->global_gram,
                                 this->ncp_local_mttkrp_t[mode], true);
    subProblem.solveNNLS();
    return subProblem.getSolutionMatrix();
  }

 public:
  DistNTFANLSBPP(const Tensor &i_tensor, const int i_k, algotype i_algo,
             const UVEC &i_global_dims, const UVEC &i_local_dims,
             const NTFMPICommunicator &i_mpicomm)
      : DistAUNTF(i_tensor, i_k, i_algo, i_global_dims, i_local_dims,
                  i_mpicomm) {}
};  // class DistNTFANLSBPP

}  // namespace planc

#endif  // DISTNTF_DISTNTFANLSBPP_HPP_
