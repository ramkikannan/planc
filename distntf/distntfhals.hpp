/* Copyright Ramakrishnan Kannan 2018 */

#ifndef DISTNTF_DISTNTFHALS_HPP_
#define DISTNTF_DISTNTFHALS_HPP_

#include "distntf/distauntf.hpp"

namespace planc {

class DistNTFHALS : public DistAUNTF {
 protected:
  MAT update(const int mode) {
    MAT H(this->m_local_ncp_factors.factor(mode));
    // iterate over all columns of H
    for (int i = 0; i < this->m_local_ncp_factors.rank(); i++) {
      VEC updHi;
      if (m_nls_sizes[mode] > 0) {
        updHi = H.col(i) + ((this->ncp_local_mttkrp_t[mode].row(i)).t() -
                            H * this->global_gram.col(i));
      } else {
        updHi = H.col(i);
        updHi.zeros();
      }

      fixNumericalError<VEC>(&updHi);
      double normHi = arma::norm(updHi, 2);
      normHi *= normHi;
      double globalnormHi = normHi;
      MPI_Allreduce(&normHi, &globalnormHi, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      if (globalnormHi > 0) {
        H.col(i) = updHi;
      }
    }
    return H.t();
  }

 public:
  DistNTFHALS(const Tensor &i_tensor, const int i_k, algotype i_algo,
              const UVEC &i_global_dims, const UVEC &i_local_dims,
              const UVEC &i_nls_sizes, const UVEC &i_nls_idxs,
              const NTFMPICommunicator &i_mpicomm)
      : DistAUNTF(i_tensor, i_k, i_algo, i_global_dims, i_local_dims,
                  i_nls_sizes, i_nls_idxs, i_mpicomm) {}
};  // class DistNTFHALS

}  // namespace planc

#endif  // DISTNTF_DISTNTFHALS_HPP_
