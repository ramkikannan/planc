/* Copyright Ramakrishnan Kannan 2018 */

#ifndef DISTNTF_DISTNTFANLSBPP_HPP_
#define DISTNTF_DISTNTFANLSBPP_HPP_

#include "distntf/distauntf.hpp"
#include "nnls/bppnnls.hpp"

namespace planc {

#define ONE_THREAD_MATRIX_SIZE 2000

class DistNTFANLSBPP : public DistAUNTF {
 protected:
  /**
   * This is openmp multithreaded ANLS/BPP update function.
   * Given the MTTKRP and the hadamard of all the grams, we 
   * determine the factor matrix to be updated. 
   * @param[in] Mode of the factor to be updated
   * @returns The new updated factor
   */
  MAT update(const int mode) {
    MAT othermat(this->m_local_ncp_factors_t.factor(mode));
    if (m_nls_sizes[mode] > 0) {
      int numThreads =
        (this->ncp_local_mttkrp_t[mode].n_cols / ONE_THREAD_MATRIX_SIZE) + 1;
      #pragma omp parallel for schedule(dynamic)
      for (UINT i = 0; i < numThreads; i++) {
        UINT spanStart = i * ONE_THREAD_MATRIX_SIZE;
        UINT spanEnd = (i + 1) * ONE_THREAD_MATRIX_SIZE - 1;
        if (spanEnd > this->ncp_local_mttkrp_t[mode].n_cols - 1) {
          spanEnd = this->ncp_local_mttkrp_t[mode].n_cols - 1;
        }
        // if it is exactly divisible, the last iteration is unnecessary.
        BPPNNLS<MAT, VEC> *subProblem;
        if (spanStart <= spanEnd) {
          if (spanStart == spanEnd) {
            subProblem = new BPPNNLS<MAT, VEC>(
              this->global_gram,
              (VEC)this->ncp_local_mttkrp_t[mode].col(spanStart), true);
          } else {  // if (spanStart < spanEnd)
            subProblem = new BPPNNLS<MAT, VEC>(
              this->global_gram,
              (MAT)this->ncp_local_mttkrp_t[mode].cols(spanStart, spanEnd),
              true);
          }
#ifdef _VERBOSE
          INFO << "Scheduling " << worh << " start=" << spanStart
             << ", end=" << spanEnd << ", tid=" << omp_get_thread_num()
             << std::endl;
#endif
          // tic();
          subProblem->solveNNLS();
          // t2 = toc();
#ifdef _VERBOSE
          INFO << "completed " << worh << " start=" << spanStart
             << ", end=" << spanEnd << ", tid=" << omp_get_thread_num()
             << " cpu=" << sched_getcpu() << " time taken=" << t2
             << " num_iterations()=" << numIter << std::endl;
#endif
          if (spanStart == spanEnd) {
            VEC solVec = subProblem->getSolutionVector();
            othermat.col(i) = solVec;
          } else {  // if (spanStart < spanEnd)
            othermat.cols(spanStart, spanEnd) = subProblem->getSolutionMatrix();
          }
          subProblem->clear();
          delete subProblem;
        }
      }
    } else {
      othermat.zeros();
    }
    return othermat;
  }

 public:
  DistNTFANLSBPP(const Tensor &i_tensor, const int i_k, algotype i_algo,
                 const UVEC &i_global_dims, const UVEC &i_local_dims,
                 const UVEC &i_nls_sizes, const UVEC &i_nls_idxs,
                 const NTFMPICommunicator &i_mpicomm)
      : DistAUNTF(i_tensor, i_k, i_algo, i_global_dims, i_local_dims,
                  i_nls_sizes, i_nls_idxs, i_mpicomm) {}
};  // class DistNTFANLSBPP

}  // namespace planc

#endif  // DISTNTF_DISTNTFANLSBPP_HPP_
