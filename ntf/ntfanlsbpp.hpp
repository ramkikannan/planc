/* Copyright Ramakrishnan Kannan 2018 */

#ifndef NTF_NTFANLSBPP_HPP_
#define NTF_NTFANLSBPP_HPP_

#include "nnls/bppnnls.hpp"
#include "ntf/auntf.hpp"

#define ONE_THREAD_MATRIX_SIZE 2000

namespace planc {

class NTFANLSBPP : public AUNTF {
 protected:
  MAT update(const int mode) {
    MAT othermat(this->m_ncp_factors.factor(mode).t());
    int numThreads =
        (this->ncp_mttkrp_t[mode].n_cols / ONE_THREAD_MATRIX_SIZE) + 1;
#pragma omp parallel for schedule(dynamic)
    for (UINT i = 0; i < numThreads; i++) {
      UINT spanStart = i * ONE_THREAD_MATRIX_SIZE;
      UINT spanEnd = (i + 1) * ONE_THREAD_MATRIX_SIZE - 1;
      if (spanEnd > this->ncp_mttkrp_t[mode].n_cols - 1) {
        spanEnd = this->ncp_mttkrp_t[mode].n_cols - 1;
      }
      // if it is exactly divisible, the last iteration is unnecessary.
      BPPNNLS<MAT, VEC> *subProblem;
      if (spanStart <= spanEnd) {
        if (spanStart == spanEnd) {
          subProblem = new BPPNNLS<MAT, VEC>(
              this->gram_without_one,
              (VEC)this->ncp_mttkrp_t[mode].col(spanStart), true);
        } else {  // if (spanStart < spanEnd)
          subProblem = new BPPNNLS<MAT, VEC>(
              this->gram_without_one,
              (MAT)this->ncp_mttkrp_t[mode].cols(spanStart, spanEnd), true);
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
          othermat.cols(spanStart, spanEnd) =
              subProblem->getSolutionMatrix();
        }
        subProblem->clear();
        delete subProblem;
      }
    }
    return othermat;
  }

 public:
  NTFANLSBPP(const Tensor &i_tensor, const int i_k, algotype i_algo)
      : AUNTF(i_tensor, i_k, i_algo) {}
};  // class NTFANLSBPP

}  // namespace planc

#endif  // NTF_NTFANLSBPP_HPP_
