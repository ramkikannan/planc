/* Copyright Ramakrishnan Kannan 2018 */

#ifndef NTF_NTFANLSBPP_HPP_
#define NTF_NTFANLSBPP_HPP_

#include "nnls/bppnnls.hpp"
#include "ntf/auntf.hpp"

namespace planc {

#define ONE_THREAD_MATRIX_SIZE 2000

class NTFANLSBPP : public AUNTF {
 protected:
  MAT update(const int mode) {
    MAT othermat(this->m_ncp_factors.factor(mode).t());
    UINT nrhs = this->ncp_mttkrp_t[mode].n_cols;
    UINT numChunks = nrhs / ONE_THREAD_MATRIX_SIZE;
    if (numChunks * ONE_THREAD_MATRIX_SIZE < nrhs) numChunks++;

// #pragma omp parallel for schedule(dynamic)
    for (UINT i = 0; i < numChunks; i++) {
      UINT spanStart = i * ONE_THREAD_MATRIX_SIZE;
      UINT spanEnd = (i + 1) * ONE_THREAD_MATRIX_SIZE - 1;
      if (spanEnd > nrhs - 1) {
        spanEnd = nrhs - 1;
      }

      BPPNNLS<MAT, VEC> subProblem(this->gram_without_one,
                        (MAT)this->ncp_mttkrp_t[mode].cols(spanStart, spanEnd),
                        true);
#ifdef _VERBOSE
      // #pragma omp critical
      {
        INFO << "Scheduling " << worh << " start=" << spanStart
             << ", end=" << spanEnd
             // << ", tid=" << omp_get_thread_num()
             << std::endl
             << "LHS ::" << std::endl
             << this->gram_without_one << std::endl
             << "RHS ::" << std::endl
             << this->ncp_mttkrp_t[mode].cols(spanStart, spanEnd) << std::endl;
      }
#endif

      subProblem.solveNNLS();

#ifdef _VERBOSE
      INFO << "completed " << worh << " start=" << spanStart
           << ", end=" << spanEnd
           // << ", tid=" << omp_get_thread_num() << " cpu=" << sched_getcpu()
           << " time taken=" << t2 << std::endl;
#endif
      othermat.cols(spanStart, spanEnd) = subProblem.getSolutionMatrix();
    }
    return othermat;
  }

 public:
  NTFANLSBPP(const Tensor &i_tensor, const int i_k, algotype i_algo)
      : AUNTF(i_tensor, i_k, i_algo) {}
};  // class NTFANLSBPP

}  // namespace planc

#endif  // NTF_NTFANLSBPP_HPP_
