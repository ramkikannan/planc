/* Copyright 2016 Ramakrishnan Kannan */

#ifndef DISTNMF_DISTANLSBPP_HPP_
#define DISTNMF_DISTANLSBPP_HPP_

#include "distnmf/aunmf.hpp"
#include "nnls/bppnnls.hpp"

/**
 * Provides the updateW and updateH for the
 * distributed ANLS/BPP algorithm.
 */

#ifdef BUILD_CUDA
#define ONE_THREAD_MATRIX_SIZE 1000
#include <omp.h>
#else
#define ONE_THREAD_MATRIX_SIZE 2000
#endif

namespace planc {

template <class INPUTMATTYPE>
class DistANLSBPP : public DistAUNMF<INPUTMATTYPE> {
 private:
  ROWVEC localWnorm;
  ROWVEC Wnorm;

  void allocateMatrices() {}

  /**
   * ANLS/BPP with chunking the RHS into smaller independent
   * subproblems
   */
  void updateOtherGivenOneMultipleRHS(const MAT& giventGiven,
                                      const MAT& giventInput, MAT* othermat) {
    UINT numChunks = giventInput.n_cols / ONE_THREAD_MATRIX_SIZE;
    if (numChunks * ONE_THREAD_MATRIX_SIZE < giventInput.n_cols) numChunks++;

// #pragma omp parallel for schedule(dynamic)
    for (UINT i = 0; i < numChunks; i++) {
      UINT spanStart = i * ONE_THREAD_MATRIX_SIZE;
      UINT spanEnd = (i + 1) * ONE_THREAD_MATRIX_SIZE - 1;
      if (spanEnd > giventInput.n_cols - 1) {
        spanEnd = giventInput.n_cols - 1;
      }
      BPPNNLS<MAT, VEC> subProblem(giventGiven,
                            (MAT)giventInput.cols(spanStart, spanEnd), true);
#ifdef _VERBOSE
      // #pragma omp critical
      {
        INFO << "Scheduling " << worh << " start=" << spanStart
             << ", end=" << spanEnd
             // << ", tid=" << omp_get_thread_num()
             << std::endl
             << "LHS ::" << std::endl
             << giventGiven << std::endl
             << "RHS ::" << std::endl
             << (MAT)giventInput.cols(spanStart, spanEnd) << std::endl;
      }
#endif

      subProblem.solveNNLS();

#ifdef _VERBOSE
      INFO << "completed " << worh << " start=" << spanStart
           << ", end=" << spanEnd
           // << ", tid=" << omp_get_thread_num() << " cpu=" << sched_getcpu()
           << " time taken=" << t2 << std::endl;
#endif
      (*othermat).rows(spanStart, spanEnd) = subProblem.getSolutionMatrix().t();
    }
  }

 protected:
  /**
   * update W given HtH and AHt
   * AHtij is of size \f$ k \times \frac{globalm}/{p}\f$.
   * this->W is of size \f$\frac{globalm}{p} \times k \f$
   * this->HtH is of size kxk
  */
  void updateW() {
    updateOtherGivenOneMultipleRHS(this->HtH, this->AHtij, &this->W);
    this->Wt = this->W.t();
  }
  /**
   * updateH given WtAij and WtW
   * WtAij is of size \f$k \times \frac{globaln}{p} \f$
   * this->H is of size \f$ \frac{globaln}{p} \times k \f$
   * this->WtW is of size kxk
   */  
  void updateH() {
    updateOtherGivenOneMultipleRHS(this->WtW, this->WtAij, &this->H);
    this->Ht = this->H.t();
  }

 public:
  DistANLSBPP(const INPUTMATTYPE& input, const MAT& leftlowrankfactor,
              const MAT& rightlowrankfactor,
              const MPICommunicator& communicator, const int numkblks)
      : DistAUNMF<INPUTMATTYPE>(input, leftlowrankfactor, rightlowrankfactor,
                                communicator, numkblks) {
    localWnorm.zeros(this->k);
    Wnorm.zeros(this->k);
    PRINTROOT("DistANLSBPP() constructor successful");
  }

  ~DistANLSBPP() {
    /*
       tempHtH.clear();
       tempWtW.clear();
       tempAHtij.clear();
       tempWtAij.clear();
     */
  }
};  // class DistANLSBPP2D

}  // namespace planc

#endif  // DISTNMF_DISTANLSBPP_HPP_
