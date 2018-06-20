/* Copyright 2016 Ramakrishnan Kannan */

#ifndef DISTNMF_DISTANLSBPP_HPP_
#define DISTNMF_DISTANLSBPP_HPP_

#include "distnmf/aunmf.hpp"
#include "nnls/bppnnls.hpp"

#ifdef BUILD_CUDA
#define ONE_THREAD_MATRIX_SIZE 1000
#include <omp.h>
#else
#define ONE_THREAD_MATRIX_SIZE giventInput.n_cols + 5
#endif

template <class INPUTMATTYPE>
class DistANLSBPP : public DistAUNMF<INPUTMATTYPE> {
 private:
  ROWVEC localWnorm;
  ROWVEC Wnorm;

  void allocateMatrices() {}

  void updateOtherGivenOneMultipleRHS(const MAT& giventGiven,
                                      const MAT& giventInput, MAT* othermat) {
    UINT numThreads = (giventInput.n_cols / ONE_THREAD_MATRIX_SIZE) + 1;
#pragma omp parallel for schedule(dynamic)
    for (UINT i = 0; i < numThreads; i++) {
      UINT spanStart = i * ONE_THREAD_MATRIX_SIZE;
      UINT spanEnd = (i + 1) * ONE_THREAD_MATRIX_SIZE - 1;
      if (spanEnd > giventInput.n_cols - 1) {
        spanEnd = giventInput.n_cols - 1;
      }
      // if it is exactly divisible, the last iteration is unnecessary.
      BPPNNLS<MAT, VEC>* subProblem;
      if (spanStart <= spanEnd) {
        if (spanStart == spanEnd) {
          subProblem = new BPPNNLS<MAT, VEC>(
              giventGiven, (VEC)giventInput.col(spanStart), true);
        } else {  // if (spanStart < spanEnd)
          subProblem = new BPPNNLS<MAT, VEC>(
              giventGiven, (MAT)giventInput.cols(spanStart, spanEnd), true);
        }
#ifdef MPI_VERBOSE
#pragma omp parallel
        {
          DISTPRINTINFO("Scheduling " << worh << " start=" << spanStart
                                      << ", end=" << spanEnd
                                      << ", tid=" << omp_get_thread_num());
        }
#endif
        subProblem->solveNNLS();
#ifdef MPI_VERBOSE
#pragma omp parallel
        {
          DISTPRINTINFO("completed " << worh << " start=" << spanStart
                                     << ", end=" << spanEnd
                                     << ", tid=" << omp_get_thread_num()
                                     << " cpu=" << sched_getcpu());
        }
#endif
        if (spanStart == spanEnd) {
          ROWVEC solVec = subProblem->getSolutionVector().t();
          (*othermat).row(i) = solVec;
        } else {  // if (spanStart < spanEnd)
          (*othermat).rows(spanStart, spanEnd) =
              subProblem->getSolutionMatrix().t();
        }
        subProblem->clear();
        delete subProblem;
      }
    }
  }

 protected:
  // updateW given HtH and AHt
  void updateW() {
    updateOtherGivenOneMultipleRHS(this->HtH, this->AHtij, &this->W);
    // BPPNNLS<MAT, VEC> subProblem(this->HtH, this->AHtij, true);
    // subProblem.solveNNLS();
    // this->Wt = subProblem.getSolutionMatrix();
    // fixNumericalError<MAT>(&(this->Wt));
    this->Wt = this->W.t();

    // localWnorm = sum(this->W % this->W);
    // tic();
    // MPI_Allreduce(localWnorm.memptr(), Wnorm.memptr(), this->k, MPI_FLOAT,
    //               MPI_SUM, MPI_COMM_WORLD);
    // double temp = toc();
    // this->time_stats.allgather_duration(temp);
    // for (int i = 0; i < this->k; i++) {
    //     this->W.col(i) = this->W.col(i) / sqrt(Wnorm(i));
    // }
    // this->Wt = this->W.t();
  }

  // updateH given WtW and WtA
  void updateH() {
    updateOtherGivenOneMultipleRHS(this->WtW, this->WtAij, &this->H);
    // BPPNNLS<MAT, VEC> subProblem1(this->WtW, this->WtAij, true);
    // subProblem1.solveNNLS();
    // this->Ht = subProblem1.getSolutionMatrix();
    // fixNumericalError<MAT>(&(this->Ht));
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

#endif  // DISTNMF_DISTANLSBPP_HPP_
