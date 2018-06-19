/* Copyright 2016 Ramakrishnan Kannan */

#ifndef DISTNMF_NAIVE_ANLS_BPP_HPP_
#define DISTNMF_NAIVE_ANLS_BPP_HPP_
#pragma once
#include "bppnnls.hpp"
#include "distnmf1D.hpp"

template <class INPUTMATTYPE>
class DistNaiveANLSBPP : public DistNMF1D<INPUTMATTYPE> {
  MAT HtH, WtW;
  MAT AcolstW, ArowsH;
  MAT ArowsHt, AcolstWt;
  ROWVEC localWnorm;
  ROWVEC Wnorm;

  void printConfig() {
    PRINTROOT("NAIVEANLSBPP constructor completed::"
              << "::A::" << this->m_globalm << "x" << this->m_globaln
              << "::norm::" << this->m_globalsqnormA << "::k::" << this->m_k
              << "::it::" << this->m_num_iterations);
  }

 public:
  DistNaiveANLSBPP(const INPUTMATTYPE &Arows, const INPUTMATTYPE &Acols,
                   const MAT &leftlowrankfactor, const MAT &rightlowrankfactor,
                   const MPICommunicator &communicator)
      : DistNMF1D<INPUTMATTYPE>(Arows, Acols, leftlowrankfactor,
                                rightlowrankfactor, communicator) {
    // allocate memory for matrices
    HtH.zeros(this->m_k, this->m_k);
    WtW.zeros(this->m_k, this->m_k);
    AcolstW.zeros(this->globaln() / this->m_mpicomm.size(), this->m_k);
    AcolstWt.zeros(this->m_k, this->globaln() / this->m_mpicomm.size());
    ArowsH.zeros(this->globalm() / this->m_mpicomm.size(), this->m_k);
    ArowsHt.zeros(this->m_k, this->globalm() / this->m_mpicomm.size());
    localWnorm.zeros(this->m_k);
    Wnorm.zeros(this->m_k);
    PRINTROOT("NAIVEANLSBPP Constructor completed");
    printConfig();
  }
  void computeNMF() {
    PRINTROOT("starting computeNMF");
#ifdef MPI_VERBOSE
    DISTPRINTINFO(PRINTMAT(this->m_Arows));
    DISTPRINTINFO(PRINTMAT(this->m_Acols));
#endif
    // we need only Acolst. So we are transposing and keeping it.
    // Also for dense matrix, having duplicate copy is costly.
#ifndef BUILD_SPARSE
    inplace_trans(this->m_Acols);
    INPUTMATTYPE Acolst(this->m_Acols.memptr(), this->m_Acols.n_rows,
                        this->m_Acols.n_cols, false, true);
#else
    INPUTMATTYPE Acolst = this->m_Acols.t();
#endif

    for (int iter = 0; iter < this->num_iterations(); iter++) {
      if (iter > 0 && this->is_compute_error()) {
        this->m_prevH = this->m_H;
        this->m_prevHtH = this->HtH;
      }
      mpitic();  // total_d W&H
      // update H given A,W
      {
        double tempTime = this->globalW();
        this->time_stats.communication_duration(tempTime);
        this->time_stats.allgather_duration(tempTime);
        mpitic();  // gramW
        WtW = this->m_globalW.t() * this->m_globalW;
        tempTime = mpitoc();  // gramW
        this->time_stats.compute_duration(tempTime);
        this->time_stats.gram_duration(tempTime);
        DISTPRINTINFO(PRINTMATINFO(WtW));
#ifdef MPI_VERBOSE
        DISTPRINTINFO(PRINTMAT(this->m_W));
        PRINTROOT(PRINTMAT(this->m_globalW));
        DISTPRINTINFO(PRINTMAT(WtW));
#endif
        tempTime = -1;
        mpitic();  // mmH
        AcolstW = Acolst * this->m_globalW;
// #if defined BUILD_SPARSE && MKL_FOUND
//                 ARMAMKLSCSCMM(this->m_Acols, 'T', this->m_globalWt,
//                 AcolstWt.memptr());
// #ifdef MPI_VERBOSE
//                 DISTPRINTINFO(PRINTMAT(AcolstWt));
// #endif
//                 AcolstW = reshape(AcolstWt.t(), this->globaln() /
//                 this->m_mpicomm.size(), this->m_k);
// #else
//                 AcolstW = Acolst * this->m_globalW;
// #endif
#ifdef MPI_VERBOSE
        DISTPRINTINFO(PRINTMAT(AcolstW));
#endif
        tempTime = mpitoc();  // mmH
        PRINTROOT(PRINTMATINFO(this->m_Acols)
                  << PRINTMATINFO(this->m_globalW) << PRINTMATINFO(AcolstW));
        this->time_stats.compute_duration(tempTime);
        this->time_stats.mm_duration(tempTime);
        this->reportTime(tempTime, "::AcolstW::");
        mpitic();  // nnlsH
        DISTPRINTINFO(PRINTMATINFO(WtW));
        DISTPRINTINFO(PRINTMATINFO(AcolstW));
        BPPNNLS<MAT, VEC> subProblem2(WtW, AcolstW, true);
        subProblem2.solveNNLS();
        this->m_Ht = subProblem2.getSolutionMatrix();
        fixNumericalError<MAT>(&(this->m_Ht));
        DISTPRINTINFO("OptimizeBlock::NNLS::" << PRINTMATINFO(this->m_Ht));
#ifdef MPI_VERBOSE
        DISTPRINTINFO(PRINTMAT(this->m_Ht));
#endif
        this->m_H = this->m_Ht.t();
        tempTime = mpitoc();  // nnlsH
        this->time_stats.compute_duration(tempTime);
        this->time_stats.nnls_duration(tempTime);
        this->reportTime(tempTime, "NNLS::H::");
      }
      // update W given A,H
      {
        double tempTime = this->globalH();
        this->time_stats.communication_duration(tempTime);
        this->time_stats.allgather_duration(tempTime);
        mpitic();  // gramH
        HtH = this->m_globalH.t() * this->m_globalH;
        DISTPRINTINFO(PRINTMATINFO(HtH));
#ifdef MPI_VERBOSE
        DISTPRINTINFO(PRINTMAT(this->m_H));
        PRINTROOT(PRINTMAT(this->m_globalH));
        DISTPRINTINFO(PRINTMAT(HtH));
#endif
        tempTime = mpitoc();  // gramH
        this->time_stats.compute_duration(tempTime);
        this->time_stats.gram_duration(tempTime);
        tempTime = -1;
        mpitic();  // mmW
        ArowsH = this->m_Arows * this->m_globalH;
// #if defined BUILD_SPARSE && MKL_FOUND
//                 ARMAMKLSCSCMM(this->m_Arows, 'N', this->m_globalHt,
//                 ArowsHt.memptr());
// #ifdef MPI_VERBOSE
//                 DISTPRINTINFO(PRINTMAT(ArowsHt));
// #endif
//                 ArowsH = ArowsHt.t();
// #else
//                 ArowsH = this->m_Arows * this->m_globalH;
// #endif
#ifdef MPI_VERBOSE
        DISTPRINTINFO(PRINTMAT(ArowsH));
#endif
        tempTime = mpitoc();  // mmW
        PRINTROOT(PRINTMATINFO(this->m_Arows)
                  << PRINTMATINFO(this->m_globalH) << PRINTMATINFO(ArowsH));
        this->time_stats.compute_duration(tempTime);
        this->time_stats.mm_duration(tempTime);
        this->reportTime(tempTime, "::ArowsH::");
        mpitic();  // nnlsW
        BPPNNLS<MAT, VEC> subProblem1(HtH, ArowsH, true);
        subProblem1.solveNNLS();
        this->m_Wt = subProblem1.getSolutionMatrix();
        fixNumericalError<MAT>(&(this->m_Wt));
        DISTPRINTINFO("OptimizeBlock::NNLS::" << PRINTMATINFO(this->m_Wt));
#ifdef MPI_VERBOSE
        DISTPRINTINFO(PRINTMAT(this->m_Wt));
#endif
        this->m_W = this->m_Wt.t();
        tempTime = mpitoc();  // nnlsW
        this->time_stats.compute_duration(tempTime);
        this->time_stats.nnls_duration(tempTime);
        this->reportTime(tempTime, "NNLS::W::");
      }
      this->time_stats.duration(mpitoc());  // total_d W&H
      if (iter > 0 && this->is_compute_error()) {
        this->computeError(WtW, this->m_prevHtH);
        PRINTROOT("it=" << iter << "::algo::" << this->m_algorithm
                        << "::k::" << this->m_k
                        << "::err::" << this->m_objective_err << "::relerr::"
                        << this->m_objective_err / this->m_globalsqnormA);
      }
      PRINTROOT("completed it=" << iter
                                << "::taken::" << this->time_stats.duration());
    }
    MPI_Barrier(MPI_COMM_WORLD);
    this->reportTime(this->time_stats.duration(), "total_d");
    this->reportTime(this->time_stats.communication_duration(), "total_comm");
    this->reportTime(this->time_stats.compute_duration(), "total_comp");
    this->reportTime(this->time_stats.allgather_duration(), "total_allgather");
    this->reportTime(this->time_stats.gram_duration(), "total_gram");
    this->reportTime(this->time_stats.mm_duration(), "total_mm");
    this->reportTime(this->time_stats.nnls_duration(), "total_nnls");
    if (this->is_compute_error()) {
      this->reportTime(this->time_stats.err_compute_duration(),
                       "total_err_compute");
      this->reportTime(this->time_stats.err_compute_duration(),
                       "total_err_communication");
    }
  }
};

#endif  // DISTNMF_NAIVE_ANLS_BPP_HPP_
