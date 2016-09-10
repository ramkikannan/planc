/* Copyright 2016 Ramakrishnan Kannan */
#ifndef MPI_NAIVE_ANLS_BPP_HPP_
#define MPI_NAIVE_ANLS_BPP_HPP_
#pragma once
#include "distnmf1D.hpp"
#include "bppnnls.hpp"


using namespace std;

template<class INPUTMATTYPE>
class DistNaiveANLSBPP : public DistNMF1D<INPUTMATTYPE> {
    FMAT HtH, WtW;
    FMAT AcolstW, ArowsH;
    FMAT ArowsHt, AcolstWt;
    // need double precision for nnls
    MAT tempHtH, tempArowsH, tempWtW, tempAcolstW;
    FROWVEC localWnorm;
    FROWVEC Wnorm;

    void printConfig() {
        PRINTROOT("NAIVEANLSBPP constructor completed::" << "::A::" \
                  << this->m_globalm <<  "x" << this->m_globaln \
                  << "::norm::" << this->m_globalsqnormA << "::k::" << this->m_k
                  << "::it::" << this->m_num_iterations);
    }

  public:
    DistNaiveANLSBPP(const INPUTMATTYPE &Arows, const INPUTMATTYPE &Acols,
                     const FMAT &leftlowrankfactor, const FMAT &rightlowrankfactor,
                     const MPICommunicator& communicator):
        DistNMF1D<INPUTMATTYPE>(Arows, Acols, leftlowrankfactor,
                                rightlowrankfactor, communicator) {
        // allocate memory for matrices
        HtH.zeros(this->m_k, this->m_k);
        WtW.zeros(this->m_k, this->m_k);
        tempHtH.zeros(this->m_k, this->m_k);
        tempWtW.zeros(this->m_k, this->m_k);
        AcolstW.zeros(this->globaln() / this->m_mpicomm.size(), this->m_k);
        AcolstWt.zeros(this->m_k, this->globaln() / this->m_mpicomm.size());
        tempAcolstW.zeros(this->globalm() / this->m_mpicomm.size(), this->m_k);
        ArowsH.zeros(this->globalm() / this->m_mpicomm.size(), this->m_k);
        ArowsHt.zeros(this->m_k, this->globalm() / this->m_mpicomm.size());
        tempArowsH.zeros(this->globalm() / this->m_mpicomm.size(), this->m_k);
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
                this->applyReg(this->regH, &this->WtW);
                tempTime = -1;
                mpitic();  // mmH
                AcolstW = Acolst * this->m_globalW;
// #if defined BUILD_SPARSE && MKL_FOUND
//                 ARMAMKLSCSCMM(this->m_Acols, 'T', this->m_globalWt, AcolstWt.memptr());
// #ifdef MPI_VERBOSE
//                 DISTPRINTINFO(PRINTMAT(AcolstWt));
// #endif
//                 AcolstW = reshape(AcolstWt.t(), this->globaln() / this->m_mpicomm.size(), this->m_k);
// #else
//                 AcolstW = Acolst * this->m_globalW;
// #endif
#ifdef MPI_VERBOSE
                DISTPRINTINFO(PRINTMAT(trans(AcolstW)));
#endif
                tempTime = mpitoc();  // mmH
                PRINTROOT(PRINTMATINFO(this->m_Acols) << PRINTMATINFO(this->m_globalW)
                          << PRINTMATINFO(AcolstW));
                this->time_stats.compute_duration(tempTime);
                this->time_stats.mm_duration(tempTime);
                this->reportTime(tempTime, "::AcolstW::");
                mpitic();  // nnlsH
                tempWtW = arma::conv_to<MAT >::from(WtW);
                tempAcolstW = arma::conv_to<MAT >::from(trans(AcolstW));
                DISTPRINTINFO(PRINTMATINFO(tempWtW));
                DISTPRINTINFO(PRINTMATINFO(tempAcolstW));
                BPPNNLS<MAT, VEC > subProblem2(tempWtW, tempAcolstW, true);
                subProblem2.solveNNLS();
                this->m_Ht = arma::conv_to<FMAT >::from(subProblem2.getSolutionMatrix());
                fixNumericalError<FMAT >(&(this->m_Ht));
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
                this->applyReg(this->regW, &this->HtH);
                tempTime = -1;
                mpitic();  // mmW
                ArowsH = this->m_Arows * this->m_globalH;
// #if defined BUILD_SPARSE && MKL_FOUND
//                 ARMAMKLSCSCMM(this->m_Arows, 'N', this->m_globalHt, ArowsHt.memptr());
// #ifdef MPI_VERBOSE
//                 DISTPRINTINFO(PRINTMAT(ArowsHt));
// #endif
//                 ArowsH = ArowsHt.t();
// #else
//                 ArowsH = this->m_Arows * this->m_globalH;
// #endif
#ifdef MPI_VERBOSE
                DISTPRINTINFO(PRINTMAT(trans(ArowsH)));
#endif
                tempTime = mpitoc();  // mmW
                PRINTROOT(PRINTMATINFO(this->m_Arows)
                          << PRINTMATINFO(this->m_globalH)
                          << PRINTMATINFO(ArowsH));
                this->time_stats.compute_duration(tempTime);
                this->time_stats.mm_duration(tempTime);
                this->reportTime(tempTime, "::ArowsH::");
                mpitic();  // nnlsW
                tempHtH = arma::conv_to<MAT >::from(HtH);
                tempArowsH = arma::conv_to<MAT >::from(trans(ArowsH));
                DISTPRINTINFO(PRINTMATINFO(tempHtH));
                DISTPRINTINFO(PRINTMATINFO(tempArowsH));
                BPPNNLS<MAT, VEC > subProblem1(tempHtH, tempArowsH, true);
                subProblem1.solveNNLS();
                this->m_Wt = arma::conv_to<FMAT >::from(subProblem1.getSolutionMatrix());
                fixNumericalError<FMAT >(&(this->m_Wt));
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
                PRINTROOT("it=" << iter << "::algo::" << this->m_algorithm \
                          << "::k::" << this->m_k \
                          << "::err::" << this->m_objective_err \
                          << "::relerr::" << this->m_objective_err / this->m_globalsqnormA);
            }
            PRINTROOT("completed it=" << iter << "::taken::"
                      << this->time_stats.duration());
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

#endif  // MPI_NAIVE_ANLS_BPP_HPP_
