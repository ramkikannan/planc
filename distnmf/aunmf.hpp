/* Copyright 2016 Ramakrishnan Kannan */

#ifndef DISTNMF_AUNMF_HPP_
#define DISTNMF_AUNMF_HPP_

#include <mpi.h>
#include <armadillo>
#include <string>
#include <vector>
#include "distnmf.hpp"
#include "mpicomm.hpp"

/*
 * There are totally prxpc process.
 * Each process will hold the following
 * An A of size (globalm/pr) x (globaln/pc)
 * Here each process m=globalm/pr and n=globaln/pc
 * H of size (globaln/p)xk
 * W of size (globalm/p)xk
 * A is mxn matrix
 * H is nxk matrix
 */

template <class INPUTMATTYPE>
class DistAUNMF : public DistNMF<INPUTMATTYPE> {
  // needed in derived algorithms to
  // call BPP routines
 protected:
  MAT HtH;    // H is of size (globaln/p)*k;
  MAT WtW;    // W is of size (globaln/p)*k;
  MAT AHtij;  // AHtij is of size k*(globalm/p)
  MAT WtAij;  // WtAij is of size k*(globaln/p)
  MAT Wt;     // Wt is of size k*(globalm/p)
  MAT Ht;     // Ht is of size k*(globaln/p)

  virtual void updateW() = 0;
  virtual void updateH() = 0;

 private:
  // Things needed while solving for W
  MAT localHtH;         // H is of size (globaln/p)*k;
  MAT Hjt, Hj;          // Hj is of size n*k;
  MAT AijHj, AijHjt;    // AijHj is of size m*k;
  INPUTMATTYPE A_ij_t;  // n*m
  // Things needed while solving for H
  MAT localWtW;        // W is of size (globalm/p)*k;
  MAT Wit, Wi;         // Wi is of size m*k;
  MAT WitAij, AijWit;  // WijtAij is of size k*n;

  // needed for error computation
  MAT prevH;        // used for error computation
  MAT prevHtH;      // used for error computation
  MAT WtAijH;       // global k*k matrix.
  MAT localWtAijH;  // local k*k matrix
  MAT errMtx;
  MAT A_errMtx;

  // needed for block implementation to save memory
  MAT Ht_blk;
  MAT AHtij_blk;
  MAT Wt_blk;
  MAT WtAij_blk;

  std::vector<int> recvWtAsize;
  std::vector<int> recvAHsize;

  int num_k_blocks;
  int perk;

  void allocateMatrices() {
    // collective call related initializations.
    // These initialization are for solving W.
    DISTPRINTINFO("k::" << this->k << "::perk::" << this->perk
                        << "::localm::" << this->m << "::localn::" << this->n
                        << "::globalm::" << this->globalm() << "::globaln::"
                        << this->globaln() << "::MPI_SIZE::" << MPI_SIZE);
    HtH.zeros(this->k, this->k);
    localHtH.zeros(this->k, this->k);
    Hj.zeros(this->n, this->perk);
    Hjt.zeros(this->perk, this->n);
    AijHj.zeros(this->m, this->perk);
    AijHjt.zeros(this->perk, this->m);
    AHtij.zeros(this->k, this->globalm() / MPI_SIZE);
    this->recvAHsize.resize(NUMCOLPROCS);
    int fillsize = this->perk * (this->globalm() / MPI_SIZE);
    fillVector<int>(fillsize, &recvAHsize);
#ifdef MPI_VERBOSE
    if (ISROOT) {
      INFO << "::recvAHsize::";
      printVector<int>(recvAHsize);
    }
#endif
    // allocated for block implementation.
    Ht_blk.zeros(this->perk, (this->globalm() / MPI_SIZE));
    AHtij_blk.zeros(this->perk, this->globalm() / MPI_SIZE);

    // These initialization are for solving H.
    Wt.zeros(this->k, this->globalm() / MPI_SIZE);
    WtW.zeros(this->k, this->k);
    localWtW.zeros(this->k, this->k);
    Wi.zeros(this->m, this->perk);
    Wit.zeros(this->perk, this->m);
    WitAij.zeros(this->perk, this->n);
    AijWit.zeros(this->n, this->perk);
    WtAij.zeros(this->k, this->globaln() / MPI_SIZE);
    this->recvWtAsize.resize(NUMROWPROCS);
    fillsize = this->perk * (this->globaln() / MPI_SIZE);
    fillVector<int>(fillsize, &recvWtAsize);

    // allocated for block implementation
    Wt_blk.zeros(this->perk, this->globalm() / MPI_SIZE);
    WtAij_blk.zeros(this->perk, this->globaln() / MPI_SIZE);
#ifdef MPI_VERBOSE
    if (ISROOT) {
      INFO << "::recvWtAsize::";
      printVector<int>(recvWtAsize);
    }
#endif
#ifndef BUILD_SPARSE
    if (this->is_compute_error()) {
      errMtx.zeros(this->m, this->n);
      A_errMtx.zeros(this->m, this->n);
    }
#endif
  }

  void freeMatrices() {
    HtH.clear();
    localHtH.clear();
    Hj.clear();
    Hjt.clear();
    AijHj.clear();
    AijHjt.clear();
    AHtij.clear();
    Wt.clear();
    WtW.clear();
    localWtW.clear();
    Wi.clear();
    Wit.clear();
    WitAij.clear();
    AijWit.clear();
    WtAij.clear();
    A_ij_t.clear();
    if (this->is_compute_error()) {
      prevH.clear();
      prevHtH.clear();
      WtAijH.clear();
      localWtAijH.clear();
    }
    Ht_blk.clear();
    AHtij_blk.clear();
    Wt_blk.clear();
    WtAij_blk.clear();
    if (this->is_compute_error()) {
      errMtx.clear();
      A_errMtx.clear();
    }
  }

 public:
  DistAUNMF(const INPUTMATTYPE &input, const MAT &leftlowrankfactor,
            const MAT &rightlowrankfactor, const MPICommunicator &communicator,
            const int numkblks)
      : DistNMF<INPUTMATTYPE>(input, leftlowrankfactor, rightlowrankfactor,
                              communicator) {
    num_k_blocks = numkblks;
    perk = this->k / num_k_blocks;
    allocateMatrices();
    this->Wt = leftlowrankfactor.t();
    this->Ht = rightlowrankfactor.t();
    A_ij_t = input.t();
    PRINTROOT("aunmf()::constructor succesful");
  }
  ~DistAUNMF() {
    // freeMatrices();
  }

  /*
   * This is a matrix multiplication routine based on
   * reduce_scatter.
   * A is mxn in column major ordering
   * W is mxk in column major ordering
   * AtW is nxk in column major ordering
   * There are totally p processes. Every process has
   * A_i as m_i * n
   * W_i as m_i * k
   * AtW_i as n_i * k
   * this->m_mpicomm.comm_subs()[0] is column communicator.
   * this->m_mpicomm.comm_subs()[1] is row communicator.
   */
  void distWtA() {
    for (int i = 0; i < num_k_blocks; i++) {
      int start_row = i * perk;
      int end_row = (i + 1) * perk - 1;
      Wt_blk = Wt.rows(start_row, end_row);
      distWtABlock();
      WtAij.rows(start_row, end_row) = WtAij_blk;
    }
  }
  void distWtABlock() {
#ifdef USE_PACOSS
    // Perform expand communication using Pacoss.
    memcpy(Wit.memptr(), Wt_blk.memptr(),
           Wt_blk.n_rows * Wt_blk.n_cols * sizeof(Wt_blk[0]));
    MPITIC;
    this->m_rowcomm->expCommBegin(Wit.memptr(), this->perk);
    this->m_rowcomm->expCommFinish(Wit.memptr(), this->perk);
#else
    int sendcnt = (this->globalm() / MPI_SIZE) * this->perk;
    int recvcnt = (this->globalm() / MPI_SIZE) * this->perk;
    Wit.zeros();
    MPITIC;  // allgather WtA
    MPI_Allgather(Wt_blk.memptr(), sendcnt, MPI_DOUBLE, Wit.memptr(), recvcnt,
                  MPI_DOUBLE, this->m_mpicomm.commSubs()[1]);
#endif
    double temp = MPITOC;  // allgather WtA
    PRINTROOT("n::" << this->n << "::k::" << this->k << PRINTMATINFO(Wt)
                    << PRINTMATINFO(Wit));
#ifdef MPI_VERBOSE
    DISTPRINTINFO(PRINTMAT(Wt_blk));
    DISTPRINTINFO(PRINTMAT(Wit));
#endif
    this->time_stats.communication_duration(temp);
    this->time_stats.allgather_duration(temp);
    MPITIC;  // mm WtA
    this->WitAij = this->Wit * this->A;
    // #if defined(MKL_FOUND) && defined(BUILD_SPARSE)
    //     // void ARMAMKLSCSCMM(const SRC &mklMat, const DESTN &Bt, const char
    //     transa,
    //     //               DESTN *Ct)
    //     ARMAMKLSCSCMM(this->A_ij_t, 'N', this->Wit, this->AijWit.memptr());
    // #ifdef MPI_VERBOSE
    //     DISTPRINTINFO(PRINTMAT(this->AijWit));
    // #endif
    //     this->WitAij = reshape(this->AijWit, this->k, this->n);
    // #else
    //     this->WitAij = this->Wit * this->A;
    // #endif
    temp = MPITOC;  // mm WtA
#ifdef MPI_VERBOSE
    DISTPRINTINFO(PRINTMAT(this->WitAij));
#endif
    PRINTROOT(PRINTMATINFO(this->A)
              << PRINTMATINFO(this->Wit) << PRINTMATINFO(this->WitAij));
    this->time_stats.compute_duration(temp);
    this->time_stats.mm_duration(temp);
    this->reportTime(temp, "WtA::");
#ifdef USE_PACOSS
    // Perform fold communication using Pacoss.
    MPITIC;
    this->m_colcomm->foldCommBegin(WitAij.memptr(), this->perk);
    this->m_colcomm->foldCommFinish(WitAij.memptr(), this->perk);
    temp = MPITOC;
    memcpy(WtAij_blk.memptr(), WitAij.memptr(),
           WtAij_blk.n_rows * WtAij_blk.n_cols * sizeof(WtAij_blk[0]));
#else
    WtAij_blk.zeros();
    MPITIC;  // reduce_scatter WtA
    MPI_Reduce_scatter(this->WitAij.memptr(), this->WtAij_blk.memptr(),
                       &(this->recvWtAsize[0]), MPI_DOUBLE, MPI_SUM,
                       this->m_mpicomm.commSubs()[0]);
    temp = MPITOC;  // reduce_scatter WtA
#endif
    this->time_stats.communication_duration(temp);
    this->time_stats.reducescatter_duration(temp);
  }
  /*
   * There are totally prxpc process.
   * Each process will hold the following
   * An A of size (m/pr) x (n/pc)
   * H of size (n/p)xk
   * find AHt kx(m/p) by reducing and scatter it using MPI_Reduce_scatter call.
   * That is, p process will hold a kx(m/p) matrix.
   * this->m_mpicomm.comm_subs()[0] is column communicator.
   * this->m_mpicomm.comm_subs()[1] is row communicator.
   * To preserve the memory for Hj, we collect only partial k
   */
  void distAH() {
    for (int i = 0; i < num_k_blocks; i++) {
      int start_row = i * perk;
      int end_row = (i + 1) * perk - 1;
      Ht_blk = Ht.rows(start_row, end_row);
      distAHBlock();
      AHtij.rows(start_row, end_row) = AHtij_blk;
    }
  }
  void distAHBlock() {
    /*
    DISTPRINTINFO("distAH::" << "::Acolst::" \
                  Acolst.n_rows<<"x"<<Acolst.n_cols \
                  << "::norm::" << arma::norm(Acolst, "fro"));
    DISTPRINTINFO("distAH::" << "::H::" \
                  << this->H.n_rows << "x" << this->H.n_cols);
    */
#ifdef USE_PACOSS
    // Perform expand communication using Pacoss.
    memcpy(Hjt.memptr(), Ht_blk.memptr(),
           Ht_blk.n_rows * Ht_blk.n_cols * sizeof(Ht_blk[0]));
    MPITIC;
    this->m_colcomm->expCommBegin(Hjt.memptr(), this->perk);
    this->m_colcomm->expCommFinish(Hjt.memptr(), this->perk);
#else
    int sendcnt = (this->globaln() / MPI_SIZE) * this->perk;
    int recvcnt = (this->globaln() / MPI_SIZE) * this->perk;
    Hjt.zeros();
    MPITIC;  // allgather AH
    MPI_Allgather(this->Ht_blk.memptr(), sendcnt, MPI_DOUBLE,
                  this->Hjt.memptr(), recvcnt, MPI_DOUBLE,
                  this->m_mpicomm.commSubs()[0]);
#endif
    PRINTROOT("n::" << this->n << "::k::" << this->k << PRINTMATINFO(Ht)
                    << PRINTMATINFO(Hjt));
    double temp = MPITOC;  // allgather AH
#ifdef MPI_VERBOSE
    DISTPRINTINFO(PRINTMAT(Ht_blk));
    DISTPRINTINFO(PRINTMAT(Hjt));
    DISTPRINTINFO(PRINTMAT(this->A_ij_t));
#endif
    this->time_stats.communication_duration(temp);
    this->time_stats.allgather_duration(temp);
    MPITIC;  // mm AH
    this->AijHjt = this->Hjt * this->A_ij_t;
    // #if defined(MKL_FOUND) && defined(BUILD_SPARSE)
    //     // void ARMAMKLSCSCMM(const SRC &mklMat, const DESTN &Bt, const char
    //     transa,
    //     //               DESTN *Ct)
    //     ARMAMKLSCSCMM(this->A, 'N', this->Hjt, this->AijHj.memptr());
    // #ifdef MPI_VERBOSE
    //     DISTPRINTINFO(PRINTMAT(this->AijHj));
    // #endif
    //     this->AijHjt = reshape(this->AijHj, this->k, this->m);
    // #else
    //     this->AijHjt = this->Hjt * this->A_ij_t;
    // #endif
    // #ifdef MPI_VERBOSE
    //     DISTPRINTINFO(PRINTMAT(this->AijHjt));
    // #endif
    temp = MPITOC;  // mm AH
    PRINTROOT(PRINTMATINFO(this->A_ij_t)
              << PRINTMATINFO(this->Hjt) << PRINTMATINFO(this->AijHjt));
    this->time_stats.compute_duration(temp);
    this->time_stats.mm_duration(temp);
    this->reportTime(temp, "AH::");
#ifdef USE_PACOSS
    // Perform fold communication using Pacoss.
    MPITIC;
    this->m_rowcomm->foldCommBegin(AijHjt.memptr(), this->perk);
    this->m_rowcomm->foldCommFinish(AijHjt.memptr(), this->perk);
    temp = MPITOC;
    memcpy(AHtij_blk.memptr(), AijHjt.memptr(),
           AHtij_blk.n_rows * AHtij_blk.n_cols * sizeof(AHtij_blk[0]));
#else
    AHtij_blk.zeros();
    MPITIC;  // reduce_scatter AH
    MPI_Reduce_scatter(this->AijHjt.memptr(), this->AHtij_blk.memptr(),
                       &(this->recvAHsize[0]), MPI_DOUBLE, MPI_SUM,
                       this->m_mpicomm.commSubs()[1]);
    temp = MPITOC;  // reduce_scatter AH
#endif
    this->time_stats.communication_duration(temp);
    this->time_stats.reducescatter_duration(temp);
  }
  /*
   * There are p processes.
   * Every process i has W in m_i * k
   * At the end of this call, all process will have
   * WtW of size k*k is symmetric. So not to worry
   * about column/row major formats.
   */
  void distInnerProduct(const MAT &X, MAT *XtX) {
    // each process computes its own kxk matrix
    MPITIC;  // gram
    localWtW = X.t() * X;
#ifdef MPI_VERBOSE
    DISTPRINTINFO("W::" << norm(X, "fro")
                        << "::localWtW::" << norm(this->localWtW, "fro"));
#endif
    double temp = MPITOC;  // gram
    this->time_stats.compute_duration(temp);
    this->time_stats.gram_duration(temp);
    (*XtX).zeros();
    if (X.n_rows == this->m) {
      this->reportTime(temp, "Gram::W::");
    } else {
      this->reportTime(temp, "Gram::H::");
    }
    MPITIC;  // allreduce gram
    MPI_Allreduce(localWtW.memptr(), (*XtX).memptr(), this->k * this->k,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    temp = MPITOC;  // allreduce gram
    this->time_stats.communication_duration(temp);
    this->time_stats.allreduce_duration(temp);
  }
  /*
   * This is the main loop function
   * Refer Algorithm 1 in Page 3 of
   * the PPoPP HPC-NMF paper.
   */
  void computeNMF() {
    PRINTROOT("computeNMF started");

#ifdef MPI_VERBOSE
    DISTPRINTINFO(PRINTMAT(this->A));
#endif
    // error computation
    if (this->is_compute_error()) {
      prevH.zeros(size(this->H));
      prevHtH.zeros(this->k, this->k);
      WtAijH.zeros(this->k, this->k);
      localWtAijH.zeros(this->k, this->k);
    }
#ifdef __WITH__BARRIER__TIMING__
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    for (int iter = 0; iter < this->num_iterations(); iter++) {
      // saving current instance for error computation.
      if (iter > 0 && this->is_compute_error()) {
        this->prevH = this->H;
        this->prevHtH = this->HtH;
      }
      MPITIC;  // total_d W&H
      // update H given WtW and WtA step 4 of the algorithm
      {
        // compute WtW
        this->distInnerProduct(this->W, &this->WtW);
        PRINTROOT(PRINTMATINFO(this->WtW));
        this->applyReg(this->regH(), &this->WtW);
#ifdef MPI_VERBOSE
        PRINTROOT(PRINTMAT(this->WtW));
#endif
        // compute WtA
        this->distWtA();
        // PRINTROOT(PRINTMATINFO(this->WtAij));
#ifdef MPI_VERBOSE
        DISTPRINTINFO(PRINTMAT(this->WtAij));
#endif
        MPITIC;  // nnls H
        // ensure both Ht and H are consistent after the update
        // some function find Ht and some H.
        updateH();
#ifdef MPI_VERBOSE
        DISTPRINTINFO("::it=" << iter << PRINTMAT(this->H));
#endif
        double temp = MPITOC;  // nnls H
        this->time_stats.compute_duration(temp);
        this->time_stats.nnls_duration(temp);
        this->reportTime(temp, "NNLS::H::");
      }
      // Update W given HtH and AH step 3 of the algorithm.
      {
        // compute HtH
        this->distInnerProduct(this->H, &this->HtH);
        PRINTROOT("HtH::" << PRINTMATINFO(this->HtH));
        this->applyReg(this->regW(), &this->HtH);
#ifdef MPI_VERBOSE
        PRINTROOT(PRINTMAT(this->HtH));
#endif
        // compute AH
        this->distAH();
        // PRINTROOT(PRINTMATINFO(this->AHtij));
#ifdef MPI_VERBOSE
        DISTPRINTINFO(PRINTMAT(this->AHtij));
#endif
        MPITIC;  // nnls W
        // Update W given HtH and AH step 3 of the algorithm.
        // ensure W and Wt are consistent. As some algorithms
        // determine W and some Wt.
        updateW();
#ifdef MPI_VERBOSE
        DISTPRINTINFO("::it=" << iter << PRINTMAT(this->W));
#endif
        double temp = MPITOC;  // nnls W
        this->time_stats.compute_duration(temp);
        this->time_stats.nnls_duration(temp);
        this->reportTime(temp, "NNLS::W::");
      }
      this->time_stats.duration(MPITOC);  // total_d W&H
      if (iter > 0 && this->is_compute_error()) {
#ifdef BUILD_SPARSE
        this->computeError(iter);
#else
        this->computeError2(iter);
#endif

        PRINTROOT("it=" << iter << "::algo::" << this->m_algorithm << "::k::"
                        << this->k << "::err::" << sqrt(this->objective_err)
                        << "::relerr::"
                        << sqrt(this->objective_err / this->m_globalsqnormA));
      }
      PRINTROOT("completed it=" << iter
                                << "::taken::" << this->time_stats.duration());
    }  // end for loop
    MPI_Barrier(MPI_COMM_WORLD);
    this->reportTime(this->time_stats.duration(), "total_d");
    this->reportTime(this->time_stats.communication_duration(), "total_comm");
    this->reportTime(this->time_stats.compute_duration(), "total_comp");
    this->reportTime(this->time_stats.allgather_duration(), "total_allgather");
    this->reportTime(this->time_stats.allreduce_duration(), "total_allreduce");
    this->reportTime(this->time_stats.reducescatter_duration(),
                     "total_reducescatter");
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

  /*
   * We assume this error function will be called in
   * every iteration before updating the block to
   * compute the error from previous iteration
   */

  void computeError(const int it) {
    // (init.norm_A)^2 - 2*trace(H*(A'*W))+trace((W'*W)*(H*H'))
    // each process owns globalsqnormA will have (init.norm_A)^2
    // each process owns WtAij is of size k*(globaln/p)
    // each process owns H is of size (globaln/p)*k
    // compute WtAij*H and do an MPI_ALL reduce to get the kxk matrix.
    // every process local computation
    MPITIC;  // computeerror
    this->localWtAijH = this->WtAij * this->prevH;
#ifdef MPI_VERBOSE
    DISTPRINTINFO("::it=" << it << PRINTMAT(this->WtAij));
    DISTPRINTINFO("::it=" << it << PRINTMAT(this->localWtAijH));
    DISTPRINTINFO("::it=" << it << PRINTMAT(this->prevH));
    PRINTROOT("::it=" << it << PRINTMAT(this->WtW));
    PRINTROOT("::it=" << it << PRINTMAT(this->prevHtH));
#endif
    double temp = MPITOC;  // computererror
    this->time_stats.err_compute_duration(temp);
    MPITIC;  // coommunication error
    MPI_Allreduce(this->localWtAijH.memptr(), this->WtAijH.memptr(),
                  this->k * this->k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    temp = MPITOC;  // communication error
#ifdef MPI_VERBOSE
    DISTPRINTINFO(PRINTMAT(WtAijH));
#endif
    this->time_stats.err_communication_duration(temp);
    double tWtAijh = trace(this->WtAijH);
    double tWtWHtH = trace(this->WtW * this->prevHtH);
    PRINTROOT("::it=" << it << "normA::" << this->m_globalsqnormA
                      << "::tWtAijH::" << 2 * tWtAijh
                      << "::tWtWHtH::" << tWtWHtH);
    this->objective_err = this->m_globalsqnormA - 2 * tWtAijh + tWtWHtH;
  }
  /*
   * Compute error the old-fashioned way
   */
  void computeError2(const int it) {
    double local_sqerror = 0.0;
    double rowcolerr = 0.0;
    PRINTROOT("::it=" << it << "::Calling compute error 2");
    MPITIC;
    // DISTPRINTINFO("::norm(Wi,fro)::" << norm(this->Wit, "fro") <<
    // "::norm(Hjt, fro)::" << norm(this->Hjt, "fro"));
    this->Wi = this->Wit.t();
    errMtx = this->Wi * this->Hjt;
    A_errMtx = this->A - errMtx;
    local_sqerror = norm(A_errMtx, "fro");
    local_sqerror *= local_sqerror;
    double temp = MPITOC;
    this->time_stats.err_compute_duration(temp);
    // DISTPRINTINFO("::it=" << it << "::local_sqerror::" << local_sqerror);
    MPITIC;
    MPI_Allreduce(&local_sqerror, &this->objective_err, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    temp = MPITOC;
    this->time_stats.err_communication_duration(temp);
  }
};
#endif  // DISTNMF_AUNMF_HPP_
