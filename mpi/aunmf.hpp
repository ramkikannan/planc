/* Copyright 2016 Ramakrishnan Kannan */
#ifndef MPI_AUNMF_HPP_
#define MPI_AUNMF_HPP_


#include <mpi.h>
#include <armadillo>
#include <vector>
#include <string>
#include "mpicomm.hpp"
#include "distnmf.hpp"

using namespace arma;
using namespace std;

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
  fmat HtH;       // H is of size (globaln/p)*k;
  fmat WtW;       // W is of size (globaln/p)*k;
  fmat AHtij;     // AHtij is of size k*(globalm/p)
  fmat WtAij;     // WtAij is of size k*(globaln/p)
  fmat Wt;        // Wt is of size k*(globalm/p)
  fmat Ht;        // Ht is of size k*(globaln/p)

  virtual void updateW() = 0;
  virtual void updateH() = 0;

 private:
  // Things needed while solving for W
  fmat localHtH;          // H is of size (globaln/p)*k;
  fmat Hjt, Hj;           // Hj is of size n*k;
  fmat AijHj, AijHjt;     // AijHj is of size m*k;
  INPUTMATTYPE A_ij_t;    // n*m
  // Things needed while solving for H
  fmat localWtW;          // W is of size (globalm/p)*k;
  fmat Wit, Wi;           // Wi is of size m*k;
  fmat WitAij, AijWit;    // WijtAij is of size k*n;

  // needed for error computation
  fmat prevH;     // used for error computation
  fmat prevHtH;   // used for error computation
  fmat WtAijH;  // global k*k matrix.
  fmat localWtAijH;  // local k*k matrix

  std::vector<int> recvWtAsize;
  std::vector<int> recvAHsize;


  void allocateMatrices() {
    // collective call related initializations.
    // These initialization are for solving W.
    HtH.zeros(this->k, this->k);
    localHtH.zeros(this->k, this->k);
    Hj.zeros(this->n, this->k);
    Hjt.zeros(this->k, this->n);
    AijHj.zeros(this->m, this->k);
    AijHjt.zeros(this->k, this->m);
    AHtij.zeros(this->k, this->globalm() / MPI_SIZE);
    this->recvAHsize.resize(NUMCOLPROCS);
    int fillsize = this->k * (this->globalm() / MPI_SIZE);
    fillVector<int>(fillsize, &recvAHsize);
#ifdef MPI_VERBOSE
    if (ISROOT) {
      INFO << "::recvAHsize::" ;
      printVector<int>(recvAHsize);
    }
#endif
    // These initialization are for solving H.
    Wt.zeros(this->k, this->globalm() / MPI_SIZE);
    WtW.zeros(this->k, this->k);
    localWtW.zeros(this->k, this->k);
    Wi.zeros(this->m, this->k);
    Wit.zeros(this->k, this->m);
    WitAij.zeros(this->k, this->n);
    AijWit.zeros(this->n, this->k);
    WtAij.zeros(this->k, this->globaln() / MPI_SIZE);
    this->recvWtAsize.resize(NUMROWPROCS);
    fillsize = this->k * (this->globaln() / MPI_SIZE);
    fillVector<int>(fillsize, &recvWtAsize);
#ifdef MPI_VERBOSE
    if (ISROOT) {
      INFO << "::recvWtAsize::";
      printVector<int>(recvWtAsize);
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

  }

 public:
  DistAUNMF(const INPUTMATTYPE &input, const fmat &leftlowrankfactor,
            const fmat &rightlowrankfactor,
            const MPICommunicator& communicator):
    DistNMF<INPUTMATTYPE>(input, leftlowrankfactor, rightlowrankfactor,
                          communicator) {
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
   * A is mxn is column major ordering
   * W is mxk is column major ordering
   * AtW is nxk is column major ordering
   * There are totally p processes. Every process has
   * A_i as m_i * n
   * W_i as m_i * k
   * AtW_i as n_i * k
   * this->m_mpicomm.comm_subs()[0] is column communicator.
   * this->m_mpicomm.comm_subs()[1] is row communicator.
   */

  void distWtA() {
    /*
    DISTPRINTINFO("distAtW::" << "::Arows::" \
                  << Arows.n_rows << "x" << Arows.n_cols \
                  << "::norm::" << arma::norm(Arows, "fro"));
    DISTPRINTINFO("distAtW::" << "::W::" \
                  << this->W.n_rows << "x" << this->W.n_cols);
    */
    int sendcnt = (this->globalm() / MPI_SIZE) * this->k;
    int recvcnt = (this->globalm() / MPI_SIZE) * this->k;
    Wit.zeros();
    mpitic();  // allgather WtA
    MPI_Allgather(Wt.memptr(), sendcnt, MPI_FLOAT,
                  Wit.memptr(), recvcnt, MPI_FLOAT,
                  this->m_mpicomm.commSubs()[1]);
    double temp = mpitoc();   // allgather WtA
    DISTPRINTINFO("n::" << this->n << "::k::" << this->k \
                  << PRINTMATINFO(Wt) << PRINTMATINFO(Wit));
#ifdef MPI_VERBOSE
    DISTPRINTINFO(PRINTMAT(Wt));
    DISTPRINTINFO(PRINTMAT(Wit));
#endif
    this->time_stats.communication_duration(temp);
    this->time_stats.allgather_duration(temp);
    mpitic();  // mm WtA
    this->WitAij = this->Wit * this->A;
// #if defined(MKL_FOUND) && defined(BUILD_SPARSE)
//     // void ARMAMKLSCSCMM(const SRC &mklMat, const DESTN &Bt, const char transa,
//     //               DESTN *Ct)
//     ARMAMKLSCSCMM(this->A_ij_t, 'N', this->Wit, this->AijWit.memptr());
// #ifdef MPI_VERBOSE
//     DISTPRINTINFO(PRINTMAT(this->AijWit));
// #endif
//     this->WitAij = reshape(this->AijWit, this->k, this->n);
// #else
//     this->WitAij = this->Wit * this->A;    
// #endif
    temp = mpitoc();  // mm WtA
#ifdef MPI_VERBOSE
    DISTPRINTINFO(PRINTMAT(this->WitAij));
#endif
    PRINTROOT(PRINTMATINFO(this->A) << PRINTMATINFO(this->Wit)
                  << PRINTMATINFO(this->WitAij));
    this->time_stats.compute_duration(temp);
    this->time_stats.mm_duration(temp);
    this->reportTime(temp, "WtA::");
    WtAij.zeros();
    mpitic();  // reduce_scatter WtA
    MPI_Reduce_scatter(this->WitAij.memptr(), this->WtAij.memptr(),
                       &(this->recvWtAsize[0]), MPI_FLOAT, MPI_SUM,
                       this->m_mpicomm.commSubs()[0]);;
#ifdef __WITH__BARRIER__TIMING__
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    temp = mpitoc();  // reduce_scatter WtA
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
   */
  void distAH() {
    /*
    DISTPRINTINFO("distAH::" << "::Acolst::" \
                  Acolst.n_rows<<"x"<<Acolst.n_cols \
                  << "::norm::" << arma::norm(Acolst, "fro"));
    DISTPRINTINFO("distAH::" << "::H::" \
                  << this->H.n_rows << "x" << this->H.n_cols);
    */
    int sendcnt = (this->globaln() / MPI_SIZE) * this->k;
    int recvcnt = (this->globaln() / MPI_SIZE) * this->k;
    Hjt.zeros();
    mpitic();  // allgather AH
    MPI_Allgather(this->Ht.memptr(), sendcnt, MPI_FLOAT,
                  this->Hjt.memptr(), recvcnt, MPI_FLOAT,
                  this->m_mpicomm.commSubs()[0]);
    DISTPRINTINFO("n::" << this->n << "::k::" << this->k \
                  << PRINTMATINFO(Ht) << PRINTMATINFO(Hjt));
#ifdef MPI_VERBOSE
    DISTPRINTINFO(PRINTMAT(Ht));
    DISTPRINTINFO(PRINTMAT(Hjt));
    DISTPRINTINFO(PRINTMAT(this->A_ij_t));
#endif
    double temp = mpitoc();  // allgather AH
    this->time_stats.communication_duration(temp);
    this->time_stats.allgather_duration(temp);
    mpitic();  // mm AH
    this->AijHjt = this->Hjt * this->A_ij_t;
// #if defined(MKL_FOUND) && defined(BUILD_SPARSE)
//     // void ARMAMKLSCSCMM(const SRC &mklMat, const DESTN &Bt, const char transa,
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
    temp = mpitoc();  // mm AH
    PRINTROOT(PRINTMATINFO(this->A_ij_t) << PRINTMATINFO(this->Hjt)
                  << PRINTMATINFO(this->AijHjt));
    this->time_stats.compute_duration(temp);
    this->time_stats.mm_duration(temp);
    this->reportTime(temp, "AH::");
    AHtij.zeros();
    mpitic();  // reduce_scatter AH
    MPI_Reduce_scatter(this->AijHjt.memptr(), this->AHtij.memptr(),
                       &(this->recvAHsize[0]), MPI_FLOAT, MPI_SUM,
                       this->m_mpicomm.commSubs()[1]);;
    temp = mpitoc();  // reduce_scatter AH
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
  void distInnerProduct(const fmat &X, fmat *XtX) {
    // each process computes its own kxk matrix
    mpitic();  // gram
    localWtW = X.t() * X;
    DISTPRINTINFO("W::" << norm(X, "fro") \
                  << "::localWtW::" << norm(this->localWtW, "fro"));
    double temp = mpitoc();  // gram
    this->time_stats.compute_duration(temp);
    this->time_stats.gram_duration(temp);
    (*XtX).zeros();
    if (X.n_rows == this->m) {
      this->reportTime(temp, "Gram::W::");
    } else {
      this->reportTime(temp, "Gram::H::");
    }
    mpitic();  // allreduce gram
    MPI_Allreduce(localWtW.memptr(), (*XtX).memptr(), this->k * this->k,
                  MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#ifdef __WITH__BARRIER__TIMING__
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    temp = mpitoc();  // allreduce gram
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
      // this->normalize_by_W();
      // saving current instance for error computation.
      if (iter > 0 && this->is_compute_error()) {
        this->prevH = this->H;
        this->prevHtH = this->HtH;
      }
      mpitic();  // total_d W&H
      // update H given WtW and WtA step 4 of the algorithm
      {
        // compute WtW
        this->distInnerProduct(this->W, &this->WtW);
        DISTPRINTINFO(PRINTMATINFO(this->WtW));
#ifdef MPI_VERBOSE
        PRINTROOT(PRINTMAT(this->WtW));
#endif
        // compute WtA
        this->distWtA();
        DISTPRINTINFO(PRINTMATINFO(this->WtAij));
#ifdef MPI_VERBOSE
        DISTPRINTINFO(PRINTMAT(this->WtAij));
#endif
        mpitic();  // nnls H
        // ensure both Ht and H are consistent after the update
        // some function find Ht and some H.
        updateH();
#ifdef MPI_VERBOSE
        DISTPRINTINFO("::it=" << iter << PRINTMAT(this->H));
#endif
        double temp = mpitoc();  // nnls H
        this->time_stats.compute_duration(temp);
        this->time_stats.nnls_duration(temp);
        this->reportTime(temp, "NNLS::H::");
      }
      // Update W given HtH and AH step 3 of the algorithm.
      {
        // compute HtH
        this->distInnerProduct(this->H, &this->HtH);
        DISTPRINTINFO("HtH::" << PRINTMATINFO(this->HtH));
#ifdef MPI_VERBOSE
        PRINTROOT(PRINTMAT(this->HtH));
#endif
        // compute AH
        this->distAH();
        DISTPRINTINFO(PRINTMATINFO(this->AHtij));
#ifdef MPI_VERBOSE
        DISTPRINTINFO(PRINTMAT(this->AHtij));
#endif
        mpitic();  // nnls W
        // Update W given HtH and AH step 3 of the algorithm.
        // ensure W and Wt are consistent. As some algorithms
        // determine W and some Wt.
        updateW();
#ifdef MPI_VERBOSE
        DISTPRINTINFO("::it=" << iter << PRINTMAT(this->W));
#endif
        double temp = mpitoc();  // nnls W
        this->time_stats.compute_duration(temp);
        this->time_stats.nnls_duration(temp);
        this->reportTime(temp, "NNLS::W::");
      }
      this->time_stats.duration(mpitoc());  // total_d W&H
      if (iter > 0 && this->is_compute_error()) {
        this->computeError(iter);
        PRINTROOT("it=" << iter << "::algo::" << this->m_algorithm \
                  << "::k::" << this->k \
                  << "::err::" << this->objective_err \
                  << "::relerr::" << this->objective_err / this->m_globalsqnormA);
      }
      PRINTROOT("completed it=" << iter << "::taken::" << this->time_stats.duration());
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
    mpitic();  // computeerror
    this->localWtAijH = this->WtAij * this->prevH;
#ifdef MPI_VERBOSE
    DISTPRINTINFO("::it=" << it << PRINTMAT(this->WtAij));
    DISTPRINTINFO("::it=" << it << PRINTMAT(this->localWtAijH));
    DISTPRINTINFO("::it=" << it << PRINTMAT(this->prevH));
    PRINTROOT("::it=" << it << PRINTMAT(this->WtW));
    PRINTROOT("::it=" << it << PRINTMAT(this->prevHtH));
#endif
    double temp = mpitoc();  // computererror
    this->time_stats.err_compute_duration(temp);
    mpitic();  // coommunication error
    MPI_Allreduce(this->localWtAijH.memptr(), this->WtAijH.memptr(), this->k * this->k,
                  MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    temp = mpitoc();  // communication error
#ifdef MPI_VERBOSE
    DISTPRINTINFO(PRINTMAT(WtAijH));
#endif
    this->time_stats.err_communication_duration(temp);
    double tWtAijh = trace(this->WtAijH);
    double tWtWHtH = trace(this->WtW * this->prevHtH);
    PRINTROOT("::it=" << it << "normA::" << this->m_globalsqnormA << "::tWtAijH::" << 2 * tWtAijh \
              << "::tWtWHtH::" << tWtWHtH);
    this->objective_err = this->m_globalsqnormA - 2 * tWtAijh + tWtWHtH;
  }

  /*void computeError(const int it) {
    int sendcnt = (this->globalm() / MPI_SIZE) * this->k;
    int recvcnt = (this->globalm() / MPI_SIZE) * this->k;
    this->Wit.zeros();
    this->Hjt.zeros();
    mpitic();
    MPI_Allgather(Wt.memptr(), sendcnt, MPI_FLOAT,
                  Wit.memptr(), recvcnt, MPI_FLOAT,
                  this->m_mpicomm.commSubs()[1]);
    sendcnt = (this->globaln() / MPI_SIZE) * this->k;
    recvcnt = (this->globaln() / MPI_SIZE) * this->k;
    MPI_Allgather(this->Ht.memptr(), sendcnt, MPI_FLOAT,
                  this->Hjt.memptr(), recvcnt, MPI_FLOAT,
                  this->m_mpicomm.commSubs()[0]);
    DISTPRINTINFO("n::" << this->n << "::k::" << this->k \
                  << PRINTMATINFO(Ht) << PRINTMATINFO(Hjt));
  #ifdef MPI_VERBOSE
    DISTPRINTINFO(PRINTMAT(Ht));
    DISTPRINTINFO(PRINTMAT(Hjt));
    DISTPRINTINFO(PRINTMAT(this->A_ij_t));
  #endif
    double temp = mpitoc();
    this->time_stats.err_communication_duration(temp);
    mpitic();
    double localerr = norm((this->A - trans(this->Wit) * this->Hjt), "fro");
    localerr *= localerr;
    temp = mpitoc();
    this->time_stats.err_compute_duration(temp);
    MPI_Allreduce(&localerr, &(this->objective_err), 1, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
  }*/
};
#endif  // MPI_AUNMF_HPP_
