/* Copyright 2016 Ramakrishnan Kannan */

#ifndef DISTNMF_DISTNMF1D_HPP_
#define DISTNMF_DISTNMF1D_HPP_

#include <string>
#include "common/distutils.hpp"
#include "common/utils.h"
#include "common/utils.hpp"

namespace planc {

template <class INPUTMATTYPE>
class DistNMF1D {
 protected:
  const MPICommunicator &m_mpicomm;
  INPUTMATTYPE m_Arows;
  INPUTMATTYPE m_Acols;
  UWORD m_globalm, m_globaln;
  MAT m_W, m_H;
  MAT m_Wt, m_Ht;
  MAT m_globalW, m_globalH;
  MAT m_globalWt, m_globalHt;
  double m_objective_err;
  double m_globalsqnormA;
  unsigned int m_num_iterations;
  unsigned int m_k;  // low rank k
  DistNMFTime time_stats;
  MAT m_prevH;    // this is needed for error computation
  MAT m_prevHtH;  // this is needed for error computation
  uint m_compute_error;
  algotype m_algorithm;

 private:
  MAT HAtW;        // needed for error computation
  MAT globalHAtW;  // needed for error computation
  MAT err_matrix;  // needed for error computation.

 public:
  DistNMF1D(const INPUTMATTYPE &Arows, const INPUTMATTYPE &Acols,
            const MAT &leftlowrankfactor, const MAT &rightlowrankfactor,
            const MPICommunicator &mpicomm)
      : m_mpicomm(mpicomm),
        m_Arows(Arows),
        m_Acols(Acols),
        m_W(leftlowrankfactor),
        m_H(rightlowrankfactor),
        time_stats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) {
    this->m_globalm = Arows.n_rows * MPI_SIZE;
    this->m_globaln = Arows.n_cols;
    this->m_Wt = this->m_W.t();
    this->m_Ht = this->m_H.t();
    this->m_k = this->m_W.n_cols;
    this->m_num_iterations = 20;
    m_globalW.zeros(this->m_globalm, this->m_k);
    err_matrix.zeros(this->m_globalm, this->m_k);
    m_globalWt.zeros(this->m_k, this->m_globalm);
    m_globalH.zeros(this->m_globaln, this->m_k);
    m_globalHt.zeros(this->m_k, this->m_globaln);
    HAtW.zeros(this->m_k, this->m_k);
    globalHAtW.zeros(this->m_k, this->m_k);
    double localsqnormA = norm(Arows, "fro");
    localsqnormA = localsqnormA * localsqnormA;
    MPI_Allreduce(&localsqnormA, &(this->m_globalsqnormA), 1, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
    DISTPRINTINFO("DistNMF1D constructor completed"
                  << "::globalm::" << this->m_globalm
                  << "::globaln::" << this->m_globaln);
  }
  /*
   * Return the communication time
   */
  double globalW() {
    int sendcnt = this->m_W.n_rows * this->m_W.n_cols;
    int recvcnt = this->m_W.n_rows * this->m_W.n_cols;
    this->m_Wt = this->m_W.t();
    mpitic();
    MPI_Allgather(this->m_Wt.memptr(), sendcnt, MPI_DOUBLE,
                  this->m_globalWt.memptr(), recvcnt, MPI_DOUBLE,
                  MPI_COMM_WORLD);
    /*MPI_Gather(this->m_Wt.memptr(), sendcnt, MPI_DOUBLE,
                  this->m_globalWt.memptr(), recvcnt, MPI_DOUBLE,
                  0, MPI_COMM_WORLD);
    sendcnt = this->m_globalWt.n_rows * this->m_globalWt.n_cols;
    MPI_Bcast(this->m_globalWt.memptr(), sendcnt, MPI_DOUBLE, 0,
    MPI_COMM_WORLD);*/
    double commTime = mpitoc();
    DISTPRINTINFO(PRINTMATINFO(this->m_Wt) << PRINTMATINFO(this->m_globalWt));
    this->m_globalW = this->m_globalWt.t();
    return commTime;
  }
  /*
   * Return the communication time
   */
  double globalH() {
    int sendcnt = this->m_H.n_rows * this->m_H.n_cols;
    int recvcnt = this->m_H.n_rows * this->m_H.n_cols;
    this->m_Ht = this->m_H.t();
    mpitic();
    MPI_Allgather(this->m_Ht.memptr(), sendcnt, MPI_DOUBLE,
                  this->m_globalHt.memptr(), recvcnt, MPI_DOUBLE,
                  MPI_COMM_WORLD);
    /*MPI_Gather(this->m_Ht.memptr(), sendcnt, MPI_DOUBLE,
                  this->m_globalHt.memptr(), recvcnt, MPI_DOUBLE,
                  0, MPI_COMM_WORLD);
    sendcnt = this->m_globalHt.n_rows * this->m_globalHt.n_cols;
    MPI_Bcast(this->m_globalHt.memptr(), sendcnt, MPI_DOUBLE, 0,
    MPI_COMM_WORLD);*/
    double commTime = mpitoc();
    this->m_globalH = this->m_globalHt.t();
    DISTPRINTINFO(PRINTMATINFO(this->m_Ht) << PRINTMATINFO(this->m_globalHt));
    return commTime;
  }
  /*
   * Assuming you have the latest globalW and globalH.
   * If other wise, call globalW and globalH before calling
   * this function.
   * (init.norm_A)^2 - 2*trace(H*(A'*W))+trace((W'*W)*(H*H'))
   * each process owns globalsqnormA will have (init.norm_A)^2
   *
   */
  void computeError(const MAT &WtW, const MAT &HtH) {
    mpitic();
    if (this->m_Acols.n_rows == this->m_globalm) {
      HAtW = this->m_prevH.t() * (this->m_Acols.t() * this->m_globalW);
    } else {
      // we assume m_Acols would have been transposed
      // by the derived classes.
      HAtW = this->m_prevH.t() * (this->m_Acols * this->m_globalW);
    }
    double temp = mpitoc();
    this->time_stats.err_compute_duration(temp);
    mpitic();
    MPI_Allreduce(HAtW.memptr(), globalHAtW.memptr(), this->m_k * this->m_k,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    temp = mpitoc();
    this->time_stats.err_communication_duration(temp);
    mpitic();
    double tHAtW = trace(globalHAtW);
    double tWtWHtH = trace(WtW * HtH);
    PRINTROOT("normA::" << this->m_globalsqnormA << "::tHAtW::" << 2 * tHAtW
                        << "::tWtWHtH::" << tWtWHtH);
    this->m_objective_err = this->m_globalsqnormA - 2 * tHAtW + tWtWHtH;
    mpitoc();
    this->time_stats.err_compute_duration(temp);
  }
  /*void computeError(const int it) {
    mpitic();
    err_matrix = this->m_globalW * this->m_prevH.t();
    err_matrix = this->m_Acols - err_matrix;
    PRINTROOT(PRINTMATINFO(this->m_globalW));
    PRINTROOT(PRINTMATINFO(this->m_prevH));
    PRINTROOT(PRINTMATINFO(err_matrix));
    double error = norm(err_matrix, "fro");
    error *= error;
    double temp = mpitoc();
    this->time_stats.err_compute_duration(temp);
    mpitic();
    MPI_Allreduce(&error, &(this->m_objective_err), 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    temp = mpitoc();
    this->time_stats.err_communication_duration(temp);
  }*/

  virtual void computeNMF() = 0;
  const unsigned int num_iterations() const { return this->m_num_iterations; }
  void num_iterations(int it) { m_num_iterations = it; }
  const UWORD globalm() const { return m_globalm; }
  const UWORD globaln() const { return m_globaln; }
  MAT getLeftLowRankFactor() { return this->m_W; }
  MAT getRightLowRankFactor() { return this->m_H; }
  void compute_error(const uint &ce) { this->m_compute_error = ce; }
  const bool is_compute_error() const { return (this->m_compute_error); }
  void algorithm(algotype dat) { this->m_algorithm = dat; }
  void reportTime(const double temp, const std::string &reportstring) {
    double mintemp, maxtemp, sumtemp;
    MPI_Allreduce(&temp, &maxtemp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&temp, &mintemp, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&temp, &sumtemp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    PRINTROOT(reportstring << "::m::" << this->m_globalm
                           << "::n::" << this->m_globaln << "::k::" << this->m_k
                           << "::SIZE::" << MPI_SIZE
                           << "::algo::" << this->m_algorithm
                           << "::root::" << temp << "::min::" << mintemp
                           << "::avg::" << (sumtemp) / (MPI_SIZE)
                           << "::max::" << maxtemp);
  }
};

}  // namespace planc

#endif  // DISTNMF_DISTNMF1D_HPP_
