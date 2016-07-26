/* Copyright 2016 Ramakrishnan Kannan */
#ifndef MPI_DISTNMF_HPP_
#define MPI_DISTNMF_HPP_

#include <string>
#include "mpicomm.hpp"
#include "nmf.hpp"
#include "distnmftime.hpp"


/*
 * There are totally prxpc process.
 * Each process will hold the following
 *   An A of size (m/pr) x (n/pc)
 * H of size (n/p)xk
 * W of size (m/p)xk
 * A is mxn matrix
 * H is nxk matrix
 */


template <typename INPUTMATTYPE>
class DistNMF : public NMF<INPUTMATTYPE> {
 protected:
  const MPICommunicator& m_mpicomm;
  uword m_globalm;
  uword m_globaln;
  double m_globalsqnormA;
  DistNMFTime time_stats;
  uint m_compute_error;
  distalgotype m_algorithm;
  frowvec localWnorm;
  frowvec Wnorm;

 public:
  DistNMF(const INPUTMATTYPE &input, const fmat &leftlowrankfactor,
          const fmat &rightlowrankfactor, const MPICommunicator& communicator):
    NMF<INPUTMATTYPE>(input, leftlowrankfactor, rightlowrankfactor),
    m_mpicomm(communicator), time_stats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) {
    double sqnorma = this->normA * this->normA;
    this->m_globalm = 0;
    this->m_globaln = 0;
    MPI_Allreduce(&sqnorma, &(this->m_globalsqnormA), 1, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&(this->m), &(this->m_globalm), 1, MPI_INT, MPI_SUM,
                  this->m_mpicomm.commSubs()[0]);
    MPI_Allreduce(&(this->n), &(this->m_globaln), 1, MPI_INT, MPI_SUM,
                  this->m_mpicomm.commSubs()[1]);
    if (ISROOT) {
      INFO << "globalsqnorma::" << this->m_globalsqnormA
           << "::globalm::" << this->m_globalm
           << "::globaln::" << this->m_globaln << endl;
    }
    this->m_compute_error = 0;
    localWnorm.zeros(this->k);
    Wnorm.zeros(this->k);
  }

  const int globalm() const {return m_globalm;}
  const int globaln() const {return m_globaln;}
  const double globalsqnorma() const {return m_globalsqnormA;}
  void compute_error(const uint &ce) {this->m_compute_error = ce;}
  const bool is_compute_error() const {return (this->m_compute_error);}
  void algorithm(distalgotype dat) {this->m_algorithm = dat;}
  void reportTime(const double temp, const std::string &reportstring) {
    double mintemp, maxtemp, sumtemp;
    MPI_Allreduce(&temp, &maxtemp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&temp, &mintemp, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&temp, &sumtemp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    PRINTROOT(reportstring \
              << "::m::" << this->m_globalm << "::n::" << this->m_globaln \
              << "::k::" << this->k << "::SIZE::" << MPI_SIZE \
              << "::algo::" << this->m_algorithm \
              << "::root::" << temp \
              << "::min::" << mintemp \
              << "::avg::" << (sumtemp) / (MPI_SIZE) \
              << "::max::" << maxtemp);
  }
  void normalize_by_W() {
    localWnorm = sum(this->W % this->W);
    mpitic();
    MPI_Allreduce(localWnorm.memptr(), Wnorm.memptr(), this->k, MPI_FLOAT,
                  MPI_SUM, MPI_COMM_WORLD);
    double temp = mpitoc();
    this->time_stats.allgather_duration(temp);
    for (int i = 0; i < this->k; i++) {
      if (Wnorm(i) > 1) {
        float norm_const = sqrt(Wnorm(i));
        this->W.col(i) = this->W.col(i) / norm_const;
        this->H.col(i) = norm_const * this->H.col(i);
      }
    }
  }
};

#endif  // MPI_DISTNMF_HPP_
