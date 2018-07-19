/* Copyright Ramakrishnan Kannan 2017 */
#ifndef COMMON_LUC_HPP_
#define COMMON_LUC_HPP_

#ifdef MPI_DISTNTF
#include <mpi.h>
#endif
#include "common/ncpfactors.hpp"
#include "nnls/bppnnls.hpp"
#include "nnls/nnls.hpp"

namespace planc {
class LUC {
 private:
  // create variable required for your local function.
  NCPFactors *factors;
  int i_k;
  const algotype m_updalgo;

  // ADMM auxiliary variables
  NCPFactors *m_local_ncp_aux;
  NCPFactors *m_local_ncp_aux_t;
  NCPFactors *m_temp_local_ncp_aux_t;
  MAT L;
  MAT Lt;
  MAT tempgram;

  int admm_iter;
  double tolerance;

  MAT updateMU(const MAT &gram, const MAT &rhs, const int mode) {
    MAT H(factors->factor(mode));
    MAT temp = H * gram + EPSILON;
    H = (H % rhs.t()) / temp;
    return H.t();
  }

  MAT updateHALS(const MAT &gram, const MAT &rhs, const int mode) {
    MAT H(factors->factor(mode));

    // iterate over all columns of H
    for (int i = 0; i < this->i_k; i++) {
      VEC updHi = H.col(i) + ((rhs.row(i)).t() - H * gram.col(i));
      fixNumericalError<VEC>(&updHi);
      double normHi = arma::norm(updHi, 2);
      normHi *= normHi;
      double globalnormHi = normHi;
#ifdef MPI_DISTNTF
      MPI_Allreduce(&normHi, &globalnormHi, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
#endif  // shared memory can just check normHi
      if (globalnormHi > 0) {
        H.col(i) = updHi;
      }
    }
    return H.t();
  }

  MAT updateBPP(const MAT &gram, const MAT &rhs) {
    BPPNNLS<MAT, VEC> subProblem(gram, rhs, true);
    subProblem.solveNNLS();
    return subProblem.getSolutionMatrix();
  }

  MAT updateAOADMM(const MAT &gram, const MAT &rhs, const int mode) {
    // return variable
    MAT updated_fac(factors->factor(mode));
    MAT prev_fac = updated_fac;

    // Set up ADMM iteration
    double alpha = arma::trace(gram) / i_k;
    alpha = (alpha > 0) ? alpha : 0.01;
    tempgram = gram;
    tempgram.diag() += alpha;
    L = arma::chol(tempgram, "lower");
    Lt = L.t();
    bool stop_iter = false;

    m_local_ncp_aux->set(mode, updated_fac);

    // Start ADMM loop from here
    for (int i = 0; i < admm_iter && !stop_iter; i++) {
      prev_fac = updated_fac;
      m_local_ncp_aux->set(mode, updated_fac);
      m_local_ncp_aux_t->set(mode, m_local_ncp_aux->factor(mode).t());

      m_temp_local_ncp_aux_t->set(
          mode, arma::solve(arma::trimatl(L),
                            rhs + (alpha * (updated_fac.t() +
                                            m_local_ncp_aux_t->factor(mode)))));
      m_local_ncp_aux_t->set(
          mode,
          arma::solve(arma::trimatu(Lt), m_temp_local_ncp_aux_t->factor(mode)));

      // Update factor matrix
      updated_fac = m_local_ncp_aux_t->factor(mode).t();
      fixNumericalError<MAT>(&(updated_fac), EPSILON_1EMINUS16);
      updated_fac = updated_fac - m_local_ncp_aux->factor(mode);
      updated_fac.for_each(
          [](MAT::elem_type &val) { val = val > 0.0 ? val : 0.0; });

      // Update dual variable
      m_local_ncp_aux->set(mode, m_local_ncp_aux->factor(mode) + updated_fac -
                                     m_local_ncp_aux_t->factor(mode).t());

      // factor norm
      double local_facnorm = arma::norm(updated_fac, "fro");
      local_facnorm *= local_facnorm;

      double global_facnorm = 0.0;
#ifdef MPI_DISTNTF
      MPI_Allreduce(&local_facnorm, &global_facnorm, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      global_facnorm = sqrt(global_facnorm);
#else
      global_facnorm = sqrt(local_facnorm);
#endif
      // dual norm
      double local_dualnorm = arma::norm(updated_fac, "fro");
      local_dualnorm *= local_dualnorm;

      double global_dualnorm = 0.0;
#ifdef MPI_DISTNTF
      MPI_Allreduce(&local_dualnorm, &global_dualnorm, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      global_dualnorm = sqrt(global_dualnorm);
#else
      global_dualnorm = sqrt(local_dualnorm);
#endif
      // Check stopping criteria (needs communication)
      double r = norm(updated_fac - m_local_ncp_aux->factor(mode), "fro");
      r *= r;
      double global_r = 0.0;
#ifdef MPI_DISTNTF
      MPI_Allreduce(&r, &global_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      global_r = sqrt(global_r);
#else
      global_r = sqrt(r);
#endif
      double s = norm(updated_fac - prev_fac, "fro");
      s *= s;
      double global_s = 0.0;
#ifdef MPI_DISTNTF
      MPI_Allreduce(&s, &global_s, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      global_s = sqrt(global_s);
#else
      global_s = sqrt(s);
#endif
      if (global_r < (tolerance * global_facnorm) &&
          global_s < (tolerance * global_dualnorm))
        stop_iter = true;
    }
#ifdef MPI_DISTNTF
    m_local_ncp_aux->distributed_normalize(mode);
#else
    m_local_ncp_aux->normalize(mode);
#endif
    return updated_fac.t();
  }

 public:
  // State creating constructor
  LUC(algotype in_updalgo, NCPFactors *localfac, const UVEC &i_local_dims,
      const int in_k)
      : m_updalgo(in_updalgo) {
    this->factors = localfac;  // pointer to local factors
    this->i_k = in_k;
    switch (m_updalgo) {
      case AOADMM:
        m_local_ncp_aux = new NCPFactors(i_local_dims, in_k, false);
        m_local_ncp_aux->zeros();
        m_local_ncp_aux_t = new NCPFactors(i_local_dims, in_k, true);
        m_local_ncp_aux_t->zeros();
        m_temp_local_ncp_aux_t = new NCPFactors(i_local_dims, in_k, true);
        m_temp_local_ncp_aux_t->zeros();
        L.zeros(in_k, in_k);
        Lt.zeros(in_k, in_k);
        tempgram.zeros(in_k, in_k);
        admm_iter = 5;
        tolerance = 0.01;
        break;
    }
  }

  // Delete the auxiliary variables
  ~LUC() {
    switch (m_updalgo) {
      case AOADMM:
        delete m_local_ncp_aux;
        delete m_local_ncp_aux_t;
        delete m_temp_local_ncp_aux_t;
    }
  }

  MAT update(algotype m_updalgo, const MAT &gram, const MAT &rhs,
             const int mode) {
    MAT rc;
    switch (m_updalgo) {
      case ANLSBPP:
        rc = updateBPP(gram, rhs);
        break;
      case AOADMM:
        rc = updateAOADMM(gram, rhs, mode);
        break;
      case HALS:
        rc = updateHALS(gram, rhs, mode);
        break;
      case MU:
        rc = updateMU(gram, rhs, mode);
        break;
      default:
        std::cout << "wrong algo type" << std::endl;
    }
    return rc;
  }
};  // LUC

}  // namespace planc
#endif  // COMMON_LUC_HPP_
