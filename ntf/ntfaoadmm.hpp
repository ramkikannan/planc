/* Copyright Ramakrishnan Kannan 2018 */

#ifndef NTF_NTFAOADMM_HPP_
#define NTF_NTFAOADMM_HPP_

#include "ntf/auntf.hpp"

namespace planc {

class NTFAOADMM : public AUNTF {
 private:
  // ADMM auxiliary variables
  NCPFactors *m_ncp_aux;
  NCPFactors *m_ncp_aux_t;
  NCPFactors *m_temp_ncp_aux_t;
  MAT L;
  MAT Lt;
  MAT tempgram;
  int admm_iter;
  double tolerance;

 protected:
  MAT update(const int mode) {
    // return variable
    MAT updated_fac(this->m_ncp_factors.factor(mode));
    MAT prev_fac = updated_fac;

    // Set up ADMM iteration
    double alpha =
        arma::trace(this->gram_without_one) / this->m_ncp_factors.rank();
    alpha = (alpha > 0) ? alpha : 0.01;
    tempgram = this->gram_without_one;
    tempgram.diag() += alpha;
    L = arma::chol(tempgram, "lower");
    Lt = L.t();
    bool stop_iter = false;

    m_ncp_aux->set(mode, updated_fac);

    // Start ADMM loop from here
    for (int i = 0; i < admm_iter && !stop_iter; i++) {
      prev_fac = updated_fac;
      m_ncp_aux->set(mode, updated_fac);
      m_ncp_aux_t->set(mode, m_ncp_aux->factor(mode).t());

      m_temp_ncp_aux_t->set(
          mode, arma::solve(arma::trimatl(L),
                            this->ncp_mttkrp_t[mode] +
                                (alpha * (updated_fac.t() +
                                          m_ncp_aux_t->factor(mode)))));
      m_ncp_aux_t->set(
          mode, arma::solve(arma::trimatu(Lt), m_temp_ncp_aux_t->factor(mode)));

      // Update factor matrix
      updated_fac = m_ncp_aux_t->factor(mode).t();
      fixNumericalError<MAT>(&(updated_fac), EPSILON_1EMINUS16);
      updated_fac = updated_fac - m_ncp_aux->factor(mode);
      updated_fac.for_each(
          [](MAT::elem_type &val) { val = val > 0.0 ? val : 0.0; });

      // Update dual variable
      m_ncp_aux->set(mode, m_ncp_aux->factor(mode) + updated_fac -
                               m_ncp_aux_t->factor(mode).t());

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
      global_dualnorm = sqrt(local_dualnorm);
      // Check stopping criteria (needs communication)
      double r = norm(updated_fac - m_ncp_aux->factor(mode), "fro");
      r *= r;
      double global_r = 0.0;
      global_r = sqrt(r);
      double s = norm(updated_fac - prev_fac, "fro");
      s *= s;
      double global_s = 0.0;
      global_s = sqrt(s);
      if (global_r < (tolerance * global_facnorm) &&
          global_s < (tolerance * global_dualnorm))
        stop_iter = true;
    }
    m_ncp_aux->normalize(mode);
    return updated_fac.t();
  }

 public:
  NTFAOADMM(const Tensor &i_tensor, const int i_k, algotype i_algo)
      : AUNTF(i_tensor, i_k, i_algo) {
    m_ncp_aux = new NCPFactors(i_tensor.dimensions(), i_k, false);
    m_ncp_aux->zeros();
    m_ncp_aux_t = new NCPFactors(i_tensor.dimensions(), i_k, true);
    m_ncp_aux_t->zeros();
    m_temp_ncp_aux_t = new NCPFactors(i_tensor.dimensions(), i_k, true);
    m_temp_ncp_aux_t->zeros();
    L.zeros(i_k, i_k);
    Lt.zeros(i_k, i_k);
    tempgram.zeros(i_k, i_k);
    admm_iter = 5;
    tolerance = 0.01;
  }
};  // class NTFAOADMM

}  // namespace planc

#endif  // NTF_NTFAOADMM_HPP_
