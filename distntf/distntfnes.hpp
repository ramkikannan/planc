/* Copyright Ramakrishnan Kannan 2018 */

#ifndef DISTNTF_DISTNTFNES_HPP_
#define DISTNTF_DISTNTFNES_HPP_

#include "distntf/distauntf.hpp"

namespace planc {

class DistNTFNES : public DistAUNTF {
 private:
  // Update Variables
  NCPFactors m_prox_t;  // Proximal Term (H_*)
  NCPFactors m_acc_t;   // Acceleration term (Y)
  NCPFactors m_grad_t;  // Gradient Term (\nabla_f(Y))
  NCPFactors m_prev_t;  // Previous Iterate
  MAT modified_gram;
  // Termination Variables
  double delta1;  // Termination value for absmax
  double delta2;  // Termination value for min
  // Acceleration Variables
  int acc_start;                   // Starting iteration for acceleration
  int acc_exp;                     // Step size exponent
  int acc_fails;                   // Acceleration Fails
  int fail_limit;                  // Acceleration Fail limit
  const int NUM_INNER_ITERS = 20;  // Capping inner iter
  double eig_time;
  double stop_iter_time;
  double proj_time;
  double norm_time;

 protected:
  inline double get_lambda(double L, double mu) {
    double q = L / mu;
    double lambda = 0.0;

    if (q > 1e6)
      lambda = 10 * mu;
    else if (q > 1e3)
      lambda = mu;
    else
      lambda = mu / 10;

    return lambda;
  }

  inline double get_alpha(double alpha, double q) {
    /* Solves the quadratic equation
      x^2 + (\alpha^2 -q)x - \alpha^2 = 0
    */
    double a, b, c, D, x;
    a = 1.0;
    b = alpha * alpha - q;
    c = -alpha * alpha;
    D = b * b - 4 * a * c;
    x = (-b + sqrt(D)) / 2;
    return x;
  }

  bool stop_iter(const int mode) {
    MPITIC;
    bool stop = false;
    double local_absmax, local_min, global_absmax, global_min;
    local_absmax = 0.0;
    if (m_nls_sizes[mode] > 0) {
      local_absmax =
          (arma::abs(m_grad_t.factor(mode) % m_acc_t.factor(mode))).max();
    }
    MPI_Allreduce(&local_absmax, &global_absmax, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);

    local_min = 0.0;
    if (m_nls_sizes[mode] > 0) {
      local_min = (m_grad_t.factor(mode)).min();
    }
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);

    if (global_absmax <= delta1 && global_min >= -delta2) stop = true;
    stop_iter_time += MPITOC;
    return stop;
  }

  void accelerate() {
    int iter = this->current_it();
    if (iter > acc_start) {
      int num_modes = m_prox_t.modes();
      double cur_err = this->current_error();

      double acc_step = std::pow(iter + 1, (1.0 / acc_exp));
      // Adjust all the factors
      // Reusing gradient/acceleration term to save memory
      m_grad_t.zeros();
      int lowrank = m_prox_t.rank();
      MAT scalecur = arma::eye<MAT>(lowrank, lowrank);
      MAT scaleprev = arma::eye<MAT>(lowrank, lowrank);
      for (int mode = 0; mode < num_modes; mode++) {
        if (mode == num_modes - 1) {
          scalecur = arma::diagmat(this->m_local_ncp_factors.lambda());
          scaleprev = arma::diagmat(m_prev_t.lambda());
        }
        // Step Matrix
        MAT acc_mat = m_grad_t.factor(mode);
        acc_mat = (scalecur * this->m_local_ncp_factors_t.factor(mode)) -
                  (scaleprev * m_prev_t.factor(mode));
        acc_mat *= acc_step;
        acc_mat += (scalecur * this->m_local_ncp_factors_t.factor(mode));

        if (m_nls_sizes[mode] == 0) {
          acc_mat.zeros();
        }
        m_acc_t.set(mode, acc_mat);
        m_acc_t.distributed_normalize_rows(mode);
      }
      // Compute Error
      // Always call with mode 0 to reuse MTTKRP if accepted
      double acc_err = this->computeError(m_acc_t, 0);

      // Acceleration Accepted
      if (acc_err < cur_err) {
        // Set proximal term to current term
        for (int mode = 0; mode < num_modes; mode++) {
          m_prox_t.set(mode, m_acc_t.factor(mode));
          m_prox_t.distributed_normalize_rows(mode);
        }
        PRINTROOT("Acceleration Successful::relative_error::" << acc_err);
      } else {
        // Acceleration Failed reset to prev iterate
        this->reset(m_prox_t, true);
        acc_fails++;
        if (acc_fails > fail_limit) {
          acc_fails = 0;
          acc_exp++;
        }
        PRINTROOT("Acceleration Failed::relative_error::" << acc_err);
      }
    }
  }

  MAT update(const int mode) {
    double L, mu, lambda, q, alpha, alpha_prev, beta;
    int outer_iter = this->current_it();
    int iter = 0;
    if (mode == 0) {
      m_prev_t.set_lambda(m_prox_t.lambda());
    }
    m_prev_t.set(mode, m_prox_t.factor(mode));
    MAT Ht(m_prox_t.factor(mode));
    MAT Htprev = Ht;
    m_acc_t.set(mode, Ht);
    modified_gram = this->global_gram;
    MPITIC;
    VEC eigval = arma::eig_sym(modified_gram);
    eig_time += MPITOC;
    L = eigval.max();
    mu = eigval.min();
    lambda = get_lambda(L, mu);
    modified_gram.diag() += lambda;

    q = (mu + lambda) / (L + lambda);

    MAT modified_local_mttkrp_t =
        (-1 * lambda) * m_acc_t.factor(mode) - this->ncp_local_mttkrp_t[mode];

    alpha = 1;
    alpha_prev = 1;
    beta = 1;

    while (iter < NUM_INNER_ITERS) {
      m_grad_t.set(mode, modified_local_mttkrp_t +
                             (modified_gram * m_acc_t.factor(mode)));
      if (stop_iter(mode)) break;

      iter++;
      Htprev = Ht;
      Ht = m_acc_t.factor(mode) - ((1 / (L + lambda)) * m_grad_t.factor(mode));
      MPITIC;
      fixNumericalError<MAT>(&Ht, EPSILON_1EMINUS16);
      Ht.for_each([](MAT::elem_type &val) { val = val > 0.0 ? val : 0.0; });
      proj_time += MPITOC;
      alpha_prev = alpha;
      alpha = get_alpha(alpha_prev, q);
      beta =
          (alpha_prev * (1 - alpha_prev)) / (alpha + alpha_prev * alpha_prev);
      m_acc_t.set(mode, Ht + beta * (Ht - Htprev));
    }
    if (m_nls_sizes[mode] == 0) {
      Ht.zeros();
      m_grad_t.set(mode, Ht);
    }
    PRINTROOT("Nesterov Update::mode::" << mode
                                        << "::outer_iter::" << outer_iter
                                        << "::NLS inner_iter::" << iter);
    m_prox_t.set(mode, Ht);
    MPITIC;
    m_prox_t.distributed_normalize_rows(mode);
    norm_time += MPITOC;
    return Ht;
  }

 public:
  DistNTFNES(const Tensor &i_tensor, const int i_k, algotype i_algo,
             const UVEC &i_global_dims, const UVEC &i_local_dims,
             const UVEC &i_nls_sizes, const UVEC &i_nls_idxs,
             const NTFMPICommunicator &i_mpicomm)
      : DistAUNTF(i_tensor, i_k, i_algo, i_global_dims, i_local_dims,
                  i_nls_sizes, i_nls_idxs, i_mpicomm),
        m_prox_t(i_nls_sizes, i_k, true),
        m_prev_t(i_nls_sizes, i_k, true),
        m_acc_t(i_nls_sizes, i_k, true),
        m_grad_t(i_nls_sizes, i_k, true) {
    m_prox_t.zeros();
    m_prev_t.zeros();
    m_acc_t.zeros();
    m_grad_t.zeros();
    modified_gram.zeros(i_k, i_k);
    delta1 = 1e-2;
    delta2 = 1e-2;
    acc_start = 5;
    acc_exp = 3;
    acc_fails = 0;
    fail_limit = 5;
    eig_time = 0.0;
    stop_iter_time = 0.0;
    proj_time = 0.0;
    norm_time = 0.0;
    // Set Accerated to be true
    this->accelerated(true);
  }
  ~DistNTFNES() {
    PRINTROOT("::eigen time::" << eig_time
                               << "::stop_iter_time::" << stop_iter_time
                               << "::proj_time::" << proj_time
                               << "::norm_time::" << norm_time);
  }
};  // class DistNTFNES

}  // namespace planc

#endif  // DISTNTF_DISTNTFNES_HPP_
