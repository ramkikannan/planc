/* Copyright 2016 Ramakrishnan Kannan */

#ifndef NMF_GNSYM_HPP_
#define NMF_GNSYM_HPP_

#include "common/nmf.hpp"

namespace planc {

template <class T>
class GNSYMNMF : public NMF<T> {
 private:
  // Not happy with this design. However to avoid computing At again and again
  // making this as private variable.
  MAT HtH;
  MAT AHt;

  // CG Matrices
  MAT Dt;           // k*(globaln/p) used as direction vector (same size as H)
  MAT grad;         // k*(globaln/p) used to store transposed
                    // matmul and gradient (same size as H)
  MAT Pk;           // k*(globaln/p) used to store conjugate direction
  MAT Y;            // k*(globaln/p) used to store the application of Hessian
  MAT XtX;          // k*k temporary to store gradient-search products

  // CG hyperparamters
  unsigned int cg_max_iters;
  bool stale_matmul;
  bool stale_gram;
  double cg_tol;
  double alpha;
  double beta;

  // counter variables
  int cg_nongrams;
  int cg_grams;
  int cg_dotprods;

  int grad_nongrams;
  int grad_grams;
  int grad_matmuls;

  int err_matmuls;
  int err_grams;

  int cg_tot_iters;

  // error variables
  double sqnormA;

  /*
   * Collected statistics are
   * iteration Htime Wtime totaltime normH normW densityH densityW relError
   */
  void allocateMatrices() {
    HtH = arma::zeros<MAT>(this->k, this->k);
    AHt = arma::zeros<MAT>(this->k, this->m);

    Pk  = arma::zeros<MAT>(this->k, this->m);
    Y   = arma::zeros<MAT>(this->k, this->m);
    Dt  = arma::zeros<MAT>(this->k, this->m);
    XtX = arma::zeros<MAT>(this->k, this->m);
  }
  void freeMatrices() {
    HtH.clear();
    AHt.clear();
    Pk.clear();
    Y.clear();
    Dt.clear();
  }

 public:
  GNSYMNMF(const T &A, int lowrank) : NMF<T>(A, lowrank) {
    this->W.clear();  //  clear W variable
    allocateMatrices();
    sqnormA = this->normA * this->normA;

    // TODO{seswar3} : Make hyperparamters user inputs
    alpha = 0.0;
    beta = 0.0;
    cg_tol = 0.0001;
    cg_max_iters = this->k;
    stale_matmul = true;
    stale_gram = true;

    // Counter variables
    cg_nongrams = 0;
    cg_grams = 0;
    cg_dotprods = 0;

    grad_nongrams = 0;
    grad_grams = 0;
    grad_matmuls = 0;

    err_matmuls = 0;
    err_grams = 0;

    cg_tot_iters = 0;
  }

  GNSYMNMF(const T &A, const MAT &llf, const MAT &rlf) : NMF<T>(A, llf, rlf) {
    this->W.clear();  //  clear W variable
    allocateMatrices();
    sqnormA = this->normA * this->normA;

    // TODO{seswar3} : Make hyperparamters user inputs
    alpha = 0.0;
    beta = 0.0;
    cg_tol = 0.0001;
    cg_max_iters = this->k;
    stale_matmul = true;
    stale_gram = true;

    // Counter variables
    cg_nongrams = 0;
    cg_grams = 0;
    cg_dotprods = 0;

    grad_nongrams = 0;
    grad_grams = 0;
    grad_matmuls = 0;

    err_matmuls = 0;
    err_grams = 0;

    cg_tot_iters = 0;
  }

  void applyHess() {
    // X is the gradient (reshaped to n/p x k)
    // Y1 = X (H^T H)
    if (stale_gram) {
      HtH = this->H.t() * this->H;
      stale_gram = false;
      cg_grams++;
    }

    // Y2 = H (X^T H)
    XtX = Pk * this->H;
    cg_dotprods++;

    // Y = 2(Y1 + Y2)
    Y = 2*((HtH * Pk) + (XtX * this->H.t()));
    cg_nongrams++;
  }

  void computeObjectiveError() {
    // Fast formula using,
    // \norm{A}_F^2 - 2Tr(AHH^T) + Tr(H^TH * H^TH)

    if (stale_gram) {
      HtH = this->H.t() * this->H;
      stale_gram = false;
      err_grams++;
    }
    if (stale_matmul) {
      AHt = this->H.t() * this->A;
      stale_matmul = false;
      err_matmuls++;
    }

    double tAHHt = arma::trace(AHt * this->H);
    double tHtHHtH = arma::trace(HtH * HtH);
    double raw_err = this->sqnormA - (2*tAHHt) + tHtHHtH;
    this->objective_err = (raw_err > 0)? raw_err : 0;
  }

  void computeNMF() {
    unsigned int currentIteration = 0;
    unsigned int cgiter = 0;
    while (currentIteration < this->num_iterations()) {
      tic();

      // Compute Gradient
      if (stale_gram) {
        HtH = this->H.t() * this->H;
        stale_gram = false;
        grad_grams++;
      }
      if (stale_matmul) {
        AHt = this->H.t() * this->A;
        stale_matmul = false;
        grad_matmuls++;
      }
      grad = -2*(AHt - (HtH * this->H.t()));
      grad_nongrams++;

      // Conjugate Gradient phase
      // CG initialization
      // using Pk for p_k and grad for r_k
      // r_0 = b - Ax_0 (r is the gradient)
      // p_0 = r_0
      Pk = grad;
      Dt.zeros();
      double rsold = arma::accu(grad % grad);

      INFO << "it=" << currentIteration
           << "::CG intial residual::" << rsold << std::endl;

      // Enter CG iterations only if residual is large
      if (rsold > cg_tol) {
        // for k = 1,2,...
        for (cgiter = 0; cgiter < this->cg_max_iters; cgiter++) {
          // compute A p_k stored in Y
          this->applyHess();

          // \alpha_k = r_k^T r_k / p_k^T A p_k
          alpha = rsold / arma::accu(Pk % Y);

          // x_{k+1} = x_k + \alpha_k p_k
          Dt = Dt + (alpha * Pk);

          // r_{k+1} = r_k - \alpha_k A p_k
          grad = grad - (alpha * Y);

          // beta_k = r_{k+1}^T r_{k+1} / r_k^T r_k
          double rsnew = arma::accu(grad % grad);

          INFO << "it=" << currentIteration << "::CG iter::" << cgiter
               << "::CG residual::" << rsnew << std::endl;

          // Stopping criteria
          if (rsnew < cg_tol)
            break;

          beta = rsnew / rsold;
          rsold = rsnew;

          // p_{k+1} = r_{k+1} + \beta_k p_k
          Pk = grad + (beta * Pk);

          cg_tot_iters++;
        }  // end for
      }

      // Project the solution
      this->H = this->H - Dt.t();
      this->H.for_each(
          [](MAT::elem_type &val) { val = val > 0.0 ? val : 0.0; });
      stale_matmul = true;
      stale_gram = true;

      INFO << "Completed It (" << currentIteration << "/"
           << this->num_iterations() << ")"
           << " time =" << toc() << std::endl;
      this->computeObjectiveError();
      INFO << "Completed it = " << currentIteration
           << " GNSYMERR=" << sqrt(this->objective_err) / this->normA
           << std::endl;
      currentIteration++;
    }
  }
  ~GNSYMNMF() {}
};

}  // namespace planc

#endif  // NMF_GNSYM_HPP_
