/* Copyright 2020 Srinivas Eswar */

#ifndef DISTNMF_DISTGNSYMNMF_HPP_
#define DISTNMF_DISTGNSYMNMF_HPP_

/**
 * Provides the update algorithm for the
 * distributed Gauss-Newton algorithm for SymNMF (SymGNCG).
 * 
 * The algorithm is implemented according to the details
 * found in the paper: 
 * "Distribued-Memory Parallel Symmetric Nonnegative Matrix
 *  Factorization" from SC 2020.
 */

namespace planc {

template <class INPUTMATTYPE>
class DistGNSym : public DistAUNMF<INPUTMATTYPE> {
 private:
  // CG hyperparamters
  unsigned int cg_max_iters;
  bool stale_matmul;
  bool stale_gram;
  double cg_tol;
  double alpha;
  double beta;
  int paired_proc;

  // counter variables
  int cg_nongrams;
  int cg_grams;
  int cg_dotprods;

  int grad_nongrams;
  int grad_grams;

  int err_matmuls;
  int err_grams;

  int cg_tot_iters;

  // Local Matrices
  MAT XY;           // k*k matrix needed for error calc
  MAT Dt;           // k*(globaln/p) used as direction vector (same size as H)
  MAT localHtAijH;  // k*k matrix needed for error calc
  MAT HtAijH;       // k*k matrix needed for error calc
  MAT grad;         // k*(globaln/p) used to store transposed
                    // matmul and gradient (same size as H)

 public:
  /**
   * Public constructor with local input matrix, local factors and communicator
   * @param[in] local input matrix of size \f$\frac{globalm}{p_r} \times \frac{globaln}{p_c}\f$.
   *            Each process owns \f$m=\frac{globalm}{p_r}\f$ and \f$n=\frac{globaln}{p_c}\f$
   * @param[in] local left low rank factor of size \f$\frac{globalm}{p} \times k \f$
   * @param[in] local right low rank factor of size \f$\frac{globaln}{p} \times k \f$
   * @param[in] MPICommunicator that has row and column communicators
   * @param[in] numkblks. the columns of the local factor can further be
   *            partitioned into numkblks
   */
  DistGNSym(const INPUTMATTYPE& input, const MAT& leftlowrankfactor,
         const MAT& rightlowrankfactor, const MPICommunicator& communicator,
         const int numkblks)
      : DistAUNMF<INPUTMATTYPE>(input, leftlowrankfactor, rightlowrankfactor,
                                communicator, numkblks) {
    XY.zeros(this->k, this->k);
    Dt.zeros(this->k, this->H.n_rows);
    grad.zeros(this->k, this->H.n_rows);
    this->Wt.zeros(arma::size(this->Ht));
    this->WtAij.zeros(arma::size(this->grad));
    localHtAijH.zeros(this->k, this->k);
    HtAijH.zeros(this->k, this->k);

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

    err_matmuls = 0;
    err_grams = 0;

    cg_tot_iters = 0;

    // Get paired processor
    int coords[2];
    coords[0] = MPI_COL_RANK;
    coords[1] = MPI_ROW_RANK;
    MPI_Cart_rank(this->m_mpicomm.gridComm(), &coords[0], &paired_proc);
    PRINTROOT("DistGNSym() constructor successful");
  }
  // Virtual functions of AUNMF
  void updateH() {}

  void updateW() {}

  /**
   * Sets the maximum number of CG iterations to run per outer iteration
   * @param[in] max_cgiters is the upper limit of CG iterations
   */
  void set_luciters(int max_cgiters) {
    this->cg_max_iters = max_cgiters;
  }

  /**
   * Computes XtH for 1-D distributed matrices X and Y
   * @param[in] reference to local matrix X of 
   *            size \f$k \times \frac{globaln}{p} \f$
   * @param[in] reference to local matrix Y of 
   *            size \f$k \times \frac{globaln}{p} \f$
   * @param[in] reference to output matrix XY of 
   *            size \f$k \times k \f$
   * used to compute XtH
   */
  void distDotProduct(const MAT &X, const MAT &Y, MAT *XY) {
    // compute local matrix
    MPITIC;  // gram
    MAT localXY = X * Y;
    // temporary memory allocation
    // MAT XtYt = localXtY;
    // XtYt.zeros();
#ifdef MPI_VERBOSE
    DISTPRINTINFO("localXY::" << norm(this->localXY, "fro"));
#endif
    double temp = MPITOC;  // gram
    this->time_stats.compute_duration(temp);
    this->time_stats.nongram_duration(temp);
    (*XY).zeros();
    this->reportTime(temp, "Dot::XY::");
    MPITIC;  // allreduce gram

    // This is because of column major ordering of Armadillo.
    MPI_Allreduce(localXY.memptr(), (*XY).memptr(), this->k * this->k,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    temp = MPITOC;  // allreduce gram
    this->time_stats.communication_duration(temp);
    this->time_stats.allreduce_duration(temp);
  }

  /**
   * Computes Gradient as shown in algorithm 3 in the paper
   * using the vectorized formula,
   * \f$\grad f = -2*vec(AH - H(HtH))\f$
   * gradient is stored in this->grad
   */
  void computeGradient() {
    // \grad f = -2*vec(AH-H(HtH)) = J^T r
    // This value will be stored in the grad
    // compute AH
    if (stale_matmul) {
      this->distAH();

      // Swap matmuls with transposed processor
      // Processor (i,j) swaps with (j,i)
      this->grad.zeros();
      int recvsize = this->grad.n_elem;
      int sendsize = this->AHtij.n_elem;
      MPITIC;

      MPI_Sendrecv(this->AHtij.memptr(), sendsize, MPI_DOUBLE,
          paired_proc, 0, this->grad.memptr(), recvsize, MPI_DOUBLE,
          paired_proc, 0, this->m_mpicomm.gridComm(), MPI_STATUS_IGNORE);
      double temp = MPITOC;  // sendrecv
      this->time_stats.communication_duration(temp);
      this->time_stats.sendrecv_duration(temp);

      stale_matmul = false;
    }

    // compute HtH
    if (stale_gram) {
      this->distInnerProduct(this->H, &this->HtH);
      stale_gram = false;
      grad_grams++;
    }

    // store the gradient in grad
    MPITIC;
    this->grad = -2*(this->grad - (this->HtH*this->Ht));
    double temp = MPITOC;
    grad_nongrams++;
    this->time_stats.nongram_duration(temp);
    this->time_stats.compute_duration(temp);
    stale_matmul = true;
  }

  /**
   * Applies approximate Hessian as shown in algorithm 4 
   * using the vectorized formula,
   * \f$Y = 2*(XHtH - H(XtH))\f$
   * Y is stored in this->WtAij
   * X is stored in this->Wt
   */
  void applyHess() {
    // X is the gradient (reshaped to n/p x k)
    // Y1 = X (H^T H)
    if (stale_gram) {
      this->distInnerProduct(this->H, &this->HtH);
      stale_gram = false;
      cg_grams++;
    }

    // Y2 = H (X^T H)
    this->distDotProduct(this->Wt, this->H, &this->XY);
    cg_dotprods++;

    // Y = 2(Y1 + Y2)
    MPITIC;
    this->WtAij = 2*((this->HtH * this->Wt) + (this->XY * this->Ht));
    double temp = MPITOC;
    cg_nongrams++;
    this->time_stats.nongram_duration(temp);
    this->time_stats.compute_duration(temp);
  }

  /**
   * Computes the dot product of the vectorized form 
   * of matrices X and Y, i.e., vec(X)^T * vec(Y)
   * @param[in] reference to local matrix X
   * @param[in] reference to local matrix Y which is the same size as X
   */
  double distDotNorm(const MAT &X, const MAT &Y) {
    double localsum = arma::accu(X % Y);
    double globalsum = 0.0;
    MPI_Allreduce(&localsum, &globalsum, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return globalsum;
  }

  /**
   * Modified error calculation to only work with a 
   * single factor matrix H
   * @param[in] iter is the current iteration number
   */
  void computeError(const int iter) {
    // Fast formula using,
    // \norm{A}_F^2 - 2Tr(AHH^T) + Tr(H^TH * H^TH)

    // Get AHH^T and reuse WtAijH to store it
    if (stale_matmul) {
      this->distAH();

      // Swap matmuls with transposed processor
      // Processor (i,j) swaps with (j,i)
      this->grad.zeros();
      int recvsize = this->grad.n_elem;
      int sendsize = this->AHtij.n_elem;
      MPITIC;

      MPI_Sendrecv(this->AHtij.memptr(), sendsize, MPI_DOUBLE,
          paired_proc, 0, this->grad.memptr(), recvsize, MPI_DOUBLE,
          paired_proc, 0, this->m_mpicomm.gridComm(), MPI_STATUS_IGNORE);
      double temp = MPITOC;  // sendrecv
      this->time_stats.communication_duration(temp);
      this->time_stats.sendrecv_duration(temp);

      stale_matmul = false;
      err_matmuls++;
    }
    // TODO{seswar3} : Make it distDotNorm
    MPITIC;
    this->localHtAijH = this->grad * this->H;
    double temp = MPITOC;
    this->time_stats.err_compute_duration(temp);

    MPITIC;
    MPI_Allreduce(this->localHtAijH.memptr(), this->HtAijH.memptr(),
                  this->k * this->k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    temp = MPITOC;
    this->time_stats.err_communication_duration(temp);

    // compute HtH (check for calling distDotNorm)
    if (stale_gram) {
      this->distInnerProduct(this->H, &this->HtH);
      stale_gram = false;
      err_grams++;
    }

    double tHtAijh = arma::trace(this->HtAijH);
    double tHtHHtH = arma::trace(this->HtH * this->HtH);

    PRINTROOT("::it=" << iter << "::normA::" << this->m_globalsqnormA
                      << "::tHtAijH::" << tHtAijh
                      << "::tHtHHtH::" << tHtHHtH);
    this->objective_err = this->m_globalsqnormA - (2*tHtAijh) + tHtHHtH;
    if (std::abs(this->objective_err) <= 1e-7f) {
      this->objective_err = 0.0;
    }
  }

  /**
   * This is the main outer loop of SymGNCG as
   * shown in algorithm 2 in the paper.
   */
  void computeNMF() {
    PRINTROOT("computeNMF started");
    double temptimer = 0.0;
    unsigned int cgiter = 0;
    for (unsigned int iter = 0; iter < this->num_iterations(); iter++) {
      MPITIC;  // iteration timer

      // Compute the gradient
      MPITIC;
      this->computeGradient();
      temptimer = MPITOC;
      this->time_stats.gradient_duration(temptimer);

      // Conjugate Gradient phase (using D as the direction vector x_k)
      // CG initialization
      // using Wt for p_k and grad for r_k
      // r_0 = b - Ax_0 (r is the gradient)
      // p_0 = r_0
      MPITIC;
      this->Wt = this->grad;
      this->Dt.zeros();
      double rsold = this->distDotNorm(this->grad, this->grad);

      if (this->is_compute_error()) {
        PRINTROOT("it=" << iter << "::algo::" << this->m_algorithm
                << "::CG initial residual::" << rsold);
      }

      // Enter CG iterations only if residual is large
      if (rsold > cg_tol) {
        // for k = 1,2,...
        for (cgiter = 0; cgiter < this->cg_max_iters; cgiter++) {
          // compute A p_k stored in WtAij
          this->applyHess();

          // \alpha_k = r_k^T r_k / p_k^T A p_k
          alpha = rsold / this->distDotNorm(this->WtAij, this->Wt);

          // x_{k+1} = x_k + \alpha_k p_k
          this->Dt = this->Dt + (alpha * this->Wt);

          // r_{k+1} = r_k - \alpha_k A p_k
          this->grad = this->grad - (alpha * this->WtAij);

          // beta_k = r_{k+1}^T r_{k+1} / r_k^T r_k
          double rsnew = this->distDotNorm(this->grad, this->grad);

          if (this->is_compute_error()) {
            PRINTROOT("it=" << iter << "::CG iter::" << cgiter
                    << "::CG residual::" << rsnew);
          }

          // Stopping criteria
          if (rsnew < cg_tol)
            break;

          beta = rsnew / rsold;
          rsold = rsnew;

          // p_{k+1} = r_{k+1} + \beta_k p_k
          this->Wt = this->grad + (beta * this->Wt);

          cg_tot_iters++;
        }  // end for
      }
      temptimer = MPITOC;
      this->time_stats.cg_duration(temptimer);

      // Project the solution
      MPITIC;
      this->Ht = this->Ht - this->Dt;
      this->Ht.for_each(
          [](MAT::elem_type &val) { val = val > 0.0 ? val : 0.0; });
      this->H = this->Ht.t();
      temptimer = MPITOC;
      this->time_stats.projection_duration(temptimer);
      stale_matmul = true;
      stale_gram = true;

      temptimer = MPITOC;  // iteration timer
      this->time_stats.duration(temptimer);
      PRINTROOT("completed it=" << iter
                                << "::taken::" << this->time_stats.duration());
      // Print Error
      if (this->is_compute_error()) {
        this->computeError(iter);

        PRINTROOT("it=" << iter << "::algo::" << this->m_algorithm << "::k::"
                        << this->k << "::cgiters::" << cgiter << "::err::"
                        << sqrt(this->objective_err) << "::relerr::"
                        << sqrt(this->objective_err / this->m_globalsqnormA));
      }
    }
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
    this->reportTime(this->time_stats.sendrecv_duration(), "total_sendrecv");
    this->reportTime(this->time_stats.nongram_duration(), "total_nongram");
    if (this->is_compute_error()) {
      this->reportTime(this->time_stats.err_compute_duration(),
                       "total_err_compute");
      this->reportTime(this->time_stats.err_compute_duration(),
                       "total_err_communication");
    }
    this->reportTime(this->time_stats.gradient_duration(), "total_gradient");
    this->reportTime(this->time_stats.cg_duration(), "total_cg");
    this->reportTime(this->time_stats.projection_duration(), "total_proj");

    // Print counters
    PRINTROOT("cg_grams::" << cg_grams);
    PRINTROOT("cg_nongrams::" << cg_nongrams);
    PRINTROOT("cg_dotprods::" << cg_dotprods);

    PRINTROOT("grad_grams::" << grad_grams);
    PRINTROOT("grad_nongrams::" << grad_nongrams);

    PRINTROOT("err_matmuls::" << err_matmuls);
    PRINTROOT("err_grams::" << err_grams);

    PRINTROOT("cg_iterations::" << cg_tot_iters);
  }
};

}  // namespace planc

#endif  // DISTNMF_DISTGNSYMNMF_HPP_
