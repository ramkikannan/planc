/* Copyright 2016 Ramakrishnan Kannan */
#ifndef COMMON_NMF_HPP_
#define COMMON_NMF_HPP_
#include <assert.h>
#include <string>
#include "common/utils.hpp"

// #ifndef _VERBOSE
// #define _VERBOSE 1;
// #endif

#define NUM_THREADS 4
#define CONV_ERR 0.000001
#define NUM_STATS 9

// #ifndef COLLECTSTATS
// #define COLLECTSTATS 1
// #endif

namespace planc {

// T must be a either an instance of MAT or sp_MAT
template <class T>
class NMF {
 protected:
  const T &A;       /// input matrix of size mxn
  MAT W, H;  /// left and low rank factors of size mxk and nxk respectively
  MAT Winit, Hinit;
  UINT m, n, k;  /// rows, columns and lowrank

  /*
   * Collected statistics are
   * iteration Htime Wtime totaltime normH normW densityH densityW relError
   */
  MAT stats;
  double objective_err, fit_err_sq;  /// objective and fit error at any particular iteration
  double normA, normW, normH, l1normW, l1normH;  /// norms of input and factor matrices
  double symmdiff;  /// difference between factors for symmetric regularization
  double densityW, densityH;
  bool cleared;
  double m_symm_reg;              /// Symmetric Regularization parameter
  unsigned int m_num_iterations;  /// number of iterations
  double m_tolerance; // error tolerance
  std::string input_file_name;
  algotype m_updalgo; // Update algorithm type
  MAT errMtx;       // used for error computation.
  T A_err_sub_mtx;  // used for error computation.
  /// The regularization is a vector of two values. The first value specifies
  /// L2 regularization values and the second is L1 regularization.
  FVEC m_regW;
  FVEC m_regH;

  void collectStats(int iteration) {
    this->normW = arma::norm(this->W, "fro");
    this->normH = arma::norm(this->H, "fro");
    UVEC nnz = find(this->W > 0);
    this->densityW = nnz.size() / (this->m * this->k);
    nnz.clear();
    nnz = find(this->H > 0);
    this->densityH = nnz.size() / (this->m * this->k);
    this->stats(iteration, 4) = this->normH;
    this->stats(iteration, 5) = this->normW;
    this->stats(iteration, 6) = this->densityH;
    this->stats(iteration, 7) = this->densityW;
    this->stats(iteration, 8) = this->objective_err;
  }

  /**
   * For both L1 and L2 regularizations we only adjust the
   * HtH or WtW. The regularization is a vector of two values.
   * The first value specifies L2 regularization values
   * and the second is L1 regularization.
   * param[in] regularization as a vector
   * param[out] Gram matrix
   */
  void applyReg(const FVEC &reg, MAT *AtA) {
    // Frobenius norm regularization
    if (reg(0) > 0) {
      MAT identity = arma::eye<MAT>(this->k, this->k);
      float lambda_l2 = reg(0);
      (*AtA) = (*AtA) + (lambda_l2 * identity);
    }

    // L1 - norm regularization
    if (reg(1) > 0) {
      MAT onematrix = arma::ones<MAT>(this->k, this->k);
      float lambda_l1 = reg(1);
      (*AtA) = (*AtA) + (lambda_l1 * onematrix);
    }
  }

  /**
   * For both L1 and L2 regularizations we only adjust the
   * HtH or WtW. This function removes the regularization for
   * error and objective calculations.
   * param[in] regularization as a vector
   * param[out] Gram matrix
   */
  void removeReg(const FVEC &reg, MAT *AtA) {
    // Frobenius norm regularization
    if (reg(0) > 0) {
      MAT identity = arma::eye<MAT>(this->k, this->k);
      float lambda_l2 = reg(0);
      (*AtA) = (*AtA) - (lambda_l2 * identity);
    }

    // L1 - norm regularization
    if (reg(1) > 0) {
      MAT onematrix = arma::ones<MAT>(this->k, this->k);
      float lambda_l1 = reg(1);
      (*AtA) = (*AtA) - (lambda_l1 * onematrix);
    }
  }

  /**
   * This is for symmetric ANLS variant.
   *
   * If we are trying to solve for H using normal equation WtWH = WtA
   * Symmetric regularization will translate to solve
   * (WtW+sym_regI)H = WtA + sym_regWt
   * In the following function, lhs is WtW, rhs is WtA and fac is Wt
   */
  void applySymmetricReg(double sym_reg, MAT *lhs, MAT *fac, MAT *rhs) {
    if (sym_reg > 0) {
      MAT identity = arma::eye<MAT>(this->k, this->k);
      (*lhs) = (*lhs) + (sym_reg * identity);
      (*rhs) = (*rhs) + (sym_reg * (*fac));
    }
  }

  /**
   * This is for symmetric ANLS variant.
   *
   * If we are trying to solve for H using normal equation WtWH = WtA
   * Symmetric regularization will translate to solve
   * (WtW+sym_regI)H = WtA + sym_regWt
   * This function removes the regularization for error and objective
   * calculations.
   */
  void removeSymmetricReg(double sym_reg, MAT *lhs, MAT *fac, MAT *rhs) {
    if (sym_reg > 0) {
      MAT identity = arma::eye<MAT>(this->k, this->k);
      (*lhs) = (*lhs) - (sym_reg * identity);
      (*rhs) = (*rhs) - (sym_reg * (*fac));
    }
  }

  /**
   *  L2 normalize column vectors of W
   */

  void normalize_by_W() {
    MAT W_square = arma::pow(this->W, 2);
    ROWVEC norm2 = arma::sqrt(arma::sum(W_square, 0));
    for (unsigned int i = 0; i < this->k; i++) {
      if (norm2(i) > 0) {
        this->W.col(i) = this->W.col(i) / norm2(i);
        this->H.col(i) = this->H.col(i) * norm2(i);
      }
    }
  }

 private:
  void otherInitializations() {
    this->stats.zeros();
    this->cleared = false;
    this->normA = arma::norm(this->A, "fro");
    this->m_num_iterations = 20;
    this->objective_err = 1000000000000;
    this->stats.resize(m_num_iterations + 1, NUM_STATS);
  }

 public:
  /**
   * Constructors with an input matrix and low rank
   * @param[in] input matrix as reference.
   * @param[in] low rank
   */
  NMF(const T &input, const unsigned int rank) : A(input) {
    // this->A = input;
    this->m = A.n_rows;
    this->n = A.n_cols;
    this->k = rank;
    // prime number closer to W.
    arma::arma_rng::set_seed(89);
    this->W = arma::randu<MAT>(m, k);
    // prime number close to H
    arma::arma_rng::set_seed(73);
    this->H = arma::randu<MAT>(n, k);
    this->m_regW = arma::zeros<FVEC>(2);
    this->m_regH = arma::zeros<FVEC>(2);
    normalize_by_W();

    // make the random MATrix positive
    // absMAT<MAT>(W);
    // absMAT<MAT>(H);
    // other intializations
    this->otherInitializations();
  }
  /**
   * Constructor with initial left and right low rank factors
   * Necessary when you want to compare algorithms starting with
   * the same initialization
   */

  NMF(const T &input, const MAT &leftlowrankfactor,
      const MAT &rightlowrankfactor): A(input) {
    assert(leftlowrankfactor.n_cols == rightlowrankfactor.n_cols);
    // this->A = input;
    this->W = leftlowrankfactor;
    this->H = rightlowrankfactor;
    this->Winit = this->W;
    this->Hinit = this->H;
    this->m = A.n_rows;
    this->n = A.n_cols;
    this->k = W.n_cols;
    this->m_regW = arma::zeros<FVEC>(2);
    this->m_regH = arma::zeros<FVEC>(2);

    // other initializations
    this->otherInitializations();
  }

  virtual void computeNMF() = 0;

  /// Returns the left low rank factor matrix W
  MAT getLeftLowRankFactor() { return W; }
  /// Returns the right low rank factor matrix H
  MAT getRightLowRankFactor() { return H; }

  /*
   * A is mxn
   * Wr is mxk will be overwritten. Must be passed with values of W.
   * Hr is nxk will be overwritten. Must be passed with values of H.
   * All MATrices are in row major forMAT
   * ||A-WH||_F^2 = over all nnz (a_ij - w_i h_j)^2 +
   *           over all zeros (w_i h_j)^2
   *         = over all nnz (a_ij - w_i h_j)^2 +
   ||WH||_F^2 - over all nnz (w_i h_j)^2
   *
   */
#if 0
    void computeObjectiveError() {
        // 1. over all nnz (a_ij - w_i h_j)^2
        // 2. over all nnz (w_i h_j)^2
        // 3. Compute R of W ahd L of H through QR
        // 4. use sgemm to compute RL
        // 5. use slange to compute ||RL||_F^2
        // 6. return nnzsse+nnzwh-||RL||_F^2
        tic();
        float nnzsse = 0;
        float nnzwh  = 0;
        MAT  Rw(this->k, this->k);
        MAT  Rh(this->k, this->k);
        MAT  Qw(this->m, this->k);
        MAT  Qh(this->n, this->k);
        MAT  RwRh(this->k, this->k);

        // #pragma omp parallel for reduction (+ : nnzsse,nnzwh)
        for (UWORD jj = 1; jj <= this->A.n_cols; jj++) {
            UWORD startIdx  = this->A.col_ptrs[jj - 1];
            UWORD endIdx    = this->A.col_ptrs[jj];
            UWORD col       = jj - 1;
            float nnzssecol = 0;
            float nnzwhcol  = 0;

            for (UWORD ii = startIdx; ii < endIdx; ii++) {
                UWORD row     = this->A.row_indices[ii];
                float tempsum = 0;

                for (UWORD kk = 0; kk < k; kk++) {
                    tempsum += (this->W(row, kk) * this->H(col, kk));
                }
                nnzwhcol  += tempsum * tempsum;
                nnzssecol += (this->A.values[ii] - tempsum)
                             * (this->A.values[ii] - tempsum);
            }
            nnzsse += nnzssecol;
            nnzwh  += nnzwhcol;
        }
        qr_econ(Qw, Rw, this->W);
        qr_econ(Qh, Rh, this->H);
        RwRh = Rw * Rh.t();
        float normWH = arma::norm(RwRh, "fro");
        Rw.clear();
        Rh.clear();
        Qw.clear();
        Qh.clear();
        RwRh.clear();
        INFO << "error compute time " << toc() << std::endl;
        float fastErr = sqrt(nnzsse + (normWH * normWH - nnzwh));
        this->objective_err = fastErr;

        // return (fastErr);
    }

#else  // ifdef BUILD_SPARSE
  // Removing blk error calculations as default method
  void computeObjectiveError_blk() {
    // (init.norm_A)^2 - 2*trace(H'*(A'*W))+trace((W'*W)*(H*H'))
    // MAT WtW = this->W.t() * this->W;
    // MAT HtH = this->H.t() * this->H;
    // MAT AtW = this->A.t() * this->W;

    // double sqnormA  = this->normA * this->normA;
    // double TrHtAtW  = arma::trace(this->H.t() * AtW);
    // double TrWtWHtH = arma::trace(WtW * HtH);

    // this->objective_err = sqnormA - (2 * TrHtAtW) + TrWtWHtH;
#ifdef _VERBOSE
    INFO << "Entering computeObjectiveError A=" << this->A.n_rows << "x"
         << this->A.n_cols << " W = " << this->W.n_rows << "x" << this->W.n_cols
         << " H=" << this->H.n_rows << "x" << this->H.n_cols << std::endl;
#endif
    tic();
    // always restrict the errMtx size to fit it in memory
    // and doesn't occupy much space.
    // For eg., the max we can have only 3 x 10^6 elements.
    // The number of columns must be chosen appropriately.
    UWORD PER_SPLIT = std::ceil((3 * 1e6) / A.n_rows);
    // UWORD PER_SPLIT = 1;
    // always colSplit. Row split is really slow as the matrix is col major
    // always
    bool colSplit = true;
    // if (this->A.n_rows > PER_SPLIT || this->A.n_cols > PER_SPLIT) {
    uint numSplits = 1;
    MAT Ht = this->H.t();
    if (this->A.n_cols > PER_SPLIT) {
      // if (this->A.n_cols < this->A.n_rows)
      //     colSplit = false;
      if (colSplit)
        numSplits = A.n_cols / PER_SPLIT;
      else
        numSplits = A.n_rows / PER_SPLIT;
      // #ifdef _VERBOSE
    } else {
      PER_SPLIT = A.n_cols;
      numSplits = 1;
    }
#ifdef _VERBOSE
    INFO << "PER_SPLIT = " << PER_SPLIT << "numSplits = " << numSplits
         << std::endl;
#endif
    // #endif
    VEC splitErr = arma::zeros<VEC>(numSplits + 1);
    // allocate one and never allocate again.
    if (colSplit && errMtx.n_rows == 0 && errMtx.n_cols == 0) {
      errMtx = arma::zeros<MAT>(A.n_rows, PER_SPLIT);
      A_err_sub_mtx = arma::zeros<T>(A.n_rows, PER_SPLIT);
    } else {
      errMtx = arma::zeros<MAT>(PER_SPLIT, A.n_cols);
      A_err_sub_mtx = arma::zeros<T>(PER_SPLIT, A.n_cols);
    }
    for (unsigned int i = 0; i <= numSplits; i++) {
      UWORD beginIdx = i * PER_SPLIT;
      UWORD endIdx = (i + 1) * PER_SPLIT - 1;
      if (colSplit) {
        if (endIdx > A.n_cols) endIdx = A.n_cols - 1;
        if (beginIdx < endIdx) {
#ifdef _VERBOSE
          INFO << "beginIdx=" << beginIdx << " endIdx= " << endIdx << std::endl;
          INFO << "Ht = " << Ht.n_rows << "x" << Ht.n_cols << std::endl;

#endif
          errMtx = W * Ht.cols(beginIdx, endIdx);
          A_err_sub_mtx = A.cols(beginIdx, endIdx);
        } else if (beginIdx == endIdx && beginIdx < A.n_cols) {
          errMtx = W * Ht.col(beginIdx);
          A_err_sub_mtx = A.col(beginIdx);
        }
      } else {
        if (endIdx > A.n_rows) endIdx = A.n_rows - 1;
#ifdef _VERBOSE
        INFO << "beginIdx=" << beginIdx << " endIdx= " << endIdx << std::endl;
#endif
        if (beginIdx < endIdx) {
          A_err_sub_mtx = A.rows(beginIdx, endIdx);
          errMtx = W.rows(beginIdx, endIdx) * Ht;
        }
      }
      A_err_sub_mtx -= errMtx;
      A_err_sub_mtx %= A_err_sub_mtx;
      splitErr(i) = arma::accu(A_err_sub_mtx);
    }
    double err_time = toc();
    INFO << "err compute time::" << err_time << std::endl;
    this->objective_err = arma::sum(splitErr);
  }

  void computeObjectiveError() {
    MAT AtW = this->A.t() * this->W;
    MAT WtW = this->W.t() * this->W;
    MAT HtH = this->H.t() * this->H;

    double sqnormA = this->normA * this->normA;
    double TrHtAtW = arma::trace(this->H.t() * AtW);
    double TrWtWHtH = arma::trace(WtW * HtH);

    // Norms of the factors
    double fro_W_sq = arma::trace(WtW);
    double fro_W_obj = this->m_regW(0) * fro_W_sq;
    this->normW = sqrt(fro_W_sq);
    double fro_H_sq = arma::trace(HtH);
    double fro_H_obj = this->m_regH(0) * fro_H_sq;
    this->normH = sqrt(fro_H_sq);

    this->l1normW = arma::norm(arma::sum(this->W, 1), 2);
    double l1_W_obj = this->m_regW(1) * this->l1normW * this->l1normW;
    this->l1normH = arma::norm(arma::sum(this->H, 1), 2);
    double l1_H_obj = this->m_regH(1) * this->l1normH * this->l1normH;

    // Fit of the NMF approximation
    this->fit_err_sq = sqnormA - (2 * TrHtAtW) + TrWtWHtH;

    double sym_obj = 0.0;
    if (this->m_symm_reg > 0) {
      this->symmdiff = arma::norm(this->W - this->H, "fro");
      sym_obj = this->m_symm_reg * symmdiff * symmdiff;
    }

    // Objective being minimized
    this->objective_err = this->fit_err_sq + fro_W_obj + fro_H_obj
        + l1_W_obj + l1_H_obj + sym_obj;
  }
#endif  // ifdef BUILD_SPARSE
  void computeObjectiveError(const T &At, const MAT &WtW, const MAT &HtH) {
    MAT AtW = At * this->W;

    double sqnormA = this->normA * this->normA;
    double TrHtAtW = arma::trace(this->H.t() * AtW);
    double TrWtWHtH = arma::trace(WtW * HtH);

    // Norms of the factors
    double fro_W_sq = arma::trace(WtW);
    double fro_W_obj = this->m_regW(0) * fro_W_sq;
    this->normW = sqrt(fro_W_sq);
    double fro_H_sq = arma::trace(HtH);
    double fro_H_obj = this->m_regH(0) * fro_H_sq;
    this->normH = sqrt(fro_H_sq);

    this->l1normW = arma::norm(arma::sum(this->W, 1), 2);
    double l1_W_obj = this->m_regW(1) * this->l1normW * this->l1normW;
    this->l1normH = arma::norm(arma::sum(this->H, 1), 2);
    double l1_H_obj = this->m_regH(1) * this->l1normH * this->l1normH;

    // Fit of the NMF approximation
    this->fit_err_sq = sqnormA - (2 * TrHtAtW) + TrWtWHtH;

    double sym_obj = 0.0;
    if (this->m_symm_reg > 0) {
      this->symmdiff = arma::norm(this->W - this->H, "fro");
      sym_obj = this->m_symm_reg * symmdiff * symmdiff;
    }

    // Objective being minimized
    this->objective_err = this->fit_err_sq + fro_W_obj + fro_H_obj
        + l1_W_obj + l1_H_obj + sym_obj;
  }

  /// Print out the objective stats
  void printObjective(const int itr) {
    double err = (this->fit_err_sq > 0)? sqrt(this->fit_err_sq) : this->normA;
    INFO << "Completed it = " << itr 
         << "::algo::" << this->m_updalgo << "::k::" << this->k << std::endl;
    INFO << "objective::" << this->objective_err 
         << "::squared error::" << this->fit_err_sq << std::endl 
         << "error::" << err 
         << "::relative error::" << err / this->normA << std::endl;
    INFO << "W frobenius norm::" << this->normW 
         << "::W L_12 norm::" << this->l1normW << std::endl
         << "H frobenius norm::" << this->normH 
         << "::H L_12 norm::" << this->l1normH << std::endl;
    if (this->m_symm_reg > 0) {
      INFO << "symmdiff::" << this->symmdiff 
           << "::relative symmdiff::" << this->symmdiff / this->normW
           << std::endl;
    }
  }

  /// Sets number of iterations for the NMF algorithms
  void num_iterations(const int it) { this->m_num_iterations = it; }
  /// Sets the relative error tolerance for NMF algorithms
  void tolerance(const double tol) { this->m_tolerance = tol; }
  // Returns the relative error tolerance for NMF algorithms
  double tolerance() { return this->m_tolerance; }
  /// Sets the regularization on left low rank factor W
  void regW(const FVEC &iregW) { this->m_regW = iregW; }
  /// Sets the regularization on right low rank H
  void regH(const FVEC &iregH) { this->m_regH = iregH; }
  /// Returns the L2 and L1 regularization parameters of W as a vector
  FVEC regW() { return this->m_regW; }
  /// Returns the L2 and L1 regularization parameters of W as a vector
  FVEC regH() { return this->m_regH; }
  /// Set the Symmetric regularization parameter
  void symm_reg(const double &i_symm_reg) { this->m_symm_reg = i_symm_reg; }
  /// Returns the Symmetric regularization parameter
  double symm_reg() { return this->m_symm_reg; }
  /// Set the update algorithm
  void updalgo(algotype dat) { this->m_updalgo = dat; }

  /// Returns the number of iterations
  const unsigned int num_iterations() const { return m_num_iterations; }

  ~NMF() { clear(); }
  /// Clear the memory for input matrix A, right low rank factor W
  /// and left low rank factor H
  void clear() {
    if (!this->cleared) {
      // this->A.clear();
      this->W.clear();
      this->H.clear();
      this->stats.clear();
      if (errMtx.n_rows != 0 && errMtx.n_cols != 0) {
        errMtx.clear();
        A_err_sub_mtx.clear();
      }
      this->cleared = true;
    }
  }
};
}  // namespace planc
#endif  // COMMON_NMF_HPP_
