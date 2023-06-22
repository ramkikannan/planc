/* Copyright 2022 Ramakrishnan Kannan */
#ifndef COMMON_JOINTNMF_HPP_
#define COMMON_JOINTNMF_HPP_
#include <assert.h>
#include <string>
#include "common/utils.hpp"

// #ifndef _VERBOSE
// #define _VERBOSE 1;
// #endif

#define NUM_THREADS 4
#define CONV_ERR 0.000001

namespace planc {

// T1, T2 must be a either an instance of MAT or SP_MAT
template <class T1, class T2>
class JointNMF {
 protected:
  const T1 &A;       /// input features matrix of size mxn
  const T2 &S;       /// input connection matrix of size nxn (should be expanded to an array)
  MAT W, H;          /// left and low rank factors of size mxk and nxk respectively
  MAT Winit, Hinit;
  UINT m, n, k;  /// rows, columns and lowrank

  algotype m_algorithm; /// Update algorithm type
  double m_alpha; /// Weighting parameter

  /*
   * Collected statistics during the computation
   */
  double objective_err, fit_err_sq;  /// objective and fit error at any particular iteration
  double normA, normS;  /// norms of the input matrices
  double normW, normH, l1normW, l1normH;  /// norms of the factor matrices
  double densityW, densityH;
  bool cleared;
  unsigned int m_num_iterations;  /// number of iterations
  double m_tolerance; // error tolerance

  /// File names for reading in input files
  std::string input_file_name;
  std::string conn_file_name;
  
  /// The regularization is a vector of two values. The first value specifies
  /// L2 regularization values and the second is L1 regularization.
  FVEC m_regW;
  FVEC m_regH;

  /**
   * For both L1 and L2 regularizations we only adjust the
   * HtH or WtW. The regularization is a vector of two values.
   * The first value specifies L2 regularization values
   * and the second is L1 regularization.
   * @param[in] regularization as a vector
   * @param[out] Gram matrix
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
   * @param[in] reg: regularization as a vector
   * @param[out] AtA: Gram matrix
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
   * @param[in] sym_reg: double for symmetric regularization
   * @param[in] lhs: LHS gram matrix
   * @param[in] fac: crossfactor matrix to add to the RHS
   * @param[in] rhs: RHS matrix, typically the bottleneck computation
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
   * @param[in] sym_reg: double for symmetric regularization
   * @param[in] lhs: LHS gram matrix
   * @param[in] fac: crossfactor matrix to add to the RHS
   * @param[in] rhs: RHS matrix, typically the bottleneck computation
   */
  void removeSymmetricReg(double sym_reg, MAT *lhs, MAT *fac, MAT *rhs) {
    if (sym_reg > 0) {
      MAT identity = arma::eye<MAT>(this->k, this->k);
      (*lhs) = (*lhs) - (sym_reg * identity);
      (*rhs) = (*rhs) - (sym_reg * (*fac));
    }
  }

  /**
   *  L2 normalize column vectors of W and push the 
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
    this->cleared = false;
    this->normA = arma::norm(this->A, "fro");
    this->normS = arma::norm(this->S, "fro");
    this->m_num_iterations = 20;
    this->m_alpha = (this->normA * this->normA) / (this->normS * this->normS);
    }

 public:
  /**
   * Constructors with an input matrices and low rank
   * @param[in] input features matrix as reference
   * @param[in] conn connection matrix as reference
   * @param[in] low rank
   */
  JointNMF(const T1 &input, const T2 &conn, const unsigned int rank) 
      : A(input), S(conn) {
    this->m = A.n_rows;
    this->n = A.n_cols;
    this->k = rank;

    // Initialize factor matrices. Note that random seeds are set in
    // the driver class
    this->W = arma::randu<MAT>(m, k);
    this->H = arma::randu<MAT>(n, k);

    this->Winit = this->W;
    this->Hinit = this->H;

    this->m_regW = arma::zeros<FVEC>(2);
    this->m_regH = arma::zeros<FVEC>(2);
    normalize_by_W();

    // other intializations
    this->otherInitializations();
  }
  /**
   * Constructor with initial left and right low rank factors
   * Necessary when you want to compare algorithms starting with
   * the same initialization.
   * @param[in] input features matrix as reference
   * @param[in] conn connection matrix as reference
   * @param[in] leftlowrankfactor  W matrix
   * @param[in] rightlowrankfactor H matrix
   */
  JointNMF(const T1 &input, const T2 &conn, const MAT &leftlowrankfactor,
      const MAT &rightlowrankfactor): A(input), S(conn) {
    assert(leftlowrankfactor.n_cols == rightlowrankfactor.n_cols);
    
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
  virtual void saveOutput(std::string outfname) = 0;

  /// Returns the left low rank factor matrix W
  MAT getLeftLowRankFactor() { return W; }
  /// Returns the right low rank factor matrix H
  MAT getRightLowRankFactor() { return H; }

  /// Sets number of iterations for the JointNMF algorithms
  void num_iterations(const int it) { this->m_num_iterations = it; }
  /// Sets the relative error tolerance for JointNMF algorithms
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
  void algorithm(algotype dat) { this->m_algorithm = dat; }
  /// Sets the alpha parameter
  void alpha(const double alp) { this->m_alpha = alp; }
  // Returns the alpha parameter
  double alpha() { return this->m_alpha; }

  /// Returns the number of iterations
  const unsigned int num_iterations() const { return m_num_iterations; }

  /// Sets the beta parameter. For anlsbpp
  void beta(const double b) { }
  // Returns the beta parameter. For anlsbpp
  double beta() { return 0; }

  /// Sets the gamma parameter. For pgd
  void gamma(const double b) { }
  // Returns the gamma parameter. For pgd
  double gamma() { return 0; }
};
} // namespace planc

#endif // COMMON_JOINTNMF_HPP
