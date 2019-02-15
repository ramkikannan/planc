/* Copyright 2017 Ramakrishnan Kannan */

#ifndef COMMON_NCPFACTORS_HPP_
#define COMMON_NCPFACTORS_HPP_

#include <cassert>
#include "common/tensor.hpp"
#include "common/utils.h"
#ifdef MPI_DISTNTF
#include <mpi.h>
#endif

/**
 *  ncp_factors contains the factors of the ncp
 * every ith factor is of size n_i * k
 * number of factors is called as mode of the tensor
 * all idxs are zero idx.
 */

namespace planc {

class NCPFactors {
  MAT *ncp_factors;   /// Array of factors .One factor for every mode.
  unsigned int m_modes;        /// Number of modes in tensor
  unsigned int m_k;            /// Low rank
  UVEC m_dimensions;  /// Vector of dimensions for every mode
  /// in the distributed mode all the processes has same lambda
  VEC m_lambda;
  /// normalize the factors of a matrix
  bool freed_ncp_factors;

 public:
  /**
   * constructor that takes the dimensions of every mode, low rank k
   * All the factors will be initialized with random uniform number
   * @param[in] vector of dimension for every mode.
   * @param[in] low rank
   * @param[in] trans. takes true or false. Transposes every factor.
   */

  NCPFactors(const UVEC &i_dimensions, const int &i_k, bool trans) {
    this->m_dimensions = i_dimensions;
    this->m_modes = i_dimensions.n_rows;
    ncp_factors = new MAT[this->m_modes];
    this->m_k = i_k;
    arma::arma_rng::set_seed(103);
    for (unsigned int i = 0; i < this->m_modes; i++) {
      // ncp_factors[i] = arma::randu<MAT>(i_dimensions[i], this->m_k);
      int rsize = (i_dimensions[i] > 0) ? i_dimensions[i] : 1;
      if (trans) {
        ncp_factors[i] = arma::randu<MAT>(this->m_k, rsize);
      } else {
        // ncp_factors[i] = arma::randi<MAT>(i_dimensions[i], this->m_k,
        //                                   arma::distr_param(0, numel));
        ncp_factors[i] = arma::randu<MAT>(rsize, this->m_k);
      }
    }
    m_lambda = arma::ones<VEC>(this->m_k);
    freed_ncp_factors = false;
  }

  // copy constructor
  /*NCPFactors(const NCPFactors &src) {
m_dimensions = src.dimensions();
m_modes = src.modes();
m_lambda = src.lambda();
m_k = src.rank();
if (ncp_factors == NULL) {
    ncp_factors = new MAT[this->m_modes];
    for (int i = 0; i < this->m_modes; i++) {
        ncp_factors[i] = arma::randu<MAT>(this->m_dimensions[i],
                                          this->m_k);
    }
}
for (int i = 0; i < this->m_modes; i++) {
    if (ncp_factors[i].n_elem == 0) {
        ncp_factors[i] = arma::zeros<MAT>(src.factor(i).n_rows,
            src.factor(i).n_cols);
    }
    ncp_factors[i] = src.factor(i);
}
}*/

  ~NCPFactors() {
    /*for (int i=0; i < this->m_modes; i++){
ncp_factors[i].clear();
}*/
    if (!freed_ncp_factors) {
      delete[] ncp_factors;
      freed_ncp_factors = true;
    }
  }

  // getters
  /// returns low rank
  int rank() const { return m_k; }
  /// dimensions of every mode
  UVEC dimensions() const { return m_dimensions; }
  /// factor matrix of a mode i_n
  MAT &factor(const int i_n) const { return ncp_factors[i_n]; }
  /// returns number of modes
  int modes() const { return m_modes; }
  /// returns the lambda vector
  VEC lambda() const { return m_lambda; }

  // setters
  /**
   * Set the mode i_n with the given factor matrix
   * @param[in] i_n mode for which given factor matrix will be updated
   * @param[in] i_factor factor matrix
   */
  void set(const int i_n, const MAT &i_factor) {
    assert(i_factor.size() == this->ncp_factors[i_n].size());
    this->ncp_factors[i_n] = i_factor;
  }
  /// sets the lambda vector
  void set_lambda(const VEC &new_lambda) { m_lambda = new_lambda; }
  // compute gram of all local factors
  /**
   * Return the hadamard of the factor grams
   * @param[out] UtU is a kxk matrix
   */
  void gram(MAT *o_UtU) {
    MAT currentGram(this->m_k, this->m_k);
    for (unsigned int i = 0; i < this->m_modes; i++) {
      currentGram = ncp_factors[i].t() * ncp_factors[i];
      (*o_UtU) = (*o_UtU) % currentGram;
    }
  }

  // find the hadamard product of all the factor grams
  // except the n. This is equation 50 of the JGO paper.
  /**
   * Returns the hadamard of the factor grams except i_n
   * @param[in] i_n ith mode that will be excluded in the factor grams
   * @param[out] UtU is a kxk matrix
   */

  void gram_leave_out_one(const unsigned int i_n, MAT *o_UtU) {
    MAT currentGram(this->m_k, this->m_k);
    (*o_UtU) = arma::ones<MAT>(this->m_k, this->m_k);
    for (unsigned int i = 0; i < this->m_modes; i++) {
      if (i != i_n) {
        currentGram = ncp_factors[i].t() * ncp_factors[i];
        (*o_UtU) = (*o_UtU) % currentGram;
      }
    }
  }
  /**
   * KRP leaving out the mode i_n
   * @param[in] mode i_n
   * @return MAT of size product of dimensions except i_n by k
   */
  MAT krp_leave_out_one(const unsigned int i_n) {
    UWORD krpsize = arma::prod(this->m_dimensions);
    krpsize /= this->m_dimensions[i_n];
    MAT krp(krpsize, this->m_k);
    krp_leave_out_one(i_n, &krp);
    return krp;
  }
  // construct low rank tensor using the factors
  /**
   * khatrirao leaving out one. we are using the implementation
   * from tensor toolbox. Always krp for mttkrp is computed in
   * reverse. Hence assuming the same. The order of the computation
   * is same a tensor tool box.
   * size of krp must be product of all dimensions leaving out nxk
   * @param[in] i_n mode that will be excluded
   * @param[out] m_dimensions[i_n]xk
   */
  void krp_leave_out_one(const unsigned int i_n, MAT *o_krp) {
    // matorder = length(A):-1:1;
    // Always krp for mttkrp is computed in
    // reverse. Hence assuming the same.
    UVEC matorder = arma::zeros<UVEC>(this->m_modes - 1);
    int j = 0;
    for (int i = this->m_modes - 1; i >= 0; i--) {
      if (i != i_n) {
        matorder(j++) = i;
      }
    }
#ifdef NTF_VERBOSE
    INFO << "::" << __PRETTY_FUNCTION__ << "::" << __LINE__
         << "::matorder::" << matorder << std::endl;
#endif
    o_krp->zeros();
    // N = ncols(1);
    // This is our k. So keep N = k in our case.
    // P = A{matorder(1)};
    // take the first factor of matorder
    /*UWORD current_nrows = ncp_factors[matorder(0)].n_rows - 1;
(*o_krp).rows(0, current_nrows) = ncp_factors[matorder(0)];
// this is factor by factor
for (int i = 1; i < this->m_modes - 1; i++) {
// remember always krp in reverse order.
// That is if A krp B krp C, we compute as
// C krp B krp A.
// prev_nrows = current_nrows;
// rightkrp.n_rows;
// we are populating column by column
MAT& rightkrp = ncp_factors[matorder[i]];
for (int j = 0; j < this->m_k; j++) {
VEC krpcol = (*o_krp)(arma::span(0, current_nrows), j);
// krpcol.each_rows*rightkrp.col(i);
for (int k = 0; k < rightkrp.n_rows; k++) {
    (*o_krp)(arma::span(k * krpcol.n_rows, (k + 1)*krpcol.n_rows -
1), j) = krpcol * rightkrp(k, j);
}
}
current_nrows *= rightkrp.n_rows;
}*/
    // Loop through all the columns
    // for n = 1:N
    //     % Loop through all the matrices
    //     ab = A{matorder(1)}(:,n);
    //     for i = matorder(2:end)
    //        % Compute outer product of nth columns
    //        ab = A{i}(:,n) * ab(:).';
    //     end
    //     % Fill nth column of P with reshaped result
    //     P(:,n) = ab(:);
    // end
    for (unsigned int n = 0; n < this->m_k; n++) {
      MAT ab = ncp_factors[matorder[0]].col(n);
      for (unsigned int i = 1; i < this->m_modes - 1; i++) {
        VEC oldabvec = arma::vectorise(ab);
        VEC currentvec = ncp_factors[matorder[i]].col(n);
        ab.clear();
        ab = currentvec * oldabvec.t();
      }
      (*o_krp).col(n) = arma::vectorise(ab);
    }
  }
  /**
   * KRP of the given vector of modes. It can be any subset of the modes.
   * @param[in] Subset of modes
   * @param[out] KRP of product of dimensions of the given modes by k
   */
  void krp(const UVEC i_modes, MAT *o_krp) {
    // matorder = length(A):-1:1;
    // Always krp for mttkrp is computed in
    // reverse. Hence assuming the same.
    UVEC matorder = arma::zeros<UVEC>(i_modes.n_rows - 1);    
    int j = 0;
    for (int i = i_modes.n_rows - 1; i >= 0; i--) {
      matorder(j++) = i_modes[i];
    }
#ifdef NTF_VERBOSE
    INFO << "::" << __PRETTY_FUNCTION__ << "::" << __LINE__
         << "::matorder::" << matorder << std::endl;
#endif
    (*o_krp).zeros();
    for (unsigned int n = 0; n < this->m_k; n++) {
      MAT ab = ncp_factors[matorder[0]].col(n);
      for (unsigned int i = 1; i < i_modes.n_rows - 1; i++) {
        VEC oldabvec = arma::vectorise(ab);
        VEC currentvec = ncp_factors[matorder[i]].col(n);
        ab.clear();
        ab = currentvec * oldabvec.t();
      }
      (*o_krp).col(n) = arma::vectorise(ab);
    }
  }

  // caller must free
  // Tensor rankk_tensor() {
  //     UWORD krpsize = arma::prod(this->m_dimensions);
  //     krpsize /= this->m_dimensions[0];
  //     MAT krpleavingzero = arma::zeros<MAT>(krpsize, this->m_k);
  //     krp_leave_out_one(0, &krpleavingzero);
  //     MAT lowranktensor(this->m_dimensions[0], krpsize);
  //     lowranktensor = this->ncp_factors[0] * krpleavingzero.t();
  //     Tensor rc(this->m_dimensions, lowranktensor.memptr());
  //     return rc;
  // }
  /**
   * Construct the rank k tensor out of the factor matrices
   * Determine the KRP of the n-1 modes leaving out 0 and multiply
   * with the mode 0 factor matrix
   * @param[out] Rank-k tensor
   */
  void rankk_tensor(Tensor &out) {
    UWORD krpsize = arma::prod(this->m_dimensions);
    krpsize /= this->m_dimensions[0];
    MAT krpleavingzero = arma::zeros<MAT>(krpsize, this->m_k);
    krp_leave_out_one(0, &krpleavingzero);
    MAT lowranktensor(this->m_dimensions[0], krpsize);
    lowranktensor = this->ncp_factors[0] * krpleavingzero.t();
    Tensor rc(this->m_dimensions, lowranktensor.memptr());
    out = rc;
  }
  /**
   *  prints just the information about the factors.
   * it will NOT print the factor
   */
  void printinfo() {
    INFO << "modes::" << this->m_modes << "::k::" << this->m_k << std::endl;
    INFO << "lambda::" << this->m_lambda << std::endl;
    INFO << "::dims::" << std::endl << this->m_dimensions << std::endl;
  }
  /// prints the entire NCPFactors including the factor matrices
  void print() {
    printinfo();
    for (unsigned int i = 0; i < this->m_modes; i++) {
      std::cout << i << "th factor" << std::endl
                << "=============" << std::endl;
      std::cout << this->ncp_factors[i];
    }
  }
  /**
   * print the ith factor matrix alone
   * @param[in] i_n the mode for which the factor matrix to be printed
   */
  void print(const unsigned int i_n) {
    std::cout << i_n << "th factor" << std::endl
              << "=============" << std::endl;
    std::cout << this->ncp_factors[i_n];
  }
  /**
   * Transposes the entire factor matrix.
   * @param[out] factor_t that contains the transpose of every factor matrix.
   */
  void trans(NCPFactors &factor_t) {
    for (unsigned int i = 0; i < this->m_modes; i++) {
      factor_t.set(i, this->ncp_factors[i].t());
    }
  }
  /// only during initialization. Reset's all lambda.
  void normalize() {
    double colNorm = 0.0;
    m_lambda.ones();
    for (unsigned int i = 0; i < this->m_modes; i++) {
      for (unsigned int j = 0; j < this->m_k; j++) {
        colNorm = arma::norm(this->ncp_factors[i].col(j));
        if (colNorm > 0) this->ncp_factors[i].col(j) /= colNorm;
        m_lambda(j) *= colNorm;
      }
    }
  }
  // replaces the existing lambdas
  /**
   * Column normalizes the factor matrix of the given mode
   * and replaces the existing lambda.
   * @param[in] mode of the factor matrix that will be column normalized
   */
  void normalize(int mode) {
    for (unsigned int i = 0; i < this->m_k; i++) {
      m_lambda(i) = arma::norm(this->ncp_factors[mode].col(i));
      if (m_lambda(i) > 0) this->ncp_factors[mode].col(i) /= m_lambda(i);
    }
  }
  // replaces the existing lambdas
  /**
   * Row normalizes the factor matrix of the given mode
   * and replaces the existing lambda.
   * @param[in] mode of the factor matrix that will be row normalized
   */
  void normalize_rows(unsigned int mode) {
    for (unsigned int i = 0; i < this->m_k; i++) {
      m_lambda(i) = arma::norm(this->ncp_factors[mode].row(i));
      if (m_lambda(i) > 0) this->ncp_factors[mode].row(i) /= m_lambda(i);
    }
  }

  /**
   * initializes the local tensor with the given seed.
   * this is for reinitializing random numbers across different
   * processors.
   * @param[in] i_seed
   */
  void randu(const int i_seed) {
    arma::arma_rng::set_seed(i_seed);
    for (unsigned int i = 0; i < this->m_modes; i++) {
      if (m_dimensions[i] > 0) {
        ncp_factors[i].randu();
      } else {
        ncp_factors[i].zeros();
      }
    }
  }
  /// this is for reinitializing zeros across different processors.
  void zeros() {
    for (unsigned int i = 0; i < this->m_modes; i++) {
      ncp_factors[i].zeros();
    }
  }
#ifdef MPI_DISTNTF
  // Distribution normalization of factor matrices
  // To be used for MPI code only
  /**
   * Distributed column normalize of all the modes
   * across different processors.
   */
  void distributed_normalize() {
    double local_colnorm;
    double global_colnorm;
    for (unsigned int i = 0; i < this->m_modes; i++) {
      for (unsigned int j = 0; j < this->m_k; j++) {
        local_colnorm = arma::norm(this->ncp_factors[i].col(j));
        local_colnorm *= local_colnorm;
        MPI_Allreduce(&local_colnorm, &global_colnorm, 1, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        global_colnorm = std::sqrt(global_colnorm);
        if (global_colnorm > 0) this->ncp_factors[i].col(j) /= global_colnorm;
        m_lambda(j) *= global_colnorm;
      }
    }
  }
  /**
   * Distributed column normalize of a given mode
   * across different processors.
   * @param[in] mode
   */
  void distributed_normalize(unsigned int mode) {
    double local_colnorm;
    double global_colnorm;
    for (unsigned int j = 0; j < this->m_k; j++) {
      local_colnorm = arma::norm(this->ncp_factors[mode].col(j));
      local_colnorm *= local_colnorm;
      MPI_Allreduce(&local_colnorm, &global_colnorm, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      global_colnorm = std::sqrt(global_colnorm);
      if (global_colnorm > 0) this->ncp_factors[mode].col(j) /= global_colnorm;
      m_lambda(j) = global_colnorm;
    }
  }
  /**
   * Distributed row normalize of a given mode
   * across different processors.
   * @param[in] mode
   */
  void distributed_normalize_rows(unsigned int mode) {
    double local_rownorm;
    double global_rownorm;
    for (unsigned int j = 0; j < this->m_k; j++) {
      local_rownorm = arma::norm(this->ncp_factors[mode].row(j));
      local_rownorm *= local_rownorm;
      MPI_Allreduce(&local_rownorm, &global_rownorm, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      global_rownorm = std::sqrt(global_rownorm);
      if (global_rownorm > 0) this->ncp_factors[mode].row(j) /= global_rownorm;
      m_lambda(j) = global_rownorm;
    }
  }
#endif
};  // NCPFactors
}  // namespace planc

#endif  // COMMON_NCPFACTORS_HPP_
