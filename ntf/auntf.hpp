/* Copyright Ramakrishnan Kannan 2017 */

#ifndef NTF_AUNTF_HPP_
#define NTF_AUNTF_HPP_

#define MPITIC tic();
#define MPITOC toc();

#ifdef MKL_FOUND
#include <mkl.h>
#else
#include <cblas.h>
#endif

#include <armadillo>
#include <vector>
#include "common/ncpfactors.hpp"
#include "common/ntf_utils.hpp"
#include "common/tensor.hpp"
#include "dimtree/ddt.hpp"

namespace planc {

#define TENSOR_DIM (m_input_tensor.dimensions())
#define TENSOR_NUMEL (m_input_tensor.numel())

// #ifndef NTF_VERBOSE
// #define NTF_VERBOSE 1
// #endif

// extern "C" void cblas_dgemm_(const CBLAS_LAYOUT Layout,
//                              const CBLAS_TRANSPOSE transa,
//                              const CBLAS_TRANSPOSE transb,
//                              const int m, const int n,
//                              const int k, const double alpha,
//                              const double *a, const int lda,
//                              const double *b, const int ldb,
//                              const double beta, double *c,
//                              const int ldc);
class AUNTF {
 protected:
  planc::NCPFactors m_ncp_factors;
  MAT *ncp_mttkrp_t;
  MAT gram_without_one;
  virtual MAT update(const int mode) = 0;

 private:
  const planc::Tensor &m_input_tensor;
  int m_num_it;
  int m_current_it;
  bool m_compute_error;

  const int m_low_rank_k;
  MAT *ncp_krp;
  const algotype m_updalgo;
  planc::Tensor *lowranktensor;
  DenseDimensionTree *kdt;
  bool m_enable_dim_tree;
  // needed for acceleration algorithms.
  bool m_accelerated;
  double m_rel_error;
  double m_normA;
  std::vector<bool> m_stale_mttkrp;

  // Ensure factor is unnormalised when calling this function
  void update_factor_mode(const int &current_mode, const MAT &factor) {
    m_ncp_factors.set(current_mode, factor);
    m_ncp_factors.normalize(current_mode);

    if (m_enable_dim_tree) {
      MAT temp = m_ncp_factors.factor(current_mode).t();
      kdt->set_factor(temp.memptr(), current_mode);
    }

    int num_modes = this->m_input_tensor.modes();
    for (int mode = 0; mode < num_modes; mode++) {
      if (mode != current_mode) this->m_stale_mttkrp[current_mode] = true;
    }
  }
  virtual void accelerate() {}

 public:
  AUNTF(const planc::Tensor &i_tensor, const int i_k, algotype i_algo)
      : m_ncp_factors(i_tensor.dimensions(), i_k, false),
        m_input_tensor(i_tensor),
        m_low_rank_k(i_k),
        m_updalgo(i_algo) {
    m_ncp_factors.normalize();
    gram_without_one.zeros(i_k, i_k);
    ncp_mttkrp_t = new MAT[i_tensor.modes()];
    ncp_krp = new MAT[i_tensor.modes()];
    for (int i = 0; i < i_tensor.modes(); i++) {
      UWORD current_size = TENSOR_NUMEL / TENSOR_DIM[i];
      ncp_krp[i].zeros(current_size, i_k);
      ncp_mttkrp_t[i].zeros(i_k, TENSOR_DIM[i]);
      this->m_stale_mttkrp.push_back(true);
    }
    lowranktensor = new planc::Tensor(i_tensor.dimensions());
    m_compute_error = false;
    m_num_it = 20;
    m_normA = i_tensor.norm();
    // INFO << "Init factors for NCP" << std::endl << "======================";
    // m_ncp_factors.print();
    this->m_enable_dim_tree = false;
  }
  ~AUNTF() {
    for (int i = 0; i < m_input_tensor.modes(); i++) {
      ncp_krp[i].clear();
      ncp_mttkrp_t[i].clear();
    }
    delete[] ncp_krp;
    delete[] ncp_mttkrp_t;
    if (this->m_enable_dim_tree) {
      delete kdt;
    }
    delete lowranktensor;
  }
  NCPFactors &ncp_factors() { return m_ncp_factors; }
  void dim_tree(bool i_dim_tree) {
    this->m_enable_dim_tree = i_dim_tree;
    if (i_dim_tree) {
      this->kdt = new DenseDimensionTree(m_input_tensor, m_ncp_factors,
                                         m_input_tensor.modes() / 2);
    }
  }
  double current_error() const { return this->m_rel_error; }
  void num_it(const int i_n) { this->m_num_it = i_n; }
  void computeNTF() {
    for (m_current_it = 0; m_current_it < m_num_it; m_current_it++) {
      INFO << "iter::" << this->m_current_it << std::endl;
      for (int j = 0; j < this->m_input_tensor.modes(); j++) {
        m_ncp_factors.gram_leave_out_one(j, &gram_without_one);
#ifdef NTF_VERBOSE
        INFO << "gram_without_" << j << "::" << arma::cond(gram_without_one)
             << std::endl
             << gram_without_one << std::endl;
#endif
        if (this->m_stale_mttkrp[j]) {
          m_ncp_factors.krp_leave_out_one(j, &ncp_krp[j]);
#ifdef NTF_VERBOSE
          INFO << "krp_leave_out_" << j << std::endl << ncp_krp[j] << std::endl;
#endif
          if (this->m_enable_dim_tree) {
            double multittv_time = 0;
            double mttkrp_time = 0;
            kdt->in_order_reuse_MTTKRP(j, ncp_mttkrp_t[j].memptr(), false,
                                       multittv_time, mttkrp_time);
          } else {
            m_input_tensor.mttkrp(j, ncp_krp[j], &ncp_mttkrp_t[j]);
          }
          this->m_stale_mttkrp[j] = false;
#ifdef NTF_VERBOSE
          INFO << "mttkrp for factor" << j << std::endl
               << ncp_mttkrp_t[j] << std::endl;
#endif
        }
        // MAT factor = update(m_updalgo, gram_without_one, ncp_mttkrp_t[j], j);
        MAT factor = update(j);
#ifdef NTF_VERBOSE
        INFO << "iter::" << i << "::factor:: " << j << std::endl
             << factor << std::endl;
#endif
        update_factor_mode(j, factor.t());
      }
      if (m_compute_error) {
        double temp_err = computeObjectiveError();
        this->m_rel_error = temp_err;
        INFO << "relative_error at it::" << this->m_current_it
             << "::" << temp_err << std::endl;
      }
      if (this->m_accelerated) accelerate();
#ifdef NTF_VERBOSE
      INFO << "ncp factors" << std::endl;
      m_ncp_factors.print();
#endif
    }
  }
  void accelerated(const bool &set_acceleration) {
    this->m_accelerated = set_acceleration;
    this->m_compute_error = true;
  }
  bool is_stale_mttkrp(const int &current_mode) const {
    return this->m_stale_mttkrp[current_mode];
  }
  void compute_error(bool i_error) { this->m_compute_error = i_error; }
  void reset(const NCPFactors &new_factors, bool trans = false) {
    int m_modes = this->m_input_tensor.modes();
    if (!trans) {
      for (int i = 0; i < m_modes; i++) {
        update_factor_mode(i, new_factors.factor(i));
      }
    } else {
      for (int i = 0; i < m_modes; i++) {
        update_factor_mode(i, new_factors.factor(i).t());
      }
    }
    m_ncp_factors.set_lambda(new_factors.lambda());
  }
  int current_it() const { return m_current_it; }
  double computeObjectiveError() {
    // current low rank tensor
    // UWORD krpsize = arma::prod(this->m_dimensions);
    // krpsize /= this->m_dimensions[0];
    // MAT krpleavingzero.zeros(krpsize, this->m_k);
    // krp_leave_out_one(0, &krpleavingzero);
    // MAT lowranktensor(this->m_dimensions[0], krpsize);
    // lowranktensor = this->ncp_factors[0] * trans(krpleavingzero);

    // compute current low rank tensor as above.
    m_ncp_factors.krp_leave_out_one(0, &ncp_krp[0]);
    // cblas_dgemm_(const CBLAS_LAYOUT Layout,
    //              const CBLAS_TRANSPOSE transa,
    //              const CBLAS_TRANSPOSE transb,
    //              const MKL_INT m, const MKL_INT n,
    //              const MKL_INT k, const double alpha,
    //              const double * a, const MKL_INT lda,
    //              const double * b, const MKL_INT ldb,
    //              const double beta, double * c,
    //              const MKL_INT ldc);
    // char transa = 'T';
    // char transb = 'N';
    int m = m_ncp_factors.factor(0).n_rows;
    int n = ncp_krp[0].n_rows;
    int k = m_ncp_factors.factor(0).n_cols;
    int lda = m;
    int ldb = n;
    int ldc = m;
    double alpha = 1;
    double beta = 0;
    char nt = 'N';
    char t = 'T';

    MAT unnorm_fac =
        m_ncp_factors.factor(0) * arma::diagmat(m_ncp_factors.lambda());
    // double *output_tensor = new double[ldc * n];
    dgemm_(&nt, &t, &m, &n, &k, &alpha, unnorm_fac.memptr(), &lda,
           ncp_krp[0].memptr(), &ldb, &beta, &lowranktensor->m_data[0], &ldc);
    // INFO << "lowrank tensor::" << std::endl;
    // lowranktensor->print();
    // for (int i=0; i < ldc*n; i++){
    //     INFO << i << ":" << output_tensor[i] << std::endl;
    // }
    double err = m_input_tensor.err(*lowranktensor);
    err = std::sqrt(err / this->m_normA);
    return err;
  }
  double computeObjectiveError(const NCPFactors &new_factors_t) {
    reset(new_factors_t, true);
    return computeObjectiveError();
  }
};  // class AUNTF
}  // namespace planc

#endif  // NTF_AUNTF_HPP_
