/* Copyright 2016 Ramakrishnan Kannan */

#ifndef NMF_ANLSBPPJNMF_HPP_
#define NMF_ANLSBPPJNMF_HPP_

#include <omp.h>

#include "common/jointnmf.hpp"
#include "nnls/bppnnls.hpp"

#define ONE_THREAD_MATRIX_SIZE 2000

namespace planc {

template <class T1, class T2>
class ANLSBPPJointNMF : public JointNMF<T1, T2> {
 private:
  double m_beta, m_symmdiff;
  double m_Afit, m_Sfit;
  
  // Auxiliary variable
  MAT H2;
  double normH2;

  /**
   * ANLS/BPP for solving the NLS problem,
   * \min \|GC - F\|_F with C >= 0.
   * @param[in] GtG: LHS of the normal equations (G^T G)
   * @param[in] GtF: RHS of the normal equations (G^T F)
   * @param[in] C  : Solution of the NLS problem
   */
  void updateFactor(const MAT& GtG, const MAT& GtF, MAT* C) {
    UINT numChunks = GtF.n_cols / ONE_THREAD_MATRIX_SIZE;
    if (numChunks * ONE_THREAD_MATRIX_SIZE < GtF.n_cols) numChunks++;

// #pragma omp parallel for schedule(dynamic)
    for (UINT i = 0; i < numChunks; i++) {
      UINT spanStart = i * ONE_THREAD_MATRIX_SIZE;
      UINT spanEnd = (i + 1) * ONE_THREAD_MATRIX_SIZE - 1;
      if (spanEnd > GtF.n_cols - 1) {
        spanEnd = GtF.n_cols - 1;
      }
      BPPNNLS<MAT, VEC> subProblem(GtG,
                            (MAT)GtF.cols(spanStart, spanEnd), true);
#ifdef _VERBOSE
      // #pragma omp critical
      {
        INFO << "Scheduling " << worh << " start=" << spanStart
             << ", end=" << spanEnd
             // << ", tid=" << omp_get_thread_num()
             << std::endl
             << "LHS ::" << std::endl
             << GtG << std::endl
             << "RHS ::" << std::endl
             << (MAT)GtF.cols(spanStart, spanEnd) << std::endl;
      }
#endif

      subProblem.solveNNLS();

#ifdef _VERBOSE
      INFO << "completed " << worh << " start=" << spanStart
           << ", end=" << spanEnd
           // << ", tid=" << omp_get_thread_num() << " cpu=" << sched_getcpu()
           << " time taken=" << t2 << std::endl;
#endif
      (*C).rows(spanStart, spanEnd) = subProblem.getSolutionMatrix().t();
    }
  }

  void computeObjectiveError(const MAT &WtA, const MAT &H2tS, const MAT &WtW,
            const MAT &HtH, const MAT &H2tH2) {
    double sqnormA = this->normA * this->normA;
    double sqnormS = this->normS * this->normS;

    // Fast error calculations for \|A - WH^T\|_F^2
    double TrWtAH   = arma::trace(WtA * this->H);
    double TrWtWHtH = arma::trace(WtW * HtH);

    // Fast error calculations for \|S - H2H^T\|_F^2
    double TrH2tSH    = arma::trace(H2tS * this->H);
    double TrH2tH2HtH = arma::trace(H2tH2 * HtH); 

    // Norms of the factors
    double fro_W_sq  = arma::trace(WtW);
    double fro_W_obj = this->m_regW(0) * fro_W_sq;
    this->normW      = sqrt(fro_W_sq);
    double fro_H_sq  = arma::trace(HtH);
    double fro_H_obj = this->m_regH(0) * fro_H_sq;
    this->normH      = sqrt(fro_H_sq);
    this->normH2     = sqrt(arma::trace(H2tH2));

    this->l1normW   = arma::norm(arma::sum(this->W, 1), 2);
    double l1_W_obj = this->m_regW(1) * this->l1normW * this->l1normW;
    this->l1normH   = arma::norm(arma::sum(this->H, 1), 2);
    double l1_H_obj = this->m_regH(1) * this->l1normH * this->l1normH;

    // Fit of the NMF approximation
    this->m_Afit     = sqnormA - (2 * TrWtAH) + TrWtWHtH;
    this->m_Sfit     = sqnormS - (2 * TrH2tSH) + TrH2tH2HtH;
    this->fit_err_sq = this->m_Afit + (this->alpha() * this->m_Sfit);

    this->m_symmdiff = arma::norm(this->H2 - this->H, "fro");
    double sym_obj   = this->m_beta * this->m_symmdiff * this->m_symmdiff;

    // Objective being minimized
    this->objective_err = this->fit_err_sq + fro_W_obj + fro_H_obj
        + l1_W_obj + l1_H_obj + sym_obj;
  }
  
  /// Print out the objective stats
  void printObjective(const int itr) {
    double Aerr = (this->m_Afit > 0)? sqrt(this->m_Afit) : this->normA;
    double Serr = (this->m_Sfit > 0)? sqrt(this->m_Sfit) : this->normS;
    INFO << "Completed it = " << itr << std::endl;
    INFO << "objective::" << this->objective_err 
         << "::squared error::" << this->fit_err_sq << std::endl 
         << "A error::" << Aerr 
         << "::A relative error::" << Aerr / this->normA << std::endl
         << "S error::" << Serr 
         << "::S relative error::" << Serr / this->normS << std::endl;
    INFO << "W frobenius norm::" << this->normW 
         << "::W L_12 norm::" << this->l1normW << std::endl
         << "H frobenius norm::" << this->normH 
         << "::H L_12 norm::" << this->l1normH << std::endl;
    INFO << "symmdiff::" << this->m_symmdiff 
         << "::relative symmdiff::" << this->m_symmdiff / this->normH
         << std::endl;
  }

 public:
  ANLSBPPJointNMF(const T1 &A, const T2 &S, int lowrank) 
      : JointNMF<T1, T2>(A, S, lowrank) {
    // Set all the other variables
    this->H2 = this->H;
    this->m_beta = this->alpha() * this->S.max();
    INFO << "Finished constructing" << std::endl;
  }
  ANLSBPPJointNMF(const T1 &A, const T2 &S, MAT W, MAT H) 
      : JointNMF<T1, T2>(A, S, W, H) {
    // Set all the other variables
    this->H2 = this->H;
    this->m_beta = this->alpha() * this->S.max();
    INFO << "Finished constructing with factors" << std::endl;
  }
  void computeNMF() {
    int curitr = 0;

    // temporary variables reused throughout the computation
    MAT lhs  = arma::zeros<MAT>(this->k, this->k);
    MAT nrhs = arma::zeros<MAT>(this->k, this->n);
    MAT fac  = arma::zeros<MAT>(this->n, this->k);

    while (curitr < this->num_iterations()) {
      // Update W
      tic();
      MAT HtH = this->H.t() * this->H;
      lhs     = HtH;
      this->applyReg(this->regW(), &lhs);
      MAT HtAt = this->H.t() * this->A.t();
      updateFactor(lhs, HtAt, &this->W);
      double tW = toc();

      // Update H2
      tic();
      lhs      = this->alpha() * HtH;
      MAT HtSt = this->H.t() * this->S.t();
      nrhs     = this->alpha() * HtSt;
      fac      = this->H.t();
      this->applySymmetricReg(this->beta(), &lhs, &fac, &nrhs);
      updateFactor(lhs, nrhs, &this->H2);
      double tH2 = toc();

      // Update H
      tic();
      MAT WtW   = this->W.t() * this->W;
      MAT H2tH2 = this->H2.t() * this->H2;
      lhs       = WtW + (this->alpha() * H2tH2);
      this->applyReg(this->regH(), &lhs);

      MAT WtA  = this->W.t() * this->A;
      MAT H2tS = this->H2.t() * this->S;
      nrhs     = WtA + (this->alpha() * H2tS);
      fac      = this->H2.t();
      this->applySymmetricReg(this->beta(), &lhs, &fac, &nrhs);
      updateFactor(lhs, nrhs, &this->H);
      double tH = toc();

      INFO << "Completed It (" << curitr << "/"
           << this->num_iterations() << ")"
           << " time =" << tW + tH2 + tH << std::endl;
      this->computeObjectiveError(WtA, H2tS, WtW, HtH, H2tH2);
      this->printObjective(curitr);

      curitr++;
    }
    INFO << "Completed JointNMF." << std::endl; 
  }
  /// Sets the beta parameter
  void beta(const double b) { this->m_beta = b; }
  // Returns the beta parameter
  double beta() { return this->m_beta; }
  /**
   * Function to output all the variables computed by the
   * ANLS version of JointNMF. Saves W, H and the auxiliary 
   * variable H2.
   * @param[in] outfname: Prefix for the output files
   */
  void saveOutput(std::string outfname) {
    std::string Wfname  = outfname + "_W";
    std::string Hfname  = outfname + "_H";
    std::string H2fname = outfname + "_H2";

    this->W.save(Wfname, arma::raw_ascii);
    this->H.save(Hfname, arma::raw_ascii);
    this->H2.save(H2fname, arma::raw_ascii);
  }
};  // class ANLSBPPJointNMF

}  // namespace planc

#endif  // NMF_ANLSBPPJNMF_HPP_
