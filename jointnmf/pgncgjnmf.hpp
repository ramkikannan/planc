/* Copyright 2016 Ramakrishnan Kannan */

#ifndef NMF_PGNCGJNMF_HPP_
#define NMF_PGNCGJNMF_HPP_

#include <omp.h>

#include "common/jointnmf.hpp"

namespace planc {

template <class T1, class T2>
class PGNCGJointNMF : public JointNMF<T1, T2> {
 private:
  double m_Afit, m_Sfit;
  MAT m_gradW;
  MAT m_gradH;

  MAT HtH;
  MAT WtW;
  MAT WtA;
  MAT HtS;

  // Variables needed for CG
  MAT xW, xH;       // x solution vector in CG
  MAT yW, yH;       // y vector in CG (y = A p)
  MAT pW, pH;       // p vector in CG
  MAT rW, rH;       // r vector in CG
  UMAT mW, mH;      // Masking matrices for W, H
  MAT pWtW, pHtH;

  // Hyperparameters
  double m_Smax, m_cgtol, m_bdtol;
  int m_maxcgiters, m_numchks;

  // NOTE: W and H will be different depending on the step size, so will need
  // to call this a lot during the line search
  void computeObjectiveError(const MAT &W, const MAT &H, const MAT &WtA, 
                            const MAT &HtS, const MAT &WtW, const MAT &HtH) {
    double sqnormA = this->normA * this->normA;
    double sqnormS = this->normS * this->normS;

    // Fast error calculations for \|A - WH^T\|_F^2
    double TrWtAH   = arma::trace(WtA * H);
    double TrWtWHtH = arma::trace(WtW * HtH);

    // Fast error calculations for \|S - HH^T\|_F^2
    double TrHtSH    = arma::trace(HtS * H);
    double TrHtHHtH = arma::trace(HtH * HtH); 

    // Norms of the factors
    double fro_W_sq  = arma::trace(WtW);
    this->normW      = sqrt(fro_W_sq);
    double fro_H_sq  = arma::trace(HtH);
    this->normH      = sqrt(fro_H_sq);

    this->l1normW   = arma::norm(arma::sum(W, 1), 2);
    this->l1normH   = arma::norm(arma::sum(H, 1), 2);

    // Fit of the NMF approximation
    this->m_Afit     = sqnormA - (2 * TrWtAH) + TrWtWHtH;
    this->m_Sfit     = sqnormS - (2 * TrHtSH) + TrHtHHtH;

    this->fit_err_sq = this->m_Afit + (this->alpha() * this->m_Sfit);

    // Objective being minimized
    this->objective_err = this->fit_err_sq;
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
  }

  // NOTE: should modify the gradient functions here to take the precomputed matrix
  // products like what we do in the computeObjectiveError function above
  void gradW(){
    this->m_gradW = (this->W * (HtH) - this->A * this->H);
  }

  void gradH(){
    this->m_gradH = (this->H * (WtW) - WtA.t()) 
                    + 2 * this->alpha() * (this->H * (HtH) - HtS.t());
  }

  void applyGramian(){
    this->pWtW = this->pW.t() * this->W;

    this->pHtH = this->pH.t() * this->H;

    this->yW = this->pW * this->HtH + this->W * pHtH; 
    this->yH = this->H * pWtW + this->pH * this->WtW
                + (2 * this->alpha())*(this->pH * this->HtH + this->H * pHtH);
  }

  // NOTE: this function does the line search and then applies the gradient to each factor matrix 
  bool applyStep(){

      MAT W_temp = this->W;
      MAT H_temp = this->H;

      double prev_error = this->objective_err;
      
      double stepsz  = 1;
      
      bool obj_dec = false;

      int stepchk = 0;
      do {
        W_temp = this->W - (stepsz * xW);
        W_temp.for_each([](MAT::elem_type &val) { val = val > 0.0 ? val : 0.0; });
        H_temp = this->H - (stepsz * xH);
        H_temp.for_each([](MAT::elem_type &val) { val = val > 0.0 ? val : 0.0; });

        HtH = H_temp.t() * H_temp;
        WtW = W_temp.t() * W_temp;
        WtA = W_temp.t() * this->A;
        HtS = H_temp.t() * this->S;

        this->computeObjectiveError(W_temp, H_temp, WtA, HtS, WtW, HtH);

        if(this->objective_err < prev_error){
          obj_dec = true;
        } else {
          stepsz = stepsz / 2;
          if (stepchk > 10) {
            break;
          } else {
            stepchk++;
          }
        } 
      } while(!obj_dec);

      INFO << "line search iter::" << stepchk << std::endl;

      if (obj_dec){
        this->W = this->W - (stepsz * xW);
        this->W.for_each([](MAT::elem_type &val) { val = val > 0.0 ? val : 0.0; });
        this->H = this->H - (stepsz * xH);
        this->H.for_each([](MAT::elem_type &val) { val = val > 0.0 ? val : 0.0; });
      } else { 
        //reset them in case step isn't taken
        HtH = this->H.t() * this->H;
        WtW = this->W.t() * this->W;
        WtA = this->W.t() * this->A;
        HtS = this->H.t() * this->S;
      }
      return obj_dec;
  }
 public:
  PGNCGJointNMF(const T1 &A, const T2 &S, int lowrank) 
      : JointNMF<T1, T2>(A, S, lowrank) {
    // Set hyperparameters
    this->m_maxcgiters = 20;
    this->m_cgtol      = 1e-8;
    this->m_bdtol      = 1e-5;
    this->m_numchks    = 20;

    // Set all the other variables
    INFO << "Finished PGNCG constructing" << std::endl;
  }
  PGNCGJointNMF(const T1 &A, const T2 &S, MAT W, MAT H) 
      : JointNMF<T1, T2>(A, S, W, H) {
    // Set hyperparameters
    this->m_maxcgiters = 20;
    this->m_cgtol      = 1e-8;
    this->m_bdtol      = 1e-5;
    this->m_numchks    = 20;

    // Set all the other variables
    INFO << "Finished PGNCG constructing with factors" << std::endl;
  }

  // NOTE: need to collect momentum parameter and implement line search...
  void computeNMF() {
    INFO << "JointNMF hyperparameters::alpha::" << this->alpha()
        << "::maxcgiters::" << this->m_maxcgiters
        << "::cgtol::" << this->m_cgtol << "::bdtol::" << this->m_bdtol
        << "::linesearch tries::" << this->m_numchks << std::endl;
    INFO << "A: " << arma::size(this->A) << std::endl;
    INFO << "S: " << arma::size(this->S) << std::endl;
    INFO << "W: " << arma::size(this->W) << std::endl;
    INFO << "H: " << arma::size(this->H) << std::endl;

    int curitr = 0;

    // implement stale_gram and stale_matmul
    HtH = this->H.t() * this->H;
    WtW = this->W.t() * this->W;
    WtA = this->W.t() * this->A;
    HtS = this->H.t() * this->S;

    // Variables for CG
    double rsold, rsnew, cgalpha, cgbeta;

    // initialize step direction variables
    this->xW.zeros(arma::size(this->W));
    this->xH.zeros(arma::size(this->H));

    while (curitr < this->num_iterations()) {
      // Update W
      // tic();

      this->computeObjectiveError(this->W, this->H, WtA, HtS, WtW, HtH);
      this->printObjective(curitr);

      gradW();
      gradH();

      // Setup the masks
      mW = (this->W <= m_bdtol) % (this->m_gradW > 0);
      mH = (this->H <= m_bdtol) % (this->m_gradH > 0);

      // r_0 = g
      rW = this->m_gradW;
      rH = this->m_gradH;
      rW.elem(arma::find(mW)).fill(0.0);
      rH.elem(arma::find(mH)).fill(0.0);

      // p_0 = r_0
      pW = rW;
      pH = rH;

      rsold = arma::dot(rW,rW) + arma::dot(rH, rH);

      INFO << "initial rsold::" << rsold << std::endl;

      // use CG to compute step direction
      if (rsold > m_cgtol) {
        // https://en.wikipedia.org/wiki/Conjugate_gradient_method
        for (int cgiter = 0; cgiter < this->m_maxcgiters; cgiter++) {
          // y_k = Ap_k = J^T J p_k
          applyGramian();

          // Mask the Hessian application
          yW.elem(arma::find(mW)).fill(0.0);
          yH.elem(arma::find(mH)).fill(0.0);

          // ptr / ptAp
          cgalpha = rsold / (arma::dot(pW, yW) + arma::dot(pH, yH));
          INFO << "cgalpha::" << cgalpha << std::endl;

          // x_{k+1} = x_k + alpha_k * p_k
          xW = xW + (cgalpha * pW);
          xH = xH + (cgalpha * pH);

          // r_{k+1} = r_k - alpha_k * y_k
          rW = rW - (cgalpha * yW);
          rH = rH - (cgalpha * yH);

          rsnew = arma::dot(rW, rW) + arma::dot(rH, rH);

          INFO << "it=" << curitr << "::CG iter::" 
            << cgiter << "::CG residual::" << rsnew << std::endl;

          // beta_k = <r_{k+1}, r_{k+1}> / <r_k, r_k>
          cgbeta = rsnew / rsold;

          // Stopping criteria
          if (rsnew < m_cgtol)
            break;

          pW = rW + (cgbeta * pW);
          pH = rH + (cgbeta * pH);

          rsold = rsnew;          
        }

        // Mask the updates 
        xW.elem(arma::find(mW)) = this->m_gradW.elem(arma::find(mW));
        xH.elem(arma::find(mH)) = this->m_gradH.elem(arma::find(mH));
      }

      // Line search and take step 
      bool objective_decrease = applyStep();

      // line search failed
      if (!objective_decrease){
        INFO << "Line search failed to decrease objective" << std::endl;
        break;
      }

      curitr++;
    }
    if (curitr == this->num_iterations()){
      this->computeObjectiveError(this->W, this->H, WtA, HtS, WtW, HtH);
      this->printObjective(curitr);
    }
    INFO << "Completed JointNMF." << std::endl; 
  }

  /**
   * Function to output all the variables computed by the
   * PGNCG version of JointNMF. Saves W, and H.
   * @param[in] outfname: Prefix for the output files
   */
  void saveOutput(std::string outfname) {
    std::string Wfname  = outfname + "_W";
    std::string Hfname  = outfname + "_H";

    this->W.save(Wfname, arma::raw_ascii);
    this->H.save(Hfname, arma::raw_ascii);
  }
};  // class PGNCGJointNMF

}  // namespace planc

#endif  // NMF_PGNCGJNMF_HPP_
