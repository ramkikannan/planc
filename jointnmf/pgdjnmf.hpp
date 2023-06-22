/* Copyright 2016 Ramakrishnan Kannan */

#ifndef NMF_PGDJNMF_HPP_
#define NMF_PGDJNMF_HPP_

#include <omp.h>

#include "common/jointnmf.hpp"

namespace planc {

template <class T1, class T2>
class PGDJointNMF : public JointNMF<T1, T2> {
 private:
  double m_Afit, m_Sfit;
  MAT m_gradW;
  MAT m_gradH;

  MAT HtH;
  MAT WtW;
  MAT WtA;
  MAT HtS;

  // Variables needed for momentum
  MAT prevstepW, prevstepH;
  double m_gamma = .9;

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
    this->m_gradW = 2 * (this->W * (HtH) - this->A * this->H);
  }

  void gradH(){
    this->m_gradH = 2 * (this->H * (WtW) - WtA.t()) 
                    + 4 * this->alpha() * (this->H * (HtH) - HtS.t());
  }

  // NOTE: this function does the line search and then applies the gradient to each factor matrix 
  bool applyGradient(){

      MAT W_temp = this->W;
      MAT H_temp = this->H;

      double prev_error = this->objective_err;
      
      double stepsz  = 1;
      
      double gnormW = arma::norm(this->m_gradW, "fro");
      double gnormH = arma::norm(this->m_gradH, "fro");

      double gnorm = 1.0 / sqrt(gnormH*gnormH + gnormW*gnormW);

      bool obj_dec = false;

      int stepchk = 0;
      do {
        W_temp = this->W - stepsz * (m_gamma*prevstepW + this->m_gradW * gnorm);
        W_temp.for_each([](MAT::elem_type &val) { val = val > 0.0 ? val : 0.0; });
        H_temp = this->H - stepsz * (m_gamma*prevstepH + this->m_gradH * gnorm);
        H_temp.for_each([](MAT::elem_type &val) { val = val > 0.0 ? val : 0.0; });

        HtH = H_temp.t() * H_temp;
        WtW = W_temp.t() * W_temp;
        WtA = W_temp.t() * this->A;
        HtS = H_temp.t() * this->S;

        this->computeObjectiveError(W_temp, H_temp, WtA, HtS, WtW, HtH);

        if(this->objective_err < prev_error){
          INFO << "objective_err::"<< this->objective_err<< "::prev_error::" << prev_error << std::endl;
          obj_dec = true;
        } else {
          stepsz = stepsz / 2;
          // TODO: change this to stop after 10 line checks...
          if (stepchk > 10) {
            break;
          } else {
            stepchk++;
          }
          // stepsz = stepsz - stepsz_dec;
        } 
        // stepsz = stepsz - stepsz_dec;
      } while(!obj_dec);
      // while(this->objective_err < prev_error);


      INFO << "line search iter::" << stepchk << std::endl;
      W_temp = this->W;
      H_temp = this->H;
      if (obj_dec){
        // based upon linesearch update W and H...
        // stepsz = stepsz + stepsz_dec;
        this->W = this->W - stepsz * (m_gamma*prevstepW + this->m_gradW * gnorm);
        this->W.for_each([](MAT::elem_type &val) { val = val > 0.0 ? val : 0.0; });
        this->H = this->H - stepsz * (m_gamma*prevstepH + this->m_gradH * gnorm);
        this->H.for_each([](MAT::elem_type &val) { val = val > 0.0 ? val : 0.0; });
      } else { 
        // stepsz = 1e-6;
        // this->W = this->W - stepsz * (m_gamma*prevstepW + this->m_gradW * gnorm);
        // this->W.for_each([](MAT::elem_type &val) { val = val > 0.0 ? val : 0.0; });
        // this->H = this->H - stepsz * (m_gamma*prevstepH + this->m_gradH * gnorm);
        // this->H.for_each([](MAT::elem_type &val) { val = val > 0.0 ? val : 0.0; });
        //reset them in case step isn't taken
        HtH = this->H.t() * this->H;
        WtW = this->W.t() * this->W;
        WtA = this->W.t() * this->A;
        HtS = this->H.t() * this->S;
      }

      // store the step...
      prevstepW = this->W - W_temp;
      prevstepH = this->H - H_temp;
      
      return obj_dec;
  }
 public:
  PGDJointNMF(const T1 &A, const T2 &S, int lowrank) 
      : JointNMF<T1, T2>(A, S, lowrank) {
    // Set all the other variables
    INFO << "Finished constructing" << std::endl;
  }
  PGDJointNMF(const T1 &A, const T2 &S, MAT W, MAT H) 
      : JointNMF<T1, T2>(A, S, W, H) {
    // Set all the other variables
    INFO << "Finished constructing with factors" << std::endl;
  }

  // NOTE: need to collect momentum parameter and implement line search...
  void computeNMF() {
    INFO << "A: " << arma::size(this->A) << std::endl;
    INFO << "S: " << arma::size(this->S) << std::endl;
    INFO << "W: " << arma::size(this->W) << std::endl;
    INFO << "H: " << arma::size(this->H) << std::endl;

    // Momentum variables
    prevstepW.zeros(this->W.n_rows, this->W.n_cols);
    prevstepH.zeros(this->H.n_rows, this->H.n_cols);

    int curitr = 0;

    // implement stale_gram and stale_matmul
    HtH = this->H.t() * this->H;
    WtW = this->W.t() * this->W;
    WtA = this->W.t() * this->A;
    HtS = this->H.t() * this->S;

    while (curitr < this->num_iterations()) {
      // Update W
      // tic();

      //TODO: figure out whether we want to add in regularization
      // this->applyReg(this->regW(), &HtH);
      // this->applyReg(this->regH(), &WtW);

      gradW();
      gradH();

      this->computeObjectiveError(this->W, this->H, WtA, HtS, WtW, HtH);
      this->printObjective(curitr);

      bool objective_decrease = applyGradient();

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
   * PGD version of JointNMF. Saves W, and H.
   * @param[in] outfname: Prefix for the output files
   */
  void saveOutput(std::string outfname) {
    std::string Wfname  = outfname + "_W";
    std::string Hfname  = outfname + "_H";

    this->W.save(Wfname, arma::raw_ascii);
    this->H.save(Hfname, arma::raw_ascii);
  }

  /// Sets the momentum parameter
  void gamma(const double b) { this->m_gamma = b; }
  // Returns the momentum parameter
  double gamma() { return this->m_gamma; }
};  // class PGDJointNMF

}  // namespace planc

#endif  // NMF_PGDJNMF_HPP_
