/* Copyright 2016 Ramakrishnan Kannan */

#ifndef DISTNMF_DISTAOADMM_HPP_
#define DISTNMF_DISTAOADMM_HPP_

#include "distnmf/aunmf.hpp"

template <class INPUTMATTYPE>
class DistAOADMM : public DistAUNMF<INPUTMATTYPE> {
 private:
  MAT tempHtH;
  MAT tempWtW;
  MAT tempAHtij;
  MAT tempWtAij;
  ROWVEC localWnorm;
  ROWVEC Wnorm;

  // Dual Variables
  MAT U;
  MAT Ut;
  MAT V;
  MAT Vt;

  // Auxiliary Variables
  MAT Haux;
  MAT Htaux;
  MAT Waux;
  MAT Wtaux;

  // Cholesky Variables
  MAT L;
  MAT Lt;
  MAT tempHtaux;
  MAT tempWtaux;

  // Hyperparameters
  double alpha, beta, tolerance;
  int admm_iter;

  void allocateMatrices() {
    this->tempHtH.zeros(this->k, this->k);
    this->tempWtW.zeros(this->k, this->k);
    this->tempAHtij.zeros(size(this->AHtij));
    this->tempWtAij.zeros(size(this->WtAij));

    // Normalise W, H
    this->W = normalise(this->W, 2, 1);
    this->Wt = this->W.t();
    this->H = normalise(this->H);
    this->Ht = this->H.t();

    // Dual Variables
    this->V.zeros(size(this->H));
    this->Vt = V.t();
    this->U.zeros(size(this->W));
    this->Ut = U.t();

    // Auxiliary Variables
    this->Waux.zeros(size(this->W));
    this->Wtaux = Waux.t();
    this->Haux.zeros(size(this->H));
    this->Htaux = Haux.t();

    // Hyperparameters
    alpha = 0.0;
    beta = 0.0;
    tolerance = 0.01;
    admm_iter = 5;

    this->L.zeros(this->k, this->k);
    this->Lt = this->L.t();
    this->tempWtaux.zeros(size(this->Wt));
    this->tempHtaux.zeros(size(this->Ht));
  }

 protected:
  // updateW given HtH and AHt
  void updateW() {
    // Calculate modified Gram Matrix
    // tempHtH = arma::conv_to<MAT >::from(this->HtH);
    tempHtH = this->HtH;
    alpha = trace(tempHtH) / this->k;
    alpha = alpha > 0 ? alpha : 0.01;
    tempHtH.diag() += alpha;
    L = arma::conv_to<MAT>::from(arma::chol(tempHtH, "lower"));
    Lt = L.t();

    bool stop_iter = false;

    // Start ADMM loop from here
    for (int i = 0; i < admm_iter && !stop_iter; i++) {
      this->Waux = this->W;

      // tempAHtij = arma::conv_to<MAT >::from(this->AHtij);
      tempAHtij = this->AHtij;
      tempAHtij = tempAHtij + (alpha * (this->Wt + this->Ut));

      // Solve least squares
      // tempWtaux = arma::conv_to<MAT >::from(
      tempWtaux = arma::solve(arma::trimatl(L), tempAHtij);
      Wtaux =
          arma::conv_to<MAT>::from(arma::solve(arma::trimatu(Lt), tempWtaux));

      // Update W
      // this->Wt = arma::conv_to<MAT >::from(Wtaux);
      this->Wt = Wtaux;
      fixNumericalError<MAT>(&(this->Wt), EPSILON_1EMINUS16);
      this->Wt = this->Wt - this->Ut;
      this->Wt.for_each(
          [](MAT::elem_type &val) { val = val > 0.0 ? val : 0.0; });
      this->W = this->Wt.t();

      // Update Dual Variable
      this->Ut = this->Ut + this->Wt - this->Wtaux;
      this->U = this->Ut.t();

      // Check stopping criteria
      double r = norm(this->Wt - this->Wtaux, "fro");
      r *= r;
      double globalr;
      mpitic();
      MPI_Allreduce(&r, &globalr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      double temp = mpitoc();
      this->time_stats.communication_duration(temp);
      this->time_stats.allreduce_duration(temp);
      globalr = sqrt(globalr);

      double s = norm(this->W - this->Waux, "fro");
      s *= s;
      double globals;
      mpitic();
      MPI_Allreduce(&s, &globals, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      temp = mpitoc();
      globals = sqrt(globals);

      double normW = norm(this->W, "fro");
      normW *= normW;
      double globalnormW;
      mpitic();
      MPI_Allreduce(&normW, &globalnormW, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      temp = mpitoc();
      globalnormW = sqrt(globalnormW);

      double normU = norm(this->U, "fro");
      normU *= normU;
      double globalnormU;
      mpitic();
      MPI_Allreduce(&normU, &globalnormU, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      temp = mpitoc();
      globalnormU = sqrt(globalnormU);

      if (globalr < (tolerance * globalnormW) &&
          globals < (tolerance * globalnormU))
        stop_iter = true;
    }
  }
  // updateH given WtW and WtA
  void updateH() {
    // Calculate the Gram Matrix
    tempWtW = arma::conv_to<MAT>::from(this->WtW);
    beta = trace(tempWtW) / this->k;
    beta = beta > 0 ? beta : 0.01;
    tempWtW.diag() += beta;
    // L = arma::conv_to<MAT >::from(arma::chol(tempWtW, "lower"));
    L = arma::chol(tempWtW, "lower");
    Lt = L.t();

    bool stop_iter = false;

    // Start ADMM loop from here
    for (int i = 0; i < admm_iter && !stop_iter; i++) {
      this->Haux = this->H;

      // tempWtAij = arma::conv_to<MAT >::from(this->WtAij);
      tempWtAij = this->WtAij;
      tempWtAij = tempWtAij + (beta * (this->Ht + this->Vt));

      // Solve least squares
      // tempHtaux = arma::conv_to<MAT >::from(
      tempHtaux = arma::solve(arma::trimatl(L), tempWtAij);
      // Htaux = arma::conv_to<MAT >::from(
      Htaux = arma::solve(arma::trimatu(Lt), tempHtaux);
      // Update H
      // this->Ht = arma::conv_to<MAT >::from(Htaux);
      this->Ht = Htaux;
      fixNumericalError<MAT>(&(this->Ht), EPSILON_1EMINUS16);
      this->Ht = this->Ht - this->Vt;
      this->Ht.for_each(
          [](MAT::elem_type &val) { val = val > 0.0 ? val : 0.0; });
      this->H = this->Ht.t();

      // Update Dual Variable
      this->Vt = this->Vt + this->Ht - this->Htaux;
      this->V = this->Vt.t();

      // Check stopping criteria
      double r = norm(this->Ht - this->Htaux, "fro");
      r *= r;
      double globalr;
      mpitic();
      MPI_Allreduce(&r, &globalr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      double temp = mpitoc();
      this->time_stats.communication_duration(temp);
      this->time_stats.allreduce_duration(temp);
      globalr = sqrt(globalr);

      double s = norm(this->H - this->Haux, "fro");
      s *= s;
      double globals;
      mpitic();
      MPI_Allreduce(&s, &globals, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      temp = mpitoc();
      globals = sqrt(globals);

      double normH = norm(this->H, "fro");
      normH *= normH;
      double globalnormH;
      mpitic();
      MPI_Allreduce(&normH, &globalnormH, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      temp = mpitoc();
      globalnormH = sqrt(globalnormH);

      double normV = norm(this->V, "fro");
      normV *= normV;
      double globalnormV;
      mpitic();
      MPI_Allreduce(&normV, &globalnormV, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      temp = mpitoc();
      globalnormV = sqrt(globalnormV);

      if (globalr < (tolerance * globalnormH) &&
          globals < (tolerance * globalnormV))
        stop_iter = true;
    }
  }

 public:
  DistAOADMM(const INPUTMATTYPE &input, const MAT &leftlowrankfactor,
             const MAT &rightlowrankfactor, const MPICommunicator &communicator,
             const int numkblks)
      : DistAUNMF<INPUTMATTYPE>(input, leftlowrankfactor, rightlowrankfactor,
                                communicator, numkblks) {
    localWnorm.zeros(this->k);
    Wnorm.zeros(this->k);
    allocateMatrices();
    PRINTROOT("DistAOADMM() constructor successful");
  }

  ~DistAOADMM() {
    /*
    tempHtH.clear();
    tempWtW.clear();
    tempAHtij.clear();
    tempWtAij.clear();
    */
  }
};  // class DistAOADMM2D

#endif  // DISTNMF_DISTAOADMM_HPP_
