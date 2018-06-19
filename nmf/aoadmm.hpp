/* Copyright 2016 Ramakrishnan Kannan */

#ifndef NMF_AOADMM_HPP_
#define NMF_AOADMM_HPP_

template <class T>
class AOADMMNMF : public NMF<T> {
 private:
  // Not happy with this design. However to avoid computing At again and again
  // making this as private variable.
  T At;
  MAT WtW;
  MAT HtH;
  MAT WtA;
  MAT AH;

  // Dual Variables
  MAT U;
  MAT V;

  // Auxiliary/Temporary Variables
  MAT Htaux;
  MAT tempHtaux;
  MAT H0;
  MAT Wtaux;
  MAT tempWtaux;
  MAT W0;
  MAT L;

  // Hyperparameters
  double alpha, beta, tolerance;
  int admm_iter;

  /*
   * Collected statistics are
   * iteration Htime Wtime totaltime normH normW densityH densityW relError
   */
  void allocateMatrices() {
    WtW = arma::zeros<MAT>(this->k, this->k);
    HtH = arma::zeros<MAT>(this->k, this->k);
    WtA = arma::zeros<MAT>(this->n, this->k);
    AH = arma::zeros<MAT>(this->m, this->k);

    // Dual Variables
    U.zeros(size(this->W));
    V.zeros(size(this->H));

    // Auxiliary/Temporary Variables
    Htaux.zeros(size(this->H.t()));
    H0.zeros(size(this->H));
    tempHtaux.zeros(size(this->H.t()));
    Wtaux.zeros(size(this->W.t()));
    W0.zeros(size(this->W));
    tempWtaux.zeros(size(this->W.t()));
    L.zeros(this->k, this->k);

    // Hyperparameters
    alpha = 0.0;
    beta = 0.0;
    tolerance = 0.01;
    admm_iter = 5;
  }
  void freeMatrices() {
    this->At.clear();
    WtW.clear();
    HtH.clear();
    WtA.clear();
    AH.clear();
  }

 public:
  AOADMMNMF(const T &A, int lowrank) : NMF<T>(A, lowrank) {
    this->normalize_by_W();
    allocateMatrices();
  }
  AOADMMNMF(const T &A, const MAT &llf, const MAT &rlf) : NMF<T>(A, llf, rlf) {
    this->normalize_by_W();
    allocateMatrices();
  }
  void computeNMF() {
    int currentIteration = 0;
    double t1, t2;
    this->At = this->A.t();
    INFO << "computed transpose At=" << PRINTMATINFO(this->At) << std::endl;
    while (currentIteration < this->num_iterations()) {
      tic();
      // update H
      tic();
      WtA = this->W.t() * this->A;
      WtW = this->W.t() * this->W;
      beta = trace(WtW) / this->k;
      beta = beta > 0 ? beta : 0.01;
      WtW.diag() += beta;

      INFO << "starting H Prereq for "
           << " took=" << toc() << PRINTMATINFO(WtW) << PRINTMATINFO(WtA)
           << std::endl;
      // to avoid divide by zero error.
      tic();
      L = arma::chol(WtW, "lower");

      bool stop_iter = false;

      // Start ADMM loop from here
      for (int i = 0; i < admm_iter && !stop_iter; i++) {
        H0 = this->H;
        tempHtaux =
            arma::solve(arma::trimatl(L), WtA + (beta * (this->H.t() + V.t())));
        Htaux = arma::solve(arma::trimatu(L.t()), tempHtaux);

        this->H = Htaux.t();
        fixNumericalError<MAT>(&(this->H), EPSILON_1EMINUS16);
        this->H = this->H - V;
        this->H.for_each(
            [](MAT::elem_type &val) { val = val > 0.0 ? val : 0.0; });
        V = V + this->H - Htaux.t();

        // Check stopping criteria
        double r = norm(this->H - Htaux.t(), "fro");
        double s = norm(this->H - H0, "fro");
        double normH = norm(this->H, "fro");
        double normV = norm(V, "fro");

        if (r < (tolerance * normH) && s < (tolerance * normV))
          stop_iter = true;
      }

      INFO << "Completed H (" << currentIteration << "/"
           << this->num_iterations() << ")"
           << " time =" << toc() << std::endl;

      // update W;
      tic();
      AH = this->A * this->H;
      HtH = this->H.t() * this->H;
      alpha = trace(HtH) / this->k;
      alpha = alpha > 0 ? alpha : 0.01;
      HtH.diag() += alpha;

      INFO << "starting W Prereq for "
           << " took=" << toc() << PRINTMATINFO(HtH) << PRINTMATINFO(AH)
           << std::endl;
      tic();
      L = arma::chol(HtH, "lower");

      stop_iter = false;

      // Start ADMM loop from here
      for (int i = 0; i < admm_iter && !stop_iter; i++) {
        W0 = this->W;
        tempWtaux = arma::solve(arma::trimatl(L),
                                AH.t() + alpha * (this->W.t() + U.t()));
        Wtaux = arma::solve(arma::trimatu(L.t()), tempWtaux);

        this->W = Wtaux.t();
        fixNumericalError<MAT>(&(this->W), EPSILON_1EMINUS16);
        this->W = this->W - U;
        this->W.for_each(
            [](MAT::elem_type &val) { val = val > 0.0 ? val : 0.0; });

        U = U + this->W - Wtaux.t();

        // Check stopping criteria
        double r = norm(this->W - Wtaux.t(), "fro");
        double s = norm(this->W - W0, "fro");
        double normW = norm(this->W, "fro");
        double normU = norm(U, "fro");

        if (r < (tolerance * normW) && s < (tolerance * normU))
          stop_iter = true;
      }

      INFO << "Completed W (" << currentIteration << "/"
           << this->num_iterations() << ")"
           << " time =" << toc() << std::endl;

      INFO << "Completed It (" << currentIteration << "/"
           << this->num_iterations() << ")"
           << " time =" << toc() << std::endl;
      this->computeObjectiveError();
      INFO << "Completed it = " << currentIteration
           << " AOADMMERR=" << sqrt(this->objective_err) / this->normA
           << std::endl;
      currentIteration++;
    }
  }
  ~AOADMMNMF() {}
};

#endif  // NMF_AOADMM_HPP_
