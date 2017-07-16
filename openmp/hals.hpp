/* Copyright 2016 Ramakrishnan Kannan */
#ifndef OPENMP_HALS_HPP_
#define OPENMP_HALS_HPP_
template <class T>
class HALSNMF: public NMF<T> {
  private:
    // Not happy with this design. However to avoid computing At again and again
    // making this as private variable.
    T At;
    FMAT WtW;
    FMAT HtH;
    FMAT WtA;
    FMAT AH;

    /*
     * Collected statistics are
     * iteration Htime Wtime totaltime normH normW densityH densityW relError
     */
    void allocateMatrices() {
        WtW = arma::zeros<FMAT >(this->k, this->k);
        HtH = arma::zeros<FMAT >(this->k, this->k);
        WtA = arma::zeros<FMAT >(this->n, this->k);
        AH = arma::zeros<FMAT >(this->m, this->k);
    }
    void freeMatrices() {
        this->At.clear();
        WtW.clear();
        HtH.clear();
        WtA.clear();
        AH.clear();
    }
  public:
    HALSNMF(const T &A, int lowrank): NMF<T>(A, lowrank) {
        this->normalize_by_W();
        allocateMatrices();
    }
    HALSNMF(const T &A, const FMAT &llf, const FMAT &rlf) :
        NMF<T>(A, llf, rlf) {
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
            INFO << "starting H Prereq for " << " took=" << toc()
                 << PRINTMATINFO(WtW) << PRINTMATINFO(WtA) << std::endl;
            // to avoid divide by zero error.
            tic();
            float normConst;
            FVEC Hx;
            for (int x = 0; x < this->k; x++) {
                // H(i,:) = max(H(i,:) + WtA(i,:) - WtW_reg(i,:) * H,epsilon);
                Hx = this->H.col(x) +
                     (((WtA.row(x)).t()) - (this->H * (WtW.col(x))));
                fixNumericalError<FVEC>(&Hx);
                normConst = norm(Hx);
                if (normConst != 0) {
                    this->H.col(x) = Hx;
                }
            }
            INFO << "Completed H ("
                 << currentIteration << "/" << this->num_iterations() << ")"
                 << " time =" << toc() << std::endl;
            // update W;
            tic();
            AH = this->A * this->H;
            HtH = this->H.t() * this->H;
            INFO << "starting W Prereq for " << " took=" << toc()
                 << PRINTMATINFO(HtH) << PRINTMATINFO(AH) << std::endl;
            tic();
            FVEC Wx;
            for (int x = 0; x < this->k; x++) {
                // FVEC Wx = W(:,x) + (AHt(:,x)-W*HHt(:,x))/HHtDiag(x);

                // W(:,i) = W(:,i) * HHt_reg(i,i) + AHt(:,i) - W * HHt_reg(:,i);
                Wx = (this->W.col(x) * HtH(x, x)) +
                     (((AH.col(x))) - (this->W * (HtH.col(x))));
                fixNumericalError<FVEC>(&Wx);
                normConst = norm(Wx);
                if (normConst != 0) {
                    Wx = Wx / normConst;
                    this->W.col(x) = Wx;
                }
            }

            INFO << "Completed W ("
                 << currentIteration << "/" << this->num_iterations() << ")"
                 << " time =" << toc() << std::endl;

            INFO << "Completed It ("
                 << currentIteration << "/" << this->num_iterations() << ")"
                 << " time =" << toc() << std::endl;
            this->computeObjectiveError();
            INFO << "Completed it = " << currentIteration << " HALSERR="
                 << sqrt(this->objective_err) / this->normA << std::endl;
            currentIteration++;
        }
    }
    ~HALSNMF() {
    }
};
#endif  // OPENMP_HALS_HPP_
