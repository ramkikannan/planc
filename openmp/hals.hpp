/* Copyright 2016 Ramakrishnan Kannan */
#ifndef OPENMP_HALS_HPP
#define OPENMP_HALS_HPP
template <class T>
class HALSNMF: public NMF<T> {
  private:
    // Not happy with this design. However to avoid computing At again and again
    // making this as private variable.
    T At;
    fmat WtW;
    fmat HtH;
    fmat WtA;
    fmat HtAt;

    /*
     * Collected statistics are
     * iteration Htime Wtime totaltime normH normW densityH densityW relError
     */
    void allocateMatrices() {
        WtW = zeros<fmat>(this->k, this->k);
        HtH = zeros<fmat>(this->k, this->k);
        WtA = zeros<fmat>(this->n, this->k);
        HtAt = zeros<fmat>(this->m, this->k);
    }
    void freeMatrices() {
        this->At.clear();
        WtW.clear();
        HtH.clear();
        WtA.clear();
        HtAt.clear();
    }
  public:
    HALSNMF(const T &A, int lowrank): NMF<T>(A, lowrank) {
    }
    HALSNMF(const T &A, const fmat &llf, const fmat &rlf) :
        NMF<T>(A, llf, rlf) {
    }
    void computeNMF() {
        int currentIteration = 0;
        double t1, t2;
        this->At = this->A.t();
        INFO << "computed transpose At=" << PRINTMATINFO(this->At) << endl;
        while (currentIteration < this->num_iterations()) {
            tic();
            // update W;
            tic();
            HtAt = this->H.t() * this->At;
            HtH = this->H.t() * this->H;
            INFO << "starting W Prereq for " << " took=" << toc()
                 << PRINTMATINFO(HtH) << PRINTMATINFO(HtAt) << endl;
            tic();
            for (int x = 0; x < this->k; x++) {
                // fvec Wx = W(:,x) + (AHt(:,x)-W*HHt(:,x))/HHtDiag(x);
                float normConst;
                // W(:,i) = W(:,i) * HHt_reg(i,i) + AHt(:,i) - W * HHt_reg(:,i);
                fvec Wx = this->W.col(x) * HtH(x, x) +
                          (((HtAt.row(x)).t()) - (this->W * (HtH.col(x))));
                fixNumericalError<fvec>(&Wx);
                Wx = Wx / norm(Wx);
                this->W.col(x) = Wx;
            }
            INFO << "Completed W ("
                 << currentIteration << "/" << this->num_iterations() << ")"
                 << " time =" << toc() << endl;

            // update H
            tic();
            WtA = this->W.t() * this->A;
            WtW = this->W.t() * this->W;
            INFO << "starting H Prereq for " << " took=" << toc()
                 << PRINTMATINFO(WtW) << PRINTMATINFO(WtA) << endl;
            // to avoid divide by zero error.
            tic();
            for (int x = 1; x < this->k; x++) {
                float normConst;
                // H(i,:) = max(H(i,:) + WtA(i,:) - WtW_reg(i,:) * H,epsilon);
                fvec Hx = this->H.col(x) +
                          (((WtA.row(x)).t()) - (this->H * (WtW.col(x))));
                fixNumericalError<fvec>(&Hx);
                this->H.col(x) = Hx;
            }
            INFO << "Completed H ("
                 << currentIteration << "/" << this->num_iterations() << ")"
                 << " time =" << toc() << endl;
            INFO << "Completed It ("
                 << currentIteration << "/" << this->num_iterations() << ")"
                 << " time =" << toc() << endl;
            this->computeObjectiveError();
            INFO << "Completed it = " << currentIteration << " HALSERR="
                 << this->objective_err / this->normA << endl;
            currentIteration++;
        }
        this->computeObjectiveError();
        INFO << "Completed it = " << currentIteration << " HALSERR="
             << this->objective_err / this->normA << endl;
    }
    ~HALSNMF() {
    }
};
#endif
