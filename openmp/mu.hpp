/* Copyright 2016 Ramakrishnan Kannan */
#ifndef OPENMP_MU_HPP
#define OPENMP_MU_HPP
template <class T>
class MUNMF: public NMF<T> {
  private:
    // Not happy with this design. However to avoid computing At again and again
    // making this as private variable.
    T At;
    fmat WtW;
    fmat HtH;
    fmat AtW;
    fmat AH;

    /*
     * Collected statistics are
     * iteration Htime Wtime totaltime normH normW densityH densityW relError
     */
    void allocateMatrices() {
        WtW = zeros<fmat>(this->k, this->k);
        HtH = zeros<fmat>(this->k, this->k);
        AtW = zeros<fmat>(this->n, this->k);
        AH = zeros<fmat>(this->m, this->k);
    }
    void freeMatrices() {
        this->At.clear();
        WtW.clear();
        HtH.clear();
        AtW.clear();
        AH.clear();
    }
  public:
    MUNMF(const T &A, int lowrank): NMF<T>(A, lowrank) {
        allocateMatrices();
    }
    MUNMF(const T &A, const fmat &llf, const fmat &rlf) : NMF<T>(A, llf, rlf) {
        allocateMatrices();
    }
    void computeNMF() {
        int currentIteration = 0;
        double t1, t2;
        this->At = this->A.t();
        INFO << "computed transpose At=" << PRINTMATINFO(this->At) << endl;
        while (currentIteration < this->numIterations) {
            tic();
            // update W;
            tic();
            AH = this->A * this->H;
            HtH = this->H.t() * this->H;
            INFO << "starting W Prereq for " << " took=" << toc()
                 << PRINTMATINFO(HtH) << PRINTMATINFO(AH) << endl;
            tic();
            // W = W.*AH./(W*HtH_reg + epsilon);
            this->W = (this->W % AH) / ((this->W * HtH) + EPSILON_1EMINUS16);
            INFO << "Completed W ("
                 << currentIteration << "/" << this->numIterations << ")"
                 << " time =" << toc() << endl;
            // update H
            tic();
            AtW = this->At * this->W;
            WtW = this->W.t() * this->W;
            INFO << "starting H Prereq for " << " took=" << toc();
                    << PRINTMATINFO(WtW) << PRINTMATINFO(WtA) << endl;
            // to avoid divide by zero error.
            tic();
            // H = H.*AtW./(WtW_reg*H + epsilon);
            this->H = (this->H % AtW) / (this->H * WtW + EPSILON_1EMINUS16);
            INFO << "Completed H ("
                 << currentIteration << "/" << this->numIterations << ")"
                 << " time =" << toc() << endl;
            INFO << "Completed It ("
                 << currentIteration << "/" << this->numIterations << ")"
                 << " time =" << toc() << endl;
            INFO << "Completed it = " << currentIteration << " MUERR="
                 << this->computeObjectiveError() / this->normA << endl;
            currentIteration++;
        }
        INFO << "Completed it = " << currentIteration << " MUERR="
             << this->computeObjectiveError() / this->normA << endl;
    }
    ~MUNMF() {
        freeMatrices();
    }
};
#endif