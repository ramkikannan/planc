/* Copyright 2016 Ramakrishnan Kannan */
#ifndef COMMON_NMF_HPP_
#define COMMON_NMF_HPP_
#include <assert.h>
#include <string>
#include "utils.hpp"

// #ifndef _VERBOSE
// #define _VERBOSE 1;
// #endif

#define NUM_THREADS 4
#define CONV_ERR 0.000001
#define NUM_STATS 9


// #ifndef COLLECTSTATS
// #define COLLECTSTATS 1
// #endif


// T must be a either an instance of MAT or sp_MAT
template <class T>
class NMF {
  protected:
    // input MATrix of size mxn
    T A;
    // low rank factors with size mxk and nxk respectively.
    FMAT W, H;
    FMAT Winit,Hinit;
    UINT m, n, k;
    /*
     * Collected statistics are
     * iteration Htime Wtime totaltime normH normW densityH densityW relError
     */
    MAT stats;
    double objective_err;
    double normA, normW, normH;
    double densityW, densityH;
    bool cleared;
    int m_num_iterations;
    std::string input_file_name;

    void collectStats(int iteration) {
        this->normW = norm(this->W, "fro");
        this->normH = norm(this->H, "fro");
        UVEC nnz = find(this->W > 0);
        this->densityW = nnz.size() / (this->m * this->k);
        nnz.clear();
        nnz = find(this->H > 0);
        this->densityH = nnz.size() / (this->m * this->k);
        this->stats(iteration, 4) = this->normH;
        this->stats(iteration, 5) = this->normW;
        this->stats(iteration, 6) = this->densityH;
        this->stats(iteration, 7) = this->densityW;
        this->stats(iteration, 8) = this->objective_err;
    }

  private:
    void otherInitializations() {
        this->stats.zeros();
        this->cleared = false;
        this->normA = norm(this->A, "fro");
        this->m_num_iterations = 20;
        this->objective_err = 1000000000000;
        this->stats.resize(m_num_iterations + 1, NUM_STATS);
    }

  public:
    NMF(const T &input, const unsigned int rank) {
        this->A = input;
        this->m = A.n_rows;
        this->n = A.n_cols;
        this->k = rank;
        this->W = arma::randu<FMAT>(m, k);
        this->H = arma::randu<FMAT>(n, k);
        // make the random MATrix positive
        // absMAT<FMAT>(W);
        // absMAT<FMAT>(H);
        // other intializations
        this->otherInitializations();
        cout << "NMF.hpp:constructor NMF(A,k) over!!!" << endl;
    }
    NMF(const T &input, const FMAT &leftlowrankfactor,
        const FMAT &rightlowrankfactor) {
        assert(leftlowrankfactor.n_cols == rightlowrankfactor.n_cols);
        this->A = input;
        this->W = leftlowrankfactor;
        this->H = rightlowrankfactor;
        this->Winit = leftlowrankfactor;
        this->Hinit = rightlowrankfactor;
        this->m = A.n_rows;
        this->n = A.n_cols;
        this->k = W.n_cols;
        // other initializations
        this->otherInitializations();
        INFO << "NMF.hpp::constructor over" << endl;
    }

    virtual void computeNMF() = 0;

    FMAT getLeftLowRankFactor() {
        return W;
    }

    FMAT getRightLowRankFactor() {
        return H;
    }
    /*
    * A is mxn
    * Wr is mxk will be overwritten. Must be passed with values of W.
    * Hr is nxk will be overwritten. Must be passed with values of H.
    * All MATrices are in row major forMAT
    * ||A-WH||_F^2 = over all nnz (a_ij - w_i h_j)^2 +
                 over all zeros (w_i h_j)^2
               = over all nnz (a_ij - w_i h_j)^2 +
                 ||WH||_F^2 - over all nnz (w_i h_j)^2

    */
#ifdef BUILD_SPARSE
    void computeObjectiveError() {
        // 1. over all nnz (a_ij - w_i h_j)^2
        // 2. over all nnz (w_i h_j)^2
        // 3. Compute R of W ahd L of H through QR
        // 4. use sgemm to compute RL
        // 5. use slange to compute ||RL||_F^2
        // 6. return nnzsse+nnzwh-||RL||_F^2
        tic();
        float nnzsse = 0;
        float nnzwh = 0;
        FMAT Rw(this->k, this->k);
        FMAT Rh(this->k, this->k);
        FMAT Qw(this->m, this->k);
        FMAT Qh(this->n, this->k);
        FMAT RwRh(this->k, this->k);
        // #pragma omp parallel for reduction (+ : nnzsse,nnzwh)
        for (UWORD jj = 1; jj <= this->A.n_cols; jj++) {
            UWORD startIdx = this->A.col_ptrs[jj - 1];
            UWORD endIdx = this->A.col_ptrs[jj];
            UWORD col = jj - 1;
            float nnzssecol = 0;
            float nnzwhcol = 0;
            for (UWORD ii = startIdx; ii < endIdx; ii++) {
                UWORD row = this->A.row_indices[ii];
                float tempsum = 0;
                for (UWORD kk = 0; kk < k; kk++) {
                    tempsum += (this->W(row, kk) * this->H(col, kk));
                }
                nnzwhcol += tempsum * tempsum;
                nnzssecol += (this->A.values[ii] - tempsum)
                             * (this->A.values[ii] - tempsum);
            }
            nnzsse += nnzssecol;
            nnzwh += nnzwhcol;
        }
        qr_econ(Qw, Rw, this->W);
        qr_econ(Qh, Rh, this->H);
        RwRh = Rw * Rh.t();
        float normWH = norm(RwRh, "fro");
        Rw.clear();
        Rh.clear();
        Qw.clear();
        Qh.clear();
        RwRh.clear();
        INFO << "error compute time " << toc() << endl;
        float fastErr = sqrt(nnzsse + (normWH * normWH - nnzwh));
        this->objective_err = fastErr;
        return (fastErr);
    }
#else
    void computeObjectiveError() {
        // (init.norm_A)^2 - 2*trace(H'*(A'*W))+trace((W'*W)*(H*H'))
        FMAT WtW = this->W.t() * this->W;
        FMAT HtH = this->H.t() * this->H;
        FMAT AtW = this->A.t() * this->W;
        this->objective_err = this->normA * this->normA
                              - 2 * trace(this->H.t() * AtW) + trace(WtW * HtH);
    }
#endif
    void computeObjectiveError(const T &At, const FMAT &WtW, const FMAT &HtH) {
        FMAT AtW = At * this->W;
        this->objective_err = this->normA * this->normA
                              - 2 * trace(this->H.t() * AtW) + trace(WtW * HtH);
    }
    void num_iterations(const int it) {this->m_num_iterations = it;}
    const int num_iterations() const {return m_num_iterations;}
    ~NMF() {
        clear();
    }
    void clear() {
        if (!this->cleared) {
            this->A.clear();
            this->W.clear();
            this->H.clear();
            this->stats.clear();
            this->cleared = true;
        }
    }
};
#endif // COMMON_NMF_HPP_
