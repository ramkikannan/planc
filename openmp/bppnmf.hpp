/* Copyright 2016 Ramakrishnan Kannan */
#ifndef OPENMP_BPPNMF_HPP_
#define OPENMP_BPPNMF_HPP_

#include <stdio.h>
#include <omp.h>
#include "nmf.hpp"
#include "bppnnls.hpp"
#include "utils.hpp"
#include "hals.hpp"
#ifdef MKL_FOUND
#include <mkl.h>
#else
#include <lapacke.h>
#endif

#define ONE_THREAD_MATRIX_SIZE 2000

template <class T>
class BPPNMF: public NMF<T> {
  private:
    T At;
    MAT giventGiven;
    // designed as if W is given and H is found.
    // The transpose is the other problem.
    void updateOtherGivenOneMultipleRHS(const T &input, const FMAT &given,
                                        char worh, FMAT *othermat) {
        double t1, t2;
        UINT numThreads = (input.n_cols / ONE_THREAD_MATRIX_SIZE) + 1;
        tic();
        FMAT givent = given.t();
        MAT  giventInput(this->k, input.n_cols);
        // This is WtW
        giventGiven = arma::conv_to<MAT >::from(givent * given);
        // This is WtA
        giventInput = arma::conv_to<MAT >::from(givent * input);
        givent.clear();
        t2 = toc();
        INFO << "starting " << worh << ". Prereq for " << worh
             << " took=" << t2 << " NumThreads=" << numThreads
             << PRINTMATINFO(giventGiven) << PRINTMATINFO(giventInput) << std::endl;
        tic();
        #pragma omp parallel for schedule(dynamic)
        for (UINT i = 0; i < numThreads; i++) {
            UINT spanStart = i * ONE_THREAD_MATRIX_SIZE;
            UINT spanEnd = (i + 1) * ONE_THREAD_MATRIX_SIZE - 1;
            if (spanEnd > input.n_cols - 1) {
                spanEnd = input.n_cols - 1;
            }
            // if it is exactly divisible, the last iteration is unnecessary.
            BPPNNLS<MAT, VEC > *subProblem;
            if (spanStart <= spanEnd) {
                if (spanStart == spanEnd) {
                    subProblem = new BPPNNLS<MAT, VEC >(giventGiven,
                                                        (VEC)giventInput.col(spanStart),
                                                        true);
                } else {  // if (spanStart < spanEnd)
                    subProblem = new BPPNNLS<MAT, VEC >(giventGiven,
                                                        (MAT)giventInput.cols(spanStart, spanEnd),
                                                        true);
                }
#ifdef _VERBOSE
                INFO << "Scheduling " << worh << " start=" << spanStart
                     << ", end=" << spanEnd << ", tid=" << omp_get_thread_num()
                     << std::endl;
#endif
                tic();
                subProblem->solveNNLS();
                t2 = toc();
#ifdef _VERBOSE
                INFO << "completed " << worh << " start="
                     << spanStart << ", end=" << spanEnd << ", tid="
                     << omp_get_thread_num()
                     << " cpu=" << sched_getcpu() << " time taken=" << t2
                     << " num_iterations()=" << numIter << std::endl;
#endif
                if (spanStart == spanEnd) {
                    FROWVEC solVec = arma::conv_to<FROWVEC>::from(
                                         subProblem->getSolutionVector().t());
                    (*othermat).row(i) = solVec;
                } else {  // if (spanStart < spanEnd)
                    (*othermat).rows(spanStart, spanEnd) = arma::conv_to<FMAT >::from(
                            subProblem->getSolutionMatrix().t());
                }
                subProblem->clear();
                delete subProblem;
            }
        }
        double totalH2 = toc();
        INFO << worh << " total time taken :" << totalH2  << std::endl;
        giventGiven.clear();
        giventInput.clear();
    }
  public:
    BPPNMF(T A, int lowrank): NMF<T>(A, lowrank) {
        giventGiven = arma::zeros<MAT >(lowrank, lowrank);
    }
    BPPNMF(const T A, const FMAT &llf, const FMAT &rlf) : NMF<T>(A, llf, rlf) {
    }
    void computeNMFSingleRHS() {
        int currentIteration = 0;
        T At = this->A.t();
        this->computeObjectiveErr();
        while (currentIteration < this->num_iterations()
                && this->objectiveErr > CONV_ERR) {
#ifdef COLLECTSTATS
            this->collectStats(currentIteration);
#endif
            // solve for H given W;
            FMAT Wt = this->W.t();
            MAT WtW = arma::conv_to<MAT >::from(Wt * this->W);
            MAT WtA = arma::conv_to<MAT >::from(Wt * this->A);
            Wt.clear();
            {
                #pragma omp parallel for
                // int i=251;
                for (UINT i = 0; i < this->n; i++) {
                    BPPNNLS<MAT, VEC > *subProblemforH = new BPPNNLS<MAT, VEC >(WtW,
                            (VEC)WtA.col(i), true);
#ifdef _VERBOSE
                    INFO << "Initialized subproblem and calling solveNNLS for "
                         << "H(" << i << "/" << this->n << ")";
#endif
                    tic();
                    int numIter = subProblemforH->solveNNLS();
                    double t2 = toc();
#ifdef _VERBOSE
                    INFO << subProblemforH->getSolutionVector();
#endif
                    this->H.row(i) = arma::conv_to<FVEC>::from(subProblemforH->getSolutionVector().t());
                    INFO << "Comp H(" << i << "/" << this->n << ") of it="
                         << currentIteration << " time taken=" << t2
                         << " num_iterations()=" << numIter << std::endl;
                }
            }
#ifdef _VERBOSE
            INFO << "H: at it = " << currentIteration << std::endl << this->H;
#endif
// #pragma omp parallel
            {
                // clear previous allocations.
                WtW.clear();
                WtA.clear();
                FMAT Ht = this->H.t();
                MAT HtH = arma::conv_to<MAT >::from(Ht * this->H);
                MAT HtAt = arma::conv_to<MAT >::from(Ht * At);
                Ht.clear();
                // solve for W given H;
                #pragma omp parallel for
                for (UINT i = 0; i < this->m; i++) {
                    BPPNNLS<MAT, VEC > *subProblemforW = new BPPNNLS<MAT, VEC >(HtH,
                            (VEC)HtAt.col(i), true);
#ifdef _VERBOSE
                    INFO << "Initialized subproblem and calling solveNNLS for "
                         << "W(" << i << "/" << this->m << ")";
#endif
                    tic();
                    int numIter = subProblemforW->solveNNLS();
                    double t2 = toc();
#ifdef _VERBOSE
                    INFO << subProblemforW->getSolutionVector();
#endif

                    this->W.row(i) = arma::conv_to<FVEC>::from(subProblemforW->getSolutionVector().t());
                    INFO << "Comp W(" << i << "/" << this->n << ") of it="
                         << currentIteration << " time taken=" << t2
                         << " num_iterations()=" << numIter << std::endl;
                }
                HtH.clear();
                HtAt.clear();
            }
#ifdef _VERBOSE
            INFO << "W: at it = " << currentIteration << std::endl << this->W;
#endif
#ifdef COLLECTSTATS
            // INFO << "iteration = " << currentIteration << " currentObjectiveError=" << this->objective_err << std::endl;
#endif
            currentIteration++;
        }
    }
    void computeNMF() {
        int currentIteration = 0;
#ifdef COLLECTSTATS
        // this->objective_err;
#endif
        this->At = this->A.t();  // do it once
        // run hals once to get proper initializations
        HALSNMF<T> tempHals(this->A, this->W, this->H);
        tempHals.num_iterations(2);
        this->W = tempHals.getLeftLowRankFactor();
        this->H = tempHals.getRightLowRankFactor();
        INFO << PRINTMATINFO(this->At);
#ifdef BUILD_SPARSE
        INFO << " nnz = " << this->At.n_nonzero << std::endl;
#endif
        INFO << "Starting BPP for num_iterations()="
             << this->num_iterations() << std::endl;
        while (currentIteration < this->num_iterations()) {
#ifdef COLLECTSTATS
            this->collectStats(currentIteration);
            this->stats(currentIteration + 1, 0) = currentIteration + 1;
#endif
            tic();
            tic();
            updateOtherGivenOneMultipleRHS(this->At, this->H, 'W', &(this->W));
            double totalW2 = toc();
            tic();
            updateOtherGivenOneMultipleRHS(this->A, this->W, 'H', &(this->H));
            double totalH2 = toc();

#ifdef COLLECTSTATS
            // end of H and start of W are almost same.
            this->stats(currentIteration + 1, 1) = totalH2;
            this->stats(currentIteration + 1, 2) = totalW2;

            this->stats(currentIteration + 1, 3) = toc();
#endif
            INFO << "completed it=" << currentIteration << " time taken = "
                 << this->stats(currentIteration + 1, 3) << std::endl;
            INFO << "error:it = " << currentIteration << "bpperr ="
                 << this->objective_err << std::endl;
            currentIteration++;
        }
#ifdef COLLECTSTATS
        this->collectStats(currentIteration);
        INFO << "NMF Statistics:" << std::endl << this->stats << std::endl;
#endif
    }
    double getObjectiveError() {
        return this->objectiveErr;
    }

    /*
     * I dont like this function here. But this seems to be the
     * easy place for having it. This function really should have been
     * in BPPNNLS.hpp. It will take some time to refactor this.
     * Given, A and W, solve for H.
     */
    FMAT solveScalableNNLS() {
        updateOtherGivenOneMultipleRHS(this->A, this->W, 'H', &(this->H));
        return this->H;
    }
    ~BPPNMF() {
        this->At.clear();
    }
};
#endif  // OPENMP_BPPNMF_HPP_
