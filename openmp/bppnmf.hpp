/*
 * BPPNMF.hpp
 *
 *  Created on: Dec 19, 2013
 *      Author: ramki
 */

#ifndef BPPNMF_HPP_
#define BPPNMF_HPP_

#include <stdio.h>
#include "nmf.hpp"
#include "bppnnls.hpp"
#include <omp.h>
#include "utils.hpp"
#include "hals.hpp"
#ifdef MKL_FOUND
#include <mkl.h>
#else
#include <lapacke.h>
#endif
//#include <cblas.h>

#define ONE_THREAD_MATRIX_SIZE 2000

template <class T>
class BPPNMF: public NMF<T>
{
private:
    T At;
    //designed as if W is given and H is found. The transpose is the other problem.
    void updateOtherGivenOneMultipleRHS(T &input, fmat &given, fmat &othermat, char worh)
    {
        double t1, t2;
        UINT numThreads = (input.n_cols / ONE_THREAD_MATRIX_SIZE) + 1;
        tic();
        fmat givent = given.t();
        mat giventGiven(this->k, this->k), giventInput(this->k, input.n_cols);
        //This is WtW
        giventGiven = conv_to<mat>::from(givent * given);
        //This is WtA
        giventInput = conv_to<mat>::from(givent * input);
        givent.clear();
        t2 = toc();
        INFO << "starting " << worh << ". Prereq for " << worh << " took=" << t2 << " NumThreads=" <<
             numThreads << " giventGiven=" << giventGiven.n_rows << "x" << giventGiven.n_cols <<
             " giventInput=" << giventInput.n_rows << "x" << giventInput.n_cols << endl;
        tic();
        //todo : change it to dynamic schedule
        #pragma omp parallel for schedule(dynamic)
        for (UINT i = 0; i < numThreads; i++)
        {
            UINT spanStart = i * ONE_THREAD_MATRIX_SIZE;
            UINT spanEnd = (i + 1) * ONE_THREAD_MATRIX_SIZE - 1;
            if (spanEnd > input.n_cols - 1)
            {
                spanEnd = input.n_cols - 1;
            }
            //if it is exactly divisible, then the last iteration is unnecessary.
            BPPNNLS<mat, vec> *subProblem;
            if (spanStart <= spanEnd)
            {
                if (spanStart == spanEnd)
                {
                    subProblem = new BPPNNLS<mat, vec>(giventGiven, (vec)giventInput.col(spanStart), true);
                }
                else //if (spanStart < spanEnd)
                {
                    subProblem = new BPPNNLS<mat, vec>(giventGiven, (mat)giventInput.cols(spanStart, spanEnd), true);
                }
#ifdef _VERBOSE
                INFO << "Scheduling " << worh << " start=" << spanStart << ", end=" << spanEnd << ", tid=" << omp_get_thread_num() << endl;
#endif
                tic();
                subProblem->solveNNLS();
                t2 = toc();
#ifdef _VERBOSE
                INFO << "completed " << worh << " start=" << spanStart << ", end=" << spanEnd << ", tid=" << omp_get_thread_num() << " cpu=" << sched_getcpu() << " time taken=" << t2 << " num_iterations()=" << numIter << endl;
#endif
                if (spanStart == spanEnd)
                {
                    frowvec solVec = conv_to<frowvec>::from(subProblem->getSolutionVector().t());
                    othermat.row(i) = solVec;
                }
                else //if (spanStart < spanEnd)
                {
                    othermat.rows(spanStart, spanEnd) = conv_to<fmat>::from(subProblem->getSolutionMatrix().t());
                }
                subProblem->clear();
                delete subProblem;
            }
        }
        double totalH2 = toc();
        INFO << worh << " total time taken :" << totalH2  << endl;
        giventGiven.clear();
        giventInput.clear();
    }

public:
    BPPNMF(T A, int lowrank): NMF<T>(A, lowrank)
    {

    }
    BPPNMF(T A, fmat &llf, fmat &rlf) : NMF<T>(A, llf, rlf)
    {
    }
    void computeNMFSingleRHS()
    {
        int currentIteration = 0;
        T At = this->A.t();
        //while(currentIteration<this->num_iterations() && norm(this->A-this->W*this->H.t(),"fro")>0.0000001 )
        //objective_err;
        this->computeObjectiveErr();
        while (currentIteration < this->num_iterations() && this->objectiveErr > CONV_ERR)
        {
#ifdef COLLECTSTATS
            this->collectStats(currentIteration);
#endif
            //solve for H given W;
            fmat Wt = this->W.t();
            mat WtW = conv_to<mat>::from(Wt * this->W);
            mat WtA = conv_to<mat>::from(Wt * this->A);
            Wt.clear();
            {
                #pragma omp parallel for
                //int i=251;
                for (UINT i = 0; i < this->n; i++)
                {
                    BPPNNLS<mat, vec> *subProblemforH = new BPPNNLS<mat, vec>(WtW, (vec)WtA.col(i), true);
#ifdef _VERBOSE
                    INFO << "Initialized subproblem and calling solveNNLS for H(" << i << "/" << this->n << ")";
#endif
                    tic();
                    int numIter = subProblemforH->solveNNLS();
                    double t2 = toc();
#ifdef _VERBOSE
                    INFO << subProblemforH->getSolutionVector();
#endif
                    this->H.row(i) = conv_to<fvec>::from(subProblemforH->getSolutionVector().t());
                    INFO << "Comp H(" << i << "/" << this->n << ") of it=" << currentIteration << " time taken=" << t2 << " num_iterations()=" << numIter << endl;
                }
            }
#ifdef _VERBOSE
            INFO << "H: at it = " << currentIteration << endl << this->H;
#endif
//#pragma omp parallel
            {
                //clear previous allocations.
                WtW.clear();
                WtA.clear();
                fmat Ht = this->H.t();
                mat HtH = conv_to<mat>::from(Ht * this->H);
                mat HtAt = conv_to<mat>::from(Ht * At);
                Ht.clear();
                //solve for W given H;
                #pragma omp parallel for
                for (UINT i = 0; i < this->m; i++)
                {
                    BPPNNLS<mat, vec> *subProblemforW = new BPPNNLS<mat, vec>(HtH, (vec)HtAt.col(i), true);
#ifdef _VERBOSE
                    INFO << "Initialized subproblem and calling solveNNLS for W(" << i << "/" << this->m << ")";
#endif
                    tic();
                    int numIter = subProblemforW->solveNNLS();
                    double t2 = toc();
#ifdef _VERBOSE
                    INFO << subProblemforW->getSolutionVector();
#endif

                    this->W.row(i) = conv_to<fvec>::from(subProblemforW->getSolutionVector().t());
                    INFO << "Comp W(" << i << "/" << this->n << ") of it=" << currentIteration << " time taken=" << t2 << " num_iterations()=" << numIter << endl;
                }
                HtH.clear();
                HtAt.clear();
            }
#ifdef _VERBOSE
            INFO << "W: at it = " << currentIteration << endl << this->W;
#endif
#ifdef COLLECTSTATS
            //INFO << "iteration = " << currentIteration << " currentObjectiveError=" << this->objective_err << endl;
#endif
            currentIteration++;
        }
    }
    void computeNMF()
    {
        int currentIteration = 0;
#ifdef COLLECTSTATS
        //this->objective_err;
#endif
        this->At = this->A.t(); //do it once
        //run hals once to get proper initializations
        HALSNMF<T> tempHals(this->A, this->W, this->H);
        tempHals.num_iterations(2);
        this->W = tempHals.getLeftLowRankFactor();
        this->H = tempHals.getRightLowRankFactor();
#ifdef BUILD_SPARSE
        INFO << "computed transpose At.m=" << this->At.n_rows << " At.n=" << this->At.n_cols << " nnz=" << this->At.n_nonzero << endl;
#else
        INFO << "computed transpose At.m=" << this->At.n_rows << " At.n=" << this->At.n_cols << endl;
#endif
        //while(currentIteration<this->num_iterations() && this->objectiveErr > CONV_ERR)
        INFO << "Starting BPP for num_iterations()=" << this->num_iterations() << endl;
        while (currentIteration < this->num_iterations())
        {
#ifdef COLLECTSTATS
            this->collectStats(currentIteration);
            this->stats(currentIteration + 1, 0) = currentIteration + 1;
#endif
            tic();
            tic();
            updateOtherGivenOneMultipleRHS(this->A, this->W, this->H, 'H');
            double totalH2 = toc();
            tic();
            updateOtherGivenOneMultipleRHS(this->At, this->H, this->W, 'W');
            double totalW2 = toc();
#ifdef COLLECTSTATS
            this->stats(currentIteration + 1, 1) = totalH2;  //end of H and start of W are almost same.
            this->stats(currentIteration + 1, 2) = totalW2; 

            this->stats(currentIteration + 1, 3) = toc() ;
#endif
            INFO << "completed it=" << currentIteration << " time taken = " << this->stats(currentIteration + 1, 3) << endl;
            INFO << "error:it = " << currentIteration << "bpperr =" << this->objective_err / this->normA << endl;
            currentIteration++;
        }
#ifdef COLLECTSTATS
        this->collectStats(currentIteration);
        INFO << "NMF Statistics:" << endl << this->stats << endl;
#endif

    }
    double getObjectiveError()
    {
        return this->objectiveErr;
    }

    /*
     * I dont like this function here. But this seems to be the
     * easy place for having it. This function really should have been
     * in BPPNNLS.hpp. It will take some time to refactor this.
     * Give, A and W, solve for H.
     */
    fmat solveScalableNNLS()
    {
        updateOtherGivenOneMultipleRHS(this->A, this->W, this->H, 'H');
        return this->H;
    }
    ~BPPNMF()
    {
        this->At.clear();
    }
};
#endif
