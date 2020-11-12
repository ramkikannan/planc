/*Copyright 2016 Ramakrishnan Kannan*/

#ifndef NNLS_BPPNNLS_HPP_
#define NNLS_BPPNNLS_HPP_

#include <assert.h>
#include "nnls.hpp"
#include "utils.hpp"
#include <set>
#include <algorithm>
#include <iomanip>
#include "SortBooleanMatrix.hpp"

template <class MATTYPE, class VECTYPE>
class BPPNNLS : public NNLS<MATTYPE, VECTYPE> {
 public:
    BPPNNLS(MATTYPE input, VECTYPE rhs, bool prodSent = false):
        NNLS<MATTYPE, VECTYPE>(input, rhs, prodSent) {
    }
    BPPNNLS(MATTYPE input, MATTYPE RHS, bool prodSent = false) :
        NNLS<MATTYPE, VECTYPE>(input, RHS, prodSent) {
    }
    int solveNNLS() {
        int rcIterations = 0;
        if (this->k == 1) {
            rcIterations = solveNNLSOneRHS();
        } else {
            // k must be greater than 1 in this case.
            // we initialized k appropriately in the
            // constructor.
            rcIterations = solveNNLSMultipleRHS();
        }
        return rcIterations;
    }

 private:
    /**
     * This implementation is based on Algorithm 1 on Page 6 of paper
     * http://www.cc.gatech.edu/~hpark/papers/SISC_082117RR_Kim_Park.pdf.
     *
     * Special case of the multi RHS solver.
     */
    int solveNNLSOneRHS() {
        // Set the RHS matrix
        this->AtB.zeros(this->n, this->k);
        this->AtB.col(0) = this->Atb;

        // Initialize the solution matrix
        this->X.zeros(this->n, this->k);
        this->X.col(0) = this->x;

        // Call matrix method
        int iter = solveNNLSMultipleRHS();

        this->x = this->X.col(0);

        return iter;
    }

    /**
     * This is the implementation of Algorithm 2 at Page 8 of the paper
     * http:// www.cc.gatech.edu/~hpark/papers/SISC_082117RR_Kim_Park.pdf.
     * 
     * Based on the nnlsm_blockpivot subroutine from the MATLAB code 
     * associated with the paper.
     */
    int solveNNLSMultipleRHS() {
        UINT iter = 0;
        UINT MAX_ITERATIONS = this->n * 5;
        bool success = true;

        // Set the initial feasible solution
        MATTYPE Y = (this->AtA * this->X) - this->AtB;
        UMAT PassiveSet = (this->X > 0);

        int pbar = 3;
        UROWVEC P(this->k);
        P.fill(pbar);

        UROWVEC Ninf(this->k);
        Ninf.fill(this->n+1);

        UMAT NonOptSet  = (Y < 0) % (PassiveSet == 0);
        UMAT InfeaSet   = (this->X < 0) % PassiveSet;
        UROWVEC NotGood = arma::sum(NonOptSet) + arma::sum(InfeaSet);
        UROWVEC NotOptCols = (NotGood > 0);

        UWORD numNonOptCols = arma::accu(NotOptCols);
#ifdef _VERBOSE
        INFO << "Rank : " << arma::rank(this->AtA) << endl;
        INFO << "Condition : " << cond(this->AtA) << endl;
        INFO << "numNonOptCols : " << numNonOptCols;
#endif
        // Temporaries needed in loop
        UROWVEC Cols1 = NotOptCols;
        UROWVEC Cols2 = NotOptCols;
        UMAT PSetBits = NonOptSet;
        UMAT POffBits = InfeaSet;
        UMAT NotOptMask = arma::ones<UMAT>(arma::size(NonOptSet));

        while (numNonOptCols > 0) {
            iter++;

            if ((MAX_ITERATIONS > 0) && (iter > MAX_ITERATIONS)) {
                success = false;
                break;
            }

            Cols1 = NotOptCols % (NotGood < Ninf);
            Cols2 = NotOptCols % (NotGood >= Ninf) % (P >= 1);
            UROWVEC Cols3Ix = arma::conv_to<UROWVEC>::from(
                        arma::find(NotOptCols % (Cols1 == 0) % (Cols2 == 0)));

            // Columns that didn't increase number of infeasible variables
            if (!Cols1.empty()) {
                // P(Cols1) = pbar;,Ninf(Cols1) = NotGood(Cols1);
                P(arma::find(Cols1)).fill(pbar);
                Ninf(arma::find(Cols1)) = NotGood(arma::find(Cols1));

                // PassiveSet(NonOptSet & repmat(Cols1,n,1)) = true;
                PSetBits = NonOptSet;
                PSetBits.each_row() %= Cols1;
                PassiveSet(arma::find(PSetBits)).fill(1u);

                // PassiveSet(InfeaSet & repmat(Cols1,n,1)) = false;
                POffBits = InfeaSet;
                POffBits.each_row() %= Cols1;
                PassiveSet(arma::find(POffBits)).fill(0u);
            }

            // Columns that did increase number of infeasible variables but full
            // exchange is still allowed
            if (!Cols2.empty()) {
                // P(Cols2) = P(Cols2)-1;
                P(arma::find(Cols2)) -= 1;

                // PassiveSet(NonOptSet & repmat(Cols2,n,1)) = true;
                PSetBits = NonOptSet;
                PSetBits.each_row() %= Cols2;
                PassiveSet(arma::find(PSetBits)).fill(1u);

                // PassiveSet(InfeaSet & repmat(Cols2,n,1)) = false;
                POffBits = InfeaSet;
                POffBits.each_row() %= Cols2;
                PassiveSet(arma::find(POffBits)).fill(0u);
            }

            // Columns using backup rule
            if (!Cols3Ix.empty()) {
                UROWVEC::iterator citr;
                for (citr = Cols3Ix.begin(); citr !=  Cols3Ix.end(); ++citr) {
                    UWORD colidx = *citr;
                    UWORD rowidx = arma::max(arma::find(
                        NonOptSet.col(colidx) + InfeaSet.col(colidx)));
                    if (PassiveSet(rowidx, colidx) > 0) {
                        PassiveSet(rowidx, colidx) = 0u;
                    } else {
                        PassiveSet(rowidx, colidx) = 1u;
                    }
                }
            }

            UVEC NotOptColsIx = arma::find(NotOptCols);
            this->X.cols(NotOptColsIx) = solveNormalEqComb(this->AtA,
                                   this->AtB.cols(NotOptColsIx),
                                   PassiveSet.cols(NotOptColsIx));
            Y.cols(NotOptColsIx) = (this->AtA * this->X.cols(NotOptColsIx))
                             - this->AtB.cols(NotOptColsIx);
            // X(abs(X)<1e-12) = 0;
            fixAbsNumericalError<MATTYPE>(&this->X, EPSILON_1EMINUS12, 0.0);
            // Y(abs(Y)<1e-12) = 0;
            fixAbsNumericalError<MATTYPE>(&Y, EPSILON_1EMINUS12, 0.0);

            // NotOptMask = repmat(NotOptCols,n,1);
            NotOptMask.ones();
            NotOptMask.each_row() %= NotOptCols;

            NonOptSet  = (Y < 0) % (PassiveSet == 0);
            InfeaSet   = (this->X < 0) % PassiveSet;
            NotGood = arma::sum(NonOptSet) + arma::sum(InfeaSet);
            NotOptCols = (NotGood > 0);
            numNonOptCols = arma::accu(NotOptCols);
        }

        if (!success) {
            ERR << "BPP failed" << std::endl;
            exit(EXIT_FAILURE);
        }

        return iter;
    }

    /**
     * This function to support the step 10 of the algorithm 2.
     * This is implementation of the paper
     * Fast algorithm for the solution of large-scale non-negativity-constrained least squares problems
     * M. H. Van Benthem and M. R. Keenan, J. Chemometrics 2004; 18: 441-450
     * Motivated out of implementation from Jingu's solveNormalEqComb.m
     * 
     * @param[in] LHS of the system of size \f$n \times n\f$
     * @param[in] RHS of the system of size \f$n \times nrhs\f$
     * @param[in] Binary matrix of size \f$n \times nrhs\f$ representing the Passive Set
     */
    MATTYPE solveNormalEqComb(MATTYPE AtA, MATTYPE AtB, UMAT PassSet) {
        MATTYPE Z;
        UVEC anyZeros = arma::find(PassSet == 0);
        if (anyZeros.empty()) {
            // Everything is the in the passive set.
            Z = arma::solve(AtA, AtB, arma::solve_opts::likely_sympd);
        } else {
            UVEC Pv = arma::find(PassSet != 0);
            Z.resize(AtB.n_rows, AtB.n_cols);
            Z.zeros();
            UINT k1 = PassSet.n_cols;
            if (k1 == 1) {
                // Single column to solve for.
                Z(Pv) = arma::solve(AtA(Pv, Pv), AtB(Pv),
                                arma::solve_opts::likely_sympd);
            } else {
                // we have to group passive set columns that are same.
                // find the correlation matrix of passive set matrix.
                std::vector<UWORD> sortedIdx, beginIdx;
                computeCorrelationScore(PassSet, sortedIdx, beginIdx);

                // Go through the groups one at a time
                for (UINT i = 1; i < beginIdx.size(); i++) {
                    UWORD sortedBeginIdx = beginIdx[i - 1];
                    UWORD sortedEndIdx = beginIdx[i];

                    // Create submatrices of indices for solve.
                    UVEC samePassiveSetCols(std::vector<UWORD>
                                            (sortedIdx.begin() + sortedBeginIdx,
                                             sortedIdx.begin() + sortedEndIdx));
                    UVEC currentPassiveSet = arma::find(
                            PassSet.col(sortedIdx[sortedBeginIdx]) == 1);
#ifdef _VERBOSE
                    INFO << "samePassiveSetCols::" << std::endl
                         <<  samePassiveSetCols << std::endl;
                    INFO << "currPassiveSet::" << std::endl
                         << currentPassiveSet << std::endl;
                    INFO << "AtA::" << std::endl
                         << AtA(currentPassiveSet, currentPassiveSet)
                         << std::endl;
                    INFO << "AtB::" << std::endl
                         << AtB(currentPassiveSet, samePassiveSetCols)
                         << std::endl;
#endif
                    Z(currentPassiveSet, samePassiveSetCols) = arma::solve(
                            AtA(currentPassiveSet, currentPassiveSet),
                            AtB(currentPassiveSet, samePassiveSetCols),
                            arma::solve_opts::likely_sympd);
                }
            }
        }
#ifdef _VERBOSE
        INFO << "Returning mat Z:" << std::endl << Z;
#endif
        return Z;
    }

   /**
    * Passset is a binary matrix where every column represents
    * one datapoint. The objective is to returns a low triangular
    * correlation matrix with 1 if the strings are equal. Zero otherwise
    * 
    * @param[in] The binary matrix being grouped
    * @param[in] Reference to the array containing lexicographically sorted
    *            columns of the binary matrix
    * @param[in] Running indices of the grouped columns in the sorted index
    *            array
    */
    void computeCorrelationScore(UMAT &PassSet, std::vector<UWORD> &sortedIdx,
                                 std::vector<UWORD> &beginIndex) {
        SortBooleanMatrix<UMAT> sbm(PassSet);
        sortedIdx = sbm.sortIndex();
        BooleanArrayComparator<UMAT> bac(PassSet);
        uint beginIdx = 0;
        beginIndex.clear();
        beginIndex.push_back(beginIdx);
        for (uint i = 0; i < sortedIdx.size(); i++) {
            if (i == sortedIdx.size() - 1 ||
                    bac(sortedIdx[i], sortedIdx[i + 1]) == true) {
                beginIdx = i + 1;
                beginIndex.push_back(beginIdx);
            }
        }
    }
};

#endif  // NNLS_BPPNNLS_HPP_
