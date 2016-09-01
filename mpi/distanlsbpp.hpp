/* Copyright 2016 Ramakrishnan Kannan */
#ifndef MPI_DISTANLSBPP_HPP_
#define MPI_DISTANLSBPP_HPP_

#include "bppnnls.hpp"
#include "aunmf.hpp"

using namespace std;

template<class INPUTMATTYPE>
class DistANLSBPP : public DistAUNMF<INPUTMATTYPE> {
  private:
    MAT tempHtH;
    MAT tempWtW;
    MAT tempAHtij;
    MAT tempWtAij;
    FROWVEC localWnorm;
    FROWVEC Wnorm;

    void allocateMatrices() {
        this->tempHtH.zeros(this->k, this->k);
        this->tempWtW.zeros(this->k, this->k);
        this->tempAHtij.zeros(size(this->AHtij));
        this->tempWtAij.zeros(size(this->WtAij));
    }

  protected:
    // updateW given HtH and AHt
    void updateW() {
        tempHtH = arma::conv_to<MAT>::from(this->HtH);
        tempAHtij = arma::conv_to<MAT>::from(this->AHtij);
        BPPNNLS<MAT, VEC > subProblem(tempHtH, tempAHtij, true);
        subProblem.solveNNLS();
        this->Wt = arma::conv_to<FMAT >::from(subProblem.getSolutionMatrix());
        fixNumericalError<FMAT >(&(this->Wt));
        this->W = this->Wt.t();
        // localWnorm = sum(this->W % this->W);
        // tic();
        // MPI_Allreduce(localWnorm.memptr(), Wnorm.memptr(), this->k, MPI_FLOAT,
        //               MPI_SUM, MPI_COMM_WORLD);
        // double temp = toc();
        // this->time_stats.allgather_duration(temp);
        // for (int i = 0; i < this->k; i++) {
        //     this->W.col(i) = this->W.col(i) / sqrt(Wnorm(i));
        // }
        // this->Wt = this->W.t();
    }
    // updateH given WtW and WtA
    void updateH() {
        tempWtW = arma::conv_to<MAT>::from(this->WtW);
        tempWtAij = arma::conv_to<MAT>::from(this->WtAij);
        BPPNNLS<MAT, VEC > subProblem1(tempWtW, tempWtAij, true);
        subProblem1.solveNNLS();
        this->Ht = arma::conv_to<FMAT >::from(subProblem1.getSolutionMatrix());
        fixNumericalError<FMAT >(&(this->Ht));
        this->H = this->Ht.t();
    }

  public:
    DistANLSBPP(const INPUTMATTYPE &input, const FMAT &leftlowrankfactor,
                const FMAT &rightlowrankfactor,
                const MPICommunicator& communicator):
        DistAUNMF<INPUTMATTYPE>(input, leftlowrankfactor,
                                rightlowrankfactor, communicator) {
        localWnorm.zeros(this->k);
        Wnorm.zeros(this->k);
        PRINTROOT("DistANLSBPP() constructor successful");
    }

    ~DistANLSBPP() {
        /*
        tempHtH.clear();
        tempWtW.clear();
        tempAHtij.clear();
        tempWtAij.clear();
        */
    }
};  // class DistANLSBPP2D

#endif  // MPI_DISTANLSBPP_HPP_
