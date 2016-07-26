/* Copyright 2016 Ramakrishnan Kannan */
#ifndef MPI_DISTANLSBPP_HPP_
#define MPI_DISTANLSBPP_HPP_

#include "bppnnls.hpp"
#include "aunmf.hpp"

using namespace std;
using namespace arma;

template<class INPUTMATTYPE>
class DistANLSBPP : public DistAUNMF<INPUTMATTYPE> {
  private:
    mat tempHtH;
    mat tempWtW;
    mat tempAHtij;
    mat tempWtAij;
    frowvec localWnorm;
    frowvec Wnorm;

    void allocateMatrices() {
        this->tempHtH.zeros(this->k, this->k);
        this->tempWtW.zeros(this->k, this->k);
        this->tempAHtij.zeros(size(this->AHtij));
        this->tempWtAij.zeros(size(this->WtAij));
    }

  protected:
    // updateW given HtH and AHt
    void updateW() {
        tempHtH = conv_to<mat>::from(this->HtH);
        tempAHtij = conv_to<mat>::from(this->AHtij);
        BPPNNLS<mat, vec> subProblem(tempHtH, tempAHtij, true);
        subProblem.solveNNLS();
        this->Wt = conv_to<fmat>::from(subProblem.getSolutionMatrix());
        fixNumericalError<fmat>(&(this->Wt));
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
        tempWtW = conv_to<mat>::from(this->WtW);
        tempWtAij = conv_to<mat>::from(this->WtAij);
        BPPNNLS<mat, vec> subProblem1(tempWtW, tempWtAij, true);
        subProblem1.solveNNLS();
        this->Ht = conv_to<fmat>::from(subProblem1.getSolutionMatrix());
        fixNumericalError<fmat>(&(this->Ht));
        this->H = this->Ht.t();
    }

  public:
    DistANLSBPP(const INPUTMATTYPE &input, const fmat &leftlowrankfactor,
                const fmat &rightlowrankfactor, const MPICommunicator& communicator):
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
}; // class DistANLSBPP2D

#endif  // MPI_DISTANLSBPP_HPP_
