/* Copyright 2016 Ramakrishnan Kannan */
#ifndef MPI_DISTMU_HPP_
#define MPI_DISTMU_HPP_
#include "aunmf.hpp"

template<class INPUTMATTYPE>
class DistMU : public DistAUNMF<INPUTMATTYPE> {
  FMAT HWtW;
  FMAT WHtH;
  FROWVEC localWnorm;
  FROWVEC Wnorm;

 protected:
  // update W given HtH and AHt
  void updateW() {
    // AHtij is of size k*(globalm/p).
    // this->W is of size (globalm/p)xk
    // this->HtH is of size kxk
    // w_ij = w_ij .* (AH)_ij/(WHtH)_ij
    // Here ij is the element of W matrix.
    WHtH = (this->W * this->HtH) + EPSILON;
#ifdef MPI_VERBOSE
    DISTPRINTINFO("::WHtH::" << endl << this->WHtH);
#endif
    this->W = (this->W % this->AHtij.t()) / WHtH;
    DISTPRINTINFO("MU::updateW::HtH::" << norm(this->HtH, "fro")\
                  << "::WHtH::" << norm(WHtH, "fro")\
                  << "::AHtij::" << norm(this->AHtij, "fro") \
                  << "::W::" << norm(this->W, "fro"));
    /*localWnorm = sum(this->W % this->W);
    mpitic();
    MPI_Allreduce(localWnorm.memptr(), Wnorm.memptr(), this->k, MPI_FLOAT,
                  MPI_SUM, MPI_COMM_WORLD);
    double temp = mpitoc();
    this->time_stats.allgather_duration(temp);
    for (int i = 0; i < this->k; i++) {
      if (Wnorm(i) > 1) {
        float norm_const = sqrt(Wnorm(i));
        this->W.col(i) = this->W.col(i) / norm_const;
        //this->H.col(i) = norm_const * this->H.col(i);
      }
    }*/
    this->Wt = this->W.t();
  }
  void updateH() {
    // WtAij is of size k*(globaln/p)
    // this->H is of size (globaln/p)xk
    // this->WtW is of size kxk
    // h_ij = h_ij .* WtAij.t()/(HWtW)_ij
    // Here ij is the element of H matrix.
    HWtW = this->H * this->WtW + EPSILON;
    this->H = (this->H % this->WtAij.t()) / HWtW;
#ifdef MPI_VERBOSE
    DISTPRINTINFO("::HWtW::" << endl << HWtW);
#endif
    // fixNumericalError<FMAT>(&this->H);
    DISTPRINTINFO("MU::updateH::WtW::" << norm(this->WtW, "fro")\
                  << "::HWtW::" << norm(HWtW, "fro")\
                  << "::WtAij::" << norm(this->WtAij, "fro")\
                  << "::H::" << norm(this->H, "fro"));
    this->Ht = this->H.t();
  }

 public:
  DistMU(const INPUTMATTYPE &input, const FMAT &leftlowrankfactor,
         const FMAT &rightlowrankfactor, const MPICommunicator& communicator):
    DistAUNMF<INPUTMATTYPE>(input, leftlowrankfactor,
                            rightlowrankfactor, communicator) {
    WHtH.zeros(this->globalm() / this->m_mpicomm.size(), this->k);
    HWtW.zeros(this->globaln() / this->m_mpicomm.size(), this->k);
    localWnorm.zeros(this->k);
    Wnorm.zeros(this->k);
  }
};
#endif  // MPI_DISTMU_HPP_
