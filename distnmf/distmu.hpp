/* Copyright 2016 Ramakrishnan Kannan */

#ifndef DISTNMF_DISTMU_HPP_
#define DISTNMF_DISTMU_HPP_
#include "distnmf/aunmf.hpp"

/**
 * Distributed MU factorization.
 * Offers implementation for the pure virtual function
 * updateW and updateH based on MU.
 */

namespace planc {

template <class INPUTMATTYPE>
class DistMU : public DistAUNMF<INPUTMATTYPE> {
  MAT HWtW;
  MAT WHtH;
  ROWVEC localWnorm;
  ROWVEC Wnorm;

 protected:
  /**
   * update W given HtH and AHt
   * AHtij is of size \f$ k \times \frac{globalm}/{p}\f$.
   * this->W is of size \f$\frac{globalm}{p} \times k\f$
   * this->HtH is of size kxk
   * \f$w_{ij} = w_{ij} .* \frac{(AH)_{ij}}{(WH^TH)_{ij}}\f$
   * Here ij is the element of W matrix.
   */
  void updateW() {
    WHtH = (this->W * this->HtH) + EPSILON;
#ifdef MPI_VERBOSE
    DISTPRINTINFO("::WHtH::" << endl << this->WHtH);
#endif  // ifdef MPI_VERBOSE
    this->W = (this->W % this->AHtij.t()) / WHtH;
    DISTPRINTINFO("MU::updateW::HtH::"
                  << PRINTMATINFO(this->HtH) << "::WHtH::" << PRINTMATINFO(WHtH)
                  << "::AHtij::" << PRINTMATINFO(this->AHtij)
                  << "::W::" << PRINTMATINFO(this->W));
    DISTPRINTINFO("MU::updateW::HtH::"
                  << norm(this->HtH, "fro") << "::WHtH::" << norm(WHtH, "fro")
                  << "::AHtij::" << norm(this->AHtij, "fro")
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
  /**
   * updateH given WtAij and WtW
   * WtAij is of size \f$k \times \frac{globaln}{p} \f$
   * this->H is of size \f$ \frac{globaln}{p} \times k\f$
   * this->WtW is of size kxk
   * \f$h_{ij} = \frac{h_{ij} .* WtAij.t()}{(HW^TW)_{ij}}\f$
   * Here ij is the element of H matrix.
   */  
  void updateH() {
    HWtW = this->H * this->WtW + EPSILON;
    this->H = (this->H % this->WtAij.t()) / HWtW;
#ifdef MPI_VERBOSE
    DISTPRINTINFO("::HWtW::" << endl << HWtW);
#endif  // ifdef MPI_VERBOSE

    // fixNumericalError<MAT>(&this->H);
    DISTPRINTINFO("MU::updateH::WtW::"
                  << PRINTMATINFO(this->WtW) << "::HWtW::" << PRINTMATINFO(HWtW)
                  << "::WtAij::" << PRINTMATINFO(this->WtAij)
                  << "::H::" << PRINTMATINFO(this->H));
    DISTPRINTINFO("MU::updateH::WtW::"
                  << norm(this->WtW, "fro") << "::HWtW::" << norm(HWtW, "fro")
                  << "::WtAij::" << norm(this->WtAij, "fro")
                  << "::H::" << norm(this->H, "fro"));
    this->Ht = this->H.t();
  }

 public:
  DistMU(const INPUTMATTYPE& input, const MAT& leftlowrankfactor,
         const MAT& rightlowrankfactor, const MPICommunicator& communicator,
         const int numkblks)
      : DistAUNMF<INPUTMATTYPE>(input, leftlowrankfactor, rightlowrankfactor,
                                communicator, numkblks) {
    WHtH.zeros(this->globalm() / this->m_mpicomm.size(), this->k);
    HWtW.zeros(this->globaln() / this->m_mpicomm.size(), this->k);
    localWnorm.zeros(this->k);
    Wnorm.zeros(this->k);
    PRINTROOT("DistMU() constructor successful");
  }
};

}  // namespace planc

#endif  // DISTNMF_DISTMU_HPP_
