/* Copyright 2016 Ramakrishnan Kannan */

#ifndef DISTNMF_DISTALS_HPP_
#define DISTNMF_DISTALS_HPP_
#include "distnmf/aunmf.hpp"

/**
 * Unconstrained least squares. Should match the SVD
 * objective error.  
 */

namespace planc {

template <class INPUTMATTYPE>
class DistALS : public DistAUNMF<INPUTMATTYPE> {
 protected:
  /**
   * update W given HtH and AHt
   * AHtij is of size \f$ k \times \frac{globalm}/{p} \f$.
   * this->W is of size \f$ \frac{globalm}{p} \times k \f$
   * this->HtH is of size kxk
  */
  void updateW() {
    this->Wt = arma::solve(arma::trimatl(this->HtH), this->AHtij);
    this->W = this->Wt.t();
    DISTPRINTINFO("ALS::updateW::HtH::" << PRINTMATINFO(this->HtH)
                                        << "::AHtij::"
                                        << PRINTMATINFO(this->AHtij)
                                        << "::W::" << PRINTMATINFO(this->W));
    DISTPRINTINFO("ALS::updateW::HtH::" << norm(this->HtH, "fro") << "::AHtij::"
                                        << norm(this->AHtij, "fro")
                                        << "::W::" << norm(this->W, "fro"));
  }
  /**
   * updateH given WtAij and WtW
   *  WtAij is of size \f$k \times \frac{globaln}{p} \f$
   * this->H is of size \f$ \frac{globaln}{p} \times k\f$
   * this->WtW is of size kxk
   */
  void updateH() {
    this->Ht = arma::solve(arma::trimatl(this->WtW), this->WtAij);
    this->H = this->Ht.t();
    DISTPRINTINFO("ALS::updateH::WtW::" << PRINTMATINFO(this->WtW)
                                        << "::WtAij::"
                                        << PRINTMATINFO(this->WtAij)
                                        << "::H::" << PRINTMATINFO(this->H));
    DISTPRINTINFO("ALS::updateH::WtW::" << norm(this->WtW, "fro") << "::WtAij::"
                                        << norm(this->WtAij, "fro")
                                        << "::H::" << norm(this->H, "fro"));
  }

 public:
  DistALS(const INPUTMATTYPE& input, const MAT& leftlowrankfactor,
          const MAT& rightlowrankfactor, const MPICommunicator& communicator,
          const int numkblks)
      : DistAUNMF<INPUTMATTYPE>(input, leftlowrankfactor, rightlowrankfactor,
                                communicator, numkblks) {
    PRINTROOT("DistALS() constructor successful");
  }
};

}  // namespace planc

#endif  // DISTNMF_DISTALS_HPP_
