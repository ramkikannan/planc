/* Copyright 2016 Ramakrishnan Kannan */

#ifndef DISTNMF_DISTALS_HPP_
#define DISTNMF_DISTALS_HPP_
#include "distnmf/aunmf.hpp"

namespace planc {

template <class INPUTMATTYPE>
class DistALS : public DistAUNMF<INPUTMATTYPE> {
 protected:
  // update W given HtH and AHt
  void updateW() {
    // AHtij is of size k*(globalm/p).
    // this->W is of size (globalm/p)xk
    // this->HtH is of size kxk
    // w_ij = w_ij .* (AH)_ij/(WHtH)_ij
    // Here ij is the element of W matrix.
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

  void updateH() {
    // WtAij is of size k*(globaln/p)
    // this->H is of size (globaln/p)xk
    // this->WtW is of size kxk
    // h_ij = h_ij .* WtAij.t()/(HWtW)_ij
    // Here ij is the element of H matrix.
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
