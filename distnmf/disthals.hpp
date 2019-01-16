/* Copyright 2016 Ramakrishnan Kannan */

#ifndef DISTNMF_DISTHALS_HPP_
#define DISTNMF_DISTHALS_HPP_

#include "distnmf/aunmf.hpp"
/**
 * emulating Jingu's code
 * https://github.com/kimjingu/nonnegfac-matlab/blob/master/nmf.m
 * function hals_iterSolver
 */

namespace planc {

template <class INPUTMATTYPE>
class DistHALS : public DistAUNMF<INPUTMATTYPE> {
 protected:
  /**
   * AHtij is of size \f$ k \times \frac{globalm}/{p}\f$.
   * this->W is of size \f$\frac{globalm}{p} \times k\f$
   * this->HtH is of size kxk
   * Eq 14(a) page 7 of JGO paper
   * \f$W(:,i)=[W(:,i) + (AH(:,i)-WH^TH(:,i))/H^TH(i,i)]_+\f$
   * column normalize W_i
   */
  void updateW() {
    for (int i = 0; i < this->k; i++) {
      // W(:,i) = max(W(:,i) * HHt_reg(i,i) + AHt(:,i) - W *
      // HHt_reg(:,i),epsilon);
      VEC updWi = this->W.col(i) * this->HtH(i, i) +
                  ((this->AHtij.row(i)).t() - this->W * this->HtH.col(i));
#ifdef MPI_VERBOSE
      DISTPRINTINFO("b4 fixNumericalError::" << endl << updWi);
#endif  // ifdef MPI_VERBOSE
      fixNumericalError<VEC>(&updWi);
#ifdef MPI_VERBOSE
      DISTPRINTINFO("after fixNumericalError::" << endl << updWi);
#endif  // ifdef MPI_VERBOSE

      // W(:,i) = W(:,i)/norm(W(:,i));
      double normWi = arma::norm(updWi, 2);
      normWi *= normWi;
      double globalnormWi;
      mpitic();
      MPI_Allreduce(&normWi, &globalnormWi, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      double temp = mpitoc();
      this->time_stats.communication_duration(temp);
      this->time_stats.allreduce_duration(temp);

      if (globalnormWi > 0) {
        this->W.col(i) = updWi / sqrt(globalnormWi);
        this->H.col(i) = this->H.col(i) * sqrt(globalnormWi);
      }
    }
    this->Wt = this->W.t();
  }

  /**
   * WtAij is of size \f$k \times \frac{globaln}{p} \f$
   * this->H is of size \f$ \frac{globaln}{p} \times k\f$
   * this->WtW is of size kxk
   * Eq 14(b) page 7 of JGO paper
   * \f$ H(:,i) = H(:,i) + WtAij(:,i) - HW^TW(:,i)\f$
   * Here ij is the element of H matrix.
   */
  void updateH() {
    for (int i = 0; i < this->k; i++) {
      // H(i,:) = max(H(i,:) + WtA(i,:) - WtW_reg(i,:) * H,epsilon);
      VEC updHi = this->H.col(i) +
                  ((this->WtAij.row(i)).t() - this->H * this->WtW.col(i));
#ifdef MPI_VERBOSE
      DISTPRINTINFO("b4 fixNumericalError::" << endl << updHi);
#endif  // ifdef MPI_VERBOSE
      fixNumericalError<VEC>(&updHi);
#ifdef MPI_VERBOSE
      DISTPRINTINFO("after fixNumericalError::" << endl << updHi);
#endif  // ifdef MPI_VERBOSE
      double normHi = arma::norm(updHi, 2);
      normHi *= normHi;
      double globalnormHi;
      mpitic();
      MPI_Allreduce(&normHi, &globalnormHi, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      double temp = mpitoc();
      this->time_stats.communication_duration(temp);
      this->time_stats.allreduce_duration(temp);

      if (globalnormHi > 0) {
        this->H.col(i) = updHi;
      }
    }
    this->Ht = this->H.t();
  }

 public:
  DistHALS(const INPUTMATTYPE& input, const MAT& leftlowrankfactor,
           const MAT& rightlowrankfactor, const MPICommunicator& communicator,
           const int numkblks)
      : DistAUNMF<INPUTMATTYPE>(input, leftlowrankfactor, rightlowrankfactor,
                                communicator, numkblks) {
    PRINTROOT("DistHALS() constructor successful");
  }
};

}  // namespace planc

#endif  // DISTNMF_DISTHALS_HPP_
