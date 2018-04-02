/* Copyright 2016 Ramakrishnan Kannan */
#ifndef MPI_DISTHALS_HPP_
#define MPI_DISTHALS_HPP_

#include "aunmf.hpp"

template<class INPUTMATTYPE>
class DistHALS : public DistAUNMF<INPUTMATTYPE>{
 protected:
  // emulating Jingu's code
  // https://github.com/kimjingu/nonnegfac-matlab/blob/master/nmf.m
  // function hals_iterSolver

  void updateW() {
    // AHtij is of size k*(globalm/p).
    // this->W is of size (globalm/p)xk
    // this->HtH is of size kxk
    // Eq 14(a) page 7 of JGO paper
    // W(:,i)=[W(:,i) + (AH(:,i)-W*HtH(:,i))/HtH(i,i)]_+
    // column normalize W_i
    // Here ij is the element of W matrix.
    for (int i = 0; i < this->k; i++) {
      // W(:,i) = max(W(:,i) * HHt_reg(i,i) + AHt(:,i) - W *
      // HHt_reg(:,i),epsilon);
      VEC updWi = this->W.col(i) * this->HtH(i, i)
                   + ((this->AHtij.row(i)).t() - this->W * this->HtH.col(i));
#ifdef MPI_VERBOSE
      DISTPRINTINFO("b4 fixNumericalError::" << endl <<  updWi);
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
      MPI_Allreduce(&normWi, &globalnormWi, 1, MPI_DOUBLE,
                    MPI_SUM, MPI_COMM_WORLD);
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

  void updateH() {
    // WtAij is of size k*(globaln/p)
    // this->H is of size (globaln/p)xk
    // this->WtW is of size kxk
    // Eq 14(b) page 7 of JGO paper
    // H(:,i) = H(:,i) + WtA(:,i) - H*WtW(:,i)
    // Here ij is the element of H matrix.
    for (int i = 0; i < this->k; i++) {
      // H(i,:) = max(H(i,:) + WtA(i,:) - WtW_reg(i,:) * H,epsilon);
      VEC updHi = this->H.col(i) +
                   ((this->WtAij.row(i)).t() - this->H * this->WtW.col(i));
#ifdef MPI_VERBOSE
      DISTPRINTINFO("b4 fixNumericalError::" << endl << updHi);
#endif // ifdef MPI_VERBOSE
      fixNumericalError<VEC>(&updHi);
#ifdef MPI_VERBOSE
      DISTPRINTINFO("after fixNumericalError::" << endl << updHi);
#endif // ifdef MPI_VERBOSE
      double normHi = arma::norm(updHi, 2);
      normHi *= normHi;
      double globalnormHi;
      mpitic();
      MPI_Allreduce(&normHi, &globalnormHi, 1, MPI_DOUBLE,
                    MPI_SUM, MPI_COMM_WORLD);
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
           const MAT& rightlowrankfactor,
           const MPICommunicator& communicator,
           const int numkblks) :
    DistAUNMF<INPUTMATTYPE>(input, leftlowrankfactor,
                            rightlowrankfactor, communicator, numkblks) {
  }
};

#endif // MPI_DISTHALS_HPP_
