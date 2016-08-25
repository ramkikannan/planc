/* Copyright 2016 Ramakrishnan Kannan */
#ifndef MPI_DISTHALS_HPP_
#define MPI_DISTHALS_HPP_

#include "aunmf.hpp"

template<class INPUTMATTYPE>
class DistHALS : public DistAUNMF<INPUTMATTYPE> {
  private:
    FMAT HWtW;
    FMAT WHtH;
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
            //W(:,i) = max(W(:,i) * HHt_reg(i,i) + AHt(:,i) - W * HHt_reg(:,i),epsilon);
            FVEC updWi = this->W.col(i) * this->HtH(i, i)
                         + ((this->AHtij.row(i)).t() - this->W * this->HtH.col(i));
#ifdef MPI_VERBOSE
            DISTPRINTINFO("b4 fixNumericalError::" << endl <<  updWi);
#endif
            fixNumericalError<FVEC>(&updWi);
#ifdef MPI_VERBOSE
            DISTPRINTINFO("after fixNumericalError::" << endl << updWi);
#endif
            //W(:,i) = W(:,i)/norm(W(:,i));
            double normWi = norm(updWi);
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
            FVEC updHi = this->H.col(i) +
                         ((this->WtAij.row(i)).t() - this->H * this->WtW.col(i));
#ifdef MPI_VERBOSE
            DISTPRINTINFO("b4 fixNumericalError::" << endl << updHi);
#endif
            fixNumericalError<FVEC>(&updHi);
#ifdef MPI_VERBOSE
            DISTPRINTINFO("after fixNumericalError::" << endl << updHi);
#endif
            this->H.col(i) = updHi;
        }
        this->Ht = this->H.t();
    }
  public:
    DistHALS(const INPUTMATTYPE &input, const FMAT &leftlowrankfactor,
             const FMAT &rightlowrankfactor, const MPICommunicator& communicator):
        DistAUNMF<INPUTMATTYPE>(input, leftlowrankfactor,
                                rightlowrankfactor, communicator) {
        WHtH.zeros(this->globalm() / this->m_mpicomm.size(), this->k);
        HWtW.zeros(this->globaln() / this->m_mpicomm.size(), this->k);
    }
};

#endif  // MPI_DISTHALS_HPP_