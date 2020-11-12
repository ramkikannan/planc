#ifndef DISTNMF_DISTR2_HPP_
#define DISTNMF_DISTR2_HPP_
#include "distnmf/aunmf.hpp"

/**
 * Distrubted Rank-2 AUNMF Factorization
 */

namespace planc {
  
  template <class INPUTMATTYPE>
  class DistR2 : public DistAUNMF<INPUTMATTYPE> {
    private:
      MAT Huv;
      MAT Wuv;
    protected:
      void updateW() {
        //HHt AHt
        MPITIC;
        update(this->HtH, this->AHtij, this->Wuv, this->Wt);
        this->W = this->Wt.t();
        double temp = MPITOC;
        this->reportTime(temp,"updateW::");
      }
      void updateH() {
        //WtW, (WtA)t
        MPITIC;
        update(this->WtW, this->WtAij, this->Huv, this->Ht);
        this->H = this->Ht.t();
        double temp = MPITOC;
        this->reportTime(temp,"updateH::");
      }

    private:
      void update(MAT& left, const MAT& right, MAT& uv, MAT& G) {
        MPITIC;
        G = solve(left, right, arma::solve_opts::fast + arma::solve_opts::allow_ugly);
        double temp = MPITOC;
        this->reportTime(temp,"G::");

        MPITIC;
        uv.row(0) = right.row(0) / left(0,0);
        uv.row(1) = right.row(1) / left(1,1);

        auto b0 = sqrt(left(0,0));
        auto b1 = sqrt(left(1,1));

        UVEC negs = find(any(G<0));

        uv.each_col( [&,b0,b1](VEC& b) {
            if (b0*b(0) >= b1*b(1)) {
              b(1) = 0;
            } else {
              b(0) = 0;
            }
          });
        
        G.cols(negs) = uv.cols(negs);
        temp = MPITOC;
        this->reportTime(temp,"UV::");
      }


    public:
      DistR2(const INPUTMATTYPE& input, const MAT& leftlowrankfactor, const MAT& rightlowrankfactor, const MPICommunicator& communicator, const int numkblks) : DistAUNMF<INPUTMATTYPE>(input, leftlowrankfactor, rightlowrankfactor, communicator, numkblks) {
        Wuv.zeros(2,this->W.n_rows);
        Huv.zeros(2,this->H.n_rows);
        
      }
  };
}
#endif
