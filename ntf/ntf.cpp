/* Copyright Ramakrishnan Kannan 2017 */

#include <armadillo>
#include <iostream>
#include "common/ncpfactors.hpp"
#include "common/ntf_utils.hpp"
#include "common/parsecommandline.hpp"
#include "common/tensor.hpp"
#include "common/utils.h"
#include "ntf/ntfhals.hpp"
#include "ntf/ntfmu.hpp"
#include "ntf/ntfanlsbpp.hpp"
#include "ntf/ntfaoadmm.hpp"

// ntf -d "2 3 4 5" -k 5 -t 20

namespace planc {

class NTFDriver {
 public:
  template <class NTFTYPE>
  void callNTF(planc::ParseCommandLine pc) {
    int test_modes = pc.num_modes();
    UVEC dimensions(test_modes);
    Tensor my_tensor(pc.dimensions());
    NTFTYPE ntfsolver(my_tensor, pc.lowrankk(), pc.lucalgo());
    ntfsolver.num_it(pc.iterations());
    if (pc.dim_tree()) {
      ntfsolver.dim_tree(true);
    }
    ntfsolver.computeNTF();
    // ntfsolver.ncp_factors().print();
  }
  NTFDriver() {}

};  // class NTF Driver

}  // namespace planc

int main(int argc, char* argv[]) {
  planc::ParseCommandLine pc(argc, argv);
  pc.parseplancopts();
  planc::NTFDriver ntfd;
  switch (pc.lucalgo()) {
    case MU:
      ntfd.callNTF<planc::NTFMU>(pc);
      break;
    case HALS:
      ntfd.callNTF<planc::NTFHALS>(pc);
      break;
    case ANLSBPP:
      ntfd.callNTF<planc::NTFANLSBPP>(pc);
      break;
    case AOADMM:
      ntfd.callNTF<planc::NTFAOADMM>(pc);
      break;
    default:
      ERR << "Wrong algorithm choice. Quitting.." << pc.lucalgo()
          << std::endl;
  }
}
