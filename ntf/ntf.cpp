/* Copyright Ramakrishnan Kannan 2017 */

#include <armadillo>
#include <iostream>
#include "auntf.hpp"
#include "ncpfactors.hpp"
#include "ntf_utils.hpp"
#include "parsecommandline.hpp"
#include "tensor.hpp"
#include "utils.h"

// ntf -d "2 3 4 5" -k 5 -t 20

int main(int argc, char* argv[]) {
  planc::ParseCommandLine pc(argc, argv);
  pc.parseplancopts();
  int test_modes = pc.num_modes();
  // int low_rank = 1;
  UVEC dimensions(test_modes);
  MAT* mttkrps = new MAT[test_modes];
  planc::Tensor my_tensor(pc.dimensions());
  INFO << "A::" << std::endl;
  my_tensor.print();
  // planc::NCPFactors cpfactors(pc.dimensions(), pc.lowrankk(), false);
  // cpfactors.normalize();
  // cpfactors.print();
  algotype ntfupdalgo = pc.lucalgo();
  planc::AUNTF auntf(my_tensor, pc.lowrankk(), ntfupdalgo);
  std::cout << "init factors" << std::endl << "--------------" << std::endl;
  auntf.ncp_factors().print();
  // std::cout << "input tensor" << std::endl << "--------------" << std::endl;
  // my_tensor.print();
  auntf.num_it(pc.iterations());
  if (pc.dim_tree()) {
    auntf.dim_tree(true);
  }
  auntf.computeNTF();
  auntf.ncp_factors().print();
  // std::cout << "input factors::" << std::endl;
  // for (int i = 0; i < test_modes; i++) {
  //     std::cout << cpfactors.factor(i);
  // }
  // std::cout << "output factors::" << std::endl;
  // for (int i = 0; i < test_modes; i++) {
  //     std::cout << solution.factor(i);
  // }
  // solution.normalize();
}