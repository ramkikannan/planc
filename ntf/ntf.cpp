#include "utils.h"
#include <armadillo>
#include <iostream>
#include "ntf_utils.hpp"
#include "ncpfactors.hpp"
#include "tensor.hpp"
#include "auntf.hpp"
#include "parsecommandline.hpp"

// ntf -d "2 3 4 5" -k 5 -t 20

int main(int argc, char* argv[]) {
    planc::ParseCommandLine pc(argc, argv);
    int test_modes = pc.num_modes();
    // int low_rank = 1;
    UVEC dimensions(test_modes);
    MAT *mttkrps = new MAT[test_modes];
    planc::NCPFactors cpfactors(pc.dimensions(), pc.lowrankk());
    cpfactors.normalize();
    cpfactors.print();
    planc::Tensor my_tensor = cpfactors.rankk_tensor();
    algotype ntfupdalgo = pc.lucalgo();
    planc::AUNTF auntf(my_tensor, pc.lowrankk(), ntfupdalgo);
    std::cout << "init factors" << std::endl << "--------------" << std::endl;
    auntf.ncp_factors().print();
    // std::cout << "input tensor" << std::endl << "--------------" << std::endl;
    // my_tensor.print();
    auntf.num_it(pc.iterations());
    auntf.computeNTF();
    planc::NCPFactors solution = auntf.ncp_factors();
    // std::cout << "input factors::" << std::endl;
    // for (int i = 0; i < test_modes; i++) {
    //     std::cout << cpfactors.factor(i);
    // }
    // std::cout << "output factors::" << std::endl;
    // for (int i = 0; i < test_modes; i++) {
    //     std::cout << solution.factor(i);
    // }
    solution.normalize();
    solution.print();
}