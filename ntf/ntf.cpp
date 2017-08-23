#include <armadillo>
#include <iostream>
#include "ntf_utils.hpp"
#include "ncpfactors.hpp"
#include "tensor.hpp"
#include "auntf.hpp"
#include "ntf_utils.h"


int main(int argc, char* argv[]) {
    int test_order = 5;
    // int low_rank = 1;
    UVEC dimensions(test_order);
    MAT *mttkrps = new MAT[test_order];
    // UVEC dimensions(4);
    for (int i = 0; i < test_order; i++) {
        dimensions(i) = i + 2;
    }
    // dimensions(3) = 6;
    PLANC::NCPFactors cpfactors(dimensions, atoi(argv[1]));
    cpfactor.normalize();
    cpfactors.print();
    PLANC::Tensor my_tensor = cpfactors.rankk_tensor();
    PLANC::ntfalgo ntfupdalgo = PLANC::NTF_BPP;
    PLANC::AUNTF auntf(my_tensor, atoi(argv[2]), ntfupdalgo);
    auntf.computeNTF();
    PLANC::NCPFactors solution = auntf.ncp_factors();
    // std::cout << "input factors::" << std::endl;
    // for (int i = 0; i < test_order; i++) {
    //     std::cout << cpfactors.factor(i);
    // }
    // std::cout << "output factors::" << std::endl;
    // for (int i = 0; i < test_order; i++) {
    //     std::cout << solution.factor(i);
    // }
    solution.print();
}