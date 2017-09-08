#include <armadillo>
#include <iostream>
#include "ntf_utils.hpp"
#include "ncpfactors.hpp"
#include "tensor.hpp"
#include "auntf.hpp"
#include "ntf_utils.h"

// ntf input_order input_low_rank_k output_low_rank_k num_iterations

void parseDimensions(const char *input, UVEC *dim) {
    stringstream ss(input);
    string s;
    int i = 0;
    float temp;
    while (getline(ss, s, ' ')) {
        temp = ::atof(s.c_str());
        (*dim)(i) = temp;
        i++;
    }
}

int main(int argc, char* argv[]) {
    int test_order = atoi(argv[1]);
    // int low_rank = 1;
    UVEC dimensions(test_order);
    MAT *mttkrps = new MAT[test_order];
    // UVEC dimensions(4);
    parseDimensions(argv[2], &dimensions);
    // dimensions(3) = 6;
    PLANC::NCPFactors cpfactors(dimensions, atoi(argv[3]));
    cpfactors.normalize();
    cpfactors.print();
    PLANC::Tensor my_tensor = cpfactors.rankk_tensor();
    PLANC::ntfalgo ntfupdalgo = PLANC::NTF_BPP;
    PLANC::AUNTF auntf(my_tensor, atoi(argv[4]), ntfupdalgo);
    std::cout << "init factors" << std::endl << "--------------" << std::endl;
    auntf.ncp_factors().print();
    // std::cout << "input tensor" << std::endl << "--------------" << std::endl;
    // my_tensor.print();
    auntf.num_it(atoi(argv[5]));
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
    solution.normalize();
    solution.print();
}