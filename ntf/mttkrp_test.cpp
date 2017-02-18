/* Copyright 2017 Ramakrishnan Kannan */

#include <armadillo>
#include "ntf_utils.hpp"
#include "ncpfactors.hpp"
#include "tensor.hpp"
#include "utils.h"

int main(int argc, char* argv[]) {
    UVEC dimensions(3);
    // UVEC dimensions(4);
    dimensions(0) = 3;
    dimensions(1) = 4;
    dimensions(2) = 5;
    // dimensions(3) = 6;
    PLANC::NCPFactors cpfactors(dimensions, 2);
    cpfactors.print();
    FMAT krp_2 = cpfactors.krp_leave_out_one(2);
    cout << "krp" << endl << "------" << endl << krp_2;
    PLANC::Tensor my_tensor = cpfactors.rankk_tensor();
    cout << "input tensor" << endl << "--------" << endl;
    my_tensor.print();
    FMAT my_mttkrp_0(3, 2);
    FMAT my_mttkrp_1(4, 2);
    FMAT my_mttkrp_2(5, 2);
    mttkrp(0, my_tensor, cpfactors, &my_mttkrp_0);
    mttkrp(1, my_tensor, cpfactors, &my_mttkrp_1);
    mttkrp(2, my_tensor, cpfactors, &my_mttkrp_2);
    cout << "mttkrp 0" << endl << "---------" << endl << my_mttkrp_0;
    cout << "mttkrp 1" << endl << "---------" << endl << my_mttkrp_1;
    cout << "mttkrp n" << endl << "---------" << endl << my_mttkrp_2;
}
