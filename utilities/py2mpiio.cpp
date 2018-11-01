/* Copyright Ramakrishnan Kannan 2018 */

#include "common/npyio.hpp"

/*
* For converting numpy array to distributed io
* Compile as g++ py2mpiio.cpp  -I/ccs/home/ramki/nmflibrary/ 
* -I$ARMADILLO_INCLUDE_DIR  -I/ccs/home/ramki/rhea/libraries/openblas/include/
*/

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << "Usage : py2mpiio inputfile.npy outputfilename" << std::endl;
    planc::NumPyArray npa;
    npa.load(argv[1]);
    npa.m_input_tensor->write(argv[2]);
  }
  return 0;
}