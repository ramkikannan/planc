1. copy libopenblas.a from openblas/lib directory to local.
2. module load matlab
3. Run mex -v CXXFLAGS="\$CXXFLAGS -std=c++11 -w" -I/ccs/home/ramki/rhea/libraries/armadillo-7.100.3/include/ matlabvsarma.cpp -l:libopenblas.a -lgomp -I/ccs/home/ramki/nmflibrary/openmp/ -I/ccs/home/ramki/nmflibrary/common/ -I/ccs/home/ramki/nmflibrary/nnls/ -I/ccs/home/ramki/rhea/libraries/openblas/include -lgfortran
