/* Copyright Ramakrishnan Kannan 2017 */

#include <matrix.h>
#include <mex.h>
#include <armadillo>
#include <iostream>
#include "matlabarmautils.hpp"
#include "bppnmf.hpp"
#include "mu.hpp"


// call from matlab as W_arma, H_arma, WtW_arma, AtW_arma, HtH_arma, AH_arma]=
// matlabvsarma(A,W_init,H_init',20);

#ifndef MWSIZE_MAX
typedef int mwSize;
typedef int mwIndex;
typedef int mwSignedIndex;

#if (defined(_LP64) || defined(_WIN64)) && !defined(MX_COMPAT_32)
/* Currently 2^48 based on hardware limitations */
# define MWSIZE_MAX    281474976710655UL
# define MWINDEX_MAX   281474976710655UL
# define MWSINDEX_MAX  281474976710655L
# define MWSINDEX_MIN -281474976710655L
#else
# define MWSIZE_MAX    2147483647UL
# define MWINDEX_MAX   2147483647UL
# define MWSINDEX_MAX  2147483647L
# define MWSINDEX_MIN -2147483647L
#endif
#define MWSIZE_MIN    0UL
#define MWINDEX_MIN   0UL
#endif

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    // declare variables
    mxArray *a_in_m, *w_in_m, *h_in_m;
    // output variables
    mxArray *w_out_m, *h_out_m, *wtw_out_m, *hth_out_m, *ah_out_m, *atw_out_m;
    const mwSize *dims;
    double *a, *w_init, *h_init;
    int dimx, dimy, numdims;
    int i, j;
    int num_iterations;

    // associate inputs
    a_in_m = mxDuplicateArray(prhs[0]);
    //b_in_m = mxDuplicateArray(prhs[1]);
    w_in_m = mxDuplicateArray(prhs[1]);
    h_in_m = mxDuplicateArray(prhs[2]);
    num_iterations = mxGetScalar(prhs[3]);

    // figure out dimensions
    dims = mxGetDimensions(prhs[0]);
    numdims = mxGetNumberOfDimensions(prhs[0]);
    size_t arows = (int)dims[0]; size_t acols = (int)dims[1];
    dims = mxGetDimensions(prhs[1]);
    size_t wrows = (int)dims[0]; size_t wcols = (int)dims[1];

    std::cout << "A::" << arows << "x" << acols
              << "::W::" << wrows << "x" << wcols << std::endl;
    // associate outputs
    w_out_m = plhs[0] = mxCreateDoubleMatrix(wrows, wcols, mxREAL);
    h_out_m = plhs[1] = mxCreateDoubleMatrix(acols, wcols, mxREAL);
    wtw_out_m = plhs[2] = mxCreateDoubleMatrix(wcols, wcols, mxREAL);
    atw_out_m = plhs[3] = mxCreateDoubleMatrix(acols, wcols, mxREAL);
    hth_out_m = plhs[4] = mxCreateDoubleMatrix(wcols, wcols, mxREAL);
    ah_out_m = plhs[5] = mxCreateDoubleMatrix(arows, wcols, mxREAL);

    // associate pointers
    a = mxGetPr(a_in_m);
    w_init = mxGetPr(w_in_m);
    h_init = mxGetPr(h_in_m);

    MAT A = arma::zeros<FMAT>(arows, acols);
    MAT W = arma::zeros<FMAT>(wrows, wcols);
    MAT H = arma::zeros<FMAT>(acols, wcols);

    matlab2arma<FMAT>(arows, acols, a, &A);
    std::cout << "copied matlab2arma A::" << arma::norm(A, "fro") << std::endl;
    matlab2arma<FMAT>(wrows, wcols, w_init, &W);
    std::cout << "copied matlab2arma W::" << norm(W, "fro") << std::endl;
    matlab2arma<FMAT>(acols, wcols, h_init, &H);
    std::cout << "copied matlab2arma H::" << norm(H, "fro") << std::endl;
    BPPNMF<FMAT> nmfAlgorithm(A, W, H);
    nmfAlgorithm.num_iterations(num_iterations);
    nmfAlgorithm.computeNMF();


    arma2matlab<MAT>(wrows, wcols, nmfAlgorithm.getLeftLowRankFactor(),
                      mxGetPr(w_out_m));
    arma2matlab<MAT>(acols, wcols, nmfAlgorithm.getRightLowRankFactor(),
                      mxGetPr(h_out_m));
    /*std::cout << "A::" <<std::endl << A;
    std::cout << "W::" << std::endl << W;
    std::cout << "H::" << std::endl << H;
    std::cout << "WtW::" << std::endl << W.t()*W;
    std::cout << "HtH::" << std::endl << H.t()*H;
    std::cout << "AtW::" << std::endl << A.t()*W;
    std::cout << "AH::" << std::endl << A*H;*/
    arma2matlab<MAT>(wcols, wcols, W.t()*W, mxGetPr(wtw_out_m));
    arma2matlab<MAT>(wcols, wcols, H.t()*H, mxGetPr(hth_out_m));
    arma2matlab<MAT>(arows, wcols, A * H, mxGetPr(ah_out_m));
    arma2matlab<MAT>(acols, wcols, A.t()*W, mxGetPr(atw_out_m));
    return;
}
