// to compile the code run as
//icc -mkl testcscmm.cpp -I ~/rhea/libraries/armadillo-7.100.3/include/ -std=c++11 -DARMA_64BIT_WORD

#include <mkl.h>
#include <armadillo>
#include <iostream>
#include <stack>

using namespace std;
using namespace arma;

static std::stack<clock_t> tictoc_stack;

inline void tic() {
    tictoc_stack.push(clock());
}

inline double toc() {
    double rc = (static_cast<double>
                 (clock() - tictoc_stack.top())) / CLOCKS_PER_SEC;
    tictoc_stack.pop();
    return rc;
}

#define PRINTMATINFO(A) cout << "::"#A"::" << (A).n_rows << "x" << (A).n_cols << endl;

/*
* mklMat is csc representation
* Bt is the row major order of the arma B matrix
* Ct is the row major order of the arma C matrix
* Once you receive Ct, transpose again to print
* C using arma
*/
void ARMAMKLSCSCMM(const sp_fmat &mklMat,
        const fmat &Bt, const char transa, float *Ct) {
    MKL_INT m, k, n, nnz;
    m = static_cast<MKL_INT>(mklMat.n_rows);
    k = static_cast<MKL_INT>(mklMat.n_cols);
    n = static_cast<MKL_INT>(Bt.n_rows);
    // fmat B = B.t();
    // C = alpha * A * B + beta * C;
    // mkl_?cscmm - https://software.MKL_INTel.com/en-us/node/468598
    // char transa = 'N';
    float alpha = 1.0;
    float beta = 0.0;
    char* matdescra = "GUNC";
    MKL_INT ldb = n;
    MKL_INT ldc = n;
    MKL_INT* pntrb = (MKL_INT *)(mklMat.col_ptrs);
    MKL_INT* pntre = pntrb + 1;
    mkl_scscmm(&transa, &m, &n, &k, &alpha, matdescra,
               mklMat.values, (MKL_INT *)mklMat.row_indices,
               pntrb, pntre,
               (float *)(Bt.memptr()), &ldb,
               &beta, Ct, &ldc);
}

/*
* multiplies as sparse mxk matrix with
* dense kxn matrix
*/

int main(int argc, char *argv[]) {
    int m = atoi(argv[1]);
    int k = atoi(argv[2]);
    int n = atoi(argv[3]);
    float sparsity = atof(argv[4]);
    sp_fmat A;
    A = sprandu<sp_fmat>(m, k, sparsity);
    fmat denseA(m, k);
    //denseA = A;
    fmat B;
    B = randu<fmat>(k, n);
    fmat C;
    C.zeros(n, m);
    PRINTMATINFO(A);
    cout << "nnz::" << A.n_nonzero << endl;
    PRINTMATINFO(B);
    tic();
    ARMAMKLSCSCMM(A, B.t(), 'N', C.memptr());
    cout << "mkl cscmm::" << toc() << endl;
    PRINTMATINFO(C);
    tic();
    C = A * B;
    cout << "arma ::" << toc() << endl;
}