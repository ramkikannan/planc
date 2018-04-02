// to compile the code run as
//g++ testmm.cpp -I$ARMADILLO_INCLUDE_DIR -L/lustre/atlas/proj-shared/csc209/ramki/rhea/libraries/openblas/lib/ -lopenblas --std=c++11

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
* multiplies as sparse mxk matrix with
* dense kxn matrix
*/

int main(int argc, char *argv[]) {
    int m = atoi(argv[1]);
    int k = atoi(argv[2]);
    int n = atoi(argv[3]);    
    fmat A;
    A = randu<fmat>(m, k);
    fmat denseA(m, k);
    //denseA = A;
    fmat B;
    B = randu<fmat>(k, n);
    fmat C;
    C.zeros(m, n);
    PRINTMATINFO(A);
    PRINTMATINFO(B);
    tic();
    C = A * B;
    cout << "arma ::" << toc() << endl;
    PRINTMATINFO(C);
}