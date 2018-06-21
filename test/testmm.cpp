// to compile the code run as
// g++ testmm.cpp -I$ARMADILLO_INCLUDE_DIR
// -L/lustre/atlas/proj-shared/csc209/ramki/rhea/libraries/openblas/lib/
// -lopenblas --std=c++11

#include <omp.h>
#include <armadillo>
#include <iostream>
#include <stack>

using namespace std;
using namespace arma;

#ifdef USE_CHRONO
static std::stack<std::chrono::steady_clock::time_point> tictoc_stack;
inline void tic() { tictoc_stack.push(std::chrono::steady_clock::now()); }

inline double toc() {
  std::chrono::duration<double> time_span =
      std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - tictoc_stack.top());
  double rc = time_span.count();
  tictoc_stack.pop();
  return rc;
}
#else
static std::stack<double> tictoc_stack;
inline void tic() { tictoc_stack.push(omp_get_wtime()); }

inline double toc() {
  double rc = (static_cast<double>(omp_get_wtime() -
                                   tictoc_stack.top()));  // / CLOCKS_PER_SEC;
  tictoc_stack.pop();
  return rc;
}
#endif

#define PRINTMATINFO(A) \
  cout << "::" #A "::" << (A).n_rows << "x" << (A).n_cols << endl;

/*
 * multiplies as sparse mxk matrix with
 * dense kxn matrix
 */

int main(int argc, char *argv[]) {
  int m = atoi(argv[1]);
  int k = atoi(argv[2]);
  int n = atoi(argv[3]);
  int t = atoi(argv[4]);
  mat A;
  A = randu<mat>(m, k);
  mat denseA(m, k);
  // denseA = A;
  mat B;
  B = randu<mat>(k, n);
  mat C;
  C.zeros(m, n);
  PRINTMATINFO(A);
  PRINTMATINFO(B);
  PRINTMATINFO(C);
  cout << "norm c::" << arma::norm(C, "fro");
  tic();
  for (int i = 0; i < t; i++) {
    C = A * B;
  }
  cout << "arma ::" << toc()/t << endl;
  PRINTMATINFO(C);
  cout << "norm c::" << arma::norm(C, "fro");
}