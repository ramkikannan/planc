#include <armadillo>
#include <iostream>
#include <ctime>
#include <stack>
#include <mpi.h>

//compile command
/*
 icc -mkl testmm.cpp -I ~/rhea/libraries/armadillo-7.100.3/include/ \
 -I/sw/rhea/openmpi/1.8.4/rhel6.6_gcc4.8.2/include \
 -lmpi -L/sw/rhea/openmpi/1.8.4/rhel6.6_gcc4.8.2/lib/
 */


using namespace std;
using namespace arma;

static std::stack<clock_t> tictoc_stack;

inline void tic() {
  tictoc_stack.push(clock());
}

inline double toc() {
  MPI_Barrier(MPI_COMM_WORLD);
  double rc = (static_cast<double>
               (clock() - tictoc_stack.top())) / CLOCKS_PER_SEC;
  tictoc_stack.pop();
  return rc;
}

#define PRINTMATINFO(A) cout << "::"#A"::" << (A).n_rows << "x" << (A).n_cols << endl;

int main(int argc, char *argv[]) {
  int m_rank;
  int m_numProcs;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &m_numProcs);
  int m = atoi(argv[1]);
  int n = atoi(argv[2]);
  int k = atoi(argv[3]);
  int pr = atoi(argv[4]);
  int pc = atoi(argv[5]);
  int p = pr * pc;
  fmat Arows = randu<fmat>(m / p, n);
  fmat Acolst = randu<fmat>(n / p, m);
  fmat A = randu<fmat>(m / pr, n / pc);
  fmat At = A.t();
  fmat ArowsH(m / p , k);
  fmat AcolstW(n / p, k);
  fmat AH(m / pr, k);
  fmat WtA(k, n / pc);
  PRINTMATINFO(Arows);
  PRINTMATINFO(Acolst);
  PRINTMATINFO(A);
  // PRINTMATINFO(W1D);
  // PRINTMATINFO(H1D);
  // PRINTMATINFO(W2D);
  // PRINTMATINFO(H2D);
  double AcolstWtime = 0;
  double ArowsHtime = 0;
  double AHtime = 0;
  double WtAtime = 0;
  for (int i = 0; i < 30; i++) {
    // cout << "iter::" << i << endl;
    fmat W1D = randu<fmat>(m, k);
    fmat H1D = randu<fmat>(n, k);
    fmat W2Dt = randu<fmat>(k, m / pr);
    fmat H2Dt = randu<fmat>(k, n / pc);
    tic();
    ArowsH = Arows * H1D;
    ArowsHtime += toc();
    tic();
    AcolstW = Acolst * W1D;
    AcolstWtime += toc();
    tic();
    AH = H2Dt * At;
    AHtime += toc();
    tic();
    WtA = W2Dt * A;
    WtAtime += toc();
  }
  if (m_rank == 0) {
    cout << "ArowsHtime::" << ArowsHtime << endl;
    cout << "AcolstWtime::" << AcolstWtime << endl;
    cout << "1D time::" << ArowsHtime + AcolstWtime << endl;
    cout << "WtAtime::" << WtAtime << endl;
    cout << "AHtime::" << AHtime << endl;
    cout << "2D time::" << WtAtime + AHtime << endl;
  }
  MPI_Finalize();
}