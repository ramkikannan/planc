/* Copyright 2020 Lawton Manning */
#ifndef HIERNMF_MATUTILS_HPP_
#define HIERNMF_MATUTILS_HPP_

#include <algorithm>
#include "common/distutils.hpp"
#include "common/utils.hpp"

namespace planc {
template <class INPUTMATTYPE>
void print(INPUTMATTYPE M, const char* name) {
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    M.print(name);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 1; i < size; i++) {
    if (i == rank) {
      M.print();
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

struct PowerTimings {
  double communication = 0;
  double matvec = 0;
  double vecmat = 0;
  double normalisation = 0;
};

template <class INPUTMATTYPE>
double powIter(const INPUTMATTYPE& A, int max_iter, double tol,
               PowerTimings * timings) {
  // MPI variables
  int size, rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Matrix/Vector sizes
  int local_m = A.n_rows;
  int global_n = A.n_cols;
  int local_n = itersplit(global_n, size, rank);

  // MPI Comm init
  int * counts = new int[size];
  for (int i = 0; i < size; i++) {
    counts[i] = itersplit(global_n, size, i);
  }
  int * displs = new int[size];
  displs[0] = 0;
  for (int i = 1; i < size; i++) {
    displs[i] = displs[i - 1] + counts[i - 1];
  }

  // Vector initialization
  VEC localQ(local_n, arma::fill::randn);
  VEC globalQ(global_n, arma::fill::zeros);
  MPITIC;
  MPI_Allgatherv(localQ.memptr(), local_n, MPI_DOUBLE, globalQ.memptr(), counts,
                 displs, MPI_DOUBLE, MPI_COMM_WORLD);
  timings->communication += MPITOC;
  MPITIC;
  globalQ /= norm(globalQ);
  timings->normalisation += MPITOC;

  // local variables
  VEC z(local_m);
  double sigma = 1;
  double s2 = sigma;
  double epsilon;
  double norm;

  // converge to first sigma value of AtA
  for (int i = 0; i < max_iter; i++) {
    MPITIC;
    z = A * globalQ;
    timings->matvec += MPITOC;
    MPITIC;
    globalQ = A.t() * z;
    timings->vecmat += MPITOC;

    // Reduce-Scatter q
    MPITIC;
    MPI_Reduce_scatter(globalQ.memptr(), localQ.memptr(), counts, MPI_DOUBLE,
                       MPI_SUM, MPI_COMM_WORLD);
    timings->communication += MPITOC;

    // Normalize q by sigma
    MPITIC;
    norm = pow(arma::norm(localQ), 2);
    MPI_Allreduce(&norm, &sigma, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    sigma = sqrt(sigma);

    localQ /= sigma;
    timings->normalisation += MPITOC;

    // All-Gather normalized q
    MPITIC;
    MPI_Allgatherv(localQ.memptr(), local_n, MPI_DOUBLE, globalQ.memptr(),
                   counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    timings->communication += MPITOC;

    // epsilon tolerance
    epsilon = abs(sigma - s2) / (sigma);
    s2 = sigma;

    if (epsilon < tol) {
      break;
    }
  }
  delete counts;
  delete displs;

  return sigma;
}

VEC maxk(VEC X, int k) {
  VEC Xs = arma::sort(X, "descend");
  if (X.n_elem <= k) {
    return Xs;
  }
  return Xs.head(k);
}

UVEC maxk_idx(VEC X, int k) {
  UVEC Xi = arma::sort_index(X, "descend");
  if (X.n_elem <= k) {
    return Xi;
  }
  return Xi.head(k);
}
}  // namespace planc
#endif  // HIERNMF_MATUTILS_HPP_
