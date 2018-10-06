/* Copyright 2016 Ramakrishnan Kannan */
#ifndef COMMON_DISTUTILS_HPP_
#define COMMON_DISTUTILS_HPP_

#include <mpi.h>
#include <string>
#include "common/distutils.h"
#include "common/utils.h"
#include "common/utils.hpp"

inline void mpitic() {
  // tictoc_stack.push(clock());
  tictoc_stack.push(std::chrono::steady_clock::now());
}

inline void mpitic(int rank) {
  // std::cout << "tic::" << rank << "::" << std::chrono::steady_clock::now() <<
  // std::endl;
  tictoc_stack.push(std::chrono::steady_clock::now());
}

inline double mpitoc(int rank) {
#ifdef __WITH__BARRIER__TIMING__
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  std::chrono::duration<double> time_span =
      std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - tictoc_stack.top());
  double rc = time_span.count();
  tictoc_stack.pop();
  std::cout << "toc::" << rank << "::" << rc << std::endl;
  return rc;
}

inline double mpitoc() {
#ifdef __WITH__BARRIER__TIMING__
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  std::chrono::duration<double> time_span =
      std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - tictoc_stack.top());
  double rc = time_span.count();
  tictoc_stack.pop();
  return rc;
}

inline void memusage(const int myrank, std::string event) {
  // Based on the answer from stackoverflow
  // http://stackoverflow.com/questions/669438/how-to-get-memory-usage-at-run-time-in-c
  int64_t rss = 0L;
  FILE *fp = NULL;
  if ((fp = fopen("/proc/self/statm", "r")) == NULL) return;
  if (fscanf(fp, "%*s%ld", &rss) != 1) {
    fclose(fp);
    return;
  }
  fclose(fp);
  int64_t current_proc_mem = (size_t)rss * (size_t)sysconf(_SC_PAGESIZE);
  // INFO << myrank << "::mem::" << current_proc_mem << std::endl;
  int64_t allprocmem;
  MPI_Reduce(&current_proc_mem, &allprocmem, 1, MPI_INT64_T, MPI_SUM, 0,
             MPI_COMM_WORLD);
  if (myrank == 0) {
    INFO << event << " total rss::" << allprocmem << std::endl;
  }
}

inline int itersplit(int n, int p, int r) {
  int split = (r < n % p) ? n / p + 1 : n / p;
  return split;
}

inline int startidx(int n, int p, int r);
{
  int rem = n % p;
  int idx = (r < rem) ? (r - 1 + 1) + (n / p + 1)
      : (rem * (n / p + 1) + ((r - rem) * (n / p)));
  return idx;
}

#endif  // COMMON_DISTUTILS_HPP_
