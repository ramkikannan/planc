/* Copyright 2016 Ramakrishnan Kannan */
#ifndef MPI_DISTUTILS_HPP_
#define MPI_DISTUTILS_HPP_

#include <mpi.h>
#include <string>
#include "utils.h"
#include "distutils.h"
#include "utils.hpp"

inline void mpitic() {
    // tictoc_stack.push(clock());
    tictoc_stack_omp_clock.push(omp_get_wtime());
}

inline void mpitic(int rank) {
    // double temp = clock();
    double temp = omp_get_wtime();
    std::cout << "tic::" << rank << "::" << temp << std::endl;
    // tictoc_stack.push(temp);
    tictoc_stack_omp_clock.push(omp_get_wtime());
}

inline double mpitoc(int rank) {
#ifdef __WITH__BARRIER__TIMING__
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    std::cout << "toc::" << rank << "::"
              << tictoc_stack_omp_clock.top() << std::endl;
    // double rc = (static_cast<double>
    //              (clock() - tictoc_stack.top())) / CLOCKS_PER_SEC;
    //tictoc_stack.pop();
    double rc = omp_get_wtime() - tictoc_stack_omp_clock.top();
    tictoc_stack_omp_clock.pop();
    return rc;
}

inline double mpitoc() {
#ifdef __WITH__BARRIER__TIMING__
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    // double rc = (static_cast<double>
    //              (clock() - tictoc_stack.top())) / CLOCKS_PER_SEC;
    // tictoc_stack.pop();
    double rc = omp_get_wtime() - tictoc_stack_omp_clock.top();
    tictoc_stack_omp_clock.pop();
    return rc;
}

inline void memusage(const int myrank, std::string event) {
    // Based on the answer from stackoverflow
    // http://stackoverflow.com/questions/669438/how-to-get-memory-usage-at-run-time-in-c
    long rss = 0L;
    FILE* fp = NULL;
    if ((fp = fopen( "/proc/self/statm", "r" )) == NULL )
        return;
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return;
    }
    fclose(fp);
    long current_proc_mem = (size_t)rss * (size_t)sysconf(_SC_PAGESIZE);
    // INFO << myrank << "::mem::" << current_proc_mem << std::endl;
    long allprocmem;
    MPI_Reduce(&current_proc_mem, &allprocmem, 1, MPI_LONG,
               MPI_SUM, 0, MPI_COMM_WORLD);
    if (myrank == 0) {
        INFO << event << " total rss::" << allprocmem << std::endl;
    }
}
#endif  // MPI_DISTUTILS_HPP_
