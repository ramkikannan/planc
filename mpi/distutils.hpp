/* Copyright 2016 Ramakrishnan Kannan */
#ifndef MPI_DISTUTILS_HPP_
#define MPI_DISTUTILS_HPP_

#include <mpi.h>
#include <string>
#include "utils.h"
#include "distutils.h"
#include "utils.hpp"

inline void mpitic() {
    tictoc_stack.push(clock());
}

inline void mpitic(int rank) {
    double temp = clock();
    cout << "tic::" << rank << "::" << temp << endl;
    tictoc_stack.push(temp);
}

inline double mpitoc(int rank) {
#ifdef __WITH__BARRIER__TIMING__
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    cout << "toc::" << rank << "::"
         << tictoc_stack.top() << endl;
    double rc = (static_cast<double>
                 (clock() - tictoc_stack.top())) / CLOCKS_PER_SEC;
    tictoc_stack.pop();
    return rc;
}

inline double mpitoc() {
#ifdef __WITH__BARRIER__TIMING__
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    double rc = (static_cast<double>
                 (clock() - tictoc_stack.top())) / CLOCKS_PER_SEC;
    tictoc_stack.pop();
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
    // INFO << myrank << "::mem::" << current_proc_mem << endl;
    long allprocmem;
    MPI_Reduce(&current_proc_mem, &allprocmem, 1, MPI_LONG,
               MPI_SUM, 0, MPI_COMM_WORLD);
    if (myrank == 0) {
        INFO << event << " total rss::" << allprocmem << endl;
    }
}
#endif  // MPI_DISTUTILS_HPP_
