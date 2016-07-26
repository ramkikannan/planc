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

inline double mpitoc() {
#ifdef __WITH__BARRIER__TIMING__
	MPI_Barrier(MPI_COMM_WORLD);
#endif
	double rc = (static_cast<double>
	             (clock() - tictoc_stack.top())) / CLOCKS_PER_SEC;
	tictoc_stack.pop();
	return rc;
}
#endif