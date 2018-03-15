/* Copyright 2016 Ramakrishnan Kannan */
#ifndef MPI_DISTUTILS_H_
#define MPI_DISTUTILS_H_

// #define MPI_VERBOSE       1
// #define WRITE_RAND_INPUT  1

// enum distalgotype {MU2D, HALS2D, ANLSBPP2D, NAIVEANLSBPP, AOADMM2D};

// ONED_DOUBLE for naive ANLS-BPP
// TWOD for HPC-NMF
enum iodistributions {ONED_ROW, ONED_COL, ONED_DOUBLE, TWOD};

#define ISROOT this->m_mpicomm.rank() == 0
#define MPI_SIZE this->m_mpicomm.size()
#define MPI_RANK this->m_mpicomm.rank()
#define MPI_ROW_RANK this->m_mpicomm.row_rank()
#define MPI_COL_RANK this->m_mpicomm.col_rank()
#define MPI_FIBER_RANK(i) this->m_mpicomm.fiber_rank(i)
#define MPI_SLICE_RANK(i) this->m_mpicomm.slice_rank(i)
#define NUMROWPROCS this->m_mpicomm.pr()
#define NUMCOLPROCS this->m_mpicomm.pc()

#define MPITIC mpitic()
#define MPITOC mpitoc()
// #define MPITIC mpitic(MPI_RANK)
// #define MPITOC mpitoc(MPI_RANK)


// #define PRINTROOT(MSG) 
#define PRINTROOT(MSG) if (ISROOT) INFO << "::" << __PRETTY_FUNCTION__ \
                                       << "::" << __LINE__ \
                                       << "::" << std::endl << MSG << std::endl;
#define DISTPRINTINFO(MSG) INFO << MPI_RANK << "::" << __PRETTY_FUNCTION__ \
                                << "::" << __LINE__ \
                                << "::" << std::endl << MSG << std::endl;
#define PRINTTICTOCTOP if (ISROOT) INFO << "tictoc::" << tictoc_stack.top() \
                                         << std::endl;

#endif  // MPI_DISTUTILS_H_

