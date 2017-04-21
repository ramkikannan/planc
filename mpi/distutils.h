/* Copyright 2016 Ramakrishnan Kannan */
#ifndef MPI_DISTUTILS_H_
#define MPI_DISTUTILS_H_

// #define MPI_VERBOSE       1
// #define WRITE_RAND_INPUT  1

enum distalgotype {MU2D, HALS2D, ANLSBPP2D, NAIVEANLSBPP};

// ONED_DOUBLE for naive ANLS-BPP
// TWOD for HPC-NMF
enum iodistributions {ONED_ROW, ONED_COL, ONED_DOUBLE, TWOD};

// usage scenarios
// mpirun -np 12 distnmf algotype lowrank AfileName numIteration pr pc
// mpirun -np 12 distnmf algotype lowrank m n numIteration pr pc
// mpirun -np 12 distnmf algotype lowrank Afile  outputfile numIteration pr pc
#define PROCROWS        2002
#define PROCCOLS        2003
#define NUMKBLOCKS      2004

// we are not supporting initfiles for distnmf.
// consistent initializations will be distIO
struct option distnmfopts[] = {
    {"input",       required_argument, 0, 'i'},
    {"algo",        required_argument, 0, 'a'},
    {"error",       optional_argument, 0, 'e'},
    {"lowrank",     required_argument, 0, 'k'},
    {"iter",        optional_argument, 0, 't'},
    {"rows",        optional_argument, 0, 'm'},
    {"columns",     optional_argument, 0, 'n'},
    {"output",      optional_argument, 0, 'o'},
    {"sparsity",    optional_argument, 0, 's'},
    {"pr",          optional_argument, 0, PROCROWS},
    {"pc",          optional_argument, 0, PROCCOLS},
    {"regw",        optional_argument, 0, REGWFLAG},
    {"regh",        optional_argument, 0, REGHFLAG},
    {"numkblocks",  optional_argument, 0, NUMKBLOCKS},
    {0,             0,                 0,  0 }
};

#define ISROOT this->m_mpicomm.rank() == 0
#define MPI_SIZE this->m_mpicomm.size()
#define MPI_RANK this->m_mpicomm.rank()
#define MPI_ROW_RANK this->m_mpicomm.row_rank()
#define MPI_COL_RANK this->m_mpicomm.col_rank()
#define NUMROWPROCS this->m_mpicomm.pr()
#define NUMCOLPROCS this->m_mpicomm.pc()

#define MPITIC mpitic()
#define MPITOC mpitoc()

// #define PRINTROOT(MSG) 
#define PRINTROOT(MSG) if (ISROOT) INFO << "::" << __PRETTY_FUNCTION__ \
                                       << "::" << __LINE__ \
                                       << "::" << MSG << endl;
#define DISTPRINTINFO(MSG) INFO << MPI_RANK << "::" << __PRETTY_FUNCTION__ \
                                << "::" << __LINE__ \
                                << "::" << MSG << endl;

#define PRINTTICTOCTOP if (ISROOT) INFO << "tictoc::" << tictoc_stack.top() \
                                         << endl;

#endif  // MPI_DISTUTILS_H_

