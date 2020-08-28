/* Copyright 2016 Ramakrishnan Kannan */

#ifndef DISTNMF_MPICOMM_HPP_
#define DISTNMF_MPICOMM_HPP_

#include <mpi.h>
#include <vector>
#include "common/distutils.hpp"

#ifdef USE_PACOSS
#include "pacoss/pacoss.h"
#endif

/**
 * Class and function for 2D MPI communicator
 * with row and column communicators
 */

namespace planc {

class MPICommunicator {
 private:
  int m_rank;  /// Local Rank
  int m_numProcs;  /// Total number of mpi process
  int m_row_rank;
  int m_row_size;
  int m_col_rank;
  int m_col_size;
  int m_pr, m_pc;
  MPI_Comm m_gridComm;

  // for 2D communicators
  // MPI Related stuffs
  MPI_Comm *m_commSubs;
  void printConfig() {
    if (rank() == 0) {
      INFO << "successfully setup MPI communicators" << std::endl;
      INFO << "size=" << size() << std::endl;
      INFO << "rowsize=" << m_row_size << ":pr=" << m_pr << std::endl;
      INFO << "colsize=" << m_col_size << ":pc=" << m_pc << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    INFO << ":rank=" << rank() << ":row_rank=" << row_rank() << ":colrank"
         << col_rank() << std::endl;
  }

 public:
  // Violating the cpp guidlines. Other functions need
  // non const pointers.
  MPICommunicator(int argc, char *argv[]) {
#ifdef USE_PACOSS
    TMPI_Init(&argc, &argv);
#else
    MPI_Init(&argc, &argv);
#endif
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &m_numProcs);
  }
  ~MPICommunicator() {
    MPI_Barrier(MPI_COMM_WORLD);
#ifdef USE_PACOSS
    TMPI_Finalize();
#else
    MPI_Finalize();
#endif
  }
  MPICommunicator(int argc, char *argv[], int pr, int pc) {
#ifdef USE_PACOSS
    TMPI_Init(&argc, &argv);
#else
    MPI_Init(&argc, &argv);
#endif
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &m_numProcs);
    int reorder = 0;
    std::vector<int> dimSizes;
    std::vector<int> periods;
    int nd = 2;
    dimSizes.resize(nd);
    this->m_pr = pr;
    this->m_pc = pc;
    dimSizes[0] = pr;
    dimSizes[1] = pc;
    periods.resize(nd);
    //MPI_Comm gridComm;
    std::vector<int> gridCoords;
    fillVector<int>(1, &periods);
    // int MPI_Cart_create(MPI_Comm comm_old, int ndims, const int dims[],
    //                const int periods[], int reorder, MPI_Comm *comm_cart)
    if (dimSizes[0] * dimSizes[1] != m_numProcs) {
      if (m_rank == 0) {
        std::cerr << "Processor grid dimensions do not"
                  << "multiply to MPI_SIZE::" << dimSizes[0] << 'x'
                  << dimSizes[1] << "::m_numProcs::" << m_numProcs << std::endl;
      }
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Cart_create(MPI_COMM_WORLD, nd, &dimSizes[0], &periods[0], reorder,
                    &m_gridComm);
    gridCoords.resize(nd);
    MPI_Cart_get(m_gridComm, nd, &dimSizes[0], &periods[0], &(gridCoords[0]));
    this->m_commSubs = new MPI_Comm[nd];
    int *keepCols = new int[nd];
    for (int i = 0; i < nd; i++) {
      std::fill_n(keepCols, nd, 0);
      keepCols[i] = 1;
      MPI_Cart_sub(m_gridComm, keepCols, &(this->m_commSubs[i]));
    }
    MPI_Comm_size(m_commSubs[0], &m_row_size);
    MPI_Comm_size(m_commSubs[1], &m_col_size);
    MPI_Comm_rank(m_commSubs[0], &m_row_rank);
    MPI_Comm_rank(m_commSubs[1], &m_col_rank);
#ifdef MPI_VERBOSE
    printConfig();
#endif
  }
  /// returns the global rank
  const int rank() const { return m_rank; }
  /// returns the total number of mpi processes
  const int size() const { return m_numProcs; }
  /// returns its rank in the row processor grid
  const int row_rank() const { return m_row_rank; }
  /// returns the rank in the column processor grid
  const int col_rank() const { return m_col_rank; }
  /// Total number of row processors
  const int pr() const { return m_pr; }
  /// Total number of column processor
  const int pc() const { return m_pc; }
  const MPI_Comm *commSubs() const { return m_commSubs; }
  const MPI_Comm gridComm() const { return m_gridComm; }
};

}  // namespace planc

#endif  // DISTNMF_MPICOMM_HPP_
