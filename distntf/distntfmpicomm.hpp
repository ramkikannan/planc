/* Copyright 2016 Ramakrishnan Kannan */

#ifndef DISTNTF_DISTNTFMPICOMM_HPP_
#define DISTNTF_DISTNTFMPICOMM_HPP_

#define MPI_CART_DIMS m_proc_grids.n_rows

#include <mpi.h>
#include <vector>
#include "distntf/distntfutils.h"
namespace planc {

class NTFMPICommunicator {
 private:
  int m_global_rank;
  int m_num_procs;
  UVEC m_proc_grids;
  MPI_Comm m_cart_comm;
  // for mode communicators (*,...,p_n,...,*)
  MPI_Comm *m_fiber_comm;
  // all communicators other than the given mode.
  // (p1,p2,...,p_n-1,*,p_n,...,p_M)
  MPI_Comm *m_slice_comm;
  UVEC m_fiber_ranks;
  UVEC m_slice_ranks;
  UVEC m_slice_sizes;
  std::vector<int> m_coords;

 public:
  void printConfig() {
    if (rank() == 0) {
      INFO << "successfully setup MPI communicators" << std::endl;
      INFO << "size=" << size() << std::endl;
      INFO << "processor grid size::" << m_proc_grids;
      int slice_size;
      MPI_Comm current_slice_comm;
      for (int i = 0; i < MPI_CART_DIMS; i++) {
        current_slice_comm = this->m_slice_comm[i];
        MPI_Comm_size(current_slice_comm, &slice_size);
        INFO << "Numprocs in slice " << i << "::" << slice_size << std::endl;
      }
      int fiber_size;
      MPI_Comm current_fiber_comm;
      for (int i = 0; i < MPI_CART_DIMS; i++) {
        current_fiber_comm = this->m_fiber_comm[i];
        MPI_Comm_size(current_fiber_comm, &fiber_size);
        INFO << "Numprocs in fiber " << i << "::" << fiber_size << std::endl;
      }
    }
    UVEC cooprint(MPI_CART_DIMS);
    for (int ii = 0; ii < MPI_CART_DIMS; ii++) cooprint[ii] = m_coords[ii];

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < size(); i++) {
      if (i == rank()) {
        INFO << "slice ranks of rank::" << m_global_rank
             << "::" << m_slice_ranks << std::endl;
        INFO << "fiber ranks of rank::" << m_global_rank
             << "::" << m_fiber_ranks << std::endl;
        INFO << "coordinates::" << m_global_rank << "::" << cooprint
             << std::endl;
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  NTFMPICommunicator(int argc, char *argv[], const UVEC &i_dims)
      : m_proc_grids(i_dims) {
    // Get the number of MPI processes
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &m_global_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &m_num_procs);
    int grid_count = m_proc_grids[0];
    if (m_num_procs != arma::prod(m_proc_grids)) {
      ERR << "number of mpi process and process grid doesn't match";
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Abort(MPI_COMM_WORLD, 1);
      exit(-1);
    }
    // Create a virtual topology MPI communicator
    std::vector<int> periods(MPI_CART_DIMS);
    std::vector<int> m_proc_grids_vec =
        arma::conv_to<std::vector<int>>::from(m_proc_grids);
    for (int i = 0; i < MPI_CART_DIMS; i++) periods[i] = 1;
    int reorder = 0;
    MPI_Cart_create(MPI_COMM_WORLD, MPI_CART_DIMS, &m_proc_grids_vec[0],
                    &periods[0], reorder, &m_cart_comm);

    // Allocate memory for subcommunicators
    m_fiber_comm = new MPI_Comm[MPI_CART_DIMS];
    m_slice_comm = new MPI_Comm[MPI_CART_DIMS];

    // Get the subcommunicators
    std::vector<int> remainDims(MPI_CART_DIMS);
    for (int i = 0; i < remainDims.size(); i++) remainDims[i] = 1;
    // initialize the fiber ranks
    m_slice_ranks = arma::zeros<UVEC>(MPI_CART_DIMS);
    int current_slice_rank;
    for (int i = 0; i < MPI_CART_DIMS; i++) {
      remainDims[i] = 0;
      MPI_Cart_sub(m_cart_comm, &remainDims[0], &(m_slice_comm[i]));
      remainDims[i] = 1;
      MPI_Comm_rank(m_slice_comm[i], &current_slice_rank);
      m_slice_ranks[i] = current_slice_rank;
    }
    for (int i = 0; i < remainDims.size(); i++) remainDims[i] = 0;
    m_fiber_ranks = arma::zeros<UVEC>(MPI_CART_DIMS);
    int current_fiber_rank;
    for (int i = 0; i < MPI_CART_DIMS; i++) {
      remainDims[i] = 1;
      MPI_Cart_sub(m_cart_comm, &remainDims[0], &(m_fiber_comm[i]));
      remainDims[i] = 0;
      MPI_Comm_rank(m_fiber_comm[i], &current_fiber_rank);
      m_fiber_ranks[i] = current_fiber_rank;
    }
    // Get the coordinates
    m_coords.resize(MPI_CART_DIMS, 0);
    MPI_Cart_coords(m_cart_comm, m_global_rank, MPI_CART_DIMS, &m_coords[0]);
    // Get the slice size
    m_slice_sizes = arma::zeros<UVEC>(MPI_CART_DIMS);
    for (int i = 0; i < MPI_CART_DIMS; i++) {
      m_slice_sizes[i] = m_num_procs / m_proc_grids[i];
    }
  }

  ~NTFMPICommunicator() {
    MPI_Barrier(MPI_COMM_WORLD);
    int finalized;
    MPI_Finalized(&finalized);

    if (!finalized) {
      for (int i = 0; i < MPI_CART_DIMS; i++) {
        MPI_Comm_free(&m_fiber_comm[i]);
        MPI_Comm_free(&m_slice_comm[i]);
      }
    }
    delete[] m_fiber_comm;
    delete[] m_slice_comm;
  }

  const MPI_Comm &cart_comm() const { return m_cart_comm; }
  void coordinates(int *o_c) const {
    MPI_Cart_coords(m_cart_comm, m_global_rank, MPI_CART_DIMS, o_c);
  }

  std::vector<int> coordinates() const { return this->m_coords; }

  const MPI_Comm &fiber(const int i) const { return m_fiber_comm[i]; }
  const MPI_Comm &slice(const int i) const { return m_slice_comm[i]; }
  int rank(const int *i_coords) const {
    int my_rank;
    MPI_Cart_rank(m_cart_comm, i_coords, &my_rank);
    return my_rank;
  }
  int size() const { return m_num_procs; }
  int size(const int i_d) const {
    int n_procs;
    MPI_Comm_size(m_slice_comm[i_d], &n_procs);
    return n_procs;
  }
  int fiber_rank(int i) const { return m_fiber_ranks[i]; }
  int slice_rank(int i) const { return m_slice_ranks[i]; }
  UVEC proc_grids() const { return this->m_proc_grids; }
  int rank() const { return m_global_rank; }
  int num_slices(int mode) const { return m_proc_grids[mode]; }
  int slice_num(int mode) const { return m_coords[mode]; }
  int slice_size(int mode) const { return m_slice_sizes[mode]; }

  /*
   * Return true only for those processors whose
   * coordinates are non-zero for the mode and zero
   * non-modes
   */
  bool isparticipating(int mode) {
    bool rc = true;
    size_t num_modes = this->m_proc_grids.n_rows;
    for (int i = 0; i < num_modes; i++) {
      if (i != mode && this->m_coords[i] != 0) {
        rc = false;
        break;
      }
    }
    return rc;
  }
};

}  // namespace planc

#endif  // DISTNTF_DISTNTFMPICOMM_HPP_
