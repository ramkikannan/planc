/* Copyright 2016 Ramakrishnan Kannan */
#ifndef DISTNTF_DISTNTFMPICOMM_HPP_
#define DISTNTF_DISTNTFMPICOMM_HPP_

#define MPI_CART_DIMS m_proc_grid_dims.n_rows

#include <mpi.h>
#include <vector>
#include "distntfutils.hpp"
namespace PLANC {

class NTFMPICommunicator {
  private:
    int m_global_rank;
    int m_num_procs;
    UVEC m_proc_grid_dims;
    MPI_Comm m_cart_comm;
    // for mode communicators (*,...,p_n,...,*)
    MPI_Comm* m_fiber_comm;
    // all communicators other than the given mode.
    // (p1,p2,...,p_n-1,*,p_n,...,p_M)
    MPI_Comm* m_slice_comm;

    void printConfig() {
        if (rank() == 0) {
            INFO << "successfully setup MPI communicators" << endl;
            INFO << "size=" << size() << endl;

        }
        MPI_Barrier(MPI_COMM_WORLD);
        cout << ":rank=" << rank() << ":row_rank=" <<
             row_rank() << ":colrank" << col_rank() << endl;
    }

  public:
    NTFMPICommunicator(int argc, char *argv[],
                       const UVEC &i_dims,
                       const MPI_Comm& comm) :
        m_proc_grid_dims(i_dims) {
        // Get the number of MPI processes
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &m_global_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &m_num_procs);
        if (m_num_procs != arma::prod(m_proc_grid_dims)) {
            ERR << "number of mpi process and process grid doesn't match";
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Abort(MPI_COMM_WORLD, 1);
            exit(-1);
        }
        // Create a virtual topology MPI communicator
        UVEC periods = arma::ones<UVEC>(MPI_CART_DIMS);
        int reorder = 0;
        MPI_Cart_create(comm, MPI_CART_DIMS,
                        m_proc_grid_dims.memptr(), periods.memptr(),
                        reorder, &m_cart_comm);

        // Allocate memory for subcommunicators
        m_fiber_comm = new MPI_Comm[MPI_CART_DIMS];
        m_slice_comm = new MPI_Comm[MPI_CART_DIMS];

        // Get the subcommunicators
        UVEC remainDims = arma::zeros<UVEC>(MPI_CART_DIMS);
        for (int i = 0; i < MPI_CART_DIMS; i++) {
            remainDims[i] = 1;
            MPI_Cart_sub(m_cart_comm, remainDims.memptr(), &(m_slice_comm[i]));
            remainDims[i] = 0;
        }
        remainDims = 1;
        for (int i = 0; i < ndims; i++) {
            remainDims[i] = 0;
            MPI_Cart_sub(m_cart_comm, remainDims.memptr(), &(m_fiber_comm[i]));
            remainDims[i] = 1;
        }
    }

    ~NTFMPICommunicator() {
        MPI_Barrier(MPI_COMM_WORLD);
        int finalized;
        MPI_Finalized(&finalized);

        if (!finalized) {
            for (int i = 0; i < MPI_CART_DIMS; i++) {
                MPI_Comm_free(m_fiber_comm + i);
                MPI_Comm_free(m_slice_comm + i);
            }
        }
        delete m_fiber_comm;
        delete m_slice_comm;
    }

    const MPI_Comm& cart_comm() const {return m_cart_comm;}
    void coordinates(UVEC *o_c) const {
        MPI_Cart_coords(m_cart_comm, m_global_rank, MPI_CART_DIMS, coords->memptr());
    }
    const MPI_Comm& fiber(const int i) const {return m_fiber_comm[d];}
    const MPI_Comm& slice(const int i) const {return m_slice_comm[d];}
    int rank(const UVEC &i_coords) const {
        MPI_Cart_rank(m_cart_comm, coords->memptr(), &rank);
    }
    int size() const {return m_num_procs;}
    int size(const int i_d) const {
        int n_procs;
        MPI_Comm_size(m_slice_comm[d], &n_procs);
        return n_procs;
    }
};
}  // namespace PLANC
#endif  // DISTNTF_DISTNTFMPICOMM_HPP_
