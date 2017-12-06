/* Copyright 2016 Ramakrishnan Kannan */
#ifndef MPI_DISTIO_HPP_
#define MPI_DISTIO_HPP_


#include <unistd.h>
#include <armadillo>
#include <string>
#include "distntfmpicomm.hpp"
#include "distutils.hpp"
#include "tensor.hpp"
#include "ncpfactors.hpp"


/*
 * File name formats
 * A is the filename
 * 1D distribution Arows_totalpartitions_rank or Acols_totalpartitions_rank
 * Double 1D distribution (both row and col distributed)
 * Arows_totalpartitions_rank and Acols_totalpartitions_rank
 * TWOD distribution A_totalpartition_rank
 * Just send the first parameter Arows and the second parameter Acols to be zero.
 */
namespace planc {
class DistNTFIO {
  private:
    const NTFMPICommunicator& m_mpicomm;
    Tensor m_A;
    // don't start getting prime number from 2;
    static const int kPrimeOffset = 10;
    // Hope no one hits on this number.
    static const int kW_seed_idx = 1210873;
    static const int kalpha = 1;
    static const int kbeta = 0;

    /*
    * Uses the pattern from the input matrix X but
    * the value is computed as low rank.
    */
    void randomLowRank(const UVEC i_global_dims, const UWORD i_k) {

        // start with the same seed_idx with the global dimensions
        // on all the MPI processor.
        arma::arma_rng::set_seed(kW_seed_idx);
        NCPFactors global_factors(i_global_dims, i_k);
        int tensor_modes = global_factors.modes();
        UVEC local_dims = i_global_dims / this->m_mpicomm.proc_grids();
        NCPFactors local_factors(local_dims, i_k);
        UWORD start_row, end_row;
        for (int i = 0; i < local_factors.modes(); i++) {
            start_row = MPI_FIBER_RANK(i) * local_dims(i);
            end_row = MPI_FIBER_RANK(i + 1) * local_dims(i);
            local_factors.factor(i) = global_factors.factor(i).rows(start_row, end_row);
        }
        m_A = local_factors.rankk_tensor();
    }

  public:
    explicit DistNTFIO(const NTFMPICommunicator &mpic): m_mpicomm(mpic) {
    }
    /*
     * We need m,n,pr,pc only for rand matrices. If otherwise we are
     * expecting the file will hold all the details.
     * If we are loading by file name we dont need distio flag.
     *
     */
    void readInput(const std::string file_name,
                   UVEC i_global_dims, UVEC i_proc_grids,
                   UWORD k = 0, double sparsity = 0) {
        // INFO << "readInput::" << file_name << "::" << distio << "::"
        //     << m << "::" << n << "::" << pr << "::" << pc
        //     << "::" << this->MPI_RANK << "::" << this->m_mpicomm.size() << std::endl;
        std::string rand_prefix("rand_");
        if (!file_name.compare(0, rand_prefix.size(), rand_prefix)) {
            if (!file_name.compare("rand_uniform")) {
                Tensor temp(i_global_dims / i_proc_grids);
                this->m_A = temp;
            } else if (!file_name.compare("rand_lowrank")) {
                randomLowRank(i_global_dims, k);
            }
        }
    }
    void writeOutput(NCPFactors &factors,
                     const std::string & output_file_name) {
        for (int i = 0; i < factors.modes(); i++) {
            std::stringstream sw, sh;
            sw << output_file_name << "_mode" << i << "_" << MPI_SIZE
               << "_" << MPI_RANK;
            factors.factor(i).save(sw.str(), arma::raw_ascii);
        }
    }
    void writeRandInput() {
    }
    const Tensor A() const {return m_A;}
    const NTFMPICommunicator& mpicomm() const {return m_mpicomm;}
};
}  // namespace planc

#endif  // MPI_DISTIO_HPP_
