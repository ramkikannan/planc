#ifndef DISTNTF_DISTAUNTF_HPP
#define DISTNTF_DISTAUNTF_HPP

#include <armadillo>
#include "updatealgos.hpp"
#include "ntf_utils.h"

/*
* Tensor A of size is M1 x M2 x... x Mn is distributed among
* P1 x P2 x ... x Pn grid of P processors. That means, every
* processor has (M1/P1) x (M2/P2) x ... x (Mn/Pn) tensor as
* m_input_tensor. Similarly every process own a portion of the
* factors as H(i,pi) of size (Mi/Pi) x k
* and collects from its neighbours as H(i,p) as (Mi/P) x k
* H(i,p) and m_input_tensor can perform local MTTKRP. The
* local MTTKRP's are reduced scattered for local NNLS.
*/


namespace PLANC {

#define TENSOR_LOCAL_DIM (m_input_tensor.dimension())
#define TENSOR_LOCAL_NUMEL (m_input_tensor.numel())


class DistAUNTF {
  private:
    const Tensor m_input_tensor;
    // local ncp factors
    NCPFactors m_local_ncp_factors;
    NCPFactors m_local_ncp_factors_t;
    NCPFactors m_gathered_ncp_factors;
    NCPFactors m_gathered_ncp_factors_t;
    //mttkrp related variables
    FMAT *ncp_krp;
    FMAT *ncp_mttkrp;
    FMAT *ncp_local_mttkrp_t;
    // gram related variables.
    FMAT factor_local_grams; // U in the algorithm.
    FMAT *factor_global_grams; // G in the algorithm
    FMAT global_gram; // hadamard of the global_grams

    // NTF related variable.
    const int m_low_rank_k;
    const int m_order;
    const distntfalgotype m_updalgo;
    const UVEC m_global_dims;
    const UVEC m_factor_local_dims;
    int m_num_it;
    int current_mode;
    int current_it;
    //communication related variables
    NTFMPICommunicator m_mpicomm;
    UVEC *recvmttkrpsize;
    //stats
    DistNTFTime time_stats;

    // do the local syrk only for the current updated factor
    // and all reduce only for the current updated factor.
    // computes G^(current_mode)
    void update_global_gram(const int current_mode) {
        // computing U
        mpitic();  // gram
        // force a ssyrk instead of gemm.
        factor_local_grams = m_local_ncp_factors.factor(current_mode) *
                             m_local_ncp_factors_t.factor(current_mode);
        double temp = mpitoc();  // gram
        this->time_stats.compute_duration(temp);
        this->time_stats.gram_duration(temp);
        factor_global_grams[current_mode].zeros();
        //Computing G.
        mpitic();  // allreduce gram
        MPI_Allreduce(factor_local_grams.memptr(),
                      factor_global_grams[current_mode].memptr(),
                      this->k * this->k, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#ifdef __WITH__BARRIER__TIMING__
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        temp = mpitoc();  // allreduce gram
        this->time_stats.communication_duration(temp);
        this->time_stats.allreduce_duration(temp);
    }

    /*
    * This iterates over all grams and find hadamard of the grams
    */
    void gram_hadamard(int current_mode) {
        global_gram.ones();
        for (int i = 0; i < m_order; i++) {
            if (i != current_mode) {
                //%= element-wise multiplication
                global_gram %= factor_global_grams[i];
            }
        }
    }

    // factor matrices all gather only on the current update factor
    void gather_ncp_factor(const int current_mode) {
        int sendcnt = m_local_ncp_factors.factor(current_mode).n_elem;
        int recvcnt = m_local_ncp_factors.factor(current_mode).n_elem;
        mpitic();
        double temp = mpitoc();
        this->time_stats.compute_duration(temp);
        this->time_stats.trans_duration(temp);
        m_gathered_ncp_factors_t.factor(current_mode).zeros();
        mpitic();  // allgather tic
        MPI_Allgather(m_local_ncp_factors_t.factor(current_mode).memptr(),
                      sendcnt, MPI_FLOAT,
                      m_gathered_ncp_factors_t.factor(current_mode).memptr(),
                      recvcnt, MPI_FLOAT,
                      // todo:: check whether it is slice or fiber while running
                      // and debugging the code.
                      this->m_mpicomm.slice(current_mode));
        temp = mpitoc();  // allgather toc
        this->time_stats.communication_duration(temp);
        this->time_stats.allgather_duration(temp);
        // keep gather_ncp_factors_t consistent.
        mpitic();  // transpose tic
        m_gathered_ncp_factors.set(current_mode, m_gathered_ncp_factors_t.t());
        temp = mpitoc();
        this->time_stats.compute_duration(temp);
        this->time_stats.trans_duration(temp);
    }

    void distmttkrp(const int &current_mode) {
        mpitic();
        m_gathered_ncp_factors.krp_leave_out_one(current_mode,
                &ncp_krp[current_mode]);
        temp = mpitoc();
        this->time_stats.compute_duration(temp);
        this->time_stats.krp_duration(temp);
        m_input_tensor.mttkrp(current_mode, &ncp_krp[current_mode],
                              &ncp_mttkrp[current_mode]);
        temp = mpitoc();  // mttkrp
        this->time_stats.compute_duration(temp);
        this->time_stats.mttkrp_duration(temp);
        mpitic();  // reduce_scatter mttkrp
        MPI_Reduce_scatter(ncp_mttkrp.memptr(), ncp_local_mttkrp_t.memptr(),
                           this->recvmttkrpsize[current_mode].memptr(),
                           MPI_FLOAT, MPI_SUM,
                           this->m_mpicomm.slice(current_mode));
        temp = mpitoc();  // reduce_scatter mttkrp
        this->time_stats.communication_duration(temp);
        this->time_stats.reducescatter_duration(temp);
    }

    void allocateMatrices() {
        //allocate matrices.
        ncp_krp = new FMAT[m_order];
        ncp_mttkrp = new FMAT[m_order];
        ncp_local_mttkrp_t = new FMAT[m_order];
        recvmttkrpsize = new UVEC[m_order];
        factor_local_grams = new FMAT[m_order];
        factor_global_grams = new FMAT[m_order];
        factor_local_grams = arma::zeros<FMAT>(i_k, i_k);

        for (int i = 0; i < m_order; i++) {
            UWORD current_size = TENSOR_NUMEL - TENSOR_DIM[i];
            ncp_krp[i] = arma::zeros <FMAT>(current_size, i_k);
            ncp_mttkrp[i] = arma::zeros<FMAT>(TENSOR_DIM[i], i_k);
            ncp_local_mttkrp_t = arma::zeros<FMAT>(m_local_ncp_factors.factor(i).size());
            recvmttkrpsize[i] = arma::UVEC(TENSOR_DIM[i] * i_k);
            factor_global_grams[i] = arma::zeros<FMAT>(i_k, i_k);
        }
    }

  public:
    DistAUNTF(const Tensor &i_tensor, const int i_k, ntfalgo i_algo,
              const UVEC &i_global_dims,
              const NTFMPICommunicator &i_mpicomm, ) :
        m_input_tensor(i_tensor.dimensions(), i_tensor.data()),
        m_low_rank_k(i_k),
        m_updalgo(i_algo),
        m_mpicomm(i_mpicomm),
        m_order(m_input_tensor.order()),
        m_global_dims(i_global_dims),
        time_stats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) {
        this->num_it = 30;

        //local factors.
        m_factor_local_dims = m_global_dims / MPI_SIZE;
        m_local_ncp_factors(m_factor_local_dims, i_k),
                            m_local_ncp_factors_t(i_k, m_factor_local_dims);
        m_local_ncp_factors.trans(m_local_ncp_factors_t);

        //input size factors
        m_gathered_ncp_factors(i_tensor.dimensions(), i_k);
        m_gathered_ncp_factors.trans(m_gathered_ncp_factors_t);
        allocateMatrices();
    }
    void num_it(const int i_n) { this->m_num_it = i_n;}
    void computeNTF() {
        //initialize everything.
        //line 3,4,5 of the algorithm
        for (int i = 1; i < m_order; i++) {
            update_global_gram(i);
            gather_ncp_factor(i);
        }
        for (int current_it = 0; current_it < m_num_it; current_it++) {
            for (int current_mode = 0; current_mode < m_order; current_mode++) {
                // line 9 and 10 of the algorithm
                distmttkrp(current_mode);
                // line 11 of the algorithm
                gram_hadamard(current_mode);
                // line 12 of the algorithm
                FMAT factor = update(m_updalgo, global_gram,
                                     ncp_local_mttkrp_t[current_mode]);
                m_local_ncp_factors_t.set(current_mode, factor);
                m_local_ncp_factors.set(current_mode, factor.t());
                // line 13 and 14
                update_global_gram(current_mode);
                // line 15
                gather_ncp_factor(current_mode);
            }
        }
    }
}  // class DistAUNTF
}  // namespace PLANC
#endif  // DISTNTF_DISTAUNTF_HPP