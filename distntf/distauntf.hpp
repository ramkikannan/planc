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
    NCPFactors m_gathered_ncp_factors;
    NCPFactors m_local_ncp_factors_t;
    NCPFactors m_gathered_ncp_factors_t;
    int m_num_it;
    FMAT local_gram;
    FMAT global_local_gram;
    DistNTFTime time_stats;
    const int m_low_rank_k;
    FMAT *ncp_krp;
    FMAT *ncp_mttkrp;
    FMAT *ncp_local_mttkrp_t;
    const distntfalgotype m_updalgo;
    NTFMPICommunicator m_mpicomm;
    const UVEC m_global_dims;
    int current_mode;
    int current_it;
    UVEC *recvmttkrpsize;

    /*
    * pre compute all the grams and keep it
    * update the gram of factor right after the new factor
    * Computing G^{(i)} as in our algorithm
    */

    void local_gram(NCPFactors &ncp_local_factor, FMAT *global_gram) {
    }

    void gram() {
        // each process computes its own kxk matrix
        mpitic();  // gram
        m_local_ncp_factors.gram_leave_out_one(j, &local_gram);
#ifdef MPI_VERBOSE
        DISTPRINTINFO("W::" << norm(local_gram, "fro") \
                      << "::localWtW::" << norm(this->localWtW, "fro"));
#endif
        double temp = mpitoc();  // gram
        this->time_stats.compute_duration(temp);
        this->time_stats.gram_duration(temp);
        PRINTROOT("it::" << current_it << "::mode::" << current_mode
                  << "::gram::" << temp);
        global_local_gram.zeros();
        mpitic();  // allreduce gram
        MPI_Allreduce(local_gram.memptr(), global_local_gram.memptr(), this->k * this->k,
                      MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#ifdef __WITH__BARRIER__TIMING__
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        temp = mpitoc();  // allreduce gram
        this->time_stats.communication_duration(temp);
        this->time_stats.allreduce_duration(temp);
    }

    void distmttkrp(const int &current_mode) {
        int sendcnt = m_local_ncp_factors.factor(current_mode).n_elem;
        int recvcnt = m_local_ncp_factors.factor(current_mode).n_elem;
        mpitic();
        m_local_ncp_factors.trans(m_local_ncp_factors_t);
        double temp = mpitoc();
        this->time_stats.compute_duration(temp);
        this->time_stats.trans_duration(temp);
        m_gathered_ncp_factors_t.factor(current_mode).zeros();
        mpitic();  // allgather tic
        MPI_Allgather(m_local_ncp_factors_t.factor(current_mode).memptr(),
                      sendcnt, MPI_FLOAT,
                      m_gathered_ncp_factors_t.factor(current_mode).memptr(),
                      recvcnt, MPI_FLOAT,
                      this->m_mpicomm.fiber(current_mode));
        temp = mpitoc();  // allgather toc
        this->time_stats.communication_duration(temp);
        this->time_stats.allgather_duration(temp);
        mpitic();  // transpose tic
        m_gathered_ncp_factors_t.trans(m_gathered_ncp_factors);
        temp = mpitoc();
        this->time_stats.compute_duration(temp);
        this->time_stats.trans_duration(temp);
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
        //todo:determine mttkrpsize.
        //todo:determine communication
        MPI_Reduce_scatter(ncp_mttkrp.memptr(), ncp_local_mttkrp_t.memptr(),
                           this->recvmttkrpsize[current_mode].memptr(),
                           MPI_FLOAT, MPI_SUM,
                           this->m_mpicomm.slice(current_mode));
        temp = mpitoc();  // reduce_scatter mttkrp
        this->time_stats.communication_duration(temp);
        this->time_stats.reducescatter_duration(temp);
    }

  public:
    DistAUNTF(const Tensor &i_tensor, const int i_k, ntfalgo i_algo,
              const UVEC &i_global_dims,
              const NTFMPICommunicator &i_mpicomm, ) :
        m_local_ncp_factors(i_tensor.dimensions(), i_k),
        m_input_tensor(i_tensor.dimensions(), i_tensor.data()),
        m_low_rank_k(i_k),
        m_updalgo(i_algo),
        m_mpicomm(i_mpicomm),
        m_global_dims(i_global_dims),
        time_stats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) {
        local_gram = arma::zeros<FMAT>(i_k, i_k);
        global_local_gram = arma::zeros<FMAT>(i_k, i_k);
        this->num_it = 30;
        ncp_mttkrp = arma::zeros<FMAT>(TENSOR_NUMEL, i_k);
        for (int i = 0; i < i_tensor.order(); i++) {
            UWORD current_size = TENSOR_NUMEL - TENSOR_DIM[i];
            ncp_krp[i] = arma::zeros <FMAT>(current_size, i_k);
            ncp_mttkrp[i] = arma::zeros<FMAT>(TENSOR_DIM[i], i_k);
            recvmttkrpsize[i] = arma::UVEC(TENSOR_DIM[i] * i_k);
        }
    }
    void num_it(const int i_n) { this->m_num_it = i_n;}
    void computeNTF() {
        for (int current_it = 0; current_it < m_num_it; current_it++) {
            for (int current_mode = 0; current_mode < this->m_input_tensor.order(); current_mode++) {
                // gram();
                // distmttkrp();
                // FMAT factor = update(m_updalgo, local_gram,
                //                      ncp_local_mttkrp_t[current_mode]);
                // m_local_ncp_factors.set(current_mode, factor.t());
                //after discussion w/ gray
                //step 9 and 10;
                // distmttkrp() without allgather
                //step 11;
                // hadamard gram without allgather and only local
                //step 12:
                // call update and set function
                // step 13 and 14:
                // call this factor gram
                // do the local syrk only for the current updated factor
                // and all reduce only for the current updated factor.
                // step 15:
                // factor matrices all gather only on the current update factor

            }
        }
    }
}
}  // namespace PLANC
#endif  // DISTNTF_DISTAUNTF_HPP