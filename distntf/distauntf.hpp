#ifndef DISTNTF_DISTAUNTF_HPP
#define DISTNTF_DISTAUNTF_HPP

#include <armadillo>
#include "ntf_utils.h"
#include "luc.hpp"
#include "distntftime.hpp"
#include "dimtree/kobydt.hpp"

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

#define DISTNTF_VERBOSE 1

namespace planc {

#define TENSOR_LOCAL_DIM (m_input_tensor.dimensions())
#define TENSOR_LOCAL_NUMEL (m_input_tensor.numel())

class DistAUNTF {
  private:
    const Tensor m_input_tensor;
    // local ncp factors
    NCPFactors m_local_ncp_factors;
    NCPFactors m_local_ncp_factors_t;
    NCPFactors m_gathered_ncp_factors;
    NCPFactors m_gathered_ncp_factors_t;
    // mttkrp related variables
    MAT *ncp_krp;
    MAT *ncp_mttkrp;
    MAT *ncp_local_mttkrp_t;
    // gram related variables.
    MAT factor_local_grams;  // U in the algorithm.
    MAT *factor_global_grams;  // G in the algorithm
    MAT global_gram;  // hadamard of the global_grams

    // NTF related variable.
    const int m_low_rank_k;
    const int m_modes;
    const algotype m_updalgo;
    const UVEC m_global_dims;
    const UVEC m_factor_local_dims;
    int m_num_it;
    int current_mode;
    int current_it;
    FVEC m_regularizers;
    bool m_compute_error;

    // update function
    LUC *m_luc_ntf_update;


    // communication related variables
    NTFMPICommunicator m_mpicomm;
    std::vector<int> recvmttkrpsize;
    // stats
    DistNTFTime time_stats;

    // computing error related;
    double m_global_sqnorm_A;
    MAT hadamard_all_grams;

    KobyDimensionTree *kdt;
    // do the local syrk only for the current updated factor
    // and all reduce only for the current updated factor.
    // computes G^(current_mode)
    void update_global_gram(const int current_mode) {
        // computing U
        mpitic();  // gram
        // force a ssyrk instead of gemm.
        factor_local_grams = m_local_ncp_factors.factor(current_mode).t() *
                             m_local_ncp_factors.factor(current_mode);
        double temp = mpitoc();  // gram
        this->time_stats.compute_duration(temp);
        this->time_stats.gram_duration(temp);
        factor_global_grams[current_mode].zeros();
        // Computing G.
        mpitic();  // allreduce gram
        MPI_Allreduce(factor_local_grams.memptr(),
                      factor_global_grams[current_mode].memptr(),
                      this->m_low_rank_k * this->m_low_rank_k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#ifdef __WITH__BARRIER__TIMING__
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        temp = mpitoc();  // allreduce gram
        applyReg(this->m_regularizers(current_mode * 2),
                 this->m_regularizers(current_mode * 2 + 1),
                 &(factor_global_grams[current_mode]));
        this->time_stats.communication_duration(temp);
        this->time_stats.allreduce_duration(temp);
    }

    void applyReg(float lambda_l2, float lambda_l1, MAT *AtA) {
        // Frobenius norm regularization
        if (lambda_l2 > 0) {
            MAT  identity  = arma::eye<MAT>(this->m_low_rank_k,
                                            this->m_low_rank_k);
            (*AtA) = (*AtA) + 2 * lambda_l2 * identity;
        }

        // L1 - norm regularization
        if (lambda_l1 > 0) {
            MAT  onematrix = arma::ones<MAT>(this->m_low_rank_k,
                                             this->m_low_rank_k);
            (*AtA) = (*AtA) + 2 * lambda_l1 * onematrix;
        }
    }

    /*
    * This iterates over all grams and find hadamard of the grams
    */
    void gram_hadamard(int current_mode) {
        global_gram.ones();
        for (int i = 0; i < m_modes; i++) {
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
        m_gathered_ncp_factors_t.factor(current_mode).zeros();
        mpitic();  // allgather tic
        MPI_Allgather(m_local_ncp_factors_t.factor(current_mode).memptr(),
                      sendcnt, MPI_DOUBLE,
                      m_gathered_ncp_factors_t.factor(current_mode).memptr(),
                      recvcnt, MPI_DOUBLE,
                      // todo:: check whether it is slice or fiber while running
                      // and debugging the code.
                      this->m_mpicomm.slice(current_mode));
        double temp = mpitoc();  // allgather toc
        this->time_stats.communication_duration(temp);
        this->time_stats.allgather_duration(temp);
        // keep gather_ncp_factors_t consistent.
        mpitic();  // transpose tic
        m_gathered_ncp_factors.set(current_mode,
                                   m_gathered_ncp_factors_t.factor(current_mode).t());
        temp = mpitoc();
        this->time_stats.compute_duration(temp);
        this->time_stats.trans_duration(temp);
    }

    void distmttkrp(const int &current_mode) {
        mpitic();
        m_gathered_ncp_factors.krp_leave_out_one(current_mode,
                &ncp_krp[current_mode]);
        double temp = mpitoc();
        this->time_stats.compute_duration(temp);
        this->time_stats.krp_duration(temp);
        mpitic();
        kdt->in_order_reuse_MTTKRP(current_mode, ncp_mttkrp[current_mode].memptr());
        // m_input_tensor.mttkrp(current_mode, ncp_krp[current_mode],
        //                       &ncp_mttkrp[current_mode]);
        temp = mpitoc();  // mttkrp
        this->time_stats.compute_duration(temp);
        this->time_stats.mttkrp_duration(temp);
        mpitic();  // reduce_scatter mttkrp
        MPI_Reduce_scatter(ncp_mttkrp[current_mode].memptr(),
                           ncp_local_mttkrp_t[current_mode].memptr(),
                           &this->recvmttkrpsize[current_mode],
                           MPI_DOUBLE, MPI_SUM,
                           this->m_mpicomm.slice(current_mode));
        temp = mpitoc();  // reduce_scatter mttkrp
        this->time_stats.communication_duration(temp);
        this->time_stats.reducescatter_duration(temp);
    }

    void allocateMatrices() {
        //allocate matrices.
        ncp_krp = new MAT[m_modes];
        ncp_mttkrp = new MAT[m_modes];
        ncp_local_mttkrp_t = new MAT[m_modes];
        UVEC temp_recvmttkrpsize = arma::zeros<UVEC>(m_modes);
        factor_local_grams = arma::zeros<MAT>(this->m_low_rank_k, this->m_low_rank_k);
        factor_global_grams = new MAT[m_modes];
        global_gram = arma::ones<MAT>(this->m_low_rank_k, this->m_low_rank_k);
        for (int i = 0; i < m_modes; i++) {
            UWORD current_size = TENSOR_LOCAL_NUMEL / TENSOR_LOCAL_DIM[i];
            ncp_krp[i] = arma::zeros <MAT>(current_size, this->m_low_rank_k);
            ncp_mttkrp[i] = arma::zeros<MAT>(TENSOR_LOCAL_DIM[i], this->m_low_rank_k);
            ncp_local_mttkrp_t[i] = arma::zeros<MAT>(m_local_ncp_factors.factor(i).n_cols,
                                    m_local_ncp_factors.factor(i).n_rows);
            temp_recvmttkrpsize[i] = TENSOR_LOCAL_DIM[i] * this->m_low_rank_k;
            factor_global_grams[i] = arma::zeros<MAT>(this->m_low_rank_k,
                                     this->m_low_rank_k);
        }
        recvmttkrpsize = arma::conv_to<std::vector<int>>::from(temp_recvmttkrpsize);
        if (m_compute_error) {
            hadamard_all_grams = arma::ones<MAT>(this->m_low_rank_k,
                                                 this->m_low_rank_k);
        }
    }

  public:
    DistAUNTF(const Tensor &i_tensor, const int i_k, algotype i_algo,
              const UVEC &i_global_dims,
              const UVEC &i_local_dims,
              const NTFMPICommunicator &i_mpicomm) :
        m_input_tensor(i_tensor.dimensions(), i_tensor.m_data),
        m_low_rank_k(i_k),
        m_updalgo(i_algo),
        m_mpicomm(i_mpicomm),
        m_modes(m_input_tensor.modes()),
        m_global_dims(i_global_dims),
        m_factor_local_dims(i_local_dims),
        m_local_ncp_factors(i_local_dims, i_k, false),
        m_local_ncp_factors_t(i_local_dims, i_k, true),
        m_gathered_ncp_factors(i_tensor.dimensions(), i_k, false),
        m_gathered_ncp_factors_t(i_tensor.dimensions(), i_k, true),
        time_stats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) {
        this->m_num_it = 30;
        //local factors.
        arma::arma_rng::set_seed(i_mpicomm.rank());
        m_local_ncp_factors.distributed_normalize();
        for (int i = 0; i < this->m_modes; i++) {
            MAT current_factor = arma::trans(m_local_ncp_factors.factor(i));
            m_local_ncp_factors_t.set(i, current_factor);
        }
        m_gathered_ncp_factors.trans(m_gathered_ncp_factors_t);
        m_luc_ntf_update = new LUC();
        allocateMatrices();
        double normA = i_tensor.norm();
        MPI_Allreduce(&normA,
                      &this->m_global_sqnorm_A,
                      1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        kdt = new KobyDimensionTree(m_input_tensor, m_gathered_ncp_factors,
                                    m_input_tensor.modes() / 2);

    }
    void num_iterations(const int i_n) { this->m_num_it = i_n;}
    void regularizers(const FVEC i_regs) {this->m_regularizers = i_regs;}
    void compute_error(bool i_error) {this->m_compute_error = i_error;}
    void computeNTF() {
        //initialize everything.
        //line 3,4,5 of the algorithm
        for (int i = 1; i < m_modes; i++) {
            update_global_gram(i);
            gather_ncp_factor(i);
        }
#ifdef DISTNTF_VERBOSE
        DISTPRINTINFO("local factor matrices::");
        this->m_local_ncp_factors.print();
        DISTPRINTINFO("local factor matrices::");
        this->m_local_ncp_factors_t.print();
        DISTPRINTINFO("gathered factor matrices::");
        this->m_gathered_ncp_factors.print();
        PRINTROOT("global_grams::" << this->global_gram);
#endif
        for (int current_it = 0; current_it < m_num_it; current_it++) {
            MAT unnorm_factor;
            for (int current_mode = 0; current_mode < m_modes; current_mode++) {
                // line 9 and 10 of the algorithm
                distmttkrp(current_mode);
                // line 11 of the algorithm
                gram_hadamard(current_mode);
                // line 12 of the algorithm
                mpitic();  // nnls_tic
#ifdef DISTNTF_VERBOSE
                DISTPRINTINFO("local factor matrix::"
                              << this->m_local_ncp_factors.factor(current_mode));
                DISTPRINTINFO("gathered factor matrix::");
                this->m_gathered_ncp_factors.print();
                PRINTROOT("global_grams::" << this->global_gram);
                DISTPRINTINFO("mttkrp::");
                this->ncp_local_mttkrp_t[current_mode].print();
#endif
                MAT factor = m_luc_ntf_update->update(m_updalgo, global_gram,
                                                      ncp_local_mttkrp_t[current_mode]);
#ifdef DISTNTF_VERBOSE
                DISTPRINTINFO("it::" << current_it << "::mode::" << current_mode
                              << std::endl << factor);
#endif
                double temp = mpitoc();  // nnls_toc
                this->time_stats.nnls_duration(temp);
                if (m_compute_error && current_mode == this->m_modes - 1) {
                    unnorm_factor = factor;
                }
                m_local_ncp_factors.set(current_mode, factor.t());
                m_local_ncp_factors.distributed_normalize(current_mode);
                factor = m_local_ncp_factors.factor(current_mode);
                m_local_ncp_factors_t.set(current_mode, factor);
                // line 13 and 14
                update_global_gram(current_mode);
                // line 15
                gather_ncp_factor(current_mode);
                kdt->set_factor(m_gathered_ncp_factors_t.factor(current_mode).memptr(),
                                current_mode);
            }
            if (m_compute_error) {
                double temp_err = computeError(unnorm_factor);
                PRINTROOT("Iter::" << current_it << "::error::" << temp_err);
            }
            PRINTROOT("completed it::" << current_it);

        }
    }
    double computeError(MAT &unnorm_factor) {

        //rel_Error = sqrt(max(init.nr_X^2 + lambda^T * Hadamard of all gram * lambda - 2 * innerprod(X,F_kten),0))/init.nr_X;
        mpitic();
        hadamard_all_grams = global_gram % factor_global_grams[this->m_modes - 1];
        VEC local_lambda = m_local_ncp_factors.lambda();
        PRINTROOT(local_lambda);
        ROWVEC temp_vec = local_lambda.t() * hadamard_all_grams;
        double norm_gram = arma::dot(temp_vec, local_lambda);
        // double norm_gram = arma::norm(hadamard_all_grams, "fro");
        // sum of the element-wise dot product between the local mttkrp and
        // the factor matrix
        double model_error = arma::dot(ncp_local_mttkrp_t[this->m_modes - 1],
                                       unnorm_factor);
        double temp = mpitoc();
        this->time_stats.err_compute_duration(temp);
        double all_model_error;
        mpitic();
        MPI_Allreduce(&model_error, &all_model_error, 1, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        temp = mpitoc();
        this->time_stats.err_communication_duration(temp);
#ifdef DISTNTF_VERBOSE
        DISTPRINTINFO ("local_model_error::" << model_error << std::endl);
        PRINTROOT("norm_A :: " << this->m_global_sqnorm_A
                  << "::global_gram_norm_sq::" << norm_gram * norm_gram
                  << "::model_error::" << 2 * all_model_error << std::endl);
#endif
        double relerr = this->m_global_sqnorm_A + norm_gram * norm_gram
                        - 2 * all_model_error;
        return relerr;
    }
};  // class DistAUNTF
}  // namespace PLANC
#endif  // DISTNTF_DISTAUNTF_HPP