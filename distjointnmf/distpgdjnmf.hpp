/* Copyright 2022 Ramakrishnan Kannan */

#ifndef DISTJNMF_DISTPGD_HPP_
#define DISTJNMF_DISTPGD_HPP_

#include "distjointnmf/distjointnmf.hpp"
#include "distjointnmf/jnmfmpicomm.hpp"
#include "distjointnmf/distjointnmfio.hpp"

namespace planc {

template <class T1, class T2>
class DistPGDJointNMF : public DistJointNMF<T1, T2> {
 protected:
  // Surrogate variables for update rules
  MAT gradW;     /// gradW is of size (globalm/p)*k
  MAT gradH;     /// gradH is of size (globaln/p)*k
  
  // Temporary variables for line search
  MAT Wa;     /// Wa is of size (globalm/p)*k
  MAT Ha;     /// Ha is of size (globaln/p)*k
  
  // S grid (SHs) --> A grid (H) communication variables  
  gridInfo SHsgrid;
  std::vector<procInfo> sendSHstoH;
  std::vector<procInfo> recvSHstoH;

  // Variables needed for momentum
  MAT prevstepW, prevstepH;
  
  // Variables needed for objective error
  MAT WtAijH, localWtAijH;      /// used for fast error computation
  MAT SHtijH, localSHtijH;
  
  double m_Afit, m_Sfit;
  int m_func_calls, m_gradW_calls, m_gradH_calls, m_lnfails;

  // Temporary variables for transfers
  MAT SHtij;

  // Hyperparameters
  double m_gamma, m_Smax;
  bool stale_WtA, stale_AH, stale_SHs, stale_WtW, stale_HtH;
  int m_numchks;

  /**
   * Allocate temporaries needed for PGD JointNMF algorithms
   */
  void allocatePGDMatrices() {
    // Gradient variables
    gradW.zeros(this->W.n_rows, this->W.n_cols);
    gradH.zeros(this->H.n_rows, this->H.n_cols);

    // Momentum variables
    prevstepW.zeros(this->W.n_rows, this->W.n_cols);
    prevstepH.zeros(this->H.n_rows, this->H.n_cols);

    // Objective calculations variables
    WtAijH.zeros(this->k, this->k); 
    localWtAijH.zeros(this->k, this->k);    
    SHtijH.zeros(this->k, this->k);
    localSHtijH.zeros(this->k, this->k);
  }

  /**
   * Communicator needed to synchronise SHs and H for computing
   * gradH
   */
  void setupSHsHComms() {
    // Setup grid
    SHsgrid.pr    = NUMROWPROCS_C(this->m_Scomm);
    SHsgrid.pc    = NUMCOLPROCS_C(this->m_Scomm);
    SHsgrid.order = 'R';

    // SHs (S grid) --> H (A grid)
    sendSHstoH = getsendinfo(this->m_globaln,
                  this->m_Scomm.row_rank(), this->m_Scomm.col_rank(),
                  this->SHsgrid, this->Hgrid);

    recvSHstoH = getrecvinfo(this->m_globaln,
                  this->m_mpicomm.row_rank(), this->m_mpicomm.col_rank(),
                  this->SHsgrid, this->Hgrid);
  }

 public:
  DistPGDJointNMF(const T1& input, const T2& conn,
          const MAT& leftlowrankfactor, const MAT& rightlowrankfactor,
          const MPICommunicatorJNMF& communicator, const int numkblks)
      : DistJointNMF<T1, T2>(input, conn, 
              leftlowrankfactor, rightlowrankfactor, communicator, numkblks) {
    // Allocate temporary matrices
    allocatePGDMatrices();
    setupSHsHComms();

    // Get S max
    double local_Smax = this->S.max();
    MPI_Allreduce(&local_Smax, &this->m_Smax, 1, MPI_DOUBLE, MPI_MAX,
          MPI_COMM_WORLD);

    // Set hyperparameters
    this->m_gamma   = 0.9;
    this->m_numchks = 20;

    // Set counters
    m_func_calls  = 0;
    m_gradW_calls = 0;
    m_gradH_calls = 0;
    m_lnfails     = 0;

    // Set all matmuls to stale
    stale_WtA = true;
    stale_AH  = true;
    stale_SHs = true;
    stale_WtW = true;
    stale_HtH = true;

#ifdef MPI_VERBOSE
    function<void()> f_SHs2H = [this] {
      std::cout << "SHs2H print." << std::endl;

      std::cout << "sendSHstoH" << std::endl;
      std::cout << this->m_Scomm.row_rank() << "x" 
                << this->m_Scomm.col_rank()
                << " (S grid) sending" << std::endl;
      for (auto i = this->sendSHstoH.begin(); 
              i != this->sendSHstoH.end(); ++i) {
        std::cout << (*i).i << "x" << (*i).j << "::startidx::"
            << (*i).start_idx << "::nrows::" << (*i).nrows
            << std::endl;
      }

      std::cout << "recvSHstoH" << std::endl;
      std::cout << this->m_mpicomm.row_rank() << "x" 
                << this->m_mpicomm.col_rank()
                << " (A grid) receiving" << std::endl;
      for (auto i = this->recvSHstoH.begin(); 
              i != this->recvSHstoH.end(); ++i) {
        std::cout << (*i).i << "x" << (*i).j << "::startidx::"
            << (*i).start_idx << "::nrows::" << (*i).nrows
            << std::endl;
      }
    };
    mpi_serial_print(f_SHs2H);
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    PRINTROOT("DistPGDJointNMF() constructor successful (single comm).");
  }

  DistPGDJointNMF(const T1& input, const T2& conn,
          const MAT& leftlowrankfactor, const MAT& rightlowrankfactor,
          const MPICommunicatorJNMF& Acomm, const MPICommunicatorJNMF& Scomm,
          const int numkblks)
      : DistJointNMF<T1, T2>(input, conn, 
              leftlowrankfactor, rightlowrankfactor, 
              Acomm, Scomm, numkblks) {
    // Allocate temporary matrices
    allocatePGDMatrices();
    setupSHsHComms();

    // Set hyperparameters
    this->m_gamma   = 0.9;
    this->m_numchks = 20;

    // Set counters
    m_func_calls  = 0;
    m_gradW_calls = 0;
    m_gradH_calls = 0;
    m_lnfails     = 0;

    // Set all matmuls to stale
    stale_WtA = true;
    stale_AH  = true;
    stale_SHs = true;
    stale_WtW = true;
    stale_HtH = true;

#ifdef MPI_VERBOSE
    function<void()> f_SHs2H = [this] {
      std::cout << "SHs2H print." << std::endl;

      std::cout << "sendSHstoH" << std::endl;
      std::cout << this->m_Scomm.row_rank() << "x" 
                << this->m_Scomm.col_rank()
                << " (S grid) sending" << std::endl;
      for (auto i = this->sendSHstoH.begin(); 
              i != this->sendSHstoH.end(); ++i) {
        std::cout << (*i).i << "x" << (*i).j << "::startidx::"
            << (*i).start_idx << "::nrows::" << (*i).nrows
            << std::endl;
      }

      std::cout << "recvSHstoH" << std::endl;
      std::cout << this->m_mpicomm.row_rank() << "x" 
                << this->m_mpicomm.col_rank()
                << " (A grid) receiving" << std::endl;
      for (auto i = this->recvSHstoH.begin(); 
              i != this->recvSHstoH.end(); ++i) {
        std::cout << (*i).i << "x" << (*i).j << "::startidx::"
            << (*i).start_idx << "::nrows::" << (*i).nrows
            << std::endl;
      }
    };
    mpi_serial_print(f_SHs2H);
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    PRINTROOT("DistPGDJointNMF() constructor successful (double comm).");
  }

  void distGradW() {
    // gradW = 2(W(HtH) - AH)
    if (stale_AH) {
      this->distAH();
      stale_AH = false;
    }
    if (stale_HtH) {
      this->distInnerProduct(this->H, &this->HtH);
      stale_HtH = false;
    }
    MPITIC;
    gradW = 2*(this->W * this->HtH - this->AHtij.t());
    double temp = MPITOC;
    this->time_stats.nongram_duration(temp);
    this->time_stats.compute_duration(temp);

    // TODO: Add support for regularisers
    m_gradW_calls++;
  }

  void distGradH() {
    // gradH = 2(H(WtW) - AtW) + 4alpha(H(HtH)-SH)
    if (stale_WtA) {
      this->distWtA();
      stale_WtA = false;
    }
    if (stale_WtW) {
      this->distInnerProduct(this->W, &this->WtW);
      stale_WtW = false;
    }
    if (stale_SHs) {
      this->distSHs();
      stale_SHs = false;
    }
    // Move SHs to A grid
    SHtij.zeros(arma::size(this->Ht));
    MPITIC;
    this->sendAtoB(this->SHstij, this->sendSHstoH, this->m_Scomm.gridComm(),
                   SHtij, this->recvSHstoH, this->m_mpicomm.gridComm());
    double temp = MPITOC;
    this->time_stats.communication_duration(temp);
    this->time_stats.sendrecv_duration(temp);
    if (stale_HtH) {
      this->distInnerProduct(this->H, &this->HtH);
      stale_HtH = false;
    }
    MPITIC;
    gradH = 2*((this->H * this->WtW) - this->WtAij.t()) +
            (4 * this->alpha()) * ((this->H * this->HtH) - SHtij.t());
    temp = MPITOC;
    this->time_stats.nongram_duration(temp);
    this->time_stats.compute_duration(temp);

    // TODO: Add support for regularisers
    m_gradH_calls++;
  }

  void computeNMF() {
#ifdef MPI_VERBOSE
    function<void()> f_print = [this] () {
      std::cout << this->m_mpicomm.rank() << "::" 
         << this->m_mpicomm.row_rank() << "x" << this->m_mpicomm.col_rank()
         << "::A::" << this->A.n_rows << "x" << this->A.n_cols 
         << std::endl;
      std::cout << this->m_Scomm.rank() << "::" 
         << this->m_Scomm.row_rank() << "x" << this->m_Scomm.col_rank()
         << "::S::" << this->S.n_rows << "x" << this->S.n_cols 
         << std::endl;
    };
    mpi_serial_print(f_print);
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    PRINTROOT("JointNMF hyperparameters::alpha::" << this->alpha()
                          << "::gamma::" << this->gamma()
                          << "::linesearch tries::" << this->m_numchks);
    PRINTROOT("ComputeNMF started.");
#ifdef MPI_VERBOSE
    DISTPRINTINFO(this->A);
    DISTPRINTINFO(this->S);
#endif
#ifdef __WITH__BARRIER__TIMING__
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    // Compute the initial objective
    // TODO: Change to work with regularisers
    MPITIC;
    this->computeObjectiveError(-1);
    double prevObj, initObj;
    initObj = this->fit_err_sq;
    prevObj = initObj;
    this->time_stats.duration(MPITOC);
    PRINTROOT("Initial objective::" << initObj 
      << "::Initial relative objective (wrt A)::" 
      << initObj / (2 * this->m_globalsqnormA));

    for (unsigned int iter = 0; iter < this->num_iterations(); iter++) {
      // Iter timer
      MPITIC;
      // Compute the gradients and their norms
      // gradW
      this->distGradW();

      MPITIC;
      double sqgradWnorm = 0.0;
      double sqlocalgW   = arma::norm(this->gradW, "fro");
      sqlocalgW = sqlocalgW * sqlocalgW;
      double temp = MPITOC;
      this->time_stats.nnls_duration(temp);
      this->time_stats.compute_duration(temp);

      MPITIC;
      MPI_Allreduce(&sqlocalgW, &sqgradWnorm, 1, 
          MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      temp = MPITOC;
      this->time_stats.allreduce_duration(temp);
      this->time_stats.communication_duration(temp);

      this->distGradH();
      
      MPITIC;
      double sqgradHnorm = 0.0;
      double sqlocalgH   = arma::norm(this->gradH, "fro");
      sqlocalgH = sqlocalgH * sqlocalgH;
      temp = MPITOC;
      this->time_stats.nnls_duration(temp);
      this->time_stats.compute_duration(temp);
      
      MPITIC;
      MPI_Allreduce(&sqlocalgH, &sqgradHnorm, 1, 
          MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      temp = MPITOC;
      this->time_stats.allreduce_duration(temp);
      this->time_stats.communication_duration(temp);

      // Compute the stepsize via linesearch
      double stepSize = 1;
      double gnorm = 1.0 / sqrt(sqgradWnorm + sqgradHnorm);
      bool objDec = false;
      int checks = 0;
      double newObj = prevObj;
      
      // Store the previous copies
      Wa = this->W;
      Ha = this->H;

      while ((!objDec) && (checks < m_numchks)) {
        // W step and project
        MPITIC;
        this->W = this->Wa 
                    - stepSize * (gnorm * gradW + this->gamma() * prevstepW);
        this->W.for_each(
          [](MAT::elem_type &val) { val = val > 0.0 ? val : 0.0; }
        );
        this->Wt = this->W.t();
        temp = MPITOC;
        this->time_stats.nnls_duration(temp);
        this->time_stats.compute_duration(temp);
        stale_WtW = true;
        stale_WtA = true;

        // H step and project
        MPITIC;
        this->H = this->Ha 
                    - stepSize * (gnorm * gradH + this->gamma() * prevstepH);
        this->H.for_each(
          [](MAT::elem_type &val) { val = val > 0.0 ? val : 0.0; }
        );
        this->Ht = this->H.t();
        temp = MPITOC;
        this->time_stats.nnls_duration(temp);
        this->time_stats.compute_duration(temp);
        
        // Transfer H (A grid) to Hs (S grid)
        MPITIC;
        this->sendAtoB(this->Ht, this->sendHtoHs, this->m_mpicomm.gridComm(),
          this->Hst, this->recvHtoHs, this->m_Scomm.gridComm());
        this->Hs = this->Hst.t();
        temp = MPITOC;
        this->time_stats.communication_duration(temp);
        this->time_stats.sendrecv_duration(temp);

        stale_HtH = true;
        stale_AH  = true;
        stale_SHs = true;

        // Compute the new objective
        this->computeObjectiveError(iter);
        newObj = this->fit_err_sq;

        if (newObj < prevObj) {
          objDec = true;
        } else {
          PRINTROOT("it=" << iter << "::line search iter::" << checks
              << "::step::" << stepSize
              << "::prevObj::" << prevObj << "::newObj::" << newObj);
          stepSize = stepSize / 2;
        }
        checks++;
      }
      if (!objDec) {
        PRINTROOT("it = " << iter << " line search failed.");
        m_lnfails++;
        stepSize   = 1e-6;
        
        // W step and project
        MPITIC;
        this->W = this->Wa 
                    - stepSize * (gnorm * gradW + this->gamma() * prevstepW);
        this->W.for_each(
          [](MAT::elem_type &val) { val = val > 0.0 ? val : 0.0; }
        );
        this->Wt = this->W.t();
        temp = MPITOC;
        this->time_stats.nnls_duration(temp);
        this->time_stats.compute_duration(temp);
        stale_WtW = true;
        stale_WtA = true;

        // H step and project
        MPITIC;
        this->H = this->Ha
                    - stepSize * (gnorm * gradH + this->gamma() * prevstepH);
        this->H.for_each(
          [](MAT::elem_type &val) { val = val > 0.0 ? val : 0.0; }
        );
        this->Ht = this->H.t();
        temp = MPITOC;
        this->time_stats.nnls_duration(temp);
        this->time_stats.compute_duration(temp);

        // Transfer H (A grid) to Hs (S grid)
        MPITIC;
        this->sendAtoB(this->Ht, this->sendHtoHs, this->m_mpicomm.gridComm(),
          this->Hst, this->recvHtoHs, this->m_Scomm.gridComm());
        this->Hs = this->Hst.t();
        temp = MPITOC;
        this->time_stats.communication_duration(temp);
        this->time_stats.sendrecv_duration(temp);

        stale_HtH = true; 
        stale_AH  = true;
        stale_SHs = true;

        // Compute the new objective
        this->computeObjectiveError(iter);
        newObj = this->fit_err_sq;
      }

      MPITIC;
      prevObj = newObj;
      prevstepW = this->W - this->Wa;
      prevstepH = this->H - this->Ha;
      temp = MPITOC;
      this->time_stats.nnls_duration(temp);
      this->time_stats.compute_duration(temp);

      // Iter timer
      this->time_stats.duration(MPITOC);

      PRINTROOT("it::" << iter 
          << "::stepsize::" << stepSize << "::invgnorm::" << gnorm
          << "::gradW norm sq::" << sqgradWnorm 
          << "::gradH norm sq::" << sqgradHnorm
          << "::objective::" << newObj);
      printObjective(iter);
      PRINTROOT("completed it=" << iter
                  << "::taken::" << this->time_stats.duration());
    }// end for loop
    MPI_Barrier(MPI_COMM_WORLD);
    this->reportTime(this->time_stats.duration(), "total_d");
    this->reportTime(this->time_stats.communication_duration(), "total_comm");
    this->reportTime(this->time_stats.compute_duration(), "total_comp");
    this->reportTime(this->time_stats.allgather_duration(), "total_allgather");
    this->reportTime(this->time_stats.allreduce_duration(), "total_allreduce");
    this->reportTime(this->time_stats.reducescatter_duration(),
                     "total_reducescatter");
    this->reportTime(this->time_stats.gram_duration(), "total_gram");
    this->reportTime(this->time_stats.mm_duration(), "total_mm");
    this->reportTime(this->time_stats.nnls_duration(), "total_nnls");
    this->reportTime(this->time_stats.sendrecv_duration(), "total_sendrecv");
    this->reportTime(this->time_stats.reg_duration(), "total_reg");
    this->reportTime(this->time_stats.err_compute_duration(),
                      "total_err_compute");
    this->reportTime(this->time_stats.err_communication_duration(),
                      "total_err_communication");
    // Matmul times
    this->reportTime(this->time_stats.WtA_compute_duration(),
                       "total_WtA_compute");
    this->reportTime(this->time_stats.WtA_communication_duration(),
                       "total_WtA_communication");
    this->reportTime(this->time_stats.AH_compute_duration(),
                       "total_AH_compute");
    this->reportTime(this->time_stats.AH_communication_duration(),
                       "total_AH_communication");
    this->reportTime(this->time_stats.SHs_compute_duration(),
                       "total_SHs_compute");
    this->reportTime(this->time_stats.SHs_communication_duration(),
                       "total_SHs_communication");
    // Counters
    PRINTROOT("Function calls::" << m_func_calls << "::Grad W calls::"
        << m_gradW_calls << "::Grad H calls::" << m_gradH_calls 
        << "::Line search failures::" << m_lnfails);
  }

  /**
   * Computes the various terms in the objective function for JointNMF
   */
  void computeObjectiveError(const int it) {
    // Reuse WtA for fast fit computation (A - WHt)
    if (stale_WtA) {
      this->distWtA();
      stale_WtA = false;
    }

    MPITIC;
    this->localWtAijH = this->WtAij * this->H;
    double temp = MPITOC;
    this->time_stats.err_compute_duration(temp);

    if (stale_WtW) {
      this->distInnerProduct(this->W, &this->WtW);
      stale_WtW = false;
    }

    // Reuse H2tS for fast fit computation (S - HHt)
    if (stale_SHs) {
      this->distSHs();
      stale_SHs = false;

      // Move SHs to A grid
      SHtij.zeros(arma::size(this->Ht));
      MPITIC;
      this->sendAtoB(this->SHstij, this->sendSHstoH, this->m_Scomm.gridComm(),
                   SHtij, this->recvSHstoH, this->m_mpicomm.gridComm());
      double temp = MPITOC;
      this->time_stats.err_communication_duration(temp);
      this->time_stats.sendrecv_duration(temp);
    }

    MPITIC;
    this->localSHtijH = SHtij * this->H;
    temp = MPITOC;
    this->time_stats.err_compute_duration(temp);

    if (stale_HtH) {
      this->distInnerProduct(this->H, &this->HtH);
      stale_HtH = false;
    }

#ifdef MPI_VERBOSE
    DISTPRINTINFO("::it=" << it << PRINTMAT(this->W));
    DISTPRINTINFO("::it=" << it << PRINTMAT(this->WtAij));
    DISTPRINTINFO("::it=" << it << PRINTMAT(this->localWtAijH));
    DISTPRINTINFO("::it=" << it << PRINTMAT(this->H));
    DISTPRINTINFO("::it=" << it << PRINTMAT(this->SHtij));
    DISTPRINTINFO("::it=" << it << PRINTMAT(this->Hs));
    DISTPRINTINFO("::it=" << it << PRINTMAT(this->SHstij));
    DISTPRINTINFO("::it=" << it << PRINTMAT(this->localSHtijH));
    DISTPRINTINFO("::it=" << it << PRINTMAT(this->HtH));
    DISTPRINTINFO("::it=" << it << PRINTMAT(this->WtW));
#endif
    
    // Compute the global WtAH
    MPITIC;
    MPI_Allreduce(this->localWtAijH.memptr(), this->WtAijH.memptr(),
        this->k * this->k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    temp = MPITOC;
    this->time_stats.err_communication_duration(temp);
    this->time_stats.allreduce_duration(temp);

    // Compute the global H2tSHs
    MPITIC;
    MPI_Allreduce(this->localSHtijH.memptr(), this->SHtijH.memptr(),
        this->k * this->k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    temp = MPITOC;
    this->time_stats.err_communication_duration(temp);
    this->time_stats.allreduce_duration(temp);

#ifdef MPI_VERBOSE
    DISTPRINTINFO("::it=" << it << PRINTMAT(WtAijH));
    DISTPRINTINFO("::it=" << it << PRINTMAT(SHtijH));
#endif

    // Compute the fits
    MPITIC;
    double tWtAijH = arma::trace(this->WtAijH);
    double tWtWHtH = arma::trace(this->WtW * this->HtH);
    this->m_Afit = this->m_globalsqnormA - (2 * tWtAijH) + tWtWHtH;
    
    double tSHtijH = arma::trace(this->SHtijH);
    double tHtHHtH = arma::trace(this->HtH * this->HtH);
    this->m_Sfit = this->m_globalsqnormS - (2 * tSHtijH) + tHtHHtH;
    temp = MPITOC;
    this->time_stats.err_compute_duration(temp);

    // Compute the regularizers
    MAT onematrix = arma::ones<MAT>(this->k, this->k);

    MPITIC;
    // Frobenius norm regularization
    double fro_W_sq = arma::trace(this->WtW);
    double fro_W_obj = this->m_regW(0) * fro_W_sq;
    this->normW = sqrt(fro_W_sq);
    double fro_H_sq = arma::trace(this->HtH);
    double fro_H_obj = this->m_regH(0) * fro_H_sq;
    this->normH = sqrt(fro_H_sq);
    
    // L_{12} norm regularization
    double l1_W_sq = arma::trace(this->WtW * onematrix);
    double l1_W_obj = this->m_regW(1) * l1_W_sq;
    this->l1normW = sqrt(l1_W_sq);
    double l1_H_sq = arma::trace(this->HtH * onematrix);
    double l1_H_obj = this->m_regH(1) * l1_H_sq;
    this->l1normH = sqrt(l1_H_sq);
    
    // Objective being minimized
    this->fit_err_sq = this->m_Afit + (this->alpha() * this->m_Sfit);
    this->objective_err = this->fit_err_sq + fro_W_obj + fro_H_obj 
              + l1_W_obj + l1_H_obj;

    m_func_calls++;
  
    temp = MPITOC;
    this->time_stats.err_compute_duration(temp);
  }

  void printObjective(int itr) {
    double gnormA = sqrt(this->m_globalsqnormA);
    double gnormS = sqrt(this->m_globalsqnormS);
    double Aerr   = (this->m_Afit > 0)? sqrt(this->m_Afit) : gnormA;
    double Serr   = (this->m_Sfit > 0)? sqrt(this->m_Sfit) : gnormS;
    PRINTROOT("Completed it = " << itr << "::algo::" << this->m_algorithm
          << "::k::" << this->k << std::endl
          << "objective::" << this->objective_err 
          << "::squared error::" << this->fit_err_sq << std::endl
          << "relative objective (wrt A)::"
          << this->objective_err / (2 * this->m_globalsqnormA) << std::endl
          << "A error::" << Aerr 
          << "::A relative error::" << Aerr / gnormA << std::endl
          << "S error::" << Serr 
          << "::S relative error::" << Serr / gnormS << std::endl
          << "W frobenius norm::" << this->normW 
          << "::W L_12 norm::" << this->l1normW << std::endl
          << "H frobenius norm::" << this->normH 
          << "::H L_12 norm::" << this->l1normH);
  }

  void saveOutput(std::string outfname) {
    INFO << "from save output function: " << outfname << std::endl;

    // NOTE: writeOutput function doesn't use DistJointNMFIO m_A, so passing in W should
    // theoretically work for now ----> might want to overload the DistJointNMFIO constructor
    // to only require a communicatior...
    DistJointNMFIO<MAT> test(this->m_mpicomm, this->W);
    test.writeOutput(this->W, this->H, outfname);
  }

  ~DistPGDJointNMF() {
    /*
       tempHtH.clear();
       tempWtW.clear();
       tempAHtij.clear();
       tempWtAij.clear();
     */
  }
  /// Sets the momentum parameter
  void gamma(const double b) { this->m_gamma = b; }
  // Returns the momentum parameter
  double gamma() { return this->m_gamma; }
};  // class DistPGDJointNMF

}  // namespace planc

#endif  // DISTJNMF_DISTPGD_HPP_
