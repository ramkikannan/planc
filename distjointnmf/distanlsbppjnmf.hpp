/* Copyright 2022 Ramakrishnan Kannan */

#ifndef DISTJNMF_DISTANLSBPP_HPP_
#define DISTJNMF_DISTANLSBPP_HPP_

#include "distjointnmf/distjointnmf.hpp"
#include "distjointnmf/jnmfmpicomm.hpp"
#include "distjointnmf/distjointnmfio.hpp"

#include "nnls/bppnnls.hpp"

/**
 * Provides the updateW and updateH for the
 * distributed ANLS/BPP algorithm.
 */

#ifdef BUILD_CUDA
#define ONE_THREAD_MATRIX_SIZE 1000
#include <omp.h>
#else
#define ONE_THREAD_MATRIX_SIZE 2000
#endif

namespace planc {

template <class T1, class T2>
class DistANLSBPPJointNMF : public DistJointNMF<T1, T2> {
 protected:
  // Surrogate variables for symmetry constraints
  MAT H2;     /// H2 is of size (globaln/p)*k
  MAT H2t;    /// H2t is of size k*(globaln/p)
  MAT H2tH2;  /// H2 is of size (globaln/p)*k
  MAT H2tSij; /// H2tSij is of size k*(globaln/p)

  gridInfo H2grid;

  // Temporaries needed for matrix multiply
  MAT H2t_blk, H2tSij_blk, H2it, H2itSij;

  // Gatherv and Reducescatter variables
  std::vector<int> gatherH2tScnts;
  std::vector<int> gatherH2tSdisp;
  std::vector<int> scatterH2tScnts;

  // S grid (H2) --> S grid (Hs) communication variables
  std::vector<procInfo> sendH2toHs;
  std::vector<procInfo> recvH2toHs;
  std::vector<procInfo> sendHstoH2;
  std::vector<procInfo> recvHstoH2;

  // Variables needed for objective error
  MAT prevH, prevHs;
  MAT prevHtH;
  MAT WtAijH, localWtAijH;      /// used for fast error computation
  MAT H2tSijHs, localH2tSijHs;  /// used for fast error computation
  double m_localdiff, m_symmdiff;
  
  double m_Afit, m_Sfit, normH2;

  // Temporary variables for NNLS
  MAT lhs;
  MAT crossFac, crossFac2;

  // Hyperparameters
  double m_beta, m_Smax;

  /**
   * Allocate temporaries needed for ANLS JointNMF algorithms
   */
  void allocateANLSMatrices() {
    // H2tS variables
    scatterH2tScnts.resize(NUMROWPROCS_C(this->m_Scomm));
    fillVector<int>(0, &scatterH2tScnts);
    gatherH2tScnts.resize(NUMCOLPROCS_C(this->m_Scomm));
    fillVector<int>(0, &gatherH2tScnts);
    gatherH2tSdisp.resize(NUMCOLPROCS_C(this->m_Scomm));
    fillVector<int>(0, &gatherH2tSdisp);

    H2t.zeros(this->k, this->H2.n_rows);
    H2tH2.zeros(this->k, this->k);
    H2tSij.zeros(this->k, this->Hs.n_rows);

    H2t_blk.zeros(this->m_perk, this->H2.n_rows);
    H2tSij_blk.zeros(this->m_perk, this->Hs.n_rows);
    H2it.zeros(this->m_perk, this->S.n_rows);
    H2itSij.zeros(this->m_perk, this->S.n_cols);

    // NNLS variables
    lhs.zeros(this->k, this->k);
  }

  /**
   * Communication needed for matmul (H2tS) computed on the 
   * the S grid (p_r2 x p_c2).
   */
  void setupANLSMatmulComms() {
    // WtA
    // Allgatherv counts
    gatherH2tScnts[0] = itersplit(this->S.n_rows, 
                          NUMCOLPROCS_C(this->m_Scomm), 0) * this->m_perk;
    gatherH2tSdisp[0] = 0;
    for (int i = 1; i < NUMCOLPROCS_C(this->m_Scomm); i++) {
      gatherH2tScnts[i] = itersplit(this->S.n_rows,
                            NUMCOLPROCS_C(this->m_Scomm), i) * this->m_perk;
      gatherH2tSdisp[i] = gatherH2tSdisp[i-1] + gatherH2tScnts[i-1];
    }
    // Reducescatter counts
    for (int i = 0; i < NUMROWPROCS_C(this->m_Scomm); i++) {
      scatterH2tScnts[i] = itersplit(this->S.n_cols,
                            NUMROWPROCS_C(this->m_Scomm), i) * this->m_perk;
    }
  }

  /**
   * Communicator needed to swap H2 (S grid) and Hs (S grid) for
   * symmetry regularisation 
   */
  void setupH2HsComms() {
    // Setup grids
    H2grid.pr    = NUMROWPROCS_C(this->m_Scomm);
    H2grid.pc    = NUMCOLPROCS_C(this->m_Scomm);
    H2grid.order = 'R';

    // Hs (S grid) --> H2 (S grid)
    sendHstoH2 = getsendinfo(this->m_globaln,
                  this->m_Scomm.row_rank(), this->m_Scomm.col_rank(),
                  this->Hsgrid, this->H2grid);
    recvHstoH2 = getrecvinfo(this->m_globaln,
                  this->m_Scomm.row_rank(), this->m_Scomm.col_rank(),
                  this->Hsgrid, this->H2grid);

    // H2 (S grid) --> Hs (S grid)
    sendH2toHs = getsendinfo(this->m_globaln,
                  this->m_Scomm.row_rank(), this->m_Scomm.col_rank(),
                  this->H2grid, this->Hsgrid);
    recvH2toHs = getrecvinfo(this->m_globaln,
                  this->m_Scomm.row_rank(), this->m_Scomm.col_rank(),
                  this->H2grid, this->Hsgrid);
  }

  /**
   * ANLS/BPP with chunking the RHS into smaller independent
   * subproblems
   */
  void updateOtherGivenOneMultipleRHS(const MAT& giventGiven,
                                      const MAT& giventInput, MAT* othermat) {
    UINT numChunks = giventInput.n_cols / ONE_THREAD_MATRIX_SIZE;
    if (numChunks * ONE_THREAD_MATRIX_SIZE < giventInput.n_cols) numChunks++;

    for (UINT i = 0; i < numChunks; i++) {
      UINT spanStart = i * ONE_THREAD_MATRIX_SIZE;
      UINT spanEnd = (i + 1) * ONE_THREAD_MATRIX_SIZE - 1;
      if (spanEnd > giventInput.n_cols - 1) {
        spanEnd = giventInput.n_cols - 1;
      }
      BPPNNLS<MAT, VEC> subProblem(giventGiven,
                            (MAT)giventInput.cols(spanStart, spanEnd), true);
      subProblem.solveNNLS();
      (*othermat).rows(spanStart, spanEnd) = subProblem.getSolutionMatrix().t();
    }
  }

 protected:
  /**
   * Updates the factor with LHS and RHS of the normal equations.
   * @param[in] AtA is the LHS of the normal equations
   * @param[in] AtB is the RHS of the normal equations
   * @param[in] X is a reference to the matrix to be updated
   * @param[in] Xt is a reference to the transpose of the matrix to be updated
   */
  void updateFactor(MAT &AtA, MAT &AtB, MAT &X, MAT &Xt) {
    updateOtherGivenOneMultipleRHS(AtA, AtB, &X);
    Xt = X.t();
  }

  /**
   * This is matrix-multiplication routine based on
   * the 2D algorithm on a p = p_r2 x p_c2 grid 
   * m_Scomm is the grid communicator
   * S is of size (n / p_r2) x (n / p_c2)
   * H2 is of size (n / p) x k
   * this->m_Scomm.comm_subs()[0] is column communicator.
   * this->m_Scomm.comm_subs()[1] is row communicator.
   */
  void distH2tS() {
    for (int i = 0; i < this->m_num_k_blocks; i++) {
      int start_row = i * this->m_perk;
      int end_row = (i + 1) * this->m_perk - 1;
      H2t_blk = H2t.rows(start_row, end_row);
      distH2tSBlock();
      H2tSij.rows(start_row, end_row) = H2tSij_blk;
    }
  }

  void distH2tSBlock() {
    int sendcnt = (this->H2.n_rows) * this->m_perk;
    H2it.zeros();
    MPITIC;  // allgather H2tS
    MPI_Allgatherv(H2t_blk.memptr(), sendcnt, MPI_DOUBLE, H2it.memptr(),
                  &(gatherH2tScnts[0]), &(gatherH2tSdisp[0]), MPI_DOUBLE,
                  this->m_Scomm.commSubs()[1]);
    double temp = MPITOC;  // allgather H2tS
#ifdef MPI_VERBOSE
    DISTPRINTINFO(PRINTMAT(H2t_blk));
    DISTPRINTINFO(PRINTMAT(H2it));
#endif
    this->time_stats.communication_duration(temp);
    this->time_stats.allgather_duration(temp);
    this->time_stats.H2tS_communication_duration(temp);
    MPITIC;  // mm H2tS
    this->H2itSij = this->H2it * this->S;
    temp = MPITOC;  // mm H2tS
#ifdef MPI_VERBOSE
    DISTPRINTINFO(PRINTMAT(this->H2itSij));
#endif
    this->time_stats.compute_duration(temp);
    this->time_stats.mm_duration(temp);
    this->time_stats.H2tS_compute_duration(temp);
    this->reportTime(temp, "H2tS::");
    H2tSij_blk.zeros();
    MPITIC;  // reduce_scatter H2tS
    MPI_Reduce_scatter(this->H2itSij.memptr(), this->H2tSij_blk.memptr(),
                       &(scatterH2tScnts[0]), MPI_DOUBLE, MPI_SUM,
                       this->m_Scomm.commSubs()[0]);
    temp = MPITOC;  // reduce_scatter H2tS
    this->time_stats.communication_duration(temp);
    this->time_stats.reducescatter_duration(temp);
    this->time_stats.H2tS_communication_duration(temp);
  }

 public:
  DistANLSBPPJointNMF(const T1& input, const T2& conn,
          const MAT& leftlowrankfactor, const MAT& rightlowrankfactor,
          const MPICommunicatorJNMF& communicator, const int numkblks)
      : DistJointNMF<T1, T2>(input, conn, 
              leftlowrankfactor, rightlowrankfactor, communicator, numkblks) {
    // Initialise the surrogate matrix
    H2 = arma::zeros<MAT>(itersplit(this->S.n_rows,
            NUMCOLPROCS_C(this->m_Scomm), MPI_COL_RANK_C(this->m_Scomm)),
            this->k);

    // Allocate temporary matrices
    allocateANLSMatrices();

    // Setup communication stuff
    setupANLSMatmulComms();
    setupH2HsComms();

#ifdef MPI_VERBOSE
    function<void()> f_Hs2H2 = [this] () {
      std::cout << "Hs2H2 print." << std::endl;
      
      std::cout << "sendHstoH2" << std::endl;
      std::cout << this->m_Scomm.row_rank() << "x" 
                << this->m_Scomm.col_rank()
                << " (S grid) sending" << std::endl;
      for (auto i = this->sendHstoH2.begin(); 
              i != this->sendHstoH2.end(); ++i) {
        std::cout << (*i).i << "x" << (*i).j << "::startidx::"
            << (*i).start_idx << "::nrows::" << (*i).nrows
            << std::endl;
      }

      std::cout << "recvHstoH2" << std::endl;
      std::cout << this->m_Scomm.row_rank() << "x" 
                << this->m_Scomm.col_rank()
                << " (S grid) receiving" << std::endl;
      for (auto i = this->recvHstoH2.begin(); 
              i != this->recvHstoH2.end(); ++i) {
        std::cout << (*i).i << "x" << (*i).j << "::startidx::"
            << (*i).start_idx << "::nrows::" << (*i).nrows
            << std::endl;
      }

      std::cout << "sendH2toHs" << std::endl;
      std::cout << this->m_Scomm.row_rank() << "x" 
                << this->m_Scomm.col_rank()
                << " (S grid) sending" << std::endl;
      for (auto i = this->sendH2toHs.begin(); 
              i != this->sendH2toHs.end(); ++i) {
        std::cout << (*i).i << "x" << (*i).j << "::startidx::"
            << (*i).start_idx << "::nrows::" << (*i).nrows
            << std::endl;
      }

      std::cout << "recvH2toHs" << std::endl;
      std::cout << this->m_Scomm.row_rank() << "x" 
                << this->m_Scomm.col_rank()
                << " (S grid) receiving" << std::endl;
      for (auto i = this->recvH2toHs.begin(); 
              i != this->recvH2toHs.end(); ++i) {
        std::cout << (*i).i << "x" << (*i).j << "::startidx::"
            << (*i).start_idx << "::nrows::" << (*i).nrows
            << std::endl;
      }
    };
    mpi_serial_print(f_Hs2H2);
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // Transfer Hs (S grid) to H2 (S grid)
    this->sendAtoB(this->Hst, this->sendHstoH2, this->m_Scomm.gridComm(),
        this->H2t, this->recvHstoH2, this->m_Scomm.gridComm());
    this->H2 = this->H2t.t();

    // Get S max
    double local_Smax = this->S.max();
    MPI_Allreduce(&local_Smax, &this->m_Smax, 1, MPI_DOUBLE, MPI_MAX,
          MPI_COMM_WORLD);

    // Set beta
    this->m_beta = this->m_alpha * this->m_Smax;

    MPI_Barrier(MPI_COMM_WORLD);
    PRINTROOT("DistANLSBPPJointNMF() constructor successful (single comm).");
  }

  DistANLSBPPJointNMF(const T1& input, const T2& conn,
          const MAT& leftlowrankfactor, const MAT& rightlowrankfactor,
          const MPICommunicatorJNMF& Acomm, const MPICommunicatorJNMF& Scomm,
          const int numkblks)
      : DistJointNMF<T1, T2>(input, conn, 
              leftlowrankfactor, rightlowrankfactor, 
              Acomm, Scomm, numkblks) {
    // Initialise the surrogate matrix
    H2 = arma::zeros<MAT>(itersplit(this->S.n_rows,
            NUMCOLPROCS_C(this->m_Scomm), MPI_COL_RANK_C(this->m_Scomm)),
            this->k);
    
    // Allocate temporary matrices
    allocateANLSMatrices();

    // Setup communication stuff
    setupANLSMatmulComms();
    setupH2HsComms();

#ifdef MPI_VERBOSE
    function<void()> f_Hs2H2 = [this] () {
      std::cout << "Hs2H2 print." << std::endl;
      
      std::cout << "sendHstoH2" << std::endl;
      std::cout << this->m_Scomm.row_rank() << "x" 
                << this->m_Scomm.col_rank()
                << " (S grid) sending" << std::endl;
      for (auto i = this->sendHstoH2.begin(); 
              i != this->sendHstoH2.end(); ++i) {
        std::cout << (*i).i << "x" << (*i).j << "::startidx::"
            << (*i).start_idx << "::nrows::" << (*i).nrows
            << std::endl;
      }

      std::cout << "recvHstoH2" << std::endl;
      std::cout << this->m_Scomm.row_rank() << "x" 
                << this->m_Scomm.col_rank()
                << " (S grid) receiving" << std::endl;
      for (auto i = this->recvHstoH2.begin(); 
              i != this->recvHstoH2.end(); ++i) {
        std::cout << (*i).i << "x" << (*i).j << "::startidx::"
            << (*i).start_idx << "::nrows::" << (*i).nrows
            << std::endl;
      }

      std::cout << "sendH2toHs" << std::endl;
      std::cout << this->m_Scomm.row_rank() << "x" 
                << this->m_Scomm.col_rank()
                << " (S grid) sending" << std::endl;
      for (auto i = this->sendH2toHs.begin(); 
              i != this->sendH2toHs.end(); ++i) {
        std::cout << (*i).i << "x" << (*i).j << "::startidx::"
            << (*i).start_idx << "::nrows::" << (*i).nrows
            << std::endl;
      }

      std::cout << "recvH2toHs" << std::endl;
      std::cout << this->m_Scomm.row_rank() << "x" 
                << this->m_Scomm.col_rank()
                << " (S grid) receiving" << std::endl;
      for (auto i = this->recvH2toHs.begin(); 
              i != this->recvH2toHs.end(); ++i) {
        std::cout << (*i).i << "x" << (*i).j << "::startidx::"
            << (*i).start_idx << "::nrows::" << (*i).nrows
            << std::endl;
      }
    };
    mpi_serial_print(f_Hs2H2);
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // Transfer Hs (S grid) to H2 (S grid)
    this->sendAtoB(this->Hst, this->sendHstoH2, this->m_Scomm.gridComm(),
        this->H2t, this->recvHstoH2, this->m_Scomm.gridComm());
    this->H2 = this->H2t.t();

    // Get S max
    double local_Smax = this->S.max();
    MPI_Allreduce(&local_Smax, &this->m_Smax, 1, MPI_DOUBLE, MPI_MAX,
          MPI_COMM_WORLD);

    // Set beta
    this->m_beta = this->m_alpha * this->m_Smax;

    MPI_Barrier(MPI_COMM_WORLD);
    PRINTROOT("DistANLSBPPJointNMF() constructor successful (double comm).");
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
                          << "::beta::" << this->beta());
    PRINTROOT("ComputeNMF started.");
#ifdef MPI_VERBOSE
    DISTPRINTINFO(this->A);
    DISTPRINTINFO(this->S);
#endif
    if (this->is_compute_error()) {
      prevH.zeros(size(this->H));
      prevHs.zeros(size(this->Hs));

      // k x k matrices
      WtAijH.zeros(this->k, this->k);
      localWtAijH.zeros(this->k, this->k);
      H2tSijHs.zeros(this->k, this->k);
      localH2tSijHs.zeros(this->k, this->k);
    }
#ifdef __WITH__BARRIER__TIMING__
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    for (unsigned int iter = 0; iter < this->num_iterations(); iter++) {
      // error comp stuff goes here
      if (iter > 0 && this->is_compute_error()) {
        MPITIC;
        this->prevH   = this->H;
        this->prevHs  = this->Hs;
        double temp = MPITOC;
        this->time_stats.err_compute_duration(temp);
      }
      // Update W
      MPITIC;
      {
        // Compute HtH
        this->distInnerProduct(this->H, &this->HtH);
        MPITIC;
        this->HstHs = this->HtH;
        lhs = this->HtH;
        double temp = MPITOC;
        this->time_stats.compute_duration(temp);
        this->time_stats.nnls_duration(temp);

        MPITIC;
        this->applyReg(this->regW(), &lhs);
        temp = MPITOC;
        this->time_stats.compute_duration(temp);
        this->time_stats.reg_duration(temp);

#ifdef MPI_VERBOSE
        PRINTROOT("::it=" << iter << PRINTMAT(lhs));
#endif
        // Compute AH
        this->distAH();

#ifdef MPI_VERBOSE
        DISTPRINTINFO("::it=" << iter << PRINTMAT(this->AHtij));
#endif
        // Solve for W
        MPITIC;
        updateFactor(lhs, this->AHtij, this->W, this->Wt);
        temp = MPITOC; // nnls W

        // Put time stats
        this->time_stats.compute_duration(temp);
        this->time_stats.nnls_duration(temp);
        this->reportTime(temp, "NNLS::W::");

        // error comp stuff
        if (iter > 0 && this->is_compute_error()) {
          MPITIC;
          this->prevHtH = this->HtH;
          double temp = MPITOC;
          this->time_stats.err_compute_duration(temp);
        }
      }
      this->time_stats.duration(MPITOC); // total_d W & H & H2
      // Update H2
      MPITIC;
      {
        // Compute HstHs (already computed)
        MPITIC;
        lhs = this->alpha() * this->HstHs;
        double temp = MPITOC;
        this->time_stats.nnls_duration(temp);
        this->time_stats.compute_duration(temp);

#ifdef MPI_VERBOSE
        PRINTROOT("::it=" << iter << PRINTMAT(lhs));
#endif
        // Compute SH
        this->distSHs();
        MPITIC;
        this->SHstij = this->alpha() * this->SHstij;
        temp = MPITOC;
        this->time_stats.compute_duration(temp);
        this->time_stats.nnls_duration(temp);

        // Swaps for symmetric regularization
        crossFac.zeros(arma::size(this->H2t));
        MPITIC;
        this->sendAtoB(this->Hst, this->sendHstoH2, this->m_Scomm.gridComm(),
          crossFac, this->recvHstoH2, this->m_Scomm.gridComm());
        temp = MPITOC;
        this->time_stats.communication_duration(temp);
        this->time_stats.sendrecv_duration(temp);

        // Compute localdiff for error calculations
        if (this->is_compute_error()) {
          MPITIC;
          this->m_localdiff = arma::norm(this->H2t - crossFac, "fro");
          temp = MPITOC;
          this->time_stats.err_compute_duration(temp);
        }

        // Make the regularization
        MPITIC;
        this->applySymmetricReg(this->beta(), &lhs, &crossFac, &this->SHstij);
        temp = MPITOC;
        this->time_stats.compute_duration(temp);
        this->time_stats.reg_duration(temp);

        // Solve for H2
        MPITIC;
        updateFactor(lhs, this->SHstij, this->H2, this->H2t);
        temp = MPITOC; // nnls H2

        // Put time stats
        this->time_stats.compute_duration(temp);
        this->time_stats.nnls_duration(temp);
        this->reportTime(temp, "NNLS::H2::");
        
        // error comp stuff goes here
        if (this->is_compute_error()) {
          MPITIC;
          // Remove the symmetric terms
          this->removeSymmetricReg(this->beta(), &lhs, &crossFac, &this->SHstij);
          // Remove scaling by alpha
          this->SHstij = this->SHstij / this->alpha();
          temp = MPITOC;
          this->time_stats.err_compute_duration(temp);
        }
      }
      this->time_stats.duration(MPITOC); // total_d W & H & H2
      // Update H
      MPITIC;
      {
        // Compute the lhs = WtW + alpha*H2tH2 + \beta I_k + reg
        // Compute WtW
        this->distInnerProduct(this->W, &this->WtW);
        this->distInnerProduct(this->H2, &this->H2tH2);
        MPITIC;
        lhs = this->WtW + (this->alpha() * this->H2tH2);
        
        this->applyReg(this->regH(), &lhs);
        double temp = MPITOC;
        this->time_stats.compute_duration(temp);
        this->time_stats.nnls_duration(temp);

#ifdef MPI_VERBOSE
        PRINTROOT("::it=" << iter << PRINTMAT(lhs));
#endif
        // Compute WtA
        this->distWtA();

        // Compute H2tS
        this->distH2tS();
        MPITIC;
        this->H2tSij = this->alpha() * this->H2tSij;
        temp = MPITOC;
        this->time_stats.compute_duration(temp);
        this->time_stats.nnls_duration(temp);

        // Add the symmetry constraints for this  
        crossFac.zeros(arma::size(this->Hst));
        MPITIC;
        this->sendAtoB(this->H2t, this->sendH2toHs, this->m_Scomm.gridComm(),
          crossFac, this->recvH2toHs, this->m_Scomm.gridComm());
        temp = MPITOC;
        this->time_stats.communication_duration(temp);
        this->time_stats.sendrecv_duration(temp);

        // Make the regularization
        MPITIC;
        this->applySymmetricReg(this->beta(), &lhs, &crossFac, &this->H2tSij);
        temp = MPITOC;
        this->time_stats.compute_duration(temp);
        this->time_stats.reg_duration(temp);

        // Make the final RHS on the A grid
        crossFac2.zeros(arma::size(this->Ht));
        MPITIC;
        this->sendAtoB(this->H2tSij, this->sendHstoH, this->m_Scomm.gridComm(),
          crossFac2, this->recvHstoH, this->m_mpicomm.gridComm());
        temp = MPITOC;
        this->time_stats.communication_duration(temp);
        this->time_stats.sendrecv_duration(temp);

        // Make the complete RHS
        MPITIC;
        this->WtAij = this->WtAij + crossFac2;
        temp = MPITOC;
        this->time_stats.compute_duration(temp);
        this->time_stats.nnls_duration(temp);

        // Solve for H
        MPITIC;
        updateFactor(lhs, this->WtAij, this->H, this->Ht);
        temp = MPITOC; // nnls H

        // Put time stats
        this->time_stats.compute_duration(temp);
        this->time_stats.nnls_duration(temp);
        this->reportTime(temp, "NNLS::H::");

        // Sync Hs and H
        MPITIC;
        this->sendAtoB(this->Ht, this->sendHtoHs, this->m_mpicomm.gridComm(),
          this->Hst, this->recvHtoHs, this->m_Scomm.gridComm());
        temp = MPITOC;
        this->time_stats.communication_duration(temp);
        this->time_stats.sendrecv_duration(temp);
        this->Hs = this->Hst.t();

        // error comp stuff goes here
        if (this->is_compute_error()) {
          MPITIC;
          // Remove the crossfactor from WtA
          this->WtAij = this->WtAij - crossFac2;
          // Remove the symmetric terms from H2tS
          this->removeSymmetricReg(this->beta(), &lhs, &crossFac, 
                  &this->H2tSij);
          // Remove the scaling terms
          this->H2tSij = this->H2tSij / this->alpha();
          temp = MPITOC;
          this->time_stats.err_compute_duration(temp);
        }
      }
      this->time_stats.duration(MPITOC); // total_d W & H & H2
      if (iter > 0 && this->is_compute_error()) {
        this->computeObjectiveError(iter);
        this->printObjective(iter);
      }
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
    if (this->is_compute_error()) {
      this->reportTime(this->time_stats.err_compute_duration(),
                       "total_err_compute");
      this->reportTime(this->time_stats.err_communication_duration(),
                       "total_err_communication");
    }
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
    this->reportTime(this->time_stats.H2tS_compute_duration(),
                       "total_H2tS_compute");
    this->reportTime(this->time_stats.H2tS_communication_duration(),
                       "total_H2tS_communication");
  }

  /**
   * Computes the various terms in the objective function for JointNMF
   */
  void computeObjectiveError(const int it) {
    // Reuse WtA for fast fit computation (A - WHt)
    MPITIC;
    this->localWtAijH = this->WtAij * this->prevH;
    double temp = MPITOC;
    this->time_stats.err_compute_duration(temp);
    
    // Reuse H2tS for fast fit computation (S - H2Hst)
    MPITIC;
    this->localH2tSijHs = this->H2tSij * this->prevHs;
    temp = MPITOC;
    this->time_stats.err_compute_duration(temp);
    
#ifdef MPI_VERBOSE
    DISTPRINTINFO("::it=" << it << PRINTMAT(this->WtAij));
    DISTPRINTINFO("::it=" << it << PRINTMAT(this->localWtAijH));
    DISTPRINTINFO("::it=" << it << PRINTMAT(this->prevH));
    DISTPRINTINFO("::it=" << it << PRINTMAT(this->H2tSij));
    DISTPRINTINFO("::it=" << it << PRINTMAT(this->localH2tSijHs));
    DISTPRINTINFO("::it=" << it << PRINTMAT(this->prevHs));
    PRINTROOT("::it=" << it << PRINTMAT(this->WtW));
    PRINTROOT("::it=" << it << PRINTMAT(this->prevHtH));
#endif
    
    // Compute the global WtAH
    MPITIC;
    MPI_Allreduce(this->localWtAijH.memptr(), this->WtAijH.memptr(),
        this->k * this->k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    temp = MPITOC;
    this->time_stats.err_communication_duration(temp);

    // Compute the global H2tSHs
    MPITIC;
    MPI_Allreduce(this->localH2tSijHs.memptr(), this->H2tSijHs.memptr(),
        this->k * this->k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    temp = MPITOC;
    this->time_stats.err_communication_duration(temp);

#ifdef MPI_VERBOSE
    DISTPRINTINFO("::it=" << it << PRINTMAT(WtAijH));
    DISTPRINTINFO("::it=" << it << PRINTMAT(H2tSijHs));
#endif

    // Compute the fits
    MPITIC;
    double tWtAijH = arma::trace(this->WtAijH);
    double tWtWHtH = arma::trace(this->WtW * this->prevHtH);
    this->m_Afit = this->m_globalsqnormA - (2 * tWtAijH) + tWtWHtH;
    
    double tH2tSijHs = arma::trace(this->H2tSijHs);
    double tH2tH2HtH = arma::trace(this->H2tH2 * this->prevHtH);
    this->m_Sfit = this->m_globalsqnormS - (2 * tH2tSijHs) + tH2tH2HtH;
    temp = MPITOC;
    this->time_stats.err_compute_duration(temp);

    // Compute the symmetric regularization
    double localdiff_sq = m_localdiff * m_localdiff;
    double globaldiff = 0.0;
    MPITIC;
    MPI_Allreduce(&localdiff_sq, &globaldiff, 1, MPI_DOUBLE, 
        MPI_SUM, MPI_COMM_WORLD);
    temp = MPITOC;
    this->time_stats.err_communication_duration(temp);
    this->m_symmdiff = sqrt(globaldiff);
    double sym_obj = this->beta() * globaldiff;

#ifdef MPI_VERBOSE
    DISTPRINTINFO("::it=" << it << "localdiff::" << m_localdiff
                      << "::localdiff_sq::" << localdiff_sq);
#endif

    // Compute the regularizers
    MAT onematrix = arma::ones<MAT>(this->k, this->k);

    MPITIC;
    // Frobenius norm regularization
    double fro_W_sq = arma::trace(this->WtW);
    double fro_W_obj = this->m_regW(0) * fro_W_sq;
    this->normW = sqrt(fro_W_sq);
    double fro_H_sq = arma::trace(this->prevHtH);
    double fro_H_obj = this->m_regH(0) * fro_H_sq;
    this->normH = sqrt(fro_H_sq);
    this->normH2 = sqrt(arma::trace(this->H2tH2));

    // L_{12} norm regularization
    double l1_W_sq = arma::trace(this->WtW * onematrix);
    double l1_W_obj = this->m_regW(1) * l1_W_sq;
    this->l1normW = sqrt(l1_W_sq);
    double l1_H_sq = arma::trace(this->prevHtH * onematrix);
    double l1_H_obj = this->m_regH(1) * l1_H_sq;
    this->l1normH = sqrt(l1_H_sq);
    
    // Objective being minimized
    this->fit_err_sq = this->m_Afit + (this->alpha() * this->m_Sfit);
    this->objective_err = this->fit_err_sq + fro_W_obj + fro_H_obj 
              + l1_W_obj + l1_H_obj + sym_obj;

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
          << "::H L_12 norm::" << this->l1normH << std::endl
          << "H2 frobenius norm::" << this->normH2 << std::endl
          << "symmdiff::" << this->m_symmdiff 
          << "::relative symmdiff::" << this->m_symmdiff / this->normH);
  }

  void saveOutput(std::string outfname) {
    INFO << "from save output function: " << outfname << std::endl;

    // NOTE: writeOutput function doesn't use DistJointNMFIO m_A, so passing in W should
    // theoretically work for now ----> might want to overload the DistJointNMFIO constructor
    // to only require a communicatior...
    DistJointNMFIO<MAT> test(this->m_mpicomm, this->W);
    test.writeOutput(this->W, this->H, outfname);

    std::stringstream h2;
    h2 << outfname << "_H2";
    test.writeOutput(this->H2, h2.str());

    // this->W.brief_print();
    // this->W.brief_print();
  }

  ~DistANLSBPPJointNMF() {
    /*
       tempHtH.clear();
       tempWtW.clear();
       tempAHtij.clear();
       tempWtAij.clear();
     */
  }
  /// Sets the beta parameter
  void beta(const double b) { this->m_beta = b; }
  // Returns the beta parameter
  double beta() { return this->m_beta; }

  MAT getHhat() { return H2; }
};  // class DistANLSBPP2D

}  // namespace planc

#endif  // DISTNMF_DISTANLSBPP_HPP_
