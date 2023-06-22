/* Copyright 2022 Ramakrishnan Kannan */

#ifndef DISTNMF_DISTJNMF_HPP_
#define DISTNMF_DISTJNMF_HPP_

#include <string>
#include "common/jointnmf.hpp"
#include "distjointnmf/jnmfmpicomm.hpp"
#include "distjointnmf/distjnmftime.hpp"

namespace planc {
template <typename T1, typename T2>
class DistJointNMF : public JointNMF<T1, T2> {
 protected:
  const MPICommunicatorJNMF &m_mpicomm;
  const MPICommunicatorJNMF &m_Scomm;

  UWORD m_ownedm;
  UWORD m_ownedn;
  UWORD m_globalm;
  UWORD m_globaln;

  double m_globalsqnormA;
  double m_globalsqnormS;

  DistJointNMFTime time_stats;
  uint m_compute_error;
  algotype m_algorithm;
  ROWVEC localWnorm;
  ROWVEC Wnorm;

  int m_num_k_blocks;
  int m_perk;

  gridInfo Wgrid, Hgrid, Hsgrid;

  // Variables needed for updates
  // Hs is a copy of H in the second grid
  MAT HtH;    /// H is of size (globaln/p)*k;
  MAT WtW;    /// W is of size (globaln/p)*k;
  MAT HstHs;  /// Hs is of size (globaln/p)*k;
  MAT AHtij;  /// AHtij is of size k*(globalm/p)
  MAT WtAij;  /// WtAij is of size k*(globaln/p)
  MAT SHstij; /// AHtij is of size k*(globaln/p)
  MAT Wt;     /// Wt is of size k*(globalm/p)
  MAT Ht;     /// Ht is of size k*(globaln/p)
  MAT Hs;     /// Hs is of size (globaln/p)*k
  MAT Hst;    /// Hst is of size k*(globaln/p)

  // Temporaries needed for matrix multiply
  // WtA block implementation
  MAT Wt_blk, WtAij_blk, Wit, WitAij;
  // AH block implementation
  MAT Ht_blk, AHtij_blk, Hjt, AijHjt;
  // SH block implementation (runs on second grid)
  MAT Hst_blk, SHstij_blk, Hsjt, SijHsjt;
  // XtX computation
  MAT localXtX;

  // Gatherv and Reducescatter variables
  std::vector<int> gatherWtAcnts;
  std::vector<int> gatherWtAdisp;
  std::vector<int> scatterWtAcnts;

  std::vector<int> gatherAHcnts;
  std::vector<int> gatherAHdisp;
  std::vector<int> scatterAHcnts;

  std::vector<int> gatherSHscnts;
  std::vector<int> gatherSHsdisp;
  std::vector<int> scatterSHscnts;

  // A grid (W, H) --> S grid (Hs) communication variables
  std::vector<procInfo> sendHstoH;
  std::vector<procInfo> recvHstoH;
  std::vector<procInfo> sendHtoHs;
  std::vector<procInfo> recvHtoHs;

  /**
   * Allocate temporary variables needed for most JointNMF algorithms
   */
  void allocateMatrices() {
    // WtA variables
    scatterWtAcnts.resize(NUMROWPROCS);
    fillVector<int>(0, &scatterWtAcnts);
    gatherWtAcnts.resize(NUMCOLPROCS);
    fillVector<int>(0, &gatherWtAcnts);
    gatherWtAdisp.resize(NUMCOLPROCS);
    fillVector<int>(0, &gatherWtAdisp);

    Wt.zeros(this->k, this->W.n_rows);
    WtW.zeros(this->k, this->k);
    WtAij.zeros(this->k, this->H.n_rows);

    Wt_blk.zeros(this->m_perk, this->W.n_rows);
    WtAij_blk.zeros(this->m_perk, this->H.n_rows);
    Wit.zeros(this->m_perk, this->A.n_rows);
    WitAij.zeros(this->m_perk, this->A.n_cols);

    // AH
    scatterAHcnts.resize(NUMCOLPROCS);
    fillVector<int>(0, &scatterAHcnts);
    gatherAHcnts.resize(NUMROWPROCS);
    fillVector<int>(0, &gatherAHcnts);
    gatherAHdisp.resize(NUMROWPROCS);
    fillVector<int>(0, &gatherAHdisp);

    Ht.zeros(this->k, this->H.n_rows);
    HtH.zeros(this->k, this->k);
    AHtij.zeros(this->k, this->W.n_rows);

    Ht_blk.zeros(this->m_perk, this->H.n_rows);
    AHtij_blk.zeros(this->m_perk, this->W.n_rows);
    Hjt.zeros(this->m_perk, this->A.n_cols);
    AijHjt.zeros(this->m_perk, this->A.n_rows);
    
    // SHs
    scatterSHscnts.resize(NUMCOLPROCS_C(m_Scomm));
    fillVector<int>(0, &scatterSHscnts);
    gatherSHscnts.resize(NUMROWPROCS_C(m_Scomm));
    fillVector<int>(0, &gatherSHscnts);
    gatherSHsdisp.resize(NUMROWPROCS_C(m_Scomm));
    fillVector<int>(0, &gatherSHsdisp);

    int H2nrows = itersplit(this->S.n_rows, 
                    NUMCOLPROCS_C(this->m_Scomm),
                    MPI_COL_RANK_C(this->m_Scomm));
    Hst.zeros(this->k, this->Hs.n_rows);
    HstHs.zeros(this->k, this->k);
    SHstij.zeros(this->k, H2nrows);

    Hst_blk.zeros(this->m_perk, this->Hs.n_rows);
    SHstij_blk.zeros(this->m_perk, H2nrows);
    Hsjt.zeros(this->m_perk, this->S.n_cols);
    SijHsjt.zeros(this->m_perk, this->S.n_rows);

    // XtX
    localXtX.zeros(this->k, this->k);
  }

  /**
   * Communication needed for matmuls (WtA, AH, SHs).
   * WtA and AH is computed on the A grid (p_r1 x p_c1).
   * SHs is computed on the S grid (p_r2 x p_c2).s
   */
  void setupMatmulComms() {
    // WtA
    // Allgatherv counts
    gatherWtAcnts[0] = itersplit(this->A.n_rows, 
                                    NUMCOLPROCS, 0) * this->m_perk;
    gatherWtAdisp[0] = 0;
    for (int i = 1; i < NUMCOLPROCS; i++) {
      gatherWtAcnts[i] = itersplit(this->A.n_rows,
                                    NUMCOLPROCS, i) * this->m_perk;
      gatherWtAdisp[i] = gatherWtAdisp[i-1] + gatherWtAcnts[i-1];
    }
    // Reducescatter counts
    for (int i = 0; i < NUMROWPROCS; i++) {
      scatterWtAcnts[i] = itersplit(this->A.n_cols,
                                    NUMROWPROCS, i) * this->m_perk;
    }
    // AH
    // Allgatherv counts
    gatherAHcnts[0] = itersplit(this->A.n_cols, 
                                    NUMROWPROCS, 0) * this->m_perk;
    gatherAHdisp[0] = 0;
    for (int i = 1; i < NUMROWPROCS; i++) {
      gatherAHcnts[i] = itersplit(this->A.n_cols, NUMROWPROCS, i) * this->m_perk;
      gatherAHdisp[i] = gatherAHdisp[i-1] + gatherAHcnts[i-1];
    }
    // Reducescatter counts
    for (int i = 0; i < NUMCOLPROCS; i++) {
      scatterAHcnts[i] = itersplit(this->A.n_rows,
                                    NUMCOLPROCS, i) * this->m_perk;
    }
    // SHs
    // Allgatherv counts
    gatherSHscnts[0] = itersplit(this->S.n_cols, 
                        NUMROWPROCS_C(this->m_Scomm), 0) * this->m_perk;
    gatherSHsdisp[0] = 0;
    for (int i = 1; i < NUMROWPROCS_C(this->m_Scomm); i++) {
      gatherSHscnts[i] = itersplit(this->S.n_cols, 
                        NUMROWPROCS_C(this->m_Scomm), i) * this->m_perk;
      gatherSHsdisp[i] = gatherSHsdisp[i-1] + gatherSHscnts[i-1];
    }
    // Reducescatter counts
    for (int i = 0; i < NUMCOLPROCS_C(this->m_Scomm); i++) {
      scatterSHscnts[i] = itersplit(this->S.n_rows,
                        NUMCOLPROCS_C(this->m_Scomm), i) * this->m_perk;
    }
  }

  /**
   * Communicator needed to synchronise H and Hs
   */
  void setupHsHComms() {
    // Setup grids
    Wgrid.pr    = NUMROWPROCS;
    Wgrid.pc    = NUMCOLPROCS;
    Wgrid.order = 'R';

    Hgrid.pr    = NUMROWPROCS;
    Hgrid.pc    = NUMCOLPROCS;
    Hgrid.order = 'C';

    Hsgrid.pr    = NUMROWPROCS_C(this->m_Scomm);
    Hsgrid.pc    = NUMCOLPROCS_C(this->m_Scomm);
    Hsgrid.order = 'C';

    // Hs (S grid) --> H (A grid)
    sendHstoH = getsendinfo(this->m_globaln, 
                  this->m_Scomm.row_rank(), this->m_Scomm.col_rank(),
                  Hsgrid, Hgrid);
    recvHstoH = getrecvinfo(this->m_globaln,
                  this->m_mpicomm.row_rank(), this->m_mpicomm.col_rank(),
                  Hsgrid, Hgrid);

    // H (A grid) --> Hs (S grid)
    sendHtoHs = getsendinfo(this->m_globaln,
                  this->m_mpicomm.row_rank(), this->m_mpicomm.col_rank(),
                  Hgrid, Hsgrid);
    recvHtoHs = getrecvinfo(this->m_globaln,
                  this->m_Scomm.row_rank(), this->m_Scomm.col_rank(),
                  Hgrid, Hsgrid);
  }

  /**
   * There are p processes.
   * Every process i has W in m_i * k
   * At the end of this call, all process will have
   * WtW of size k*k is symmetric. So not to worry
   * about column/row major formats.
   * @param[in] X is of size m_i x k
   * @param[out] XtX Every process owns the same kxk global gram matrix of X
   */
  void distInnerProduct(const MAT &X, MAT *XtX) {
    // each process computes its own kxk matrix
    MPITIC;  // gram
    localXtX = X.t() * X;
    double temp = MPITOC;  // gram
#ifdef MPI_VERBOSE
    DISTPRINTINFO("X::" << arma::norm(X, "fro")
                      << "::localXtX::" << arma::norm(localXtX, "fro"));
#endif
    this->time_stats.compute_duration(temp);
    this->time_stats.gram_duration(temp);
    (*XtX).zeros();
    this->reportTime(temp, "Gram::X::");
    MPITIC;  // allreduce gram
    MPI_Allreduce(localXtX.memptr(), (*XtX).memptr(), this->k * this->k,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    temp = MPITOC;  // allreduce gram
    this->time_stats.communication_duration(temp);
    this->time_stats.allreduce_duration(temp);
  }

  /**
   * Computes XtY for 1-D distributed matrices X and Y
   * @param[in] reference to local matrix X of 
   *            size \f$\frac{globaln}{p} \times k \f$
   * @param[in] reference to local matrix Y of 
   *            size \f$\frac{globaln}{p} \times k \f$
   * @param[in] reference to output matrix XtY of 
   *            size \f$k \times k \f$
   */
  void distDotProduct(const MAT &X, const MAT &Y, MAT *XtY) {
    // compute local matrix
    MPITIC;  // gram
    MAT localXtY = X.t() * Y;
    double temp = MPITOC;  // gram
#ifdef MPI_VERBOSE
    DISTPRINTINFO("localXtY::" << arma::norm(localXtY, "fro"));
#endif
    this->time_stats.compute_duration(temp);
    this->time_stats.nongram_duration(temp);
    (*XtY).zeros();
    this->reportTime(temp, "Dot::XY::");
    MPITIC;  // allreduce gram

    // This is because of column major ordering of Armadillo.
    MPI_Allreduce(localXtY.memptr(), (*XtY).memptr(), this->k * this->k,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    temp = MPITOC;  // allreduce gram
    this->time_stats.communication_duration(temp);
    this->time_stats.allreduce_duration(temp);
  }

  /**
   * This is matrix-multiplication routine based on
   * the 2D algorithm on a p = p_r1 x p_c1 grid 
   * m_mpicomm is the grid communicator
   * A is of size (m / p_r1) x (n / p_c1)
   * W is of size (m / p) x k
   * this->m_mpicomm.comm_subs()[0] is column communicator.
   * this->m_mpicomm.comm_subs()[1] is row communicator.
   */
  void distWtA() {
    for (int i = 0; i < m_num_k_blocks; i++) {
      int start_row = i * m_perk;
      int end_row = (i + 1) * m_perk - 1;
      Wt_blk = Wt.rows(start_row, end_row);
      distWtABlock();
      WtAij.rows(start_row, end_row) = WtAij_blk;
    }
  }

  void distWtABlock() {
    int sendcnt = (this->W.n_rows) * this->m_perk;
    Wit.zeros();
    MPITIC;  // allgather WtA
    MPI_Allgatherv(Wt_blk.memptr(), sendcnt, MPI_DOUBLE, Wit.memptr(),
                  &(gatherWtAcnts[0]), &(gatherWtAdisp[0]), MPI_DOUBLE,
                  this->m_mpicomm.commSubs()[1]);
    double temp = MPITOC;  // allgather WtA
#ifdef MPI_VERBOSE
    DISTPRINTINFO(PRINTMAT(Wt_blk));
    DISTPRINTINFO(PRINTMAT(Wit));
#endif
    this->time_stats.communication_duration(temp);
    this->time_stats.allgather_duration(temp);
    this->time_stats.WtA_communication_duration(temp);
    MPITIC;  // mm WtA
    this->WitAij = this->Wit * this->A;
    temp = MPITOC;  // mm WtA
#ifdef MPI_VERBOSE
    DISTPRINTINFO(PRINTMAT(this->WitAij));
#endif
    this->time_stats.compute_duration(temp);
    this->time_stats.mm_duration(temp);
    this->time_stats.WtA_compute_duration(temp);
    this->reportTime(temp, "WtA::");
    WtAij_blk.zeros();
    MPITIC;  // reduce_scatter WtA
    MPI_Reduce_scatter(this->WitAij.memptr(), this->WtAij_blk.memptr(),
                       &(scatterWtAcnts[0]), MPI_DOUBLE, MPI_SUM,
                       this->m_mpicomm.commSubs()[0]);
    temp = MPITOC;  // reduce_scatter WtA
    this->time_stats.communication_duration(temp);
    this->time_stats.reducescatter_duration(temp);
    this->time_stats.WtA_communication_duration(temp);
  }

  /**
   * This is matrix-multiplication routine based on
   * the 2D algorithm on a p = p_r1 x p_c1 grid 
   * m_mpicomm is the grid communicator
   * A is of size (m / p_r1) x (n / p_c1)
   * H is of size (n / p) x k
   * this->m_mpicomm.comm_subs()[0] is column communicator.
   * this->m_mpicomm.comm_subs()[1] is row communicator.
   */
  void distAH() {
    for (int i = 0; i < m_num_k_blocks; i++) {
      int start_row = i * m_perk;
      int end_row = (i + 1) * m_perk - 1;
      Ht_blk = Ht.rows(start_row, end_row);
      distAHBlock();
      AHtij.rows(start_row, end_row) = AHtij_blk;
    }
  }

  void distAHBlock() {
    int sendcnt = (this->H.n_rows) * this->m_perk;
    Hjt.zeros();
    MPITIC;  // allgather AH
    MPI_Allgatherv(this->Ht_blk.memptr(), sendcnt, MPI_DOUBLE,
                  this->Hjt.memptr(), &(gatherAHcnts[0]), &(gatherAHdisp[0]),
                  MPI_DOUBLE, this->m_mpicomm.commSubs()[0]);
    double temp = MPITOC;  // allgather AH
#ifdef MPI_VERBOSE
    DISTPRINTINFO(PRINTMAT(Ht_blk));
    DISTPRINTINFO(PRINTMAT(Hjt));
#endif
    this->time_stats.communication_duration(temp);
    this->time_stats.allgather_duration(temp);
    this->time_stats.AH_communication_duration(temp);
    MPITIC;  // mm AH
    this->AijHjt = this->Hjt * this->A.t();
    temp = MPITOC;  // mm AH
#ifdef MPI_VERBOSE
    DISTPRINTINFO(PRINTMAT(this->AijHjt));
#endif
    this->time_stats.compute_duration(temp);
    this->time_stats.mm_duration(temp);
    this->time_stats.AH_compute_duration(temp);
    this->reportTime(temp, "AH::");
    AHtij_blk.zeros();
    MPITIC;  // reduce_scatter AH
    MPI_Reduce_scatter(this->AijHjt.memptr(), this->AHtij_blk.memptr(),
                       &(this->scatterAHcnts[0]), MPI_DOUBLE, MPI_SUM,
                       this->m_mpicomm.commSubs()[1]);
    temp = MPITOC;  // reduce_scatter AH
    this->time_stats.communication_duration(temp);
    this->time_stats.reducescatter_duration(temp);
    this->time_stats.AH_communication_duration(temp);
  }

  /**
   * Sending matrix A (in A grid) with sending information to 
   * matrix B (in B grid) with receiving information.
   * Note matrices sent/received are k x n or k x m in size and contiguous
   * columns are sent out. (Use Ht, Wt, Hst in the communication)
   * @param[in] A is a reference to the matrix to be sent
   * @param[in] Asendinfo is a vector of procInfo structs with send counts
   * @param[in] Acomm is A grid's communicator
   * @param[in] B is a reference to the matrix to be received
   * @param[in] Brecvinfo is a vector of procInfo structs with receive counts
   * @param[in] Bcomm is B grid's communicator
   */
  void sendAtoB(MAT &A, std::vector<procInfo> Asendinfo, 
                  const MPI_Comm Acomm, MAT &B,
                  std::vector<procInfo> Brecvinfo, const MPI_Comm Bcomm) {
    MPI_Request sendreq[Asendinfo.size()];
    MPI_Status sendstat[Asendinfo.size()];
    MPI_Status recvstat[Brecvinfo.size()];
    
    // Send all the rows (non-blocking)
    for (int ii = 0; ii < Asendinfo.size(); ii++) {
      // MPI_Isend(const void *buf, int count, MPI_Datatype datatype,
      //    int dest, int tag, MPI_Comm comm, MPI_Request *request);
      std::vector<int> destcoords = {Asendinfo[ii].i, Asendinfo[ii].j};
      int destproc = mpisub2ind(Bcomm, destcoords);

      MPI_Isend(A.colptr(Asendinfo[ii].start_idx), Asendinfo[ii].nrows * this->k, 
        MPI_DOUBLE, destproc, 0, MPI_COMM_WORLD, &sendreq[ii]);
    }

    for (int jj = 0; jj < Brecvinfo.size(); jj++) {
      // int MPI_Recv(void *buf, int count, MPI_Datatype datatype,
      //  int source, int tag, MPI_Comm comm, MPI_Status *status)
      std::vector<int> srccoords = {Brecvinfo[jj].i, Brecvinfo[jj].j};
      int srcproc = mpisub2ind(Acomm, srccoords);

      MPI_Recv(B.colptr(Brecvinfo[jj].start_idx), Brecvinfo[jj].nrows * this->k,
        MPI_DOUBLE, srcproc, 0, MPI_COMM_WORLD, &recvstat[jj]);
#ifdef MPI_VERBOSE
      DISTPRINTINFO("Recv status::source::" << recvstat[jj].MPI_SOURCE 
          << "::tag::" << recvstat[jj].MPI_TAG
          << "::error::" << recvstat[jj].MPI_ERROR);
#endif
    }

    // Wait for the Isends
    for (int ii=0; ii < Asendinfo.size(); ii++) {
      MPI_Wait(&sendreq[ii], &sendstat[ii]);
#ifdef MPI_VERBOSE
      DISTPRINTINFO("Send status::source::" << sendstat[ii].MPI_SOURCE 
          << "::tag::" << sendstat[ii].MPI_TAG
          << "::error::" << sendstat[ii].MPI_ERROR);
#endif
    }
  }

  /**
   * This is matrix-multiplication routine based on
   * the 2D algorithm on a p = p_r2 x p_c2 grid 
   * m_mpicomm is the grid communicator
   * A is of size (m / p_r2) x (n / p_c2)
   * H is of size (n / p) x k
   * this->m_Scomm.comm_subs()[0] is column communicator.
   * this->m_Scomm.comm_subs()[1] is row communicator.
   */
  void distSHs() {
    for (int i = 0; i < m_num_k_blocks; i++) {
      int start_row = i * m_perk;
      int end_row = (i + 1) * m_perk - 1;
      Hst_blk = Hst.rows(start_row, end_row);
      distSHsBlock();
      SHstij.rows(start_row, end_row) = SHstij_blk;
    }
  }

  void distSHsBlock() {
    int sendcnt = (this->Hs.n_rows) * this->m_perk;
    Hsjt.zeros();
    MPITIC;  // allgather SHs
    MPI_Allgatherv(this->Hst_blk.memptr(), sendcnt, MPI_DOUBLE,
                  this->Hsjt.memptr(), &(gatherSHscnts[0]),
                  &(gatherSHsdisp[0]), MPI_DOUBLE,
                  this->m_Scomm.commSubs()[0]);
    double temp = MPITOC;  // allgather SHs
#ifdef MPI_VERBOSE
    DISTPRINTINFO(PRINTMAT(Hst_blk));
    DISTPRINTINFO(PRINTMAT(Hsjt));
#endif
    this->time_stats.communication_duration(temp);
    this->time_stats.allgather_duration(temp);
    this->time_stats.SHs_communication_duration(temp);
    MPITIC;  // mm SHs
    this->SijHsjt = this->Hsjt * this->S.t();
    temp = MPITOC;  // mm SHs
#ifdef MPI_VERBOSE
    DISTPRINTINFO(PRINTMAT(this->SijHsjt));
#endif
    this->time_stats.compute_duration(temp);
    this->time_stats.mm_duration(temp);
    this->time_stats.SHs_compute_duration(temp);
    this->reportTime(temp, "SHs::");
    SHstij_blk.zeros();
    MPITIC;  // reduce_scatter SHs
    MPI_Reduce_scatter(this->SijHsjt.memptr(), this->SHstij_blk.memptr(),
                       &(this->scatterSHscnts[0]), MPI_DOUBLE, MPI_SUM,
                       this->m_Scomm.commSubs()[1]);
    temp = MPITOC;  // reduce_scatter SHs
    this->time_stats.communication_duration(temp);
    this->time_stats.reducescatter_duration(temp);
    this->time_stats.SHs_communication_duration(temp);
  }

 public:
  /**
   * There are totally p processes arranged in a single
   * logical \f$p_r \times p_c\f$ grid. Each process will hold the following
   * @param[in] input Features matrix of 
   *          size \f$\frac{globalm}{p_r} \times \frac{globaln}{p_c}\f$
   * @param[in] conn Connection matrix of 
   *          size \f$\frac{globaln}{p_r} \times \frac{globaln}{p_c}\f$
   * @param[in] leftlowrankfactor W of
   *          size \f$\frac{globalm}{p} \times k\f$
   * @param[in] rightlowrankfactor H of 
   *          size \f$\frac{globaln}{p} \times k\f$
   * @param[in] communicator Single communicator for both matrices
   * @param[in] numblks Block size on k for distributed matrix-multiply 
   */  
  DistJointNMF(const T1 &input, const T2 &conn, 
          const MAT &leftlowrankfactor, const MAT &rightlowrankfactor, 
          const MPICommunicatorJNMF &communicator,
          const int numblks)
      : JointNMF<T1, T2>(input, conn, leftlowrankfactor, rightlowrankfactor),
        m_mpicomm(communicator), m_Scomm(communicator) {
    this->m_num_k_blocks = numblks;
    this->m_perk = this->k / this->m_num_k_blocks;
    // Get the global norms
    double sqnormA = this->normA * this->normA;
    MPI_Allreduce(&sqnormA, &(this->m_globalsqnormA), 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    double sqnormS = this->normS * this->normS;
    MPI_Allreduce(&sqnormS, &(this->m_globalsqnormS), 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    // Get the global row and column counts
    this->m_globalm = 0;
    this->m_globaln = 0;
    this->m_ownedm = this->W.n_rows;
    this->m_ownedn = this->H.n_rows;

    // Use m_mpicomm as the only global communicator
    MPI_Allreduce(&(this->m), &(this->m_globalm), 1, MPI_INT, MPI_SUM,
                  this->m_mpicomm.commSubs()[0]);
    MPI_Allreduce(&(this->n), &(this->m_globaln), 1, MPI_INT, MPI_SUM,
                  this->m_mpicomm.commSubs()[1]);
    if (ISROOT) {
      INFO << "globalsqnormA::" << this->m_globalsqnormA
           << "::globalsqnormS::" << this->m_globalsqnormS
           << "::globalm::" << this->m_globalm
           << "::globaln::" << this->m_globaln << std::endl;
    }
    this->m_compute_error = 0;
    localWnorm.zeros(this->k);
    Wnorm.zeros(this->k);

    // Set default value for alpha
    this->m_alpha = this->m_globalsqnormA / this->m_globalsqnormS;

    // Initialise Hs (need to copy from A grid)
    this->Hs = arma::zeros<MAT>(itersplit(this->S.n_cols, 
                  NUMROWPROCS_C(this->m_Scomm), MPI_ROW_RANK_C(this->m_Scomm)),
                  this->k);

    // Allocate temporary matrices
    allocateMatrices();

    // Setup communication stuff
    setupMatmulComms();
    setupHsHComms();

#ifdef MPI_VERBOSE
    function<void()> f_Hs2H = [this] () {
      std::cout << "Hs2H print." << std::endl;
      
      std::cout << "sendHstoH" << std::endl;
      std::cout << this->m_Scomm.row_rank() << "x" 
                << this->m_Scomm.col_rank()
                << " (S grid) sending" << std::endl;
      for (auto i = this->sendHstoH.begin(); 
              i != this->sendHstoH.end(); ++i) {
        std::cout << (*i).i << "x" << (*i).j << "::startidx::"
            << (*i).start_idx << "::nrows::" << (*i).nrows
            << std::endl;
      }

      std::cout << "recvHstoH" << std::endl;
      std::cout << this->m_mpicomm.row_rank() << "x" 
                << this->m_mpicomm.col_rank()
                << " (A grid) receiving" << std::endl;
      for (auto i = this->recvHstoH.begin(); 
              i != this->recvHstoH.end(); ++i) {
        std::cout << (*i).i << "x" << (*i).j << "::startidx::"
            << (*i).start_idx << "::nrows::" << (*i).nrows
            << std::endl;
      }

      std::cout << "sendHtoHs" << std::endl;
      std::cout << this->m_mpicomm.row_rank() << "x" 
                << this->m_mpicomm.col_rank()
                << " (A grid) sending" << std::endl;
      for (auto i = this->sendHtoHs.begin(); 
              i != this->sendHtoHs.end(); ++i) {
        std::cout << (*i).i << "x" << (*i).j << "::startidx::"
            << (*i).start_idx << "::nrows::" << (*i).nrows
            << std::endl;
      }

      std::cout << "recvHtoHs" << std::endl;
      std::cout << this->m_Scomm.row_rank() << "x" 
                << this->m_Scomm.col_rank()
                << " (S grid) receiving" << std::endl;
      for (auto i = this->recvHtoHs.begin(); 
              i != this->recvHtoHs.end(); ++i) {
        std::cout << (*i).i << "x" << (*i).j << "::startidx::"
            << (*i).start_idx << "::nrows::" << (*i).nrows
            << std::endl;
      }
    };
    mpi_serial_print(f_Hs2H);
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // Initialise factors
    this->Wt = this->W.t();
    this->Ht = this->H.t();

    // Transfer H (A grid) to Hs (S grid)
    sendAtoB(this->Ht, this->sendHtoHs, this->m_mpicomm.gridComm(),
        this->Hst, this->recvHtoHs, this->m_Scomm.gridComm());
    this->Hs = this->Hst.t();

    MPI_Barrier(MPI_COMM_WORLD);
  }
  /**
   * There are totally p processes arranged in to two logical grids.
   * The logical \f$p_r1 \times p_c1\f$ grid is for the features matrix
   * and a logical \f$p_r2 \times p_c2\f$ for the connections matrix.
   * Each process will hold the following
   * @param[in] input Features matrix of 
   *          size \f$\frac{globalm}{p_r1} \times \frac{globaln}{p_c1}\f$
   * @param[in] conn Connection matrix of 
   *          size \f$\frac{globaln}{p_r2} \times \frac{globaln}{p_c2}\f$
   * @param[in] leftlowrankfactor W of
   *          size \f$\frac{globalm}{p} \times k\f$
   * @param[in] rightlowrankfactor H of 
   *          size \f$\frac{globaln}{p} \times k\f$
   * @param[in] Acomm Single communicator for the features matrix
   * @param[in] Scomm Single communicator for the connection matrix
   * @param[in] numblks Block size on k for distributed matrix-multiply 
   */  
  DistJointNMF(const T1 &input, const T2 &conn, 
          const MAT &leftlowrankfactor, const MAT &rightlowrankfactor, 
          const MPICommunicatorJNMF &Acomm, const MPICommunicatorJNMF &Scomm,
          const int numblks)
      : JointNMF<T1, T2>(input, conn, leftlowrankfactor, rightlowrankfactor),
        m_mpicomm(Acomm), m_Scomm(Scomm) {
    this->m_num_k_blocks = numblks;
    this->m_perk = this->k / this->m_num_k_blocks;
    // Get the global norms
    double sqnormA = this->normA * this->normA;
    MPI_Allreduce(&sqnormA, &(this->m_globalsqnormA), 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    double sqnormS = this->normS * this->normS;
    MPI_Allreduce(&sqnormS, &(this->m_globalsqnormS), 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    // Get the global row and column counts
    this->m_globalm = 0;
    this->m_globaln = 0;
    this->m_ownedm = this->W.n_rows;
    this->m_ownedn = this->H.n_rows;

    // Use m_mpicomm as the communicator for W, H. It should be in sync with
    // the m_Scomm view of W, H.
    MPI_Allreduce(&(this->m), &(this->m_globalm), 1, MPI_INT, MPI_SUM,
                  this->m_mpicomm.commSubs()[0]);
    MPI_Allreduce(&(this->n), &(this->m_globaln), 1, MPI_INT, MPI_SUM,
                  this->m_mpicomm.commSubs()[1]);
    if (ISROOT) {
      INFO << "globalsqnormA::" << this->m_globalsqnormA
           << "::globalsqnormS::" << this->m_globalsqnormS
           << "::globalm::" << this->m_globalm
           << "::globaln::" << this->m_globaln << std::endl;
    }
    this->m_compute_error = 0;
    localWnorm.zeros(this->k);
    Wnorm.zeros(this->k);

    // Set default value for alpha
    this->m_alpha = this->m_globalsqnormA / this->m_globalsqnormS;

    // Initialise Hs (need to copy from A grid)
    this->Hs = arma::zeros<MAT>(itersplit(this->S.n_cols,
                  NUMROWPROCS_C(this->m_Scomm), MPI_ROW_RANK_C(this->m_Scomm)),
                  this->k);

    // Allocate temporary matrices
    allocateMatrices();

    // Setup communication stuff
    setupMatmulComms();
    setupHsHComms();

#ifdef MPI_VERBOSE
    function<void()> f_Hs2H = [this] () {
      std::cout << "Hs2H print." << std::endl;

      std::cout << "sendHstoH" << std::endl;
      std::cout << this->m_Scomm.row_rank() << "x" 
                << this->m_Scomm.col_rank()
                << " (S grid) sending" << std::endl;
      for (auto i = this->sendHstoH.begin(); 
              i != this->sendHstoH.end(); ++i) {
        std::cout << (*i).i << "x" << (*i).j << "::startidx::"
            << (*i).start_idx << "::nrows::" << (*i).nrows
            << std::endl;
      }

      std::cout << "recvHstoH" << std::endl;
      std::cout << this->m_mpicomm.row_rank() << "x" 
                << this->m_mpicomm.col_rank()
                << " (A grid) receiving" << std::endl;
      for (auto i = this->recvHstoH.begin(); 
              i != this->recvHstoH.end(); ++i) {
        std::cout << (*i).i << "x" << (*i).j << "::startidx::"
            << (*i).start_idx << "::nrows::" << (*i).nrows
            << std::endl;
      }

      std::cout << "sendHtoHs" << std::endl;
      std::cout << this->m_mpicomm.row_rank() << "x" 
                << this->m_mpicomm.col_rank()
                << " (A grid) sending" << std::endl;
      for (auto i = this->sendHtoHs.begin(); 
              i != this->sendHtoHs.end(); ++i) {
        std::cout << (*i).i << "x" << (*i).j << "::startidx::"
            << (*i).start_idx << "::nrows::" << (*i).nrows
            << std::endl;
      }

      std::cout << "recvHtoHs" << std::endl;
      std::cout << this->m_Scomm.row_rank() << "x" 
                << this->m_Scomm.col_rank()
                << " (S grid) receiving" << std::endl;
      for (auto i = this->recvHtoHs.begin(); 
              i != this->recvHtoHs.end(); ++i) {
        std::cout << (*i).i << "x" << (*i).j << "::startidx::"
            << (*i).start_idx << "::nrows::" << (*i).nrows
            << std::endl;
      }
    };
    mpi_serial_print(f_Hs2H);
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // Initialise factors
    this->Wt = this->W.t();
    this->Ht = this->H.t();

    // Transfer H (A grid) to Hs (S grid)
    sendAtoB(this->Ht, this->sendHtoHs, this->m_mpicomm.gridComm(),
        this->Hst, this->recvHtoHs, this->m_Scomm.gridComm());
    this->Hs = this->Hst.t();

    PRINTROOT("Acomm::pr::" << Acomm.pr() << "::pc::" << Acomm.pc());
    PRINTROOT("Scomm::pr::" << Scomm.pr() << "::pc::" << Scomm.pc());
    MPI_Barrier(MPI_COMM_WORLD);
  }

  /// returns globalm
  const int globalm() const { return m_globalm; }
  /// returns globaln
  const int globaln() const { return m_globaln; }
  /// returns global squared norm of A
  const double globalsqnorma() const { return m_globalsqnormA; }
  /// return the current error
  void compute_error(const uint &ce) { this->m_compute_error = ce; }
  /// returns the flag to compute error or not.
  const bool is_compute_error() const { return (this->m_compute_error); }
  /// returns the NMF algorithm
  void algorithm(algotype dat) { this->m_algorithm = dat; }
  /// Reports the time
  void reportTime(const double temp, const std::string &reportstring) {
    double mintemp, maxtemp, sumtemp;
    MPI_Allreduce(&temp, &maxtemp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&temp, &mintemp, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&temp, &sumtemp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    PRINTROOT(reportstring << "::m::" << this->m_globalm
                           << "::n::" << this->m_globaln << "::k::" << this->k
                           << "::SIZE::" << MPI_SIZE
                           << "::algo::" << this->m_algorithm
                           << "::root::" << temp << "::min::" << mintemp
                           << "::avg::" << (sumtemp) / (MPI_SIZE)
                           << "::max::" << maxtemp);
  }
  /// Column Normalizes the distributed W matrix
  void normalize_by_W() {
    localWnorm = sum(this->W % this->W);
    mpitic();
    MPI_Allreduce(localWnorm.memptr(), Wnorm.memptr(), this->k, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
    double temp = mpitoc();
    this->time_stats.allgather_duration(temp);
    for (int i = 0; i < this->k; i++) {
      if (Wnorm(i) > 1) {
        double norm_const = sqrt(Wnorm(i));
        this->W.col(i) = this->W.col(i) / norm_const;
        this->H.col(i) = norm_const * this->H.col(i);
      }
    }
  }

  /*
   * Hyperparameter empty getters and setters
   */
  /// Sets the momentum parameter
  void beta(const double b) { return; }
  // Returns the momentum parameter
  double beta() { return 0.0; }
  /// Sets the momentum parameter
  void gamma(const double b) { return; }
  // Returns the momentum parameter
  double gamma() { return 0.0; }
  /// Sets the inner iterations parameters (if needed)
  void set_luciters(const int max_luciters) {}
};

}  // namespace planc

#endif  // DISTNMF_DISTJNMF_HPP_
