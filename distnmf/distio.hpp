/* Copyright 2016 Ramakrishnan Kannan */

#ifndef DISTNMF_DISTIO_HPP_
#define DISTNMF_DISTIO_HPP_

#include <unistd.h>

#include <armadillo>
#include <string>

#include "common/distutils.hpp"
#include "distnmf/mpicomm.hpp"

/**
 * File name formats
 * A is the filename
 * 1D distribution Arows_totalpartitions_rank or Acols_totalpartitions_rank
 * Double 1D distribution (both row and col distributed)
 * Arows_totalpartitions_rank and Acols_totalpartitions_rank
 * TWOD distribution A_totalpartition_rank
 * Just send the first parameter Arows and the second parameter Acols to be
 * zero.
 */

namespace planc {

template <class MATTYPE>
class DistIO {
 private:
  const MPICommunicator& m_mpicomm;
  MATTYPE m_Arows;  /// ONED_ROW and ONED_DOUBLE
  MATTYPE m_Acols;  /// ONED_COL and ONED_DOUBLE
  MATTYPE& m_A;     /// TWOD
  // don't start getting prime number from 2;
  static const int kPrimeOffset = 10;
  // Hope no one hits on this number.
  static const int kW_seed_idx = 1210873;
  static const int k_row_seed = 160669;
  static const int k_col_seed = 162287;
  static const int k_seed_off = 617237;
#ifdef BUILD_SPARSE
  static const int kalpha = 5;
  static const int kbeta = 10;
#else
  static const int kalpha = 1;
  static const int kbeta = 0;
#endif
  IVEC rcounts;  // vector to hold row counts for uneven splits
  IVEC ccounts;  // vector to hold column counts for uneven splits

  const iodistributions m_distio;
  /**
   * A random matrix is always needed for sparse case
   * to get the pattern. That is., the indices where
   * numbers are being filled. We will use this pattern
   * and change the number later.
   */
  void randMatrix(const std::string type, const int primeseedidx,
                  double sparsity, const int symm,
                  const bool adj_rand, MATTYPE* X) {
    int row_rank = m_mpicomm.row_rank();
    int col_rank = m_mpicomm.col_rank();

    int n_rows = (*X).n_rows;
    int n_cols = (*X).n_cols;

    // Set seed
    if (symm) {  // Special case: symmetric matrix
       int symseed = 0;
       if (row_rank < col_rank) {
        symseed = (k_row_seed * row_rank) + (k_col_seed * col_rank)
                  + k_seed_off;
       } else {
        symseed = (k_row_seed * col_rank) + (k_col_seed * row_rank)
                  + k_seed_off;
       }
       arma::arma_rng::set_seed(symseed);

      // Diagonal case
       if (row_rank == col_rank) {
         sparsity = sparsity * 0.5;
       }
    } else if (primeseedidx == -1) {
      arma::arma_rng::set_seed_random();
    } else {
      arma::arma_rng::set_seed(random_sieve(primeseedidx));
    }

#ifdef DEBUG_VERBOSE
    DISTPRINTINFO("randMatrix::" << primeseedidx << "::sp=" << sparsity);
#endif
#ifdef BUILD_SPARSE
    if (type == "uniform" || type == "lowrank") {
      if (symm) {
        if (row_rank == col_rank) {
          // Diagonal blocks
          (*X) = arma::sprandu(n_rows, n_cols, sparsity);
          (*X) = 0.5 * ((*X)  + (*X).t());
        } else if (row_rank < col_rank) {
        // Lower triangular blocks
          (*X) = arma::sprandu(n_cols, n_rows, sparsity).t();
        } else {
          (*X) = arma::sprandu(n_rows, n_cols, sparsity);
        }
      } else {
        (*X).sprandu((*X).n_rows, (*X).n_cols, sparsity);
      }
    } else if (type == "normal") {
      if (symm) {
        if (row_rank == col_rank) {
          // Diagonal blocks
          (*X) = arma::sprandn(n_rows, n_cols, sparsity);
          (*X) = 0.5 * ((*X)  + (*X).t());
        } else if (row_rank < col_rank) {
        // Lower triangular blocks
          (*X) = arma::sprandn(n_cols, n_rows, sparsity).t();
        } else {
          (*X) = arma::sprandn(n_rows, n_cols, sparsity);
        }
      } else {
        (*X).sprandn((*X).n_rows, (*X).n_cols, sparsity);
      }
    }
    SP_MAT::iterator start_it = (*X).begin();
    SP_MAT::iterator end_it = (*X).end();
    if (adj_rand) {
      for (SP_MAT::iterator it = start_it; it != end_it; ++it) {
        double currentValue = (*it);
        (*it) = ceil(kalpha * currentValue + kbeta);
        if ((*it) < 0) (*it) = kbeta;
      }
    }
    // for (uint i=0; i<(*X).n_nonzero;i++){
    //     double currentValue = (*X).values[i];
    //     currentValue *= kalpha;
    //     currentValue += kbeta;
    //     currentValue = ceil(currentValue);
    //     if (currentValue <= 0) currentValue=kbeta;
    //     (*X).values[i]=currentValue;
    // }
#else
    if (type == "uniform") {
      if (symm) {
        if (row_rank == col_rank) {
          // Diagonal blocks
          (*X) = arma::randu(n_rows, n_cols);
          (*X) = 0.5 * ((*X)  + (*X).t());
        } else if (row_rank < col_rank) {
        // Lower triangular blocks
          (*X) = arma::randu(n_cols, n_rows).t();
        } else {
          (*X) = arma::randu(n_rows, n_cols);
        }
      } else {
        (*X).randu();
      }
    } else if (type == "normal") {
      if (symm) {
        if (row_rank == col_rank) {
          // Diagonal blocks
          (*X) = arma::randn(n_rows, n_cols);
          (*X) = 0.5 * ((*X)  + (*X).t());
        } else if (row_rank < col_rank) {
        // Lower triangular blocks
          (*X) = arma::randn(n_cols, n_rows).t();
        } else {
          (*X) = arma::randn(n_rows, n_cols);
        }
      } else {
        (*X).randn();
      }
    }
    if (adj_rand) {
      (*X) = kalpha * (*X) + kbeta;
      (*X) = ceil(*X);
    }
    (*X).elem(find((*X) < 0)).zeros();
#endif
  }

#ifndef BUILD_SPARSE
  /* normalization */
  void normalize(normtype i_normtype) {
    ROWVEC globalnormA = arma::zeros<ROWVEC>(m_A.n_cols);
    ROWVEC normc = arma::zeros<ROWVEC>(m_A.n_cols);
    MATTYPE normmat = arma::zeros<MATTYPE>(m_A.n_rows, m_A.n_cols);
    switch (m_distio) {
      case ONED_ROW:
        if (i_normtype == L2NORM) {
          normc = arma::sum(arma::square(m_A));
          MPI_Allreduce(normc.memptr(), globalnormA.memptr(), m_A.n_cols,
                        MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        } else if (i_normtype == MAXNORM) {
          normc = arma::max(m_A);
          MPI_Allreduce(normc.memptr(), globalnormA.memptr(), m_A.n_cols,
                        MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        }

        break;
      case ONED_COL:
      case ONED_DOUBLE:
        if (i_normtype == L2NORM) {
          globalnormA = arma::sum(arma::square(m_Arows));
        } else if (i_normtype == MAXNORM) {
          globalnormA = arma::max(arma::square(m_Arows));
        }
        break;
      case TWOD:
        if (i_normtype == L2NORM) {
          normc = arma::sum(arma::square(m_A));
          MPI_Allreduce(normc.memptr(), globalnormA.memptr(), m_A.n_cols,
                        MPI_DOUBLE, MPI_SUM, this->m_mpicomm.commSubs()[1]);
        } else if (i_normtype == MAXNORM) {
          normc = arma::max(m_A);
          MPI_Allreduce(normc.memptr(), globalnormA.memptr(), m_A.n_cols,
                        MPI_DOUBLE, MPI_SUM, this->m_mpicomm.commSubs()[1]);
        }
        break;
      default:
        INFO << "cannot normalize" << std::endl;
    }
    normmat = arma::repmat(globalnormA, m_A.n_rows, 1);
    m_A /= normmat;
  }
#endif

  /**
   * Uses the pattern from the input matrix X but
   * the value is computed as low rank.
   */
  void randomLowRank(const UWORD m, const UWORD n, const UWORD k,
                     const int symm, const bool adj_rand, MATTYPE* X) {
    uint start_row = 0, start_col = 0;
    uint end_row = 0, end_col = 0;

    int pr = m_mpicomm.pr();
    int pc = m_mpicomm.pc();

    int row_rank = m_mpicomm.row_rank();
    int col_rank = m_mpicomm.col_rank();

    switch (m_distio) {
      case ONED_ROW:
        start_row = startidx(m, pr, row_rank);
        end_row = start_row + (*X).n_rows - 1;
        start_col = 0;
        end_col = (*X).n_cols - 1;
        break;
      case ONED_COL:
        start_row = 0;
        end_row = (*X).n_rows - 1;
        start_col = startidx(n, pc, col_rank);
        end_col = start_col + (*X).n_cols - 1;
        break;
      // in the case of ONED_DOUBLE we are stitching
      // m/p x n/p matrices and in TWOD we are
      // stitching m/pr * n/pc matrices.
      case ONED_DOUBLE:
        if ((*X).n_cols == n) {  //  m_Arows
          start_row = startidx(m, pr, row_rank);
          end_row = start_row + (*X).n_rows - 1;
          start_col = 0;
          end_col = n - 1;
        }
        if ((*X).n_rows == m) {  // m_Acols
          start_row = 0;
          end_row = m - 1;
          start_col = startidx(n, pc, col_rank);
          end_col = start_col + (*X).n_cols - 1;
        }
        break;
      case TWOD:
        start_row = startidx(m, pr, row_rank);
        end_row = start_row + (*X).n_rows - 1;
        start_col = startidx(n, pc, col_rank);
        end_col = start_col + (*X).n_cols - 1;
        break;
    }
    // all machines will generate same Wrnd and Hrnd
    // at all times.
    arma::arma_rng::set_seed(kW_seed_idx);
    MAT Wrnd(m, k);
    Wrnd.randu();
    MAT Hrnd(k, n);

    if (symm) {
      Hrnd = Wrnd.t();
    } else {
      Hrnd.randu();
    }
#ifdef BUILD_SPARSE
    SP_MAT::iterator start_it = (*X).begin();
    SP_MAT::iterator end_it = (*X).end();
    double tempVal = 0.0;
    for (SP_MAT::iterator it = start_it; it != end_it; ++it) {
      VEC Wrndi = vectorise(Wrnd.row(start_row + it.row()));
      VEC Hrndj = Hrnd.col(start_col + it.col());
      tempVal = dot(Wrndi, Hrndj);
      if (adj_rand) {
        (*it) = ceil(kalpha * tempVal + kbeta);
      } else {
        (*it) = tempVal;
      }
    }
#else
    // MAT templr;
    if ((*X).n_cols == n) {  // ONED_ROW
      MAT myWrnd = Wrnd.rows(start_row, end_row);
      (*X) = myWrnd * Hrnd;
    } else if ((*X).n_rows == m) {  // ONED_COL
      MAT myHcols = Hrnd.cols(start_col, end_col);
      (*X) = Wrnd * myHcols;
    } else {  // TWOD
      MAT myWrnd = Wrnd.rows(start_row, end_row);
      MAT myHcols = Hrnd.cols(start_col, end_col);
      (*X) = myWrnd * myHcols;
    }
    if (adj_rand) {
      (*X).for_each(
          [](MAT::elem_type &val) { val = ceil(kalpha * val + kbeta);});
    }
#endif
  }

#ifdef BUILD_SPARSE
  void uniform_dist_matrix(MATTYPE& A) {
    // make the matrix of ONED distribution
    // sometimes in the pacoss it gives nearly equal
    // distribution. we have to make sure everyone of
    // same size;
    unsigned int max_rows = 0, max_cols = 0;
    unsigned int my_rows = A.n_rows;
    unsigned int my_cols = A.n_cols;
    bool last_exist = false;
    double my_min_value = 0.0;
    double overall_min;
    UWORD my_correct_nnz = 0;
    if (A.n_nonzero > 0) {
      my_min_value = A.values[0];
    }
    MPI_Allreduce(&my_rows, &max_rows, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&my_cols, &max_cols, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    // choose max rows and max cols nicely
    max_rows -= (max_rows % m_mpicomm.pr());
    max_cols -= (max_cols % m_mpicomm.pc());
    // count the number of nnz's that satisfies this
    // criteria
    try {
      if (A.n_nonzero > 0) {
        SP_MAT::iterator start_it = A.begin();
        SP_MAT::iterator end_it = A.end();
        for (SP_MAT::iterator it = start_it; it != end_it; ++it) {
          if (it.row() < max_rows && it.col() < max_cols) {
            my_correct_nnz++;
            if (*it != 0 && my_min_value < *it) {
              my_min_value = *it;
            }
          }
          if (it.row() == max_rows - 1 && it.col() == max_cols - 1) {
            last_exist = true;
          }
        }
      }
      if (!last_exist) {
        my_correct_nnz++;
      }

      if (A.n_nonzero == 0) {
        my_correct_nnz++;
      }
      MPI_Allreduce(&my_min_value, &overall_min, 1, MPI_INT, MPI_MIN,
                    MPI_COMM_WORLD);
      arma::umat locs;
      VEC vals;
      DISTPRINTINFO("max_rows::" << max_rows << "::max_cols::" << max_cols
                                 << "::my_rows::" << my_rows << "::my_cols::"
                                 << my_cols << "::last_exist::" << last_exist
                                 << "::my_nnz::" << A.n_nonzero
                                 << "::my_correct_nnz::" << my_correct_nnz);
      locs = arma::zeros<arma::umat>(2, my_correct_nnz);
      vals = arma::zeros<VEC>(my_correct_nnz);
      if (A.n_nonzero > 0) {
        SP_MAT::iterator start_it = A.begin();
        SP_MAT::iterator end_it = A.end();
        double idx = 0;
        for (SP_MAT::iterator it = start_it; it != end_it; ++it) {
          if (it.row() < max_rows && it.col() < max_cols) {
            locs(0, idx) = it.row();
            locs(1, idx) = it.col();
            vals(idx++) = *it;
          }
        }
      } else {
        locs(0, 0) = 0;
        locs(1, 0) = 0;
        vals(0) = overall_min;
      }
      if (A.n_nonzero > 0 && !last_exist) {
        locs(0, my_correct_nnz - 1) = max_rows - 1;
        locs(1, my_correct_nnz - 1) = max_cols - 1;
        vals(my_correct_nnz - 1) = overall_min;
      }
      SP_MAT A_new(locs, vals);
      A.clear();
      A = A_new;
    } catch (const std::exception& e) {
      DISTPRINTINFO(e.what()
                    << "max_rows::" << max_rows << "::max_cols::" << max_cols
                    << "::my_rows::" << my_rows << "::my_cols::" << my_cols
                    << "::last_exist::" << last_exist << "::my_nnz::"
                    << A.n_nonzero << "::my_correct_nnz::" << my_correct_nnz);
    }
  }
#endif
 public:
  DistIO<MATTYPE>(const MPICommunicator& mpic, const iodistributions& iod,
                  MATTYPE& A)
      : m_mpicomm(mpic), m_distio(iod), m_A(A) {}

  /**
   * We need m,n,pr,pc only for rand matrices. If otherwise we are
   * expecting the file will hold all the details.
   * If we are loading by file name we dont need distio flag.
   * If the filename is rand_lowrank/rand_uniform, appropriate
   * random functions will be called. Otherwise, it will be loaded from file.
   * @param[in] file_name. For random matrices rand_lowrank/rand_uniform
   * @param[in] m - globalm. Needed only for random matrices.
   *                otherwise, we will know from file.
   * @param[in] n - globaln. Needed only for random matrices
   * @param[in] k - low rank. Used for generating synthetic lowrank matrices.
   * @param[in] sparsity - sparsity factor between 0-1 for sparse matrices.
   * @param[in] pr - Number of row processors in the 2D processor grid
   * @param[in] pc - Number of columnn processors in the 2D processor grid
   * @param[in] symm - Flag to create symmetric matrix
   * @param[in] adj_rand - Flag to run elementwise scaling on each entry.
   * @param[in] normalization - L2 column normalization of input matrix.
   */
  void readInput(const std::string file_name, UWORD m = 0, UWORD n = 0,
                 UWORD k = 0, double sparsity = 0, UWORD pr = 0, UWORD pc = 0,
                 int symm = 0, bool adj_rand = false,
                 normtype i_normalization = NONE) {
    // INFO << "readInput::" << file_name << "::" << distio << "::"
    //     << m << "::" << n << "::" << pr << "::" << pc
    //     << "::" << this->MPI_RANK << "::" << this->m_mpicomm.size() <<
    //     std::endl;
    std::string rand_prefix("rand_");
    // Check file sizes
    if (symm) {
      assert(m == n);
    }

    if (!file_name.compare(0, rand_prefix.size(), rand_prefix)) {
      std::string type = file_name.substr(rand_prefix.size());
      assert(type == "normal" || type == "lowrank" || type == "uniform");

      // Initialise the jagged splits
      rcounts.zeros(pr);
      ccounts.zeros(pc);

      for (int r = 0; r < pr; r++) rcounts[r] = itersplit(m, pr, r);

      for (int r = 0; r < pc; r++) ccounts[r] = itersplit(n, pc, r);

      int row_rank = m_mpicomm.row_rank();
      int col_rank = m_mpicomm.col_rank();

      switch (m_distio) {
        // TODO{seswar3} Have to check the ONED initialisations
        case ONED_ROW:
          m_Arows.zeros(rcounts[row_rank], n);
          randMatrix(type, MPI_RANK + kPrimeOffset, sparsity, symm, adj_rand,
                     &m_Arows);
          if (type == "lowrank") {
            randomLowRank(m, n, k, symm, adj_rand, &m_Arows);
          }
          break;
        case ONED_COL:
          m_Acols.zeros(m, ccounts[col_rank]);
          randMatrix(type, MPI_RANK + kPrimeOffset, sparsity, symm, adj_rand,
                     &m_Acols);
          if (type == "lowrank") {
            randomLowRank(m, n, k, symm, adj_rand, &m_Acols);
          }
          break;
        case ONED_DOUBLE: {
          int p = MPI_SIZE;
          m_Arows.set_size(rcounts[row_rank], n);
          m_Acols.set_size(m, ccounts[col_rank]);
          randMatrix(type, MPI_RANK + kPrimeOffset, sparsity, symm, adj_rand,
                     &m_Arows);
          if (type == "lowrank") {
            randomLowRank(m, n, k, symm, adj_rand, &m_Arows);
          }
          randMatrix(type, MPI_RANK + kPrimeOffset, sparsity, symm, adj_rand,
                     &m_Acols);
          if (type == "lowrank") {
            randomLowRank(m, n, k, symm, adj_rand, &m_Acols);
          }
          break;
        }
        case TWOD:
          m_A.zeros(rcounts[row_rank], ccounts[col_rank]);
          randMatrix(type, MPI_RANK + kPrimeOffset, sparsity, symm,
                     adj_rand, &m_A);
          if (type == "lowrank") {
            randomLowRank(m, n, k, symm, adj_rand, &m_A);
          }
          break;
      }
    } else {
      std::stringstream sr, sc;
      if (m_distio == ONED_ROW || m_distio == ONED_DOUBLE) {
        sr << file_name << "rows_" << MPI_SIZE << "_" << MPI_RANK;
#ifdef BUILD_SPARSE
        m_Arows.load(sr.str(), arma::coord_ascii);
        uniform_dist_matrix(m_Arows);
#else
        m_Arows.load(sr.str());
#endif
      }
      if (m_distio == ONED_COL || m_distio == ONED_DOUBLE) {
        sc << file_name << "cols_" << MPI_SIZE << "_" << MPI_RANK;
#ifdef BUILD_SPARSE
        m_Acols.load(sc.str(), arma::coord_ascii);
        uniform_dist_matrix(m_Acols);
#else
        m_Acols.load(sc.str());
#endif
        m_Acols = m_Acols.t();
      }
      if (m_distio == TWOD) {
        // sr << file_name << "_" << MPI_SIZE << "_" << MPI_RANK;
        // sr << file_name << MPI_RANK;
        // The global rank to pr, pc is not deterministic and can be different.
        // Hence naming the file for 2D distributed with pr, pc.
        sr << file_name << MPI_ROW_RANK << "_" << MPI_COL_RANK;
        int pr = m_mpicomm.pr();
        int pc = m_mpicomm.pc();

        int srow = itersplit(m, pr, MPI_ROW_RANK);
        int scol = itersplit(n, pc, MPI_COL_RANK);
#ifdef BUILD_SPARSE
        MAT temp_ijv;
        temp_ijv.load(sr.str(), arma::raw_ascii);
        if (temp_ijv.n_rows > 0 && temp_ijv.n_cols > 0) {
          MAT vals(2, temp_ijv.n_rows);
          MAT idxs_only = temp_ijv.cols(0, 1);
          arma::umat idxs = arma::conv_to<arma::umat>::from(idxs_only);
          arma::umat idxst = idxs.t();
          vals = temp_ijv.col(2);
          SP_MAT temp_spmat(idxst, vals);
          m_A = temp_spmat;
        } else {
          arma::umat idxs = arma::zeros<arma::umat>(2, 1);
          VEC vals = arma::zeros<VEC>(1);
          SP_MAT temp_spmat(idxs, vals);
          m_A = temp_spmat;
        }
        // m_A.load(sr.str(), arma::coord_ascii);
        // uniform_dist_matrix(m_A);
        m_A.resize(srow, scol);
#else
        m_A.load(sr.str());
#endif
      }
    }
#ifndef BUILD_SPARSE
    if (i_normalization != NONE) {
      normalize(i_normalization);
    }
#endif
  }
  /**
   * Writes the factor matrix as output_file_name_W_MPISIZE_MPIRANK
   * @param[in] Local W factor matrix
   * @param[in] Local H factor matrix
   * @param[in] output file name
   */
  void writeOutput(const MAT& W, const MAT& H,
                   const std::string& output_file_name) {
    std::stringstream sw, sh;
    if (m_distio == TWOD) {
      sw << output_file_name << "_W_" << MPI_ROW_RANK << "_" << MPI_COL_RANK;
      sh << output_file_name << "_H_" << MPI_ROW_RANK << "_" << MPI_COL_RANK;
    } else {
      sw << output_file_name << "_W_" << MPI_SIZE << "_" << MPI_RANK;
      sh << output_file_name << "_H_" << MPI_SIZE << "_" << MPI_RANK;
    }
    W.save(sw.str(), arma::raw_ascii);
    H.save(sh.str(), arma::raw_ascii);
  }
  void writeRandInput() {
    std::string file_name("Arnd");
    std::stringstream sr, sc;
    if (m_distio == TWOD) {
      sr << file_name << "_" << MPI_SIZE << "_" << MPI_RANK;
      DISTPRINTINFO("Writing rand input file " << sr.str()
                                               << PRINTMATINFO(m_A));

#ifdef BUILD_SPARSE
      this->m_A.save(sr.str(), arma::coord_ascii);
#else
      this->m_A.save(sr.str(), arma::raw_ascii);
#endif
    }
    if (m_distio == ONED_ROW || m_distio == ONED_DOUBLE) {
      sr << file_name << "rows_" << MPI_SIZE << "_" << MPI_RANK;
      DISTPRINTINFO("Writing rand input file " << sr.str()
                                               << PRINTMATINFO(m_Arows));
#ifdef BUILD_SPARSE
      this->m_Arows.save(sr.str(), arma::coord_ascii);
#else
      this->m_Arows.save(sr.str(), arma::raw_ascii);
#endif
    }
    if (m_distio == ONED_COL || m_distio == ONED_DOUBLE) {
      sc << file_name << "cols_" << MPI_SIZE << "_" << MPI_RANK;
      DISTPRINTINFO("Writing rand input file " << sc.str()
                                               << PRINTMATINFO(m_Acols));
#ifdef BUILD_SPARSE
      this->m_Acols.save(sc.str(), arma::coord_ascii);
#else
      this->m_Acols.save(sc.str(), arma::raw_ascii);
#endif
    }
  }
  const MATTYPE& Arows() const { return m_Arows; }
  const MATTYPE& Acols() const { return m_Acols; }
  const MATTYPE& A() const { return m_A; }
  const MPICommunicator& mpicomm() const { return m_mpicomm; }
  const IVEC& row_counts() const { return rcounts; }
  const IVEC& col_counts() const { return ccounts; }
};

}  // namespace planc

// run with mpi run 3.
void testDistIO(char argc, char* argv[]) {

  planc::MPICommunicator mpicomm(argc, argv);
#ifdef BUILD_SPARSE
  SP_MAT A;
  planc::DistIO<SP_MAT> dio(mpicomm, ONED_DOUBLE, A);
#else
  MAT A;
  planc::DistIO<MAT> dio(mpicomm, ONED_DOUBLE, A);
#endif
  dio.readInput("rand", 12, 9, 0.5);
  INFO << "Arows:" << mpicomm.rank() << std::endl
       << arma::conv_to<MAT>::from(dio.Arows()) << std::endl;
  INFO << "Acols:" << mpicomm.rank() << std::endl
       << arma::conv_to<MAT>::from(dio.Acols()) << std::endl;
}

#endif  // DISTNMF_DISTIO_HPP_
