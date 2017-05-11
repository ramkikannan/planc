/* Copyright 2016 Ramakrishnan Kannan */
#ifndef MPI_DISTIO_HPP_
#define MPI_DISTIO_HPP_


#include <unistd.h>
#include <armadillo>
#include <string>
#include "mpicomm.hpp"
#include "distutils.hpp"

using namespace std;

/*
 * File name formats
 * A is the filename
 * 1D distribution Arows_totalpartitions_rank or Acols_totalpartitions_rank
 * Double 1D distribution (both row and col distributed)
 * Arows_totalpartitions_rank and Acols_totalpartitions_rank
 * TWOD distribution A_totalpartition_rank
 * Just send the first parameter Arows and the second parameter Acols to be zero.
 */

template <class MATTYPE>
class DistIO {
  private:
    const MPICommunicator& m_mpicomm;
    MATTYPE m_Arows;        // ONED_ROW and ONED_DOUBLE
    MATTYPE m_Acols;        // ONED_COL and ONED_DOUBLE
    MATTYPE m_A;            // TWOD
    // don't start getting prime number from 2;
    static const int kPrimeOffset = 10;
    // Hope no one hits on this number.
    static const int kW_seed_idx = 1210873;
#ifdef BUILD_SPARSE
    static const int kalpha = 5;
    static const int kbeta = 10;
#else
    static const int kalpha = 1;
    static const int kbeta = 0;
#endif

    const iodistributions m_distio;
    /*
    * A random matrix is always needed for sparse case
    * to get the pattern. That is., the indices where
    * numbers are being filled. We will use this pattern
    * and change the number later.
    */
    void randMatrix(const std::string type, const int primeseedidx,
                    const double sparsity,
                    MATTYPE *X) {
        if (primeseedidx == -1) {
            arma::arma_rng::set_seed_random();
        } else {
            arma::arma_rng::set_seed(random_sieve(primeseedidx));
        }
#ifdef DEBUG_VERBOSE
        DISTPRINTINFO("randMatrix::" << primeseedidx << "::sp=" << sparsity);
#endif
#ifdef BUILD_SPARSE
        if (type == "uniform" || type == "lowrank") {
            (*X).sprandu((*X).n_rows, (*X).n_cols, sparsity);
        } else if (type == "normal") {
            (*X).sprandn((*X).n_rows, (*X).n_cols, sparsity);
        }
        SP_FMAT::iterator start_it = (*X).begin();
        SP_FMAT::iterator end_it = (*X).end();
        for (SP_FMAT::iterator it = start_it; it != end_it; ++it) {
            float currentValue = (*it);
            (*it) = ceil(kalpha * currentValue + kbeta);
            if ((*it) < 0) (*it) = kbeta;
        }
        // for (uint i=0; i<(*X).n_nonzero;i++){
        //     float currentValue = (*X).values[i];
        //     currentValue *= kalpha;
        //     currentValue += kbeta;
        //     currentValue = ceil(currentValue);
        //     if (currentValue <= 0) currentValue=kbeta;
        //     (*X).values[i]=currentValue;
        // }
#else
        if (type == "uniform") {
            (*X).randu();
        } else if (type == "normal") {
            (*X).randn();
        }
        (*X) = kalpha * (*X) + kbeta;
        (*X) = ceil(*X);
        (*X).elem(find((*X) < 0)).zeros();
#endif
    }
    /*
    * Uses the pattern from the input matrix X but
    * the value is computed as low rank.
    */
    void randomLowRank(const UWORD m, const UWORD n, const UWORD k,
                       MATTYPE *X) {
        uint start_row = 0, end_row = 0, start_col = 0, end_col = 0;
        switch (m_distio) {
        case ONED_ROW:
            start_row = MPI_RANK * (*X).n_rows;
            end_row = ((MPI_RANK + 1) * (*X).n_rows) - 1;
            start_col = 0;
            end_col = (*X).n_cols - 1;
            break;
        case ONED_COL:
            start_row = 0;
            end_row = (*X).n_rows - 1;
            start_col = MPI_RANK * (*X).n_cols;
            end_col = ((MPI_RANK + 1) * (*X).n_cols) - 1;
            break;
        // in the case of ONED_DOUBLE we are stitching
        // m/p x n/p matrices and in TWOD we are
        // stitching m/pr * n/pc matrices.
        case ONED_DOUBLE:
            if ((*X).n_cols == n) {  //  m_Arows
                start_row = MPI_RANK * (*X).n_rows;
                end_row = ((MPI_RANK + 1) * (*X).n_rows) - 1;
                start_col = 0;
                end_col = n - 1;
            }
            if ((*X).n_rows == m) {  // m_Acols
                start_row = 0;
                end_row = m - 1;
                start_col = MPI_RANK * (*X).n_cols;
                end_col = ((MPI_RANK + 1) * (*X).n_cols) - 1;
            }
            break;
        case TWOD:
            start_row = MPI_ROW_RANK * (*X).n_rows;
            end_row = ((MPI_ROW_RANK + 1) * (*X).n_rows) - 1;
            start_col = MPI_COL_RANK * (*X).n_cols;
            end_col = ((MPI_COL_RANK + 1) * (*X).n_cols) - 1;
            break;
        }
        // all machines will generate same Wrnd and Hrnd
        // at all times.
        arma::arma_rng::set_seed(kW_seed_idx);
        FMAT Wrnd(m, k);
        Wrnd.randu();
        FMAT Hrnd(k, n);
        Hrnd.randu();
#ifdef BUILD_SPARSE
        SP_FMAT::iterator start_it = (*X).begin();
        SP_FMAT::iterator end_it = (*X).end();
        float tempVal = 0.0;
        for (SP_FMAT::iterator it = start_it; it != end_it; ++it) {
            FVEC Wrndi = vectorise(Wrnd.row(start_row + it.row()));
            FVEC Hrndj = Hrnd.col(start_col + it.col());
            tempVal =  dot(Wrndi, Hrndj);
            (*it) = ceil(kalpha * tempVal + kbeta);
        }
#else
        FMAT templr;
        if ((*X).n_cols == n) {  // ONED_ROW
            FMAT myWrnd = Wrnd.rows(start_row, end_row);
            templr =  myWrnd * Hrnd;
        } else if ((*X).n_rows == m) {  // ONED_COL
            FMAT myHcols = Hrnd.cols(start_col, end_col);
            templr = Wrnd * myHcols;
        } else if ((*X).n_rows == (m / MPI_SIZE) &&
                   (*X).n_cols == (n / MPI_SIZE) ||
                   ((*X).n_rows == (m / this->m_mpicomm.pr()) &&
                    (*X).n_cols == (n / this->m_mpicomm.pc()))) {
            FMAT myWrnd = Wrnd.rows(start_row, end_row);
            FMAT myHcols = Hrnd.cols(start_col, end_col);
            templr = myWrnd * myHcols;
        }
        (*X) = ceil(kalpha * templr + kbeta);
#endif
    }

    void uniform_dist_matrix(MATTYPE &A) {
        // make the matrix of ONED distribution
        // sometimes in the pacoss it gives nearly equal
        // distribution. we have to make sure everyone of
        // same size;
        int max_rows = 0, max_cols = 0;
        int my_rows = A.n_rows;
        int my_cols = A.n_cols;
        bool last_exist = false;
        float my_min_value = 0.0;
        if (A.n_nonzero > 0) {
            my_min_value = A.min();
        }
        float overall_min;
        MPI_Allreduce(&my_rows, &max_rows, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&my_cols, &max_cols, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&my_min_value, &overall_min, 1, MPI_INT,
                      MPI_MIN, MPI_COMM_WORLD);
        //choose max rows and max cols nicely
        max_rows -= (max_rows % m_mpicomm.pr());
        max_cols -= (max_cols % m_mpicomm.pc());
        // count the number of nnz's that satisfies this
        // criteria
        UWORD my_correct_nnz = 0;
        if (A.n_nonzero > 0) {
            SP_FMAT::iterator start_it = A.begin();
            SP_FMAT::iterator end_it = A.end();
            for (SP_FMAT::iterator it = start_it; it != end_it; ++it) {
                if (it.row() < max_rows && it.col() < max_cols) {
                    my_correct_nnz++;
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
        DISTPRINTINFO("max_rows::" << max_rows
                      << "::max_cols::" << max_cols
                      << "::my_rows::" << my_rows
                      << "::my_cols::" << my_cols
                      << "::last_exist::" << last_exist
                      << "::my_nnz::" << A.n_nonzero
                      << "::my_correct_nnz::" << my_correct_nnz);
        arma::umat locs;
        FVEC vals;
        locs = arma::zeros<arma::umat>(2, my_correct_nnz);
        vals = arma::zeros<FVEC>(my_correct_nnz);
        if (A.n_nonzero > 0) {
            SP_FMAT::iterator start_it = A.begin();
            SP_FMAT::iterator end_it = A.end();
            float idx = 0;
            for (SP_FMAT::iterator it = start_it; it != end_it; ++it) {
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
        if (!last_exist) {
            locs(0, my_correct_nnz - 1) = max_rows - 1;
            locs(1, my_correct_nnz - 1) = max_cols - 1;
            vals(my_correct_nnz - 1) = overall_min;
        }
        SP_FMAT A_new(locs, vals);
        A.clear();
        A = A_new;
    }
  public:
    DistIO<MATTYPE>(const MPICommunicator &mpic, const iodistributions &iod):
        m_mpicomm(mpic), m_distio(iod) {
    }
    /*
     * We need m,n,pr,pc only for rand matrices. If otherwise we are
     * expecting the file will hold all the details.
     * If we are loading by file name we dont need distio flag.
     *
     */
    void readInput(const std::string file_name,
                   UWORD m = 0, UWORD n = 0, UWORD k = 0, double sparsity = 0,
                   UWORD pr = 0, UWORD pc = 0) {
        // INFO << "readInput::" << file_name << "::" << distio << "::"
        //     << m << "::" << n << "::" << pr << "::" << pc
        //     << "::" << this->MPI_RANK << "::" << this->m_mpicomm.size() << endl;
        std::string rand_prefix("rand_");
        if (!file_name.compare(0, rand_prefix.size(), rand_prefix)) {
            std::string type = file_name.substr(rand_prefix.size());
            assert(type == "normal" || type == "lowrank" || type == "uniform");
            switch (m_distio) {
            case ONED_ROW:
                m_Arows.zeros(m / MPI_SIZE, n);
                randMatrix(type, MPI_RANK + kPrimeOffset, sparsity, &m_Arows);
                if (type == "lowrank") {
                    randomLowRank(m, n, k, &m_Arows);
                }
                break;
            case ONED_COL:
                m_Acols.zeros(m , n / MPI_SIZE);
                randMatrix(type, MPI_RANK + kPrimeOffset, sparsity, &m_Acols);
                if (type == "lowrank") {
                    randomLowRank(m, n, k, &m_Acols);
                }
                break;
            case ONED_DOUBLE: {
                int p = MPI_SIZE;
                m_Arows.set_size(m / p, n);
                m_Acols.set_size(m, n / p);
                randMatrix(type, MPI_RANK + kPrimeOffset, sparsity, &m_Arows);
                if (type == "lowrank") {
                    randomLowRank(m, n, k, &m_Arows);
                }
                randMatrix(type, MPI_RANK + kPrimeOffset, sparsity, &m_Acols);
                if (type == "lowrank") {
                    randomLowRank(m, n, k, &m_Acols);
                }
                break;
            }
            case TWOD:
                m_A.zeros(m / pr, n / pc);
                randMatrix(type, MPI_RANK + kPrimeOffset, sparsity, &m_A);
                if (type == "lowrank") {
                    randomLowRank(m, n, k, &m_A);
                }
                break;
            }
        } else {
            stringstream sr, sc;
            if (m_distio == ONED_ROW || m_distio == ONED_DOUBLE) {
                sr << file_name << "rows_" << MPI_SIZE << "_" << MPI_RANK;
#ifdef BUILD_SPARSE
                m_Arows.load(sr.str(), arma::coord_ascii);
#else
                m_Arows.load(sr.str(), arma::raw_ascii);
#endif
                uniform_dist_matrix(m_Arows);
            }
            if (m_distio == ONED_COL || m_distio == ONED_DOUBLE) {
                sc << file_name << "cols_" << MPI_SIZE << "_" << MPI_RANK;
#ifdef BUILD_SPARSE
                m_Acols.load(sc.str(), arma::coord_ascii);
#else
                m_Acols.load(sc.str(), arma::raw_ascii);
#endif
                m_Acols = m_Acols.t();
                uniform_dist_matrix(m_Acols);
            }
            if (m_distio == TWOD) {
                //sr << file_name << "_" << MPI_SIZE << "_" << MPI_RANK;
                sr << file_name << MPI_RANK;
#ifdef BUILD_SPARSE
                FMAT temp_ijv;
                temp_ijv.load(sr.str(), arma::raw_ascii);
                if (temp_ijv.n_rows > 0 && temp_ijv.n_cols > 0) {
                    arma::umat idxs(2, temp_ijv.n_rows);
                    FMAT vals(2, temp_ijv.n_rows);
                    FMAT idxs_only = temp_ijv.cols(0, 1);
                    idxs = arma::conv_to<arma::umat>::from(idxs_only);
                    arma::umat idxst = idxs.t();
                    vals = temp_ijv.col(2);
                    SP_FMAT temp_spfmat(idxst, vals);
                    m_A = temp_spfmat;
                } else {
                    arma::umat idxs = arma::zeros<arma::umat>(2,1);
                    FVEC vals = arma::zeros<FVEC>(1);
                    SP_FMAT temp_spfmat(idxs,vals);
                    m_A = temp_spfmat;
                }
                // m_A.load(sr.str(), arma::coord_ascii);
#else
                m_A.load(sr.str(), arma::raw_ascii);
#endif
                uniform_dist_matrix(m_A);
            }
        }
    }
    void writeOutput(const FMAT & W, const FMAT & H,
                     const std::string & output_file_name) {
        stringstream sw, sh;
        sw << output_file_name << "_W_" << MPI_SIZE << "_" << MPI_RANK;
        sh << output_file_name << "_H_" << MPI_SIZE << "_" << MPI_RANK;
        W.save(sw, arma::raw_ascii);
        H.save(sh, arma::raw_ascii);
    }
    void writeRandInput() {
        std::string file_name("Arnd");
        stringstream sr, sc;
        if (m_distio == TWOD) {
            sr << file_name << "_" << MPI_SIZE << "_" << MPI_RANK;
            DISTPRINTINFO("Writing rand input file " << sr.str() \
                          << PRINTMATINFO(m_A));

#ifdef BUILD_SPARSE
            this->m_A.save(sr.str(), arma::coord_ascii);
#else
            this->m_A.save(sr.str(), arma::raw_ascii);
#endif
        }
        if (m_distio == ONED_ROW || m_distio == ONED_DOUBLE) {
            sr << file_name << "rows_" << MPI_SIZE << "_" << MPI_RANK;
            DISTPRINTINFO("Writing rand input file " << sr.str() \
                          << PRINTMATINFO(m_Arows));
#ifdef BUILD_SPARSE
            this->m_Arows.save(sr.str(), arma::coord_ascii);
#else
            this->m_Arows.save(sr.str(), arma::raw_ascii);
#endif
        }
        if (m_distio == ONED_COL || m_distio == ONED_DOUBLE) {
            sc << file_name << "cols_" << MPI_SIZE << "_" << MPI_RANK;
            DISTPRINTINFO("Writing rand input file " << sc.str()\
                          << PRINTMATINFO(m_Acols));
#ifdef BUILD_SPARSE
            this->m_Acols.save(sc.str(), arma::coord_ascii);
#else
            this->m_Acols.save(sc.str(), arma::raw_ascii);
#endif
        }
    }
    const MATTYPE& Arows() const {return m_Arows;}
    const MATTYPE& Acols() const {return m_Acols;}
    const MATTYPE& A() const {return m_A;}
    const MPICommunicator& mpicomm() const {return m_mpicomm;}
};

// run with mpi run 3.
void testDistIO(char argc, char *argv[]) {
    MPICommunicator mpicomm(argc, argv);
#ifdef BUILD_SPARSE
    DistIO<SP_FMAT> dio(mpicomm, ONED_DOUBLE);
#else
    DistIO<FMAT> dio(mpicomm, ONED_DOUBLE);
#endif
    dio.readInput("rand", 12, 9, 0.5);
    cout << "Arows:" << mpicomm.rank() << endl
         << arma::conv_to<FMAT >::from(dio.Arows()) << endl;
    cout << "Acols:" << mpicomm.rank() << endl
         << arma::conv_to<FMAT >::from(dio.Acols()) << endl;
}
#endif  // MPI_DISTIO_HPP_
