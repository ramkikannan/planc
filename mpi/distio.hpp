/* Copyright 2016 Ramakrishnan Kannan */
#ifndef MPI_DISTIO_HPP_
#define MPI_DISTIO_HPP_


#include <unistd.h>
#include <armadillo>
#include <string>
#include "mpicomm.hpp"
#include "distutils.hpp"

using namespace std;
using namespace arma;

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
    //Hope no one hits on this number.
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
            arma_rng::set_seed_random();
        } else {
            arma_rng::set_seed(random_sieve(primeseedidx));
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
        sp_fmat::iterator start_it = (*X).begin();
        sp_fmat::iterator end_it = (*X).end();
        for (sp_fmat::iterator it = start_it; it != end_it; ++it) {
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
    void randomLowRank(const uword m, const uword n, const uword k,
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
            if ((*X).n_cols == n) { //  m_Arows
                start_row = MPI_RANK * (*X).n_rows;
                end_row = ((MPI_RANK + 1) * (*X).n_rows) - 1;
                start_col = 0;
                end_col = n - 1;

            }
            if ((*X).n_rows == m) { // m_Acols
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
        arma_rng::set_seed(kW_seed_idx);
        fmat Wrnd(m, k);
        Wrnd.randu();
        fmat Hrnd(k, n);
        Hrnd.randu();
#ifdef BUILD_SPARSE
        sp_fmat::iterator start_it = (*X).begin();
        sp_fmat::iterator end_it = (*X).end();
        float tempVal = 0.0;
        for (sp_fmat::iterator it = start_it; it != end_it; ++it) {
            fvec Wrndi = vectorise(Wrnd.row(start_row + it.row()));
            fvec Hrndj = Hrnd.col(start_col + it.col());
            tempVal =  dot(Wrndi, Hrndj);
            (*it) = ceil(kalpha * tempVal + kbeta);
        }
#else
        fmat templr;
        if ((*X).n_cols == n) { // ONED_ROW
            fmat myWrnd = Wrnd.rows(start_row, end_row);
            templr =  myWrnd * Hrnd;
        } else if ((*X).n_rows == m) {  // ONED_COL
            fmat myHcols = Hrnd.cols(start_col, end_col);
            templr = Wrnd * myHcols;
        } else if ((*X).n_rows == (m / MPI_SIZE) &&
                   (*X).n_cols == (n / MPI_SIZE) ||
                   ((*X).n_rows == (m / this->m_mpicomm.pr()) &&
                    (*X).n_cols == (n / this->m_mpicomm.pc()))) {
            fmat myWrnd = Wrnd.rows(start_row, end_row);
            fmat myHcols = Hrnd.cols(start_col, end_col);
            templr = myWrnd * myHcols;
        }
        (*X) = ceil(kalpha * templr + kbeta);
#endif
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
                   uword m = 0, uword n = 0, uword k = 0, double sparsity = 0,
                   uword pr = 0, uword pc = 0) {
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
                m_Arows.load(sr.str(), coord_ascii);
#else
                m_Arows.load(sr.str(), raw_ascii);
#endif
            }
            if (m_distio == ONED_COL || m_distio == ONED_DOUBLE) {
                sc << file_name << "cols_" << MPI_SIZE << "_" << MPI_RANK;
#ifdef BUILD_SPARSE
                m_Acols.load(sc.str(), coord_ascii);                
#else
                m_Acols.load(sc.str(), raw_ascii);
#endif
                m_Acols=m_Acols.t();
            }
            if (m_distio == TWOD) {
                sr << file_name << "_" << MPI_SIZE << "_" << MPI_RANK;
#ifdef BUILD_SPARSE
                m_A.load(sr.str(), coord_ascii);
#else
                m_A.load(sr.str(), raw_ascii);
#endif
            }
        }
    }
    void writeOutput(const fmat & W, const fmat & H,
                     const std::string & output_file_name) {
        stringstream sw, sh;
        sw << output_file_name << "_W_" << MPI_SIZE << "_" << MPI_RANK;
        sh << output_file_name << "_H_" << MPI_SIZE << "_" << MPI_RANK;
        W.save(sw, raw_ascii);
        H.save(sh, raw_ascii);
    }
    void writeRandInput() {
        std::string file_name("Arnd");
        stringstream sr, sc;
        if (m_distio == TWOD) {
            sr << file_name << "_" << MPI_SIZE << "_" << MPI_RANK;
            DISTPRINTINFO("Writing rand input file " << sr.str() \
                          << PRINTMATINFO(m_A));

#ifdef BUILD_SPARSE
            this->m_A.save(sr.str(), coord_ascii);
#else
            this->m_A.save(sr.str(), raw_ascii);
#endif
        }
        if (m_distio == ONED_ROW || m_distio == ONED_DOUBLE) {
            sr << file_name << "rows_" << MPI_SIZE << "_" << MPI_RANK;
            DISTPRINTINFO("Writing rand input file " << sr.str() \
                          << PRINTMATINFO(m_Arows));
#ifdef BUILD_SPARSE
            this->m_Arows.save(sr.str(), coord_ascii);
#else
            this->m_Arows.save(sr.str(), raw_ascii);
#endif
        }
        if (m_distio == ONED_COL || m_distio == ONED_DOUBLE) {
            sc << file_name << "cols_" << MPI_SIZE << "_" << MPI_RANK;
            DISTPRINTINFO("Writing rand input file " << sc.str()\
                          << PRINTMATINFO(m_Acols));
#ifdef BUILD_SPARSE
            this->m_Acols.save(sc.str(), coord_ascii);
#else
            this->m_Acols.save(sc.str(), raw_ascii);
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
    DistIO<sp_fmat> dio(mpicomm, ONED_DOUBLE);
#else
    DistIO<fmat> dio(mpicomm, ONED_DOUBLE);
#endif
    dio.readInput("rand", 12, 9, 0.5);
    cout << "Arows:" << mpicomm.rank() << endl
         << conv_to<fmat>::from(dio.Arows()) << endl;
    cout << "Acols:" << mpicomm.rank() << endl
         << conv_to<fmat>::from(dio.Acols()) << endl;
}
#endif  // MPI_DISTIO_HPP_
