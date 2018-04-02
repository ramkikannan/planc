/* Copyright 2017 Ramakrishnan Kannan */

#ifndef NTF_TENSOR_HPP_
#define NTF_TENSOR_HPP_

#include <armadillo>
#include <random>
#include <type_traits>
#include <ios>
#include <fstream>
#include <cblas.h>

namespace planc {
/*
 * Data is stored such that the unfolding \f$Y_0\f$ is column
 * major.  This means the flattening \f$Y_{N-1}\f$ is row-major,
 * and any other flattening \f$Y_n\f$ can be represented as a set
 * of \f$\prod\limits_{k=n+1}^{N-1}I_k\f$ row major matrices, each
 * of which is \f$I_n \times \prod\limits_{k=0}^{n-1}I_k\f$.
 */

// sgemm (TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
//extern "C" void dgemm_(const char*, const char*, const int*,
//                      const int*, const int*, const double*, const double*, const int*,
//                      const double*, const int*, const double*, double*, const int*);

class Tensor {
  private:
    int m_modes;
    UVEC m_dimensions;
    UWORD m_numel;
    unsigned int rand_seed;
    bool freed_on_destruction;

  public:
    double* m_data;
    Tensor() {
        this->m_modes = 0;
        this->m_numel = 0;
        this->m_data = NULL;
        freed_on_destruction = false;
    }
    Tensor(const UVEC& i_dimensions):
        m_dimensions(i_dimensions),
        m_modes(i_dimensions.n_rows),
        m_numel(arma::prod(i_dimensions)),
        rand_seed(103) {
        m_data = new double[this->m_numel];
        freed_on_destruction = false;
        randu();
    }
    // Need when copying from matrix to Tensor.
    // otherwise copy constructor will be called.
    Tensor(const UVEC& i_dimensions, double *i_data):
        m_dimensions(i_dimensions),
        m_modes(i_dimensions.n_rows),
        m_numel(arma::prod(i_dimensions)),
        rand_seed(103) {
        m_data = new double[this->m_numel];
        memcpy(this->m_data, i_data, sizeof(double)*this->m_numel);
        freed_on_destruction = false;
    }
    ~Tensor() {
        if (!freed_on_destruction) {
            delete[] m_data;
            freed_on_destruction = true;
        }
    }
    //copy constructor
    Tensor(const Tensor &src) {
        clear();
        this->m_numel = src.numel();
        this->m_modes = src.modes();
        this->m_dimensions = src.dimensions();
        this->rand_seed  = 103;
        m_data = new double[this->m_numel];
        memcpy(this->m_data, src.m_data, sizeof(double)*this->m_numel);
    }
    Tensor& operator=(const Tensor& other) { // copy assignment
        if (this != &other) { // self-assignment check expected
            clear();
            this->m_data = new double[other.numel()]; // create storage in this
            this->m_numel = other.numel();
            this->m_modes = other.modes();
            this->m_dimensions = other.dimensions();
            this->freed_on_destruction = false;
            memcpy(this->m_data, other.m_data, sizeof(double)*this->m_numel);
        }
        return *this;
    }

    void clear() {
        if (this->m_numel > 0 && this->m_data != NULL) {
            delete[] this->m_data;  // destroy storage in this
            this->m_numel = 0;
            this->m_data = NULL;
            freed_on_destruction = false;
        }
    }

    int modes() const {return m_modes;}
    UVEC dimensions() const {return m_dimensions;}
    int dimension(int i) const {return m_dimensions[i];}
    int numel() const {return m_numel;}

    UWORD dimensions_leave_out_one(int i_n) const {
        UWORD rc = arma::prod(this->m_dimensions);
        return rc - this->m_dimensions(i_n);
    }
    void zeros() {
        for (UWORD i = 0; i < this->m_numel; i++) {
            this->m_data[i] = 0;
        }
    }

    void rand() {
        for (UWORD i = 0; i < this->m_numel; i++) {
            unsigned int *temp = const_cast<unsigned int *>(&rand_seed);
            this->m_data[i] = static_cast <double> (rand_r(temp))
                              / static_cast <double> (RAND_MAX);
        }
    }

    void randi() {
        std::random_device rd;
        std::mt19937 gen(rd());
        int max_randi = this->m_numel;
        std::uniform_int_distribution<> dis(0, max_randi);
        for (UWORD i = 0; i < this->m_numel; i++) {
            this->m_data[i] = dis(gen);
        }
    }

    void randu(const int i_seed = -1) {
        std::random_device rd;
        std::uniform_real_distribution<> dis(0, 1);
        if (i_seed == -1) {
            std::mt19937 gen(rand_seed);
            for (int i = 0; i < this->m_numel; i++) {
                m_data[i] = dis(gen);
            }
        } else {
            std::mt19937 gen(i_seed);
            for (int i = 0; i < this->m_numel; i++) {
                m_data[i] = dis(gen);
            }
        }
    }

    // size of krp must be product of all dimensions leaving out nxk
    // o_mttkrp will be of size dimension[n]xk
    // implementation of mttkrp from tensor toolbox
    // const at the end as this does not change any local data.
    void mttkrp(const int i_n, const MAT& i_krp, MAT *o_mttkrp) const {
        (*o_mttkrp).zeros();
        if (i_n == 0) {
            //if n == 1
            //Ur = khatrirao(U{2: N}, 'r');
            //Y = reshape(X.data, szn, szr);
            //V =  Y * Ur;
            // Compute number of columns of Y_n
            // Technically, we could divide the total number of entries by n,
            // but that seems like a bad decision
            // size_t ncols = arma::prod(this->m_dimensions);
            // ncols /= this->m_dimensions[0];
            // Call matrix matrix multiply
            // call dgemm (TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
            // C := alpha*op( A )*op( B ) + beta*C
            // A, B and C are matrices, with op( A ) an m by k matrix,
            // op( B ) a k by n matrix and C an m by n matrix.
            // matricized tensor is m x k is in column major format
            // krp is k x n is in column major format
            // output is m x n in column major format
            char transa = 'N';
            char transb = 'N';
            int m = this->m_dimensions[0];
            int n = i_krp.n_cols;
            int k = i_krp.n_rows;
            int lda = m;
            int ldb = k;
            int ldc = m;
            double alpha = 1;
            double beta = 0;
            // sgemm (TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
            //dgemm_(&transa, &transb, &m, &n, &k, &alpha, this->m_data,
            // &lda, i_krp.memptr(), &ldb, &beta, o_mttkrp->memptr() , &ldc);
            // printf("mode=%d,i=%d,m=%d,n=%d,k=%d,alpha=%lf,T_stride=%d,lda=%d,krp_stride=%d,ldb=%d,beat=%lf,mttkrp=!!!,ldc=%d\n",0, 0, m, n, k,alpha,0*k*m,m,0*n*k,i_krp.n_rows,beta,n);
            cblas_dgemm( CblasRowMajor, CblasTrans, CblasTrans, m, n, k, alpha, this->m_data, m, i_krp.memptr(), k, beta, o_mttkrp->memptr(), n );

        } else  {
            int ncols = 1;
            int nmats = 1;
            int lowrankk = i_krp.n_cols;

            // Count the number of columns
            for (int i = 0; i < i_n; i++) {
                ncols *= this->m_dimensions[i];
            }

            // Count the number of matrices
            for (int i = i_n + 1; i < this->m_modes; i++) {
                nmats *= this->m_dimensions[i];
            }
            // For each matrix...
            for (int i = 0; i < nmats; i++) {
                char transa = 'T';
                char transb = 'N';
                int m = this->m_dimensions[i_n];
                int n = lowrankk;
                int k = ncols;
                int lda = k; // not sure. could be m. higher confidence on k.
                int ldb = i_krp.n_rows;
                int ldc = m;
                double alpha = 1;
                double beta = (i == 0) ? 0 : 1;
                double *A = this->m_data + i * k * m;
                double *B = const_cast<double *>(i_krp.memptr()) + i * k;

                // for KRP move ncols*lowrankk
                // for tensor X move as n_cols*blas_n
                // For output matrix don't move anything as beta=1;
                // for reference from gram while moving input tensor like Y->data()+i*nrows*ncols
                // sgemm (TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
                //dgemm_(&transa, &transb, &m, &n, &k, &alpha, A,
                //&lda, B , &ldb, &beta, (*o_mttkrp).memptr() , &ldc);
                // printf("mode=%d,i=%d,m=%d,n=%d,k=%d,alpha=%lf,T_stride=%d,lda=%d,krp_stride=%d,ldb=%d,beat=%lf,mttkrp=!!!,ldc=%d\n",i_n, i, m, n, k,alpha,i*k*m,k,i*n*k,nmats*ncols,beta,n);
                cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, this->m_data + i * k * m, ncols, i_krp.memptr() + i * k, ncols * nmats, beta, o_mttkrp->memptr(), n );
            }
        }
    }

    void print() const {
        for (int i = 0; i < this->m_numel; i++) {
            std::cout << i << " : " << this->m_data[i] << std::endl;
        }
    }
    double norm() const {
        double norm_fro = 0;
        for (int i = 0; i < this->m_numel; i++) {
            norm_fro += (this->m_data[i] * this->m_data[i]);
        }
        return norm_fro;
    }
    double err(const Tensor &b) const {
        double norm_fro = 0;
        double err_diff;
        for (int i = 0; i < this->m_numel; i++) {
            err_diff = this->m_data[i] - b.m_data[i];
            norm_fro += err_diff * err_diff;
        }
        return norm_fro;
    }
    template <typename NumericType>
    void scale(NumericType scale) {
        // static_assert(std::is_arithmetic<NumericType>::value,
        //               "NumericType for scale operation must be numeric");
        for (int i = 0; i < this->m_numel; i++) {
            this->m_data[i] = this->m_data[i] * scale;
        }
    }
    template <typename NumericType>
    void shift(NumericType scale) {
        // static_assert(std::is_arithmetic<NumericType>::value,
        //               "NumericType for shift operation must be numeric");
        for (int i = 0; i < this->m_numel; i++) {
            this->m_data[i] = this->m_data[i] + scale;
        }
    }
    template <typename NumericType>
    void bound(NumericType min, NumericType max) {
        // static_assert(std::is_arithmetic<NumericType>::value,
        //               "NumericType for bound operation must be numeric");
        for (int i = 0; i < this->m_numel; i++) {
            if (this->m_data[i] < min) this->m_data[i] = min;
            if (this->m_data[i] > max) this->m_data[i] = max;
        }
    }
    template <typename NumericType>
    void lower_bound(NumericType min) {
        // static_assert(std::is_arithmetic<NumericType>::value,
        //               "NumericType for bound operation must be numeric");
        for (int i = 0; i < this->m_numel; i++) {
            if (this->m_data[i] < min) this->m_data[i] = min;
        }
    }

    void save(std::string filename, std::ios_base::openmode mode = std::ios_base::out) {
        std::string filename_no_extension = filename.substr(0,
                                            filename.find_last_of("."));
        // info file always in text mode
        filename_no_extension.append(".info");
        std::ofstream ofs;
        ofs.open(filename_no_extension, std::ios_base::out);
        // write modes
        ofs << this->m_modes << std::endl;
        // dimension of modes
        for (int i = 0; i < this->m_modes; i++) {
            ofs << this->m_dimensions[i] << " ";
        }
        ofs << std::endl;
        ofs.close();
        ofs.open(filename, mode);
        // write elements
        for (size_t i = 0; i < this->m_numel; i++) {
            ofs << this->m_data[i] << std::endl;
        }
        // Close the file
        ofs.close();
    }
    void read(std::string filename, std::ios_base::openmode mode = std::ios_base::in) {
        // clear existing tensor
        if (this->m_numel > 0) {
            delete[] this->m_data;  // destroy storage in this
            this->m_numel = 0;
            this->m_data = NULL;  // preserve invariants in case next line throws
        }
        std::string filename_no_extension = filename.substr(0,
                                            filename.find_last_of("."));
        filename_no_extension.append(".info");

        std::ifstream ifs;
        // info file always in text mode
        ifs.open(filename_no_extension, std::ios_base::in);
        // write modes
        ifs >> this->m_modes;
        // dimension of modes
        this->m_dimensions = arma::zeros<UVEC>(this->m_modes);
        for (int i = 0; i < this->m_modes; i++) {
            ifs >> this->m_dimensions[i];
        }
        ifs.close();
        ifs.open(filename, mode);
        this->m_numel = arma::prod(this->m_dimensions);
        this->m_data = new double[this->m_numel];
        for (int i = 0; i < this->m_numel; i++) {
            ifs >> this->m_data[i];
        }
        // Close the file
        ifs.close();
    }

    size_t sub2ind(UVEC sub) {
        assert(sub.n_cols == this->m_dimensions.n_cols);
        UVEC cumprod_dims = arma::cumprod(this->m_dimensions);
        UVEC cumprod_dims_shifted = arma::shift(cumprod_dims, 1);
        cumprod_dims_shifted(0) = 1;
        size_t idx =  arma::dot(cumprod_dims_shifted, sub);
        return idx;
    }
    double at(UVEC sub) {
        return m_data[sub2ind(sub)];
    }

};  // class Tensor
}  // namespace planc

#endif  // NTF_TENSOR_HPP_
