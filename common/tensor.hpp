/* Copyright 2017 Ramakrishnan Kannan */

#ifndef NTF_TENSOR_HPP_
#define NTF_TENSOR_HPP_

#include <armadillo>
#include <random>
#include <type_traits>

namespace planc {
/*
 * Data is stored such that the unfolding \f$Y_0\f$ is column
 * major.  This means the flattening \f$Y_{N-1}\f$ is row-major,
 * and any other flattening \f$Y_n\f$ can be represented as a set
 * of \f$\prod\limits_{k=n+1}^{N-1}I_k\f$ row major matrices, each
 * of which is \f$I_n \times \prod\limits_{k=0}^{n-1}I_k\f$.
 */

// sgemm (TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
extern "C" void dgemm_(const char*, const char*, const int*,
                       const int*, const int*, const double*, const double*, const int*,
                       const double*, const int*, const double*, double*, const int*);

class Tensor {
  private:
    int m_modes;
    UVEC m_dimensions;
    UWORD m_numel;
    unsigned int rand_seed;

  public:
    double* m_data;
    Tensor(){
        this->m_modes = 0;        
        this->m_numel = 0;
        this->m_data = NULL;
    }  
    Tensor(const UVEC& i_dimensions):
        m_dimensions(i_dimensions),
        m_modes(i_dimensions.n_rows),
        m_numel(arma::prod(i_dimensions)),
        rand_seed(103) {
        m_data = new double[this->m_numel];
        randu();
    }
    Tensor(const UVEC& i_dimensions, double *i_data):
        m_dimensions(i_dimensions),
        m_modes(i_dimensions.n_rows),
        m_numel(arma::prod(i_dimensions)),
        rand_seed(103) {
        m_data = new double[this->m_numel];
        memcpy(this->m_data, i_data, sizeof(double)*this->m_numel);
    }
    ~Tensor() {
        delete m_data;
    }

    //copy constructor
    Tensor(const Tensor &src) {
        if (this->m_numel > 0 && this->m_data != NULL) {
            if (this->m_numel == src.numel()) {
                memcpy(this->m_data, src.m_data, sizeof(double)*this->m_numel);
            } else {
                if (this->m_data != NULL) free(this->m_data);
                this->m_numel = src.numel();
                m_data = new double[this->m_numel];
                memcpy(this->m_data, src.m_data, sizeof(double)*this->m_numel);
            }
        } else {
            this->m_numel = src.numel();
            m_data = new double[this->m_numel];
            memcpy(this->m_data, src.m_data, sizeof(double)*this->m_numel);
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

    void randu() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 1);
        for (int i = 0; i < this->m_numel; i++) {
            m_data[i] = dis(gen);
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
            dgemm_(&transa, &transb, &m, &n, &k, &alpha, this->m_data,
                   &lda, i_krp.memptr(), &ldb, &beta, o_mttkrp->memptr() , &ldc);
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
                dgemm_(&transa, &transb, &m, &n, &k, &alpha, A,
                       &lda, B , &ldb, &beta, (*o_mttkrp).memptr() , &ldc);
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
};  // class Tensor
}  // namespace planc

#endif  // NTF_TENSOR_HPP_
