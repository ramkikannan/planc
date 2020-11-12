/* Copyright 2016 Ramakrishnan Kannan */

#ifndef NNLS_NNLS_HPP_
#define NNLS_NNLS_HPP_
#include "utils.hpp"

// #ifndef _VERBOSE
// #define _VERBOSE 1;
// #endif

template <class MATTYPE, class VECTYPE>
class NNLS {
 protected:
    MATTYPE AtA;   // input matrix is mxn. Hence AtA is nxn.
    VECTYPE Atb;   // right hand side vector b is nx1. Hence Atb is nx1.
    MATTYPE AtB;   // multiple RHS B is mxk. Hence AtB is nxk.
    UINT m, n, k;  // dimension of matrix.
    VECTYPE x;  // solution vector nx1;
    MATTYPE X;  // solution matrix nxk;
    // If true The A matrix is AtA and b vector is Atb.
    bool inputProd;
    bool cleared;

 public:
  /**
   * Public constructor for NNLS solver via Normal Equations with single RHS. 
   * Base class for solving,
   * \f$|| AX - b ||_F^2\f$ with \f$X >= 0\f$ via \f$A^TA X = A^Tb\f$.
   * 
   * @param[in] lhs of the normal equation. Sent as either A of size \f$m \times n\f$
   *            or AtA of size \f$n \times n\f$ depending on prodSent
   * @param[in] rhs of the normal equation for a single RHS. Sent as either
   *            b of size \f$m \times 1\f$ or Atb of size \f$n \times 1\f$
   * @param[in] Boolean signifying if AtA and Atb are sent
   */
    NNLS(const MATTYPE& inputMat, const VECTYPE& rhs, bool prodSent) {
        this->inputProd = prodSent;
        if (inputProd) {
            this->AtA = inputMat;
            this->Atb = rhs;
            this->n = rhs.n_rows;
        } else {
            this->AtA = inputMat.t() * inputMat;
            this->Atb = inputMat.t() * rhs;
            this->m = inputMat.n_rows;
            this->n = inputMat.n_cols;
        }
        this->k = 1;
        x.zeros(this->n);
#ifdef _VERBOSE
        INFO << "NNLS::Constructor with RHS vector" <<  endl;
#endif
        this->cleared = false;
    }

/**
   * Public constructor for NNLS solver via Normal Equations with single RHS. 
   * Base class for solving,
   * \f$|| AX - b ||_F^2\f$ with \f$X >= 0\f$ via \f$A^TA X = A^Tb\f$.
   * 
   * @param[in] lhs of the normal equation. Sent as either A of size \f$m \times n\f$
   *            or AtA of size \f$n \times n\f$ depending on prodSent
   * @param[in] rhs of the normal equation for a single RHS. Sent as either
   *            b of size \f$m \times 1\f$ or Atb of size \f$n \times 1\f$
   * @param[in] initial value for x of size \f$n\f$
   * @param[in] Boolean signifying if AtA and Atb are sent
   */
    NNLS(const MATTYPE& inputMat, const VECTYPE& rhs, const VECTYPE& initx,
            bool prodSent) {
        this->inputProd = prodSent;
        if (inputProd) {
            this->AtA = inputMat;
            this->Atb = rhs;
            this->n = rhs.n_rows;
        } else {
            this->AtA = inputMat.t() * inputMat;
            this->Atb = inputMat.t() * rhs;
            this->m = inputMat.n_rows;
            this->n = inputMat.n_cols;
        }
        this->k = 1;
        x = initx;
#ifdef _VERBOSE
        INFO << "NNLS::Constructor with RHS vector" <<  endl;
#endif
        this->cleared = false;
    }

  /**
   * Public constructor for NNLS solver via Normal Equations with multiple RHS. 
   * Base class for solving,
   * \f$|| AX - B ||_F^2\f$ with \f$X >= 0\f$ via \f$A^TA X = A^TB\f$.
   * 
   * @param[in] lhs of the normal equation. Sent as either A of size \f$m \times n\f$
   *            or AtA of size \f$n \times n\f$ depending on prodSent
   * @param[in] rhs of the normal equation for multiple RHS. Sent as either
   *            B of size \f$m \times k\f$ or AtB of size \f$n \times k\f$
   * @param[in] Boolean signifying if AtA and Atb are sent
   */
    NNLS(const MATTYPE& inputMat, const MATTYPE& RHS, bool prodSent) {
        this->inputProd = prodSent;
        if (this->inputProd) {
            this->AtA = inputMat;
            // bug raised by oguz
            if (RHS.n_cols == 1) {
                // user has called RHS as mattype instead
                // of vec type. Take just the first col
                // of AtB in this case.
                this->Atb = RHS.col(0);
            } else {
                this->AtB = RHS;
            }
            this->n = RHS.n_rows;
        } else {
            this->AtA = inputMat.t() * inputMat;
            this->AtB = inputMat.t() * RHS;
            this->m = inputMat.n_rows;
            this->n = inputMat.n_cols;
        }
        this->k = RHS.n_cols;
        X.resize(this->n, this->k);
        X.zeros();
#ifdef _VERBOSE
        INFO << "NNLS::Constructor with multiple RHS vector"
             << "k=" << k << std::endl;
#endif
        this->cleared = false;
    }

  /**
   * Public constructor for NNLS solver via Normal Equations with multiple RHS. 
   * Base class for solving,
   * \f$|| AX - B ||_F^2\f$ with \f$X >= 0\f$ via \f$A^TA X = A^TB\f$.
   * 
   * @param[in] lhs of the normal equation. Sent as either A of size \f$m \times n\f$
   *            or AtA of size \f$n \times n\f$ depending on prodSent
   * @param[in] rhs of the normal equation for multiple RHS. Sent as either
   *            B of size \f$m \times k\f$ or AtB of size \f$n \times k\f$
   * @param[in] initial value for X of size \f$n \times k\f$
   * @param[in] Boolean signifying if AtA and Atb are sent
   */
    NNLS(const MATTYPE& inputMat, const MATTYPE& RHS, const MATTYPE& initX,
            bool prodSent) {
        this->inputProd = prodSent;
        if (this->inputProd) {
            this->AtA = inputMat;
            // bug raised by oguz
            if (RHS.n_cols == 1) {
                // user has called RHS as mattype instead
                // of vec type. Take just the first col
                // of AtB in this case.
                this->Atb = RHS.col(0);
            } else {
                this->AtB = RHS;
            }
            this->n = RHS.n_rows;
        } else {
            this->AtA = inputMat.t() * inputMat;
            this->AtB = inputMat.t() * RHS;
            this->m = inputMat.n_rows;
            this->n = inputMat.n_cols;
        }
        this->k = RHS.n_cols;
        X.resize(this->n, this->k);
        X = initX;
#ifdef _VERBOSE
        INFO << "NNLS::Constructor with multiple RHS vector"
             << "k=" << k << std::endl;
#endif
        this->cleared = false;
    }

    ~NNLS() {
    }

    virtual int solveNNLS() = 0;

    VECTYPE getSolutionVector() {
        return this->x;
    }
    MATTYPE getSolutionMatrix() {
        return this->X;
    }
    void clear() {
        if (!this->cleared) {
            this->AtA.clear();
            this->Atb.clear();
            this->AtB.clear();
            this->x.clear();
            this->X.clear();
            this->cleared = true;
        }
    }
};
#endif  // NNLS_NNLS_HPP_
