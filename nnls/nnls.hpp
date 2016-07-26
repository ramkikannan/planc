/*
 * NNLS.hpp
 *
 *  Created on: Dec 11, 2013
 *      Author: ramki
 */

#ifndef NNLS_HPP_
#define NNLS_HPP_
#include "utils.hpp"

//#ifndef _VERBOSE
//#define _VERBOSE 1;
//#endif

template <class MATTYPE, class VECTYPE>
class NNLS
{
protected:
    MATTYPE CtC; //input matrix is pxq. Hence CtC is qxq.
    VECTYPE Ctb; //right hand side vector b is px1. Hence Ctb is qx1.
    MATTYPE CtB; //multiple RHS B is pxr. Hence CtB is qxr.
    UINT p, q, r; //dimension of matrix.
    VECTYPE x;//solution vector qx1;
    MATTYPE X; //solution matrix qxr;
    //If true The C matrix is CtC and b vector is Ctb.
    bool inputProd;
    bool cleared;
public:
    NNLS(MATTYPE& inputMat, VECTYPE& rhs, bool prodSent)
    {
        this->inputProd = prodSent;
        if (inputProd)
        {
            this->CtC = inputMat;
            this->Ctb = rhs;
            this->q = rhs.n_rows;
        }
        else
        {
            this->CtC = inputMat.t() * inputMat;
            this->Ctb = inputMat.t() * rhs;
            this->p = inputMat.n_rows;
            this->q = inputMat.n_cols;
        }
        this->r = 1;
        x.zeros(this->q);
#ifdef _VERBOSE
        INFO << "NNLS::Constructor with RHS vector" <<  endl;
#endif
        this->cleared = false;
    }
    NNLS(MATTYPE& inputMat, MATTYPE& RHS, bool prodSent)
    {
        this->inputProd = prodSent;
        if (this->inputProd)
        {
            this->CtC = inputMat;
            this->CtB = RHS;
            this->q = RHS.n_rows;
        }
        else
        {
            this->CtC = inputMat.t() * inputMat;
            this->CtB = inputMat.t() * RHS;
            this->p = inputMat.n_rows;
            this->q = inputMat.n_cols;
        }
        this->r = RHS.n_cols;
        X.resize(this->q, this->r);
        X.zeros();
#ifdef _VERBOSE
        INFO << "NNLS::Constructor with multiple RHS vector" << "r=" << r << endl;
#endif
        this->cleared = false;
    }
    ~NNLS()
    {
        // TODO Auto-generated destructor stub
        //clear();
    }

    virtual int solveNNLS() = 0;

    VECTYPE getSolutionVector()
    {
        return this->x;
    }
    MATTYPE getSolutionMatrix()
    {
        return this->X;
    }
    void clear()
    {
        if (!this->cleared)
        {
            this->CtC.clear();
            this->Ctb.clear();
            this->CtB.clear();
            this->x.clear();
            this->X.clear();
            this->cleared = true;
        }
    }
};
#endif /*NNLS_HPP_*/
