//  $Id: nnls.cc,v 1.1.2.1 2003/03/06 16:28:07 suvrit Exp $
// File: nnls.cc
// Implements the Lawson-Hanson NNLS algorithm

// https://software.sandia.gov/appspack/version3/nnls_8c-source.html

// Copyright (C) 2004 Lawson-Hanson
// Modifications to adapt to c++ by Suvrit Sra.

// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

#ifndef NNLS_ACTIVESETNNLS_H_
#define NNLS_ACTIVESETNNLS_H_

#include <stdio.h>
#include <math.h>
#define nnls_max(a, b) ((a) >= (b) ? (a) : (b))
#define nnls_abs(x) ((x) >= 0 ? (x) : -(x))

// has definitions of nnls, g1, h12 function
// defined them static to avoid multiple definitions
// error
#include "ActiveSetNNLS.hpp"

typedef int integer;
typedef double doublereal;

/*     SUBROUTINE NNLS  (A,MDA,M,N,B,X,RNORM,W,ZZ,INDEX,MODE) */

/*  Algorithm NNLS: NONNEGATIVE LEAST SQUARES */

/*  The original version of this code was developed by */
/*  Charles L. Lawson and Richard J. Hanson at Jet Propulsion Laboratory */
/*  1973 JUN 15, and published in the book */
/*  "SOLVING LEAST SQUARES PROBLEMS", Prentice-HalL, 1974. */
/*  Revised FEB 1995 to accompany reprinting of the book by SIAM. */

/*     GIVEN AN M BY N MATRIX, A, AND AN M-VECTOR, B,  COMPUTE AN */
/*     N-VECTOR, X, THAT SOLVES THE LEAST SQUARES PROBLEM */

/*                      A * X = B  SUBJECT TO X .GE. 0 */
/*     ------------------------------------------------------------------ */
/*                     Subroutine Arguments */

/*     A(),MDA,M,N     MDA IS THE FIRST DIMENSIONING PARAMETER FOR THE */
/*                     ARRAY, A().   ON ENTRY A() CONTAINS THE M BY N */
/*                     MATRIX, A.           ON EXIT A() CONTAINS */
/*                     THE PRODUCT MATRIX, Q*A , WHERE Q IS AN */
/*                     M BY M ORTHOGONAL MATRIX GENERATED IMPLICITLY BY */
/*                     THIS SUBROUTINE. */
/*     B()     ON ENTRY B() CONTAINS THE M-VECTOR, B.   ON EXIT B() CON- */
/*             TAINS Q*B. */
/*     X()     ON ENTRY X() NEED NOT BE INITIALIZED.  ON EXIT X() WILL */
/*             CONTAIN THE SOLUTION VECTOR. */
/*     RNORM   ON EXIT RNORM CONTAINS THE EUCLIDEAN NORM OF THE */
/*             RESIDUAL VECTOR. */
/*     W()     AN N-ARRAY OF WORKING SPACE.  ON EXIT W() WILL CONTAIN */
/*             THE DUAL SOLUTION VECTOR.   W WILL SATISFY W(I) = 0. */
/*             FOR ALL I IN SET P  AND W(I) .LE. 0. FOR ALL I IN SET Z */
/*     ZZ()     AN M-ARRAY OF WORKING SPACE. */
/*     INDEX()     AN INT WORKING ARRAY OF LENGTH AT LEAST N. */
/*                 ON EXIT THE CONTENTS OF THIS ARRAY DEFINE THE SETS */
/*                 P AND Z AS FOLLOWS.. */

/*                 INDEX(1)   THRU INDEX(NSETP) = SET P. */
/*                 INDEX(IZ1) THRU INDEX(IZ2)   = SET Z. */
/*                 IZ1 = NSETP + 1 = NPP1 */
/*                 IZ2 = N */
/*     MODE    THIS IS A SUCCESS-FAILURE FLAG WITH THE FOLLOWING */
/*             MEANINGS. */
/*             1     THE SOLUTION HAS BEEN COMPUTED SUCCESSFULLY. */
/*             2     THE DIMENSIONS OF THE PROBLEM ARE BAD. */
/*                   EITHER M .LE. 0 OR N .LE. 0. */
/*             3    ITERATION COUNT EXCEEDED.  MORE THAN 3*N ITERATIONS. */

/*     ------------------------------------------------------------------ */
/* Subroutine */
//int nnls(double* a, int mda, int m, int n,
//    double* b, double* x, double* rnorm,
//    double* w, double* zz, int* index, int* mode, int maxIter);



/*     SUBROUTINE H12 (MODE,LPIVOT,L1,M,U,IUE,UP,C,ICE,ICV,NCV) */

/*  CONSTRUCTION AND/OR APPLICATION OF A SINGLE */
/*  HOUSEHOLDER TRANSFORMATION..     Q = I + U*(U**T)/B */

/*  The original version of this code was developed by */
/*  Charles L. Lawson and Richard J. Hanson at Jet Propulsion Laboratory */
/*  1973 JUN 12, and published in the book */
/*  "SOLVING LEAST SQUARES PROBLEMS", Prentice-HalL, 1974. */
/*  Revised FEB 1995 to accompany reprinting of the book by SIAM. */
/*     ------------------------------------------------------------------ */
/*                     Subroutine Arguments */

/*     MODE   = 1 OR 2   Selects Algorithm H1 to construct and apply a */
/*            Householder transformation, or Algorithm H2 to apply a */
/*            previously constructed transformation. */
/*     LPIVOT IS THE INDEX OF THE PIVOT ELEMENT. */
/*     L1,M   IF L1 .LE. M   THE TRANSFORMATION WILL BE CONSTRUCTED TO */
/*            ZERO ELEMENTS INDEXED FROM L1 THROUGH M.   IF L1 GT. M */
/*            THE SUBROUTINE DOES AN IDENTITY TRANSFORMATION. */
/*     U(),IUE,UP    On entry with MODE = 1, U() contains the pivot */
/*            vector.  IUE is the storage increment between elements. */
/*            On exit when MODE = 1, U() and UP contain quantities */
/*            defining the vector U of the Householder transformation. */
/*            on entry with MODE = 2, U() and UP should contain */
/*            quantities previously computed with MODE = 1.  These will */
/*            not be modified during the entry with MODE = 2. */
/*     C()    ON ENTRY with MODE = 1 or 2, C() CONTAINS A MATRIX WHICH */
/*            WILL BE REGARDED AS A SET OF VECTORS TO WHICH THE */
/*            HOUSEHOLDER TRANSFORMATION IS TO BE APPLIED. */
/*            ON EXIT C() CONTAINS THE SET OF TRANSFORMED VECTORS. */
/*     ICE    STORAGE INCREMENT BETWEEN ELEMENTS OF VECTORS IN C(). */
/*     ICV    STORAGE INCREMENT BETWEEN VECTORS IN C(). */
/*     NCV    NUMBER OF VECTORS IN C() TO BE TRANSFORMED. IF NCV .LE. 0 */
/*            NO OPERATIONS WILL BE DONE ON C(). */
/*     ------------------------------------------------------------------ */
/* Subroutine */
//int h12(int mode, int* lpivot, int* l1,
//   int m, double* u, int* iue, double* up, double* c__,
//   int* ice, int* icv, int* ncv);


/*     COMPUTE ORTHOGONAL ROTATION MATRIX.. */

/*  The original version of this code was developed by */
/*  Charles L. Lawson and Richard J. Hanson at Jet Propulsion Laboratory
 */
/*  1973 JUN 12, and published in the book */
/*  "SOLVING LEAST SQUARES PROBLEMS", Prentice-HalL, 1974. */
/*  Revised FEB 1995 to accompany reprinting of the book by SIAM. */

/*     COMPUTE.. MATRIX   (C, S) SO THAT (C, S)(A) = (SQRT(A**2+B**2)) */
/*                        (-S,C)         (-S,C)(B)   (   0          ) */
/*     COMPUTE SIG = SQRT(A**2+B**2) */
/*        SIG IS COMPUTED LAST TO ALLOW FOR THE POSSIBILITY THAT */
/*        SIG MAY BE IN THE SAME LOCATION AS A OR B . */
//int g1(double* a, double* b, double* cterm, double* sterm, double* sig);
//
// June 18, 2008 -- Added a helper class to the bottom - Ted Kim
//
// Included string for support of memcpy
#include <string.h>
template<class T>
class ActiveSetNNLS {
  public:
    ActiveSetNNLS(int rows, int maxCols) :
        _rows( rows ),
        _maxCols( maxCols ) {
        _workA     = new T[_rows * _maxCols];
        _workW     = new T[_maxCols];
        _workZ     = new T[_rows];
        _workIndex = new int[_maxCols];

        _maxIter = 1;
    }

    ~ActiveSetNNLS() {
        delete[] _workA;
        delete[] _workW;
        delete[] _workZ;
        delete[] _workIndex;
    }

    bool solve(T* A, int numCols, T* b, T* x, T& rNorm) {
        int workMode;

        // make copy of A, since nnls() clobbers it
        memcpy(_workA, A, numCols * _rows * sizeof(T));

        // do it
        nnls(_workA, _rows, _rows, numCols, b, x, &rNorm,
             _workW, _workZ, _workIndex, &workMode, _maxIter);

        return (workMode == 1);
    }

    int& maxIter() { return _maxIter;}

    T* getDual() {return _workW;}

  private:
    int _rows;
    int _maxCols;

    T* _workA;
    T* _workW;
    T* _workZ;
    int* _workIndex;

    // maximum iterations of NNLS
    int _maxIter;
};
#endif  // NNLS_ACTIVESETNNLS_H_
