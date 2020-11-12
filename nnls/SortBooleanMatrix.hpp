/* Copyright 2016 Ramakrishnan Kannan */

#ifndef NNLS_SORTBOOLEANMATRIX_HPP_
#define NNLS_SORTBOOLEANMATRIX_HPP_
#include <armadillo>
#include <vector>
#include <algorithm>

template <class T>
class BooleanArrayComparator {
    const T &X;
 public:
    explicit BooleanArrayComparator(const T &input): X(input) {
    }
    /*
    * if idxi < idxj return true;
    * if idxi >= idxj return false;
    */
    bool operator() (UWORD idxi, UWORD idxj) {
        for (uint i = 0; i < X.n_rows; i++) {
            if (this->X(i, idxi) < this->X(i, idxj))
                return true;
            else if (this->X(i, idxi) > this->X(i, idxj))
                return false;
        }
        return false;
    }
};

template <class T>
class SortBooleanMatrix {
    const T &X;
    std::vector<UWORD> idxs;
 public:
    explicit SortBooleanMatrix(const T &input) : X(input), idxs(X.n_cols) {
        for (uint i = 0; i < X.n_cols; i++) {
            idxs[i] = i;
        }
    }
    std::vector<UWORD> sortIndex() {
        std::sort(this->idxs.begin(), this->idxs.end(),
             BooleanArrayComparator<T>(this->X));
        return this->idxs;
    }
};
#endif  // NNLS_SORTBOOLEANMATRIX_HPP_
