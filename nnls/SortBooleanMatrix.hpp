#include <armadillo>
#include <vector>
#include <algorithm>

using namespace std;

template <class T>
class BooleanArrayComparator
{
        const T &X;
public:
        BooleanArrayComparator(const T &input): X(input)
        {
        }
        /*
        * if idxi < idxj return true;
        * if idxi >= idxj return false;
        */
        bool operator () (uword idxi, uword idxj)
        {
                for (uint i = 0; i < X.n_rows; i++)
                {
                        if (this->X(i, idxi) < this->X(i, idxj))
                                return true;
                        else if (this->X(i, idxi) > this->X(i, idxj))
                                return false;
                }
                return false;
        }
};

template <class T>
class SortBooleanMatrix
{
        const T &X;
        vector<uword> idxs;
public:
        SortBooleanMatrix(const T &input) : X(input), idxs(X.n_cols)
        {
                for (uint i = 0; i < X.n_cols; i++)
                {
                        idxs[i] = i;
                }
        }
        vector<uword> sortIndex()
        {
                sort(this->idxs.begin(), this->idxs.end(), BooleanArrayComparator<T>(this->X));
                return this->idxs;
        }

};
