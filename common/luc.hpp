/* Copyright Ramakrishnan Kannan 2017 */
#ifndef NTF_LUC_HPP_
#define NTF_LUC_HPP_

#include "ntf_utils.h"
#include "nnls.hpp"
#include "bppnnls.hpp"

namespace planc {
class LUC {
  private:
    // create variable required for your local function.
    MAT updateBPP(const MAT &gram, const MAT &rhs) {
        MAT nnlsgram;
        MAT nnlsrhs;
        BPPNNLS<MAT, VEC > subProblem(gram, rhs, true);
        subProblem.solveNNLS();
        return subProblem.getSolutionMatrix();
    }
  public:
    MAT update(algotype m_updalgo, const MAT &gram, const MAT &rhs) {
        MAT rc;
        switch (m_updalgo) {
        case ANLSBPP:
            rc = updateBPP(gram, rhs);
            break;
        default:
            std::cout << "wrong algo type" << std::endl;
        }
        return rc;
    }
};  // LUC

}  // planc
#endif  // NTF_LUC_HPP_
