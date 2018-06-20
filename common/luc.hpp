/* Copyright Ramakrishnan Kannan 2017 */
#ifndef COMMON_LUC_HPP_
#define COMMON_LUC_HPP_

#include "nnls/bppnnls.hpp"
#include "nnls/nnls.hpp"

namespace planc {
class LUC {
 private:
  // create variable required for your local function.
  MAT updateBPP(const MAT &gram, const MAT &rhs) {
    MAT nnlsgram;
    MAT nnlsrhs;
    BPPNNLS<MAT, VEC> subProblem(gram, rhs, true);
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

}  // namespace planc
#endif  // COMMON_LUC_HPP_
