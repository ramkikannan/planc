/* Copyright 2020 Lawton Manning */
#ifndef HIERNMF_NODE_HPP_
#define HIERNMF_NODE_HPP_
#include <queue>
#include <vector>
#include "common/parsecommandline.hpp"
#include "distnmf/distnmftime.hpp"
#include "distnmf/distr2.hpp"
#include "distnmf/mpicomm.hpp"
#include "hiernmf/matutils.hpp"

namespace planc {

struct NodeTimings {
  PowerTimings sigma;
  DistNMFTime * nmf;
  double total;
};
template <class INPUTMATTYPE>
class Node {
 public:
  Node *lchild = NULL;
  Node *rchild = NULL;
  bool lvalid, rvalid;
  Node *parent = NULL;
  const INPUTMATTYPE * A0;
  INPUTMATTYPE A;
  VEC W;
  VEC H;
  double sigma;
  double score;
  UVEC cols;
  bool accepted = false;
  uint64_t index;
  uint64_t level;

  uint64_t global_m, global_n;

  UVEC top_words;

  MPICommunicator *mpicomm;
  ParseCommandLine *pc;

  NodeTimings timings;

  void allocate() {
#ifdef BUILD_SPARSE
    int n_cols = this->cols.n_elem;

    arma::umat locs(2, n_cols);
    for (int i = 0; i < n_cols; i++) {
      locs(1, i) = i;
      locs(0, i) = this->cols(i);
    }
    VEC vals(n_cols);
    vals.fill(1);

    SP_MAT S(locs, vals, this->A0->n_cols, n_cols);

    this->A = *this->A0 * S;
#else
    this->A = this->A0->cols(this->cols);
#endif
  }

  void compute_sigma() {
    this->sigma = powIter(this->A, this->pc->iterations(),
                          this->pc->tolerance(), &this->timings.sigma);
  }

  void compute_score() {
    if (this->lvalid && this->rvalid) {
      this->score = (this->lchild->sigma + this->rchild->sigma) - this->sigma;
    } else if (this->lvalid) {
      this->score = this->lchild->sigma - this->sigma;
    } else if (this->rvalid) {
      this->score = this->rchild->sigma - this->sigma;
    } else {
      this->score = 0;
    }
  }

  Node() {}

  Node(const INPUTMATTYPE * A, VEC W, VEC H, const UVEC & cols, Node * parent,
       uint64_t index) {
    this->cols = cols;
    this->A0 = A;
    this->global_m = parent->global_m;
    this->global_n = cols.n_elem;
    this->W = W;
    this->H = H;
    this->parent = parent;
    this->index = index;
    this->level = parent->level + 1;
    this->mpicomm = parent->mpicomm;
    this->pc = parent->pc;
    this->allocate();
    this->compute_sigma();
    this->lvalid = false;
    this->rvalid = false;
  }

  void split() {
    this->accepted = true;

    /*
    arma::arma_rng::set_seed(pc->initseed()+this->index);
    MAT gW = arma::randu<MAT>(A.n_rows, 2);
    MAT gH = arma::randu<MAT>(A.n_cols, 2);

    int startW = startidx(A.n_rows, mpicomm->pc(), mpicomm->col_rank());
    int stopW = startW +
                itersplit(A.n_rows, mpicomm->pc(), mpicomm->col_rank()) - 1;
    MAT W = gW.rows(startW, stopW);

    int startH = startidx(A.n_cols, mpicomm->pr(), mpicomm->row_rank());
    int stopH = startH +
                itersplit(A.n_cols, mpicomm->pr(), mpicomm->row_rank()) - 1;
    MAT H = gH.rows(startH, stopH);
    */

    arma::arma_rng::set_seed(pc->initseed()*this->index + mpicomm->rank());
    MAT W = arma::randu<MAT>(
        itersplit(A.n_rows, mpicomm->pc(), mpicomm->col_rank()), 2);
    MAT H = arma::randu<MAT>(
        itersplit(A.n_cols, mpicomm->pr(), mpicomm->row_rank()), 2);


    DistR2<INPUTMATTYPE> nmf(A, W, H, *mpicomm, 1);
    nmf.symm_reg(pc->symm_reg());
    nmf.num_iterations(pc->iterations());
    nmf.compute_error(pc->compute_error());
    nmf.tolerance(pc->tolerance());
    nmf.algorithm(R2);
    nmf.regW(pc->regW());
    nmf.regH(pc->regH());

    mpitic();
    nmf.computeNMF();
    timings.total = mpitoc();
    timings.nmf = nmf.times();

    W = nmf.getLeftLowRankFactor();
    H = nmf.getRightLowRankFactor();

    UVEC lleft = H.col(0) >= H.col(1);
    UVEC left(A.n_cols, arma::fill::zeros);
    int * recvcnts = new int[this->mpicomm->size()];
    int * displs = new int[this->mpicomm->size()];
    recvcnts[0] = itersplit(A.n_cols, pc->pr(), 0);
    displs[0] = 0;
    for (int i = 1; i < this->mpicomm->size(); i++) {
      recvcnts[i] = itersplit(A.n_cols, pc->pr(), i);
      displs[i] = displs[i - 1] + recvcnts[i - 1];
    }
    MPI_Allgatherv(lleft.memptr(), lleft.n_elem, MPI_UNSIGNED_LONG_LONG,
                   left.memptr(), recvcnts, displs, MPI_UNSIGNED_LONG_LONG,
                   MPI_COMM_WORLD);
    delete recvcnts;
    delete displs;

    int lidx = 0;
    UVEC lcols = this->cols(find(left == 1));
    UVEC rcols = this->cols(find(left == 0));

    if (rcols.n_elem > lcols.n_elem) {
      lcols = this->cols(find(left == 0));
      rcols = this->cols(find(left == 1));
      lidx = 1;
    }

    this->lvalid = !lcols.is_empty();
    this->rvalid = !rcols.is_empty();

    A.clear();

    if (this->lvalid) {
      this->lchild = new Node(this->A0, W.col(lidx), H.col(lidx), lcols, this,
                              2 * this->index + 1);
    }
    if (this->rvalid) {
      this->rchild = new Node(this->A0, W.col(1 - lidx), H.col(1 - lidx), rcols,
                              this, 2 * this->index + 2);
    }

    this->compute_score();
  }

  void accept() {
    if (this->lvalid) {
      this->lchild->split();
    }
    if (this->rvalid) {
      this->rchild->split();
    }
  }

  template <class QUEUE>
  void enqueue_children(QUEUE * queue) {
    if (this->lvalid) {
      queue->push(this->lchild);
    }
    if (this->rvalid) {
      queue->push(this->rchild);
    }
  }

  bool operator>(const Node<INPUTMATTYPE> &rhs) const {
    return (this->score > rhs.score);
  }

  bool operator<(const Node<INPUTMATTYPE> &rhs) const {
    return (this->score < rhs.score);
  }
};

class ScoreCompare {
 public:
  template <typename T>
  bool operator()(T *a, T *b) {
    return a->score < b->score;
  }
};

template <class INPUTMATTYPE>
class RootNode : public Node<INPUTMATTYPE> {
 public:
  RootNode(const INPUTMATTYPE * A, uint64_t global_m,
           uint64_t global_n, const UVEC & cols, MPICommunicator * mpicomm,
           ParseCommandLine * pc)
      : Node<INPUTMATTYPE>() {
    this->cols = cols;
    this->A0 = A;
    this->global_m = global_m;
    this->global_n = global_n;
    this->parent = NULL;
    this->mpicomm = mpicomm;
    this->pc = pc;
    this->A = INPUTMATTYPE(*A);
    this->sigma = 0.0;
    this->index = 0;
    this->level = 0;
    this->lvalid = false;
    this->rvalid = false;
  }
};
}  // namespace planc
#endif  // HIERNMF_NODE_HPP_
