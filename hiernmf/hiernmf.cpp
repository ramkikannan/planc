/* Copyright 2020 Lawton Manning */
#include <fstream>
#include <queue>
#include <string>
#include <vector>
#include <cinttypes>
#include "common/distutils.hpp"
#include "common/parsecommandline.hpp"
#include "common/utils.hpp"
#include "distnmf/distio.hpp"
#include "distnmf/distr2.hpp"
#include "distnmf/mpicomm.hpp"
#include "hiernmf/node.hpp"

namespace planc {

class HierNMFDriver {
 private:
  int m_argc;
  char **m_argv;
  int m_k;
  UWORD m_globalm, m_globaln;
  std::string m_Afile_name;
  std::string m_outputfile_name;
  int m_num_it;
  int m_pr;
  int m_pc = 1;
  FVEC m_regW;
  FVEC m_regH;
  double m_sparsity;
  iodistributions m_distio;
  uint m_compute_error;
  int m_num_k_blocks;
  static const int kprimeoffset = 17;
  normtype m_input_normalization;
  MPICommunicator *mpicomm;
  ParseCommandLine *pc;

#ifdef BUILD_SPARSE
  RootNode<SP_MAT> *root;
#else
  RootNode<MAT> *root;
#endif

  void parseCommandLine() {
    pc = new ParseCommandLine(this->m_argc, this->m_argv);
    pc->parseplancopts();
    this->m_k = pc->lowrankk();
    this->m_Afile_name = pc->input_file_name();
    this->m_pr = pc->pr();
    this->m_pc = 1;
    this->m_sparsity = pc->sparsity();
    this->m_num_it = pc->iterations();
    this->m_distio = TWOD;
    this->m_regW = pc->regW();
    this->m_regH = pc->regH();
    this->m_num_k_blocks = 1;
    this->m_globalm = pc->globalm();
    this->m_globaln = pc->globaln();
    this->m_compute_error = pc->compute_error();
    this->m_outputfile_name = pc->output_file_name();
    pc->printConfig();
  }

  void buildTree() {
    std::string rand_prefix("rand_");
    this->mpicomm =
        new MPICommunicator(this->m_argc, this->m_argv, this->m_pr, this->m_pc);

#ifdef BUILD_SPARSE
    SP_MAT A;
    DistIO<SP_MAT> dio(*mpicomm, m_distio, A);
#else
    MAT A;
    DistIO<MAT> dio(*mpicomm, m_distio, A);
#endif

    if (m_Afile_name.compare(0, rand_prefix.size(), rand_prefix) == 0) {
      dio.readInput(m_Afile_name, this->m_globalm, this->m_globaln, this->m_k,
                    this->m_sparsity, this->m_pr, this->m_pc, 0, false, NONE);
    } else {
      dio.readInput(m_Afile_name, this->m_globalm, this->m_globaln, this->m_k,
                    this->m_sparsity, this->m_pr, this->m_pc, 0, false, NONE);
    }
    A = dio.A();

    MPI_Allreduce(&A.n_rows, &this->m_globalm, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, this->mpicomm->commSubs()[0]);
    MPI_Allreduce(&A.n_cols, &this->m_globaln, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, this->mpicomm->commSubs()[1]);

    int rank = this->mpicomm->rank();
    arma::uvec cols(this->m_globaln);
    for (unsigned int i = 0; i < this->m_globaln; i++) {
      cols[i] = i;
    }

#ifdef BUILD_SPARSE
    std::priority_queue<Node<SP_MAT> *, std::vector<Node<SP_MAT> *>,
                        ScoreCompare>
        frontiers;
    Node<SP_MAT> *frontier;
    Node<SP_MAT> *node;

    std::queue<Node<SP_MAT> *> nodes;
#else
    std::priority_queue<Node<MAT> *, std::vector<Node<MAT> *>, ScoreCompare>
        frontiers;
    Node<MAT> *frontier;
    Node<MAT> *node;

    std::queue<Node<MAT> *> nodes;
#endif

    mpitic();

    MPI_Barrier(MPI_COMM_WORLD);
    memusage(mpicomm->rank(), "before root initialization");

#ifdef BUILD_SPARSE
    this->root = new RootNode<SP_MAT>(&A, this->m_globalm, this->m_globaln,
                                      cols, this->mpicomm, this->pc);
#else
    this->root = new RootNode<MAT>(&A, this->m_globalm, this->m_globaln, cols,
                                   this->mpicomm, this->pc);
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    memusage(mpicomm->rank(), "after root initialization");

    nodes.push(root);

    this->root->split();
    this->root->accept();
    this->root->enqueue_children(&frontiers);
    this->root->enqueue_children(&nodes);

    for (int i = 2; i < pc->nodes(); i++) {
      if (frontiers.empty()) {
        break;
      }
      frontier = frontiers.top();
      frontiers.pop();
      frontier->accept();
      frontier->enqueue_children(&frontiers);
      frontier->enqueue_children(&nodes);
    }

    double temp = mpitoc();
    if (this->mpicomm->rank() == 0) {
      printf("H2NMF took %.3lf secs.\n", temp);
    }

    if (this->mpicomm->rank() == 0) {
      printf(
          "idx level m n NMF MATVEC(SIGMA) VECMAT(SIGMA) COMM(SIGMA) "
          "NORM(SIGMA)\n");
    }
    while (!nodes.empty()) {
      node = nodes.front();
      nodes.pop();
      if (this->mpicomm->rank() == 0) {
        printf("%" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64
                " %f %f %f %f %f\n",
              node->index, node->level,
              node->global_m, node->global_n, node->timings.total,
              node->timings.sigma.matvec, node->timings.sigma.vecmat,
              node->timings.sigma.communication,
              node->timings.sigma.normalisation);
      }
      if (node->index == 0) {
        continue;
      }
      std::stringstream s;
      s << this->m_outputfile_name << "_" << node->index;
      dio.writeOutput(node->W, node->H, s.str());
      if (this->mpicomm->rank() == 0) {
        s << "_cols";
        node->cols.save(s.str(), arma::raw_ascii);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    delete this->mpicomm;
  }

 public:
  HierNMFDriver(int argc, char *argv[]) {
    this->m_argc = argc;
    this->m_argv = argv;
    this->parseCommandLine();
    this->buildTree();
  }
};

}  // namespace planc

int main(int argc, char *argv[]) {
  try {
    planc::HierNMFDriver hd(argc, argv);
    fflush(stdout);
  } catch (const std::exception &e) {
    INFO << "Exception with stack trace " << std::endl;
    INFO << e.what();
  }
}
