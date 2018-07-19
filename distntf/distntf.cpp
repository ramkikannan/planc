/* Copyright 2016 Ramakrishnan Kannan */

#include <string>
#include "common/distutils.hpp"
#include "common/parsecommandline.hpp"
#include "common/utils.hpp"
#include "distntf/distauntf.hpp"
#include "distntf/distntfio.hpp"
#include "distntf/distntfmpicomm.hpp"

class DistNTF {
 private:
  int m_argc;
  char **m_argv;
  int m_k;
  std::string m_Afile_name;
  std::string m_outputfile_name;
  int m_num_it;
  UVEC m_proc_grids;
  FVEC m_regs;
  algotype m_ntfalgo;
  double m_sparsity;
  uint m_compute_error;
  int m_num_k_blocks;
  UVEC m_global_dims;
  UVEC m_factor_local_dims;
  bool m_enable_dim_tree;
  static const int kprimeoffset = 17;

  void printConfig() {
    std::cout << "a::" << this->m_ntfalgo << "::i::" << this->m_Afile_name
              << "::k::" << this->m_k << "::dims::" << this->m_global_dims
              << "::t::" << this->m_num_it
              << "::proc_grids::" << this->m_proc_grids
              << "::error::" << this->m_compute_error
              << "::regs::" << this->m_regs
              << "::num_k_blocks::" << m_num_k_blocks
              << "::dim_tree::" << m_enable_dim_tree << std::endl;
  }

  void callDistNTF() {
    std::string rand_prefix("rand_");
    planc::NTFMPICommunicator mpicomm(this->m_argc, this->m_argv,
                                      this->m_proc_grids);
    mpicomm.printConfig();
    planc::DistNTFIO dio(mpicomm);
    if (m_Afile_name.compare(0, rand_prefix.size(), rand_prefix) == 0) {
      dio.readInput(m_Afile_name, this->m_global_dims, this->m_proc_grids,
                    this->m_k, this->m_sparsity);
    } else {
      dio.readInput(m_Afile_name);
    }
    // planc::Tensor A(dio.A()->dimensions(), dio.A()->m_data);
    planc::Tensor A;
    A = dio.A();
    if (m_Afile_name.compare(0, rand_prefix.size(), rand_prefix) != 0) {
      UVEC local_dims = A.dimensions();
      /*MPI_Allreduce(&localm, &(this->m_globalm), 1, MPI_INT,
       *            MPI_SUM, mpicomm.commSubs()[0]);
       * MPI_Allreduce(&localn, &(this->m_globaln), 1, MPI_INT,
       *            MPI_SUM, mpicomm.commSubs()[1]);*/
      this->m_global_dims = local_dims % this->m_proc_grids;
    }
    INFO << mpicomm.rank()
         << "::Completed generating tensor A=" << A.dimensions()
         << "::global dims::" << this->m_global_dims << std::endl;
#ifdef DISTNTF_VERBOSE
    A.print();
#endif
#ifdef WRITE_RAND_INPUT
    dio.writeRandInput();
#endif  // ifdef WRITE_RAND_INPUT
        // same matrix as only one of them will be used.

    // sometimes for really very large matrices starting w/
    // rand initialization hurts ANLS BPP running time. For a better
    // initializer we run couple of iterations of HALS.
#ifdef MPI_VERBOSE
    INFO << mpicomm.rank() << "::" << __PRETTY_FUNCTION__
         << "::" << factors.printinfo() << std::endl;
#endif  // ifdef MPI_VERBOSE
    MPI_Barrier(MPI_COMM_WORLD);
    memusage(mpicomm.rank(), "b4 constructor ");
    // TODO(ramkikannan): I was here. Need to modify the reallocations by
    // using localOwnedRowCount instead of m_globalm.
    // DistAUNTF(const Tensor &i_tensor, const int i_k, algotype i_algo,
    //   const UVEC &i_global_dims,
    //   const UVEC &i_local_dims,
    //   const NTFMPICommunicator &i_mpicomm)
    this->m_factor_local_dims = this->m_global_dims / mpicomm.size();
    planc::DistAUNTF ntfsolver(A, this->m_k, this->m_ntfalgo,
                               this->m_global_dims, this->m_factor_local_dims,
                               mpicomm);
    memusage(mpicomm.rank(), "after constructor ");
    ntfsolver.num_iterations(this->m_num_it);
    ntfsolver.compute_error(this->m_compute_error);
    if (this->m_enable_dim_tree) {
      ntfsolver.dim_tree(this->m_enable_dim_tree);
    }
    ntfsolver.regularizers(this->m_regs);
    MPI_Barrier(MPI_COMM_WORLD);
    // try {
    mpitic();
    ntfsolver.computeNTF();
    double temp = mpitoc();

    if (mpicomm.rank() == 0) printf("NTF took %.3lf secs.\n", temp);
    // } catch (std::exception& e) {
    //     printf("Failed rank %d: %s\n", mpicomm.rank(), e.what());
    //     MPI_Abort(MPI_COMM_WORLD, 1);
    // }
  }

  void parseCommandLine() {
    planc::ParseCommandLine pc(this->m_argc, this->m_argv);
    pc.parseplancopts();
    this->m_ntfalgo = pc.lucalgo();
    this->m_k = pc.lowrankk();
    this->m_Afile_name = pc.input_file_name();
    this->m_proc_grids = pc.processor_grids();
    this->m_sparsity = pc.sparsity();
    this->m_num_it = pc.iterations();
    this->m_num_k_blocks = 1;
    this->m_regs = pc.regularizers();
    this->m_global_dims = pc.dimensions();
    this->m_compute_error = pc.compute_error();
    this->m_enable_dim_tree = pc.dim_tree();
    printConfig();
    callDistNTF();
  }

 public:
  DistNTF(int argc, char *argv[]) {
    this->m_argc = argc;
    this->m_argv = argv;
    this->parseCommandLine();
  }
};

int main(int argc, char *argv[]) {
  DistNTF dnd(argc, argv);
  fflush(stdout);
}
