/* Copyright 2016 Ramakrishnan Kannan */

#include <string>
#include "common/distutils.hpp"
#include "common/parsecommandline.hpp"
#include "common/utils.hpp"
#include "distnmf/distals.hpp"
#include "distnmf/distanlsbpp.hpp"
#include "distnmf/distaoadmm.hpp"
#include "distnmf/disthals.hpp"
#include "distnmf/distio.hpp"
#include "distnmf/distmu.hpp"
#include "distnmf/mpicomm.hpp"
#include "distnmf/naiveanlsbpp.hpp"
#include "distnmf/distgnsymnmf.hpp"
#include "distnmf/distr2.hpp"
#ifdef BUILD_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace planc {

class DistNMFDriver {
 private:
  int m_argc;
  char **m_argv;
  int m_k;
  UWORD m_globalm, m_globaln;
  std::string m_Afile_name;
  std::string m_outputfile_name;
  int m_num_it;
  int m_pr;
  int m_pc;
  FVEC m_regW;
  FVEC m_regH;
  double m_symm_reg;
  int m_symm_flag;
  bool m_adj_rand;
  algotype m_nmfalgo;
  double m_sparsity;
  iodistributions m_distio;
  uint m_compute_error;
  double m_tolerance;
  int m_num_k_blocks;
  static const int kprimeoffset = 17;
  normtype m_input_normalization;
  int m_max_luciters;
  int m_initseed;

#ifdef BUILD_CUDA
  void printDevProp(cudaDeviceProp devProp) {
    printf("Major revision number: %d\n", devProp.major);
    printf("Minor revision number: %d\n", devProp.minor);
    printf("Name: %s\n", devProp.name);
    printf("Total global memory: %u\n", devProp.totalGlobalMem);
    printf("Total shared memory per block: %u\n", devProp.sharedMemPerBlock);
    printf("Total registers per block: %d\n", devProp.regsPerBlock);
    printf("Warp size: %d\n", devProp.warpSize);
    printf("Maximum memory pitch: %u\n", devProp.memPitch);
    printf("Maximum threads per block: %d\n", devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
      printf("Maximum dimension %d of block: %d\n", i,
             devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
      printf("Maximum dimension %d of grid: %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate: %d\n", devProp.clockRate);
    printf("Total constant memory: %u\n", devProp.totalConstMem);
    printf("Texture alignment: %u\n", devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",
           (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors: %d\n", devProp.multiProcessorCount);
    printf("Kernel execution timeout: %s\n",
           (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    return;
  }
  void gpuQuery() {
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("CUDA Device Query...\n");
    printf("There are %d CUDA devices.\n", devCount);
    // Iterate through devices
    for (int i = 0; i < devCount; ++i) {
      // Get device properties
      printf("\nCUDA Device #%d\n", i);
      cudaDeviceProp devProp;
      cudaGetDeviceProperties(&devProp, i);
      printDevProp(devProp);
    }
  }
#endif

  void printConfig() {
    INFO << "a::" << this->m_nmfalgo << "::i::" << this->m_Afile_name
         << "::k::" << this->m_k << "::m::" << this->m_globalm
         << "::n::" << this->m_globaln << "::t::" << this->m_num_it
         << "::pr::" << this->m_pr << "::pc::" << this->m_pc
         << "::error::" << this->m_compute_error
         << "::distio::" << this->m_distio << "::regW::"
         << "l2::" << this->m_regW(0) << "::l1::" << this->m_regW(1)
         << "::regH::"
         << "l2::" << this->m_regH(0) << "::l1::" << this->m_regH(1)
         << "symm_reg::" << this->m_symm_reg
         << "symm_flag::" << this->m_symm_flag
         << "::num_k_blocks::" << this->m_num_k_blocks
         << "::normtype::" << this->m_input_normalization << std::endl;
  }

  template <class NMFTYPE>
  void callDistNMF1D() {
    std::string rand_prefix("rand_");
    MPICommunicator mpicomm(this->m_argc, this->m_argv);
#ifdef BUILD_SPARSE
    SP_MAT A;
    DistIO<SP_MAT> dio(mpicomm, m_distio, A);
#else   // ifdef BUILD_SPARSE
    MAT A;
    DistIO<MAT> dio(mpicomm, m_distio, A);
#endif  // ifdef BUILD_SPARSE

    if (m_Afile_name.compare(0, rand_prefix.size(), rand_prefix) == 0) {
      dio.readInput(m_Afile_name, this->m_globalm, this->m_globaln, this->m_k,
                    this->m_sparsity, this->m_pr, this->m_pc, this->m_symm_flag,
                    this->m_adj_rand, this->m_input_normalization);
    } else {
      dio.readInput(m_Afile_name);
    }
#ifdef BUILD_SPARSE
    SP_MAT Arows = dio.Arows();
    SP_MAT Acols = dio.Acols();
#else   // ifdef BUILD_SPARSE
    MAT Arows = dio.Arows();
    MAT Acols = dio.Acols();
#endif  // ifdef BUILD_SPARSE

    if (m_Afile_name.compare(0, rand_prefix.size(), rand_prefix) != 0) {
      this->m_globaln = Arows.n_cols;
      this->m_globalm = Acols.n_rows;
    }
    INFO << mpicomm.rank()
         << "::Completed generating 1D rand Arows=" << PRINTMATINFO(Arows)
         << "::Acols=" << PRINTMATINFO(Acols) << std::endl;
#ifdef WRITE_RAND_INPUT
    dio.writeRandInput();
#endif  // ifdef WRITE_RAND_INPUT
    arma::arma_rng::set_seed(random_sieve(mpicomm.rank() + kprimeoffset));
    MAT W = arma::randu<MAT>(this->m_globalm / mpicomm.size(), this->m_k);
    MAT H = arma::randu<MAT>(this->m_globaln / mpicomm.size(), this->m_k);
    sleep(10);
    MPI_Barrier(MPI_COMM_WORLD);
    memusage(mpicomm.rank(), "b4 constructor ");
    NMFTYPE nmfAlgorithm(Arows, Acols, W, H, mpicomm);
    sleep(10);
    memusage(mpicomm.rank(), "after constructor ");
    nmfAlgorithm.num_iterations(this->m_num_it);
    nmfAlgorithm.compute_error(this->m_compute_error);
    nmfAlgorithm.algorithm(this->m_nmfalgo);
    MPI_Barrier(MPI_COMM_WORLD);
    try {
      nmfAlgorithm.computeNMF();
    } catch (std::exception &e) {
      printf("Failed rank %d: %s\n", mpicomm.rank(), e.what());
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (!m_outputfile_name.empty()) {
      dio.writeOutput(nmfAlgorithm.getLeftLowRankFactor(),
                      nmfAlgorithm.getRightLowRankFactor(), m_outputfile_name);
    }
  }

  template <class NMFTYPE>
  void callDistNMF2D() {
    std::string rand_prefix("rand_");
    MPICommunicator mpicomm(this->m_argc, this->m_argv, this->m_pr, this->m_pc);
// #ifdef BUILD_CUDA
//         if (mpicomm.rank()==0){
//             gpuQuery();
//         }
// #endif
#ifdef USE_PACOSS
    std::string dim_part_file_name = this->m_Afile_name;
    dim_part_file_name += ".dpart.part" + std::to_string(mpicomm.rank());
    this->m_Afile_name += ".part" + std::to_string(mpicomm.rank());
    INFO << mpicomm.rank() << ":: part_file_name::" << dim_part_file_name
         << "::m_Afile_name::" << this->m_Afile_name << std::endl;
    Pacoss_SparseStruct<double> ss;
    ss.load(m_Afile_name.c_str());
    std::vector<std::vector<Pacoss_IntPair> > dim_part;
    Pacoss_Communicator<double>::loadDistributedDimPart(
        dim_part_file_name.c_str(), dim_part);
    Pacoss_Communicator<double> *rowcomm = new Pacoss_Communicator<double>(
        MPI_COMM_WORLD, ss._idx[0], dim_part[0]);
    Pacoss_Communicator<double> *colcomm = new Pacoss_Communicator<double>(
        MPI_COMM_WORLD, ss._idx[1], dim_part[1]);
    this->m_globalm = ss._dimSize[0];
    this->m_globaln = ss._dimSize[1];
    arma::umat locations(2, ss._idx[0].size());

    for (Pacoss_Int i = 0; i < ss._idx[0].size(); i++) {
      locations(0, i) = ss._idx[0][i];
      locations(1, i) = ss._idx[1][i];
    }
    arma::vec values(ss._idx[0].size());

    for (Pacoss_Int i = 0; i < values.size(); i++) values[i] = ss._val[i];
    SP_MAT A(locations, values);
    A.resize(rowcomm->localRowCount(), colcomm->localRowCount());
#else  // ifdef USE_PACOSS

    if ((this->m_pr > 0) && (this->m_pc > 0) &&
        (this->m_pr * this->m_pc != mpicomm.size())) {
      ERR << "pr*pc is not MPI_SIZE" << std::endl;
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
#ifdef BUILD_SPARSE
    SP_MAT A;
    DistIO<SP_MAT> dio(mpicomm, m_distio, A);

    if (mpicomm.rank() == 0) {
      INFO << "sparse case::sparsity::" << this->m_sparsity << std::endl;
    }
#else   // ifdef BUILD_SPARSE
    MAT A;
    DistIO<MAT> dio(mpicomm, m_distio, A);
#endif  // ifdef BUILD_SPARSE. One outstanding PACOSS

    if (m_Afile_name.compare(0, rand_prefix.size(), rand_prefix) == 0) {
      dio.readInput(m_Afile_name, this->m_globalm, this->m_globaln, this->m_k,
                    this->m_sparsity, this->m_pr, this->m_pc, this->m_symm_flag,
                    this->m_adj_rand, this->m_input_normalization);
    } else {
      dio.readInput(m_Afile_name, this->m_globalm, this->m_globaln, this->m_k,
                    this->m_sparsity, this->m_pr, this->m_pc, this->m_symm_flag,
                    this->m_adj_rand, this->m_input_normalization);
    }
    A = dio.A();

    //if (m_Afile_name.compare(0, rand_prefix.size(), rand_prefix) != 0) {
    //  UWORD localm = A.n_rows;
    //  UWORD localn = A.n_cols;
    //
    //  /*MPI_Allreduce(&localm, &(this->m_globalm), 1, MPI_INT,
    //   *            MPI_SUM, mpicomm.commSubs()[0]);
    //   * MPI_Allreduce(&localn, &(this->m_globaln), 1, MPI_INT,
    //   *            MPI_SUM, mpicomm.commSubs()[1]);*/
    //  this->m_globalm = localm * m_pr;
    //  this->m_globaln = localn * m_pc;
    //}
#ifdef WRITE_RAND_INPUT
    dio.writeRandInput();
#endif  // ifdef WRITE_RAND_INPUT
#endif  // ifdef USE_PACOSS. Everything over. No more outstanding ifdef's.
    // don't worry about initializing with the
    // same matrix as only one of them will be used.
    arma::arma_rng::set_seed(this->m_initseed + mpicomm.rank());
#ifdef USE_PACOSS
    MAT W = arma::randu<MAT>(rowcomm->localOwnedRowCount(), this->m_k);
    MAT H = arma::randu<MAT>(colcomm->localOwnedRowCount(), this->m_k);
#else   // ifdef USE_PACOSS
    MAT W = arma::randu<MAT>(itersplit(A.n_rows, m_pc,
                              mpicomm.col_rank()), this->m_k);
    MAT H = arma::randu<MAT>(itersplit(A.n_cols, m_pr,
                              mpicomm.row_rank()), this->m_k);
#endif  // ifdef USE_PACOSS
        // sometimes for really very large matrices starting w/
        // rand initialization hurts ANLS BPP running time. For a better
        // initializer we run couple of iterations of HALS.
#ifndef USE_PACOSS
#ifdef BUILD_SPARSE
    if (m_nmfalgo == ANLSBPP && this->m_symm_reg < 0) {
      DistHALS<SP_MAT> lrinitializer(A, W, H, mpicomm, this->m_num_k_blocks);
      lrinitializer.num_iterations(4);
      lrinitializer.algorithm(HALS);
      lrinitializer.computeNMF();
      W = lrinitializer.getLeftLowRankFactor();
      H = lrinitializer.getRightLowRankFactor();
    }
#endif  // ifdef BUILD_SPARSE
#endif  // ifndef USE_PACOSS

#ifdef MPI_VERBOSE
    INFO << mpicomm.rank() << "::" << __PRETTY_FUNCTION__
         << "::" << PRINTMATINFO(W) << std::endl;
    INFO << mpicomm.rank() << "::" << __PRETTY_FUNCTION__
         << "::" << PRINTMATINFO(H) << std::endl;
#endif  // ifdef MPI_VERBOSE
    MPI_Barrier(MPI_COMM_WORLD);
    memusage(mpicomm.rank(), "b4 constructor ");
    // TODO(ramkikannan): I was here. Need to modify the reallocations by using
    // localOwnedRowCount instead of m_globalm.
    double global_A_sum = 0.0;
    double global_A_mean = 0.0;
    double global_A_max = 0.0;
    if (this->m_symm_reg >= 0) {
      double local_A_sum = arma::accu(A);
      MPI_Allreduce(&local_A_sum, &global_A_sum, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      global_A_mean = global_A_sum / (this->m_globalm * this->m_globalm);
      // standard symm nmf initialization as per Da Kuang's code.
      int rowprime = 17;
      int colprime = 29;

      int hseed = this->m_initseed
                  + rowprime*mpicomm.row_rank() + colprime*mpicomm.col_rank();
      arma::arma_rng::set_seed(hseed);
      H.randu();
      H = 2 * std::sqrt(global_A_mean / this->m_k) * H;

      int wseed = this->m_initseed
                  + colprime*mpicomm.row_rank() + rowprime*mpicomm.col_rank();
      arma::arma_rng::set_seed(wseed);
      W.randu();
      W = 2 * std::sqrt(global_A_mean / this->m_k) * W;
    }
    NMFTYPE nmfAlgorithm(A, W, H, mpicomm, this->m_num_k_blocks);
#ifdef USE_PACOSS
    nmfAlgorithm.set_rowcomm(rowcomm);
    nmfAlgorithm.set_colcomm(colcomm);
#endif  // ifdef USE_PACOSS
    memusage(mpicomm.rank(), "after constructor ");
    nmfAlgorithm.num_iterations(this->m_num_it);
    nmfAlgorithm.compute_error(this->m_compute_error);
    nmfAlgorithm.algorithm(this->m_nmfalgo);
    nmfAlgorithm.regW(this->m_regW);
    nmfAlgorithm.regH(this->m_regH);
    if (this->m_symm_reg == 0) {
      double local_A_max = A.max();
      MPI_Allreduce(&local_A_max, &global_A_max, 1, MPI_DOUBLE, MPI_MAX,
                    MPI_COMM_WORLD);
      this->m_symm_reg = global_A_max * global_A_max;
    }
    nmfAlgorithm.symm_reg(this->m_symm_reg);

    // Optional LUC Algorithm params
    nmfAlgorithm.set_luciters(this->m_max_luciters);

    MPI_Barrier(MPI_COMM_WORLD);
    try {
      mpitic();
      nmfAlgorithm.computeNMF();
      double temp = mpitoc();

      if (mpicomm.rank() == 0) printf("NMF took %.3lf secs.\n", temp);
    } catch (std::exception &e) {
      printf("Failed rank %d: %s\n", mpicomm.rank(), e.what());
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (!m_outputfile_name.empty()) {
      dio.writeOutput(nmfAlgorithm.getLeftLowRankFactor(),
                      nmfAlgorithm.getRightLowRankFactor(), m_outputfile_name);
    }
  }
  void parseCommandLine() {
    ParseCommandLine pc(this->m_argc, this->m_argv);
    pc.parseplancopts();
    this->m_nmfalgo = pc.lucalgo();
    this->m_k = pc.lowrankk();
    this->m_Afile_name = pc.input_file_name();
    this->m_pr = pc.pr();
    this->m_pc = pc.pc();
    this->m_sparsity = pc.sparsity();
    this->m_num_it = pc.iterations();
    this->m_distio = TWOD;
    this->m_regW = pc.regW();
    this->m_regH = pc.regH();
    this->m_num_k_blocks = 1;
    this->m_globalm = pc.globalm();
    this->m_globaln = pc.globaln();
    this->m_compute_error = pc.compute_error();
    this->m_tolerance = pc.tolerance();
    this->m_symm_reg = pc.symm_reg();
    this->m_symm_flag = 0;
    this->m_adj_rand = pc.adj_rand();
    this->m_max_luciters = pc.max_luciters();
    this->m_initseed = pc.initseed();
    this->m_outputfile_name = pc.output_file_name();

    // Put in the default LUC iterations
    if (this->m_max_luciters == -1) {
      this->m_max_luciters = this->m_k;
    }

    // check the conditions for symm nmf
    if (this->m_symm_reg != -1) {
      this->m_symm_flag = 1;
      if (this->m_pr != this->m_pc) {
        ERR << "Symmetric Regularization enabled"
            << " and process grid is not square"
            << "::pr::" << this->m_pr << "::pc::" << this->m_pc << std::endl;
        return;
      }
      if (this->m_globalm != this->m_globaln) {
        // TODO: Add file prefix check
        ERR << "Symmetric Regularization enabled"
            << " and input matrix is not square"
            << "::m::" << this->m_globalm << "::n::" << this->m_globaln
            << std::endl;
        return;
      }
      if ((this->m_nmfalgo != ANLSBPP) && (this->m_nmfalgo != GNSYM)) {
        ERR << "Symmetric Regularization enabled "
            << "is only enabled for ANLSBPP and GNSYM"
            << std::endl;
        return;
      }
    }
    if (this->m_nmfalgo == NAIVEANLSBPP) {
      this->m_distio = ONED_DOUBLE;
    } else {
      this->m_distio = TWOD;
    }
    this->m_input_normalization = pc.input_normalization();
    pc.printConfig();
    switch (this->m_nmfalgo) {
      case MU:
#ifdef BUILD_SPARSE
        callDistNMF2D<DistMU<SP_MAT> >();
#else   // ifdef BUILD_SPARSE
        callDistNMF2D<DistMU<MAT> >();
#endif  // ifdef BUILD_SPARSE
        break;
      case HALS:
#ifdef BUILD_SPARSE
        callDistNMF2D<DistHALS<SP_MAT> >();
#else   // ifdef BUILD_SPARSE
        callDistNMF2D<DistHALS<MAT> >();
#endif  // ifdef BUILD_SPARSE
        break;
      case ANLSBPP:
#ifdef BUILD_SPARSE
        callDistNMF2D<DistANLSBPP<SP_MAT> >();
#else   // ifdef BUILD_SPARSE
        callDistNMF2D<DistANLSBPP<MAT> >();
#endif  // ifdef BUILD_SPARSE
        break;
      case NAIVEANLSBPP:
#ifdef BUILD_SPARSE
        callDistNMF1D<DistNaiveANLSBPP<SP_MAT> >();
#else   // ifdef BUILD_SPARSE
        callDistNMF1D<DistNaiveANLSBPP<MAT> >();
#endif  // ifdef BUILD_SPARSE
        break;
      case AOADMM:
#ifdef BUILD_SPARSE
        callDistNMF2D<DistAOADMM<SP_MAT> >();
#else   // ifdef BUILD_SPARSE
        callDistNMF2D<DistAOADMM<MAT> >();
#endif  // ifdef BUILD_SPARSE
        break;
      case CPALS:
#ifdef BUILD_SPARSE
        callDistNMF2D<DistALS<SP_MAT> >();
#else   // ifdef BUILD_SPARSE
        callDistNMF2D<DistALS<MAT> >();
#endif  // ifdef BUILD_SPARSE
        break;
      case GNSYM:
#ifdef BUILD_SPARSE
        callDistNMF2D<DistGNSym<SP_MAT> >();
#else   // ifdef BUILD_SPARSE
        callDistNMF2D<DistGNSym<MAT> >();
#endif  // ifdef BUILD_SPARSE
        break;
      case R2:
#ifdef BUILD_SPARSE
        callDistNMF2D<DistR2<SP_MAT> >();
#else   // ifdef BUILD_SPARSE
        callDistNMF2D<DistR2<MAT> >();
#endif  // ifdef BUILD_SPARSE
        break;
      default:
        ERR << "Unsupported algorithm " << this->m_nmfalgo << std::endl;
    }
  }

 public:
  DistNMFDriver(int argc, char *argv[]) {
    this->m_argc = argc;
    this->m_argv = argv;
    this->parseCommandLine();
  }
};

}  // namespace planc

int main(int argc, char *argv[]) {
  try {
    planc::DistNMFDriver dnd(argc, argv);
    fflush(stdout);
  } catch (const std::exception &e) {
    INFO << "Exception with stack trace " << std::endl;
    INFO << e.what();
  }
}
