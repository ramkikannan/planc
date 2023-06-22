/* Copyright 2022 Ramakrishnan Kannan */

#include <distjointnmf_driver.hpp>

namespace planc {

class DistJointNMFDriver {
 private:
  int m_argc;
  char **m_argv;
  
  // Matrix dimensions
  int m_k;
  UWORD m_globalm, m_globaln;
  
  // Input names (and creation variables)
  std::string m_Afile_name, m_Sfile_name, m_outputfile_name;
  double m_sparsity;
  iodistributions m_distio;
  bool m_adj_rand;
  normtype m_input_normalization;

  // Grid sizes
  int m_pr, m_pc, m_cpr, m_cpc;
  
  // Regularisation
  FVEC m_regW, m_regH;
  
  // Algorithm variables
  algotype m_nmfalgo;
  int m_num_it;
  uint m_compute_error;
  double m_tolerance;
  int m_num_k_blocks;
  
  // Seeds for matrix creation
  int m_initseed;
  static const int kprimeoffset = 17;
  
  // Joint NMF variables
  double m_alpha, m_beta;
  bool feat_type, conn_type;
  double m_gamma;
  int m_unpartitioned = 1;
  int m_maxluciters;

#ifdef INIT_TRUE_LR
  int kW_true_seed = WTRUE_SEED;
  int kH_true_seed = HTRUE_SEED;
#endif

  void printConfig() {
    INFO << "a::" << this->m_nmfalgo << "::i::" << this->m_Afile_name
         << "::k::" << this->m_k << "::m::" << this->m_globalm
         << "::n::" << this->m_globaln << "::t::" << this->m_num_it
         << "::pr::" << this->m_pr << "::pc::" << this->m_pc
         << "::cpr::" << this->m_cpr << "::cpc::" << this->m_cpc
         << "::error::" << this->m_compute_error
         << "::distio::" << this->m_distio << "::regW::"
         << "l2::" << this->m_regW(0) << "::l1::" << this->m_regW(1)
         << "::regH::"
         << "l2::" << this->m_regH(0) << "::l1::" << this->m_regH(1)
         << "::alpha::" << this->m_alpha << "::beta::" << this->m_beta
         << "::num_k_blocks::" << this->m_num_k_blocks
         << "::normtype::" << this->m_input_normalization << std::endl;
  }

  template <class NMFTYPE, class T1, class T2>
  void callDistJointNMF() {
    T1 A;
    T2 S;

    printConfig();

    std::string rand_prefix("rand_");
    MPICommunicatorJNMF Acomm(this->m_argc, this->m_argv, this->m_pr, this->m_pc);
    if ((this->m_pr > 0) && (this->m_pc > 0) &&
        (this->m_pr * this->m_pc != Acomm.size())) {
      ERR << "pr*pc is not MPI_SIZE" << std::endl;
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPICommunicatorJNMF Scomm(this->m_argc, this->m_argv, 
        this->m_cpr, this->m_cpc, Acomm.gridComm(), Acomm.commSubs());

    DistJointNMFIO<T1> Adio(Acomm, A);

    //NOTE: put lock after barrier for now... 
    int i = 1;
#ifdef MPI_VERBOSE
    if (Scomm.rank() == 0) {
        pid_t pid = getpid();
        INFO << "Process ID: " << pid << " is the process you are looking for" << std::endl;
    }
#endif
    while(i<1){/* infinite while-loop for debugger */}
    
    mpitic();
// NOTE: we assume that both X and S are either partitioned or not partitioned
    Adio.readInput(m_Afile_name, this->m_globalm, this->m_globaln, this->m_k,
                this->m_sparsity, Acomm.pr(), Acomm.pc(), 0,
                this->m_adj_rand, this->m_input_normalization, this->m_unpartitioned);

    double temp_a_input = mpitoc();
    if(Acomm.rank() == 0){printf("A readinput took %.3lf secs.\n", temp_a_input);}

    A = Adio.A();

    //A.brief_print();
    INFO << Acomm.rank() << "::Calling with Scomm::" << std::endl;

    DistJointNMFIO<T2> Sdio(Scomm, S);

    int h = 1;
    while(h<1){/* infinite while-loop for debugger */}

    mpitic();
#ifdef APPEND_TIME
tic();
#endif
    Sdio.readInput(m_Sfile_name, this->m_globaln, this->m_globaln, this->m_k,
                this->m_sparsity, Scomm.pr(), Scomm.pc(), 1,
                this->m_adj_rand, this->m_input_normalization, this->m_unpartitioned);
#ifdef APPEND_TIME
double sri_time = toc();
if(Scomm.rank() == 0){printf("S readinput estimated by chrono took %.3lf secs.\n", sri_time);}
#endif

    double temp_s_input = mpitoc();
    if(Scomm.rank() == 0){printf("S readinput took %.3lf secs.\n", temp_s_input);}

    S = Sdio.A();

#ifdef MPI_VERBOSE
    MPI_Barrier(MPI_COMM_WORLD);
    auto f_config_A = [this, &A, &Acomm](){
      cout  << "A: (" << Acomm.row_rank() << ", " << Acomm.col_rank() << ")" << endl;
      A.print();
    };
    mpi_serial_print(f_config_A);
    MPI_Barrier(MPI_COMM_WORLD);
    auto f_config_S = [this, &S, &Scomm](){
      cout  << "S: (" << Scomm.row_rank() << ", " << Scomm.col_rank() << ")" << endl;
      S.print();
    };
    mpi_serial_print(f_config_S);
    MPI_Barrier(MPI_COMM_WORLD);
#endif

#ifdef WRITE_RAND_INPUT
    string prefix = "Arnd";
    Adio.writeRandInput(prefix, this->feat_type);
    prefix = "Srnd";
    Sdio.writeRandInput(prefix, this->conn_type);
#endif  // ifdef WRITE_RAND_INPUT

    arma::arma_rng::set_seed(this->m_initseed + Acomm.rank());

    MAT W = arma::zeros<MAT>(itersplit(A.n_rows, m_pc,
                              Acomm.col_rank()), this->m_k);
    MAT H = arma::zeros<MAT>(itersplit(A.n_cols, m_pr,
                              Acomm.row_rank()), this->m_k);

    // Get starting indices
    procInfo Winfo = getrownum(this->m_globalm,
                        Acomm.row_rank(), Acomm.col_rank(),
                        Acomm.pr(), Acomm.pc(), 'R');
    procInfo Hinfo = getrownum(this->m_globaln,
                        Acomm.row_rank(), Acomm.col_rank(),
                        Acomm.pr(), Acomm.pc(), 'C');

#ifdef INIT_TRUE_LR
    // Make the noise matrices
    MAT nW = arma::zeros<MAT>(W.n_rows, W.n_cols);
    int nW_seed = 1811;
    MAT nH = arma::zeros<MAT>(H.n_rows, H.n_cols);
    int nH_seed = 2503;

    // Set the matrix as the true low rank with tiny noise
    gen_discard(Winfo.start_idx, Winfo.nrows, this->m_k,
           W, false, kW_true_seed);
    gen_discard(Winfo.start_idx, Winfo.nrows, this->m_k,
           nW, false, nW_seed);
    W = W + (0.001 * nW);
    gen_discard(Hinfo.start_idx, Hinfo.nrows, this->m_k,
           H, false, kH_true_seed);
    gen_discard(Hinfo.start_idx, Hinfo.nrows, this->m_k,
           nH, false, nH_seed);
    H = H + (0.001 * nH);
#else
    gen_discard(Winfo.start_idx, Winfo.nrows, this->m_k,
           W, false, this->m_initseed + 17);
    gen_discard(Hinfo.start_idx, Hinfo.nrows, this->m_k,
           H, false, this->m_initseed + 23);
#endif

#ifdef WRITE_RAND_INPUT
    Adio.writeOutput(W, H, "initfac");
#endif

#ifdef MPI_VERBOSE
    INFO << Acomm.rank() << "::" << __PRETTY_FUNCTION__
         << "::" << PRINTMATINFO(W) << std::endl;
    INFO << Acomm.rank() << "::" << __PRETTY_FUNCTION__
         << "::" << PRINTMATINFO(H) << std::endl;
#endif  // ifdef MPI_VERBOSE
    MPI_Barrier(MPI_COMM_WORLD);
    memusage(Acomm.rank(), "b4 constructor ");
    // TODO(ramkikannan): I was here. Need to modify the reallocations by using
    // localOwnedRowCount instead of m_globalm.
    double global_A_sum = 0.0;
    double global_A_mean = 0.0;
    double global_A_max = 0.0;
    
    NMFTYPE nmfAlgorithm(A, S, W, H, Acomm, Scomm, this->m_num_k_blocks);
    memusage(Acomm.rank(), "after constructor ");
    nmfAlgorithm.num_iterations(this->m_num_it);
    nmfAlgorithm.compute_error(this->m_compute_error);
    nmfAlgorithm.algorithm(this->m_nmfalgo);
    nmfAlgorithm.regW(this->m_regW);
    nmfAlgorithm.regH(this->m_regH);

    /*
     * Hyperparameter settings
     */
    // Set alpha (0 for default settings)
    // Default alpha = norm(A, 'fro')^2 / norm(S, 'fro')^2
    if (this->m_alpha > 0) {
      nmfAlgorithm.alpha(this->m_alpha);
    }
    // Set the algorithm specific hyperparameters
    // Set beta (0 for default settings)
    // Default beta = alpha * max(S)
    if (this->m_beta > 0) {
      nmfAlgorithm.beta(this->m_beta);
    }
    // -1 is the input signifying default momentum (0.9)
    if (this->m_gamma >= 0) {
      nmfAlgorithm.gamma(this->m_gamma);
    }
    // Number of CG iterations for PGNCG  (-1 for default settings)
    // Default is 20 iterations
    if (this->m_maxluciters >= 0) {
      nmfAlgorithm.set_luciters(this->m_maxluciters);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    try {
      mpitic();
      nmfAlgorithm.computeNMF();
      double temp = mpitoc();

      if (Acomm.rank() == 0) printf("JointNMF took %.3lf secs.\n", temp);
    } catch (std::exception &e) {
      printf("Failed rank %d: %s\n", Acomm.rank(), e.what());
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Write out the factor matrices (if needed)
    if (!m_outputfile_name.empty()) {
      nmfAlgorithm.saveOutput(this->m_outputfile_name);
    }
  }
  void parseCommandLine() {
    ParseCommandLine pc(this->m_argc, this->m_argv);
    pc.parseplancopts(DISTJOINTNMF);
    this->m_nmfalgo = pc.lucalgo();
    this->m_k = pc.lowrankk();
    this->m_Afile_name = pc.input_file_name();
    this->m_Sfile_name = pc.conn_file_name();
    this->m_pr = pc.pr();
    this->m_pc = pc.pc();
    this->m_cpr = pc.cpr();
    this->m_cpc = pc.cpc();
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
    this->m_adj_rand = pc.adj_rand();
    this->m_initseed = pc.initseed();
    this->m_outputfile_name = pc.output_file_name();
    this->m_alpha = pc.joint_alpha();
    this->m_beta = pc.joint_beta();
    this->m_gamma = pc.gamma();
    this->m_unpartitioned = pc.unpartitioned();
    this->m_maxluciters = pc.max_luciters();
    this->feat_type = pc.feat_typesity();
    this->conn_type = pc.conn_typesity();

    this->m_distio = TWOD;
    this->m_input_normalization = pc.input_normalization();

    if((this->m_pr * this->m_pc) > this->m_globalm || (this->m_pr * this->m_pc) > this->m_globaln){
      stringstream error;
      error << "p > n or m. number of processors must be greater than both m and n" << endl;
      cerr << error.str() << endl;
      throw error.str();
    }

    pc.printConfig(DISTJOINTNMF);
    switch (this->m_nmfalgo) {
      case ANLSBPP:
        build_opts<DistANLSBPPJointNMF>();
        break;
      case PGD:
        build_opts<DistPGDJointNMF>();
        break;
      case PGNCG:
        build_opts<DistPGNCGJointNMF>();
        break;
      default:
        ERR << "Unsupported algorithm " << this->m_nmfalgo << std::endl;
    }
  }

  template <template <class, class> class NMFTYPE>
  void build_opts(){
        // TODO: wrap all this in a templated<ANLSBPPJointNMF> function
#ifdef BUILD_SPDEN
        callDistJointNMF<NMFTYPE<SP_MAT, MAT>, SP_MAT, MAT>();
#elif BUILD_DENSP
        callDistJointNMF<NMFTYPE<MAT, SP_MAT>, MAT, SP_MAT>();
#elif BUILD_DENDEN
        callDistJointNMF<NMFTYPE<MAT,MAT>, MAT,MAT>();
#elif BUILD_SPSP
        callDistJointNMF<NMFTYPE<SP_MAT, SP_MAT>, SP_MAT, SP_MAT>();
#else // build all templates version here
      INFO << "feat_type: " << this->feat_type << "  conn_type: " << this->conn_type << std::endl;
      if (this->feat_type && this->conn_type) {
        callDistJointNMF<NMFTYPE<MAT, MAT>, MAT, MAT>();
      } else if (!this->feat_type && this->conn_type) {
        callDistJointNMF<NMFTYPE<SP_MAT, MAT>, SP_MAT, MAT>();
      } else if (this->feat_type && !this->conn_type) {
        callDistJointNMF<NMFTYPE<MAT, SP_MAT>, MAT, SP_MAT>();
      } else if (!this->feat_type && !this->conn_type) {
        callDistJointNMF<NMFTYPE<SP_MAT, SP_MAT>, SP_MAT, SP_MAT>();
      }
#endif // ifdef BUILD_TEMPLATES
  }

 public:
  DistJointNMFDriver(int argc, char *argv[]) {
    this->m_argc = argc;
    this->m_argv = argv;
    this->parseCommandLine();
  }
};

}  // namespace planc

int main(int argc, char *argv[]) {
  try {
    planc::DistJointNMFDriver dnd(argc, argv);
    fflush(stdout);
  } catch (const std::exception &e) {
    INFO << "Exception with stack trace " << std::endl;
    INFO << e.what();
  }
}
