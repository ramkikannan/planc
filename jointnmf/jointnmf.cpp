#include <jointnmf_driver.hpp>

namespace planc {

class JointNMFDriver {
 private:
  int m_argc;
  char **m_argv;
  int m_k;
  UWORD m_m, m_n;

  std::string m_Afile_name, m_Sfile_name, m_outputfile_name;
  
  int m_num_it, m_initseed;
  
  FVEC m_regW, m_regH;
  
  bool m_adj_rand;
  algotype m_nmfalgo;
  
  double m_sparsity, m_tolerance;
  
  uint m_compute_error;
  normtype m_input_normalization;
  
  double alpha, beta;
  bool feat_type, conn_type;

  // Variables for creating random matrix
  static const int kW_seed_idx = 1210873;
  static const int kprimeoffset = 17;

  double m_gamma;

  template <class NMFTYPE, class T1, class T2>
  void callJointNMF() {
    T1 A;
    T2 S;

    // Generate/Read data matrix
    double t2;

    if (!this->m_Afile_name.empty() &&
      this->m_Afile_name.compare(0, 4, "rand") &&
      !this->m_Sfile_name.empty() &&
      this->m_Sfile_name.compare(0, 4, "rand")) {
      INFO << "Reading in from file" << std::endl;
      tic();
      read_input_matrix(A, this->m_Afile_name);
      read_input_matrix(S, this->m_Sfile_name);
      t2 = toc();
      INFO << "Successfully loaded input matrices A, S: " << PRINTMATINFO(A) 
         << "::" << PRINTMATINFO(S) << "::"
         << "(" << t2 << " s)" << std::endl;
      this->m_m = A.n_rows;
      this->m_n = A.n_cols;
    } else {
      INFO << "Generating random matrix" << std::endl;
      arma::arma_rng::set_seed(this->kW_seed_idx);
      std::string rand_prefix("rand_");
      std::string atype = this->m_Afile_name.substr(rand_prefix.size());
      assert(atype == "normal" || atype == "lowrank" || atype == "uniform");
      std::string stype = this->m_Sfile_name.substr(rand_prefix.size());
      assert(stype == "normal" || stype == "lowrank" || stype == "uniform");

      tic();
      generate_rand_matrix(A, atype, this->m_m, this->m_n, this->m_k,
          this->m_sparsity, false, this->m_adj_rand);
      generate_rand_matrix(S, stype, this->m_n, this->m_n, this->m_k,
          this->m_sparsity, true, this->m_adj_rand);
      t2 = toc();
      INFO << "generated random matrices A, S: " << PRINTMATINFO(A)
           << "::" << PRINTMATINFO(S) << "::"
           << "(" << t2 << " s)" << std::endl;

      // INFO << "A number non-zeros: " << size(nonzeros(A)) << std::endl;
      // INFO << "S number non-zeros: " << size(nonzeros(S)) << std::endl;
    }

    // Normalize the input matrix
    if (this->m_input_normalization != NONE) {
      tic();
      if (this->m_input_normalization == L2NORM) {
        A = arma::normalise(A);
      } else if (this->m_input_normalization == MAXNORM) {
        double maxnorm = 1 / A.max();
        A = maxnorm * A;
      }
      t2 = toc();
      INFO << "Normalized A (" << t2 << "s)" << std::endl;
    }

    // Set parameters and call NMF
    arma::arma_rng::set_seed(this->m_initseed);
    MAT W = arma::randu<MAT>(this->m_m, this->m_k);
    MAT H = arma::randu<MAT>(this->m_n, this->m_k);
    
    NMFTYPE nmfAlgorithm(A, S, W, H);
    nmfAlgorithm.num_iterations(this->m_num_it);
    nmfAlgorithm.algorithm(this->m_nmfalgo);
    // Always compute error for the shared memory case
    // nmfAlgorithm.compute_error(this->m_compute_error);

    // Set alpha parameter (if it is an input)
    if (this->alpha > 0) {
      nmfAlgorithm.alpha(this->alpha);
    }

    // -1 is the input signifying default momentum (0.9)
    if (this->m_gamma >= 0) {
      nmfAlgorithm.gamma(this->m_gamma);
    }

    // Optional parameters (per algorithm)
    switch (this->m_nmfalgo) {
      case ANLSBPP:
        if (this->beta > 0) {
          nmfAlgorithm.beta(this->beta);
        }
        if (!this->m_regW.empty()) {
          nmfAlgorithm.regW(this->m_regW);
        }
        if (!this->m_regH.empty()) {
          nmfAlgorithm.regH(this->m_regH);
        }
      break;
      case PGD:
        if (!this->m_regW.empty()) {
          nmfAlgorithm.regW(this->m_regW);
        }
        if (!this->m_regH.empty()) {
          nmfAlgorithm.regH(this->m_regH);
        }
      break;
    }

    INFO << "completed constructor" << PRINTMATINFO(A) << std::endl;
    tic();
    nmfAlgorithm.computeNMF();
    t2 = toc();
    printf("JointNMF took %.3lf secs.\n", t2);

    // Save the factor matrices
    if (!this->m_outputfile_name.empty()) {
      nmfAlgorithm.saveOutput(this->m_outputfile_name);
    }
  }

  void parseCommandLine() {
    ParseCommandLine pc(this->m_argc, this->m_argv);
    pc.parseplancopts(JOINTNMF);
    this->m_nmfalgo = pc.lucalgo();
    this->m_k = pc.lowrankk();
    this->m_Afile_name = pc.input_file_name();
    this->m_sparsity = pc.sparsity();
    this->m_num_it = pc.iterations();
    this->m_regW = pc.regW();
    this->m_regH = pc.regH();
    this->m_m = pc.globalm();
    this->m_n = pc.globaln();
    this->m_compute_error = pc.compute_error();
    this->m_tolerance = pc.tolerance();
    this->m_adj_rand = pc.adj_rand();
    this->m_initseed = pc.initseed();
    this->m_outputfile_name = pc.output_file_name();
    this->m_Sfile_name = pc.conn_file_name();
    this->alpha = pc.joint_alpha();
    this->beta = pc.joint_beta();
    this->m_gamma = pc.gamma();
    this->feat_type = pc.feat_typesity();
    this->conn_type = pc.conn_typesity();

    pc.printConfig(JOINTNMF);
    switch (this->m_nmfalgo) {
      case ANLSBPP:
        build_opts<ANLSBPPJointNMF>();
        break;
      case PGD:
        build_opts<PGDJointNMF>();
        break;
      case PGNCG:
        build_opts<PGNCGJointNMF>();
        break;
      default:
        ERR << "Unsupported algorithm " << this->m_nmfalgo << std::endl;
    }
  }

  template <template <class, class> class NMFTYPE>
  void build_opts(){
        // TODO: wrap all this in a templated<ANLSBPPJointNMF> function
#ifdef BUILD_SPDEN
        callJointNMF<NMFTYPE<SP_MAT, MAT>, SP_MAT, MAT>();
#elif BUILD_DENSP
        callJointNMF<NMFTYPE<MAT, SP_MAT>, MAT, SP_MAT>();
#elif BUILD_DENDEN
        callJointNMF<NMFTYPE<MAT,MAT>, MAT,MAT>();
#elif BUILD_SPSP
        callJointNMF<NMFTYPE<SP_MAT, SP_MAT>, SP_MAT, SP_MAT>();
#else // build all templates version here
      if(this->feat_type && this->conn_type){
        callJointNMF<NMFTYPE<MAT, MAT>, MAT, MAT>();
      }else if(!this->feat_type && this->conn_type){
        callJointNMF<NMFTYPE<SP_MAT, MAT>, SP_MAT, MAT>();
      }else if(this->feat_type && !this->conn_type){
        callJointNMF<NMFTYPE<MAT, SP_MAT>, MAT, SP_MAT>();
      }else if(!this->feat_type && !this->conn_type){
        callJointNMF<NMFTYPE<SP_MAT, SP_MAT>, SP_MAT, SP_MAT>();
      }
#endif // ifdef BUILD_TEMPLATES
}

 public:
  JointNMFDriver(int argc, char *argv[]) {
    this->m_argc = argc;
    this->m_argv = argv;
    this->parseCommandLine();
  }
};  // class NMFDriver

}  // namespace planc

int main(int argc, char *argv[]) {
  try {
    planc::JointNMFDriver dnd(argc, argv);
    fflush(stdout);
  } catch (const std::exception &e) {
    INFO << "Exception with stack trace " << std::endl;
    INFO << e.what();
  }
}