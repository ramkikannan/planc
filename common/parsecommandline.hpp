/* Copyright Ramakrishnan Kannan 2017 */

#ifndef COMMON_PARSECOMMANDLINE_HPP_
#define COMMON_PARSECOMMANDLINE_HPP_

#include <getopt.h>
#include <armadillo>
#include <iostream>
#include <sstream>
#include <string>
#include "common/parsecommandline.h"

namespace planc {
class ParseCommandLine {
 private:
  int m_argc;
  char **m_argv;

  // common to all algorithms.
  algotype m_lucalgo;
  normtype m_input_normalization;
  bool m_compute_error;
  int m_num_it;
  int m_num_k_blocks;
  bool m_dim_tree;

  // file names
  std::string m_Afile_name;
  std::string m_outputfile_name;
  // std::string m_init_file_name;

  // nmf related values
  UWORD m_k;
  UWORD m_globalm;
  UWORD m_globaln;

  // algo related values
  FVEC m_regW;
  FVEC m_regH;
  float m_sparsity;

  // distnmf related values
  int m_pr;
  int m_pc;

  // dist ntf
  int m_num_modes;
  UVEC m_dimensions;
  UVEC m_proc_grids;
  FVEC m_regularizers;

  void parseArrayofString(const char opt, const char *input) {
    std::stringstream ss(input);
    std::string s;
    int i = 0;
    if (this->m_num_modes == 0) {
      while (getline(ss, s, ' ')) {
        i++;
      }
      this->m_num_modes = i;
      this->m_dimensions = arma::zeros<UVEC>(this->m_num_modes);
      this->m_regularizers = arma::zeros<FVEC>(2 * this->m_num_modes);
      this->m_proc_grids = arma::ones<UVEC>(this->m_num_modes);
    }
    i = 0;
    ss.clear();
    ss.str(input);
    while (getline(ss, s, ' ')) {
      switch (opt) {
        case 'd':
          this->m_dimensions(i++) = ::atoi(s.c_str());
          break;
        case 'r':
          this->m_regularizers(i++) = ::atof(s.c_str());
          break;
        case 'p':
          this->m_proc_grids(i++) = ::atoi(s.c_str());
          break;
        default:
          INFO << "wrong option::" << opt << "::values::" << input << std::endl;
      }
    }
  }

 public:
  /**
   * Constructor that takes the number of arguments and the 
   * command line parameters.
   * @param[in] argc - number of arguments
   * @param[in] **argv - command line parameters.
   */ 
  ParseCommandLine(int argc, char **argv) : m_argc(argc), m_argv(argv) {
    this->m_num_modes = 0;
    this->m_pr = 1;
    this->m_pc = 1;
    this->m_regW = arma::zeros<FVEC>(2);
    this->m_regH = arma::zeros<FVEC>(2);
    this->m_num_k_blocks = 1;
    this->m_k = 20;
    this->m_num_it = 20;
    this->m_lucalgo = ANLSBPP;
    this->m_compute_error = 0;
    this->m_input_normalization = NONE;
    this->m_dim_tree = 1;
  }
  /// parses the command line parameters
  void parseplancopts() {
    int opt, long_index;
    while ((opt = getopt_long(this->m_argc, this->m_argv,
                              "a:d:e:i:k:o:p:r:s:t:", plancopts,
                              &long_index)) != -1) {
      switch (opt) {
        case 'a':
          this->m_lucalgo = static_cast<algotype>(atoi(optarg));
          break;
        case 'e':
          this->m_compute_error = atoi(optarg);
          break;
        case 'i': {
          std::string temp = std::string(optarg);
          this->m_Afile_name = temp;
          break;
        }
        case 'k':
          this->m_k = atoi(optarg);
          break;
        case 'o': {
          std::string temp = std::string(optarg);
          this->m_outputfile_name = temp;
          break;
        }
        case 'd':
        case 'r':
        case 'p':
          parseArrayofString(opt, optarg);
          break;
        case 's':
          this->m_sparsity = atof(optarg);
          break;
        case 't':
          this->m_num_it = atoi(optarg);
          break;
        case NUMKBLOCKS:
          this->m_num_k_blocks = atoi(optarg);
          break;
        case NORMALIZATION: {
          std::string temp = std::string(optarg);
          if (temp.compare("l2") == 0) {
            this->m_input_normalization = normtype::L2NORM;
          } else if (temp.compare("max") == 0) {
            this->m_input_normalization = normtype::MAXNORM;
          }
          break;
        }
        case DIMTREE:
          this->m_dim_tree = atoi(optarg);
          break;
        default:
          std::cout << "failed while processing argument:" << optarg
                    << std::endl;
          print_usage();
          exit(EXIT_FAILURE);
      }
    }
    // properly initialize the values for nmf cases now.
    if (this->m_num_modes == 2) {
      this->m_globalm = this->m_dimensions(0);
      this->m_globaln = this->m_dimensions(1);
      this->m_regW(0) = this->m_regularizers(0);
      this->m_regW(1) = this->m_regularizers(1);
      this->m_regH(0) = this->m_regularizers(2);
      this->m_regH(1) = this->m_regularizers(3);
      this->m_pr = this->m_proc_grids(0);
      this->m_pc = this->m_proc_grids(1);
    }
  }

  /// print the configuration received through the command line paramters

  void printConfig() {
    std::cout << "a::" << this->m_lucalgo << "::i::" << this->m_Afile_name
              << "::k::" << this->m_k << "::m::" << this->m_globalm
              << "::n::" << this->m_globaln << "::t::" << this->m_num_it
              << "::pr::" << this->m_pr << "::pc::" << this->m_pc
              << "::error::" << this->m_compute_error << "::regW::"
              << "l2::" << this->m_regW(0) << "::l1::" << this->m_regW(1)
              << "::regH::"
              << "l2::" << this->m_regH(0) << "::l1::" << this->m_regH(1)
              << "::num_k_blocks::" << m_num_k_blocks
              << "::dimensions::" << this->m_dimensions
              << "::procs::" << this->m_proc_grids
              << "::regularizers::" << this->m_regularizers
              << "::input normalization::" << this->m_input_normalization
              << "::dimtree::" << this->m_dim_tree << std::endl;
  }

  void print_usage() {
    INFO << std::endl;
    INFO << "distnmf usage:" << std::endl;
    INFO << "for short arguments like -i do not use equals sign, eg -t 10"
         << std::endl
         << "for long arguments like --pr give key=value pair, eg --pr=4"
         << std::endl
         << "algorithm codes 0-MU2D, 1-HALS2D, 2-ANLSBPP2D, 3-NAIVEANLSBPP"
         << std::endl;
    // mpirun -np 12 distnmf algotype lowrank m n numIteration pr pc
    INFO << "Usage 1: mpirun -np 6 distnmf -a 0/1/2/3 -k 50"
         << "-i rand_uniform/rand_normal/rand_lowrank --dimtree 1"
         << "-d \"21600 14400\" -t 10 -p \"3 2\" "
         << "--normalization \"l2\" "
         << "-r \"0.0001 0 0 0.0001\" " << std::endl;
    // mpirun -np 12 distnmf algotype lowrank AfileName numIteration pr pc
    INFO << "Usage 2: mpirun -np 6 distnmf -a 0/1/2/3 -k 50 --dimtree 1"
         << "-i Ainput -t 10 -p \"3 2\" "
         << "--normalization \"max\" "
         << "-r \"0.0001 0 0 0.0001\" " << std::endl;
    // mpirun -np 12 distnmf algotype lowrank Afile nmfoutput numIteration pr pc
    INFO << "Usage 3: mpirun -np 6 distnmf -a 0/1/2/3 -k 50 --dimtree 1"
         << "-i Ainput -o nmfoutput -t 10 -p \"3 2\" "
         << "-r \"0.0001 0 0 0.0001\" " << std::endl;
    // mpirun -np 12 distnmf algotype lowrank Afile nmfoutput numIteration pr pc
    // s
    INFO << "Usage 4: mpirun -np 6 distnmf -a 0/1/2/3 -k 50 --dimtree 1"
         << "-i Ainput -o nmfoutput -t 10 -p \"3 2\" --sparsity=0.3"
         << "-r \"0.0001 0 0 0.0001\" " << std::endl;
  }
  /// returns the low rank. Passed as parameter --lowrank or -k
  UWORD lowrankk() { return m_k; }
  /// return global rows. Passed as parameter -d
  UWORD globalm() { return m_globalm; }
  //// returns the global columns. Passed as parameter -d
  UWORD globaln() { return m_globaln; }
  /**
   * L2 regularization as the first parameter and L1 as second 
   * for left lowrank factor W. Passed as parameter --regularizer 
   * with pair of values in double quotes for W and H "l2W l1W l2H l1H"
   */  
  FVEC regW() { return m_regW; }
  /**
   * L2 regularization as the first parameter and L1 as second 
   * for right lowrank factor H. Passed as parameter --regularizer 
   * with pair of values in double quotes for W and H "l2W l1W l2H l1H"
   */  
  FVEC regH() { return m_regH; }
  /// Returns the NMF algorithm to run. Passed as parameter --algo or -a
  algotype lucalgo() { return m_lucalgo; }
  /**
   *  Returns the process grid configuration. 
   * Passed as parameter --processors or -p
   */

  UVEC processor_grids() { return m_proc_grids; }
  /**
   * Returns the vector regularizers for all the modes.
   * It will 2 times the mode values. The first entry is
   * L2 regularization and second value is L1 for every mode. 
   * Passed as parameter --regularizers "L2 L1" for every mode.
   */
  FVEC regularizers() { return m_regularizers; }
  /**
   *  Returns vector of dimensions for every mode.
   * Passed as parameter -d or --dimensions
   */
  UVEC dimensions() { return m_dimensions; }
  int num_k_blocks() { return m_num_k_blocks; }
  /// Returns number of iterations. passed as -t or --iter
  int iterations() { return m_num_it; }
  /// Input parameter for generating sparse matrix. Passed as -s or --sparsity
  float sparsity() { return m_sparsity; }
  /// Returns input file name. Passed as -i or --input
  std::string input_file_name() { return m_Afile_name; }
  /**
   * Returns output file name. Passed as -o or --output. 
   * Every mode will appended as _mode. 
   */
  std::string output_file_name() { return m_outputfile_name; }
  /**
   * Returns the number of processor rows. 
   * Used for distributed NMF. The first parameter of -p. 
   */
  int pr() { return m_pr; }
    /**
   * Returns the number of processor columns. 
   * Used for distributed NMF. The second parameter of -p. 
   */
  int pc() { return m_pc; }
  /// Returns number of modes in tensors. For matrix it is two. 
  int num_modes() { return m_num_modes; }
  /**
   * Enable dimension tree or not. By default we use dimension trees
   * for more than three modes. Passed as parameter --dimtree 1
   */
  bool dim_tree() { return m_dim_tree; }
  /// Returns whether to compute error not. Passed as parameter -e or --error
  bool compute_error() { return m_compute_error; }
  /// To column normalize the input matrix.
  normtype input_normalization() { return this->m_input_normalization; }
};  // ParseCommandLine
}  // namespace planc

#endif  // COMMON_PARSECOMMANDLINE_HPP_
