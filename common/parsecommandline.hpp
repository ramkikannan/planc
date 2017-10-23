#ifndef COMMON_PARSE_COMMAND_LINE_HPP_
#define COMMON_PARSE_COMMAND_LINE_HPP_

#include <getopt.h>
#include "parsecommandline.h"
#include <iostream>
#include <sstream>
#include <string>
#include <armadillo>

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
    int     m_num_modes;
    UVEC    m_dimensions;
    UVEC    m_proc_grids;
    FVEC    m_regularizers;

    void parseArrayofString(const char opt, const char *input) {
        std::stringstream ss(input);
        std::string s;
        int    i = 0;
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
                this->m_regularizers(i++) = ::atof(s.c_str());;
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
    ParseCommandLine(int argc, char **argv):
        m_argc(argc), m_argv(argv) {
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
    }

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
            case NORMALIZATION:
                std::string temp = std::string(optarg);
                if (temp.compare("l2") == 0) {
                    this->m_input_normalization = L2;
                } else if (temp.compare("max") == 0) {
                    this->m_input_normalization = MAX;
                }
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

    void printConfig() {
        std::cout << "a::" << this->m_lucalgo << "::i::" << this->m_Afile_name
                  << "::k::" << this->m_k << "::m::" << this->m_globalm
                  << "::n::" << this->m_globaln << "::t::" << this->m_num_it
                  << "::pr::" << this->m_pr << "::pc::" << this->m_pc
                  << "::error::" << this->m_compute_error
                  << "::regW::" << "l2::" << this->m_regW(0)
                  << "::l1::" << this->m_regW(1)
                  << "::regH::" << "l2::" << this->m_regH(0)
                  << "::l1::" << this->m_regH(1)
                  << "::num_k_blocks::" << m_num_k_blocks
                  << "::dimensions::" << this->m_dimensions
                  << "::procs::" << this->m_proc_grids
                  << "::regularizers::" << this->m_regularizers
                  << "::input normalization::" << this->m_input_normalization
                  << std::endl;
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
             << "-i rand_uniform/rand_normal/rand_lowrank "
             << "-d \"21600 14400\" -t 10 -p \"3 2\" "
             << "--normalization \"l2\" "
             << "-r \"0.0001 0 0 0.0001\" " << std::endl;
        // mpirun -np 12 distnmf algotype lowrank AfileName numIteration pr pc
        INFO << "Usage 2: mpirun -np 6 distnmf -a 0/1/2/3 -k 50"
             << "-i Ainput -t 10 -p \"3 2\" "
             << "--normalization \"max\" "
             << "-r \"0.0001 0 0 0.0001\" " << std::endl;
        // mpirun -np 12 distnmf algotype lowrank Afile nmfoutput numIteration pr pc
        INFO << "Usage 3: mpirun -np 6 distnmf -a 0/1/2/3 -k 50"
             << "-i Ainput -o nmfoutput -t 10 -p \"3 2\" "
             << "-r \"0.0001 0 0 0.0001\" " << std::endl;
        // mpirun -np 12 distnmf algotype lowrank Afile nmfoutput numIteration pr pc s
        INFO << "Usage 4: mpirun -np 6 distnmf -a 0/1/2/3 -k 50"
             << "-i Ainput -o nmfoutput -t 10 -p \"3 2\" --sparsity=0.3"
             << "-r \"0.0001 0 0 0.0001\" " << std::endl;
    }

    UWORD lowrankk() {return m_k;}
    UWORD globalm() {return m_globalm;}
    UWORD globaln() {return m_globaln;}
    FVEC regW() {return m_regW;}
    FVEC regH() {return m_regH;}
    algotype lucalgo() {return m_lucalgo;}
    UVEC processor_grids() {return m_proc_grids;}
    FVEC regularizers() {return m_regularizers;}
    UVEC dimensions() {return m_dimensions;}
    int num_k_blocks() {return m_num_k_blocks;}
    int iterations() {return m_num_it;}
    float sparsity() {return m_sparsity;}
    std::string input_file_name() {return m_Afile_name;}
    std::string output_file_name() {return m_outputfile_name;}
    int pr() {return m_pr;}
    int pc() {return m_pc;}
    int num_modes() {return m_num_modes;}
    bool compute_error() {return m_compute_error;}
    normtype input_normalization() {return this->m_input_normalization;}

};  // ParseCommandLine
}  // planc


#endif  // COMMON_PARSE_COMMAND_LINE_HPP_
