/* Copyright 2016 Ramakrishnan Kannan */
#include <string>
#include "distntfutils.hpp"
#include "mpicomm.hpp"
#include "utils.hpp"

class DistNTFDriver {
  private:
    int m_argc;
    char **m_argv;
    int m_k;
    UVEC m_globaldims;
    std::string m_Afile_name;
    std::string m_outputfile_name;
    int m_num_it;
    UVEC m_proc_grid_dims;
    FVEC m_regW;
    FVEC m_regH;
    distntfalgotype m_ntfalgo;
    float m_sparsity;
    uint m_compute_error;
    int m_num_k_blocks;
    static const int kprimeoffset = 17;

    void printConfig() {
        cout << "a::" << this->m_ntfalgo <<  "::i::" << this->m_Afile_name
             << "::k::" << this->m_k << "::m::" << this->m_globaldims
             << "::t::" << this->m_num_it << "::pr::" << this->m_proc_grid_dims
             << "::error::" << this->m_compute_error
             << "::distio::" << this->m_distio
             << "::regW::" << "l2::" << this->m_regW(0)
             << "::l1::" << this->m_regW(1)
             << "::regH::" << "l2::" << this->m_regH(0)
             << "::l1::" << this->m_regH(1)
             << "::num_k_blocks::" << m_num_k_blocks
             << endl;
    }
    void parseRegularizedParameter(const char *input, FVEC *reg) {
        stringstream ss(input);
        string s;
        int i = 0;
        float temp;
        while (getline(ss, s, ' ')) {
            temp = ::atof(s.c_str());
            (*reg)(i) = temp;
            i++;
        }
    }
    void parseDimensions(const char *input, UVEC *dims) {
        stringstream ss(input);
        string s;
        int i = 0;
        float temp;
        while (getline(ss, s, ' ')) {
            temp = ::atoi(s.c_str());
            (*dims)(i) = temp;
            i++;
        }
    }

  public:
    void parseCommandLine() {
        int opt, long_index;
        this->m_nmfalgo = static_cast<distntfalgotype>(2);  // defaults to ANLS/BPP
        this->m_k = 20;
        this->m_Afile_name = "rand";
        this->m_pr = 0;
        this->m_pc = 0;
        this->m_sparsity = 0.01;
        this->m_num_it = 10;
        this->m_distio = TWOD;
        this->m_compute_error = 0;
        this->m_regW = arma::zeros<FVEC>(2);
        this->m_regH = arma::zeros<FVEC>(2);
        this->m_num_k_blocks = 1;
        while ((opt = getopt_long(this->m_argc, this->m_argv,
                                  "a:i:e:k:m:n:o:t:s:", distnmfopts,
                                  &long_index)) != -1) {
            switch (opt) {
            case 'a' :
                this->m_nmfalgo = static_cast<distalgotype>(atoi(optarg));
                break;
            case 'd':
                parseDimensions(optarg, &this->m_globaldims);
                break;
            case 'i' : {
                std::string temp = std::string(optarg);
                this->m_Afile_name = temp;
            }
            break;
            case 'e':
                this->m_compute_error = atoi(optarg);
                break;

            case 'k':
                this->m_k = atoi(optarg);
                break;
            case 'o': {
                std::string temp = std::string(optarg);
                this->m_outputfile_name = temp;
            }
            break;
            case 's':
                this->m_sparsity = atof(optarg);
                break;
            case 't':
                this->m_num_it = atoi(optarg);
                break;
            case 'p':
                parseDimensions(optarg, &this->m_proc_grid_dims);
                break;
            case REGWFLAG:
                parseRegularizedParameter(optarg, &this->m_regW);
                break;
            case REGHFLAG:
                parseRegularizedParameter(optarg, &this->m_regH);
                break;
            case NUMKBLOCKS:
                this->m_num_k_blocks = atoi(optarg);
                break;
            default:
                cout << "failed while processing argument:" << optarg << endl;
                print_usage();
                exit(EXIT_FAILURE);
            }
        }
        printConfig();
    }
    DistNTFDriver(int argc, char *argv[]) {
        this->m_argc = argc;
        this->m_argv = argv;
    }

    void print_usage() {
        INFO << "for short arguments like -i no equals sign"
             << "for long arguments like --pr give key=value pair"
             << "-a 0 for MU, 1-HALS, 2-ANLSBPP" << endl;
        // mpirun -np 12 distnmf algotype lowrank m n numIteration pr pc
        INFO << "Usage 1: mpirun -np 6 distnmf -a 0/1/2/3 -k 50"
             <<  "-i rand_uniform/rand_normal/rand_lowrank "
             << "-m \"3 4 5 6\" -t 10 -p\" 2 2 2 2 \" "
             <<  "--regW=\"0.0001 0\" --regH=\"0 0.0001\""  << endl;
        // mpirun -np 12 distnmf algotype lowrank AfileName numIteration pr pc
        // INFO << "Usage 1: mpirun -np 6 distnmf -a 0/1/2/3 -k 50"
        //      <<  "-i Ainput -t 10 --pr 3 --pc 2"
        //      <<  "--regW=\"0.0001 0\" --regH=\"0 0.0001\""  << endl;
        // mpirun -np 12 distnmf algotype lowrank Afile nmfoutput numIteration pr pc
        // INFO << "Usage 1: mpirun -np 6 distnmf -a 0/1/2/3 -k 50"
        //      <<  "-i Ainput -o nmfoutput -t 10 --pr 3 --pc 2"
        //      <<  "--regW=\"0.0001 0\" --regH=\"0 0.0001\""  << endl;
        // mpirun -np 12 distnmf algotype lowrank Afile nmfoutput numIteration pr pc s
        // INFO << "Usage 1: mpirun -np 6 distnmf -a 0/1/2/3 -k 50"
        //      <<  "-i Ainput -o nmfoutput -t 10 --pr 3 --pc 2 --sparsity=0.3"
        //      <<  "--regW=\"0.0001 0\" --regH=\"0 0.0001\""  << endl;
    }
};

int main(int argc, char* argv[]) {
    DistNTFDriver dnd(argc, argv);
    dnd.parseCommandLine();
}
