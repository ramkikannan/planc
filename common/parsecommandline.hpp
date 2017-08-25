#ifndef COMMON_PARSE_COMMAND_LINE_HPP_
#define COMMON_PARSE_COMMAND_LINE_HPP_

#include <iostream>
#include <armadillo>

namespace PLANC {
class ParseCommandLine {
  private:
    const int m_argc;
    const char *m_argv[];
    UWORD m_globalm, m_globaln;
    const factorizationtype m_facttype;

    // file names
    std::string m_Afile_name;
    std::string m_outputfile_name;
    std::string m_init_file_name;

    // nmf related values
    UWORD m_k;
    UWORD m_globalm;
    UWORD m_globaln;
    UWORD m_localm;
    UWORD m_localn;

    //algo related values
    int m_num_it;
    FVEC m_regW;
    FVEC m_regH;
    distalgotype m_nmfalgo;
    float m_sparsity;
    bool m_compute_error;

    //distnmf related values
    distalgotype m_distnmfalgo;
    int m_pr;
    int m_pc;
    int m_num_k_blocks;

    iodistributions m_distio;

    void parseRegularizedParameter(const char *input, FVEC *reg) {
        stringstream ss(input);
        string s;
        int    i = 0;
        float  temp;

        while (getline(ss, s, ' ')) {
            temp      = ::atof(s.c_str());
            (*reg)(i) = temp;
            i++;
        }
    }

    parseDistNMF() {
        while ((opt = getopt_long(this->m_argc, this->m_argv,
                                  "a:i:e:k:m:n:o:t:s:", distnmfopts,
                                  &long_index)) != -1) {
            switch (opt) {
            case 'a':
                this->m_distnmfalgo = static_cast<distalgotype>(atoi(optarg));
                break;
            case 'i': {
                std::string temp = std::string(optarg);
                this->m_Afile_name = temp;
                break;
            }
            case 'e':
                this->m_compute_error = atoi(optarg);
                break;
            case 'k':
                this->m_k = atoi(optarg);
                break;
            case 'm':
                this->m_globalm = atoi(optarg);
                break;
            case 'n':
                this->m_globaln = atoi(optarg);
                break;
            case 'o': {
                std::string temp = std::string(optarg);
                this->m_outputfile_name = temp;
                break;
            }
            case 's':
                this->m_sparsity = atof(optarg);
                break;
            case 't':
                this->m_num_it = atoi(optarg);
                break;
            case PROCROWS:
                this->m_pr = atoi(optarg);
                break;
            case PROCCOLS:
                this->m_pc = atoi(optarg);
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
                cout << "failed while processing argument:" << optarg
                     << std::endl;
                print_usage();
                exit(EXIT_FAILURE);
            }
        }
    }

  public:
    ParseCommandLine(int argc, char *argv[], factorizationtype facttype):
        m_argc(argc), m_argv(argv), m_facttype(facttype) {
    }

    void parse() {
        switch (m_facttype) {
        case FT_NMF:
            parseNMF();
            break;
        case FT_DISTNMF:
            parseDistNMF();
            break;
        case FT_NTF:
            parseNTF();
            break;
        case FT_DISTNTF:
            parseDistNTF();
            break;
        }
    }
    UWORD lowrankk() {return m_k;}
    UWORD globalm() {return m_globalm;}
    UWORD globaln() {return m_globaln;}
    UWORD localm() {return m_localm;}
    UWORD localn() {return m_localn;}
    FVEC regW() {return m_regW;}
    FVEC regH() {return m_regH;}
    distalgotype dist_nmf_algo() {}

    int num_args() {return m_argc;}
    char **arguments() {return m_argv;}


};  // ParseCommandLine


}  // PLANC


#endif  // COMMON_PARSE_COMMAND_LINE_HPP_
