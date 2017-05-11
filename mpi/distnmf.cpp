/* Copyright 2016 Ramakrishnan Kannan */
#include <string>
#include "distutils.hpp"
#include "distio.hpp"
#include "distanlsbpp.hpp"
#include "distmu.hpp"
#include "disthals.hpp"
#include "mpicomm.hpp"
#include "naiveanlsbpp.hpp"
#include "utils.hpp"

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
    distalgotype m_nmfalgo;
    float m_sparsity;
    iodistributions m_distio;
    uint m_compute_error;
    int m_num_k_blocks;
    static const int kprimeoffset = 17;

    void printConfig() {
        cout << "a::" << this->m_nmfalgo <<  "::i::" << this->m_Afile_name
             << "::k::" << this->m_k << "::m::" << this->m_globalm
             << "::n::" << this->m_globaln << "::t::" << this->m_num_it
             << "::pr::" << this->m_pr << "::pc::" << this->m_pc
             << "::error::" << this->m_compute_error
             << "::distio::" << this->m_distio
             << "::regW::" << "l2::" << this->m_regW(0)
             << "::l1::" << this->m_regW(1)
             << "::regH::" << "l2::" << this->m_regH(0)
             << "::l1::" << this->m_regH(1)
             << "::num_k_blocks::" << m_num_k_blocks
             << endl;
    }
    template<class NMFTYPE>
    void callDistNMF1D() {
        std::string rand_prefix("rand_");
        MPICommunicator mpicomm(this->m_argc, this->m_argv);
#ifdef BUILD_SPARSE
        DistIO<SP_FMAT> dio(mpicomm, m_distio);
#else
        DistIO<FMAT> dio(mpicomm, m_distio);
#endif
        if (m_Afile_name.compare(0, rand_prefix.size(), rand_prefix) == 0) {
            dio.readInput(m_Afile_name, this->m_globalm,
                          this->m_globaln, this->m_k, this->m_sparsity,
                          this->m_pr, this->m_pc);
        } else {
            dio.readInput(m_Afile_name);
        }
#ifdef BUILD_SPARSE
        // SP_FMAT Arows(dio.Arows().row_indices, dio.Arows().col_ptrs,
        //               dio.Arows().values,
        //               dio.Arows().n_rows, dio.Arows().n_cols);
        // SP_FMAT Acols(dio.Acols().row_indices, dio.Acols().col_ptrs,
        //               dio.Acols().values,
        //               dio.Acols().n_rows, dio.Acols().n_cols);
        SP_FMAT Arows(dio.Arows());
        SP_FMAT Acols(dio.Acols());
#else
        FMAT Arows(dio.Arows());
        FMAT Acols(dio.Acols());
#endif
        if (m_Afile_name.compare(0, rand_prefix.size(), rand_prefix) != 0) {
            this->m_globaln = Arows.n_cols;
            this->m_globalm = Acols.n_rows;
        }
        INFO << mpicomm.rank() <<  "::Completed generating 1D rand Arows="
             << PRINTMATINFO(Arows) << "::Acols="
             << PRINTMATINFO(Acols) << endl;
#ifdef WRITE_RAND_INPUT
        dio.writeRandInput();
#endif
        arma::arma_rng::set_seed(random_sieve(mpicomm.rank() + kprimeoffset));
        FMAT W = arma::randu<FMAT>(this->m_globalm / mpicomm.size(), this->m_k);
        FMAT H = arma::randu<FMAT>(this->m_globaln / mpicomm.size(), this->m_k);
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
        nmfAlgorithm.computeNMF();
        if (!m_outputfile_name.empty()) {
            dio.writeOutput(nmfAlgorithm.getLeftLowRankFactor(),
                            nmfAlgorithm.getRightLowRankFactor(),
                            m_outputfile_name);
        }
    }

    template <class NMFTYPE>
    void callDistNMF2D() {
        std::string rand_prefix("rand_");
        MPICommunicator mpicomm(this->m_argc, this->m_argv,
                                this->m_pr, this->m_pc);
#ifdef USE_PACOSS
        std::string dim_part_file_name = this->m_Afile_name;
        dim_part_file_name += ".dpart.part" + std::to_string(mpicomm.rank());
        this->m_Afile_name += ".part" + std::to_string(mpicomm.rank());
        Pacoss_SparseStruct<float> ss;
        ss.load(m_Afile_name.c_str());
        std::vector<std::vector<Pacoss_IntPair>> dim_part;
        Pacoss_Communicator<float>::loadDistributedDimPart(dim_part_file_name.c_str(), dim_part);
        Pacoss_Communicator<float> *rowcomm = new Pacoss_Communicator<float>(MPI_COMM_WORLD, ss._idx[0], dim_part[0]);
        Pacoss_Communicator<float> *colcomm = new Pacoss_Communicator<float>(MPI_COMM_WORLD, ss._idx[1], dim_part[1]);
        this->m_globalm = ss._dimSize[0];
        this->m_globaln = ss._dimSize[1];
        arma::umat locations(2, ss._idx[0].size());
        for (Pacoss_Int i = 0; i < ss._idx[0].size(); i++) {
          locations(0,i) = ss._idx[0][i];
          locations(1,i) = ss._idx[1][i];
        }
        arma::fvec values(ss._idx[0].size());
        for (Pacoss_Int i = 0; i < values.size(); i++) { values[i] = ss._val[i]; }
        SP_FMAT A(locations, values); A.resize(rowcomm->localRowCount(), colcomm->localRowCount());
#else
        if (this->m_pr > 0 && this->m_pc > 0
                && this->m_pr * this->m_pc != mpicomm.size()) {
            ERR << "pr*pc is not MPI_SIZE" << endl;
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
#ifdef BUILD_SPARSE
        UWORD nnz;
        DistIO<SP_FMAT> dio(mpicomm, m_distio);
        if (mpicomm.rank() == 0) {
            INFO << "sparse case" << endl;
        }
#else
        DistIO<FMAT> dio(mpicomm, m_distio);
#endif
        if (m_Afile_name.compare(0, rand_prefix.size(), rand_prefix) == 0) {
            dio.readInput(m_Afile_name, this->m_globalm,
                          this->m_globaln, this->m_k, this->m_sparsity,
                          this->m_pr, this->m_pc);
        } else {
            dio.readInput(m_Afile_name);
        }
#ifdef BUILD_SPARSE
        // SP_FMAT A(dio.A().row_indices, dio.A().col_ptrs, dio.A().values,
        //           dio.A().n_rows, dio.A().n_cols);
        SP_FMAT A(dio.A());
#else
        FMAT A(dio.A());
#endif
        if (m_Afile_name.compare(0, rand_prefix.size(), rand_prefix) != 0) {
            UWORD localm = A.n_rows;
            UWORD localn = A.n_cols;
            /*MPI_Allreduce(&localm, &(this->m_globalm), 1, MPI_INT,
                          MPI_SUM, mpicomm.commSubs()[0]);
            MPI_Allreduce(&localn, &(this->m_globaln), 1, MPI_INT,
                          MPI_SUM, mpicomm.commSubs()[1]);*/
            this->m_globalm = localm * m_pr;
            this->m_globaln = localn * m_pc;
        }
        INFO << mpicomm.rank() <<  "::Completed generating 2D random matrix A="
             << PRINTMATINFO(A)
             << "::globalm::" << this->m_globalm
             << "::globaln::" << this->m_globaln
             << endl;
#ifdef WRITE_RAND_INPUT
        dio.writeRandInput();
#endif
#endif
        // don't worry about initializing with the
        // same matrix as only one of them will be used.
        arma::arma_rng::set_seed(mpicomm.rank());
#ifdef USE_PACOSS
        FMAT W = arma::randu<FMAT>(rowcomm->localOwnedRowCount(), this->m_k);
        FMAT H = arma::randu<FMAT>(colcomm->localOwnedRowCount(), this->m_k);
#else
        FMAT W = arma::randu<FMAT >(this->m_globalm / mpicomm.size(), this->m_k);
        FMAT H = arma::randu<FMAT >(this->m_globaln / mpicomm.size(), this->m_k);
#ifdef BUILD_SPARSE
        // sometimes for really very large matrices starting w/
        // rand initialization hurts ANLS BPP running time. For a better
        // initializer we run couple of iterations of HALS.
        if (m_nmfalgo == ANLSBPP2D) {
          DistHALS<SP_FMAT> lrinitializer(A, W, H, mpicomm, this->m_num_k_blocks);
          lrinitializer.num_iterations(4);
          lrinitializer.algorithm(HALS2D);
          lrinitializer.computeNMF();
          W = lrinitializer.getLeftLowRankFactor();
          H = lrinitializer.getRightLowRankFactor();
        }
#endif

#ifdef MPI_VERBOSE
        INFO << mpicomm.rank() << "::" << __PRETTY_FUNCTION__ << "::" \
             << PRINTMATINFO(W) << endl;
        INFO << mpicomm.rank() << "::" << __PRETTY_FUNCTION__ << "::" \
             << PRINTMATINFO(H) << endl;
#endif
        MPI_Barrier(MPI_COMM_WORLD);
        memusage(mpicomm.rank(), "b4 constructor ");
        // TODO: I was here. Need to modify the reallocations by using localOwnedRowCount instead of m_globalm.
        NMFTYPE nmfAlgorithm(A, W, H, mpicomm, this->m_num_k_blocks);
#ifdef USE_PACOSS
        nmfAlgorithm.set_rowcomm(rowcomm);
        nmfAlgorithm.set_colcomm(colcomm);
#endif
        memusage(mpicomm.rank(), "after constructor ");
        nmfAlgorithm.num_iterations(this->m_num_it);
        nmfAlgorithm.compute_error(this->m_compute_error);
        nmfAlgorithm.algorithm(this->m_nmfalgo);
        nmfAlgorithm.regW(this->m_regW);
        nmfAlgorithm.regH(this->m_regH);
        MPI_Barrier(MPI_COMM_WORLD);
        try {
          mpitic();
          nmfAlgorithm.computeNMF();
          double temp = mpitoc();
          if (mpicomm.rank() == 0) { printf("NMF took %.3lf secs.\n", temp); }
        } catch (std::exception &e) {
          printf("Failed rank %d\n", mpicomm.rank());
          MPI_Abort(MPI_COMM_WORLD, 1);
        }
#ifndef USE_PACOSS
        if (!m_outputfile_name.empty()) {
            dio.writeOutput(nmfAlgorithm.getLeftLowRankFactor(),
                            nmfAlgorithm.getRightLowRankFactor(),
                            m_outputfile_name);
        }
#endif
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

    void parseCommandLine() {
        int opt, long_index;
        this->m_nmfalgo = static_cast<distalgotype>(2);  // defaults to ANLS/BPP
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
            case 'm':
                this->m_globalm = atoi(optarg);
                break;
            case 'n':
                this->m_globaln = atoi(optarg);
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
                cout << "failed while processing argument:" << optarg << endl;
                print_usage();
                exit(EXIT_FAILURE);
            }
        }
        if (this->m_nmfalgo == NAIVEANLSBPP) {
            this->m_distio = ONED_DOUBLE;
        } else {
            this->m_distio = TWOD;
        }
        printConfig();
        switch (this->m_nmfalgo) {
        case MU2D:
#ifdef BUILD_SPARSE
            callDistNMF2D<DistMU<SP_FMAT> >();
#else
            callDistNMF2D<DistMU<FMAT> >();
#endif
            break;
        case HALS2D:
#ifdef BUILD_SPARSE
            callDistNMF2D<DistHALS<SP_FMAT> >();
#else
            callDistNMF2D<DistHALS<FMAT> >();
#endif
            break;
        case ANLSBPP2D:
#ifdef BUILD_SPARSE
            callDistNMF2D<DistANLSBPP<SP_FMAT> >();
#else
            callDistNMF2D<DistANLSBPP<FMAT> >();
#endif
            break;
        case NAIVEANLSBPP:
#ifdef BUILD_SPARSE
            callDistNMF1D<DistNaiveANLSBPP<SP_FMAT> >();
#else
            callDistNMF1D<DistNaiveANLSBPP<FMAT> >();
#endif
        }
    }

 public:
    DistNMFDriver(int argc, char *argv[]) {
        this->m_argc = argc;
        this->m_argv = argv;
    }

    void print_usage() {
        INFO << "for short arguments like -i no equals sign"
             << "for long arguments like --pr give key=value pair"
             << "-a 0 for MU2D, 1-HALS2D, 2-ANLSBPP2D, 3-NAIVEANLSBPP " << endl;
        // mpirun -np 12 distnmf algotype lowrank m n numIteration pr pc
        INFO << "Usage 1: mpirun -np 6 distnmf -a 0/1/2/3 -k 50"
             <<  "-i rand_uniform/rand_normal/rand_lowrank "
             << "-m 21600 -n 14400 -t 10 --pr 3 --pc 2"
             <<  "--regW=\"0.0001 0\" --regH=\"0 0.0001\""  << endl;
        // mpirun -np 12 distnmf algotype lowrank AfileName numIteration pr pc
        INFO << "Usage 1: mpirun -np 6 distnmf -a 0/1/2/3 -k 50"
             <<  "-i Ainput -t 10 --pr 3 --pc 2"
             <<  "--regW=\"0.0001 0\" --regH=\"0 0.0001\""  << endl;
        // mpirun -np 12 distnmf algotype lowrank Afile nmfoutput numIteration pr pc
        INFO << "Usage 1: mpirun -np 6 distnmf -a 0/1/2/3 -k 50"
             <<  "-i Ainput -o nmfoutput -t 10 --pr 3 --pc 2"
             <<  "--regW=\"0.0001 0\" --regH=\"0 0.0001\""  << endl;
        // mpirun -np 12 distnmf algotype lowrank Afile nmfoutput numIteration pr pc s
        INFO << "Usage 1: mpirun -np 6 distnmf -a 0/1/2/3 -k 50"
             <<  "-i Ainput -o nmfoutput -t 10 --pr 3 --pc 2 --sparsity=0.3"
             <<  "--regW=\"0.0001 0\" --regH=\"0 0.0001\""  << endl;
    }
};

int main(int argc, char* argv[]) {
    DistNMFDriver dnd(argc, argv);
    dnd.parseCommandLine();
}
