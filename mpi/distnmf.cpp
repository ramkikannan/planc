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
    uword m_globalm, m_globaln;
    std::string m_Afile_name;
    std::string m_outputfile_name;
    int m_num_it;
    int m_pr;
    int m_pc;
    distalgotype m_nmfalgo;
    float m_sparsity;
    iodistributions m_distio;
    uint m_compute_error;
    static const int kprimeoffset = 17;

    void printConfig() {
        cout << "a::" << this->m_nmfalgo <<  "::i::" << this->m_Afile_name
             << "::k::" << this->m_k << "::m::" << this->m_globalm
             << "::n::" << this->m_globaln << "::t::" << this->m_num_it
             << "::pr::" << this->m_pr << "::pc::" << this->m_pc
             << "::error::" << this->m_compute_error
             << "::distio::" << this->m_distio << endl;
    }

  public:
    DistNMFDriver(int argc, char *argv[]) {
        this->m_argc = argc;
        this->m_argv = argv;
    }
    template<class NMFTYPE>
    void callDistNMF1D() {
        std::string rand_prefix("rand_");
        MPICommunicator mpicomm(this->m_argc, this->m_argv);
#ifdef BUILD_SPARSE
        DistIO<sp_fmat> dio(mpicomm, m_distio);
#else
        DistIO<fmat> dio(mpicomm, m_distio);
#endif
        if (m_Afile_name.compare(0, rand_prefix.size(), rand_prefix) == 0) {
            dio.readInput(m_Afile_name, this->m_globalm,
                          this->m_globaln, this->m_k, this->m_sparsity,
                          this->m_pr, this->m_pc);
        } else {
            dio.readInput(m_Afile_name);
        }
#ifdef BUILD_SPARSE
        // sp_fmat Arows(dio.Arows().row_indices, dio.Arows().col_ptrs,
        //               dio.Arows().values,
        //               dio.Arows().n_rows, dio.Arows().n_cols);
        // sp_fmat Acols(dio.Acols().row_indices, dio.Acols().col_ptrs,
        //               dio.Acols().values,
        //               dio.Acols().n_rows, dio.Acols().n_cols);
        sp_fmat Arows(dio.Arows());
        sp_fmat Acols(dio.Acols());
#else
        fmat Arows(dio.Arows());
        fmat Acols(dio.Acols());
#endif
        if (m_Afile_name.compare(0, rand_prefix.size(), rand_prefix) != 0) {
            this->m_globaln = Arows.n_cols;
            this->m_globalm = Acols.n_rows;
        }
        INFO << mpicomm.rank() <<  "::Completed generating 1D rand Arows="
             << PRINTMATINFO(Arows) << "::Acols=" << PRINTMATINFO(Acols) << endl;

#ifdef WRITE_RAND_INPUT
        dio.writeRandInput();
#endif
        arma_rng::set_seed(random_sieve(mpicomm.rank() + kprimeoffset));
        fmat W = randu<fmat>(this->m_globalm / mpicomm.size(), this->m_k);
        fmat H = randu<fmat>(this->m_globaln / mpicomm.size(), this->m_k);
        MPI_Barrier(MPI_COMM_WORLD);
        NMFTYPE nmfAlgorithm(Arows, Acols, W, H, mpicomm);
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
        if (this->m_pr > 0 && this->m_pc > 0
                && this->m_pr * this->m_pc != mpicomm.size()) {
            ERR << "pr*pc is not MPI_SIZE" << endl;
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
#ifdef BUILD_SPARSE
        uword nnz;
        DistIO<sp_fmat> dio(mpicomm, m_distio);
        if (mpicomm.rank() == 0) {
            INFO << "sparse case" << endl;
        }
#else
        DistIO<fmat> dio(mpicomm, m_distio);
#endif
        if (m_Afile_name.compare(0, rand_prefix.size(), rand_prefix) == 0) {
            dio.readInput(m_Afile_name, this->m_globalm,
                          this->m_globaln, this->m_k, this->m_sparsity,
                          this->m_pr, this->m_pc);
        } else {
            dio.readInput(m_Afile_name);
        }
#ifdef BUILD_SPARSE
        // sp_fmat A(dio.A().row_indices, dio.A().col_ptrs, dio.A().values,
        //           dio.A().n_rows, dio.A().n_cols);
        sp_fmat A(dio.A());
#else
        fmat A(dio.A());
#endif
        if (m_Afile_name.compare(0, rand_prefix.size(), rand_prefix) != 0) {
            uword localm = A.n_rows;
            uword localn = A.n_cols;
            MPI_Allreduce(&localm, &(this->m_globalm), 1, MPI_INT,
                          MPI_SUM, mpicomm.commSubs()[0]);
            MPI_Allreduce(&localn, &(this->m_globaln), 1, MPI_INT,
                          MPI_SUM, mpicomm.commSubs()[1]);
        }
        INFO << mpicomm.rank() <<  "::Completed generating 2D random matrix A="
             << PRINTMATINFO(A) << endl;
#ifdef WRITE_RAND_INPUT
        dio.writeRandInput();
#endif
        // don't worry about initializing with the
        // same matrix as only one of them will be used.
        arma_rng::set_seed(mpicomm.rank());
        fmat W = randu<fmat>(this->m_globalm / mpicomm.size(), this->m_k);
        fmat H = randu<fmat>(this->m_globaln / mpicomm.size(), this->m_k);
#ifdef MPI_VERBOSE
        INFO << mpicomm.rank() << "::" << __PRETTY_FUNCTION__ << "::" \
             << PRINTMATINFO(W) << endl;
        INFO << mpicomm.rank() << "::" << __PRETTY_FUNCTION__ << "::" \
             << PRINTMATINFO(H) << endl;
#endif
        MPI_Barrier(MPI_COMM_WORLD);
        NMFTYPE nmfAlgorithm(A, W, H, mpicomm);
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
    void print_usage() {
        INFO << "for short arguments like -i no equals sign"
             << "for long arguments like --pr give key=value pair"
             << "-a 0 for MU2D, 1-HALS2D, 2-ANLSBPP2D, 3-NAIVEANLSBPP " << endl;
        // mpirun -np 12 distnmf algotype lowrank m n numIteration pr pc
        INFO << "Usage 1: mpirun -np 6 distnmf -a 0/1/2/3 -k 50"
             <<  "-i rand_uniform/rand_normal/rand_lowrank "
             << "-m 21600 -n 14400 -t 10 --pr 3 --pc 2"  << endl;
        // mpirun -np 12 distnmf algotype lowrank AfileName numIteration pr pc
        INFO << "Usage 1: mpirun -np 6 distnmf -a 0/1/2/3 -k 50"
             <<  "-i Ainput -t 10 --pr 3 --pc 2"  << endl;
        // mpirun -np 12 distnmf algotype lowrank Afile nmfoutput numIteration pr pc
        INFO << "Usage 1: mpirun -np 6 distnmf -a 0/1/2/3 -k 50"
             <<  "-i Ainput -o nmfoutput -t 10 --pr 3 --pc 2"  << endl;
        // mpirun -np 12 distnmf algotype lowrank Afile nmfoutput numIteration pr pc s
        INFO << "Usage 1: mpirun -np 6 distnmf -a 0/1/2/3 -k 50"
             <<  "-i Ainput -o nmfoutput -t 10 --pr 3 --pc 2 --sparsity=0.3"  << endl;
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
            callDistNMF2D<DistMU<sp_fmat> >();
#else
            callDistNMF2D<DistMU<fmat> >();
#endif
            break;
        case HALS2D:
#ifdef BUILD_SPARSE
            callDistNMF2D<DistHALS<sp_fmat> >();
#else
            callDistNMF2D<DistHALS<fmat> >();
#endif
            break;
        case ANLSBPP2D:
#ifdef BUILD_SPARSE
            callDistNMF2D<DistANLSBPP<sp_fmat> >();
#else
            callDistNMF2D<DistANLSBPP<fmat> >();
#endif
            break;
        case NAIVEANLSBPP:
#ifdef BUILD_SPARSE
            callDistNMF1D<DistNaiveANLSBPP<sp_fmat> >();
#else
            callDistNMF1D<DistNaiveANLSBPP<fmat> >();
#endif
        }
    }
};

int main(int argc, char* argv[]) {
    DistNMFDriver dnd(argc, argv);
    dnd.parseCommandLine();
}