/* Copyright 2016 Ramakrishnan Kannan */
#include <string>
#include "distutils.hpp"
#include "distio.hpp"
#include "distanlsbpp.hpp"
#include "distmu.hpp"
#include "disthals.hpp"
#include "distaoadmm.hpp"
#include "mpicomm.hpp"
#include "naiveanlsbpp.hpp"
#include "utils.hpp"
#include "parsecommandline.hpp"

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
    algotype m_nmfalgo;
    double m_sparsity;
    iodistributions m_distio;
    uint m_compute_error;
    int m_num_k_blocks;
    static const int kprimeoffset = 17;
    normtype m_input_normalization;

    void printConfig() {
        cout  << "a::" << this->m_nmfalgo << "::i::" << this->m_Afile_name
              << "::k::" << this->m_k << "::m::" << this->m_globalm
              << "::n::" << this->m_globaln << "::t::" << this->m_num_it
              << "::pr::" << this->m_pr << "::pc::" << this->m_pc
              << "::error::" << this->m_compute_error
              << "::distio::" << this->m_distio
              << "::regW::" << "l2::" << this->m_regW(0)
              << "::l1::" << this->m_regW(1)
              << "::regH::" << "l2::" << this->m_regH(0)
              << "::l1::" << this->m_regH(1)
              << "::num_k_blocks::" << this->m_num_k_blocks
              << "::normtype::" << this->m_input_normalization
              << std::endl;
    }

    template<class NMFTYPE>
    void callDistNMF1D() {
        std::string rand_prefix("rand_");
        MPICommunicator mpicomm(this->m_argc, this->m_argv);
#ifdef BUILD_SPARSE
        DistIO<SP_MAT> dio(mpicomm, m_distio);
#else  // ifdef BUILD_SPARSE
        DistIO<MAT> dio(mpicomm, m_distio);
#endif  // ifdef BUILD_SPARSE

        if (m_Afile_name.compare(0, rand_prefix.size(), rand_prefix) == 0) {
            dio.readInput(m_Afile_name, this->m_globalm,
                          this->m_globaln, this->m_k, this->m_sparsity,
                          this->m_pr, this->m_pc, this->m_input_normalization);
        } else {
            dio.readInput(m_Afile_name);
        }
#ifdef BUILD_SPARSE
        SP_MAT Arows(dio.Arows());
        SP_MAT Acols(dio.Acols());
#else  // ifdef BUILD_SPARSE
        MAT Arows(dio.Arows());
        MAT Acols(dio.Acols());
#endif  // ifdef BUILD_SPARSE

        if (m_Afile_name.compare(0, rand_prefix.size(), rand_prefix) != 0) {
            this->m_globaln = Arows.n_cols;
            this->m_globalm = Acols.n_rows;
        }
        INFO << mpicomm.rank() << "::Completed generating 1D rand Arows="
             << PRINTMATINFO(Arows) << "::Acols="
             << PRINTMATINFO(Acols) << std::endl;
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
        } catch (std::exception& e) {
            printf("Failed rank %d: %s\n", mpicomm.rank(), e.what());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (!m_outputfile_name.empty()) {
            dio.writeOutput(nmfAlgorithm.getLeftLowRankFactor(),
                            nmfAlgorithm.getRightLowRankFactor(),
                            m_outputfile_name);
        }
    }

    template<class NMFTYPE>
    void callDistNMF2D() {
        std::string rand_prefix("rand_");
        MPICommunicator mpicomm(this->m_argc, this->m_argv,
                                this->m_pr, this->m_pc);
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
            dim_part_file_name.c_str(),
            dim_part);
        Pacoss_Communicator<double> *rowcomm = new Pacoss_Communicator<double>(
            MPI_COMM_WORLD,
            ss._idx[0],
            dim_part[0]);
        Pacoss_Communicator<double> *colcomm = new Pacoss_Communicator<double>(
            MPI_COMM_WORLD,
            ss._idx[1],
            dim_part[1]);
        this->m_globalm = ss._dimSize[0];
        this->m_globaln = ss._dimSize[1];
        arma::umat locations(2, ss._idx[0].size());

        for (Pacoss_Int i = 0; i < ss._idx[0].size(); i++) {
            locations(0, i) = ss._idx[0][i];
            locations(1, i) = ss._idx[1][i];
        }
        arma::vec values(ss._idx[0].size());

        for (Pacoss_Int i = 0; i < values.size(); i++) values[i] = ss._val[i];
        SP_MAT A(locations, values); A.resize(
            rowcomm->localRowCount(), colcomm->localRowCount());
#else  // ifdef USE_PACOSS

        if ((this->m_pr > 0) && (this->m_pc > 0)
                && (this->m_pr * this->m_pc != mpicomm.size())) {
            ERR << "pr*pc is not MPI_SIZE" << std::endl;
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
# ifdef BUILD_SPARSE
        UWORD nnz;
        DistIO<SP_MAT> dio(mpicomm, m_distio);

        if (mpicomm.rank() == 0) {
            INFO << "sparse case" << std::endl;
        }
# else   // ifdef BUILD_SPARSE
        DistIO<MAT> dio(mpicomm, m_distio);
# endif  // ifdef BUILD_SPARSE. One outstanding PACOSS

        if (m_Afile_name.compare(0, rand_prefix.size(), rand_prefix) == 0) {
            dio.readInput(m_Afile_name, this->m_globalm,
                          this->m_globaln, this->m_k, this->m_sparsity,
                          this->m_pr, this->m_pc, this->m_input_normalization);
        } else {
            dio.readInput(m_Afile_name);
        }
#ifdef BUILD_SPARSE
        // SP_MAT A(dio.A().row_indices, dio.A().col_ptrs, dio.A().values,
        // dio.A().n_rows, dio.A().n_cols);
        SP_MAT A(dio.A());
# else  // ifdef BUILD_SPARSE
        MAT A(dio.A());
# endif  // ifdef BUILD_SPARSE. One outstanding PACOSS

        if (m_Afile_name.compare(0, rand_prefix.size(), rand_prefix) != 0) {
            UWORD localm = A.n_rows;
            UWORD localn = A.n_cols;

            /*MPI_Allreduce(&localm, &(this->m_globalm), 1, MPI_INT,
             *            MPI_SUM, mpicomm.commSubs()[0]);
             * MPI_Allreduce(&localn, &(this->m_globaln), 1, MPI_INT,
             *            MPI_SUM, mpicomm.commSubs()[1]);*/
            this->m_globalm = localm * m_pr;
            this->m_globaln = localn * m_pc;
        }
# ifdef WRITE_RAND_INPUT
        dio.writeRandInput();
#endif  // ifdef WRITE_RAND_INPUT
#endif  // ifdef USE_PACOSS. Everything over. No more outstanding ifdef's.

        // don't worry about initializing with the
        // same matrix as only one of them will be used.
        arma::arma_rng::set_seed(mpicomm.rank());
#ifdef USE_PACOSS
        MAT W = arma::randu<MAT>(rowcomm->localOwnedRowCount(), this->m_k);
        MAT H = arma::randu<MAT>(colcomm->localOwnedRowCount(), this->m_k);
#else  // ifdef USE_PACOSS
        MAT W = arma::randu<MAT>(this->m_globalm / mpicomm.size(), this->m_k);
        MAT H = arma::randu<MAT>(this->m_globaln / mpicomm.size(), this->m_k);
#endif  // ifdef USE_PACOSS
        // sometimes for really very large matrices starting w/
        // rand initialization hurts ANLS BPP running time. For a better
        // initializer we run couple of iterations of HALS.
#ifndef USE_PACOSS
#ifdef BUILD_SPARSE
        if (m_nmfalgo == ANLSBPP) {
            DistHALS<SP_MAT> lrinitializer(A,
                                           W,
                                           H,
                                           mpicomm,
                                           this->m_num_k_blocks);
            lrinitializer.num_iterations(4);
            lrinitializer.algorithm(HALS);
            lrinitializer.computeNMF();
            W = lrinitializer.getLeftLowRankFactor();
            H = lrinitializer.getRightLowRankFactor();
        }
#endif  // ifdef BUILD_SPARSE
#endif  //ifndef USE_PACOSS

#ifdef MPI_VERBOSE
        INFO << mpicomm.rank() << "::" << __PRETTY_FUNCTION__ << "::" \
             << PRINTMATINFO(W) << std::endl;
        INFO << mpicomm.rank() << "::" << __PRETTY_FUNCTION__ << "::" \
             << PRINTMATINFO(H) << std::endl;
#endif  // ifdef MPI_VERBOSE
        MPI_Barrier(MPI_COMM_WORLD);
        memusage(mpicomm.rank(), "b4 constructor ");
        // TODO: I was here. Need to modify the reallocations by using localOwnedRowCount instead of m_globalm.
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
        MPI_Barrier(MPI_COMM_WORLD);
        try {
            mpitic();
            nmfAlgorithm.computeNMF();
            double temp = mpitoc();

            if (mpicomm.rank() == 0) printf("NMF took %.3lf secs.\n", temp);
        } catch (std::exception& e) {
            printf("Failed rank %d: %s\n", mpicomm.rank(), e.what());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
#ifndef USE_PACOSS
        if (!m_outputfile_name.empty()) {
            dio.writeOutput(nmfAlgorithm.getLeftLowRankFactor(),
                            nmfAlgorithm.getRightLowRankFactor(),
                            m_outputfile_name);
        }
#endif  // ifndef USE_PACOSS
    }
    void parseCommandLine() {
        planc::ParseCommandLine pc(this->m_argc, this->m_argv);
        pc.parseplancopts();
        this->m_nmfalgo       = pc.lucalgo();
        this->m_k             = pc.lowrankk();
        this->m_Afile_name    = pc.input_file_name();
        this->m_pr            = pc.pr();
        this->m_pc            = pc.pc();
        this->m_sparsity      = pc.sparsity();
        this->m_num_it        = pc.iterations();
        this->m_distio        = TWOD;
        this->m_regW          = pc.regW();
        this->m_regH          = pc.regH();
        this->m_num_k_blocks  = 1;
        this->m_globalm       = pc.globalm();
        this->m_globaln       = pc.globaln();
        this->m_compute_error = pc.compute_error();
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
#else  // ifdef BUILD_SPARSE
            callDistNMF2D<DistMU<MAT> >();
#endif  // ifdef BUILD_SPARSE
            break;
        case HALS:
#ifdef BUILD_SPARSE
            callDistNMF2D<DistHALS<SP_MAT> >();
#else  // ifdef BUILD_SPARSE
            callDistNMF2D<DistHALS<MAT> >();
#endif  // ifdef BUILD_SPARSE
            break;
        case ANLSBPP:
#ifdef BUILD_SPARSE
            callDistNMF2D<DistANLSBPP<SP_MAT> >();
#else  // ifdef BUILD_SPARSE
            callDistNMF2D<DistANLSBPP<MAT> >();
#endif  // ifdef BUILD_SPARSE
            break;
        case NAIVEANLSBPP:
#ifdef BUILD_SPARSE
            callDistNMF1D<DistNaiveANLSBPP<SP_MAT> >();
#else  // ifdef BUILD_SPARSE
            callDistNMF1D<DistNaiveANLSBPP<MAT> >();
#endif  // ifdef BUILD_SPARSE
            break;
        case AOADMM:
#ifdef BUILD_SPARSE
            callDistNMF2D<DistAOADMM<SP_MAT> >();
#else  // ifdef BUILD_SPARSE
            callDistNMF2D<DistAOADMM<MAT> >();
#endif  // ifdef BUILD_SPARSE
        }
    }

  public:
    DistNMFDriver(int argc, char *argv[]) {
        this->m_argc = argc;
        this->m_argv = argv;
        this->parseCommandLine();
    }
};

int main(int argc, char *argv[]) {
    try {
        DistNMFDriver dnd(argc, argv);
        fflush(stdout);
    } catch (const std::exception &e) {
        INFO << "Exception with stack trace " << std::endl;
        INFO << e.what();
    }
}
