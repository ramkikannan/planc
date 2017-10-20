/* Copyright 2016 Ramakrishnan Kannan */
#include <string>
#include "distutils.hpp"
#include "distntfio.hpp"
#include "distntfmpicomm.hpp"
#include "utils.hpp"
#include "parsecommandline.hpp"
#include "distauntf.hpp"

class DistNTF {
  private:
    int m_argc;
    char **m_argv;
    int m_k;
    std::string m_Afile_name;
    std::string m_outputfile_name;
    int m_num_it;
    UVEC m_proc_grids;
    FVEC m_regs;
    algotype m_ntfalgo;
    double m_sparsity;
    uint m_compute_error;
    int m_num_k_blocks;
    UVEC m_global_dims;
    UVEC m_local_dims;
    static const int kprimeoffset = 17;

    void printConfig() {
        cout << "a::" << this->m_ntfalgo << "::i::" << this->m_Afile_name
             << "::k::" << this->m_k << "::dims::" << this->m_global_dims
             << "::t::" << this->m_num_it
             << "::proc_grids::" << this->m_proc_grids
             << "::error::" << this->m_compute_error
             << "::regs::" << this->m_regs
             << "::num_k_blocks::" << m_num_k_blocks
             << std::endl;
    }

    void callDistNTF() {
        std::string rand_prefix("rand_");
        planc::NTFMPICommunicator mpicomm(this->m_argc, this->m_argv,
                                          this->m_global_dims);
        planc::DistNTFIO<planc::Tensor> dio(mpicomm);
        if (m_Afile_name.compare(0, rand_prefix.size(), rand_prefix) == 0) {
            dio.readInput(m_Afile_name, this->m_global_dims, this->m_k,
                          this->m_sparsity, this->m_proc_grids);
        } else {
            dio.readInput(m_Afile_name);
        }
        planc::Tensor A(dio.A());
        if (m_Afile_name.compare(0, rand_prefix.size(), rand_prefix) != 0) {
            UVEC local_dims = A.dimensions();
            /*MPI_Allreduce(&localm, &(this->m_globalm), 1, MPI_INT,
             *            MPI_SUM, mpicomm.commSubs()[0]);
             * MPI_Allreduce(&localn, &(this->m_globaln), 1, MPI_INT,
             *            MPI_SUM, mpicomm.commSubs()[1]);*/
            this->m_global_dims = local_dims % this->m_proc_grids;
        }
        INFO << mpicomm.rank() << "::Completed generating tensor A="
             << A.dimensions()
             << "::global dims::" << this->m_global_dims
             << std::endl;
# ifdef WRITE_RAND_INPUT
        dio.writeRandInput();
#endif  // ifdef WRITE_RAND_INPUT
        // same matrix as only one of them will be used.

        // sometimes for really very large matrices starting w/
        // rand initialization hurts ANLS BPP running time. For a better
        // initializer we run couple of iterations of HALS.
#ifdef MPI_VERBOSE
        INFO << mpicomm.rank() << "::" << __PRETTY_FUNCTION__ << "::" \
             << factors.printinfo() << std::endl;
#endif  // ifdef MPI_VERBOSE
        MPI_Barrier(MPI_COMM_WORLD);
        memusage(mpicomm.rank(), "b4 constructor ");
        // TODO: I was here. Need to modify the reallocations by
        // using localOwnedRowCount instead of m_globalm.
        planc::DistAUNTF ntfsolver(A, this->m_k, this->m_ntfalgo,
                                   this->m_global_dims, mpicomm);


        memusage(mpicomm.rank(), "after constructor ");
        ntfsolver.num_iterations(this->m_num_it);
        ntfsolver.compute_error(this->m_compute_error);
        ntfsolver.regularizers(this->m_regs);
        MPI_Barrier(MPI_COMM_WORLD);
        try {
            mpitic();
            ntfsolver.computeNTF();
            double temp = mpitoc();

            if (mpicomm.rank() == 0) printf("NMF took %.3lf secs.\n", temp);
        } catch (std::exception& e) {
            printf("Failed rank %d: %s\n", mpicomm.rank(), e.what());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    void parseRegularizedParameter(const char *input, FVEC *reg) {
        stringstream ss(input);
        string s;
        int    i = 0;
        double  temp;

        while (getline(ss, s, ' ')) {
            temp      = ::atof(s.c_str());
            (*reg)(i) = temp;
            i++;
        }
    }
    void parseCommandLine() {
        PLANC::ParseCommandLine pc(this->m_argc, this->m_argv);
        pc.parseplancopts();
        this->m_ntfalgo       = pc.lucalgo();
        this->m_k             = pc.lowrankk();
        this->m_Afile_name    = pc.input_file_name();
        this->m_proc_grids    = pc.processor_grids();
        this->m_sparsity      = pc.sparsity();
        this->m_num_it        = pc.iterations();
        this->m_distio        = TWOD;
        this->m_regW          = pc.regW();
        this->m_regH          = pc.regH();
        this->m_num_k_blocks  = 1;
        this->m_global_dims   = pc.dimensions();
        this->m_compute_error = pc.compute_error();
        if (this->m_ntfalgo == NAIVEANLSBPP) {
            this->m_distio = ONED_DOUBLE;
        } else {
            this->m_distio = TWOD;
        }
        printConfig();
        callDistNTF();
    }

  public:
    DistNTF(int argc, char *argv[]) {
        this->m_argc = argc;
        this->m_argv = argv;
        this->parseCommandLine();
    }

    void print_usage() {
        INFO << std::endl;
        INFO << "distnmf usage:" << std::endl;
        INFO << "for short arguments like -i do not use equals sign, eg -t 10"
             << std::endl
             << "for long arguments like --pr give key=value pair, eg --pr=4"
             << std::endl
             << "algorithm codes 0-MU2D, 1-HALS, 2-ANLSBPP2D, 3-NAIVEANLSBPP"
             << std::endl;
        // mpirun -np 12 distnmf algotype lowrank m n numIteration pr pc
        INFO << "Usage 1: mpirun -np 6 distnmf -a 0/1/2/3 -k 50"
             << "-i rand_uniform/rand_normal/rand_lowrank "
             << "-m 21600 -n 14400 -t 10 --pr 3 --pc 2"
             << "--regW=\"0.0001 0\" --regH=\"0 0.0001\"" << std::endl;
        // mpirun -np 12 distnmf algotype lowrank AfileName numIteration pr pc
        INFO << "Usage 2: mpirun -np 6 distnmf -a 0/1/2/3 -k 50"
             << "-i Ainput -t 10 --pr 3 --pc 2"
             << "--regW=\"0.0001 0\" --regH=\"0 0.0001\"" << std::endl;
        // mpirun -np 12 distnmf algotype lowrank Afile nmfoutput numIteration pr pc
        INFO << "Usage 3: mpirun -np 6 distnmf -a 0/1/2/3 -k 50"
             << "-i Ainput -o nmfoutput -t 10 --pr 3 --pc 2"
             << "--regW=\"0.0001 0\" --regH=\"0 0.0001\"" << std::endl;
        // mpirun -np 12 distnmf algotype lowrank Afile nmfoutput numIteration pr pc s
        INFO << "Usage 4: mpirun -np 6 distnmf -a 0/1/2/3 -k 50"
             << "-i Ainput -o nmfoutput -t 10 --pr 3 --pc 2 --sparsity=0.3"
             << "--regW=\"0.0001 0\" --regH=\"0 0.0001\"" << std::endl;
    }
};

int main(int argc, char *argv[]) {
    DistNTF dnd(argc, argv);
    fflush(stdout);
}
