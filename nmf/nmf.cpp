/* Copyright 2016 Ramakrishnan Kannan */

#include "parsecommandline.hpp"
#include "bppnmf.hpp"
#include "utils.hpp"
#include "hals.hpp"
#include "mu.hpp"
#include <string>
#include <omp.h>

template <class NMFTYPE>
void NMFDriver(int k, UWORD m, UWORD n, std::string AfileName,
               std::string WfileName, std::string HfileName, int numIt) {
#ifdef BUILD_SPARSE
    SP_MAT A;
    UWORD nnz;
#else
    MAT A;
#endif
    double t1, t2;
    if (!AfileName.empty() &&
            !AfileName.compare(AfileName.size() - 4, 4, "rand")) {
#ifdef BUILD_SPARSE
        A.load(AfileName, arma::coord_ascii);
        INFO << "Successfully loaded the input matrix" << std::endl;
#else
        tic();
        A.load(AfileName);
        t2 = toc();
        INFO << "Successfully loaded dense input matrix. A=" << PRINTMATINFO(A)
             << " took=" << t2 << std::endl;
        m = A.n_rows;
        n = A.n_cols;
#endif
    } else {
#ifdef BUILD_SPARSE
        A = arma::sprandu<SP_MAT>(m, n, 0.001);
#else
        A = arma::randu<MAT>(m, n);
#endif
        INFO << "generated random matrix A=" << PRINTMATINFO(A) << std::endl;
    }
    MAT W, H;
    NMFTYPE nmfAlgorithm(A, k);
    nmfAlgorithm.num_iterations(numIt);
    INFO << "completed constructor" << PRINTMATINFO(A) << std::endl;
    tic();
    nmfAlgorithm.computeNMF();
    t2 = toc();
    INFO << "time taken:" << t2 << std::endl;
    if (!WfileName.empty()) {
        nmfAlgorithm.getLeftLowRankFactor().save(WfileName,
                arma::raw_ascii);
    }
    if (!HfileName.empty()) {
        nmfAlgorithm.getRightLowRankFactor().save(HfileName,
                arma::raw_ascii);
    }
}
#ifdef BUILD_SPARSE
void incrementalGraph(std::string AfileName, std::string WfileName) {
    SP_MAT A;
    UWORD m, n, nnz;
    A.load(AfileName);
    INFO << "Loaded input matrix A=" << PRINTMATINFO(A) << std::endl;
    MAT W, H;
    W.load(WfileName);
    INFO << "Loaded input matrix W=" << PRINTMATINFO(W) << std::endl;
    H.ones(A.n_cols, W.n_cols);
    BPPNMF<SP_MAT > bppnmf(A, W, H);
    H = bppnmf.solveScalableNNLS();
    OUTPUT << H << std::endl;
}
#endif
void print_usage() {
    cout << "Usage1 : NMFLibrary --algo=[0/1/2] --lowrank=20 "
         << "--input=filename --iter=20" << std::endl;
    cout << "Usage2 : NMFLibrary --algo=[0/1/2] --lowrank=20 "
         << "--rows=20000 --columns=10000 --iter=20" << std::endl;
    cout << "Usage3 : NMFLibrary --algo=[0/1/2] --lowrank=20 "
         <<  "--input=filename  --winit=filename --hinit=filename "
         <<   "--iter=20" << std::endl;
    cout << "Usage4 : --algo=[0/1/2] --lowrank=20 "
         << "--input=filename --winit=filename --hinit=filename "
         << "--w=woutputfilename --h=outputfilename --iter=20" << std::endl;
    cout << "Usage5: NMFLibrary --input=filename" << std::endl;
}
void parseCommandLineandCallNMF(int argc, char *argv[]) {
    PLANC::ParseCommandLine pc(argc, argv);
    pc.parseplancopts();
    switch (pc.nmfalgo()) {
    case MU:
#ifdef BUILD_SPARSE
        NMFDriver<MUNMF<SP_MAT > >(pc.lowrankk(), pc.globalm(), pc.globaln(),
                                   pc.input_file_name(),
                                   pc.output_file_name() + "_w",
                                   pc.output_file_name() + "_h",
                                   pc.iterations());
#else
        NMFDriver<MUNMF<MAT > >(pc.lowrankk(), pc.globalm(), pc.globaln(),
                                pc.input_file_name(),
                                pc.output_file_name() + "_w",
                                pc.output_file_name() + "_h",
                                pc.iterations());
#endif
        break;
    case HALS:
#ifdef BUILD_SPARSE
        NMFDriver<HALSNMF<SP_MAT > >(pc.lowrankk(), pc.globalm(), pc.globaln(),
                                     pc.input_file_name(),
                                     pc.output_file_name() + "_w",
                                     pc.output_file_name() + "_h",
                                     pc.iterations());
#else
        NMFDriver<HALSNMF<MAT > >(pc.lowrankk(), pc.globalm(), pc.globaln(),
                                  pc.input_file_name(),
                                  pc.output_file_name() + "_w",
                                  pc.output_file_name() + "_h",
                                  pc.iterations());
#endif
        break;
    case ANLSBPP:
#ifdef BUILD_SPARSE
        NMFDriver<BPPNMF<SP_MAT > >(pc.lowrankk(), pc.globalm(), pc.globaln(),
                                    pc.input_file_name(),
                                    pc.output_file_name() + "_w",
                                    pc.output_file_name() + "_h",
                                    pc.iterations());
#else
        NMFDriver<BPPNMF<MAT > >(pc.lowrankk(), pc.globalm(), pc.globaln(),
                                 pc.input_file_name(),
                                 pc.output_file_name() + "_w",
                                 pc.output_file_name() + "_h",
                                 pc.iterations());
#endif
        break;
    }
}


int main(int argc, char* argv[]) {
    parseCommandLineandCallNMF(argc, argv);
}
