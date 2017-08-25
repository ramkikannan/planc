/* Copyright 2016 Ramakrishnan Kannan */

#include "parsecommandline.h"
#include "bppnmf.hpp"
#include "utils.hpp"
#include "hals.hpp"
#include "mu.hpp"
#include <string>
#include <omp.h>

template <class NMFTYPE>
void NMFDriver(int k, UWORD m, UWORD n, std::string AfileName,
               std::string WinitFileName, std::string HinitFileName,
               std::string WfileName, std::string HfileName, int numIt) {
#ifdef BUILD_SPARSE
    SP_MAT A;
    UWORD nnz;
#else
    MAT A;
#endif
    double t1, t2;
    if (!AfileName.empty()) {
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
    if (!WinitFileName.empty()) {
        INFO << "Winitfilename = " << WinitFileName << std::endl;
        W.load(WinitFileName);
        INFO << "Loaded W." << PRINTMATINFO(W) << std::endl;
    }
    if (!HinitFileName.empty()) {
        INFO << "HInitfilename=" << HinitFileName << std::endl;
        H.load(HinitFileName);
        INFO << "Loaded H." << PRINTMATINFO(H) << std::endl;
    }
    if (!WinitFileName.empty()) {
        NMFTYPE nmfAlgorithm(A, W, H);
        nmfAlgorithm.num_iterations(numIt);
        INFO << "completed constructor" << std::endl;
        tic();
        nmfAlgorithm.computeNMF();
        t2 = toc();
        INFO << "time taken:" << t2 << std::endl;
        if (!WfileName.empty()) {
            nmfAlgorithm.getLeftLowRankFactor().save(WfileName, arma::raw_ascii);
        }
        if (!HfileName.empty()) {
            nmfAlgorithm.getRightLowRankFactor().save(HfileName, arma::raw_ascii);
        }
    } else {
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
    algotype nmfalgo = BPP_NMF;
    int lowRank = 50;
    int numIt = 20;
    std::string AfileName;
    std::string WInitfileName;
    std::string HInitfileName;
    std::string WfileName;
    std::string HfileName;
    UWORD m = 0, n = 0;
    int opt, long_index;
    while ((opt = getopt_long(argc, argv, "a:h:i:k:m:n:t:w:", nmfopts,
                              &long_index)) != -1) {
        switch (opt) {
        case 'a' :
            nmfalgo = static_cast<algotype>(atoi(optarg));
            break;
        case 'h' : {
            std::string temp = std::string(optarg);
            HfileName = temp;
        }
        break;
        case 'i' : {
            std::string temp = std::string(optarg);
            AfileName = temp;
        }
        break;
        case 'k':
            lowRank = atoi(optarg);
            break;
        case 'm':
            m = atoi(optarg);
            break;
        case 'n':
            n = atoi(optarg);
            break;
        case 't':
            numIt = atoi(optarg);
            break;
        case 'w': {
            std::string temp = std::string(optarg);
            HfileName = temp;
        }
        break;
        case WINITFLAG: {
            std::string temp = std::string(optarg);
            WInitfileName = temp;
        }
        break;
        case HINITFLAG: {
            std::string temp = std::string(optarg);
            HInitfileName = temp;
        }
        break;
        default:
            print_usage();
            exit(EXIT_FAILURE);
        }
    }

    switch (nmfalgo) {
    case MU_NMF:
#ifdef BUILD_SPARSE
        NMFDriver<MUNMF<SP_MAT > >(lowRank, m, n, AfileName, WInitfileName,
                                   HInitfileName, WfileName, HfileName, numIt);
#else
        NMFDriver<MUNMF<MAT > >(lowRank, m, n, AfileName, WInitfileName,
                                HInitfileName, WfileName, HfileName, numIt);
#endif
        break;
    case HALS_NMF:
#ifdef BUILD_SPARSE
        NMFDriver<HALSNMF<SP_MAT > >(lowRank, m, n, AfileName, WInitfileName,
                                     HInitfileName, WfileName, HfileName, numIt);
#else
        NMFDriver<HALSNMF<MAT > >(lowRank, m, n, AfileName, WInitfileName,
                                  HInitfileName, WfileName, HfileName, numIt);
#endif
        break;
    case BPP_NMF:
#ifdef BUILD_SPARSE
        NMFDriver<BPPNMF<SP_MAT > >(lowRank, m, n, AfileName, WInitfileName,
                                    HInitfileName, WfileName, HfileName, numIt);
#else
        NMFDriver<BPPNMF<MAT > >(lowRank, m, n, AfileName, WInitfileName,
                                 HInitfileName, WfileName, HfileName, numIt);
#endif
        break;
    }
}


int main(int argc, char* argv[]) {
    parseCommandLineandCallNMF(argc, argv);
}
