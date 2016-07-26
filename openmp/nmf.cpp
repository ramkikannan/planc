/* Copyright 2016 Ramakrishnan Kannan */

#include "bppnmf.hpp"
#include "utils.hpp"
#include "hals.hpp"
#include "mu.hpp"
#include <string>
#include <omp.h>

template <class NMFTYPE>
void NMFDriver(int k, uword m, uword n, std::string AfileName,
               std::string WinitFileName, std::string HinitFileName,
               std::string WfileName, std::string HfileName, int numIt) {
#ifdef BUILD_SPARSE
    sp_fmat A;
    uword nnz;
#else
    fmat A;
#endif
    double t1, t2;
    if (!AfileName.empty()) {
#ifdef BUILD_SPARSE
        INFO << "calling load matrix market " << AfileName << endl;
        uword uim, uin, uinnz;
        LoadMatrixMarketFile<sp_fmat, fvec, uword>(AfileName, uim, uin, uinnz, A, false);
        m = uim;
        n = uin;
        nnz = uinnz;
        INFO << "Successfully loaded the input matrix" << endl;
#else
        t1 = omp_get_wtime();
        A.load(AfileName, raw_ascii);
        t2 = omp_get_wtime();
        INFO << "Successfully loaded dense input matrix. A=" << A.n_rows << "x" << A.n_cols << " took=" << t2 - t1 << endl;
        m = A.n_rows;
        n = A.n_cols;
#endif
    } else {
        A = randu<fmat>(m, n);
        INFO << "Completed generating random matrix A=" << A.n_rows << "x" << A.n_cols << endl;
    }
    fmat W, H;
    if (!WinitFileName.empty()) {
        INFO << "Winitfilename = " << WinitFileName << endl;
        W.load(WinitFileName, raw_ascii);
        INFO << "Successfully loaded W. W=" << W.n_rows << "x" << W.n_cols << endl;
    }
    if (!HinitFileName.empty()) {
        INFO << "HInitfilename=" << HinitFileName << endl;
        H.load(HinitFileName, raw_ascii);
        INFO << "Successfully loaded H. H=" << H.n_rows << "x" << H.n_cols << endl;
    }
    if (!WinitFileName.empty()) {
        NMFTYPE nmfAlgorithm (A, W, H);
        nmfAlgorithm.num_iterations(numIt);
        INFO << "completed constructor" << endl;
        t1 = omp_get_wtime();
        nmfAlgorithm.computeNMF();
        t2 = omp_get_wtime();
        INFO << "time taken:" << (t2 - t1) << endl;
        if (!WfileName.empty()) {
            nmfAlgorithm.getLeftLowRankFactor().save(WfileName, raw_ascii);
        }
        if (!HfileName.empty()) {
            nmfAlgorithm.getRightLowRankFactor().save(HfileName, raw_ascii);
        }
    } else {
        NMFTYPE nmfAlgorithm(A, k);
        nmfAlgorithm.num_iterations(numIt);
        INFO << "completed constructor" << " A = " << A.n_rows << "x" << A.n_cols << endl; //" W=" << this->W.n_rows << "x" << this->W.n_cols << " H=" << this->H.n_rows "x" << this->H.n_cols <<  endl;
        t1 = omp_get_wtime();
        nmfAlgorithm.computeNMF();
        t2 = omp_get_wtime();
        INFO << "time taken:" << (t2 - t1) << endl;
        if (!WfileName.empty()) {
            nmfAlgorithm.getLeftLowRankFactor().save(WfileName, raw_ascii);
        }
        if (!HfileName.empty()) {
            nmfAlgorithm.getRightLowRankFactor().save(HfileName, raw_ascii);
        }
    }
}
#ifdef BUILD_SPARSE
void incrementalGraph(std::string AfileName, std::string WfileName) {
    sp_fmat A;
    uword m, n, nnz;
    LoadMatrixMarketFile<sp_fmat, fvec, uword>(AfileName, m, n, nnz, A, false);
    INFO << "m=" << m << " n=" << n << " nnz=" << nnz << endl;
    INFO << "Loaded input matrix A=" << A.n_rows << "x" << A.n_cols << endl;
    fmat W, H;
    W.load(WfileName, raw_ascii);
    INFO << "Loaded input matrix W=" << W.n_rows << "x" << W.n_cols << endl;
    H.ones(A.n_cols, W.n_cols);
    BPPNMF<sp_fmat> bppnmf(A, W, H);
    H = bppnmf.solveScalableNNLS();
    OUTPUT << H << endl;
}
#endif
void print_usage() {
    cout << "Usage1 : NMFLibrary --algo=[0/1/2] --lowrank=20 "
         << "--input=filename --iter=20" << endl;
    cout << "Usage2 : NMFLibrary --algo=[0/1/2] --lowrank=20 "
         << "--rows=20000 --columns=10000 --iter=20" << endl;
    cout << "Usage3 : NMFLibrary --algo=[0/1/2] --lowrank=20 "
         <<  "--input=filename  --winit=filename --hinit=filename "
         <<   "--iter=20" << endl;
    cout << "Usage4 : --algo=[0/1/2] --lowrank=20 "
         << "--input=filename --winit=filename --hinit=filename "
         << "--w=woutputfilename --h=outputfilename --iter=20" << endl;
    cout << "Usage5: NMFLibrary --input=filename" << endl;
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
    uword m = 0, n = 0;
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
        NMFDriver<MUNMF<sp_fmat> >(lowRank, m, n, AfileName, WInitfileName,
                                   HInitfileName, WfileName, HfileName, numIt);
#else
        NMFDriver<MUNMF<fmat> >(lowRank, m, n, AfileName, WInitfileName,
                                HInitfileName, WfileName, HfileName, numIt);
#endif
        break;
    case HALS_NMF:
#ifdef BUILD_SPARSE
        NMFDriver<HALSNMF<sp_fmat> >(lowRank, m, n, AfileName, WInitfileName,
                                     HInitfileName, WfileName, HfileName, numIt);
#else
        NMFDriver<HALSNMF<fmat> >(lowRank, m, n, AfileName, WInitfileName,
                                  HInitfileName, WfileName, HfileName, numIt);
#endif
        break;
    case BPP_NMF:
#ifdef BUILD_SPARSE
        NMFDriver<BPPNMF<sp_fmat> >(lowRank, m, n, AfileName, WInitfileName,
                                    HInitfileName, WfileName, HfileName, numIt);
#else
        NMFDriver<BPPNMF<fmat> >(lowRank, m, n, AfileName, WInitfileName,
                                 HInitfileName, WfileName, HfileName, numIt);
#endif
        break;
    }
}


int main(int argc, char* argv[]) {
    parseCommandLineandCallNMF(argc, argv);
}
