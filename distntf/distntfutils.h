/* Copyright 2016 Ramakrishnan Kannan */
#ifndef DISTNTF_DISTNTFUTILS_H_
#define DISTNTF_DISTNTFUTILS_H_

#include "distutils.h"

namespace PLANC {
enum distntfalgotype {NTF_MU, NTF_HALS, NTF_BPP};;

// usage scenarios
// mpirun -np 12 distnmf algotype lowrank AfileName numIteration pr pc
// mpirun -np 12 distnmf algotype lowrank m n numIteration pr pc
// mpirun -np 12 distnmf algotype lowrank Afile  outputfile numIteration pr pc

struct option distntfopts[] = {
    {"input",       required_argument, 0, 'i'},
    {"algo",        required_argument, 0, 'a'},
    {"dimensions",  optional_argument, 0, 'd'},
    {"error",       optional_argument, 0, 'e'},
    {"lowrank",     required_argument, 0, 'k'},
    {"iter",        optional_argument, 0, 't'},
    {"output",      optional_argument, 0, 'o'},
    {"sparsity",    optional_argument, 0, 's'},
    {"processors",  optional_argument, 0, 'p'},
    {"regw",        optional_argument, 0, REGWFLAG},
    {"regh",        optional_argument, 0, REGHFLAG},
    {"numkblocks",  optional_argument, 0, NUMKBLOCKS},
    {0,             0,                 0,  0 }
};
}  // namespace PLANC
#endif  // DISTNTF_DISTNTFUTILS_H_

