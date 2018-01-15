/* Copyright Ramakrishnan Kannan 2017 */

#ifndef COMMON_PARSECOMMANDLINE_H_
#define COMMON_PARSECOMMANDLINE_H_

#include "utils.h"
#include <getopt.h>

// usage scenarios
// NMFLibrary algotype lowrank AfileName numIteration
// NMFLibrary algotype lowrank m n numIteration
// NMFLibrary algotype lowrank Afile WInitfile HInitfile numIteration
// NMFLibrary algotype lowrank Afile WInitfile HInitfile WoutputFile HoutputFile numIteration
// #define WINITFLAG 1000
// #define HINITFLAG 1001
// #define REGWFLAG  1002
// #define REGHFLAG  1003

//distnmf related defines
// #define PROCROWS        2002
// #define PROCCOLS        2003
#define NUMKBLOCKS      2004
#define NORMALIZATION   2005
#define DIMTREE         2006 

// enum factorizationtype{FT_NMF, FT_DISTNMF, FT_NTF, FT_DISTNTF};

// struct option nmfopts[] = {
//   {"input",       required_argument, 0, 'i'},
//   {"algo",        required_argument, 0, 'a'},
//   {"lowrank",     optional_argument, 0, 'k'},
//   {"iter",        optional_argument, 0, 't'},
//   {"rows",        optional_argument, 0, 'm'},
//   {"columns",     optional_argument, 0, 'n'},
//   {"winit",       optional_argument, 0, WINITFLAG},
//   {"hinit",       optional_argument, 0, HINITFLAG},
//   {"wout",        optional_argument, 0, 'w'},
//   {"hout",        optional_argument, 0, 'h'},
//   {"regw",        optional_argument, 0, REGWFLAG},
//   {"regh",        optional_argument, 0, REGHFLAG},
//   {0,         0,                 0,  0 }
// };

// usage scenarios
// mpirun -np 12 distnmf algotype lowrank AfileName numIteration pr pc
// mpirun -np 12 distnmf algotype lowrank m n numIteration pr pc
// mpirun -np 12 distnmf algotype lowrank Afile  outputfile numIteration pr pc

// we are not supporting initfiles for distnmf.
// consistent initializations will be distIO
// struct option distnmfopts[] = {
//     {"input",       required_argument, 0, 'i'},
//     {"algo",        required_argument, 0, 'a'},
//     {"error",       optional_argument, 0, 'e'},
//     {"lowrank",     required_argument, 0, 'k'},
//     {"iter",        optional_argument, 0, 't'},
//     {"rows",        optional_argument, 0, 'm'},
//     {"columns",     optional_argument, 0, 'n'},
//     {"output",      optional_argument, 0, 'o'},
//     {"sparsity",    optional_argument, 0, 's'},
//     {"pr",          optional_argument, 0, PROCROWS},
//     {"pc",          optional_argument, 0, PROCCOLS},
//     {"regw",        optional_argument, 0, REGWFLAG},
//     {"regh",        optional_argument, 0, REGHFLAG},
//     {"numkblocks",  optional_argument, 0, NUMKBLOCKS},
//     {0,             0,                 0,  0 }
// };

// usage scenarios
// mpirun -np 12 distnmf algotype lowrank AfileName numIteration pr pc
// mpirun -np 12 distnmf algotype lowrank m n numIteration pr pc
// mpirun -np 12 distnmf algotype lowrank Afile  outputfile numIteration pr pc

struct option plancopts[] = {
    {"algo",          optional_argument, 0, 'a'},
    {"dimensions",    required_argument, 0, 'd'},
    {"error",         optional_argument, 0, 'e'},
    {"input",         required_argument, 0, 'i'},
    {"lowrank",       required_argument, 0, 'k'},    
    {"output",        optional_argument, 0, 'o'},    
    {"processors",    optional_argument, 0, 'p'},
    {"regularizer",   optional_argument, 0, 'r'},
    {"sparsity",      optional_argument, 0, 's'},
    {"iter",          optional_argument, 0, 't'},
    {"numkblocks",    optional_argument, 0, NUMKBLOCKS},
    {"normalization", optional_argument, 0, NORMALIZATION}, 
    {"dimtree",       optional_argument, 0, DIMTREE}, 
    {0,             0,                 0,  0 }
};

#endif  // COMMON_PARSECOMMANDLINE_H_
