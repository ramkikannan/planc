#ifndef PACOSS_PARTITIONER_H
#define PACOSS_PARTITIONER_H

#include "pacoss-common.h"
#include "pacoss-sparse-struct.h"

#define PACOSS_PARTITIONER_DIM_PARTITION_SUFFIX ".dpart"
#define PACOSS_PARTITIONER_NZ_PARTITION_SUFFIX ".nzpart"

template <class DataType, class IntType, class IdxType = IntType>
class Pacoss_Partitioner
{
  public:
    // I/O routines for part vectors
    static void loadDimPart(
        const IntType &numParts,
        std::vector<std::vector<IntType>> &dimPart,
        const char * const dimPartFileName);

    static void saveDimPart(
        const IntType numParts,
        const std::vector<std::vector<IntType>> &dimPart,
        const char * const dimPartFileName);

    static void loadNzPart(
        std::vector<IntType> &nzPart,
        const char * const nzPartFileName);

    static void saveNzPart(
        const std::vector<IntType> &nzPart,
        const char * const nzPartFileName);

    // Partitioning routines for dimensions
    static void partitionDimsRandom(
        const Pacoss_SparseStruct<DataType> &spStruct,
        const IntType numParts,
        std::vector<std::vector<IntType>> &dimPart);

    static void partitionDimsBalanced(
        Pacoss_SparseStruct<DataType> &spStruct,
        const IntType numParts,
        const std::vector<IntType> &nzPart,
        std::vector<std::vector<IntType>> &dimPart,
        std::vector<double> maxRowImbalance,
        std::vector<std::vector<IntType>> procCtorIdx = std::vector<std::vector<IntType>>());

    // Partitioning routines for nonzeros
    static void partitionNzFineRandom(
        const Pacoss_SparseStruct<DataType> &spStruct,
        const IntType numParts,
        std::vector<IntType> &nzPart);

    static void partitionNzCheckerboardRandom(
        const Pacoss_SparseStruct<DataType> &spStruct,
        const IntType numRowParts,
        const IntType numColParts,
        std::vector<IntType> &nzPart,
        std::vector<std::vector<IntType>> &partCtorIdx);

    static void partitionNzCheckerboardUniform(
        const Pacoss_SparseStruct<DataType> &spStruct,
        const std::vector<IntType> &numParts,
        std::vector<IntType> &nzPart,
        std::vector<std::vector<IntType>> &dimPart,
        bool randomize = true);

    static void partitionHypergraphFast(
        const int numVtx,
        const int numHedges,
        const int * const pinBegin,
        const int * const pinIdx,
        const int * const vertexWeights,
        const int numParts,
        int *partVec);

#ifdef PACOSS_USE_PATOH
    static void partitionHypergraphPatoh(
        const int numVtx,
        const int numHedges,
        int * pinBegin,
        int * pinIdx,
        int * vertexWeights,
        int * hedgeWeights,
        const int numConsts,
        const int numParts,
        const char * const patohSettings,
        int *partVec);

    static void partitionNzFinePatoh(
        const Pacoss_SparseStruct<DataType> &spStruct,
        const IntType numParts,
        const char * const patohSettings,
        std::vector<IntType> &nzPart);

    static void partitionNzCheckerboardPatoh(
        const Pacoss_SparseStruct<DataType> &spStruct,
        const IntType numRowParts,
        const IntType numColParts,
        const char * const patohSettings,
        const char * const colPartitionType,
        std::vector<IntType> &nzPart,
        std::vector<std::vector<IntType>> &partCtorIdx);
#endif

    // Distribution routines
    static void printPartitionStats(
        const Pacoss_SparseStruct<DataType> &spStruct,
        const IntType numParts,
        const std::vector<IntType> &nzPart,
        const std::vector<std::vector<IntType>> &dimPart);

    static void distributeSparseStruct(
        const Pacoss_SparseStruct<DataType> &spStruct,
        const IntType numParts,
        const std::vector<IntType> &nzPart,
        const char * const sparseStructOutputFileNamePrefix);

    static void distributeDimPartition(
        const char * const sparseStructOutputFileNamePrefix,
        const IntType numParts,
        const std::vector<std::vector<IntType>> & dimPart,
        const char * const dimOutputFileNamePrefix);
};

#endif
