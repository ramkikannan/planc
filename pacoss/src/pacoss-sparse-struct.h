#ifndef PACOSS_SPARSE_STRUCT_H
#define PACOSS_SPARSE_STRUCT_H

#include "pacoss-common.h"

template <class DataType>
class Pacoss_SparseStruct
{
  public:
    Pacoss_Int _order;
    Pacoss_Idx _nnz;
    std::vector<DataType> _val;
    Pacoss_IntVector _dimSize;
    std::vector<Pacoss_IntVector> _idx;

    void load(const char * const fileName);
    void save(const char * const fileName);
    void print(FILE *outputFile = stdout);
    void form(
        const std::vector<Pacoss_IntVector> &idx,
        const std::vector<DataType> &val,
        const bool oneBased = true
        );
    void fix();
};

#endif
