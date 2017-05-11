#include "pacoss-sparse-struct.h"

template <class DataType>
void Pacoss_SparseStruct<DataType>::load(const char * const fileName)
{
  FILE *file = fopene(fileName, "r");

  // Read _order, _nnz, and _dimSize.
  fscanfe(file, " %" SCNINT " %" SCNIDX, &_order, &_nnz);
  _dimSize.resize(_order);
  for (Pacoss_Int i = 0; i < _order; i++) { fscanfe(file, " %" SCNINT, &_dimSize[i]); }

  // Now allocate and read the nonzero entries.
  _idx.resize(_order);
  for (Pacoss_Int i = 0; i < _order; i++) { _idx[i].resize(_nnz); }
  _val.resize(_nnz);
  for (Pacoss_Idx i = 0; i < _nnz; i++) {
    for (Pacoss_Int j = 0; j < _order; j++) {
      fscanfe(file, " %" SCNINT, &_idx[j][i]);
      _idx[j][i]--; // 0-based conversion
    }
    fscanfe(file, SCNMOD<DataType>(), &_val[i]);
  }

  fclosee(file);
}

template <class DataType>
void Pacoss_SparseStruct<DataType>::print(FILE *outputFile)
{
  fprintfe(outputFile, "%" PRIINT " %" PRIIDX "\n", _order, _nnz);
  for (Pacoss_Int i = 0; i < _order; i++) { fprintfe(outputFile, "%" PRIINT " ", _dimSize[i]); }
  fprintfe(outputFile, "\n");
  for (Pacoss_Idx i = 0; i < _nnz; i++) {
    for (Pacoss_Int j = 0; j < _order; j++) { 
      fprintfe(outputFile, "%" PRIINT " ", _idx[j][i] + 1); // 1-based conversion
    }
    fprintfe(outputFile, PRIMOD<DataType>(), _val[i]);
    fprintfe(outputFile, "\n");
  }
}

template <class DataType>
void Pacoss_SparseStruct<DataType>::save(const char * const fileName)
{
  FILE *outputFile = fopene(fileName, "w");

  print(outputFile);

  fclosee(outputFile);
}

template <class DataType>
void Pacoss_SparseStruct<DataType>::form(
    const std::vector<Pacoss_IntVector> &idx,
    const std::vector<DataType> &val,
    const bool oneBased
    )
{
  _order = (Pacoss_Int)idx.size();
  _nnz = (Pacoss_Idx)idx[0].size();
  _idx.resize(_order);
  _dimSize.resize(_order);
  for (Pacoss_Int i = 0, ie = _order; i < ie; i++) { 
    if (!oneBased) { _idx[i] = idx[i]; }
    else { 
      _idx[i].reserve(_nnz);
      for (Pacoss_Idx j = 0, je = _nnz; j < je; j++) { _idx[i].push_back(idx[i][j] - 1); }
    }
    _dimSize[i] = *std::max_element(_idx[i].begin(), _idx[i].end()) + 1;
  }
  _val = val;
  if (_val.size() == 0) { _val.resize(_nnz, (DataType)1.0); }
}

template class Pacoss_SparseStruct<double>;
template class Pacoss_SparseStruct<float>;
