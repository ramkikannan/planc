#include "pacoss-partitioner.h"
#include "keyvalsorter.h"

// I/O routines for part vectors
template <class DataType, class IntType, class IdxType>
void Pacoss_Partitioner<DataType, IntType, IdxType>::loadDimPart(
    const IntType &numParts,
    std::vector<std::vector<IntType>> &dimPart,
    const char * const dimPartFileName)
{
  FILE *file = fopene(dimPartFileName, "r"); 
  IntType order;
  fscanfe(file, STRCAT(SCNMOD<IntType>(), SCNMOD<IntType>()), &order, &numParts);
  dimPart.resize(order);
  for (IntType i = 0, ie = (IntType)dimPart.size(); i < ie; i++) { 
    IntType dimSize;
    fscanfe(file, SCNMOD<IntType>(), &dimSize);
    dimPart[i].resize(dimSize);
  }
  for (auto &curDimPart : dimPart) {
    for (auto &i : curDimPart) {
      fscanfe(file, SCNMOD<IntType>(), &i);
      i--; // Convert to 0-based
    }
  }
  fclosee(file);
}

template <class DataType, class IntType, class IdxType>
void Pacoss_Partitioner<DataType, IntType, IdxType>::saveDimPart(
    const IntType numParts,
    const std::vector<std::vector<IntType>> &dimPart,
    const char * const dimPartFileName)
{
  FILE *file = fopene(dimPartFileName, "w"); 
  fprintfe(file, STRCAT("%zu ", PRIMOD<IntType>(), "\n"), dimPart.size(), numParts);
  for (auto &i : dimPart) { fprintfe(file, "%zu ", i.size()); }
  fprintfe(file, "\n");
  for (auto &curDimPart : dimPart) {
    fprintfe(file, "\n");
    for (auto i : curDimPart) { fprintfe(file, STRCAT(PRIMOD<IntType>(), "\n"), i + 1); }
  }
  fclosee(file);
}

template <class DataType, class IntType, class IdxType>
void Pacoss_Partitioner<DataType, IntType, IdxType>::loadNzPart(
    std::vector<IntType> &nzPart,
    const char * const nzPartFileName)
{
  FILE *file = fopene(nzPartFileName, "r"); 
  IntType nzPartSize;
  fscanf(file, SCNMOD<IntType>(), &nzPartSize); 
  nzPart.resize(nzPartSize);
  for (IntType i = 0; i < nzPartSize; i++) {
    fscanf(file, SCNMOD<IntType>(), &nzPart[i]);
    nzPart[i]--; // Convert to 0-based
  }
  fclosee(file);
}

template <class DataType, class IntType, class IdxType>
void Pacoss_Partitioner<DataType, IntType, IdxType>::saveNzPart(
    const std::vector<IntType> &nzPart,
    const char * const nzPartFileName)
{
  FILE *file = fopene(nzPartFileName, "w"); 
  fprintfe(file, "%zu\n\n", nzPart.size());
  for (auto i : nzPart) { fprintfe(file, STRCAT(PRIMOD<IntType>(), "\n"), i + 1); }
  fclosee(file);
}

// Partitioning routines for dimensions
template <class DataType, class IntType, class IdxType>
void Pacoss_Partitioner<DataType, IntType, IdxType>::partitionDimsRandom(
    const Pacoss_SparseStruct<DataType> &spStruct,
    const IntType numParts,
    std::vector<std::vector<IntType>> &dimPart)
{
  dimPart.resize(spStruct._order);
  for (IntType i = 0; i < spStruct._order; i++) { dimPart[i].resize(spStruct._dimSize[i]); }
  srand((unsigned)time(NULL));
  for (IntType i = 0; i < spStruct._order; i++) {
    for (IntType j = 0; j < spStruct._dimSize[i]; j++) {
      dimPart[i][j] = rand() % numParts;
    }
  }
}

template <class DataType, class IntType, class IdxType>
void Pacoss_Partitioner<DataType, IntType, IdxType>::partitionDimsBalanced(
    Pacoss_SparseStruct<DataType> &spStruct,
    const IntType numParts,
    const std::vector<IntType> &nzPart,
    std::vector<std::vector<IntType>> &dimPart,
    std::vector<double> maxRowImbalance,
    std::vector<std::vector<IntType>> partCtorIdx)
{
  if (maxRowImbalance.size() == 0) { // If no imbalance is specified, then set allowed imbalance to maximum.
    for (IntType i = 0, ie = spStruct._order; i < ie; i++) { maxRowImbalance.push_back((double)numParts + 1.0); }
  }
  // Set up communicator groups, if not provided, then push each part to the list of available parts for that group
  if (partCtorIdx.size() == 0) {
    printfe("No communicator info specified; putting all partessors to the same communicator.\n");
    for (IntType dim = 0, dime = spStruct._order; dim < dime; dim++) {
      partCtorIdx.push_back(std::vector<IntType>(numParts)); // Put all procs to ctor 0
    }
  }
  std::vector<std::vector<std::vector<IntType>>> availProcs(spStruct._order);
  for (IntType dim = 0, dime = spStruct._order; dim < dime; dim++) {
    auto & dimProcCtorIdx = partCtorIdx[dim];
    auto & dimAvailProcs = availProcs[dim];
    IntType dimNumCtors = (*std::max_element(dimProcCtorIdx.begin(), dimProcCtorIdx.end())) + 1; // Ctors are 0-based
    availProcs[dim].resize(dimNumCtors);
    for (IntType part = 0, parte = numParts; part < parte; part++) {
      dimAvailProcs[dimProcCtorIdx[part]].push_back(part); // Add part to its ctor's list of parts
    }
  }

  // Find the set of owner candidate parts for each row that 'touch' that row.
  std::vector<std::vector<std::vector<IntType>>> ownerCandidate;
  for (IntType i = 0; i < spStruct._order; i++) { ownerCandidate.emplace_back(spStruct._dimSize[i]); }
  std::vector<IdxType> sortedNzIdx; sortedNzIdx.reserve(spStruct._nnz);
  for (IdxType i = 0; i < spStruct._nnz; i++) { sortedNzIdx.push_back(i); }
  std::vector<std::vector<IntType>> temp; temp.push_back(nzPart);
  KeyValSorter::sort<IntType>(temp, sortedNzIdx);
  for (IntType i = 0; i < spStruct._nnz; i++) {
    IdxType nzIdx = sortedNzIdx[i];
    IntType partIdx = temp[0][i];
    for (IntType j = 0; j < spStruct._order; j++) {
      IntType rowIdx = spStruct._idx[j][nzIdx];
      if (ownerCandidate[j][rowIdx].empty() || ownerCandidate[j][rowIdx].back() != partIdx) {
        ownerCandidate[j][rowIdx].push_back(partIdx);
      }
    }
  }

  // Find greedyRowOrder sorted with small-to-large number of owner candidates, then assign rows to parts greedily
  srand((int)time(0));
  dimPart.clear();
  for (IntType i = 0; i < spStruct._order; i++) { dimPart.emplace_back(spStruct._dimSize[i]); }
  for (IntType j = 0; j < spStruct._order; j++) {
    // Sort rows with increasing number of ownerCandidates.
    std::vector<IntType> partRowCount(numParts);
    IntType maxPartRowCount = (IntType)((spStruct._dimSize[j] * maxRowImbalance[j]) / (double)numParts);
    std::vector<IntType> greedyRowOrder; greedyRowOrder.reserve(spStruct._dimSize[j]);
    for (IntType i = 0; i < spStruct._dimSize[j]; i++) { greedyRowOrder.push_back(i); }
    std::vector<std::vector<IntType>> temp;
    temp.emplace_back(spStruct._dimSize[j]);
    for (IntType i = 0, ie = (IntType)temp[0].size(); i < ie; i++) {
      temp[0][i] = (IntType)ownerCandidate[j][i].size();
    }
    KeyValSorter::sort<IntType>(temp, greedyRowOrder);
    // Send and receive volumes correspond to fold communication. It is the inverse for expand communication.
    std::vector<IntType> partRecvCommVol(numParts);
    std::vector<IntType> partSendCommVol(numParts);
    // Assign each row to parts by balancing the send comm load
    std::vector<IntType> cands; cands.reserve(128);
    auto & dimProcCtorIdx = partCtorIdx[j];
    auto & dimAvailProcs = availProcs[j];
    for (auto i : greedyRowOrder) {
      IntType minPartIdx = -1;
      // Filter parts in ownerCandidate[j][i] that are overloaded
      auto &curOwnerCandidate = ownerCandidate[j][i];
      cands.clear();
      for (auto cand : curOwnerCandidate) {
        if (partRowCount[cand] < maxPartRowCount) { cands.push_back(cand); }
      }
      if (cands.size() > 0) { // Try to assign to one of these parts if possible
        minPartIdx = cands[rand() % (int)cands.size()];
      } else {
        if (curOwnerCandidate.size() > 0) {
          auto &partList = dimAvailProcs[dimProcCtorIdx[curOwnerCandidate[0]]];
          if (partList.size() > 0) { // Try to assign to an underloaded part in the same ctor if possible
            auto minPartIdxIdx = rand() % (int)partList.size();
            minPartIdx = partList[minPartIdxIdx];
            if (partRowCount[minPartIdx] + 1 >= maxPartRowCount) {
              std::swap(partList[minPartIdxIdx], partList[partList.size() - 1]);
              partList.resize(partList.size() - 1);
            }
          } else { // All parts are overloaded in the current ctor; assign the row to one of them randomly.
            auto minPartIdxIdx = rand() % (int)curOwnerCandidate.size();
            minPartIdx = curOwnerCandidate[minPartIdxIdx];
          }
        } else { // Row has no owner candidates; assign to the part having minimum number of rows.
          minPartIdx = rand() % (int)numParts;
        }
      }
      Pacoss_AssertGe(minPartIdx, 0); Pacoss_AssertLt(minPartIdx, numParts);
      // Assign to minPartIdx, increase the recv comm volume accordingly
      dimPart[j][i] = minPartIdx;
      partRowCount[minPartIdx]++;
      partRecvCommVol[minPartIdx] += (IntType)std::max((size_t)0, curOwnerCandidate.size() - 1);
      // Increase send comm volume of other parts
      for (auto partIdx : curOwnerCandidate) {
        if (partIdx != minPartIdx) { partSendCommVol[partIdx]++; }
      }
    }
  }
}


// Partitioning routines for nonzeros
template <class DataType, class IntType, class IdxType>
void Pacoss_Partitioner<DataType, IntType, IdxType>::partitionNzFineRandom(
    const Pacoss_SparseStruct<DataType> &spStruct,
    const IntType numParts,
    std::vector<IntType> &nzPart)
{
  srand((unsigned)time(NULL));
  nzPart.resize(spStruct._nnz);
  for (IdxType i = 0; i < spStruct._nnz; i++) {
    nzPart[i] = rand() % numParts;
  }
}

template <class DataType, class IntType, class IdxType>
void Pacoss_Partitioner<DataType, IntType, IdxType>::partitionNzCheckerboardRandom(
    const Pacoss_SparseStruct<DataType> &spStruct,
    const IntType numRowParts,
    const IntType numColParts,
    std::vector<IntType> &nzPart,
    std::vector<std::vector<IntType>> &partCtorIdx)
{
  Pacoss_AssertEq(spStruct._order, 2);
  printfe(STRCAT("Partitioning with a ", PRIMOD<IntType>(), "x", PRIMOD<IntType>(), " checkerboard topology.\n"),
      numRowParts, numColParts);
  std::vector<IntType> rowPart(spStruct._dimSize[0]);
  std::vector<IntType> colPart(spStruct._dimSize[1]);
  auto &rowIdx = spStruct._idx[0];
  auto &colIdx = spStruct._idx[1];
  srand((unsigned)time(NULL));
  for (auto &i : rowPart) { i = rand() % numRowParts; }
  for (auto &i : colPart) { i = rand() % numColParts; }
  // Finally, generate the nzPart vector according to the 2-D checkerboard partition
  nzPart.resize(spStruct._nnz);
  for (IdxType i = 0; i < spStruct._nnz; i++) {
    nzPart[i] = rowPart[rowIdx[i]] * numColParts + colPart[colIdx[i]];
    Pacoss_AssertGe(nzPart[i], 0); Pacoss_AssertLt(nzPart[i], numRowParts * numColParts);
  }
  // Put each process to its row and column communicator
  partCtorIdx.resize(2);
  partCtorIdx[0].resize(numRowParts * numColParts);
  partCtorIdx[1].resize(numRowParts * numColParts);
  for (IntType i = 0, ie = numRowParts; i < ie; i++) {
    for (IntType j = 0, je = numColParts; j < je; j++) {
      IntType procIdx = i * numColParts + j; 
      partCtorIdx[0][procIdx] = i;
      partCtorIdx[1][procIdx] = j;
    }
  }
}

template <class DataType, class IntType, class IdxType>
void Pacoss_Partitioner<DataType, IntType, IdxType>::partitionNzCheckerboardUniform(
        const Pacoss_SparseStruct<DataType> &spStruct,
        const std::vector<IntType> &procDims,
        std::vector<IntType> &nzPart,
        std::vector<std::vector<IntType>> &dimPart,
        bool randomize)
{
  std::vector<IntType> rowsPerProcDim(spStruct._order);
  std::vector<IntType> dimProcIdMultiplier(spStruct._order);
  std::vector<std::vector<IntType>> dimPerm(spStruct._order);
  for (IntType dim = 0, dime = spStruct._order; dim < dime; dim++) {
    auto &perm = dimPerm[dim];
    for (IntType i = 0, ie = spStruct._dimSize[dim]; i < ie; i++) { perm.push_back(i); }
    if (randomize) {
      srand((unsigned int)time(NULL));
      for (IntType i = 0, ie = spStruct._dimSize[dim]; i < ie; i++) {
        IntType j = rand() % spStruct._dimSize[dim];
        std::swap(perm[i], perm[j]);
      }
    }
  }
  dimProcIdMultiplier[spStruct._order - 1] = 1;
  for (IntType dim = spStruct._order - 2, dime = 0; dim >= dime; dim--) {
    dimProcIdMultiplier[dim] = dimProcIdMultiplier[dim + 1] * procDims[dim + 1];
  }
  for (IntType dim = 0, dime = spStruct._order; dim < dime; dim++) {
    rowsPerProcDim[dim] = (spStruct._dimSize[dim] + procDims[dim] - 1) / procDims[dim];
  }
  // Partition dimensions
  dimPart.resize(spStruct._order);
  std::vector<std::vector<std::vector<IntType>>> procList(spStruct._order);
  IntType numProcs = 1;
  for (IntType dim = 0, dime = spStruct._order; dim < dime; dim++) { 
    dimPart[dim].resize(spStruct._dimSize[dim]); 
    procList[dim].resize(procDims[dim]);
    numProcs *= procDims[dim];
  }
  for (IntType i = 0, ie = numProcs; i < ie; i++) {
    for (IntType dim = 0, dime = spStruct._order; dim < dime; dim++) {
      IntType procRowIdx = (i / dimProcIdMultiplier[dim]) % procDims[dim];
      procList[dim][procRowIdx].push_back(i);
    }
  }
  printVector(procList[1][0]);
  for (IntType dim = 0, dime = spStruct._order; dim < dime; dim++) {
    auto &curDimPerm = dimPerm[dim];
    for (IntType row = 0, rowe = procDims[dim]; row < rowe; row++) {
      IntType rowNumProcs = (IntType)procList[dim][row].size();
      IntType rowBeg = row * rowsPerProcDim[dim];
      IntType rowEnd = std::min((row + 1) * rowsPerProcDim[dim], (IntType)spStruct._dimSize[dim]);
      IntType rowsPerProc = (rowEnd - rowBeg + rowNumProcs - 1) / rowNumProcs;
      for (IntType i = 0, ie = (IntType)procList[dim][row].size(); i < ie; i++) {
        for (IntType j = i * rowsPerProc, je = std::min((i + 1) * rowsPerProc, rowEnd); j < je; j++) {
          dimPart[dim][curDimPerm[rowBeg + j]] = procList[dim][row][i];
        }
      }
    }
  }
  // Partition nonzeros
  nzPart.resize(spStruct._nnz);
  for (IdxType i = 0, ie = (IdxType)spStruct._nnz; i < ie; i++) {
    nzPart[i] = 0;
    for (IntType dim = 0, dime = spStruct._order; dim < dime; dim++) {
      IntType idx = dimPerm[dim][spStruct._idx[dim][i]];
      nzPart[i] += (idx / rowsPerProcDim[dim]) * dimProcIdMultiplier[dim];
    }
  }
}

template <class DataType, class IntType, class IdxType>
void Pacoss_Partitioner<DataType, IntType, IdxType>::partitionHypergraphFast(
    const int numVtx,
    const int numHedges,
    const int * const _pinBegin,
    const int * const _pinIdx,
    const int * const vertexWeights,
    const int numParts,
    int *_partVec)
{
}

#ifdef PACOSS_USE_PATOH
#include "patoh.h"

template <class DataType, class IntType, class IdxType>
void Pacoss_Partitioner<DataType, IntType, IdxType>::partitionHypergraphPatoh(
    const int numVtx,
    const int numHedges,
    int * pinBegin,
    int * pinIdx,
    int * vertexWeights,
    int * hedgeWeights,
    const int numConsts,
    const int numParts,
    const char * const patohSettings,
    int *partVec)
{
  if (numParts == 1) { // Patoh seems to have a bug for numParts = 1
    for (int i = 0; i < numParts; i++) { partVec[i] = 0; }
    return;
  }
  int patohParams;
  if (strcmp(patohSettings, "speed") == 0) {
    patohParams = PATOH_SUGPARAM_SPEED;
  } else if (strcmp(patohSettings, "quality") == 0) {
    patohParams = PATOH_SUGPARAM_QUALITY;
  } else {
    patohParams = PATOH_SUGPARAM_DEFAULT;
  }
  PaToH_Parameters args;
  std::vector<float> targetWeights(numParts * numConsts, 1.0f / (float)numParts); // load balance constraints for parts
  std::vector<int> partWeights(numParts * numConsts);
  PaToH_Initialize_Parameters(&args, PATOH_CONPART, patohParams);
  args._k = numParts;
  // PaToH memory parameters with default values. Increase when PaToH fails due to getting out of memory. 
  // args.MemMul_CellNet = 11; 
  // args.MemMul_Pins *= 5;
  // args.MemMul_General = 1;
  int patohErr = PaToH_Alloc(&args, numVtx, numHedges, numConsts, vertexWeights, hedgeWeights, pinBegin, pinIdx);
  if (patohErr != 0) {
    char msg[1000]; sprintf(msg, "ERROR: PaToH_Alloc returned with the error code %d.\n", patohErr);
    throw Pacoss_Error(msg);
  }
  timerTic();
  int cutSize;
  patohErr = PaToH_Part(&args, numVtx, numHedges, numConsts, 0, vertexWeights, hedgeWeights, pinBegin, pinIdx,
      &targetWeights[0], partVec, &partWeights[0], &cutSize);
  if (patohErr != 0) {
    char msg[1000]; sprintf(msg, "ERROR: PaToH_Part returned with the error code %d.\n", patohErr);
    throw Pacoss_Error(msg);
  }
  printfe("Partitioning is finished with cutsize %d.\n", cutSize);
  patohErr = PaToH_Free();
  for (int i = 0; i < numVtx; i++) { Pacoss_AssertLt(partVec[i], numParts); }
  if (patohErr != 0) {
    char msg[1000]; sprintf(msg, "ERROR: PaToH_Free returned with the error code %d.\n", patohErr);
    throw Pacoss_Error(msg);
  }
  printfe("Call to PaToH took %lf seconds\n", timerToc());
}

template <class DataType, class IntType, class IdxType>
void Pacoss_Partitioner<DataType, IntType, IdxType>::partitionNzFinePatoh(
    const Pacoss_SparseStruct<DataType> &spStruct,
    const IntType numParts,
    const char * const patohSettings,
    std::vector<IntType> &nzPart)
{
  int numVertex, numHedges;  // number of vertices and hyperedges of the hypergraph
  std::vector<int> pinBegin, // pointers for the hyperedge pinIdx
    pinIdx,                  // hyperedge pin indices
    partVec,                 // partition array for the vertices
    vertexWeights;           // vertex weights (set to 1)
  numVertex = (int)spStruct._nnz; // There is one vertex per nonzero in the fine-grain hypergraph

  printfe("Creating the hypergraph...\n"); 
  // Compute the number of hyperedges in hypergraph. There is one hedge for each vector row (across all dimensions).
  numHedges = 0;
  for (int i = 0; i < spStruct._order; i++) { numHedges += spStruct._dimSize[i]; }
  // hedgeSumPrefix[k] corresponds to the number of hedges up to k.th dimension (hence the begin index for dim k)
  std::vector<int> hedgeSumPrefix(spStruct._order + 1);
  for (int i = 1; i <= spStruct._order; i++) { hedgeSumPrefix[i] = hedgeSumPrefix[i - 1] + spStruct._dimSize[i - 1]; }
  numHedges = hedgeSumPrefix[spStruct._order];
  // Allocate and compute pinBegin and pinIdx (vertices that each hyperedge connects).
  pinBegin.resize(numHedges + 1, 0);
  pinIdx.resize(spStruct._nnz * spStruct._order);
  // Find the number of vertices belonging to each hedge
  for (int i = 0; i < spStruct._nnz; i++) {
    for (int j = 0; j < spStruct._order; j++) {
      int hedgeIdx = hedgeSumPrefix[j] + spStruct._idx[j][i];
      pinBegin[hedgeIdx]++;
    }
  }
  // Take prefix sum to form CSR pointers for the pins of hedges
  for (int i = 1; i <= numHedges; i++) { pinBegin[i] += pinBegin[i - 1]; }
  // Fill in the pinIdx array
  for (int i = 0; i < spStruct._nnz; i++) {
    for (int j = 0; j < spStruct._order; j++) {
      int hedgeIdx = hedgeSumPrefix[j] + spStruct._idx[j][i];
      pinIdx[--pinBegin[hedgeIdx]] = i;
    }
  }
  Pacoss_AssertEq(pinBegin[0], 0); Pacoss_AssertEq(pinBegin[numHedges], (int)pinIdx.size());
  vertexWeights.resize(numVertex, 1); // Set vertex weights to 1
  partVec.resize(numVertex);

  printfe("Calling PaToH wrapper to partition the hypergraph...\n"); 
  partitionHypergraphPatoh(numVertex, numHedges, &pinBegin[0], &pinIdx[0], &vertexWeights[0], NULL, 1, (int)numParts,
      patohSettings, &partVec[0]);
  nzPart.reserve(partVec.size());
  for (int i = 0, ie = (int)partVec.size(); i < ie; i++) { nzPart.push_back((IntType)partVec[i]); }
}

template <class DataType, class IntType, class IdxType>
void Pacoss_Partitioner<DataType, IntType, IdxType>::partitionNzCheckerboardPatoh(
    const Pacoss_SparseStruct<DataType> &spStruct,
    const IntType numRowParts,
    const IntType numColParts,
    const char * const patohSettings,
    const char * const colPartitionType,
    std::vector<IntType> &nzPart,
    std::vector<std::vector<IntType>> &partCtorIdx)
{
  Pacoss_AssertEq(spStruct._order, 2); // Checkerboard partitioning works only for matrices
  printfe(STRCAT("Partitioning with a ", PRIMOD<IntType>(), "x", PRIMOD<IntType>(), " checkerboard topology.\n"),
        numRowParts, numColParts);
  int numVertex = spStruct._dimSize[0]; // number of vertices
  int numHedges = spStruct._dimSize[1]; // number of hyperedges
  std::vector<int> vertexWeights(numVertex); // vertex weights
  std::vector<int> pinBegin(numHedges + 1); // hedge pointers
  std::vector<int> pinIdx(spStruct._nnz); // hedge pinIdx
  std::vector<int> rowPart(spStruct._dimSize[0]); // part vector of rows
  std::vector<int> colPart(spStruct._dimSize[1]); // part vector of cols
  auto &rowIdx = spStruct._idx[0];
  auto &colIdx = spStruct._idx[1];
  printfe("Creating the column-net hypergraph...\n");
  // Make the first pass to compute vertex weights and pinBegin
  std::vector<bool> sample(spStruct._nnz);
  srand((unsigned)time(NULL));
  for (IdxType i = 0; i < spStruct._nnz; i++) { sample[i] = 1 - (rand() % 1); }
  std::fill(vertexWeights.begin(), vertexWeights.end(), 1); // Balancing the number of rows
  for (IdxType i = 0; i < spStruct._nnz ; i++) {
    if (sample[i]) {
//      vertexWeights[rowIdx[i]]++; // Balancing the number of nonzeros
      pinBegin[colIdx[i]]++;
    }
  }
  // Perform a prefix sum to compute the pinBegin pointers
  for (IntType i = 1; i <= numHedges ; i++) { pinBegin[i] += pinBegin[i - 1]; }
  // Fill in the pinIdx array in the second pass
  for (IdxType i = 0; i < spStruct._nnz; i++) { 
    if (sample[i]) { pinIdx[--pinBegin[colIdx[i]]] = (int)rowIdx[i]; }
  }
  printfe("Partitioning the column-net hypergraph using PaToH...\n"); 
  partitionHypergraphPatoh(numVertex, numHedges, &pinBegin[0], &pinIdx[0], &vertexWeights[0], NULL, 1, (int)numRowParts,
      patohSettings, &rowPart[0]);
  for (auto i : rowPart) { Pacoss_AssertLt(i, numRowParts); }
  printfe("DONE!\n");

  if (strcmp(colPartitionType, "1d") == 0) { // Partition the columns randomly
    srand((unsigned)time(NULL));
    // Compute the comm cost of each column
    std::vector<IntType> colCommCost(spStruct._dimSize[1], -1); // connectivity-1 is the comm cost for each hedge
    std::vector<IntType> partVisited(numRowParts, -1);
    for (IntType i = 0, ie = spStruct._dimSize[1]; i < ie; i++) {
      for (int j = pinBegin[i], je = pinBegin[i + 1]; j < je; j++) {
        int pinPart = rowPart[pinIdx[j]];
        if (partVisited[pinPart] < i) { // a part in the cut for this hedge
          partVisited[pinPart] = i;
          colCommCost[i]++;
        }
      }
      if (colCommCost[i] == -1) { colCommCost[i] = 0; }
    }
//    std::vector<IntType> colCommCost(spStruct._dimSize[1], 0); // Cost of a column is the number of its nonzeros.
//    for (IdxType i = 0, ie = (IdxType)spStruct._nnz; i < ie; i++) { colCommCost[spStruct._idx[1][i]]++; }
    IntType totalCommCost = std::accumulate(colCommCost.begin(), colCommCost.end(), 0);
    IntType commPerPart = (totalCommCost + numColParts - 1) / numColParts;
    std::vector<IntType> randColOrder(spStruct._dimSize[1]);
    for (IntType i = 0, ie = spStruct._dimSize[1]; i < ie; i++) { randColOrder[i] = i; }
    srand((int)time(0));
    for (IntType i = 0, ie = spStruct._dimSize[1]; i < ie; i++) { 
      std::swap(randColOrder[i], randColOrder[rand() % spStruct._dimSize[1]]);
    }
    int curPartIdx = 0;
    IntType curPartLoad = 0;
    for (IntType i = 0, ie = spStruct._dimSize[1]; i < ie; i++) {
      IntType colIdx = randColOrder[i];
      colPart[colIdx] = curPartIdx;
      curPartLoad += colCommCost[colIdx];
      if (curPartLoad > commPerPart) {
        curPartIdx++;
        curPartLoad = 0;
      }
    }
//    for (IntType i = 0; i < spStruct._dimSize[1]; i++) { colPart[i] = rand() % (int)numColParts; }
  } else { // Partition the columns using multi-constraint PaToH
    printfe("Creating the row-net hypergraph...\n"); 
    numVertex = spStruct._dimSize[1];
    numHedges = spStruct._dimSize[0];
    vertexWeights.resize(numVertex * numRowParts); std::fill(vertexWeights.begin(), vertexWeights.end(), 0);
    pinBegin.resize(numHedges + 1); std::fill(pinBegin.begin(), pinBegin.end(), 0);
    // Make the first pass to compute vertex weights and pinBegin
    for (IdxType i = 0; i < spStruct._nnz; i++) {
//      vertexWeights[colIdx[i] * numRowParts + rowPart[rowIdx[i]]]++;
      pinBegin[rowIdx[i]]++;
    }
    // Perform a prefix sum to compute the pinBegin pointers
    for (IntType i = 1; i <= numHedges ; i++) { pinBegin[i] += pinBegin[i - 1]; }
    // Fill in the pinIdx array in the second pass
    for (IdxType i = 0; i < spStruct._nnz; i++) { pinIdx[--pinBegin[rowIdx[i]]] = (int)colIdx[i]; }
    Pacoss_AssertEq(pinBegin[0], 0); Pacoss_AssertEq(pinBegin[numHedges], (int)pinIdx.size());
    printfe("Partitioning the row-net hypergraph using multi-constraint PaToH...\n"); 
    partitionHypergraphPatoh(numVertex, numHedges, &pinBegin[0], &pinIdx[0], &vertexWeights[0], NULL, (int)numRowParts,
        (int)numColParts, patohSettings, &colPart[0]);
  }
  // Generate the nzPart vector according to the 2-D checkerboard partition
  nzPart.resize(spStruct._nnz);
  for (IdxType i = 0; i < spStruct._nnz; i++) {
    nzPart[i] = rowPart[rowIdx[i]] * numColParts + colPart[colIdx[i]];
    Pacoss_AssertLt(colPart[colIdx[i]], numColParts);
    Pacoss_AssertLt(rowPart[rowIdx[i]], numRowParts);
    Pacoss_AssertGe(nzPart[i], 0); Pacoss_AssertLt(nzPart[i], numRowParts * numColParts);
  }
  // Put each process to its row and column communicator
  partCtorIdx.resize(2);
  partCtorIdx[0].resize(numRowParts * numColParts);
  partCtorIdx[1].resize(numRowParts * numColParts);
  for (IntType i = 0, ie = numRowParts; i < ie; i++) {
    for (IntType j = 0, je = numColParts; j < je; j++) {
      IntType procIdx = i * numColParts + j; 
      partCtorIdx[0][procIdx] = i;
      partCtorIdx[1][procIdx] = j;
    }
  }
}
#endif

template <class DataType, class IntType, class IdxType>
void Pacoss_Partitioner<DataType, IntType, IdxType>::printPartitionStats(
    const Pacoss_SparseStruct<DataType> &spStruct,
    const IntType numParts,
    const std::vector<IntType> &nzPart,
    const std::vector<std::vector<IntType>> &dimPart)
{
  std::cout << "\n**************** Partition Stats *********************************************\n";
  std::cout << "Printing partition stats in order MAX AVG TOTAL IMBALANCE...\n";
  std::cout << "Computing part loads...\n";
  std::vector<IntType> partLoad(numParts);
  for (IdxType i = 0, ie = (IdxType)nzPart.size(); i < ie; i++) { partLoad[nzPart[i]]++; } 
  IntType maxLoad = *std::max_element(partLoad.begin(), partLoad.end());
  IntType totalLoad = std::accumulate(partLoad.begin(), partLoad.end(), 0);
  IntType avgLoad = totalLoad / numParts;
  std::cout << "Computational load: " << maxLoad << " " << avgLoad << " " << totalLoad << " " << (double)maxLoad /
    (double)avgLoad << std::endl;
  std::cout << "Computing part row counts...\n";
  std::vector<std::vector<IntType>> partRowCount(spStruct._order);
  std::vector<IntType> dimMaxRowCount(spStruct._order);
  IntType totalMaxRowCount = 0;
  IntType totalTotalRowCount = 0;
  for (IntType dim = 0; dim < spStruct._order; dim++) {
    partRowCount[dim].resize(numParts);
    for (IntType i = 0; i < spStruct._dimSize[dim]; i++) {
      partRowCount[dim][dimPart[dim][i]]++; 
    }
    IntType maxRowCount  = *std::max_element(partRowCount[dim].begin(), partRowCount[dim].end()); 
    IntType totalRowCount = std::accumulate(partRowCount[dim].begin(), partRowCount[dim].end(), 0);
    IntType avgRowCount = totalRowCount / numParts;
    totalMaxRowCount += maxRowCount;
    totalTotalRowCount += totalRowCount;
    std::cout << "Dim " << dim << " row count: " << maxRowCount << " " << avgRowCount << " " << totalRowCount << " " <<
      (double)maxRowCount / (double)avgRowCount << std::endl;
  }
  std::cout << "Overall row count: " << totalMaxRowCount << " " << totalTotalRowCount / numParts << " " <<
    totalTotalRowCount << " " << (double)totalMaxRowCount * (double)numParts / (double)totalTotalRowCount << std::endl;

  std::cout << "Computing fold communication volume...\n";
  // nzOrder corresponds to a permutation order of nzPart with increasing part ids.
  std::vector<IdxType> nzOrder; nzOrder.reserve(spStruct._nnz);
  IntType maxFoldSendVol = 0, maxFoldRecvVol = 0, totalFoldSendVol = 0, totalFoldRecvVol = 0;
  IntType maxFoldSendMsgCount = 0, maxFoldRecvMsgCount = 0, totalFoldSendMsgCount = 0, totalFoldRecvMsgCount = 0;
  for (IdxType i = 0; i < spStruct._nnz; i++) { nzOrder.push_back(i); }
  // Sort nzOrder with increasing nzPart
  std::vector<std::vector<IntType>> key; key.push_back(nzPart);
  KeyValSorter::sort(0, nzPart.size() - 1, key, nzOrder);
  std::vector<IntType> &orderedNzPart = key[0];
  for (IntType dim = 0; dim < spStruct._order; dim++) {
    std::vector<IntType> rowLastPart(spStruct._dimSize[dim], -1);
    std::vector<IntType> dimPartFoldSendVol(numParts);
    std::vector<IntType> dimPartFoldRecvVol(numParts);
    auto &idx = spStruct._idx[dim];
    auto &rowPart = dimPart[dim];
    std::vector<IntType> dimPartFoldSendMsgCount(numParts);
    std::vector<IntType> dimPartFoldRecvMsgCount(numParts);
    std::vector<std::unordered_map<IntType, IntType>> dimPartFoldSendMsg(numParts);
    std::vector<std::unordered_map<IntType, IntType>> dimPartFoldRecvMsg(numParts);
    for (IdxType i = 0, ie = (IdxType)nzOrder.size(); i < ie; i++) {
      IdxType nzIdx = nzOrder[i];
      IntType rowIdx = idx[nzIdx];
      IntType nzPartIdx = orderedNzPart[i];
      IntType rowPartIdx = rowPart[rowIdx];
      if (nzPartIdx > rowLastPart[rowIdx] && nzPartIdx != rowPartIdx) { // Hedge i is cut; increase the comm vol
        dimPartFoldRecvVol[rowPartIdx]++;
        dimPartFoldSendVol[nzPartIdx]++;
        dimPartFoldSendMsg[nzPartIdx][rowPartIdx] = 1;
        dimPartFoldRecvMsg[rowPartIdx][nzPartIdx] = 1;
        rowLastPart[rowIdx] = nzPartIdx;
      }
    }
    for (IntType i = 0; i < numParts; i++) { // Compute msg counts by checking the size of send/recv hashtables.
      dimPartFoldSendMsgCount[i] = (IntType)dimPartFoldSendMsg[i].size();
      dimPartFoldRecvMsgCount[i] = (IntType)dimPartFoldRecvMsg[i].size();
    }
    // Now that we have comm vol and msg count for each part, find the total and max.
    IntType dimMaxFoldSendVol = *std::max_element(dimPartFoldSendVol.begin(), dimPartFoldSendVol.end());
    IntType dimTotalFoldSendVol = std::accumulate(dimPartFoldSendVol.begin(), dimPartFoldSendVol.end(), 0);
    IntType dimMaxFoldRecvVol = *std::max_element(dimPartFoldRecvVol.begin(), dimPartFoldRecvVol.end());
    IntType dimTotalFoldRecvVol = std::accumulate(dimPartFoldRecvVol.begin(), dimPartFoldRecvVol.end(), 0);
    Pacoss_AssertEq(dimTotalFoldSendVol, dimTotalFoldRecvVol);
    IntType dimMaxFoldSendMsgCount = *std::max_element(dimPartFoldSendMsgCount.begin(),
        dimPartFoldSendMsgCount.end());
    IntType dimTotalFoldSendMsgCount = std::accumulate(dimPartFoldSendMsgCount.begin(),
        dimPartFoldSendMsgCount.end(), 0);
    IntType dimMaxFoldRecvMsgCount = *std::max_element(dimPartFoldRecvMsgCount.begin(),
        dimPartFoldRecvMsgCount.end());
    IntType dimTotalFoldRecvMsgCount = std::accumulate(dimPartFoldRecvMsgCount.begin(),
        dimPartFoldRecvMsgCount.end(), 0);
    std::cout << "Dim " << dim << " send volume: " << dimMaxFoldSendVol << " " << dimTotalFoldSendVol / numParts << " "
      << dimTotalFoldSendVol << " " << (double)dimMaxFoldSendVol * (double)numParts / (double)dimTotalFoldSendVol <<
      std::endl;
    std::cout << "Dim " << dim << " recv volume: " << dimMaxFoldRecvVol << " " << dimTotalFoldRecvVol / numParts << " "
      << dimTotalFoldRecvVol << " " << (double)dimMaxFoldRecvVol * (double)numParts / (double)dimTotalFoldRecvVol <<
      std::endl;
    std::cout << "Dim " << dim << " send #msg: " << dimMaxFoldSendMsgCount << " " << dimTotalFoldSendMsgCount /
      numParts << " " << dimTotalFoldSendMsgCount << " " << (double)dimMaxFoldSendMsgCount * (double)numParts /
      (double)dimTotalFoldSendMsgCount << std::endl;
    std::cout << "Dim " << dim << " recv #msg: " << dimMaxFoldRecvMsgCount << " " << dimTotalFoldRecvMsgCount /
      numParts << " " << dimTotalFoldRecvMsgCount << " " << (double)dimMaxFoldRecvMsgCount * (double)numParts /
      (double)dimTotalFoldRecvMsgCount << std::endl;
    maxFoldSendVol += dimMaxFoldSendVol;
    maxFoldRecvVol += dimMaxFoldRecvVol;
    maxFoldSendMsgCount += dimMaxFoldSendMsgCount;
    maxFoldRecvMsgCount += dimMaxFoldRecvMsgCount;
    totalFoldSendVol += dimTotalFoldSendVol;
    totalFoldRecvVol += dimTotalFoldRecvVol;
    totalFoldSendMsgCount += dimTotalFoldSendMsgCount;
    totalFoldRecvMsgCount += dimTotalFoldRecvMsgCount;
  }
  std::cout << "Overall send volume: " << maxFoldSendVol << " " << totalFoldSendVol / numParts << " " <<
    totalFoldSendVol << " " << (double)maxFoldSendVol * (double)numParts / (double)totalFoldSendVol << std::endl;
  std::cout << "Overall recv volume: " << maxFoldRecvVol << " " << totalFoldRecvVol / numParts << " " <<
    totalFoldRecvVol << " " << (double)maxFoldRecvVol * (double)numParts / (double)totalFoldRecvVol << std::endl;
  std::cout << "Overall send+recv volume: " << maxFoldSendVol + maxFoldRecvVol << " " << (totalFoldSendVol +
      totalFoldRecvVol) / numParts << " " << totalFoldSendVol + totalFoldRecvVol << " " << (double)(maxFoldSendVol +
        maxFoldRecvVol) * (double)numParts / (double)(totalFoldSendVol + totalFoldRecvVol) << std::endl;
  std::cout << "Overall send #msg: " << maxFoldSendMsgCount << " " << totalFoldSendMsgCount / numParts << " " <<
    totalFoldSendMsgCount << " " << (double)maxFoldSendMsgCount * (double)numParts / (double)totalFoldSendMsgCount <<
    std::endl;
  std::cout << "Overall recv #msg: " << maxFoldRecvMsgCount << " " << totalFoldRecvMsgCount / numParts << " " <<
    totalFoldRecvMsgCount << " " << (double)maxFoldRecvMsgCount * (double)numParts / (double)totalFoldRecvMsgCount <<
    std::endl;
  std::cout << "Overall send+recv #msg: " << maxFoldSendMsgCount + maxFoldRecvMsgCount << " " << (totalFoldSendMsgCount
      + totalFoldRecvMsgCount) / numParts << " " << totalFoldSendMsgCount + totalFoldRecvMsgCount << " " <<
    (double)(maxFoldSendMsgCount + maxFoldRecvMsgCount) * (double)numParts / (double)(totalFoldSendMsgCount +
        totalFoldRecvMsgCount) << std::endl;
  std::cout << "**************** End of Partition Stats **************************************\n\n";
}

// Distribution routines
template <class DataType, class IntType, class IdxType>
void Pacoss_Partitioner<DataType, IntType, IdxType>::distributeSparseStruct(
    const Pacoss_SparseStruct<DataType> &spStruct,
    const IntType numParts,
    const std::vector<IntType> &nzPart,
    const char * const outputFileNamePrefix)
{
  std::vector<IdxType> partNnz(numParts);
  for (auto i : nzPart) { partNnz[i]++; }
  // Open partitioned spStruct files, write their headers
  IntType maxOpenFileCount = 512;
  std::vector<FILE *> partTensorFile(numParts);
  for (IntType k = 0; k < (numParts + maxOpenFileCount - 1) / maxOpenFileCount; k++) {
    IntType partBegin = k * maxOpenFileCount;
    IntType partEnd = std::min((k + 1) * maxOpenFileCount, numParts);
    for (IntType i = partBegin, ie = partEnd; i < ie; i++) {
      std::string partTensorFileName(outputFileNamePrefix);
      partTensorFileName += ".part";
      { char numStr[100]; sprintfe(numStr, PRIMOD<IntType>(), i); partTensorFileName += numStr; }
      partTensorFile[i] = fopene(partTensorFileName.c_str(), "w");
      fprintfe(partTensorFile[i], STRCAT(PRIMOD<IntType>(), " ", PRIMOD<IdxType>(), "\n"), spStruct._order, partNnz[i]);
      for (IntType j = 0; j < spStruct._order; j++) { 
        fprintfe(partTensorFile[i], STRCAT(PRIMOD<IntType>(), " "), spStruct._dimSize[j]);
      }
      fprintfe(partTensorFile[i], "\n");
    }
    for (IdxType j = 0; j < spStruct._nnz; j++) {
      IntType partIdx = nzPart[j];
      if (partIdx >= partBegin && partIdx < partEnd) {
        for (IntType k = 0; k < spStruct._order; k++) {
          fprintfe(partTensorFile[partIdx], STRCAT(PRIMOD<IntType>(), " "), spStruct._idx[k][j] + 1); // 1-based conversion
        }
        fprintfe(partTensorFile[partIdx], PRIMOD<DataType>(), spStruct._val[j]);
        fprintfe(partTensorFile[partIdx], "\n");
      }
    }
    for (IntType i = partBegin, ie = partEnd; i < ie; i++) { fclosee(partTensorFile[i]); }
  }
}

template <class DataType, class IntType, class IdxType>
void Pacoss_Partitioner<DataType, IntType, IdxType>::distributeDimPartition(
    const char * const ssOutputFileNamePrefix,
    const IntType numParts,
    const std::vector<std::vector<IntType>> & dimPart,
    const char * const dimOutputFileNamePrefix)
{
  IntType order = (IntType)dimPart.size();

  // Find the set of owned rows in each dimension by each part.
  std::vector<std::vector<std::vector<IntType>>> ownedRows(order);
  for (IntType i = 0; i < order; i++) { ownedRows[i].resize(numParts); }
  for (IntType i = 0; i < order; i++) {
    auto & owri = ownedRows[i];
    for (IntType j = 0, je = (IntType)dimPart[i].size(); j < je; j++) {
      owri[dimPart[i][j]].push_back(j);
    }
  }

  // Create the dimension partition file for each part.
  for (IntType partIdx = 0; partIdx < numParts; partIdx++) {
    std::string partSsFileName(ssOutputFileNamePrefix);
    partSsFileName += ".part";
    { char numStr[100]; sprintfe(numStr, PRIMOD<IntType>(), partIdx); partSsFileName += numStr; }
    Pacoss_SparseStruct<DataType> ss;
    ss.load(partSsFileName.c_str());
    std::string partDimOutputFileName(dimOutputFileNamePrefix);
    partDimOutputFileName += ".part";
    { char numStr[100]; sprintfe(numStr, PRIMOD<IntType>(), partIdx); partDimOutputFileName += numStr; }
    FILE *partDimOutputFile = fopene(partDimOutputFileName.c_str(), "w");
    fprintfe(partDimOutputFile, STRCAT(PRIMOD<IntType>(), "\n"), order);
    std::vector<std::unordered_map<IntType, IntType>> map(order);
    for (IntType dim = 0; dim < order; dim++) {
      for (IdxType j = 0; j < ss._nnz; j++) { // Mark rows touched by nonzeros.
        IntType k = ss._idx[dim][j];
        map[dim][k] = dimPart[dim][k];
      }
      for (auto &i : ownedRows[dim][partIdx]) { map[dim][i] = partIdx; } // Mark rows that are owned by this part.
    }
    for (IntType dim = 0; dim < order; dim++) { 
      fprintfe(partDimOutputFile, STRCAT(PRIMOD<IntType>(), " "), map[dim].size());
    }
    fprintfe(partDimOutputFile, "\n\n");
    for (IntType dim = 0; dim < ss._order; dim++) {
      for (auto &i : map[dim]) {
        fprintfe(partDimOutputFile, STRCAT(PRIMOD<IntType>(), " ", PRIMOD<IntType>(), "\n"), i.first + 1, i.second);
      }
      fprintfe(partDimOutputFile, "\n");
    }
    fclosee(partDimOutputFile);
  }
}

template class Pacoss_Partitioner<double, int32_t>;
template class Pacoss_Partitioner<double, int32_t, int64_t>;
template class Pacoss_Partitioner<double, int64_t>;
template class Pacoss_Partitioner<float, int32_t>;
template class Pacoss_Partitioner<float, int32_t, int64_t>;
template class Pacoss_Partitioner<float, int64_t>;
