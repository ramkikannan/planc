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
    for (auto i : curDimPart) { fprintfe(file, STRCAT(PRIMOD<IntType>(), "\n"), i); }
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
  for (IntType i = 0; i < nzPartSize; i++) { fscanf(file, SCNMOD<IntType>(), &nzPart[i]); }
  fclosee(file);
}

template <class DataType, class IntType, class IdxType>
void Pacoss_Partitioner<DataType, IntType, IdxType>::saveNzPart(
    const std::vector<IntType> &nzPart,
    const char * const nzPartFileName)
{
  FILE *file = fopene(nzPartFileName, "w"); 
  fprintfe(file, "%zu\n\n", nzPart.size());
  for (auto i : nzPart) { fprintfe(file, STRCAT(PRIMOD<IntType>(), "\n"), i); }
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
    const Pacoss_SparseStruct<DataType> &spStruct,
    const IntType numParts,
    const std::vector<IntType> &nzPart,
    std::vector<std::vector<IntType>> &dimPart,
    double maxAllowedRowImbalance)
{
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
  dimPart.clear();
  for (IntType i = 0; i < spStruct._order; i++) { dimPart.emplace_back(spStruct._dimSize[i]); }
  for (IntType j = 0; j < spStruct._order; j++) {
    // Initialize rowCountList that keeps track of the parts having minimum number of rows at any instant.
    std::vector<IntType> rowCountListHead(spStruct._dimSize[j], -1);
    std::vector<IntType> rowCountListPrev(numParts, -1), rowCountListNext(numParts, -1);
    IntType minRowCount = 0;
    rowCountListNext[0] = rowCountListHead[0];
    rowCountListHead[0] = 0;
    for (IntType i = 1; i < numParts; i++) {
      rowCountListNext[i] = rowCountListHead[0];
      rowCountListPrev[rowCountListNext[i]] = i;
      rowCountListHead[0] = i;
    }
    // Sort rows with increasing number of ownerCandidates.
    std::vector<IntType> partRowCount(numParts);
    IntType maxPartRowCount = (IntType)((spStruct._dimSize[j] * maxAllowedRowImbalance) / (double)numParts);
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
    for (auto i : greedyRowOrder) {
      IntType minPartIdx = -1;
      // Assign the row to the part having the least recv communication volume
      for (auto partIdx : ownerCandidate[j][i]) {
        if (partRowCount[partIdx] < maxPartRowCount) { 
          if (minPartIdx == -1) { minPartIdx = partIdx; }
          else if (partRecvCommVol[partIdx] < partRecvCommVol[minPartIdx]) { minPartIdx = partIdx; }
        }
      }
      // If could not assign the row(due to imbalance or empty slice), assign it to a part having minimal num of rows
      if (minPartIdx == -1) { minPartIdx = rowCountListHead[minRowCount]; }
      // Assign to minPartIdx, increase the recv comm volume accordingly
      dimPart[j][i] = minPartIdx;
      partRowCount[minPartIdx]++;
      partRecvCommVol[minPartIdx] += (IntType)std::max((size_t)0, ownerCandidate[j][i].size() - 1);
      // Increase send comm volume of other parts
      for (auto partIdx : ownerCandidate[j][i]) {
        if (partIdx != minPartIdx) { partSendCommVol[partIdx]++; }
      }
      // Update rowCountList according to the added row to minPartIdx
      // Add to the new list first
      IntType prev = rowCountListPrev[minPartIdx];
      IntType next = rowCountListNext[minPartIdx];
      if (rowCountListHead[partRowCount[minPartIdx]] != -1) {
        rowCountListPrev[rowCountListHead[partRowCount[minPartIdx]]] = minPartIdx;
      }
      rowCountListNext[minPartIdx] = rowCountListHead[partRowCount[minPartIdx]];
      rowCountListPrev[minPartIdx] = -1;
      rowCountListHead[partRowCount[minPartIdx]] = minPartIdx;
      // Remove from the old list
      if (prev != -1) { rowCountListNext[prev] = next; }
      else { 
        rowCountListHead[partRowCount[minPartIdx] - 1] = next;
        if (partRowCount[minPartIdx] - 1 == minRowCount) {
          while (rowCountListHead[minRowCount] == -1) { minRowCount++; }
        }
      }
      if (next != -1) { rowCountListPrev[next] = prev; }
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
    std::vector<IntType> &nzPart)
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
  }
}


#define PACOSS_PARTITIONER_TOPK_PART 5

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
    std::vector<IntType> &nzPart)
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
  for (IdxType i = 0; i < spStruct._nnz ; i++) {
    if (sample[i]) {
      vertexWeights[rowIdx[i]]++;
      pinBegin[colIdx[i]]++;
    }
  }
  // Perform a prefix sum to compute the pinBegin pointers
  for (IntType i = 1; i <= numHedges ; i++) { pinBegin[i] += pinBegin[i - 1]; }
  // Fill in the pinIdx array in the second pass
  for (IdxType i = 0; i < spStruct._nnz; i++) { 
    if (sample[i]) { pinIdx[--pinBegin[colIdx[i]]] = (int)rowIdx[i]; }
  }
//  Pacoss_AssertEq(pinBegin[0], 0); Pacoss_AssertEq(pinBegin[numHedges], pinIdx.size());
  printfe("Partitioning the column-net hypergraph using PaToH...\n"); 
  partitionHypergraphPatoh(numVertex, numHedges, &pinBegin[0], &pinIdx[0], &vertexWeights[0], NULL, 1, (int)numRowParts,
      patohSettings, &rowPart[0]);

  if (strcmp(colPartitionType, "1d") == 0) { // Partition the columns randomly
    srand((unsigned)time(NULL));
    for (IntType i = 0; i < spStruct._dimSize[1]; i++) { colPart[i] = rand() % (int)numColParts; }
  } else { // Partition the columns using multi-constraint PaToH
    printfe("Creating the row-net hypergraph...\n"); 
    numVertex = spStruct._dimSize[1];
    numHedges = spStruct._dimSize[0];
    vertexWeights.resize(numVertex * numRowParts); std::fill(vertexWeights.begin(), vertexWeights.end(), 0);
    pinBegin.resize(numHedges + 1); std::fill(pinBegin.begin(), pinBegin.end(), 0);
    // Make the first pass to compute vertex weights and pinBegin
    for (IdxType i = 0; i < spStruct._nnz; i++) {
      vertexWeights[colIdx[i] * numRowParts + rowPart[rowIdx[i]]]++;
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
  // Finally, generate the nzPart vector according to the 2-D checkerboard partition
  nzPart.resize(spStruct._nnz);
  for (IdxType i = 0; i < spStruct._nnz; i++) {
    nzPart[i] = rowPart[rowIdx[i]] * numColParts + colPart[colIdx[i]];
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
  std::vector<FILE *> partTensorFile(numParts);
  for (IntType i = 0; i < numParts; i++) {
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

  for (IdxType i = 0; i < spStruct._nnz; i++) {
    IntType partIdx = nzPart[i];
    for (IntType k = 0; k < spStruct._order; k++) {
      fprintfe(partTensorFile[partIdx], STRCAT(PRIMOD<IntType>(), " "), spStruct._idx[k][i] + 1); // 1-based conversion
    }
    fprintfe(partTensorFile[partIdx], PRIMOD<DataType>(), spStruct._val[i]);
    fprintfe(partTensorFile[partIdx], "\n");
  }

  for (IntType i = 0; i < numParts; i++) { fclosee(partTensorFile[i]); }
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
