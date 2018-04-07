#include "pacoss-communicator.h"

template <class DataType>
Pacoss_Communicator<DataType>::Pacoss_Communicator(
    MPI_Comm comm,
    Pacoss_IntVector &idx,
    std::vector<Pacoss_IntPair> &vecOwnerMap)
{ 
  _comm = comm;
  TMPI_Comm_rank(_comm, &_procRank);
  TMPI_Comm_size(_comm, &_numProcs);

  std::vector<MPI_Request> _sendRequest(_numProcs);
  std::vector<MPI_Request> _recvRequest(_numProcs);
  std::vector<MPI_Status> _sendStatus(_numProcs);
  std::vector<MPI_Status> _recvStatus(_numProcs);

  // Compute list of processes for foldSend, find local proc indices, map each vector row with its local proc index
  _foldSendProcMapG2L[_procRank] = 0;
  for (Pacoss_Int i = 0, ie = (Pacoss_Int)vecOwnerMap.size(); i < ie; i++) {
    Pacoss_Int rowIdx, partIdx;
    if (_numProcs == 1) {
      rowIdx = i; partIdx = 0; // Assign all rows to part 0 for the sequential execution
    } else {
      rowIdx = vecOwnerMap[i].first; Pacoss_AssertGe(rowIdx, 0);
      partIdx = vecOwnerMap[i].second; Pacoss_AssertGe(partIdx, 0);
    }
    auto search = _foldSendProcMapG2L.find(partIdx);
    if (search == _foldSendProcMapG2L.end()) { // partIdx is not yet in the list of local procs, so add it
      Pacoss_Int mapSize = (Pacoss_Int)_foldSendProcMapG2L.size();
      _foldSendProcMapG2L[partIdx] = mapSize;
    }
    _foldSendRowToProcMapG2L[rowIdx] = partIdx; // Map vector row to the local index of its owner proc
  }
  _foldSendRowCount = (Pacoss_Int)_foldSendRowToProcMapG2L.size();
  _foldSendProcCount = (Pacoss_Int)_foldSendProcMapG2L.size();

  // Compute global-to-local and local-to-global process indices in foldSend
  // Set _foldSendProcMapG2L in a way to assure round-robin send order
  {
    Pacoss_IntVector procs; procs.reserve(_foldSendProcMapL2G.size());
    for (auto &i : _foldSendProcMapG2L) { procs.push_back(i.first); }
    std::sort(procs.begin(), procs.end());
    Pacoss_Int markerProcRank = 0, locProcIdx = 0;;
    while (procs[markerProcRank] != _procRank) { markerProcRank++; }
    // Put the processes with increasing index after _procRank first
    for (Pacoss_Int i = markerProcRank, ie = (Pacoss_Int)procs.size(); i < ie; i++) { 
      _foldSendProcMapG2L[procs[i]] = locProcIdx++;
    }
    // Then put the rest of the processes in increasing process rank
    for (Pacoss_Int i = 0 ; i < markerProcRank; i++) { _foldSendProcMapG2L[procs[i]] = locProcIdx++; }
  }
  // Allocate and compute _foldSendProcMapL2G by inverting _foldSendProcMapG2L
  _foldSendProcMapL2G.resize(_foldSendProcCount);
  for (auto i : _foldSendProcMapG2L) {
    _foldSendProcMapL2G[i.second] = i.first;
  }
  // Now that local proc indices are determined, finalize _foldSendProcMapG2L with computed local proc indices.
  for (auto &i : _foldSendRowToProcMapG2L) {
    i.second = _foldSendProcMapG2L[i.second];
  }

  // Compute the number of row indices belonging to each part
  // We put counts to _foldSendBegin initially. Then we perform prefix sum to obtain CSR-like pointers for each set.
  _foldSendBegin.resize(_foldSendProcCount + 1);
  for (auto i : _foldSendRowToProcMapG2L) {
    Pacoss_Int rowLocPart = i.second;
    _foldSendBegin[rowLocPart]++;
  }
  for (Pacoss_Int i = 1; i <= _foldSendProcCount; i++) {
    _foldSendBegin[i] += _foldSendBegin[i - 1];
  }
  Pacoss_AssertEq(_foldSendBegin[_foldSendProcCount], _foldSendRowCount);

  // Allocate and fill out _foldSendRowMapL2G using CSR-like pointers in _foldSendBegin for each proc
  _foldSendRowMapL2G.resize(_foldSendRowCount);
  for (auto i : _foldSendRowToProcMapG2L) {
    Pacoss_Int rowIdx = i.first;
    Pacoss_Int rowLocPart = i.second;
    _foldSendRowMapL2G[--_foldSendBegin[rowLocPart]] = rowIdx;
  }
  for (Pacoss_Int i = 0; i < _foldSendProcCount; i++) {
    std::sort(&_foldSendRowMapL2G[0] + _foldSendBegin[i], &_foldSendRowMapL2G[0] + _foldSendBegin[i + 1]);
  }

  // Now that we have _foldSendRowMapL2G, we can construct _foldSendRowMapG2L by inverting it
  for (Pacoss_Int i = 0; i < _foldSendRowCount; i++) {
    _foldSendRowMapG2L[_foldSendRowMapL2G[i]] = i;
  }
  // Convert global nonzero row indices to local
  for (Pacoss_Idx i = 0, ie = (Pacoss_Idx)idx.size(); i < ie; i++) { idx[i] = _foldSendRowMapG2L[idx[i]]; }

  // Start of communication phase to obtain info for foldRecv communication
  Pacoss_IntVector foldSendCount(_numProcs);
  Pacoss_IntVector foldRecvCount(_numProcs);
  // Notify all comm procs for their receive counts. Count the #procs that'll send a message to the current proc
  memset(&foldSendCount[0], 0, _numProcs * sizeof(foldSendCount[0]));
  for (Pacoss_Int i = 0; i < _foldSendProcCount; i++) {
    foldSendCount[_foldSendProcMapL2G[i]] = _foldSendBegin[i + 1] - _foldSendBegin[i];
  }
  // Send/recv foldSendCount/foldRecvCount to/from all processes. Also keep track of _foldRecvProcCount and
  // _foldRecvRowCount
  foldSendCount[_procRank] = 0; // Do not send anything to self
  _foldRecvRowCount = 0;
  _foldRecvProcCount = 0;
  TMPI_Alltoall(&foldSendCount[0], 1, &foldRecvCount[0], 1, _comm);
  for (Pacoss_Int i = 0; i < _numProcs; i++) {
    if (foldRecvCount[i] > 0) { // Current proc receives a message from proc i
      _foldRecvProcCount++;
      _foldRecvRowCount += foldRecvCount[i];
    }
  }

  // Allocate and compute _foldRecvProcMapL2G now that we have _foldRecvCount for all procs
  // We use inverse round-robin order for _foldRecv local process indices for efficiency
  _foldRecvProcMapL2G.resize(_foldRecvProcCount);
  for (Pacoss_Int i = (_procRank - 1 + _numProcs) % _numProcs, j = 0; i != _procRank; i = (i - 1 + _numProcs) % _numProcs)
  {
    if (foldRecvCount[i] > 0) _foldRecvProcMapL2G[j++] = i; // Add proc i to list of _foldRecv procs
  }

  // Now is time to send the indices of the rows that parts should receive. First Isend the indices
  // Skip self for send(i = 0)
  for (Pacoss_Int i = 1; i < _foldSendProcCount; i++) {
    Pacoss_Int sendProcIdx = _foldSendProcMapL2G[i];
    TMPI_Isend(&_foldSendRowMapL2G[_foldSendBegin[i]], _foldSendBegin[i + 1] -
        _foldSendBegin[i], sendProcIdx, (Pacoss_Int)0, _comm, &_sendRequest[i]);
  }

  // Allocate and compute _foldRecvBegin
  _foldRecvBegin.resize(_foldRecvProcCount + 1);
  _foldRecvBegin[0] = 0;
  for (Pacoss_Int i = 0; i < _foldRecvProcCount; i++) {
    _foldRecvBegin[i + 1] = _foldRecvBegin[i] + foldRecvCount[_foldRecvProcMapL2G[i]];
  }
  // Allocate and Irecv _foldRecvReduceMap indices, then convert them to local indices
  _foldRecvReduceMap.resize(_foldRecvRowCount);
  for (Pacoss_Int i = 0; i < _foldRecvProcCount; i++) {
    Pacoss_Int recvProcIdx = _foldRecvProcMapL2G[i];
    TMPI_Irecv(&_foldRecvReduceMap[_foldRecvBegin[i]], _foldRecvBegin[i + 1] - _foldRecvBegin[i], recvProcIdx,
        (Pacoss_Int)0, _comm, &_recvRequest[i]);
  }
  TMPI_Waitall(_foldSendProcCount - 1, &_sendRequest[1], &_sendStatus[1]);
  for (Pacoss_Int i = 0; i < _foldRecvProcCount; i++) {
    Pacoss_Int locRecvProcIdx;
    TMPI_Waitany(_foldRecvProcCount, &_recvRequest[0], &locRecvProcIdx, &_recvStatus[0]);
    for (Pacoss_Int i = _foldRecvBegin[locRecvProcIdx], ie = _foldRecvBegin[locRecvProcIdx + 1]; i < ie; i++) {
      // Convert received global row indices to local
      _foldRecvReduceMap[i] = _foldSendRowMapG2L[_foldRecvReduceMap[i]]; 
    }
  }  

  _totalSendRecvProcCount = _foldSendProcCount + _foldRecvProcCount - 1;
  _request.resize(_totalSendRecvProcCount);
  _status.resize(_totalSendRecvProcCount);

  // Setup all-to-all communication structures.
  _foldSendCounts.resize(_numProcs, 0);
  _foldSendDispls.resize(_numProcs, 0);
  _foldRecvCounts.resize(_numProcs, 0);
  _foldRecvDispls.resize(_numProcs, 0);
  for (Pacoss_Int i = 1; i < _foldSendProcCount; i++) {
    Pacoss_Int procIdx = _foldSendProcMapL2G[i];
    _foldSendCounts[procIdx] = _foldSendBegin[i + 1] - _foldSendBegin[i];
    _foldSendDispls[procIdx] = _foldSendBegin[i];
  }
  for (Pacoss_Int i = 0; i < _foldRecvProcCount; i++) {
    Pacoss_Int procIdx = _foldRecvProcMapL2G[i];
    _foldRecvCounts[procIdx] = _foldRecvBegin[i + 1] - _foldRecvBegin[i];
    _foldRecvDispls[procIdx] = _foldRecvBegin[i];
  }
}

template <class DataType>
void Pacoss_Communicator<DataType>::foldCommA2A(
    DataType *commBuf,
    DataType *auxBuf,
    Pacoss_Int unitMsgSize)
{
  Pacoss_Int *sendCounts, *sendDispls, *recvCounts, *recvDispls;
  if (unitMsgSize == 1) { // No need for scaling
    sendCounts = &_foldSendCounts[0];
    sendDispls = &_foldSendDispls[0];
    recvCounts = &_foldRecvCounts[0];
    recvDispls = &_foldRecvDispls[0];
  } else { // Scale the send/receive counts and displacements by unitMsgSize
    if ((Pacoss_Int)_scSendCounts.size() < _numProcs) {
      _scSendCounts.resize(_numProcs);
      _scSendDispls.resize(_numProcs);
      _scRecvCounts.resize(_numProcs);
      _scRecvDispls.resize(_numProcs);
    }
    for (Pacoss_Int i = 0; i < _numProcs; i++) {
      _scSendCounts[i] = _foldSendCounts[i] * unitMsgSize;
      _scSendDispls[i] = _foldSendDispls[i] * unitMsgSize;
      _scRecvCounts[i] = _foldRecvCounts[i] * unitMsgSize;
      _scRecvDispls[i] = _foldRecvDispls[i] * unitMsgSize;
    }
    sendCounts = &_scSendCounts[0];
    sendDispls = &_scSendDispls[0];
    recvCounts = &_scRecvCounts[0];
    recvDispls = &_scRecvDispls[0];
  }
  // Now perform the communication.
  TMPI_Alltoallv(&commBuf[0], &sendCounts[0], &sendDispls[0], &auxBuf[0], &recvCounts[0], &recvDispls[0], _comm);
  // Now perform the reduction for each received message.
  for (Pacoss_Int locProcIdx = 0; locProcIdx < _foldRecvProcCount; locProcIdx++) {
    for (Pacoss_Int rowIdx = _foldRecvBegin[locProcIdx]; rowIdx < _foldRecvBegin[locProcIdx + 1]; rowIdx++) {
      DataType *auxBufRow = &auxBuf[rowIdx * unitMsgSize];
      DataType *commBufRow = &commBuf[_foldRecvReduceMap[rowIdx] * unitMsgSize];
      for (Pacoss_Int k = 0; k < unitMsgSize; k++) { commBufRow[k] += auxBufRow[k]; }
    }
  }
}

template <class DataType>
void Pacoss_Communicator<DataType>::foldCommA2A(
    DataType *commBuf,
    Pacoss_Int unitMsgSize)
{
  if (_internalBuf.size() < _foldRecvBegin[_foldRecvProcCount] * unitMsgSize * sizeof(DataType)) { 
    _internalBuf.resize(_foldRecvBegin[_foldRecvProcCount] * unitMsgSize * sizeof(DataType));
  }
  foldCommA2A(commBuf, (DataType *)&_internalBuf[0], unitMsgSize);
}

template <class DataType>
void Pacoss_Communicator<DataType>::foldCommBegin(
    DataType *commBuf,
    DataType *auxBuf,
    Pacoss_Int unitMsgSize)
{
  // Perform all Isends
  for (Pacoss_Int i = 1; i < _foldSendProcCount; i++) { // Skip the self, which is the process 0.
    Pacoss_Int procIdx = _foldSendProcMapL2G[i];
    DataType *msgBeg = &commBuf[(size_t)_foldSendBegin[i] * unitMsgSize];
    Pacoss_Int msgSize = (_foldSendBegin[i + 1] - _foldSendBegin[i]) * unitMsgSize;
    TMPI_Isend(msgBeg, msgSize, procIdx, (Pacoss_Int)0, _comm, &_request[_foldRecvProcCount + i - 1]);
  }
  // Perform all Irecvs
  for (Pacoss_Int i = 0; i < _foldRecvProcCount; i++) {
    Pacoss_Int procIdx = _foldRecvProcMapL2G[i];
    DataType *msgBeg = &auxBuf[(size_t)_foldRecvBegin[i] * unitMsgSize];
    Pacoss_Int msgSize = (_foldRecvBegin[i + 1] - _foldRecvBegin[i]) * unitMsgSize;
    TMPI_Irecv(msgBeg, msgSize, procIdx, (Pacoss_Int)0, _comm, &_request[i]);
  }
}

template <class DataType>
void Pacoss_Communicator<DataType>::foldCommFinish(
    DataType *commBuf,
    DataType *auxBuf,
    Pacoss_Int unitMsgSize)
{
  for (Pacoss_Int i = 0; i < _totalSendRecvProcCount; i++) { // Wait for all unfinished requests to finish.
    Pacoss_Int reqIdx;
    MPI_Status status;
    TMPI_Waitany(_totalSendRecvProcCount, &_request[0], &reqIdx, &status);
    if (reqIdx < _foldRecvProcCount) { // A receive request is finished; process the received message.
      // Reduce each row of the message to the corresponding row in the commBuf.
      for (Pacoss_Int rowIdx = _foldRecvBegin[reqIdx]; rowIdx < _foldRecvBegin[reqIdx + 1]; rowIdx++) {
        DataType *auxBufRow = &auxBuf[(size_t)rowIdx * unitMsgSize];
        DataType *commBufRow = &commBuf[(size_t)_foldRecvReduceMap[rowIdx] * unitMsgSize];
        for (Pacoss_Int k = 0; k < unitMsgSize; k++) { commBufRow[k] += auxBufRow[k]; }
      }
    }
  }
}

template <class DataType>
void Pacoss_Communicator<DataType>::foldCommBegin(
    DataType *commBuf,
    Pacoss_Int unitMsgSize)
{
  if (_internalBuf.size() < _foldRecvBegin[(size_t)_foldRecvProcCount] * unitMsgSize * sizeof(DataType)) { 
    _internalBuf.resize(_foldRecvBegin[(size_t)_foldRecvProcCount] * unitMsgSize * sizeof(DataType));
  }
  foldCommBegin(commBuf, (DataType *)&_internalBuf[0], unitMsgSize);
}

template <class DataType>
void Pacoss_Communicator<DataType>::foldCommFinish(
    DataType *commBuf,
    Pacoss_Int unitMsgSize)
{
  if (_internalBuf.size() < _foldRecvBegin[(size_t)_foldRecvProcCount] * unitMsgSize * sizeof(DataType)) { 
    _internalBuf.resize(_foldRecvBegin[(size_t)_foldRecvProcCount] * unitMsgSize * sizeof(DataType));
  }
  foldCommFinish(commBuf, (DataType *)&_internalBuf[0], unitMsgSize);
}

template <class DataType>
void Pacoss_Communicator<DataType>::expCommA2A(
    DataType *commBuf,
    DataType *auxBuf,
    Pacoss_Int unitMsgSize)
{
  // Scatter expand data to auxBuf.
  for (Pacoss_Int rowIdx = 0, rowEnd = _expSendBegin[_expSendProcCount]; rowIdx < rowEnd; rowIdx++) {
    DataType *auxBufRow = &auxBuf[rowIdx * unitMsgSize];
    DataType *commBufRow = &commBuf[_expSendReduceMap[rowIdx] * unitMsgSize];
    for (Pacoss_Int k = 0; k < unitMsgSize; k++) { auxBufRow[k] = commBufRow[k]; }
  }
  Pacoss_Int *sendCounts, *sendDispls, *recvCounts, *recvDispls;
  if (unitMsgSize == 1) { // No need for scaling
    sendCounts = &_expSendCounts[0];
    sendDispls = &_expSendDispls[0];
    recvCounts = &_expRecvCounts[0];
    recvDispls = &_expRecvDispls[0];
  } else { // Scale the send/receive counts and displacements by unitMsgSize
    if ((Pacoss_Int)_scSendCounts.size() < _numProcs) {
      _scSendCounts.resize(_numProcs);
      _scSendDispls.resize(_numProcs);
      _scRecvCounts.resize(_numProcs);
      _scRecvDispls.resize(_numProcs);
    }
    for (Pacoss_Int i = 0; i < _numProcs; i++) {
      _scSendCounts[i] = _expSendCounts[i] * unitMsgSize;
      _scSendDispls[i] = _expSendDispls[i] * unitMsgSize;
      _scRecvCounts[i] = _expRecvCounts[i] * unitMsgSize;
      _scRecvDispls[i] = _expRecvDispls[i] * unitMsgSize;
    }
    sendCounts = &_scSendCounts[0];
    sendDispls = &_scSendDispls[0];
    recvCounts = &_scRecvCounts[0];
    recvDispls = &_scRecvDispls[0];
  }
  // Now perform the communication.
  TMPI_Alltoallv(&auxBuf[0], &sendCounts[0], &sendDispls[0], &commBuf[0], &recvCounts[0], &recvDispls[0], _comm);
}

template <class DataType>
void Pacoss_Communicator<DataType>::expCommA2A(
    DataType *commBuf,
    Pacoss_Int unitMsgSize)
{
  if (_internalBuf.size() < _expRecvBegin[_expRecvProcCount] * unitMsgSize * sizeof(DataType)) { 
    _internalBuf.resize(_expRecvBegin[_expRecvProcCount] * unitMsgSize * sizeof(DataType));
  }
  expCommA2A(commBuf, (DataType *)&_internalBuf[0], unitMsgSize);
}

template <class DataType>
void Pacoss_Communicator<DataType>::expCommBegin(
    DataType *commBuf,
    DataType *auxBuf,
    Pacoss_Int unitMsgSize)
{
  // Scatter expand data to auxBuf.
  for (Pacoss_Int rowIdx = 0, rowEnd = _expSendBegin[_expSendProcCount]; rowIdx < rowEnd; rowIdx++) {
    DataType *auxBufRow = &auxBuf[(size_t)rowIdx * unitMsgSize];
    DataType *commBufRow = &commBuf[(size_t)_expSendReduceMap[rowIdx] * unitMsgSize];
    for (Pacoss_Int k = 0; k < unitMsgSize; k++) { auxBufRow[k] = commBufRow[k]; }
  }
  // Perform all Isends.
  for (Pacoss_Int i = 0; i < _expSendProcCount; i++) {
    Pacoss_Int procIdx = _expSendProcMapL2G[i];
    DataType *msgBeg = &auxBuf[(size_t)_expSendBegin[i] * unitMsgSize];
    Pacoss_Int msgSize = (_expSendBegin[i + 1] - _expSendBegin[i]) * unitMsgSize;
    TMPI_Isend(msgBeg, msgSize, procIdx, (Pacoss_Int)0, _comm, &_request[i]);
  }
  // Perform all Irecvs into commBuf in-place.
  for (Pacoss_Int i = 1; i < _expRecvProcCount; i++) {
    Pacoss_Int procIdx = _expRecvProcMapL2G[i];
    DataType *msgBeg = &commBuf[(size_t)_expRecvBegin[i] * unitMsgSize];
    Pacoss_Int msgSize = (_expRecvBegin[i + 1] - _expRecvBegin[i]) * unitMsgSize;
    TMPI_Irecv(msgBeg, msgSize, procIdx, (Pacoss_Int)0, _comm, &_request[_expSendProcCount + i - 1]);
  }
}

template <class DataType>
void Pacoss_Communicator<DataType>::expCommFinish(
    DataType *commBuf,
    DataType *auxBuf,
    Pacoss_Int unitMsgSize)
{
  // Wait for all outstanding requests to finish.
  TMPI_Waitall(_totalSendRecvProcCount, &_request[0], &_status[0]);
}

template <class DataType>
void Pacoss_Communicator<DataType>::expCommBegin(
    DataType *commBuf,
    Pacoss_Int unitMsgSize)
{
  if (_internalBuf.size() < _expSendBegin[(size_t)_expSendProcCount] * unitMsgSize * sizeof(DataType)) {
    _internalBuf.resize(_expSendBegin[(size_t)_expSendProcCount] * unitMsgSize * sizeof(DataType));
  }
  expCommBegin(commBuf, (DataType *)&_internalBuf[0], unitMsgSize);
}

template <class DataType>
void Pacoss_Communicator<DataType>::expCommFinish(
    DataType *commBuf,
    Pacoss_Int unitMsgSize)
{
  if (_internalBuf.size() < _expSendBegin[(size_t)_expSendProcCount] * unitMsgSize * sizeof(DataType)) {
    _internalBuf.resize(_expSendBegin[(size_t)_expSendProcCount] * unitMsgSize * sizeof(DataType));
  }
  expCommFinish(commBuf, (DataType *)&_internalBuf[0], unitMsgSize);
}

template <class DataType>
void Pacoss_Communicator<DataType>::gatherData(
    DataType *commBuf,
    DataType *gatherBuf,
    Pacoss_Int unitMsgSize,
    Pacoss_Int root)
{
  Pacoss_Int ownedCount = localOwnedRowCount();
  Pacoss_Int totalRows = 0;
  Pacoss_IntVector rowIdx;
  Pacoss_IntVector recvCount; if (_procRank == root) { recvCount.resize(_numProcs); }
  Pacoss_IntVector recvBegin; if (_procRank == root) { recvBegin.resize(_numProcs + 1); }
  std::vector<DataType> permGatherBuf;

  TMPI_Gather(&ownedCount, (Pacoss_Int)1, &recvCount[0], (Pacoss_Int)1, root, _comm); 
  if (_procRank == root) {
    recvBegin[0] = 0;
    for (Pacoss_Int i = 0, ie = (Pacoss_Int)recvCount.size(); i < ie; i++) { 
      recvBegin[i + 1] = recvBegin[i] + recvCount[i];
    }
    totalRows = recvBegin[_numProcs];
    rowIdx.resize(totalRows);
  }
  TMPI_Gatherv(&_foldSendRowMapL2G[0], ownedCount, &rowIdx[0], &recvCount[0], &recvBegin[0], root, _comm);
  if (_procRank == root) {
    for (Pacoss_Int i = 0, ie = (Pacoss_Int)recvCount.size(); i < ie; i++) {
      recvBegin[i] *= unitMsgSize;
      recvCount[i] *= unitMsgSize;
    }
    permGatherBuf.resize(totalRows * unitMsgSize);
  }
  TMPI_Gatherv(commBuf, ownedCount * unitMsgSize, &permGatherBuf[0], &recvCount[0], &recvBegin[0], root, _comm);
  if (_procRank == root) {
    for (Pacoss_Int i = 0; i < totalRows; i++) {
      DataType *gatherBufRow = &gatherBuf[rowIdx[i] * unitMsgSize];
      Pacoss_AssertLt(rowIdx[i], totalRows);
      DataType *permGatherBufRow = &permGatherBuf[i * unitMsgSize];
      for (Pacoss_Int j = 0; j < unitMsgSize; j++) {
        gatherBufRow[j] = permGatherBufRow[j];
      }
    }
  }
}

template <class DataType>
void Pacoss_Communicator<DataType>::scatterData(
    DataType *commBuf,
    DataType *scatterBuf,
    Pacoss_Int unitMsgSize,
    Pacoss_Int root)
{
  Pacoss_Int ownedCount = localOwnedRowCount();
  Pacoss_Int totalRows = 0;
  Pacoss_IntVector rowIdx;
  Pacoss_IntVector recvCount; if (_procRank == root) { recvCount.resize(_numProcs); }
  Pacoss_IntVector recvBegin; if (_procRank == root) { recvBegin.resize(_numProcs + 1); }
  std::vector<DataType> permGatherBuf;

  TMPI_Gather(&ownedCount, (Pacoss_Int)1, &recvCount[0], (Pacoss_Int)1, root, _comm); 
  if (_procRank == root) {
    recvBegin[0] = 0;
    for (Pacoss_Int i = 0, ie = (Pacoss_Int)recvCount.size(); i < ie; i++) {
      recvBegin[i + 1] = recvBegin[i] + recvCount[i];
    }
    totalRows = recvBegin[_numProcs];
    rowIdx.resize(totalRows);
  }
  TMPI_Gatherv(&_foldSendRowMapL2G[0], ownedCount, &rowIdx[0], &recvCount[0], &recvBegin[0], root, _comm);
  if (_procRank == root) {
    for (Pacoss_Int i = 0, ie = (Pacoss_Int)recvCount.size(); i < ie; i++) {
      recvBegin[i] *= unitMsgSize;
      recvCount[i] *= unitMsgSize;
    }
    permGatherBuf.resize(totalRows * unitMsgSize);
  }
  if (_procRank == root) {
    for (Pacoss_Int i = 0; i < totalRows; i++) {
      DataType *scatterBufRow = &scatterBuf[rowIdx[i] * unitMsgSize];
      Pacoss_AssertLt(rowIdx[i], totalRows);
      DataType *permGatherBufRow = &permGatherBuf[i * unitMsgSize];
      for (Pacoss_Int j = 0; j < unitMsgSize; j++) {
        permGatherBufRow[j] = scatterBufRow[j];
      }
    }
  }
  TMPI_Scatterv(&permGatherBuf[0], &recvCount[0], &recvBegin[0], commBuf, ownedCount * unitMsgSize, root, _comm);
}

template <class DataType>
Pacoss_Int Pacoss_Communicator<DataType>::localRowCount()
{ return (Pacoss_Int)_foldSendRowMapL2G.size(); }

template <class DataType>
Pacoss_Int Pacoss_Communicator<DataType>::localOwnedRowCount()
{ return _foldSendBegin[1]; }

template <class DataType>
void Pacoss_Communicator<DataType>::getCommStats(std::unordered_map<std::string, Pacoss_Int> &stats)
{
//  Pacoss_Int order = _foldSendProcMapL2G.size();
//  for (Pacoss_Int i = 0; i < order; i++) {
//    Pacoss_Int temp;
//    std::string modeString = std::to_string((long long)i);
//
//    stats["TOTAL_ROW_COUNT_" + modeString] =
//      stats["MAX_ROW_COUNT_" + modeString] =
//      stats["MAX_ROW_COUNT_" + modeString] = _foldSendBegin[i][_foldSendProcCount[i]];
//    TMPI_Allreduce((int *)MPI_IN_PLACE, &stats["TOTAL_ROW_COUNT_" + modeString], 1, MPI_SUM,
//        _comm);
//    TMPI_Allreduce((int *)MPI_IN_PLACE, &stats["MAX_ROW_COUNT_" + modeString], 1, MPI_MAX,
//        _comm);
//
//    // Fold send/receive communication volume
//    temp = _foldSendBegin[i][_foldSendProcCount[i]] - _foldSendBegin[i][1];
//    stats["TOTAL_FOLD_SEND_VOLUME_" + modeString] =
//      stats["MAX_FOLD_SEND_VOLUME_" + modeString] =
//      stats["MAX_FOLD_SEND_VOLUME_" + modeString] = temp;
//    TMPI_Allreduce((int *)MPI_IN_PLACE, &stats["TOTAL_FOLD_SEND_VOLUME_" + modeString], 1, MPI_SUM,
//        _comm);
//    TMPI_Allreduce((int *)MPI_IN_PLACE, &stats["MAX_FOLD_SEND_VOLUME_" + modeString], 1, MPI_MAX,
//        _comm);
//    temp = _foldRecvBegin[i][_foldRecvProcCount[i]] - _foldRecvBegin[i][0];
//    stats["TOTAL_FOLD_RECV_VOLUME_" + modeString] =
//      stats["MAX_FOLD_RECV_VOLUME_" + modeString] =
//      stats["MAX_FOLD_RECV_VOLUME_" + modeString] = temp;
//    TMPI_Allreduce((int *)MPI_IN_PLACE, &stats["TOTAL_FOLD_RECV_VOLUME_" + modeString], 1, MPI_SUM,
//        _comm);
//    TMPI_Allreduce((int *)MPI_IN_PLACE, &stats["MAX_FOLD_RECV_VOLUME_" + modeString], 1, MPI_MAX,
//        _comm);
//    // Total/max communication volume. Multiply by two to reflect both fold and expand phases
//    stats["TOTAL_COMM_VOLUME_" + modeString] = 2 * (stats["TOTAL_FOLD_SEND_VOLUME_" +
//      modeString] + stats["TOTAL_FOLD_RECV_VOLUME_" + modeString]);
//    stats["MAX_COMM_VOLUME_" + modeString] = 2 * (stats["MAX_FOLD_SEND_VOLUME_" +
//      modeString] + stats["MAX_FOLD_RECV_VOLUME_" + modeString]);
//    stats["MAX_COMM_VOLUME_" + modeString] = 2 * (stats["MAX_FOLD_SEND_VOLUME_" +
//      modeString] + stats["MAX_FOLD_RECV_VOLUME_" + modeString]);
//
//    // Communication latency (number of messages)
//    temp = _foldSendProcCount[i] - 1;
//    stats["TOTAL_FOLD_SEND_MSG_COUNT_" + modeString] =
//      stats["MAX_FOLD_SEND_MSG_COUNT_" + modeString] =
//      stats["MAX_FOLD_SEND_MSG_COUNT_" + modeString] = temp;
//    TMPI_Allreduce((int *)MPI_IN_PLACE, &stats["TOTAL_FOLD_SEND_MSG_COUNT_" + modeString], 1, MPI_SUM,
//        _comm);
//    TMPI_Allreduce((int *)MPI_IN_PLACE, &stats["MAX_FOLD_SEND_MSG_COUNT_" + modeString], 1, MPI_MAX,
//        _comm);
//    temp = _foldRecvProcCount[i];
//    stats["TOTAL_FOLD_RECV_MSG_COUNT_" + modeString] =
//      stats["MAX_FOLD_RECV_MSG_COUNT_" + modeString] =
//      stats["MAX_FOLD_RECV_MSG_COUNT_" + modeString] = temp;
//    TMPI_Allreduce((int *)MPI_IN_PLACE, &stats["TOTAL_FOLD_RECV_MSG_COUNT_" + modeString], 1, MPI_SUM,
//        _comm);
//    TMPI_Allreduce((int *)MPI_IN_PLACE, &stats["MAX_FOLD_RECV_MSG_COUNT_" + modeString], 1, MPI_MAX,
//        _comm);
//    stats["TOTAL_MSG_COUNT_" + modeString] = stats["TOTAL_FOLD_SEND_MSG_COUNT_" +
//      modeString] + stats["TOTAL_FOLD_RECV_MSG_COUNT_" + modeString];
//    stats["MAX_MSG_COUNT_" + modeString] = stats["MAX_FOLD_SEND_MSG_COUNT_" +
//      modeString] + stats["MAX_FOLD_RECV_MSG_COUNT_" + modeString];
//    stats["MAX_MSG_COUNT_" + modeString] = stats["MAX_FOLD_SEND_MSG_COUNT_" +
//      modeString] + stats["MAX_FOLD_RECV_MSG_COUNT_" + modeString];
//  }
}

template <class DataType>
void Pacoss_Communicator<DataType>::loadDistributedDimPart(
    const char * const fileName,
    std::vector<std::vector<Pacoss_IntPair>> &dimPart)
{
  FILE *file = fopene(fileName, "r");
  Pacoss_Int order;
  fscanfe(file, " %" PRIINT, &order);
  dimPart.resize(order);
  Pacoss_IntVector mapSize(order);
  for (Pacoss_Int i = 0; i < order; i++) { fscanf(file, " %" PRIINT, &mapSize[i]); }
  for (Pacoss_Int i = 0; i < order; i++) {
    dimPart[i].resize(mapSize[i]);
    for (Pacoss_Int j = 0; j < mapSize[i]; j++) {
      fscanfe(file, " %" PRIINT " %" PRIINT, &(dimPart[i][j].first), &(dimPart[i][j].second));
      dimPart[i][j].first--; // Convert to 0-based
    }
  }
  fclosee(file);
}

template class Pacoss_Communicator<double>;
template class Pacoss_Communicator<float>;
template class Pacoss_Communicator<int32_t>;
template class Pacoss_Communicator<int64_t>;
