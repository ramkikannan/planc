#include "pacoss-common.h"
#include "tmpi.h"

template <class DataType>
class Pacoss_Communicator
{
  public:
    MPI_Comm _comm;
    Pacoss_Int _procRank,
        _numProcs;
    std::vector<char> _internalBuf;

    std::vector<MPI_Request> _request;
    std::vector<MPI_Status> _status;
    // Structs for fold communication
    Pacoss_Int _totalSendRecvProcCount;
    Pacoss_Int _foldSendProcCount;
    Pacoss_Int _foldSendRowCount;
    Pacoss_IntVector _foldSendProcMapL2G;
    Pacoss_IntVector _foldSendRowMapL2G;
    Pacoss_IntVector _foldSendBegin;
    Pacoss_IntMap _foldSendProcMapG2L;
    Pacoss_IntMap _foldSendRowToProcMapG2L;
    Pacoss_IntMap _foldSendRowMapG2L;
    Pacoss_Int _foldRecvProcCount;
    Pacoss_Int _foldRecvRowCount;
    Pacoss_IntVector _foldRecvBegin;
    Pacoss_IntVector _foldRecvProcMapL2G;
    Pacoss_IntVector _foldRecvReduceMap;
    // Structs for expand communication, which is the inverse of fold communication.
    Pacoss_Int & _expRecvProcCount = _foldSendProcCount;
    Pacoss_Int & _expRecvRowCount = _foldSendRowCount;
    Pacoss_IntVector & _expRecvProcMapL2G = _foldSendProcMapL2G;
    Pacoss_IntVector & _expRecvRowMapL2G = _foldSendRowMapL2G;
    Pacoss_IntVector & _expRecvBegin = _foldSendBegin;
    Pacoss_IntMap & _expRecvProcMapG2L = _foldSendProcMapG2L;
    Pacoss_IntMap & _expRecvRowToProcMapG2L = _foldSendRowToProcMapG2L;
    Pacoss_IntMap & _expRecvRowMapG2L = _foldSendRowMapG2L;
    Pacoss_Int & _expSendProcCount = _foldRecvProcCount;
    Pacoss_Int & _expSendRowCount = _foldRecvRowCount;
    Pacoss_IntVector & _expSendBegin = _foldRecvBegin;
    Pacoss_IntVector & _expSendProcMapL2G = _foldRecvProcMapL2G;
    Pacoss_IntVector & _expSendReduceMap = _foldRecvReduceMap;
    // Structs for all-to-all communication
    Pacoss_IntVector _foldSendCounts;
    Pacoss_IntVector _foldSendDispls;
    Pacoss_IntVector _foldRecvCounts;
    Pacoss_IntVector _foldRecvDispls;
    Pacoss_IntVector & _expSendCounts = _foldRecvCounts;
    Pacoss_IntVector & _expSendDispls = _foldRecvDispls;
    Pacoss_IntVector & _expRecvCounts = _foldSendCounts;
    Pacoss_IntVector & _expRecvDispls = _foldSendDispls;
    Pacoss_IntVector _scSendCounts;
    Pacoss_IntVector _scSendDispls;
    Pacoss_IntVector _scRecvCounts;
    Pacoss_IntVector _scRecvDispls;

    Pacoss_Communicator(MPI_Comm comm, Pacoss_IntVector &idx, std::vector<Pacoss_IntPair> &vecOwnerMap);

    void foldCommBegin(DataType *commBuf, DataType *auxBuf, Pacoss_Int unitMsgSize);
    void foldCommFinish(DataType *commBuf, DataType *auxBuf, Pacoss_Int unitMsgSize);
    void foldCommBegin(DataType *commBuf, Pacoss_Int unitMsgSize);
    void foldCommFinish(DataType *commBuf, Pacoss_Int unitMsgSize);

    void expCommBegin(DataType *commBuf, DataType *auxBuf, Pacoss_Int unitMsgSize);
    void expCommFinish(DataType *commBuf, DataType *auxBuf, Pacoss_Int unitMsgSize);
    void expCommBegin(DataType *commBuf, Pacoss_Int unitMsgSize);
    void expCommFinish(DataType *commBuf, Pacoss_Int unitMsgSize);

    void foldCommA2A(DataType *commBuf, DataType *auxBuf, Pacoss_Int unitMsgSize);
    void foldCommA2A(DataType *commBuf, Pacoss_Int unitMsgSize);

    void expCommA2A(DataType *commBuf, DataType *auxBuf, Pacoss_Int unitMsgSize);
    void expCommA2A(DataType *commBuf, Pacoss_Int unitMsgSize);

    void gatherData(DataType *commBuf, DataType *gatherBuf, Pacoss_Int unitMsgSize, Pacoss_Int root);
    void scatterData(DataType *commBuf, DataType *gatherBuf, Pacoss_Int unitMsgSize, Pacoss_Int root);

    Pacoss_Int localRowCount();
    Pacoss_Int localOwnedRowCount();

    static void loadDistributedDimPart(const char * const fileName, std::vector<std::vector<Pacoss_IntPair>> &dimPart);

    void getCommStats(std::unordered_map<std::string, Pacoss_Int> &stats);
};
