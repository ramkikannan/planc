#ifndef TMPI_H
#define TMPI_H

#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#ifndef TMPI_NOMPI
#include "mpi.h"
#else
#include "mpi-backup.hb"
#endif

// The rank of the current process and the total number of processes
extern int TMPI_ProcRank;
extern int TMPI_NumProcs;

// Normal output and test output files for each MPI rank
extern FILE *TMPI_OutFile;
extern FILE *TMPI_TestFile;
extern char TMPI_OutFileName[];
extern char TMPI_TestFileName[];

// Exception class for TMPI
class tmpi_exception : public std::exception
{
  private:
    std::string _what;

  public:
    tmpi_exception(const char * const msg) { _what = std::string(msg); }
    tmpi_exception(const char * const msg, int sourceLine, const char * const sourceFile, const char * const function) { 
      _what = "\n";
      if (TMPI_NumProcs > 1) { _what += "rank" + std::to_string(TMPI_ProcRank) + ":"; }
      _what += std::string(sourceFile) + std::string(":");
      _what += std::to_string(sourceLine) + std::string(":");
      _what += std::string(function) + std::string(": ");
      _what += std::string(msg);
    }
    tmpi_exception() { _what = std::string("Uninitialized exception."); }
    ~tmpi_exception() {  }
    virtual const char* what() const throw() {
      return _what.c_str();
    }
};

// Print routine that is only executed at the root process (with rank 0).
#define trprintf(...) { \
  if (TMPI_ProcRank == 0) { \
    printf(__VA_ARGS__); \
  } \
}

// Print routine that outputs to TMPI_OutFile. For the process with rank 0 it also outputs to the stdout.
#define tmprintf(...) { \
  fprintf(TMPI_OutFile, __VA_ARGS__);  \
  fflush(TMPI_OutFile); \
  if (TMPI_ProcRank == 0) { \
    printf(__VA_ARGS__); \
  } \
}

// Print routine that outputs to TMPI_TestOutFile.
#define tmtprintf(...) { \
  fprintf(TMPI_TestFile, __VA_ARGS__);  \
}

// Check whether an MPI statement returns MPI_SUCCESS. It throws a tmpi_exception otherwise.
#ifdef TMPI_NOMPI
#define TMPI_CHECK_ERROR(stmt) 
#else
#define TMPI_CHECK_ERROR(stmt) { \
  auto errCode = stmt; \
  if (errCode != MPI_SUCCESS) { \
    char msg[100]; \
    sprintf(msg, "Failed with the error code %d.\n", errCode); \
    throw tmpi_exception(msg, __LINE__, __FILE__, __func__); \
  } \
}
#endif

//// MPI datatypes and templated routines that return the corresponding datatypes
//typedef enum
//{
//  type_int = MPI_INT,
//  type_long = MPI_LONG,
//  type_longlong = MPI_LONG_LONG,
//  type_float = MPI_FLOAT,
//  type_double = MPI_DOUBLE
//} TMPI_Datatype;

template <class T>
MPI_Datatype getTMPIDataType()
{ throw tmpi_exception("Data type is not supported by TMPI."); }

template <>
inline MPI_Datatype getTMPIDataType<int>()
{ return MPI_INT; }

template <>
inline MPI_Datatype getTMPIDataType<long>()
{ return MPI_LONG; }

template <>
inline MPI_Datatype getTMPIDataType<long long>()
{ return MPI_LONG_LONG; }

template <>
inline MPI_Datatype getTMPIDataType<float>()
{ return MPI_FLOAT; }

template <>
inline MPI_Datatype getTMPIDataType<double>()
{ return MPI_DOUBLE; }

// TMPI functions

void TMPI_Init(int *argc, char ***argv);

void TMPI_Finalize();

void TMPI_TestInit();

void TMPI_TestFinalize();

void TMPI_Barrier(MPI_Comm comm);

void TMPI_Wait(MPI_Request *request, MPI_Status *status);

template <class IntType>
void TMPI_Comm_rank(MPI_Comm comm, IntType *rank)
{
#ifdef TMPI_NOMPI
  *rank = 0;
#else
  int irank;
  TMPI_CHECK_ERROR(MPI_Comm_rank(comm, &irank));
  *rank = (IntType)irank;
#endif
}

int TMPI_Comm_rank(MPI_Comm comm);

template <class IntType>
void TMPI_Comm_size(MPI_Comm comm, IntType *size)
{
#ifdef TMPI_NOMPI
  *size = 1;
#else
  int isize;
  TMPI_CHECK_ERROR(MPI_Comm_size(comm, &isize));
  *size = (IntType)isize;
#endif
}

int TMPI_Comm_size(MPI_Comm comm);

template <class IntType>
void TMPI_Waitany(IntType count, MPI_Request array_of_requests[], IntType *indx, MPI_Status *status)
{
  int iindx;
#ifdef TMPI_NOMPI
  iindx = MPI_UNDEFINED;
#else
  TMPI_CHECK_ERROR(MPI_Waitany((int)count, array_of_requests, &iindx, status));
#endif
  *indx = (IntType)iindx;
}

template <class IntType>
void TMPI_Waitall(IntType count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[])
{
  TMPI_CHECK_ERROR(MPI_Waitall((int)count, array_of_requests, array_of_statuses));
}

template <class DataType, class IntType>
void TMPI_Send(DataType *buf, IntType count, IntType dest, IntType tag, MPI_Comm comm)
{
  TMPI_CHECK_ERROR(MPI_Send((void **)buf, (int)count, getTMPIDataType<DataType>(), (int)dest, (int)tag, comm));
}

template <class DataType, class IntType>
void TMPI_Recv(DataType *buf, IntType count, IntType dest, IntType tag, MPI_Comm comm, MPI_Status *status)
{
  TMPI_CHECK_ERROR(MPI_Recv((void **)buf, (int)count, getTMPIDataType<DataType>(), (int)dest, (int)tag, comm,
        status));
}

template <class DataType, class IntType>
void TMPI_Isend(DataType *buf, IntType count, IntType dest, IntType tag, MPI_Comm comm, MPI_Request *request)
{
  TMPI_CHECK_ERROR(MPI_Isend((void **)buf, (int)count, getTMPIDataType<DataType>(), (int)dest, (int)tag, comm,
        request));
}

template <class DataType, class IntType>
void TMPI_Irecv(DataType *buf, IntType count, IntType dest, IntType tag, MPI_Comm comm, MPI_Request *request)
{
  TMPI_CHECK_ERROR(MPI_Irecv((void **)buf, (int)count, getTMPIDataType<DataType>(), (int)dest, (int)tag, comm,
      request));
}

template <class DataType, class IntType>
void TMPI_Bcast(DataType *buf, IntType count, IntType root, MPI_Comm comm)
{
  TMPI_CHECK_ERROR(MPI_Bcast((void *)buf, (int)count, getTMPIDataType<DataType>(), (int)root, comm));
}

template <class DataType, class IntType>
void TMPI_Reduce(DataType *sendbuf, DataType *recvbuf, IntType count, MPI_Op op, IntType root, MPI_Comm comm)
{
#ifdef TMPI_NOMPI
  if (sendbuf != MPI_IN_PLACE) {
    memcpy(recvbuf, sendbuf, count * sizeof(DataType));
  }
#else
  TMPI_CHECK_ERROR(MPI_Reduce((void *)sendbuf, (void *)recvbuf, (int)count, getTMPIDataType<DataType>(), op,
        (int)root, comm));
#endif
}

template <class DataType, class IntType>
void TMPI_Allreduce(DataType *sendbuf, DataType *recvbuf, IntType count, MPI_Op op, MPI_Comm comm)
{
#ifdef TMPI_NOMPI
  if (sendbuf != MPI_IN_PLACE) {
    memcpy(recvbuf, sendbuf, count * sizeof(DataType));
  }
#else
  TMPI_CHECK_ERROR(MPI_Allreduce((void *)sendbuf, (void *)recvbuf, (int)count, getTMPIDataType<DataType>(), op,
        comm));
#endif
}

template <class SendDataType, class RecvDataType, class IntType>
void TMPI_Scatter(SendDataType *sendbuf, IntType sendcount, RecvDataType *recvbuf, IntType recvcount, IntType root,
    MPI_Comm comm)
{
#ifdef TMPI_NOMPI
  if (recvbuf != MPI_IN_PLACE) {
    memcpy(recvbuf, sendbuf, sendcount * sizeof(SendDataType));
  }
#else
  TMPI_CHECK_ERROR(MPI_Scatter((void *)sendbuf, (int)sendcount, getTMPIDataType<SendDataType>(), (void *)recvbuf,
        (int)recvcount, getTMPIDataType<RecvDataType>(), (int)root, comm));
#endif
}

template <class SendDataType, class RecvDataType, class IntType>
void TMPI_Scatterv(SendDataType *sendbuf, IntType *sendcounts, IntType *displs, RecvDataType *recvbuf, IntType
    recvcount, IntType root, MPI_Comm comm)
{
#ifdef TMPI_NOMPI
  if (recvbuf != MPI_IN_PLACE) {
    memcpy(recvbuf, sendbuf + displs[0], sendcounts[0] * sizeof(SendDataType));
  }
#else
  if (sizeof(IntType) != sizeof(int)) { // Integer sizes are different; need to copy to int array.
    int myRank = TMPI_Comm_rank(comm);
    int numProcs = TMPI_Comm_size(comm);
    std::vector<int> isendcounts, idispls;
    if (myRank == root) {
      isendcounts.reserve(numProcs); idispls.reserve(numProcs);
      for (int i = 0; i < numProcs; i++) {
        isendcounts.push_back(sendcounts[i]);
        idispls.push_back(displs[i]);
      }
    }
    TMPI_CHECK_ERROR(MPI_Scatterv((void *)sendbuf, &isendcounts[0], &idispls[0], getTMPIDataType<SendDataType>(),
          (void *)recvbuf, (int)recvcount, getTMPIDataType<RecvDataType>(), root, comm));
  } else { // Integer sizes are the same; use the pointer directly.
    TMPI_CHECK_ERROR(MPI_Scatterv((void *)sendbuf, (int *)sendcounts, (int *)displs, getTMPIDataType<SendDataType>(),
          (void *)recvbuf, (int)recvcount, getTMPIDataType<RecvDataType>(), root, comm));
  }
#endif
}

template <class SendDataType, class RecvDataType, class IntType>
void TMPI_Gather(SendDataType *sendbuf, IntType sendcount, RecvDataType *recvbuf, IntType recvcount, IntType root,
    MPI_Comm comm)
{
#ifdef TMPI_NOMPI
  if (sendbuf != MPI_IN_PLACE) {
    memcpy(recvbuf, sendbuf, sendcount * sizeof(SendDataType));
  }
#else
  TMPI_CHECK_ERROR(MPI_Gather((void *)sendbuf, (int)sendcount, getTMPIDataType<SendDataType>(), (void *)recvbuf,
        (int)recvcount, getTMPIDataType<RecvDataType>(), (int)root, comm));
#endif
}

template <class SendDataType, class RecvDataType, class IntType>
void TMPI_Gatherv(SendDataType *sendbuf, IntType sendcount, RecvDataType *recvbuf, IntType *recvcounts, IntType *displs,
    IntType root, MPI_Comm comm)
{
#ifdef TMPI_NOMPI
  if (sendbuf != MPI_IN_PLACE) {
    memcpy(recvbuf + displs[0], sendbuf, sendcount * sizeof(SendDataType));
  }
#else
  if (sizeof(IntType) != sizeof(int)) { // Integer sizes are different; need to copy to int array.
    int myRank = TMPI_Comm_rank(comm);
    int numProcs = TMPI_Comm_size(comm);
    std::vector<int> irecvcounts, idispls;
    if (myRank == root) {
      irecvcounts.reserve(numProcs); idispls.reserve(numProcs);
      for (int i = 0; i < numProcs; i++) {
        irecvcounts.push_back(recvcounts[i]);
        idispls.push_back(displs[i]);
      }
    }
    TMPI_CHECK_ERROR(MPI_Gatherv((void *)sendbuf, (int)sendcount, getTMPIDataType<SendDataType>(), (void *)recvbuf,
          &irecvcounts[0], &idispls[0], getTMPIDataType<RecvDataType>(), (int)root, comm));
  } else { // Integer sizes are the same; use the pointer directly.
    TMPI_CHECK_ERROR(MPI_Gatherv((void *)sendbuf, (int)sendcount, getTMPIDataType<SendDataType>(), (void *)recvbuf,
          (int *)recvcounts, (int *)displs, getTMPIDataType<RecvDataType>(), root, comm));
  }
#endif
}

template <class SendDataType, class RecvDataType, class IntType>
void TMPI_Allgather(SendDataType *sendbuf, IntType sendcount, RecvDataType *recvbuf, IntType recvcount, MPI_Comm comm)
{
#ifdef TMPI_NOMPI
  if (sendbuf != MPI_IN_PLACE) {
    memcpy(recvbuf, sendbuf, sendcount * sizeof(SendDataType));
  }
#else
  TMPI_CHECK_ERROR(MPI_Allgather((void *)sendbuf, (int)sendcount, getTMPIDataType<SendDataType>(), (void *)recvbuf,
        (int)recvcount, getTMPIDataType<RecvDataType>(), comm));
#endif
}

template <class SendDataType, class RecvDataType, class IntType>
void TMPI_Allgatherv(SendDataType *sendbuf, IntType sendcount, RecvDataType *recvbuf, IntType *recvcounts, IntType
    *displs, MPI_Comm comm)
{
#ifdef TMPI_NOMPI
  if (sendbuf != MPI_IN_PLACE) {
    memcpy(recvbuf + displs[0], sendbuf, sendcount * sizeof(SendDataType));
  }
#else
  if (sizeof(IntType) != sizeof(int)) { // Integer sizes are different; need to copy to int array.
    int numProcs = TMPI_Comm_size(comm);
    std::vector<int> irecvcounts, idispls;
    irecvcounts.reserve(numProcs); idispls.reserve(numProcs);
    for (int i = 0; i < numProcs; i++) {
      irecvcounts.push_back(recvcounts[i]);
      idispls.push_back(displs[i]);
    }
    TMPI_CHECK_ERROR(MPI_Allgatherv((void *)sendbuf, (int)sendcount, getTMPIDataType<SendDataType>(), (void *)recvbuf,
          &irecvcounts[0], &idispls[0], getTMPIDataType<RecvDataType>(), comm));
  }
  else {
    TMPI_CHECK_ERROR(MPI_Allgatherv((void *)sendbuf, (int)sendcount, getTMPIDataType<SendDataType>(), (void *)recvbuf,
          (int *)recvcounts, (int *)displs, getTMPIDataType<RecvDataType>(), comm));
  }
#endif
}

template <class SendDataType, class RecvDataType, class IntType>
void TMPI_Alltoall(SendDataType *sendbuf, IntType sendcount, RecvDataType *recvbuf, IntType recvcount, MPI_Comm comm)
{
#ifdef TMPI_NOMPI
  memcpy(recvbuf, sendbuf, sendcount * sizeof(SendDataType));
#else
  TMPI_CHECK_ERROR(MPI_Alltoall((void *)sendbuf, (int)sendcount, getTMPIDataType<SendDataType>(), (void *)recvbuf,
        (int)recvcount, getTMPIDataType<RecvDataType>(), comm));
#endif
}

template <class SendDataType, class RecvDataType, class IntType>
void TMPI_Alltoallv(SendDataType *sendbuf, IntType *sendcounts, IntType *sdispls, RecvDataType *recvbuf, IntType
    *recvcounts, IntType *rdispls, MPI_Comm comm)
{
#ifdef TMPI_NOMPI
  memcpy(recvbuf + rdispls[0], sendbuf + sdispls[0], sendcounts[0] * sizeof(SendDataType));
#else
  if (sizeof(IntType) != sizeof(int)) { // Integer sizes are different; need to copy to int array.
    int numProcs = TMPI_Comm_size(comm);
    std::vector<int> isendcounts, isdispls;
    std::vector<int> irecvcounts, irdispls;
    isendcounts.reserve(numProcs); isdispls.reserve(numProcs);
    irecvcounts.reserve(numProcs); irdispls.reserve(numProcs);
    for (int i = 0; i < numProcs; i++) {
      isendcounts.push_back(sendcounts[i]);
      isdispls.push_back(sdispls[i]);
      irecvcounts.push_back(recvcounts[i]);
      irdispls.push_back(rdispls[i]);
    }
    TMPI_CHECK_ERROR(MPI_Alltoallv((void *)sendbuf, &isendcounts[0], &isdispls[0], getTMPIDataType<SendDataType>(),
          (void *)recvbuf, &irecvcounts[0], &irdispls[0], getTMPIDataType<RecvDataType>(), comm));
  } else {
    TMPI_CHECK_ERROR(MPI_Alltoallv((void *)sendbuf, (int *)sendcounts, (int *)sdispls, getTMPIDataType<SendDataType>(),
          (void *)recvbuf, (int *)recvcounts, (int *)rdispls, getTMPIDataType<RecvDataType>(), comm));
  }
#endif
}

#endif
