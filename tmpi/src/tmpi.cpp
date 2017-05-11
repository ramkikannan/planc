#include "tmpi.h"

int TMPI_ProcRank;
int TMPI_NumProcs;
FILE *TMPI_OutFile;
FILE *TMPI_TestFile;
char TMPI_OutFileName[50];
char TMPI_TestFileName[50];

void TMPI_Init(int *argc, char ***argv)
{
#ifdef TMPI_NOMPI
  TMPI_ProcRank = 0;
  TMPI_NumProcs = 1;
#else
  TMPI_CHECK_ERROR(MPI_Init(argc, argv));
  TMPI_CHECK_ERROR(MPI_Comm_rank(MPI_COMM_WORLD, &TMPI_ProcRank));
  TMPI_CHECK_ERROR(MPI_Comm_size(MPI_COMM_WORLD, &TMPI_NumProcs));
#endif
  sprintf(TMPI_OutFileName, "mpiout-%d.out", TMPI_ProcRank);
  if ((TMPI_OutFile = fopen(TMPI_OutFileName, "w")) == NULL) {
    char msg[1000];
    sprintf(msg, "ERROR: Unable to open the file %s.\n", TMPI_OutFileName);
    throw tmpi_exception(msg);
  }
}

void TMPI_Finalize()
{
  TMPI_CHECK_ERROR(MPI_Finalize());
  if (fclose(TMPI_OutFile) != 0) {
    char msg[1000];
    sprintf(msg, "ERROR: Unable to close the file %s.\n", TMPI_OutFileName);
    throw tmpi_exception(msg);
  }
}

void TMPI_TestInit()
{
  sprintf(TMPI_TestFileName, "mpitestout-%d.out", TMPI_ProcRank);
  if ((TMPI_TestFile = fopen(TMPI_TestFileName, "w")) == NULL) {
    char msg[1000];
    sprintf(msg, "ERROR: Unable to open the file %s.\n", TMPI_TestFileName);
    throw tmpi_exception(msg);
  }
}

void TMPI_TestFinalize()
{
  if (fclose(TMPI_TestFile) != 0) {
    char msg[1000];
    sprintf(msg, "ERROR: Unable to close the file %s.\n", TMPI_TestFileName);
    throw tmpi_exception(msg);
  }
}

void TMPI_Barrier(MPI_Comm comm)
{
  TMPI_CHECK_ERROR(MPI_Barrier(comm));
}

void TMPI_Wait(MPI_Request *request, MPI_Status *status)
{
#ifdef TMPI_NOMPI
  status->MPI_ERROR = MPI_SUCCESS;
#else
  TMPI_CHECK_ERROR(MPI_Wait(request, status));
#endif
}

int TMPI_Comm_rank(MPI_Comm comm)
{
  int irank;
#ifdef TMPI_NOMPI
  irank = 0;
#else
  TMPI_CHECK_ERROR(MPI_Comm_rank(comm, &irank));
#endif
  return irank;
}

int TMPI_Comm_size(MPI_Comm comm)
{
  int isize;
#ifdef TMPI_NOMPI
  isize = 1;
#else
  TMPI_CHECK_ERROR(MPI_Comm_size(comm, &isize));
#endif
  return isize;
}
