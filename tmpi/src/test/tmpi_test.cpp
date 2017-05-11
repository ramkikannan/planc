#include "tmpi.h"
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>

int main(int argc, char **argv) {
  TMPI_Init(&argc, &argv); 

  int numProc, rank;
  TMPI_Comm_size(MPI_COMM_WORLD, &numProc);
  TMPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    trprintf("tmpi_test began.\n");
  }
  TMPI_Barrier(MPI_COMM_WORLD);

  double data = (double)rank;
  double recvData;
  MPI_Status status;
  MPI_Request request;

  if (numProc % 2 == 0) {
    trprintf("Testing TMPI_Send/Recv...");
    if (rank < numProc / 2) {
      TMPI_Send(&data, 1, numProc - rank - 1, rank, MPI_COMM_WORLD);
      TMPI_Recv(&recvData, 1, numProc - rank - 1, numProc - rank - 1, MPI_COMM_WORLD, &status);
    } else {
      TMPI_Recv(&recvData, 1, numProc - rank - 1, numProc - rank - 1, MPI_COMM_WORLD, &status);
      TMPI_Send(&data, 1, numProc - rank - 1, rank, MPI_COMM_WORLD);
    }
    assert(recvData == (double)numProc - rank - 1);
    TMPI_Barrier(MPI_COMM_WORLD);
    trprintf("SUCCESS!\n");

    trprintf("Testing TMPI_Isend/Irecv...");
    if (rank < numProc / 2) {
      TMPI_Irecv(&recvData, 1, numProc - rank - 1, numProc - rank - 1, MPI_COMM_WORLD, &request);
      TMPI_Send(&data, 1, numProc - rank - 1, rank, MPI_COMM_WORLD);
      TMPI_Wait(&request, &status);
    } else {
      TMPI_Isend(&data, 1, numProc - rank - 1, rank, MPI_COMM_WORLD, &request);
      TMPI_Recv(&recvData, 1, numProc - rank - 1, numProc - rank - 1, MPI_COMM_WORLD, &status);
      TMPI_Wait(&request, &status);
    }
    assert(recvData == (double)numProc - rank - 1);
    TMPI_Barrier(MPI_COMM_WORLD);
    trprintf("SUCCESS!\n");
  } else {
    trprintf("WARNING: The number of processes is not even; skipping tests for TMPI_Send/Recv/Isend/Irecv.\n");
  }

  if (rank == 0) {
    trprintf("Testing TMPI_Bcast...");
    data = 1.2345;
  }
  TMPI_Bcast(&data, 1, 0, MPI_COMM_WORLD);
  assert(data == 1.2345);
  TMPI_Barrier(MPI_COMM_WORLD);
  trprintf("SUCCESS!\n");

  trprintf("Testing TMPI_Reduce...");
  data = rank;
  TMPI_Reduce(&data, &recvData, 1, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    assert(recvData == numProc * (numProc - 1) / 2);
  }
  TMPI_Barrier(MPI_COMM_WORLD);
  trprintf("SUCCESS!\n");

  trprintf("Testing TMPI_Allreduce...");
  std::vector<double> arr(numProc * numProc);
  std::fill(arr.begin(), arr.end(), rank);
  std::vector<double> arr2(numProc * numProc);
  for (int i = 0; i < numProc; i++) { arr[i] = (double)rank; }
  TMPI_Allreduce(&arr[0], &arr2[0], numProc, MPI_SUM, MPI_COMM_WORLD);
  for (int i = 0; i < numProc; i++) { assert(arr2[i] == (numProc * (numProc - 1) / 2)); }
  trprintf("SUCCESS!\n");

  trprintf("Testing TMPI_Scatter...");
  for (int i = 0; i < numProc; i++) { arr[i] = (double)i; }
  TMPI_Scatter(&arr[0], 1, &data, 1, 0, MPI_COMM_WORLD);
  assert(data == (double)rank);
  trprintf("SUCCESS!\n");

  trprintf("Testing TMPI_Scatterv...");
  std::vector<int> iarr(numProc);
  std::vector<int> iarr2(numProc);
  iarr[0] = iarr2[0] = 0;
  for (int i = 1; i < numProc; i++) {
    iarr[i] = i;
    iarr2[i] = iarr2[i - 1] + (i - 1); 
    for (int j = 0; j < i; j++) arr[iarr2[i] + j] = (double)i;
  }
  TMPI_Scatterv(&arr[0], &iarr[0], &iarr2[0], &arr2[0], rank, 0, MPI_COMM_WORLD);
  // Proc rank i should get the double value i, i times
  for (int i = 0; i < rank; i++) { assert(arr2[i] == (double)rank); }
  trprintf("SUCCESS!\n");

  trprintf("Testing TMPI_Gather...");
  data = (double)rank;
  TMPI_Gather(&data, 1, &arr2[0], 1, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    for (int i = 0; i < numProc; i++) { assert(arr2[i] == (double)i); }
    trprintf("SUCCESS!\n");
  }

  trprintf("Testing TMPI_Gatherv...");
  for (int i = 0; i < rank; i++) { arr[i] = (double)rank; }
  if (rank == 0) {
    iarr[0] = iarr2[0] = 0;
    for(int i = 1; i < numProc; i++) { 
      iarr[i] = i;
      iarr2[i] = iarr2[i - 1] + (i - 1);
    }
  }
  TMPI_Gatherv(&arr[0], rank, &arr2[0], &iarr[0], &iarr2[0], 0, MPI_COMM_WORLD);
  if (rank == 0) {
    double *ptr = &arr2[0];
    for (int i = 0; i < numProc; i++) {
      for (int j = 0; j < i; j++) {
        assert(*ptr == i);
        ptr++;
      }
    }
    trprintf("SUCCESS!\n");
  }

  trprintf("Testing TMPI_Allgather...");
  for (int i = 0; i < numProc; i++) { arr[i] = (double)-1.0; }
  arr[rank] = rank;
  TMPI_Allgather((double *)MPI_IN_PLACE, 1, &arr[0], 1, MPI_COMM_WORLD);
  for (int i = 0; i < numProc; i++) { assert(arr[i] == i); }
  trprintf("SUCCESS!\n");

  trprintf("Testing TMPI_Allgatherv...");
  int begIdx = rank * (rank - 1) / 2;
  for (int i = 0; i < rank; i++) { arr[begIdx + i] = rank; }
  iarr[0] = 0;
  iarr2[0] = 0;
  for (int i = 1; i < numProc; i++) {
    iarr[i] = i; // receive counts; process i sends i messages.
    iarr2[i] = iarr2[i - 1] + iarr[i - 1]; // prefix sum of iarr; used for displs.
  }
  TMPI_Allgatherv((double *)MPI_IN_PLACE, rank, &arr[0], &iarr[0], &iarr2[0], MPI_COMM_WORLD);
  double *ptr = &arr[0];
  for (int i = 0; i < numProc; i++) {
    for (int j = 0; j < i; j++) {
      assert(*ptr == i);
      ptr++;
    }
  }
  trprintf("SUCCESS!\n");

  trprintf("Testing TMPI_Alltoall...");
  for (int i = 0; i < numProc; i++) { arr[i] = (double)rank; arr2[i] = -1.0; }
  TMPI_Alltoall(&arr[0], 1, &arr2[0], 1, MPI_COMM_WORLD);
  for (int i = 0; i < numProc; i++) { assert(arr2[i] == i); }
  trprintf("SUCCESS!\n");

  trprintf("Testing TMPI_Alltoallv...");
  iarr[0] = iarr2[0] = 0;
  for (int i = 1; i < numProc; i++) {
    iarr[i] = i;
    iarr2[i] = iarr2[i - 1] + iarr[i - 1];
    for (int j = 0; j < i; j++) arr[iarr2[i] + j] = (double)rank;
  }
  int *iarr3 = (int *)malloc(numProc * sizeof(iarr3[0]));
  int *iarr4 = (int *)malloc(numProc * sizeof(iarr4[0]));
  iarr3[0] = rank;
  iarr4[0] = 0;
  for (int i = 1; i < numProc; i++) { 
    iarr3[i] = rank;
    iarr4[i] = iarr4[i - 1] + iarr3[i - 1];
  }
  TMPI_Alltoallv(&arr[0], &iarr[0], &iarr2[0], &arr2[0], &iarr3[0], &iarr4[0], MPI_COMM_WORLD);
  ptr = &arr2[0];
  for (int i = 0; i < numProc; i++) {
    for (int j = 0; j < rank; j++) { 
      assert(*ptr == (double)i);
      ptr++;
    }
  }
  trprintf("SUCCESS!\n");

  trprintf("tmpi_test ended.\n");
  TMPI_Finalize(); 

  return 0;
}
