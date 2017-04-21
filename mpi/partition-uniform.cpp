#include <cstdio>
#include <cstdlib>
#include <vector>
#include <ctime>
#include <string>
#include <iostream>

int main(int argc, char **argv)
{
  printf("partition-uniform began.\n");
  if (argc < 4) {
    printf("Usage: partition-uniform [matrix-file-name] [row-proc-count] [col-proc-count]\n");
    return 0;
  }
  int rowProcCount = atoi(argv[2]);
  int colProcCount = atoi(argv[3]);
  int procCount = rowProcCount * colProcCount;
  printf("Creating a %d x %d uniform partition.\n", rowProcCount, colProcCount);

  printf("Opening input and output files...\n");
  FILE *file = fopen(argv[1], "r");
  if (file == NULL) { 
    printf("Unable to open file %s.\n", argv[1]);
    return 0;
  }

  std::vector< std::vector<int> > procRowIdxs(procCount);
  std::vector< std::vector<int> > procColIdxs(procCount);
  std::vector< std::vector<double> > procVals(procCount);
  printf("Partitioning nonzeros and outputting to files...\n");
  int order, nnz;
  int rowCount, colCount;
  fscanf(file, " %d %d", &order, &nnz);
  fscanf(file, " %d %d", &rowCount, &colCount);
  int rowsPerProc = rowCount / rowProcCount;
  int colsPerProc = colCount / colProcCount;
  printf("Assigning %d rows and %d columns per part.\n", rowsPerProc, colsPerProc);
  for (int i = 0; i < nnz; i++) {
    if (i % 100000 == 0 && i > 0) { printf("Processing %d.th nonzero...\n", i); }
    int rowIdx, colIdx;
    double val;
    fscanf(file, " %d %d %lf", &rowIdx, &colIdx, &val); rowIdx--; colIdx--;
    int procRowIdx = rowIdx / rowsPerProc;
    int procColIdx = colIdx / colsPerProc;
    if (procRowIdx >= rowProcCount || procColIdx >= colProcCount) { continue; } // Prune matrix
    int procIdx = procRowIdx * colProcCount + procColIdx;
    int localRowIdx = rowIdx % rowsPerProc;
    int localColIdx = colIdx % colsPerProc;
    procRowIdxs[procIdx].push_back(localRowIdx);
    procColIdxs[procIdx].push_back(localColIdx);
    procVals[procIdx].push_back(val);
  }
  fclose(file);

  for (int i = 0; i < procCount; i++) { 
    printf("Writing the matrix for part %d...\n", i);
    std::string outFileName(argv[1]);
    outFileName += std::to_string(i);
    file = fopen(outFileName.c_str(), "w");
    if (file == NULL) { 
      printf("Unable to open file %s.\n", outFileName.c_str());
      return 0;
    }
    auto &curRowIdxs = procRowIdxs[i];
    auto &curColIdxs = procColIdxs[i];
    auto &curVals = procVals[i];
    for (int j = 0; j < curRowIdxs.size(); j++) {
      fprintf(file, "%d %d %e\n", curRowIdxs[j], curColIdxs[j], curVals[j]);
    }
    fclose(file);
  }

  printf("partition-uniform finished.\n");
  return 0;
}

