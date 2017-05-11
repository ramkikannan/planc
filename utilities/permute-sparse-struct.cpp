#include <cstdio>
#include <cstdlib>
#include <vector>
#include <ctime>

int main(int argc, char **argv)
{
  printf("permute-sparse-struct began.\n");
  if (argc < 3) {
    printf("Usage: permute-sparse-struct [sparse-struct-file-name] [permuted-file-name]\n");
    return 0;
  }

  FILE *file = fopen(argv[1], "r");
  if (file == NULL) { 
    printf("Unable to open file %s.\n", argv[1]);
    return 0;
  }
  int order, nnz;
  printf("Reading the sparse struct...\n");
  fscanf(file, " %d %d", &order, &nnz);
  std::vector<int> dimSize(order);
  std::vector< std::vector<int> > idx(order);
  std::vector<double> val;
  for (int i = 0; i < order; i++) { fscanf(file, " %d", &dimSize[i]); }
  for (int i = 0; i < order; i++) { idx[i].resize(nnz); }
  val.resize(nnz);
  for (int i = 0; i < nnz; i++) {
    for (int j = 0; j < order; j++) { 
      fscanf(file, " %d", &idx[j][i]); idx[j][i]--; // Convert to 0-based
    }
    fscanf(file, " %lf", &val[i]);
  }
  fclose(file);
  std::vector< std::vector<int> > perm(order);
  printf("Permuting the sparse struct...\n");
  srand((int)time(0));
  for (int i = 0; i < order; i++) {
    perm[i].resize(dimSize[i]);
    for (int j = 0; j < dimSize[i]; j++) { perm[i][j] = j; }
    for (int j = 0; j < dimSize[i]; j++) { std::swap(perm[i][j], perm[i][rand() % dimSize[i]]); }
  }
  printf("Writing the permuted sparse struct...\n");
  file = fopen(argv[2], "w");
  if (file == NULL) { 
    printf("Unable to open file %s.\n", argv[2]);
    return 0;
  }
  fprintf(file, "%d %d\n", order, nnz);
  for (int i = 0; i < order; i++) { fprintf(file, "%d ", dimSize[i]); }
  for (int i = 0; i < nnz; i++) {
    fprintf(file, "\n");
    for (int j = 0; j < order; j++) { fprintf(file, "%d ", perm[j][idx[j][i]] + 1); }
    fprintf(file, "%e", val[i]);
  }
  fclose(file);

  printf("permute-sparse-struct finished.\n");
  return 0;
}
