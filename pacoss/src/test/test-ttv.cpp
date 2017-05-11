#include "pacoss.h"
#include "tmpi.h"

char * sparseStructFileName;
char * dimPartFileName;
char dimPartFileNameDefault[1000];

void printUsage()
{
}

void processArguments(int argc, char **argv)
{
  for (Pacoss_Int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-f") == 0) {
      sparseStructFileName = argv[i + 1];
      tmprintf("sparseStructFileName: %s\n", sparseStructFileName);
      i++;
    } else if (strcmp(argv[i], "-d") == 0) {
      dimPartFileName = argv[i + 1];
      i++; 
    } else {
      tmprintf("WARNING: Ignoring argument: %s\n", argv[i]);
    }
  }
  if (dimPartFileName == NULL) {
    strcpy(dimPartFileNameDefault, sparseStructFileName);
    strcat(dimPartFileNameDefault, PACOSS_PARTITIONER_DIM_PARTITION_SUFFIX);
    dimPartFileName = dimPartFileNameDefault;
  }
}

int main(int argc, char **argv)
{
  TMPI_Init(&argc, &argv);
  tmprintf("Pacoss_TestTtv began.\n");

  processArguments(argc, argv);

  Pacoss_SparseStruct<double> ss;
  std::string ssFileName(sparseStructFileName); ssFileName += ".part" + std::to_string(TMPI_ProcRank);
  ss.load(ssFileName.c_str());
//  if (TMPI_ProcRank == 0) { ss.print(); }

  std::vector<std::vector<Pacoss_IntPair>> dimPart;
  std::string dimPartFileNameStr(dimPartFileName); dimPartFileNameStr += ".part" + std::to_string(TMPI_ProcRank);
  Pacoss_Communicator<double>::loadDistributedDimPart(dimPartFileNameStr.c_str(), dimPart);

  std::vector<Pacoss_Communicator<double>> comms;
  std::vector<std::vector<double>> vecs;
  comms.reserve(ss._order);
  for (Pacoss_Int dim = 0; dim < ss._order; dim++) {
    comms.emplace_back(MPI_COMM_WORLD, ss._idx[dim], dimPart[dim]);
    vecs.emplace_back(comms[dim].localRowCount(), (dim + 1));
  }

  // Initialize vectors, perform TTV at each dimension
  for (Pacoss_Int iter = 0; iter < 2; iter++) {
    for (Pacoss_Int dim = 0; dim < ss._order; dim++) {
      std::fill(vecs[dim].begin(), vecs[dim].end(), 0.0);
      for (Pacoss_Int i = 0; i < ss._nnz; i++) {
        double mult = ss._val[i];
        for (Pacoss_Int j = 0; j < ss._order; j++) {
          if (j == dim) { continue; }
          mult *= vecs[j][ss._idx[j][i]];
        }
        vecs[dim][ss._idx[dim][i]] += mult;
      }
      comms[dim].foldCommBegin(&vecs[dim][0], 1);
      comms[dim].foldCommFinish(&vecs[dim][0], 1);
      comms[dim].expCommBegin(&vecs[dim][0], 1);
      comms[dim].expCommFinish(&vecs[dim][0], 1);

      double norm = 0.0;
      for (Pacoss_Int i = 0; i < comms[dim]._foldSendBegin[1]; i++) {
        norm += vecs[dim][i];
      }
      TMPI_Allreduce((double *)MPI_IN_PLACE, &norm, 1, MPI_SUM, MPI_COMM_WORLD);
      tmprintf("norm: %lf\n", norm);
      std::vector<double> vec(ss._dimSize[dim]);
      comms[dim].gatherData(&vecs[dim][0], &vec[0], 1, 0); 
      comms[dim].scatterData(&vecs[dim][0], &vec[0], 1, 0); 
      comms[dim].gatherData(&vecs[dim][0], &vec[0], 1, 0); 
      if (comms[dim]._procRank == 0) {
        double norm = 0.0;
        for (Pacoss_Int i = 0, ie = (Pacoss_Int)vec.size(); i < ie; i++) 
          norm += vec[i];
        tmprintf("gathered norm: %lf\n", norm);
      }
      for (Pacoss_Int i = 0; i < comms[dim].localRowCount(); i++) {
        vecs[dim][i] /= norm;
      }
    }
  }

  TMPI_Barrier(MPI_COMM_WORLD);
  tmprintf("Pacoss_TestTtv ended.\n");
  TMPI_Finalize();
  return 0;
}
