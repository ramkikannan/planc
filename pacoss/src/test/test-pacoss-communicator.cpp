#include "pacoss-communicator.h"

#define THROWFAIL(msg) { \
  char throwfail_str[1000]; \
  sprintfe(throwfail_str, "\n%s\nLine:%d, File:%s", msg, __LINE__, __FILE__); \
  throw Pacoss_UnitTestFail(throwfail_str); \
}

class Pacoss_UnitTestFail : public std::exception
{
  private:
    std::string _what;

  public:
    Pacoss_UnitTestFail(const char * const msg) { _what = std::string(msg); }
    Pacoss_UnitTestFail() { _what = std::string("Uninitialized exception."); }
    virtual const char* what() const throw() {
      return _what.c_str();
    }
};

class Pacoss_Communicator_Test
{
  public:
    static void testPacossComm(int argc, char **argv) {
      tmprintf("Testing Pacoss_Communicator...\n");
      if (TMPI_ProcRank == 0 && TMPI_NumProcs != 3) {
        THROWFAIL("Please run the test with 3 MPI processes.\n");
      }
      for (Pacoss_Int fileIdx = 1; fileIdx < 2; fileIdx++) {
        tmprintf("   Testing for file %" PRIINT "...", fileIdx);
        TMPI_TestInit();
        Pacoss_Int procRank;
        TMPI_Comm_rank(MPI_COMM_WORLD, &procRank);
        char fileName[100];

        // Read the idx array
        sprintfe(fileName, "dat/test/test%" PRIINT "-%" PRIINT ".idx", fileIdx, procRank);
        FILE *file = fopene(fileName, "r");
        Pacoss_Idx idxSize;
        Pacoss_Int idxMax;
        fscanfe(file, " %" PRIINT " %" PRIIDX, &idxMax, &idxSize);
        Pacoss_IntVector idx; idx.reserve(idxSize);
        for (Pacoss_Idx i = 0; i < idxSize; i++) {
          Pacoss_Int j; fscanfe(file, " %" SCNINT, &j);
          idx.push_back(j - 1);
        }
        fclosee(file);

        // Read the vecOwnerMap
        std::vector<Pacoss_IntPair> vecOwnerMap;
        sprintfe(fileName, "dat/test/test%" PRIINT "-%" PRIINT ".part", fileIdx, procRank);
        file = fopene(fileName, "r");
        while (!feof(file)) {
          Pacoss_IntPair pair;
          fscanfe(file, "%" PRIINT " %" PRIINT " ", &pair.first, &pair.second);
          vecOwnerMap.emplace_back(pair.first - 1, pair.second); // Convert vector idx to 0-based
        }
        fclosee(file);

        // Now create the sparse communicator
        Pacoss_Communicator<Pacoss_Int> comm(MPI_COMM_WORLD, idx, vecOwnerMap);

        // Output for the test case
        tmtprintf("Fold send proc count: %" PRIINT "\n", comm._foldSendProcCount);
        for (Pacoss_Int i = 0; i < comm._foldSendProcCount; i++) {
          tmtprintf("Proc %" PRIINT ": ", comm._foldSendProcMapL2G[i]);
          for (Pacoss_Int j = comm._foldSendBegin[i], je = comm._foldSendBegin[i + 1]; j < je; j++) {
            tmtprintf("%" PRIINT " ", comm._foldSendRowMapL2G[j] + 1);
          }
          tmtprintf("\n");
        }
        tmtprintf("Fold recv proc count: %" PRIINT "\n", comm._foldRecvProcCount);
        for (Pacoss_Int i = 0; i < comm._foldRecvProcCount; i++) {
          tmtprintf("Proc %" PRIINT ": ", comm._foldRecvProcMapL2G[i]);
          for (Pacoss_Int j = comm._foldRecvBegin[i], je = comm._foldRecvBegin[i + 1]; j < je; j++) {
            tmtprintf("%" PRIINT " ", comm._foldSendRowMapL2G[comm._foldRecvReduceMap[j]] + 1);
          }
          tmtprintf("\n");
        }

        // Now test a basic sparse fold/expand communication.
        // Testing P2P communication
        // Initialize the vector and perform a fold communication.
        std::vector<Pacoss_Int> vec;
        vec.resize(comm._foldSendBegin[comm._foldSendProcCount], comm._procRank + 1);
        tmtprintf("Initial vector:\n");
        for (Pacoss_Int i = 0, ie = (Pacoss_Int)vec.size(); i < ie; i++) {
          tmtprintf("%" PRIINT " %" PRIINT "\n", comm._foldSendRowMapL2G[i] + 1, vec[i]);
        }
        comm.foldCommBegin(&vec[0], (Pacoss_Int)1);
        comm.foldCommFinish(&vec[0], (Pacoss_Int)1);
        // Output the result after fold.
        tmtprintf("After fold:\n");
        for (Pacoss_Int i = 0, ie = (Pacoss_Int)vec.size(); i < ie; i++) {
          tmtprintf("%" PRIINT " %" PRIINT "\n", comm._foldSendRowMapL2G[i] + 1, vec[i]);
        }
        // Perform an expand communication to deliver the final results.
        comm.expCommBegin(&vec[0], (Pacoss_Int)1);
        comm.expCommFinish(&vec[0], (Pacoss_Int)1);
        // Output the new vector after expand.
        tmtprintf("After expand:\n");
        for (Pacoss_Int i = 0, ie = (Pacoss_Int)vec.size(); i < ie; i++) {
          tmtprintf("%" PRIINT " %" PRIINT "\n", comm._foldSendRowMapL2G[i] + 1, vec[i]);
        }

        // Testing A2A communication with unitMsgSize=1
        // Initialize the vector and perform a fold communication.
        std::fill(vec.begin(), vec.end(), comm._procRank + 1);
        tmtprintf("Initial vector:\n");
        for (Pacoss_Int i = 0, ie = (Pacoss_Int)vec.size(); i < ie; i++) {
          tmtprintf("%" PRIINT " %" PRIINT "\n", comm._foldSendRowMapL2G[i] + 1, vec[i]);
        }
        comm.foldCommA2A(&vec[0], (Pacoss_Int)1);
        // Output the result after fold.
        tmtprintf("After fold:\n");
        for (Pacoss_Int i = 0, ie = (Pacoss_Int)vec.size(); i < ie; i++) {
          tmtprintf("%" PRIINT " %" PRIINT "\n", comm._foldSendRowMapL2G[i] + 1, vec[i]);
        }
        // Perform an expand communication to deliver the final results.
        comm.expCommA2A(&vec[0], (Pacoss_Int)1);
        // Output the new vector after expand.
        tmtprintf("After expand:\n");
        for (Pacoss_Int i = 0, ie = (Pacoss_Int)vec.size(); i < ie; i++) {
          tmtprintf("%" PRIINT " %" PRIINT "\n", comm._foldSendRowMapL2G[i] + 1, vec[i]);
        }

        // Testing P2P communication with unitMsgSize=2
        // Initialize the vector and perform a fold communication.
        Pacoss_Int unitMsgSize = 2;
        vec.resize(comm._foldSendBegin[comm._foldSendProcCount] * unitMsgSize);
        std::fill(vec.begin(), vec.end(), comm._procRank + 1);
        tmtprintf("Initial vector:\n");
        for (Pacoss_Int i = 0, ie = (Pacoss_Int)vec.size(); i < ie; i++) {
          tmtprintf("%" PRIINT " %" PRIINT "\n", comm._foldSendRowMapL2G[i / unitMsgSize] + 1, vec[i]);
        }
        comm.foldCommBegin(&vec[0], unitMsgSize);
        comm.foldCommFinish(&vec[0], unitMsgSize);
        // Output the result after fold.
        tmtprintf("After fold:\n");
        for (Pacoss_Int i = 0, ie = (Pacoss_Int)vec.size(); i < ie; i++) {
          tmtprintf("%" PRIINT " %" PRIINT "\n", comm._foldSendRowMapL2G[i / unitMsgSize] + 1, vec[i]);
        }
        // Perform an expand communication to deliver the final results.
        comm.expCommBegin(&vec[0], unitMsgSize);
        comm.expCommFinish(&vec[0], unitMsgSize);
        // Output the new vector after expand.
        tmtprintf("After expand:\n");
        for (Pacoss_Int i = 0, ie = (Pacoss_Int)vec.size(); i < ie; i++) {
          tmtprintf("%" PRIINT " %" PRIINT "\n", comm._foldSendRowMapL2G[i / unitMsgSize] + 1, vec[i]);
        }

        // Testing A2A communication with unitMsgSize=3
        // Initialize the vector and perform a fold communication.
        unitMsgSize = 3;
        vec.resize(comm._foldSendBegin[comm._foldSendProcCount] * unitMsgSize);
        std::fill(vec.begin(), vec.end(), comm._procRank + 1);
        tmtprintf("Initial vector:\n");
        for (Pacoss_Int i = 0, ie = (Pacoss_Int)vec.size(); i < ie; i++) {
          tmtprintf("%" PRIINT " %" PRIINT "\n", comm._foldSendRowMapL2G[i / unitMsgSize] + 1, vec[i]);
        }
        comm.foldCommA2A(&vec[0], unitMsgSize);
        // Output the result after fold.
        tmtprintf("After fold:\n");
        for (Pacoss_Int i = 0, ie = (Pacoss_Int)vec.size(); i < ie; i++) {
          tmtprintf("%" PRIINT " %" PRIINT "\n", comm._foldSendRowMapL2G[i / unitMsgSize] + 1, vec[i]);
        }
        // Perform an expand communication to deliver the final results.
        comm.expCommA2A(&vec[0], unitMsgSize);
        // Output the new vector after expand.
        tmtprintf("After expand:\n");
        for (Pacoss_Int i = 0, ie = (Pacoss_Int)vec.size(); i < ie; i++) {
          tmtprintf("%" PRIINT " %" PRIINT "\n", comm._foldSendRowMapL2G[i / unitMsgSize] + 1, vec[i]);
        }

        // Testing gather and scatter of data at the root process
        unitMsgSize = 3;
        vec.resize(comm._foldSendBegin[comm._foldSendProcCount] * unitMsgSize);
        std::fill(vec.begin(), vec.end(), comm._procRank + 1);
        Pacoss_IntVector fullVec;
        if (comm._procRank == 0) { fullVec.resize(idxMax * unitMsgSize); }
        comm.gatherData(&vec[0], &fullVec[0], unitMsgSize, 0);
        std::fill(vec.begin(), vec.end(), 0);
        comm.scatterData(&vec[0], &fullVec[0], unitMsgSize, 0);
        comm.expCommA2A(&vec[0], unitMsgSize);
        tmtprintf("After gather scatter expand:\n");
        for (Pacoss_Int i = 0, ie = (Pacoss_Int)vec.size(); i < ie; i++) {
          tmtprintf("%" PRIINT " %" PRIINT "\n", comm._foldSendRowMapL2G[i / unitMsgSize] + 1, vec[i]);
        }

        TMPI_TestFinalize();

        // Compare the output with the correct result
        sprintfe(fileName, "dat/test/test%" PRIINT "-%" PRIINT ".out", fileIdx, procRank);
        FILE *testRes = fopene(fileName, "r"); // Correct result of this test case
        FILE *outFile = fopene(TMPI_TestFileName, "r"); // Output of this execution
        char line1[1000], line2[1000];
        for (Pacoss_Int i = 1; !feof(testRes) && !feof(outFile); i++) { // Compare outFile and testRes line-by-line
          fgets(line1, 1000, testRes);
          fgets(line2, 1000, outFile);
          if (strcmp(line1, line2) != 0) { 
            char msg[1000];
            sprintfe(msg, "Unit test failed!: Line %" PRIINT " of the output is"
                " incorrect.\nExpected:\n%s\nOutput:\n%s\n", i, line1, line2);
            THROWFAIL(msg);
          }
        }
        if (!feof(testRes)) {
          char msg[1000];
          sprintfe(msg, "Unit test failed!: Test output file is bigger.\n");
          THROWFAIL(msg);
        }
        if (!feof(outFile)) {
          char msg[1000];
          sprintfe(msg, "Unit test failed!: Expected output file is bigger.\n");
          THROWFAIL(msg);
        }
        fclosee(testRes);
        fclosee(outFile);
        tmprintf("SUCCESS!\n");
      }
    }
};

int main(int argc, char **argv)
{
  TMPI_Init(&argc, &argv);

  Pacoss_Communicator_Test::testPacossComm(argc, argv);

  TMPI_Finalize();
  return 0;
}
