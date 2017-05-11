#include "cpp-utils.h"

char STRCAT_STR[1000];

std::chrono::time_point<std::chrono::high_resolution_clock> tic_global;

FILE *_fopene(
    const char * const fileName,
    const char * const mode,
    const int sourceLineNo,
    const char * const sourceFileName)
{ 
  FILE *file = fopen(fileName, mode);
  if (file == NULL) { 
    char msg[1000];
    sprintf(msg, "Unable to open the file %s.", fileName);
    throw _CppUtils_Error(msg, sourceLineNo, sourceFileName);
  }
  return file;
}

void _fclosee(
    FILE *file,
    const int sourceLineNo,
    const char * const sourceFileName)
{ 
  if (file != NULL) { 
    int ret = fclose(file);
    if (ret != 0) { // File is not closed successfully.
      char msg[1000];
      sprintf(msg, "The file was not closed successfully.");
      throw _CppUtils_Error(msg, sourceLineNo, sourceFileName);
    }
  } else {
    char msg[1000];
    sprintf(msg, "File pointer in fclosee is NULL.");
    throw _CppUtils_Error(msg, sourceLineNo, sourceFileName);
  }
}

std::chrono::time_point<std::chrono::high_resolution_clock> timerTic()
{
  tic_global = std::chrono::high_resolution_clock::now();
  return std::chrono::high_resolution_clock::now();
}

double timerToc(const std::chrono::time_point<std::chrono::high_resolution_clock> &ticPoint)
{
  return (double)(std::chrono::high_resolution_clock::now() - ticPoint).count() / 1000000000.0;
}

const char * STRCAT(const char * t)
// recursive variadic function
{
  strcpy(STRCAT_STR, t);
  return STRCAT_STR;
}
