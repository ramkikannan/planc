#ifndef PACOSS_COMMON_H
#define PACOSS_COMMON_H

// User libraries
#include "pacoss-cpp-libs.h"
#include "cpp-utils.h"

// The sizes of default integer and index types used in the library.
#define PACOSS_INTWIDTH 32
#define PACOSS_IDXWIDTH 64

#if PACOSS_INTWIDTH == 32
#define Pacoss_Int int32_t
#define PACOSS_INT_MAX INT32_MAX
#define PACOSS_INT_MIN INT32_MIN
#define PRIINT PRId32
#define SCNINT SCNd32
#else
#define Pacoss_Int int64_t
#define PACOSS_INT_MAX INT64_MAX
#define PACOSS_INT_MIN INT64_MIN
#define PRIINT PRId64
#define SCNINT SCNd64
#endif

#if PACOSS_IDXWIDTH == 32
#define Pacoss_Idx int32_t
#define PACOSS_IDX_MAX INT32_MAX
#define PACOSS_IDX_MIN INT32_MIN
#define PRIIDX PRId32
#define SCNIDX SCNd32
#else
#define Pacoss_Idx int64_t
#define PACOSS_IDX_MAX INT64_MAX
#define PACOSS_IDX_MIN INT64_MIN
#define PRIIDX PRId64
#define SCNIDX SCNd64
#endif

typedef std::pair<Pacoss_Int, Pacoss_Int> Pacoss_IntPair;
typedef std::unordered_map<Pacoss_Int, Pacoss_Int> Pacoss_IntMap;
typedef std::vector<Pacoss_Int> Pacoss_IntVector;
typedef std::unique_ptr<Pacoss_Int []> Pacoss_IntArray;

#define Pacoss_Error(msg) _Pacoss_Error(msg, __LINE__, __FILE__)

class _Pacoss_Error : public std::exception
{
  private:
    std::string _what;

  public:
    _Pacoss_Error(const char * const msg) { _what = std::string(msg); }
    _Pacoss_Error(const char * const msg, int sourceLine, const char * const sourceFile) { 
      _what = "\n";
#ifdef TMPI_H // Append the process rank if TMPI is enabled.
      if (TMPI_NumProcs > 1) { _what += "rank" + std::to_string(TMPI_ProcRank) + ":"; }
#endif
      _what += std::string(sourceFile) + std::string(":");
      _what += std::to_string(sourceLine) + std::string(": ");
      _what += std::string(msg);
    }
    _Pacoss_Error() { _what = std::string("Uninitialized exception."); }
    ~_Pacoss_Error() noexcept { }
    virtual const char* what() const throw() {
      return _what.c_str();
    }
};

#define Pacoss_AssertGe(a, b) { \
  if ((a) < (b)) {  \
    std::string msg; \
    msg = "The assertion " + std::to_string(a) + " >= " + std::to_string(b) + " has failed.\n"; \
    throw Pacoss_Error(msg.c_str()); \
  } \
}

#define Pacoss_AssertGt(a, b) { \
  if ((a) <= (b)) {  \
    std::string msg; \
    msg = "The assertion " + std::to_string(a) + " > " + std::to_string(b) + " has failed.\n"; \
    throw Pacoss_Error(msg.c_str()); \
  } \
}

#define Pacoss_AssertLe(a, b) { \
  if ((a) > (b)) {  \
    std::string msg; \
    msg = "The assertion " + std::to_string(a) + " <= " + std::to_string(b) + " has failed.\n"; \
    throw Pacoss_Error(msg.c_str()); \
  } \
}

#define Pacoss_AssertLt(a, b) { \
  if ((a) >= (b)) {  \
    std::string msg; \
    msg = "The assertion " + std::to_string(a) + " < " + std::to_string(b) + " has failed.\n"; \
    throw Pacoss_Error(msg.c_str()); \
  } \
}

#define Pacoss_AssertEq(a, b) { \
  if ((a) != (b)) {  \
    std::string msg; \
    msg = "The assertion " + std::to_string(a) + " == " + std::to_string(b) + " has failed.\n"; \
    throw Pacoss_Error(msg.c_str()); \
  } \
}

#define Pacoss_AssertNe(a, b) { \
  if ((a) == (b)) {  \
    std::string msg; \
    msg = "The assertion " + std::to_string(a) + " != " + std::to_string(b) + " has failed.\n"; \
    throw Pacoss_Error(msg.c_str()); \
  } \
}

#endif
