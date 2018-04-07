#ifndef CPP_UTILS_H
#define CPP_UTILS_H

#include <cstdio>
#include <string>
#include <cstring>
#include <exception>
#include <chrono>
#include <vector>
#include <iostream>

#define CppUtils_Error(msg) _CppUtils_Error(msg, __LINE__, __FILE__)

class _CppUtils_Error : public std::exception
{
  private:
    std::string _what;

  public:
    _CppUtils_Error(const char * const msg) { _what = std::string(msg); }
    _CppUtils_Error(const char * const msg, const int sourceLineNo, const char * const sourceFileName) { 
      _what = "\n";
      _what += std::string(sourceFileName) + std::string(":");
      _what += std::to_string(sourceLineNo) + std::string(": ");
      _what += std::string(msg);
    }
    _CppUtils_Error() { _what = std::string("Uninitialized exception."); }
    ~_CppUtils_Error() noexcept { }
    virtual const char* what() const throw() {
      return _what.c_str();
    }
};

// Error checking versions of some standard C library functions
#define printfe(...) \
{ \
  int err = printf(__VA_ARGS__); \
  if (err < 0) { \
    char msg[1000]; \
    sprintf(msg, "printf() failed with the error code %d.", err); \
    throw CppUtils_Error(msg); \
  } \
}

#define scanfe(...) \
{ \
  int err = scanf(__VA_ARGS__); \
  if (err == EOF || err < 0) { \
    char msg[1000]; \
    sprintf(msg, "scanf() failed with the error code %d.", err); \
    throw CppUtils_Error(msg); \
  } \
}

#define fprintfe(...) \
{ \
  int err = fprintf(__VA_ARGS__); \
  if (err < 0) { \
    char msg[1000]; \
    sprintf(msg, "fprintf() failed with the error code %d.", err); \
    throw CppUtils_Error(msg); \
  } \
}

#define fscanfe(...) \
{ \
  int err = fscanf(__VA_ARGS__); \
  if (err == EOF || err < 0) { \
    char msg[1000]; \
    sprintf(msg, "fscanf() failed with the error code %d.", err); \
    throw CppUtils_Error(msg); \
  } \
}

#define sprintfe(...) \
{ \
  int err = sprintf(__VA_ARGS__); \
  if (err < 0) { \
    char msg[1000]; \
    sprintf(msg, "sprintf() failed with the error code %d.", err); \
    throw CppUtils_Error(msg); \
  } \
}

#define sscanfe(...) \
{ \
  int err = sscanf(__VA_ARGS__); \
  if (err == EOF || err < 0) { \
    char msg[1000]; \
    sprintf(msg, "sscanf() failed with the error code %d.", err); \
    throw CppUtils_Error(msg); \
  } \
}

#define fopene(fileName,mode) _fopene(fileName, mode, __LINE__, __FILE__)
FILE *_fopene(
    const char * const fileName,
    const char * const mode,
    const int sourceLineNo,
    const char * const sourceFileName);

#define fclosee(fileName) _fclosee(fileName, __LINE__, __FILE__)
void _fclosee(
    FILE *file,
    const int sourceLineNo,
    const char * const sourceFileName);

template <class T>
inline const char * PRIMOD()
{ throw CppUtils_Error("Data type is not supported by PRIMOD()."); }

template <>
inline const char * PRIMOD<long double>() { return "%Lf"; }

template <>
inline const char * PRIMOD<double>() { return "%lf"; }

template <>
inline const char * PRIMOD<float>() { return "%f"; }

template <>
inline const char * PRIMOD<int>() { return "%d"; }

template <>
inline const char * PRIMOD<long>() { return "%ld"; }

template <>
inline const char * PRIMOD<long long>() { return "%Ld"; }

template <class T>
inline const char * SCNMOD()
{ throw CppUtils_Error("Data type is not supported by SCNMOD()."); }

template <>
inline const char * SCNMOD<long double>() { return " %Lf"; }

template <>
inline const char * SCNMOD<double>() { return " %lf"; }

template <>
inline const char * SCNMOD<float>() { return " %f"; }

template <>
inline const char * SCNMOD<int>() { return " %d"; }

template <>
inline const char * SCNMOD<long>() { return " %ld"; }

template <>
inline const char * SCNMOD<long long>() { return " %Ld"; }

const char * STRCAT(const char * t);

extern char STRCAT_STR[1000];

template<typename T, typename... Args>
const char * STRCAT(T t, Args... args) // recursive variadic function
{
  char str[1000];
  strcpy(STRCAT_STR, "");
  STRCAT(args...);
  strcpy(str, STRCAT_STR);
  strcpy(STRCAT_STR, t);
  strcat(STRCAT_STR, str);
  return STRCAT_STR;
}

template<class T>
static void printVector(std::vector<T> vec) {
  for (auto &i : vec) { std::cout << i << " "; }
  std::cout << std::endl;
}

template<class T>
static void printlnVector(std::vector<T> vec) {
  for (auto &i : vec) { std::cout << i << " "; }
  std::cout << std::endl;
}

// Timer tic-toc routines
extern std::chrono::time_point<std::chrono::high_resolution_clock> tic_global;

std::chrono::time_point<std::chrono::high_resolution_clock> timerTic();

double timerToc(const std::chrono::time_point<std::chrono::high_resolution_clock> &ticPoint = tic_global);

#endif
