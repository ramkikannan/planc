# Copyright 2016 Ramakrishnan Kannan
# Find Armadillo, BLAS and LAPACK. 

find_path(ARMADILLO_INCLUDE_DIR
  NAMES armadillo
  PATHS "$ENV{ProgramFiles}/Armadillo/include"
  )

message(STATUS "     ARMA_FOUND = ${ARMADILLO_INCLUDE_DIR}")

set(NMFLIB_USE_LAPACK           false)
set(NMFLIB_USE_BLAS             false)
set(NMFLIB_USE_ATLAS            false)

OPTION(CMAKE_IGNORE_MKL "Build Ignoring MKL" ON)

#to build sparse comment or uncomment this line.
OPTION(CMAKE_BUILD_SPARSE "Build Sparse" OFF)
if(CMAKE_BUILD_SPARSE)
  add_definitions(-DBUILD_SPARSE=1)
endif()

OPTION(CMAKE_WITH_BARRIER_TIMING "Barrier placed to collect time" ON)
if(CMAKE_WITH_BARRIER_TIMING)
  add_definitions(-D__WITH__BARRIER__TIMING__=1)
endif()

#while exception get the stack trace
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -rdynamic -g3 -O0 ")

set(CMAKE_MODULE_PATH ${ARMADILLO_INCLUDE_DIR}/../cmake_aux/Modules/)
message(STATUS "CMAKE_MODULE_PATH = ${CMAKE_MODULE_PATH}" )

include(CheckIncludeFileCXX)
include(CheckLibraryExists)

##
## Find LAPACK and BLAS libraries, or their optimised versions
##

set(NMFLIB_OS unix)
set(ARMADILLO_INCLUDE_DIRS ${ARMADILLO_INCLUDE_DIR}/../)
set(ARMADILLO_LIBRARY_DIRS ${ARMADILLO_INCLUDE_DIR}/../)

if(NOT CMAKE_IGNORE_MKL)
  include(${ARMADILLO_INCLUDE_DIR}/../cmake_aux/Modules/ARMA_FindMKL.cmake)
endif()
include(${ARMADILLO_INCLUDE_DIR}/../cmake_aux/Modules/ARMA_FindOpenBLAS.cmake)
include(${ARMADILLO_INCLUDE_DIR}/../cmake_aux/Modules/ARMA_FindBLAS.cmake)
include(${ARMADILLO_INCLUDE_DIR}/../cmake_aux/Modules/ARMA_FindLAPACK.cmake)

message(STATUS "     MKL_FOUND = ${MKL_ROOT}"     )
message(STATUS "OpenBLAS_FOUND = ${OpenBLAS_FOUND}")
message(STATUS "    BLAS_FOUND = ${BLAS_FOUND}"    )
message(STATUS "  LAPACK_FOUND = ${LAPACK_FOUND}"  )

if(MKL_FOUND)
  set(NMFLIB_USE_LAPACK true)
  set(NMFLIB_USE_BLAS   true)
  add_definitions(-DMKL_FOUND=1)
  set(MKL_INCLUDE_DIR ${MKL_ROOT}/include)
  link_directories(${MKL_ROOT}/lib/intel64)
  #just linked with mkl_rt didn't work fine for syrk call. 
  #https://software.intel.com/en-us/forums/intel-math-kernel-library/topic/737425
  #overriding MKL_LIBRARIES and CXX flags
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMKL_ILP64 -m64")
  #set(MKL_LIBRARIES "-Wl,--start-group ${MKL_ROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKL_ROOT}/lib/intel64/libmkl_gnu_thread.a ${MKL_ROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl")
  set(MKL_LIBRARIES "-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_rt -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl") 
  set(NMFLIB_LIBS ${NMFLIB_LIBS} ${MKL_LIBRARIES})
else()
  if(OpenBLAS_FOUND AND BLAS_FOUND)
    message(STATUS "")
    message(STATUS "*** WARNING: found both OpenBLAS and BLAS. BLAS will not be used")
  endif()

  if(OpenBLAS_FOUND)
    set(NMFLIB_USE_BLAS true)      
    set(NMFLIB_LIBS ${NMFLIB_LIBS} ${OpenBLAS_LIBRARIES})
    get_filename_component(OPENBLAS_DIR ${OpenBLAS_LIBRARIES} DIRECTORY)
    set(OPENBLAS_INCLUDE_DIR ${OPENBLAS_DIR}/../include)
  else()

    if(BLAS_FOUND)
      set(NMFLIB_USE_BLAS true)
      set(NMFLIB_LIBS ${NMFLIB_LIBS} ${BLAS_LIBRARIES})
    endif()

  endif()

  if(LAPACK_FOUND)
    set(NMFLIB_USE_LAPACK true)
    set(NMFLIB_LIBS ${NMFLIB_LIBS} ${LAPACK_LIBRARIES})
  endif()

endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
if(DEFINED CMAKE_CXX_COMPILER_ID AND DEFINED CMAKE_CXX_COMPILER_VERSION)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT ${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 4.8.3)
    set(NMFLIB_USE_EXTERN_CXX11_RNG true)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    message(STATUS "Detected gcc 4.8.3 or later. Added '-std=c++11' to compiler flags")
  endif()
endif()

OPTION(CMAKE_BUILD_CUDA "Build with CUDA/NVBLAS" OFF)
if(CMAKE_BUILD_CUDA)
  add_definitions(-DBUILD_CUDA=1)
  find_package(CUDA REQUIRED)
  message(STATUS " CUDA_FOUND = ${CUDA_FOUND}" )
  if (CUDA_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp") 
    set(NMFLIB_LIBS ${CUDA_LIBRARIES} ${NMFLIB_LIBS})
    set(NMFLIB_LIBS ${CUDA_CUBLAS_LIBRARIES} ${NMFLIB_LIBS})
    find_library(CUDA_NVBLAS_LIBRARY 
                 NAMES nvblas
                 PATHS ${CUDA_TOOLKIT_ROOT_DIR}
                 PATH_SUFFIXES lib64
                 NO_DEFAULT_PATH)
    set(NMFLIB_LIBS ${CUDA_NVBLAS_LIBRARY} ${NMFLIB_LIBS})
    include_directories(${CUDA_INCLUDE_DIRS})
  endif()
endif()


set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3 -O0" CACHE STRING "CXX_DFLAGS_DEBUG" FORCE )
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3" CACHE STRING "CXX_FLAGS_RELEASE" FORCE )
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-as-needed")