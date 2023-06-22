#Copyright 2022 Ramakrishnan Kannan

cmake_minimum_required(VERSION 3.6 FATAL_ERROR)

set(ARMA_USE_WRAPPER true)

project(DISTJOINTNMF CXX)

set(NMFLIB_COMMON ${DISTJOINTNMF_SOURCE_DIR}/../common)
set(NNLS_SRC ${DISTJOINTNMF_SOURCE_DIR}/../nnls/)
message(STATUS "DISTJOINTNMF_SOURCE_DIR =  ${DISTJOINTNMF_SOURCE_DIR}")
message(STATUS "NMFLIB_COMMON =  ${NMFLIB_COMMON}")

include(${NMFLIB_COMMON}/cmake_aux/NMFLIB_FindARMA.cmake)

find_package(MPI REQUIRED)

message(STATUS " MPI_CXX_FOUND = ${MPI_CXX_FOUND}" )

if(MPI_CXX_FOUND)
  set(NMFLIB_LIBS ${NMFLIB_LIBS} ${MPI_CXX_LIBRARIES})
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3 -O0 -rdynamic")

message(STATUS "CMAKE_CXX_FLAGS           = ${CMAKE_CXX_FLAGS}"          )
message(STATUS "CMAKE_SHARED_LINKER_FLAGS = ${CMAKE_SHARED_LINKER_FLAGS}")
message(STATUS "CMAKE_REQUIRED_INCLUDES   = ${CMAKE_REQUIRED_INCLUDES}"  )

message(STATUS "")
message(STATUS "*** libraries:")
message(STATUS "*** NMFLIB_LIBS = ${NMFLIB_LIBS}")
message(STATUS "")

include_directories(
  ${ARMADILLO_INCLUDE_DIR}
  ${ARMADILLO_INCLUDE_DIRS}
  ${DISTJOINTNMF_SOURCE_DIR}
  ${DISTJOINTNMF_SOURCE_DIR}/../
  ${NMFLIB_COMMON}
  ${NNLS_SRC}
  ${MPI_CXX_INCLUDE_PATH}
  ${MKL_INCLUDE_DIR}
  ${OPENBLAS_INCLUDE_DIR}
)

# set(BLA_VENDOR Intel)
if (ENABLE_MKL AND MKL_LIBS)
  message(STATUS "ENABLED MKL AT: ${MKL_LIBS}")
  add_compile_definitions(MKL_FOUND)
  set(BLAS_LIBRARIES ${MKL_LIBS}:${BLAS_LIBRARIES})
elseif (ENABLE_MKL AND NOT MKL_LIBS)
  message(STATUS "ENABLED MKL, BUT MKL PATH NOT PROVIDED. SEARCHING FOR MKL WITH FIND_LIBRARY")
  FIND_LIBRARY(MKL 
    NAMES mkl 
    HINTS [${MKLROOT}, ${MKL_ROOT}])

  message(STATUS MKL: ${MKL})

  if (${MKL} STREQUAL MKL-NOTFOUND)
    message(STATUS "UNABLE TO FIND MKL, CONTINUING ON WITHOUT IT")
  else ()
    message(STATUS "FOUND MKL AT: ${MKL}") #${MKL_LIBS}
    add_compile_definitions(MKL_FOUND)
    set(BLAS_LIBRARIES ${MKL}:${BLAS_LIBRARIES})
  endif (${MKL} STREQUAL MKL-NOTFOUND)
  
else (NOT ENABLE_MKL)
  message(STATUS "MKL WAS NOT ENABLED")
endif (ENABLE_MKL AND MKL_LIBS)

# Check Barrier timings
if(CMAKE_WITH_BARRIER_TIMING)
  message(STATUS "MPI Barrier timings enabled.")
endif()

if(CMAKE_BUILD_SPDEN)
  add_definitions(-DBUILD_SPDEN=1)
  set(JNMF_TYPE spden_)
elseif(CMAKE_BUILD_DENSP)
  add_definitions(-DBUILD_DENSP=1)
  set(JNMF_TYPE densp_)
elseif(CMAKE_BUILD_DENDEN)
  add_definitions(-DBUILD_DENDEN=1)
  set(JNMF_TYPE denden_)
elseif(CMAKE_BUILD_SPSP)
  add_definitions(-DBUILD_SPSP=1)
  set(JNMF_TYPE spsp_)
else()
  set(JNMF_TYPE)
endif()

add_executable(${JNMF_TYPE}distjointnmf distjointnmf.cpp)
target_link_libraries(${JNMF_TYPE}distjointnmf ${NMFLIB_LIBS})

install(TARGETS ${JNMF_TYPE}distjointnmf
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} )