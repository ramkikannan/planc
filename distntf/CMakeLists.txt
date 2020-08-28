#Copyright 2016 Ramakrishnan Kannan

cmake_minimum_required(VERSION 3.6 FATAL_ERROR)

set(ARMA_USE_WRAPPER true)

# the settings below will be automatically configured by the rest of this script

project(DISTNTF CXX)
set(NMFLIB_COMMON ${DISTNTF_SOURCE_DIR}/../common)
set(NNLS_SRC ${DISTNTF_SOURCE_DIR}/../nnls/)
message(STATUS "DISTNTF_SOURCE_DIR =  ${DISTNTF_SOURCE_DIR}")
message(STATUS "NMFLIB_COMMON =  ${NMFLIB_COMMON}")

include(${NMFLIB_COMMON}/cmake_aux/NMFLIB_FindARMA.cmake)

add_definitions(-DMPI_DISTNTF=1)
add_definitions(-fopenmp)

find_package(MPI REQUIRED)

message(STATUS "   Boost_FOUND = ${Boost_FOUND}"   )
message(STATUS " MPI_CXX_FOUND = ${MPI_CXX_FOUND}" )

if(MPI_CXX_FOUND)
  set(NMFLIB_LIBS ${NMFLIB_LIBS} ${MPI_CXX_LIBRARIES})
endif()

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
  ${DISTNTF_SOURCE_DIR}
  ${DISTNTF_SOURCE_DIR}/../
  ${NMFLIB_COMMON}
  ${NNLS_SRC}
  ${MPI_CXX_INCLUDE_PATH}
  ${Boost_INCLUDE_DIR}
  ${MKL_INCLUDE_DIR}
  ${OPENBLAS_INCLUDE_DIR}
)

add_executable(dense_distntf
  distntf.cpp
)

target_link_libraries(dense_distntf ${NMFLIB_LIBS})
install(TARGETS dense_distntf
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} )