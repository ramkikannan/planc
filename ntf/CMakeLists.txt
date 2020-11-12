#Copyright 2016 Ramakrishnan Kannan

cmake_minimum_required(VERSION 3.6 FATAL_ERROR)

set(ARMA_USE_WRAPPER true)

# the settings below will be automatically configured by the rest of this script

project(NTF CXX)
set(NTFLIB_COMMON ${NTF_SOURCE_DIR}/../common)
set(NNLS_SRC ${NTF_SOURCE_DIR}/../nnls/)
message(STATUS "NTF_SOURCE_DIR =  ${NTF_SOURCE_DIR}")
message(STATUS "NTFLIB_COMMON =  ${NTFLIB_COMMON}")

include(${NTFLIB_COMMON}/cmake_aux/NMFLIB_FindARMA.cmake)

message(STATUS "CMAKE_CXX_FLAGS           = ${CMAKE_CXX_FLAGS}"          )
message(STATUS "CMAKE_SHARED_LINKER_FLAGS = ${CMAKE_SHARED_LINKER_FLAGS}")
message(STATUS "CMAKE_REQUIRED_INCLUDES   = ${CMAKE_REQUIRED_INCLUDES}"  )

message(STATUS "*** libraries:")
message(STATUS "*** NTFLIB_LIBS = ${NTFLIB_LIBS} ${NMFLIB_LIBS}")

add_definitions(-fopenmp)

include_directories(
  ${ARMADILLO_INCLUDE_DIR}
  ${ARMADILLO_INCLUDE_DIRS}
  ${NTF_SOURCE_DIR}
  ${NTF_SOURCE_DIR}/../
  ${NTFLIB_COMMON}
  ${NNLS_SRC}
  ${Boost_INCLUDE_DIR}
  ${MKL_INCLUDE_DIR}
  ${OPENBLAS_INCLUDE_DIR}
)

add_executable(dense_ntf
ntf.cpp
)

target_link_libraries(dense_ntf ${NTFLIB_LIBS} ${NMFLIB_LIBS})
install(TARGETS dense_ntf
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} )