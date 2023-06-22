#Copyright 2022 Ramakrishnan Kannan

cmake_minimum_required(VERSION 3.6 FATAL_ERROR)

set(ARMA_USE_WRAPPER true)

# the settings below will be automatically configured by the rest of this script

project(NMF CXX)
set(NMFLIB_COMMON ${NMF_SOURCE_DIR}/../common)
set(NNLS_SRC ${NMF_SOURCE_DIR}/../nnls/)
message(STATUS "NMF_SOURCE_DIR =  ${NMF_SOURCE_DIR}")
message(STATUS "NMFLIB_COMMON =  ${NMFLIB_COMMON}")

include(${NMFLIB_COMMON}/cmake_aux/NMFLIB_FindARMA.cmake)

message(STATUS "CMAKE_CXX_FLAGS           = ${CMAKE_CXX_FLAGS}"          )
message(STATUS "CMAKE_SHARED_LINKER_FLAGS = ${CMAKE_SHARED_LINKER_FLAGS}")
message(STATUS "CMAKE_REQUIRED_INCLUDES   = ${CMAKE_REQUIRED_INCLUDES}"  )

message(STATUS "*** libraries:")
message(STATUS "*** NMFLIB_LIBS = ${NMFLIB_LIBS}")

include_directories(
  ${ARMADILLO_INCLUDE_DIR}
  ${ARMADILLO_INCLUDE_DIRS}
  ${NMF_SOURCE_DIR}
  ${NMF_SOURCE_DIR}/../
  ${NMFLIB_COMMON}
  ${NNLS_SRC}
  ${Boost_INCLUDE_DIR}
  ${MKL_INCLUDE_DIR}
  ${OPENBLAS_INCLUDE_DIR}
)

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

add_executable(${JNMF_TYPE}jointnmf jointnmf.cpp jointnmf_driver.hpp)
target_link_libraries(${JNMF_TYPE}jointnmf ${NMFLIB_LIBS})
install(TARGETS ${JNMF_TYPE}jointnmf
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} )
