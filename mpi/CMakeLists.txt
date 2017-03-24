#Copyright 2016 Ramakrishnan Kannan

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(ARMA_USE_WRAPPER true)

# the settings below will be automatically configured by the rest of this script

# set(ARMADILLO_LIB_DIR /home/okaya/armadillo-6.600.5/install/usr/lib)

project(DISTNMF CXX)

set(NMFLIB_COMMON ${DISTNMF_SOURCE_DIR}/../common)
set(NNLS_SRC ${DISTNMF_SOURCE_DIR}/../nnls/)
message(STATUS "DISTNMF_SOURCE_DIR =  ${DISTNMF_SOURCE_DIR}")
message(STATUS "NMFLIB_COMMON =  ${NMFLIB_COMMON}")

include(${NMFLIB_COMMON}/cmake_aux/NMFLIB_FindARMA.cmake)

#BOOST package needed for activeset NNLS
set(BOOST false)
#As advised by Wlliam Renaud note dated 4/22. There is an issue on Rhea
#in which the system boost is found before the version in modules.
#Ignore system boost and use module system boost
set(Boost_NO_BOOST_CMAKE TRUE)
find_package(Boost REQUIRED)

find_package(MPI REQUIRED)

message(STATUS "   Boost_FOUND = ${Boost_FOUND}"   )
message(STATUS " MPI_CXX_FOUND = ${MPI_CXX_FOUND}" )

if(MKL_FOUND)
  add_definitions(-DMKL_FOUND=1)
  set(MKL_INCLUDE_DIR $ENV{MKLROOT}/include)
endif()

if(MPI_CXX_FOUND)
  set(NMFLIB_LIBS ${NMFLIB_LIBS} ${MPI_CXX_LIBRARIES})
endif()

#to build sparse comment or uncomment this line.
OPTION(CMAKE_BUILD_SPARSE "Build Sparse" OFF)
if(CMAKE_BUILD_SPARSE)
  add_definitions(-DBUILD_SPARSE=1)
endif()

OPTION(CMAKE_WITH_BARRIER_TIMING "Barrier placed to collect time" ON)
if(CMAKE_WITH_BARRIER_TIMING)
  add_definitions(-D__WITH__BARRIER__TIMING__=1)
endif()

OPTION(CMAKE_BUILD_PACOSS "Build with PACOSS" OFF)
if(CMAKE_WITH_PACOSS)
  add_definitions(-DUSE_PACOSS=1)
  set(PACOSS_DIR ${DISTNMF_SOURCE_DIR}/../pacoss)
  set(PACOSS_INCLUDE_DIR ${DISTNMF_SOURCE_DIR}/../pacoss/src) 
  set(PACOSS_INCLUDE_DIR ${PACOSS_INCLUDE_DIR} ${DISTNMF_SOURCE_DIR}/../tmpi/src)
  set(PACOSS_LIB_DIR ${DISTNMF_SOURCE_DIR}/../pacoss/lib)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3 -O0")

message(STATUS "CMAKE_CXX_FLAGS           = ${CMAKE_CXX_FLAGS}"          )
message(STATUS "CMAKE_SHARED_LINKER_FLAGS = ${CMAKE_SHARED_LINKER_FLAGS}")
message(STATUS "CMAKE_REQUIRED_INCLUDES   = ${CMAKE_REQUIRED_INCLUDES}"  )

message(STATUS "")
message(STATUS "*** libraries:")
message(STATUS "*** NMFLIB_LIBS = ${NMFLIB_LIBS}")
message(STATUS "")

include_directories(
  if(CMAKE_WITH_PACOSS)
    ${PACOSS_INCLUDE_DIR}
  endif()
  ${ARMADILLO_INCLUDE_DIR}
  ${ARMADILLO_INCLUDE_DIRS}
  ${DISTNMF_SOURCE_DIR}
  ${NMFLIB_COMMON}
  ${NNLS_SRC}
  if(MPI_CXX_FOUND)
    ${MPI_CXX_INCLUDE_PATH}
  endif()
  ${Boost_INCLUDE_DIR}
  ${MKL_INCLUDE_DIR}
)


if(CMAKE_WITH_PACOSS)
  link_directories(
    ${PACOSS_LIB_DIR}  
  )
endif()

add_executable(distnmf
  distnmf.cpp
)

add_definitions(-std=c++11 -w)

if(CMAKE_WITH_PACOSS)
  target_link_libraries(distnmf ${NMFLIB_LIBS} pacoss)
else()
  target_link_libraries(distnmf ${NMFLIB_LIBS} pacoss)
endif()
