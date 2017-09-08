#Copyright 2016 Ramakrishnan Kannan

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(ARMA_USE_WRAPPER true)

# the settings below will be automatically configured by the rest of this script

project(NTF CXX)
set(NTFLIB_COMMON ${NTF_SOURCE_DIR}/../common)
set(NNLS_SRC ${NTF_SOURCE_DIR}/../nnls/)
message(STATUS "NTF_SOURCE_DIR =  ${NTF_SOURCE_DIR}")
message(STATUS "NTFLIB_COMMON =  ${NTFLIB_COMMON}")

include(${NTFLIB_COMMON}/cmake_aux/NMFLIB_FindARMA.cmake)

#BOOST package needed for activeset NNLS
#set(BOOST false)
#As advised by Wlliam Renaud note dated 4/22. There is an issue on Rhea
#in which the system boost is found before the version in modules.
#Ignore system boost and use module system boost
#set(Boost_NO_BOOST_CMAKE TRUE)
#find_package(Boost REQUIRED)

message(STATUS "   Boost_FOUND = ${Boost_FOUND}"   )

#to build sparse comment or uncomment this line.
OPTION(CMAKE_BUILD_SPARSE "Build Sparse" OFF)
if(CMAKE_BUILD_SPARSE)
  add_definitions(-DBUILD_SPARSE=1)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -std=gnu++11")

message(STATUS "CMAKE_CXX_FLAGS           = ${CMAKE_CXX_FLAGS}"          )
message(STATUS "CMAKE_SHARED_LINKER_FLAGS = ${CMAKE_SHARED_LINKER_FLAGS}")
message(STATUS "CMAKE_REQUIRED_INCLUDES   = ${CMAKE_REQUIRED_INCLUDES}"  )

message(STATUS "*** libraries:")
message(STATUS "*** NTFLIB_LIBS = ${NTFLIB_LIBS} ${NMFLIB_LIBS}")

include_directories(
  ${ARMADILLO_INCLUDE_DIR}
  ${ARMADILLO_INCLUDE_DIRS}
  ${NTF_SOURCE_DIR}
  ${NTFLIB_COMMON}
  ${NNLS_SRC}
  ${Boost_INCLUDE_DIR}
  ${MKL_INCLUDE_DIR}
  ${OPENBLAS_INCLUDE_DIR}
)

# add_executable(ntf
#  ntf.cpp
# )

add_executable(ntf
#mttkrp_test.cpp
ntf.cpp
)

target_link_libraries(ntf ${NTFLIB_LIBS} ${NMFLIB_LIBS})
