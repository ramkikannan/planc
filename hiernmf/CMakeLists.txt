#Copyright 2016 Ramakrishnan Kannan

cmake_minimum_required(VERSION 3.6 FATAL_ERROR)

set(ARMA_USE_WRAPPER true)

# the settings below will be automatically configured by the rest of this script

# set(ARMADILLO_LIB_DIR /home/okaya/armadillo-6.600.5/install/usr/lib)

project(HIERNMF CXX)

set(NMFLIB_COMMON ${HIERNMF_SOURCE_DIR}/../common)
set(NNLS_SRC ${HIERNMF_SOURCE_DIR}/../nnls/)
message(STATUS "HIERNMF_SOURCE_DIR =  ${HIERNMF_SOURCE_DIR}")
message(STATUS "NMFLIB_COMMON =  ${NMFLIB_COMMON}")

include(${NMFLIB_COMMON}/cmake_aux/NMFLIB_FindARMA.cmake)

#BOOST package needed for activeset NNLS
#set(BOOST false)
#As advised by Wlliam Renaud note dated 4/22. There is an issue on Rhea
#in which the system boost is found before the version in modules.
#Ignore system boost and use module system boost
#set(Boost_NO_BOOST_CMAKE TRUE)
#find_package(Boost REQUIRED)

find_package(MPI REQUIRED)

message(STATUS " MPI_CXX_FOUND = ${MPI_CXX_FOUND}" )

if(MPI_CXX_FOUND)
  set(NMFLIB_LIBS ${NMFLIB_LIBS} ${MPI_CXX_LIBRARIES})
endif()

OPTION(CMAKE_BUILD_PACOSS "Build with PACOSS" OFF)
if(CMAKE_WITH_PACOSS)
  add_definitions(-DUSE_PACOSS=1)
  set(PACOSS_DIR ${DISTNMF_SOURCE_DIR}/../pacoss)
  set(PACOSS_INCLUDE_DIR ${DISTNMF_SOURCE_DIR}/../pacoss/src) 
  set(PACOSS_INCLUDE_DIR ${PACOSS_INCLUDE_DIR} ${DISTNMF_SOURCE_DIR}/../tmpi/src)
  set(PACOSS_LIB_DIR ${DISTNMF_SOURCE_DIR}/../pacoss/lib)
  link_directories(
    ${PACOSS_LIB_DIR}  
  )
  set(NMFLIB_LIBS "${NMFLIB_LIBS} pacoss")
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
  ${PACOSS_INCLUDE_DIR}
  ${ARMADILLO_INCLUDE_DIR}
  ${ARMADILLO_INCLUDE_DIRS}
  ${HIERNMF_SOURCE_DIR}
  ${HIERNMF_SOURCE_DIR}/../
  ${NMFLIB_COMMON}
  ${NNLS_SRC}
  ${MPI_CXX_INCLUDE_PATH}
  ${MKL_INCLUDE_DIR}
  ${OPENBLAS_INCLUDE_DIR}
)

if(CMAKE_WITH_PACOSS)
  set(DENSE_OR_SPARSE sparse)
  add_executable(sparse_hiernmf
    hiernmf.cpp
  )
  target_link_libraries(sparse_hiernmf ${NMFLIB_LIBS} pacoss)
elseif(CMAKE_BUILD_SPARSE)
  set(DENSE_OR_SPARSE sparse)
  add_executable(sparse_hiernmf
    hiernmf.cpp
  )
  target_link_libraries(sparse_hiernmf ${NMFLIB_LIBS})
else()
  set(DENSE_OR_SPARSE dense)
  add_executable(dense_hiernmf
    hiernmf.cpp
  )
  target_link_libraries(dense_hiernmf ${NMFLIB_LIBS})
endif()

install(TARGETS ${DENSE_OR_SPARSE}_hiernmf
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} )
