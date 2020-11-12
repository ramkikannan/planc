cmake_minimum_required(VERSION 3.6 FATAL_ERROR)

# the settings below will be automatically configured by the rest of this script

project(PLANC CXX)
set(PLANC_MAJOR_VERSION 0)
set(PLANC_MINOR_VERSION 8)
set(PLANC_PATCH_VERSION 1)
set(PLANC_VERSION "${PLANC_MAJOR_VERSION}.${PLANC_MINOR_VERSION}.${PLANC_PATCH_VERSION}")

# Adhere to GNU filesystem layout conventions
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 11)
cmake_host_system_information(RESULT _host_name QUERY FQDN)

message(STATUS, "host name = ${_host_name}")

string(FIND ${_host_name} "rhea" RHEA_FOUND)
string(FIND ${_host_name} "eos" EOS_FOUND)
string(FIND ${_host_name} "titan" TITAN_FOUND)
string(FIND ${_host_name} "summit" SUMMIT_FOUND)

message(STATUS, "RHEA = ${RHEA_FOUND}")
message(STATUS, "EOS = ${EOS_FOUND}")
message(STATUS, "TITAN = ${TITAN_FOUND}")
message(STATUS, "SUMMIT = ${SUMMIT_FOUND}")

if(NOT (TITAN_FOUND EQUAL -1 AND SUMMIT_FOUND EQUAL -1))
    set(CMAKE_BUILD_CUDA 1)
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

set(NMF_SOURCE_DIR ${PROJECT_SOURCE_DIR}/nmf)
set(NTF_SOURCE_DIR ${PROJECT_SOURCE_DIR}/ntf)
set(DISTNMF_SOURCE_DIR ${PROJECT_SOURCE_DIR}/distnmf)
set(DISTNTF_SOURCE_DIR ${PROJECT_SOURCE_DIR}/distntf)
set(HIERNMF_SOURCE_DIR ${PROJECT_SOURCE_DIR}/hiernmf)

set(DENSEDIRS nmf distnmf distntf hiernmf)
set(SPARSEDIRS nmf distnmf hiernmf)

set(CMAKE_BUILD_SPARSE 0)
foreach (DENSEDIR ${DENSEDIRS})
  add_subdirectory(${DENSEDIR} dense_${DENSEDIR})
endforeach()
  
set(CMAKE_BUILD_SPARSE 1)
set(CMAKE_BUILD_CUDA 0)
foreach (SPARSEDIR ${SPARSEDIRS})
  add_subdirectory(${SPARSEDIR} sparse_${SPARSEDIR})
endforeach()