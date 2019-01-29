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

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

set(NMF_SOURCE_DIR ${PROJECT_SOURCE_DIR}/nmf)
set(NTF_SOURCE_DIR ${PROJECT_SOURCE_DIR}/ntf)
set(DISTNMF_SOURCE_DIR ${PROJECT_SOURCE_DIR}/nmf)
set(DISTNTF_SOURCE_DIR ${PROJECT_SOURCE_DIR}/nmf)

set(DENSEDIRS nmf distnmf distntf)
set(SPARSEDIRS nmf distnmf)

set(CMAKE_BUILD_SPARSE 0)
foreach (DENSEDIR ${DENSEDIRS})
  add_subdirectory(${DENSEDIR} dense_${DENSEDIR})
endforeach()
  
set(CMAKE_BUILD_SPARSE 1)
foreach (SPARSEDIR ${SPARSEDIRS})
  add_subdirectory(${SPARSEDIR} sparse_${SPARSEDIR})
endforeach()
