#Copyright 2019 Ramakrishnan Kannan

# This builds all the executable for the planc project

cmake_minimum_required(VERSION 3.6 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 11)
cmake_host_system_information(RESULT _host_name QUERY HOSTNAME)

message(STATUS, "host name = ${_host_name}")

string(FIND ${_host_name} "rhea" RHEA_FOUND)
string(FIND ${_host_name} "eos" EOS_FOUND)
string(FIND ${_host_name} "titan" TITAN_FOUND)
string(FIND ${_host_name} "summit" SUMMIT_FOUND)

message(STATUS, "RHEA = ${RHEA_FOUND}")
message(STATUS, "EOS = ${EOS_FOUND}")
message(STATUS, "TITAN = ${TITAN_FOUND}")
message(STATUS, "SUMMIT = ${SUMMIT_FOUND}")


if(TITAN_FOUND EQUAL 0)
    set(CMAKE_BUILD_CUDA 1)
endif()

if(SUMMIT_FOUND EQUAL 0)
    set(CMAKE_BUILD_CUDA 1)
endif()

add_subdirectory(ntf/)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ntf/dense_ntf RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX})
add_subdirectory(nmf/)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/nmf/dense_nmf RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX})
add_subdirectory(distntf/)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/distntf/dense_distntf RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX})
add_subdirectory(distnmf/)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/distnmf/dense_distnmf RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX})
set(CMAKE_BUILD_SPARSE 1)
add_subdirectory(nmf/ sparse_nmf/)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/sparse_nmf/sparse_nmf RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX})
add_subdirectory(distnmf/ sparse_distnmf/)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/sparse_distnmf/sparse_distnmf RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX})
