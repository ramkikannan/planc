#Copyright 2019 Ramakrishnan Kannan

# This builds all the executable for the planc project

cmake_minimum_required(VERSION 3.6 FATAL_ERROR)

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

add_subdirectory(ntf/)
add_subdirectory(nmf/)
add_subdirectory(distntf/)
add_subdirectory(distnmf/)
set(CMAKE_BUILD_SPARSE 1)
add_subdirectory(nmf/ sparse_nmf/)
add_subdirectory(distnmf/ sparse_distnmf/)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/nmf/dense_nmf DESTINATION ${CMAKE_INSTALL_PREFIX})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ntf/dense_ntf DESTINATION ${CMAKE_INSTALL_PREFIX})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/distnmf/dense_distnmf DESTINATION ${CMAKE_INSTALL_PREFIX})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/distntf/dense_distntf DESTINATION ${CMAKE_INSTALL_PREFIX})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/sparse_distnmf/sparse_distnmf DESTINATION ${CMAKE_INSTALL_PREFIX})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/sparse_nmf/sparse_nmf DESTINATION ${CMAKE_INSTALL_PREFIX})

